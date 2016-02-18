use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::sync::Arc;

use device::Device;
use framebuffer::Subpass;
use shader::FragmentShaderEntryPoint;
use shader::VertexShaderEntryPoint;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

use pipeline::GenericPipeline;
use pipeline::blend::Blend;
use pipeline::input_assembly::InputAssembly;
use pipeline::multisample::Multisample;
use pipeline::raster::Rasterization;
use pipeline::vertex::MultiVertex;
use pipeline::vertex::Vertex;

///
///
/// The template parameter contains the descriptor set to use with this pipeline, and the
/// renderpass layout.
pub struct GraphicsPipeline<MultiVertex> {
    device: Arc<Device>,
    pipeline: vk::Pipeline,

    dynamic_line_width: bool,

    marker: PhantomData<(MultiVertex,)>
}

impl<MV> GraphicsPipeline<MV>
    where MV: MultiVertex
{
    /// Builds a new graphics pipeline object.
    ///
    /// # Panic
    ///
    /// - Panicks if primitive restart is enabled and the topology doesn't support this feature.
    /// - Panicks if the `rasterization_samples` parameter of `multisample` is not >= 1.
    /// - Panicks if the `sample_shading` parameter of `multisample` is not between 0.0 and 1.0.
    ///
    // TODO: check all the device's limits
    pub fn new<V, F, R>(device: &Arc<Device>, vertex_shader: &VertexShaderEntryPoint<V>,
                        input_assembly: &InputAssembly, raster: &Rasterization,
                        multisample: &Multisample, blend: &Blend,
                        fragment_shader: &FragmentShaderEntryPoint<F>, render_pass: &Subpass<R>)
                        -> Result<Arc<GraphicsPipeline<MV>>, OomError>
    {
        let vk = device.pointers();

        let pipeline = unsafe {
            let mut dynamic_states: Vec<vk::DynamicState> = Vec::new();
            let mut stages = Vec::with_capacity(5);

            stages.push(vk::PipelineShaderStageCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                stage: vk::SHADER_STAGE_VERTEX_BIT,
                module: vertex_shader.module().internal_object(),
                pName: vertex_shader.name().as_ptr(),
                pSpecializationInfo: ptr::null(),       // TODO:
            });

            stages.push(vk::PipelineShaderStageCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                stage: vk::SHADER_STAGE_FRAGMENT_BIT,
                module: fragment_shader.module().internal_object(),
                pName: fragment_shader.name().as_ptr(),
                pSpecializationInfo: ptr::null(),       // TODO:
            });

            let binding_descriptions = (0 .. MV::num_buffers()).map(|num| {
                let (stride, rate) = MV::buffer_info(num);
                vk::VertexInputBindingDescription {
                    binding: num,
                    stride: stride,
                    inputRate: rate as u32,
                }
            }).collect::<Vec<_>>();

            let attribute_descriptions = vertex_shader.attributes().iter().enumerate().map(|(loc, name)| {
                let (binding, info) = MV::attrib(name).expect("missing attr");       // TODO: error
                vk::VertexInputAttributeDescription {
                    location: loc as u32,
                    binding: binding,
                    format: info.format as u32,
                    offset: info.offset as u32,
                }
            }).collect::<Vec<_>>();

            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                vertexBindingDescriptionCount: binding_descriptions.len() as u32,
                pVertexBindingDescriptions: binding_descriptions.as_ptr(),
                vertexAttributeDescriptionCount: attribute_descriptions.len() as u32,
                pVertexAttributeDescriptions: attribute_descriptions.as_ptr(),
            };

            assert!(!input_assembly.primitive_restart_enable ||
                    input_assembly.topology.supports_primitive_restart());
            let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                topology: input_assembly.topology as u32,
                primitiveRestartEnable: if input_assembly.primitive_restart_enable { vk::TRUE } else { vk::FALSE },
            };

            let vp = vk::Viewport { x: 0.0, y: 0.0, width: 1244.0, height: 699.0, minDepth: 0.0, maxDepth: 0.0 };
            let sc = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: 1244, height: 699 } };
            let viewport = vk::PipelineViewportStateCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                viewportCount: 1,       // FIXME:
                pViewports: &vp,        // FIXME:
                scissorCount: 1,        // FIXME:
                pScissors: &sc,     // FIXME:
            };

            if raster.line_width.is_none() {
                dynamic_states.push(vk::DYNAMIC_STATE_LINE_WIDTH);
            }

            let rasterization = vk::PipelineRasterizationStateCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                depthClampEnable: if raster.depth_clamp { vk::TRUE } else { vk::FALSE },
                rasterizerDiscardEnable: if raster.rasterizer_discard { vk::TRUE } else { vk::FALSE },
                polygonMode: raster.polygon_mode as u32,
                cullMode: raster.cull_mode as u32,
                frontFace: raster.front_face as u32,
                depthBiasEnable: if raster.depthBiasEnable { vk::TRUE } else { vk::FALSE },
                depthBiasConstantFactor: raster.depthBiasConstantFactor,
                depthBiasClamp: raster.depthBiasClamp,
                depthBiasSlopeFactor: raster.depthBiasSlopeFactor,
                lineWidth: raster.line_width.unwrap_or(1.0),
            };

            assert!(multisample.rasterization_samples >= 1);
            if let Some(s) = multisample.sample_shading { assert!(s >= 0.0 && s <= 1.0); }
            let multisample = vk::PipelineMultisampleStateCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                rasterizationSamples: multisample.rasterization_samples,
                sampleShadingEnable: if multisample.sample_shading.is_some() { vk::TRUE } else { vk::FALSE },
                minSampleShading: multisample.sample_shading.unwrap_or(1.0),
                pSampleMask: multisample.sample_mask.as_ptr(),
                alphaToCoverageEnable: if multisample.alpha_to_coverage { vk::TRUE } else { vk::FALSE },
                alphaToOneEnable: if multisample.alpha_to_one { vk::TRUE } else { vk::FALSE },
            };

            let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                depthTestEnable: vk::FALSE,          // FIXME:
                depthWriteEnable: vk::FALSE,         // FIXME:
                depthCompareOp: 0,           // FIXME:
                depthBoundsTestEnable: vk::FALSE,            // FIXME:
                stencilTestEnable: vk::FALSE,            // FIXME:
                front: vk::StencilOpState {
                    failOp: 0,           // FIXME:
                    passOp: 0,           // FIXME:
                    depthFailOp: 0,          // FIXME:
                    compareOp: 0,            // FIXME:
                    compareMask: 0,          // FIXME:
                    writeMask: 0,            // FIXME:
                    reference: 0,            // FIXME:
                },
                back: vk::StencilOpState {
                    failOp: 0,           // FIXME:
                    passOp: 0,           // FIXME:
                    depthFailOp: 0,          // FIXME:
                    compareOp: 0,            // FIXME:
                    compareMask: 0,          // FIXME:
                    writeMask: 0,            // FIXME:
                    reference: 0,            // FIXME:
                },
                minDepthBounds: 0.0,           // FIXME:
                maxDepthBounds: 1.0,           // FIXME:
            };

            let atch = vk::PipelineColorBlendAttachmentState {
                blendEnable: 0,
                srcColorBlendFactor: 0,
                dstColorBlendFactor: 0,
                colorBlendOp: 0,
                srcAlphaBlendFactor: 0,
                dstAlphaBlendFactor: 0,
                alphaBlendOp: 0,
                colorWriteMask: 0xf,
            };
            let blend = vk::PipelineColorBlendStateCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                logicOpEnable: if blend.logic_op.is_some() { vk::TRUE } else { vk::FALSE },
                logicOp: blend.logic_op.unwrap_or(Default::default()) as u32,
                attachmentCount: 1,         // FIXME:
                pAttachments: &atch,      // FIXME:
                blendConstants: blend.blend_constants.unwrap_or([0.0, 0.0, 0.0, 0.0]),
            };

            let dynamic_states = vk::PipelineDynamicStateCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                dynamicStateCount:  dynamic_states.len() as u32,
                pDynamicStates: dynamic_states.as_ptr(),
            };

            // FIXME: hack with leaking pipeline layout
            let layout = {
                let infos = vk::PipelineLayoutCreateInfo {
                    sType: vk::STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    pNext: ptr::null(),
                    flags: 0,
                    setLayoutCount: 0,
                    pSetLayouts: ptr::null(),
                    pushConstantRangeCount: 0,
                    pPushConstantRanges: ptr::null(),
                };
                let mut out = mem::uninitialized();
                try!(check_errors(vk.CreatePipelineLayout(device.internal_object(), &infos, 
                                                          ptr::null(), &mut out)));
                out
            };

            let infos = vk::GraphicsPipelineCreateInfo {
                sType: vk::STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,       // TODO: some flags are available but none are critical
                stageCount: stages.len() as u32,
                pStages: stages.as_ptr(),
                pVertexInputState: &vertex_input_state,
                pInputAssemblyState: &input_assembly,
                pTessellationState: ptr::null(),        // FIXME:
                pViewportState: &viewport,
                pRasterizationState: &rasterization,
                pMultisampleState: &multisample,
                pDepthStencilState: &depth_stencil,
                pColorBlendState: &blend,
                pDynamicState: &dynamic_states,
                layout: layout,      // FIXME:
                renderPass: render_pass.renderpass().internal_object(),
                subpass: render_pass.index(),
                basePipelineHandle: 0,    // TODO:
                basePipelineIndex: 0,       // TODO:
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateGraphicsPipelines(device.internal_object(), 0,
                                                         1, &infos, ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(GraphicsPipeline {
            device: device.clone(),
            pipeline: pipeline,

            dynamic_line_width: raster.line_width.is_none(),

            marker: PhantomData,
        }))
    }
}

impl<MultiVertex> GraphicsPipeline<MultiVertex> {
    /// Returns true if the line width used by this pipeline is dynamic.
    #[inline]
    pub fn has_dynamic_line_width(&self) -> bool {
        self.dynamic_line_width
    }
}

impl<MultiVertex> GenericPipeline for GraphicsPipeline<MultiVertex> {
}

impl<MultiVertex> VulkanObject for GraphicsPipeline<MultiVertex> {
    type Object = vk::Pipeline;

    #[inline]
    fn internal_object(&self) -> vk::Pipeline {
        self.pipeline
    }
}

impl<MultiVertex> Drop for GraphicsPipeline<MultiVertex> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyPipeline(self.device.internal_object(), self.pipeline, ptr::null());
        }
    }
}
