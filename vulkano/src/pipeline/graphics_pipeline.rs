use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::sync::Arc;

use device::Device;
use descriptor_set::PipelineLayout;
use descriptor_set::Layout as PipelineLayoutDesc;
use descriptor_set::LayoutPossibleSuperset as PipelineLayoutPossibleSuperset;
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
use pipeline::raster::DepthBiasControl;
use pipeline::raster::Rasterization;
use pipeline::vertex::MultiVertex;
use pipeline::vertex::Vertex;
use pipeline::viewport::ViewportsState;

///
///
/// The template parameter contains the descriptor set to use with this pipeline, and the
/// renderpass layout.
pub struct GraphicsPipeline<MultiVertex, Layout> {
    device: Arc<Device>,
    pipeline: vk::Pipeline,
    layout: Arc<PipelineLayout<Layout>>,

    dynamic_line_width: bool,
    dynamic_viewport: bool,
    dynamic_scissor: bool,
    dynamic_depth_bias: bool,

    num_viewports: u32,

    marker: PhantomData<(MultiVertex,)>
}

impl<MV, L> GraphicsPipeline<MV, L>
    where MV: MultiVertex, L: PipelineLayoutDesc
{
    /// Builds a new graphics pipeline object.
    ///
    /// # Panic
    ///
    /// - Panicks if primitive restart is enabled and the topology doesn't support this feature.
    /// - Panicks if the `rasterization_samples` parameter of `multisample` is not >= 1.
    /// - Panicks if the `sample_shading` parameter of `multisample` is not between 0.0 and 1.0.
    /// - Panicks if the line width is different from 1.0 and the `wide_lines` feature is not enabled.
    ///
    // TODO: check all the device's limits
    pub fn new<Vi, Fo, R, Vl, Fl>
              (device: &Arc<Device>, vertex_shader: &VertexShaderEntryPoint<Vi, Vl>,
               input_assembly: &InputAssembly, viewport: &ViewportsState,
               raster: &Rasterization, multisample: &Multisample, blend: &Blend,
               fragment_shader: &FragmentShaderEntryPoint<Fo, Fl>,
               layout: &Arc<PipelineLayout<L>>, render_pass: Subpass<R>)
               -> Result<Arc<GraphicsPipeline<MV, L>>, OomError>
        where L: PipelineLayoutDesc + PipelineLayoutPossibleSuperset<Vl> + PipelineLayoutPossibleSuperset<Fl>,
              Vl: PipelineLayoutDesc, Fl: PipelineLayoutDesc
    {
        let vk = device.pointers();

        assert!(PipelineLayoutPossibleSuperset::is_superset_of(layout.layout(), vertex_shader.layout()));
        assert!(PipelineLayoutPossibleSuperset::is_superset_of(layout.layout(), fragment_shader.layout()));

        let pipeline = unsafe {
            // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
            let mut dynamic_states: Vec<vk::DynamicState> = Vec::new();
            // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
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

            // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
            let binding_descriptions = (0 .. MV::num_buffers()).map(|num| {
                let (stride, rate) = MV::buffer_info(num);
                vk::VertexInputBindingDescription {
                    binding: num,
                    stride: stride,
                    inputRate: rate as u32,
                }
            }).collect::<Vec<_>>();

            // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
            let attribute_descriptions = vertex_shader.attributes().iter().map(|&(loc, ref name)| {
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

            let (vp_vp, vp_sc, vp_num) = match *viewport {
                ViewportsState::Fixed { ref data } => (
                    data.iter().map(|e| e.0.clone().into()).collect::<Vec<vk::Viewport>>(),
                    data.iter().map(|e| e.1.clone().into()).collect::<Vec<vk::Rect2D>>(),
                    data.len() as u32
                ),
                ViewportsState::DynamicViewports { ref scissors } => {
                    let num = scissors.len() as u32;
                    let scissors = scissors.iter().map(|e| e.clone().into())
                                           .collect::<Vec<vk::Rect2D>>();
                    dynamic_states.push(vk::DYNAMIC_STATE_VIEWPORT);
                    (vec![], scissors, num)
                },
                ViewportsState::DynamicScissors { ref viewports } => {
                    let num = viewports.len() as u32;
                    let viewports = viewports.iter().map(|e| e.clone().into())
                                             .collect::<Vec<vk::Viewport>>();
                    dynamic_states.push(vk::DYNAMIC_STATE_SCISSOR);
                    (viewports, vec![], num)
                },
                ViewportsState::Dynamic { num } => {
                    dynamic_states.push(vk::DYNAMIC_STATE_VIEWPORT);
                    dynamic_states.push(vk::DYNAMIC_STATE_SCISSOR);
                    (vec![], vec![], num)
                },
            };

            let viewport_info = vk::PipelineViewportStateCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                viewportCount: vp_num,
                pViewports: vp_vp.as_ptr(),
                scissorCount: vp_num,
                pScissors: vp_sc.as_ptr(),
            };

            if let Some(line_width) = raster.line_width {
                if line_width != 1.0 {
                    assert!(device.enabled_features().wide_lines);
                }
            } else {
                dynamic_states.push(vk::DYNAMIC_STATE_LINE_WIDTH);
            }

            let (db_enable, db_const, db_clamp, db_slope) = match raster.depth_bias {
                DepthBiasControl::Dynamic => {
                    dynamic_states.push(vk::DYNAMIC_STATE_DEPTH_BIAS);
                    (vk::TRUE, 0.0, 0.0, 0.0)
                },
                DepthBiasControl::Disabled => {
                    (vk::FALSE, 0.0, 0.0, 0.0)
                },
                DepthBiasControl::Static(bias) => {
                    // TODO: check the depthBiasClamp feature if clamp != 0.0
                    (vk::TRUE, bias.constant_factor, bias.clamp, bias.slope_factor)
                },
            };

            let rasterization = vk::PipelineRasterizationStateCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                depthClampEnable: if raster.depth_clamp { vk::TRUE } else { vk::FALSE },
                rasterizerDiscardEnable: if raster.rasterizer_discard { vk::TRUE } else { vk::FALSE },
                polygonMode: raster.polygon_mode as u32,
                cullMode: raster.cull_mode as u32,
                frontFace: raster.front_face as u32,
                depthBiasEnable: db_enable,
                depthBiasConstantFactor: db_const,
                depthBiasClamp: db_clamp,
                depthBiasSlopeFactor: db_slope,
                lineWidth: raster.line_width.unwrap_or(1.0),
            };

            assert!(multisample.rasterization_samples >= 1);
            // FIXME: check that rasterization_samples is equal to what's in the renderpass
            if let Some(s) = multisample.sample_shading { assert!(s >= 0.0 && s <= 1.0); }
            let multisample = vk::PipelineMultisampleStateCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                rasterizationSamples: multisample.rasterization_samples,
                sampleShadingEnable: if multisample.sample_shading.is_some() { vk::TRUE } else { vk::FALSE },
                minSampleShading: multisample.sample_shading.unwrap_or(1.0),
                pSampleMask: ptr::null(),   //multisample.sample_mask.as_ptr(),     // FIXME:
                alphaToCoverageEnable: if multisample.alpha_to_coverage { vk::TRUE } else { vk::FALSE },
                alphaToOneEnable: if multisample.alpha_to_one { vk::TRUE } else { vk::FALSE },
            };

            let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                depthTestEnable: vk::TRUE,          // FIXME:
                depthWriteEnable: vk::TRUE,         // FIXME:
                depthCompareOp: vk::COMPARE_OP_LESS,           // FIXME:
                depthBoundsTestEnable: vk::FALSE,            // FIXME:
                stencilTestEnable: vk::FALSE,            // FIXME:
                front: vk::StencilOpState {
                    failOp: vk::STENCIL_OP_KEEP,           // FIXME:
                    passOp: vk::STENCIL_OP_KEEP,           // FIXME:
                    depthFailOp: vk::STENCIL_OP_KEEP,          // FIXME:
                    compareOp: 0,            // FIXME:
                    compareMask: 0,          // FIXME:
                    writeMask: 0,            // FIXME:
                    reference: 0,            // FIXME:
                },
                back: vk::StencilOpState {
                    failOp: vk::STENCIL_OP_KEEP,           // FIXME:
                    passOp: vk::STENCIL_OP_KEEP,           // FIXME:
                    depthFailOp: vk::STENCIL_OP_KEEP,          // FIXME:
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

            let infos = vk::GraphicsPipelineCreateInfo {
                sType: vk::STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,       // TODO: some flags are available but none are critical
                stageCount: stages.len() as u32,
                pStages: stages.as_ptr(),
                pVertexInputState: &vertex_input_state,
                pInputAssemblyState: &input_assembly,
                pTessellationState: ptr::null(),        // FIXME:
                pViewportState: &viewport_info,
                pRasterizationState: &rasterization,
                pMultisampleState: &multisample,
                pDepthStencilState: &depth_stencil,
                pColorBlendState: &blend,
                pDynamicState: &dynamic_states,
                layout: layout.internal_object(),
                renderPass: render_pass.render_pass().internal_object(),
                subpass: render_pass.index(),
                basePipelineHandle: 0,    // TODO:
                basePipelineIndex: -1,       // TODO:
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateGraphicsPipelines(device.internal_object(), 0,
                                                         1, &infos, ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(GraphicsPipeline {
            device: device.clone(),
            pipeline: pipeline,
            layout: layout.clone(),

            dynamic_line_width: raster.line_width.is_none(),
            dynamic_viewport: viewport.dynamic_viewports(),
            dynamic_scissor: viewport.dynamic_scissors(),
            dynamic_depth_bias: raster.depth_bias.is_dynamic(),

            num_viewports: viewport.num_viewports(),

            marker: PhantomData,
        }))
    }
}

impl<MV, L> GraphicsPipeline<MV, L>
    where MV: MultiVertex, L: PipelineLayoutDesc
{
    /// Returns the pipeline layout used in the constructor.
    #[inline]
    pub fn layout(&self) -> &Arc<PipelineLayout<L>> {
        &self.layout
    }
}

impl<MultiVertex, Layout> GraphicsPipeline<MultiVertex, Layout> {
    /// Returns true if the line width used by this pipeline is dynamic.
    #[inline]
    pub fn has_dynamic_line_width(&self) -> bool {
        self.dynamic_line_width
    }
}

impl<MultiVertex, Layout> GenericPipeline for GraphicsPipeline<MultiVertex, Layout> {
}

unsafe impl<MultiVertex, Layout> VulkanObject for GraphicsPipeline<MultiVertex, Layout> {
    type Object = vk::Pipeline;

    #[inline]
    fn internal_object(&self) -> vk::Pipeline {
        self.pipeline
    }
}

impl<MultiVertex, Layout> Drop for GraphicsPipeline<MultiVertex, Layout> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyPipeline(self.device.internal_object(), self.pipeline, ptr::null());
        }
    }
}
