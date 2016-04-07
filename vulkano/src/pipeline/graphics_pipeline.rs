// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::u32;
use smallvec::SmallVec;

use device::Device;
use descriptor::PipelineLayout;
use descriptor::pipeline_layout::PipelineLayoutDesc;
use descriptor::pipeline_layout::PipelineLayoutSuperset;
use framebuffer::RenderPass;
use framebuffer::Subpass;
use Error;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

use pipeline::blend::Blend;
use pipeline::depth_stencil::Compare;
use pipeline::depth_stencil::DepthStencil;
use pipeline::depth_stencil::DepthBounds;
use pipeline::input_assembly::InputAssembly;
use pipeline::input_assembly::PrimitiveTopology;
use pipeline::multisample::Multisample;
use pipeline::raster::DepthBiasControl;
use pipeline::raster::PolygonMode;
use pipeline::raster::Rasterization;
use pipeline::shader::FragmentShaderEntryPoint;
use pipeline::shader::VertexShaderEntryPoint;
use pipeline::vertex::Definition as VertexDefinition;
use pipeline::vertex::Vertex;
use pipeline::viewport::ViewportsState;

pub struct GraphicsPipelineParams<'a, Vdef, Vsp, Vi, Vl, Fs, Fo, Fl, L, Rp> where L: 'a, Rp: 'a {
    pub vertex: Vdef,
    pub vertex_shader: VertexShaderEntryPoint<'a, Vsp, Vi, Vl>,
    pub input_assembly: InputAssembly,
    pub viewport: ViewportsState,
    pub raster: Rasterization,
    pub multisample: Multisample,
    pub fragment_shader: FragmentShaderEntryPoint<'a, Fs, Fo, Fl>,
    pub depth_stencil: DepthStencil,
    pub blend: Blend,
    pub layout: &'a Arc<L>,
    pub render_pass: Subpass<'a, Rp>,
}

///
///
/// The template parameter contains the descriptor set to use with this pipeline, and the
/// renderpass layout.
pub struct GraphicsPipeline<VertexDefinition, Layout, RenderP> {
    device: Arc<Device>,
    pipeline: vk::Pipeline,
    layout: Arc<Layout>,

    render_pass: Arc<RenderP>,
    render_pass_subpass: u32,

    vertex_definition: VertexDefinition,

    dynamic_line_width: bool,
    dynamic_viewport: bool,
    dynamic_scissor: bool,
    dynamic_depth_bias: bool,
    dynamic_depth_bounds: bool,
    dynamic_stencil_compare_mask: bool,
    dynamic_stencil_write_mask: bool,
    dynamic_stencil_reference: bool,

    num_viewports: u32,
}

impl<Vdef, L, Rp> GraphicsPipeline<Vdef, L, Rp>
    where Vdef: VertexDefinition, L: PipelineLayout, Rp: RenderPass
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
    pub fn new<'a, Vsp, Vi, Vl, Fs, Fo, Fl>
              (device: &Arc<Device>,
               params: GraphicsPipelineParams<'a, Vdef, Vsp, Vi, Vl, Fs, Fo, Fl, L, Rp>)
              -> Result<Arc<GraphicsPipeline<Vdef, L, Rp>>, GraphicsPipelineCreationError>
        where L: PipelineLayout + PipelineLayoutSuperset<Vl> + PipelineLayoutSuperset<Fl>,
              Vl: PipelineLayoutDesc, Fl: PipelineLayoutDesc
    {
        let vk = device.pointers();

        // FIXME: check
        //assert!(PipelineLayoutSuperset::is_superset_of(layout.layout(), params.vertex_shader.layout()));
        //assert!(PipelineLayoutSuperset::is_superset_of(layout.layout(), params.fragment_shader.layout()));

        // Will contain the list of dynamic states. Filled throughout this function.
        let mut dynamic_states: SmallVec<[vk::DynamicState; 8]> = SmallVec::new();

        // List of shader stages.
        let stages = {
            let mut stages = SmallVec::<[_; 5]>::new();

            stages.push(vk::PipelineShaderStageCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                stage: vk::SHADER_STAGE_VERTEX_BIT,
                module: params.vertex_shader.module().internal_object(),
                pName: params.vertex_shader.name().as_ptr(),
                pSpecializationInfo: ptr::null(),       // TODO:
            });

            stages.push(vk::PipelineShaderStageCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                stage: vk::SHADER_STAGE_FRAGMENT_BIT,
                module: params.fragment_shader.module().internal_object(),
                pName: params.fragment_shader.name().as_ptr(),
                pSpecializationInfo: ptr::null(),       // TODO:
            });

            stages
        };

        // Vertex bindings.
        let binding_descriptions = {
            let mut binding_descriptions = SmallVec::<[_; 8]>::new();
            for (num, (stride, rate)) in params.vertex.buffers().enumerate() {
                if stride > device.physical_device().limits().max_vertex_input_binding_stride() as usize {
                    return Err(GraphicsPipelineCreationError::MaxVertexInputBindingStrideExceeded {
                        binding: num,
                        max: device.physical_device().limits().max_vertex_input_binding_stride() as usize,
                        obtained: stride,
                    });
                }

                binding_descriptions.push(vk::VertexInputBindingDescription {
                    binding: num as u32,
                    stride: stride as u32,
                    inputRate: rate as u32,
                });
            }

            binding_descriptions
        };

        if binding_descriptions.len() > device.physical_device().limits()
                                              .max_vertex_input_bindings() as usize
        {
            return Err(GraphicsPipelineCreationError::MaxVertexInputBindingsExceeded {
                max: device.physical_device().limits().max_vertex_input_bindings() as usize,
                obtained: binding_descriptions.len(),
            });
        }

        // Vertex attributes.
        // TODO: check vertex attribute formats somewhere (match and support)?
        let attribute_descriptions = {
            let mut attribute_descriptions = SmallVec::<[_; 8]>::new();

            for &(loc, ref name) in params.vertex_shader.attributes().iter() {
                let (binding, info) = match params.vertex.attrib(name) {
                    Some(i) => i,
                    None => return Err(GraphicsPipelineCreationError::MissingVertexAttribute {
                        name: name.clone().into_owned()
                    })
                };

                // TODO: check attribute format support

                if info.offset > device.physical_device().limits().max_vertex_input_attribute_offset() as usize {
                    return Err(GraphicsPipelineCreationError::MaxVertexInputAttributeOffsetExceeded {
                        max: device.physical_device().limits().max_vertex_input_attribute_offset() as usize,
                        obtained: info.offset,
                    });
                }

                debug_assert!(binding < params.vertex.buffers().len());

                attribute_descriptions.push(vk::VertexInputAttributeDescription {
                    location: loc as u32,
                    binding: binding as u32,
                    format: info.format as u32,
                    offset: info.offset as u32,
                });
            }

            attribute_descriptions
        };

        if attribute_descriptions.len() > device.physical_device().limits()
                                                .max_vertex_input_attributes() as usize
        {
            return Err(GraphicsPipelineCreationError::MaxVertexInputAttributesExceeded {
                max: device.physical_device().limits().max_vertex_input_attributes() as usize,
                obtained: attribute_descriptions.len(),
            });
        }

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
            sType: vk::STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,   // reserved
            vertexBindingDescriptionCount: binding_descriptions.len() as u32,
            pVertexBindingDescriptions: binding_descriptions.as_ptr(),
            vertexAttributeDescriptionCount: attribute_descriptions.len() as u32,
            pVertexAttributeDescriptions: attribute_descriptions.as_ptr(),
        };

        if params.input_assembly.primitive_restart_enable &&
           !params.input_assembly.topology.supports_primitive_restart()
        {
            return Err(GraphicsPipelineCreationError::PrimitiveDoesntSupportPrimitiveRestart {
                primitive: params.input_assembly.topology
            });
        }

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            sType: vk::STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,   // reserved
            topology: params.input_assembly.topology as u32,
            primitiveRestartEnable: if params.input_assembly.primitive_restart_enable {
                vk::TRUE
            } else {
                vk::FALSE
            },
        };

        let (vp_vp, vp_sc, vp_num) = match params.viewport {
            ViewportsState::Fixed { ref data } => (
                data.iter().map(|e| e.0.clone().into()).collect::<SmallVec<[vk::Viewport; 4]>>(),
                data.iter().map(|e| e.1.clone().into()).collect::<SmallVec<[vk::Rect2D; 4]>>(),
                data.len() as u32
            ),
            ViewportsState::DynamicViewports { ref scissors } => {
                let num = scissors.len() as u32;
                let scissors = scissors.iter().map(|e| e.clone().into())
                                       .collect::<SmallVec<[vk::Rect2D; 4]>>();
                dynamic_states.push(vk::DYNAMIC_STATE_VIEWPORT);
                (SmallVec::new(), scissors, num)
            },
            ViewportsState::DynamicScissors { ref viewports } => {
                let num = viewports.len() as u32;
                let viewports = viewports.iter().map(|e| e.clone().into())
                                         .collect::<SmallVec<[vk::Viewport; 4]>>();
                dynamic_states.push(vk::DYNAMIC_STATE_SCISSOR);
                (viewports, SmallVec::new(), num)
            },
            ViewportsState::Dynamic { num } => {
                dynamic_states.push(vk::DYNAMIC_STATE_VIEWPORT);
                dynamic_states.push(vk::DYNAMIC_STATE_SCISSOR);
                (SmallVec::new(), SmallVec::new(), num)
            },
        };

        if vp_num > 1 && !device.enabled_features().multi_viewport {
            return Err(GraphicsPipelineCreationError::MultiViewportFeatureNotEnabled);
        }

        if vp_num > device.physical_device().limits().max_viewports() {
            return Err(GraphicsPipelineCreationError::MaxViewportsExceeded {
                obtained: vp_num,
                max: device.physical_device().limits().max_viewports(),
            });
        }

        for vp in vp_vp.iter() {
            if vp.width > device.physical_device().limits().max_viewport_dimensions()[0] as f32 ||
               vp.height > device.physical_device().limits().max_viewport_dimensions()[1] as f32
            {
                return Err(GraphicsPipelineCreationError::MaxViewportDimensionsExceeded);
            }

            if vp.x < device.physical_device().limits().viewport_bounds_range()[0] ||
               vp.x + vp.width > device.physical_device().limits().viewport_bounds_range()[1] ||
               vp.y < device.physical_device().limits().viewport_bounds_range()[0] ||
               vp.y + vp.height > device.physical_device().limits().viewport_bounds_range()[1]
            {
                return Err(GraphicsPipelineCreationError::ViewportBoundsExceeded);
            }
        }

        let viewport_info = vk::PipelineViewportStateCreateInfo {
            sType: vk::STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,   // reserved
            viewportCount: vp_num,
            pViewports: if vp_vp.is_empty() { ptr::null() } else { vp_vp.as_ptr() },    // validation layer crashes if you just pass the pointer
            scissorCount: vp_num,
            pScissors: if vp_sc.is_empty() { ptr::null() } else { vp_sc.as_ptr() },     // validation layer crashes if you just pass the pointer
        };

        if let Some(line_width) = params.raster.line_width {
            if line_width != 1.0 && !device.enabled_features().wide_lines {
                return Err(GraphicsPipelineCreationError::WideLinesFeatureNotEnabled);
            }
        } else {
            dynamic_states.push(vk::DYNAMIC_STATE_LINE_WIDTH);
        }

        let (db_enable, db_const, db_clamp, db_slope) = match params.raster.depth_bias {
            DepthBiasControl::Dynamic => {
                dynamic_states.push(vk::DYNAMIC_STATE_DEPTH_BIAS);
                (vk::TRUE, 0.0, 0.0, 0.0)
            },
            DepthBiasControl::Disabled => {
                (vk::FALSE, 0.0, 0.0, 0.0)
            },
            DepthBiasControl::Static(bias) => {
                if bias.clamp != 0.0 && !device.enabled_features().depth_bias_clamp {
                    return Err(GraphicsPipelineCreationError::DepthBiasClampFeatureNotEnabled);
                }

                (vk::TRUE, bias.constant_factor, bias.clamp, bias.slope_factor)
            },
        };

        if params.raster.depth_clamp && !device.enabled_features().depth_clamp {
            return Err(GraphicsPipelineCreationError::DepthClampFeatureNotEnabled);
        }

        if params.raster.polygon_mode != PolygonMode::Fill &&
           !device.enabled_features().fill_mode_non_solid
        {
            return Err(GraphicsPipelineCreationError::FillModeNonSolidFeatureNotEnabled);
        }

        let rasterization = vk::PipelineRasterizationStateCreateInfo {
            sType: vk::STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,   // reserved
            depthClampEnable: if params.raster.depth_clamp { vk::TRUE } else { vk::FALSE },
            rasterizerDiscardEnable: if params.raster.rasterizer_discard { vk::TRUE } else { vk::FALSE },
            polygonMode: params.raster.polygon_mode as u32,
            cullMode: params.raster.cull_mode as u32,
            frontFace: params.raster.front_face as u32,
            depthBiasEnable: db_enable,
            depthBiasConstantFactor: db_const,
            depthBiasClamp: db_clamp,
            depthBiasSlopeFactor: db_slope,
            lineWidth: params.raster.line_width.unwrap_or(1.0),
        };

        assert!(params.multisample.rasterization_samples >= 1);
        // FIXME: check that rasterization_samples is equal to what's in the renderpass
        if let Some(s) = params.multisample.sample_shading { assert!(s >= 0.0 && s <= 1.0); }
        let multisample = vk::PipelineMultisampleStateCreateInfo {
            sType: vk::STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,   // reserved
            rasterizationSamples: params.multisample.rasterization_samples,
            sampleShadingEnable: if params.multisample.sample_shading.is_some() { vk::TRUE } else { vk::FALSE },
            minSampleShading: params.multisample.sample_shading.unwrap_or(1.0),
            pSampleMask: ptr::null(),   //params.multisample.sample_mask.as_ptr(),     // FIXME:
            alphaToCoverageEnable: if params.multisample.alpha_to_coverage { vk::TRUE } else { vk::FALSE },
            alphaToOneEnable: if params.multisample.alpha_to_one { vk::TRUE } else { vk::FALSE },
        };

        let depth_stencil = {
            let db = match params.depth_stencil.depth_bounds_test {
                DepthBounds::Disabled => (vk::FALSE, 0.0, 0.0),
                DepthBounds::Fixed(ref range) => {
                    if !device.enabled_features().depth_bounds {
                        return Err(GraphicsPipelineCreationError::DepthBoundsFeatureNotEnabled);
                    }

                    (vk::TRUE, range.start, range.end)
                },
                DepthBounds::Dynamic => {
                    if !device.enabled_features().depth_bounds {
                        return Err(GraphicsPipelineCreationError::DepthBoundsFeatureNotEnabled);
                    }

                    dynamic_states.push(vk::DYNAMIC_STATE_DEPTH_BOUNDS);

                    (vk::TRUE, 0.0, 1.0)
                },
            };

            match (params.depth_stencil.stencil_front.compare_mask,
                   params.depth_stencil.stencil_back.compare_mask)
            {
                (Some(_), Some(_)) => (),
                (None, None) => {
                    dynamic_states.push(vk::DYNAMIC_STATE_STENCIL_COMPARE_MASK);
                },
                _ => return Err(GraphicsPipelineCreationError::WrongStencilState)
            };

            match (params.depth_stencil.stencil_front.write_mask,
                   params.depth_stencil.stencil_back.write_mask)
            {
                (Some(_), Some(_)) => (),
                (None, None) => {
                    dynamic_states.push(vk::DYNAMIC_STATE_STENCIL_WRITE_MASK);
                },
                _ => return Err(GraphicsPipelineCreationError::WrongStencilState)
            };

            match (params.depth_stencil.stencil_front.reference,
                   params.depth_stencil.stencil_back.reference)
            {
                (Some(_), Some(_)) => (),
                (None, None) => {
                    dynamic_states.push(vk::DYNAMIC_STATE_STENCIL_REFERENCE);
                },
                _ => return Err(GraphicsPipelineCreationError::WrongStencilState)
            };

            vk::PipelineDepthStencilStateCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                depthTestEnable: if !params.depth_stencil.depth_write &&
                                    params.depth_stencil.depth_compare == Compare::Always
                                 { vk::FALSE } else { vk::TRUE },
                depthWriteEnable: if params.depth_stencil.depth_write { vk::TRUE }
                                  else { vk::FALSE },
                depthCompareOp: params.depth_stencil.depth_compare as u32,
                depthBoundsTestEnable: db.0,
                stencilTestEnable: vk::FALSE,            // FIXME:
                front: vk::StencilOpState {
                    failOp: params.depth_stencil.stencil_front.fail_op as u32,
                    passOp: params.depth_stencil.stencil_front.pass_op as u32,
                    depthFailOp: params.depth_stencil.stencil_front.depth_fail_op as u32,
                    compareOp: params.depth_stencil.stencil_front.compare as u32,
                    compareMask: params.depth_stencil.stencil_front.compare_mask.unwrap_or(u32::MAX),
                    writeMask: params.depth_stencil.stencil_front.write_mask.unwrap_or(u32::MAX),
                    reference: params.depth_stencil.stencil_front.reference.unwrap_or(0),
                },
                back: vk::StencilOpState {
                    failOp: params.depth_stencil.stencil_back.fail_op as u32,
                    passOp: params.depth_stencil.stencil_back.pass_op as u32,
                    depthFailOp: params.depth_stencil.stencil_back.depth_fail_op as u32,
                    compareOp: params.depth_stencil.stencil_back.compare as u32,
                    compareMask: params.depth_stencil.stencil_back.compare_mask.unwrap_or(u32::MAX),
                    writeMask: params.depth_stencil.stencil_back.write_mask.unwrap_or(u32::MAX),
                    reference: params.depth_stencil.stencil_back.reference.unwrap_or(0)
                },
                minDepthBounds: db.1,
                maxDepthBounds: db.2,
            }
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
            logicOpEnable: if params.blend.logic_op.is_some() { vk::TRUE } else { vk::FALSE },
            logicOp: params.blend.logic_op.unwrap_or(Default::default()) as u32,
            attachmentCount: 1,         // FIXME:
            pAttachments: &atch,      // FIXME:
            blendConstants: params.blend.blend_constants.unwrap_or([0.0, 0.0, 0.0, 0.0]),
        };

        let dynamic_states = vk::PipelineDynamicStateCreateInfo {
            sType: vk::STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,   // reserved
            dynamicStateCount: dynamic_states.len() as u32,
            pDynamicStates: dynamic_states.as_ptr(),
        };

        let pipeline = unsafe {
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
                layout: PipelineLayout::inner_pipeline_layout(&**params.layout).internal_object(),
                renderPass: params.render_pass.render_pass().render_pass().internal_object(),
                subpass: params.render_pass.index(),
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
            layout: params.layout.clone(),

            vertex_definition: params.vertex,

            render_pass: params.render_pass.render_pass().clone(),
            render_pass_subpass: params.render_pass.index(),

            dynamic_line_width: params.raster.line_width.is_none(),
            dynamic_viewport: params.viewport.dynamic_viewports(),
            dynamic_scissor: params.viewport.dynamic_scissors(),
            dynamic_depth_bias: params.raster.depth_bias.is_dynamic(),
            dynamic_depth_bounds: params.depth_stencil.depth_bounds_test.is_dynamic(),
            dynamic_stencil_compare_mask: params.depth_stencil.stencil_back.compare_mask.is_none(),
            dynamic_stencil_write_mask: params.depth_stencil.stencil_back.write_mask.is_none(),
            dynamic_stencil_reference: params.depth_stencil.stencil_back.reference.is_none(),

            num_viewports: params.viewport.num_viewports(),
        }))
    }
}

impl<Mv, L, Rp> GraphicsPipeline<Mv, L, Rp>
    where Mv: VertexDefinition
{
    /// Returns the vertex definition used in the constructor.
    #[inline]
    pub fn vertex_definition(&self) -> &Mv {
        &self.vertex_definition
    }
}

impl<Mv, L, Rp> GraphicsPipeline<Mv, L, Rp>
    where L: PipelineLayout
{
    /// Returns the pipeline layout used in the constructor.
    #[inline]
    pub fn layout(&self) -> &Arc<L> {
        &self.layout
    }
}

impl<Mv, L, Rp> GraphicsPipeline<Mv, L, Rp>
    where Rp: RenderPass
{
    /// Returns the pass used in the constructor.
    #[inline]
    pub fn subpass(&self) -> Subpass<Rp> {
        Subpass::from(&self.render_pass, self.render_pass_subpass).unwrap()
    }
}

impl<Mv, L, Rp> GraphicsPipeline<Mv, L, Rp> {
    /// Returns true if the line width used by this pipeline is dynamic.
    #[inline]
    pub fn has_dynamic_line_width(&self) -> bool {
        self.dynamic_line_width
    }

    /// Returns the number of viewports and scissors of this pipeline.
    #[inline]
    pub fn num_viewports(&self) -> u32 {
        self.num_viewports
    }

    /// Returns true if the viewports used by this pipeline are dynamic.
    #[inline]
    pub fn has_dynamic_viewports(&self) -> bool {
        self.dynamic_viewport
    }

    /// Returns true if the scissors used by this pipeline are dynamic.
    #[inline]
    pub fn has_dynamic_scissors(&self) -> bool {
        self.dynamic_scissor
    }
}

unsafe impl<Mv, L, Rp> VulkanObject for GraphicsPipeline<Mv, L, Rp> {
    type Object = vk::Pipeline;

    #[inline]
    fn internal_object(&self) -> vk::Pipeline {
        self.pipeline
    }
}

impl<Mv, L, Rp> Drop for GraphicsPipeline<Mv, L, Rp> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyPipeline(self.device.internal_object(), self.pipeline, ptr::null());
        }
    }
}

/// Error that can happen when creating a graphics pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphicsPipelineCreationError {
    /// Not enough memory.
    OomError(OomError),

    /// The pipeline layout is not compatible with what the shaders expect.
    IncompatiblePipelineLayout,

    MissingVertexAttribute { name: String },

    MaxVertexInputBindingStrideExceeded { binding: usize, max: usize, obtained: usize },

    MaxVertexInputBindingsExceeded { max: usize, obtained: usize },

    MaxVertexInputAttributeOffsetExceeded { max: usize, obtained: usize },

    MaxVertexInputAttributesExceeded { max: usize, obtained: usize },

    PrimitiveDoesntSupportPrimitiveRestart { primitive: PrimitiveTopology },

    MultiViewportFeatureNotEnabled,

    MaxViewportsExceeded { max: u32, obtained: u32 },

    MaxViewportDimensionsExceeded,

    ViewportBoundsExceeded,

    WideLinesFeatureNotEnabled,

    DepthClampFeatureNotEnabled,

    DepthBiasClampFeatureNotEnabled,

    FillModeNonSolidFeatureNotEnabled,

    DepthBoundsFeatureNotEnabled,

    WrongStencilState,
}

impl error::Error for GraphicsPipelineCreationError {
    #[inline]
    // TODO: finish
    fn description(&self) -> &str {
        match *self {
            GraphicsPipelineCreationError::OomError(_) => "not enough memory available",
            GraphicsPipelineCreationError::IncompatiblePipelineLayout => "the pipeline layout is \
                                                                          not compatible with what \
                                                                          the shaders expect",
            GraphicsPipelineCreationError::MissingVertexAttribute { .. } => "",
            GraphicsPipelineCreationError::MaxVertexInputBindingStrideExceeded { .. } => "",
            GraphicsPipelineCreationError::MaxVertexInputBindingsExceeded { .. } => "",
            GraphicsPipelineCreationError::MaxVertexInputAttributeOffsetExceeded { .. } => "",
            GraphicsPipelineCreationError::MaxVertexInputAttributesExceeded { .. } => "",
            GraphicsPipelineCreationError::PrimitiveDoesntSupportPrimitiveRestart { .. } => "",
            GraphicsPipelineCreationError::MultiViewportFeatureNotEnabled => "",
            GraphicsPipelineCreationError::MaxViewportsExceeded { .. } => "",
            GraphicsPipelineCreationError::MaxViewportDimensionsExceeded => "",
            GraphicsPipelineCreationError::ViewportBoundsExceeded => "",
            GraphicsPipelineCreationError::WideLinesFeatureNotEnabled => "",
            GraphicsPipelineCreationError::DepthClampFeatureNotEnabled => "",
            GraphicsPipelineCreationError::DepthBiasClampFeatureNotEnabled => "",
            GraphicsPipelineCreationError::FillModeNonSolidFeatureNotEnabled => "",
            GraphicsPipelineCreationError::DepthBoundsFeatureNotEnabled => "",
            GraphicsPipelineCreationError::WrongStencilState => "",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            GraphicsPipelineCreationError::OomError(ref err) => Some(err),
            _ => None
        }
    }
}

impl fmt::Display for GraphicsPipelineCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<OomError> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: OomError) -> GraphicsPipelineCreationError {
        GraphicsPipelineCreationError::OomError(err)
    }
}

impl From<Error> for GraphicsPipelineCreationError {
    #[inline]
    fn from(err: Error) -> GraphicsPipelineCreationError {
        match err {
            err @ Error::OutOfHostMemory => {
                GraphicsPipelineCreationError::OomError(OomError::from(err))
            },
            err @ Error::OutOfDeviceMemory => {
                GraphicsPipelineCreationError::OomError(OomError::from(err))
            },
            _ => panic!("unexpected error: {:?}", err)
        }
    }
}
