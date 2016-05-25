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
use descriptor::pipeline_layout::EmptyPipeline;
use framebuffer::RenderPass;
use framebuffer::RenderPassDesc;
use framebuffer::RenderPassSubpassInterface;
use framebuffer::Subpass;
use Error;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

use pipeline::blend::Blend;
use pipeline::blend::AttachmentsBlend;
use pipeline::depth_stencil::Compare;
use pipeline::depth_stencil::DepthStencil;
use pipeline::depth_stencil::DepthBounds;
use pipeline::input_assembly::InputAssembly;
use pipeline::input_assembly::PrimitiveTopology;
use pipeline::multisample::Multisample;
use pipeline::raster::DepthBiasControl;
use pipeline::raster::PolygonMode;
use pipeline::raster::Rasterization;
use pipeline::shader::ShaderInterfaceDef;
use pipeline::shader::ShaderInterfaceDefMatch;
use pipeline::shader::VertexShaderEntryPoint;
use pipeline::shader::TessControlShaderEntryPoint;
use pipeline::shader::TessEvaluationShaderEntryPoint;
use pipeline::shader::GeometryShaderEntryPoint;
use pipeline::shader::FragmentShaderEntryPoint;
use pipeline::vertex::Definition as VertexDefinition;
use pipeline::viewport::ViewportsState;

pub struct GraphicsPipelineParams<'a, Vdef, Vsp, Vi, Vo, Vl, Tcs, Tci, Tco, Tcl, Tes, Tei, Teo,
                                  Tel, Gs, Gi, Go, Gl, Fs, Fi, Fo, Fl, L, Rp>
    where L: 'a, Rp: 'a
{
    pub vertex_input: Vdef,
    pub vertex_shader: VertexShaderEntryPoint<'a, Vsp, Vi, Vo, Vl>,
    pub input_assembly: InputAssembly,
    pub tessellation: Option<GraphicsPipelineParamsTess<'a, Tcs, Tci, Tco, Tcl, Tes, Tei, Teo, Tel>>,
    pub geometry_shader: Option<GeometryShaderEntryPoint<'a, Gs, Gi, Go, Gl>>,
    pub viewport: ViewportsState,
    pub raster: Rasterization,
    pub multisample: Multisample,
    pub fragment_shader: FragmentShaderEntryPoint<'a, Fs, Fi, Fo, Fl>,
    pub depth_stencil: DepthStencil,
    pub blend: Blend,
    pub layout: &'a Arc<L>,
    pub render_pass: Subpass<'a, Rp>,
}

pub struct GraphicsPipelineParamsTess<'a, Tcs, Tci, Tco, Tcl, Tes, Tei, Teo, Tel> {
    pub tessellation_control_shader: TessControlShaderEntryPoint<'a, Tcs, Tci, Tco, Tcl>,
    pub tessellation_evaluation_shader: TessEvaluationShaderEntryPoint<'a, Tes, Tei, Teo, Tel>,
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
    dynamic_blend_constants: bool,

    num_viewports: u32,
}

impl<Vdef, L, Rp> GraphicsPipeline<Vdef, L, Rp>
    where L: PipelineLayout, Rp: RenderPass + RenderPassDesc
{
    /// Builds a new graphics pipeline object.
    #[inline]
    pub fn new<'a, Vsp, Vi, Vo, Vl, Fs, Fi, Fo, Fl>
              (device: &Arc<Device>,
               params: GraphicsPipelineParams<'a, Vdef, Vsp, Vi, Vo, Vl, (), (), (), EmptyPipeline,
                                              (), (), (), EmptyPipeline, (), (), (), EmptyPipeline,
                                              Fs, Fi, Fo, Fl, L, Rp>)
              -> Result<Arc<GraphicsPipeline<Vdef, L, Rp>>, GraphicsPipelineCreationError>
        where Vdef: VertexDefinition<Vi>,
              L: PipelineLayout + PipelineLayoutSuperset<Vl> + PipelineLayoutSuperset<Fl>,
              Vl: PipelineLayoutDesc, Fl: PipelineLayoutDesc,
              Fi: ShaderInterfaceDefMatch<Vo>,
              Fo: ShaderInterfaceDef,
              Vo: ShaderInterfaceDef,
              Rp: RenderPassSubpassInterface<Fo>,
    {
        // TODO: return proper errors
        assert!(params.fragment_shader.input().matches(params.vertex_shader.output()));
        GraphicsPipeline::new_inner::<_, _, _, _, (), (), (), EmptyPipeline, (), (), (),
                                      EmptyPipeline, (), (), (), EmptyPipeline, _, _, _, _>
                                      (device, params)
    }

    /// Builds a new graphics pipeline object with a geometry shader.
    #[inline]
    pub fn with_geometry_shader<'a, Vsp, Vi, Vo, Vl, Gsp, Gi, Go, Gl, Fs, Fi, Fo, Fl>
              (device: &Arc<Device>,
               params: GraphicsPipelineParams<'a, Vdef, Vsp, Vi, Vo, Vl, (), (), (), EmptyPipeline,
                                              (), (), (), EmptyPipeline, Gsp, Gi, Go, Gl, Fs, Fi,
                                              Fo, Fl, L, Rp>)
              -> Result<Arc<GraphicsPipeline<Vdef, L, Rp>>, GraphicsPipelineCreationError>
        where Vdef: VertexDefinition<Vi>,
              L: PipelineLayout + PipelineLayoutSuperset<Vl> + PipelineLayoutSuperset<Fl> +
                 PipelineLayoutSuperset<Gl>,
              Vl: PipelineLayoutDesc,
              Fl: PipelineLayoutDesc,
              Gl: PipelineLayoutDesc,
              Gi: ShaderInterfaceDefMatch<Vo>,
              Vo: ShaderInterfaceDef,
              Fi: ShaderInterfaceDefMatch<Go> + ShaderInterfaceDefMatch<Vo>,
              Fo: ShaderInterfaceDef,
              Go: ShaderInterfaceDef,
              Rp: RenderPassSubpassInterface<Fo>,
    {
        // TODO: return proper errors
        if let Some(ref geometry_shader) = params.geometry_shader {
            assert!(geometry_shader.input().matches(params.vertex_shader.output()));
            assert!(params.fragment_shader.input().matches(geometry_shader.output()));
        } else {
            assert!(params.fragment_shader.input().matches(params.vertex_shader.output()));
        }

        GraphicsPipeline::new_inner(device, params)
    }

    /// Builds a new graphics pipeline object with tessellation shaders.
    #[inline]
    pub fn with_tessellation<'a, Vsp, Vi, Vo, Vl, Tcs, Tci, Tco, Tcl, Tes, Tei, Teo, Tel, Fs, Fi,
                            Fo, Fl>
              (device: &Arc<Device>,
               params: GraphicsPipelineParams<'a, Vdef, Vsp, Vi, Vo, Vl, Tcs, Tci, Tco, Tcl, Tes,
                                              Tei, Teo, Tel, (), (), (), EmptyPipeline, Fs, Fi,
                                              Fo, Fl, L, Rp>)
              -> Result<Arc<GraphicsPipeline<Vdef, L, Rp>>, GraphicsPipelineCreationError>
        where Vdef: VertexDefinition<Vi>,
              L: PipelineLayout + PipelineLayoutSuperset<Vl> + PipelineLayoutSuperset<Fl> +
                 PipelineLayoutSuperset<Tcl> + PipelineLayoutSuperset<Tel>,
              Vl: PipelineLayoutDesc,
              Fl: PipelineLayoutDesc,
              Tcl: PipelineLayoutDesc,
              Tel: PipelineLayoutDesc,
              Tci: ShaderInterfaceDefMatch<Vo>,
              Tei: ShaderInterfaceDefMatch<Tco>,
              Vo: ShaderInterfaceDef,
              Tco: ShaderInterfaceDef,
              Teo: ShaderInterfaceDef,
              Fi: ShaderInterfaceDefMatch<Teo> + ShaderInterfaceDefMatch<Vo>,
              Fo: ShaderInterfaceDef,
              Rp: RenderPassSubpassInterface<Fo>,
    {
        // TODO: return proper errors
        if let Some(ref tess) = params.tessellation {
            assert!(tess.tessellation_control_shader.input().matches(params.vertex_shader.output()));
            assert!(tess.tessellation_evaluation_shader.input().matches(tess.tessellation_control_shader.output()));
            assert!(params.fragment_shader.input().matches(tess.tessellation_evaluation_shader.output()));
        } else {
            assert!(params.fragment_shader.input().matches(params.vertex_shader.output()));
        }

        GraphicsPipeline::new_inner(device, params)
    }

    fn new_inner<'a, Vsp, Vi, Vo, Vl, Tcs, Tci, Tco, Tcl, Tes, Tei, Teo, Tel, Gsp, Gi, Go, Gl, Fs,
                 Fi, Fo, Fl>
                (device: &Arc<Device>,
                 params: GraphicsPipelineParams<'a, Vdef, Vsp, Vi, Vo, Vl, Tcs, Tci, Tco, Tcl, Tes,
                                                Tei, Teo, Tel, Gsp, Gi, Go, Gl, Fs, Fi, Fo, Fl, L,
                                                Rp>)
                 -> Result<Arc<GraphicsPipeline<Vdef, L, Rp>>, GraphicsPipelineCreationError>
        where Vdef: VertexDefinition<Vi>,
              Fo: ShaderInterfaceDef,
              L: PipelineLayout + PipelineLayoutSuperset<Vl> + PipelineLayoutSuperset<Fl> +
                 PipelineLayoutSuperset<Tcl> + PipelineLayoutSuperset<Tel> +
                 PipelineLayoutSuperset<Gl>,
              Vl: PipelineLayoutDesc,
              Fl: PipelineLayoutDesc,
              Gl: PipelineLayoutDesc,
              Tcl: PipelineLayoutDesc,
              Tel: PipelineLayoutDesc,
              Rp: RenderPassSubpassInterface<Fo>,
    {
        let vk = device.pointers();

        // Checking that the pipeline layout matches the shader stages.
        // TODO: more details in the errors
        if !PipelineLayoutSuperset::is_superset_of(&**params.layout,
                                                   params.vertex_shader.layout())
        {
            return Err(GraphicsPipelineCreationError::IncompatiblePipelineLayout);
        }
        if !PipelineLayoutSuperset::is_superset_of(&**params.layout,
                                                   params.fragment_shader.layout())
        {
            return Err(GraphicsPipelineCreationError::IncompatiblePipelineLayout);
        }
        if let Some(ref geometry_shader) = params.geometry_shader {
            if !PipelineLayoutSuperset::is_superset_of(&**params.layout,
                                                       geometry_shader.layout())
            {
                return Err(GraphicsPipelineCreationError::IncompatiblePipelineLayout);
            }
        }
        if let Some(ref tess) = params.tessellation {
            if !PipelineLayoutSuperset::is_superset_of(&**params.layout,
                                                       tess.tessellation_control_shader.layout())
            {
                return Err(GraphicsPipelineCreationError::IncompatiblePipelineLayout);
            }
            if !PipelineLayoutSuperset::is_superset_of(&**params.layout,
                                                       tess.tessellation_evaluation_shader.layout())
            {
                return Err(GraphicsPipelineCreationError::IncompatiblePipelineLayout);
            }
        }

        // Check that the subpass can accept the output of the fragment shader.
        if !params.render_pass.render_pass().is_compatible_with(params.render_pass.index(),
                                                                params.fragment_shader.output())
        {
            return Err(GraphicsPipelineCreationError::FragmentShaderRenderPassIncompatible);
        }

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

            if let Some(ref gs) = params.geometry_shader {
                if !device.enabled_features().geometry_shader {
                    return Err(GraphicsPipelineCreationError::GeometryShaderFeatureNotEnabled);
                }

                stages.push(vk::PipelineShaderStageCreateInfo {
                    sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    pNext: ptr::null(),
                    flags: 0,   // reserved
                    stage: vk::SHADER_STAGE_GEOMETRY_BIT,
                    module: gs.module().internal_object(),
                    pName: gs.name().as_ptr(),
                    pSpecializationInfo: ptr::null(),       // TODO:
                });
            }

            if let Some(ref tess) = params.tessellation {
                // FIXME: must check that the control shader and evaluation shader are compatible

                if !device.enabled_features().tessellation_shader {
                    return Err(GraphicsPipelineCreationError::TessellationShaderFeatureNotEnabled);
                }

                stages.push(vk::PipelineShaderStageCreateInfo {
                    sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    pNext: ptr::null(),
                    flags: 0,   // reserved
                    stage: vk::SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                    module: tess.tessellation_control_shader.module().internal_object(),
                    pName: tess.tessellation_control_shader.name().as_ptr(),
                    pSpecializationInfo: ptr::null(),       // TODO:
                });

                stages.push(vk::PipelineShaderStageCreateInfo {
                    sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    pNext: ptr::null(),
                    flags: 0,   // reserved
                    stage: vk::SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                    module: tess.tessellation_evaluation_shader.module().internal_object(),
                    pName: tess.tessellation_evaluation_shader.name().as_ptr(),
                    pSpecializationInfo: ptr::null(),       // TODO:
                });
            }

            stages
        };

        // Vertex bindings.
        let (binding_descriptions, attribute_descriptions) = {
            let (buffers_iter, attribs_iter) = params.vertex_input.definition(params.vertex_shader.input_definition());

            let mut binding_descriptions = SmallVec::<[_; 8]>::new();
            for (num, stride, rate) in buffers_iter {
                if stride > device.physical_device().limits().max_vertex_input_binding_stride() as usize {
                    return Err(GraphicsPipelineCreationError::MaxVertexInputBindingStrideExceeded {
                        binding: num as usize,
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

            let mut attribute_descriptions = SmallVec::<[_; 8]>::new();
            for (loc, binding, info) in attribs_iter {
                // TODO: check attribute format support

                if info.offset > device.physical_device().limits().max_vertex_input_attribute_offset() as usize {
                    return Err(GraphicsPipelineCreationError::MaxVertexInputAttributeOffsetExceeded {
                        max: device.physical_device().limits().max_vertex_input_attribute_offset() as usize,
                        obtained: info.offset,
                    });
                }

                debug_assert!(binding_descriptions.iter().find(|b| b.binding == binding).is_some());

                attribute_descriptions.push(vk::VertexInputAttributeDescription {
                    location: loc as u32,
                    binding: binding as u32,
                    format: info.format as u32,
                    offset: info.offset as u32,
                });
            }

            (binding_descriptions, attribute_descriptions)
        };

        if binding_descriptions.len() > device.physical_device().limits()
                                              .max_vertex_input_bindings() as usize
        {
            return Err(GraphicsPipelineCreationError::MaxVertexInputBindingsExceeded {
                max: device.physical_device().limits().max_vertex_input_bindings() as usize,
                obtained: binding_descriptions.len(),
            });
        }

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

        // TODO: should check from the tess eval shader instead of the input assembly
        if let Some(ref gs) = params.geometry_shader {
            if !gs.primitives().matches(params.input_assembly.topology) {
                return Err(GraphicsPipelineCreationError::TopologyNotMatchingGeometryShader);
            }
        }

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            sType: vk::STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,   // reserved
            topology: params.input_assembly.topology.into(),
            primitiveRestartEnable: if params.input_assembly.primitive_restart_enable {
                vk::TRUE
            } else {
                vk::FALSE
            },
        };

        let tessellation = match params.input_assembly.topology {
            PrimitiveTopology::PatchList { vertices_per_patch } => {
                if params.tessellation.is_none() {
                    return Err(GraphicsPipelineCreationError::InvalidPrimitiveTopology);
                }
                if vertices_per_patch > device.physical_device().limits()
                                              .max_tessellation_patch_size()
                {
                    return Err(GraphicsPipelineCreationError::MaxTessellationPatchSizeExceeded);
                }

                Some(vk::PipelineTessellationStateCreateInfo {
                    sType: vk::STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
                    pNext: ptr::null(),
                    flags: 0,   // reserved,
                    patchControlPoints: vertices_per_patch,
                })
            },
            _ => {
                if params.tessellation.is_some() {
                    return Err(GraphicsPipelineCreationError::InvalidPrimitiveTopology);
                }

                None
            }
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

            if params.depth_stencil.depth_write && !params.render_pass.has_writable_depth() {
                return Err(GraphicsPipelineCreationError::NoDepthAttachment);
            }

            if params.depth_stencil.depth_compare != Compare::Always &&
               !params.render_pass.has_depth()
            {
                return Err(GraphicsPipelineCreationError::NoDepthAttachment);
            }

            if (!params.depth_stencil.stencil_front.always_keep() ||
                !params.depth_stencil.stencil_back.always_keep()) &&
                !params.render_pass.has_stencil()
            {
                return Err(GraphicsPipelineCreationError::NoStencilAttachment);
            }

            // FIXME: stencil writability

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
                stencilTestEnable: if params.depth_stencil.stencil_front.always_keep() &&
                                      params.depth_stencil.stencil_back.always_keep()
                                      { vk::FALSE } else { vk::TRUE },
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

        let blend_atch: SmallVec<[vk::PipelineColorBlendAttachmentState; 8]> = {
            let num_atch = params.render_pass.num_color_attachments();

            match params.blend.attachments {
                AttachmentsBlend::Collective(blend) => {
                    (0 .. num_atch).map(|_| blend.clone().into()).collect()
                },
                AttachmentsBlend::Individual(blend) => {
                    if blend.len() != num_atch as usize {
                        return Err(GraphicsPipelineCreationError::MismatchBlendingAttachmentsCount);
                    }

                    if !device.enabled_features().independent_blend {
                        return Err(GraphicsPipelineCreationError::IndependentBlendFeatureNotEnabled);
                    }

                    blend.iter().map(|b| b.clone().into()).collect()
                },
            }
        };

        let blend = vk::PipelineColorBlendStateCreateInfo {
            sType: vk::STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,   // reserved
            logicOpEnable: if params.blend.logic_op.is_some() {
                if !device.enabled_features().logic_op {
                    return Err(GraphicsPipelineCreationError::LogicOpFeatureNotEnabled);
                }
                vk::TRUE
            } else {
                vk::FALSE
            },
            logicOp: params.blend.logic_op.unwrap_or(Default::default()) as u32,
            attachmentCount: blend_atch.len() as u32,
            pAttachments: blend_atch.as_ptr(),
            blendConstants: if let Some(c) = params.blend.blend_constants {
                c
            } else {
                dynamic_states.push(vk::DYNAMIC_STATE_BLEND_CONSTANTS);
                [0.0, 0.0, 0.0, 0.0]
            }
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
                pTessellationState: tessellation.as_ref().map(|t| t as *const _)
                                                .unwrap_or(ptr::null()),
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

            vertex_definition: params.vertex_input,

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
            dynamic_blend_constants: params.blend.blend_constants.is_none(),

            num_viewports: params.viewport.num_viewports(),
        }))
    }
}

impl<Mv, L, Rp> GraphicsPipeline<Mv, L, Rp> {
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
    where Rp: RenderPass + RenderPassDesc
{
    /// Returns the render pass used in the constructor.
    #[inline]
    pub fn render_pass(&self) -> &Arc<Rp> {
        &self.render_pass
    }

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

    /// Returns true if the depth bounds used by this pipeline are dynamic.
    #[inline]
    pub fn has_dynamic_depth_bounds(&self) -> bool {
        self.dynamic_depth_bounds
    }

    /// Returns true if the stencil compare masks used by this pipeline are dynamic.
    #[inline]
    pub fn has_dynamic_stencil_compare_mask(&self) -> bool {
        self.dynamic_stencil_compare_mask
    }

    /// Returns true if the stencil write masks used by this pipeline are dynamic.
    #[inline]
    pub fn has_dynamic_stencil_write_mask(&self) -> bool {
        self.dynamic_stencil_write_mask
    }

    /// Returns true if the stencil references used by this pipeline are dynamic.
    #[inline]
    pub fn has_dynamic_stencil_reference(&self) -> bool {
        self.dynamic_stencil_reference
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

    /// The output of the fragment shader is not compatible with what the render pass subpass
    /// expects.
    FragmentShaderRenderPassIncompatible,

    /// One of the vertex attributes requested by the vertex shader is missing from the vertex
    /// input.
    MissingVertexAttribute {
        /// Name of the attribute.
        name: String
    },

    /// The maximum stride value for vertex input (ie. the distance between two vertex elements)
    /// has been exceeded.
    MaxVertexInputBindingStrideExceeded {
        /// Index of the faulty binding.
        binding: usize,
        /// Maximum allowed value.
        max: usize,
        /// Value that was passed.
        obtained: usize,
    },

    /// The maximum number of vertex sources has been exceeded.
    MaxVertexInputBindingsExceeded {
        /// Maximum allowed value.
        max: usize,
        /// Value that was passed.
        obtained: usize,
    },

    /// The maximum offset for a vertex attribute has been exceeded. This means that your vertex
    /// struct is too large.
    MaxVertexInputAttributeOffsetExceeded {
        /// Maximum allowed value.
        max: usize,
        /// Value that was passed.
        obtained: usize,
    },

    /// The maximum number of vertex attributes has been exceeded.
    MaxVertexInputAttributesExceeded {
        /// Maximum allowed value.
        max: usize,
        /// Value that was passed.
        obtained: usize,
    },

    /// The user requested to use primitive restart, but the primitive topology doesn't support it.
    PrimitiveDoesntSupportPrimitiveRestart {
        /// The topology that doesn't support primitive restart.
        primitive: PrimitiveTopology
    },

    /// The `multi_viewport` feature must be enabled in order to use multiple viewports at once.
    MultiViewportFeatureNotEnabled,

    /// The maximum number of viewports has been exceeded.
    MaxViewportsExceeded {
        /// Maximum allowed value.
        max: u32,
        /// Value that was passed.
        obtained: u32
    },

    /// The maximum dimensions of viewports has been exceeded.
    MaxViewportDimensionsExceeded,

    /// The minimum or maximum bounds of viewports have been exceeded.
    ViewportBoundsExceeded,

    /// The `wide_lines` feature must be enabled in order to use a line width superior to 1.0.
    WideLinesFeatureNotEnabled,

    /// The `depth_clamp` feature must be enabled in order to use depth clamping.
    DepthClampFeatureNotEnabled,

    /// The `depth_bias_clamp` feature must be enabled in order to use a depth bias clamp different
    /// from 0.0.
    DepthBiasClampFeatureNotEnabled,

    /// The `fill_mode_non_solid` feature must be enabled in order to use a polygon mode different
    /// from `Fill`.
    FillModeNonSolidFeatureNotEnabled,

    /// The `depth_bounds` feature must be enabled in order to use depth bounds testing.
    DepthBoundsFeatureNotEnabled,

    /// The requested stencil test is invalid.
    WrongStencilState,

    /// The primitives topology does not match what the geometry shader expects.
    TopologyNotMatchingGeometryShader,

    /// The `geometry_shader` feature must be enabled in order to use geometry shaders.
    GeometryShaderFeatureNotEnabled,

    /// The `tessellation_shader` feature must be enabled in order to use tessellation shaders.
    TessellationShaderFeatureNotEnabled,

    /// The number of attachments specified in the blending does not match the number of
    /// attachments in the subpass.
    MismatchBlendingAttachmentsCount,

    /// The `independent_blend` feature must be enabled in order to use different blending
    /// operations per attachment.
    IndependentBlendFeatureNotEnabled,

    /// The `logic_op` feature must be enabled in order to use logic operations.
    LogicOpFeatureNotEnabled,

    /// The depth test requires a depth attachment but render pass has no depth attachment, or
    /// depth writing is enabled and the depth attachment is read-only.
    NoDepthAttachment,

    /// The stencil test requires a stencil attachment but render pass has no stencil attachment, or
    /// stencil writing is enabled and the stencil attachment is read-only.
    NoStencilAttachment,

    /// Tried to use a patch list without a tessellation shader, or a non-patch-list with a
    /// tessellation shader.
    InvalidPrimitiveTopology,

    /// The `maxTessellationPatchSize` limit was exceeded.
    MaxTessellationPatchSizeExceeded,
}

impl error::Error for GraphicsPipelineCreationError {
    #[inline]
    // TODO: finish
    fn description(&self) -> &str {
        match *self {
            GraphicsPipelineCreationError::OomError(_) => "not enough memory available",
            GraphicsPipelineCreationError::IncompatiblePipelineLayout => {
                "the pipeline layout is not compatible with what the shaders expect"
            },
            GraphicsPipelineCreationError::FragmentShaderRenderPassIncompatible => {
                "the output of the fragment shader is not compatible with what the render pass \
                 subpass expects"
            },
            GraphicsPipelineCreationError::MissingVertexAttribute { .. } => {
                "one of the vertex attributes requested by the vertex shader is missing from the \
                 vertex input"
            },
            GraphicsPipelineCreationError::MaxVertexInputBindingStrideExceeded { .. } => {
                "the maximum stride value for vertex input (ie. the distance between two vertex \
                 elements) has been exceeded"
            },
            GraphicsPipelineCreationError::MaxVertexInputBindingsExceeded { .. } => {
                "the maximum number of vertex sources has been exceeded"
            },
            GraphicsPipelineCreationError::MaxVertexInputAttributeOffsetExceeded { .. } => {
                "the maximum offset for a vertex attribute has been exceeded"
            },
            GraphicsPipelineCreationError::MaxVertexInputAttributesExceeded { .. } => {
                "the maximum number of vertex attributes has been exceeded"
            },
            GraphicsPipelineCreationError::PrimitiveDoesntSupportPrimitiveRestart { .. } => {
                "the user requested to use primitive restart, but the primitive topology \
                 doesn't support it"
            },
            GraphicsPipelineCreationError::MultiViewportFeatureNotEnabled => {
                "the `multi_viewport` feature must be enabled in order to use multiple viewports \
                 at once"
            },
            GraphicsPipelineCreationError::MaxViewportsExceeded { .. } => {
                "the maximum number of viewports has been exceeded"
            },
            GraphicsPipelineCreationError::MaxViewportDimensionsExceeded => {
                "the maximum dimensions of viewports has been exceeded"
            },
            GraphicsPipelineCreationError::ViewportBoundsExceeded => {
                "the minimum or maximum bounds of viewports have been exceeded"
            },
            GraphicsPipelineCreationError::WideLinesFeatureNotEnabled => {
                "the `wide_lines` feature must be enabled in order to use a line width \
                 superior to 1.0"
            },
            GraphicsPipelineCreationError::DepthClampFeatureNotEnabled => {
                "the `depth_clamp` feature must be enabled in order to use depth clamping"
            },
            GraphicsPipelineCreationError::DepthBiasClampFeatureNotEnabled => {
                "the `depth_bias_clamp` feature must be enabled in order to use a depth bias \
                 clamp different from 0.0."
            },
            GraphicsPipelineCreationError::FillModeNonSolidFeatureNotEnabled => {
                "the `fill_mode_non_solid` feature must be enabled in order to use a polygon mode \
                 different from `Fill`"
            },
            GraphicsPipelineCreationError::DepthBoundsFeatureNotEnabled => {
                "the `depth_bounds` feature must be enabled in order to use depth bounds testing"
            },
            GraphicsPipelineCreationError::WrongStencilState => {
                "the requested stencil test is invalid"
            },
            GraphicsPipelineCreationError::TopologyNotMatchingGeometryShader => {
                "the primitives topology does not match what the geometry shader expects"
            },
            GraphicsPipelineCreationError::GeometryShaderFeatureNotEnabled => {
                "the `geometry_shader` feature must be enabled in order to use geometry shaders"
            },
            GraphicsPipelineCreationError::TessellationShaderFeatureNotEnabled => {
                "the `tessellation_shader` feature must be enabled in order to use tessellation \
                 shaders"
            },
            GraphicsPipelineCreationError::MismatchBlendingAttachmentsCount => {
                "the number of attachments specified in the blending does not match the number of \
                 attachments in the subpass"
            },
            GraphicsPipelineCreationError::IndependentBlendFeatureNotEnabled => {
                "the `independent_blend` feature must be enabled in order to use different \
                 blending operations per attachment"
            },
            GraphicsPipelineCreationError::LogicOpFeatureNotEnabled => {
                "the `logic_op` feature must be enabled in order to use logic operations"
            },
            GraphicsPipelineCreationError::NoDepthAttachment => {
                "the depth attachment of the render pass does not match the depth test"
            },
            GraphicsPipelineCreationError::NoStencilAttachment => {
                "the stencil attachment of the render pass does not match the stencil test"
            },
            GraphicsPipelineCreationError::InvalidPrimitiveTopology => {
                "trying to use a patch list without a tessellation shader, or a non-patch-list \
                 with a tessellation shader"
            },
            GraphicsPipelineCreationError::MaxTessellationPatchSizeExceeded => {
                "the maximum tessellation patch size was exceeded"
            },
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

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use format::Format;
    use framebuffer::Subpass;
    use descriptor::pipeline_layout::EmptyPipeline;
    use descriptor::pipeline_layout::EmptyPipelineDesc;
    use pipeline::GraphicsPipeline;
    use pipeline::GraphicsPipelineParams;
    use pipeline::GraphicsPipelineCreationError;
    use pipeline::blend::Blend;
    use pipeline::depth_stencil::DepthStencil;
    use pipeline::input_assembly::InputAssembly;
    use pipeline::input_assembly::PrimitiveTopology;
    use pipeline::multisample::Multisample;
    use pipeline::shader::ShaderModule;
    use pipeline::shader::EmptyShaderInterfaceDef;
    use pipeline::vertex::SingleBufferDefinition;
    use pipeline::viewport::ViewportsState;
    use pipeline::viewport::Viewport;
    use pipeline::viewport::Scissor;

    #[test]
    fn create() {
        let (device, _) = gfx_dev_and_queue!();

        let vs = unsafe { ShaderModule::new(&device, &BASIC_VS).unwrap() };
        let fs = unsafe { ShaderModule::new(&device, &BASIC_FS).unwrap() };

        let _ = GraphicsPipeline::new(&device, GraphicsPipelineParams {
            vertex_input: SingleBufferDefinition::<()>::new(),
            vertex_shader: unsafe {
                vs.vertex_shader_entry_point::<(), _, _, _>(&CString::new("main").unwrap(),
                                                            EmptyShaderInterfaceDef,
                                                            EmptyShaderInterfaceDef,
                                                            EmptyPipelineDesc)
            },
            input_assembly: InputAssembly::triangle_list(),
            tessellation: None,
            geometry_shader: None,
            viewport: ViewportsState::Dynamic { num: 1 },
            raster: Default::default(),
            multisample: Multisample::disabled(),
            fragment_shader: unsafe {
                fs.fragment_shader_entry_point::<(), _, _, _>(&CString::new("main").unwrap(),
                                                              EmptyShaderInterfaceDef,
                                                              EmptyShaderInterfaceDef,
                                                              EmptyPipelineDesc)
            },
            depth_stencil: DepthStencil::disabled(),
            blend: Blend::pass_through(),
            layout: &EmptyPipeline::new(&device).unwrap(),
            render_pass: Subpass::from(&simple_rp::CustomRenderPass::new(&device, &{
                simple_rp::Formats { color: (Format::R8G8B8A8Unorm, 1) }
            }).unwrap(), 0).unwrap(),
        }).unwrap();
    }

    #[test]
    fn bad_primitive_restart() {
        let (device, _) = gfx_dev_and_queue!();

        let vs = unsafe { ShaderModule::new(&device, &BASIC_VS).unwrap() };
        let fs = unsafe { ShaderModule::new(&device, &BASIC_FS).unwrap() };

        let result = GraphicsPipeline::new(&device, GraphicsPipelineParams {
            vertex_input: SingleBufferDefinition::<()>::new(),
            vertex_shader: unsafe {
                vs.vertex_shader_entry_point::<(), _, _, _>(&CString::new("main").unwrap(),
                                                            EmptyShaderInterfaceDef,
                                                            EmptyShaderInterfaceDef,
                                                            EmptyPipelineDesc)
            },
            input_assembly: InputAssembly {
                topology: PrimitiveTopology::TriangleList,
                primitive_restart_enable: true,
            },
            tessellation: None,
            geometry_shader: None,
            viewport: ViewportsState::Dynamic { num: 1 },
            raster: Default::default(),
            multisample: Multisample::disabled(),
            fragment_shader: unsafe {
                fs.fragment_shader_entry_point::<(), _, _, _>(&CString::new("main").unwrap(),
                                                              EmptyShaderInterfaceDef,
                                                              EmptyShaderInterfaceDef,
                                                              EmptyPipelineDesc)
            },
            depth_stencil: DepthStencil::disabled(),
            blend: Blend::pass_through(),
            layout: &EmptyPipeline::new(&device).unwrap(),
            render_pass: Subpass::from(&simple_rp::CustomRenderPass::new(&device, &{
                simple_rp::Formats { color: (Format::R8G8B8A8Unorm, 1) }
            }).unwrap(), 0).unwrap(),
        });

        match result {
            Err(GraphicsPipelineCreationError::PrimitiveDoesntSupportPrimitiveRestart { .. }) => (),
            _ => panic!()
        }
    }

    #[test]
    fn multi_viewport_feature() {
        let (device, _) = gfx_dev_and_queue!();

        let vs = unsafe { ShaderModule::new(&device, &BASIC_VS).unwrap() };
        let fs = unsafe { ShaderModule::new(&device, &BASIC_FS).unwrap() };

        let result = GraphicsPipeline::new(&device, GraphicsPipelineParams {
            vertex_input: SingleBufferDefinition::<()>::new(),
            vertex_shader: unsafe {
                vs.vertex_shader_entry_point::<(), _, _, _>(&CString::new("main").unwrap(),
                                                            EmptyShaderInterfaceDef,
                                                            EmptyShaderInterfaceDef,
                                                            EmptyPipelineDesc)
            },
            input_assembly: InputAssembly::triangle_list(),
            tessellation: None,
            geometry_shader: None,
            viewport: ViewportsState::Dynamic { num: 2 },
            raster: Default::default(),
            multisample: Multisample::disabled(),
            fragment_shader: unsafe {
                fs.fragment_shader_entry_point::<(), _, _, _>(&CString::new("main").unwrap(),
                                                              EmptyShaderInterfaceDef,
                                                              EmptyShaderInterfaceDef,
                                                              EmptyPipelineDesc)
            },
            depth_stencil: DepthStencil::disabled(),
            blend: Blend::pass_through(),
            layout: &EmptyPipeline::new(&device).unwrap(),
            render_pass: Subpass::from(&simple_rp::CustomRenderPass::new(&device, &{
                simple_rp::Formats { color: (Format::R8G8B8A8Unorm, 1) }
            }).unwrap(), 0).unwrap(),
        });

        match result {
            Err(GraphicsPipelineCreationError::MultiViewportFeatureNotEnabled) => (),
            _ => panic!()
        }
    }

    #[test]
    fn max_viewports() {
        let (device, _) = gfx_dev_and_queue!(multi_viewport);

        let vs = unsafe { ShaderModule::new(&device, &BASIC_VS).unwrap() };
        let fs = unsafe { ShaderModule::new(&device, &BASIC_FS).unwrap() };

        let result = GraphicsPipeline::new(&device, GraphicsPipelineParams {
            vertex_input: SingleBufferDefinition::<()>::new(),
            vertex_shader: unsafe {
                vs.vertex_shader_entry_point::<(), _, _, _>(&CString::new("main").unwrap(),
                                                            EmptyShaderInterfaceDef,
                                                            EmptyShaderInterfaceDef,
                                                            EmptyPipelineDesc)
            },
            input_assembly: InputAssembly::triangle_list(),
            tessellation: None,
            geometry_shader: None,
            viewport: ViewportsState::Dynamic { num: !0 },
            raster: Default::default(),
            multisample: Multisample::disabled(),
            fragment_shader: unsafe {
                fs.fragment_shader_entry_point::<(), _, _, _>(&CString::new("main").unwrap(),
                                                              EmptyShaderInterfaceDef,
                                                              EmptyShaderInterfaceDef,
                                                              EmptyPipelineDesc)
            },
            depth_stencil: DepthStencil::disabled(),
            blend: Blend::pass_through(),
            layout: &EmptyPipeline::new(&device).unwrap(),
            render_pass: Subpass::from(&simple_rp::CustomRenderPass::new(&device, &{
                simple_rp::Formats { color: (Format::R8G8B8A8Unorm, 1) }
            }).unwrap(), 0).unwrap(),
        });

        match result {
            Err(GraphicsPipelineCreationError::MaxViewportsExceeded { .. }) => (),
            _ => panic!()
        }
    }

    #[test]
    fn no_depth_attachment() {
        let (device, _) = gfx_dev_and_queue!();

        let vs = unsafe { ShaderModule::new(&device, &BASIC_VS).unwrap() };
        let fs = unsafe { ShaderModule::new(&device, &BASIC_FS).unwrap() };

        let result = GraphicsPipeline::new(&device, GraphicsPipelineParams {
            vertex_input: SingleBufferDefinition::<()>::new(),
            vertex_shader: unsafe {
                vs.vertex_shader_entry_point::<(), _, _, _>(&CString::new("main").unwrap(),
                                                            EmptyShaderInterfaceDef,
                                                            EmptyShaderInterfaceDef,
                                                            EmptyPipelineDesc)
            },
            input_assembly: InputAssembly::triangle_list(),
            tessellation: None,
            geometry_shader: None,
            viewport: ViewportsState::Dynamic { num: 1 },
            raster: Default::default(),
            multisample: Multisample::disabled(),
            fragment_shader: unsafe {
                fs.fragment_shader_entry_point::<(), _, _, _>(&CString::new("main").unwrap(),
                                                              EmptyShaderInterfaceDef,
                                                              EmptyShaderInterfaceDef,
                                                              EmptyPipelineDesc)
            },
            depth_stencil: DepthStencil::simple_depth_test(),
            blend: Blend::pass_through(),
            layout: &EmptyPipeline::new(&device).unwrap(),
            render_pass: Subpass::from(&simple_rp::CustomRenderPass::new(&device, &{
                simple_rp::Formats { color: (Format::R8G8B8A8Unorm, 1) }
            }).unwrap(), 0).unwrap(),
        });

        match result {
            Err(GraphicsPipelineCreationError::NoDepthAttachment) => (),
            _ => panic!()
        }
    }


    mod simple_rp {
        use format::Format;

        single_pass_renderpass!{
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: Format,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        }
    }

    /*
        #version 450

        #extension GL_ARB_separate_shader_objects : enable
        #extension GL_ARB_shading_language_420pack : enable

        layout(location = 0) in vec2 position;

        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
        }
    */
    const BASIC_VS: [u8; 912] = [3, 2, 35, 7, 0, 0, 1, 0, 1, 0, 8, 0, 27, 0, 0, 0, 0, 0, 0, 0, 17,
                                 0, 2, 0, 1, 0, 0, 0, 17, 0, 2, 0, 32, 0, 0, 0, 17, 0, 2, 0, 33, 0,
                                 0, 0, 11, 0, 6, 0, 1, 0, 0, 0, 71, 76, 83, 76, 46, 115, 116, 100,
                                 46, 52, 53, 48, 0, 0, 0, 0, 14, 0, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                                 15, 0, 7, 0, 0, 0, 0, 0, 4, 0, 0, 0, 109, 97, 105, 110, 0, 0, 0,
                                 0, 13, 0, 0, 0, 18, 0, 0, 0, 3, 0, 3, 0, 2, 0, 0, 0, 194, 1, 0,
                                 0, 4, 0, 9, 0, 71, 76, 95, 65, 82, 66, 95, 115, 101, 112, 97,
                                 114, 97, 116, 101, 95, 115, 104, 97, 100, 101, 114, 95, 111, 98,
                                 106, 101, 99, 116, 115, 0, 0, 4, 0, 9, 0, 71, 76, 95, 65, 82, 66,
                                 95, 115, 104, 97, 100, 105, 110, 103, 95, 108, 97, 110, 103, 117,
                                 97, 103, 101, 95, 52, 50, 48, 112, 97, 99, 107, 0, 5, 0, 4, 0, 4,
                                 0, 0, 0, 109, 97, 105, 110, 0, 0, 0, 0, 5, 0, 6, 0, 11, 0, 0, 0,
                                 103, 108, 95, 80, 101, 114, 86, 101, 114, 116, 101, 120, 0, 0, 0,
                                 0, 6, 0, 6, 0, 11, 0, 0, 0, 0, 0, 0, 0, 103, 108, 95, 80, 111,
                                 115, 105, 116, 105, 111, 110, 0, 6, 0, 7, 0, 11, 0, 0, 0, 1, 0,
                                 0, 0, 103, 108, 95, 80, 111, 105, 110, 116, 83, 105, 122, 101, 0,
                                 0, 0, 0, 6, 0, 7, 0, 11, 0, 0, 0, 2, 0, 0, 0, 103, 108, 95, 67,
                                 108, 105, 112, 68, 105, 115, 116, 97, 110, 99, 101, 0, 6, 0, 7,
                                 0, 11, 0, 0, 0, 3, 0, 0, 0, 103, 108, 95, 67, 117, 108, 108, 68,
                                 105, 115, 116, 97, 110, 99, 101, 0, 5, 0, 3, 0, 13, 0, 0, 0, 0, 0,
                                 0, 0, 5, 0, 5, 0, 18, 0, 0, 0, 112, 111, 115, 105, 116, 105, 111,
                                 110, 0, 0, 0, 0, 72, 0, 5, 0, 11, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0,
                                 0, 0, 0, 0, 0, 72, 0, 5, 0, 11, 0, 0, 0, 1, 0, 0, 0, 11, 0, 0, 0,
                                 1, 0, 0, 0, 72, 0, 5, 0, 11, 0, 0, 0, 2, 0, 0, 0, 11, 0, 0, 0, 3,
                                 0, 0, 0, 72, 0, 5, 0, 11, 0, 0, 0, 3, 0, 0, 0, 11, 0, 0, 0, 4, 0,
                                 0, 0, 71, 0, 3, 0, 11, 0, 0, 0, 2, 0, 0, 0, 71, 0, 4, 0, 18, 0, 0,
                                 0, 30, 0, 0, 0, 0, 0, 0, 0, 19, 0, 2, 0, 2, 0, 0, 0, 33, 0, 3, 0,
                                 3, 0, 0, 0, 2, 0, 0, 0, 22, 0, 3, 0, 6, 0, 0, 0, 32, 0, 0, 0, 23,
                                 0, 4, 0, 7, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 21, 0, 4, 0, 8, 0, 0,
                                 0, 32, 0, 0, 0, 0, 0, 0, 0, 43, 0, 4, 0, 8, 0, 0, 0, 9, 0, 0, 0,
                                 1, 0, 0, 0, 28, 0, 4, 0, 10, 0, 0, 0, 6, 0, 0, 0, 9, 0, 0, 0, 30,
                                 0, 6, 0, 11, 0, 0, 0, 7, 0, 0, 0, 6, 0, 0, 0, 10, 0, 0, 0, 10, 0,
                                 0, 0, 32, 0, 4, 0, 12, 0, 0, 0, 3, 0, 0, 0, 11, 0, 0, 0, 59, 0,
                                 4, 0, 12, 0, 0, 0, 13, 0, 0, 0, 3, 0, 0, 0, 21, 0, 4, 0, 14, 0,
                                 0, 0, 32, 0, 0, 0, 1, 0, 0, 0, 43, 0, 4, 0, 14, 0, 0, 0, 15, 0,
                                 0, 0, 0, 0, 0, 0, 23, 0, 4, 0, 16, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0,
                                 0, 32, 0, 4, 0, 17, 0, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0, 59, 0, 4,
                                 0, 17, 0, 0, 0, 18, 0, 0, 0, 1, 0, 0, 0, 43, 0, 4, 0, 6, 0, 0,
                                 0, 20, 0, 0, 0, 0, 0, 0, 0, 43, 0, 4, 0, 6, 0, 0, 0, 21, 0, 0,
                                 0, 0, 0, 128, 63, 32, 0, 4, 0, 25, 0, 0, 0, 3, 0, 0, 0, 7, 0, 0,
                                 0, 54, 0, 5, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0,
                                 248, 0, 2, 0, 5, 0, 0, 0, 61, 0, 4, 0, 16, 0, 0, 0, 19, 0, 0, 0,
                                 18, 0, 0, 0, 81, 0, 5, 0, 6, 0, 0, 0, 22, 0, 0, 0, 19, 0, 0, 0,
                                 0, 0, 0, 0, 81, 0, 5, 0, 6, 0, 0, 0, 23, 0, 0, 0, 19, 0, 0, 0, 1,
                                 0, 0, 0, 80, 0, 7, 0, 7, 0, 0, 0, 24, 0, 0, 0, 22, 0, 0, 0, 23,
                                 0, 0, 0, 20, 0, 0, 0, 21, 0, 0, 0, 65, 0, 5, 0, 25, 0, 0, 0, 26,
                                 0, 0, 0, 13, 0, 0, 0, 15, 0, 0, 0, 62, 0, 3, 0, 26, 0, 0, 0, 24,
                                 0, 0, 0, 253, 0, 1, 0, 56, 0, 1, 0];

    /*
        #version 450

        #extension GL_ARB_separate_shader_objects : enable
        #extension GL_ARB_shading_language_420pack : enable

        layout(location = 0) out vec4 f_color;

        void main() {
            f_color = vec4(1.0, 0.0, 0.0, 1.0);
        }
    */
    const BASIC_FS: [u8; 420] = [3, 2, 35, 7, 0, 0, 1, 0, 1, 0, 8, 0, 13, 0, 0, 0, 0, 0, 0, 0, 17,
                                 0, 2, 0, 1, 0, 0, 0, 11, 0, 6, 0, 1, 0, 0, 0, 71, 76, 83, 76, 46,
                                 115, 116, 100, 46, 52, 53,48, 0, 0, 0, 0, 14, 0, 3, 0, 0, 0, 0,
                                 0, 1, 0, 0, 0, 15, 0, 6, 0, 4, 0, 0, 0, 4, 0, 0, 0, 109, 97,
                                 105, 110, 0, 0, 0, 0, 9, 0, 0, 0, 16, 0, 3, 0, 4, 0, 0, 0, 7, 0,
                                 0, 0, 3, 0, 3, 0, 2, 0, 0, 0, 194, 1, 0, 0, 4, 0, 9, 0, 71, 76,
                                 95, 65, 82, 66, 95, 115, 101, 112, 97, 114, 97, 116, 101, 95,
                                 115, 104, 97, 100, 101, 114, 95, 111, 98, 106, 101, 99, 116, 115,
                                 0, 0, 4, 0, 9, 0, 71, 76, 95, 65, 82, 66, 95, 115, 104, 97, 100,
                                 105, 110, 103, 95, 108, 97, 110, 103, 117, 97, 103, 101, 95, 52,
                                 50, 48, 112, 97, 99, 107, 0, 5, 0, 4, 0, 4, 0, 0, 0, 109, 97,
                                 105, 110, 0, 0, 0, 0, 5, 0, 4, 0, 9, 0, 0, 0, 102, 95, 99, 111,
                                 108, 111, 114, 0, 71, 0, 4, 0, 9, 0, 0, 0, 30, 0, 0, 0, 0, 0,
                                 0, 0, 19, 0, 2, 0, 2, 0, 0, 0, 33, 0, 3, 0, 3, 0, 0, 0, 2, 0, 0,
                                 0, 22, 0, 3, 0, 6, 0, 0, 0, 32, 0, 0, 0, 23, 0, 4, 0, 7, 0, 0,
                                 0, 6, 0, 0, 0, 4, 0, 0, 0, 32, 0, 4, 0, 8, 0, 0, 0, 3, 0, 0, 0,
                                 7, 0, 0, 0, 59, 0, 4, 0, 8, 0, 0, 0, 9, 0, 0, 0, 3, 0, 0, 0, 43,
                                 0, 4, 0, 6, 0, 0, 0, 10, 0, 0, 0, 0, 0, 128, 63, 43, 0, 4, 0, 6,
                                 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 44, 0, 7, 0, 7, 0, 0, 0, 12, 0,
                                 0, 0, 10, 0, 0, 0, 11, 0, 0, 0, 11, 0, 0, 0, 10, 0, 0, 0, 54, 0,
                                 5, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 248, 0, 2,
                                 0, 5, 0, 0, 0, 62, 0, 3, 0, 9, 0, 0, 0, 12, 0, 0, 0, 253, 0, 1,
                                 0, 56, 0, 1, 0];
}
