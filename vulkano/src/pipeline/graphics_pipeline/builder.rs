// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// TODO: graphics pipeline params are deprecated, but are still the primary implementation in order
// to avoid duplicating code, so we hide the warnings for now
#![allow(deprecated)]

use smallvec::SmallVec;
use std::mem;
use std::ptr;
use std::sync::Arc;
use std::u32;

use descriptor::pipeline_layout::PipelineLayoutAbstract;
use device::Device;
use framebuffer::RenderPassAbstract;
use framebuffer::Subpass;
use pipeline::blend::AttachmentBlend;
use pipeline::blend::AttachmentsBlend;
use pipeline::blend::Blend;
use pipeline::blend::LogicOp;
use pipeline::depth_stencil::Compare;
use pipeline::depth_stencil::DepthBounds;
use pipeline::depth_stencil::DepthStencil;
use pipeline::graphics_pipeline::GraphicsPipeline;
use pipeline::graphics_pipeline::GraphicsPipelineCreationError;
use pipeline::graphics_pipeline::Inner as GraphicsPipelineInner;
use pipeline::input_assembly::PrimitiveTopology;
use pipeline::raster::CullMode;
use pipeline::raster::DepthBiasControl;
use pipeline::raster::FrontFace;
use pipeline::raster::PolygonMode;
use pipeline::raster::Rasterization;
use pipeline::shader::EmptyEntryPointDummy;
use pipeline::shader::GraphicsEntryPointAbstract;
use pipeline::shader::GraphicsShaderType;
use pipeline::shader::ShaderInterfaceDefMatch;
use pipeline::shader::SpecializationConstants;
use pipeline::vertex::BufferlessDefinition;
use pipeline::vertex::SingleBufferDefinition;
use pipeline::vertex::VertexDefinition;
use pipeline::viewport::Scissor;
use pipeline::viewport::Viewport;
use pipeline::viewport::ViewportsState;

use VulkanObject;
use check_errors;
use descriptor::pipeline_layout::PipelineLayoutDesc;
use descriptor::pipeline_layout::PipelineLayoutSuperset;
use framebuffer::RenderPassSubpassInterface;
use vk;

/// Prototype for a `GraphicsPipeline`.
// TODO: we can optimize this by filling directly the raw vk structs
pub struct GraphicsPipelineBuilder<Vdef, Vs, Vss, Tcs, Tcss, Tes, Tess, Gs, Gss, Fs, Fss, Rp> {
    vertex_input: Vdef,
    vertex_shader: Option<(Vs, Vss)>,
    input_assembly: vk::PipelineInputAssemblyStateCreateInfo,
    // Note: the `input_assembly_topology` member is temporary in order to not lose information
    // about the number of patches per primitive.
    input_assembly_topology: PrimitiveTopology,
    tessellation: Option<TessInfo<Tcs, Tcss, Tes, Tess>>,
    geometry_shader: Option<(Gs, Gss)>,
    viewport: Option<ViewportsState>,
    raster: Rasterization,
    multisample: vk::PipelineMultisampleStateCreateInfo,
    fragment_shader: Option<(Fs, Fss)>,
    depth_stencil: DepthStencil,
    blend: Blend,
    render_pass: Option<Subpass<Rp>>,
}

// Additional parameters if tessellation is used.
#[derive(Copy, Clone)]
struct TessInfo<Tcs, Tcss, Tes, Tess> {
    tessellation_control_shader: (Tcs, Tcss),
    tessellation_evaluation_shader: (Tes, Tess),
}

impl
    GraphicsPipelineBuilder<BufferlessDefinition,
                            EmptyEntryPointDummy,
                            (),
                            EmptyEntryPointDummy,
                            (),
                            EmptyEntryPointDummy,
                            (),
                            EmptyEntryPointDummy,
                            (),
                            EmptyEntryPointDummy,
                            (),
                            ()> {
    /// Builds a new empty builder.
    pub(super) fn new() -> Self {
        unsafe {
            GraphicsPipelineBuilder {
                vertex_input: BufferlessDefinition,
                vertex_shader: None,
                input_assembly: vk::PipelineInputAssemblyStateCreateInfo {
                    sType: vk::STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                    topology: PrimitiveTopology::TriangleList.into(),
                    ..mem::zeroed()
                },
                input_assembly_topology: PrimitiveTopology::TriangleList,
                tessellation: None,
                geometry_shader: None,
                viewport: None,
                raster: Default::default(),
                multisample: vk::PipelineMultisampleStateCreateInfo {
                    sType: vk::STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                    ..mem::zeroed()
                },
                fragment_shader: None,
                depth_stencil: DepthStencil::disabled(),
                blend: Blend::pass_through(),
                render_pass: None,
            }
        }
    }
}

impl<Vdef, Vs, Vss, Tcs, Tcss, Tes, Tess, Gs, Gss, Fs, Fss, Rp>
    GraphicsPipelineBuilder<Vdef, Vs, Vss, Tcs, Tcss, Tes, Tess, Gs, Gss, Fs, Fss, Rp>
    where Vdef: VertexDefinition<Vs::InputDefinition>,
          Vs: GraphicsEntryPointAbstract,
          Fs: GraphicsEntryPointAbstract,
          Gs: GraphicsEntryPointAbstract,
          Tcs: GraphicsEntryPointAbstract,
          Tes: GraphicsEntryPointAbstract,
          Vss: SpecializationConstants,
          Tcss: SpecializationConstants,
          Tess: SpecializationConstants,
          Gss: SpecializationConstants,
          Fss: SpecializationConstants,
          Vs::PipelineLayout: Clone + 'static + Send + Sync, // TODO: shouldn't be required
          Fs::PipelineLayout: Clone + 'static + Send + Sync, // TODO: shouldn't be required
          Tcs::PipelineLayout: Clone + 'static + Send + Sync, // TODO: shouldn't be required
          Tes::PipelineLayout: Clone + 'static + Send + Sync, // TODO: shouldn't be required
          Gs::PipelineLayout: Clone + 'static + Send + Sync, // TODO: shouldn't be required
          Tcs::InputDefinition: ShaderInterfaceDefMatch<Vs::OutputDefinition>,
          Tes::InputDefinition: ShaderInterfaceDefMatch<Tcs::OutputDefinition>,
          Gs::InputDefinition: ShaderInterfaceDefMatch<Tes::OutputDefinition>
                                   + ShaderInterfaceDefMatch<Vs::OutputDefinition>,
          Fs::InputDefinition: ShaderInterfaceDefMatch<Gs::OutputDefinition>
                                   + ShaderInterfaceDefMatch<Tes::OutputDefinition>
                                   + ShaderInterfaceDefMatch<Vs::OutputDefinition>,
          Rp: RenderPassAbstract + RenderPassSubpassInterface<Fs::OutputDefinition>
{
    /// Builds the graphics pipeline.
    // TODO: replace Box<PipelineLayoutAbstract> with a PipelineUnion struct without template params
    pub fn build(self, device: Arc<Device>)
                 -> Result<GraphicsPipeline<Vdef, Box<PipelineLayoutAbstract + Send + Sync>, Rp>,
                           GraphicsPipelineCreationError> {
        let pipeline_layout;

        if let Some(ref tess) = self.tessellation {
            if let Some(ref gs) = self.geometry_shader {
                if let Err(err) = tess.tessellation_control_shader
                    .0
                    .input()
                    .matches(self.vertex_shader.as_ref().unwrap().0.output())
                {
                    return Err(GraphicsPipelineCreationError::VertexTessControlStagesMismatch(err));
                }
                if let Err(err) = tess.tessellation_evaluation_shader
                    .0
                    .input()
                    .matches(tess.tessellation_control_shader.0.output())
                {
                    return Err(GraphicsPipelineCreationError::TessControlTessEvalStagesMismatch(err));
                }
                if let Err(err) = gs.0
                    .input()
                    .matches(tess.tessellation_evaluation_shader.0.output())
                {
                    return Err(GraphicsPipelineCreationError::TessEvalGeometryStagesMismatch(err));
                }
                if let Err(err) = self.fragment_shader
                    .as_ref()
                    .unwrap()
                    .0
                    .input()
                    .matches(gs.0.output())
                {
                    return Err(GraphicsPipelineCreationError::GeometryFragmentStagesMismatch(err));
                }

                pipeline_layout = Box::new(self.vertex_shader.as_ref().unwrap().0.layout().clone()
                    .union(self.fragment_shader.as_ref().unwrap().0.layout().clone())
                    .union(self.tessellation.as_ref().unwrap().tessellation_control_shader.0.layout().clone())    // FIXME: unwrap()
                    .union(self.tessellation.as_ref().unwrap().tessellation_evaluation_shader.0.layout().clone())    // FIXME: unwrap()
                    .union(self.geometry_shader.as_ref().unwrap().0.layout().clone())    // FIXME: unwrap()
                    .build(device.clone()).unwrap()) as Box<_>; // TODO: error

            } else {
                if let Err(err) = tess.tessellation_control_shader
                    .0
                    .input()
                    .matches(self.vertex_shader.as_ref().unwrap().0.output())
                {
                    return Err(GraphicsPipelineCreationError::VertexTessControlStagesMismatch(err));
                }
                if let Err(err) = tess.tessellation_evaluation_shader
                    .0
                    .input()
                    .matches(tess.tessellation_control_shader.0.output())
                {
                    return Err(GraphicsPipelineCreationError::TessControlTessEvalStagesMismatch(err));
                }
                if let Err(err) = self.fragment_shader
                    .as_ref()
                    .unwrap()
                    .0
                    .input()
                    .matches(tess.tessellation_evaluation_shader.0.output())
                {
                    return Err(GraphicsPipelineCreationError::TessEvalFragmentStagesMismatch(err));
                }

                pipeline_layout = Box::new(self.vertex_shader.as_ref().unwrap().0.layout().clone()
                    .union(self.fragment_shader.as_ref().unwrap().0.layout().clone())
                    .union(self.tessellation.as_ref().unwrap().tessellation_control_shader.0.layout().clone())    // FIXME: unwrap()
                    .union(self.tessellation.as_ref().unwrap().tessellation_evaluation_shader.0.layout().clone())    // FIXME: unwrap()
                    .build(device.clone()).unwrap()) as Box<_>; // TODO: error
            }

        } else {
            if let Some(ref geometry_shader) = self.geometry_shader {
                if let Err(err) = geometry_shader
                    .0
                    .input()
                    .matches(self.vertex_shader.as_ref().unwrap().0.output())
                {
                    return Err(GraphicsPipelineCreationError::VertexGeometryStagesMismatch(err));
                }
                if let Err(err) = self.fragment_shader
                    .as_ref()
                    .unwrap()
                    .0
                    .input()
                    .matches(geometry_shader.0.output())
                {
                    return Err(GraphicsPipelineCreationError::GeometryFragmentStagesMismatch(err));
                }

                pipeline_layout = Box::new(self.vertex_shader.as_ref().unwrap().0.layout().clone()
                    .union(self.fragment_shader.as_ref().unwrap().0.layout().clone())
                    .union(self.geometry_shader.as_ref().unwrap().0.layout().clone())    // FIXME: unwrap()
                    .build(device.clone()).unwrap()) as Box<_>; // TODO: error

            } else {
                if let Err(err) = self.fragment_shader
                    .as_ref()
                    .unwrap()
                    .0
                    .input()
                    .matches(self.vertex_shader.as_ref().unwrap().0.output())
                {
                    return Err(GraphicsPipelineCreationError::VertexFragmentStagesMismatch(err));
                }

                pipeline_layout =
                    Box::new(self.vertex_shader
                                 .as_ref()
                                 .unwrap()
                                 .0
                                 .layout()
                                 .clone()
                                 .union(self.fragment_shader.as_ref().unwrap().0.layout().clone())
                                 .build(device.clone())
                                 .unwrap()) as Box<_>; // TODO: error
            }
        }

        self.with_pipeline_layout(device, pipeline_layout)
    }

    /// Builds the graphics pipeline.
    ///
    /// Does the same as `build`, except that `build` automatically builds the pipeline layout
    /// object corresponding to the union of your shaders while this function allows you to specify
    /// the pipeline layout.
    pub fn with_pipeline_layout<Pl>(mut self, device: Arc<Device>, pipeline_layout: Pl)
                                    -> Result<GraphicsPipeline<Vdef, Pl, Rp>,
                                              GraphicsPipelineCreationError>
        where Pl: PipelineLayoutAbstract
    {
        // TODO: return errors instead of panicking if missing param

        let vk = device.pointers();

        // Checking that the pipeline layout matches the shader stages.
        // TODO: more details in the errors
        PipelineLayoutSuperset::ensure_superset_of(&pipeline_layout,
                                                   self.vertex_shader
                                                       .as_ref()
                                                       .unwrap()
                                                       .0
                                                       .layout())?;
        PipelineLayoutSuperset::ensure_superset_of(&pipeline_layout,
                                                   self.fragment_shader
                                                       .as_ref()
                                                       .unwrap()
                                                       .0
                                                       .layout())?;
        if let Some(ref geometry_shader) = self.geometry_shader {
            PipelineLayoutSuperset::ensure_superset_of(&pipeline_layout,
                                                       geometry_shader.0.layout())?;
        }
        if let Some(ref tess) = self.tessellation {
            PipelineLayoutSuperset::ensure_superset_of(&pipeline_layout,
                                                       tess.tessellation_control_shader
                                                           .0
                                                           .layout())?;
            PipelineLayoutSuperset::ensure_superset_of(&pipeline_layout,
                                                       tess.tessellation_evaluation_shader
                                                           .0
                                                           .layout())?;
        }

        // Check that the subpass can accept the output of the fragment shader.
        if !RenderPassSubpassInterface::is_compatible_with(&self.render_pass
                                                               .as_ref()
                                                               .unwrap()
                                                               .render_pass(),
                                                           self.render_pass
                                                               .as_ref()
                                                               .unwrap()
                                                               .index(),
                                                           self.fragment_shader
                                                               .as_ref()
                                                               .unwrap()
                                                               .0
                                                               .output())
        {
            return Err(GraphicsPipelineCreationError::FragmentShaderRenderPassIncompatible);
        }

        // Will contain the list of dynamic states. Filled throughout this function.
        let mut dynamic_states: SmallVec<[vk::DynamicState; 8]> = SmallVec::new();

        // Creating the specialization constants of the various stages.
        let vertex_shader_specialization = {
            let spec_descriptors = Vss::descriptors();
            let constants = &self.vertex_shader.as_ref().unwrap().1;
            vk::SpecializationInfo {
                mapEntryCount: spec_descriptors.len() as u32,
                pMapEntries: spec_descriptors.as_ptr() as *const _,
                dataSize: mem::size_of_val(constants),
                pData: constants as *const Vss as *const _,
            }
        };
        let tess_shader_specialization = if let Some(ref tess) = self.tessellation {
            let tcs_spec = {
                let spec_descriptors = Tcss::descriptors();
                let constants = &tess.tessellation_control_shader.1;
                vk::SpecializationInfo {
                    mapEntryCount: spec_descriptors.len() as u32,
                    pMapEntries: spec_descriptors.as_ptr() as *const _,
                    dataSize: mem::size_of_val(constants),
                    pData: constants as *const Tcss as *const _,
                }
            };
            let tes_spec = {
                let spec_descriptors = Tess::descriptors();
                let constants = &tess.tessellation_evaluation_shader.1;
                vk::SpecializationInfo {
                    mapEntryCount: spec_descriptors.len() as u32,
                    pMapEntries: spec_descriptors.as_ptr() as *const _,
                    dataSize: mem::size_of_val(constants),
                    pData: constants as *const Tess as *const _,
                }
            };
            Some((tcs_spec, tes_spec))
        } else {
            None
        };
        let geometry_shader_specialization = if let Some(ref gs) = self.geometry_shader {
            let spec_descriptors = Gss::descriptors();
            let constants = &gs.1;
            Some(vk::SpecializationInfo {
                     mapEntryCount: spec_descriptors.len() as u32,
                     pMapEntries: spec_descriptors.as_ptr() as *const _,
                     dataSize: mem::size_of_val(constants),
                     pData: constants as *const Gss as *const _,
                 })
        } else {
            None
        };
        let fragment_shader_specialization = {
            let spec_descriptors = Fss::descriptors();
            let constants = &self.fragment_shader.as_ref().unwrap().1;
            vk::SpecializationInfo {
                mapEntryCount: spec_descriptors.len() as u32,
                pMapEntries: spec_descriptors.as_ptr() as *const _,
                dataSize: mem::size_of_val(constants),
                pData: constants as *const Fss as *const _,
            }
        };

        // List of shader stages.
        let stages = {
            let mut stages = SmallVec::<[_; 5]>::new();

            match self.vertex_shader.as_ref().unwrap().0.ty() {
                GraphicsShaderType::Vertex => {},
                _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
            };

            stages.push(vk::PipelineShaderStageCreateInfo {
                            sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                            pNext: ptr::null(),
                            flags: 0, // reserved
                            stage: vk::SHADER_STAGE_VERTEX_BIT,
                            module: self.vertex_shader
                                .as_ref()
                                .unwrap()
                                .0
                                .module()
                                .internal_object(),
                            pName: self.vertex_shader.as_ref().unwrap().0.name().as_ptr(),
                            pSpecializationInfo: &vertex_shader_specialization as *const _,
                        });

            match self.fragment_shader.as_ref().unwrap().0.ty() {
                GraphicsShaderType::Fragment => {},
                _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
            };

            stages.push(vk::PipelineShaderStageCreateInfo {
                            sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                            pNext: ptr::null(),
                            flags: 0, // reserved
                            stage: vk::SHADER_STAGE_FRAGMENT_BIT,
                            module: self.fragment_shader
                                .as_ref()
                                .unwrap()
                                .0
                                .module()
                                .internal_object(),
                            pName: self.fragment_shader.as_ref().unwrap().0.name().as_ptr(),
                            pSpecializationInfo: &fragment_shader_specialization as *const _,
                        });

            if let Some(ref gs) = self.geometry_shader {
                if !device.enabled_features().geometry_shader {
                    return Err(GraphicsPipelineCreationError::GeometryShaderFeatureNotEnabled);
                }

                stages.push(vk::PipelineShaderStageCreateInfo {
                                sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                pNext: ptr::null(),
                                flags: 0, // reserved
                                stage: vk::SHADER_STAGE_GEOMETRY_BIT,
                                module: gs.0.module().internal_object(),
                                pName: gs.0.name().as_ptr(),
                                pSpecializationInfo: geometry_shader_specialization
                                    .as_ref()
                                    .unwrap() as
                                    *const _,
                            });
            }

            if let Some(ref tess) = self.tessellation {
                // FIXME: must check that the control shader and evaluation shader are compatible

                if !device.enabled_features().tessellation_shader {
                    return Err(GraphicsPipelineCreationError::TessellationShaderFeatureNotEnabled);
                }

                match tess.tessellation_control_shader.0.ty() {
                    GraphicsShaderType::TessellationControl => {},
                    _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
                };

                match tess.tessellation_evaluation_shader.0.ty() {
                    GraphicsShaderType::TessellationControl => {},
                    _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
                };

                stages.push(vk::PipelineShaderStageCreateInfo {
                                sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                pNext: ptr::null(),
                                flags: 0, // reserved
                                stage: vk::SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                                module: tess.tessellation_control_shader
                                    .0
                                    .module()
                                    .internal_object(),
                                pName: tess.tessellation_control_shader.0.name().as_ptr(),
                                pSpecializationInfo: &tess_shader_specialization
                                    .as_ref()
                                    .unwrap()
                                    .0 as
                                    *const _,
                            });

                stages.push(vk::PipelineShaderStageCreateInfo {
                                sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                pNext: ptr::null(),
                                flags: 0, // reserved
                                stage: vk::SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                                module: tess.tessellation_evaluation_shader
                                    .0
                                    .module()
                                    .internal_object(),
                                pName: tess.tessellation_evaluation_shader.0.name().as_ptr(),
                                pSpecializationInfo: &tess_shader_specialization
                                    .as_ref()
                                    .unwrap()
                                    .1 as
                                    *const _,
                            });
            }

            stages
        };

        // Vertex bindings.
        let (binding_descriptions, attribute_descriptions) = {
            let (buffers_iter, attribs_iter) =
                self.vertex_input
                    .definition(self.vertex_shader.as_ref().unwrap().0.input())?;

            let mut binding_descriptions = SmallVec::<[_; 8]>::new();
            for (num, stride, rate) in buffers_iter {
                if stride >
                    device
                        .physical_device()
                        .limits()
                        .max_vertex_input_binding_stride() as usize
                {
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

                if info.offset >
                    device
                        .physical_device()
                        .limits()
                        .max_vertex_input_attribute_offset() as usize
                {
                    return Err(GraphicsPipelineCreationError::MaxVertexInputAttributeOffsetExceeded {
                        max: device.physical_device().limits().max_vertex_input_attribute_offset() as usize,
                        obtained: info.offset,
                    });
                }

                debug_assert!(binding_descriptions
                                  .iter()
                                  .find(|b| b.binding == binding)
                                  .is_some());

                attribute_descriptions.push(vk::VertexInputAttributeDescription {
                                                location: loc as u32,
                                                binding: binding as u32,
                                                format: info.format as u32,
                                                offset: info.offset as u32,
                                            });
            }

            (binding_descriptions, attribute_descriptions)
        };

        if binding_descriptions.len() >
            device
                .physical_device()
                .limits()
                .max_vertex_input_bindings() as usize
        {
            return Err(GraphicsPipelineCreationError::MaxVertexInputBindingsExceeded {
                           max: device
                               .physical_device()
                               .limits()
                               .max_vertex_input_bindings() as
                               usize,
                           obtained: binding_descriptions.len(),
                       });
        }

        if attribute_descriptions.len() >
            device
                .physical_device()
                .limits()
                .max_vertex_input_attributes() as usize
        {
            return Err(GraphicsPipelineCreationError::MaxVertexInputAttributesExceeded {
                           max: device
                               .physical_device()
                               .limits()
                               .max_vertex_input_attributes() as
                               usize,
                           obtained: attribute_descriptions.len(),
                       });
        }

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
            sType: vk::STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0, // reserved
            vertexBindingDescriptionCount: binding_descriptions.len() as u32,
            pVertexBindingDescriptions: binding_descriptions.as_ptr(),
            vertexAttributeDescriptionCount: attribute_descriptions.len() as u32,
            pVertexAttributeDescriptions: attribute_descriptions.as_ptr(),
        };

        if self.input_assembly.primitiveRestartEnable != vk::FALSE &&
            !self.input_assembly_topology.supports_primitive_restart()
        {
            return Err(GraphicsPipelineCreationError::PrimitiveDoesntSupportPrimitiveRestart {
                           primitive: self.input_assembly_topology,
                       });
        }

        // TODO: should check from the tess eval shader instead of the input assembly
        if let Some(ref gs) = self.geometry_shader {
            match gs.0.ty() {
                GraphicsShaderType::Geometry(primitives) => {
                    if !primitives.matches(self.input_assembly_topology) {
                        return Err(GraphicsPipelineCreationError::TopologyNotMatchingGeometryShader);
                    }
                },
                _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
            }
        }

        let tessellation = match self.input_assembly_topology {
            PrimitiveTopology::PatchList { vertices_per_patch } => {
                if self.tessellation.is_none() {
                    return Err(GraphicsPipelineCreationError::InvalidPrimitiveTopology);
                }
                if vertices_per_patch >
                    device
                        .physical_device()
                        .limits()
                        .max_tessellation_patch_size()
                {
                    return Err(GraphicsPipelineCreationError::MaxTessellationPatchSizeExceeded);
                }

                Some(vk::PipelineTessellationStateCreateInfo {
                         sType: vk::STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
                         pNext: ptr::null(),
                         flags: 0, // reserved,
                         patchControlPoints: vertices_per_patch,
                     })
            },
            _ => {
                if self.tessellation.is_some() {
                    return Err(GraphicsPipelineCreationError::InvalidPrimitiveTopology);
                }

                None
            },
        };

        let (vp_vp, vp_sc, vp_num) = match *self.viewport.as_ref().unwrap() {
            ViewportsState::Fixed { ref data } => (data.iter()
                                                       .map(|e| e.0.clone().into_vulkan_viewport())
                                                       .collect::<SmallVec<[vk::Viewport; 4]>>(),
                                                   data.iter()
                                                       .map(|e| e.1.clone().into_vulkan_rect())
                                                       .collect::<SmallVec<[vk::Rect2D; 4]>>(),
                                                   data.len() as u32),
            ViewportsState::DynamicViewports { ref scissors } => {
                let num = scissors.len() as u32;
                let scissors = scissors
                    .iter()
                    .map(|e| e.clone().into_vulkan_rect())
                    .collect::<SmallVec<[vk::Rect2D; 4]>>();
                dynamic_states.push(vk::DYNAMIC_STATE_VIEWPORT);
                (SmallVec::new(), scissors, num)
            },
            ViewportsState::DynamicScissors { ref viewports } => {
                let num = viewports.len() as u32;
                let viewports = viewports
                    .iter()
                    .map(|e| e.clone().into_vulkan_viewport())
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
            flags: 0, // reserved
            viewportCount: vp_num,
            pViewports: if vp_vp.is_empty() {
                ptr::null()
            } else {
                vp_vp.as_ptr()
            }, // validation layer crashes if you just pass the pointer
            scissorCount: vp_num,
            pScissors: if vp_sc.is_empty() {
                ptr::null()
            } else {
                vp_sc.as_ptr()
            }, // validation layer crashes if you just pass the pointer
        };

        if let Some(line_width) = self.raster.line_width {
            if line_width != 1.0 && !device.enabled_features().wide_lines {
                return Err(GraphicsPipelineCreationError::WideLinesFeatureNotEnabled);
            }
        } else {
            dynamic_states.push(vk::DYNAMIC_STATE_LINE_WIDTH);
        }

        let (db_enable, db_const, db_clamp, db_slope) = match self.raster.depth_bias {
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

        if self.raster.depth_clamp && !device.enabled_features().depth_clamp {
            return Err(GraphicsPipelineCreationError::DepthClampFeatureNotEnabled);
        }

        if self.raster.polygon_mode != PolygonMode::Fill &&
            !device.enabled_features().fill_mode_non_solid
        {
            return Err(GraphicsPipelineCreationError::FillModeNonSolidFeatureNotEnabled);
        }

        let rasterization = vk::PipelineRasterizationStateCreateInfo {
            sType: vk::STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0, // reserved
            depthClampEnable: if self.raster.depth_clamp {
                vk::TRUE
            } else {
                vk::FALSE
            },
            rasterizerDiscardEnable: if self.raster.rasterizer_discard {
                vk::TRUE
            } else {
                vk::FALSE
            },
            polygonMode: self.raster.polygon_mode as u32,
            cullMode: self.raster.cull_mode as u32,
            frontFace: self.raster.front_face as u32,
            depthBiasEnable: db_enable,
            depthBiasConstantFactor: db_const,
            depthBiasClamp: db_clamp,
            depthBiasSlopeFactor: db_slope,
            lineWidth: self.raster.line_width.unwrap_or(1.0),
        };

        self.multisample.rasterizationSamples = self.render_pass
            .as_ref()
            .unwrap()
            .num_samples()
            .unwrap_or(1);
        if self.multisample.sampleShadingEnable != vk::FALSE {
            debug_assert!(self.multisample.minSampleShading >= 0.0 &&
                              self.multisample.minSampleShading <= 1.0);
            if !device.enabled_features().sample_rate_shading {
                return Err(GraphicsPipelineCreationError::SampleRateShadingFeatureNotEnabled);
            }
        }
        if self.multisample.alphaToOneEnable != vk::FALSE {
            if !device.enabled_features().alpha_to_one {
                return Err(GraphicsPipelineCreationError::AlphaToOneFeatureNotEnabled);
            }
        }

        let depth_stencil = {
            let db = match self.depth_stencil.depth_bounds_test {
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

            match (self.depth_stencil.stencil_front.compare_mask,
                     self.depth_stencil.stencil_back.compare_mask) {
                (Some(_), Some(_)) => (),
                (None, None) => {
                    dynamic_states.push(vk::DYNAMIC_STATE_STENCIL_COMPARE_MASK);
                },
                _ => return Err(GraphicsPipelineCreationError::WrongStencilState),
            };

            match (self.depth_stencil.stencil_front.write_mask,
                     self.depth_stencil.stencil_back.write_mask) {
                (Some(_), Some(_)) => (),
                (None, None) => {
                    dynamic_states.push(vk::DYNAMIC_STATE_STENCIL_WRITE_MASK);
                },
                _ => return Err(GraphicsPipelineCreationError::WrongStencilState),
            };

            match (self.depth_stencil.stencil_front.reference,
                     self.depth_stencil.stencil_back.reference) {
                (Some(_), Some(_)) => (),
                (None, None) => {
                    dynamic_states.push(vk::DYNAMIC_STATE_STENCIL_REFERENCE);
                },
                _ => return Err(GraphicsPipelineCreationError::WrongStencilState),
            };

            if self.depth_stencil.depth_write &&
                !self.render_pass.as_ref().unwrap().has_writable_depth()
            {
                return Err(GraphicsPipelineCreationError::NoDepthAttachment);
            }

            if self.depth_stencil.depth_compare != Compare::Always &&
                !self.render_pass.as_ref().unwrap().has_depth()
            {
                return Err(GraphicsPipelineCreationError::NoDepthAttachment);
            }

            if (!self.depth_stencil.stencil_front.always_keep() ||
                    !self.depth_stencil.stencil_back.always_keep()) &&
                !self.render_pass.as_ref().unwrap().has_stencil()
            {
                return Err(GraphicsPipelineCreationError::NoStencilAttachment);
            }

            // FIXME: stencil writability

            vk::PipelineDepthStencilStateCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0, // reserved
                depthTestEnable: if !self.depth_stencil.depth_write &&
                    self.depth_stencil.depth_compare == Compare::Always
                {
                    vk::FALSE
                } else {
                    vk::TRUE
                },
                depthWriteEnable: if self.depth_stencil.depth_write {
                    vk::TRUE
                } else {
                    vk::FALSE
                },
                depthCompareOp: self.depth_stencil.depth_compare as u32,
                depthBoundsTestEnable: db.0,
                stencilTestEnable: if self.depth_stencil.stencil_front.always_keep() &&
                    self.depth_stencil.stencil_back.always_keep()
                {
                    vk::FALSE
                } else {
                    vk::TRUE
                },
                front: vk::StencilOpState {
                    failOp: self.depth_stencil.stencil_front.fail_op as u32,
                    passOp: self.depth_stencil.stencil_front.pass_op as u32,
                    depthFailOp: self.depth_stencil.stencil_front.depth_fail_op as u32,
                    compareOp: self.depth_stencil.stencil_front.compare as u32,
                    compareMask: self.depth_stencil
                        .stencil_front
                        .compare_mask
                        .unwrap_or(u32::MAX),
                    writeMask: self.depth_stencil
                        .stencil_front
                        .write_mask
                        .unwrap_or(u32::MAX),
                    reference: self.depth_stencil.stencil_front.reference.unwrap_or(0),
                },
                back: vk::StencilOpState {
                    failOp: self.depth_stencil.stencil_back.fail_op as u32,
                    passOp: self.depth_stencil.stencil_back.pass_op as u32,
                    depthFailOp: self.depth_stencil.stencil_back.depth_fail_op as u32,
                    compareOp: self.depth_stencil.stencil_back.compare as u32,
                    compareMask: self.depth_stencil
                        .stencil_back
                        .compare_mask
                        .unwrap_or(u32::MAX),
                    writeMask: self.depth_stencil
                        .stencil_back
                        .write_mask
                        .unwrap_or(u32::MAX),
                    reference: self.depth_stencil.stencil_back.reference.unwrap_or(0),
                },
                minDepthBounds: db.1,
                maxDepthBounds: db.2,
            }
        };

        let blend_atch: SmallVec<[vk::PipelineColorBlendAttachmentState; 8]> = {
            let num_atch = self.render_pass.as_ref().unwrap().num_color_attachments();

            match self.blend.attachments {
                AttachmentsBlend::Collective(blend) => {
                    (0 .. num_atch)
                        .map(|_| blend.clone().into_vulkan_state())
                        .collect()
                },
                AttachmentsBlend::Individual(blend) => {
                    if blend.len() != num_atch as usize {
                        return Err(GraphicsPipelineCreationError::MismatchBlendingAttachmentsCount);
                    }

                    if !device.enabled_features().independent_blend {
                        return Err(GraphicsPipelineCreationError::IndependentBlendFeatureNotEnabled);
                    }

                    blend
                        .iter()
                        .map(|b| b.clone().into_vulkan_state())
                        .collect()
                },
            }
        };

        let blend = vk::PipelineColorBlendStateCreateInfo {
            sType: vk::STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0, // reserved
            logicOpEnable: if self.blend.logic_op.is_some() {
                if !device.enabled_features().logic_op {
                    return Err(GraphicsPipelineCreationError::LogicOpFeatureNotEnabled);
                }
                vk::TRUE
            } else {
                vk::FALSE
            },
            logicOp: self.blend.logic_op.unwrap_or(Default::default()) as u32,
            attachmentCount: blend_atch.len() as u32,
            pAttachments: blend_atch.as_ptr(),
            blendConstants: if let Some(c) = self.blend.blend_constants {
                c
            } else {
                dynamic_states.push(vk::DYNAMIC_STATE_BLEND_CONSTANTS);
                [0.0, 0.0, 0.0, 0.0]
            },
        };

        let dynamic_states = if !dynamic_states.is_empty() {
            Some(vk::PipelineDynamicStateCreateInfo {
                     sType: vk::STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                     pNext: ptr::null(),
                     flags: 0, // reserved
                     dynamicStateCount: dynamic_states.len() as u32,
                     pDynamicStates: dynamic_states.as_ptr(),
                 })
        } else {
            None
        };

        let pipeline = unsafe {
            let infos = vk::GraphicsPipelineCreateInfo {
                sType: vk::STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0, // TODO: some flags are available but none are critical
                stageCount: stages.len() as u32,
                pStages: stages.as_ptr(),
                pVertexInputState: &vertex_input_state,
                pInputAssemblyState: &self.input_assembly,
                pTessellationState: tessellation
                    .as_ref()
                    .map(|t| t as *const _)
                    .unwrap_or(ptr::null()),
                pViewportState: &viewport_info,
                pRasterizationState: &rasterization,
                pMultisampleState: &self.multisample,
                pDepthStencilState: &depth_stencil,
                pColorBlendState: &blend,
                pDynamicState: dynamic_states
                    .as_ref()
                    .map(|s| s as *const _)
                    .unwrap_or(ptr::null()),
                layout: PipelineLayoutAbstract::sys(&pipeline_layout).internal_object(),
                renderPass: self.render_pass
                    .as_ref()
                    .unwrap()
                    .render_pass()
                    .inner()
                    .internal_object(),
                subpass: self.render_pass.as_ref().unwrap().index(),
                basePipelineHandle: 0, // TODO:
                basePipelineIndex: -1, // TODO:
            };

            let mut output = mem::uninitialized();
            check_errors(vk.CreateGraphicsPipelines(device.internal_object(),
                                                    0,
                                                    1,
                                                    &infos,
                                                    ptr::null(),
                                                    &mut output))?;
            output
        };

        let (render_pass, render_pass_subpass) = self.render_pass.take().unwrap().into();

        Ok(GraphicsPipeline {
               inner: GraphicsPipelineInner {
                   device: device.clone(),
                   pipeline: pipeline,
               },
               layout: pipeline_layout,

               vertex_definition: self.vertex_input,

               render_pass: render_pass,
               render_pass_subpass: render_pass_subpass,

               dynamic_line_width: self.raster.line_width.is_none(),
               dynamic_viewport: self.viewport.as_ref().unwrap().dynamic_viewports(),
               dynamic_scissor: self.viewport.as_ref().unwrap().dynamic_scissors(),
               dynamic_depth_bias: self.raster.depth_bias.is_dynamic(),
               dynamic_depth_bounds: self.depth_stencil.depth_bounds_test.is_dynamic(),
               dynamic_stencil_compare_mask: self.depth_stencil.stencil_back.compare_mask.is_none(),
               dynamic_stencil_write_mask: self.depth_stencil.stencil_back.write_mask.is_none(),
               dynamic_stencil_reference: self.depth_stencil.stencil_back.reference.is_none(),
               dynamic_blend_constants: self.blend.blend_constants.is_none(),

               num_viewports: self.viewport.as_ref().unwrap().num_viewports(),
           })
    }

    // TODO: add build_with_cache method
}

impl<Vdef, Vs, Vss, Tcs, Tcss, Tes, Tess, Gs, Gss, Fs, Fss, Rp>
    GraphicsPipelineBuilder<Vdef, Vs, Vss, Tcs, Tcss, Tes, Tess, Gs, Gss, Fs, Fss, Rp> {
    // TODO: add pipeline derivate system

    /// Sets the vertex input.
    #[inline]
    pub fn vertex_input<T>(
        self, vertex_input: T)
        -> GraphicsPipelineBuilder<T, Vs, Vss, Tcs, Tcss, Tes, Tess, Gs, Gss, Fs, Fss, Rp> {
        GraphicsPipelineBuilder {
            vertex_input: vertex_input,
            vertex_shader: self.vertex_shader,
            input_assembly: self.input_assembly,
            input_assembly_topology: self.input_assembly_topology,
            tessellation: self.tessellation,
            geometry_shader: self.geometry_shader,
            viewport: self.viewport,
            raster: self.raster,
            multisample: self.multisample,
            fragment_shader: self.fragment_shader,
            depth_stencil: self.depth_stencil,
            blend: self.blend,
            render_pass: self.render_pass,
        }
    }

    /// Sets the vertex input to a single vertex buffer.
    ///
    /// You will most likely need to explicitely specify the template parameter to the type of a
    /// vertex.
    #[inline]
    pub fn vertex_input_single_buffer<V>(self)
                                         -> GraphicsPipelineBuilder<SingleBufferDefinition<V>,
                                                                    Vs,
                                                                    Vss,
                                                                    Tcs,
                                                                    Tcss,
                                                                    Tes,
                                                                    Tess,
                                                                    Gs,
                                                                    Gss,
                                                                    Fs,
                                                                    Fss,
                                                                    Rp> {
        self.vertex_input(SingleBufferDefinition::<V>::new())
    }

    /// Sets the vertex shader to use.
    // TODO: correct specialization constants
    #[inline]
    pub fn vertex_shader<Vs2, Vss2>(
        self, shader: Vs2, specialization_constants: Vss2)
        -> GraphicsPipelineBuilder<Vdef, Vs2, Vss2, Tcs, Tcss, Tes, Tess, Gs, Gss, Fs, Fss, Rp>
        where Vs2: GraphicsEntryPointAbstract<SpecializationConstants = Vss2>,
              Vss2: SpecializationConstants
    {
        GraphicsPipelineBuilder {
            vertex_input: self.vertex_input,
            vertex_shader: Some((shader, specialization_constants)),
            input_assembly: self.input_assembly,
            input_assembly_topology: self.input_assembly_topology,
            tessellation: self.tessellation,
            geometry_shader: self.geometry_shader,
            viewport: self.viewport,
            raster: self.raster,
            multisample: self.multisample,
            fragment_shader: self.fragment_shader,
            depth_stencil: self.depth_stencil,
            blend: self.blend,
            render_pass: self.render_pass,
        }
    }

    /// Sets whether primitive restart if enabled.
    #[inline]
    pub fn primitive_restart(mut self, enabled: bool) -> Self {
        self.input_assembly.primitiveRestartEnable = if enabled {
            vk::TRUE
        } else {
            vk::FALSE
        };

        self
    }

    /// Sets the topology of the primitives that are expected by the pipeline.
    #[inline]
    pub fn primitive_topology(mut self, topology: PrimitiveTopology) -> Self {
        self.input_assembly_topology = topology;
        self.input_assembly.topology = topology.into();
        self
    }

    /// Sets the topology of the primitives to a list of points.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::PointList)`.
    #[inline]
    pub fn point_list(self) -> Self {
        self.primitive_topology(PrimitiveTopology::PointList)
    }

    /// Sets the topology of the primitives to a list of lines.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::LineList)`.
    #[inline]
    pub fn line_list(self) -> Self {
        self.primitive_topology(PrimitiveTopology::LineList)
    }

    /// Sets the topology of the primitives to a line strip.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::LineStrip)`.
    #[inline]
    pub fn line_strip(self) -> Self {
        self.primitive_topology(PrimitiveTopology::LineStrip)
    }

    /// Sets the topology of the primitives to a list of triangles. Note that this is the default.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::TriangleList)`.
    #[inline]
    pub fn triangle_list(self) -> Self {
        self.primitive_topology(PrimitiveTopology::TriangleList)
    }

    /// Sets the topology of the primitives to a triangle strip.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::TriangleStrip)`.
    #[inline]
    pub fn triangle_strip(self) -> Self {
        self.primitive_topology(PrimitiveTopology::TriangleStrip)
    }

    /// Sets the topology of the primitives to a fan of triangles.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::TriangleFan)`.
    #[inline]
    pub fn triangle_fan(self) -> Self {
        self.primitive_topology(PrimitiveTopology::TriangleFan)
    }

    /// Sets the topology of the primitives to a list of lines with adjacency information.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::LineListWithAdjacency)`.
    #[inline]
    pub fn line_list_with_adjacency(self) -> Self {
        self.primitive_topology(PrimitiveTopology::LineListWithAdjacency)
    }

    /// Sets the topology of the primitives to a line strip with adjacency information.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::LineStripWithAdjacency)`.
    #[inline]
    pub fn line_strip_with_adjacency(self) -> Self {
        self.primitive_topology(PrimitiveTopology::LineStripWithAdjacency)
    }

    /// Sets the topology of the primitives to a list of triangles with adjacency information.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::TriangleListWithAdjacency)`.
    #[inline]
    pub fn triangle_list_with_adjacency(self) -> Self {
        self.primitive_topology(PrimitiveTopology::TriangleListWithAdjacency)
    }

    /// Sets the topology of the primitives to a triangle strip with adjacency information`
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::TriangleStripWithAdjacency)`.
    #[inline]
    pub fn triangle_strip_with_adjacency(self) -> Self {
        self.primitive_topology(PrimitiveTopology::TriangleStripWithAdjacency)
    }

    /// Sets the topology of the primitives to a list of patches. Can only be used and must be used
    /// with a tessellation shader.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::PatchList { vertices_per_patch })`.
    #[inline]
    pub fn patch_list(self, vertices_per_patch: u32) -> Self {
        self.primitive_topology(PrimitiveTopology::PatchList { vertices_per_patch })
    }

    /// Sets the tessellation shaders to use.
    // TODO: correct specialization constants
    #[inline]
    pub fn tessellation_shaders<Tcs2, Tcss2, Tes2, Tess2>(
        self, tessellation_control_shader: Tcs2,
        tessellation_control_shader_spec_constants: Tcss2, tessellation_evaluation_shader: Tes2,
        tessellation_evaluation_shader_spec_constants: Tess2)
        -> GraphicsPipelineBuilder<Vdef, Vs, Vss, Tcs2, Tcss2, Tes2, Tess2, Gs, Gss, Fs, Fss, Rp>
        where Tcs2: GraphicsEntryPointAbstract<SpecializationConstants = Tcss2>,
              Tes2: GraphicsEntryPointAbstract<SpecializationConstants = Tess2>,
              Tcss2: SpecializationConstants,
              Tess2: SpecializationConstants
    {
        GraphicsPipelineBuilder {
            vertex_input: self.vertex_input,
            vertex_shader: self.vertex_shader,
            input_assembly: self.input_assembly,
            input_assembly_topology: self.input_assembly_topology,
            tessellation: Some(TessInfo {
                                   tessellation_control_shader:
                                       (tessellation_control_shader,
                                        tessellation_control_shader_spec_constants),
                                   tessellation_evaluation_shader:
                                       (tessellation_evaluation_shader,
                                        tessellation_evaluation_shader_spec_constants),
                               }),
            geometry_shader: self.geometry_shader,
            viewport: self.viewport,
            raster: self.raster,
            multisample: self.multisample,
            fragment_shader: self.fragment_shader,
            depth_stencil: self.depth_stencil,
            blend: self.blend,
            render_pass: self.render_pass,
        }
    }

    /// Sets the tessellation shaders stage as disabled. This is the default.
    #[inline]
    pub fn tessellation_shaders_disabled(mut self) -> Self {
        self.tessellation = None;
        self
    }

    /// Sets the geometry shader to use.
    // TODO: correct specialization constants
    #[inline]
    pub fn geometry_shader<Gs2, Gss2>(
        self, shader: Gs2, specialization_constants: Gss2)
        -> GraphicsPipelineBuilder<Vdef, Vs, Vss, Tcs, Tcss, Tes, Tess, Gs2, Gss2, Fs, Fss, Rp>
        where Gs2: GraphicsEntryPointAbstract<SpecializationConstants = Gss2>,
              Gss2: SpecializationConstants
    {
        GraphicsPipelineBuilder {
            vertex_input: self.vertex_input,
            vertex_shader: self.vertex_shader,
            input_assembly: self.input_assembly,
            input_assembly_topology: self.input_assembly_topology,
            tessellation: self.tessellation,
            geometry_shader: Some((shader, specialization_constants)),
            viewport: self.viewport,
            raster: self.raster,
            multisample: self.multisample,
            fragment_shader: self.fragment_shader,
            depth_stencil: self.depth_stencil,
            blend: self.blend,
            render_pass: self.render_pass,
        }
    }

    /// Sets the geometry shader stage as disabled. This is the default.
    #[inline]
    pub fn geometry_shader_disabled(mut self) -> Self {
        self.geometry_shader = None;
        self
    }

    /// Sets the viewports to some value, and the scissor boxes to boxes that always cover the
    /// whole viewport.
    #[inline]
    pub fn viewports<I>(self, viewports: I) -> Self
        where I: IntoIterator<Item = Viewport>
    {
        self.viewports_scissors(viewports.into_iter().map(|v| (v, Scissor::irrelevant())))
    }

    /// Sets the characteristics of viewports and scissor boxes in advance.
    #[inline]
    pub fn viewports_scissors<I>(mut self, viewports: I) -> Self
        where I: IntoIterator<Item = (Viewport, Scissor)>
    {
        self.viewport = Some(ViewportsState::Fixed { data: viewports.into_iter().collect() });
        self
    }

    /// Sets the scissor boxes to some values, and viewports to dynamic. The viewports will
    /// need to be set before drawing.
    #[inline]
    pub fn viewports_dynamic_scissors_fixed<I>(mut self, scissors: I) -> Self
        where I: IntoIterator<Item = Scissor>
    {
        self.viewport =
            Some(ViewportsState::DynamicViewports { scissors: scissors.into_iter().collect() });
        self
    }

    /// Sets the viewports to dynamic, and the scissor boxes to boxes that always cover the whole
    /// viewport. The viewports will need to be set before drawing.
    #[inline]
    pub fn viewports_dynamic_scissors_irrelevant(mut self, num: u32) -> Self {
        self.viewport = Some(ViewportsState::DynamicViewports {
                                 scissors: (0 .. num).map(|_| Scissor::irrelevant()).collect(),
                             });
        self
    }

    /// Sets the viewports to some values, and scissor boxes to dynamic. The scissor boxes will
    /// need to be set before drawing.
    #[inline]
    pub fn viewports_fixed_scissors_dynamic<I>(mut self, viewports: I) -> Self
        where I: IntoIterator<Item = Viewport>
    {
        self.viewport =
            Some(ViewportsState::DynamicScissors { viewports: viewports.into_iter().collect() });
        self
    }

    /// Sets the viewports and scissor boxes to dynamic. They will both need to be set before
    /// drawing.
    #[inline]
    pub fn viewports_scissors_dynamic(mut self, num: u32) -> Self {
        self.viewport = Some(ViewportsState::Dynamic { num: num });
        self
    }

    /// If true, then the depth value of the vertices will be clamped to the range `[0.0 ; 1.0]`.
    /// If false, fragments whose depth is outside of this range will be discarded before the
    /// fragment shader even runs.
    #[inline]
    pub fn depth_clamp(mut self, clamp: bool) -> Self {
        self.raster.depth_clamp = clamp;
        self
    }

    // TODO: this won't work correctly
    /*/// Disables the fragment shader stage.
    #[inline]
    pub fn rasterizer_discard(mut self) -> Self {
        self.rasterization.rasterizer_discard. = true;
        self
    }*/

    /// Sets the front-facing faces to couner-clockwise faces. This is the default.
    ///
    /// Triangles whose vertices are oriented counter-clockwise on the screen will be considered
    /// as facing their front. Otherwise they will be considered as facing their back.
    #[inline]
    pub fn front_face_counter_clockwise(mut self) -> Self {
        self.raster.front_face = FrontFace::CounterClockwise;
        self
    }

    /// Sets the front-facing faces to clockwise faces.
    ///
    /// Triangles whose vertices are oriented clockwise on the screen will be considered
    /// as facing their front. Otherwise they will be considered as facing their back.
    #[inline]
    pub fn front_face_clockwise(mut self) -> Self {
        self.raster.front_face = FrontFace::Clockwise;
        self
    }

    /// Sets backface culling as disabled. This is the default.
    #[inline]
    pub fn cull_mode_disabled(mut self) -> Self {
        self.raster.cull_mode = CullMode::None;
        self
    }

    /// Sets backface culling to front faces. The front faces (as chosen with the `front_face_*`
    /// methods) will be discarded by the GPU when drawing.
    #[inline]
    pub fn cull_mode_front(mut self) -> Self {
        self.raster.cull_mode = CullMode::Front;
        self
    }

    /// Sets backface culling to back faces. Faces that are not facing the front (as chosen with
    /// the `front_face_*` methods) will be discarded by the GPU when drawing.
    #[inline]
    pub fn cull_mode_back(mut self) -> Self {
        self.raster.cull_mode = CullMode::Back;
        self
    }

    /// Sets backface culling to both front and back faces. All the faces will be discarded.
    ///
    /// > **Note**: This option exists for the sake of completeness. It has no known practical
    /// > usage.
    #[inline]
    pub fn cull_mode_front_and_back(mut self) -> Self {
        self.raster.cull_mode = CullMode::FrontAndBack;
        self
    }

    /// Sets the polygon mode to "fill". This is the default.
    #[inline]
    pub fn polygon_mode_fill(mut self) -> Self {
        self.raster.polygon_mode = PolygonMode::Fill;
        self
    }

    /// Sets the polygon mode to "line". Triangles will each be turned into three lines.
    #[inline]
    pub fn polygon_mode_line(mut self) -> Self {
        self.raster.polygon_mode = PolygonMode::Line;
        self
    }

    /// Sets the polygon mode to "point". Triangles and lines will each be turned into three points.
    #[inline]
    pub fn polygon_mode_point(mut self) -> Self {
        self.raster.polygon_mode = PolygonMode::Point;
        self
    }

    /// Sets the width of the lines, if the GPU needs to draw lines. The default is `1.0`.
    #[inline]
    pub fn line_width(mut self, value: f32) -> Self {
        self.raster.line_width = Some(value);
        self
    }

    /// Sets the width of the lines as dynamic, which means that you will need to set this value
    /// when drawing.
    #[inline]
    pub fn line_width_dynamic(mut self) -> Self {
        self.raster.line_width = None;
        self
    }

    // TODO: missing DepthBiasControl

    /// Disables sample shading. The fragment shader will only be run once per fragment (ie. per
    /// pixel) and not once by sample. The output will then be copied in all of the covered
    /// samples.
    ///
    /// Sample shading is disabled by default.
    #[inline]
    pub fn sample_shading_disabled(mut self) -> Self {
        self.multisample.sampleShadingEnable = vk::FALSE;
        self
    }

    /// Enables sample shading. The fragment shader will be run once per sample at the borders of
    /// the object you're drawing.
    ///
    /// Enabling sampling shading requires the `sample_rate_shading` feature to be enabled on the
    /// device.
    ///
    /// The `min_fract` parameter is the minimum fraction of samples shading. For example if its
    /// value is 0.5, then the fragment shader will run for at least half of the samples. The other
    /// half of the samples will get their values determined automatically.
    ///
    /// Sample shading is disabled by default.
    ///
    /// # Panic
    ///
    /// - Panics if `min_fract` is not between 0.0 and 1.0.
    ///
    #[inline]
    pub fn sample_shading_enabled(mut self, min_fract: f32) -> Self {
        assert!(min_fract >= 0.0 && min_fract <= 1.0);
        self.multisample.sampleShadingEnable = vk::TRUE;
        self.multisample.minSampleShading = min_fract;
        self
    }

    // TODO: doc
    pub fn alpha_to_coverage_disabled(mut self) -> Self {
        self.multisample.alphaToCoverageEnable = vk::FALSE;
        self
    }

    // TODO: doc
    pub fn alpha_to_coverage_enabled(mut self) -> Self {
        self.multisample.alphaToCoverageEnable = vk::TRUE;
        self
    }

    /// Disables alpha-to-one.
    ///
    /// Alpha-to-one is disabled by default.
    #[inline]
    pub fn alpha_to_one_disabled(mut self) -> Self {
        self.multisample.alphaToOneEnable = vk::FALSE;
        self
    }

    /// Enables alpha-to-one. The alpha component of the first color output of the fragment shader
    /// will be replaced by the value `1.0`.
    ///
    /// Enabling alpha-to-one requires the `alpha_to_one` feature to be enabled on the device.
    ///
    /// Alpha-to-one is disabled by default.
    #[inline]
    pub fn alpha_to_one_enabled(mut self) -> Self {
        self.multisample.alphaToOneEnable = vk::TRUE;
        self
    }

    // TODO: rasterizationSamples and pSampleMask

    /// Sets the fragment shader to use.
    ///
    /// The fragment shader is run once for each pixel that is covered by each primitive.
    // TODO: correct specialization constants
    #[inline]
    pub fn fragment_shader<Fs2, Fss2>(
        self, shader: Fs2, specialization_constants: Fss2)
        -> GraphicsPipelineBuilder<Vdef, Vs, Vss, Tcs, Tcss, Tes, Tess, Gs, Gss, Fs2, Fss2, Rp>
        where Fs2: GraphicsEntryPointAbstract<SpecializationConstants = Fss2>,
              Fss2: SpecializationConstants
    {
        GraphicsPipelineBuilder {
            vertex_input: self.vertex_input,
            vertex_shader: self.vertex_shader,
            input_assembly: self.input_assembly,
            input_assembly_topology: self.input_assembly_topology,
            tessellation: self.tessellation,
            geometry_shader: self.geometry_shader,
            viewport: self.viewport,
            raster: self.raster,
            multisample: self.multisample,
            fragment_shader: Some((shader, specialization_constants)),
            depth_stencil: self.depth_stencil,
            blend: self.blend,
            render_pass: self.render_pass,
        }
    }

    /// Sets the depth/stencil configuration. This function may be removed in the future.
    #[inline]
    pub fn depth_stencil(mut self, depth_stencil: DepthStencil) -> Self {
        self.depth_stencil = depth_stencil;
        self
    }

    /// Sets the depth/stencil tests as disabled.
    ///
    /// > **Note**: This is a shortcut for all the other `depth_*` and `depth_stencil_*` methods
    /// > of the builder.
    #[inline]
    pub fn depth_stencil_disabled(mut self) -> Self {
        self.depth_stencil = DepthStencil::disabled();
        self
    }

    /// Sets the depth/stencil tests as a simple depth test and no stencil test.
    ///
    /// > **Note**: This is a shortcut for setting the depth test to `Less`, the depth write Into
    /// > ` true` and disable the stencil test.
    #[inline]
    pub fn depth_stencil_simple_depth(mut self) -> Self {
        self.depth_stencil = DepthStencil::simple_depth_test();
        self
    }

    /// Sets whether the depth buffer will be written.
    #[inline]
    pub fn depth_write(mut self, write: bool) -> Self {
        self.depth_stencil.depth_write = write;
        self
    }

    // TODO: missing tons of depth-stencil stuff


    #[inline]
    pub fn blend_collective(mut self, blend: AttachmentBlend) -> Self {
        self.blend.attachments = AttachmentsBlend::Collective(blend);
        self
    }

    #[inline]
    pub fn blend_individual<I>(mut self, blend: I) -> Self
        where I: IntoIterator<Item = AttachmentBlend>
    {
        self.blend.attachments = AttachmentsBlend::Individual(blend.into_iter().collect());
        self
    }

    /// Each fragment shader output will have its value directly written to the framebuffer
    /// attachment. This is the default.
    #[inline]
    pub fn blend_pass_through(self) -> Self {
        self.blend_collective(AttachmentBlend::pass_through())
    }

    #[inline]
    pub fn blend_alpha_blending(self) -> Self {
        self.blend_collective(AttachmentBlend::alpha_blending())
    }

    #[inline]
    pub fn blend_logic_op(mut self, logic_op: LogicOp) -> Self {
        self.blend.logic_op = Some(logic_op);
        self
    }

    /// Sets the logic operation as disabled. This is the default.
    #[inline]
    pub fn blend_logic_op_disabled(mut self) -> Self {
        self.blend.logic_op = None;
        self
    }

    /// Sets the blend constant. The default is `[0.0, 0.0, 0.0, 0.0]`.
    ///
    /// The blend constant is used for some blending calculations. It is irrelevant otherwise.
    #[inline]
    pub fn blend_constants(mut self, constants: [f32; 4]) -> Self {
        self.blend.blend_constants = Some(constants);
        self
    }

    /// Sets the blend constant value as dynamic. Its value will need to be set before drawing.
    ///
    /// The blend constant is used for some blending calculations. It is irrelevant otherwise.
    #[inline]
    pub fn blend_constants_dynamic(mut self) -> Self {
        self.blend.blend_constants = None;
        self
    }

    /// Sets the render pass subpass to use.
    #[inline]
    pub fn render_pass<Rp2>(
        self, subpass: Subpass<Rp2>)
        -> GraphicsPipelineBuilder<Vdef, Vs, Vss, Tcs, Tcss, Tes, Tess, Gs, Gss, Fs, Fss, Rp2> {
        GraphicsPipelineBuilder {
            vertex_input: self.vertex_input,
            vertex_shader: self.vertex_shader,
            input_assembly: self.input_assembly,
            input_assembly_topology: self.input_assembly_topology,
            tessellation: self.tessellation,
            geometry_shader: self.geometry_shader,
            viewport: self.viewport,
            raster: self.raster,
            multisample: self.multisample,
            fragment_shader: self.fragment_shader,
            depth_stencil: self.depth_stencil,
            blend: self.blend,
            render_pass: Some(subpass),
        }
    }
}

impl<Vdef, Vs, Vss, Tcs, Tcss, Tes, Tess, Gs, Gss, Fs, Fss, Rp> Clone
    for GraphicsPipelineBuilder<Vdef, Vs, Vss, Tcs, Tcss, Tes, Tess, Gs, Gss, Fs, Fss, Rp>
    where Vdef: Clone,
          Vs: Clone,
          Vss: Clone,
          Tcs: Clone,
          Tcss: Clone,
          Tes: Clone,
          Tess: Clone,
          Gs: Clone,
          Gss: Clone,
          Fs: Clone,
          Fss: Clone,
          Rp: Clone
{
    fn clone(&self) -> Self {
        GraphicsPipelineBuilder {
            vertex_input: self.vertex_input.clone(),
            vertex_shader: self.vertex_shader.clone(),
            input_assembly: unsafe { ptr::read(&self.input_assembly) },
            input_assembly_topology: self.input_assembly_topology,
            tessellation: self.tessellation.clone(),
            geometry_shader: self.geometry_shader.clone(),
            viewport: self.viewport.clone(),
            raster: self.raster.clone(),
            multisample: vk::PipelineMultisampleStateCreateInfo {
                sType: self.multisample.sType,
                pNext: self.multisample.pNext,
                flags: self.multisample.flags,
                rasterizationSamples: self.multisample.rasterizationSamples,
                sampleShadingEnable: self.multisample.sampleShadingEnable,
                minSampleShading: self.multisample.minSampleShading,
                pSampleMask: self.multisample.pSampleMask,
                alphaToCoverageEnable: self.multisample.alphaToCoverageEnable,
                alphaToOneEnable: self.multisample.alphaToOneEnable,
            },
            fragment_shader: self.fragment_shader.clone(),
            depth_stencil: self.depth_stencil.clone(),
            blend: self.blend.clone(),
            render_pass: self.render_pass.clone(),
        }
    }
}
