// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// TODO: graphics pipeline params are deprecated, but are still the primary implementation in order
// to avoid duplicating code, so we hide the warnings for now
#![allow(deprecated)]

use crate::check_errors;
use crate::descriptor_set::layout::DescriptorSetDesc;
use crate::descriptor_set::layout::DescriptorSetLayout;
use crate::device::Device;
use crate::image::SampleCount;
use crate::pipeline::blend::AttachmentBlend;
use crate::pipeline::blend::AttachmentsBlend;
use crate::pipeline::blend::Blend;
use crate::pipeline::blend::LogicOp;
use crate::pipeline::cache::PipelineCache;
use crate::pipeline::depth_stencil::Compare;
use crate::pipeline::depth_stencil::DepthBounds;
use crate::pipeline::depth_stencil::DepthStencil;
use crate::pipeline::graphics_pipeline::GraphicsPipeline;
use crate::pipeline::graphics_pipeline::GraphicsPipelineCreationError;
use crate::pipeline::graphics_pipeline::Inner as GraphicsPipelineInner;
use crate::pipeline::input_assembly::PrimitiveTopology;
use crate::pipeline::layout::PipelineLayout;
use crate::pipeline::layout::PipelineLayoutPcRange;
use crate::pipeline::raster::CullMode;
use crate::pipeline::raster::DepthBiasControl;
use crate::pipeline::raster::FrontFace;
use crate::pipeline::raster::PolygonMode;
use crate::pipeline::raster::Rasterization;
use crate::pipeline::shader::EntryPointAbstract;
use crate::pipeline::shader::GraphicsEntryPoint;
use crate::pipeline::shader::GraphicsShaderType;
use crate::pipeline::shader::SpecializationConstants;
use crate::pipeline::vertex::BufferlessDefinition;
use crate::pipeline::vertex::BuffersDefinition;
use crate::pipeline::vertex::Vertex;
use crate::pipeline::vertex::VertexDefinition;
use crate::pipeline::vertex::VertexInputRate;
use crate::pipeline::viewport::Scissor;
use crate::pipeline::viewport::Viewport;
use crate::pipeline::viewport::ViewportsState;
use crate::render_pass::Subpass;
use crate::OomError;
use crate::VulkanObject;
use smallvec::SmallVec;
use std::mem;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;
use std::u32;

/// Prototype for a `GraphicsPipeline`.
// TODO: we can optimize this by filling directly the raw vk structs
pub struct GraphicsPipelineBuilder<'vs, 'tcs, 'tes, 'gs, 'fs, Vdef, Vss, Tcss, Tess, Gss, Fss> {
    vertex_definition: Vdef,
    vertex_shader: Option<(GraphicsEntryPoint<'vs>, Vss)>,
    input_assembly: ash::vk::PipelineInputAssemblyStateCreateInfo,
    // Note: the `input_assembly_topology` member is temporary in order to not lose information
    // about the number of patches per primitive.
    input_assembly_topology: PrimitiveTopology,
    tessellation: Option<TessInfo<'tcs, 'tes, Tcss, Tess>>,
    geometry_shader: Option<(GraphicsEntryPoint<'gs>, Gss)>,
    viewport: Option<ViewportsState>,
    raster: Rasterization,
    multisample: ash::vk::PipelineMultisampleStateCreateInfo,
    fragment_shader: Option<(GraphicsEntryPoint<'fs>, Fss)>,
    depth_stencil: DepthStencil,
    blend: Blend,
    subpass: Option<Subpass>,
    cache: Option<Arc<PipelineCache>>,
}

// Additional parameters if tessellation is used.
#[derive(Clone, Debug)]
struct TessInfo<'tcs, 'tes, Tcss, Tess> {
    tessellation_control_shader: (GraphicsEntryPoint<'tcs>, Tcss),
    tessellation_evaluation_shader: (GraphicsEntryPoint<'tes>, Tess),
}

impl
    GraphicsPipelineBuilder<
        'static,
        'static,
        'static,
        'static,
        'static,
        BufferlessDefinition,
        (),
        (),
        (),
        (),
        (),
    >
{
    /// Builds a new empty builder.
    pub(super) fn new() -> Self {
        GraphicsPipelineBuilder {
            vertex_definition: BufferlessDefinition,
            vertex_shader: None,
            input_assembly: ash::vk::PipelineInputAssemblyStateCreateInfo {
                topology: PrimitiveTopology::TriangleList.into(),
                ..Default::default()
            },
            input_assembly_topology: PrimitiveTopology::TriangleList,
            tessellation: None,
            geometry_shader: None,
            viewport: None,
            raster: Default::default(),
            multisample: ash::vk::PipelineMultisampleStateCreateInfo::default(),
            fragment_shader: None,
            depth_stencil: DepthStencil::disabled(),
            blend: Blend::pass_through(),
            subpass: None,
            cache: None,
        }
    }
}

impl<'vs, 'tcs, 'tes, 'gs, 'fs, Vdef, Vss, Tcss, Tess, Gss, Fss>
    GraphicsPipelineBuilder<'vs, 'tcs, 'tes, 'gs, 'fs, Vdef, Vss, Tcss, Tess, Gss, Fss>
where
    Vdef: VertexDefinition,
    Vss: SpecializationConstants,
    Tcss: SpecializationConstants,
    Tess: SpecializationConstants,
    Gss: SpecializationConstants,
    Fss: SpecializationConstants,
{
    /// Builds the graphics pipeline, using an inferred a pipeline layout.
    pub fn build(
        self,
        device: Arc<Device>,
    ) -> Result<GraphicsPipeline<Vdef>, GraphicsPipelineCreationError> {
        self.with_auto_layout(device, &[])
    }

    /// Builds the graphics pipeline, using an inferred pipeline layout with some dynamic buffers.
    ///
    /// Configures the inferred layout for each descriptor `(set, binding)` in `dynamic_buffers` to accept dynamic
    /// buffers.
    pub fn with_auto_layout(
        self,
        device: Arc<Device>,
        dynamic_buffers: &[(usize, usize)],
    ) -> Result<GraphicsPipeline<Vdef>, GraphicsPipelineCreationError> {
        let (descriptor_set_layout_descs, push_constant_ranges) = {
            let stages: SmallVec<[&GraphicsEntryPoint; 5]> = std::array::IntoIter::new([
                self.vertex_shader.as_ref().map(|s| &s.0),
                self.tessellation
                    .as_ref()
                    .map(|s| &s.tessellation_control_shader.0),
                self.tessellation
                    .as_ref()
                    .map(|s| &s.tessellation_evaluation_shader.0),
                self.geometry_shader.as_ref().map(|s| &s.0),
                self.fragment_shader.as_ref().map(|s| &s.0),
            ])
            .flatten()
            .collect();

            for (output, input) in stages.iter().zip(stages.iter().skip(1)) {
                if let Err(err) = input.input().matches(output.output()) {
                    return Err(GraphicsPipelineCreationError::ShaderStagesMismatch(err));
                }
            }

            let mut descriptor_set_layout_descs = stages
                .iter()
                .try_fold(vec![], |total, shader| -> Result<_, ()> {
                    DescriptorSetDesc::union_multiple(&total, shader.descriptor_set_layout_descs())
                })
                .expect("Can't be union'd");
            DescriptorSetDesc::tweak_multiple(
                &mut descriptor_set_layout_descs,
                dynamic_buffers.into_iter().cloned(),
            );

            let push_constant_ranges = stages.iter().fold(vec![], |total, shader| {
                PipelineLayoutPcRange::union_multiple(&total, shader.push_constant_ranges())
            });

            (descriptor_set_layout_descs, push_constant_ranges)
        };

        let descriptor_set_layouts = descriptor_set_layout_descs
            .into_iter()
            .map(|desc| Ok(Arc::new(DescriptorSetLayout::new(device.clone(), desc)?)))
            .collect::<Result<Vec<_>, OomError>>()?;
        let pipeline_layout = Arc::new(
            PipelineLayout::new(device.clone(), descriptor_set_layouts, push_constant_ranges)
                .unwrap(),
        );
        self.with_pipeline_layout(device, pipeline_layout)
    }

    /// Builds the graphics pipeline.
    ///
    /// Does the same as `build`, except that `build` automatically builds the pipeline layout
    /// object corresponding to the union of your shaders while this function allows you to specify
    /// the pipeline layout.
    pub fn with_pipeline_layout(
        mut self,
        device: Arc<Device>,
        pipeline_layout: Arc<PipelineLayout>,
    ) -> Result<GraphicsPipeline<Vdef>, GraphicsPipelineCreationError> {
        // TODO: return errors instead of panicking if missing param

        let fns = device.fns();

        // Checking that the pipeline layout matches the shader stages.
        // TODO: more details in the errors

        {
            let shader = &self.vertex_shader.as_ref().unwrap().0;
            pipeline_layout.ensure_superset_of(
                shader.descriptor_set_layout_descs(),
                shader.push_constant_ranges(),
            )?;
        }

        if let Some(ref geometry_shader) = self.geometry_shader {
            let shader = &geometry_shader.0;
            pipeline_layout.ensure_superset_of(
                shader.descriptor_set_layout_descs(),
                shader.push_constant_ranges(),
            )?;
        }

        if let Some(ref tess) = self.tessellation {
            {
                let shader = &tess.tessellation_control_shader.0;
                pipeline_layout.ensure_superset_of(
                    shader.descriptor_set_layout_descs(),
                    shader.push_constant_ranges(),
                )?;
            }

            {
                let shader = &tess.tessellation_evaluation_shader.0;
                pipeline_layout.ensure_superset_of(
                    shader.descriptor_set_layout_descs(),
                    shader.push_constant_ranges(),
                )?;
            }
        }

        if let Some(ref fragment_shader) = self.fragment_shader {
            let shader = &fragment_shader.0;
            pipeline_layout.ensure_superset_of(
                shader.descriptor_set_layout_descs(),
                shader.push_constant_ranges(),
            )?;

            // Check that the subpass can accept the output of the fragment shader.
            // TODO: If there is no fragment shader, what should be checked then? The previous stage?
            if !self
                .subpass
                .as_ref()
                .unwrap()
                .is_compatible_with(shader.output())
            {
                return Err(GraphicsPipelineCreationError::FragmentShaderRenderPassIncompatible);
            }
        }

        // Will contain the list of dynamic states. Filled throughout this function.
        let mut dynamic_states: SmallVec<[ash::vk::DynamicState; 8]> = SmallVec::new();

        // Creating the specialization constants of the various stages.
        let vertex_shader_specialization = {
            let shader = self.vertex_shader.as_ref().unwrap();
            let spec_descriptors = Vss::descriptors();
            if spec_descriptors != shader.0.spec_constants() {
                return Err(GraphicsPipelineCreationError::IncompatibleSpecializationConstants);
            }

            let constants = &shader.1;
            ash::vk::SpecializationInfo {
                map_entry_count: spec_descriptors.len() as u32,
                p_map_entries: spec_descriptors.as_ptr() as *const _,
                data_size: mem::size_of_val(constants),
                p_data: constants as *const Vss as *const _,
            }
        };

        let tess_shader_specialization = if let Some(ref tess) = self.tessellation {
            let tcs_spec = {
                let shader = &tess.tessellation_control_shader;
                let spec_descriptors = Tcss::descriptors();
                if spec_descriptors != shader.0.spec_constants() {
                    return Err(GraphicsPipelineCreationError::IncompatibleSpecializationConstants);
                }

                let constants = &shader.1;
                ash::vk::SpecializationInfo {
                    map_entry_count: spec_descriptors.len() as u32,
                    p_map_entries: spec_descriptors.as_ptr() as *const _,
                    data_size: mem::size_of_val(constants),
                    p_data: constants as *const Tcss as *const _,
                }
            };
            let tes_spec = {
                let shader = &tess.tessellation_evaluation_shader;
                let spec_descriptors = Tess::descriptors();
                if spec_descriptors != shader.0.spec_constants() {
                    return Err(GraphicsPipelineCreationError::IncompatibleSpecializationConstants);
                }

                let constants = &shader.1;
                ash::vk::SpecializationInfo {
                    map_entry_count: spec_descriptors.len() as u32,
                    p_map_entries: spec_descriptors.as_ptr() as *const _,
                    data_size: mem::size_of_val(constants),
                    p_data: constants as *const Tess as *const _,
                }
            };
            Some((tcs_spec, tes_spec))
        } else {
            None
        };

        let geometry_shader_specialization = if let Some(ref shader) = self.geometry_shader {
            let spec_descriptors = Gss::descriptors();
            if spec_descriptors != shader.0.spec_constants() {
                return Err(GraphicsPipelineCreationError::IncompatibleSpecializationConstants);
            }

            let constants = &shader.1;
            Some(ash::vk::SpecializationInfo {
                map_entry_count: spec_descriptors.len() as u32,
                p_map_entries: spec_descriptors.as_ptr() as *const _,
                data_size: mem::size_of_val(constants),
                p_data: constants as *const Gss as *const _,
            })
        } else {
            None
        };

        let fragment_shader_specialization = if let Some(ref shader) = self.fragment_shader {
            let spec_descriptors = Fss::descriptors();
            if spec_descriptors != shader.0.spec_constants() {
                return Err(GraphicsPipelineCreationError::IncompatibleSpecializationConstants);
            }

            let constants = &shader.1;
            Some(ash::vk::SpecializationInfo {
                map_entry_count: spec_descriptors.len() as u32,
                p_map_entries: spec_descriptors.as_ptr() as *const _,
                data_size: mem::size_of_val(constants),
                p_data: constants as *const Fss as *const _,
            })
        } else {
            None
        };

        // List of shader stages.
        let stages = {
            let mut stages = SmallVec::<[_; 5]>::new();

            match self.vertex_shader.as_ref().unwrap().0.ty() {
                GraphicsShaderType::Vertex => {}
                _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
            };

            stages.push(ash::vk::PipelineShaderStageCreateInfo {
                flags: ash::vk::PipelineShaderStageCreateFlags::empty(),
                stage: ash::vk::ShaderStageFlags::VERTEX,
                module: self
                    .vertex_shader
                    .as_ref()
                    .unwrap()
                    .0
                    .module()
                    .internal_object(),
                p_name: self.vertex_shader.as_ref().unwrap().0.name().as_ptr(),
                p_specialization_info: &vertex_shader_specialization as *const _,
                ..Default::default()
            });

            if let Some(ref tess) = self.tessellation {
                // FIXME: must check that the control shader and evaluation shader are compatible

                if !device.enabled_features().tessellation_shader {
                    return Err(GraphicsPipelineCreationError::TessellationShaderFeatureNotEnabled);
                }

                match tess.tessellation_control_shader.0.ty() {
                    GraphicsShaderType::TessellationControl => {}
                    _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
                };

                match tess.tessellation_evaluation_shader.0.ty() {
                    GraphicsShaderType::TessellationEvaluation => {}
                    _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
                };

                stages.push(ash::vk::PipelineShaderStageCreateInfo {
                    flags: ash::vk::PipelineShaderStageCreateFlags::empty(),
                    stage: ash::vk::ShaderStageFlags::TESSELLATION_CONTROL,
                    module: tess
                        .tessellation_control_shader
                        .0
                        .module()
                        .internal_object(),
                    p_name: tess.tessellation_control_shader.0.name().as_ptr(),
                    p_specialization_info: &tess_shader_specialization.as_ref().unwrap().0
                        as *const _,
                    ..Default::default()
                });

                stages.push(ash::vk::PipelineShaderStageCreateInfo {
                    flags: ash::vk::PipelineShaderStageCreateFlags::empty(),
                    stage: ash::vk::ShaderStageFlags::TESSELLATION_EVALUATION,
                    module: tess
                        .tessellation_evaluation_shader
                        .0
                        .module()
                        .internal_object(),
                    p_name: tess.tessellation_evaluation_shader.0.name().as_ptr(),
                    p_specialization_info: &tess_shader_specialization.as_ref().unwrap().1
                        as *const _,
                    ..Default::default()
                });
            }

            if let Some(ref geometry_shader) = self.geometry_shader {
                if !device.enabled_features().geometry_shader {
                    return Err(GraphicsPipelineCreationError::GeometryShaderFeatureNotEnabled);
                }

                match geometry_shader.0.ty() {
                    GraphicsShaderType::Geometry(_) => {}
                    _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
                };

                stages.push(ash::vk::PipelineShaderStageCreateInfo {
                    flags: ash::vk::PipelineShaderStageCreateFlags::empty(),
                    stage: ash::vk::ShaderStageFlags::GEOMETRY,
                    module: geometry_shader.0.module().internal_object(),
                    p_name: geometry_shader.0.name().as_ptr(),
                    p_specialization_info: geometry_shader_specialization.as_ref().unwrap()
                        as *const _,
                    ..Default::default()
                });
            }

            if let Some(ref fragment_shader) = self.fragment_shader {
                match fragment_shader.0.ty() {
                    GraphicsShaderType::Fragment => {}
                    _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
                };

                stages.push(ash::vk::PipelineShaderStageCreateInfo {
                    flags: ash::vk::PipelineShaderStageCreateFlags::empty(),
                    stage: ash::vk::ShaderStageFlags::FRAGMENT,
                    module: fragment_shader.0.module().internal_object(),
                    p_name: fragment_shader.0.name().as_ptr(),
                    p_specialization_info: fragment_shader_specialization.as_ref().unwrap()
                        as *const _,
                    ..Default::default()
                });
            }

            stages
        };

        // Vertex input.
        let vertex_input = self
            .vertex_definition
            .definition(self.vertex_shader.as_ref().unwrap().0.input())?;

        let (binding_descriptions, binding_divisor_descriptions) = {
            let mut binding_descriptions = SmallVec::<[_; 8]>::new();
            let mut binding_divisor_descriptions = SmallVec::<[_; 8]>::new();

            for (binding, binding_desc) in vertex_input.bindings() {
                if binding_desc.stride
                    > device
                        .physical_device()
                        .properties()
                        .max_vertex_input_binding_stride
                        .unwrap()
                {
                    return Err(
                        GraphicsPipelineCreationError::MaxVertexInputBindingStrideExceeded {
                            binding,
                            max: device
                                .physical_device()
                                .properties()
                                .max_vertex_input_binding_stride
                                .unwrap(),
                            obtained: binding_desc.stride,
                        },
                    );
                }

                binding_descriptions.push(ash::vk::VertexInputBindingDescription {
                    binding,
                    stride: binding_desc.stride,
                    input_rate: binding_desc.input_rate.into(),
                });

                if let VertexInputRate::Instance { divisor } = binding_desc.input_rate {
                    if divisor != 1 {
                        if !device
                            .enabled_features()
                            .vertex_attribute_instance_rate_divisor
                        {
                            return Err(GraphicsPipelineCreationError::VertexAttributeInstanceRateDivisorFeatureNotEnabled);
                        }

                        if divisor == 0
                            && !device
                                .enabled_features()
                                .vertex_attribute_instance_rate_zero_divisor
                        {
                            return Err(GraphicsPipelineCreationError::VertexAttributeInstanceRateZeroDivisorFeatureNotEnabled);
                        }

                        if divisor
                            > device
                                .physical_device()
                                .properties()
                                .max_vertex_attrib_divisor
                                .unwrap()
                        {
                            return Err(
                                GraphicsPipelineCreationError::MaxVertexAttribDivisorExceeded {
                                    binding,
                                    max: device
                                        .physical_device()
                                        .properties()
                                        .max_vertex_attrib_divisor
                                        .unwrap(),
                                    obtained: divisor,
                                },
                            );
                        }

                        binding_divisor_descriptions.push(
                            ash::vk::VertexInputBindingDivisorDescriptionEXT { binding, divisor },
                        )
                    }
                }
            }

            if binding_descriptions.len()
                > device
                    .physical_device()
                    .properties()
                    .max_vertex_input_bindings
                    .unwrap() as usize
            {
                return Err(
                    GraphicsPipelineCreationError::MaxVertexInputBindingsExceeded {
                        max: device
                            .physical_device()
                            .properties()
                            .max_vertex_input_bindings
                            .unwrap(),
                        obtained: binding_descriptions.len(),
                    },
                );
            }

            (binding_descriptions, binding_divisor_descriptions)
        };

        let attribute_descriptions = {
            let mut attribute_descriptions = SmallVec::<[_; 8]>::new();

            for (location, attribute_desc) in vertex_input.attributes() {
                // TODO: check attribute format support

                if attribute_desc.offset
                    > device
                        .physical_device()
                        .properties()
                        .max_vertex_input_attribute_offset
                        .unwrap()
                {
                    return Err(
                        GraphicsPipelineCreationError::MaxVertexInputAttributeOffsetExceeded {
                            max: device
                                .physical_device()
                                .properties()
                                .max_vertex_input_attribute_offset
                                .unwrap(),
                            obtained: attribute_desc.offset,
                        },
                    );
                }

                attribute_descriptions.push(ash::vk::VertexInputAttributeDescription {
                    location,
                    binding: attribute_desc.binding,
                    format: attribute_desc.format.into(),
                    offset: attribute_desc.offset,
                });
            }

            if attribute_descriptions.len()
                > device
                    .physical_device()
                    .properties()
                    .max_vertex_input_attributes
                    .unwrap() as usize
            {
                return Err(
                    GraphicsPipelineCreationError::MaxVertexInputAttributesExceeded {
                        max: device
                            .physical_device()
                            .properties()
                            .max_vertex_input_attributes
                            .unwrap(),
                        obtained: attribute_descriptions.len(),
                    },
                );
            }

            attribute_descriptions
        };

        let vertex_input_divisor_state = if !binding_divisor_descriptions.is_empty() {
            Some(ash::vk::PipelineVertexInputDivisorStateCreateInfoEXT {
                vertex_binding_divisor_count: binding_divisor_descriptions.len() as u32,
                p_vertex_binding_divisors: binding_divisor_descriptions.as_ptr(),
                ..Default::default()
            })
        } else {
            None
        };

        let vertex_input_state = ash::vk::PipelineVertexInputStateCreateInfo {
            p_next: if let Some(next) = vertex_input_divisor_state.as_ref() {
                next as *const _ as *const _
            } else {
                ptr::null()
            },
            flags: ash::vk::PipelineVertexInputStateCreateFlags::empty(),
            vertex_binding_description_count: binding_descriptions.len() as u32,
            p_vertex_binding_descriptions: binding_descriptions.as_ptr(),
            vertex_attribute_description_count: attribute_descriptions.len() as u32,
            p_vertex_attribute_descriptions: attribute_descriptions.as_ptr(),
            ..Default::default()
        };

        if self.input_assembly.primitive_restart_enable != ash::vk::FALSE
            && !self.input_assembly_topology.supports_primitive_restart()
        {
            return Err(
                GraphicsPipelineCreationError::PrimitiveDoesntSupportPrimitiveRestart {
                    primitive: self.input_assembly_topology,
                },
            );
        }

        // TODO: should check from the tess eval shader instead of the input assembly
        if let Some(ref gs) = self.geometry_shader {
            match gs.0.ty() {
                GraphicsShaderType::Geometry(primitives) => {
                    if !primitives.matches(self.input_assembly_topology) {
                        return Err(
                            GraphicsPipelineCreationError::TopologyNotMatchingGeometryShader,
                        );
                    }
                }
                _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
            }
        }

        let tessellation = match self.input_assembly_topology {
            PrimitiveTopology::PatchList { vertices_per_patch } => {
                if self.tessellation.is_none() {
                    return Err(GraphicsPipelineCreationError::InvalidPrimitiveTopology);
                }
                if vertices_per_patch
                    > device
                        .physical_device()
                        .properties()
                        .max_tessellation_patch_size
                        .unwrap()
                {
                    return Err(GraphicsPipelineCreationError::MaxTessellationPatchSizeExceeded);
                }

                Some(ash::vk::PipelineTessellationStateCreateInfo {
                    flags: ash::vk::PipelineTessellationStateCreateFlags::empty(),
                    patch_control_points: vertices_per_patch,
                    ..Default::default()
                })
            }
            _ => {
                if self.tessellation.is_some() {
                    return Err(GraphicsPipelineCreationError::InvalidPrimitiveTopology);
                }

                None
            }
        };

        let (vp_vp, vp_sc, vp_num) = match *self.viewport.as_ref().unwrap() {
            ViewportsState::Fixed { ref data } => (
                data.iter()
                    .map(|e| e.0.clone().into())
                    .collect::<SmallVec<[ash::vk::Viewport; 4]>>(),
                data.iter()
                    .map(|e| e.1.clone().into())
                    .collect::<SmallVec<[ash::vk::Rect2D; 4]>>(),
                data.len() as u32,
            ),
            ViewportsState::DynamicViewports { ref scissors } => {
                let num = scissors.len() as u32;
                let scissors = scissors
                    .iter()
                    .map(|e| e.clone().into())
                    .collect::<SmallVec<[ash::vk::Rect2D; 4]>>();
                dynamic_states.push(ash::vk::DynamicState::VIEWPORT);
                (SmallVec::new(), scissors, num)
            }
            ViewportsState::DynamicScissors { ref viewports } => {
                let num = viewports.len() as u32;
                let viewports = viewports
                    .iter()
                    .map(|e| e.clone().into())
                    .collect::<SmallVec<[ash::vk::Viewport; 4]>>();
                dynamic_states.push(ash::vk::DynamicState::SCISSOR);
                (viewports, SmallVec::new(), num)
            }
            ViewportsState::Dynamic { num } => {
                dynamic_states.push(ash::vk::DynamicState::VIEWPORT);
                dynamic_states.push(ash::vk::DynamicState::SCISSOR);
                (SmallVec::new(), SmallVec::new(), num)
            }
        };

        if vp_num > 1 && !device.enabled_features().multi_viewport {
            return Err(GraphicsPipelineCreationError::MultiViewportFeatureNotEnabled);
        }

        if vp_num > device.physical_device().properties().max_viewports.unwrap() {
            return Err(GraphicsPipelineCreationError::MaxViewportsExceeded {
                obtained: vp_num,
                max: device.physical_device().properties().max_viewports.unwrap(),
            });
        }

        for vp in vp_vp.iter() {
            if vp.width
                > device
                    .physical_device()
                    .properties()
                    .max_viewport_dimensions
                    .unwrap()[0] as f32
                || vp.height
                    > device
                        .physical_device()
                        .properties()
                        .max_viewport_dimensions
                        .unwrap()[1] as f32
            {
                return Err(GraphicsPipelineCreationError::MaxViewportDimensionsExceeded);
            }

            if vp.x
                < device
                    .physical_device()
                    .properties()
                    .viewport_bounds_range
                    .unwrap()[0]
                || vp.x + vp.width
                    > device
                        .physical_device()
                        .properties()
                        .viewport_bounds_range
                        .unwrap()[1]
                || vp.y
                    < device
                        .physical_device()
                        .properties()
                        .viewport_bounds_range
                        .unwrap()[0]
                || vp.y + vp.height
                    > device
                        .physical_device()
                        .properties()
                        .viewport_bounds_range
                        .unwrap()[1]
            {
                return Err(GraphicsPipelineCreationError::ViewportBoundsExceeded);
            }
        }

        let viewport_info = ash::vk::PipelineViewportStateCreateInfo {
            flags: ash::vk::PipelineViewportStateCreateFlags::empty(),
            viewport_count: vp_num,
            p_viewports: if vp_vp.is_empty() {
                ptr::null()
            } else {
                vp_vp.as_ptr()
            }, // validation layer crashes if you just pass the pointer
            scissor_count: vp_num,
            p_scissors: if vp_sc.is_empty() {
                ptr::null()
            } else {
                vp_sc.as_ptr()
            }, // validation layer crashes if you just pass the pointer
            ..Default::default()
        };

        if let Some(line_width) = self.raster.line_width {
            if line_width != 1.0 && !device.enabled_features().wide_lines {
                return Err(GraphicsPipelineCreationError::WideLinesFeatureNotEnabled);
            }
        } else {
            dynamic_states.push(ash::vk::DynamicState::LINE_WIDTH);
        }

        let (db_enable, db_const, db_clamp, db_slope) = match self.raster.depth_bias {
            DepthBiasControl::Dynamic => {
                dynamic_states.push(ash::vk::DynamicState::DEPTH_BIAS);
                (ash::vk::TRUE, 0.0, 0.0, 0.0)
            }
            DepthBiasControl::Disabled => (ash::vk::FALSE, 0.0, 0.0, 0.0),
            DepthBiasControl::Static(bias) => {
                if bias.clamp != 0.0 && !device.enabled_features().depth_bias_clamp {
                    return Err(GraphicsPipelineCreationError::DepthBiasClampFeatureNotEnabled);
                }

                (
                    ash::vk::TRUE,
                    bias.constant_factor,
                    bias.clamp,
                    bias.slope_factor,
                )
            }
        };

        if self.raster.depth_clamp && !device.enabled_features().depth_clamp {
            return Err(GraphicsPipelineCreationError::DepthClampFeatureNotEnabled);
        }

        if self.raster.polygon_mode != PolygonMode::Fill
            && !device.enabled_features().fill_mode_non_solid
        {
            return Err(GraphicsPipelineCreationError::FillModeNonSolidFeatureNotEnabled);
        }

        let rasterization = ash::vk::PipelineRasterizationStateCreateInfo {
            flags: ash::vk::PipelineRasterizationStateCreateFlags::empty(),
            depth_clamp_enable: if self.raster.depth_clamp {
                ash::vk::TRUE
            } else {
                ash::vk::FALSE
            },
            rasterizer_discard_enable: if self.raster.rasterizer_discard {
                ash::vk::TRUE
            } else {
                ash::vk::FALSE
            },
            polygon_mode: self.raster.polygon_mode.into(),
            cull_mode: self.raster.cull_mode.into(),
            front_face: self.raster.front_face.into(),
            depth_bias_enable: db_enable,
            depth_bias_constant_factor: db_const,
            depth_bias_clamp: db_clamp,
            depth_bias_slope_factor: db_slope,
            line_width: self.raster.line_width.unwrap_or(1.0),
            ..Default::default()
        };

        self.multisample.rasterization_samples = self
            .subpass
            .as_ref()
            .unwrap()
            .num_samples()
            .unwrap_or(SampleCount::Sample1)
            .into();
        if self.multisample.sample_shading_enable != ash::vk::FALSE {
            debug_assert!(
                self.multisample.min_sample_shading >= 0.0
                    && self.multisample.min_sample_shading <= 1.0
            );
            if !device.enabled_features().sample_rate_shading {
                return Err(GraphicsPipelineCreationError::SampleRateShadingFeatureNotEnabled);
            }
        }
        if self.multisample.alpha_to_one_enable != ash::vk::FALSE {
            if !device.enabled_features().alpha_to_one {
                return Err(GraphicsPipelineCreationError::AlphaToOneFeatureNotEnabled);
            }
        }

        let depth_stencil = {
            let db = match self.depth_stencil.depth_bounds_test {
                DepthBounds::Disabled => (ash::vk::FALSE, 0.0, 0.0),
                DepthBounds::Fixed(ref range) => {
                    if !device.enabled_features().depth_bounds {
                        return Err(GraphicsPipelineCreationError::DepthBoundsFeatureNotEnabled);
                    }

                    (ash::vk::TRUE, range.start, range.end)
                }
                DepthBounds::Dynamic => {
                    if !device.enabled_features().depth_bounds {
                        return Err(GraphicsPipelineCreationError::DepthBoundsFeatureNotEnabled);
                    }

                    dynamic_states.push(ash::vk::DynamicState::DEPTH_BOUNDS);

                    (ash::vk::TRUE, 0.0, 1.0)
                }
            };

            match (
                self.depth_stencil.stencil_front.compare_mask,
                self.depth_stencil.stencil_back.compare_mask,
            ) {
                (Some(_), Some(_)) => (),
                (None, None) => {
                    dynamic_states.push(ash::vk::DynamicState::STENCIL_COMPARE_MASK);
                }
                _ => return Err(GraphicsPipelineCreationError::WrongStencilState),
            };

            match (
                self.depth_stencil.stencil_front.write_mask,
                self.depth_stencil.stencil_back.write_mask,
            ) {
                (Some(_), Some(_)) => (),
                (None, None) => {
                    dynamic_states.push(ash::vk::DynamicState::STENCIL_WRITE_MASK);
                }
                _ => return Err(GraphicsPipelineCreationError::WrongStencilState),
            };

            match (
                self.depth_stencil.stencil_front.reference,
                self.depth_stencil.stencil_back.reference,
            ) {
                (Some(_), Some(_)) => (),
                (None, None) => {
                    dynamic_states.push(ash::vk::DynamicState::STENCIL_REFERENCE);
                }
                _ => return Err(GraphicsPipelineCreationError::WrongStencilState),
            };

            if self.depth_stencil.depth_write
                && !self.subpass.as_ref().unwrap().has_writable_depth()
            {
                return Err(GraphicsPipelineCreationError::NoDepthAttachment);
            }

            if self.depth_stencil.depth_compare != Compare::Always
                && !self.subpass.as_ref().unwrap().has_depth()
            {
                return Err(GraphicsPipelineCreationError::NoDepthAttachment);
            }

            if (!self.depth_stencil.stencil_front.always_keep()
                || !self.depth_stencil.stencil_back.always_keep())
                && !self.subpass.as_ref().unwrap().has_stencil()
            {
                return Err(GraphicsPipelineCreationError::NoStencilAttachment);
            }

            // FIXME: stencil writability

            ash::vk::PipelineDepthStencilStateCreateInfo {
                flags: ash::vk::PipelineDepthStencilStateCreateFlags::empty(),
                depth_test_enable: if !self.depth_stencil.depth_write
                    && self.depth_stencil.depth_compare == Compare::Always
                {
                    ash::vk::FALSE
                } else {
                    ash::vk::TRUE
                },
                depth_write_enable: if self.depth_stencil.depth_write {
                    ash::vk::TRUE
                } else {
                    ash::vk::FALSE
                },
                depth_compare_op: self.depth_stencil.depth_compare.into(),
                depth_bounds_test_enable: db.0,
                stencil_test_enable: if self.depth_stencil.stencil_front.always_keep()
                    && self.depth_stencil.stencil_back.always_keep()
                {
                    ash::vk::FALSE
                } else {
                    ash::vk::TRUE
                },
                front: ash::vk::StencilOpState {
                    fail_op: self.depth_stencil.stencil_front.fail_op.into(),
                    pass_op: self.depth_stencil.stencil_front.pass_op.into(),
                    depth_fail_op: self.depth_stencil.stencil_front.depth_fail_op.into(),
                    compare_op: self.depth_stencil.stencil_front.compare.into(),
                    compare_mask: self
                        .depth_stencil
                        .stencil_front
                        .compare_mask
                        .unwrap_or(u32::MAX),
                    write_mask: self
                        .depth_stencil
                        .stencil_front
                        .write_mask
                        .unwrap_or(u32::MAX),
                    reference: self.depth_stencil.stencil_front.reference.unwrap_or(0),
                },
                back: ash::vk::StencilOpState {
                    fail_op: self.depth_stencil.stencil_back.fail_op.into(),
                    pass_op: self.depth_stencil.stencil_back.pass_op.into(),
                    depth_fail_op: self.depth_stencil.stencil_back.depth_fail_op.into(),
                    compare_op: self.depth_stencil.stencil_back.compare.into(),
                    compare_mask: self
                        .depth_stencil
                        .stencil_back
                        .compare_mask
                        .unwrap_or(u32::MAX),
                    write_mask: self
                        .depth_stencil
                        .stencil_back
                        .write_mask
                        .unwrap_or(u32::MAX),
                    reference: self.depth_stencil.stencil_back.reference.unwrap_or(0),
                },
                min_depth_bounds: db.1,
                max_depth_bounds: db.2,
                ..Default::default()
            }
        };

        let blend_atch: SmallVec<[ash::vk::PipelineColorBlendAttachmentState; 8]> = {
            let num_atch = self.subpass.as_ref().unwrap().num_color_attachments();

            match self.blend.attachments {
                AttachmentsBlend::Collective(blend) => {
                    (0..num_atch).map(|_| blend.clone().into()).collect()
                }
                AttachmentsBlend::Individual(blend) => {
                    if blend.len() != num_atch as usize {
                        return Err(
                            GraphicsPipelineCreationError::MismatchBlendingAttachmentsCount,
                        );
                    }

                    if !device.enabled_features().independent_blend {
                        return Err(
                            GraphicsPipelineCreationError::IndependentBlendFeatureNotEnabled,
                        );
                    }

                    blend.iter().map(|b| b.clone().into()).collect()
                }
            }
        };

        let blend = ash::vk::PipelineColorBlendStateCreateInfo {
            flags: ash::vk::PipelineColorBlendStateCreateFlags::empty(),
            logic_op_enable: if self.blend.logic_op.is_some() {
                if !device.enabled_features().logic_op {
                    return Err(GraphicsPipelineCreationError::LogicOpFeatureNotEnabled);
                }
                ash::vk::TRUE
            } else {
                ash::vk::FALSE
            },
            logic_op: self.blend.logic_op.unwrap_or(Default::default()).into(),
            attachment_count: blend_atch.len() as u32,
            p_attachments: blend_atch.as_ptr(),
            blend_constants: if let Some(c) = self.blend.blend_constants {
                c
            } else {
                dynamic_states.push(ash::vk::DynamicState::BLEND_CONSTANTS);
                [0.0, 0.0, 0.0, 0.0]
            },
            ..Default::default()
        };

        let dynamic_states = if !dynamic_states.is_empty() {
            Some(ash::vk::PipelineDynamicStateCreateInfo {
                flags: ash::vk::PipelineDynamicStateCreateFlags::empty(),
                dynamic_state_count: dynamic_states.len() as u32,
                p_dynamic_states: dynamic_states.as_ptr(),
                ..Default::default()
            })
        } else {
            None
        };

        if let Some(multiview) = self
            .subpass
            .as_ref()
            .unwrap()
            .render_pass()
            .desc()
            .multiview()
            .as_ref()
        {
            if multiview.used_layer_count() > 0 {
                if self.geometry_shader.is_some()
                    && !device
                        .physical_device()
                        .supported_features()
                        .multiview_geometry_shader
                {
                    return Err(GraphicsPipelineCreationError::MultiviewGeometryShaderNotSupported);
                }

                if self.tessellation.is_some()
                    && !device
                        .physical_device()
                        .supported_features()
                        .multiview_tessellation_shader
                {
                    return Err(
                        GraphicsPipelineCreationError::MultiviewTessellationShaderNotSupported,
                    );
                }
            }
        }

        let pipeline = unsafe {
            let infos = ash::vk::GraphicsPipelineCreateInfo {
                flags: ash::vk::PipelineCreateFlags::empty(), // TODO: some flags are available but none are critical
                stage_count: stages.len() as u32,
                p_stages: stages.as_ptr(),
                p_vertex_input_state: &vertex_input_state,
                p_input_assembly_state: &self.input_assembly,
                p_tessellation_state: tessellation
                    .as_ref()
                    .map(|t| t as *const _)
                    .unwrap_or(ptr::null()),
                p_viewport_state: &viewport_info,
                p_rasterization_state: &rasterization,
                p_multisample_state: &self.multisample,
                p_depth_stencil_state: &depth_stencil,
                p_color_blend_state: &blend,
                p_dynamic_state: dynamic_states
                    .as_ref()
                    .map(|s| s as *const _)
                    .unwrap_or(ptr::null()),
                layout: pipeline_layout.internal_object(),
                render_pass: self
                    .subpass
                    .as_ref()
                    .unwrap()
                    .render_pass()
                    .inner()
                    .internal_object(),
                subpass: self.subpass.as_ref().unwrap().index(),
                base_pipeline_handle: ash::vk::Pipeline::null(), // TODO:
                base_pipeline_index: -1,                         // TODO:
                ..Default::default()
            };

            let cache_handle = match self.cache {
                Some(cache) => cache.internal_object(),
                None => ash::vk::PipelineCache::null(),
            };

            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_graphics_pipelines(
                device.internal_object(),
                cache_handle,
                1,
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        // Some drivers return `VK_SUCCESS` but provide a null handle if they
        // fail to create the pipeline (due to invalid shaders, etc)
        // This check ensures that we don't create an invalid `GraphicsPipeline` instance
        if pipeline == ash::vk::Pipeline::null() {
            panic!("vkCreateGraphicsPipelines provided a NULL handle");
        }

        Ok(GraphicsPipeline {
            inner: GraphicsPipelineInner {
                device: device.clone(),
                pipeline,
            },
            layout: pipeline_layout,
            subpass: self.subpass.take().unwrap(),
            vertex_definition: self.vertex_definition,
            vertex_input,

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

impl<'vs, 'tcs, 'tes, 'gs, 'fs, Vdef, Vss, Tcss, Tess, Gss, Fss>
    GraphicsPipelineBuilder<'vs, 'tcs, 'tes, 'gs, 'fs, Vdef, Vss, Tcss, Tess, Gss, Fss>
{
    // TODO: add pipeline derivate system

    /// Sets the vertex input.
    #[inline]
    pub fn vertex_input<T>(
        self,
        vertex_definition: T,
    ) -> GraphicsPipelineBuilder<'vs, 'tcs, 'tes, 'gs, 'fs, T, Vss, Tcss, Tess, Gss, Fss> {
        GraphicsPipelineBuilder {
            vertex_definition,
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
            subpass: self.subpass,
            cache: self.cache,
        }
    }

    /// Sets the vertex input to a single vertex buffer.
    ///
    /// You will most likely need to explicitly specify the template parameter to the type of a
    /// vertex.
    #[inline]
    pub fn vertex_input_single_buffer<V: Vertex>(
        self,
    ) -> GraphicsPipelineBuilder<
        'vs,
        'tcs,
        'tes,
        'gs,
        'fs,
        BuffersDefinition,
        Vss,
        Tcss,
        Tess,
        Gss,
        Fss,
    > {
        self.vertex_input(BuffersDefinition::new().vertex::<V>())
    }

    /// Sets the vertex shader to use.
    // TODO: correct specialization constants
    #[inline]
    pub fn vertex_shader<'vs2, Vss2>(
        self,
        shader: GraphicsEntryPoint<'vs2>,
        specialization_constants: Vss2,
    ) -> GraphicsPipelineBuilder<'vs2, 'tcs, 'tes, 'gs, 'fs, Vdef, Vss2, Tcss, Tess, Gss, Fss>
    where
        Vss2: SpecializationConstants,
    {
        GraphicsPipelineBuilder {
            vertex_definition: self.vertex_definition,
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
            subpass: self.subpass,
            cache: self.cache,
        }
    }

    /// Sets whether primitive restart if enabled.
    #[inline]
    pub fn primitive_restart(mut self, enabled: bool) -> Self {
        self.input_assembly.primitive_restart_enable = if enabled {
            ash::vk::TRUE
        } else {
            ash::vk::FALSE
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
    pub fn tessellation_shaders<'tcs2, 'tes2, Tcss2, Tess2>(
        self,
        tessellation_control_shader: GraphicsEntryPoint<'tcs2>,
        tessellation_control_shader_spec_constants: Tcss2,
        tessellation_evaluation_shader: GraphicsEntryPoint<'tes2>,
        tessellation_evaluation_shader_spec_constants: Tess2,
    ) -> GraphicsPipelineBuilder<'vs, 'tcs2, 'tes2, 'gs, 'fs, Vdef, Vss, Tcss2, Tess2, Gss, Fss>
    where
        Tcss2: SpecializationConstants,
        Tess2: SpecializationConstants,
    {
        GraphicsPipelineBuilder {
            vertex_definition: self.vertex_definition,
            vertex_shader: self.vertex_shader,
            input_assembly: self.input_assembly,
            input_assembly_topology: self.input_assembly_topology,
            tessellation: Some(TessInfo {
                tessellation_control_shader: (
                    tessellation_control_shader,
                    tessellation_control_shader_spec_constants,
                ),
                tessellation_evaluation_shader: (
                    tessellation_evaluation_shader,
                    tessellation_evaluation_shader_spec_constants,
                ),
            }),
            geometry_shader: self.geometry_shader,
            viewport: self.viewport,
            raster: self.raster,
            multisample: self.multisample,
            fragment_shader: self.fragment_shader,
            depth_stencil: self.depth_stencil,
            blend: self.blend,
            subpass: self.subpass,
            cache: self.cache,
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
    pub fn geometry_shader<'gs2, Gss2>(
        self,
        shader: GraphicsEntryPoint<'gs2>,
        specialization_constants: Gss2,
    ) -> GraphicsPipelineBuilder<'vs, 'tcs, 'tes, 'gs2, 'fs, Vdef, Vss, Tcss, Tess, Gss2, Fss>
    where
        Gss2: SpecializationConstants,
    {
        GraphicsPipelineBuilder {
            vertex_definition: self.vertex_definition,
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
            subpass: self.subpass,
            cache: self.cache,
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
    where
        I: IntoIterator<Item = Viewport>,
    {
        self.viewports_scissors(viewports.into_iter().map(|v| (v, Scissor::irrelevant())))
    }

    /// Sets the characteristics of viewports and scissor boxes in advance.
    #[inline]
    pub fn viewports_scissors<I>(mut self, viewports: I) -> Self
    where
        I: IntoIterator<Item = (Viewport, Scissor)>,
    {
        self.viewport = Some(ViewportsState::Fixed {
            data: viewports.into_iter().collect(),
        });
        self
    }

    /// Sets the scissor boxes to some values, and viewports to dynamic. The viewports will
    /// need to be set before drawing.
    #[inline]
    pub fn viewports_dynamic_scissors_fixed<I>(mut self, scissors: I) -> Self
    where
        I: IntoIterator<Item = Scissor>,
    {
        self.viewport = Some(ViewportsState::DynamicViewports {
            scissors: scissors.into_iter().collect(),
        });
        self
    }

    /// Sets the viewports to dynamic, and the scissor boxes to boxes that always cover the whole
    /// viewport. The viewports will need to be set before drawing.
    #[inline]
    pub fn viewports_dynamic_scissors_irrelevant(mut self, num: u32) -> Self {
        self.viewport = Some(ViewportsState::DynamicViewports {
            scissors: (0..num).map(|_| Scissor::irrelevant()).collect(),
        });
        self
    }

    /// Sets the viewports to some values, and scissor boxes to dynamic. The scissor boxes will
    /// need to be set before drawing.
    #[inline]
    pub fn viewports_fixed_scissors_dynamic<I>(mut self, viewports: I) -> Self
    where
        I: IntoIterator<Item = Viewport>,
    {
        self.viewport = Some(ViewportsState::DynamicScissors {
            viewports: viewports.into_iter().collect(),
        });
        self
    }

    /// Sets the viewports and scissor boxes to dynamic. They will both need to be set before
    /// drawing.
    #[inline]
    pub fn viewports_scissors_dynamic(mut self, num: u32) -> Self {
        self.viewport = Some(ViewportsState::Dynamic { num });
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

    /// Sets the front-facing faces to counter-clockwise faces. This is the default.
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
        self.multisample.sample_shading_enable = ash::vk::FALSE;
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
        self.multisample.sample_shading_enable = ash::vk::TRUE;
        self.multisample.min_sample_shading = min_fract;
        self
    }

    // TODO: doc
    pub fn alpha_to_coverage_disabled(mut self) -> Self {
        self.multisample.alpha_to_coverage_enable = ash::vk::FALSE;
        self
    }

    // TODO: doc
    pub fn alpha_to_coverage_enabled(mut self) -> Self {
        self.multisample.alpha_to_coverage_enable = ash::vk::TRUE;
        self
    }

    /// Disables alpha-to-one.
    ///
    /// Alpha-to-one is disabled by default.
    #[inline]
    pub fn alpha_to_one_disabled(mut self) -> Self {
        self.multisample.alpha_to_one_enable = ash::vk::FALSE;
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
        self.multisample.alpha_to_one_enable = ash::vk::TRUE;
        self
    }

    // TODO: rasterizationSamples and pSampleMask

    /// Sets the fragment shader to use.
    ///
    /// The fragment shader is run once for each pixel that is covered by each primitive.
    // TODO: correct specialization constants
    #[inline]
    pub fn fragment_shader<'fs2, Fss2>(
        self,
        shader: GraphicsEntryPoint<'fs2>,
        specialization_constants: Fss2,
    ) -> GraphicsPipelineBuilder<'vs, 'tcs, 'tes, 'gs, 'fs2, Vdef, Vss, Tcss, Tess, Gss, Fss2>
    where
        Fss2: SpecializationConstants,
    {
        GraphicsPipelineBuilder {
            vertex_definition: self.vertex_definition,
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
            subpass: self.subpass,
            cache: self.cache,
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
    where
        I: IntoIterator<Item = AttachmentBlend>,
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
    pub fn render_pass(self, subpass: Subpass) -> Self {
        GraphicsPipelineBuilder {
            vertex_definition: self.vertex_definition,
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
            subpass: Some(subpass),
            cache: self.cache,
        }
    }

    /// Enable caching of this pipeline via a PipelineCache object.
    ///
    /// If this pipeline already exists in the cache it will be used, if this is a new
    /// pipeline it will be inserted into the cache. The implementation handles the
    /// PipelineCache.
    #[inline]
    pub fn build_with_cache(mut self, pipeline_cache: Arc<PipelineCache>) -> Self {
        self.cache = Some(pipeline_cache);
        self
    }
}

impl<'vs, 'tcs, 'tes, 'gs, 'fs, Vdef, Vss, Tcss, Tess, Gss, Fss> Clone
    for GraphicsPipelineBuilder<'vs, 'tcs, 'tes, 'gs, 'fs, Vdef, Vss, Tcss, Tess, Gss, Fss>
where
    Vdef: Clone,
    Vss: Clone,
    Tcss: Clone,
    Tess: Clone,
    Gss: Clone,
    Fss: Clone,
{
    fn clone(&self) -> Self {
        GraphicsPipelineBuilder {
            vertex_definition: self.vertex_definition.clone(),
            vertex_shader: self.vertex_shader.clone(),
            input_assembly: unsafe { ptr::read(&self.input_assembly) },
            input_assembly_topology: self.input_assembly_topology,
            tessellation: self.tessellation.clone(),
            geometry_shader: self.geometry_shader.clone(),
            viewport: self.viewport.clone(),
            raster: self.raster.clone(),
            multisample: self.multisample,
            fragment_shader: self.fragment_shader.clone(),
            depth_stencil: self.depth_stencil.clone(),
            blend: self.blend.clone(),
            subpass: self.subpass.clone(),
            cache: self.cache.clone(),
        }
    }
}
