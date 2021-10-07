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
use crate::descriptor_set::layout::{DescriptorSetDesc, DescriptorSetLayout};
use crate::device::Device;
use crate::image::SampleCount;
use crate::pipeline::cache::PipelineCache;
use crate::pipeline::color_blend::{
    AttachmentBlend, BlendFactor, ColorBlendAttachmentState, ColorBlendState, ColorComponents,
    LogicOp,
};
use crate::pipeline::depth_stencil::DepthStencilState;
use crate::pipeline::graphics_pipeline::{
    GraphicsPipeline, GraphicsPipelineCreationError, Inner as GraphicsPipelineInner,
};
use crate::pipeline::input_assembly::{InputAssemblyState, PrimitiveTopology};
use crate::pipeline::layout::{PipelineLayout, PipelineLayoutCreationError, PipelineLayoutPcRange};
use crate::pipeline::multisample::MultisampleState;
use crate::pipeline::rasterization::{CullMode, FrontFace, PolygonMode, RasterizationState};
use crate::pipeline::shader::{
    EntryPointAbstract, GraphicsEntryPoint, GraphicsShaderType, ShaderStage,
    SpecializationConstants,
};
use crate::pipeline::tessellation::TessellationState;
use crate::pipeline::vertex::{BuffersDefinition, Vertex, VertexDefinition, VertexInputRate};
use crate::pipeline::viewport::{Scissor, Viewport, ViewportState};
use crate::pipeline::{DynamicState, StateMode};
use crate::render_pass::Subpass;
use crate::VulkanObject;
use fnv::FnvHashMap;
use smallvec::SmallVec;
use std::collections::hash_map::{Entry, HashMap};
use std::mem::{self, MaybeUninit};
use std::ptr;
use std::sync::Arc;
use std::u32;

/// Prototype for a `GraphicsPipeline`.
#[derive(Debug)]
pub struct GraphicsPipelineBuilder<'vs, 'tcs, 'tes, 'gs, 'fs, Vdef, Vss, Tcss, Tess, Gss, Fss> {
    subpass: Option<Subpass>,
    cache: Option<Arc<PipelineCache>>,

    vertex_shader: Option<(GraphicsEntryPoint<'vs>, Vss)>,
    tessellation_shaders: Option<TessellationShaders<'tcs, 'tes, Tcss, Tess>>,
    geometry_shader: Option<(GraphicsEntryPoint<'gs>, Gss)>,
    fragment_shader: Option<(GraphicsEntryPoint<'fs>, Fss)>,

    vertex_definition: Vdef,
    input_assembly_state: InputAssemblyState,
    tessellation_state: TessellationState,
    viewport_state: ViewportState,
    rasterization_state: RasterizationState,
    multisample_state: MultisampleState,
    depth_stencil_state: DepthStencilState,
    color_blend_state: ColorBlendState,
}

// Additional parameters if tessellation is used.
#[derive(Clone, Debug)]
struct TessellationShaders<'tcs, 'tes, Tcss, Tess> {
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
        BuffersDefinition,
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
            vertex_shader: None,
            tessellation_shaders: None,
            geometry_shader: None,
            fragment_shader: None,

            vertex_definition: BuffersDefinition::new(),
            input_assembly_state: Default::default(),
            tessellation_state: Default::default(),
            viewport_state: Default::default(),
            rasterization_state: Default::default(),
            multisample_state: Default::default(),
            depth_stencil_state: Default::default(),
            color_blend_state: Default::default(),

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
    ) -> Result<GraphicsPipeline, GraphicsPipelineCreationError> {
        self.with_auto_layout(device, |_| {})
    }

    /// The same as `new`, but allows you to provide a closure that is given a mutable reference to
    /// the inferred descriptor set definitions. This can be used to make changes to the layout
    /// before it's created, for example to add dynamic buffers or immutable samplers.
    pub fn with_auto_layout<F>(
        self,
        device: Arc<Device>,
        func: F,
    ) -> Result<GraphicsPipeline, GraphicsPipelineCreationError>
    where
        F: FnOnce(&mut [DescriptorSetDesc]),
    {
        let (descriptor_set_layout_descs, push_constant_ranges) = {
            let stages: SmallVec<[&GraphicsEntryPoint; 5]> = std::array::IntoIter::new([
                self.vertex_shader.as_ref().map(|s| &s.0),
                self.tessellation_shaders
                    .as_ref()
                    .map(|s| &s.tessellation_control_shader.0),
                self.tessellation_shaders
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
            func(&mut descriptor_set_layout_descs);

            // We want to union each push constant range into a set of ranges that do not have intersecting stage flags.
            // e.g. The range [0, 16) is either made available to Vertex | Fragment or we only make [0, 16) available to
            // Vertex and a subrange available to Fragment, like [0, 8)
            let mut range_map = HashMap::new();
            for stage in stages.iter() {
                if let Some(range) = stage.push_constant_range() {
                    match range_map.entry((range.offset, range.size)) {
                        Entry::Vacant(entry) => {
                            entry.insert(range.stages);
                        }
                        Entry::Occupied(mut entry) => {
                            *entry.get_mut() = *entry.get() | range.stages;
                        }
                    }
                }
            }
            let push_constant_ranges: Vec<_> = range_map
                .iter()
                .map(|((offset, size), stages)| PipelineLayoutPcRange {
                    offset: *offset,
                    size: *size,
                    stages: *stages,
                })
                .collect();

            (descriptor_set_layout_descs, push_constant_ranges)
        };

        let descriptor_set_layouts = descriptor_set_layout_descs
            .into_iter()
            .map(|desc| Ok(Arc::new(DescriptorSetLayout::new(device.clone(), desc)?)))
            .collect::<Result<Vec<_>, PipelineLayoutCreationError>>()?;
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
    ) -> Result<GraphicsPipeline, GraphicsPipelineCreationError> {
        // TODO: return errors instead of panicking if missing param

        let fns = device.fns();
        let subpass = self.subpass.take().expect("Missing subpass");

        // Checking that the pipeline layout matches the shader stages.
        // TODO: more details in the errors

        {
            let shader = &self.vertex_shader.as_ref().unwrap().0;
            pipeline_layout.ensure_compatible_with_shader(
                shader.descriptor_set_layout_descs(),
                shader.push_constant_range(),
            )?;
        }

        if let Some(ref geometry_shader) = self.geometry_shader {
            let shader = &geometry_shader.0;
            pipeline_layout.ensure_compatible_with_shader(
                shader.descriptor_set_layout_descs(),
                shader.push_constant_range(),
            )?;
        }

        if let Some(ref tess) = self.tessellation_shaders {
            {
                let shader = &tess.tessellation_control_shader.0;
                pipeline_layout.ensure_compatible_with_shader(
                    shader.descriptor_set_layout_descs(),
                    shader.push_constant_range(),
                )?;
            }

            {
                let shader = &tess.tessellation_evaluation_shader.0;
                pipeline_layout.ensure_compatible_with_shader(
                    shader.descriptor_set_layout_descs(),
                    shader.push_constant_range(),
                )?;
            }
        }

        if let Some(ref fragment_shader) = self.fragment_shader {
            let shader = &fragment_shader.0;
            pipeline_layout.ensure_compatible_with_shader(
                shader.descriptor_set_layout_descs(),
                shader.push_constant_range(),
            )?;

            // Check that the subpass can accept the output of the fragment shader.
            // TODO: If there is no fragment shader, what should be checked then? The previous stage?
            if !subpass.is_compatible_with(shader.output()) {
                return Err(GraphicsPipelineCreationError::FragmentShaderRenderPassIncompatible);
            }
        }

        // Will contain the list of dynamic states. Filled throughout this function.
        let mut dynamic_state_modes: FnvHashMap<DynamicState, bool> = HashMap::default();

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

        let tess_shader_specialization = if let Some(ref tess) = self.tessellation_shaders {
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
        let mut shaders = HashMap::default();
        let stages = {
            let mut stages = SmallVec::<[_; 5]>::new();

            match self.vertex_shader.as_ref().unwrap().0.ty() {
                GraphicsShaderType::Vertex => {}
                _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
            };

            shaders.insert(ShaderStage::Vertex, ());
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

            if let Some(ref tess) = self.tessellation_shaders {
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

                shaders.insert(ShaderStage::TessellationControl, ());
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

                shaders.insert(ShaderStage::TessellationEvaluation, ());
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

                let shader_execution_mode = match geometry_shader.0.ty() {
                    GraphicsShaderType::Geometry(mode) => mode,
                    _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
                };

                if let StateMode::Fixed(topology) = self.input_assembly_state.topology {
                    if !shader_execution_mode.matches(topology) {
                        return Err(
                            GraphicsPipelineCreationError::TopologyNotMatchingGeometryShader,
                        );
                    }
                }

                // TODO: VUID-VkGraphicsPipelineCreateInfo-pStages-00739
                // If the pipeline is being created with pre-rasterization shader state and pStages
                // includes a geometry shader stage, and also includes tessellation shader stages,
                // its shader code must contain an OpExecutionMode instruction that specifies an
                // input primitive type that is compatible with the primitive topology that is
                // output by the tessellation stages

                shaders.insert(ShaderStage::Geometry, ());
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

                shaders.insert(ShaderStage::Fragment, ());
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

        // Vertex input state
        let vertex_input = self
            .vertex_definition
            .definition(self.vertex_shader.as_ref().unwrap().0.input())?;

        let (binding_descriptions, binding_divisor_descriptions) = {
            let mut binding_descriptions = SmallVec::<[_; 8]>::new();
            let mut binding_divisor_descriptions = SmallVec::<[_; 8]>::new();

            for (binding, binding_desc) in vertex_input.bindings() {
                if binding
                    >= device
                        .physical_device()
                        .properties()
                        .max_vertex_input_bindings
                {
                    return Err(
                        GraphicsPipelineCreationError::MaxVertexInputBindingsExceeded {
                            max: device
                                .physical_device()
                                .properties()
                                .max_vertex_input_bindings,
                            obtained: binding,
                        },
                    );
                }

                if binding_desc.stride
                    > device
                        .physical_device()
                        .properties()
                        .max_vertex_input_binding_stride
                {
                    return Err(
                        GraphicsPipelineCreationError::MaxVertexInputBindingStrideExceeded {
                            binding,
                            max: device
                                .physical_device()
                                .properties()
                                .max_vertex_input_binding_stride,
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
                    .max_vertex_input_bindings as usize
            {
                return Err(
                    GraphicsPipelineCreationError::MaxVertexInputBindingsExceeded {
                        max: device
                            .physical_device()
                            .properties()
                            .max_vertex_input_bindings,
                        obtained: binding_descriptions.len() as u32,
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
                {
                    return Err(
                        GraphicsPipelineCreationError::MaxVertexInputAttributeOffsetExceeded {
                            max: device
                                .physical_device()
                                .properties()
                                .max_vertex_input_attribute_offset,
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
                    .max_vertex_input_attributes as usize
            {
                return Err(
                    GraphicsPipelineCreationError::MaxVertexInputAttributesExceeded {
                        max: device
                            .physical_device()
                            .properties()
                            .max_vertex_input_attributes,
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

        let vertex_input_state = Some(ash::vk::PipelineVertexInputStateCreateInfo {
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
        });

        // Input assembly state
        let input_assembly_state = if self.vertex_shader.is_some() {
            let topology = match self.input_assembly_state.topology {
                StateMode::Fixed(topology) => {
                    match topology {
                        PrimitiveTopology::LineListWithAdjacency
                        | PrimitiveTopology::LineStripWithAdjacency
                        | PrimitiveTopology::TriangleListWithAdjacency
                        | PrimitiveTopology::TriangleStripWithAdjacency => {
                            if !device.enabled_features().geometry_shader {
                                return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                                    feature: "geometry_shader",
                                    reason: "InputAssemblyState::topology was set to a WithAdjacency PrimitiveTopology",
                                });
                            }
                        }
                        PrimitiveTopology::PatchList => {
                            if !device.enabled_features().tessellation_shader {
                                return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                                    feature: "tessellation_shader",
                                    reason: "InputAssemblyState::topology was set to PrimitiveTopology::PatchList",
                                });
                            }
                        }
                        _ => (),
                    }
                    topology.into()
                }
                StateMode::Dynamic => {
                    if !device.enabled_features().extended_dynamic_state {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "extended_dynamic_state",
                            reason: "InputAssemblyState::topology was set to Dynamic",
                        });
                    }
                    dynamic_state_modes.insert(DynamicState::PrimitiveTopology, true);
                    Default::default()
                }
            };

            let primitive_restart_enable = match self.input_assembly_state.primitive_restart_enable
            {
                StateMode::Fixed(primitive_restart_enable) => {
                    if primitive_restart_enable {
                        match self.input_assembly_state.topology {
                            StateMode::Fixed(
                                PrimitiveTopology::PointList
                                | PrimitiveTopology::LineList
                                | PrimitiveTopology::TriangleList
                                | PrimitiveTopology::LineListWithAdjacency
                                | PrimitiveTopology::TriangleListWithAdjacency,
                            ) => {
                                if !device.enabled_features().primitive_topology_list_restart {
                                    return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                                        feature: "primitive_topology_list_restart",
                                        reason: "InputAssemblyState::primitive_restart_enable was set to true in combination with a List PrimitiveTopology",
                                    });
                                }
                            }
                            StateMode::Fixed(PrimitiveTopology::PatchList) => {
                                if !device
                                    .enabled_features()
                                    .primitive_topology_patch_list_restart
                                {
                                    return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                                        feature: "primitive_topology_patch_list_restart",
                                        reason: "InputAssemblyState::primitive_restart_enable was set to true in combination with PrimitiveTopology::PatchList",
                                    });
                                }
                            }
                            _ => (),
                        }
                    }

                    primitive_restart_enable as ash::vk::Bool32
                }
                StateMode::Dynamic => {
                    if !device.enabled_features().extended_dynamic_state2 {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "extended_dynamic_state2",
                            reason:
                                "InputAssemblyState::primitive_restart_enable was set to Dynamic",
                        });
                    }
                    dynamic_state_modes.insert(DynamicState::PrimitiveRestartEnable, true);
                    Default::default()
                }
            };

            Some(ash::vk::PipelineInputAssemblyStateCreateInfo {
                flags: ash::vk::PipelineInputAssemblyStateCreateFlags::empty(),
                topology,
                primitive_restart_enable,
                ..Default::default()
            })
        } else {
            None
        };

        let tessellation_state = if self.tessellation_shaders.is_some() {
            if !matches!(
                self.input_assembly_state.topology,
                StateMode::Dynamic | StateMode::Fixed(PrimitiveTopology::PatchList)
            ) {
                return Err(GraphicsPipelineCreationError::InvalidPrimitiveTopology);
            }

            let patch_control_points = match self.tessellation_state.patch_control_points {
                StateMode::Fixed(patch_control_points) => {
                    if patch_control_points <= 0
                        || patch_control_points
                            > device
                                .physical_device()
                                .properties()
                                .max_tessellation_patch_size
                    {
                        return Err(GraphicsPipelineCreationError::InvalidNumPatchControlPoints);
                    }

                    patch_control_points
                }
                StateMode::Dynamic => {
                    if !device
                        .enabled_features()
                        .extended_dynamic_state2_patch_control_points
                    {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "extended_dynamic_state2_patch_control_points",
                            reason: "TessellationState::patch_control_points was set to Dynamic",
                        });
                    }

                    dynamic_state_modes.insert(DynamicState::PatchControlPoints, true);
                    Default::default()
                }
            };

            Some(ash::vk::PipelineTessellationStateCreateInfo {
                flags: ash::vk::PipelineTessellationStateCreateFlags::empty(),
                patch_control_points,
                ..Default::default()
            })
        } else {
            None
        };

        let (viewport_count, viewports, scissor_count, scissors) = match self.viewport_state {
            ViewportState::Fixed { ref data } => {
                let count = data.len() as u32;
                assert!(count != 0); // TODO: return error?
                let viewports = data
                    .iter()
                    .map(|e| e.0.clone().into())
                    .collect::<SmallVec<[ash::vk::Viewport; 2]>>();
                let scissors = data
                    .iter()
                    .map(|e| e.1.clone().into())
                    .collect::<SmallVec<[ash::vk::Rect2D; 2]>>();

                (count, viewports, count, scissors)
            }
            ViewportState::FixedViewport {
                ref viewports,
                scissor_count_dynamic,
            } => {
                let viewport_count = viewports.len() as u32;
                assert!(viewport_count != 0); // TODO: return error?
                let viewports = viewports
                    .iter()
                    .map(|e| e.clone().into())
                    .collect::<SmallVec<[ash::vk::Viewport; 2]>>();

                let scissor_count = if scissor_count_dynamic {
                    if !device.enabled_features().extended_dynamic_state {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "extended_dynamic_state",
                            reason: "ViewportState::FixedViewport::scissor_count_dynamic was set to true",
                        });
                    }
                    dynamic_state_modes.insert(DynamicState::ScissorWithCount, true);
                    0
                } else {
                    dynamic_state_modes.insert(DynamicState::Scissor, true);
                    viewport_count
                };

                (viewport_count, viewports, scissor_count, SmallVec::new())
            }
            ViewportState::FixedScissor {
                ref scissors,
                viewport_count_dynamic,
            } => {
                let scissor_count = scissors.len() as u32;
                assert!(scissor_count != 0); // TODO: return error?
                let scissors = scissors
                    .iter()
                    .map(|e| e.clone().into())
                    .collect::<SmallVec<[ash::vk::Rect2D; 2]>>();

                let viewport_count = if viewport_count_dynamic {
                    if !device.enabled_features().extended_dynamic_state {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "extended_dynamic_state",
                            reason: "ViewportState::FixedScissor::viewport_count_dynamic was set to true",
                        });
                    }
                    dynamic_state_modes.insert(DynamicState::ViewportWithCount, true);
                    0
                } else {
                    dynamic_state_modes.insert(DynamicState::Viewport, true);
                    scissor_count
                };

                (viewport_count, SmallVec::new(), scissor_count, scissors)
            }
            ViewportState::Dynamic {
                count,
                viewport_count_dynamic,
                scissor_count_dynamic,
            } => {
                if !(viewport_count_dynamic && scissor_count_dynamic) {
                    assert!(count != 0); // TODO: return error?
                }

                let viewport_count = if viewport_count_dynamic {
                    if !device.enabled_features().extended_dynamic_state {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "extended_dynamic_state",
                            reason:
                                "ViewportState::Dynamic::viewport_count_dynamic was set to true",
                        });
                    }
                    dynamic_state_modes.insert(DynamicState::ViewportWithCount, true);
                    0
                } else {
                    dynamic_state_modes.insert(DynamicState::Viewport, true);
                    count
                };

                let scissor_count = if scissor_count_dynamic {
                    if !device.enabled_features().extended_dynamic_state {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "extended_dynamic_state",
                            reason: "ViewportState::Dynamic::scissor_count_dynamic was set to true",
                        });
                    }
                    dynamic_state_modes.insert(DynamicState::ScissorWithCount, true);
                    0
                } else {
                    dynamic_state_modes.insert(DynamicState::Scissor, true);
                    count
                };

                (
                    viewport_count,
                    SmallVec::new(),
                    scissor_count,
                    SmallVec::new(),
                )
            }
        };

        let viewport_state = {
            let viewport_scissor_count = u32::max(viewport_count, scissor_count);

            if viewport_scissor_count > 1 && !device.enabled_features().multi_viewport {
                return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                    feature: "multi_viewport",
                    reason: "ViewportState viewport/scissor count was greater than 1",
                });
            }

            if viewport_scissor_count > device.physical_device().properties().max_viewports {
                return Err(GraphicsPipelineCreationError::MaxViewportsExceeded {
                    obtained: viewport_scissor_count,
                    max: device.physical_device().properties().max_viewports,
                });
            }

            for vp in viewports.iter() {
                if vp.width
                    > device
                        .physical_device()
                        .properties()
                        .max_viewport_dimensions[0] as f32
                    || vp.height
                        > device
                            .physical_device()
                            .properties()
                            .max_viewport_dimensions[1] as f32
                {
                    return Err(GraphicsPipelineCreationError::MaxViewportDimensionsExceeded);
                }

                if vp.x < device.physical_device().properties().viewport_bounds_range[0]
                    || vp.x + vp.width
                        > device.physical_device().properties().viewport_bounds_range[1]
                    || vp.y < device.physical_device().properties().viewport_bounds_range[0]
                    || vp.y + vp.height
                        > device.physical_device().properties().viewport_bounds_range[1]
                {
                    return Err(GraphicsPipelineCreationError::ViewportBoundsExceeded);
                }
            }

            Some(ash::vk::PipelineViewportStateCreateInfo {
                flags: ash::vk::PipelineViewportStateCreateFlags::empty(),
                viewport_count,
                p_viewports: if viewports.is_empty() {
                    ptr::null()
                } else {
                    viewports.as_ptr()
                }, // validation layer crashes if you just pass the pointer
                scissor_count,
                p_scissors: if scissors.is_empty() {
                    ptr::null()
                } else {
                    scissors.as_ptr()
                }, // validation layer crashes if you just pass the pointer
                ..Default::default()
            })
        };

        let rasterization_state = {
            if self.rasterization_state.depth_clamp_enable && !device.enabled_features().depth_clamp
            {
                return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                    feature: "depth_clamp",
                    reason: "RasterizationState::depth_clamp_enable was true",
                });
            }

            let rasterizer_discard_enable = match self.rasterization_state.rasterizer_discard_enable
            {
                StateMode::Fixed(rasterizer_discard_enable) => {
                    rasterizer_discard_enable as ash::vk::Bool32
                }
                StateMode::Dynamic => {
                    if !device.enabled_features().extended_dynamic_state2 {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "extended_dynamic_state2",
                            reason:
                                "RasterizationState::rasterizer_discard_enable was set to Dynamic",
                        });
                    }
                    dynamic_state_modes.insert(DynamicState::RasterizerDiscardEnable, true);
                    ash::vk::FALSE
                }
            };

            if self.rasterization_state.polygon_mode != PolygonMode::Fill
                && !device.enabled_features().fill_mode_non_solid
            {
                return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                    feature: "fill_mode_non_solid",
                    reason: "RasterizationState::polygon_mode was not Fill",
                });
            }

            let cull_mode = match self.rasterization_state.cull_mode {
                StateMode::Fixed(cull_mode) => cull_mode.into(),
                StateMode::Dynamic => {
                    if !device.enabled_features().extended_dynamic_state {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "extended_dynamic_state",
                            reason: "RasterizationState::cull_mode was set to Dynamic",
                        });
                    }
                    dynamic_state_modes.insert(DynamicState::CullMode, true);
                    CullMode::default().into()
                }
            };

            let front_face = match self.rasterization_state.front_face {
                StateMode::Fixed(front_face) => front_face.into(),
                StateMode::Dynamic => {
                    if !device.enabled_features().extended_dynamic_state {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "extended_dynamic_state",
                            reason: "RasterizationState::front_face was set to Dynamic",
                        });
                    }
                    dynamic_state_modes.insert(DynamicState::FrontFace, true);
                    FrontFace::default().into()
                }
            };

            let (
                depth_bias_enable,
                depth_bias_constant_factor,
                depth_bias_clamp,
                depth_bias_slope_factor,
            ) = if let Some(depth_bias_state) = self.rasterization_state.depth_bias {
                if depth_bias_state.enable_dynamic {
                    if !device.enabled_features().extended_dynamic_state2 {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "extended_dynamic_state2",
                            reason: "DepthBiasState::enable_dynamic was true",
                        });
                    }
                    dynamic_state_modes.insert(DynamicState::DepthTestEnable, true);
                }

                let (constant_factor, clamp, slope_factor) = match depth_bias_state.bias {
                    StateMode::Fixed(bias) => {
                        if bias.clamp != 0.0 && !device.enabled_features().depth_bias_clamp {
                            return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                                feature: "depth_bias_clamp",
                                reason: "DepthBias::clamp was not 0.0",
                            });
                        }
                        (bias.constant_factor, bias.clamp, bias.slope_factor)
                    }
                    StateMode::Dynamic => {
                        dynamic_state_modes.insert(DynamicState::DepthBias, true);
                        (0.0, 0.0, 0.0)
                    }
                };

                (ash::vk::TRUE, constant_factor, clamp, slope_factor)
            } else {
                (ash::vk::FALSE, 0.0, 0.0, 0.0)
            };

            let line_width = match self.rasterization_state.line_width {
                StateMode::Fixed(line_width) => {
                    if line_width != 1.0 && !device.enabled_features().wide_lines {
                        return Err(GraphicsPipelineCreationError::WideLinesFeatureNotEnabled);
                    }
                    line_width
                }
                StateMode::Dynamic => {
                    dynamic_state_modes.insert(DynamicState::LineWidth, true);
                    1.0
                }
            };

            Some(ash::vk::PipelineRasterizationStateCreateInfo {
                flags: ash::vk::PipelineRasterizationStateCreateFlags::empty(),
                depth_clamp_enable: self.rasterization_state.depth_clamp_enable as ash::vk::Bool32,
                rasterizer_discard_enable,
                polygon_mode: self.rasterization_state.polygon_mode.into(),
                cull_mode,
                front_face,
                depth_bias_enable,
                depth_bias_constant_factor,
                depth_bias_clamp,
                depth_bias_slope_factor,
                line_width,
                ..Default::default()
            })
        };

        let has_fragment_shader_state =
            self.rasterization_state.rasterizer_discard_enable != StateMode::Fixed(true);

        let multisample_state = if has_fragment_shader_state {
            let rasterization_samples =
                subpass.num_samples().unwrap_or(SampleCount::Sample1).into();

            let (sample_shading_enable, min_sample_shading) =
                if let Some(min_sample_shading) = self.multisample_state.sample_shading {
                    if !device.enabled_features().sample_rate_shading {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "sample_rate_shading",
                            reason: "MultisampleState::sample_shading was Some",
                        });
                    }
                    assert!(min_sample_shading >= 0.0 && min_sample_shading <= 1.0); // TODO: return error?
                    (ash::vk::TRUE, min_sample_shading)
                } else {
                    (ash::vk::FALSE, 0.0)
                };

            let alpha_to_one_enable = {
                if self.multisample_state.alpha_to_one_enable {
                    if !device.enabled_features().alpha_to_one {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "alpha_to_one",
                            reason: "MultisampleState::alpha_to_one was true",
                        });
                    }
                }
                self.multisample_state.alpha_to_one_enable as ash::vk::Bool32
            };

            Some(ash::vk::PipelineMultisampleStateCreateInfo {
                flags: ash::vk::PipelineMultisampleStateCreateFlags::empty(),
                rasterization_samples,
                sample_shading_enable,
                min_sample_shading,
                p_sample_mask: ptr::null(),
                alpha_to_coverage_enable: self.multisample_state.alpha_to_coverage_enable
                    as ash::vk::Bool32,
                alpha_to_one_enable,
                ..Default::default()
            })
        } else {
            None
        };

        let depth_stencil_state = if has_fragment_shader_state
            && subpass.subpass_desc().depth_stencil.is_some()
        {
            let (depth_test_enable, depth_write_enable, depth_compare_op) =
                if let Some(depth_state) = &self.depth_stencil_state.depth {
                    if !subpass.has_depth() {
                        return Err(GraphicsPipelineCreationError::NoDepthAttachment);
                    }

                    if depth_state.enable_dynamic {
                        if !device.enabled_features().extended_dynamic_state {
                            return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                                feature: "extended_dynamic_state",
                                reason: "DepthState::enable_dynamic was true",
                            });
                        }
                        dynamic_state_modes.insert(DynamicState::DepthTestEnable, true);
                    }

                    let write_enable = match depth_state.write_enable {
                        StateMode::Fixed(write_enable) => {
                            if write_enable && !subpass.has_writable_depth() {
                                return Err(GraphicsPipelineCreationError::NoDepthAttachment);
                            }
                            write_enable as ash::vk::Bool32
                        }
                        StateMode::Dynamic => {
                            if !device.enabled_features().extended_dynamic_state {
                                return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                                    feature: "extended_dynamic_state",
                                    reason: "DepthState::write_enable was set to Dynamic",
                                });
                            }
                            dynamic_state_modes.insert(DynamicState::DepthWriteEnable, true);
                            ash::vk::TRUE
                        }
                    };

                    let compare_op = match depth_state.compare_op {
                        StateMode::Fixed(compare_op) => compare_op.into(),
                        StateMode::Dynamic => {
                            if !device.enabled_features().extended_dynamic_state {
                                return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                                    feature: "extended_dynamic_state",
                                    reason: "DepthState::compare_op was set to Dynamic",
                                });
                            }
                            dynamic_state_modes.insert(DynamicState::DepthCompareOp, true);
                            ash::vk::CompareOp::ALWAYS
                        }
                    };

                    (ash::vk::TRUE, write_enable, compare_op)
                } else {
                    (ash::vk::FALSE, ash::vk::FALSE, ash::vk::CompareOp::ALWAYS)
                };

            let (depth_bounds_test_enable, min_depth_bounds, max_depth_bounds) = if let Some(
                depth_bounds_state,
            ) =
                &self.depth_stencil_state.depth_bounds
            {
                if !device.enabled_features().depth_bounds {
                    return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                        feature: "depth_bounds",
                        reason: "DepthStencilState::depth_bounds was Some",
                    });
                }

                if depth_bounds_state.enable_dynamic {
                    if !device.enabled_features().extended_dynamic_state {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "extended_dynamic_state",
                            reason: "DepthBoundsState::enable_dynamic was true",
                        });
                    }
                    dynamic_state_modes.insert(DynamicState::DepthBoundsTestEnable, true);
                }

                let (min_bounds, max_bounds) = match depth_bounds_state.bounds.clone() {
                    StateMode::Fixed(bounds) => {
                        if !device.enabled_extensions().ext_depth_range_unrestricted
                            && !(0.0..1.0).contains(bounds.start())
                            && !(0.0..1.0).contains(bounds.end())
                        {
                            return Err(GraphicsPipelineCreationError::ExtensionNotEnabled {
                            extension: "ext_depth_range_unrestricted",
                            reason: "DepthBoundsState::bounds were not both between 0.0 and 1.0 inclusive",
                        });
                        }
                        bounds.into_inner()
                    }
                    StateMode::Dynamic => {
                        dynamic_state_modes.insert(DynamicState::DepthBounds, true);
                        (0.0, 1.0)
                    }
                };

                (ash::vk::TRUE, min_bounds, max_bounds)
            } else {
                (ash::vk::FALSE, 0.0, 1.0)
            };

            let (stencil_test_enable, front, back) = if let Some(stencil_state) =
                &self.depth_stencil_state.stencil
            {
                if !subpass.has_stencil() {
                    return Err(GraphicsPipelineCreationError::NoStencilAttachment);
                }

                // TODO: if stencil buffer can potentially be written, check if it is writable

                if stencil_state.enable_dynamic {
                    if !device.enabled_features().extended_dynamic_state {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "extended_dynamic_state",
                            reason: "StencilState::enable_dynamic was true",
                        });
                    }
                    dynamic_state_modes.insert(DynamicState::StencilTestEnable, true);
                }

                match (stencil_state.front.ops, stencil_state.back.ops) {
                    (StateMode::Fixed(_), StateMode::Fixed(_)) => (),
                    (StateMode::Dynamic, StateMode::Dynamic) => {
                        if !device.enabled_features().extended_dynamic_state {
                            return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                                feature: "extended_dynamic_state",
                                reason: "StencilState::ops was set to Dynamic",
                            });
                        }
                        dynamic_state_modes.insert(DynamicState::StencilOp, true);
                    }
                    _ => return Err(GraphicsPipelineCreationError::WrongStencilState),
                };

                match (
                    stencil_state.front.compare_mask,
                    stencil_state.back.compare_mask,
                ) {
                    (StateMode::Fixed(_), StateMode::Fixed(_)) => (),
                    (StateMode::Dynamic, StateMode::Dynamic) => {
                        dynamic_state_modes.insert(DynamicState::StencilCompareMask, true);
                    }
                    _ => return Err(GraphicsPipelineCreationError::WrongStencilState),
                };

                match (
                    stencil_state.front.write_mask,
                    stencil_state.back.write_mask,
                ) {
                    (StateMode::Fixed(_), StateMode::Fixed(_)) => (),
                    (StateMode::Dynamic, StateMode::Dynamic) => {
                        dynamic_state_modes.insert(DynamicState::StencilWriteMask, true);
                    }
                    _ => return Err(GraphicsPipelineCreationError::WrongStencilState),
                };

                match (stencil_state.front.reference, stencil_state.back.reference) {
                    (StateMode::Fixed(_), StateMode::Fixed(_)) => (),
                    (StateMode::Dynamic, StateMode::Dynamic) => {
                        dynamic_state_modes.insert(DynamicState::StencilReference, true);
                    }
                    _ => return Err(GraphicsPipelineCreationError::WrongStencilState),
                };

                let [front, back] = [&stencil_state.front, &stencil_state.back].map(|ops_state| {
                    let ops = match ops_state.ops {
                        StateMode::Fixed(x) => x,
                        StateMode::Dynamic => Default::default(),
                    };
                    let compare_mask = match ops_state.compare_mask {
                        StateMode::Fixed(x) => x,
                        StateMode::Dynamic => Default::default(),
                    };
                    let write_mask = match ops_state.write_mask {
                        StateMode::Fixed(x) => x,
                        StateMode::Dynamic => Default::default(),
                    };
                    let reference = match ops_state.reference {
                        StateMode::Fixed(x) => x,
                        StateMode::Dynamic => Default::default(),
                    };

                    ash::vk::StencilOpState {
                        fail_op: ops.fail_op.into(),
                        pass_op: ops.pass_op.into(),
                        depth_fail_op: ops.depth_fail_op.into(),
                        compare_op: ops.compare_op.into(),
                        compare_mask,
                        write_mask,
                        reference,
                    }
                });

                (ash::vk::TRUE, front, back)
            } else {
                (ash::vk::FALSE, Default::default(), Default::default())
            };

            Some(ash::vk::PipelineDepthStencilStateCreateInfo {
                flags: ash::vk::PipelineDepthStencilStateCreateFlags::empty(),
                depth_test_enable,
                depth_write_enable,
                depth_compare_op,
                depth_bounds_test_enable,
                stencil_test_enable,
                front,
                back,
                min_depth_bounds,
                max_depth_bounds,
                ..Default::default()
            })
        } else {
            None
        };

        let color_blend_attachments: SmallVec<[ash::vk::PipelineColorBlendAttachmentState; 4]> = {
            let num_atch = subpass.num_color_attachments();
            let to_vulkan = |state: &ColorBlendAttachmentState| {
                let blend = if let Some(blend) = &state.blend {
                    if !device.enabled_features().dual_src_blend
                        && std::array::IntoIter::new([
                            blend.color_source,
                            blend.color_destination,
                            blend.alpha_source,
                            blend.alpha_destination,
                        ])
                        .any(|blend_factor| {
                            matches!(
                                blend_factor,
                                BlendFactor::Src1Color
                                    | BlendFactor::OneMinusSrc1Color
                                    | BlendFactor::Src1Alpha
                                    | BlendFactor::OneMinusSrc1Alpha
                            )
                        })
                    {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "dual_src_blend",
                            reason:
                                "One of the BlendFactor members of AttachmentBlend was set to Src1",
                        });
                    }

                    blend.clone().into()
                } else {
                    Default::default()
                };
                let color_write_mask = state.color_write_mask.into();

                Ok(ash::vk::PipelineColorBlendAttachmentState {
                    color_write_mask,
                    ..blend
                })
            };

            if self.color_blend_state.attachments.len() == 1 {
                std::iter::repeat(to_vulkan(&self.color_blend_state.attachments[0])?)
                    .take(num_atch as usize)
                    .collect()
            } else {
                if self.color_blend_state.attachments.len() != num_atch as usize {
                    return Err(GraphicsPipelineCreationError::MismatchBlendingAttachmentsCount);
                }

                if self.color_blend_state.attachments.len() > 1
                    && !device.enabled_features().independent_blend
                {
                    // Ensure that all `blend` and `color_write_mask` are identical.
                    let mut iter = self
                        .color_blend_state
                        .attachments
                        .iter()
                        .map(|state| (&state.blend, &state.color_write_mask));
                    let first = iter.next().unwrap();

                    if !iter.all(|state| state == first) {
                        return Err(
                            GraphicsPipelineCreationError::FeatureNotEnabled {
                                feature: "independent_blend",
                                reason: "The blend and color_write_mask members of all elements of ColorBlendState::attachments were not identical",
                            },
                        );
                    }
                }

                self.color_blend_state
                    .attachments
                    .iter()
                    .map(to_vulkan)
                    .collect::<Result<_, _>>()?
            }
        };

        let color_blend_state = if has_fragment_shader_state {
            let (logic_op_enable, logic_op) =
                if let Some(logic_op) = self.color_blend_state.logic_op {
                    if !device.enabled_features().logic_op {
                        return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "logic_op",
                            reason: "ColorBlendState::logic_op was set to Some",
                        });
                    }

                    let logic_op = match logic_op {
                        StateMode::Fixed(logic_op) => logic_op,
                        StateMode::Dynamic => {
                            if !device.enabled_features().extended_dynamic_state2_logic_op {
                                return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                                    feature: "extended_dynamic_state2_logic_op",
                                    reason: "ColorBlendState::logic_op was set to Some(Dynamic)",
                                });
                            }
                            dynamic_state_modes.insert(DynamicState::LogicOp, true);
                            Default::default()
                        }
                    };

                    (ash::vk::TRUE, logic_op.into())
                } else {
                    (ash::vk::FALSE, Default::default())
                };

            let blend_constants = match self.color_blend_state.blend_constants {
                StateMode::Fixed(blend_constants) => blend_constants,
                StateMode::Dynamic => {
                    dynamic_state_modes.insert(DynamicState::BlendConstants, true);
                    Default::default()
                }
            };

            Some(ash::vk::PipelineColorBlendStateCreateInfo {
                flags: ash::vk::PipelineColorBlendStateCreateFlags::empty(),
                logic_op_enable,
                logic_op,
                attachment_count: color_blend_attachments.len() as u32,
                p_attachments: color_blend_attachments.as_ptr(),
                blend_constants,
                ..Default::default()
            })
        } else {
            None
        };

        // Dynamic state
        let dynamic_state_list: Vec<ash::vk::DynamicState> = dynamic_state_modes
            .iter()
            .filter(|(_, d)| **d)
            .map(|(&state, _)| state.into())
            .collect();

        let dynamic_state = if !dynamic_state_list.is_empty() {
            Some(ash::vk::PipelineDynamicStateCreateInfo {
                flags: ash::vk::PipelineDynamicStateCreateFlags::empty(),
                dynamic_state_count: dynamic_state_list.len() as u32,
                p_dynamic_states: dynamic_state_list.as_ptr(),
                ..Default::default()
            })
        } else {
            None
        };

        // Set any remaining states to fixed, if the corresponding state is enabled.
        if vertex_input_state.is_some() {
            dynamic_state_modes
                .entry(DynamicState::VertexInput)
                .or_insert(false);
        }

        if input_assembly_state.is_some() {
            dynamic_state_modes
                .entry(DynamicState::PrimitiveTopology)
                .or_insert(false);
            dynamic_state_modes
                .entry(DynamicState::PrimitiveRestartEnable)
                .or_insert(false);
        }

        if tessellation_state.is_some() {
            dynamic_state_modes
                .entry(DynamicState::PatchControlPoints)
                .or_insert(false);
        }

        if viewport_state.is_some() {
            dynamic_state_modes
                .entry(DynamicState::Viewport)
                .or_insert(false);
            dynamic_state_modes
                .entry(DynamicState::Scissor)
                .or_insert(false);
            dynamic_state_modes
                .entry(DynamicState::ViewportWithCount)
                .or_insert(false);
            dynamic_state_modes
                .entry(DynamicState::ScissorWithCount)
                .or_insert(false);
        }

        if rasterization_state.is_some() {
            dynamic_state_modes
                .entry(DynamicState::RasterizerDiscardEnable)
                .or_insert(false);
            dynamic_state_modes
                .entry(DynamicState::CullMode)
                .or_insert(false);
            dynamic_state_modes
                .entry(DynamicState::FrontFace)
                .or_insert(false);
            dynamic_state_modes
                .entry(DynamicState::DepthBiasEnable)
                .or_insert(false);
            dynamic_state_modes
                .entry(DynamicState::DepthBias)
                .or_insert(false);
            dynamic_state_modes
                .entry(DynamicState::LineWidth)
                .or_insert(false);
        }

        if depth_stencil_state.is_some() {
            dynamic_state_modes
                .entry(DynamicState::DepthTestEnable)
                .or_insert(false);
            dynamic_state_modes
                .entry(DynamicState::DepthWriteEnable)
                .or_insert(false);
            dynamic_state_modes
                .entry(DynamicState::DepthCompareOp)
                .or_insert(false);
            dynamic_state_modes
                .entry(DynamicState::DepthBoundsTestEnable)
                .or_insert(false);
            dynamic_state_modes
                .entry(DynamicState::StencilTestEnable)
                .or_insert(false);
            dynamic_state_modes
                .entry(DynamicState::StencilCompareMask)
                .or_insert(false);
            dynamic_state_modes
                .entry(DynamicState::StencilWriteMask)
                .or_insert(false);
            dynamic_state_modes
                .entry(DynamicState::StencilReference)
                .or_insert(false);
            dynamic_state_modes
                .entry(DynamicState::DepthBounds)
                .or_insert(false);
        }

        if color_blend_state.is_some() {
            dynamic_state_modes
                .entry(DynamicState::LogicOp)
                .or_insert(false);
            dynamic_state_modes
                .entry(DynamicState::BlendConstants)
                .or_insert(false);
        }

        // StateMode::Dynamic states not handled yet:
        // - ViewportWScaling (VkPipelineViewportWScalingStateCreateInfoNV)
        // - DiscardRectangle (VkPipelineDiscardRectangleStateCreateInfoEXT)
        // - SampleLocations (VkPipelineSampleLocationsStateCreateInfoEXT)
        // - ViewportShadingRatePalette (VkPipelineViewportShadingRateImageStateCreateInfoNV)
        // - ViewportCoarseSampleOrder (VkPipelineViewportCoarseSampleOrderStateCreateInfoNV)
        // - ExclusiveScissor (VkPipelineViewportExclusiveScissorStateCreateInfoNV)
        // - FragmentShadingRate (VkPipelineFragmentShadingRateStateCreateInfoKHR)
        // - LineStipple (VkPipelineRasterizationLineStateCreateInfoEXT)
        // - ColorWriteEnable (VkPipelineColorWriteCreateInfoEXT)

        if let Some(multiview) = subpass.render_pass().desc().multiview().as_ref() {
            if multiview.used_layer_count() > 0 {
                if self.geometry_shader.is_some()
                    && !device
                        .physical_device()
                        .supported_features()
                        .multiview_geometry_shader
                {
                    return Err(GraphicsPipelineCreationError::MultiviewGeometryShaderNotSupported);
                }

                if self.tessellation_shaders.is_some()
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
                p_vertex_input_state: vertex_input_state
                    .as_ref()
                    .map(|p| p as *const _)
                    .unwrap_or(ptr::null()),
                p_input_assembly_state: input_assembly_state
                    .as_ref()
                    .map(|p| p as *const _)
                    .unwrap_or(ptr::null()),
                p_tessellation_state: tessellation_state
                    .as_ref()
                    .map(|p| p as *const _)
                    .unwrap_or(ptr::null()),
                p_viewport_state: viewport_state
                    .as_ref()
                    .map(|p| p as *const _)
                    .unwrap_or(ptr::null()),
                p_rasterization_state: rasterization_state
                    .as_ref()
                    .map(|p| p as *const _)
                    .unwrap_or(ptr::null()),
                p_multisample_state: multisample_state
                    .as_ref()
                    .map(|p| p as *const _)
                    .unwrap_or(ptr::null()),
                p_depth_stencil_state: depth_stencil_state
                    .as_ref()
                    .map(|p| p as *const _)
                    .unwrap_or(ptr::null()),
                p_color_blend_state: color_blend_state
                    .as_ref()
                    .map(|p| p as *const _)
                    .unwrap_or(ptr::null()),
                p_dynamic_state: dynamic_state
                    .as_ref()
                    .map(|s| s as *const _)
                    .unwrap_or(ptr::null()),
                layout: pipeline_layout.internal_object(),
                render_pass: subpass.render_pass().inner().internal_object(),
                subpass: subpass.index(),
                base_pipeline_handle: ash::vk::Pipeline::null(), // TODO:
                base_pipeline_index: -1,                         // TODO:
                ..Default::default()
            };

            let cache_handle = match self.cache.as_ref() {
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
            subpass,
            shaders,

            vertex_input,
            input_assembly_state: self.input_assembly_state,
            tessellation_state: if tessellation_state.is_some() {
                Some(self.tessellation_state)
            } else {
                None
            },
            viewport_state: if viewport_state.is_some() {
                Some(self.viewport_state)
            } else {
                None
            },
            rasterization_state: self.rasterization_state,
            multisample_state: if multisample_state.is_some() {
                Some(self.multisample_state)
            } else {
                None
            },
            depth_stencil_state: if depth_stencil_state.is_some() {
                Some(self.depth_stencil_state)
            } else {
                None
            },
            color_blend_state: if color_blend_state.is_some() {
                Some(self.color_blend_state)
            } else {
                None
            },
            dynamic_state: dynamic_state_modes,
        })
    }

    // TODO: add build_with_cache method
}

impl<'vs, 'tcs, 'tes, 'gs, 'fs, Vdef, Vss, Tcss, Tess, Gss, Fss>
    GraphicsPipelineBuilder<'vs, 'tcs, 'tes, 'gs, 'fs, Vdef, Vss, Tcss, Tess, Gss, Fss>
{
    // TODO: add pipeline derivate system

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
            vertex_shader: Some((shader, specialization_constants)),
            tessellation_shaders: self.tessellation_shaders,
            geometry_shader: self.geometry_shader,
            fragment_shader: self.fragment_shader,

            vertex_definition: self.vertex_definition,
            input_assembly_state: self.input_assembly_state,
            tessellation_state: self.tessellation_state,
            viewport_state: self.viewport_state,
            rasterization_state: self.rasterization_state,
            multisample_state: self.multisample_state,
            depth_stencil_state: self.depth_stencil_state,
            color_blend_state: self.color_blend_state,

            subpass: self.subpass,
            cache: self.cache,
        }
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
            vertex_shader: self.vertex_shader,
            tessellation_shaders: Some(TessellationShaders {
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
            fragment_shader: self.fragment_shader,

            vertex_definition: self.vertex_definition,
            input_assembly_state: self.input_assembly_state,
            tessellation_state: self.tessellation_state,
            viewport_state: self.viewport_state,
            rasterization_state: self.rasterization_state,
            multisample_state: self.multisample_state,
            depth_stencil_state: self.depth_stencil_state,
            color_blend_state: self.color_blend_state,

            subpass: self.subpass,
            cache: self.cache,
        }
    }

    /// Sets the tessellation shaders stage as disabled. This is the default.
    #[deprecated(since = "0.27")]
    #[inline]
    pub fn tessellation_shaders_disabled(mut self) -> Self {
        self.tessellation_shaders = None;
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
            vertex_shader: self.vertex_shader,
            tessellation_shaders: self.tessellation_shaders,
            geometry_shader: Some((shader, specialization_constants)),
            fragment_shader: self.fragment_shader,

            vertex_definition: self.vertex_definition,
            input_assembly_state: self.input_assembly_state,
            tessellation_state: self.tessellation_state,
            viewport_state: self.viewport_state,
            rasterization_state: self.rasterization_state,
            multisample_state: self.multisample_state,
            depth_stencil_state: self.depth_stencil_state,
            color_blend_state: self.color_blend_state,

            subpass: self.subpass,
            cache: self.cache,
        }
    }

    /// Sets the geometry shader stage as disabled. This is the default.
    #[deprecated(since = "0.27")]
    #[inline]
    pub fn geometry_shader_disabled(mut self) -> Self {
        self.geometry_shader = None;
        self
    }

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
            vertex_shader: self.vertex_shader,
            tessellation_shaders: self.tessellation_shaders,
            geometry_shader: self.geometry_shader,
            fragment_shader: Some((shader, specialization_constants)),

            vertex_definition: self.vertex_definition,
            input_assembly_state: self.input_assembly_state,
            tessellation_state: self.tessellation_state,
            viewport_state: self.viewport_state,
            rasterization_state: self.rasterization_state,
            multisample_state: self.multisample_state,
            depth_stencil_state: self.depth_stencil_state,
            color_blend_state: self.color_blend_state,

            subpass: self.subpass,
            cache: self.cache,
        }
    }

    /// Sets the vertex input.
    #[inline]
    pub fn vertex_input<T>(
        self,
        vertex_definition: T,
    ) -> GraphicsPipelineBuilder<'vs, 'tcs, 'tes, 'gs, 'fs, T, Vss, Tcss, Tess, Gss, Fss> {
        GraphicsPipelineBuilder {
            vertex_shader: self.vertex_shader,
            tessellation_shaders: self.tessellation_shaders,
            geometry_shader: self.geometry_shader,
            fragment_shader: self.fragment_shader,

            vertex_definition,
            input_assembly_state: self.input_assembly_state,
            tessellation_state: self.tessellation_state,
            viewport_state: self.viewport_state,
            rasterization_state: self.rasterization_state,
            multisample_state: self.multisample_state,
            depth_stencil_state: self.depth_stencil_state,
            color_blend_state: self.color_blend_state,

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

    /// Sets the input assembly state.
    ///
    /// The default value is [`InputAssemblyState::default()`].
    #[inline]
    pub fn input_assembly_state(mut self, input_assembly_state: InputAssemblyState) -> Self {
        self.input_assembly_state = input_assembly_state;
        self
    }

    /// Sets whether primitive restart is enabled.
    #[deprecated(since = "0.27", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn primitive_restart(mut self, enabled: bool) -> Self {
        self.input_assembly_state.primitive_restart_enable = StateMode::Fixed(enabled);
        self
    }

    /// Sets the topology of the primitives that are expected by the pipeline.
    #[deprecated(since = "0.27", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn primitive_topology(mut self, topology: PrimitiveTopology) -> Self {
        self.input_assembly_state.topology = StateMode::Fixed(topology);
        self
    }

    /// Sets the topology of the primitives to a list of points.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::PointList)`.
    #[deprecated(since = "0.27", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn point_list(self) -> Self {
        self.primitive_topology(PrimitiveTopology::PointList)
    }

    /// Sets the topology of the primitives to a list of lines.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::LineList)`.
    #[deprecated(since = "0.27", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn line_list(self) -> Self {
        self.primitive_topology(PrimitiveTopology::LineList)
    }

    /// Sets the topology of the primitives to a line strip.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::LineStrip)`.
    #[deprecated(since = "0.27", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn line_strip(self) -> Self {
        self.primitive_topology(PrimitiveTopology::LineStrip)
    }

    /// Sets the topology of the primitives to a list of triangles. Note that this is the default.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::TriangleList)`.
    #[deprecated(since = "0.27", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn triangle_list(self) -> Self {
        self.primitive_topology(PrimitiveTopology::TriangleList)
    }

    /// Sets the topology of the primitives to a triangle strip.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::TriangleStrip)`.
    #[deprecated(since = "0.27", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn triangle_strip(self) -> Self {
        self.primitive_topology(PrimitiveTopology::TriangleStrip)
    }

    /// Sets the topology of the primitives to a fan of triangles.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::TriangleFan)`.
    #[deprecated(since = "0.27", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn triangle_fan(self) -> Self {
        self.primitive_topology(PrimitiveTopology::TriangleFan)
    }

    /// Sets the topology of the primitives to a list of lines with adjacency information.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::LineListWithAdjacency)`.
    #[deprecated(since = "0.27", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn line_list_with_adjacency(self) -> Self {
        self.primitive_topology(PrimitiveTopology::LineListWithAdjacency)
    }

    /// Sets the topology of the primitives to a line strip with adjacency information.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::LineStripWithAdjacency)`.
    #[deprecated(since = "0.27", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn line_strip_with_adjacency(self) -> Self {
        self.primitive_topology(PrimitiveTopology::LineStripWithAdjacency)
    }

    /// Sets the topology of the primitives to a list of triangles with adjacency information.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::TriangleListWithAdjacency)`.
    #[deprecated(since = "0.27", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn triangle_list_with_adjacency(self) -> Self {
        self.primitive_topology(PrimitiveTopology::TriangleListWithAdjacency)
    }

    /// Sets the topology of the primitives to a triangle strip with adjacency information`
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::TriangleStripWithAdjacency)`.
    #[deprecated(since = "0.27", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn triangle_strip_with_adjacency(self) -> Self {
        self.primitive_topology(PrimitiveTopology::TriangleStripWithAdjacency)
    }

    /// Sets the topology of the primitives to a list of patches. Can only be used and must be used
    /// with a tessellation shader.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::PatchList)`.
    #[deprecated(since = "0.27", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn patch_list(self) -> Self {
        self.primitive_topology(PrimitiveTopology::PatchList)
    }

    /// Sets the tessellation state. This is required if the pipeline contains tessellation shaders,
    /// and ignored otherwise.
    ///
    /// The default value is [`TessellationState::default()`].
    #[inline]
    pub fn tessellation_state(mut self, tessellation_state: TessellationState) -> Self {
        self.tessellation_state = tessellation_state;
        self
    }

    /// Sets the viewport state.
    ///
    /// The default value is [`ViewportState::default()`].
    #[inline]
    pub fn viewport_state(mut self, viewport_state: ViewportState) -> Self {
        self.viewport_state = viewport_state;
        self
    }

    /// Sets the viewports to some value, and the scissor boxes to boxes that always cover the
    /// whole viewport.
    #[deprecated(since = "0.27", note = "Use `viewport_state` instead")]
    #[inline]
    pub fn viewports<I>(self, viewports: I) -> Self
    where
        I: IntoIterator<Item = Viewport>,
    {
        self.viewports_scissors(viewports.into_iter().map(|v| (v, Scissor::irrelevant())))
    }

    /// Sets the characteristics of viewports and scissor boxes in advance.
    #[deprecated(since = "0.27", note = "Use `viewport_state` instead")]
    #[inline]
    pub fn viewports_scissors<I>(mut self, viewports: I) -> Self
    where
        I: IntoIterator<Item = (Viewport, Scissor)>,
    {
        self.viewport_state = ViewportState::Fixed {
            data: viewports.into_iter().collect(),
        };
        self
    }

    /// Sets the scissor boxes to some values, and viewports to dynamic. The viewports will
    /// need to be set before drawing.
    #[deprecated(since = "0.27", note = "Use `viewport_state` instead")]
    #[inline]
    pub fn viewports_dynamic_scissors_fixed<I>(mut self, scissors: I) -> Self
    where
        I: IntoIterator<Item = Scissor>,
    {
        self.viewport_state = ViewportState::FixedScissor {
            scissors: scissors.into_iter().collect(),
            viewport_count_dynamic: false,
        };
        self
    }

    /// Sets the viewports to dynamic, and the scissor boxes to boxes that always cover the whole
    /// viewport. The viewports will need to be set before drawing.
    #[deprecated(since = "0.27", note = "Use `viewport_state` instead")]
    #[inline]
    pub fn viewports_dynamic_scissors_irrelevant(mut self, num: u32) -> Self {
        self.viewport_state = ViewportState::FixedScissor {
            scissors: (0..num).map(|_| Scissor::irrelevant()).collect(),
            viewport_count_dynamic: false,
        };
        self
    }

    /// Sets the viewports to some values, and scissor boxes to dynamic. The scissor boxes will
    /// need to be set before drawing.
    #[deprecated(since = "0.27", note = "Use `viewport_state` instead")]
    #[inline]
    pub fn viewports_fixed_scissors_dynamic<I>(mut self, viewports: I) -> Self
    where
        I: IntoIterator<Item = Viewport>,
    {
        self.viewport_state = ViewportState::FixedViewport {
            viewports: viewports.into_iter().collect(),
            scissor_count_dynamic: false,
        };
        self
    }

    /// Sets the viewports and scissor boxes to dynamic. They will both need to be set before
    /// drawing.
    #[deprecated(since = "0.27", note = "Use `viewport_state` instead")]
    #[inline]
    pub fn viewports_scissors_dynamic(mut self, count: u32) -> Self {
        self.viewport_state = ViewportState::Dynamic {
            count,
            viewport_count_dynamic: false,
            scissor_count_dynamic: false,
        };
        self
    }

    /// Sets the rasterization state.
    ///
    /// The default value is [`RasterizationState::default()`].
    #[inline]
    pub fn rasterization_state(mut self, rasterization_state: RasterizationState) -> Self {
        self.rasterization_state = rasterization_state;
        self
    }

    /// If true, then the depth value of the vertices will be clamped to the range `[0.0 ; 1.0]`.
    /// If false, fragments whose depth is outside of this range will be discarded before the
    /// fragment shader even runs.
    #[deprecated(since = "0.27", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn depth_clamp(mut self, clamp: bool) -> Self {
        self.rasterization_state.depth_clamp_enable = clamp;
        self
    }

    /// Sets the front-facing faces to counter-clockwise faces. This is the default.
    ///
    /// Triangles whose vertices are oriented counter-clockwise on the screen will be considered
    /// as facing their front. Otherwise they will be considered as facing their back.
    #[deprecated(since = "0.27", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn front_face_counter_clockwise(mut self) -> Self {
        self.rasterization_state.front_face = StateMode::Fixed(FrontFace::CounterClockwise);
        self
    }

    /// Sets the front-facing faces to clockwise faces.
    ///
    /// Triangles whose vertices are oriented clockwise on the screen will be considered
    /// as facing their front. Otherwise they will be considered as facing their back.
    #[deprecated(since = "0.27", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn front_face_clockwise(mut self) -> Self {
        self.rasterization_state.front_face = StateMode::Fixed(FrontFace::Clockwise);
        self
    }

    /// Sets backface culling as disabled. This is the default.
    #[deprecated(since = "0.27", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn cull_mode_disabled(mut self) -> Self {
        self.rasterization_state.cull_mode = StateMode::Fixed(CullMode::None);
        self
    }

    /// Sets backface culling to front faces. The front faces (as chosen with the `front_face_*`
    /// methods) will be discarded by the GPU when drawing.
    #[deprecated(since = "0.27", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn cull_mode_front(mut self) -> Self {
        self.rasterization_state.cull_mode = StateMode::Fixed(CullMode::Front);
        self
    }

    /// Sets backface culling to back faces. Faces that are not facing the front (as chosen with
    /// the `front_face_*` methods) will be discarded by the GPU when drawing.
    #[deprecated(since = "0.27", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn cull_mode_back(mut self) -> Self {
        self.rasterization_state.cull_mode = StateMode::Fixed(CullMode::Back);
        self
    }

    /// Sets backface culling to both front and back faces. All the faces will be discarded.
    ///
    /// > **Note**: This option exists for the sake of completeness. It has no known practical
    /// > usage.
    #[deprecated(since = "0.27", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn cull_mode_front_and_back(mut self) -> Self {
        self.rasterization_state.cull_mode = StateMode::Fixed(CullMode::FrontAndBack);
        self
    }

    /// Sets the polygon mode to "fill". This is the default.
    #[deprecated(since = "0.27", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn polygon_mode_fill(mut self) -> Self {
        self.rasterization_state.polygon_mode = PolygonMode::Fill;
        self
    }

    /// Sets the polygon mode to "line". Triangles will each be turned into three lines.
    #[deprecated(since = "0.27", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn polygon_mode_line(mut self) -> Self {
        self.rasterization_state.polygon_mode = PolygonMode::Line;
        self
    }

    /// Sets the polygon mode to "point". Triangles and lines will each be turned into three points.
    #[deprecated(since = "0.27", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn polygon_mode_point(mut self) -> Self {
        self.rasterization_state.polygon_mode = PolygonMode::Point;
        self
    }

    /// Sets the width of the lines, if the GPU needs to draw lines. The default is `1.0`.
    #[deprecated(since = "0.27", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn line_width(mut self, value: f32) -> Self {
        self.rasterization_state.line_width = StateMode::Fixed(value);
        self
    }

    /// Sets the width of the lines as dynamic, which means that you will need to set this value
    /// when drawing.
    #[deprecated(since = "0.27", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn line_width_dynamic(mut self) -> Self {
        self.rasterization_state.line_width = StateMode::Dynamic;
        self
    }

    /// Sets the multisample state.
    ///
    /// The default value is [`MultisampleState::default()`].
    #[inline]
    pub fn multisample_state(mut self, multisample_state: MultisampleState) -> Self {
        self.multisample_state = multisample_state;
        self
    }

    /// Disables sample shading. The fragment shader will only be run once per fragment (ie. per
    /// pixel) and not once by sample. The output will then be copied in all of the covered
    /// samples.
    ///
    /// Sample shading is disabled by default.
    #[deprecated(since = "0.27", note = "Use `multisample_state` instead")]
    #[inline]
    pub fn sample_shading_disabled(mut self) -> Self {
        self.multisample_state.sample_shading = None;
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
    #[deprecated(since = "0.27", note = "Use `multisample_state` instead")]
    #[inline]
    pub fn sample_shading_enabled(mut self, min_fract: f32) -> Self {
        assert!(min_fract >= 0.0 && min_fract <= 1.0);
        self.multisample_state.sample_shading = Some(min_fract);
        self
    }

    // TODO: doc
    #[deprecated(since = "0.27", note = "Use `multisample_state` instead")]
    pub fn alpha_to_coverage_disabled(mut self) -> Self {
        self.multisample_state.alpha_to_coverage_enable = false;
        self
    }

    // TODO: doc
    #[deprecated(since = "0.27", note = "Use `multisample_state` instead")]
    pub fn alpha_to_coverage_enabled(mut self) -> Self {
        self.multisample_state.alpha_to_coverage_enable = true;
        self
    }

    /// Disables alpha-to-one.
    ///
    /// Alpha-to-one is disabled by default.
    #[deprecated(since = "0.27", note = "Use `multisample_state` instead")]
    #[inline]
    pub fn alpha_to_one_disabled(mut self) -> Self {
        self.multisample_state.alpha_to_one_enable = false;
        self
    }

    /// Enables alpha-to-one. The alpha component of the first color output of the fragment shader
    /// will be replaced by the value `1.0`.
    ///
    /// Enabling alpha-to-one requires the `alpha_to_one` feature to be enabled on the device.
    ///
    /// Alpha-to-one is disabled by default.
    #[deprecated(since = "0.27", note = "Use `multisample_state` instead")]
    #[inline]
    pub fn alpha_to_one_enabled(mut self) -> Self {
        self.multisample_state.alpha_to_one_enable = true;
        self
    }

    /// Sets the depth/stencil state.
    ///
    /// The default value is [`DepthStencilState::default()`].
    #[inline]
    pub fn depth_stencil_state(mut self, depth_stencil_state: DepthStencilState) -> Self {
        self.depth_stencil_state = depth_stencil_state;
        self
    }

    /// Sets the depth/stencil state.
    #[deprecated(since = "0.27", note = "Use `depth_stencil_state` instead")]
    #[inline]
    pub fn depth_stencil(self, depth_stencil_state: DepthStencilState) -> Self {
        self.depth_stencil_state(depth_stencil_state)
    }

    /// Sets the depth/stencil tests as disabled.
    ///
    /// > **Note**: This is a shortcut for all the other `depth_*` and `depth_stencil_*` methods
    /// > of the builder.
    #[deprecated(since = "0.27", note = "Use `depth_stencil_state` instead")]
    #[inline]
    pub fn depth_stencil_disabled(mut self) -> Self {
        self.depth_stencil_state = DepthStencilState::disabled();
        self
    }

    /// Sets the depth/stencil tests as a simple depth test and no stencil test.
    ///
    /// > **Note**: This is a shortcut for setting the depth test to `Less`, the depth write Into
    /// > ` true` and disable the stencil test.
    #[deprecated(since = "0.27", note = "Use `depth_stencil_state` instead")]
    #[inline]
    pub fn depth_stencil_simple_depth(mut self) -> Self {
        self.depth_stencil_state = DepthStencilState::simple_depth_test();
        self
    }

    /// Sets whether the depth buffer will be written.
    #[deprecated(since = "0.27", note = "Use `depth_stencil_state` instead")]
    #[inline]
    pub fn depth_write(mut self, write: bool) -> Self {
        let depth_state = self
            .depth_stencil_state
            .depth
            .get_or_insert(Default::default());
        depth_state.write_enable = StateMode::Fixed(write);
        self
    }

    /// Sets the color blend state.
    ///
    /// The default value is [`ColorBlendState::default()`].
    #[inline]
    pub fn color_blend_state(mut self, color_blend_state: ColorBlendState) -> Self {
        self.color_blend_state = color_blend_state;
        self
    }

    #[deprecated(since = "0.27", note = "Use `color_blend_state` instead")]
    #[inline]
    pub fn blend_collective(mut self, blend: AttachmentBlend) -> Self {
        self.color_blend_state.attachments = vec![ColorBlendAttachmentState {
            blend: Some(blend),
            color_write_mask: ColorComponents::all(),
        }];
        self
    }

    #[deprecated(since = "0.27", note = "Use `color_blend_state` instead")]
    #[inline]
    pub fn blend_individual<I>(mut self, blend: I) -> Self
    where
        I: IntoIterator<Item = AttachmentBlend>,
    {
        self.color_blend_state.attachments = blend
            .into_iter()
            .map(|x| ColorBlendAttachmentState {
                blend: Some(x),
                color_write_mask: ColorComponents::all(),
            })
            .collect();
        self
    }

    /// Each fragment shader output will have its value directly written to the framebuffer
    /// attachment. This is the default.
    #[deprecated(since = "0.27", note = "Use `color_blend_state` instead")]
    #[inline]
    pub fn blend_pass_through(mut self) -> Self {
        self.color_blend_state.attachments = vec![ColorBlendAttachmentState {
            blend: None,
            color_write_mask: ColorComponents::all(),
        }];
        self
    }

    #[deprecated(since = "0.27", note = "Use `color_blend_state` instead")]
    #[inline]
    pub fn blend_alpha_blending(mut self) -> Self {
        self.color_blend_state.attachments = vec![ColorBlendAttachmentState {
            blend: Some(AttachmentBlend::alpha()),
            color_write_mask: ColorComponents::all(),
        }];
        self
    }

    #[deprecated(since = "0.27", note = "Use `color_blend_state` instead")]
    #[inline]
    pub fn blend_logic_op(mut self, logic_op: LogicOp) -> Self {
        self.color_blend_state.logic_op = Some(StateMode::Fixed(logic_op));
        self
    }

    /// Sets the logic operation as disabled. This is the default.
    #[deprecated(since = "0.27", note = "Use `color_blend_state` instead")]
    #[inline]
    pub fn blend_logic_op_disabled(mut self) -> Self {
        self.color_blend_state.logic_op = None;
        self
    }

    /// Sets the blend constant. The default is `[0.0, 0.0, 0.0, 0.0]`.
    ///
    /// The blend constant is used for some blending calculations. It is irrelevant otherwise.
    #[deprecated(since = "0.27", note = "Use `color_blend_state` instead")]
    #[inline]
    pub fn blend_constants(mut self, constants: [f32; 4]) -> Self {
        self.color_blend_state.blend_constants = StateMode::Fixed(constants);
        self
    }

    /// Sets the blend constant value as dynamic. Its value will need to be set before drawing.
    ///
    /// The blend constant is used for some blending calculations. It is irrelevant otherwise.
    #[deprecated(since = "0.27", note = "Use `color_blend_state` instead")]
    #[inline]
    pub fn blend_constants_dynamic(mut self) -> Self {
        self.color_blend_state.blend_constants = StateMode::Dynamic;
        self
    }

    /// Sets the render pass subpass to use.
    #[inline]
    pub fn render_pass(self, subpass: Subpass) -> Self {
        GraphicsPipelineBuilder {
            vertex_shader: self.vertex_shader,
            tessellation_shaders: self.tessellation_shaders,
            geometry_shader: self.geometry_shader,
            fragment_shader: self.fragment_shader,

            vertex_definition: self.vertex_definition,
            input_assembly_state: self.input_assembly_state,
            tessellation_state: self.tessellation_state,
            viewport_state: self.viewport_state,
            rasterization_state: self.rasterization_state,
            multisample_state: self.multisample_state,
            depth_stencil_state: self.depth_stencil_state,
            color_blend_state: self.color_blend_state,

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
            vertex_shader: self.vertex_shader.clone(),
            tessellation_shaders: self.tessellation_shaders.clone(),
            geometry_shader: self.geometry_shader.clone(),
            fragment_shader: self.fragment_shader.clone(),

            vertex_definition: self.vertex_definition.clone(),
            input_assembly_state: self.input_assembly_state,
            tessellation_state: self.tessellation_state,
            viewport_state: self.viewport_state.clone(),
            rasterization_state: self.rasterization_state.clone(),
            multisample_state: self.multisample_state,
            depth_stencil_state: self.depth_stencil_state.clone(),
            color_blend_state: self.color_blend_state.clone(),

            subpass: self.subpass.clone(),
            cache: self.cache.clone(),
        }
    }
}
