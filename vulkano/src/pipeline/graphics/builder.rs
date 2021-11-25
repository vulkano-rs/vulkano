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
use crate::format::NumericType;
use crate::pipeline::cache::PipelineCache;
use crate::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorBlendState, ColorComponents, LogicOp,
};
use crate::pipeline::graphics::depth_stencil::DepthStencilState;
use crate::pipeline::graphics::discard_rectangle::DiscardRectangleState;
use crate::pipeline::graphics::input_assembly::{InputAssemblyState, PrimitiveTopology};
use crate::pipeline::graphics::multisample::MultisampleState;
use crate::pipeline::graphics::rasterization::{
    CullMode, FrontFace, PolygonMode, RasterizationState,
};
use crate::pipeline::graphics::tessellation::TessellationState;
use crate::pipeline::graphics::vertex_input::{
    BuffersDefinition, Vertex, VertexDefinition, VertexInputState,
};
use crate::pipeline::graphics::viewport::{Scissor, Viewport, ViewportState};
use crate::pipeline::graphics::{GraphicsPipeline, GraphicsPipelineCreationError};
use crate::pipeline::layout::{PipelineLayout, PipelineLayoutCreationError, PipelineLayoutPcRange};
use crate::pipeline::{DynamicState, PartialStateMode, StateMode};
use crate::render_pass::Subpass;
use crate::shader::{
    DescriptorRequirements, EntryPoint, ShaderExecution, ShaderStage, SpecializationConstants,
    SpecializationMapEntry,
};
use crate::DeviceSize;
use crate::VulkanObject;
use fnv::FnvHashMap;
use smallvec::SmallVec;
use std::collections::hash_map::{Entry, HashMap};
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;
use std::u32;

/// Prototype for a `GraphicsPipeline`.
#[derive(Debug)]
pub struct GraphicsPipelineBuilder<'vs, 'tcs, 'tes, 'gs, 'fs, Vdef, Vss, Tcss, Tess, Gss, Fss> {
    subpass: Option<Subpass>,
    cache: Option<Arc<PipelineCache>>,

    vertex_shader: Option<(EntryPoint<'vs>, Vss)>,
    tessellation_shaders: Option<TessellationShaders<'tcs, 'tes, Tcss, Tess>>,
    geometry_shader: Option<(EntryPoint<'gs>, Gss)>,
    fragment_shader: Option<(EntryPoint<'fs>, Fss)>,

    vertex_input_state: Vdef,
    input_assembly_state: InputAssemblyState,
    tessellation_state: TessellationState,
    viewport_state: ViewportState,
    discard_rectangle_state: DiscardRectangleState,
    rasterization_state: RasterizationState,
    multisample_state: MultisampleState,
    depth_stencil_state: DepthStencilState,
    color_blend_state: ColorBlendState,
}

// Additional parameters if tessellation is used.
#[derive(Clone, Debug)]
struct TessellationShaders<'tcs, 'tes, Tcss, Tess> {
    control: (EntryPoint<'tcs>, Tcss),
    evaluation: (EntryPoint<'tes>, Tess),
}

impl
    GraphicsPipelineBuilder<
        'static,
        'static,
        'static,
        'static,
        'static,
        VertexInputState,
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
            subpass: None,
            cache: None,

            vertex_shader: None,
            tessellation_shaders: None,
            geometry_shader: None,
            fragment_shader: None,

            vertex_input_state: Default::default(),
            input_assembly_state: Default::default(),
            tessellation_state: Default::default(),
            viewport_state: Default::default(),
            discard_rectangle_state: Default::default(),
            rasterization_state: Default::default(),
            multisample_state: Default::default(),
            depth_stencil_state: Default::default(),
            color_blend_state: Default::default(),
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
    ) -> Result<Arc<GraphicsPipeline>, GraphicsPipelineCreationError> {
        self.with_auto_layout(device, |_| {})
    }

    /// The same as `new`, but allows you to provide a closure that is given a mutable reference to
    /// the inferred descriptor set definitions. This can be used to make changes to the layout
    /// before it's created, for example to add dynamic buffers or immutable samplers.
    pub fn with_auto_layout<F>(
        self,
        device: Arc<Device>,
        func: F,
    ) -> Result<Arc<GraphicsPipeline>, GraphicsPipelineCreationError>
    where
        F: FnOnce(&mut [DescriptorSetDesc]),
    {
        let (descriptor_set_layout_descs, push_constant_ranges) = {
            let stages: SmallVec<[&EntryPoint; 5]> = std::array::IntoIter::new([
                self.vertex_shader.as_ref().map(|s| &s.0),
                self.tessellation_shaders.as_ref().map(|s| &s.control.0),
                self.tessellation_shaders.as_ref().map(|s| &s.evaluation.0),
                self.geometry_shader.as_ref().map(|s| &s.0),
                self.fragment_shader.as_ref().map(|s| &s.0),
            ])
            .flatten()
            .collect();

            // Produce `DescriptorRequirements` for each binding, by iterating over all shaders
            // and adding the requirements of each.
            let mut descriptor_requirements: FnvHashMap<(u32, u32), DescriptorRequirements> =
                HashMap::default();

            for (loc, reqs) in stages
                .iter()
                .map(|shader| shader.descriptor_requirements())
                .flatten()
            {
                match descriptor_requirements.entry(loc) {
                    Entry::Occupied(entry) => {
                        // Previous shaders already added requirements, so we produce the
                        // intersection of the previous requirements and those of the
                        // current shader.
                        let previous = entry.into_mut();
                        *previous = previous.intersection(reqs).expect("Could not produce an intersection of the shader descriptor requirements");
                    }
                    Entry::Vacant(entry) => {
                        // No previous shader had this descriptor yet, so we just insert the
                        // requirements.
                        entry.insert(reqs.clone());
                    }
                }
            }

            // Build a description of a descriptor set layout from the shader requirements, then
            // feed it to the user-provided closure to allow tweaking.
            let mut descriptor_set_layout_descs = DescriptorSetDesc::from_requirements(
                descriptor_requirements
                    .iter()
                    .map(|(&loc, reqs)| (loc, reqs)),
            );
            func(&mut descriptor_set_layout_descs);

            // We want to union each push constant range into a set of ranges that do not have intersecting stage flags.
            // e.g. The range [0, 16) is either made available to Vertex | Fragment or we only make [0, 16) available to
            // Vertex and a subrange available to Fragment, like [0, 8)
            let mut range_map = HashMap::new();
            for stage in stages.iter() {
                if let Some(range) = stage.push_constant_requirements() {
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
            .map(|desc| Ok(DescriptorSetLayout::new(device.clone(), desc)?))
            .collect::<Result<Vec<_>, PipelineLayoutCreationError>>()?;
        let pipeline_layout =
            PipelineLayout::new(device.clone(), descriptor_set_layouts, push_constant_ranges)
                .unwrap();
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
    ) -> Result<Arc<GraphicsPipeline>, GraphicsPipelineCreationError> {
        // TODO: return errors instead of panicking if missing param

        let fns = device.fns();
        let subpass = self.subpass.take().expect("Missing subpass");
        let self_vertex_input_state = self
            .vertex_input_state
            .definition(self.vertex_shader.as_ref().unwrap().0.input_interface())?;

        // Check individual shaders
        if let Some(vertex_shader) = &self.vertex_shader {
            match self.vertex_shader.as_ref().unwrap().0.execution() {
                ShaderExecution::Vertex => (),
                _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
            }

            // Check that the vertex input state contains attributes for all the shader's input
            // variables.
            for element in vertex_shader.0.input_interface().elements() {
                assert!(!element.ty.is_64bit); // TODO: implement
                let location_range =
                    element.location..element.location + element.ty.num_locations();

                for location in location_range {
                    let attribute_desc = match self_vertex_input_state.attributes.get(&location) {
                        Some(attribute_desc) => attribute_desc,
                        None => {
                            return Err(
                                GraphicsPipelineCreationError::VertexInputAttributeMissing {
                                    location,
                                },
                            )
                        }
                    };

                    // TODO: Check component assignments too. Multiple variables can occupy the same
                    // location but in different components.

                    let shader_type = element.ty.to_format().type_color().unwrap();
                    let attribute_type = attribute_desc.format.type_color().unwrap();

                    if !matches!(
                        (shader_type, attribute_type),
                        (
                            NumericType::SFLOAT
                                | NumericType::UFLOAT
                                | NumericType::SNORM
                                | NumericType::UNORM
                                | NumericType::SSCALED
                                | NumericType::USCALED
                                | NumericType::SRGB,
                            NumericType::SFLOAT
                                | NumericType::UFLOAT
                                | NumericType::SNORM
                                | NumericType::UNORM
                                | NumericType::SSCALED
                                | NumericType::USCALED
                                | NumericType::SRGB,
                        ) | (NumericType::SINT, NumericType::SINT)
                            | (NumericType::UINT, NumericType::UINT)
                    ) {
                        return Err(
                            GraphicsPipelineCreationError::VertexInputAttributeIncompatibleFormat {
                                location,
                                shader_type,
                                attribute_type,
                            },
                        );
                    }
                }
            }
        } else {
            panic!("Missing vertex shader"); // TODO: return error
        }

        if let Some(tessellation_shaders) = &self.tessellation_shaders {
            // FIXME: must check that the control shader and evaluation shader are compatible

            if !device.enabled_features().tessellation_shader {
                return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                    feature: "tessellation_shader",
                    reason: "a tessellation shader was provided",
                });
            }

            match tessellation_shaders.control.0.execution() {
                ShaderExecution::TessellationControl => (),
                _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
            }

            match tessellation_shaders.evaluation.0.execution() {
                ShaderExecution::TessellationEvaluation => (),
                _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
            }

            if let Some(multiview) = subpass.render_pass().desc().multiview().as_ref() {
                if multiview.used_layer_count() > 0
                    && !device.enabled_features().multiview_tessellation_shader
                {
                    return Err(
                        GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "multiview_tessellation_shader",
                            reason: "a tessellation shader was provided, and the render pass has multiview enabled with more than zero layers used",
                        },
                    );
                }
            }
        }

        if let Some(geometry_shader) = &self.geometry_shader {
            if !device.enabled_features().geometry_shader {
                return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                    feature: "geometry_shader",
                    reason: "a geometry shader was provided",
                });
            }

            let input = match geometry_shader.0.execution() {
                ShaderExecution::Geometry(execution) => execution.input,
                _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
            };

            if let PartialStateMode::Fixed(topology) = self.input_assembly_state.topology {
                if !input.is_compatible_with(topology) {
                    return Err(GraphicsPipelineCreationError::TopologyNotMatchingGeometryShader);
                }
            }

            if let Some(multiview) = subpass.render_pass().desc().multiview().as_ref() {
                if multiview.used_layer_count() > 0
                    && !device.enabled_features().multiview_geometry_shader
                {
                    return Err(
                        GraphicsPipelineCreationError::FeatureNotEnabled {
                            feature: "multiview_geometry_shader",
                            reason: "a geometry shader was provided, and the render pass has multiview enabled with more than zero layers used",
                        },
                    );
                }
            }

            // TODO: VUID-VkGraphicsPipelineCreateInfo-pStages-00739
            // If the pipeline is being created with pre-rasterization shader state and pStages
            // includes a geometry shader stage, and also includes tessellation shader stages,
            // its shader code must contain an OpExecutionMode instruction that specifies an
            // input primitive type that is compatible with the primitive topology that is
            // output by the tessellation stages
        }

        if let Some(fragment_shader) = &self.fragment_shader {
            match fragment_shader.0.execution() {
                ShaderExecution::Fragment => (),
                _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
            }

            // Check that the subpass can accept the output of the fragment shader.
            // TODO: If there is no fragment shader, what should be checked then? The previous stage?
            if !subpass.is_compatible_with(fragment_shader.0.output_interface()) {
                return Err(GraphicsPipelineCreationError::FragmentShaderRenderPassIncompatible);
            }
        } else {
            // TODO: should probably error out here at least under some circumstances?
        }

        // Collect all the shader stages for operations not specific to any shader type.
        struct ShaderStageInfo<'a> {
            entry_point: &'a EntryPoint<'a>,
            specialization_map_entries: &'a [SpecializationMapEntry],
            specialization_data: &'a [u8],
        }

        let stages_info: SmallVec<[ShaderStageInfo; 5]> = std::array::IntoIter::new([
            self.vertex_shader
                .as_ref()
                .map(|(entry_point, spec_consts)| ShaderStageInfo {
                    entry_point,
                    specialization_map_entries: Vss::descriptors(),
                    specialization_data: unsafe {
                        std::slice::from_raw_parts(
                            spec_consts as *const _ as *const u8,
                            std::mem::size_of_val(spec_consts),
                        )
                    },
                }),
            self.tessellation_shaders.as_ref().map(
                |TessellationShaders {
                     control: (entry_point, spec_consts),
                     ..
                 }| ShaderStageInfo {
                    entry_point,
                    specialization_map_entries: Tcss::descriptors(),
                    specialization_data: unsafe {
                        std::slice::from_raw_parts(
                            spec_consts as *const _ as *const u8,
                            std::mem::size_of_val(spec_consts),
                        )
                    },
                },
            ),
            self.tessellation_shaders.as_ref().map(
                |TessellationShaders {
                     evaluation: (entry_point, spec_consts),
                     ..
                 }| ShaderStageInfo {
                    entry_point,
                    specialization_map_entries: Tess::descriptors(),
                    specialization_data: unsafe {
                        std::slice::from_raw_parts(
                            spec_consts as *const _ as *const u8,
                            std::mem::size_of_val(spec_consts),
                        )
                    },
                },
            ),
            self.geometry_shader
                .as_ref()
                .map(|(entry_point, spec_consts)| ShaderStageInfo {
                    entry_point,
                    specialization_map_entries: Gss::descriptors(),
                    specialization_data: unsafe {
                        std::slice::from_raw_parts(
                            spec_consts as *const _ as *const u8,
                            std::mem::size_of_val(spec_consts),
                        )
                    },
                }),
            self.fragment_shader
                .as_ref()
                .map(|(entry_point, spec_consts)| ShaderStageInfo {
                    entry_point,
                    specialization_map_entries: Fss::descriptors(),
                    specialization_data: unsafe {
                        std::slice::from_raw_parts(
                            spec_consts as *const _ as *const u8,
                            std::mem::size_of_val(spec_consts),
                        )
                    },
                }),
        ])
        .flatten()
        .collect();

        let mut descriptor_requirements: FnvHashMap<(u32, u32), DescriptorRequirements> =
            HashMap::default();

        // Check that the pipeline layout is compatible with the shader requirements, and collect
        // an intersection of the requirements.
        // TODO: more details in the errors
        for stage_info in &stages_info {
            pipeline_layout.ensure_compatible_with_shader(
                stage_info.entry_point.descriptor_requirements(),
                stage_info.entry_point.push_constant_requirements(),
            )?;

            for (loc, reqs) in stage_info.entry_point.descriptor_requirements() {
                match descriptor_requirements.entry(loc) {
                    Entry::Occupied(entry) => {
                        let previous = entry.into_mut();
                        *previous = previous.intersection(reqs).expect("Could not produce an intersection of the shader descriptor requirements");
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(reqs.clone());
                    }
                }
            }
        }

        let num_used_descriptor_sets = descriptor_requirements
            .keys()
            .map(|loc| loc.0)
            .max()
            .map(|x| x + 1)
            .unwrap_or(0);

        // Check that the input interface of each shader matches the output interface of the
        // previous shader.
        // TODO: this check is too strict; the output only has to be a superset, any variables
        // not used in the input of the next shader are just ignored.
        for (output, input) in stages_info.iter().zip(stages_info.iter().skip(1)) {
            if let Err(err) = input
                .entry_point
                .input_interface()
                .matches(output.entry_point.output_interface())
            {
                return Err(GraphicsPipelineCreationError::ShaderStagesMismatch(err));
            }
        }

        // Creating the specialization for each stage.
        let specialization: SmallVec<[_; 5]> = stages_info
            .iter()
            .map(|stage_info| {
                for (constant_id, reqs) in stage_info
                    .entry_point
                    .specialization_constant_requirements()
                {
                    let map_entry = stage_info
                        .specialization_map_entries
                        .iter()
                        .find(|desc| desc.constant_id == constant_id)
                        .ok_or(
                            GraphicsPipelineCreationError::IncompatibleSpecializationConstants,
                        )?;

                    if map_entry.size as DeviceSize != reqs.size {
                        return Err(
                            GraphicsPipelineCreationError::IncompatibleSpecializationConstants,
                        );
                    }
                }

                Ok(ash::vk::SpecializationInfo {
                    map_entry_count: stage_info.specialization_map_entries.len() as u32,
                    p_map_entries: stage_info.specialization_map_entries.as_ptr() as *const _,
                    data_size: stage_info.specialization_data.len(),
                    p_data: stage_info.specialization_data.as_ptr() as *const _,
                })
            })
            .collect::<Result<_, _>>()?;

        debug_assert_eq!(specialization.len(), stages_info.len());

        // Creating the shader stages themselves.
        let stages: SmallVec<[_; 5]> = stages_info
            .iter()
            .zip(&specialization)
            .map(|(stage_info, specialization)| {
                let stage = ShaderStage::from(*stage_info.entry_point.execution());

                ash::vk::PipelineShaderStageCreateInfo {
                    flags: ash::vk::PipelineShaderStageCreateFlags::empty(),
                    stage: stage.into(),
                    module: stage_info.entry_point.module().internal_object(),
                    p_name: stage_info.entry_point.name().as_ptr(),
                    p_specialization_info: specialization as *const _,
                    ..Default::default()
                }
            })
            .collect();

        debug_assert_eq!(stages.len(), stages_info.len());

        // Will contain the list of dynamic states. Filled while checking the states.
        let mut dynamic_state_modes: FnvHashMap<DynamicState, bool> = HashMap::default();

        // Vertex input state
        let binding_descriptions = self_vertex_input_state.to_vulkan_bindings(&device)?;
        let attribute_descriptions = self_vertex_input_state.to_vulkan_attributes(&device)?;
        let binding_divisor_descriptions =
            self_vertex_input_state.to_vulkan_binding_divisors(&device)?;
        let binding_divisor_state = self_vertex_input_state
            .to_vulkan_binding_divisor_state(binding_divisor_descriptions.as_ref());
        let vertex_input_state = Some(self_vertex_input_state.to_vulkan(
            &device,
            &mut dynamic_state_modes,
            &binding_descriptions,
            &attribute_descriptions,
            binding_divisor_state.as_ref(),
        ));

        // Input assembly state
        let input_assembly_state = if self.vertex_shader.is_some() {
            Some(
                self.input_assembly_state
                    .to_vulkan(&device, &mut dynamic_state_modes)?,
            )
        } else {
            None
        };

        // Tessellation state
        let tessellation_state = if self.tessellation_shaders.is_some() {
            Some(self.tessellation_state.to_vulkan(
                &device,
                &mut dynamic_state_modes,
                &self.input_assembly_state,
            )?)
        } else {
            None
        };

        // Viewport state
        let (viewport_count, viewports, scissor_count, scissors) = self
            .viewport_state
            .to_vulkan_viewports_scissors(&device, &mut dynamic_state_modes)?;
        let viewport_state = Some(self.viewport_state.to_vulkan(
            &device,
            &mut dynamic_state_modes,
            viewport_count,
            &viewports,
            scissor_count,
            &scissors,
        )?);

        // Discard rectangle state
        let discard_rectangles = self
            .discard_rectangle_state
            .to_vulkan_rectangles(&device, &mut dynamic_state_modes)?;
        let mut discard_rectangle_state = self.discard_rectangle_state.to_vulkan(
            &device,
            &mut dynamic_state_modes,
            &discard_rectangles,
        )?;

        // Rasterization state
        let mut rasterization_line_state = self
            .rasterization_state
            .to_vulkan_line_state(&device, &mut dynamic_state_modes)?;
        let rasterization_state = Some(self.rasterization_state.to_vulkan(
            &device,
            &mut dynamic_state_modes,
            rasterization_line_state.as_mut(),
        )?);

        // Fragment shader state
        let has_fragment_shader_state =
            self.rasterization_state.rasterizer_discard_enable != StateMode::Fixed(true);

        // Multisample state
        let multisample_state = if has_fragment_shader_state {
            Some(
                self.multisample_state
                    .to_vulkan(&device, &mut dynamic_state_modes, &subpass)?,
            )
        } else {
            None
        };

        // Depth/stencil state
        let depth_stencil_state = if has_fragment_shader_state
            && subpass.subpass_desc().depth_stencil.is_some()
        {
            Some(
                self.depth_stencil_state
                    .to_vulkan(&device, &mut dynamic_state_modes, &subpass)?,
            )
        } else {
            None
        };

        // Color blend state
        let (color_blend_attachments, color_write_enables) = self
            .color_blend_state
            .to_vulkan_attachments(&device, &mut dynamic_state_modes, &subpass)?;
        let mut color_write = self.color_blend_state.to_vulkan_color_write(
            &device,
            &mut dynamic_state_modes,
            &color_write_enables,
        )?;
        let color_blend_state = if has_fragment_shader_state {
            Some(self.color_blend_state.to_vulkan(
                &device,
                &mut dynamic_state_modes,
                &color_blend_attachments,
                color_write.as_mut(),
            )?)
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

        // Dynamic states not handled yet:
        // - ViewportWScaling (VkPipelineViewportWScalingStateCreateInfoNV)
        // - DiscardRectangle (VkPipelineDiscardRectangleStateCreateInfoEXT)
        // - SampleLocations (VkPipelineSampleLocationsStateCreateInfoEXT)
        // - ViewportShadingRatePalette (VkPipelineViewportShadingRateImageStateCreateInfoNV)
        // - ViewportCoarseSampleOrder (VkPipelineViewportCoarseSampleOrderStateCreateInfoNV)
        // - ExclusiveScissor (VkPipelineViewportExclusiveScissorStateCreateInfoNV)
        // - FragmentShadingRate (VkPipelineFragmentShadingRateStateCreateInfoKHR)

        // Finally, create the pipeline itself.
        let handle = unsafe {
            let mut create_info = ash::vk::GraphicsPipelineCreateInfo {
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
                render_pass: subpass.render_pass().internal_object(),
                subpass: subpass.index(),
                base_pipeline_handle: ash::vk::Pipeline::null(), // TODO:
                base_pipeline_index: -1,                         // TODO:
                ..Default::default()
            };

            if let Some(discard_rectangle_state) = discard_rectangle_state.as_mut() {
                discard_rectangle_state.p_next = create_info.p_next;
                create_info.p_next = discard_rectangle_state as *const _ as *const _;
            }

            let cache_handle = match self.cache.as_ref() {
                Some(cache) => cache.internal_object(),
                None => ash::vk::PipelineCache::null(),
            };

            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_graphics_pipelines(
                device.internal_object(),
                cache_handle,
                1,
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        // Some drivers return `VK_SUCCESS` but provide a null handle if they
        // fail to create the pipeline (due to invalid shaders, etc)
        // This check ensures that we don't create an invalid `GraphicsPipeline` instance
        if handle == ash::vk::Pipeline::null() {
            panic!("vkCreateGraphicsPipelines provided a NULL handle");
        }

        Ok(Arc::new(GraphicsPipeline {
            handle,
            device: device.clone(),
            layout: pipeline_layout,
            subpass,
            shaders: stages_info
                .iter()
                .map(|stage_info| (ShaderStage::from(*stage_info.entry_point.execution()), ()))
                .collect(),
            descriptor_requirements,
            num_used_descriptor_sets,

            vertex_input_state: self_vertex_input_state, // Can be None if there's a mesh shader, but we don't support that yet
            input_assembly_state: self.input_assembly_state, // Can be None if there's a mesh shader, but we don't support that yet
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
            discard_rectangle_state: if discard_rectangle_state.is_some() {
                Some(self.discard_rectangle_state)
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
        }))
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
        shader: EntryPoint<'vs2>,
        specialization_constants: Vss2,
    ) -> GraphicsPipelineBuilder<'vs2, 'tcs, 'tes, 'gs, 'fs, Vdef, Vss2, Tcss, Tess, Gss, Fss>
    where
        Vss2: SpecializationConstants,
    {
        GraphicsPipelineBuilder {
            subpass: self.subpass,
            cache: self.cache,

            vertex_shader: Some((shader, specialization_constants)),
            tessellation_shaders: self.tessellation_shaders,
            geometry_shader: self.geometry_shader,
            fragment_shader: self.fragment_shader,

            vertex_input_state: self.vertex_input_state,
            input_assembly_state: self.input_assembly_state,
            tessellation_state: self.tessellation_state,
            viewport_state: self.viewport_state,
            discard_rectangle_state: self.discard_rectangle_state,
            rasterization_state: self.rasterization_state,
            multisample_state: self.multisample_state,
            depth_stencil_state: self.depth_stencil_state,
            color_blend_state: self.color_blend_state,
        }
    }

    /// Sets the tessellation shaders to use.
    // TODO: correct specialization constants
    #[inline]
    pub fn tessellation_shaders<'tcs2, 'tes2, Tcss2, Tess2>(
        self,
        control_shader: EntryPoint<'tcs2>,
        control_specialization_constants: Tcss2,
        evaluation_shader: EntryPoint<'tes2>,
        evaluation_specialization_constants: Tess2,
    ) -> GraphicsPipelineBuilder<'vs, 'tcs2, 'tes2, 'gs, 'fs, Vdef, Vss, Tcss2, Tess2, Gss, Fss>
    where
        Tcss2: SpecializationConstants,
        Tess2: SpecializationConstants,
    {
        GraphicsPipelineBuilder {
            subpass: self.subpass,
            cache: self.cache,

            vertex_shader: self.vertex_shader,
            tessellation_shaders: Some(TessellationShaders {
                control: (control_shader, control_specialization_constants),
                evaluation: (evaluation_shader, evaluation_specialization_constants),
            }),
            geometry_shader: self.geometry_shader,
            fragment_shader: self.fragment_shader,

            vertex_input_state: self.vertex_input_state,
            input_assembly_state: self.input_assembly_state,
            tessellation_state: self.tessellation_state,
            viewport_state: self.viewport_state,
            discard_rectangle_state: self.discard_rectangle_state,
            rasterization_state: self.rasterization_state,
            multisample_state: self.multisample_state,
            depth_stencil_state: self.depth_stencil_state,
            color_blend_state: self.color_blend_state,
        }
    }

    /// Sets the geometry shader to use.
    // TODO: correct specialization constants
    #[inline]
    pub fn geometry_shader<'gs2, Gss2>(
        self,
        shader: EntryPoint<'gs2>,
        specialization_constants: Gss2,
    ) -> GraphicsPipelineBuilder<'vs, 'tcs, 'tes, 'gs2, 'fs, Vdef, Vss, Tcss, Tess, Gss2, Fss>
    where
        Gss2: SpecializationConstants,
    {
        GraphicsPipelineBuilder {
            subpass: self.subpass,
            cache: self.cache,

            vertex_shader: self.vertex_shader,
            tessellation_shaders: self.tessellation_shaders,
            geometry_shader: Some((shader, specialization_constants)),
            fragment_shader: self.fragment_shader,

            vertex_input_state: self.vertex_input_state,
            input_assembly_state: self.input_assembly_state,
            tessellation_state: self.tessellation_state,
            viewport_state: self.viewport_state,
            discard_rectangle_state: self.discard_rectangle_state,
            rasterization_state: self.rasterization_state,
            multisample_state: self.multisample_state,
            depth_stencil_state: self.depth_stencil_state,
            color_blend_state: self.color_blend_state,
        }
    }

    /// Sets the fragment shader to use.
    ///
    /// The fragment shader is run once for each pixel that is covered by each primitive.
    // TODO: correct specialization constants
    #[inline]
    pub fn fragment_shader<'fs2, Fss2>(
        self,
        shader: EntryPoint<'fs2>,
        specialization_constants: Fss2,
    ) -> GraphicsPipelineBuilder<'vs, 'tcs, 'tes, 'gs, 'fs2, Vdef, Vss, Tcss, Tess, Gss, Fss2>
    where
        Fss2: SpecializationConstants,
    {
        GraphicsPipelineBuilder {
            subpass: self.subpass,
            cache: self.cache,

            vertex_shader: self.vertex_shader,
            tessellation_shaders: self.tessellation_shaders,
            geometry_shader: self.geometry_shader,
            fragment_shader: Some((shader, specialization_constants)),

            vertex_input_state: self.vertex_input_state,
            input_assembly_state: self.input_assembly_state,
            tessellation_state: self.tessellation_state,
            viewport_state: self.viewport_state,
            discard_rectangle_state: self.discard_rectangle_state,
            rasterization_state: self.rasterization_state,
            multisample_state: self.multisample_state,
            depth_stencil_state: self.depth_stencil_state,
            color_blend_state: self.color_blend_state,
        }
    }

    /// Sets the vertex input state.
    ///
    /// The default value is [`VertexInputState::default()`].
    #[inline]
    pub fn vertex_input_state<T>(
        self,
        vertex_input_state: T,
    ) -> GraphicsPipelineBuilder<'vs, 'tcs, 'tes, 'gs, 'fs, T, Vss, Tcss, Tess, Gss, Fss>
    where
        T: VertexDefinition,
    {
        GraphicsPipelineBuilder {
            subpass: self.subpass,
            cache: self.cache,

            vertex_shader: self.vertex_shader,
            tessellation_shaders: self.tessellation_shaders,
            geometry_shader: self.geometry_shader,
            fragment_shader: self.fragment_shader,

            vertex_input_state,
            input_assembly_state: self.input_assembly_state,
            tessellation_state: self.tessellation_state,
            viewport_state: self.viewport_state,
            discard_rectangle_state: self.discard_rectangle_state,
            rasterization_state: self.rasterization_state,
            multisample_state: self.multisample_state,
            depth_stencil_state: self.depth_stencil_state,
            color_blend_state: self.color_blend_state,
        }
    }

    /// Sets the input assembly state.
    ///
    /// The default value is [`InputAssemblyState::default()`].
    #[inline]
    pub fn input_assembly_state(mut self, input_assembly_state: InputAssemblyState) -> Self {
        self.input_assembly_state = input_assembly_state;
        self
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

    /// Sets the discard rectangle state.
    ///
    /// The default value is [`DiscardRectangleState::default()`].
    #[inline]
    pub fn discard_rectangle_state(
        mut self,
        discard_rectangle_state: DiscardRectangleState,
    ) -> Self {
        self.discard_rectangle_state = discard_rectangle_state;
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

    /// Sets the multisample state.
    ///
    /// The default value is [`MultisampleState::default()`].
    #[inline]
    pub fn multisample_state(mut self, multisample_state: MultisampleState) -> Self {
        self.multisample_state = multisample_state;
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

    /// Sets the color blend state.
    ///
    /// The default value is [`ColorBlendState::default()`].
    #[inline]
    pub fn color_blend_state(mut self, color_blend_state: ColorBlendState) -> Self {
        self.color_blend_state = color_blend_state;
        self
    }

    /// Sets the tessellation shaders stage as disabled. This is the default.
    #[deprecated(since = "0.27")]
    #[inline]
    pub fn tessellation_shaders_disabled(mut self) -> Self {
        self.tessellation_shaders = None;
        self
    }

    /// Sets the geometry shader stage as disabled. This is the default.
    #[deprecated(since = "0.27")]
    #[inline]
    pub fn geometry_shader_disabled(mut self) -> Self {
        self.geometry_shader = None;
        self
    }

    /// Sets the vertex input to a single vertex buffer.
    ///
    /// You will most likely need to explicitly specify the template parameter to the type of a
    /// vertex.
    #[deprecated(since = "0.27", note = "Use `vertex_input_state` instead")]
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
        self.vertex_input_state(BuffersDefinition::new().vertex::<V>())
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
        self.input_assembly_state.topology = PartialStateMode::Fixed(topology);
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

    #[deprecated(since = "0.27", note = "Use `color_blend_state` instead")]
    // TODO: When this method is removed, also remove the special casing in `with_pipeline_layout`
    // for self.color_blend_state.attachments.len() == 1
    #[inline]
    pub fn blend_collective(mut self, blend: AttachmentBlend) -> Self {
        self.color_blend_state.attachments = vec![ColorBlendAttachmentState {
            blend: Some(blend),
            color_write_mask: ColorComponents::all(),
            color_write_enable: StateMode::Fixed(true),
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
                color_write_enable: StateMode::Fixed(true),
            })
            .collect();
        self
    }

    /// Each fragment shader output will have its value directly written to the framebuffer
    /// attachment. This is the default.
    #[deprecated(since = "0.27", note = "Use `color_blend_state` instead")]
    // TODO: When this method is removed, also remove the special casing in `with_pipeline_layout`
    // for self.color_blend_state.attachments.len() == 1
    #[inline]
    pub fn blend_pass_through(mut self) -> Self {
        self.color_blend_state.attachments = vec![ColorBlendAttachmentState {
            blend: None,
            color_write_mask: ColorComponents::all(),
            color_write_enable: StateMode::Fixed(true),
        }];
        self
    }

    #[deprecated(since = "0.27", note = "Use `color_blend_state` instead")]
    // TODO: When this method is removed, also remove the special casing in `with_pipeline_layout`
    // for self.color_blend_state.attachments.len() == 1
    #[inline]
    pub fn blend_alpha_blending(mut self) -> Self {
        self.color_blend_state.attachments = vec![ColorBlendAttachmentState {
            blend: Some(AttachmentBlend::alpha()),
            color_write_mask: ColorComponents::all(),
            color_write_enable: StateMode::Fixed(true),
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
            subpass: Some(subpass),
            cache: self.cache,

            vertex_shader: self.vertex_shader,
            tessellation_shaders: self.tessellation_shaders,
            geometry_shader: self.geometry_shader,
            fragment_shader: self.fragment_shader,

            vertex_input_state: self.vertex_input_state,
            input_assembly_state: self.input_assembly_state,
            tessellation_state: self.tessellation_state,
            viewport_state: self.viewport_state,
            rasterization_state: self.rasterization_state,
            multisample_state: self.multisample_state,
            depth_stencil_state: self.depth_stencil_state,
            color_blend_state: self.color_blend_state,

            discard_rectangle_state: self.discard_rectangle_state,
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
            subpass: self.subpass.clone(),
            cache: self.cache.clone(),

            vertex_shader: self.vertex_shader.clone(),
            tessellation_shaders: self.tessellation_shaders.clone(),
            geometry_shader: self.geometry_shader.clone(),
            fragment_shader: self.fragment_shader.clone(),

            vertex_input_state: self.vertex_input_state.clone(),
            input_assembly_state: self.input_assembly_state,
            tessellation_state: self.tessellation_state,
            viewport_state: self.viewport_state.clone(),
            rasterization_state: self.rasterization_state.clone(),
            multisample_state: self.multisample_state,
            depth_stencil_state: self.depth_stencil_state.clone(),
            color_blend_state: self.color_blend_state.clone(),

            discard_rectangle_state: self.discard_rectangle_state.clone(),
        }
    }
}
