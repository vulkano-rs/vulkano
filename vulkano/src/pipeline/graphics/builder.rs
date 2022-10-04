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

use super::{
    color_blend::{
        AttachmentBlend, ColorBlendAttachmentState, ColorBlendState, ColorComponents, LogicOp,
    },
    depth_stencil::{DepthStencilState, StencilOps},
    discard_rectangle::DiscardRectangleState,
    input_assembly::{InputAssemblyState, PrimitiveTopology, PrimitiveTopologyClass},
    multisample::MultisampleState,
    rasterization::{
        CullMode, DepthBiasState, FrontFace, LineRasterizationMode, PolygonMode, RasterizationState,
    },
    render_pass::{PipelineRenderPassType, PipelineRenderingCreateInfo},
    tessellation::TessellationState,
    vertex_input::{
        BuffersDefinition, Vertex, VertexDefinition, VertexInputAttributeDescription,
        VertexInputBindingDescription, VertexInputState,
    },
    viewport::{Scissor, Viewport, ViewportState},
    GraphicsPipeline, GraphicsPipelineCreationError,
};
use crate::{
    descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutCreateInfo},
    device::{Device, DeviceOwned},
    format::NumericType,
    pipeline::{
        cache::PipelineCache,
        graphics::{
            color_blend::BlendFactor,
            depth_stencil::{DepthBoundsState, DepthState, StencilOpState, StencilState},
            vertex_input::VertexInputRate,
        },
        layout::{PipelineLayoutCreateInfo, PushConstantRange},
        DynamicState, PartialStateMode, PipelineLayout, StateMode,
    },
    shader::{
        DescriptorRequirements, EntryPoint, ShaderExecution, ShaderStage, SpecializationConstants,
        SpecializationMapEntry,
    },
    DeviceSize, RequiresOneOf, Version, VulkanError, VulkanObject,
};
use smallvec::SmallVec;
use std::{
    collections::{hash_map::Entry, HashMap},
    mem::{size_of_val, MaybeUninit},
    ptr, slice,
    sync::Arc,
};

/// Prototype for a `GraphicsPipeline`.
#[derive(Debug)]
pub struct GraphicsPipelineBuilder<'vs, 'tcs, 'tes, 'gs, 'fs, Vdef, Vss, Tcss, Tess, Gss, Fss> {
    render_pass: Option<PipelineRenderPassType>,
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
            render_pass: None,
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

#[derive(Clone, Copy, Debug)]
struct Has {
    vertex_input_state: bool,
    pre_rasterization_shader_state: bool,
    tessellation_state: bool,
    viewport_state: bool,
    fragment_shader_state: bool,
    depth_stencil_state: bool,
    fragment_output_state: bool,
    color_blend_state: bool,
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
        F: FnOnce(&mut [DescriptorSetLayoutCreateInfo]),
    {
        let (set_layout_create_infos, push_constant_ranges) = {
            let stages: SmallVec<[&EntryPoint<'_>; 5]> = [
                self.vertex_shader.as_ref().map(|s| &s.0),
                self.tessellation_shaders.as_ref().map(|s| &s.control.0),
                self.tessellation_shaders.as_ref().map(|s| &s.evaluation.0),
                self.geometry_shader.as_ref().map(|s| &s.0),
                self.fragment_shader.as_ref().map(|s| &s.0),
            ]
            .into_iter()
            .flatten()
            .collect();

            // Produce `DescriptorRequirements` for each binding, by iterating over all shaders
            // and adding the requirements of each.
            let mut descriptor_requirements: HashMap<(u32, u32), DescriptorRequirements> =
                HashMap::default();

            for (loc, reqs) in stages
                .iter()
                .flat_map(|shader| shader.descriptor_requirements())
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
            let mut set_layout_create_infos = DescriptorSetLayoutCreateInfo::from_requirements(
                descriptor_requirements
                    .iter()
                    .map(|(&loc, reqs)| (loc, reqs)),
            );
            func(&mut set_layout_create_infos);

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
                .map(|((offset, size), stages)| PushConstantRange {
                    stages: *stages,
                    offset: *offset,
                    size: *size,
                })
                .collect();

            (set_layout_create_infos, push_constant_ranges)
        };

        let set_layouts = set_layout_create_infos
            .into_iter()
            .map(|desc| DescriptorSetLayout::new(device.clone(), desc))
            .collect::<Result<Vec<_>, _>>()?;
        let pipeline_layout = PipelineLayout::new(
            device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts,
                push_constant_ranges,
                ..Default::default()
            },
        )
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
        let vertex_input_state = self
            .vertex_input_state
            .definition(self.vertex_shader.as_ref().unwrap().0.input_interface())?;

        // If there is one element, duplicate it for all attachments.
        // TODO: this is undocumented and only exists for compatibility with some of the
        // deprecated builder methods. Remove it when those methods are gone.
        if self.color_blend_state.attachments.len() == 1 {
            let color_attachment_count =
                match self.render_pass.as_ref().expect("Missing render pass") {
                    PipelineRenderPassType::BeginRenderPass(subpass) => {
                        subpass.subpass_desc().color_attachments.len()
                    }
                    PipelineRenderPassType::BeginRendering(rendering_info) => {
                        rendering_info.color_attachment_formats.len()
                    }
                };
            let element = self.color_blend_state.attachments.pop().unwrap();
            self.color_blend_state
                .attachments
                .extend(std::iter::repeat(element).take(color_attachment_count));
        }

        let has = {
            let &Self {
                ref render_pass,
                cache: _,

                ref vertex_shader,
                ref tessellation_shaders,
                geometry_shader: _,
                fragment_shader: _,

                vertex_input_state: _,
                input_assembly_state: _,
                tessellation_state: _,
                viewport_state: _,
                discard_rectangle_state: _,
                ref rasterization_state,
                multisample_state: _,
                depth_stencil_state: _,
                color_blend_state: _,
            } = &self;

            let render_pass = render_pass.as_ref().expect("Missing render pass");

            let has_pre_rasterization_shader_state = true;
            let has_vertex_input_state = vertex_shader.is_some();
            let has_fragment_shader_state =
                rasterization_state.rasterizer_discard_enable != StateMode::Fixed(true);
            let has_fragment_output_state =
                rasterization_state.rasterizer_discard_enable != StateMode::Fixed(true);

            let has_tessellation_state =
                has_pre_rasterization_shader_state && tessellation_shaders.is_some();
            let has_viewport_state =
                has_pre_rasterization_shader_state && has_fragment_shader_state;
            let has_depth_stencil_state = has_fragment_shader_state
                && match render_pass {
                    PipelineRenderPassType::BeginRenderPass(subpass) => {
                        subpass.subpass_desc().depth_stencil_attachment.is_some()
                    }
                    PipelineRenderPassType::BeginRendering(rendering_info) => {
                        !has_fragment_output_state
                            || rendering_info.depth_attachment_format.is_some()
                            || rendering_info.stencil_attachment_format.is_some()
                    }
                };
            let has_color_blend_state = has_fragment_output_state
                && match render_pass {
                    PipelineRenderPassType::BeginRenderPass(subpass) => {
                        !subpass.subpass_desc().color_attachments.is_empty()
                    }
                    PipelineRenderPassType::BeginRendering(rendering_info) => {
                        !rendering_info.color_attachment_formats.is_empty()
                    }
                };

            Has {
                vertex_input_state: has_vertex_input_state,
                pre_rasterization_shader_state: has_pre_rasterization_shader_state,
                tessellation_state: has_tessellation_state,
                viewport_state: has_viewport_state,
                fragment_shader_state: has_fragment_shader_state,
                depth_stencil_state: has_depth_stencil_state,
                fragment_output_state: has_fragment_output_state,
                color_blend_state: has_color_blend_state,
            }
        };

        self.validate_create(&device, &pipeline_layout, &vertex_input_state, has)?;

        let (handle, descriptor_requirements, dynamic_state, shaders) =
            unsafe { self.record_create(&device, &pipeline_layout, &vertex_input_state, has)? };

        let Self {
            mut render_pass,
            cache: _,
            vertex_shader: _,
            tessellation_shaders: _,
            geometry_shader: _,
            fragment_shader: _,
            vertex_input_state: _,
            input_assembly_state,
            tessellation_state,
            viewport_state,
            discard_rectangle_state,
            rasterization_state,
            multisample_state,
            depth_stencil_state,
            color_blend_state,
        } = self;

        let num_used_descriptor_sets = descriptor_requirements
            .keys()
            .map(|loc| loc.0)
            .max()
            .map(|x| x + 1)
            .unwrap_or(0);

        Ok(Arc::new(GraphicsPipeline {
            handle,
            device,
            layout: pipeline_layout,
            render_pass: render_pass.take().expect("Missing render pass"),
            shaders,
            descriptor_requirements,
            num_used_descriptor_sets,

            vertex_input_state, // Can be None if there's a mesh shader, but we don't support that yet
            input_assembly_state, // Can be None if there's a mesh shader, but we don't support that yet
            tessellation_state: has.tessellation_state.then_some(tessellation_state),
            viewport_state: has.viewport_state.then_some(viewport_state),
            discard_rectangle_state: has
                .pre_rasterization_shader_state
                .then_some(discard_rectangle_state),
            rasterization_state,
            multisample_state: has.fragment_output_state.then_some(multisample_state),
            depth_stencil_state: has.depth_stencil_state.then_some(depth_stencil_state),
            color_blend_state: has.color_blend_state.then_some(color_blend_state),
            dynamic_state,
        }))
    }

    fn validate_create(
        &self,
        device: &Device,
        pipeline_layout: &PipelineLayout,
        vertex_input_state: &VertexInputState,
        has: Has,
    ) -> Result<(), GraphicsPipelineCreationError> {
        let physical_device = device.physical_device();
        let properties = physical_device.properties();

        let &Self {
            ref render_pass,
            cache: _,

            ref vertex_shader,
            ref tessellation_shaders,
            ref geometry_shader,
            ref fragment_shader,

            vertex_input_state: _,
            ref input_assembly_state,
            ref tessellation_state,
            ref viewport_state,
            ref discard_rectangle_state,
            ref rasterization_state,
            ref multisample_state,
            ref depth_stencil_state,
            ref color_blend_state,
        } = self;

        let render_pass = render_pass.as_ref().expect("Missing render pass");

        let mut shader_stages: SmallVec<[_; 5]> = SmallVec::new();

        // VUID-VkGraphicsPipelineCreateInfo-layout-01688
        // Checked at pipeline layout creation time.

        /*
            Render pass
        */

        match render_pass {
            PipelineRenderPassType::BeginRenderPass(subpass) => {
                // VUID-VkGraphicsPipelineCreateInfo-commonparent
                assert_eq!(device, subpass.render_pass().device().as_ref());
            }
            PipelineRenderPassType::BeginRendering(rendering_info) => {
                let &PipelineRenderingCreateInfo {
                    view_mask,
                    ref color_attachment_formats,
                    depth_attachment_format,
                    stencil_attachment_format,
                    _ne: _,
                } = rendering_info;

                // VUID-VkGraphicsPipelineCreateInfo-dynamicRendering-06576
                if !device.enabled_features().dynamic_rendering {
                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                        required_for: "`render_pass` is `PipelineRenderPassType::BeginRendering`",
                        requires_one_of: RequiresOneOf {
                            features: &["dynamic_rendering"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkGraphicsPipelineCreateInfo-multiview-06577
                if view_mask != 0 && !device.enabled_features().multiview {
                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                        required_for: "`render_pass` is `PipelineRenderPassType::BeginRendering` where `view_mask` is not `0`",
                        requires_one_of: RequiresOneOf {
                            features: &["multiview"],
                            ..Default::default()
                        },
                    });
                }

                let view_count = u32::BITS - view_mask.leading_zeros();

                // VUID-VkGraphicsPipelineCreateInfo-renderPass-06578
                if view_count > properties.max_multiview_view_count.unwrap_or(0) {
                    return Err(
                        GraphicsPipelineCreationError::MaxMultiviewViewCountExceeded {
                            view_count,
                            max: properties.max_multiview_view_count.unwrap_or(0),
                        },
                    );
                }

                if has.fragment_output_state {
                    for (attachment_index, format) in color_attachment_formats
                        .iter()
                        .enumerate()
                        .flat_map(|(i, f)| f.map(|f| (i, f)))
                    {
                        let attachment_index = attachment_index as u32;

                        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06580
                        format.validate_device(device)?;

                        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06582
                        // Use unchecked, because all validation has been done above.
                        if !unsafe { physical_device.format_properties_unchecked(format) }
                            .potential_format_features()
                            .color_attachment
                        {
                            return Err(
                                GraphicsPipelineCreationError::ColorAttachmentFormatUsageNotSupported {
                                    attachment_index,
                                },
                            );
                        }
                    }

                    if let Some(format) = depth_attachment_format {
                        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06583
                        format.validate_device(device)?;

                        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06585
                        // Use unchecked, because all validation has been done above.
                        if !unsafe { physical_device.format_properties_unchecked(format) }
                            .potential_format_features()
                            .depth_stencil_attachment
                        {
                            return Err(
                                GraphicsPipelineCreationError::DepthAttachmentFormatUsageNotSupported,
                            );
                        }

                        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06587
                        if !format.aspects().depth {
                            return Err(
                                GraphicsPipelineCreationError::DepthAttachmentFormatUsageNotSupported,
                            );
                        }
                    }

                    if let Some(format) = stencil_attachment_format {
                        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06584
                        format.validate_device(device)?;

                        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06586
                        // Use unchecked, because all validation has been done above.
                        if !unsafe { physical_device.format_properties_unchecked(format) }
                            .potential_format_features()
                            .depth_stencil_attachment
                        {
                            return Err(
                                GraphicsPipelineCreationError::StencilAttachmentFormatUsageNotSupported,
                            );
                        }

                        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06588
                        if !format.aspects().stencil {
                            return Err(
                                GraphicsPipelineCreationError::StencilAttachmentFormatUsageNotSupported,
                            );
                        }
                    }

                    if let (Some(depth_format), Some(stencil_format)) =
                        (depth_attachment_format, stencil_attachment_format)
                    {
                        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06589
                        if depth_format != stencil_format {
                            return Err(
                                GraphicsPipelineCreationError::DepthStencilAttachmentFormatMismatch,
                            );
                        }
                    }
                }
            }
        }

        /*
            Vertex input state
        */

        if has.vertex_input_state {
            // Vertex input state
            // VUID-VkGraphicsPipelineCreateInfo-pVertexInputState-04910
            {
                let &VertexInputState {
                    ref bindings,
                    ref attributes,
                } = vertex_input_state;

                // VUID-VkPipelineVertexInputStateCreateInfo-vertexBindingDescriptionCount-00613
                if bindings.len() > properties.max_vertex_input_bindings as usize {
                    return Err(
                        GraphicsPipelineCreationError::MaxVertexInputBindingsExceeded {
                            max: properties.max_vertex_input_bindings,
                            obtained: bindings.len() as u32,
                        },
                    );
                }

                // VUID-VkPipelineVertexInputStateCreateInfo-pVertexBindingDescriptions-00616
                // Ensured by HashMap.

                for (&binding, binding_desc) in bindings {
                    let &VertexInputBindingDescription { stride, input_rate } = binding_desc;

                    // VUID-VkVertexInputBindingDescription-binding-00618
                    if binding >= properties.max_vertex_input_bindings {
                        return Err(
                            GraphicsPipelineCreationError::MaxVertexInputBindingsExceeded {
                                max: properties.max_vertex_input_bindings,
                                obtained: binding,
                            },
                        );
                    }

                    // VUID-VkVertexInputBindingDescription-stride-00619
                    if stride > properties.max_vertex_input_binding_stride {
                        return Err(
                            GraphicsPipelineCreationError::MaxVertexInputBindingStrideExceeded {
                                binding,
                                max: properties.max_vertex_input_binding_stride,
                                obtained: stride,
                            },
                        );
                    }

                    match input_rate {
                        VertexInputRate::Instance { divisor } if divisor != 1 => {
                            // VUID-VkVertexInputBindingDivisorDescriptionEXT-vertexAttributeInstanceRateDivisor-02229
                            if !device
                                .enabled_features()
                                .vertex_attribute_instance_rate_divisor
                            {
                                return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                    required_for: "`vertex_input_state.bindings` has an element where `input_rate` is `VertexInputRate::Instance`, where `divisor` is not `1`",
                                    requires_one_of: RequiresOneOf {
                                        features: &["vertex_attribute_instance_rate_divisor"],
                                        ..Default::default()
                                    },
                                });
                            }

                            // VUID-VkVertexInputBindingDivisorDescriptionEXT-vertexAttributeInstanceRateZeroDivisor-02228
                            if divisor == 0
                                && !device
                                    .enabled_features()
                                    .vertex_attribute_instance_rate_zero_divisor
                            {
                                return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                    required_for: "`vertex_input_state.bindings` has an element where `input_rate` is `VertexInputRate::Instance`, where `divisor` is `0`",
                                    requires_one_of: RequiresOneOf {
                                        features: &["vertex_attribute_instance_rate_zero_divisor"],
                                        ..Default::default()
                                    },
                                });
                            }

                            // VUID-VkVertexInputBindingDivisorDescriptionEXT-divisor-01870
                            if divisor > properties.max_vertex_attrib_divisor.unwrap() {
                                return Err(
                                    GraphicsPipelineCreationError::MaxVertexAttribDivisorExceeded {
                                        binding,
                                        max: properties.max_vertex_attrib_divisor.unwrap(),
                                        obtained: divisor,
                                    },
                                );
                            }
                        }
                        _ => (),
                    }
                }

                // VUID-VkPipelineVertexInputStateCreateInfo-vertexAttributeDescriptionCount-00614
                if attributes.len() > properties.max_vertex_input_attributes as usize {
                    return Err(
                        GraphicsPipelineCreationError::MaxVertexInputAttributesExceeded {
                            max: properties.max_vertex_input_attributes,
                            obtained: attributes.len(),
                        },
                    );
                }

                // VUID-VkPipelineVertexInputStateCreateInfo-pVertexAttributeDescriptions-00617
                // Ensured by HashMap.

                for (&location, attribute_desc) in attributes {
                    let &VertexInputAttributeDescription {
                        binding,
                        format,
                        offset,
                    } = attribute_desc;

                    // VUID-VkVertexInputAttributeDescription-format-parameter
                    format.validate_device(device)?;

                    // TODO:
                    // VUID-VkVertexInputAttributeDescription-location-00620

                    // VUID-VkPipelineVertexInputStateCreateInfo-binding-00615
                    if !bindings.contains_key(&binding) {
                        return Err(
                            GraphicsPipelineCreationError::VertexInputAttributeInvalidBinding {
                                location,
                                binding,
                            },
                        );
                    }

                    // VUID-VkVertexInputAttributeDescription-offset-00622
                    if offset > properties.max_vertex_input_attribute_offset {
                        return Err(
                            GraphicsPipelineCreationError::MaxVertexInputAttributeOffsetExceeded {
                                max: properties.max_vertex_input_attribute_offset,
                                obtained: offset,
                            },
                        );
                    }

                    // Use unchecked, because all validation has been done above.
                    let format_features = unsafe {
                        device
                            .physical_device()
                            .format_properties_unchecked(format)
                            .buffer_features
                    };

                    // VUID-VkVertexInputAttributeDescription-format-00623
                    if !format_features.vertex_buffer {
                        return Err(
                            GraphicsPipelineCreationError::VertexInputAttributeUnsupportedFormat {
                                location,
                                format,
                            },
                        );
                    }
                }
            }

            // Input assembly state
            // VUID-VkGraphicsPipelineCreateInfo-pStages-02098
            {
                let &InputAssemblyState {
                    topology,
                    primitive_restart_enable,
                } = input_assembly_state;

                match topology {
                    PartialStateMode::Fixed(topology) => {
                        // VUID-VkPipelineInputAssemblyStateCreateInfo-topology-parameter
                        topology.validate_device(device)?;

                        match topology {
                            PrimitiveTopology::LineListWithAdjacency
                            | PrimitiveTopology::LineStripWithAdjacency
                            | PrimitiveTopology::TriangleListWithAdjacency
                            | PrimitiveTopology::TriangleStripWithAdjacency => {
                                // VUID-VkPipelineInputAssemblyStateCreateInfo-topology-00429
                                if !device.enabled_features().geometry_shader {
                                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                        required_for: "`input_assembly_state.topology` is `StateMode::Fixed(PrimitiveTopology::*WithAdjacency)`",
                                        requires_one_of: RequiresOneOf {
                                            features: &["geometry_shader"],
                                            ..Default::default()
                                        },
                                    });
                                }
                            }
                            PrimitiveTopology::PatchList => {
                                // VUID-VkPipelineInputAssemblyStateCreateInfo-topology-00430
                                if !device.enabled_features().tessellation_shader {
                                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                        required_for: "`input_assembly_state.topology` is `StateMode::Fixed(PrimitiveTopology::PatchList)`",
                                        requires_one_of: RequiresOneOf {
                                            features: &["tessellation_shader"],
                                            ..Default::default()
                                        },
                                    });
                                }

                                // TODO:
                                // VUID-VkGraphicsPipelineCreateInfo-topology-00737
                            }
                            _ => (),
                        }
                    }
                    PartialStateMode::Dynamic(topology_class) => {
                        // VUID-VkPipelineInputAssemblyStateCreateInfo-topology-parameter
                        topology_class.example().validate_device(device)?;

                        // VUID?
                        if !(device.api_version() >= Version::V1_3
                            || device.enabled_features().extended_dynamic_state)
                        {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for:
                                    "`input_assembly_state.topology` is `PartialStateMode::Dynamic`",
                                requires_one_of: RequiresOneOf {
                                    api_version: Some(Version::V1_3),
                                    features: &["extended_dynamic_state"],
                                    ..Default::default()
                                },
                            });
                        }
                    }
                }

                match primitive_restart_enable {
                    StateMode::Fixed(primitive_restart_enable) => {
                        if primitive_restart_enable {
                            match topology {
                                PartialStateMode::Fixed(
                                    PrimitiveTopology::PointList
                                    | PrimitiveTopology::LineList
                                    | PrimitiveTopology::TriangleList
                                    | PrimitiveTopology::LineListWithAdjacency
                                    | PrimitiveTopology::TriangleListWithAdjacency,
                                ) => {
                                    // VUID-VkPipelineInputAssemblyStateCreateInfo-topology-06252
                                    if !device.enabled_features().primitive_topology_list_restart {
                                        return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                            required_for: "`input_assembly_state.primitive_restart_enable` is `StateMode::Fixed(true)` and `input_assembly_state.topology` is `StateMode::Fixed(PrimitiveTopology::*List)`",
                                            requires_one_of: RequiresOneOf {
                                                features: &["primitive_topology_list_restart"],
                                                ..Default::default()
                                            },
                                        });
                                    }
                                }
                                PartialStateMode::Fixed(PrimitiveTopology::PatchList) => {
                                    // VUID-VkPipelineInputAssemblyStateCreateInfo-topology-06253
                                    if !device
                                        .enabled_features()
                                        .primitive_topology_patch_list_restart
                                    {
                                        return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                            required_for: "`input_assembly_state.primitive_restart_enable` is `StateMode::Fixed(true)` and `input_assembly_state.topology` is `StateMode::Fixed(PrimitiveTopology::PatchList)`",
                                            requires_one_of: RequiresOneOf {
                                                features: &["primitive_topology_patch_list_restart"],
                                                ..Default::default()
                                            },
                                        });
                                    }
                                }
                                _ => (),
                            }
                        }
                    }
                    StateMode::Dynamic => {
                        // VUID?
                        if !(device.api_version() >= Version::V1_3
                            || device.enabled_features().extended_dynamic_state2)
                        {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for: "`input_assembly_state.primitive_restart_enable` is `StateMode::Dynamic`",
                                requires_one_of: RequiresOneOf {
                                    api_version: Some(Version::V1_3),
                                    features: &["extended_dynamic_state2"],
                                    ..Default::default()
                                },
                            });
                        }
                    }
                };
            }
        }

        /*
            Pre-rasterization shader state
        */

        if has.pre_rasterization_shader_state {
            // Vertex shader
            if let Some((entry_point, specialization_data)) = vertex_shader {
                shader_stages.push(ShaderStageInfo {
                    entry_point,
                    specialization_map_entries: Vss::descriptors(),
                    _specialization_data: unsafe {
                        std::slice::from_raw_parts(
                            specialization_data as *const _ as *const u8,
                            size_of_val(specialization_data),
                        )
                    },
                });

                match entry_point.execution() {
                    ShaderExecution::Vertex => (),
                    _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
                }

                // VUID?
                // Check that the vertex input state contains attributes for all the shader's input
                // variables.
                for element in entry_point.input_interface().elements() {
                    assert!(!element.ty.is_64bit); // TODO: implement
                    let location_range =
                        element.location..element.location + element.ty.num_locations();

                    for location in location_range {
                        let attribute_desc =
                            match vertex_input_state.attributes.get(&location) {
                                Some(attribute_desc) => attribute_desc,
                                None => return Err(
                                    GraphicsPipelineCreationError::VertexInputAttributeMissing {
                                        location,
                                    },
                                ),
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

                // TODO:
                // VUID-VkPipelineShaderStageCreateInfo-stage-00712
            } else {
                // VUID-VkGraphicsPipelineCreateInfo-stage-02096
                panic!("Missing vertex shader"); // TODO: return error
            }

            // Tessellation shaders & tessellation state
            if let Some(tessellation_shaders) = tessellation_shaders {
                // VUID-VkGraphicsPipelineCreateInfo-pStages-00729
                // VUID-VkGraphicsPipelineCreateInfo-pStages-00730
                // Ensured by the definition of TessellationShaders.

                // FIXME: must check that the control shader and evaluation shader are compatible

                // VUID-VkPipelineShaderStageCreateInfo-stage-00705
                if !device.enabled_features().tessellation_shader {
                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                        required_for: "`tessellation_shaders` are provided",
                        requires_one_of: RequiresOneOf {
                            features: &["tessellation_shader"],
                            ..Default::default()
                        },
                    });
                }

                {
                    let (entry_point, specialization_data) = &tessellation_shaders.control;

                    shader_stages.push(ShaderStageInfo {
                        entry_point,
                        specialization_map_entries: Tcss::descriptors(),
                        _specialization_data: unsafe {
                            std::slice::from_raw_parts(
                                specialization_data as *const _ as *const u8,
                                size_of_val(specialization_data),
                            )
                        },
                    });

                    match entry_point.execution() {
                        ShaderExecution::TessellationControl => (),
                        _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
                    }
                }

                {
                    let (entry_point, specialization_data) = &tessellation_shaders.evaluation;

                    shader_stages.push(ShaderStageInfo {
                        entry_point,
                        specialization_map_entries: Tess::descriptors(),
                        _specialization_data: unsafe {
                            std::slice::from_raw_parts(
                                specialization_data as *const _ as *const u8,
                                size_of_val(specialization_data),
                            )
                        },
                    });

                    match entry_point.execution() {
                        ShaderExecution::TessellationEvaluation => (),
                        _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
                    }
                }

                if !device.enabled_features().multiview_tessellation_shader {
                    let view_mask = match render_pass {
                        PipelineRenderPassType::BeginRenderPass(subpass) => {
                            subpass.render_pass().views_used()
                        }
                        PipelineRenderPassType::BeginRendering(rendering_info) => {
                            rendering_info.view_mask
                        }
                    };

                    // VUID-VkGraphicsPipelineCreateInfo-renderPass-06047
                    // VUID-VkGraphicsPipelineCreateInfo-renderPass-06057
                    if view_mask != 0 {
                        return Err(GraphicsPipelineCreationError::RequirementNotMet {
                            required_for: "`tessellation_shaders` are provided and `render_pass` has a subpass where `view_mask` is not `0`",
                            requires_one_of: RequiresOneOf {
                                features: &["multiview_tessellation_shader"],
                                ..Default::default()
                            },
                        });
                    }
                }

                // TODO:
                // VUID-VkPipelineShaderStageCreateInfo-stage-00713
                // VUID-VkGraphicsPipelineCreateInfo-pStages-00732
                // VUID-VkGraphicsPipelineCreateInfo-pStages-00733
                // VUID-VkGraphicsPipelineCreateInfo-pStages-00734
                // VUID-VkGraphicsPipelineCreateInfo-pStages-00735
            }

            // Geometry shader
            if let Some((entry_point, specialization_data)) = geometry_shader {
                shader_stages.push(ShaderStageInfo {
                    entry_point,
                    specialization_map_entries: Gss::descriptors(),
                    _specialization_data: unsafe {
                        std::slice::from_raw_parts(
                            specialization_data as *const _ as *const u8,
                            size_of_val(specialization_data),
                        )
                    },
                });

                // VUID-VkPipelineShaderStageCreateInfo-stage-00704
                if !device.enabled_features().geometry_shader {
                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                        required_for: "`geometry_shader` is provided",
                        requires_one_of: RequiresOneOf {
                            features: &["geometry_shader"],
                            ..Default::default()
                        },
                    });
                }

                let input = match entry_point.execution() {
                    ShaderExecution::Geometry(execution) => execution.input,
                    _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
                };

                if let PartialStateMode::Fixed(topology) = input_assembly_state.topology {
                    // VUID-VkGraphicsPipelineCreateInfo-pStages-00738
                    if !input.is_compatible_with(topology) {
                        return Err(
                            GraphicsPipelineCreationError::TopologyNotMatchingGeometryShader,
                        );
                    }
                }

                if !device.enabled_features().multiview_geometry_shader {
                    let view_mask = match render_pass {
                        PipelineRenderPassType::BeginRenderPass(subpass) => {
                            subpass.render_pass().views_used()
                        }
                        PipelineRenderPassType::BeginRendering(rendering_info) => {
                            rendering_info.view_mask
                        }
                    };

                    // VUID-VkGraphicsPipelineCreateInfo-renderPass-06048
                    // VUID-VkGraphicsPipelineCreateInfo-renderPass-06058
                    if view_mask != 0 {
                        return Err(GraphicsPipelineCreationError::RequirementNotMet {
                            required_for: "`geometry_shader` is provided and `render_pass` has a subpass where `view_mask` is not `0`",
                            requires_one_of: RequiresOneOf {
                                features: &["multiview_geometry_shader"],
                                ..Default::default()
                            },
                        });
                    }
                }

                // TODO:
                // VUID-VkPipelineShaderStageCreateInfo-stage-00714
                // VUID-VkPipelineShaderStageCreateInfo-stage-00715
                // VUID-VkGraphicsPipelineCreateInfo-pStages-00739
            }

            // Rasterization state
            // VUID?
            {
                let &RasterizationState {
                    depth_clamp_enable,
                    rasterizer_discard_enable,
                    polygon_mode,
                    cull_mode,
                    front_face,
                    depth_bias,
                    line_width,
                    line_rasterization_mode,
                    line_stipple,
                } = rasterization_state;

                // VUID-VkPipelineRasterizationStateCreateInfo-polygonMode-parameter
                polygon_mode.validate_device(device)?;

                // VUID-VkPipelineRasterizationStateCreateInfo-depthClampEnable-00782
                if depth_clamp_enable && !device.enabled_features().depth_clamp {
                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                        required_for: "`rasterization_state.depth_clamp_enable` is set",
                        requires_one_of: RequiresOneOf {
                            features: &["depth_clamp"],
                            ..Default::default()
                        },
                    });
                }

                // VUID?
                if matches!(rasterizer_discard_enable, StateMode::Dynamic)
                    && !(device.api_version() >= Version::V1_3
                        || device.enabled_features().extended_dynamic_state2)
                {
                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                        required_for: "`rasterization_state.rasterizer_discard_enable` is `StateMode::Dynamic`",
                        requires_one_of: RequiresOneOf {
                            api_version: Some(Version::V1_3),
                            features: &["extended_dynamic_state"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkPipelineRasterizationStateCreateInfo-polygonMode-01507
                if polygon_mode != PolygonMode::Fill
                    && !device.enabled_features().fill_mode_non_solid
                {
                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                        required_for:
                            "`rasterization_state.polygon_mode` is not `PolygonMode::Fill`",
                        requires_one_of: RequiresOneOf {
                            features: &["fill_mode_non_solid"],
                            ..Default::default()
                        },
                    });
                }

                match cull_mode {
                    StateMode::Fixed(cull_mode) => {
                        // VUID-VkPipelineRasterizationStateCreateInfo-cullMode-parameter
                        cull_mode.validate_device(device)?;
                    }
                    StateMode::Dynamic => {
                        // VUID?
                        if !(device.api_version() >= Version::V1_3
                            || device.enabled_features().extended_dynamic_state)
                        {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for:
                                    "`rasterization_state.cull_mode` is `StateMode::Dynamic`",
                                requires_one_of: RequiresOneOf {
                                    api_version: Some(Version::V1_3),
                                    features: &["extended_dynamic_state"],
                                    ..Default::default()
                                },
                            });
                        }
                    }
                }

                match front_face {
                    StateMode::Fixed(front_face) => {
                        // VUID-VkPipelineRasterizationStateCreateInfo-frontFace-parameter
                        front_face.validate_device(device)?;
                    }
                    StateMode::Dynamic => {
                        // VUID?
                        if !(device.api_version() >= Version::V1_3
                            || device.enabled_features().extended_dynamic_state)
                        {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for:
                                    "`rasterization_state.front_face` is `StateMode::Dynamic`",
                                requires_one_of: RequiresOneOf {
                                    api_version: Some(Version::V1_3),
                                    features: &["extended_dynamic_state"],
                                    ..Default::default()
                                },
                            });
                        }
                    }
                }

                if let Some(depth_bias_state) = depth_bias {
                    let DepthBiasState {
                        enable_dynamic,
                        bias,
                    } = depth_bias_state;

                    // VUID?
                    if enable_dynamic
                        && !(device.api_version() >= Version::V1_3
                            || device.enabled_features().extended_dynamic_state2)
                    {
                        return Err(GraphicsPipelineCreationError::RequirementNotMet {
                            required_for: "`rasterization_state.depth_bias` is `Some(depth_bias_state)`, where `depth_bias_state.enable_dynamic` is set",
                            requires_one_of: RequiresOneOf {
                                api_version: Some(Version::V1_3),
                                features: &["extended_dynamic_state2"],
                                ..Default::default()
                            },
                        });
                    }

                    // VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-00754
                    if matches!(bias, StateMode::Fixed(bias) if bias.clamp != 0.0)
                        && !device.enabled_features().depth_bias_clamp
                    {
                        return Err(GraphicsPipelineCreationError::RequirementNotMet {
                            required_for: "`rasterization_state.depth_bias` is `Some(depth_bias_state)`, where `depth_bias_state.bias` is `StateMode::Fixed(bias)`, where `bias.clamp` is not `0.0`",
                            requires_one_of: RequiresOneOf {
                                features: &["depth_bias_clamp"],
                                ..Default::default()
                            },
                        });
                    }
                }

                // VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-00749
                if matches!(line_width, StateMode::Fixed(line_width) if line_width != 1.0)
                    && !device.enabled_features().wide_lines
                {
                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                        required_for: "`rasterization_state.line_width` is `StateMode::Fixed(line_width)`, where `line_width` is not `1.0`",
                        requires_one_of: RequiresOneOf {
                            features: &["wide_lines"],
                            ..Default::default()
                        },
                    });
                }

                if device.enabled_extensions().ext_line_rasterization {
                    // VUID-VkPipelineRasterizationLineStateCreateInfoEXT-lineRasterizationMode-parameter
                    line_rasterization_mode.validate_device(device)?;

                    match line_rasterization_mode {
                        LineRasterizationMode::Default => (),
                        LineRasterizationMode::Rectangular => {
                            // VUID-VkPipelineRasterizationLineStateCreateInfoEXT-lineRasterizationMode-02768
                            if !device.enabled_features().rectangular_lines {
                                return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                    required_for: "`rasterization_state.line_rasterization_mode` is `LineRasterizationMode::Rectangular`",
                                    requires_one_of: RequiresOneOf {
                                        features: &["rectangular_lines"],
                                        ..Default::default()
                                    },
                                });
                            }
                        }
                        LineRasterizationMode::Bresenham => {
                            // VUID-VkPipelineRasterizationLineStateCreateInfoEXT-lineRasterizationMode-02769
                            if !device.enabled_features().bresenham_lines {
                                return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                    required_for: "`rasterization_state.line_rasterization_mode` is `LineRasterizationMode::Bresenham`",
                                    requires_one_of: RequiresOneOf {
                                        features: &["bresenham_lines"],
                                        ..Default::default()
                                    },
                                });
                            }
                        }
                        LineRasterizationMode::RectangularSmooth => {
                            // VUID-VkPipelineRasterizationLineStateCreateInfoEXT-lineRasterizationMode-02770
                            if !device.enabled_features().smooth_lines {
                                return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                    required_for: "`rasterization_state.line_rasterization_mode` is `LineRasterizationMode::RectangularSmooth`",
                                    requires_one_of: RequiresOneOf {
                                        features: &["smooth_lines"],
                                        ..Default::default()
                                    },
                                });
                            }
                        }
                    }

                    if let Some(line_stipple) = line_stipple {
                        match line_rasterization_mode {
                            LineRasterizationMode::Default => {
                                // VUID-VkPipelineRasterizationLineStateCreateInfoEXT-stippledLineEnable-02774
                                if !device.enabled_features().stippled_rectangular_lines {
                                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                        required_for: "`rasterization_state.line_stipple` is `Some` and `rasterization_state.line_rasterization_mode` is `LineRasterizationMode::Default`",
                                        requires_one_of: RequiresOneOf {
                                            features: &["stippled_rectangular_lines"],
                                            ..Default::default()
                                        },
                                    });
                                }

                                // VUID-VkPipelineRasterizationLineStateCreateInfoEXT-stippledLineEnable-02774
                                if !properties.strict_lines {
                                    return Err(
                                        GraphicsPipelineCreationError::StrictLinesNotSupported,
                                    );
                                }
                            }
                            LineRasterizationMode::Rectangular => {
                                // VUID-VkPipelineRasterizationLineStateCreateInfoEXT-stippledLineEnable-02771
                                if !device.enabled_features().stippled_rectangular_lines {
                                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                        required_for: "`rasterization_state.line_stipple` is `Some` and `rasterization_state.line_rasterization_mode` is `LineRasterizationMode::Rectangular`",
                                        requires_one_of: RequiresOneOf {
                                            features: &["stippled_rectangular_lines"],
                                            ..Default::default()
                                        },
                                    });
                                }
                            }
                            LineRasterizationMode::Bresenham => {
                                // VUID-VkPipelineRasterizationLineStateCreateInfoEXT-stippledLineEnable-02772
                                if !device.enabled_features().stippled_bresenham_lines {
                                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                        required_for: "`rasterization_state.line_stipple` is `Some` and `rasterization_state.line_rasterization_mode` is `LineRasterizationMode::Bresenham`",
                                        requires_one_of: RequiresOneOf {
                                            features: &["stippled_bresenham_lines"],
                                            ..Default::default()
                                        },
                                    });
                                }
                            }
                            LineRasterizationMode::RectangularSmooth => {
                                // VUID-VkPipelineRasterizationLineStateCreateInfoEXT-stippledLineEnable-02773
                                if !device.enabled_features().stippled_smooth_lines {
                                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                        required_for: "`rasterization_state.line_stipple` is `Some` and `rasterization_state.line_rasterization_mode` is `LineRasterizationMode::RectangularSmooth`",
                                        requires_one_of: RequiresOneOf {
                                            features: &["stippled_smooth_lines"],
                                            ..Default::default()
                                        },
                                    });
                                }
                            }
                        }

                        if let StateMode::Fixed(line_stipple) = line_stipple {
                            // VUID-VkGraphicsPipelineCreateInfo-stippledLineEnable-02767
                            assert!(line_stipple.factor >= 1 && line_stipple.factor <= 256);
                            // TODO: return error?
                        }
                    }
                } else {
                    if line_rasterization_mode != LineRasterizationMode::Default {
                        return Err(GraphicsPipelineCreationError::RequirementNotMet {
                            required_for: "`rasterization_state.line_rasterization_mode` is not `LineRasterizationMode::Default`",
                            requires_one_of: RequiresOneOf {
                                device_extensions: &["ext_line_rasterization"],
                                ..Default::default()
                            },
                        });
                    }

                    if line_stipple.is_some() {
                        return Err(GraphicsPipelineCreationError::RequirementNotMet {
                            required_for: "`rasterization_state.line_stipple` is `Some`",
                            requires_one_of: RequiresOneOf {
                                device_extensions: &["ext_line_rasterization"],
                                ..Default::default()
                            },
                        });
                    }
                }
            }

            // Discard rectangle state
            {
                let &DiscardRectangleState {
                    mode,
                    ref rectangles,
                } = discard_rectangle_state;

                if device.enabled_extensions().ext_discard_rectangles {
                    // VUID-VkPipelineDiscardRectangleStateCreateInfoEXT-discardRectangleMode-parameter
                    mode.validate_device(device)?;

                    let discard_rectangle_count = match *rectangles {
                        PartialStateMode::Dynamic(count) => count,
                        PartialStateMode::Fixed(ref rectangles) => rectangles.len() as u32,
                    };

                    // VUID-VkPipelineDiscardRectangleStateCreateInfoEXT-discardRectangleCount-00582
                    if discard_rectangle_count > properties.max_discard_rectangles.unwrap() {
                        return Err(
                            GraphicsPipelineCreationError::MaxDiscardRectanglesExceeded {
                                max: properties.max_discard_rectangles.unwrap(),
                                obtained: discard_rectangle_count,
                            },
                        );
                    }
                } else {
                    let error = match *rectangles {
                        PartialStateMode::Dynamic(_) => true,
                        PartialStateMode::Fixed(ref rectangles) => !rectangles.is_empty(),
                    };

                    if error {
                        return Err(GraphicsPipelineCreationError::RequirementNotMet {
                            required_for: "`discard_rectangle_state.rectangles` is not `PartialStateMode::Fixed(vec![])`",
                            requires_one_of: RequiresOneOf {
                                device_extensions: &["ext_discard_rectangles"],
                                ..Default::default()
                            },
                        });
                    }
                }
            }

            // TODO:
            // VUID-VkPipelineShaderStageCreateInfo-stage-02596
            // VUID-VkPipelineShaderStageCreateInfo-stage-02597
            // VUID-VkGraphicsPipelineCreateInfo-pStages-00740
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06049
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06050
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06059
        }

        // VUID-VkGraphicsPipelineCreateInfo-pStages-00731
        if has.tessellation_state {
            let &TessellationState {
                patch_control_points,
            } = tessellation_state;

            // VUID-VkGraphicsPipelineCreateInfo-pStages-00736
            if !matches!(
                input_assembly_state.topology,
                PartialStateMode::Dynamic(PrimitiveTopologyClass::Patch)
                    | PartialStateMode::Fixed(PrimitiveTopology::PatchList)
            ) {
                return Err(GraphicsPipelineCreationError::InvalidPrimitiveTopology);
            }

            match patch_control_points {
                StateMode::Fixed(patch_control_points) => {
                    // VUID-VkPipelineTessellationStateCreateInfo-patchControlPoints-01214
                    if patch_control_points == 0
                        || patch_control_points > properties.max_tessellation_patch_size
                    {
                        return Err(GraphicsPipelineCreationError::InvalidNumPatchControlPoints);
                    }
                }
                StateMode::Dynamic => {
                    // VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-04870
                    if !device
                        .enabled_features()
                        .extended_dynamic_state2_patch_control_points
                    {
                        return Err(GraphicsPipelineCreationError::RequirementNotMet {
                            required_for:
                                "`tessellation_state.patch_control_points` is `StateMode::Dynamic`",
                            requires_one_of: RequiresOneOf {
                                features: &["extended_dynamic_state2_patch_control_points"],
                                ..Default::default()
                            },
                        });
                    }
                }
            };
        }

        // Viewport state
        // VUID-VkGraphicsPipelineCreateInfo-rasterizerDiscardEnable-00750
        // VUID-VkGraphicsPipelineCreateInfo-pViewportState-04892
        if has.viewport_state {
            let (viewport_count, scissor_count) = match *viewport_state {
                ViewportState::Fixed { ref data } => {
                    let count = data.len() as u32;
                    assert!(count != 0); // TODO: return error?

                    for (viewport, _) in data {
                        for i in 0..2 {
                            if viewport.dimensions[i] > properties.max_viewport_dimensions[i] as f32
                            {
                                return Err(
                                    GraphicsPipelineCreationError::MaxViewportDimensionsExceeded,
                                );
                            }

                            if viewport.origin[i] < properties.viewport_bounds_range[0]
                                || viewport.origin[i] + viewport.dimensions[i]
                                    > properties.viewport_bounds_range[1]
                            {
                                return Err(GraphicsPipelineCreationError::ViewportBoundsExceeded);
                            }
                        }
                    }

                    // TODO:
                    // VUID-VkPipelineViewportStateCreateInfo-offset-02822
                    // VUID-VkPipelineViewportStateCreateInfo-offset-02823

                    (count, count)
                }
                ViewportState::FixedViewport {
                    ref viewports,
                    scissor_count_dynamic,
                } => {
                    let viewport_count = viewports.len() as u32;

                    // VUID-VkPipelineViewportStateCreateInfo-scissorCount-04136
                    assert!(viewport_count != 0); // TODO: return error?

                    for viewport in viewports {
                        for i in 0..2 {
                            if viewport.dimensions[i] > properties.max_viewport_dimensions[i] as f32
                            {
                                return Err(
                                    GraphicsPipelineCreationError::MaxViewportDimensionsExceeded,
                                );
                            }

                            if viewport.origin[i] < properties.viewport_bounds_range[0]
                                || viewport.origin[i] + viewport.dimensions[i]
                                    > properties.viewport_bounds_range[1]
                            {
                                return Err(GraphicsPipelineCreationError::ViewportBoundsExceeded);
                            }
                        }
                    }

                    // VUID-VkPipelineViewportStateCreateInfo-scissorCount-04136
                    let scissor_count = if scissor_count_dynamic {
                        if !(device.api_version() >= Version::V1_3
                            || device.enabled_features().extended_dynamic_state)
                        {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for: "`viewport_state` is `ViewportState::FixedViewport`, where `scissor_count_dynamic` is set",
                                requires_one_of: RequiresOneOf {
                                    api_version: Some(Version::V1_3),
                                    features: &["extended_dynamic_state"],
                                    ..Default::default()
                                },
                            });
                        }

                        // VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-03380
                        0
                    } else {
                        viewport_count
                    };

                    (viewport_count, scissor_count)
                }
                ViewportState::FixedScissor {
                    ref scissors,
                    viewport_count_dynamic,
                } => {
                    let scissor_count = scissors.len() as u32;

                    // VUID-VkPipelineViewportStateCreateInfo-viewportCount-04135
                    assert!(scissor_count != 0); // TODO: return error?

                    // VUID-VkPipelineViewportStateCreateInfo-viewportCount-04135
                    let viewport_count = if viewport_count_dynamic {
                        if !(device.api_version() >= Version::V1_3
                            || device.enabled_features().extended_dynamic_state)
                        {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for: "`viewport_state` is `ViewportState::FixedScissor`, where `viewport_count_dynamic` is set",
                                requires_one_of: RequiresOneOf {
                                    api_version: Some(Version::V1_3),
                                    features: &["extended_dynamic_state"],
                                    ..Default::default()
                                },
                            });
                        }

                        // VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-03379
                        0
                    } else {
                        scissor_count
                    };

                    // TODO:
                    // VUID-VkPipelineViewportStateCreateInfo-offset-02822
                    // VUID-VkPipelineViewportStateCreateInfo-offset-02823

                    (viewport_count, scissor_count)
                }
                ViewportState::Dynamic {
                    count,
                    viewport_count_dynamic,
                    scissor_count_dynamic,
                } => {
                    // VUID-VkPipelineViewportStateCreateInfo-viewportCount-04135
                    // VUID-VkPipelineViewportStateCreateInfo-scissorCount-04136
                    if !(viewport_count_dynamic && scissor_count_dynamic) {
                        assert!(count != 0); // TODO: return error?
                    }

                    // VUID-VkPipelineViewportStateCreateInfo-viewportCount-04135
                    let viewport_count = if viewport_count_dynamic {
                        if !(device.api_version() >= Version::V1_3
                            || device.enabled_features().extended_dynamic_state)
                        {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for: "`viewport_state` is `ViewportState::Dynamic`, where `viewport_count_dynamic` is set",
                                requires_one_of: RequiresOneOf {
                                    api_version: Some(Version::V1_3),
                                    features: &["extended_dynamic_state"],
                                    ..Default::default()
                                },
                            });
                        }

                        // VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-03379
                        0
                    } else {
                        count
                    };

                    // VUID-VkPipelineViewportStateCreateInfo-scissorCount-04136
                    let scissor_count = if scissor_count_dynamic {
                        if !(device.api_version() >= Version::V1_3
                            || device.enabled_features().extended_dynamic_state)
                        {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for: "`viewport_state` is `ViewportState::Dynamic`, where `scissor_count_dynamic` is set",
                                requires_one_of: RequiresOneOf {
                                    api_version: Some(Version::V1_3),
                                    features: &["extended_dynamic_state"],
                                    ..Default::default()
                                },
                            });
                        }

                        // VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-03380
                        0
                    } else {
                        count
                    };

                    (viewport_count, scissor_count)
                }
            };

            // VUID-VkPipelineViewportStateCreateInfo-scissorCount-04134
            // Ensured by the definition of `ViewportState`.

            let viewport_scissor_count = u32::max(viewport_count, scissor_count);

            // VUID-VkPipelineViewportStateCreateInfo-viewportCount-01216
            // VUID-VkPipelineViewportStateCreateInfo-scissorCount-01217
            if viewport_scissor_count > 1 && !device.enabled_features().multi_viewport {
                return Err(GraphicsPipelineCreationError::RequirementNotMet {
                    required_for: "`viewport_state` has a fixed viewport/scissor count that is greater than `1`",
                    requires_one_of: RequiresOneOf {
                        features: &["multi_viewport"],
                        ..Default::default()
                    },
                });
            }

            // VUID-VkPipelineViewportStateCreateInfo-viewportCount-01218
            // VUID-VkPipelineViewportStateCreateInfo-scissorCount-01219
            if viewport_scissor_count > properties.max_viewports {
                return Err(GraphicsPipelineCreationError::MaxViewportsExceeded {
                    obtained: viewport_scissor_count,
                    max: properties.max_viewports,
                });
            }

            // TODO:
            // VUID-VkGraphicsPipelineCreateInfo-primitiveFragmentShadingRateWithMultipleViewports-04503
            // VUID-VkGraphicsPipelineCreateInfo-primitiveFragmentShadingRateWithMultipleViewports-04504
        }

        /*
            Fragment shader state
        */

        if has.fragment_shader_state {
            // Fragment shader
            if let Some((entry_point, specialization_data)) = fragment_shader {
                shader_stages.push(ShaderStageInfo {
                    entry_point,
                    specialization_map_entries: Fss::descriptors(),
                    _specialization_data: unsafe {
                        std::slice::from_raw_parts(
                            specialization_data as *const _ as *const u8,
                            size_of_val(specialization_data),
                        )
                    },
                });

                match entry_point.execution() {
                    ShaderExecution::Fragment => (),
                    _ => return Err(GraphicsPipelineCreationError::WrongShaderType),
                }

                // Check that the subpass can accept the output of the fragment shader.
                match render_pass {
                    PipelineRenderPassType::BeginRenderPass(subpass) => {
                        if !subpass.is_compatible_with(entry_point.output_interface()) {
                            return Err(
                                GraphicsPipelineCreationError::FragmentShaderRenderPassIncompatible,
                            );
                        }
                    }
                    PipelineRenderPassType::BeginRendering(_) => {
                        // TODO:
                    }
                }

                // TODO:
                // VUID-VkPipelineShaderStageCreateInfo-stage-00718
                // VUID-VkPipelineShaderStageCreateInfo-stage-06685
                // VUID-VkPipelineShaderStageCreateInfo-stage-06686
                // VUID-VkGraphicsPipelineCreateInfo-pStages-01565
                // VUID-VkGraphicsPipelineCreateInfo-renderPass-06056
                // VUID-VkGraphicsPipelineCreateInfo-renderPass-06061
            } else {
                // TODO: should probably error out here at least under some circumstances?
                // VUID?
            }

            // TODO:
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06038
        }

        // Depth/stencil state
        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06043
        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06053
        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06590
        if has.depth_stencil_state {
            let &DepthStencilState {
                ref depth,
                ref depth_bounds,
                ref stencil,
            } = depth_stencil_state;

            if let Some(depth_state) = depth {
                let &DepthState {
                    enable_dynamic,
                    write_enable,
                    compare_op,
                } = depth_state;

                let has_depth_attachment = match render_pass {
                    PipelineRenderPassType::BeginRenderPass(subpass) => subpass.has_depth(),
                    PipelineRenderPassType::BeginRendering(rendering_info) => {
                        rendering_info.depth_attachment_format.is_none()
                    }
                };

                // VUID?
                if !has_depth_attachment {
                    return Err(GraphicsPipelineCreationError::NoDepthAttachment);
                }

                // VUID?
                if enable_dynamic
                    && !(device.api_version() >= Version::V1_3
                        || device.enabled_features().extended_dynamic_state)
                {
                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                        required_for:
                            "`depth_stencil_state.depth` is `Some(depth_state)`, where `depth_state.enable_dynamic` is set",
                        requires_one_of: RequiresOneOf {
                            api_version: Some(Version::V1_3),
                            features: &["extended_dynamic_state"],
                            ..Default::default()
                        },
                    });
                }

                match write_enable {
                    StateMode::Fixed(write_enable) => {
                        match render_pass {
                            PipelineRenderPassType::BeginRenderPass(subpass) => {
                                // VUID-VkGraphicsPipelineCreateInfo-renderPass-06039
                                if write_enable && !subpass.has_writable_depth() {
                                    return Err(GraphicsPipelineCreationError::NoDepthAttachment);
                                }
                            }
                            PipelineRenderPassType::BeginRendering(_) => {
                                // No VUID?
                            }
                        }
                    }
                    StateMode::Dynamic => {
                        // VUID?
                        if !(device.api_version() >= Version::V1_3
                            || device.enabled_features().extended_dynamic_state)
                        {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for: "`depth_stencil_state.depth` is `Some(depth_state)`, where `depth_state.write_enable` is `StateMode::Dynamic`",
                                requires_one_of: RequiresOneOf {
                                    api_version: Some(Version::V1_3),
                                    features: &["extended_dynamic_state"],
                                    ..Default::default()
                                },
                            });
                        }
                    }
                }

                match compare_op {
                    StateMode::Fixed(compare_op) => {
                        // VUID-VkPipelineDepthStencilStateCreateInfo-depthCompareOp-parameter
                        compare_op.validate_device(device)?;
                    }
                    StateMode::Dynamic => {
                        // VUID?
                        if !(device.api_version() >= Version::V1_3
                            || device.enabled_features().extended_dynamic_state)
                        {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for: "`depth_stencil_state.depth` is `Some(depth_state)`, where `depth_state.compare_op` is `StateMode::Dynamic`",
                                requires_one_of: RequiresOneOf {
                                    api_version: Some(Version::V1_3),
                                    features: &["extended_dynamic_state"],
                                    ..Default::default()
                                },
                            });
                        }
                    }
                }
            }

            if let Some(depth_bounds_state) = depth_bounds {
                let &DepthBoundsState {
                    enable_dynamic,
                    ref bounds,
                } = depth_bounds_state;

                // VUID-VkPipelineDepthStencilStateCreateInfo-depthBoundsTestEnable-00598
                if !device.enabled_features().depth_bounds {
                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                        required_for: "`depth_stencil_state.depth_bounds` is `Some`",
                        requires_one_of: RequiresOneOf {
                            features: &["depth_bounds"],
                            ..Default::default()
                        },
                    });
                }

                // VUID?
                if enable_dynamic
                    && !(device.api_version() >= Version::V1_3
                        || device.enabled_features().extended_dynamic_state)
                {
                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                        required_for: "`depth_stencil_state.depth_bounds` is `Some(depth_bounds_state)`, where `depth_bounds_state.enable_dynamic` is set",
                        requires_one_of: RequiresOneOf {
                            api_version: Some(Version::V1_3),
                            features: &["extended_dynamic_state"],
                            ..Default::default()
                        },
                    });
                }

                if let StateMode::Fixed(bounds) = bounds {
                    // VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-02510
                    if !device.enabled_extensions().ext_depth_range_unrestricted
                        && !(0.0..1.0).contains(bounds.start())
                        && !(0.0..1.0).contains(bounds.end())
                    {
                        return Err(GraphicsPipelineCreationError::RequirementNotMet {
                            required_for: "`depth_stencil_state.depth_bounds` is `Some(depth_bounds_state)`, where `depth_bounds_state.bounds` is not between `0.0` and `1.0` inclusive",
                            requires_one_of: RequiresOneOf {
                                device_extensions: &["ext_depth_range_unrestricted"],
                                ..Default::default()
                            },
                        });
                    }
                }
            }

            if let Some(stencil_state) = stencil {
                let &StencilState {
                    enable_dynamic,
                    ref front,
                    ref back,
                } = stencil_state;

                let has_stencil_attachment = match render_pass {
                    PipelineRenderPassType::BeginRenderPass(subpass) => subpass.has_stencil(),
                    PipelineRenderPassType::BeginRendering(rendering_info) => {
                        rendering_info.stencil_attachment_format.is_none()
                    }
                };

                if !has_stencil_attachment {
                    return Err(GraphicsPipelineCreationError::NoDepthAttachment);
                }

                // VUID?
                if enable_dynamic
                    && !(device.api_version() >= Version::V1_3
                        || device.enabled_features().extended_dynamic_state)
                {
                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                        required_for:
                            "`depth_stencil_state.stencil` is `Some(stencil_state)`, where `stencil_state.enable_dynamic` is set",
                        requires_one_of: RequiresOneOf {
                            api_version: Some(Version::V1_3),
                            features: &["extended_dynamic_state"],
                            ..Default::default()
                        },
                    });
                }

                match (front.ops, back.ops) {
                    (StateMode::Fixed(front_ops), StateMode::Fixed(back_ops)) => {
                        for ops in [front_ops, back_ops] {
                            let StencilOps {
                                fail_op,
                                pass_op,
                                depth_fail_op,
                                compare_op,
                            } = ops;

                            // VUID-VkStencilOpState-failOp-parameter
                            fail_op.validate_device(device)?;

                            // VUID-VkStencilOpState-passOp-parameter
                            pass_op.validate_device(device)?;

                            // VUID-VkStencilOpState-depthFailOp-parameter
                            depth_fail_op.validate_device(device)?;

                            // VUID-VkStencilOpState-compareOp-parameter
                            compare_op.validate_device(device)?;
                        }
                    }
                    (StateMode::Dynamic, StateMode::Dynamic) => {
                        // VUID?
                        if !(device.api_version() >= Version::V1_3
                            || device.enabled_features().extended_dynamic_state)
                        {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for: "`depth_stencil_state.stencil` is `Some(stencil_state)`, where `stencil_state.front.ops` and `stencil_state.back.ops` are `StateMode::Dynamic`",
                                requires_one_of: RequiresOneOf {
                                    api_version: Some(Version::V1_3),
                                    features: &["extended_dynamic_state"],
                                    ..Default::default()
                                },
                            });
                        }
                    }
                    _ => return Err(GraphicsPipelineCreationError::WrongStencilState),
                }

                if !matches!(
                    (front.compare_mask, back.compare_mask),
                    (StateMode::Fixed(_), StateMode::Fixed(_))
                        | (StateMode::Dynamic, StateMode::Dynamic)
                ) {
                    return Err(GraphicsPipelineCreationError::WrongStencilState);
                }

                if !matches!(
                    (front.write_mask, back.write_mask),
                    (StateMode::Fixed(_), StateMode::Fixed(_))
                        | (StateMode::Dynamic, StateMode::Dynamic)
                ) {
                    return Err(GraphicsPipelineCreationError::WrongStencilState);
                }

                if !matches!(
                    (front.reference, back.reference),
                    (StateMode::Fixed(_), StateMode::Fixed(_))
                        | (StateMode::Dynamic, StateMode::Dynamic)
                ) {
                    return Err(GraphicsPipelineCreationError::WrongStencilState);
                }

                // TODO:
                // VUID-VkGraphicsPipelineCreateInfo-renderPass-06040
            }
        }

        /*
            Fragment output state
        */

        if has.fragment_output_state {
            // Multisample state
            // VUID-VkGraphicsPipelineCreateInfo-rasterizerDiscardEnable-00751
            {
                let &MultisampleState {
                    rasterization_samples,
                    sample_shading,
                    sample_mask: _,
                    alpha_to_coverage_enable: _,
                    alpha_to_one_enable,
                } = multisample_state;

                // VUID-VkPipelineMultisampleStateCreateInfo-rasterizationSamples-parameter
                rasterization_samples.validate_device(device)?;

                match render_pass {
                    PipelineRenderPassType::BeginRenderPass(subpass) => {
                        if let Some(samples) = subpass.num_samples() {
                            // VUID-VkGraphicsPipelineCreateInfo-subpass-00757
                            if rasterization_samples != samples {
                                return Err(GraphicsPipelineCreationError::MultisampleRasterizationSamplesMismatch);
                            }
                        }

                        // TODO:
                        // VUID-VkGraphicsPipelineCreateInfo-subpass-00758
                        // VUID-VkGraphicsPipelineCreateInfo-subpass-01505
                        // VUID-VkGraphicsPipelineCreateInfo-subpass-01411
                        // VUID-VkGraphicsPipelineCreateInfo-subpass-01412
                    }
                    PipelineRenderPassType::BeginRendering(_) => {
                        // No equivalent VUIDs for dynamic rendering, as no sample count information
                        // is provided until `begin_rendering`.
                    }
                }

                if let Some(min_sample_shading) = sample_shading {
                    // VUID-VkPipelineMultisampleStateCreateInfo-sampleShadingEnable-00784
                    if !device.enabled_features().sample_rate_shading {
                        return Err(GraphicsPipelineCreationError::RequirementNotMet {
                            required_for: "`multisample_state.sample_shading` is `Some`",
                            requires_one_of: RequiresOneOf {
                                features: &["sample_rate_shading"],
                                ..Default::default()
                            },
                        });
                    }

                    // VUID-VkPipelineMultisampleStateCreateInfo-minSampleShading-00786
                    // TODO: return error?
                    assert!((0.0..=1.0).contains(&min_sample_shading));
                }

                // VUID-VkPipelineMultisampleStateCreateInfo-alphaToOneEnable-00785
                if alpha_to_one_enable && !device.enabled_features().alpha_to_one {
                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                        required_for: "`multisample_state.alpha_to_one_enable` is set",
                        requires_one_of: RequiresOneOf {
                            features: &["alpha_to_one"],
                            ..Default::default()
                        },
                    });
                }

                // TODO:
                // VUID-VkGraphicsPipelineCreateInfo-lineRasterizationMode-02766
            }
        }

        // Color blend state
        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06044
        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06054
        if has.color_blend_state {
            let &ColorBlendState {
                logic_op,
                ref attachments,
                blend_constants: _,
            } = color_blend_state;

            if let Some(logic_op) = logic_op {
                // VUID-VkPipelineColorBlendStateCreateInfo-logicOpEnable-00606
                if !device.enabled_features().logic_op {
                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                        required_for: "`color_blend_state.logic_op` is `Some`",
                        requires_one_of: RequiresOneOf {
                            features: &["logic_op"],
                            ..Default::default()
                        },
                    });
                }

                match logic_op {
                    StateMode::Fixed(logic_op) => {
                        // VUID-VkPipelineColorBlendStateCreateInfo-logicOpEnable-00607
                        logic_op.validate_device(device)?
                    }
                    StateMode::Dynamic => {
                        // VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-04869
                        if !device.enabled_features().extended_dynamic_state2_logic_op {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for:
                                    "`color_blend_state.logic_op` is `Some(StateMode::Dynamic)`",
                                requires_one_of: RequiresOneOf {
                                    features: &["extended_dynamic_state2_logic_op"],
                                    ..Default::default()
                                },
                            });
                        }
                    }
                }
            }

            let color_attachment_count = match render_pass {
                PipelineRenderPassType::BeginRenderPass(subpass) => {
                    subpass.subpass_desc().color_attachments.len()
                }
                PipelineRenderPassType::BeginRendering(rendering_info) => {
                    rendering_info.color_attachment_formats.len()
                }
            };

            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06042
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06055
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06060
            if color_attachment_count != attachments.len() {
                return Err(GraphicsPipelineCreationError::MismatchBlendingAttachmentsCount);
            }

            if attachments.len() > 1 && !device.enabled_features().independent_blend {
                // Ensure that all `blend` and `color_write_mask` are identical.
                let mut iter = attachments
                    .iter()
                    .map(|state| (&state.blend, &state.color_write_mask));
                let first = iter.next().unwrap();

                // VUID-VkPipelineColorBlendStateCreateInfo-pAttachments-00605
                if !iter.all(|state| state == first) {
                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                        required_for: "`color_blend_state.attachments` has elements where `blend` and `color_write_mask` do not match the other elements",
                        requires_one_of: RequiresOneOf {
                            features: &["independent_blend"],
                            ..Default::default()
                        },
                    });
                }
            }

            for (attachment_index, state) in attachments.iter().enumerate() {
                let &ColorBlendAttachmentState {
                    blend,
                    color_write_mask: _,
                    color_write_enable,
                } = state;

                if let Some(blend) = blend {
                    let AttachmentBlend {
                        color_op,
                        color_source,
                        color_destination,
                        alpha_op,
                        alpha_source,
                        alpha_destination,
                    } = blend;

                    // VUID-VkPipelineColorBlendAttachmentState-colorBlendOp-parameter
                    color_op.validate_device(device)?;

                    // VUID-VkPipelineColorBlendAttachmentState-srcColorBlendFactor-parameter
                    color_source.validate_device(device)?;

                    // VUID-VkPipelineColorBlendAttachmentState-dstColorBlendFactor-parameter
                    color_destination.validate_device(device)?;

                    // VUID-VkPipelineColorBlendAttachmentState-alphaBlendOp-parameter
                    alpha_op.validate_device(device)?;

                    // VUID-VkPipelineColorBlendAttachmentState-srcAlphaBlendFactor-parameter
                    alpha_source.validate_device(device)?;

                    // VUID-VkPipelineColorBlendAttachmentState-dstAlphaBlendFactor-parameter
                    alpha_destination.validate_device(device)?;

                    // VUID?
                    if !device.enabled_features().dual_src_blend
                        && [
                            color_source,
                            color_destination,
                            alpha_source,
                            alpha_destination,
                        ]
                        .into_iter()
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
                        return Err(GraphicsPipelineCreationError::RequirementNotMet {
                            required_for: "`color_blend_state.attachments` has an element where `blend` is `Some(blend)`, where `blend.color_source`, `blend.color_destination`, `blend.alpha_source` or `blend.alpha_destination` is `BlendFactor::Src1*`",
                            requires_one_of: RequiresOneOf {
                                features: &["dual_src_blend"],
                                ..Default::default()
                            },
                        });
                    }

                    let attachment_format = match render_pass {
                        PipelineRenderPassType::BeginRenderPass(subpass) => subpass
                            .subpass_desc()
                            .color_attachments[attachment_index]
                            .as_ref()
                            .and_then(|atch_ref| {
                                subpass.render_pass().attachments()[atch_ref.attachment as usize]
                                    .format
                            }),
                        PipelineRenderPassType::BeginRendering(rendering_info) => {
                            rendering_info.color_attachment_formats[attachment_index]
                        }
                    };

                    // VUID-VkGraphicsPipelineCreateInfo-renderPass-06041
                    // VUID-VkGraphicsPipelineCreateInfo-renderPass-06062
                    // Use unchecked, because all validation has been done above or by the
                    // render pass creation.
                    if !attachment_format.map_or(false, |format| unsafe {
                        physical_device
                            .format_properties_unchecked(format)
                            .potential_format_features()
                            .color_attachment_blend
                    }) {
                        return Err(
                            GraphicsPipelineCreationError::ColorAttachmentFormatBlendNotSupported {
                                attachment_index: attachment_index as u32,
                            },
                        );
                    }
                }

                match color_write_enable {
                    StateMode::Fixed(enable) => {
                        // VUID-VkPipelineColorWriteCreateInfoEXT-pAttachments-04801
                        if !enable && !device.enabled_features().color_write_enable {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for: "`color_blend_state.attachments` has an element where `color_write_enable` is `StateMode::Fixed(false)`",
                                requires_one_of: RequiresOneOf {
                                    features: &["color_write_enable"],
                                    ..Default::default()
                                },
                            });
                        }
                    }
                    StateMode::Dynamic => {
                        // VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-04800
                        if !device.enabled_features().color_write_enable {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for: "`color_blend_state.attachments` has an element where `color_write_enable` is `StateMode::Dynamic`",
                                requires_one_of: RequiresOneOf {
                                    features: &["color_write_enable"],
                                    ..Default::default()
                                },
                            });
                        }
                    }
                }
            }
        }

        /*
            Generic shader checks
        */

        for stage_info in &shader_stages {
            // VUID-VkGraphicsPipelineCreateInfo-layout-00756
            pipeline_layout.ensure_compatible_with_shader(
                stage_info.entry_point.descriptor_requirements(),
                stage_info.entry_point.push_constant_requirements(),
            )?;

            for (constant_id, reqs) in stage_info
                .entry_point
                .specialization_constant_requirements()
            {
                let map_entry = stage_info
                    .specialization_map_entries
                    .iter()
                    .find(|desc| desc.constant_id == constant_id)
                    .ok_or(GraphicsPipelineCreationError::IncompatibleSpecializationConstants)?;

                if map_entry.size as DeviceSize != reqs.size {
                    return Err(GraphicsPipelineCreationError::IncompatibleSpecializationConstants);
                }
            }
        }

        // VUID-VkGraphicsPipelineCreateInfo-pStages-00742
        // VUID-VkGraphicsPipelineCreateInfo-None-04889
        // TODO: this check is too strict; the output only has to be a superset, any variables
        // not used in the input of the next shader are just ignored.
        for (output, input) in shader_stages.iter().zip(shader_stages.iter().skip(1)) {
            if let Err(err) = input
                .entry_point
                .input_interface()
                .matches(output.entry_point.output_interface())
            {
                return Err(GraphicsPipelineCreationError::ShaderStagesMismatch(err));
            }
        }

        // TODO:
        // VUID-VkPipelineShaderStageCreateInfo-maxClipDistances-00708
        // VUID-VkPipelineShaderStageCreateInfo-maxCullDistances-00709
        // VUID-VkPipelineShaderStageCreateInfo-maxCombinedClipAndCullDistances-00710
        // VUID-VkPipelineShaderStageCreateInfo-maxSampleMaskWords-00711

        // Dynamic states not handled yet:
        // - ViewportWScaling (VkPipelineViewportWScalingStateCreateInfoNV)
        // - SampleLocations (VkPipelineSampleLocationsStateCreateInfoEXT)
        // - ViewportShadingRatePalette (VkPipelineViewportShadingRateImageStateCreateInfoNV)
        // - ViewportCoarseSampleOrder (VkPipelineViewportCoarseSampleOrderStateCreateInfoNV)
        // - ExclusiveScissor (VkPipelineViewportExclusiveScissorStateCreateInfoNV)
        // - FragmentShadingRate (VkPipelineFragmentShadingRateStateCreateInfoKHR)

        Ok(())
    }

    unsafe fn record_create(
        &self,
        device: &Device,
        pipeline_layout: &PipelineLayout,
        vertex_input_state: &VertexInputState,
        has: Has,
    ) -> Result<
        (
            ash::vk::Pipeline,
            HashMap<(u32, u32), DescriptorRequirements>,
            HashMap<DynamicState, bool>,
            HashMap<ShaderStage, ()>,
        ),
        GraphicsPipelineCreationError,
    > {
        let Self {
            render_pass,
            cache,

            vertex_shader,
            tessellation_shaders,
            geometry_shader,
            fragment_shader,

            vertex_input_state: _,
            input_assembly_state,
            tessellation_state,
            viewport_state,
            discard_rectangle_state,
            rasterization_state,
            multisample_state,
            depth_stencil_state,
            color_blend_state,
        } = self;

        let render_pass = render_pass.as_ref().unwrap();

        let mut descriptor_requirements: HashMap<(u32, u32), DescriptorRequirements> =
            HashMap::new();
        let mut dynamic_state: HashMap<DynamicState, bool> = HashMap::default();
        let mut stages = HashMap::default();
        let mut stages_vk: SmallVec<[_; 5]> = SmallVec::new();

        /*
            Render pass
        */

        let mut render_pass_vk = ash::vk::RenderPass::null();
        let mut subpass_vk = 0;
        let mut color_attachment_formats_vk: SmallVec<[_; 4]> = SmallVec::new();
        let mut rendering_create_info_vk = None;

        match render_pass {
            PipelineRenderPassType::BeginRenderPass(subpass) => {
                render_pass_vk = subpass.render_pass().internal_object();
                subpass_vk = subpass.index();
            }
            PipelineRenderPassType::BeginRendering(rendering_info) => {
                let &PipelineRenderingCreateInfo {
                    view_mask,
                    ref color_attachment_formats,
                    depth_attachment_format,
                    stencil_attachment_format,
                    _ne: _,
                } = rendering_info;

                color_attachment_formats_vk.extend(
                    color_attachment_formats
                        .iter()
                        .map(|format| format.map_or(ash::vk::Format::UNDEFINED, Into::into)),
                );

                let _ = rendering_create_info_vk.insert(ash::vk::PipelineRenderingCreateInfo {
                    view_mask,
                    color_attachment_count: color_attachment_formats_vk.len() as u32,
                    p_color_attachment_formats: color_attachment_formats_vk.as_ptr(),
                    depth_attachment_format: depth_attachment_format
                        .map_or(ash::vk::Format::UNDEFINED, Into::into),
                    stencil_attachment_format: stencil_attachment_format
                        .map_or(ash::vk::Format::UNDEFINED, Into::into),
                    ..Default::default()
                });
            }
        }

        /*
            Vertex input state
        */

        let mut vertex_binding_descriptions_vk: SmallVec<[_; 8]> = SmallVec::new();
        let mut vertex_attribute_descriptions_vk: SmallVec<[_; 8]> = SmallVec::new();
        let mut vertex_binding_divisor_descriptions_vk: SmallVec<[_; 8]> = SmallVec::new();
        let mut vertex_binding_divisor_state_vk = None;
        let mut vertex_input_state_vk = None;
        let mut input_assembly_state_vk = None;

        if has.vertex_input_state {
            // Vertex input state
            {
                dynamic_state.insert(DynamicState::VertexInput, false);

                let &VertexInputState {
                    ref bindings,
                    ref attributes,
                } = vertex_input_state;

                vertex_binding_descriptions_vk.extend(bindings.iter().map(
                    |(&binding, binding_desc)| ash::vk::VertexInputBindingDescription {
                        binding,
                        stride: binding_desc.stride,
                        input_rate: binding_desc.input_rate.into(),
                    },
                ));

                vertex_attribute_descriptions_vk.extend(attributes.iter().map(
                    |(&location, attribute_desc)| ash::vk::VertexInputAttributeDescription {
                        location,
                        binding: attribute_desc.binding,
                        format: attribute_desc.format.into(),
                        offset: attribute_desc.offset,
                    },
                ));

                let vertex_input_state =
                    vertex_input_state_vk.insert(ash::vk::PipelineVertexInputStateCreateInfo {
                        flags: ash::vk::PipelineVertexInputStateCreateFlags::empty(),
                        vertex_binding_description_count: vertex_binding_descriptions_vk.len()
                            as u32,
                        p_vertex_binding_descriptions: vertex_binding_descriptions_vk.as_ptr(),
                        vertex_attribute_description_count: vertex_attribute_descriptions_vk.len()
                            as u32,
                        p_vertex_attribute_descriptions: vertex_attribute_descriptions_vk.as_ptr(),
                        ..Default::default()
                    });

                {
                    vertex_binding_divisor_descriptions_vk.extend(
                        bindings
                            .iter()
                            .filter_map(|(&binding, binding_desc)| match binding_desc.input_rate {
                                VertexInputRate::Instance { divisor } if divisor != 1 => {
                                    Some((binding, divisor))
                                }
                                _ => None,
                            })
                            .map(|(binding, divisor)| {
                                ash::vk::VertexInputBindingDivisorDescriptionEXT {
                                    binding,
                                    divisor,
                                }
                            }),
                    );

                    // VUID-VkPipelineVertexInputDivisorStateCreateInfoEXT-vertexBindingDivisorCount-arraylength
                    if !vertex_binding_divisor_descriptions_vk.is_empty() {
                        vertex_input_state.p_next =
                            vertex_binding_divisor_state_vk.insert(
                                ash::vk::PipelineVertexInputDivisorStateCreateInfoEXT {
                                    vertex_binding_divisor_count:
                                        vertex_binding_divisor_descriptions_vk.len() as u32,
                                    p_vertex_binding_divisors:
                                        vertex_binding_divisor_descriptions_vk.as_ptr(),
                                    ..Default::default()
                                },
                            ) as *const _ as *const _;
                    }
                }
            }

            // Input assembly state
            {
                let &InputAssemblyState {
                    topology,
                    primitive_restart_enable,
                } = input_assembly_state;

                let topology = match topology {
                    PartialStateMode::Fixed(topology) => {
                        dynamic_state.insert(DynamicState::PrimitiveTopology, false);
                        topology.into()
                    }
                    PartialStateMode::Dynamic(topology_class) => {
                        dynamic_state.insert(DynamicState::PrimitiveTopology, true);
                        topology_class.example().into()
                    }
                };

                let primitive_restart_enable = match primitive_restart_enable {
                    StateMode::Fixed(primitive_restart_enable) => {
                        dynamic_state.insert(DynamicState::PrimitiveRestartEnable, false);
                        primitive_restart_enable as ash::vk::Bool32
                    }
                    StateMode::Dynamic => {
                        dynamic_state.insert(DynamicState::PrimitiveRestartEnable, true);
                        Default::default()
                    }
                };

                let _ =
                    input_assembly_state_vk.insert(ash::vk::PipelineInputAssemblyStateCreateInfo {
                        flags: ash::vk::PipelineInputAssemblyStateCreateFlags::empty(),
                        topology,
                        primitive_restart_enable,
                        ..Default::default()
                    });
            }
        }

        /*
            Pre-rasterization shader state
        */

        let mut vertex_shader_specialization_vk = None;
        let mut tessellation_control_shader_specialization_vk = None;
        let mut tessellation_evaluation_shader_specialization_vk = None;
        let mut tessellation_state_vk = None;
        let mut geometry_shader_specialization_vk = None;
        let mut viewports_vk: SmallVec<[_; 2]> = SmallVec::new();
        let mut scissors_vk: SmallVec<[_; 2]> = SmallVec::new();
        let mut viewport_state_vk = None;
        let mut rasterization_line_state_vk = None;
        let mut rasterization_state_vk = None;
        let mut discard_rectangles: SmallVec<[_; 2]> = SmallVec::new();
        let mut discard_rectangle_state_vk = None;

        if has.pre_rasterization_shader_state {
            // Vertex shader
            if let Some((entry_point, specialization_data)) = vertex_shader {
                let specialization_map_entries = Vss::descriptors();
                let specialization_data = slice::from_raw_parts(
                    specialization_data as *const _ as *const u8,
                    size_of_val(specialization_data),
                );

                let specialization_info_vk =
                    vertex_shader_specialization_vk.insert(ash::vk::SpecializationInfo {
                        map_entry_count: specialization_map_entries.len() as u32,
                        p_map_entries: specialization_map_entries.as_ptr() as *const _,
                        data_size: specialization_data.len(),
                        p_data: specialization_data.as_ptr() as *const _,
                    });

                for (loc, reqs) in entry_point.descriptor_requirements() {
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

                stages.insert(ShaderStage::Vertex, ());
                stages_vk.push(ash::vk::PipelineShaderStageCreateInfo {
                    flags: ash::vk::PipelineShaderStageCreateFlags::empty(),
                    stage: ash::vk::ShaderStageFlags::VERTEX,
                    module: entry_point.module().internal_object(),
                    p_name: entry_point.name().as_ptr(),
                    p_specialization_info: specialization_info_vk as *const _,
                    ..Default::default()
                });
            }

            // Tessellation shaders
            if let Some(tessellation_shaders) = tessellation_shaders {
                {
                    let (entry_point, specialization_data) = &tessellation_shaders.control;
                    let specialization_map_entries = Tcss::descriptors();
                    let specialization_data = slice::from_raw_parts(
                        specialization_data as *const _ as *const u8,
                        size_of_val(specialization_data),
                    );

                    let specialization_info_vk = tessellation_control_shader_specialization_vk
                        .insert(ash::vk::SpecializationInfo {
                            map_entry_count: specialization_map_entries.len() as u32,
                            p_map_entries: specialization_map_entries.as_ptr() as *const _,
                            data_size: specialization_data.len(),
                            p_data: specialization_data.as_ptr() as *const _,
                        });

                    for (loc, reqs) in entry_point.descriptor_requirements() {
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

                    stages.insert(ShaderStage::TessellationControl, ());
                    stages_vk.push(ash::vk::PipelineShaderStageCreateInfo {
                        flags: ash::vk::PipelineShaderStageCreateFlags::empty(),
                        stage: ash::vk::ShaderStageFlags::TESSELLATION_CONTROL,
                        module: entry_point.module().internal_object(),
                        p_name: entry_point.name().as_ptr(),
                        p_specialization_info: specialization_info_vk as *const _,
                        ..Default::default()
                    });
                }

                {
                    let (entry_point, specialization_data) = &tessellation_shaders.evaluation;
                    let specialization_map_entries = Tess::descriptors();
                    let specialization_data = slice::from_raw_parts(
                        specialization_data as *const _ as *const u8,
                        size_of_val(specialization_data),
                    );

                    let specialization_info_vk = tessellation_evaluation_shader_specialization_vk
                        .insert(ash::vk::SpecializationInfo {
                            map_entry_count: specialization_map_entries.len() as u32,
                            p_map_entries: specialization_map_entries.as_ptr() as *const _,
                            data_size: specialization_data.len(),
                            p_data: specialization_data.as_ptr() as *const _,
                        });

                    for (loc, reqs) in entry_point.descriptor_requirements() {
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

                    stages.insert(ShaderStage::TessellationEvaluation, ());
                    stages_vk.push(ash::vk::PipelineShaderStageCreateInfo {
                        flags: ash::vk::PipelineShaderStageCreateFlags::empty(),
                        stage: ash::vk::ShaderStageFlags::TESSELLATION_EVALUATION,
                        module: entry_point.module().internal_object(),
                        p_name: entry_point.name().as_ptr(),
                        p_specialization_info: specialization_info_vk as *const _,
                        ..Default::default()
                    });
                }
            }

            // Geometry shader
            if let Some((entry_point, specialization_data)) = geometry_shader {
                let specialization_map_entries = Gss::descriptors();
                let specialization_data = slice::from_raw_parts(
                    specialization_data as *const _ as *const u8,
                    size_of_val(specialization_data),
                );

                let specialization_info_vk =
                    geometry_shader_specialization_vk.insert(ash::vk::SpecializationInfo {
                        map_entry_count: specialization_map_entries.len() as u32,
                        p_map_entries: specialization_map_entries.as_ptr() as *const _,
                        data_size: specialization_data.len(),
                        p_data: specialization_data.as_ptr() as *const _,
                    });

                for (loc, reqs) in entry_point.descriptor_requirements() {
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

                stages.insert(ShaderStage::Geometry, ());
                stages_vk.push(ash::vk::PipelineShaderStageCreateInfo {
                    flags: ash::vk::PipelineShaderStageCreateFlags::empty(),
                    stage: ash::vk::ShaderStageFlags::GEOMETRY,
                    module: entry_point.module().internal_object(),
                    p_name: entry_point.name().as_ptr(),
                    p_specialization_info: specialization_info_vk as *const _,
                    ..Default::default()
                });
            }

            // Rasterization state
            {
                let &RasterizationState {
                    depth_clamp_enable,
                    rasterizer_discard_enable,
                    polygon_mode,
                    cull_mode,
                    front_face,
                    depth_bias,
                    line_width,
                    line_rasterization_mode,
                    line_stipple,
                } = rasterization_state;

                let rasterizer_discard_enable = match rasterizer_discard_enable {
                    StateMode::Fixed(rasterizer_discard_enable) => {
                        dynamic_state.insert(DynamicState::RasterizerDiscardEnable, false);
                        rasterizer_discard_enable as ash::vk::Bool32
                    }
                    StateMode::Dynamic => {
                        dynamic_state.insert(DynamicState::RasterizerDiscardEnable, true);
                        ash::vk::FALSE
                    }
                };

                let cull_mode = match cull_mode {
                    StateMode::Fixed(cull_mode) => {
                        dynamic_state.insert(DynamicState::CullMode, false);
                        cull_mode.into()
                    }
                    StateMode::Dynamic => {
                        dynamic_state.insert(DynamicState::CullMode, true);
                        CullMode::default().into()
                    }
                };

                let front_face = match front_face {
                    StateMode::Fixed(front_face) => {
                        dynamic_state.insert(DynamicState::FrontFace, false);
                        front_face.into()
                    }
                    StateMode::Dynamic => {
                        dynamic_state.insert(DynamicState::FrontFace, true);
                        FrontFace::default().into()
                    }
                };

                let (
                    depth_bias_enable,
                    depth_bias_constant_factor,
                    depth_bias_clamp,
                    depth_bias_slope_factor,
                ) = if let Some(depth_bias_state) = depth_bias {
                    if depth_bias_state.enable_dynamic {
                        dynamic_state.insert(DynamicState::DepthBiasEnable, true);
                    } else {
                        dynamic_state.insert(DynamicState::DepthBiasEnable, false);
                    }

                    let (constant_factor, clamp, slope_factor) = match depth_bias_state.bias {
                        StateMode::Fixed(bias) => {
                            dynamic_state.insert(DynamicState::DepthBias, false);
                            (bias.constant_factor, bias.clamp, bias.slope_factor)
                        }
                        StateMode::Dynamic => {
                            dynamic_state.insert(DynamicState::DepthBias, true);
                            (0.0, 0.0, 0.0)
                        }
                    };

                    (ash::vk::TRUE, constant_factor, clamp, slope_factor)
                } else {
                    (ash::vk::FALSE, 0.0, 0.0, 0.0)
                };

                let line_width = match line_width {
                    StateMode::Fixed(line_width) => {
                        dynamic_state.insert(DynamicState::LineWidth, false);
                        line_width
                    }
                    StateMode::Dynamic => {
                        dynamic_state.insert(DynamicState::LineWidth, true);
                        1.0
                    }
                };

                let rasterization_state =
                    rasterization_state_vk.insert(ash::vk::PipelineRasterizationStateCreateInfo {
                        flags: ash::vk::PipelineRasterizationStateCreateFlags::empty(),
                        depth_clamp_enable: depth_clamp_enable as ash::vk::Bool32,
                        rasterizer_discard_enable,
                        polygon_mode: polygon_mode.into(),
                        cull_mode,
                        front_face,
                        depth_bias_enable,
                        depth_bias_constant_factor,
                        depth_bias_clamp,
                        depth_bias_slope_factor,
                        line_width,
                        ..Default::default()
                    });

                if device.enabled_extensions().ext_line_rasterization {
                    let (stippled_line_enable, line_stipple_factor, line_stipple_pattern) =
                        if let Some(line_stipple) = line_stipple {
                            let (factor, pattern) = match line_stipple {
                                StateMode::Fixed(line_stipple) => {
                                    dynamic_state.insert(DynamicState::LineStipple, false);
                                    (line_stipple.factor, line_stipple.pattern)
                                }
                                StateMode::Dynamic => {
                                    dynamic_state.insert(DynamicState::LineStipple, true);
                                    (1, 0)
                                }
                            };

                            (ash::vk::TRUE, factor, pattern)
                        } else {
                            (ash::vk::FALSE, 1, 0)
                        };

                    rasterization_state.p_next = rasterization_line_state_vk.insert(
                        ash::vk::PipelineRasterizationLineStateCreateInfoEXT {
                            line_rasterization_mode: line_rasterization_mode.into(),
                            stippled_line_enable,
                            line_stipple_factor,
                            line_stipple_pattern,
                            ..Default::default()
                        },
                    ) as *const _ as *const _;
                }
            }

            // Discard rectangle state
            if device.enabled_extensions().ext_discard_rectangles {
                let &DiscardRectangleState {
                    mode,
                    ref rectangles,
                } = discard_rectangle_state;

                let discard_rectangle_count = match *rectangles {
                    PartialStateMode::Fixed(ref rectangles) => {
                        dynamic_state.insert(DynamicState::DiscardRectangle, false);
                        discard_rectangles.extend(rectangles.iter().map(|&rect| rect.into()));
                        discard_rectangles.len() as u32
                    }
                    PartialStateMode::Dynamic(count) => {
                        dynamic_state.insert(DynamicState::DiscardRectangle, true);
                        count
                    }
                };

                let _ = discard_rectangle_state_vk.insert(
                    ash::vk::PipelineDiscardRectangleStateCreateInfoEXT {
                        flags: ash::vk::PipelineDiscardRectangleStateCreateFlagsEXT::empty(),
                        discard_rectangle_mode: mode.into(),
                        discard_rectangle_count,
                        p_discard_rectangles: discard_rectangles.as_ptr(),
                        ..Default::default()
                    },
                );
            }
        }

        // Tessellation state
        if has.tessellation_state {
            let &TessellationState {
                patch_control_points,
            } = tessellation_state;

            let patch_control_points = match patch_control_points {
                StateMode::Fixed(patch_control_points) => {
                    dynamic_state.insert(DynamicState::PatchControlPoints, false);
                    patch_control_points
                }
                StateMode::Dynamic => {
                    dynamic_state.insert(DynamicState::PatchControlPoints, true);
                    Default::default()
                }
            };

            let _ = tessellation_state_vk.insert(ash::vk::PipelineTessellationStateCreateInfo {
                flags: ash::vk::PipelineTessellationStateCreateFlags::empty(),
                patch_control_points,
                ..Default::default()
            });
        }

        // Viewport state
        if has.viewport_state {
            let (viewport_count, scissor_count) = match *viewport_state {
                ViewportState::Fixed { ref data } => {
                    let count = data.len() as u32;
                    viewports_vk.extend(data.iter().map(|e| e.0.clone().into()));
                    dynamic_state.insert(DynamicState::Viewport, false);
                    dynamic_state.insert(DynamicState::ViewportWithCount, false);

                    scissors_vk.extend(data.iter().map(|e| e.1.into()));
                    dynamic_state.insert(DynamicState::Scissor, false);
                    dynamic_state.insert(DynamicState::ScissorWithCount, false);

                    (count, count)
                }
                ViewportState::FixedViewport {
                    ref viewports,
                    scissor_count_dynamic,
                } => {
                    let viewport_count = viewports.len() as u32;
                    viewports_vk.extend(viewports.iter().map(|e| e.clone().into()));
                    dynamic_state.insert(DynamicState::Viewport, false);
                    dynamic_state.insert(DynamicState::ViewportWithCount, false);

                    let scissor_count = if scissor_count_dynamic {
                        dynamic_state.insert(DynamicState::Scissor, false);
                        dynamic_state.insert(DynamicState::ScissorWithCount, true);
                        0
                    } else {
                        dynamic_state.insert(DynamicState::Scissor, true);
                        dynamic_state.insert(DynamicState::ScissorWithCount, false);
                        viewport_count
                    };

                    (viewport_count, scissor_count)
                }
                ViewportState::FixedScissor {
                    ref scissors,
                    viewport_count_dynamic,
                } => {
                    let scissor_count = scissors.len() as u32;
                    scissors_vk.extend(scissors.iter().map(|e| (*e).into()));
                    dynamic_state.insert(DynamicState::Scissor, false);
                    dynamic_state.insert(DynamicState::ScissorWithCount, false);

                    let viewport_count = if viewport_count_dynamic {
                        dynamic_state.insert(DynamicState::Viewport, false);
                        dynamic_state.insert(DynamicState::ViewportWithCount, true);
                        0
                    } else {
                        dynamic_state.insert(DynamicState::Viewport, true);
                        dynamic_state.insert(DynamicState::ViewportWithCount, false);
                        scissor_count
                    };

                    (viewport_count, scissor_count)
                }
                ViewportState::Dynamic {
                    count,
                    viewport_count_dynamic,
                    scissor_count_dynamic,
                } => {
                    let viewport_count = if viewport_count_dynamic {
                        dynamic_state.insert(DynamicState::Viewport, false);
                        dynamic_state.insert(DynamicState::ViewportWithCount, true);
                        0
                    } else {
                        dynamic_state.insert(DynamicState::Viewport, true);
                        dynamic_state.insert(DynamicState::ViewportWithCount, false);
                        count
                    };
                    let scissor_count = if scissor_count_dynamic {
                        dynamic_state.insert(DynamicState::Scissor, false);
                        dynamic_state.insert(DynamicState::ScissorWithCount, true);
                        0
                    } else {
                        dynamic_state.insert(DynamicState::Scissor, true);
                        dynamic_state.insert(DynamicState::ScissorWithCount, false);
                        count
                    };

                    (viewport_count, scissor_count)
                }
            };

            let _ = viewport_state_vk.insert(ash::vk::PipelineViewportStateCreateInfo {
                flags: ash::vk::PipelineViewportStateCreateFlags::empty(),
                viewport_count,
                p_viewports: if viewports_vk.is_empty() {
                    ptr::null()
                } else {
                    viewports_vk.as_ptr()
                }, // validation layer crashes if you just pass the pointer
                scissor_count,
                p_scissors: if scissors_vk.is_empty() {
                    ptr::null()
                } else {
                    scissors_vk.as_ptr()
                }, // validation layer crashes if you just pass the pointer
                ..Default::default()
            });
        }

        /*
            Fragment shader state
        */

        let mut fragment_shader_specialization_vk = None;
        let mut depth_stencil_state_vk = None;

        if has.fragment_shader_state {
            // Fragment shader
            if let Some((entry_point, specialization_data)) = fragment_shader {
                let specialization_map_entries = Fss::descriptors();
                let specialization_data = slice::from_raw_parts(
                    specialization_data as *const _ as *const u8,
                    size_of_val(specialization_data),
                );

                let specialization_info_vk =
                    fragment_shader_specialization_vk.insert(ash::vk::SpecializationInfo {
                        map_entry_count: specialization_map_entries.len() as u32,
                        p_map_entries: specialization_map_entries.as_ptr() as *const _,
                        data_size: specialization_data.len(),
                        p_data: specialization_data.as_ptr() as *const _,
                    });

                for (loc, reqs) in entry_point.descriptor_requirements() {
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

                stages.insert(ShaderStage::Fragment, ());
                stages_vk.push(ash::vk::PipelineShaderStageCreateInfo {
                    flags: ash::vk::PipelineShaderStageCreateFlags::empty(),
                    stage: ash::vk::ShaderStageFlags::FRAGMENT,
                    module: entry_point.module().internal_object(),
                    p_name: entry_point.name().as_ptr(),
                    p_specialization_info: specialization_info_vk as *const _,
                    ..Default::default()
                });
            }
        }

        // Depth/stencil state
        if has.depth_stencil_state {
            let &DepthStencilState {
                ref depth,
                ref depth_bounds,
                ref stencil,
            } = depth_stencil_state;

            let (depth_test_enable, depth_write_enable, depth_compare_op) =
                if let Some(depth_state) = depth {
                    let &DepthState {
                        enable_dynamic,
                        write_enable,
                        compare_op,
                    } = depth_state;

                    if enable_dynamic {
                        dynamic_state.insert(DynamicState::DepthTestEnable, true);
                    } else {
                        dynamic_state.insert(DynamicState::DepthTestEnable, false);
                    }

                    let write_enable = match write_enable {
                        StateMode::Fixed(write_enable) => {
                            dynamic_state.insert(DynamicState::DepthWriteEnable, false);
                            write_enable as ash::vk::Bool32
                        }
                        StateMode::Dynamic => {
                            dynamic_state.insert(DynamicState::DepthWriteEnable, true);
                            ash::vk::TRUE
                        }
                    };

                    let compare_op = match compare_op {
                        StateMode::Fixed(compare_op) => {
                            dynamic_state.insert(DynamicState::DepthCompareOp, false);
                            compare_op.into()
                        }
                        StateMode::Dynamic => {
                            dynamic_state.insert(DynamicState::DepthCompareOp, true);
                            ash::vk::CompareOp::ALWAYS
                        }
                    };

                    (ash::vk::TRUE, write_enable, compare_op)
                } else {
                    (ash::vk::FALSE, ash::vk::FALSE, ash::vk::CompareOp::ALWAYS)
                };

            let (depth_bounds_test_enable, min_depth_bounds, max_depth_bounds) =
                if let Some(depth_bounds_state) = depth_bounds {
                    let &DepthBoundsState {
                        enable_dynamic,
                        ref bounds,
                    } = depth_bounds_state;

                    if enable_dynamic {
                        dynamic_state.insert(DynamicState::DepthBoundsTestEnable, true);
                    } else {
                        dynamic_state.insert(DynamicState::DepthBoundsTestEnable, false);
                    }

                    let (min_bounds, max_bounds) = match bounds.clone() {
                        StateMode::Fixed(bounds) => {
                            dynamic_state.insert(DynamicState::DepthBounds, false);
                            bounds.into_inner()
                        }
                        StateMode::Dynamic => {
                            dynamic_state.insert(DynamicState::DepthBounds, true);
                            (0.0, 1.0)
                        }
                    };

                    (ash::vk::TRUE, min_bounds, max_bounds)
                } else {
                    (ash::vk::FALSE, 0.0, 1.0)
                };

            let (stencil_test_enable, front, back) = if let Some(stencil_state) = stencil {
                let &StencilState {
                    enable_dynamic,
                    ref front,
                    ref back,
                } = stencil_state;

                if enable_dynamic {
                    dynamic_state.insert(DynamicState::StencilTestEnable, true);
                } else {
                    dynamic_state.insert(DynamicState::StencilTestEnable, false);
                }

                match (front.ops, back.ops) {
                    (StateMode::Fixed(_), StateMode::Fixed(_)) => {
                        dynamic_state.insert(DynamicState::StencilOp, false);
                    }
                    (StateMode::Dynamic, StateMode::Dynamic) => {
                        dynamic_state.insert(DynamicState::StencilOp, true);
                    }
                    _ => unreachable!(),
                };

                match (front.compare_mask, back.compare_mask) {
                    (StateMode::Fixed(_), StateMode::Fixed(_)) => {
                        dynamic_state.insert(DynamicState::StencilCompareMask, false);
                    }
                    (StateMode::Dynamic, StateMode::Dynamic) => {
                        dynamic_state.insert(DynamicState::StencilCompareMask, true);
                    }
                    _ => unreachable!(),
                };

                match (front.write_mask, back.write_mask) {
                    (StateMode::Fixed(_), StateMode::Fixed(_)) => {
                        dynamic_state.insert(DynamicState::StencilWriteMask, false);
                    }
                    (StateMode::Dynamic, StateMode::Dynamic) => {
                        dynamic_state.insert(DynamicState::StencilWriteMask, true);
                    }
                    _ => unreachable!(),
                };

                match (front.reference, back.reference) {
                    (StateMode::Fixed(_), StateMode::Fixed(_)) => {
                        dynamic_state.insert(DynamicState::StencilReference, false);
                    }
                    (StateMode::Dynamic, StateMode::Dynamic) => {
                        dynamic_state.insert(DynamicState::StencilReference, true);
                    }
                    _ => unreachable!(),
                };

                let [front, back] = [front, back].map(|stencil_op_state| {
                    let &StencilOpState {
                        ops,
                        compare_mask,
                        write_mask,
                        reference,
                    } = stencil_op_state;

                    let ops = match ops {
                        StateMode::Fixed(x) => x,
                        StateMode::Dynamic => Default::default(),
                    };
                    let compare_mask = match compare_mask {
                        StateMode::Fixed(x) => x,
                        StateMode::Dynamic => Default::default(),
                    };
                    let write_mask = match write_mask {
                        StateMode::Fixed(x) => x,
                        StateMode::Dynamic => Default::default(),
                    };
                    let reference = match reference {
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

            let _ = depth_stencil_state_vk.insert(ash::vk::PipelineDepthStencilStateCreateInfo {
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
            });
        }

        /*
            Fragment output state
        */

        let mut multisample_state_vk = None;
        let mut color_blend_attachments_vk: SmallVec<[_; 4]> = SmallVec::new();
        let mut color_write_enables_vk: SmallVec<[_; 4]> = SmallVec::new();
        let mut color_write_vk = None;
        let mut color_blend_state_vk = None;

        if has.fragment_output_state {
            // Multisample state
            {
                let &MultisampleState {
                    rasterization_samples,
                    sample_shading,
                    ref sample_mask,
                    alpha_to_coverage_enable,
                    alpha_to_one_enable,
                } = multisample_state;

                let (sample_shading_enable, min_sample_shading) =
                    if let Some(min_sample_shading) = sample_shading {
                        (ash::vk::TRUE, min_sample_shading)
                    } else {
                        (ash::vk::FALSE, 0.0)
                    };

                let _ = multisample_state_vk.insert(ash::vk::PipelineMultisampleStateCreateInfo {
                    flags: ash::vk::PipelineMultisampleStateCreateFlags::empty(),
                    rasterization_samples: rasterization_samples.into(),
                    sample_shading_enable,
                    min_sample_shading,
                    p_sample_mask: sample_mask as _,
                    alpha_to_coverage_enable: alpha_to_coverage_enable as ash::vk::Bool32,
                    alpha_to_one_enable: alpha_to_one_enable as ash::vk::Bool32,
                    ..Default::default()
                });
            }
        }

        // Color blend state
        if has.color_blend_state {
            let &ColorBlendState {
                logic_op,
                ref attachments,
                blend_constants,
            } = color_blend_state;

            color_blend_attachments_vk.extend(attachments.iter().map(
                |color_blend_attachment_state| {
                    let &ColorBlendAttachmentState {
                        blend,
                        color_write_mask,
                        color_write_enable: _,
                    } = color_blend_attachment_state;

                    let blend = if let Some(blend) = blend {
                        blend.into()
                    } else {
                        Default::default()
                    };

                    ash::vk::PipelineColorBlendAttachmentState {
                        color_write_mask: color_write_mask.into(),
                        ..blend
                    }
                },
            ));

            let (logic_op_enable, logic_op) = if let Some(logic_op) = logic_op {
                let logic_op = match logic_op {
                    StateMode::Fixed(logic_op) => {
                        dynamic_state.insert(DynamicState::LogicOp, false);
                        logic_op.into()
                    }
                    StateMode::Dynamic => {
                        dynamic_state.insert(DynamicState::LogicOp, true);
                        Default::default()
                    }
                };

                (ash::vk::TRUE, logic_op)
            } else {
                (ash::vk::FALSE, Default::default())
            };

            let blend_constants = match blend_constants {
                StateMode::Fixed(blend_constants) => {
                    dynamic_state.insert(DynamicState::BlendConstants, false);
                    blend_constants
                }
                StateMode::Dynamic => {
                    dynamic_state.insert(DynamicState::BlendConstants, true);
                    Default::default()
                }
            };

            let mut color_blend_state_vk =
                color_blend_state_vk.insert(ash::vk::PipelineColorBlendStateCreateInfo {
                    flags: ash::vk::PipelineColorBlendStateCreateFlags::empty(),
                    logic_op_enable,
                    logic_op,
                    attachment_count: color_blend_attachments_vk.len() as u32,
                    p_attachments: color_blend_attachments_vk.as_ptr(),
                    blend_constants,
                    ..Default::default()
                });

            if device.enabled_extensions().ext_color_write_enable {
                color_write_enables_vk.extend(attachments.iter().map(
                    |color_blend_attachment_state| {
                        let &ColorBlendAttachmentState {
                            blend: _,
                            color_write_mask: _,
                            color_write_enable,
                        } = color_blend_attachment_state;

                        match color_write_enable {
                            StateMode::Fixed(enable) => {
                                dynamic_state.insert(DynamicState::ColorWriteEnable, false);
                                enable as ash::vk::Bool32
                            }
                            StateMode::Dynamic => {
                                dynamic_state.insert(DynamicState::ColorWriteEnable, true);
                                ash::vk::TRUE
                            }
                        }
                    },
                ));

                color_blend_state_vk.p_next =
                    color_write_vk.insert(ash::vk::PipelineColorWriteCreateInfoEXT {
                        attachment_count: color_write_enables_vk.len() as u32,
                        p_color_write_enables: color_write_enables_vk.as_ptr(),
                        ..Default::default()
                    }) as *const _ as *const _;
            }
        }

        /*
            Dynamic state
        */

        let mut dynamic_state_list: SmallVec<[_; 4]> = SmallVec::new();
        let mut dynamic_state_vk = None;

        {
            dynamic_state_list.extend(
                dynamic_state
                    .iter()
                    .filter(|(_, d)| **d)
                    .map(|(&state, _)| state.into()),
            );

            if !dynamic_state_list.is_empty() {
                let _ = dynamic_state_vk.insert(ash::vk::PipelineDynamicStateCreateInfo {
                    flags: ash::vk::PipelineDynamicStateCreateFlags::empty(),
                    dynamic_state_count: dynamic_state_list.len() as u32,
                    p_dynamic_states: dynamic_state_list.as_ptr(),
                    ..Default::default()
                });
            }
        }

        /*
            Create
        */

        let mut create_info = ash::vk::GraphicsPipelineCreateInfo {
            flags: ash::vk::PipelineCreateFlags::empty(), // TODO: some flags are available but none are critical
            stage_count: stages_vk.len() as u32,
            p_stages: stages_vk.as_ptr(),
            p_vertex_input_state: vertex_input_state_vk
                .as_ref()
                .map(|p| p as *const _)
                .unwrap_or(ptr::null()),
            p_input_assembly_state: input_assembly_state_vk
                .as_ref()
                .map(|p| p as *const _)
                .unwrap_or(ptr::null()),
            p_tessellation_state: tessellation_state_vk
                .as_ref()
                .map(|p| p as *const _)
                .unwrap_or(ptr::null()),
            p_viewport_state: viewport_state_vk
                .as_ref()
                .map(|p| p as *const _)
                .unwrap_or(ptr::null()),
            p_rasterization_state: rasterization_state_vk
                .as_ref()
                .map(|p| p as *const _)
                .unwrap_or(ptr::null()),
            p_multisample_state: multisample_state_vk
                .as_ref()
                .map(|p| p as *const _)
                .unwrap_or(ptr::null()),
            p_depth_stencil_state: depth_stencil_state_vk
                .as_ref()
                .map(|p| p as *const _)
                .unwrap_or(ptr::null()),
            p_color_blend_state: color_blend_state_vk
                .as_ref()
                .map(|p| p as *const _)
                .unwrap_or(ptr::null()),
            p_dynamic_state: dynamic_state_vk
                .as_ref()
                .map(|s| s as *const _)
                .unwrap_or(ptr::null()),
            layout: pipeline_layout.internal_object(),
            render_pass: render_pass_vk,
            subpass: subpass_vk,
            base_pipeline_handle: ash::vk::Pipeline::null(), // TODO:
            base_pipeline_index: -1,                         // TODO:
            ..Default::default()
        };

        if let Some(info) = discard_rectangle_state_vk.as_mut() {
            info.p_next = create_info.p_next;
            create_info.p_next = info as *const _ as *const _;
        }

        if let Some(info) = rendering_create_info_vk.as_mut() {
            info.p_next = create_info.p_next;
            create_info.p_next = info as *const _ as *const _;
        }

        let cache_handle = match cache.as_ref() {
            Some(cache) => cache.internal_object(),
            None => ash::vk::PipelineCache::null(),
        };

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_graphics_pipelines)(
                device.internal_object(),
                cache_handle,
                1,
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        // Some drivers return `VK_SUCCESS` but provide a null handle if they
        // fail to create the pipeline (due to invalid shaders, etc)
        // This check ensures that we don't create an invalid `GraphicsPipeline` instance
        if handle == ash::vk::Pipeline::null() {
            panic!("vkCreateGraphicsPipelines provided a NULL handle");
        }

        Ok((handle, descriptor_requirements, dynamic_state, stages))
    }

    // TODO: add build_with_cache method
}

struct ShaderStageInfo<'a> {
    entry_point: &'a EntryPoint<'a>,
    specialization_map_entries: &'a [SpecializationMapEntry],
    _specialization_data: &'a [u8],
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
            render_pass: self.render_pass,
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
            render_pass: self.render_pass,
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
            render_pass: self.render_pass,
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
            render_pass: self.render_pass,
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
            render_pass: self.render_pass,
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
    #[deprecated(since = "0.27.0")]
    #[inline]
    pub fn tessellation_shaders_disabled(mut self) -> Self {
        self.tessellation_shaders = None;
        self
    }

    /// Sets the geometry shader stage as disabled. This is the default.
    #[deprecated(since = "0.27.0")]
    #[inline]
    pub fn geometry_shader_disabled(mut self) -> Self {
        self.geometry_shader = None;
        self
    }

    /// Sets the vertex input to a single vertex buffer.
    ///
    /// You will most likely need to explicitly specify the template parameter to the type of a
    /// vertex.
    #[deprecated(since = "0.27.0", note = "Use `vertex_input_state` instead")]
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
    #[deprecated(since = "0.27.0", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn primitive_restart(mut self, enabled: bool) -> Self {
        self.input_assembly_state.primitive_restart_enable = StateMode::Fixed(enabled);
        self
    }

    /// Sets the topology of the primitives that are expected by the pipeline.
    #[deprecated(since = "0.27.0", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn primitive_topology(mut self, topology: PrimitiveTopology) -> Self {
        self.input_assembly_state.topology = PartialStateMode::Fixed(topology);
        self
    }

    /// Sets the topology of the primitives to a list of points.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::PointList)`.
    #[deprecated(since = "0.27.0", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn point_list(self) -> Self {
        self.primitive_topology(PrimitiveTopology::PointList)
    }

    /// Sets the topology of the primitives to a list of lines.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::LineList)`.
    #[deprecated(since = "0.27.0", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn line_list(self) -> Self {
        self.primitive_topology(PrimitiveTopology::LineList)
    }

    /// Sets the topology of the primitives to a line strip.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::LineStrip)`.
    #[deprecated(since = "0.27.0", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn line_strip(self) -> Self {
        self.primitive_topology(PrimitiveTopology::LineStrip)
    }

    /// Sets the topology of the primitives to a list of triangles. Note that this is the default.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::TriangleList)`.
    #[deprecated(since = "0.27.0", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn triangle_list(self) -> Self {
        self.primitive_topology(PrimitiveTopology::TriangleList)
    }

    /// Sets the topology of the primitives to a triangle strip.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::TriangleStrip)`.
    #[deprecated(since = "0.27.0", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn triangle_strip(self) -> Self {
        self.primitive_topology(PrimitiveTopology::TriangleStrip)
    }

    /// Sets the topology of the primitives to a fan of triangles.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::TriangleFan)`.
    #[deprecated(since = "0.27.0", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn triangle_fan(self) -> Self {
        self.primitive_topology(PrimitiveTopology::TriangleFan)
    }

    /// Sets the topology of the primitives to a list of lines with adjacency information.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::LineListWithAdjacency)`.
    #[deprecated(since = "0.27.0", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn line_list_with_adjacency(self) -> Self {
        self.primitive_topology(PrimitiveTopology::LineListWithAdjacency)
    }

    /// Sets the topology of the primitives to a line strip with adjacency information.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::LineStripWithAdjacency)`.
    #[deprecated(since = "0.27.0", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn line_strip_with_adjacency(self) -> Self {
        self.primitive_topology(PrimitiveTopology::LineStripWithAdjacency)
    }

    /// Sets the topology of the primitives to a list of triangles with adjacency information.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::TriangleListWithAdjacency)`.
    #[deprecated(since = "0.27.0", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn triangle_list_with_adjacency(self) -> Self {
        self.primitive_topology(PrimitiveTopology::TriangleListWithAdjacency)
    }

    /// Sets the topology of the primitives to a triangle strip with adjacency information`
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::TriangleStripWithAdjacency)`.
    #[deprecated(since = "0.27.0", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn triangle_strip_with_adjacency(self) -> Self {
        self.primitive_topology(PrimitiveTopology::TriangleStripWithAdjacency)
    }

    /// Sets the topology of the primitives to a list of patches. Can only be used and must be used
    /// with a tessellation shader.
    ///
    /// > **Note**: This is equivalent to
    /// > `self.primitive_topology(PrimitiveTopology::PatchList)`.
    #[deprecated(since = "0.27.0", note = "Use `input_assembly_state` instead")]
    #[inline]
    pub fn patch_list(self) -> Self {
        self.primitive_topology(PrimitiveTopology::PatchList)
    }

    /// Sets the viewports to some value, and the scissor boxes to boxes that always cover the
    /// whole viewport.
    #[deprecated(since = "0.27.0", note = "Use `viewport_state` instead")]
    #[inline]
    pub fn viewports<I>(self, viewports: I) -> Self
    where
        I: IntoIterator<Item = Viewport>,
    {
        self.viewports_scissors(viewports.into_iter().map(|v| (v, Scissor::irrelevant())))
    }

    /// Sets the characteristics of viewports and scissor boxes in advance.
    #[deprecated(since = "0.27.0", note = "Use `viewport_state` instead")]
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
    #[deprecated(since = "0.27.0", note = "Use `viewport_state` instead")]
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
    #[deprecated(since = "0.27.0", note = "Use `viewport_state` instead")]
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
    #[deprecated(since = "0.27.0", note = "Use `viewport_state` instead")]
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
    #[deprecated(since = "0.27.0", note = "Use `viewport_state` instead")]
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
    #[deprecated(since = "0.27.0", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn depth_clamp(mut self, clamp: bool) -> Self {
        self.rasterization_state.depth_clamp_enable = clamp;
        self
    }

    /// Sets the front-facing faces to counter-clockwise faces. This is the default.
    ///
    /// Triangles whose vertices are oriented counter-clockwise on the screen will be considered
    /// as facing their front. Otherwise they will be considered as facing their back.
    #[deprecated(since = "0.27.0", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn front_face_counter_clockwise(mut self) -> Self {
        self.rasterization_state.front_face = StateMode::Fixed(FrontFace::CounterClockwise);
        self
    }

    /// Sets the front-facing faces to clockwise faces.
    ///
    /// Triangles whose vertices are oriented clockwise on the screen will be considered
    /// as facing their front. Otherwise they will be considered as facing their back.
    #[deprecated(since = "0.27.0", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn front_face_clockwise(mut self) -> Self {
        self.rasterization_state.front_face = StateMode::Fixed(FrontFace::Clockwise);
        self
    }

    /// Sets backface culling as disabled. This is the default.
    #[deprecated(since = "0.27.0", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn cull_mode_disabled(mut self) -> Self {
        self.rasterization_state.cull_mode = StateMode::Fixed(CullMode::None);
        self
    }

    /// Sets backface culling to front faces. The front faces (as chosen with the `front_face_*`
    /// methods) will be discarded by the GPU when drawing.
    #[deprecated(since = "0.27.0", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn cull_mode_front(mut self) -> Self {
        self.rasterization_state.cull_mode = StateMode::Fixed(CullMode::Front);
        self
    }

    /// Sets backface culling to back faces. Faces that are not facing the front (as chosen with
    /// the `front_face_*` methods) will be discarded by the GPU when drawing.
    #[deprecated(since = "0.27.0", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn cull_mode_back(mut self) -> Self {
        self.rasterization_state.cull_mode = StateMode::Fixed(CullMode::Back);
        self
    }

    /// Sets backface culling to both front and back faces. All the faces will be discarded.
    ///
    /// > **Note**: This option exists for the sake of completeness. It has no known practical
    /// > usage.
    #[deprecated(since = "0.27.0", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn cull_mode_front_and_back(mut self) -> Self {
        self.rasterization_state.cull_mode = StateMode::Fixed(CullMode::FrontAndBack);
        self
    }

    /// Sets the polygon mode to "fill". This is the default.
    #[deprecated(since = "0.27.0", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn polygon_mode_fill(mut self) -> Self {
        self.rasterization_state.polygon_mode = PolygonMode::Fill;
        self
    }

    /// Sets the polygon mode to "line". Triangles will each be turned into three lines.
    #[deprecated(since = "0.27.0", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn polygon_mode_line(mut self) -> Self {
        self.rasterization_state.polygon_mode = PolygonMode::Line;
        self
    }

    /// Sets the polygon mode to "point". Triangles and lines will each be turned into three points.
    #[deprecated(since = "0.27.0", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn polygon_mode_point(mut self) -> Self {
        self.rasterization_state.polygon_mode = PolygonMode::Point;
        self
    }

    /// Sets the width of the lines, if the GPU needs to draw lines. The default is `1.0`.
    #[deprecated(since = "0.27.0", note = "Use `rasterization_state` instead")]
    #[inline]
    pub fn line_width(mut self, value: f32) -> Self {
        self.rasterization_state.line_width = StateMode::Fixed(value);
        self
    }

    /// Sets the width of the lines as dynamic, which means that you will need to set this value
    /// when drawing.
    #[deprecated(since = "0.27.0", note = "Use `rasterization_state` instead")]
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
    #[deprecated(since = "0.27.0", note = "Use `multisample_state` instead")]
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
    #[deprecated(since = "0.27.0", note = "Use `multisample_state` instead")]
    #[inline]
    pub fn sample_shading_enabled(mut self, min_fract: f32) -> Self {
        assert!((0.0..=1.0).contains(&min_fract));
        self.multisample_state.sample_shading = Some(min_fract);
        self
    }

    // TODO: doc
    #[deprecated(since = "0.27.0", note = "Use `multisample_state` instead")]
    pub fn alpha_to_coverage_disabled(mut self) -> Self {
        self.multisample_state.alpha_to_coverage_enable = false;
        self
    }

    // TODO: doc
    #[deprecated(since = "0.27.0", note = "Use `multisample_state` instead")]
    pub fn alpha_to_coverage_enabled(mut self) -> Self {
        self.multisample_state.alpha_to_coverage_enable = true;
        self
    }

    /// Disables alpha-to-one.
    ///
    /// Alpha-to-one is disabled by default.
    #[deprecated(since = "0.27.0", note = "Use `multisample_state` instead")]
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
    #[deprecated(since = "0.27.0", note = "Use `multisample_state` instead")]
    #[inline]
    pub fn alpha_to_one_enabled(mut self) -> Self {
        self.multisample_state.alpha_to_one_enable = true;
        self
    }

    /// Sets the depth/stencil state.
    #[deprecated(since = "0.27.0", note = "Use `depth_stencil_state` instead")]
    #[inline]
    pub fn depth_stencil(self, depth_stencil_state: DepthStencilState) -> Self {
        self.depth_stencil_state(depth_stencil_state)
    }

    /// Sets the depth/stencil tests as disabled.
    ///
    /// > **Note**: This is a shortcut for all the other `depth_*` and `depth_stencil_*` methods
    /// > of the builder.
    #[deprecated(since = "0.27.0", note = "Use `depth_stencil_state` instead")]
    #[inline]
    pub fn depth_stencil_disabled(mut self) -> Self {
        self.depth_stencil_state = DepthStencilState::disabled();
        self
    }

    /// Sets the depth/stencil tests as a simple depth test and no stencil test.
    ///
    /// > **Note**: This is a shortcut for setting the depth test to `Less`, the depth write Into
    /// > ` true` and disable the stencil test.
    #[deprecated(since = "0.27.0", note = "Use `depth_stencil_state` instead")]
    #[inline]
    pub fn depth_stencil_simple_depth(mut self) -> Self {
        self.depth_stencil_state = DepthStencilState::simple_depth_test();
        self
    }

    /// Sets whether the depth buffer will be written.
    #[deprecated(since = "0.27.0", note = "Use `depth_stencil_state` instead")]
    #[inline]
    pub fn depth_write(mut self, write: bool) -> Self {
        let depth_state = self
            .depth_stencil_state
            .depth
            .get_or_insert(Default::default());
        depth_state.write_enable = StateMode::Fixed(write);
        self
    }

    #[deprecated(since = "0.27.0", note = "Use `color_blend_state` instead")]
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

    #[deprecated(since = "0.27.0", note = "Use `color_blend_state` instead")]
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
    #[deprecated(since = "0.27.0", note = "Use `color_blend_state` instead")]
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

    #[deprecated(since = "0.27.0", note = "Use `color_blend_state` instead")]
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

    #[deprecated(since = "0.27.0", note = "Use `color_blend_state` instead")]
    #[inline]
    pub fn blend_logic_op(mut self, logic_op: LogicOp) -> Self {
        self.color_blend_state.logic_op = Some(StateMode::Fixed(logic_op));
        self
    }

    /// Sets the logic operation as disabled. This is the default.
    #[deprecated(since = "0.27.0", note = "Use `color_blend_state` instead")]
    #[inline]
    pub fn blend_logic_op_disabled(mut self) -> Self {
        self.color_blend_state.logic_op = None;
        self
    }

    /// Sets the blend constant. The default is `[0.0, 0.0, 0.0, 0.0]`.
    ///
    /// The blend constant is used for some blending calculations. It is irrelevant otherwise.
    #[deprecated(since = "0.27.0", note = "Use `color_blend_state` instead")]
    #[inline]
    pub fn blend_constants(mut self, constants: [f32; 4]) -> Self {
        self.color_blend_state.blend_constants = StateMode::Fixed(constants);
        self
    }

    /// Sets the blend constant value as dynamic. Its value will need to be set before drawing.
    ///
    /// The blend constant is used for some blending calculations. It is irrelevant otherwise.
    #[deprecated(since = "0.27.0", note = "Use `color_blend_state` instead")]
    #[inline]
    pub fn blend_constants_dynamic(mut self) -> Self {
        self.color_blend_state.blend_constants = StateMode::Dynamic;
        self
    }

    /// Sets the render pass subpass to use.
    #[inline]
    pub fn render_pass(self, render_pass: impl Into<PipelineRenderPassType>) -> Self {
        GraphicsPipelineBuilder {
            render_pass: Some(render_pass.into()),
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
            render_pass: self.render_pass.clone(),
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
