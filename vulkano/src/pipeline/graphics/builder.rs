// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{
    color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState},
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
        VertexDefinition, VertexInputAttributeDescription, VertexInputBindingDescription,
        VertexInputState,
    },
    viewport::ViewportState,
    GraphicsPipeline, GraphicsPipelineCreationError,
};
use crate::{
    descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutCreateInfo},
    device::{Device, DeviceOwned},
    format::{FormatFeatures, NumericType},
    image::ImageAspects,
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
        DescriptorBindingRequirements, FragmentShaderExecution, FragmentTestsStages,
        PipelineShaderStageCreateInfo, ShaderExecution, ShaderScalarType, ShaderStage,
        ShaderStages,
    },
    DeviceSize, RequiresOneOf, Version, VulkanError, VulkanObject,
};
use ahash::HashMap;
use smallvec::SmallVec;
use std::{collections::hash_map::Entry, ffi::CString, mem::MaybeUninit, ptr, sync::Arc};

/// Prototype for a `GraphicsPipeline`.
#[derive(Clone, Debug)]
pub struct GraphicsPipelineBuilder<Vdef> {
    render_pass: Option<PipelineRenderPassType>,
    cache: Option<Arc<PipelineCache>>,
    stages: SmallVec<[PipelineShaderStageCreateInfo; 5]>,

    vertex_input_state: Option<Vdef>,
    input_assembly_state: Option<InputAssemblyState>,
    tessellation_state: Option<TessellationState>,
    viewport_state: Option<ViewportState>,
    discard_rectangle_state: Option<DiscardRectangleState>,
    rasterization_state: Option<RasterizationState>,
    multisample_state: Option<MultisampleState>,
    depth_stencil_state: Option<DepthStencilState>,
    color_blend_state: Option<ColorBlendState>,
}

impl GraphicsPipelineBuilder<VertexInputState> {
    /// Builds a new empty builder.
    pub(super) fn new() -> Self {
        GraphicsPipelineBuilder {
            render_pass: None,
            cache: None,
            stages: SmallVec::new(),

            vertex_input_state: None,
            input_assembly_state: None,
            tessellation_state: None,
            viewport_state: None,
            discard_rectangle_state: None,
            rasterization_state: None,
            multisample_state: None,
            depth_stencil_state: None,
            color_blend_state: None,
        }
    }
}

impl<Vdef> GraphicsPipelineBuilder<Vdef>
where
    Vdef: VertexDefinition,
{
    /// Builds the graphics pipeline, using an inferred a pipeline layout.
    #[inline]
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
            // Produce `DescriptorBindingRequirements` for each binding, by iterating over all
            // shaders and adding the requirements of each.
            let mut descriptor_binding_requirements: HashMap<
                (u32, u32),
                DescriptorBindingRequirements,
            > = HashMap::default();

            for (&loc, reqs) in self
                .stages
                .iter()
                .flat_map(|stage| &stage.entry_point.info().descriptor_binding_requirements)
            {
                match descriptor_binding_requirements.entry(loc) {
                    Entry::Occupied(entry) => {
                        // Previous shaders already added requirements, so we merge requirements of
                        // the current shader into the requirements of the previous one.
                        entry.into_mut().merge(reqs).expect("Could not produce an intersection of the shader descriptor requirements");
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
                descriptor_binding_requirements
                    .iter()
                    .map(|(&loc, reqs)| (loc, reqs)),
            );
            func(&mut set_layout_create_infos);

            // We want to union each push constant range into a set of ranges that do not have intersecting stage flags.
            // e.g. The range [0, 16) is either made available to Vertex | Fragment or we only make [0, 16) available to
            // Vertex and a subrange available to Fragment, like [0, 8)
            let mut range_map = HashMap::default();
            for info in self.stages.iter() {
                if let Some(range) = &info.entry_point.info().push_constant_requirements {
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
    #[inline]
    pub fn with_pipeline_layout(
        self,
        device: Arc<Device>,
        pipeline_layout: Arc<PipelineLayout>,
    ) -> Result<Arc<GraphicsPipeline>, GraphicsPipelineCreationError> {
        let vertex_input_state = self
            .stages
            .iter()
            .find(|stage| matches!(stage.entry_point.info().execution, ShaderExecution::Vertex))
            .zip(self.vertex_input_state.as_ref())
            .map(|(stage, vertex_input_state)| {
                vertex_input_state.definition(&stage.entry_point.info().input_interface)
            })
            .transpose()?;

        self.validate_create(&device, &pipeline_layout, vertex_input_state.as_ref())?;

        let (handle, descriptor_requirements, dynamic_state, shaders, fragment_tests_stages) =
            unsafe { self.record_create(&device, &pipeline_layout, vertex_input_state.as_ref())? };

        let Self {
            mut render_pass,
            cache: _,
            stages: _,

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
            id: GraphicsPipeline::next_id(),
            layout: pipeline_layout,
            render_pass: render_pass.take().expect("Missing render pass"),

            shaders,
            descriptor_binding_requirements: descriptor_requirements,
            num_used_descriptor_sets,
            fragment_tests_stages,

            vertex_input_state: vertex_input_state.unwrap(), // Can be None if there's a mesh shader, but we don't support that yet
            input_assembly_state: input_assembly_state.unwrap(), // Can be None if there's a mesh shader, but we don't support that yet
            tessellation_state,
            viewport_state,
            discard_rectangle_state,
            rasterization_state: rasterization_state.unwrap(), // Can be None for pipeline libraries, but we don't support that yet
            multisample_state,
            depth_stencil_state,
            color_blend_state,
            dynamic_state,
        }))
    }

    fn validate_create(
        &self,
        device: &Device,
        pipeline_layout: &PipelineLayout,
        vertex_input_state: Option<&VertexInputState>,
    ) -> Result<(), GraphicsPipelineCreationError> {
        let physical_device = device.physical_device();
        let properties = physical_device.properties();

        let Self {
            render_pass,
            cache: _,
            stages,

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

        /*
            Gather shader stages
        */

        let mut stages_present = ShaderStages::empty();
        let mut vertex_stage = None;
        let mut tessellation_control_stage = None;
        let mut tessellation_evaluation_stage = None;
        let mut geometry_stage = None;
        let mut fragment_stage = None;

        for (stage_index, stage) in stages.iter().enumerate() {
            let entry_point_info = stage.entry_point.info();
            let stage_enum = ShaderStage::from(&entry_point_info.execution);
            let stage_flag = ShaderStages::from(stage_enum);

            // VUID-VkGraphicsPipelineCreateInfo-stage-06897
            if stages_present.intersects(stage_flag) {
                return Err(GraphicsPipelineCreationError::ShaderStageDuplicate {
                    stage_index,
                    stage: stage_enum,
                });
            }

            // VUID-VkGraphicsPipelineCreateInfo-pStages-02095
            // VUID-VkGraphicsPipelineCreateInfo-pStages-06896
            // VUID-VkPipelineShaderStageCreateInfo-stage-parameter
            let stage_slot = match stage_enum {
                ShaderStage::Vertex => &mut vertex_stage,
                ShaderStage::TessellationControl => &mut tessellation_control_stage,
                ShaderStage::TessellationEvaluation => &mut tessellation_evaluation_stage,
                ShaderStage::Geometry => &mut geometry_stage,
                ShaderStage::Fragment => &mut fragment_stage,
                _ => {
                    return Err(GraphicsPipelineCreationError::ShaderStageInvalid {
                        stage_index,
                        stage: stage_enum,
                    })
                }
            };

            *stage_slot = Some(stage);
            stages_present |= stage_flag;
        }

        /*
            Validate needed/unused state
        */

        let need_pre_rasterization_shader_state = true;

        // Check this first because everything else depends on it.
        // VUID?
        match (
            rasterization_state.is_some(),
            need_pre_rasterization_shader_state,
        ) {
            (true, false) => {
                return Err(GraphicsPipelineCreationError::StateUnused {
                    state: "rasterization_state",
                })
            }
            (false, true) => {
                return Err(GraphicsPipelineCreationError::StateMissing {
                    state: "rasterization_state",
                })
            }
            _ => (),
        }

        let need_vertex_input_state = need_pre_rasterization_shader_state
            && stages
                .iter()
                .any(|stage| matches!(stage.entry_point.info().execution, ShaderExecution::Vertex));
        let need_fragment_shader_state = need_pre_rasterization_shader_state
            && rasterization_state
                .as_ref()
                .unwrap()
                .rasterizer_discard_enable
                != StateMode::Fixed(true);
        let need_fragment_output_state = need_pre_rasterization_shader_state
            && rasterization_state
                .as_ref()
                .unwrap()
                .rasterizer_discard_enable
                != StateMode::Fixed(true);

        // VUID-VkGraphicsPipelineCreateInfo-stage-02096
        // VUID-VkGraphicsPipelineCreateInfo-pStages-06895
        match (vertex_stage.is_some(), need_pre_rasterization_shader_state) {
            (true, false) => {
                return Err(GraphicsPipelineCreationError::ShaderStageUnused {
                    stage: ShaderStage::Vertex,
                })
            }
            (false, true) => return Err(GraphicsPipelineCreationError::VertexShaderStageMissing),
            _ => (),
        }

        // VUID-VkGraphicsPipelineCreateInfo-pStages-06895
        match (
            tessellation_control_stage.is_some(),
            need_pre_rasterization_shader_state,
        ) {
            (true, false) => {
                return Err(GraphicsPipelineCreationError::ShaderStageUnused {
                    stage: ShaderStage::Vertex,
                })
            }
            (false, true) => (),
            _ => (),
        }

        // VUID-VkGraphicsPipelineCreateInfo-pStages-06895
        match (
            tessellation_evaluation_stage.is_some(),
            need_pre_rasterization_shader_state,
        ) {
            (true, false) => {
                return Err(GraphicsPipelineCreationError::ShaderStageUnused {
                    stage: ShaderStage::Vertex,
                })
            }
            (false, true) => (),
            _ => (),
        }

        // VUID-VkGraphicsPipelineCreateInfo-pStages-00729
        // VUID-VkGraphicsPipelineCreateInfo-pStages-00730
        if stages_present
            .intersects(ShaderStages::TESSELLATION_CONTROL | ShaderStages::TESSELLATION_EVALUATION)
            && !stages_present.contains(
                ShaderStages::TESSELLATION_CONTROL | ShaderStages::TESSELLATION_EVALUATION,
            )
        {
            return Err(GraphicsPipelineCreationError::OtherTessellationShaderStageMissing);
        }

        // VUID-VkGraphicsPipelineCreateInfo-pStages-06895
        match (
            geometry_stage.is_some(),
            need_pre_rasterization_shader_state,
        ) {
            (true, false) => {
                return Err(GraphicsPipelineCreationError::ShaderStageUnused {
                    stage: ShaderStage::Vertex,
                })
            }
            (false, true) => (),
            _ => (),
        }

        // VUID-VkGraphicsPipelineCreateInfo-pStages-06894
        match (fragment_stage.is_some(), need_fragment_shader_state) {
            (true, false) => {
                return Err(GraphicsPipelineCreationError::ShaderStageUnused {
                    stage: ShaderStage::Vertex,
                })
            }
            (false, true) => (),
            _ => (),
        }

        // VUID-VkGraphicsPipelineCreateInfo-pVertexInputState-04910
        match (vertex_input_state.is_some(), need_vertex_input_state) {
            (true, false) => {
                return Err(GraphicsPipelineCreationError::StateUnused {
                    state: "vertex_input_state",
                })
            }
            (false, true) => {
                return Err(GraphicsPipelineCreationError::StateMissing {
                    state: "vertex_input_state",
                })
            }
            _ => (),
        }

        // VUID-VkGraphicsPipelineCreateInfo-pStages-02098
        match (input_assembly_state.is_some(), need_vertex_input_state) {
            (true, false) => {
                return Err(GraphicsPipelineCreationError::StateUnused {
                    state: "input_assembly_state",
                })
            }
            (false, true) => {
                return Err(GraphicsPipelineCreationError::StateMissing {
                    state: "input_assembly_state",
                })
            }
            _ => (),
        }

        // VUID-VkGraphicsPipelineCreateInfo-pStages-00731
        match (
            tessellation_state.is_some(),
            need_pre_rasterization_shader_state
                && stages_present.contains(
                    ShaderStages::TESSELLATION_CONTROL | ShaderStages::TESSELLATION_EVALUATION,
                ),
        ) {
            (true, false) => {
                return Err(GraphicsPipelineCreationError::StateUnused {
                    state: "tessellation_state",
                })
            }
            (false, true) => {
                return Err(GraphicsPipelineCreationError::StateMissing {
                    state: "tessellation_state",
                })
            }
            _ => (),
        }

        // VUID-VkGraphicsPipelineCreateInfo-rasterizerDiscardEnable-00750
        // VUID-VkGraphicsPipelineCreateInfo-pViewportState-04892
        match (
            viewport_state.is_some(),
            need_pre_rasterization_shader_state
                && rasterization_state
                    .as_ref()
                    .unwrap()
                    .rasterizer_discard_enable
                    != StateMode::Fixed(true),
        ) {
            (true, false) => {
                return Err(GraphicsPipelineCreationError::StateUnused {
                    state: "viewport_state",
                })
            }
            (false, true) => {
                return Err(GraphicsPipelineCreationError::StateMissing {
                    state: "viewport_state",
                })
            }
            _ => (),
        }

        // VUID-VkGraphicsPipelineCreateInfo-rasterizerDiscardEnable-00751
        match (multisample_state.is_some(), need_fragment_output_state) {
            (true, false) => {
                return Err(GraphicsPipelineCreationError::StateUnused {
                    state: "multisample_state",
                })
            }
            (false, true) => {
                return Err(GraphicsPipelineCreationError::StateMissing {
                    state: "multisample_state",
                })
            }
            _ => (),
        }

        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06590
        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06043
        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06053
        match (
            depth_stencil_state.is_some(),
            !need_fragment_output_state
                || match render_pass {
                    Some(PipelineRenderPassType::BeginRenderPass(subpass)) => {
                        subpass.subpass_desc().depth_stencil_attachment.is_some()
                    }
                    Some(PipelineRenderPassType::BeginRendering(rendering_info)) => {
                        rendering_info.depth_attachment_format.is_some()
                            || rendering_info.stencil_attachment_format.is_some()
                    }
                    None => false,
                },
        ) {
            (true, false) => {
                return Err(GraphicsPipelineCreationError::StateUnused {
                    state: "depth_stencil_state",
                })
            }
            (false, true) => {
                return Err(GraphicsPipelineCreationError::StateMissing {
                    state: "depth_stencil_state",
                })
            }
            _ => (),
        }

        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06044
        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06054
        match (
            color_blend_state.is_some(),
            need_fragment_output_state
                && match render_pass {
                    Some(PipelineRenderPassType::BeginRenderPass(subpass)) => {
                        !subpass.subpass_desc().color_attachments.is_empty()
                    }
                    Some(PipelineRenderPassType::BeginRendering(rendering_info)) => {
                        !rendering_info.color_attachment_formats.is_empty()
                    }
                    None => false,
                },
        ) {
            (true, false) => {
                return Err(GraphicsPipelineCreationError::StateUnused {
                    state: "color_blend_state",
                })
            }
            (false, true) => {
                return Err(GraphicsPipelineCreationError::StateMissing {
                    state: "color_blend_state",
                })
            }
            _ => (),
        }

        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06575
        match (
            render_pass.is_some(),
            need_pre_rasterization_shader_state
                || need_fragment_shader_state
                || need_fragment_output_state,
        ) {
            (true, false) => {
                return Err(GraphicsPipelineCreationError::StateUnused {
                    state: "render_pass",
                })
            }
            (false, true) => {
                return Err(GraphicsPipelineCreationError::StateMissing {
                    state: "render_pass",
                })
            }
            _ => (),
        }

        // VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-04058 (partly)
        match (
            discard_rectangle_state.is_some(),
            need_pre_rasterization_shader_state,
        ) {
            (true, false) => {
                return Err(GraphicsPipelineCreationError::StateUnused {
                    state: "discard_rectangle_state",
                })
            }
            (false, true) => (),
            _ => (),
        }

        /*
            Validate shader stages individually
        */

        for (stage_index, stage) in stages.iter().enumerate() {
            let &PipelineShaderStageCreateInfo {
                flags,
                ref entry_point,
                ref specialization_info,
                _ne: _,
            } = stage;

            // VUID-VkPipelineShaderStageCreateInfo-flags-parameter
            flags.validate_device(device)?;

            let entry_point_info = entry_point.info();
            let stage_enum = ShaderStage::from(&entry_point_info.execution);

            // VUID-VkPipelineShaderStageCreateInfo-pName-00707
            // Guaranteed by definition of `EntryPoint`.

            // TODO:
            // VUID-VkPipelineShaderStageCreateInfo-maxClipDistances-00708
            // VUID-VkPipelineShaderStageCreateInfo-maxCullDistances-00709
            // VUID-VkPipelineShaderStageCreateInfo-maxCombinedClipAndCullDistances-00710
            // VUID-VkPipelineShaderStageCreateInfo-maxSampleMaskWords-00711

            match stage_enum {
                ShaderStage::Vertex => {
                    vertex_stage = Some(stage);

                    // VUID-VkPipelineShaderStageCreateInfo-stage-00712
                    // TODO:
                }
                ShaderStage::TessellationControl | ShaderStage::TessellationEvaluation => {
                    // VUID-VkPipelineShaderStageCreateInfo-stage-00705
                    if !device.enabled_features().tessellation_shader {
                        return Err(GraphicsPipelineCreationError::RequirementNotMet {
                            required_for: "`stages` contains a `TessellationControl` or \
                                `TessellationEvaluation` shader stage",
                            requires_one_of: RequiresOneOf {
                                features: &["tessellation_shader"],
                                ..Default::default()
                            },
                        });
                    }

                    // VUID-VkPipelineShaderStageCreateInfo-stage-00713
                    // TODO:
                }
                ShaderStage::Geometry => {
                    // VUID-VkPipelineShaderStageCreateInfo-stage-00704
                    if !device.enabled_features().geometry_shader {
                        return Err(GraphicsPipelineCreationError::RequirementNotMet {
                            required_for: "`stages` contains a `Geometry` shader stage",
                            requires_one_of: RequiresOneOf {
                                features: &["geometry_shader"],
                                ..Default::default()
                            },
                        });
                    }

                    // VUID-VkPipelineShaderStageCreateInfo-stage-00714
                    // VUID-VkPipelineShaderStageCreateInfo-stage-00715
                    // TODO:
                }
                ShaderStage::Fragment => {
                    fragment_stage = Some(stage);

                    // VUID-VkPipelineShaderStageCreateInfo-stage-00718
                    // VUID-VkPipelineShaderStageCreateInfo-stage-06685
                    // VUID-VkPipelineShaderStageCreateInfo-stage-06686
                    // TODO:
                }
                _ => unreachable!(),
            }

            // TODO:
            // VUID-VkPipelineShaderStageCreateInfo-stage-02596
            // VUID-VkPipelineShaderStageCreateInfo-stage-02597

            for (&constant_id, provided_value) in specialization_info {
                // Per `VkSpecializationMapEntry` spec:
                // "If a constantID value is not a specialization constant ID used in the shader,
                // that map entry does not affect the behavior of the pipeline."
                // We *may* want to be stricter than this for the sake of catching user errors?
                if let Some(default_value) =
                    entry_point_info.specialization_constants.get(&constant_id)
                {
                    // VUID-VkSpecializationMapEntry-constantID-00776
                    // Check for equal types rather than only equal size.
                    if !provided_value.eq_type(default_value) {
                        return Err(
                            GraphicsPipelineCreationError::ShaderSpecializationConstantTypeMismatch {
                                stage_index,
                                constant_id,
                                default_value: *default_value,
                                provided_value: *provided_value,
                            },
                        );
                    }
                }
            }

            // VUID-VkGraphicsPipelineCreateInfo-layout-00756
            pipeline_layout.ensure_compatible_with_shader(
                entry_point_info
                    .descriptor_binding_requirements
                    .iter()
                    .map(|(k, v)| (*k, v)),
                entry_point_info.push_constant_requirements.as_ref(),
            )?;
        }

        let ordered_stages: SmallVec<[_; 5]> = [
            vertex_stage,
            tessellation_control_stage,
            tessellation_evaluation_stage,
            geometry_stage,
            fragment_stage,
        ]
        .into_iter()
        .flatten()
        .collect();

        // VUID-VkGraphicsPipelineCreateInfo-pStages-00742
        // VUID-VkGraphicsPipelineCreateInfo-None-04889

        // TODO: this check is too strict; the output only has to be a superset, any variables
        // not used in the input of the next shader are just ignored.
        for (output, input) in ordered_stages.iter().zip(ordered_stages.iter().skip(1)) {
            if let Err(err) = (input.entry_point.info().input_interface)
                .matches(&output.entry_point.info().output_interface)
            {
                return Err(GraphicsPipelineCreationError::ShaderStagesMismatch(err));
            }
        }

        // VUID-VkGraphicsPipelineCreateInfo-layout-01688
        // Checked at pipeline layout creation time.

        /*
            Validate states individually
        */

        if let Some(vertex_input_state) = vertex_input_state {
            let VertexInputState {
                bindings,
                attributes,
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

                // VUID-VkVertexInputBindingDescription-stride-04456
                if device.enabled_extensions().khr_portability_subset
                    && (stride == 0
                        || stride
                            % properties
                                .min_vertex_input_binding_stride_alignment
                                .unwrap()
                            != 0)
                {
                    return Err(GraphicsPipelineCreationError::MinVertexInputBindingStrideAlignmentExceeded {
                            binding,
                            max: properties.min_vertex_input_binding_stride_alignment.unwrap(),
                            obtained: binding,
                        });
                }

                match input_rate {
                    VertexInputRate::Instance { divisor } if divisor != 1 => {
                        // VUID-VkVertexInputBindingDivisorDescriptionEXT-vertexAttributeInstanceRateDivisor-02229
                        if !device
                            .enabled_features()
                            .vertex_attribute_instance_rate_divisor
                        {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for: "`vertex_input_state.bindings` has an element \
                                    where `input_rate` is `VertexInputRate::Instance`, where \
                                    `divisor` is not `1`",
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
                                required_for: "`vertex_input_state.bindings` has an element \
                                    where `input_rate` is `VertexInputRate::Instance`, where \
                                    `divisor` is `0`",
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
            // Ensured by HashMap with the exception of formats exceeding a single location.
            // When a format exceeds a single location the location following it (e.g.
            // R64B64G64_SFLOAT) needs to be unassigned.
            let unassigned_locations = attributes
                .iter()
                .filter(|&(_, attribute_desc)| attribute_desc.format.block_size().unwrap() > 16)
                .map(|(location, _)| location + 1);
            for location in unassigned_locations {
                if !attributes.get(&location).is_none() {
                    return Err(GraphicsPipelineCreationError::VertexInputAttributeInvalidAssignedLocation {
                        location,
                    });
                }
            }

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
                let binding_desc = bindings.get(&binding).ok_or(
                    GraphicsPipelineCreationError::VertexInputAttributeInvalidBinding {
                        location,
                        binding,
                    },
                )?;

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
                if !format_features.intersects(FormatFeatures::VERTEX_BUFFER) {
                    return Err(
                        GraphicsPipelineCreationError::VertexInputAttributeUnsupportedFormat {
                            location,
                            format,
                        },
                    );
                }

                // VUID-VkVertexInputAttributeDescription-vertexAttributeAccessBeyondStride-04457
                if device.enabled_extensions().khr_portability_subset
                    && !device
                        .enabled_features()
                        .vertex_attribute_access_beyond_stride
                    && offset as DeviceSize + format.block_size().unwrap()
                        > binding_desc.stride as DeviceSize
                {
                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                        required_for: "this device is a portability subset device, and \
                            `vertex_input_state.attributes` has an element where \
                            `offset + format.block_size()` is greater than the `stride` of \
                            `binding`",
                        requires_one_of: RequiresOneOf {
                            features: &["vertex_attribute_access_beyond_stride"],
                            ..Default::default()
                        },
                    });
                }
            }
        }

        if let Some(input_assembly_state) = input_assembly_state {
            let &InputAssemblyState {
                topology,
                primitive_restart_enable,
            } = input_assembly_state;

            match topology {
                PartialStateMode::Fixed(topology) => {
                    // VUID-VkPipelineInputAssemblyStateCreateInfo-topology-parameter
                    topology.validate_device(device)?;

                    match topology {
                        PrimitiveTopology::TriangleFan => {
                            // VUID-VkPipelineInputAssemblyStateCreateInfo-triangleFans-04452
                            if device.enabled_extensions().khr_portability_subset
                                && !device.enabled_features().triangle_fans
                            {
                                return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                    required_for: "this device is a portability subset \
                                            device, and `input_assembly_state.topology` is \
                                            `StateMode::Fixed(PrimitiveTopology::TriangleFan)`",
                                    requires_one_of: RequiresOneOf {
                                        features: &["triangle_fans"],
                                        ..Default::default()
                                    },
                                });
                            }
                        }
                        PrimitiveTopology::LineListWithAdjacency
                        | PrimitiveTopology::LineStripWithAdjacency
                        | PrimitiveTopology::TriangleListWithAdjacency
                        | PrimitiveTopology::TriangleStripWithAdjacency => {
                            // VUID-VkPipelineInputAssemblyStateCreateInfo-topology-00429
                            if !device.enabled_features().geometry_shader {
                                return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                    required_for: "`input_assembly_state.topology` is \
                                            `StateMode::Fixed(PrimitiveTopology::*WithAdjacency)`",
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
                                    required_for: "`input_assembly_state.topology` is \
                                            `StateMode::Fixed(PrimitiveTopology::PatchList)`",
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
                            required_for: "`input_assembly_state.topology` is \
                                    `PartialStateMode::Dynamic`",
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
                                        required_for:
                                            "`input_assembly_state.primitive_restart_enable` \
                                                is `StateMode::Fixed(true)` and \
                                                `input_assembly_state.topology` is \
                                                `StateMode::Fixed(PrimitiveTopology::*List)`",
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
                                        required_for:
                                            "`input_assembly_state.primitive_restart_enable` \
                                                is `StateMode::Fixed(true)` and \
                                                `input_assembly_state.topology` is \
                                                `StateMode::Fixed(PrimitiveTopology::PatchList)`",
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
                            required_for: "`input_assembly_state.primitive_restart_enable` is \
                                    `StateMode::Dynamic`",
                            requires_one_of: RequiresOneOf {
                                api_version: Some(Version::V1_3),
                                features: &["extended_dynamic_state2"],
                                ..Default::default()
                            },
                        });
                    }
                }
            }
        }

        if let Some(tessellation_state) = tessellation_state {
            let &TessellationState {
                patch_control_points,
            } = tessellation_state;

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
                            required_for: "`tessellation_state.patch_control_points` is \
                                `StateMode::Dynamic`",
                            requires_one_of: RequiresOneOf {
                                features: &["extended_dynamic_state2_patch_control_points"],
                                ..Default::default()
                            },
                        });
                    }
                }
            };
        }

        if let Some(viewport_state) = viewport_state {
            let (viewport_count, scissor_count) = match viewport_state {
                ViewportState::Fixed { data } => {
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
                    viewports,
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
                    let scissor_count = if *scissor_count_dynamic {
                        if !(device.api_version() >= Version::V1_3
                            || device.enabled_features().extended_dynamic_state)
                        {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for: "`viewport_state` is \
                                    `ViewportState::FixedViewport`, where `scissor_count_dynamic` \
                                    is set",
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
                    scissors,
                    viewport_count_dynamic,
                } => {
                    let scissor_count = scissors.len() as u32;

                    // VUID-VkPipelineViewportStateCreateInfo-viewportCount-04135
                    assert!(scissor_count != 0); // TODO: return error?

                    // VUID-VkPipelineViewportStateCreateInfo-viewportCount-04135
                    let viewport_count = if *viewport_count_dynamic {
                        if !(device.api_version() >= Version::V1_3
                            || device.enabled_features().extended_dynamic_state)
                        {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for: "`viewport_state` is \
                                    `ViewportState::FixedScissor`, where `viewport_count_dynamic` \
                                    is set",
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
                &ViewportState::Dynamic {
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
                                required_for: "`viewport_state` is \
                                    `ViewportState::Dynamic`, where `viewport_count_dynamic` \
                                    is set",
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
                                required_for: "`viewport_state` is \
                                    `ViewportState::Dynamic`, where `scissor_count_dynamic` \
                                    is set",
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
                    required_for: "`viewport_state` has a fixed viewport/scissor count that is \
                        greater than `1`",
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

        if let Some(rasterization_state) = rasterization_state {
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

            match rasterizer_discard_enable {
                StateMode::Dynamic => {
                    // VUID?
                    if !(device.api_version() >= Version::V1_3
                        || device.enabled_features().extended_dynamic_state2)
                    {
                        return Err(GraphicsPipelineCreationError::RequirementNotMet {
                            required_for: "`rasterization_state.rasterizer_discard_enable` is \
                                    `StateMode::Dynamic`",
                            requires_one_of: RequiresOneOf {
                                api_version: Some(Version::V1_3),
                                features: &["extended_dynamic_state"],
                                ..Default::default()
                            },
                        });
                    }
                }
                StateMode::Fixed(false) => {
                    // VUID-VkPipelineRasterizationStateCreateInfo-pointPolygons-04458
                    if device.enabled_extensions().khr_portability_subset
                        && !device.enabled_features().point_polygons
                        && polygon_mode == PolygonMode::Point
                    {
                        return Err(GraphicsPipelineCreationError::RequirementNotMet {
                            required_for: "this device is a portability subset device, \
                                    `rasterization_state.rasterizer_discard_enable` is \
                                    `StateMode::Fixed(false)` and \
                                    `rasterization_state.polygon_mode` is `PolygonMode::Point`",
                            requires_one_of: RequiresOneOf {
                                features: &["point_polygons"],
                                ..Default::default()
                            },
                        });
                    }
                }
                _ => (),
            }

            // VUID-VkPipelineRasterizationStateCreateInfo-polygonMode-01507
            if polygon_mode != PolygonMode::Fill && !device.enabled_features().fill_mode_non_solid {
                return Err(GraphicsPipelineCreationError::RequirementNotMet {
                    required_for: "`rasterization_state.polygon_mode` is not \
                            `PolygonMode::Fill`",
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
                            required_for: "`rasterization_state.cull_mode` is \
                                    `StateMode::Dynamic`",
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
                            required_for: "`rasterization_state.front_face` is \
                                    `StateMode::Dynamic`",
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
                        required_for: "`rasterization_state.depth_bias` is \
                                `Some(depth_bias_state)`, where `depth_bias_state.enable_dynamic` \
                                is set",
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
                        required_for: "`rasterization_state.depth_bias` is \
                            `Some(depth_bias_state)`, where `depth_bias_state.bias` is \
                            `StateMode::Fixed(bias)`, where `bias.clamp` is not `0.0`",
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
                    required_for: "`rasterization_state.line_width` is \
                            `StateMode::Fixed(line_width)`, where `line_width` is not `1.0`",
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
                                required_for: "`rasterization_state.line_rasterization_mode` \
                                        is `LineRasterizationMode::Rectangular`",
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
                                required_for: "`rasterization_state.line_rasterization_mode` \
                                        is `LineRasterizationMode::Bresenham`",
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
                                required_for: "`rasterization_state.line_rasterization_mode` \
                                        is `LineRasterizationMode::RectangularSmooth`",
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
                                    required_for: "`rasterization_state.line_stipple` is \
                                            `Some` and \
                                            `rasterization_state.line_rasterization_mode` \
                                            is `LineRasterizationMode::Default`",
                                    requires_one_of: RequiresOneOf {
                                        features: &["stippled_rectangular_lines"],
                                        ..Default::default()
                                    },
                                });
                            }

                            // VUID-VkPipelineRasterizationLineStateCreateInfoEXT-stippledLineEnable-02774
                            if !properties.strict_lines {
                                return Err(GraphicsPipelineCreationError::StrictLinesNotSupported);
                            }
                        }
                        LineRasterizationMode::Rectangular => {
                            // VUID-VkPipelineRasterizationLineStateCreateInfoEXT-stippledLineEnable-02771
                            if !device.enabled_features().stippled_rectangular_lines {
                                return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                    required_for: "`rasterization_state.line_stipple` is \
                                            `Some` and \
                                            `rasterization_state.line_rasterization_mode` \
                                            is `LineRasterizationMode::Rectangular`",
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
                                    required_for: "`rasterization_state.line_stipple` is \
                                            `Some` and \
                                            `rasterization_state.line_rasterization_mode` \
                                            is `LineRasterizationMode::Bresenham`",
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
                                    required_for: "`rasterization_state.line_stipple` is \
                                            `Some` and \
                                            `rasterization_state.line_rasterization_mode` \
                                            is `LineRasterizationMode::RectangularSmooth`",
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
                        required_for: "`rasterization_state.line_rasterization_mode` is not \
                                `LineRasterizationMode::Default`",
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

            // TODO:
            // VUID-VkGraphicsPipelineCreateInfo-pStages-00740
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06049
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06050
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06059
        }

        if let Some(multisample_state) = multisample_state {
            let &MultisampleState {
                rasterization_samples,
                sample_shading,
                sample_mask: _,
                alpha_to_coverage_enable: _,
                alpha_to_one_enable,
            } = multisample_state;

            // VUID-VkPipelineMultisampleStateCreateInfo-rasterizationSamples-parameter
            rasterization_samples.validate_device(device)?;

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

        if let Some(depth_stencil_state) = depth_stencil_state {
            let DepthStencilState {
                depth,
                depth_bounds,
                stencil,
            } = depth_stencil_state;

            if let Some(depth_state) = depth {
                let &DepthState {
                    enable_dynamic,
                    write_enable,
                    compare_op,
                } = depth_state;

                // VUID?
                if enable_dynamic
                    && !(device.api_version() >= Version::V1_3
                        || device.enabled_features().extended_dynamic_state)
                {
                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                        required_for: "`depth_stencil_state.depth` is `Some(depth_state)`, where \
                             `depth_state.enable_dynamic` is set",
                        requires_one_of: RequiresOneOf {
                            api_version: Some(Version::V1_3),
                            features: &["extended_dynamic_state"],
                            ..Default::default()
                        },
                    });
                }

                match write_enable {
                    StateMode::Fixed(_) => (),
                    StateMode::Dynamic => {
                        // VUID?
                        if !(device.api_version() >= Version::V1_3
                            || device.enabled_features().extended_dynamic_state)
                        {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for: "`depth_stencil_state.depth` is \
                                    `Some(depth_state)`, where `depth_state.write_enable` is \
                                    `StateMode::Dynamic`",
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
                                required_for: "`depth_stencil_state.depth` is \
                                    `Some(depth_state)`, where `depth_state.compare_op` is \
                                        `StateMode::Dynamic`",
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
                let DepthBoundsState {
                    enable_dynamic,
                    bounds,
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
                if *enable_dynamic
                    && !(device.api_version() >= Version::V1_3
                        || device.enabled_features().extended_dynamic_state)
                {
                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                        required_for: "`depth_stencil_state.depth_bounds` is \
                            `Some(depth_bounds_state)`, where `depth_bounds_state.enable_dynamic` \
                            is set",
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
                            required_for: "`depth_stencil_state.depth_bounds` is \
                                `Some(depth_bounds_state)`, where `depth_bounds_state.bounds` is \
                                not between `0.0` and `1.0` inclusive",
                            requires_one_of: RequiresOneOf {
                                device_extensions: &["ext_depth_range_unrestricted"],
                                ..Default::default()
                            },
                        });
                    }
                }
            }

            if let Some(stencil_state) = stencil {
                let StencilState {
                    enable_dynamic,
                    front,
                    back,
                } = stencil_state;

                // VUID?
                if *enable_dynamic
                    && !(device.api_version() >= Version::V1_3
                        || device.enabled_features().extended_dynamic_state)
                {
                    return Err(GraphicsPipelineCreationError::RequirementNotMet {
                        required_for: "`depth_stencil_state.stencil` is `Some(stencil_state)`, \
                            where `stencil_state.enable_dynamic` is set",
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
                                required_for: "`depth_stencil_state.stencil` is \
                                    `Some(stencil_state)`, where `stencil_state.front.ops` and \
                                    `stencil_state.back.ops` are `StateMode::Dynamic`",
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

        if let Some(color_blend_state) = color_blend_state {
            let ColorBlendState {
                logic_op,
                attachments,
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
                                required_for: "`color_blend_state.logic_op` is \
                                    `Some(StateMode::Dynamic)`",
                                requires_one_of: RequiresOneOf {
                                    features: &["extended_dynamic_state2_logic_op"],
                                    ..Default::default()
                                },
                            });
                        }
                    }
                }
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
                        required_for: "`color_blend_state.attachments` has elements where \
                            `blend` and `color_write_mask` do not match the other elements",
                        requires_one_of: RequiresOneOf {
                            features: &["independent_blend"],
                            ..Default::default()
                        },
                    });
                }
            }

            for state in attachments {
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
                            required_for: "`color_blend_state.attachments` has an element where \
                                `blend` is `Some(blend)`, where `blend.color_source`, \
                                `blend.color_destination`, `blend.alpha_source` or \
                                `blend.alpha_destination` is `BlendFactor::Src1*`",
                            requires_one_of: RequiresOneOf {
                                features: &["dual_src_blend"],
                                ..Default::default()
                            },
                        });
                    }

                    // VUID-VkPipelineColorBlendAttachmentState-constantAlphaColorBlendFactors-04454
                    // VUID-VkPipelineColorBlendAttachmentState-constantAlphaColorBlendFactors-04455
                    if device.enabled_extensions().khr_portability_subset
                        && !device.enabled_features().constant_alpha_color_blend_factors
                        && (matches!(
                            color_source,
                            BlendFactor::ConstantAlpha | BlendFactor::OneMinusConstantAlpha
                        ) || matches!(
                            color_destination,
                            BlendFactor::ConstantAlpha | BlendFactor::OneMinusConstantAlpha
                        ))
                    {
                        return Err(GraphicsPipelineCreationError::RequirementNotMet {
                            required_for: "this device is a portability subset device, and \
                                `color_blend_state.attachments` has an element where `blend` is \
                                `Some(blend)`, where \
                                `blend.color_source` or `blend.color_destination` is \
                                `BlendFactor::ConstantAlpha` or \
                                `BlendFactor::OneMinusConstantAlpha`",
                            requires_one_of: RequiresOneOf {
                                features: &["constant_alpha_color_blend_factors"],
                                ..Default::default()
                            },
                        });
                    }
                }

                match color_write_enable {
                    StateMode::Fixed(enable) => {
                        // VUID-VkPipelineColorWriteCreateInfoEXT-pAttachments-04801
                        if !enable && !device.enabled_features().color_write_enable {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for: "`color_blend_state.attachments` has an element \
                                    where `color_write_enable` is `StateMode::Fixed(false)`",
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
                                required_for: "`color_blend_state.attachments` has an element \
                                    where `color_write_enable` is `StateMode::Dynamic`",
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

        if let Some(render_pass) = render_pass {
            match render_pass {
                PipelineRenderPassType::BeginRenderPass(subpass) => {
                    // VUID-VkGraphicsPipelineCreateInfo-commonparent
                    assert_eq!(device, subpass.render_pass().device().as_ref());

                    if subpass.render_pass().views_used() != 0 {
                        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06047
                        if stages_present.intersects(
                            ShaderStages::TESSELLATION_CONTROL
                                | ShaderStages::TESSELLATION_EVALUATION,
                        ) && !device.enabled_features().multiview_tessellation_shader
                        {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for:
                                    "`tessellation_shaders` are provided and `render_pass` has a \
                                    subpass where `view_mask` is not `0`",
                                requires_one_of: RequiresOneOf {
                                    features: &["multiview_tessellation_shader"],
                                    ..Default::default()
                                },
                            });
                        }

                        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06048
                        if stages_present.intersects(ShaderStages::GEOMETRY)
                            && !device.enabled_features().multiview_geometry_shader
                        {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for:
                                    "`geometry_shader` is provided and `render_pass` has a \
                                    subpass where `view_mask` is not `0`",
                                requires_one_of: RequiresOneOf {
                                    features: &["multiview_geometry_shader"],
                                    ..Default::default()
                                },
                            });
                        }
                    }
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
                            required_for:
                                "`render_pass` is `PipelineRenderPassType::BeginRendering`",
                            requires_one_of: RequiresOneOf {
                                features: &["dynamic_rendering"],
                                ..Default::default()
                            },
                        });
                    }

                    // VUID-VkGraphicsPipelineCreateInfo-multiview-06577
                    if view_mask != 0 && !device.enabled_features().multiview {
                        return Err(GraphicsPipelineCreationError::RequirementNotMet {
                            required_for:
                                "`render_pass` is `PipelineRenderPassType::BeginRendering` \
                            where `view_mask` is not `0`",
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

                    if need_fragment_output_state {
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
                                .intersects(FormatFeatures::COLOR_ATTACHMENT)
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
                                .intersects(FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
                            {
                                return Err(
                                GraphicsPipelineCreationError::DepthAttachmentFormatUsageNotSupported,
                            );
                            }

                            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06587
                            if !format.aspects().intersects(ImageAspects::DEPTH) {
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
                                .intersects(FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
                            {
                                return Err(
                                GraphicsPipelineCreationError::StencilAttachmentFormatUsageNotSupported,
                            );
                            }

                            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06588
                            if !format.aspects().intersects(ImageAspects::STENCIL) {
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

                    if view_mask != 0 {
                        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06057
                        if stages_present.intersects(
                            ShaderStages::TESSELLATION_CONTROL
                                | ShaderStages::TESSELLATION_EVALUATION,
                        ) && !device.enabled_features().multiview_tessellation_shader
                        {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for:
                                    "`tessellation_shaders` are provided and `render_pass` has a \
                                    subpass where `view_mask` is not `0`",
                                requires_one_of: RequiresOneOf {
                                    features: &["multiview_tessellation_shader"],
                                    ..Default::default()
                                },
                            });
                        }

                        // VUID-VkGraphicsPipelineCreateInfo-renderPass-06058
                        if stages_present.intersects(ShaderStages::GEOMETRY)
                            && !device.enabled_features().multiview_geometry_shader
                        {
                            return Err(GraphicsPipelineCreationError::RequirementNotMet {
                                required_for:
                                    "`geometry_shader` is provided and `render_pass` has a \
                                    subpass where `view_mask` is not `0`",
                                requires_one_of: RequiresOneOf {
                                    features: &["multiview_geometry_shader"],
                                    ..Default::default()
                                },
                            });
                        }
                    }
                }
            }
        }

        if let Some(discard_rectangle_state) = discard_rectangle_state {
            if !device.enabled_extensions().ext_discard_rectangles {
                return Err(GraphicsPipelineCreationError::RequirementNotMet {
                    required_for: "`discard_rectangle_state` is `Some`",
                    requires_one_of: RequiresOneOf {
                        device_extensions: &["ext_discard_rectangles"],
                        ..Default::default()
                    },
                });
            }

            let DiscardRectangleState { mode, rectangles } = discard_rectangle_state;

            // VUID-VkPipelineDiscardRectangleStateCreateInfoEXT-discardRectangleMode-parameter
            mode.validate_device(device)?;

            let discard_rectangle_count = match rectangles {
                PartialStateMode::Dynamic(count) => *count,
                PartialStateMode::Fixed(rectangles) => rectangles.len() as u32,
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
        }

        /*
            Checks that rely on multiple pieces of state
        */

        if let (Some(vertex_stage), Some(vertex_input_state)) = (vertex_stage, vertex_input_state) {
            for element in vertex_stage.entry_point.info().input_interface.elements() {
                assert!(!element.ty.is_64bit); // TODO: implement
                let location_range =
                    element.location..element.location + element.ty.num_locations();

                for location in location_range {
                    // VUID-VkGraphicsPipelineCreateInfo-Input-07905
                    let attribute_desc = match vertex_input_state.attributes.get(&location) {
                        Some(attribute_desc) => attribute_desc,
                        None => {
                            return Err(
                                GraphicsPipelineCreationError::VertexInputAttributeMissing {
                                    location,
                                },
                            )
                        }
                    };

                    // TODO: Check component assignments too. Multiple variables can occupy the
                    // same location but in different components.

                    let shader_type = element.ty.base_type;
                    let attribute_type = attribute_desc.format.type_color().unwrap();

                    // VUID?
                    if !matches!(
                        (shader_type, attribute_type),
                        (
                            ShaderScalarType::Float,
                            NumericType::SFLOAT
                                | NumericType::UFLOAT
                                | NumericType::SNORM
                                | NumericType::UNORM
                                | NumericType::SSCALED
                                | NumericType::USCALED
                                | NumericType::SRGB,
                        ) | (ShaderScalarType::Sint, NumericType::SINT)
                            | (ShaderScalarType::Uint, NumericType::UINT)
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
        }

        if let (Some(_), Some(_)) = (tessellation_control_stage, tessellation_evaluation_stage) {
            // FIXME: must check that the control shader and evaluation shader are compatible

            // TODO:
            // VUID-VkGraphicsPipelineCreateInfo-pStages-00732
            // VUID-VkGraphicsPipelineCreateInfo-pStages-00733
            // VUID-VkGraphicsPipelineCreateInfo-pStages-00734
            // VUID-VkGraphicsPipelineCreateInfo-pStages-00735
        }

        if let (Some(_), Some(_)) = (tessellation_evaluation_stage, geometry_stage) {
            // TODO:
            // VUID-VkGraphicsPipelineCreateInfo-pStages-00739
        }

        if let (None, Some(geometry_stage), Some(input_assembly_state)) = (
            tessellation_evaluation_stage,
            geometry_stage,
            input_assembly_state,
        ) {
            let entry_point_info = geometry_stage.entry_point.info();
            let input = match entry_point_info.execution {
                ShaderExecution::Geometry(execution) => execution.input,
                _ => unreachable!(),
            };

            if let PartialStateMode::Fixed(topology) = input_assembly_state.topology {
                // VUID-VkGraphicsPipelineCreateInfo-pStages-00738
                if !input.is_compatible_with(topology) {
                    return Err(GraphicsPipelineCreationError::TopologyNotMatchingGeometryShader);
                }
            }
        }

        if let (Some(fragment_stage), Some(render_pass)) = (fragment_stage, render_pass) {
            let entry_point_info = fragment_stage.entry_point.info();

            // Check that the subpass can accept the output of the fragment shader.
            match render_pass {
                PipelineRenderPassType::BeginRenderPass(subpass) => {
                    if !subpass.is_compatible_with(&entry_point_info.output_interface) {
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
            // VUID-VkGraphicsPipelineCreateInfo-pStages-01565
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06038
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06056
            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06061
        }

        if let (Some(input_assembly_state), Some(_)) = (input_assembly_state, tessellation_state) {
            // VUID-VkGraphicsPipelineCreateInfo-pStages-00736
            if !matches!(
                input_assembly_state.topology,
                PartialStateMode::Dynamic(PrimitiveTopologyClass::Patch)
                    | PartialStateMode::Fixed(PrimitiveTopology::PatchList)
            ) {
                return Err(GraphicsPipelineCreationError::InvalidPrimitiveTopology);
            }
        }

        if let (Some(rasterization_state), Some(depth_stencil_state)) =
            (rasterization_state, depth_stencil_state)
        {
            if let Some(stencil_state) = &depth_stencil_state.stencil {
                if let (StateMode::Fixed(front_reference), StateMode::Fixed(back_reference)) =
                    (stencil_state.front.reference, stencil_state.back.reference)
                {
                    // VUID-VkPipelineDepthStencilStateCreateInfo-separateStencilMaskRef-04453
                    if device.enabled_extensions().khr_portability_subset
                        && !device.enabled_features().separate_stencil_mask_ref
                        && matches!(
                            rasterization_state.cull_mode,
                            StateMode::Fixed(CullMode::None)
                        )
                        && front_reference != back_reference
                    {
                        return Err(GraphicsPipelineCreationError::RequirementNotMet {
                            required_for: "this device is a portability subset device, \
                                    `rasterization_state.cull_mode` is \
                                    `StateMode::Fixed(CullMode::None)`, and \
                                    `depth_stencil_state.stencil` is `Some(stencil_state)`, \
                                    where `stencil_state.front.reference` does not equal \
                                    `stencil_state.back.reference`",
                            requires_one_of: RequiresOneOf {
                                features: &["separate_stencil_mask_ref"],
                                ..Default::default()
                            },
                        });
                    }
                }
            }
        }

        if let (Some(multisample_state), Some(render_pass)) = (multisample_state, render_pass) {
            match render_pass {
                PipelineRenderPassType::BeginRenderPass(subpass) => {
                    if let Some(samples) = subpass.num_samples() {
                        // VUID-VkGraphicsPipelineCreateInfo-subpass-00757
                        if multisample_state.rasterization_samples != samples {
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
        }

        if let (Some(depth_stencil_state), Some(render_pass)) = (depth_stencil_state, render_pass) {
            if let Some(depth_state) = &depth_stencil_state.depth {
                let has_depth_attachment = match render_pass {
                    PipelineRenderPassType::BeginRenderPass(subpass) => subpass.has_depth(),
                    PipelineRenderPassType::BeginRendering(rendering_info) => {
                        rendering_info.depth_attachment_format.is_some()
                    }
                };

                // VUID?
                if !has_depth_attachment {
                    return Err(GraphicsPipelineCreationError::NoDepthAttachment);
                }

                if let StateMode::Fixed(true) = depth_state.write_enable {
                    match render_pass {
                        PipelineRenderPassType::BeginRenderPass(subpass) => {
                            // VUID-VkGraphicsPipelineCreateInfo-renderPass-06039
                            if !subpass.has_writable_depth() {
                                return Err(GraphicsPipelineCreationError::NoDepthAttachment);
                            }
                        }
                        PipelineRenderPassType::BeginRendering(_) => {
                            // No VUID?
                        }
                    }
                }
            }

            if depth_stencil_state.stencil.is_some() {
                let has_stencil_attachment = match render_pass {
                    PipelineRenderPassType::BeginRenderPass(subpass) => subpass.has_stencil(),
                    PipelineRenderPassType::BeginRendering(rendering_info) => {
                        rendering_info.stencil_attachment_format.is_some()
                    }
                };

                if !has_stencil_attachment {
                    return Err(GraphicsPipelineCreationError::NoStencilAttachment);
                }
            }
        }

        if let (Some(color_blend_state), Some(render_pass)) = (color_blend_state, render_pass) {
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
            if color_attachment_count != color_blend_state.attachments.len() {
                return Err(GraphicsPipelineCreationError::MismatchBlendingAttachmentsCount);
            }

            for (attachment_index, state) in color_blend_state.attachments.iter().enumerate() {
                if state.blend.is_some() {
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
                            .intersects(FormatFeatures::COLOR_ATTACHMENT_BLEND)
                    }) {
                        return Err(
                            GraphicsPipelineCreationError::ColorAttachmentFormatBlendNotSupported {
                                attachment_index: attachment_index as u32,
                            },
                        );
                    }
                }
            }
        }

        Ok(())
    }

    unsafe fn record_create(
        &self,
        device: &Device,
        pipeline_layout: &PipelineLayout,
        vertex_input_state: Option<&VertexInputState>,
    ) -> Result<
        (
            ash::vk::Pipeline,
            HashMap<(u32, u32), DescriptorBindingRequirements>,
            HashMap<DynamicState, bool>,
            HashMap<ShaderStage, ()>,
            Option<FragmentTestsStages>,
        ),
        GraphicsPipelineCreationError,
    > {
        let Self {
            render_pass,
            cache,
            stages,

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

        let mut descriptor_binding_requirements: HashMap<
            (u32, u32),
            DescriptorBindingRequirements,
        > = HashMap::default();
        let mut stages2 = HashMap::default();
        let mut fragment_tests_stages = None;
        let mut dynamic_state: HashMap<DynamicState, bool> = HashMap::default();

        struct PerPipelineShaderStageCreateInfo {
            name_vk: CString,
            specialization_info_vk: ash::vk::SpecializationInfo,
            specialization_map_entries_vk: Vec<ash::vk::SpecializationMapEntry>,
            specialization_data_vk: Vec<u8>,
        }

        let (mut stages_vk, mut per_stage_vk): (SmallVec<[_; 5]>, SmallVec<[_; 5]>) = stages
            .iter()
            .map(|stage| {
                let &PipelineShaderStageCreateInfo {
                    flags,
                    ref entry_point,
                    ref specialization_info,
                    _ne: _,
                } = stage;

                let entry_point_info = entry_point.info();
                let stage = ShaderStage::from(&entry_point_info.execution);
                stages2.insert(stage, ());

                if let ShaderExecution::Fragment(
                    FragmentShaderExecution {
                        fragment_tests_stages: s,
                        ..
                    }
                ) = entry_point_info.execution {
                    fragment_tests_stages = Some(s)
                }

                for (&loc, reqs) in &entry_point_info.descriptor_binding_requirements {
                    match descriptor_binding_requirements.entry(loc) {
                        Entry::Occupied(entry) => {
                            entry.into_mut().merge(reqs).expect("Could not produce an intersection of the shader descriptor requirements");
                        }
                        Entry::Vacant(entry) => {
                            entry.insert(reqs.clone());
                        }
                    }
                }

                let mut specialization_data_vk: Vec<u8> = Vec::new();
                let specialization_map_entries_vk: Vec<_> = specialization_info
                    .iter()
                    .map(|(&constant_id, value)| {
                        let data = value.as_bytes();
                        let offset = specialization_data_vk.len() as u32;
                        let size = data.len();
                        specialization_data_vk.extend(data);

                        ash::vk::SpecializationMapEntry {
                            constant_id,
                            offset,
                            size,
                        }
                    })
                    .collect();

                (
                    ash::vk::PipelineShaderStageCreateInfo {
                        flags: flags.into(),
                        stage: stage.into(),
                        module: entry_point.module().handle(),
                        p_name: ptr::null(),
                        p_specialization_info: ptr::null(),
                        ..Default::default()
                    },
                    PerPipelineShaderStageCreateInfo {
                        name_vk: CString::new(entry_point_info.name.as_str()).unwrap(),
                        specialization_info_vk: ash::vk::SpecializationInfo {
                            map_entry_count: specialization_map_entries_vk.len() as u32,
                            p_map_entries: ptr::null(),
                            data_size: specialization_data_vk.len(),
                            p_data: ptr::null(),
                        },
                        specialization_map_entries_vk,
                        specialization_data_vk,
                    },
                )
            })
            .unzip();

        for (
            stage_vk,
            PerPipelineShaderStageCreateInfo {
                name_vk,
                specialization_info_vk,
                specialization_map_entries_vk,
                specialization_data_vk,
            },
        ) in (stages_vk.iter_mut()).zip(per_stage_vk.iter_mut())
        {
            *stage_vk = ash::vk::PipelineShaderStageCreateInfo {
                p_name: name_vk.as_ptr(),
                p_specialization_info: specialization_info_vk,
                ..*stage_vk
            };

            *specialization_info_vk = ash::vk::SpecializationInfo {
                p_map_entries: specialization_map_entries_vk.as_ptr(),
                p_data: specialization_data_vk.as_ptr() as _,
                ..*specialization_info_vk
            };
        }

        let mut vertex_input_state_vk = None;
        let mut vertex_binding_descriptions_vk: SmallVec<[_; 8]> = SmallVec::new();
        let mut vertex_attribute_descriptions_vk: SmallVec<[_; 8]> = SmallVec::new();
        let mut vertex_binding_divisor_state_vk = None;
        let mut vertex_binding_divisor_descriptions_vk: SmallVec<[_; 8]> = SmallVec::new();

        if let Some(vertex_input_state) = vertex_input_state {
            dynamic_state.insert(DynamicState::VertexInput, false);

            let VertexInputState {
                bindings,
                attributes,
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
                    vertex_binding_description_count: vertex_binding_descriptions_vk.len() as u32,
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
                            ash::vk::VertexInputBindingDivisorDescriptionEXT { binding, divisor }
                        }),
                );

                // VUID-VkPipelineVertexInputDivisorStateCreateInfoEXT-vertexBindingDivisorCount-arraylength
                if !vertex_binding_divisor_descriptions_vk.is_empty() {
                    vertex_input_state.p_next = vertex_binding_divisor_state_vk.insert(
                        ash::vk::PipelineVertexInputDivisorStateCreateInfoEXT {
                            vertex_binding_divisor_count: vertex_binding_divisor_descriptions_vk
                                .len()
                                as u32,
                            p_vertex_binding_divisors: vertex_binding_divisor_descriptions_vk
                                .as_ptr(),
                            ..Default::default()
                        },
                    ) as *const _ as *const _;
                }
            }
        }

        let mut input_assembly_state_vk = None;

        if let Some(input_assembly_state) = input_assembly_state {
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

            let _ = input_assembly_state_vk.insert(ash::vk::PipelineInputAssemblyStateCreateInfo {
                flags: ash::vk::PipelineInputAssemblyStateCreateFlags::empty(),
                topology,
                primitive_restart_enable,
                ..Default::default()
            });
        }

        let mut tessellation_state_vk = None;

        if let Some(tessellation_state) = tessellation_state {
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

        let mut viewport_state_vk = None;
        let mut viewports_vk: SmallVec<[_; 2]> = SmallVec::new();
        let mut scissors_vk: SmallVec<[_; 2]> = SmallVec::new();

        if let Some(viewport_state) = viewport_state {
            let (viewport_count, scissor_count) = match viewport_state {
                ViewportState::Fixed { data } => {
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
                    viewports,
                    scissor_count_dynamic,
                } => {
                    let viewport_count = viewports.len() as u32;
                    viewports_vk.extend(viewports.iter().map(|e| e.clone().into()));
                    dynamic_state.insert(DynamicState::Viewport, false);
                    dynamic_state.insert(DynamicState::ViewportWithCount, false);

                    let scissor_count = if *scissor_count_dynamic {
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
                    scissors,
                    viewport_count_dynamic,
                } => {
                    let scissor_count = scissors.len() as u32;
                    scissors_vk.extend(scissors.iter().map(|&e| e.into()));
                    dynamic_state.insert(DynamicState::Scissor, false);
                    dynamic_state.insert(DynamicState::ScissorWithCount, false);

                    let viewport_count = if *viewport_count_dynamic {
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
                &ViewportState::Dynamic {
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

        let mut rasterization_state_vk = None;
        let mut rasterization_line_state_vk = None;

        if let Some(rasterization_state) = rasterization_state {
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

        let mut multisample_state_vk = None;

        if let Some(multisample_state) = multisample_state {
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

        let mut depth_stencil_state_vk = None;

        if let Some(depth_stencil_state) = depth_stencil_state {
            let DepthStencilState {
                depth,
                depth_bounds,
                stencil,
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
                    let DepthBoundsState {
                        enable_dynamic,
                        bounds,
                    } = depth_bounds_state;

                    if *enable_dynamic {
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
                let StencilState {
                    enable_dynamic,
                    front,
                    back,
                } = stencil_state;

                if *enable_dynamic {
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

        let mut color_blend_state_vk = None;
        let mut color_blend_attachments_vk: SmallVec<[_; 4]> = SmallVec::new();
        let mut color_write_vk = None;
        let mut color_write_enables_vk: SmallVec<[_; 4]> = SmallVec::new();

        if let Some(color_blend_state) = color_blend_state {
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

        let render_pass = render_pass.as_ref().unwrap();
        let mut render_pass_vk = ash::vk::RenderPass::null();
        let mut subpass_vk = 0;
        let mut color_attachment_formats_vk: SmallVec<[_; 4]> = SmallVec::new();
        let mut rendering_create_info_vk = None;

        match render_pass {
            PipelineRenderPassType::BeginRenderPass(subpass) => {
                render_pass_vk = subpass.render_pass().handle();
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

        let mut discard_rectangle_state_vk = None;
        let mut discard_rectangles: SmallVec<[_; 2]> = SmallVec::new();

        if let Some(discard_rectangle_state) = discard_rectangle_state {
            let DiscardRectangleState { mode, rectangles } = discard_rectangle_state;

            let discard_rectangle_count = match rectangles {
                PartialStateMode::Fixed(rectangles) => {
                    dynamic_state.insert(DynamicState::DiscardRectangle, false);
                    discard_rectangles.extend(rectangles.iter().map(|&rect| rect.into()));

                    discard_rectangles.len() as u32
                }
                PartialStateMode::Dynamic(count) => {
                    dynamic_state.insert(DynamicState::DiscardRectangle, true);

                    *count
                }
            };

            let _ = discard_rectangle_state_vk.insert(
                ash::vk::PipelineDiscardRectangleStateCreateInfoEXT {
                    flags: ash::vk::PipelineDiscardRectangleStateCreateFlagsEXT::empty(),
                    discard_rectangle_mode: (*mode).into(),
                    discard_rectangle_count,
                    p_discard_rectangles: discard_rectangles.as_ptr(),
                    ..Default::default()
                },
            );
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
            layout: pipeline_layout.handle(),
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
            Some(cache) => cache.handle(),
            None => ash::vk::PipelineCache::null(),
        };

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_graphics_pipelines)(
                device.handle(),
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

        Ok((
            handle,
            descriptor_binding_requirements,
            dynamic_state,
            stages2,
            fragment_tests_stages,
        ))
    }

    // TODO: add build_with_cache method
}

impl<Vdef> GraphicsPipelineBuilder<Vdef> {
    /// Sets the shader stages to use.
    ///
    /// A vertex shader must always be included. Other stages are optional.
    #[inline]
    pub fn stages(
        mut self,
        stages: impl IntoIterator<Item = PipelineShaderStageCreateInfo>,
    ) -> Self {
        self.stages = stages.into_iter().collect();
        self
    }

    /// Sets the vertex input state.
    ///
    /// This state is always used, and must be provided.
    ///
    /// The default value is `None`.
    pub fn vertex_input_state<T>(self, vertex_input_state: T) -> GraphicsPipelineBuilder<T>
    where
        T: VertexDefinition,
    {
        GraphicsPipelineBuilder {
            render_pass: self.render_pass,
            cache: self.cache,
            stages: self.stages,

            vertex_input_state: Some(vertex_input_state),
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
    /// This state is always used, and must be provided.
    ///
    /// The default value is `None`.
    #[inline]
    pub fn input_assembly_state(mut self, input_assembly_state: InputAssemblyState) -> Self {
        self.input_assembly_state = Some(input_assembly_state);
        self
    }

    /// Sets the tessellation state.
    ///
    /// This state is used if `stages` contains tessellation shaders.
    ///
    /// The default value is `None`.
    #[inline]
    pub fn tessellation_state(mut self, tessellation_state: TessellationState) -> Self {
        self.tessellation_state = Some(tessellation_state);
        self
    }

    /// Sets the viewport state.
    ///
    /// This state is used if [rasterizer discarding] is not enabled.
    ///
    /// The default value is `None`.
    ///
    /// [rasterizer discarding]: RasterizationState::rasterizer_discard_enable
    #[inline]
    pub fn viewport_state(mut self, viewport_state: ViewportState) -> Self {
        self.viewport_state = Some(viewport_state);
        self
    }

    /// Sets the rasterization state.
    ///
    /// This state is always used, and must be provided.
    ///
    /// The default value is `None`.
    #[inline]
    pub fn rasterization_state(mut self, rasterization_state: RasterizationState) -> Self {
        self.rasterization_state = Some(rasterization_state);
        self
    }

    /// Sets the multisample state.
    ///
    /// This state is used if [rasterizer discarding] is not enabled.
    ///
    /// The default value is `None`.
    ///
    /// [rasterizer discarding]: RasterizationState::rasterizer_discard_enable
    #[inline]
    pub fn multisample_state(mut self, multisample_state: MultisampleState) -> Self {
        self.multisample_state = Some(multisample_state);
        self
    }

    /// Sets the depth/stencil state.
    ///
    /// This state is used if `render_pass` has depth/stencil attachments, or if
    /// [rasterizer discarding] is enabled.
    ///
    /// The default value is `None`.
    ///
    /// [rasterizer discarding]: RasterizationState::rasterizer_discard_enable
    #[inline]
    pub fn depth_stencil_state(mut self, depth_stencil_state: DepthStencilState) -> Self {
        self.depth_stencil_state = Some(depth_stencil_state);
        self
    }

    /// Sets the color blend state.
    ///
    /// This state is used if `render_pass` has color attachments, and [rasterizer discarding] is
    /// not enabled.
    ///
    /// The default value is `None`.
    ///
    /// [rasterizer discarding]: RasterizationState::rasterizer_discard_enable
    #[inline]
    pub fn color_blend_state(mut self, color_blend_state: ColorBlendState) -> Self {
        self.color_blend_state = Some(color_blend_state);
        self
    }

    /// Sets the render pass subpass to use.
    ///
    /// This state is always used, and must be provided.
    ///
    /// The default value is `None`.
    pub fn render_pass(mut self, render_pass: impl Into<PipelineRenderPassType>) -> Self {
        self.render_pass = Some(render_pass.into());
        self
    }

    /// Sets the discard rectangle state.
    ///
    /// This state is always used if it is provided.
    ///
    /// The default value is `None`.
    #[inline]
    pub fn discard_rectangle_state(
        mut self,
        discard_rectangle_state: DiscardRectangleState,
    ) -> Self {
        self.discard_rectangle_state = Some(discard_rectangle_state);
        self
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
