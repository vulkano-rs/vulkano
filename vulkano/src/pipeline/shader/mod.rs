use crate::{
    device::Device,
    macros::vulkan_bitflags,
    shader::{
        spirv::{BuiltIn, Decoration, ExecutionMode, Id, Instruction},
        EntryPoint, ShaderStage,
    },
    Requires, RequiresAllOf, RequiresOneOf, ValidationError,
};

pub(crate) mod inout_interface;

/// Specifies a single shader stage when creating a pipeline.
#[derive(Clone, Debug)]
pub struct PipelineShaderStageCreateInfo {
    /// Additional properties of the shader stage.
    ///
    /// The default value is empty.
    pub flags: PipelineShaderStageCreateFlags,

    /// The shader entry point for the stage, which includes any specialization constants.
    ///
    /// There is no default value.
    pub entry_point: EntryPoint,

    /// The required subgroup size.
    ///
    /// Requires [`subgroup_size_control`](crate::device::Features::subgroup_size_control). The
    /// shader stage must be included in
    /// [`required_subgroup_size_stages`](crate::device::Properties::required_subgroup_size_stages).
    /// Subgroup size must be power of 2 and within
    /// [`min_subgroup_size`](crate::device::Properties::min_subgroup_size)
    /// and [`max_subgroup_size`](crate::device::Properties::max_subgroup_size).
    ///
    /// For compute shaders, `max_compute_workgroup_subgroups * required_subgroup_size` must be
    /// greater than or equal to `workgroup_size.x * workgroup_size.y * workgroup_size.z`.
    ///
    /// The default value is None.
    pub required_subgroup_size: Option<u32>,

    pub _ne: crate::NonExhaustive,
}

impl PipelineShaderStageCreateInfo {
    /// Returns a `PipelineShaderStageCreateInfo` with the specified `entry_point`.
    #[inline]
    pub fn new(entry_point: EntryPoint) -> Self {
        Self {
            flags: PipelineShaderStageCreateFlags::empty(),
            entry_point,
            required_subgroup_size: None,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            ref entry_point,
            required_subgroup_size,
            _ne: _,
        } = self;

        let spirv = entry_point.module().spirv();
        let properties = device.physical_device().properties();

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkPipelineShaderStageCreateInfo-flags-parameter"])
        })?;

        let execution_model = entry_point.info().execution_model;
        let stage_enum = ShaderStage::from(execution_model);

        stage_enum.validate_device(device).map_err(|err| {
            err.add_context("entry_point.info().execution")
                .set_vuids(&["VUID-VkPipelineShaderStageCreateInfo-stage-parameter"])
        })?;

        // VUID-VkPipelineShaderStageCreateInfo-pName-00707
        // Guaranteed by definition of `EntryPoint`.

        // TODO:
        // VUID-VkPipelineShaderStageCreateInfo-maxClipDistances-00708
        // VUID-VkPipelineShaderStageCreateInfo-maxCullDistances-00709
        // VUID-VkPipelineShaderStageCreateInfo-maxCombinedClipAndCullDistances-00710
        // VUID-VkPipelineShaderStageCreateInfo-maxSampleMaskWords-00711
        // VUID-VkPipelineShaderStageCreateInfo-stage-02596
        // VUID-VkPipelineShaderStageCreateInfo-stage-02597

        match stage_enum {
            ShaderStage::Vertex => {
                // VUID-VkPipelineShaderStageCreateInfo-stage-00712
                // TODO:
            }
            ShaderStage::TessellationControl | ShaderStage::TessellationEvaluation => {
                if !device.enabled_features().tessellation_shader {
                    return Err(Box::new(ValidationError {
                        context: "entry_point".into(),
                        problem: "specifies a `ShaderStage::TessellationControl` or \
                            `ShaderStage::TessellationEvaluation` entry point"
                            .into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "tessellation_shader",
                        )])]),
                        vuids: &["VUID-VkPipelineShaderStageCreateInfo-stage-00705"],
                    }));
                }

                // VUID-VkPipelineShaderStageCreateInfo-stage-00713
                // TODO:
            }
            ShaderStage::Geometry => {
                if !device.enabled_features().geometry_shader {
                    return Err(Box::new(ValidationError {
                        context: "entry_point".into(),
                        problem: "specifies a `ShaderStage::Geometry` entry point".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "geometry_shader",
                        )])]),
                        vuids: &["VUID-VkPipelineShaderStageCreateInfo-stage-00704"],
                    }));
                }

                // TODO:
                // VUID-VkPipelineShaderStageCreateInfo-stage-00714
                // VUID-VkPipelineShaderStageCreateInfo-stage-00715
            }
            ShaderStage::Fragment => {
                // TODO:
                // VUID-VkPipelineShaderStageCreateInfo-stage-00718
                // VUID-VkPipelineShaderStageCreateInfo-stage-06685
                // VUID-VkPipelineShaderStageCreateInfo-stage-06686
            }
            ShaderStage::Compute => (),
            ShaderStage::Raygen => (),
            ShaderStage::AnyHit => (),
            ShaderStage::ClosestHit => (),
            ShaderStage::Miss => (),
            ShaderStage::Intersection => (),
            ShaderStage::Callable => (),
            ShaderStage::Mesh => {
                if !device.enabled_features().mesh_shader {
                    return Err(Box::new(ValidationError {
                        context: "entry_point".into(),
                        problem: "specifies a `ShaderStage::Mesh` entry point".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "mesh_shader",
                        )])]),
                        vuids: &["VUID-VkPipelineShaderStageCreateInfo-stage-02091"],
                    }));
                }
            }
            ShaderStage::Task => {
                if !device.enabled_features().task_shader {
                    return Err(Box::new(ValidationError {
                        context: "entry_point".into(),
                        problem: "specifies a `ShaderStage::Task` entry point".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "task_shader",
                        )])]),
                        vuids: &["VUID-VkPipelineShaderStageCreateInfo-stage-02092"],
                    }));
                }
            }
            ShaderStage::SubpassShading => (),
        }

        let mut clip_distance_array_size = 0;
        let mut cull_distance_array_size = 0;

        for instruction in spirv.decorations() {
            if let Instruction::Decorate {
                target,
                decoration: Decoration::BuiltIn { built_in },
            } = *instruction
            {
                let variable_array_size = |variable| {
                    let result_type_id = match *spirv.id(variable).instruction() {
                        Instruction::Variable { result_type_id, .. } => result_type_id,
                        _ => return None,
                    };

                    let length = match *spirv.id(result_type_id).instruction() {
                        Instruction::TypeArray { length, .. } => length,
                        _ => return None,
                    };

                    let value = match *spirv.id(length).instruction() {
                        Instruction::Constant { ref value, .. } => {
                            if value.len() > 1 {
                                u32::MAX
                            } else {
                                value[0]
                            }
                        }
                        _ => return None,
                    };

                    Some(value)
                };

                match built_in {
                    BuiltIn::ClipDistance => {
                        clip_distance_array_size = variable_array_size(target).unwrap();

                        if clip_distance_array_size > properties.max_clip_distances {
                            return Err(Box::new(ValidationError {
                                context: "entry_point".into(),
                                problem: "the number of elements in the `ClipDistance` built-in \
                                    variable is greater than the \
                                    `max_clip_distances` device limit"
                                    .into(),
                                vuids: &[
                                    "VUID-VkPipelineShaderStageCreateInfo-maxClipDistances-00708",
                                ],
                                ..Default::default()
                            }));
                        }
                    }
                    BuiltIn::CullDistance => {
                        cull_distance_array_size = variable_array_size(target).unwrap();

                        if cull_distance_array_size > properties.max_cull_distances {
                            return Err(Box::new(ValidationError {
                                context: "entry_point".into(),
                                problem: "the number of elements in the `CullDistance` built-in \
                                    variable is greater than the \
                                    `max_cull_distances` device limit"
                                    .into(),
                                vuids: &[
                                    "VUID-VkPipelineShaderStageCreateInfo-maxCullDistances-00709",
                                ],
                                ..Default::default()
                            }));
                        }
                    }
                    BuiltIn::SampleMask => {
                        if variable_array_size(target).unwrap() > properties.max_sample_mask_words {
                            return Err(Box::new(ValidationError {
                                context: "entry_point".into(),
                                problem: "the number of elements in the `SampleMask` built-in \
                                    variable is greater than the \
                                    `max_sample_mask_words` device limit"
                                    .into(),
                                vuids: &[
                                    "VUID-VkPipelineShaderStageCreateInfo-maxSampleMaskWords-00711",
                                ],
                                ..Default::default()
                            }));
                        }
                    }
                    _ => (),
                }
            }
        }

        if clip_distance_array_size
            .checked_add(cull_distance_array_size)
            .map_or(true, |sum| {
                sum > properties.max_combined_clip_and_cull_distances
            })
        {
            return Err(Box::new(ValidationError {
                context: "entry_point".into(),
                problem: "the sum of the number of elements in the `ClipDistance` and \
                    `CullDistance` built-in variables is greater than the \
                    `max_combined_clip_and_cull_distances` device limit"
                    .into(),
                vuids: &[
                    "VUID-VkPipelineShaderStageCreateInfo-maxCombinedClipAndCullDistances-00710",
                ],
                ..Default::default()
            }));
        }

        let entry_point_function = spirv.function(entry_point.id());

        for instruction in entry_point_function.execution_modes() {
            if let Instruction::ExecutionMode {
                mode: ExecutionMode::OutputVertices { vertex_count },
                ..
            } = *instruction
            {
                match stage_enum {
                    ShaderStage::TessellationControl | ShaderStage::TessellationEvaluation => {
                        if vertex_count == 0 {
                            return Err(Box::new(ValidationError {
                                context: "entry_point".into(),
                                problem: "the `vertex_count` of the \
                                    `ExecutionMode::OutputVertices` is zero"
                                    .into(),
                                vuids: &["VUID-VkPipelineShaderStageCreateInfo-stage-00713"],
                                ..Default::default()
                            }));
                        }

                        if vertex_count > properties.max_tessellation_patch_size {
                            return Err(Box::new(ValidationError {
                                context: "entry_point".into(),
                                problem: "the `vertex_count` of the \
                                    `ExecutionMode::OutputVertices` is greater than the \
                                    `max_tessellation_patch_size` device limit"
                                    .into(),
                                vuids: &["VUID-VkPipelineShaderStageCreateInfo-stage-00713"],
                                ..Default::default()
                            }));
                        }
                    }
                    ShaderStage::Geometry => {
                        if vertex_count == 0 {
                            return Err(Box::new(ValidationError {
                                context: "entry_point".into(),
                                problem: "the `vertex_count` of the \
                                    `ExecutionMode::OutputVertices` is zero"
                                    .into(),
                                vuids: &["VUID-VkPipelineShaderStageCreateInfo-stage-00714"],
                                ..Default::default()
                            }));
                        }

                        if vertex_count > properties.max_geometry_output_vertices {
                            return Err(Box::new(ValidationError {
                                context: "entry_point".into(),
                                problem: "the `vertex_count` of the \
                                    `ExecutionMode::OutputVertices` is greater than the \
                                    `max_geometry_output_vertices` device limit"
                                    .into(),
                                vuids: &["VUID-VkPipelineShaderStageCreateInfo-stage-00714"],
                                ..Default::default()
                            }));
                        }
                    }
                    _ => (),
                }
            }
        }

        let local_size = (spirv
            .decorations()
            .iter()
            .find_map(|instruction| match *instruction {
                Instruction::Decorate {
                    target,
                    decoration:
                        Decoration::BuiltIn {
                            built_in: BuiltIn::WorkgroupSize,
                        },
                } => {
                    let constituents: &[Id; 3] = match *spirv.id(target).instruction() {
                        Instruction::ConstantComposite {
                            ref constituents, ..
                        } => constituents.as_slice().try_into().unwrap(),
                        _ => unreachable!(),
                    };

                    let local_size = constituents.map(|id| match *spirv.id(id).instruction() {
                        Instruction::Constant { ref value, .. } => {
                            assert!(value.len() == 1);
                            value[0]
                        }
                        _ => unreachable!(),
                    });

                    Some(local_size)
                }
                _ => None,
            }))
        .or_else(|| {
            entry_point_function
                .execution_modes()
                .iter()
                .find_map(|instruction| match *instruction {
                    Instruction::ExecutionMode {
                        mode:
                            ExecutionMode::LocalSize {
                                x_size,
                                y_size,
                                z_size,
                            },
                        ..
                    } => Some([x_size, y_size, z_size]),
                    Instruction::ExecutionModeId {
                        mode:
                            ExecutionMode::LocalSizeId {
                                x_size,
                                y_size,
                                z_size,
                            },
                        ..
                    } => Some([x_size, y_size, z_size].map(
                        |id| match *spirv.id(id).instruction() {
                            Instruction::Constant { ref value, .. } => {
                                assert!(value.len() == 1);
                                value[0]
                            }
                            _ => unreachable!(),
                        },
                    )),
                    _ => None,
                })
        })
        .unwrap_or_default();
        let workgroup_size = local_size
            .into_iter()
            .try_fold(1, u32::checked_mul)
            .unwrap();

        if let Some(required_subgroup_size) = required_subgroup_size {
            if !device.enabled_features().subgroup_size_control {
                return Err(Box::new(ValidationError {
                    context: "required_subgroup_size".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "subgroup_size_control",
                    )])]),
                    vuids: &["VUID-VkPipelineShaderStageCreateInfo-pNext-02755"],
                }));
            }

            if !properties
                .required_subgroup_size_stages
                .unwrap_or_default()
                .contains_enum(stage_enum)
            {
                return Err(Box::new(ValidationError {
                    problem: "`required_subgroup_size` is `Some`, but the \
                        `required_subgroup_size_stages` device property does not contain the \
                        shader stage of `entry_point`"
                        .into(),
                    vuids: &["VUID-VkPipelineShaderStageCreateInfo-pNext-02755"],
                    ..Default::default()
                }));
            }

            if !required_subgroup_size.is_power_of_two() {
                return Err(Box::new(ValidationError {
                    context: "required_subgroup_size".into(),
                    problem: "is not a power of 2".into(),
                    vuids: &["VUID-VkPipelineShaderStageRequiredSubgroupSizeCreateInfo-requiredSubgroupSize-02760"],
                    ..Default::default()
                }));
            }

            if required_subgroup_size < properties.min_subgroup_size.unwrap_or(1) {
                return Err(Box::new(ValidationError {
                    context: "required_subgroup_size".into(),
                    problem: "is less than the `min_subgroup_size` device limit".into(),
                    vuids: &["VUID-VkPipelineShaderStageRequiredSubgroupSizeCreateInfo-requiredSubgroupSize-02761"],
                    ..Default::default()
                }));
            }

            if required_subgroup_size > properties.max_subgroup_size.unwrap_or(128) {
                return Err(Box::new(ValidationError {
                    context: "required_subgroup_size".into(),
                    problem: "is greater than the `max_subgroup_size` device limit".into(),
                    vuids: &["VUID-VkPipelineShaderStageRequiredSubgroupSizeCreateInfo-requiredSubgroupSize-02762"],
                    ..Default::default()
                }));
            }

            if matches!(
                stage_enum,
                ShaderStage::Compute | ShaderStage::Mesh | ShaderStage::Task
            ) && workgroup_size
                > required_subgroup_size
                    .checked_mul(
                        properties
                            .max_compute_workgroup_subgroups
                            .unwrap_or_default(),
                    )
                    .unwrap_or(u32::MAX)
            {
                return Err(Box::new(ValidationError {
                    problem: "the product of the `local_size_x`, `local_size_y` and \
                        `local_size_z` of `entry_point` is greater than the the product \
                        of `required_subgroup_size` and the \
                        `max_compute_workgroup_subgroups` device limit"
                        .into(),
                    vuids: &["VUID-VkPipelineShaderStageCreateInfo-pNext-02756"],
                    ..Default::default()
                }));
            }
        }

        // TODO:
        // VUID-VkPipelineShaderStageCreateInfo-module-08987

        Ok(())
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a pipeline shader stage.
    PipelineShaderStageCreateFlags = PipelineShaderStageCreateFlags(u32);

    /* TODO: enable
    // TODO: document
    ALLOW_VARYING_SUBGROUP_SIZE = ALLOW_VARYING_SUBGROUP_SIZE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_subgroup_size_control)]),
    ]),
    */

    /* TODO: enable
    // TODO: document
    REQUIRE_FULL_SUBGROUPS = REQUIRE_FULL_SUBGROUPS
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_subgroup_size_control)]),
    ]),
    */
}
