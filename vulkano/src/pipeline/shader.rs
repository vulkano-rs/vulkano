use crate::{
    device::Device,
    format::NumericType,
    macros::vulkan_bitflags,
    pipeline::graphics::color_blend::ColorComponents,
    shader::{
        reflect::get_constant,
        spirv::{
            BuiltIn, Decoration, ExecutionMode, ExecutionModel, Id, Instruction, Spirv,
            StorageClass,
        },
        EntryPoint, ShaderStage,
    },
    Requires, RequiresAllOf, RequiresOneOf, ValidationError,
};
use ahash::HashMap;
use std::collections::hash_map::Entry;

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

        let properties = device.physical_device().properties();

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkPipelineShaderStageCreateInfo-flags-parameter"])
        })?;

        let entry_point_info = entry_point.info();
        let stage_enum = ShaderStage::from(entry_point_info.execution_model);

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
            ShaderStage::SubpassShading => (),
        }

        let spirv = entry_point.module().spirv();
        let entry_point_function = spirv.function(entry_point.id());

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
        let workgroup_size = local_size.into_iter().try_fold(1, u32::checked_mul);

        match stage_enum {
            ShaderStage::Compute => {
                if local_size[0] > properties.max_compute_work_group_size[0] {
                    return Err(Box::new(ValidationError {
                        problem: "the `local_size_x` of `entry_point` is greater than \
                            `max_compute_work_group_size[0]`"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-x-06429"],
                        ..Default::default()
                    }));
                }

                if local_size[1] > properties.max_compute_work_group_size[1] {
                    return Err(Box::new(ValidationError {
                        problem: "the `local_size_y` of `entry_point` is greater than \
                            `max_compute_work_group_size[1]`"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-x-06430"],
                        ..Default::default()
                    }));
                }

                if local_size[2] > properties.max_compute_work_group_size[2] {
                    return Err(Box::new(ValidationError {
                        problem: "the `local_size_x` of `entry_point` is greater than \
                            `max_compute_work_group_size[2]`"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-x-06431"],
                        ..Default::default()
                    }));
                }

                if workgroup_size.map_or(true, |size| {
                    size > properties.max_compute_work_group_invocations
                }) {
                    return Err(Box::new(ValidationError {
                        problem: "the product of the `local_size_x`, `local_size_y` and \
                            `local_size_z` of `entry_point` is greater than the \
                            `max_compute_work_group_invocations` device limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-x-06432"],
                        ..Default::default()
                    }));
                }
            }
            ShaderStage::Task => {
                if local_size[0] > properties.max_task_work_group_size.unwrap_or_default()[0] {
                    return Err(Box::new(ValidationError {
                        problem: "the `local_size_x` of `entry_point` is greater than \
                            `max_task_work_group_size[0]`"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-TaskEXT-07291"],
                        ..Default::default()
                    }));
                }

                if local_size[1] > properties.max_task_work_group_size.unwrap_or_default()[1] {
                    return Err(Box::new(ValidationError {
                        problem: "the `local_size_y` of `entry_point` is greater than \
                            `max_task_work_group_size[1]`"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-TaskEXT-07292"],
                        ..Default::default()
                    }));
                }

                if local_size[2] > properties.max_task_work_group_size.unwrap_or_default()[2] {
                    return Err(Box::new(ValidationError {
                        problem: "the `local_size_x` of `entry_point` is greater than \
                            `max_task_work_group_size[2]`"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-TaskEXT-07293"],
                        ..Default::default()
                    }));
                }

                if workgroup_size.map_or(true, |size| {
                    size > properties
                        .max_task_work_group_invocations
                        .unwrap_or_default()
                }) {
                    return Err(Box::new(ValidationError {
                        problem: "the product of the `local_size_x`, `local_size_y` and \
                            `local_size_z` of `entry_point` is greater than the \
                            `max_task_work_group_invocations` device limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-TaskEXT-07294"],
                        ..Default::default()
                    }));
                }
            }
            ShaderStage::Mesh => {
                if local_size[0] > properties.max_mesh_work_group_size.unwrap_or_default()[0] {
                    return Err(Box::new(ValidationError {
                        problem: "the `local_size_x` of `entry_point` is greater than \
                            `max_mesh_work_group_size[0]`"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-MeshEXT-07295"],
                        ..Default::default()
                    }));
                }

                if local_size[1] > properties.max_mesh_work_group_size.unwrap_or_default()[1] {
                    return Err(Box::new(ValidationError {
                        problem: "the `local_size_y` of `entry_point` is greater than \
                            `max_mesh_work_group_size[1]`"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-MeshEXT-07296"],
                        ..Default::default()
                    }));
                }

                if local_size[2] > properties.max_mesh_work_group_size.unwrap_or_default()[2] {
                    return Err(Box::new(ValidationError {
                        problem: "the `local_size_x` of `entry_point` is greater than \
                            `max_mesh_work_group_size[2]`"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-MeshEXT-07297"],
                        ..Default::default()
                    }));
                }

                if workgroup_size.map_or(true, |size| {
                    size > properties
                        .max_mesh_work_group_invocations
                        .unwrap_or_default()
                }) {
                    return Err(Box::new(ValidationError {
                        problem: "the product of the `local_size_x`, `local_size_y` and \
                            `local_size_z` of `entry_point` is greater than the \
                            `max_mesh_work_group_invocations` device limit"
                            .into(),
                        vuids: &["VUID-RuntimeSpirv-MeshEXT-07298"],
                        ..Default::default()
                    }));
                }
            }
            _ => (),
        }

        let workgroup_size = workgroup_size.unwrap();

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

struct InterfaceVariableInfo<'a> {
    variable_decorations: &'a [Instruction],
    pointer_type_decorations: &'a [Instruction],
    block_type_info: Option<InterfaceVariableBlockInfo<'a>>,
    type_id: Id,
}

struct InterfaceVariableBlockInfo<'a> {
    block_type_decorations: &'a [Instruction],
    block_member_decorations: &'a [Instruction],
}

pub(crate) fn validate_interfaces_compatible(
    out_spirv: &Spirv,
    out_execution_model: ExecutionModel,
    out_interface: &[Id],
    in_spirv: &Spirv,
    in_execution_model: ExecutionModel,
    in_interface: &[Id],
    allow_larger_output_vector: bool,
) -> Result<(), Box<ValidationError>> {
    let out_variables_by_location = get_variables_by_location(
        out_spirv,
        out_execution_model,
        out_interface,
        StorageClass::Output,
    );
    let in_variables_by_location = get_variables_by_location(
        in_spirv,
        in_execution_model,
        in_interface,
        StorageClass::Input,
    );

    for (location, in_variable_info) in in_variables_by_location {
        let out_variable_info = out_variables_by_location.get(&location).ok_or_else(|| {
            Box::new(ValidationError {
                problem: format!(
                    "the input interface includes a variable at location {}, component {}, \
                    but the output interface does not contain a variable \
                    with the same location and component",
                    location.0, location.1,
                )
                .into(),
                vuids: &["VUID-RuntimeSpirv-OpEntryPoint-08743"],
                ..Default::default()
            })
        })?;

        if !are_interface_decoration_sets_compatible(
            out_spirv,
            out_variable_info.variable_decorations,
            in_spirv,
            in_variable_info.variable_decorations,
            decoration_filter_variable,
        ) {
            return Err(Box::new(ValidationError {
                problem: format!(
                    "for interface location {}, component {}, \
                    the input variable doesn't have the same decorations as the output variable",
                    location.0, location.1,
                )
                .into(),
                vuids: &["VUID-RuntimeSpirv-OpVariable-08746"],
                ..Default::default()
            }));
        }

        if !are_interface_decoration_sets_compatible(
            out_spirv,
            out_variable_info.pointer_type_decorations,
            in_spirv,
            in_variable_info.pointer_type_decorations,
            decoration_filter,
        ) {
            return Err(Box::new(ValidationError {
                problem: format!(
                    "for interface location {}, component {}, \
                    the input variable's pointer type doesn't have the same decorations as \
                    the output variable's pointer type",
                    location.0, location.1,
                )
                .into(),
                // vuids?
                ..Default::default()
            }));
        }

        match (
            &out_variable_info.block_type_info,
            &in_variable_info.block_type_info,
        ) {
            (Some(out_block_type_info), Some(in_block_type_info)) => {
                if !are_interface_decoration_sets_compatible(
                    out_spirv,
                    out_block_type_info.block_type_decorations,
                    in_spirv,
                    in_block_type_info.block_type_decorations,
                    decoration_filter,
                ) {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "for interface location {}, component {}, \
                            the input block structure type doesn't have the same decorations as \
                            the output block structure type",
                            location.0, location.1,
                        )
                        .into(),
                        vuids: &["VUID-RuntimeSpirv-OpVariable-08746"],
                        ..Default::default()
                    }));
                }

                if !are_interface_decoration_sets_compatible(
                    out_spirv,
                    out_block_type_info.block_member_decorations,
                    in_spirv,
                    in_block_type_info.block_member_decorations,
                    decoration_filter,
                ) {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "for interface location {}, component {}, \
                            the input block structure member doesn't have the same decorations as \
                            the output block structure member",
                            location.0, location.1,
                        )
                        .into(),
                        vuids: &["VUID-RuntimeSpirv-OpVariable-08746"],
                        ..Default::default()
                    }));
                }
            }
            (None, None) => (),
            (Some(_), None) | (None, Some(_)) => {
                // TODO: this may be allowed, depending on the outcome of this discussion:
                // https://github.com/KhronosGroup/Vulkan-Docs/issues/2242
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "for interface location {}, component {}, \
                        the input variable doesn't have the same or a compatible type as \
                        the output variable",
                        location.0, location.1,
                    )
                    .into(),
                    vuids: &[
                        "VUID-RuntimeSpirv-OpEntryPoint-07754",
                        "VUID-RuntimeSpirv-maintenance4-06817",
                    ],
                    ..Default::default()
                }));
            }
        }

        if !are_interface_types_compatible(
            out_spirv,
            out_variable_info.type_id,
            in_spirv,
            in_variable_info.type_id,
            allow_larger_output_vector,
        ) {
            return Err(Box::new(ValidationError {
                problem: format!(
                    "for interface location {}, component {}, \
                    the input variable doesn't have the same or a compatible type as \
                    the output variable",
                    location.0, location.1,
                )
                .into(),
                vuids: &[
                    "VUID-RuntimeSpirv-OpEntryPoint-07754",
                    "VUID-RuntimeSpirv-maintenance4-06817",
                ],
                ..Default::default()
            }));
        }
    }

    Ok(())
}

fn get_variables_by_location<'a>(
    spirv: &'a Spirv,
    execution_model: ExecutionModel,
    interface: &[Id],
    filter_storage_class: StorageClass,
) -> HashMap<(u32, u32), InterfaceVariableInfo<'a>> {
    // Collect all variables into a hashmap indexed by location.
    let mut variables_by_location: HashMap<_, _> = HashMap::default();

    for variable_id in interface.iter().copied() {
        let variable_id_info = spirv.id(variable_id);
        let (pointer_type_id, storage_class) = match *variable_id_info.instruction() {
            Instruction::Variable {
                result_type_id,
                storage_class,
                ..
            } if storage_class == filter_storage_class => (result_type_id, storage_class),
            _ => continue,
        };
        let pointer_type_id_info = spirv.id(pointer_type_id);
        let type_id = match *pointer_type_id_info.instruction() {
            Instruction::TypePointer { ty, .. } => {
                strip_array(spirv, storage_class, execution_model, variable_id, ty)
            }
            _ => unreachable!(),
        };

        let interface_variable_info = InterfaceVariableInfo {
            variable_decorations: variable_id_info.decorations(),
            pointer_type_decorations: pointer_type_id_info.decorations(),
            block_type_info: None,
            type_id,
        };

        let mut variable_location = None;
        let mut variable_component = 0;

        for instruction in variable_id_info.decorations() {
            if let Instruction::Decorate { ref decoration, .. } = *instruction {
                match *decoration {
                    Decoration::Location { location } => variable_location = Some(location),
                    Decoration::Component { component } => variable_component = component,
                    _ => (),
                }
            }
        }

        if let Some(variable_location) = variable_location {
            variables_by_location.insert(
                (variable_location, variable_component),
                interface_variable_info,
            );
        } else {
            let block_type_id_info = spirv.id(type_id);
            let block_type_decorations = block_type_id_info.decorations();
            let member_types = match block_type_id_info.instruction() {
                Instruction::TypeStruct { member_types, .. } => member_types,
                _ => continue,
            };

            for (&type_id, member_info) in member_types.iter().zip(block_type_id_info.members()) {
                let block_member_decorations = member_info.decorations();
                let mut member_location = None;
                let mut member_component = 0;

                for instruction in block_member_decorations {
                    if let Instruction::MemberDecorate { ref decoration, .. } = *instruction {
                        match *decoration {
                            Decoration::Location { location } => member_location = Some(location),
                            Decoration::Component { component } => member_component = component,
                            _ => (),
                        }
                    }
                }

                if let Some(member_location) = member_location {
                    variables_by_location.insert(
                        (member_location, member_component),
                        InterfaceVariableInfo {
                            block_type_info: Some(InterfaceVariableBlockInfo {
                                block_type_decorations,
                                block_member_decorations,
                            }),
                            type_id,
                            ..interface_variable_info
                        },
                    );
                }
            }
        }
    }

    variables_by_location
}

fn are_interface_types_compatible(
    out_spirv: &Spirv,
    out_type_id: Id,
    in_spirv: &Spirv,
    in_type_id: Id,
    allow_larger_output_vector: bool,
) -> bool {
    let out_id_info = out_spirv.id(out_type_id);
    let in_id_info = in_spirv.id(in_type_id);

    // Decorations must be compatible.
    if !are_interface_decoration_sets_compatible(
        out_spirv,
        out_id_info.decorations(),
        in_spirv,
        in_id_info.decorations(),
        decoration_filter,
    ) {
        return false;
    }

    // Type definitions must be compatible, potentially recursively.
    // TODO: Add more types. What else can appear in a shader interface?
    match (out_id_info.instruction(), in_id_info.instruction()) {
        (
            &Instruction::TypeInt {
                width: out_width,
                signedness: out_signedness,
                ..
            },
            &Instruction::TypeInt {
                width: in_width,
                signedness: in_signedness,
                ..
            },
        ) => out_width == in_width && out_signedness == in_signedness,
        (
            &Instruction::TypeFloat {
                width: out_width, ..
            },
            &Instruction::TypeFloat {
                width: in_width, ..
            },
        ) => out_width == in_width,
        (
            &Instruction::TypeVector {
                component_type: out_component_type,
                component_count: out_component_count,
                ..
            },
            &Instruction::TypeVector {
                component_type: in_component_type,
                component_count: in_component_count,
                ..
            },
        ) => {
            let is_component_count_compatible = if allow_larger_output_vector {
                out_component_count >= in_component_count
            } else {
                out_component_count == in_component_count
            };

            is_component_count_compatible
                && are_interface_types_compatible(
                    out_spirv,
                    out_component_type,
                    in_spirv,
                    in_component_type,
                    allow_larger_output_vector,
                )
        }
        (
            &Instruction::TypeMatrix {
                column_type: out_column_type,
                column_count: out_column_count,
                ..
            },
            &Instruction::TypeMatrix {
                column_type: in_column_type,
                column_count: in_column_count,
                ..
            },
        ) => {
            out_column_count == in_column_count
                && are_interface_types_compatible(
                    out_spirv,
                    out_column_type,
                    in_spirv,
                    in_column_type,
                    allow_larger_output_vector,
                )
        }
        (
            &Instruction::TypeArray {
                element_type: out_element_type,
                length: out_length,
                ..
            },
            &Instruction::TypeArray {
                element_type: in_element_type,
                length: in_length,
                ..
            },
        ) => {
            let out_length = match *out_spirv.id(out_length).instruction() {
                Instruction::Constant { ref value, .. } => {
                    value.iter().rev().fold(0, |a, &b| (a << 32) | b as u64)
                }
                _ => unreachable!(),
            };
            let in_length = match *in_spirv.id(in_length).instruction() {
                Instruction::Constant { ref value, .. } => {
                    value.iter().rev().fold(0, |a, &b| (a << 32) | b as u64)
                }
                _ => unreachable!(),
            };

            out_length == in_length
                && are_interface_types_compatible(
                    out_spirv,
                    out_element_type,
                    in_spirv,
                    in_element_type,
                    allow_larger_output_vector,
                )
        }
        (
            &Instruction::TypeStruct {
                member_types: ref out_member_types,
                ..
            },
            &Instruction::TypeStruct {
                member_types: ref in_member_types,
                ..
            },
        ) => {
            out_member_types.len() == in_member_types.len()
                && (out_member_types.iter().zip(out_id_info.members()))
                    .zip(in_member_types.iter().zip(in_id_info.members()))
                    .all(
                        |(
                            (&out_member_type, out_member_info),
                            (&in_member_type, in_member_info),
                        )| {
                            are_interface_decoration_sets_compatible(
                                out_spirv,
                                out_member_info.decorations(),
                                in_spirv,
                                in_member_info.decorations(),
                                decoration_filter,
                            ) && are_interface_types_compatible(
                                out_spirv,
                                out_member_type,
                                in_spirv,
                                in_member_type,
                                allow_larger_output_vector,
                            )
                        },
                    )
        }
        _ => false,
    }
}

fn are_interface_decoration_sets_compatible(
    out_spirv: &Spirv,
    out_instructions: &[Instruction],
    in_spirv: &Spirv,
    in_instructions: &[Instruction],
    filter: fn(&Instruction) -> bool,
) -> bool {
    if out_instructions.is_empty() && in_instructions.is_empty() {
        return true;
    }

    // This is O(n²), but instructions are not expected to have very many decorations.
    out_instructions
        .iter()
        .filter(|i| filter(i))
        .all(|out_instruction| {
            in_instructions.iter().any(|in_instruction| {
                are_interface_decorations_compatible(
                    out_spirv,
                    out_instruction,
                    in_spirv,
                    in_instruction,
                )
            })
        })
        && in_instructions
            .iter()
            .filter(|i| filter(i))
            .all(|in_instruction| {
                out_instructions.iter().any(|out_instruction| {
                    are_interface_decorations_compatible(
                        out_spirv,
                        out_instruction,
                        in_spirv,
                        in_instruction,
                    )
                })
            })
}

fn are_interface_decorations_compatible(
    out_spirv: &Spirv,
    out_instruction: &Instruction,
    in_spirv: &Spirv,
    in_instruction: &Instruction,
) -> bool {
    match (out_instruction, in_instruction) {
        // Regular decorations are equal if the decorations match.
        (
            Instruction::Decorate {
                decoration: out_decoration,
                ..
            },
            Instruction::Decorate {
                decoration: in_decoration,
                ..
            },
        )
        | (
            Instruction::MemberDecorate {
                decoration: out_decoration,
                ..
            },
            Instruction::MemberDecorate {
                decoration: in_decoration,
                ..
            },
        )
        | (
            Instruction::DecorateString {
                decoration: out_decoration,
                ..
            },
            Instruction::DecorateString {
                decoration: in_decoration,
                ..
            },
        )
        | (
            Instruction::MemberDecorateString {
                decoration: out_decoration,
                ..
            },
            Instruction::MemberDecorateString {
                decoration: in_decoration,
                ..
            },
        ) => out_decoration == in_decoration,

        // DecorateId needs more care, because the Ids must first be resolved before comparing.
        (
            Instruction::DecorateId {
                decoration: out_decoration,
                ..
            },
            Instruction::DecorateId {
                decoration: in_decoration,
                ..
            },
        ) => match (out_decoration, in_decoration) {
            (
                &Decoration::UniformId {
                    execution: out_execution,
                },
                &Decoration::UniformId {
                    execution: in_execution,
                },
            ) => match (
                out_spirv.id(out_execution).instruction(),
                in_spirv.id(in_execution).instruction(),
            ) {
                (
                    Instruction::Constant {
                        value: out_value, ..
                    },
                    Instruction::Constant {
                        value: in_value, ..
                    },
                ) => out_value == in_value,
                _ => unimplemented!("the Ids of `Decoration::UniformId` are not both constants"),
            },
            (&Decoration::AlignmentId { .. }, &Decoration::AlignmentId { .. }) => {
                unreachable!("requires the `Kernel` capability, which Vulkan does not support")
            }
            (
                &Decoration::MaxByteOffsetId {
                    max_byte_offset: out_max_byte_offset,
                },
                &Decoration::MaxByteOffsetId {
                    max_byte_offset: in_max_byte_offset,
                },
            ) => match (
                out_spirv.id(out_max_byte_offset).instruction(),
                in_spirv.id(in_max_byte_offset).instruction(),
            ) {
                (
                    Instruction::Constant {
                        value: out_value, ..
                    },
                    Instruction::Constant {
                        value: in_value, ..
                    },
                ) => out_value == in_value,
                _ => unimplemented!(
                    "the Ids of `Decoration::MaxByteOffsetId` are not both constants"
                ),
            },
            (&Decoration::CounterBuffer { .. }, &Decoration::CounterBuffer { .. }) => {
                unreachable!("can decorate only the `Uniform` storage class")
            }
            (
                &Decoration::AliasScopeINTEL {
                    aliasing_scopes_list: _out_aliasing_scopes_list,
                },
                &Decoration::AliasScopeINTEL {
                    aliasing_scopes_list: _in_aliasing_scopes_list,
                },
            ) => unimplemented!(),
            (
                &Decoration::NoAliasINTEL {
                    aliasing_scopes_list: _out_aliasing_scopes_list,
                },
                &Decoration::NoAliasINTEL {
                    aliasing_scopes_list: _in_aliasing_scopes_list,
                },
            ) => unimplemented!(),
            _ => false,
        },
        _ => false,
    }
}

fn decoration_filter(instruction: &Instruction) -> bool {
    match instruction {
        Instruction::Decorate { decoration, .. }
        | Instruction::MemberDecorate { decoration, .. }
        | Instruction::DecorateString { decoration, .. }
        | Instruction::MemberDecorateString { decoration, .. }
        | Instruction::DecorateId { decoration, .. } => !matches!(
            decoration,
            Decoration::Location { .. }
                | Decoration::XfbBuffer { .. }
                | Decoration::XfbStride { .. }
                | Decoration::Offset { .. }
                | Decoration::Stream { .. }
                | Decoration::Component { .. }
                | Decoration::NoPerspective
                | Decoration::Flat
                | Decoration::Centroid
                | Decoration::Sample
                | Decoration::PerVertexKHR
        ),
        _ => false,
    }
}

fn decoration_filter_variable(instruction: &Instruction) -> bool {
    match instruction {
        Instruction::Decorate { decoration, .. }
        | Instruction::MemberDecorate { decoration, .. }
        | Instruction::DecorateString { decoration, .. }
        | Instruction::MemberDecorateString { decoration, .. }
        | Instruction::DecorateId { decoration, .. } => !matches!(
            decoration,
            Decoration::Location { .. }
                | Decoration::XfbBuffer { .. }
                | Decoration::XfbStride { .. }
                | Decoration::Offset { .. }
                | Decoration::Stream { .. }
                | Decoration::Component { .. }
                | Decoration::NoPerspective
                | Decoration::Flat
                | Decoration::Centroid
                | Decoration::Sample
                | Decoration::PerVertexKHR
                | Decoration::RelaxedPrecision
        ),
        _ => false,
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ShaderInterfaceLocationInfo {
    pub(crate) numeric_type: NumericType,
    pub(crate) width: ShaderInterfaceLocationWidth,
    pub(crate) components: [ColorComponents; 2], // Index 0 and 1
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ShaderInterfaceLocationWidth {
    Bits32,
    Bits64,
}

impl From<u32> for ShaderInterfaceLocationWidth {
    #[inline]
    fn from(value: u32) -> Self {
        if value > 32 {
            Self::Bits64
        } else {
            Self::Bits32
        }
    }
}

pub(crate) fn shader_interface_location_info(
    spirv: &Spirv,
    entry_point_id: Id,
    filter_storage_class: StorageClass,
) -> HashMap<u32, ShaderInterfaceLocationInfo> {
    fn add_type(
        locations: &mut HashMap<u32, ShaderInterfaceLocationInfo>,
        spirv: &Spirv,
        mut location: u32,
        mut component: u32,
        index: u32,
        type_id: Id,
    ) -> (u32, u32) {
        debug_assert!(component < 4);

        let mut add_scalar = |numeric_type: NumericType, width: u32| -> (u32, u32) {
            let width = ShaderInterfaceLocationWidth::from(width);
            let components_to_add = match width {
                ShaderInterfaceLocationWidth::Bits32 => {
                    ColorComponents::from_index(component as usize)
                }
                ShaderInterfaceLocationWidth::Bits64 => {
                    debug_assert!(component & 1 == 0);
                    ColorComponents::from_index(component as usize)
                        | ColorComponents::from_index(component as usize + 1)
                }
            };

            let location_info = match locations.entry(location) {
                Entry::Occupied(entry) => {
                    let location_info = entry.into_mut();
                    debug_assert_eq!(location_info.numeric_type, numeric_type);
                    debug_assert_eq!(location_info.width, width);
                    location_info
                }
                Entry::Vacant(entry) => entry.insert(ShaderInterfaceLocationInfo {
                    numeric_type,
                    width,
                    components: [ColorComponents::empty(); 2],
                }),
            };

            let components = &mut location_info.components[index as usize];
            debug_assert!(!components.intersects(components_to_add));
            *components |= components_to_add;

            (components_to_add.count(), 1)
        };

        match *spirv.id(type_id).instruction() {
            Instruction::TypeInt {
                width, signedness, ..
            } => {
                let numeric_type = if signedness == 1 {
                    NumericType::Int
                } else {
                    NumericType::Uint
                };

                add_scalar(numeric_type, width)
            }
            Instruction::TypeFloat { width, .. } => add_scalar(NumericType::Float, width),
            Instruction::TypeVector {
                component_type,
                component_count,
                ..
            } => {
                let mut total_locations_added = 1;

                for _ in 0..component_count {
                    // Overflow into next location
                    if component == 4 {
                        component = 0;
                        location += 1;
                        total_locations_added += 1;
                    } else {
                        debug_assert!(component < 4);
                    }

                    let (_, components_added) =
                        add_type(locations, spirv, location, component, index, component_type);
                    component += components_added;
                }

                (total_locations_added, 0)
            }
            Instruction::TypeMatrix {
                column_type,
                column_count,
                ..
            } => {
                let mut total_locations_added = 0;

                for _ in 0..column_count {
                    let (locations_added, _) =
                        add_type(locations, spirv, location, component, index, column_type);
                    location += locations_added;
                    total_locations_added += locations_added;
                }

                (total_locations_added, 0)
            }
            Instruction::TypeArray {
                element_type,
                length,
                ..
            } => {
                let length = get_constant(spirv, length).unwrap();
                let mut total_locations_added = 0;

                for _ in 0..length {
                    let (locations_added, _) =
                        add_type(locations, spirv, location, component, index, element_type);
                    location += locations_added;
                    total_locations_added += locations_added;
                }

                (total_locations_added, 0)
            }
            Instruction::TypeStruct {
                ref member_types, ..
            } => {
                let mut total_locations_added = 0;

                for &member_type in member_types {
                    let (locations_added, _) =
                        add_type(locations, spirv, location, component, index, member_type);
                    location += locations_added;
                    total_locations_added += locations_added;
                }

                (total_locations_added, 0)
            }
            _ => unimplemented!(),
        }
    }

    let (execution_model, interface) = match spirv.function(entry_point_id).entry_point() {
        Some(&Instruction::EntryPoint {
            execution_model,
            ref interface,
            ..
        }) => (execution_model, interface),
        _ => unreachable!(),
    };

    let mut locations = HashMap::default();

    for &variable_id in interface {
        let variable_id_info = spirv.id(variable_id);
        let (pointer_type_id, storage_class) = match *variable_id_info.instruction() {
            Instruction::Variable {
                result_type_id,
                storage_class,
                ..
            } if storage_class == filter_storage_class => (result_type_id, storage_class),
            _ => continue,
        };
        let pointer_type_id_info = spirv.id(pointer_type_id);
        let type_id = match *pointer_type_id_info.instruction() {
            Instruction::TypePointer { ty, .. } => {
                strip_array(spirv, storage_class, execution_model, variable_id, ty)
            }
            _ => unreachable!(),
        };

        let mut variable_location = None;
        let mut variable_component = 0;
        let mut variable_index = 0;

        for instruction in variable_id_info.decorations() {
            if let Instruction::Decorate { ref decoration, .. } = *instruction {
                match *decoration {
                    Decoration::Location { location } => variable_location = Some(location),
                    Decoration::Component { component } => variable_component = component,
                    Decoration::Index { index } => variable_index = index,
                    _ => (),
                }
            }
        }

        if let Some(variable_location) = variable_location {
            add_type(
                &mut locations,
                spirv,
                variable_location,
                variable_component,
                variable_index,
                type_id,
            );
        } else {
            let block_type_id_info = spirv.id(type_id);
            let member_types = match block_type_id_info.instruction() {
                Instruction::TypeStruct { member_types, .. } => member_types,
                _ => continue,
            };

            for (&type_id, member_info) in member_types.iter().zip(block_type_id_info.members()) {
                let mut member_location = None;
                let mut member_component = 0;
                let mut member_index = 0;

                for instruction in member_info.decorations() {
                    if let Instruction::MemberDecorate { ref decoration, .. } = *instruction {
                        match *decoration {
                            Decoration::Location { location } => member_location = Some(location),
                            Decoration::Component { component } => member_component = component,
                            Decoration::Index { index } => member_index = index,
                            _ => (),
                        }
                    }
                }

                if let Some(member_location) = member_location {
                    add_type(
                        &mut locations,
                        spirv,
                        member_location,
                        member_component,
                        member_index,
                        type_id,
                    );
                }
            }
        }
    }

    locations
}

fn strip_array(
    spirv: &Spirv,
    storage_class: StorageClass,
    execution_model: ExecutionModel,
    variable_id: Id,
    pointed_type_id: Id,
) -> Id {
    let variable_decorations = spirv.id(variable_id).decorations();
    let variable_has_decoration = |has_decoration: Decoration| -> bool {
        variable_decorations.iter().any(|instruction| {
            matches!(
                instruction,
                Instruction::Decorate {
                    decoration,
                    ..
                } if *decoration == has_decoration
            )
        })
    };

    let must_strip_array = match storage_class {
        StorageClass::Output => match execution_model {
            ExecutionModel::TaskEXT | ExecutionModel::MeshEXT | ExecutionModel::MeshNV => true,
            ExecutionModel::TessellationControl => !variable_has_decoration(Decoration::Patch),
            ExecutionModel::TaskNV => !variable_has_decoration(Decoration::PerTaskNV),
            _ => false,
        },
        StorageClass::Input => match execution_model {
            ExecutionModel::Geometry | ExecutionModel::MeshEXT => true,
            ExecutionModel::TessellationControl | ExecutionModel::TessellationEvaluation => {
                !variable_has_decoration(Decoration::Patch)
            }
            ExecutionModel::Fragment => variable_has_decoration(Decoration::PerVertexKHR),
            ExecutionModel::MeshNV => !variable_has_decoration(Decoration::PerTaskNV),
            _ => false,
        },
        _ => unreachable!(),
    };

    if must_strip_array {
        match spirv.id(pointed_type_id).instruction() {
            &Instruction::TypeArray { element_type, .. } => element_type,
            _ => pointed_type_id,
        }
    } else {
        pointed_type_id
    }
}
