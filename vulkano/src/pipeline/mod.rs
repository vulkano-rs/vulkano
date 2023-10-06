// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Describes a processing operation that will execute on the Vulkan device.
//!
//! In Vulkan, before you can add a draw or a compute command to a command buffer you have to
//! create a *pipeline object* that describes this command.
//!
//! When you create a pipeline object, the implementation will usually generate some GPU machine
//! code that will execute the operation (similar to a compiler that generates an executable for
//! the CPU). Consequently it is a CPU-intensive operation that should be performed at
//! initialization or during a loading screen.

pub use self::{compute::ComputePipeline, graphics::GraphicsPipeline, layout::PipelineLayout};
use crate::{
    device::{Device, DeviceOwned},
    macros::{vulkan_bitflags, vulkan_enum},
    shader::{
        spirv::{BuiltIn, Decoration, ExecutionMode, Id, Instruction},
        DescriptorBindingRequirements, EntryPoint, ShaderStage,
    },
    Requires, RequiresAllOf, RequiresOneOf, ValidationError,
};
use ahash::HashMap;
use std::sync::Arc;

pub mod cache;
pub mod compute;
pub mod graphics;
pub mod layout;

/// A trait for operations shared between pipeline types.
pub trait Pipeline: DeviceOwned {
    /// Returns the bind point of this pipeline.
    fn bind_point(&self) -> PipelineBindPoint;

    /// Returns the pipeline layout used in this pipeline.
    fn layout(&self) -> &Arc<PipelineLayout>;

    /// Returns the number of descriptor sets actually accessed by this pipeline. This may be less
    /// than the number of sets in the pipeline layout.
    fn num_used_descriptor_sets(&self) -> u32;

    /// Returns a reference to the descriptor binding requirements for this pipeline.
    fn descriptor_binding_requirements(
        &self,
    ) -> &HashMap<(u32, u32), DescriptorBindingRequirements>;
}

vulkan_enum! {
    #[non_exhaustive]

    /// The type of a pipeline.
    ///
    /// When binding a pipeline or descriptor sets in a command buffer, the state for each bind point
    /// is independent from the others. This means that it is possible, for example, to bind a graphics
    /// pipeline without disturbing any bound compute pipeline. Likewise, binding descriptor sets for
    /// the `Compute` bind point does not affect sets that were bound to the `Graphics` bind point.
    PipelineBindPoint = PipelineBindPoint(i32);

    // TODO: document
    Compute = COMPUTE,

    // TODO: document
    Graphics = GRAPHICS,

    /* TODO: enable
    // TODO: document
    RayTracing = RAY_TRACING_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    SubpassShading = SUBPASS_SHADING_HUAWEI
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(huawei_subpass_shading)]),
    ]),*/
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a pipeline.
    PipelineCreateFlags = PipelineCreateFlags(u32);

    /// The pipeline will not be optimized.
    DISABLE_OPTIMIZATION = DISABLE_OPTIMIZATION,

    /// Derivative pipelines can be created using this pipeline as a base.
    ALLOW_DERIVATIVES = ALLOW_DERIVATIVES,

    /// Create the pipeline by deriving from a base pipeline.
    DERIVATIVE = DERIVATIVE,

    /* TODO: enable
    // TODO: document
    VIEW_INDEX_FROM_DEVICE_INDEX = VIEW_INDEX_FROM_DEVICE_INDEX
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_device_group)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    DISPATCH_BASE = DISPATCH_BASE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_device_group)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    FAIL_ON_PIPELINE_COMPILE_REQUIRED = FAIL_ON_PIPELINE_COMPILE_REQUIRED
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_pipeline_creation_cache_control)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    EARLY_RETURN_ON_FAILURE = EARLY_RETURN_ON_FAILURE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_pipeline_creation_cache_control)]),
    ]),
    */

    /* TODO: enable
    // TODO: document
    RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT = RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_KHR {
        // Provided by VK_KHR_dynamic_rendering with VK_KHR_fragment_shading_rate
    },*/

    /* TODO: enable
    // TODO: document
    RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT = RENDERING_FRAGMENT_DENSITY_MAP_ATTACHMENT_EXT {
        // Provided by VK_KHR_dynamic_rendering with VK_EXT_fragment_density_map
    },*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_NO_NULL_ANY_HIT_SHADERS = RAY_TRACING_NO_NULL_ANY_HIT_SHADERS_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS = RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_NO_NULL_MISS_SHADERS = RAY_TRACING_NO_NULL_MISS_SHADERS_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_NO_NULL_INTERSECTION_SHADERS = RAY_TRACING_NO_NULL_INTERSECTION_SHADERS_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_SKIP_TRIANGLES = RAY_TRACING_SKIP_TRIANGLES_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_SKIP_AABBS = RAY_TRACING_SKIP_AABBS_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY = RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    DEFER_COMPILE = DEFER_COMPILE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    CAPTURE_STATISTICS = CAPTURE_STATISTICS_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_pipeline_executable_properties)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    CAPTURE_INTERNAL_REPRESENTATIONS = CAPTURE_INTERNAL_REPRESENTATIONS_KHR{
        device_extensions: [khr_pipeline_executable_properties],
    },*/

    /* TODO: enable
    // TODO: document
    INDIRECT_BINDABLE = INDIRECT_BINDABLE_NV{
        device_extensions: [nv_device_generated_commands],
    },*/

    /* TODO: enable
    // TODO: document
    LIBRARY = LIBRARY_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_pipeline_library)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    DESCRIPTOR_BUFFER = DESCRIPTOR_BUFFER_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_descriptor_buffer)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RETAIN_LINK_TIME_OPTIMIZATION_INFO = RETAIN_LINK_TIME_OPTIMIZATION_INFO_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_graphics_pipeline_library)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    LINK_TIME_OPTIMIZATION = LINK_TIME_OPTIMIZATION_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_graphics_pipeline_library)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_ALLOW_MOTION = RAY_TRACING_ALLOW_MOTION_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_ray_tracing_motion_blur)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    COLOR_ATTACHMENT_FEEDBACK_LOOP = COLOR_ATTACHMENT_FEEDBACK_LOOP_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_attachment_feedback_loop_layout)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP = DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_attachment_feedback_loop_layout)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_OPACITY_MICROMAP = RAY_TRACING_OPACITY_MICROMAP_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_opacity_micromap)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_DISPLACEMENT_MICROMAP = RAY_TRACING_DISPLACEMENT_MICROMAP_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_displacement_micromap)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    NO_PROTECTED_ACCESS = NO_PROTECTED_ACCESS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_pipeline_protected_access)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    PROTECTED_ACCESS_ONLY = PROTECTED_ACCESS_ONLY_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_pipeline_protected_access)]),
    ]),*/
}

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

        for instruction in spirv.iter_decoration() {
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

        for instruction in entry_point_function.iter_execution_mode() {
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
            .iter_decoration()
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
                .iter_execution_mode()
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

vulkan_enum! {
    #[non_exhaustive]

    /// A particular state value within a graphics pipeline that can be dynamically set by a command
    /// buffer.
    DynamicState = DynamicState(i32);

    /// The elements, but not the count, of
    /// [`ViewportState::viewports`](crate::pipeline::graphics::viewport::ViewportState::viewports).
    ///
    /// Set with
    /// [`set_viewport`](crate::command_buffer::AutoCommandBufferBuilder::set_viewport).
    Viewport = VIEWPORT,

    /// The elements, but not the count, of
    /// [`ViewportState::scissors`](crate::pipeline::graphics::viewport::ViewportState::scissors).
    ///
    /// Set with
    /// [`set_scissor`](crate::command_buffer::AutoCommandBufferBuilder::set_scissor).
    Scissor = SCISSOR,

    /// The value of
    /// [`RasterizationState::line_width`](crate::pipeline::graphics::rasterization::RasterizationState::line_width).
    ///
    /// Set with
    /// [`set_line_width`](crate::command_buffer::AutoCommandBufferBuilder::set_line_width).
    LineWidth = LINE_WIDTH,

    /// The value of
    /// [`RasterizationState::depth_bias`](crate::pipeline::graphics::rasterization::RasterizationState::depth_bias).
    ///
    /// Set with
    /// [`set_depth_bias`](crate::command_buffer::AutoCommandBufferBuilder::set_depth_bias).
    DepthBias = DEPTH_BIAS,

    /// The value of
    /// [`ColorBlendState::blend_constants`](crate::pipeline::graphics::color_blend::ColorBlendState::blend_constants).
    ///
    /// Set with
    /// [`set_blend_constants`](crate::command_buffer::AutoCommandBufferBuilder::set_blend_constants).
    BlendConstants = BLEND_CONSTANTS,

    /// The value of
    /// [`DepthBoundsState::bounds`](crate::pipeline::graphics::depth_stencil::DepthBoundsState::bounds).
    ///
    /// Set with
    /// [`set_depth_bounds`](crate::command_buffer::AutoCommandBufferBuilder::set_depth_bounds).
    DepthBounds = DEPTH_BOUNDS,

    /// The value of
    /// [`StencilOpState::compare_mask`](crate::pipeline::graphics::depth_stencil::StencilOpState::compare_mask)
    /// for both the front and back face.
    ///
    /// Set with
    /// [`set_stencil_compare_mask`](crate::command_buffer::AutoCommandBufferBuilder::set_stencil_compare_mask).
    StencilCompareMask = STENCIL_COMPARE_MASK,

    /// The value of
    /// [`StencilOpState::write_mask`](crate::pipeline::graphics::depth_stencil::StencilOpState::write_mask)
    /// for both the front and back face.
    ///
    /// Set with
    /// [`set_stencil_write_mask`](crate::command_buffer::AutoCommandBufferBuilder::set_stencil_write_mask).
    StencilWriteMask = STENCIL_WRITE_MASK,

    /// The value of
    /// [`StencilOpState::reference`](crate::pipeline::graphics::depth_stencil::StencilOpState::reference)
    /// for both the front and back face.
    ///
    /// Set with
    /// [`set_stencil_reference`](crate::command_buffer::AutoCommandBufferBuilder::set_stencil_reference).
    StencilReference = STENCIL_REFERENCE,

    /// The value of
    /// [`RasterizationState::cull_mode`](crate::pipeline::graphics::rasterization::RasterizationState::cull_mode).
    ///
    /// Set with
    /// [`set_cull_mode`](crate::command_buffer::AutoCommandBufferBuilder::set_cull_mode).
    CullMode = CULL_MODE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// The value of
    /// [`RasterizationState::front_face`](crate::pipeline::graphics::rasterization::RasterizationState::front_face).
    ///
    /// Set with
    /// [`set_front_face`](crate::command_buffer::AutoCommandBufferBuilder::set_front_face).
    FrontFace = FRONT_FACE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// The value of
    /// [`InputAssemblyState::topology`](crate::pipeline::graphics::input_assembly::InputAssemblyState::topology).
    ///
    /// Set with
    /// [`set_primitive_topology`](crate::command_buffer::AutoCommandBufferBuilder::set_primitive_topology).
    PrimitiveTopology = PRIMITIVE_TOPOLOGY
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// Both the elements and the count of
    /// [`ViewportState::viewports`](crate::pipeline::graphics::viewport::ViewportState::viewports).
    ///
    /// Set with
    /// [`set_viewport_with_count`](crate::command_buffer::AutoCommandBufferBuilder::set_viewport_with_count).
    ViewportWithCount = VIEWPORT_WITH_COUNT
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// Both the elements and the count of
    /// [`ViewportState::scissors`](crate::pipeline::graphics::viewport::ViewportState::scissors).
    ///
    /// Set with
    /// [`set_scissor_with_count`](crate::command_buffer::AutoCommandBufferBuilder::set_scissor_with_count).
    ScissorWithCount = SCISSOR_WITH_COUNT
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /* TODO: enable
    // TODO: document
    VertexInputBindingStride = VERTEX_INPUT_BINDING_STRIDE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),*/

    /// The `Option` variant of
    /// [`DepthStencilState::depth`](crate::pipeline::graphics::depth_stencil::DepthStencilState::depth).
    ///
    /// Set with
    /// [`set_depth_test_enable`](crate::command_buffer::AutoCommandBufferBuilder::set_depth_test_enable).
    DepthTestEnable = DEPTH_TEST_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// The value of
    /// [`DepthState::write_enable`](crate::pipeline::graphics::depth_stencil::DepthState::write_enable).
    ///
    /// Set with
    /// [`set_depth_write_enable`](crate::command_buffer::AutoCommandBufferBuilder::set_depth_write_enable).
    DepthWriteEnable = DEPTH_WRITE_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// The value of
    /// [`DepthState::compare_op`](crate::pipeline::graphics::depth_stencil::DepthState::compare_op).
    ///
    /// Set with
    /// [`set_depth_compare_op`](crate::command_buffer::AutoCommandBufferBuilder::set_depth_compare_op).
    DepthCompareOp = DEPTH_COMPARE_OP
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// The `Option` variant of
    /// [`DepthStencilState::depth_bounds`](crate::pipeline::graphics::depth_stencil::DepthStencilState::depth_bounds).
    ///
    /// Set with
    /// [`set_depth_bounds_test_enable`](crate::command_buffer::AutoCommandBufferBuilder::set_depth_bounds_test_enable).
    DepthBoundsTestEnable = DEPTH_BOUNDS_TEST_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// The `Option` variant of
    /// [`DepthStencilState::stencil`](crate::pipeline::graphics::depth_stencil::DepthStencilState::stencil).
    ///
    /// Set with
    /// [`set_stencil_test_enable`](crate::command_buffer::AutoCommandBufferBuilder::set_stencil_test_enable).
    StencilTestEnable = STENCIL_TEST_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// The value of
    /// [`StencilOpState::ops`](crate::pipeline::graphics::depth_stencil::StencilOpState::ops)
    /// for both the front and back face.
    ///
    /// Set with
    /// [`set_stencil_op`](crate::command_buffer::AutoCommandBufferBuilder::set_stencil_op).
    StencilOp = STENCIL_OP
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// The value of
    /// [`RasterizationState::rasterizer_discard_enable`](crate::pipeline::graphics::rasterization::RasterizationState::rasterizer_discard_enable).
    ///
    /// Set with
    /// [`set_rasterizer_discard_enable`](crate::command_buffer::AutoCommandBufferBuilder::set_rasterizer_discard_enable).
    RasterizerDiscardEnable = RASTERIZER_DISCARD_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state2)]),
    ]),

    /// The `Option` variant of
    /// [`RasterizationState::depth_bias`](crate::pipeline::graphics::rasterization::RasterizationState::depth_bias).
    ///
    /// Set with
    /// [`set_depth_bias_enable`](crate::command_buffer::AutoCommandBufferBuilder::set_depth_bias_enable).
    DepthBiasEnable = DEPTH_BIAS_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state2)]),
    ]),

    /// The value of
    /// [`InputAssemblyState::primitive_restart_enable`](crate::pipeline::graphics::input_assembly::InputAssemblyState::primitive_restart_enable).
    ///
    /// Set with
    /// [`set_primitive_restart_enable`](crate::command_buffer::AutoCommandBufferBuilder::set_primitive_restart_enable).
    PrimitiveRestartEnable = PRIMITIVE_RESTART_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state2)]),
    ]),

    /* TODO: enable
    // TODO: document
    ViewportWScaling = VIEWPORT_W_SCALING_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_clip_space_w_scaling)]),
    ]), */

    /// The elements, but not count, of
    /// [`DiscardRectangleState::rectangles`](crate::pipeline::graphics::discard_rectangle::DiscardRectangleState::rectangles).
    ///
    /// Set with
    /// [`set_discard_rectangle`](crate::command_buffer::AutoCommandBufferBuilder::set_discard_rectangle).
    DiscardRectangle = DISCARD_RECTANGLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_discard_rectangles)]),
    ]),

    /* TODO: enable
    // TODO: document
    SampleLocations = SAMPLE_LOCATIONS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_sample_locations)]),
    ]), */

    /* TODO: enable
    // TODO: document
    RayTracingPipelineStackSize = RAY_TRACING_PIPELINE_STACK_SIZE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ViewportShadingRatePalette = VIEWPORT_SHADING_RATE_PALETTE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_shading_rate_image)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ViewportCoarseSampleOrder = VIEWPORT_COARSE_SAMPLE_ORDER_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_shading_rate_image)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ExclusiveScissor = EXCLUSIVE_SCISSOR_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_scissor_exclusive)]),
    ]), */

    /* TODO: enable
    // TODO: document
    FragmentShadingRate = FRAGMENT_SHADING_RATE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_fragment_shading_rate)]),
    ]), */

    /// The value of
    /// [`RasterizationState::line_stipple`](crate::pipeline::graphics::rasterization::RasterizationState::line_stipple).
    ///
    /// Set with
    /// [`set_line_stipple`](crate::command_buffer::AutoCommandBufferBuilder::set_line_stipple).
    LineStipple = LINE_STIPPLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_line_rasterization)]),
    ]),

    /* TODO: enable
    // TODO: document
    VertexInput = VERTEX_INPUT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_vertex_input_dynamic_state)]),
    ]), */

    /// The value of
    /// [`TessellationState::patch_control_points`](crate::pipeline::graphics::tessellation::TessellationState::patch_control_points).
    ///
    /// Set with
    /// [`set_patch_control_points`](crate::command_buffer::AutoCommandBufferBuilder::set_patch_control_points).
    PatchControlPoints = PATCH_CONTROL_POINTS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state2)]),
    ]),

    /// The value of
    /// [`ColorBlendState::logic_op`](crate::pipeline::graphics::color_blend::ColorBlendState::logic_op).
    ///
    /// Set with
    /// [`set_logic_op`](crate::command_buffer::AutoCommandBufferBuilder::set_logic_op).
    LogicOp = LOGIC_OP_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state2)]),
    ]),

    /// The value of
    /// [`ColorBlendAttachmentState::color_write_enable`](crate::pipeline::graphics::color_blend::ColorBlendAttachmentState::color_write_enable)
    /// for every attachment.
    ///
    /// Set with
    /// [`set_color_write_enable`](crate::command_buffer::AutoCommandBufferBuilder::set_color_write_enable).
    ColorWriteEnable = COLOR_WRITE_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_color_write_enable)]),
    ]),

    /* TODO: enable
    // TODO: document
    TessellationDomainOrigin = TESSELLATION_DOMAIN_ORIGIN_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    DepthClampEnable = DEPTH_CLAMP_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    PolygonMode = POLYGON_MODE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    RasterizationSamples = RASTERIZATION_SAMPLES_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    SampleMask = SAMPLE_MASK_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    AlphaToCoverageEnable = ALPHA_TO_COVERAGE_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    AlphaToOneEnable = ALPHA_TO_ONE_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    LogicOpEnable = LOGIC_OP_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ColorBlendEnable = COLOR_BLEND_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ColorBlendEquation = COLOR_BLEND_EQUATION_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ColorWriteMask = COLOR_WRITE_MASK_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    RasterizationStream = RASTERIZATION_STREAM_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ConservativeRasterizationMode = CONSERVATIVE_RASTERIZATION_MODE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ExtraPrimitiveOverestimationSize = EXTRA_PRIMITIVE_OVERESTIMATION_SIZE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    DepthClipEnable = DEPTH_CLIP_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    SampleLocationsEnable = SAMPLE_LOCATIONS_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ColorBlendAdvanced = COLOR_BLEND_ADVANCED_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ProvokingVertexMode = PROVOKING_VERTEX_MODE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    LineRasterizationMode = LINE_RASTERIZATION_MODE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    LineStippleEnable = LINE_STIPPLE_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    DepthClipNegativeOneToOne = DEPTH_CLIP_NEGATIVE_ONE_TO_ONE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ViewportWScalingEnable = VIEWPORT_W_SCALING_ENABLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ViewportSwizzle = VIEWPORT_SWIZZLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    CoverageToColorEnable = COVERAGE_TO_COLOR_ENABLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    CoverageToColorLocation = COVERAGE_TO_COLOR_LOCATION_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    CoverageModulationMode = COVERAGE_MODULATION_MODE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    CoverageModulationTableEnable = COVERAGE_MODULATION_TABLE_ENABLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    CoverageModulationTable = COVERAGE_MODULATION_TABLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ShadingRateImageEnable = SHADING_RATE_IMAGE_ENABLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    RepresentativeFragmentTestEnable = REPRESENTATIVE_FRAGMENT_TEST_ENABLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */

    /* TODO: enable
    // TODO: document
    CoverageReductionMode = COVERAGE_REDUCTION_MODE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]), */
}
