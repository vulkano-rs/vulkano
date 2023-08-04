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
        DescriptorBindingRequirements, EntryPoint, ShaderStage, ShaderStages,
        SpecializationConstant,
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

    /* TODO: enable
    // TODO: document
    ALLOW_DERIVATIVES = ALLOW_DERIVATIVES,*/

    /* TODO: enable
    // TODO: document
    DERIVATIVE = DERIVATIVE,*/

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

    /// The shader entry point for the stage.
    ///
    /// There is no default value.
    pub entry_point: EntryPoint,

    /// Values for the specialization constants in the shader, indexed by their `constant_id`.
    ///
    /// Specialization constants are constants whose value can be overridden when you create
    /// a pipeline. When provided, they must have the same type as defined in the shader.
    /// Constants that are not given a value here will have the default value that was specified
    /// for them in the shader code.
    ///
    /// The default value is empty.
    pub specialization_info: HashMap<u32, SpecializationConstant>,

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
            specialization_info: HashMap::default(),
            required_subgroup_size: None,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            ref entry_point,
            ref specialization_info,
            ref required_subgroup_size,
            _ne: _,
        } = self;

        flags
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "flags".into(),
                vuids: &["VUID-VkPipelineShaderStageCreateInfo-flags-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        let entry_point_info = entry_point.info();
        let stage_enum = ShaderStage::from(&entry_point_info.execution);

        stage_enum
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "entry_point.info().execution".into(),
                vuids: &["VUID-VkPipelineShaderStageCreateInfo-stage-parameter"],
                ..ValidationError::from_requirement(err)
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

        for (&constant_id, provided_value) in specialization_info {
            // Per `VkSpecializationMapEntry` spec:
            // "If a constantID value is not a specialization constant ID used in the shader,
            // that map entry does not affect the behavior of the pipeline."
            // We *may* want to be stricter than this for the sake of catching user errors?
            if let Some(default_value) = entry_point_info.specialization_constants.get(&constant_id)
            {
                // Check for equal types rather than only equal size.
                if !provided_value.eq_type(default_value) {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`specialization_info[{0}]` does not have the same type as \
                            `entry_point.info().specialization_constants[{0}]`",
                            constant_id
                        )
                        .into(),
                        vuids: &["VUID-VkSpecializationMapEntry-constantID-00776"],
                        ..Default::default()
                    }));
                }
            }
        }

        let local_size = entry_point_info.local_size(specialization_info)?;
        let workgroup_size = local_size.map(|[x, y, z]| x * y * z);
        if workgroup_size == Some(0) {
            todo!("LocalSize {:?} cannot be 0!", local_size.unwrap());
        }

        if let Some(required_subgroup_size) = required_subgroup_size {
            validate_required_subgroup_size(
                device,
                stage_enum,
                workgroup_size,
                *required_subgroup_size,
            )?;
        }

        Ok(())
    }
}

pub(crate) fn validate_required_subgroup_size(
    device: &Device,
    stage: ShaderStage,
    workgroup_size: Option<u32>,
    subgroup_size: u32,
) -> Result<(), Box<ValidationError>> {
    if !device.enabled_features().subgroup_size_control {
        return Err(Box::new(ValidationError {
            context: "required_subgroup_size".into(),
            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                "subgroup_size_control",
            )])]),
            vuids: &["VUID-VkPipelineShaderStageCreateInfo-pNext-02755"],
            ..Default::default()
        }));
    }
    let stages: ShaderStages = [stage].into_iter().collect();
    let properties = device.physical_device().properties();
    if !properties
        .required_subgroup_size_stages
        .unwrap_or_default()
        .contains(stages)
    {
        return Err(Box::new(ValidationError {
            context: "required_subgroup_size".into(),
            problem: format!("`shader stage` {stage:?} is not in `required_subgroup_size_stages`")
                .into(),
            vuids: &["VUID-VkPipelineShaderStageCreateInfo-pNext-02755"],
            ..Default::default()
        }));
    }
    if !subgroup_size.is_power_of_two() {
        return Err(Box::new(ValidationError {
            context: "required_subgroup_size".into(),
            problem: format!("`subgroup_size` {subgroup_size} is not a power of 2").into(),
            vuids: &["VUID-VkPipelineShaderStageRequiredSubgroupSizeCreateInfo-requiredSubgroupSize-02760"],
            ..Default::default()
        }));
    }
    let min_subgroup_size = properties.min_subgroup_size.unwrap_or(1);
    if subgroup_size < min_subgroup_size {
        return Err(Box::new(ValidationError {
            context: "required_subgroup_size".into(),
            problem: format!(
                "`subgroup_size` {subgroup_size} is less than `min_subgroup_size` {min_subgroup_size}"
            )
            .into(),
            vuids: &["VUID-VkPipelineShaderStageRequiredSubgroupSizeCreateInfo-requiredSubgroupSize-02761"],
            ..Default::default()
        }));
    }
    let max_subgroup_size = properties.max_subgroup_size.unwrap_or(128);
    if subgroup_size > max_subgroup_size {
        return Err(Box::new(ValidationError {
            context: "required_subgroup_size".into(),
            problem:
                format!("`subgroup_size` {subgroup_size} is greater than `max_subgroup_size` {max_subgroup_size}")
                    .into(),
            vuids: &["VUID-VkPipelineShaderStageRequiredSubgroupSizeCreateInfo-requiredSubgroupSize-02762"],
            ..Default::default()
        }));
    }
    if let Some(workgroup_size) = workgroup_size {
        let max_compute_workgroup_subgroups = properties
            .max_compute_workgroup_subgroups
            .unwrap_or_default();
        if max_compute_workgroup_subgroups * subgroup_size < workgroup_size {
            return Err(Box::new(ValidationError {
                context: "required_subgroup_size".into(),
                problem: format!("`subgroup_size` {subgroup_size} creates more than {max_compute_workgroup_subgroups} subgroups").into(),
                vuids: &["VUID-VkPipelineShaderStageCreateInfo-pNext-02756"],
                ..Default::default()
            }));
        }
    }
    Ok(())
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

    // TODO: document
    Viewport = VIEWPORT,

    // TODO: document
    Scissor = SCISSOR,

    // TODO: document
    LineWidth = LINE_WIDTH,

    // TODO: document
    DepthBias = DEPTH_BIAS,

    // TODO: document
    BlendConstants = BLEND_CONSTANTS,

    // TODO: document
    DepthBounds = DEPTH_BOUNDS,

    // TODO: document
    StencilCompareMask = STENCIL_COMPARE_MASK,

    // TODO: document
    StencilWriteMask = STENCIL_WRITE_MASK,

    // TODO: document
    StencilReference = STENCIL_REFERENCE,

    // TODO: document
    CullMode = CULL_MODE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    // TODO: document
    FrontFace = FRONT_FACE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    // TODO: document
    PrimitiveTopology = PRIMITIVE_TOPOLOGY
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    // TODO: document
    ViewportWithCount = VIEWPORT_WITH_COUNT
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    // TODO: document
    ScissorWithCount = SCISSOR_WITH_COUNT
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    // TODO: document
    VertexInputBindingStride = VERTEX_INPUT_BINDING_STRIDE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    // TODO: document
    DepthTestEnable = DEPTH_TEST_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    // TODO: document
    DepthWriteEnable = DEPTH_WRITE_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    // TODO: document
    DepthCompareOp = DEPTH_COMPARE_OP
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    // TODO: document
    DepthBoundsTestEnable = DEPTH_BOUNDS_TEST_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    // TODO: document
    StencilTestEnable = STENCIL_TEST_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    // TODO: document
    StencilOp = STENCIL_OP
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    // TODO: document
    RasterizerDiscardEnable = RASTERIZER_DISCARD_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state2)]),
    ]),

    // TODO: document
    DepthBiasEnable = DEPTH_BIAS_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state2)]),
    ]),

    // TODO: document
    PrimitiveRestartEnable = PRIMITIVE_RESTART_ENABLE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state2)]),
    ]),

    // TODO: document
    ViewportWScaling = VIEWPORT_W_SCALING_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_clip_space_w_scaling)]),
    ]),

    // TODO: document
    DiscardRectangle = DISCARD_RECTANGLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_discard_rectangles)]),
    ]),

    // TODO: document
    SampleLocations = SAMPLE_LOCATIONS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_sample_locations)]),
    ]),

    // TODO: document
    RayTracingPipelineStackSize = RAY_TRACING_PIPELINE_STACK_SIZE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
    ]),

    // TODO: document
    ViewportShadingRatePalette = VIEWPORT_SHADING_RATE_PALETTE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_shading_rate_image)]),
    ]),

    // TODO: document
    ViewportCoarseSampleOrder = VIEWPORT_COARSE_SAMPLE_ORDER_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_shading_rate_image)]),
    ]),

    // TODO: document
    ExclusiveScissor = EXCLUSIVE_SCISSOR_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_scissor_exclusive)]),
    ]),

    // TODO: document
    FragmentShadingRate = FRAGMENT_SHADING_RATE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_fragment_shading_rate)]),
    ]),

    // TODO: document
    LineStipple = LINE_STIPPLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_line_rasterization)]),
    ]),

    // TODO: document
    VertexInput = VERTEX_INPUT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_vertex_input_dynamic_state)]),
    ]),

    // TODO: document
    PatchControlPoints = PATCH_CONTROL_POINTS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state2)]),
    ]),

    // TODO: document
    LogicOp = LOGIC_OP_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state2)]),
    ]),

    // TODO: document
    ColorWriteEnable = COLOR_WRITE_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_color_write_enable)]),
    ]),

    // TODO: document
    TessellationDomainOrigin = TESSELLATION_DOMAIN_ORIGIN_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    DepthClampEnable = DEPTH_CLAMP_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    PolygonMode = POLYGON_MODE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    RasterizationSamples = RASTERIZATION_SAMPLES_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    SampleMask = SAMPLE_MASK_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    AlphaToCoverageEnable = ALPHA_TO_COVERAGE_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    AlphaToOneEnable = ALPHA_TO_ONE_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    LogicOpEnable = LOGIC_OP_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    ColorBlendEnable = COLOR_BLEND_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    ColorBlendEquation = COLOR_BLEND_EQUATION_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    ColorWriteMask = COLOR_WRITE_MASK_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    RasterizationStream = RASTERIZATION_STREAM_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    ConservativeRasterizationMode = CONSERVATIVE_RASTERIZATION_MODE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    ExtraPrimitiveOverestimationSize = EXTRA_PRIMITIVE_OVERESTIMATION_SIZE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    DepthClipEnable = DEPTH_CLIP_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    SampleLocationsEnable = SAMPLE_LOCATIONS_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    ColorBlendAdvanced = COLOR_BLEND_ADVANCED_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    ProvokingVertexMode = PROVOKING_VERTEX_MODE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    LineRasterizationMode = LINE_RASTERIZATION_MODE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    LineStippleEnable = LINE_STIPPLE_ENABLE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    DepthClipNegativeOneToOne = DEPTH_CLIP_NEGATIVE_ONE_TO_ONE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    ViewportWScalingEnable = VIEWPORT_W_SCALING_ENABLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    ViewportSwizzle = VIEWPORT_SWIZZLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    CoverageToColorEnable = COVERAGE_TO_COLOR_ENABLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    CoverageToColorLocation = COVERAGE_TO_COLOR_LOCATION_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    CoverageModulationMode = COVERAGE_MODULATION_MODE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    CoverageModulationTableEnable = COVERAGE_MODULATION_TABLE_ENABLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    CoverageModulationTable = COVERAGE_MODULATION_TABLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    ShadingRateImageEnable = SHADING_RATE_IMAGE_ENABLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    RepresentativeFragmentTestEnable = REPRESENTATIVE_FRAGMENT_TEST_ENABLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    // TODO: document
    CoverageReductionMode = COVERAGE_REDUCTION_MODE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),
}

/// Specifies how a dynamic state is handled by a graphics pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateMode<F> {
    /// The pipeline has a fixed value for this state. Previously set dynamic state will be lost
    /// when binding it, and will have to be re-set after binding a pipeline that uses it.
    Fixed(F),

    /// The pipeline expects a dynamic value to be set by a command buffer. Previously set dynamic
    /// state is not disturbed when binding it.
    Dynamic,
}

impl<T> From<Option<T>> for StateMode<T> {
    fn from(val: Option<T>) -> Self {
        match val {
            Some(x) => StateMode::Fixed(x),
            None => StateMode::Dynamic,
        }
    }
}

impl<T> From<StateMode<T>> for Option<T> {
    fn from(val: StateMode<T>) -> Self {
        match val {
            StateMode::Fixed(x) => Some(x),
            StateMode::Dynamic => None,
        }
    }
}

/// A variant of `StateMode` that is used for cases where some value is still needed when the state
/// is dynamic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartialStateMode<F, D> {
    Fixed(F),
    Dynamic(D),
}
