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
    device::DeviceOwned,
    macros::{vulkan_bitflags, vulkan_enum},
    shader::DescriptorBindingRequirements,
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
    RayTracing = RAY_TRACING_KHR {
        device_extensions: [khr_ray_tracing_pipeline, nv_ray_tracing],
    },*/

    /* TODO: enable
    // TODO: document
    SubpassShading = SUBPASS_SHADING_HUAWEI {
        device_extensions: [huawei_subpass_shading],
    },*/
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags that control how a pipeline is created.
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
    VIEW_INDEX_FROM_DEVICE_INDEX = VIEW_INDEX_FROM_DEVICE_INDEX {
        api_version: V1_1,
    },*/

    /* TODO: enable
    // TODO: document
    DISPATCH_BASE = DISPATCH_BASE {
        api_version: V1_1,
    },*/

    /* TODO: enable
    // TODO: document
    FAIL_ON_PIPELINE_COMPILE_REQUIRED = FAIL_ON_PIPELINE_COMPILE_REQUIRED {
        api_version: V1_3,
    },*/

    /* TODO: enable
    // TODO: document
    EARLY_RETURN_ON_FAILURE = EARLY_RETURN_ON_FAILURE {
        api_version: V1_3,
    },
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
    RAY_TRACING_NO_NULL_ANY_HIT_SHADERS = RAY_TRACING_NO_NULL_ANY_HIT_SHADERS_KHR {
        device_extensions: [khr_ray_tracing_pipeline],
    },*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS = RAY_TRACING_NO_NULL_CLOSEST_HIT_SHADERS_KHR {
        device_extensions: [khr_ray_tracing_pipeline],
    },*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_NO_NULL_MISS_SHADERS = RAY_TRACING_NO_NULL_MISS_SHADERS_KHR {
        device_extensions: [khr_ray_tracing_pipeline],
    },*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_NO_NULL_INTERSECTION_SHADERS = RAY_TRACING_NO_NULL_INTERSECTION_SHADERS_KHR {
        device_extensions: [khr_ray_tracing_pipeline],
    },*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_SKIP_TRIANGLES = RAY_TRACING_SKIP_TRIANGLES_KHR {
        device_extensions: [khr_ray_tracing_pipeline],
    },*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_SKIP_AABBS = RAY_TRACING_SKIP_AABBS_KHR {
        device_extensions: [khr_ray_tracing_pipeline],
    },*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY = RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_KHR {
        device_extensions: [khr_ray_tracing_pipeline],
    },*/

    /* TODO: enable
    // TODO: document
    DEFER_COMPILE = DEFER_COMPILE_NV {
        device_extensions: [nv_ray_tracing],
    },*/

    /* TODO: enable
    // TODO: document
    CAPTURE_STATISTICS = CAPTURE_STATISTICS_KHR {
        device_extensions: [khr_pipeline_executable_properties],
    },*/

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
    LIBRARY = LIBRARY_KHR {
        device_extensions: [khr_pipeline_library],
    },*/

    /* TODO: enable
    // TODO: document
    DESCRIPTOR_BUFFER = DESCRIPTOR_BUFFER_EXT {
        device_extensions: [ext_descriptor_buffer],
    },*/

    /* TODO: enable
    // TODO: document
    RETAIN_LINK_TIME_OPTIMIZATION_INFO = RETAIN_LINK_TIME_OPTIMIZATION_INFO_EXT {
        device_extensions: [ext_graphics_pipeline_library],
    },*/

    /* TODO: enable
    // TODO: document
    LINK_TIME_OPTIMIZATION = LINK_TIME_OPTIMIZATION_EXT {
        device_extensions: [ext_graphics_pipeline_library],
    },*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_ALLOW_MOTION = RAY_TRACING_ALLOW_MOTION_NV {
        device_extensions: [nv_ray_tracing_motion_blur],
    },*/

    /* TODO: enable
    // TODO: document
    COLOR_ATTACHMENT_FEEDBACK_LOOP = COLOR_ATTACHMENT_FEEDBACK_LOOP_EXT {
        device_extensions: [ext_attachment_feedback_loop_layout],
    },*/

    /* TODO: enable
    // TODO: document
    DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP = DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_EXT {
        device_extensions: [ext_attachment_feedback_loop_layout],
    },*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_OPACITY_MICROMAP = RAY_TRACING_OPACITY_MICROMAP_EXT {
        device_extensions: [ext_opacity_micromap],
    },*/

    /* TODO: enable
    // TODO: document
    RAY_TRACING_DISPLACEMENT_MICROMAP = RAY_TRACING_DISPLACEMENT_MICROMAP_NV {
        device_extensions: [nv_displacement_micromap],
    },*/

    /* TODO: enable
    // TODO: document
    NO_PROTECTED_ACCESS = NO_PROTECTED_ACCESS_EXT {
        device_extensions: [ext_pipeline_protected_access],
    },*/

    /* TODO: enable
    // TODO: document
    PROTECTED_ACCESS_ONLY = PROTECTED_ACCESS_ONLY_EXT {
        device_extensions: [ext_pipeline_protected_access],
    },*/
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
    CullMode = CULL_MODE {
        api_version: V1_3,
        device_extensions: [ext_extended_dynamic_state],
    },

    // TODO: document
    FrontFace = FRONT_FACE {
        api_version: V1_3,
        device_extensions: [ext_extended_dynamic_state],
    },

    // TODO: document
    PrimitiveTopology = PRIMITIVE_TOPOLOGY {
        api_version: V1_3,
        device_extensions: [ext_extended_dynamic_state],
    },

    // TODO: document
    ViewportWithCount = VIEWPORT_WITH_COUNT {
        api_version: V1_3,
        device_extensions: [ext_extended_dynamic_state],
    },

    // TODO: document
    ScissorWithCount = SCISSOR_WITH_COUNT {
        api_version: V1_3,
        device_extensions: [ext_extended_dynamic_state],
    },

    // TODO: document
    VertexInputBindingStride = VERTEX_INPUT_BINDING_STRIDE {
        api_version: V1_3,
        device_extensions: [ext_extended_dynamic_state],
    },

    // TODO: document
    DepthTestEnable = DEPTH_TEST_ENABLE {
        api_version: V1_3,
        device_extensions: [ext_extended_dynamic_state],
    },

    // TODO: document
    DepthWriteEnable = DEPTH_WRITE_ENABLE {
        api_version: V1_3,
        device_extensions: [ext_extended_dynamic_state],
    },

    // TODO: document
    DepthCompareOp = DEPTH_COMPARE_OP {
        api_version: V1_3,
        device_extensions: [ext_extended_dynamic_state],
    },

    // TODO: document
    DepthBoundsTestEnable = DEPTH_BOUNDS_TEST_ENABLE {
        api_version: V1_3,
        device_extensions: [ext_extended_dynamic_state],
    },

    // TODO: document
    StencilTestEnable = STENCIL_TEST_ENABLE {
        api_version: V1_3,
        device_extensions: [ext_extended_dynamic_state],
    },

    // TODO: document
    StencilOp = STENCIL_OP {
        api_version: V1_3,
        device_extensions: [ext_extended_dynamic_state],
    },

    // TODO: document
    RasterizerDiscardEnable = RASTERIZER_DISCARD_ENABLE {
        api_version: V1_3,
        device_extensions: [ext_extended_dynamic_state2],
    },

    // TODO: document
    DepthBiasEnable = DEPTH_BIAS_ENABLE {
        api_version: V1_3,
        device_extensions: [ext_extended_dynamic_state2],
    },

    // TODO: document
    PrimitiveRestartEnable = PRIMITIVE_RESTART_ENABLE {
        api_version: V1_3,
        device_extensions: [ext_extended_dynamic_state2],
    },

    // TODO: document
    ViewportWScaling = VIEWPORT_W_SCALING_NV {
        device_extensions: [nv_clip_space_w_scaling],
    },

    // TODO: document
    DiscardRectangle = DISCARD_RECTANGLE_EXT {
        device_extensions: [ext_discard_rectangles],
    },

    // TODO: document
    SampleLocations = SAMPLE_LOCATIONS_EXT {
        device_extensions: [ext_sample_locations],
    },

    // TODO: document
    RayTracingPipelineStackSize = RAY_TRACING_PIPELINE_STACK_SIZE_KHR {
        device_extensions: [khr_ray_tracing_pipeline],
    },

    // TODO: document
    ViewportShadingRatePalette = VIEWPORT_SHADING_RATE_PALETTE_NV {
        device_extensions: [nv_shading_rate_image],
    },

    // TODO: document
    ViewportCoarseSampleOrder = VIEWPORT_COARSE_SAMPLE_ORDER_NV {
        device_extensions: [nv_shading_rate_image],
    },

    // TODO: document
    ExclusiveScissor = EXCLUSIVE_SCISSOR_NV {
        device_extensions: [nv_scissor_exclusive],
    },

    // TODO: document
    FragmentShadingRate = FRAGMENT_SHADING_RATE_KHR {
        device_extensions: [khr_fragment_shading_rate],
    },

    // TODO: document
    LineStipple = LINE_STIPPLE_EXT {
        device_extensions: [ext_line_rasterization],
    },

    // TODO: document
    VertexInput = VERTEX_INPUT_EXT {
        device_extensions: [ext_vertex_input_dynamic_state],
    },

    // TODO: document
    PatchControlPoints = PATCH_CONTROL_POINTS_EXT {
        device_extensions: [ext_extended_dynamic_state2],
    },

    // TODO: document
    LogicOp = LOGIC_OP_EXT {
        device_extensions: [ext_extended_dynamic_state2],
    },

    // TODO: document
    ColorWriteEnable = COLOR_WRITE_ENABLE_EXT {
        device_extensions: [ext_color_write_enable],
    },

    // TODO: document
    TessellationDomainOrigin = TESSELLATION_DOMAIN_ORIGIN_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    DepthClampEnable = DEPTH_CLAMP_ENABLE_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    PolygonMode = POLYGON_MODE_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    RasterizationSamples = RASTERIZATION_SAMPLES_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    SampleMask = SAMPLE_MASK_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    AlphaToCoverageEnable = ALPHA_TO_COVERAGE_ENABLE_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    AlphaToOneEnable = ALPHA_TO_ONE_ENABLE_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    LogicOpEnable = LOGIC_OP_ENABLE_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    ColorBlendEnable = COLOR_BLEND_ENABLE_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    ColorBlendEquation = COLOR_BLEND_EQUATION_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    ColorWriteMask = COLOR_WRITE_MASK_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    RasterizationStream = RASTERIZATION_STREAM_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    ConservativeRasterizationMode = CONSERVATIVE_RASTERIZATION_MODE_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    ExtraPrimitiveOverestimationSize = EXTRA_PRIMITIVE_OVERESTIMATION_SIZE_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    DepthClipEnable = DEPTH_CLIP_ENABLE_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    SampleLocationsEnable = SAMPLE_LOCATIONS_ENABLE_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    ColorBlendAdvanced = COLOR_BLEND_ADVANCED_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    ProvokingVertexMode = PROVOKING_VERTEX_MODE_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    LineRasterizationMode = LINE_RASTERIZATION_MODE_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    LineStippleEnable = LINE_STIPPLE_ENABLE_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    DepthClipNegativeOneToOne = DEPTH_CLIP_NEGATIVE_ONE_TO_ONE_EXT {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    ViewportWScalingEnable = VIEWPORT_W_SCALING_ENABLE_NV {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    ViewportSwizzle = VIEWPORT_SWIZZLE_NV {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    CoverageToColorEnable = COVERAGE_TO_COLOR_ENABLE_NV {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    CoverageToColorLocation = COVERAGE_TO_COLOR_LOCATION_NV {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    CoverageModulationMode = COVERAGE_MODULATION_MODE_NV {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    CoverageModulationTableEnable = COVERAGE_MODULATION_TABLE_ENABLE_NV {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    CoverageModulationTable = COVERAGE_MODULATION_TABLE_NV {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    ShadingRateImageEnable = SHADING_RATE_IMAGE_ENABLE_NV {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    RepresentativeFragmentTestEnable = REPRESENTATIVE_FRAGMENT_TEST_ENABLE_NV {
        device_extensions: [ext_extended_dynamic_state3],
    },

    // TODO: document
    CoverageReductionMode = COVERAGE_REDUCTION_MODE_NV {
        device_extensions: [ext_extended_dynamic_state3],
    },
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
