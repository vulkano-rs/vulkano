//! Describes a processing operation that will execute on the Vulkan device.
//!
//! In Vulkan, before you can add a draw or a compute command to a command buffer you have to
//! create a *pipeline object* that describes this command.
//!
//! When you create a pipeline object, the implementation will usually generate some GPU machine
//! code that will execute the operation (similar to a compiler that generates an executable for
//! the CPU). Consequently it is a CPU-intensive operation that should be performed at
//! initialization or during a loading screen.

pub use self::{
    compute::ComputePipeline, graphics::GraphicsPipeline, layout::PipelineLayout, shader::*,
};
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
pub(crate) mod shader;

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

vulkan_enum! {
    #[non_exhaustive]

    /// A particular state value within a pipeline that can be dynamically set by a command buffer.
    ///
    /// Whenever a particular state is set to be dynamic while creating the pipeline,
    /// the corresponding predefined value in the pipeline's create info is ignored, unless
    /// specified otherwise here.
    ///
    /// If the dynamic state is used to enable/disable a certain functionality,
    /// and the value in the create info is an `Option`
    /// (for example, [`DynamicState::DepthTestEnable`] and [`DepthStencilState::depth`]),
    /// then that `Option` must be `Some` when creating the pipeline,
    /// in order to provide settings to use when the functionality is enabled.
    ///
    /// [`DepthStencilState::depth`]: (crate::pipeline::graphics::depth_stencil::DepthStencilState::depth)
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
    /// [`ColorBlendState::blend_constants`](graphics::color_blend::ColorBlendState::blend_constants).
    ///
    /// Set with
    /// [`set_blend_constants`](crate::command_buffer::AutoCommandBufferBuilder::set_blend_constants).
    BlendConstants = BLEND_CONSTANTS,

    /// The value, but not the `Option` variant, of
    /// [`DepthStencilState::depth_bounds`](crate::pipeline::graphics::depth_stencil::DepthStencilState::depth_bounds).
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
    /// [`RasterizationState::cull_mode`](graphics::rasterization::RasterizationState::cull_mode).
    ///
    /// Set with
    /// [`set_cull_mode`](crate::command_buffer::AutoCommandBufferBuilder::set_cull_mode).
    CullMode = CULL_MODE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// The value of
    /// [`RasterizationState::front_face`](graphics::rasterization::RasterizationState::front_face).
    ///
    /// Set with
    /// [`set_front_face`](crate::command_buffer::AutoCommandBufferBuilder::set_front_face).
    FrontFace = FRONT_FACE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state)]),
    ]),

    /// The value of
    /// [`InputAssemblyState::topology`](graphics::input_assembly::InputAssemblyState::topology).
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
    /// [`InputAssemblyState::primitive_restart_enable`](graphics::input_assembly::InputAssemblyState::primitive_restart_enable).
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

    /// The `Option` variant of
    /// [`GraphicsPipelineCreateInfo::vertex_input_state`](crate::pipeline::graphics::GraphicsPipelineCreateInfo::vertex_input_state).
    ///
    /// Set with
    /// [`set_vertex_input`](crate::command_buffer::AutoCommandBufferBuilder::set_vertex_input).
    VertexInput = VERTEX_INPUT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_vertex_input_dynamic_state)]),
    ]),

    /// The value of
    /// [`TessellationState::patch_control_points`](graphics::tessellation::TessellationState::patch_control_points).
    ///
    /// Set with
    /// [`set_patch_control_points`](crate::command_buffer::AutoCommandBufferBuilder::set_patch_control_points).
    PatchControlPoints = PATCH_CONTROL_POINTS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state2)]),
    ]),

    /// The value of
    /// [`ColorBlendState::logic_op`](graphics::color_blend::ColorBlendState::logic_op).
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

    /// The value of
    /// [`ConservativeRasterizationState::mode`](crate::pipeline::graphics::rasterization::RasterizationConservativeState::mode)
    ///
    /// Set with
    /// [`set_conservative_rasterization_mode`](crate::command_buffer::AutoCommandBufferBuilder::set_conservative_rasterization_mode).
    ConservativeRasterizationMode = CONSERVATIVE_RASTERIZATION_MODE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

    /// The value of
    /// [`ConservativeRasterizationState::overestimation_size`](crate::pipeline::graphics::rasterization::RasterizationConservativeState::overestimation_size)
    ///
    /// Set with
    /// [`set_extra_primitive_overestimation_size`](crate::command_buffer::AutoCommandBufferBuilder::set_extra_primitive_overestimation_size).
    ExtraPrimitiveOverestimationSize = EXTRA_PRIMITIVE_OVERESTIMATION_SIZE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_extended_dynamic_state3)]),
    ]),

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
