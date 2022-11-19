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
use crate::{device::DeviceOwned, macros::vulkan_enum, shader::DescriptorBindingRequirements};
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

    /*
    // TODO: document
    RayTracing = RAY_TRACING_KHR {
        device_extensions: [khr_ray_tracing_pipeline, nv_ray_tracing],
    },

    // TODO: document
    SubpassShading = SUBPASS_SHADING_HUAWEI {
        device_extensions: [huawei_subpass_shading],
    },
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
