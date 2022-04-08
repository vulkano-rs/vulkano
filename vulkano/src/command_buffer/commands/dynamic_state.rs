// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    command_buffer::{
        synced::{Command, SyncCommandBufferBuilder},
        sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder,
    },
    device::DeviceOwned,
    pipeline::{
        graphics::{
            color_blend::LogicOp,
            depth_stencil::{CompareOp, StencilFaces, StencilOp, StencilOps},
            input_assembly::PrimitiveTopology,
            rasterization::{CullMode, DepthBias, FrontFace, LineStipple},
            viewport::{Scissor, Viewport},
        },
        DynamicState,
    },
    Version,
};
use parking_lot::Mutex;
use smallvec::SmallVec;

/// # Commands to set dynamic state for pipelines.
///
/// These commands require a queue with a pipeline type that uses the given state.
impl<L, P> AutoCommandBufferBuilder<L, P> {
    // Helper function for dynamic state setting.
    fn has_fixed_state(&self, state: DynamicState) -> bool {
        self.state()
            .pipeline_graphics()
            .map(|pipeline| matches!(pipeline.dynamic_state(state), Some(false)))
            .unwrap_or(false)
    }

    /// Sets the dynamic blend constants for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    pub fn set_blend_constants(&mut self, constants: [f32; 4]) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            !self.has_fixed_state(DynamicState::BlendConstants),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        unsafe {
            self.inner.set_blend_constants(constants);
        }

        self
    }

    /// Sets whether dynamic color writes should be enabled for each attachment in the
    /// framebuffer.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the [`color_write_enable`](crate::device::Features::color_write_enable)
    ///   feature is not enabled on the device.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    /// - If there is a graphics pipeline with color blend state bound, `enables.len()` must equal
    /// - [`attachments.len()`](crate::pipeline::graphics::color_blend::ColorBlendState::attachments).
    #[inline]
    pub fn set_color_write_enable<I>(&mut self, enables: I) -> &mut Self
    where
        I: IntoIterator<Item = bool>,
        I::IntoIter: ExactSizeIterator,
    {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            self.device().enabled_features().color_write_enable,
            "the color_write_enable feature must be enabled on the device"
        );
        assert!(
            !self.has_fixed_state(DynamicState::ColorWriteEnable),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        let enables = enables.into_iter();

        if let Some(color_blend_state) = self
            .state()
            .pipeline_graphics()
            .and_then(|pipeline| pipeline.color_blend_state())
        {
            assert!(
					enables.len() == color_blend_state.attachments.len(),
					"if there is a graphics pipeline with color blend state bound, enables.len() must equal attachments.len()"
				);
        }

        unsafe {
            self.inner.set_color_write_enable(enables);
        }

        self
    }

    /// Sets the dynamic cull mode for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the device API version is less than 1.3 and the
    ///   [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature is
    ///   not enabled on the device.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    #[inline]
    pub fn set_cull_mode(&mut self, cull_mode: CullMode) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            self.device().api_version() >= Version::V1_3
                || self.device().enabled_features().extended_dynamic_state,
            "the extended_dynamic_state feature must be enabled on the device"
        );
        assert!(
            !self.has_fixed_state(DynamicState::CullMode),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        unsafe {
            self.inner.set_cull_mode(cull_mode);
        }

        self
    }

    /// Sets the dynamic depth bias values for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    /// - If the [`depth_bias_clamp`](crate::device::Features::depth_bias_clamp)
    ///   feature is not enabled on the device, panics if `clamp` is not 0.0.
    #[inline]
    pub fn set_depth_bias(
        &mut self,
        constant_factor: f32,
        clamp: f32,
        slope_factor: f32,
    ) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            !self.has_fixed_state(DynamicState::DepthBias),
            "the currently bound graphics pipeline must not contain this state internally"
        );
        assert!(
            clamp == 0.0 || self.device().enabled_features().depth_bias_clamp,
            "if the depth_bias_clamp feature is not enabled, clamp must be 0.0"
        );

        unsafe {
            self.inner
                .set_depth_bias(constant_factor, clamp, slope_factor);
        }

        self
    }

    /// Sets whether dynamic depth bias is enabled for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the device API version is less than 1.3 and the
    ///   [`extended_dynamic_state2`](crate::device::Features::extended_dynamic_state2) feature is
    ///   not enabled on the device.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    #[inline]
    pub fn set_depth_bias_enable(&mut self, enable: bool) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            self.device().api_version() >= Version::V1_3
                || self.device().enabled_features().extended_dynamic_state2,
            "the extended_dynamic_state2 feature must be enabled on the device"
        );
        assert!(
            !self.has_fixed_state(DynamicState::DepthBiasEnable),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        unsafe {
            self.inner.set_depth_bias_enable(enable);
        }

        self
    }

    /// Sets the dynamic depth bounds for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    /// - If the
    ///   [`ext_depth_range_unrestricted`](crate::device::DeviceExtensions::ext_depth_range_unrestricted)
    ///   device extension is not enabled, panics if `min` or `max` is not between 0.0 and 1.0 inclusive.
    pub fn set_depth_bounds(&mut self, min: f32, max: f32) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            !self.has_fixed_state(DynamicState::DepthBounds),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        if !self
            .device()
            .enabled_extensions()
            .ext_depth_range_unrestricted
        {
            assert!(
					min >= 0.0 && min <= 1.0 && max >= 0.0 && max <= 1.0,
					"if the ext_depth_range_unrestricted device extension is not enabled, depth bounds values must be between 0.0 and 1.0"
				);
        }

        unsafe {
            self.inner.set_depth_bounds(min, max);
        }

        self
    }

    /// Sets whether dynamic depth bounds testing is enabled for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the device API version is less than 1.3 and the
    ///   [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature is
    ///   not enabled on the device.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    #[inline]
    pub fn set_depth_bounds_test_enable(&mut self, enable: bool) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            self.device().api_version() >= Version::V1_3
                || self.device().enabled_features().extended_dynamic_state,
            "the extended_dynamic_state feature must be enabled on the device"
        );
        assert!(
            !self.has_fixed_state(DynamicState::DepthBoundsTestEnable),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        unsafe {
            self.inner.set_depth_bounds_test_enable(enable);
        }

        self
    }

    /// Sets the dynamic depth compare op for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the device API version is less than 1.3 and the
    ///   [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature is
    ///   not enabled on the device.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    #[inline]
    pub fn set_depth_compare_op(&mut self, compare_op: CompareOp) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            self.device().api_version() >= Version::V1_3
                || self.device().enabled_features().extended_dynamic_state,
            "the extended_dynamic_state feature must be enabled on the device"
        );
        assert!(
            !self.has_fixed_state(DynamicState::DepthCompareOp),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        unsafe {
            self.inner.set_depth_compare_op(compare_op);
        }

        self
    }

    /// Sets whether dynamic depth testing is enabled for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the device API version is less than 1.3 and the
    ///   [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature is
    ///   not enabled on the device.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    #[inline]
    pub fn set_depth_test_enable(&mut self, enable: bool) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            self.device().api_version() >= Version::V1_3
                || self.device().enabled_features().extended_dynamic_state,
            "the extended_dynamic_state feature must be enabled on the device"
        );
        assert!(
            !self.has_fixed_state(DynamicState::DepthTestEnable),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        unsafe {
            self.inner.set_depth_test_enable(enable);
        }

        self
    }

    /// Sets whether dynamic depth write is enabled for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the device API version is less than 1.3 and the
    ///   [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature is
    ///   not enabled on the device.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    #[inline]
    pub fn set_depth_write_enable(&mut self, enable: bool) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            self.device().api_version() >= Version::V1_3
                || self.device().enabled_features().extended_dynamic_state,
            "the extended_dynamic_state feature must be enabled on the device"
        );
        assert!(
            !self.has_fixed_state(DynamicState::DepthWriteEnable),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        unsafe {
            self.inner.set_depth_write_enable(enable);
        }

        self
    }

    /// Sets the dynamic discard rectangles for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the
    ///   [`ext_discard_rectangles`](crate::device::DeviceExtensions::ext_discard_rectangles)
    ///   extension is not enabled on the device.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    /// - Panics if the highest discard rectangle slot being set is greater than the
    ///   [`max_discard_rectangles`](crate::device::Properties::max_discard_rectangles) device
    ///   property.
    pub fn set_discard_rectangle<I>(&mut self, first_rectangle: u32, rectangles: I) -> &mut Self
    where
        I: IntoIterator<Item = Scissor>,
    {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            self.device().enabled_extensions().ext_discard_rectangles,
            "the ext_discard_rectangles extension must be enabled on the device"
        );
        assert!(
            !self.has_fixed_state(DynamicState::DiscardRectangle),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        let rectangles: SmallVec<[Scissor; 2]> = rectangles.into_iter().collect();

        assert!(
				first_rectangle + rectangles.len() as u32 <= self.device().physical_device().properties().max_discard_rectangles.unwrap(),
				"the highest discard rectangle slot being set must not be higher than the max_discard_rectangles device property"
			);

        // TODO: VUID-vkCmdSetDiscardRectangleEXT-viewportScissor2D-04788
        // If this command is recorded in a secondary command buffer with
        // VkCommandBufferInheritanceViewportScissorInfoNV::viewportScissor2D enabled, then this
        // function must not be called

        unsafe {
            self.inner
                .set_discard_rectangle(first_rectangle, rectangles);
        }

        self
    }

    /// Sets the dynamic front face for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the device API version is less than 1.3 and the
    ///   [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature is
    ///   not enabled on the device.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    #[inline]
    pub fn set_front_face(&mut self, face: FrontFace) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            self.device().api_version() >= Version::V1_3
                || self.device().enabled_features().extended_dynamic_state,
            "the extended_dynamic_state feature must be enabled on the device"
        );
        assert!(
            !self.has_fixed_state(DynamicState::FrontFace),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        unsafe {
            self.inner.set_front_face(face);
        }

        self
    }

    /// Sets the dynamic line stipple values for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the [`ext_line_rasterization`](crate::device::DeviceExtensions::ext_line_rasterization)
    ///   extension is not enabled on the device.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    /// - Panics if `factor` is not between 1 and 256 inclusive.
    #[inline]
    pub fn set_line_stipple(&mut self, factor: u32, pattern: u16) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            self.device().enabled_extensions().ext_line_rasterization,
            "the ext_line_rasterization extension must be enabled on the device"
        );
        assert!(
            !self.has_fixed_state(DynamicState::LineStipple),
            "the currently bound graphics pipeline must not contain this state internally"
        );
        assert!(
            factor >= 1 && factor <= 256,
            "factor must be between 1 and 256 inclusive"
        );

        unsafe {
            self.inner.set_line_stipple(factor, pattern);
        }

        self
    }

    /// Sets the dynamic line width for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    /// - If the [`wide_lines`](crate::device::Features::wide_lines) feature is not enabled, panics
    ///   if `line_width` is not 1.0.
    pub fn set_line_width(&mut self, line_width: f32) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            !self.has_fixed_state(DynamicState::LineWidth),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        if !self.device().enabled_features().wide_lines {
            assert!(
                line_width == 1.0,
                "if the wide_line features is not enabled, line width must be 1.0"
            );
        }

        unsafe {
            self.inner.set_line_width(line_width);
        }

        self
    }

    /// Sets the dynamic logic op for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the
    ///   [`extended_dynamic_state2_logic_op`](crate::device::Features::extended_dynamic_state2_logic_op)
    ///   feature is not enabled on the device.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    #[inline]
    pub fn set_logic_op(&mut self, logic_op: LogicOp) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            self.device()
                .enabled_features()
                .extended_dynamic_state2_logic_op,
            "the extended_dynamic_state2_logic_op feature must be enabled on the device"
        );
        assert!(
            !self.has_fixed_state(DynamicState::LogicOp),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        unsafe {
            self.inner.set_logic_op(logic_op);
        }

        self
    }

    /// Sets the dynamic number of patch control points for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the
    ///   [`extended_dynamic_state2_patch_control_points`](crate::device::Features::extended_dynamic_state2_patch_control_points)
    ///   feature is not enabled on the device.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    /// - Panics if `num` is 0.
    /// - Panics if `num` is greater than the
    ///   [`max_tessellation_patch_size`](crate::device::Properties::max_tessellation_patch_size)
    ///   property of the device.
    #[inline]
    pub fn set_patch_control_points(&mut self, num: u32) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
				self.device().enabled_features().extended_dynamic_state2_patch_control_points,
				"the extended_dynamic_state2_patch_control_points feature must be enabled on the device"
			);
        assert!(
            !self.has_fixed_state(DynamicState::PatchControlPoints),
            "the currently bound graphics pipeline must not contain this state internally"
        );
        assert!(num > 0, "num must be greater than 0");
        assert!(
            num <= self
                .device()
                .physical_device()
                .properties()
                .max_tessellation_patch_size,
            "num must be less than or equal to max_tessellation_patch_size"
        );

        unsafe {
            self.inner.set_patch_control_points(num);
        }

        self
    }

    /// Sets whether dynamic primitive restart is enabled for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the device API version is less than 1.3 and the
    ///   [`extended_dynamic_state2`](crate::device::Features::extended_dynamic_state2) feature is
    ///   not enabled on the device.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    #[inline]
    pub fn set_primitive_restart_enable(&mut self, enable: bool) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            self.device().api_version() >= Version::V1_3
                || self.device().enabled_features().extended_dynamic_state2,
            "the extended_dynamic_state2 feature must be enabled on the device"
        );
        assert!(
            !self.has_fixed_state(DynamicState::PrimitiveRestartEnable),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        unsafe {
            self.inner.set_primitive_restart_enable(enable);
        }

        self
    }

    /// Sets the dynamic primitive topology for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the device API version is less than 1.3 and the
    ///   [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature is
    ///   not enabled on the device.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    /// - If the [`geometry_shader`](crate::device::Features::geometry_shader) feature is not
    ///   enabled, panics if `topology` is a `WithAdjacency` topology.
    /// - If the [`tessellation_shader`](crate::device::Features::tessellation_shader) feature is
    ///   not enabled, panics if `topology` is `PatchList`.
    #[inline]
    pub fn set_primitive_topology(&mut self, topology: PrimitiveTopology) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            self.device().api_version() >= Version::V1_3
                || self.device().enabled_features().extended_dynamic_state,
            "the extended_dynamic_state feature must be enabled on the device"
        );
        assert!(
            !self.has_fixed_state(DynamicState::PrimitiveTopology),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        if !self.device().enabled_features().geometry_shader {
            assert!(!matches!(topology, PrimitiveTopology::LineListWithAdjacency
				| PrimitiveTopology::LineStripWithAdjacency
				| PrimitiveTopology::TriangleListWithAdjacency
				| PrimitiveTopology::TriangleStripWithAdjacency), "if the geometry_shader feature is not enabled, topology must not be a WithAdjacency topology");
        }

        if !self.device().enabled_features().tessellation_shader {
            assert!(
                !matches!(topology, PrimitiveTopology::PatchList),
                "if the tessellation_shader feature is not enabled, topology must not be PatchList"
            );
        }

        unsafe {
            self.inner.set_primitive_topology(topology);
        }

        self
    }

    /// Sets whether dynamic rasterizer discard is enabled for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the device API version is less than 1.3 and the
    ///   [`extended_dynamic_state2`](crate::device::Features::extended_dynamic_state2) feature is
    ///   not enabled on the device.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    #[inline]
    pub fn set_rasterizer_discard_enable(&mut self, enable: bool) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            self.device().api_version() >= Version::V1_3
                || self.device().enabled_features().extended_dynamic_state2,
            "the extended_dynamic_state2 feature must be enabled on the device"
        );
        assert!(
            !self.has_fixed_state(DynamicState::RasterizerDiscardEnable),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        unsafe {
            self.inner.set_rasterizer_discard_enable(enable);
        }

        self
    }

    /// Sets the dynamic scissors for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    /// - Panics if the highest scissor slot being set is greater than the
    ///   [`max_viewports`](crate::device::Properties::max_viewports) device property.
    /// - If the [`multi_viewport`](crate::device::Features::multi_viewport) feature is not enabled,
    ///   panics if `first_scissor` is not 0, or if more than 1 scissor is provided.
    pub fn set_scissor<I>(&mut self, first_scissor: u32, scissors: I) -> &mut Self
    where
        I: IntoIterator<Item = Scissor>,
    {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            !self.has_fixed_state(DynamicState::Scissor),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        let scissors: SmallVec<[Scissor; 2]> = scissors.into_iter().collect();

        assert!(
				first_scissor + scissors.len() as u32 <= self.device().physical_device().properties().max_viewports,
				"the highest scissor slot being set must not be higher than the max_viewports device property"
			);

        if !self.device().enabled_features().multi_viewport {
            assert!(
                first_scissor == 0,
                "if the multi_viewport feature is not enabled, first_scissor must be 0"
            );

            assert!(
					scissors.len() <= 1,
					"if the multi_viewport feature is not enabled, no more than 1 scissor must be provided"
				);
        }

        // TODO:
        // If this command is recorded in a secondary command buffer with
        // VkCommandBufferInheritanceViewportScissorInfoNV::viewportScissor2D enabled, then this
        // function must not be called

        unsafe {
            self.inner.set_scissor(first_scissor, scissors);
        }

        self
    }

    /// Sets the dynamic scissors with count for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the device API version is less than 1.3 and the
    ///   [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature is
    ///   not enabled on the device.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    /// - Panics if the highest scissor slot being set is greater than the
    ///   [`max_viewports`](crate::device::Properties::max_viewports) device property.
    /// - If the [`multi_viewport`](crate::device::Features::multi_viewport) feature is not enabled,
    ///   panics if more than 1 scissor is provided.
    #[inline]
    pub fn set_scissor_with_count<I>(&mut self, scissors: I) -> &mut Self
    where
        I: IntoIterator<Item = Scissor>,
    {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            self.device().api_version() >= Version::V1_3
                || self.device().enabled_features().extended_dynamic_state,
            "the extended_dynamic_state feature must be enabled on the device"
        );
        assert!(
            !self.has_fixed_state(DynamicState::ScissorWithCount),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        let scissors: SmallVec<[Scissor; 2]> = scissors.into_iter().collect();

        assert!(
				scissors.len() as u32 <= self.device().physical_device().properties().max_viewports,
				"the highest scissor slot being set must not be higher than the max_viewports device property"
			);

        if !self.device().enabled_features().multi_viewport {
            assert!(
					scissors.len() <= 1,
					"if the multi_viewport feature is not enabled, no more than 1 scissor must be provided"
				);
        }

        // TODO: VUID-vkCmdSetScissorWithCountEXT-commandBuffer-04820
        // commandBuffer must not have
        // VkCommandBufferInheritanceViewportScissorInfoNV::viewportScissor2D enabled

        unsafe {
            self.inner.set_scissor_with_count(scissors);
        }

        self
    }

    /// Sets the dynamic stencil compare mask on one or both faces for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    pub fn set_stencil_compare_mask(
        &mut self,
        faces: StencilFaces,
        compare_mask: u32,
    ) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            !self.has_fixed_state(DynamicState::StencilCompareMask),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        unsafe {
            self.inner.set_stencil_compare_mask(faces, compare_mask);
        }

        self
    }

    /// Sets the dynamic stencil ops on one or both faces for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the device API version is less than 1.3 and the
    ///   [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature is
    ///   not enabled on the device.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    #[inline]
    pub fn set_stencil_op(
        &mut self,
        faces: StencilFaces,
        fail_op: StencilOp,
        pass_op: StencilOp,
        depth_fail_op: StencilOp,
        compare_op: CompareOp,
    ) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            self.device().api_version() >= Version::V1_3
                || self.device().enabled_features().extended_dynamic_state,
            "the extended_dynamic_state feature must be enabled on the device"
        );
        assert!(
            !self.has_fixed_state(DynamicState::StencilOp),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        unsafe {
            self.inner
                .set_stencil_op(faces, fail_op, pass_op, depth_fail_op, compare_op);
        }

        self
    }

    /// Sets the dynamic stencil reference on one or both faces for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    pub fn set_stencil_reference(&mut self, faces: StencilFaces, reference: u32) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            !self.has_fixed_state(DynamicState::StencilReference),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        unsafe {
            self.inner.set_stencil_reference(faces, reference);
        }

        self
    }

    /// Sets whether dynamic stencil testing is enabled for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the device API version is less than 1.3 and the
    ///   [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature is
    ///   not enabled on the device.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    #[inline]
    pub fn set_stencil_test_enable(&mut self, enable: bool) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            self.device().api_version() >= Version::V1_3
                || self.device().enabled_features().extended_dynamic_state,
            "the extended_dynamic_state feature must be enabled on the device"
        );
        assert!(
            !self.has_fixed_state(DynamicState::StencilTestEnable),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        unsafe {
            self.inner.set_stencil_test_enable(enable);
        }

        self
    }

    /// Sets the dynamic stencil write mask on one or both faces for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    pub fn set_stencil_write_mask(&mut self, faces: StencilFaces, write_mask: u32) -> &mut Self {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            !self.has_fixed_state(DynamicState::StencilWriteMask),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        unsafe {
            self.inner.set_stencil_write_mask(faces, write_mask);
        }

        self
    }

    /// Sets the dynamic viewports for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    /// - Panics if the highest viewport slot being set is greater than the
    ///   [`max_viewports`](crate::device::Properties::max_viewports) device property.
    /// - If the [`multi_viewport`](crate::device::Features::multi_viewport) feature is not enabled,
    ///   panics if `first_viewport` is not 0, or if more than 1 viewport is provided.
    pub fn set_viewport<I>(&mut self, first_viewport: u32, viewports: I) -> &mut Self
    where
        I: IntoIterator<Item = Viewport>,
    {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            !self.has_fixed_state(DynamicState::Viewport),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        let viewports: SmallVec<[Viewport; 2]> = viewports.into_iter().collect();

        assert!(
				first_viewport + viewports.len() as u32 <= self.device().physical_device().properties().max_viewports,
				"the highest viewport slot being set must not be higher than the max_viewports device property"
			);

        if !self.device().enabled_features().multi_viewport {
            assert!(
                first_viewport == 0,
                "if the multi_viewport feature is not enabled, first_viewport must be 0"
            );

            assert!(
					viewports.len() <= 1,
					"if the multi_viewport feature is not enabled, no more than 1 viewport must be provided"
				);
        }

        // TODO:
        // commandBuffer must not have
        // VkCommandBufferInheritanceViewportScissorInfoNV::viewportScissor2D enabled

        unsafe {
            self.inner.set_viewport(first_viewport, viewports);
        }

        self
    }

    /// Sets the dynamic viewports with count for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the device API version is less than 1.3 and the
    ///   [`extended_dynamic_state`](crate::device::Features::extended_dynamic_state) feature is
    ///   not enabled on the device.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    /// - Panics if the highest viewport slot being set is greater than the
    ///   [`max_viewports`](crate::device::Properties::max_viewports) device property.
    /// - If the [`multi_viewport`](crate::device::Features::multi_viewport) feature is not enabled,
    ///   panics if more than 1 viewport is provided.
    #[inline]
    pub fn set_viewport_with_count<I>(&mut self, viewports: I) -> &mut Self
    where
        I: IntoIterator<Item = Viewport>,
    {
        assert!(
            self.queue_family().supports_graphics(),
            "the queue family of the command buffer must support graphics operations"
        );
        assert!(
            self.device().api_version() >= Version::V1_3
                || self.device().enabled_features().extended_dynamic_state,
            "the extended_dynamic_state feature must be enabled on the device"
        );
        assert!(
            !self.has_fixed_state(DynamicState::ViewportWithCount),
            "the currently bound graphics pipeline must not contain this state internally"
        );

        let viewports: SmallVec<[Viewport; 2]> = viewports.into_iter().collect();

        assert!(
				viewports.len() as u32 <= self.device().physical_device().properties().max_viewports,
				"the highest viewport slot being set must not be higher than the max_viewports device property"
			);

        if !self.device().enabled_features().multi_viewport {
            assert!(
					viewports.len() <= 1,
					"if the multi_viewport feature is not enabled, no more than 1 viewport must be provided"
				);
        }

        // TODO: VUID-vkCmdSetViewportWithCountEXT-commandBuffer-04819
        // commandBuffer must not have
        // VkCommandBufferInheritanceViewportScissorInfoNV::viewportScissor2D enabled

        unsafe {
            self.inner.set_viewport_with_count(viewports);
        }

        self
    }
}

impl SyncCommandBufferBuilder {
    /// Calls `vkCmdSetBlendConstants` on the builder.
    #[inline]
    pub unsafe fn set_blend_constants(&mut self, constants: [f32; 4]) {
        struct Cmd {
            constants: [f32; 4],
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_blend_constants"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_blend_constants(self.constants);
            }
        }

        self.commands.push(Box::new(Cmd { constants }));
        self.current_state.blend_constants = Some(constants);
    }

    /// Calls `vkCmdSetColorWriteEnableEXT` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_color_write_enable<I>(&mut self, enables: I)
    where
        I: IntoIterator<Item = bool>,
    {
        struct Cmd<I> {
            enables: Mutex<Option<I>>,
        }

        impl<I> Command for Cmd<I>
        where
            I: IntoIterator<Item = bool> + Send + Sync,
        {
            fn name(&self) -> &'static str {
                "set_color_write_enable"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_color_write_enable(self.enables.lock().take().unwrap());
            }
        }

        let enables: SmallVec<[bool; 4]> = enables.into_iter().collect();
        self.current_state.color_write_enable = Some(enables.clone());
        self.commands.push(Box::new(Cmd {
            enables: Mutex::new(Some(enables)),
        }));
    }

    /// Calls `vkCmdSetCullModeEXT` on the builder.
    #[inline]
    pub unsafe fn set_cull_mode(&mut self, cull_mode: CullMode) {
        struct Cmd {
            cull_mode: CullMode,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_cull_mode"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_cull_mode(self.cull_mode);
            }
        }

        self.commands.push(Box::new(Cmd { cull_mode }));
        self.current_state.cull_mode = Some(cull_mode);
    }

    /// Calls `vkCmdSetDepthBias` on the builder.
    #[inline]
    pub unsafe fn set_depth_bias(&mut self, constant_factor: f32, clamp: f32, slope_factor: f32) {
        struct Cmd {
            constant_factor: f32,
            clamp: f32,
            slope_factor: f32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_depth_bias"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_depth_bias(self.constant_factor, self.clamp, self.slope_factor);
            }
        }

        self.commands.push(Box::new(Cmd {
            constant_factor,
            clamp,
            slope_factor,
        }));
        self.current_state.depth_bias = Some(DepthBias {
            constant_factor,
            clamp,
            slope_factor,
        });
    }

    /// Calls `vkCmdSetDepthBiasEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_depth_bias_enable(&mut self, enable: bool) {
        struct Cmd {
            enable: bool,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_depth_bias_enable"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_depth_bias_enable(self.enable);
            }
        }

        self.commands.push(Box::new(Cmd { enable }));
        self.current_state.depth_bias_enable = Some(enable);
    }

    /// Calls `vkCmdSetDepthBounds` on the builder.
    #[inline]
    pub unsafe fn set_depth_bounds(&mut self, min: f32, max: f32) {
        struct Cmd {
            min: f32,
            max: f32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_depth_bounds"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_depth_bounds(self.min, self.max);
            }
        }

        self.commands.push(Box::new(Cmd { min, max }));
        self.current_state.depth_bounds = Some((min, max));
    }

    /// Calls `vkCmdSetDepthBoundsTestEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_depth_bounds_test_enable(&mut self, enable: bool) {
        struct Cmd {
            enable: bool,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_depth_bounds_test_enable"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_depth_bounds_test_enable(self.enable);
            }
        }

        self.commands.push(Box::new(Cmd { enable }));
        self.current_state.depth_bounds_test_enable = Some(enable);
    }

    /// Calls `vkCmdSetDepthCompareOpEXT` on the builder.
    #[inline]
    pub unsafe fn set_depth_compare_op(&mut self, compare_op: CompareOp) {
        struct Cmd {
            compare_op: CompareOp,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_depth_compare_op"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_depth_compare_op(self.compare_op);
            }
        }

        self.commands.push(Box::new(Cmd { compare_op }));
        self.current_state.depth_compare_op = Some(compare_op);
    }

    /// Calls `vkCmdSetDepthTestEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_depth_test_enable(&mut self, enable: bool) {
        struct Cmd {
            enable: bool,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_depth_test_enable"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_depth_test_enable(self.enable);
            }
        }

        self.commands.push(Box::new(Cmd { enable }));
        self.current_state.depth_test_enable = Some(enable);
    }

    /// Calls `vkCmdSetDepthWriteEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_depth_write_enable(&mut self, enable: bool) {
        struct Cmd {
            enable: bool,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_depth_write_enable"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_depth_write_enable(self.enable);
            }
        }

        self.commands.push(Box::new(Cmd { enable }));
        self.current_state.depth_write_enable = Some(enable);
    }

    /// Calls `vkCmdSetDiscardRectangle` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_discard_rectangle<I>(&mut self, first_rectangle: u32, rectangles: I)
    where
        I: IntoIterator<Item = Scissor>,
    {
        struct Cmd {
            first_rectangle: u32,
            rectangles: Mutex<SmallVec<[Scissor; 2]>>,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_discard_rectangle"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_discard_rectangle(self.first_rectangle, self.rectangles.lock().drain(..));
            }
        }

        let rectangles: SmallVec<[Scissor; 2]> = rectangles.into_iter().collect();

        for (num, rectangle) in rectangles.iter().enumerate() {
            let num = num as u32 + first_rectangle;
            self.current_state
                .discard_rectangle
                .insert(num, rectangle.clone());
        }

        self.commands.push(Box::new(Cmd {
            first_rectangle,
            rectangles: Mutex::new(rectangles),
        }));
    }

    /// Calls `vkCmdSetFrontFaceEXT` on the builder.
    #[inline]
    pub unsafe fn set_front_face(&mut self, face: FrontFace) {
        struct Cmd {
            face: FrontFace,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_front_face"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_front_face(self.face);
            }
        }

        self.commands.push(Box::new(Cmd { face }));
        self.current_state.front_face = Some(face);
    }

    /// Calls `vkCmdSetLineStippleEXT` on the builder.
    #[inline]
    pub unsafe fn set_line_stipple(&mut self, factor: u32, pattern: u16) {
        struct Cmd {
            factor: u32,
            pattern: u16,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_line_stipple"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_line_stipple(self.factor, self.pattern);
            }
        }

        self.commands.push(Box::new(Cmd { factor, pattern }));
        self.current_state.line_stipple = Some(LineStipple { factor, pattern });
    }

    /// Calls `vkCmdSetLineWidth` on the builder.
    #[inline]
    pub unsafe fn set_line_width(&mut self, line_width: f32) {
        struct Cmd {
            line_width: f32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_line_width"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_line_width(self.line_width);
            }
        }

        self.commands.push(Box::new(Cmd { line_width }));
        self.current_state.line_width = Some(line_width);
    }

    /// Calls `vkCmdSetLogicOpEXT` on the builder.
    #[inline]
    pub unsafe fn set_logic_op(&mut self, logic_op: LogicOp) {
        struct Cmd {
            logic_op: LogicOp,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_logic_op"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_logic_op(self.logic_op);
            }
        }

        self.commands.push(Box::new(Cmd { logic_op }));
        self.current_state.logic_op = Some(logic_op);
    }

    /// Calls `vkCmdSetPatchControlPointsEXT` on the builder.
    #[inline]
    pub unsafe fn set_patch_control_points(&mut self, num: u32) {
        struct Cmd {
            num: u32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_patch_control_points"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_patch_control_points(self.num);
            }
        }

        self.commands.push(Box::new(Cmd { num }));
        self.current_state.patch_control_points = Some(num);
    }

    /// Calls `vkCmdSetPrimitiveRestartEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_primitive_restart_enable(&mut self, enable: bool) {
        struct Cmd {
            enable: bool,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_primitive_restart_enable"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_primitive_restart_enable(self.enable);
            }
        }

        self.commands.push(Box::new(Cmd { enable }));
        self.current_state.primitive_restart_enable = Some(enable);
    }

    /// Calls `vkCmdSetPrimitiveTopologyEXT` on the builder.
    #[inline]
    pub unsafe fn set_primitive_topology(&mut self, topology: PrimitiveTopology) {
        struct Cmd {
            topology: PrimitiveTopology,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_primitive_topology"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_primitive_topology(self.topology);
            }
        }

        self.commands.push(Box::new(Cmd { topology }));
        self.current_state.primitive_topology = Some(topology);
    }

    /// Calls `vkCmdSetRasterizerDiscardEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_rasterizer_discard_enable(&mut self, enable: bool) {
        struct Cmd {
            enable: bool,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_rasterizer_discard_enable"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_rasterizer_discard_enable(self.enable);
            }
        }

        self.commands.push(Box::new(Cmd { enable }));
        self.current_state.rasterizer_discard_enable = Some(enable);
    }

    /// Calls `vkCmdSetStencilCompareMask` on the builder.
    #[inline]
    pub unsafe fn set_stencil_compare_mask(&mut self, faces: StencilFaces, compare_mask: u32) {
        struct Cmd {
            faces: StencilFaces,
            compare_mask: u32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_stencil_compare_mask"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_stencil_compare_mask(self.faces, self.compare_mask);
            }
        }

        self.commands.push(Box::new(Cmd {
            faces,
            compare_mask,
        }));

        let faces = ash::vk::StencilFaceFlags::from(faces);

        if faces.intersects(ash::vk::StencilFaceFlags::FRONT) {
            self.current_state.stencil_compare_mask.front = Some(compare_mask);
        }

        if faces.intersects(ash::vk::StencilFaceFlags::BACK) {
            self.current_state.stencil_compare_mask.back = Some(compare_mask);
        }
    }

    /// Calls `vkCmdSetStencilOpEXT` on the builder.
    #[inline]
    pub unsafe fn set_stencil_op(
        &mut self,
        faces: StencilFaces,
        fail_op: StencilOp,
        pass_op: StencilOp,
        depth_fail_op: StencilOp,
        compare_op: CompareOp,
    ) {
        struct Cmd {
            faces: StencilFaces,
            fail_op: StencilOp,
            pass_op: StencilOp,
            depth_fail_op: StencilOp,
            compare_op: CompareOp,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_stencil_op"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_stencil_op(
                    self.faces,
                    self.fail_op,
                    self.pass_op,
                    self.depth_fail_op,
                    self.compare_op,
                );
            }
        }

        self.commands.push(Box::new(Cmd {
            faces,
            fail_op,
            pass_op,
            depth_fail_op,
            compare_op,
        }));

        let faces = ash::vk::StencilFaceFlags::from(faces);

        if faces.intersects(ash::vk::StencilFaceFlags::FRONT) {
            self.current_state.stencil_op.front = Some(StencilOps {
                fail_op,
                pass_op,
                depth_fail_op,
                compare_op,
            });
        }

        if faces.intersects(ash::vk::StencilFaceFlags::BACK) {
            self.current_state.stencil_op.back = Some(StencilOps {
                fail_op,
                pass_op,
                depth_fail_op,
                compare_op,
            });
        }
    }

    /// Calls `vkCmdSetStencilReference` on the builder.
    #[inline]
    pub unsafe fn set_stencil_reference(&mut self, faces: StencilFaces, reference: u32) {
        struct Cmd {
            faces: StencilFaces,
            reference: u32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_stencil_reference"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_stencil_reference(self.faces, self.reference);
            }
        }

        self.commands.push(Box::new(Cmd { faces, reference }));

        let faces = ash::vk::StencilFaceFlags::from(faces);

        if faces.intersects(ash::vk::StencilFaceFlags::FRONT) {
            self.current_state.stencil_reference.front = Some(reference);
        }

        if faces.intersects(ash::vk::StencilFaceFlags::BACK) {
            self.current_state.stencil_reference.back = Some(reference);
        }
    }

    /// Calls `vkCmdSetStencilTestEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_stencil_test_enable(&mut self, enable: bool) {
        struct Cmd {
            enable: bool,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_stencil_test_enable"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_stencil_test_enable(self.enable);
            }
        }

        self.commands.push(Box::new(Cmd { enable }));
        self.current_state.stencil_test_enable = Some(enable);
    }

    /// Calls `vkCmdSetStencilWriteMask` on the builder.
    #[inline]
    pub unsafe fn set_stencil_write_mask(&mut self, faces: StencilFaces, write_mask: u32) {
        struct Cmd {
            faces: StencilFaces,
            write_mask: u32,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_stencil_write_mask"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_stencil_write_mask(self.faces, self.write_mask);
            }
        }

        self.commands.push(Box::new(Cmd { faces, write_mask }));

        let faces = ash::vk::StencilFaceFlags::from(faces);

        if faces.intersects(ash::vk::StencilFaceFlags::FRONT) {
            self.current_state.stencil_write_mask.front = Some(write_mask);
        }

        if faces.intersects(ash::vk::StencilFaceFlags::BACK) {
            self.current_state.stencil_write_mask.back = Some(write_mask);
        }
    }

    /// Calls `vkCmdSetScissor` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_scissor<I>(&mut self, first_scissor: u32, scissors: I)
    where
        I: IntoIterator<Item = Scissor>,
    {
        struct Cmd {
            first_scissor: u32,
            scissors: Mutex<SmallVec<[Scissor; 2]>>,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_scissor"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_scissor(self.first_scissor, self.scissors.lock().drain(..));
            }
        }

        let scissors: SmallVec<[Scissor; 2]> = scissors.into_iter().collect();

        for (num, scissor) in scissors.iter().enumerate() {
            let num = num as u32 + first_scissor;
            self.current_state.scissor.insert(num, scissor.clone());
        }

        self.commands.push(Box::new(Cmd {
            first_scissor,
            scissors: Mutex::new(scissors),
        }));
    }

    /// Calls `vkCmdSetScissorWithCountEXT` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_scissor_with_count<I>(&mut self, scissors: I)
    where
        I: IntoIterator<Item = Scissor>,
    {
        struct Cmd {
            scissors: Mutex<SmallVec<[Scissor; 2]>>,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_scissor_with_count"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_scissor_with_count(self.scissors.lock().drain(..));
            }
        }

        let scissors: SmallVec<[Scissor; 2]> = scissors.into_iter().collect();
        self.current_state.scissor_with_count = Some(scissors.clone());
        self.commands.push(Box::new(Cmd {
            scissors: Mutex::new(scissors),
        }));
    }

    /// Calls `vkCmdSetViewport` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_viewport<I>(&mut self, first_viewport: u32, viewports: I)
    where
        I: IntoIterator<Item = Viewport>,
    {
        struct Cmd {
            first_viewport: u32,
            viewports: Mutex<SmallVec<[Viewport; 2]>>,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_viewport"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_viewport(self.first_viewport, self.viewports.lock().drain(..));
            }
        }

        let viewports: SmallVec<[Viewport; 2]> = viewports.into_iter().collect();

        for (num, viewport) in viewports.iter().enumerate() {
            let num = num as u32 + first_viewport;
            self.current_state.viewport.insert(num, viewport.clone());
        }

        self.commands.push(Box::new(Cmd {
            first_viewport,
            viewports: Mutex::new(viewports),
        }));
    }

    /// Calls `vkCmdSetViewportWithCountEXT` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_viewport_with_count<I>(&mut self, viewports: I)
    where
        I: IntoIterator<Item = Viewport>,
    {
        struct Cmd {
            viewports: Mutex<SmallVec<[Viewport; 2]>>,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_viewport_with_count"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_viewport_with_count(self.viewports.lock().drain(..));
            }
        }

        let viewports: SmallVec<[Viewport; 2]> = viewports.into_iter().collect();
        self.current_state.viewport_with_count = Some(viewports.clone());
        self.commands.push(Box::new(Cmd {
            viewports: Mutex::new(viewports),
        }));
    }
}

impl UnsafeCommandBufferBuilder {
    /// Calls `vkCmdSetBlendConstants` on the builder.
    #[inline]
    pub unsafe fn set_blend_constants(&mut self, constants: [f32; 4]) {
        let fns = self.device.fns();
        fns.v1_0.cmd_set_blend_constants(self.handle, &constants);
    }

    /// Calls `vkCmdSetColorWriteEnableEXT` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_color_write_enable(&mut self, enables: impl IntoIterator<Item = bool>) {
        debug_assert!(self.device.enabled_extensions().ext_color_write_enable);
        debug_assert!(self.device.enabled_features().color_write_enable);

        let enables = enables
            .into_iter()
            .map(|v| v as ash::vk::Bool32)
            .collect::<SmallVec<[_; 4]>>();
        if enables.is_empty() {
            return;
        }

        let fns = self.device.fns();
        fns.ext_color_write_enable.cmd_set_color_write_enable_ext(
            self.handle,
            enables.len() as u32,
            enables.as_ptr(),
        );
    }

    /// Calls `vkCmdSetCullModeEXT` on the builder.
    #[inline]
    pub unsafe fn set_cull_mode(&mut self, cull_mode: CullMode) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            fns.v1_3.cmd_set_cull_mode(self.handle, cull_mode.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            debug_assert!(self.device.enabled_features().extended_dynamic_state);
            fns.ext_extended_dynamic_state
                .cmd_set_cull_mode_ext(self.handle, cull_mode.into());
        }
    }

    /// Calls `vkCmdSetDepthBias` on the builder.
    #[inline]
    pub unsafe fn set_depth_bias(&mut self, constant_factor: f32, clamp: f32, slope_factor: f32) {
        debug_assert!(clamp == 0.0 || self.device.enabled_features().depth_bias_clamp);
        let fns = self.device.fns();
        fns.v1_0
            .cmd_set_depth_bias(self.handle, constant_factor, clamp, slope_factor);
    }

    /// Calls `vkCmdSetDepthBiasEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_depth_bias_enable(&mut self, enable: bool) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            fns.v1_3
                .cmd_set_depth_bias_enable(self.handle, enable.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state2);
            debug_assert!(self.device.enabled_features().extended_dynamic_state2);
            fns.ext_extended_dynamic_state2
                .cmd_set_depth_bias_enable_ext(self.handle, enable.into());
        }
    }

    /// Calls `vkCmdSetDepthBounds` on the builder.
    #[inline]
    pub unsafe fn set_depth_bounds(&mut self, min: f32, max: f32) {
        debug_assert!(min >= 0.0 && min <= 1.0);
        debug_assert!(max >= 0.0 && max <= 1.0);
        let fns = self.device.fns();
        fns.v1_0.cmd_set_depth_bounds(self.handle, min, max);
    }

    /// Calls `vkCmdSetDepthBoundsTestEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_depth_bounds_test_enable(&mut self, enable: bool) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            fns.v1_3
                .cmd_set_depth_bounds_test_enable(self.handle, enable.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            debug_assert!(self.device.enabled_features().extended_dynamic_state);
            fns.ext_extended_dynamic_state
                .cmd_set_depth_bounds_test_enable_ext(self.handle, enable.into());
        }
    }

    /// Calls `vkCmdSetDepthCompareOpEXT` on the builder.
    #[inline]
    pub unsafe fn set_depth_compare_op(&mut self, compare_op: CompareOp) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            fns.v1_3
                .cmd_set_depth_compare_op(self.handle, compare_op.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            debug_assert!(self.device.enabled_features().extended_dynamic_state);
            fns.ext_extended_dynamic_state
                .cmd_set_depth_compare_op_ext(self.handle, compare_op.into());
        }
    }

    /// Calls `vkCmdSetDepthTestEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_depth_test_enable(&mut self, enable: bool) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            fns.v1_3
                .cmd_set_depth_test_enable(self.handle, enable.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            debug_assert!(self.device.enabled_features().extended_dynamic_state);
            fns.ext_extended_dynamic_state
                .cmd_set_depth_test_enable_ext(self.handle, enable.into());
        }
    }

    /// Calls `vkCmdSetDepthWriteEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_depth_write_enable(&mut self, enable: bool) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            fns.v1_3
                .cmd_set_depth_write_enable(self.handle, enable.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            debug_assert!(self.device.enabled_features().extended_dynamic_state);
            fns.ext_extended_dynamic_state
                .cmd_set_depth_write_enable_ext(self.handle, enable.into());
        }
    }

    /// Calls `vkCmdSetDiscardRectangleEXT` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_discard_rectangle(
        &mut self,
        first_rectangle: u32,
        rectangles: impl IntoIterator<Item = Scissor>,
    ) {
        debug_assert!(self.device.enabled_extensions().ext_discard_rectangles);

        let rectangles = rectangles
            .into_iter()
            .map(|v| v.clone().into())
            .collect::<SmallVec<[_; 2]>>();
        if rectangles.is_empty() {
            return;
        }

        debug_assert!(
            first_rectangle + rectangles.len() as u32
                <= self
                    .device
                    .physical_device()
                    .properties()
                    .max_discard_rectangles
                    .unwrap()
        );

        let fns = self.device.fns();
        fns.ext_discard_rectangles.cmd_set_discard_rectangle_ext(
            self.handle,
            first_rectangle,
            rectangles.len() as u32,
            rectangles.as_ptr(),
        );
    }

    /// Calls `vkCmdSetFrontFaceEXT` on the builder.
    #[inline]
    pub unsafe fn set_front_face(&mut self, face: FrontFace) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            fns.v1_3.cmd_set_front_face(self.handle, face.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            debug_assert!(self.device.enabled_features().extended_dynamic_state);
            fns.ext_extended_dynamic_state
                .cmd_set_front_face_ext(self.handle, face.into());
        }
    }

    /// Calls `vkCmdSetLineStippleEXT` on the builder.
    #[inline]
    pub unsafe fn set_line_stipple(&mut self, factor: u32, pattern: u16) {
        debug_assert!(self.device.enabled_extensions().ext_line_rasterization);
        debug_assert!(factor >= 1 && factor <= 256);
        let fns = self.device.fns();
        fns.ext_line_rasterization
            .cmd_set_line_stipple_ext(self.handle, factor, pattern);
    }

    /// Calls `vkCmdSetLineWidth` on the builder.
    #[inline]
    pub unsafe fn set_line_width(&mut self, line_width: f32) {
        debug_assert!(line_width == 1.0 || self.device.enabled_features().wide_lines);
        let fns = self.device.fns();
        fns.v1_0.cmd_set_line_width(self.handle, line_width);
    }

    /// Calls `vkCmdSetLogicOpEXT` on the builder.
    #[inline]
    pub unsafe fn set_logic_op(&mut self, logic_op: LogicOp) {
        debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state2);
        debug_assert!(
            self.device
                .enabled_features()
                .extended_dynamic_state2_logic_op
        );
        let fns = self.device.fns();

        fns.ext_extended_dynamic_state2
            .cmd_set_logic_op_ext(self.handle, logic_op.into());
    }

    /// Calls `vkCmdSetPatchControlPointsEXT` on the builder.
    #[inline]
    pub unsafe fn set_patch_control_points(&mut self, num: u32) {
        debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state2);
        debug_assert!(
            self.device
                .enabled_features()
                .extended_dynamic_state2_patch_control_points
        );
        debug_assert!(num > 0);
        debug_assert!(
            num as u32
                <= self
                    .device
                    .physical_device()
                    .properties()
                    .max_tessellation_patch_size
        );
        let fns = self.device.fns();
        fns.ext_extended_dynamic_state2
            .cmd_set_patch_control_points_ext(self.handle, num);
    }

    /// Calls `vkCmdSetPrimitiveRestartEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_primitive_restart_enable(&mut self, enable: bool) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            fns.v1_3
                .cmd_set_primitive_restart_enable(self.handle, enable.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state2);
            debug_assert!(self.device.enabled_features().extended_dynamic_state2);
            fns.ext_extended_dynamic_state2
                .cmd_set_primitive_restart_enable_ext(self.handle, enable.into());
        }
    }

    /// Calls `vkCmdSetPrimitiveTopologyEXT` on the builder.
    #[inline]
    pub unsafe fn set_primitive_topology(&mut self, topology: PrimitiveTopology) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            fns.v1_3
                .cmd_set_primitive_topology(self.handle, topology.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            debug_assert!(self.device.enabled_features().extended_dynamic_state);
            fns.ext_extended_dynamic_state
                .cmd_set_primitive_topology_ext(self.handle, topology.into());
        }
    }

    /// Calls `vkCmdSetRasterizerDiscardEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_rasterizer_discard_enable(&mut self, enable: bool) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            fns.v1_3
                .cmd_set_rasterizer_discard_enable(self.handle, enable.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state2);
            debug_assert!(self.device.enabled_features().extended_dynamic_state2);
            fns.ext_extended_dynamic_state2
                .cmd_set_rasterizer_discard_enable_ext(self.handle, enable.into());
        }
    }

    /// Calls `vkCmdSetStencilCompareMask` on the builder.
    #[inline]
    pub unsafe fn set_stencil_compare_mask(&mut self, face_mask: StencilFaces, compare_mask: u32) {
        let fns = self.device.fns();
        fns.v1_0
            .cmd_set_stencil_compare_mask(self.handle, face_mask.into(), compare_mask);
    }

    /// Calls `vkCmdSetStencilOpEXT` on the builder.
    #[inline]
    pub unsafe fn set_stencil_op(
        &mut self,
        face_mask: StencilFaces,
        fail_op: StencilOp,
        pass_op: StencilOp,
        depth_fail_op: StencilOp,
        compare_op: CompareOp,
    ) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            fns.v1_3.cmd_set_stencil_op(
                self.handle,
                face_mask.into(),
                fail_op.into(),
                pass_op.into(),
                depth_fail_op.into(),
                compare_op.into(),
            );
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            debug_assert!(self.device.enabled_features().extended_dynamic_state);
            fns.ext_extended_dynamic_state.cmd_set_stencil_op_ext(
                self.handle,
                face_mask.into(),
                fail_op.into(),
                pass_op.into(),
                depth_fail_op.into(),
                compare_op.into(),
            );
        }
    }

    /// Calls `vkCmdSetStencilReference` on the builder.
    #[inline]
    pub unsafe fn set_stencil_reference(&mut self, face_mask: StencilFaces, reference: u32) {
        let fns = self.device.fns();
        fns.v1_0
            .cmd_set_stencil_reference(self.handle, face_mask.into(), reference);
    }

    /// Calls `vkCmdSetStencilTestEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_stencil_test_enable(&mut self, enable: bool) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            fns.v1_3
                .cmd_set_stencil_test_enable(self.handle, enable.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            debug_assert!(self.device.enabled_features().extended_dynamic_state);
            fns.ext_extended_dynamic_state
                .cmd_set_stencil_test_enable_ext(self.handle, enable.into());
        }
    }

    /// Calls `vkCmdSetStencilWriteMask` on the builder.
    #[inline]
    pub unsafe fn set_stencil_write_mask(&mut self, face_mask: StencilFaces, write_mask: u32) {
        let fns = self.device.fns();
        fns.v1_0
            .cmd_set_stencil_write_mask(self.handle, face_mask.into(), write_mask);
    }

    /// Calls `vkCmdSetScissor` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_scissor(
        &mut self,
        first_scissor: u32,
        scissors: impl IntoIterator<Item = Scissor>,
    ) {
        let scissors = scissors
            .into_iter()
            .map(|v| ash::vk::Rect2D::from(v.clone()))
            .collect::<SmallVec<[_; 2]>>();
        if scissors.is_empty() {
            return;
        }

        debug_assert!(scissors.iter().all(|s| s.offset.x >= 0 && s.offset.y >= 0));
        debug_assert!(scissors.iter().all(|s| {
            s.extent.width < i32::MAX as u32
                && s.extent.height < i32::MAX as u32
                && s.offset.x.checked_add(s.extent.width as i32).is_some()
                && s.offset.y.checked_add(s.extent.height as i32).is_some()
        }));
        debug_assert!(
            (first_scissor == 0 && scissors.len() == 1)
                || self.device.enabled_features().multi_viewport
        );
        debug_assert!(
            first_scissor + scissors.len() as u32
                <= self.device.physical_device().properties().max_viewports
        );

        let fns = self.device.fns();
        fns.v1_0.cmd_set_scissor(
            self.handle,
            first_scissor,
            scissors.len() as u32,
            scissors.as_ptr(),
        );
    }

    /// Calls `vkCmdSetScissorWithCountEXT` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_scissor_with_count(&mut self, scissors: impl IntoIterator<Item = Scissor>) {
        let scissors = scissors
            .into_iter()
            .map(|v| ash::vk::Rect2D::from(v.clone()))
            .collect::<SmallVec<[_; 2]>>();
        if scissors.is_empty() {
            return;
        }

        debug_assert!(scissors.iter().all(|s| s.offset.x >= 0 && s.offset.y >= 0));
        debug_assert!(scissors.iter().all(|s| {
            s.extent.width < i32::MAX as u32
                && s.extent.height < i32::MAX as u32
                && s.offset.x.checked_add(s.extent.width as i32).is_some()
                && s.offset.y.checked_add(s.extent.height as i32).is_some()
        }));
        debug_assert!(scissors.len() == 1 || self.device.enabled_features().multi_viewport);
        debug_assert!(
            scissors.len() as u32 <= self.device.physical_device().properties().max_viewports
        );

        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            fns.v1_3.cmd_set_scissor_with_count(
                self.handle,
                scissors.len() as u32,
                scissors.as_ptr(),
            );
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            debug_assert!(self.device.enabled_features().extended_dynamic_state);
            fns.ext_extended_dynamic_state
                .cmd_set_scissor_with_count_ext(
                    self.handle,
                    scissors.len() as u32,
                    scissors.as_ptr(),
                );
        }
    }

    /// Calls `vkCmdSetViewport` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_viewport(
        &mut self,
        first_viewport: u32,
        viewports: impl IntoIterator<Item = Viewport>,
    ) {
        let viewports = viewports
            .into_iter()
            .map(|v| v.clone().into())
            .collect::<SmallVec<[_; 2]>>();
        if viewports.is_empty() {
            return;
        }

        debug_assert!(
            (first_viewport == 0 && viewports.len() == 1)
                || self.device.enabled_features().multi_viewport
        );
        debug_assert!(
            first_viewport + viewports.len() as u32
                <= self.device.physical_device().properties().max_viewports
        );

        let fns = self.device.fns();
        fns.v1_0.cmd_set_viewport(
            self.handle,
            first_viewport,
            viewports.len() as u32,
            viewports.as_ptr(),
        );
    }

    /// Calls `vkCmdSetViewportWithCountEXT` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_viewport_with_count(
        &mut self,
        viewports: impl IntoIterator<Item = Viewport>,
    ) {
        let viewports = viewports
            .into_iter()
            .map(|v| v.clone().into())
            .collect::<SmallVec<[_; 2]>>();
        if viewports.is_empty() {
            return;
        }

        debug_assert!(viewports.len() == 1 || self.device.enabled_features().multi_viewport);
        debug_assert!(
            viewports.len() as u32 <= self.device.physical_device().properties().max_viewports
        );

        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            fns.v1_3.cmd_set_viewport_with_count(
                self.handle,
                viewports.len() as u32,
                viewports.as_ptr(),
            );
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            debug_assert!(self.device.enabled_features().extended_dynamic_state);
            fns.ext_extended_dynamic_state
                .cmd_set_viewport_with_count_ext(
                    self.handle,
                    viewports.len() as u32,
                    viewports.as_ptr(),
                );
        }
    }
}
