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
        allocator::CommandBufferAllocator,
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
    RequirementNotMet, RequiresOneOf, Version,
};
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    ops::RangeInclusive,
};

/// # Commands to set dynamic state for pipelines.
///
/// These commands require a queue with a pipeline type that uses the given state.
impl<L, A> AutoCommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    // Helper function for dynamic state setting.
    fn validate_pipeline_fixed_state(
        &self,
        state: DynamicState,
    ) -> Result<(), SetDynamicStateError> {
        // VUID-vkCmdDispatch-None-02859
        if self.state().pipeline_graphics().map_or(false, |pipeline| {
            matches!(pipeline.dynamic_state(state), Some(false))
        }) {
            return Err(SetDynamicStateError::PipelineHasFixedState);
        }

        Ok(())
    }

    /// Sets the dynamic blend constants for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    pub fn set_blend_constants(&mut self, constants: [f32; 4]) -> &mut Self {
        self.validate_set_blend_constants(constants).unwrap();

        unsafe {
            self.inner.set_blend_constants(constants);
        }

        self
    }

    fn validate_set_blend_constants(
        &self,
        _constants: [f32; 4],
    ) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::BlendConstants)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetBlendConstants-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        Ok(())
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
    pub fn set_color_write_enable<I>(&mut self, enables: I) -> &mut Self
    where
        I: IntoIterator<Item = bool>,
        I::IntoIter: ExactSizeIterator,
    {
        let enables = enables.into_iter();

        self.validate_set_color_write_enable(&enables).unwrap();

        unsafe {
            self.inner.set_color_write_enable(enables);
        }

        self
    }

    fn validate_set_color_write_enable(
        &self,
        enables: &impl ExactSizeIterator,
    ) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::ColorWriteEnable)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetColorWriteEnableEXT-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetColorWriteEnableEXT-None-04803
        if !self.device().enabled_features().color_write_enable {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`set_color_write_enable`",
                requires_one_of: RequiresOneOf {
                    device_extensions: &["ext_color_write_enable"],
                    ..Default::default()
                },
            });
        }

        if let Some(color_blend_state) = self
            .state()
            .pipeline_graphics()
            .and_then(|pipeline| pipeline.color_blend_state())
        {
            // VUID-vkCmdSetColorWriteEnableEXT-attachmentCount-06656
            // Indirectly checked
            if enables.len() != color_blend_state.attachments.len() {
                return Err(
                    SetDynamicStateError::PipelineColorBlendAttachmentCountMismatch {
                        provided_count: enables.len() as u32,
                        required_count: color_blend_state.attachments.len() as u32,
                    },
                );
            }
        }

        Ok(())
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
    pub fn set_cull_mode(&mut self, cull_mode: CullMode) -> &mut Self {
        self.validate_set_cull_mode(cull_mode).unwrap();

        unsafe {
            self.inner.set_cull_mode(cull_mode);
        }

        self
    }

    fn validate_set_cull_mode(&self, cull_mode: CullMode) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::CullMode)?;

        // VUID-vkCmdSetCullMode-cullMode-parameter
        cull_mode.validate_device(self.device())?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetCullMode-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetCullMode-None-03384
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`set_cull_mode`",
                requires_one_of: RequiresOneOf {
                    api_version: Some(Version::V1_3),
                    features: &["extended_dynamic_state"],
                    ..Default::default()
                },
            });
        }

        Ok(())
    }

    /// Sets the dynamic depth bias values for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    /// - If the [`depth_bias_clamp`](crate::device::Features::depth_bias_clamp)
    ///   feature is not enabled on the device, panics if `clamp` is not 0.0.
    pub fn set_depth_bias(
        &mut self,
        constant_factor: f32,
        clamp: f32,
        slope_factor: f32,
    ) -> &mut Self {
        self.validate_set_depth_bias(constant_factor, clamp, slope_factor)
            .unwrap();

        unsafe {
            self.inner
                .set_depth_bias(constant_factor, clamp, slope_factor);
        }

        self
    }

    fn validate_set_depth_bias(
        &self,
        _constant_factor: f32,
        clamp: f32,
        _slope_factor: f32,
    ) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::DepthBias)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetDepthBias-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetDepthBias-depthBiasClamp-00790
        if clamp != 0.0 && !self.device().enabled_features().depth_bias_clamp {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`clamp` is not `0.0`",
                requires_one_of: RequiresOneOf {
                    features: &["depth_bias_clamp"],
                    ..Default::default()
                },
            });
        }

        Ok(())
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
    pub fn set_depth_bias_enable(&mut self, enable: bool) -> &mut Self {
        self.validate_set_depth_bias_enable(enable).unwrap();

        unsafe {
            self.inner.set_depth_bias_enable(enable);
        }

        self
    }

    fn validate_set_depth_bias_enable(&self, _enable: bool) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::DepthBiasEnable)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetDepthBiasEnable-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetDepthBiasEnable-None-04872
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state2)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`set_depth_bias_enable`",
                requires_one_of: RequiresOneOf {
                    api_version: Some(Version::V1_3),
                    features: &["extended_dynamic_state2"],
                    ..Default::default()
                },
            });
        }

        Ok(())
    }

    /// Sets the dynamic depth bounds for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    /// - If the
    ///   [`ext_depth_range_unrestricted`](crate::device::DeviceExtensions::ext_depth_range_unrestricted)
    ///   device extension is not enabled, panics if the start and end of `bounds` are not between
    ///   0.0 and 1.0 inclusive.
    pub fn set_depth_bounds(&mut self, bounds: RangeInclusive<f32>) -> &mut Self {
        self.validate_set_depth_bounds(bounds.clone()).unwrap();

        unsafe {
            self.inner.set_depth_bounds(bounds);
        }

        self
    }

    fn validate_set_depth_bounds(
        &self,
        bounds: RangeInclusive<f32>,
    ) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::DepthBounds)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetDepthBounds-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetDepthBounds-minDepthBounds-00600
        // VUID-vkCmdSetDepthBounds-maxDepthBounds-00601
        if !self
            .device()
            .enabled_extensions()
            .ext_depth_range_unrestricted
            && !((0.0..=1.0).contains(bounds.start()) && (0.0..=1.0).contains(bounds.end()))
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`bounds` is not between `0.0` and `1.0` inclusive",
                requires_one_of: RequiresOneOf {
                    device_extensions: &["ext_depth_range_unrestricted"],
                    ..Default::default()
                },
            });
        }

        Ok(())
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
    pub fn set_depth_bounds_test_enable(&mut self, enable: bool) -> &mut Self {
        self.validate_set_depth_bounds_test_enable(enable).unwrap();

        unsafe {
            self.inner.set_depth_bounds_test_enable(enable);
        }

        self
    }

    fn validate_set_depth_bounds_test_enable(
        &self,
        _enable: bool,
    ) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::DepthBoundsTestEnable)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetDepthBoundsTestEnable-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetDepthBoundsTestEnable-None-03349
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`set_depth_bounds_test_enable`",
                requires_one_of: RequiresOneOf {
                    api_version: Some(Version::V1_3),
                    features: &["extended_dynamic_state"],
                    ..Default::default()
                },
            });
        }

        Ok(())
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
    pub fn set_depth_compare_op(&mut self, compare_op: CompareOp) -> &mut Self {
        self.validate_set_depth_compare_op(compare_op).unwrap();

        unsafe {
            self.inner.set_depth_compare_op(compare_op);
        }

        self
    }

    fn validate_set_depth_compare_op(
        &self,
        compare_op: CompareOp,
    ) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::DepthCompareOp)?;

        // VUID-vkCmdSetDepthCompareOp-depthCompareOp-parameter
        compare_op.validate_device(self.device())?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetDepthCompareOp-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetDepthCompareOp-None-03353
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`set_depth_compare_op`",
                requires_one_of: RequiresOneOf {
                    api_version: Some(Version::V1_3),
                    features: &["extended_dynamic_state"],
                    ..Default::default()
                },
            });
        }

        Ok(())
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
    pub fn set_depth_test_enable(&mut self, enable: bool) -> &mut Self {
        self.validate_set_depth_test_enable(enable).unwrap();

        unsafe {
            self.inner.set_depth_test_enable(enable);
        }

        self
    }

    fn validate_set_depth_test_enable(&self, _enable: bool) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::DepthTestEnable)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetDepthTestEnable-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetDepthTestEnable-None-03352
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`set_depth_test_enable`",
                requires_one_of: RequiresOneOf {
                    api_version: Some(Version::V1_3),
                    features: &["extended_dynamic_state"],
                    ..Default::default()
                },
            });
        }

        Ok(())
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
    pub fn set_depth_write_enable(&mut self, enable: bool) -> &mut Self {
        self.validate_set_depth_write_enable(enable).unwrap();

        unsafe {
            self.inner.set_depth_write_enable(enable);
        }

        self
    }

    fn validate_set_depth_write_enable(&self, _enable: bool) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::DepthWriteEnable)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetDepthWriteEnable-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetDepthWriteEnable-None-03354
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`set_depth_write_enable`",
                requires_one_of: RequiresOneOf {
                    api_version: Some(Version::V1_3),
                    features: &["extended_dynamic_state"],
                    ..Default::default()
                },
            });
        }

        Ok(())
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
    pub fn set_discard_rectangle(
        &mut self,
        first_rectangle: u32,
        rectangles: impl IntoIterator<Item = Scissor>,
    ) -> &mut Self {
        let rectangles: SmallVec<[Scissor; 2]> = rectangles.into_iter().collect();
        self.validate_set_discard_rectangle(first_rectangle, &rectangles)
            .unwrap();

        unsafe {
            self.inner
                .set_discard_rectangle(first_rectangle, rectangles);
        }

        self
    }

    fn validate_set_discard_rectangle(
        &self,
        first_rectangle: u32,
        rectangles: &[Scissor],
    ) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::DiscardRectangle)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetDiscardRectangle-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        if self.device().enabled_extensions().ext_discard_rectangles {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`set_discard_rectangle`",
                requires_one_of: RequiresOneOf {
                    device_extensions: &["ext_discard_rectangles"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkCmdSetDiscardRectangleEXT-firstDiscardRectangle-00585
        if first_rectangle + rectangles.len() as u32
            > self
                .device()
                .physical_device()
                .properties()
                .max_discard_rectangles
                .unwrap()
        {
            return Err(SetDynamicStateError::MaxDiscardRectanglesExceeded {
                provided: first_rectangle + rectangles.len() as u32,
                max: self
                    .device()
                    .physical_device()
                    .properties()
                    .max_discard_rectangles
                    .unwrap(),
            });
        }

        Ok(())
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
    pub fn set_front_face(&mut self, face: FrontFace) -> &mut Self {
        self.validate_set_front_face(face).unwrap();

        unsafe {
            self.inner.set_front_face(face);
        }

        self
    }

    fn validate_set_front_face(&self, face: FrontFace) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::FrontFace)?;

        // VUID-vkCmdSetFrontFace-frontFace-parameter
        face.validate_device(self.device())?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetFrontFace-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetFrontFace-None-03383
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`set_front_face`",
                requires_one_of: RequiresOneOf {
                    api_version: Some(Version::V1_3),
                    features: &["extended_dynamic_state"],
                    ..Default::default()
                },
            });
        }

        Ok(())
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
    pub fn set_line_stipple(&mut self, factor: u32, pattern: u16) -> &mut Self {
        self.validate_set_line_stipple(factor, pattern).unwrap();

        unsafe {
            self.inner.set_line_stipple(factor, pattern);
        }

        self
    }

    fn validate_set_line_stipple(
        &self,
        factor: u32,
        _pattern: u16,
    ) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::LineStipple)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetLineStippleEXT-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        if !self.device().enabled_extensions().ext_line_rasterization {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`set_line_stipple`",
                requires_one_of: RequiresOneOf {
                    device_extensions: &["ext_line_rasterization"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkCmdSetLineStippleEXT-lineStippleFactor-02776
        if !(1..=256).contains(&factor) {
            return Err(SetDynamicStateError::FactorOutOfRange);
        }

        Ok(())
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
        self.validate_set_line_width(line_width).unwrap();

        unsafe {
            self.inner.set_line_width(line_width);
        }

        self
    }

    fn validate_set_line_width(&self, line_width: f32) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::LineWidth)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetLineWidth-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetLineWidth-lineWidth-00788
        if !self.device().enabled_features().wide_lines && line_width != 1.0 {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`line_width` is not `1.0`",
                requires_one_of: RequiresOneOf {
                    features: &["wide_lines"],
                    ..Default::default()
                },
            });
        }

        Ok(())
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
    pub fn set_logic_op(&mut self, logic_op: LogicOp) -> &mut Self {
        self.validate_set_logic_op(logic_op).unwrap();

        unsafe {
            self.inner.set_logic_op(logic_op);
        }

        self
    }

    fn validate_set_logic_op(&self, logic_op: LogicOp) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::LogicOp)?;

        // VUID-vkCmdSetLogicOpEXT-logicOp-parameter
        logic_op.validate_device(self.device())?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetLogicOpEXT-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetLogicOpEXT-None-04867
        if !self
            .device()
            .enabled_features()
            .extended_dynamic_state2_logic_op
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`set_logic_op`",
                requires_one_of: RequiresOneOf {
                    features: &["extended_dynamic_state2_logic_op"],
                    ..Default::default()
                },
            });
        }

        Ok(())
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
    pub fn set_patch_control_points(&mut self, num: u32) -> &mut Self {
        self.validate_set_patch_control_points(num).unwrap();

        unsafe {
            self.inner.set_patch_control_points(num);
        }

        self
    }

    fn validate_set_patch_control_points(&self, num: u32) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::PatchControlPoints)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetPatchControlPointsEXT-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetPatchControlPointsEXT-None-04873
        if !self
            .device()
            .enabled_features()
            .extended_dynamic_state2_patch_control_points
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`set_patch_control_points`",
                requires_one_of: RequiresOneOf {
                    features: &["extended_dynamic_state2_patch_control_points"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkCmdSetPatchControlPointsEXT-patchControlPoints-04874
        assert!(num > 0, "num must be greater than 0");

        // VUID-vkCmdSetPatchControlPointsEXT-patchControlPoints-04874
        if num
            > self
                .device()
                .physical_device()
                .properties()
                .max_tessellation_patch_size
        {
            return Err(SetDynamicStateError::MaxTessellationPatchSizeExceeded {
                provided: num,
                max: self
                    .device()
                    .physical_device()
                    .properties()
                    .max_tessellation_patch_size,
            });
        }

        Ok(())
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
    pub fn set_primitive_restart_enable(&mut self, enable: bool) -> &mut Self {
        self.validate_set_primitive_restart_enable(enable).unwrap();

        unsafe {
            self.inner.set_primitive_restart_enable(enable);
        }

        self
    }

    fn validate_set_primitive_restart_enable(
        &self,
        _enable: bool,
    ) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::PrimitiveRestartEnable)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetPrimitiveRestartEnable-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetPrimitiveRestartEnable-None-04866
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state2)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`set_primitive_restart_enable`",
                requires_one_of: RequiresOneOf {
                    api_version: Some(Version::V1_3),
                    features: &["extended_dynamic_state2"],
                    ..Default::default()
                },
            });
        }

        Ok(())
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
    pub fn set_primitive_topology(&mut self, topology: PrimitiveTopology) -> &mut Self {
        self.validate_set_primitive_topology(topology).unwrap();

        unsafe {
            self.inner.set_primitive_topology(topology);
        }

        self
    }

    fn validate_set_primitive_topology(
        &self,
        topology: PrimitiveTopology,
    ) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::PrimitiveTopology)?;

        // VUID-vkCmdSetPrimitiveTopology-primitiveTopology-parameter
        topology.validate_device(self.device())?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetPrimitiveTopology-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetPrimitiveTopology-None-03347
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`set_primitive_topology`",
                requires_one_of: RequiresOneOf {
                    api_version: Some(Version::V1_3),
                    features: &["extended_dynamic_state"],
                    ..Default::default()
                },
            });
        }

        // VUID?
        // Since these requirements exist for fixed state when creating the pipeline,
        // I assume they exist for dynamic state as well.
        match topology {
            PrimitiveTopology::TriangleFan => {
                if self.device().enabled_extensions().khr_portability_subset
                    && !self.device().enabled_features().triangle_fans
                {
                    return Err(SetDynamicStateError::RequirementNotMet {
                        required_for:
                            "the `khr_portability_subset` extension is enabled on the device, and \
                            `topology` is `PrimitiveTopology::TriangleFan`",
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
                if !self.device().enabled_features().geometry_shader {
                    return Err(SetDynamicStateError::RequirementNotMet {
                        required_for: "`topology` is `PrimitiveTopology::*WithAdjacency`",
                        requires_one_of: RequiresOneOf {
                            features: &["geometry_shader"],
                            ..Default::default()
                        },
                    });
                }
            }
            PrimitiveTopology::PatchList => {
                if !self.device().enabled_features().tessellation_shader {
                    return Err(SetDynamicStateError::RequirementNotMet {
                        required_for: "`topology` is `PrimitiveTopology::PatchList`",
                        requires_one_of: RequiresOneOf {
                            features: &["tessellation_shader"],
                            ..Default::default()
                        },
                    });
                }
            }
            _ => (),
        }

        Ok(())
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
    pub fn set_rasterizer_discard_enable(&mut self, enable: bool) -> &mut Self {
        self.validate_set_rasterizer_discard_enable(enable).unwrap();

        unsafe {
            self.inner.set_rasterizer_discard_enable(enable);
        }

        self
    }

    fn validate_set_rasterizer_discard_enable(
        &self,
        _enable: bool,
    ) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::RasterizerDiscardEnable)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetRasterizerDiscardEnable-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetRasterizerDiscardEnable-None-04871
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state2)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`set_rasterizer_discard_enable`",
                requires_one_of: RequiresOneOf {
                    api_version: Some(Version::V1_3),
                    features: &["extended_dynamic_state2"],
                    ..Default::default()
                },
            });
        }

        Ok(())
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
    pub fn set_scissor(
        &mut self,
        first_scissor: u32,
        scissors: impl IntoIterator<Item = Scissor>,
    ) -> &mut Self {
        let scissors: SmallVec<[Scissor; 2]> = scissors.into_iter().collect();
        self.validate_set_scissor(first_scissor, &scissors).unwrap();

        unsafe {
            self.inner.set_scissor(first_scissor, scissors);
        }

        self
    }

    fn validate_set_scissor(
        &self,
        first_scissor: u32,
        scissors: &[Scissor],
    ) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::Scissor)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetScissor-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetScissor-firstScissor-00592
        if first_scissor + scissors.len() as u32
            > self.device().physical_device().properties().max_viewports
        {
            return Err(SetDynamicStateError::MaxViewportsExceeded {
                provided: first_scissor + scissors.len() as u32,
                max: self.device().physical_device().properties().max_viewports,
            });
        }

        if !self.device().enabled_features().multi_viewport {
            // VUID-vkCmdSetScissor-firstScissor-00593
            if first_scissor != 0 {
                return Err(SetDynamicStateError::RequirementNotMet {
                    required_for: "`first_scissor` is not `0`",
                    requires_one_of: RequiresOneOf {
                        features: &["multi_viewport"],
                        ..Default::default()
                    },
                });
            }

            // VUID-vkCmdSetScissor-scissorCount-00594
            if scissors.len() > 1 {
                return Err(SetDynamicStateError::RequirementNotMet {
                    required_for: "`scissors.len()` is greater than `1`",
                    requires_one_of: RequiresOneOf {
                        features: &["multi_viewport"],
                        ..Default::default()
                    },
                });
            }
        }

        Ok(())
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
    pub fn set_scissor_with_count(
        &mut self,
        scissors: impl IntoIterator<Item = Scissor>,
    ) -> &mut Self {
        let scissors: SmallVec<[Scissor; 2]> = scissors.into_iter().collect();
        self.validate_set_scissor_with_count(&scissors).unwrap();

        unsafe {
            self.inner.set_scissor_with_count(scissors);
        }

        self
    }

    fn validate_set_scissor_with_count(
        &self,
        scissors: &[Scissor],
    ) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::ScissorWithCount)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetScissorWithCount-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetScissorWithCount-None-03396
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`set_scissor_with_count`",
                requires_one_of: RequiresOneOf {
                    api_version: Some(Version::V1_3),
                    features: &["extended_dynamic_state"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkCmdSetScissorWithCount-scissorCount-03397
        if scissors.len() as u32 > self.device().physical_device().properties().max_viewports {
            return Err(SetDynamicStateError::MaxViewportsExceeded {
                provided: scissors.len() as u32,
                max: self.device().physical_device().properties().max_viewports,
            });
        }

        // VUID-vkCmdSetScissorWithCount-scissorCount-03398
        if !self.device().enabled_features().multi_viewport && scissors.len() > 1 {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`scissors.len()` is greater than `1`",
                requires_one_of: RequiresOneOf {
                    features: &["multi_viewport"],
                    ..Default::default()
                },
            });
        }

        Ok(())
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
        self.validate_set_stencil_compare_mask(faces, compare_mask)
            .unwrap();

        unsafe {
            self.inner.set_stencil_compare_mask(faces, compare_mask);
        }

        self
    }

    fn validate_set_stencil_compare_mask(
        &self,
        faces: StencilFaces,
        _compare_mask: u32,
    ) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::StencilCompareMask)?;

        // VUID-vkCmdSetStencilCompareMask-faceMask-parameter
        faces.validate_device(self.device())?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetStencilCompareMask-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        Ok(())
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
    pub fn set_stencil_op(
        &mut self,
        faces: StencilFaces,
        fail_op: StencilOp,
        pass_op: StencilOp,
        depth_fail_op: StencilOp,
        compare_op: CompareOp,
    ) -> &mut Self {
        self.validate_set_stencil_op(faces, fail_op, pass_op, depth_fail_op, compare_op)
            .unwrap();

        unsafe {
            self.inner
                .set_stencil_op(faces, fail_op, pass_op, depth_fail_op, compare_op);
        }

        self
    }

    fn validate_set_stencil_op(
        &self,
        faces: StencilFaces,
        fail_op: StencilOp,
        pass_op: StencilOp,
        depth_fail_op: StencilOp,
        compare_op: CompareOp,
    ) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::StencilOp)?;

        // VUID-vkCmdSetStencilOp-faceMask-parameter
        faces.validate_device(self.device())?;

        // VUID-vkCmdSetStencilOp-failOp-parameter
        fail_op.validate_device(self.device())?;

        // VUID-vkCmdSetStencilOp-passOp-parameter
        pass_op.validate_device(self.device())?;

        // VUID-vkCmdSetStencilOp-depthFailOp-parameter
        depth_fail_op.validate_device(self.device())?;

        // VUID-vkCmdSetStencilOp-compareOp-parameter
        compare_op.validate_device(self.device())?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetStencilOp-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetStencilOp-None-03351
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`set_stencil_op`",
                requires_one_of: RequiresOneOf {
                    api_version: Some(Version::V1_3),
                    features: &["extended_dynamic_state"],
                    ..Default::default()
                },
            });
        }

        Ok(())
    }

    /// Sets the dynamic stencil reference on one or both faces for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    pub fn set_stencil_reference(&mut self, faces: StencilFaces, reference: u32) -> &mut Self {
        self.validate_set_stencil_reference(faces, reference)
            .unwrap();

        unsafe {
            self.inner.set_stencil_reference(faces, reference);
        }

        self
    }

    fn validate_set_stencil_reference(
        &self,
        faces: StencilFaces,
        _reference: u32,
    ) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::StencilReference)?;

        // VUID-vkCmdSetStencilReference-faceMask-parameter
        faces.validate_device(self.device())?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetStencilReference-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        Ok(())
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
    pub fn set_stencil_test_enable(&mut self, enable: bool) -> &mut Self {
        self.validate_set_stencil_test_enable(enable).unwrap();

        unsafe {
            self.inner.set_stencil_test_enable(enable);
        }

        self
    }

    fn validate_set_stencil_test_enable(&self, _enable: bool) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::StencilTestEnable)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetStencilTestEnable-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetStencilTestEnable-None-03350
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`set_stencil_test_enable`",
                requires_one_of: RequiresOneOf {
                    api_version: Some(Version::V1_3),
                    features: &["extended_dynamic_state"],
                    ..Default::default()
                },
            });
        }

        Ok(())
    }

    /// Sets the dynamic stencil write mask on one or both faces for future draw calls.
    ///
    /// # Panics
    ///
    /// - Panics if the queue family of the command buffer does not support graphics operations.
    /// - Panics if the currently bound graphics pipeline already contains this state internally.
    pub fn set_stencil_write_mask(&mut self, faces: StencilFaces, write_mask: u32) -> &mut Self {
        self.validate_set_stencil_write_mask(faces, write_mask)
            .unwrap();

        unsafe {
            self.inner.set_stencil_write_mask(faces, write_mask);
        }

        self
    }

    fn validate_set_stencil_write_mask(
        &self,
        faces: StencilFaces,
        _write_mask: u32,
    ) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::StencilWriteMask)?;

        // VUID-vkCmdSetStencilWriteMask-faceMask-parameter
        faces.validate_device(self.device())?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetStencilWriteMask-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        Ok(())
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
    pub fn set_viewport(
        &mut self,
        first_viewport: u32,
        viewports: impl IntoIterator<Item = Viewport>,
    ) -> &mut Self {
        let viewports: SmallVec<[Viewport; 2]> = viewports.into_iter().collect();
        self.validate_set_viewport(first_viewport, &viewports)
            .unwrap();

        unsafe {
            self.inner.set_viewport(first_viewport, viewports);
        }

        self
    }

    fn validate_set_viewport(
        &self,
        first_viewport: u32,
        viewports: &[Viewport],
    ) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::Viewport)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetViewport-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetViewport-firstViewport-01223
        if first_viewport + viewports.len() as u32
            > self.device().physical_device().properties().max_viewports
        {
            return Err(SetDynamicStateError::MaxViewportsExceeded {
                provided: first_viewport + viewports.len() as u32,
                max: self.device().physical_device().properties().max_viewports,
            });
        }

        if !self.device().enabled_features().multi_viewport {
            // VUID-vkCmdSetViewport-firstViewport-01224
            if first_viewport != 0 {
                return Err(SetDynamicStateError::RequirementNotMet {
                    required_for: "`first_scissors` is not `0`",
                    requires_one_of: RequiresOneOf {
                        features: &["multi_viewport"],
                        ..Default::default()
                    },
                });
            }

            // VUID-vkCmdSetViewport-viewportCount-01225
            if viewports.len() > 1 {
                return Err(SetDynamicStateError::RequirementNotMet {
                    required_for: "`viewports.len()` is greater than `1`",
                    requires_one_of: RequiresOneOf {
                        features: &["multi_viewport"],
                        ..Default::default()
                    },
                });
            }
        }

        Ok(())
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
    pub fn set_viewport_with_count(
        &mut self,
        viewports: impl IntoIterator<Item = Viewport>,
    ) -> &mut Self {
        let viewports: SmallVec<[Viewport; 2]> = viewports.into_iter().collect();
        self.validate_set_viewport_with_count(&viewports).unwrap();

        unsafe {
            self.inner.set_viewport_with_count(viewports);
        }

        self
    }

    fn validate_set_viewport_with_count(
        &self,
        viewports: &[Viewport],
    ) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::ViewportWithCount)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetViewportWithCount-commandBuffer-cmdpool
        if !queue_family_properties.queue_flags.graphics {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetViewportWithCount-None-03393
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`set_viewport_with_count`",
                requires_one_of: RequiresOneOf {
                    api_version: Some(Version::V1_3),
                    features: &["extended_dynamic_state"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkCmdSetViewportWithCount-viewportCount-03394
        if viewports.len() as u32 > self.device().physical_device().properties().max_viewports {
            return Err(SetDynamicStateError::MaxViewportsExceeded {
                provided: viewports.len() as u32,
                max: self.device().physical_device().properties().max_viewports,
            });
        }

        // VUID-vkCmdSetViewportWithCount-viewportCount-03395
        if !self.device().enabled_features().multi_viewport && viewports.len() > 1 {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`viewports.len()` is greater than `1`",
                requires_one_of: RequiresOneOf {
                    features: &["multi_viewport"],
                    ..Default::default()
                },
            });
        }

        Ok(())
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
    pub unsafe fn set_color_write_enable(&mut self, enables: impl IntoIterator<Item = bool>) {
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
    pub unsafe fn set_depth_bounds(&mut self, bounds: RangeInclusive<f32>) {
        struct Cmd {
            bounds: RangeInclusive<f32>,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "set_depth_bounds"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.set_depth_bounds(self.bounds.clone());
            }
        }

        self.commands.push(Box::new(Cmd {
            bounds: bounds.clone(),
        }));
        self.current_state.depth_bounds = Some(bounds);
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
    pub unsafe fn set_discard_rectangle(
        &mut self,
        first_rectangle: u32,
        rectangles: impl IntoIterator<Item = Scissor>,
    ) {
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
            self.current_state.discard_rectangle.insert(num, *rectangle);
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
    pub unsafe fn set_scissor(
        &mut self,
        first_scissor: u32,
        scissors: impl IntoIterator<Item = Scissor>,
    ) {
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
            self.current_state.scissor.insert(num, *scissor);
        }

        self.commands.push(Box::new(Cmd {
            first_scissor,
            scissors: Mutex::new(scissors),
        }));
    }

    /// Calls `vkCmdSetScissorWithCountEXT` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    pub unsafe fn set_scissor_with_count(&mut self, scissors: impl IntoIterator<Item = Scissor>) {
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
    pub unsafe fn set_viewport(
        &mut self,
        first_viewport: u32,
        viewports: impl IntoIterator<Item = Viewport>,
    ) {
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
    pub unsafe fn set_viewport_with_count(
        &mut self,
        viewports: impl IntoIterator<Item = Viewport>,
    ) {
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
        (fns.v1_0.cmd_set_blend_constants)(self.handle, &constants);
    }

    /// Calls `vkCmdSetColorWriteEnableEXT` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    pub unsafe fn set_color_write_enable(&mut self, enables: impl IntoIterator<Item = bool>) {
        debug_assert!(self.device.enabled_extensions().ext_color_write_enable);

        let enables = enables
            .into_iter()
            .map(|v| v as ash::vk::Bool32)
            .collect::<SmallVec<[_; 4]>>();
        if enables.is_empty() {
            return;
        }

        let fns = self.device.fns();
        (fns.ext_color_write_enable.cmd_set_color_write_enable_ext)(
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
            (fns.v1_3.cmd_set_cull_mode)(self.handle, cull_mode.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            (fns.ext_extended_dynamic_state.cmd_set_cull_mode_ext)(self.handle, cull_mode.into());
        }
    }

    /// Calls `vkCmdSetDepthBias` on the builder.
    #[inline]
    pub unsafe fn set_depth_bias(&mut self, constant_factor: f32, clamp: f32, slope_factor: f32) {
        let fns = self.device.fns();
        (fns.v1_0.cmd_set_depth_bias)(self.handle, constant_factor, clamp, slope_factor);
    }

    /// Calls `vkCmdSetDepthBiasEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_depth_bias_enable(&mut self, enable: bool) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_depth_bias_enable)(self.handle, enable.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state2);
            (fns.ext_extended_dynamic_state2
                .cmd_set_depth_bias_enable_ext)(self.handle, enable.into());
        }
    }

    /// Calls `vkCmdSetDepthBounds` on the builder.
    #[inline]
    pub unsafe fn set_depth_bounds(&mut self, bounds: RangeInclusive<f32>) {
        let fns = self.device.fns();
        (fns.v1_0.cmd_set_depth_bounds)(self.handle, *bounds.start(), *bounds.end());
    }

    /// Calls `vkCmdSetDepthBoundsTestEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_depth_bounds_test_enable(&mut self, enable: bool) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_depth_bounds_test_enable)(self.handle, enable.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            (fns.ext_extended_dynamic_state
                .cmd_set_depth_bounds_test_enable_ext)(self.handle, enable.into());
        }
    }

    /// Calls `vkCmdSetDepthCompareOpEXT` on the builder.
    #[inline]
    pub unsafe fn set_depth_compare_op(&mut self, compare_op: CompareOp) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_depth_compare_op)(self.handle, compare_op.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            (fns.ext_extended_dynamic_state.cmd_set_depth_compare_op_ext)(
                self.handle,
                compare_op.into(),
            );
        }
    }

    /// Calls `vkCmdSetDepthTestEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_depth_test_enable(&mut self, enable: bool) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_depth_test_enable)(self.handle, enable.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            (fns.ext_extended_dynamic_state.cmd_set_depth_test_enable_ext)(
                self.handle,
                enable.into(),
            );
        }
    }

    /// Calls `vkCmdSetDepthWriteEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_depth_write_enable(&mut self, enable: bool) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_depth_write_enable)(self.handle, enable.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            (fns.ext_extended_dynamic_state
                .cmd_set_depth_write_enable_ext)(self.handle, enable.into());
        }
    }

    /// Calls `vkCmdSetDiscardRectangleEXT` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    pub unsafe fn set_discard_rectangle(
        &mut self,
        first_rectangle: u32,
        rectangles: impl IntoIterator<Item = Scissor>,
    ) {
        debug_assert!(self.device.enabled_extensions().ext_discard_rectangles);

        let rectangles = rectangles
            .into_iter()
            .map(|v| v.into())
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
        (fns.ext_discard_rectangles.cmd_set_discard_rectangle_ext)(
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
            (fns.v1_3.cmd_set_front_face)(self.handle, face.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            (fns.ext_extended_dynamic_state.cmd_set_front_face_ext)(self.handle, face.into());
        }
    }

    /// Calls `vkCmdSetLineStippleEXT` on the builder.
    #[inline]
    pub unsafe fn set_line_stipple(&mut self, factor: u32, pattern: u16) {
        debug_assert!(self.device.enabled_extensions().ext_line_rasterization);
        let fns = self.device.fns();
        (fns.ext_line_rasterization.cmd_set_line_stipple_ext)(self.handle, factor, pattern);
    }

    /// Calls `vkCmdSetLineWidth` on the builder.
    #[inline]
    pub unsafe fn set_line_width(&mut self, line_width: f32) {
        let fns = self.device.fns();
        (fns.v1_0.cmd_set_line_width)(self.handle, line_width);
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

        (fns.ext_extended_dynamic_state2.cmd_set_logic_op_ext)(self.handle, logic_op.into());
    }

    /// Calls `vkCmdSetPatchControlPointsEXT` on the builder.
    #[inline]
    pub unsafe fn set_patch_control_points(&mut self, num: u32) {
        debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state2);
        let fns = self.device.fns();
        (fns.ext_extended_dynamic_state2
            .cmd_set_patch_control_points_ext)(self.handle, num);
    }

    /// Calls `vkCmdSetPrimitiveRestartEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_primitive_restart_enable(&mut self, enable: bool) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_primitive_restart_enable)(self.handle, enable.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state2);
            (fns.ext_extended_dynamic_state2
                .cmd_set_primitive_restart_enable_ext)(self.handle, enable.into());
        }
    }

    /// Calls `vkCmdSetPrimitiveTopologyEXT` on the builder.
    #[inline]
    pub unsafe fn set_primitive_topology(&mut self, topology: PrimitiveTopology) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_primitive_topology)(self.handle, topology.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            (fns.ext_extended_dynamic_state
                .cmd_set_primitive_topology_ext)(self.handle, topology.into());
        }
    }

    /// Calls `vkCmdSetRasterizerDiscardEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_rasterizer_discard_enable(&mut self, enable: bool) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_rasterizer_discard_enable)(self.handle, enable.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state2);
            (fns.ext_extended_dynamic_state2
                .cmd_set_rasterizer_discard_enable_ext)(self.handle, enable.into());
        }
    }

    /// Calls `vkCmdSetStencilCompareMask` on the builder.
    #[inline]
    pub unsafe fn set_stencil_compare_mask(&mut self, face_mask: StencilFaces, compare_mask: u32) {
        let fns = self.device.fns();
        (fns.v1_0.cmd_set_stencil_compare_mask)(self.handle, face_mask.into(), compare_mask);
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
            (fns.v1_3.cmd_set_stencil_op)(
                self.handle,
                face_mask.into(),
                fail_op.into(),
                pass_op.into(),
                depth_fail_op.into(),
                compare_op.into(),
            );
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            (fns.ext_extended_dynamic_state.cmd_set_stencil_op_ext)(
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
        (fns.v1_0.cmd_set_stencil_reference)(self.handle, face_mask.into(), reference);
    }

    /// Calls `vkCmdSetStencilTestEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_stencil_test_enable(&mut self, enable: bool) {
        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_stencil_test_enable)(self.handle, enable.into());
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            (fns.ext_extended_dynamic_state
                .cmd_set_stencil_test_enable_ext)(self.handle, enable.into());
        }
    }

    /// Calls `vkCmdSetStencilWriteMask` on the builder.
    #[inline]
    pub unsafe fn set_stencil_write_mask(&mut self, face_mask: StencilFaces, write_mask: u32) {
        let fns = self.device.fns();
        (fns.v1_0.cmd_set_stencil_write_mask)(self.handle, face_mask.into(), write_mask);
    }

    /// Calls `vkCmdSetScissor` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    pub unsafe fn set_scissor(
        &mut self,
        first_scissor: u32,
        scissors: impl IntoIterator<Item = Scissor>,
    ) {
        let scissors = scissors
            .into_iter()
            .map(ash::vk::Rect2D::from)
            .collect::<SmallVec<[_; 2]>>();
        if scissors.is_empty() {
            return;
        }

        let fns = self.device.fns();
        (fns.v1_0.cmd_set_scissor)(
            self.handle,
            first_scissor,
            scissors.len() as u32,
            scissors.as_ptr(),
        );
    }

    /// Calls `vkCmdSetScissorWithCountEXT` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    pub unsafe fn set_scissor_with_count(&mut self, scissors: impl IntoIterator<Item = Scissor>) {
        let scissors = scissors
            .into_iter()
            .map(ash::vk::Rect2D::from)
            .collect::<SmallVec<[_; 2]>>();
        if scissors.is_empty() {
            return;
        }

        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_scissor_with_count)(
                self.handle,
                scissors.len() as u32,
                scissors.as_ptr(),
            );
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            (fns.ext_extended_dynamic_state
                .cmd_set_scissor_with_count_ext)(
                self.handle,
                scissors.len() as u32,
                scissors.as_ptr(),
            );
        }
    }

    /// Calls `vkCmdSetViewport` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    pub unsafe fn set_viewport(
        &mut self,
        first_viewport: u32,
        viewports: impl IntoIterator<Item = Viewport>,
    ) {
        let viewports = viewports
            .into_iter()
            .map(|v| v.into())
            .collect::<SmallVec<[_; 2]>>();
        if viewports.is_empty() {
            return;
        }

        let fns = self.device.fns();
        (fns.v1_0.cmd_set_viewport)(
            self.handle,
            first_viewport,
            viewports.len() as u32,
            viewports.as_ptr(),
        );
    }

    /// Calls `vkCmdSetViewportWithCountEXT` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    pub unsafe fn set_viewport_with_count(
        &mut self,
        viewports: impl IntoIterator<Item = Viewport>,
    ) {
        let viewports = viewports
            .into_iter()
            .map(|v| v.into())
            .collect::<SmallVec<[_; 2]>>();
        if viewports.is_empty() {
            return;
        }

        let fns = self.device.fns();

        if self.device.api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_viewport_with_count)(
                self.handle,
                viewports.len() as u32,
                viewports.as_ptr(),
            );
        } else {
            debug_assert!(self.device.enabled_extensions().ext_extended_dynamic_state);
            (fns.ext_extended_dynamic_state
                .cmd_set_viewport_with_count_ext)(
                self.handle,
                viewports.len() as u32,
                viewports.as_ptr(),
            );
        }
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
enum SetDynamicStateError {
    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// The provided `factor` is not between 1 and 256 inclusive.
    FactorOutOfRange,

    /// The [`max_discard_rectangles`](crate::device::Properties::max_discard_rectangles)
    /// limit has been exceeded.
    MaxDiscardRectanglesExceeded { provided: u32, max: u32 },

    /// The [`max_tessellation_patch_size`](crate::device::Properties::max_tessellation_patch_size)
    /// limit has been exceeded.
    MaxTessellationPatchSizeExceeded { provided: u32, max: u32 },

    /// The [`max_viewports`](crate::device::Properties::max_viewports)
    /// limit has been exceeded.
    MaxViewportsExceeded { provided: u32, max: u32 },

    /// The queue family doesn't allow this operation.
    NotSupportedByQueueFamily,

    /// The provided item count is different from the number of attachments in the color blend
    /// state of the currently bound pipeline.
    PipelineColorBlendAttachmentCountMismatch {
        provided_count: u32,
        required_count: u32,
    },

    /// The currently bound pipeline contains this state as internally fixed state, which cannot be
    /// overridden with dynamic state.
    PipelineHasFixedState,
}

impl Error for SetDynamicStateError {}

impl Display for SetDynamicStateError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
            Self::FactorOutOfRange => write!(
                f,
                "the provided `factor` is not between 1 and 256 inclusive",
            ),
            Self::MaxDiscardRectanglesExceeded { .. } => {
                write!(f, "the `max_discard_rectangles` limit has been exceeded")
            }
            Self::MaxTessellationPatchSizeExceeded { .. } => write!(
                f,
                "the `max_tessellation_patch_size` limit has been exceeded",
            ),
            Self::MaxViewportsExceeded { .. } => {
                write!(f, "the `max_viewports` limit has been exceeded")
            }
            Self::NotSupportedByQueueFamily => {
                write!(f, "the queue family doesn't allow this operation")
            }
            Self::PipelineColorBlendAttachmentCountMismatch {
                provided_count,
                required_count,
            } => write!(
                f,
                "the provided item count ({}) is different from the number of attachments in the \
                color blend state of the currently bound pipeline ({})",
                provided_count, required_count,
            ),
            Self::PipelineHasFixedState => write!(
                f,
                "the currently bound pipeline contains this state as internally fixed state, which \
                cannot be overridden with dynamic state",
            ),
        }
    }
}

impl From<RequirementNotMet> for SetDynamicStateError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}
