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
        allocator::CommandBufferAllocator, sys::UnsafeCommandBufferBuilder,
        AutoCommandBufferBuilder,
    },
    device::{DeviceOwned, QueueFlags},
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
    RequirementNotMet, Requires, RequiresAllOf, RequiresOneOf, Version, VulkanObject,
};
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
        if self
            .builder_state
            .pipeline_graphics
            .as_ref()
            .map_or(false, |pipeline| {
                matches!(pipeline.dynamic_state(state), Some(false))
            })
        {
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
            self.set_blend_constants_unchecked(constants);
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
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_blend_constants_unchecked(&mut self, constants: [f32; 4]) -> &mut Self {
        self.builder_state.blend_constants = Some(constants);
        self.add_command(
            "set_blend_constants",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_blend_constants(constants);
            },
        );

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
    pub fn set_color_write_enable(&mut self, enables: SmallVec<[bool; 4]>) -> &mut Self {
        self.validate_set_color_write_enable(&enables).unwrap();

        unsafe {
            self.set_color_write_enable_unchecked(enables);
        }

        self
    }

    fn validate_set_color_write_enable(
        &self,
        enables: &[bool],
    ) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::ColorWriteEnable)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetColorWriteEnableEXT-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetColorWriteEnableEXT-None-04803
        if !self.device().enabled_features().color_write_enable {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`AutoCommandBufferBuilder::set_color_write_enable`",
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "ext_color_write_enable",
                )])]),
            });
        }

        if let Some(color_blend_state) = self
            .builder_state
            .pipeline_graphics
            .as_ref()
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

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_color_write_enable_unchecked(
        &mut self,
        enables: SmallVec<[bool; 4]>,
    ) -> &mut Self {
        self.builder_state.color_write_enable = Some(enables.clone());
        self.add_command(
            "set_color_write_enable",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_color_write_enable(&enables);
            },
        );

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
    pub fn set_cull_mode(&mut self, cull_mode: CullMode) -> &mut Self {
        self.validate_set_cull_mode(cull_mode).unwrap();

        unsafe {
            self.set_cull_mode_unchecked(cull_mode);
        }

        self
    }

    fn validate_set_cull_mode(&self, cull_mode: CullMode) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::CullMode)?;

        // VUID-vkCmdSetCullMode-cullMode-parameter
        cull_mode.validate_device(self.device())?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetCullMode-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetCullMode-None-03384
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`AutoCommandBufferBuilder::set_cull_mode`",
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_cull_mode_unchecked(&mut self, cull_mode: CullMode) -> &mut Self {
        self.builder_state.cull_mode = Some(cull_mode);
        self.add_command(
            "set_cull_mode",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_cull_mode(cull_mode);
            },
        );

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
    pub fn set_depth_bias(
        &mut self,
        constant_factor: f32,
        clamp: f32,
        slope_factor: f32,
    ) -> &mut Self {
        self.validate_set_depth_bias(constant_factor, clamp, slope_factor)
            .unwrap();

        unsafe {
            self.set_depth_bias_unchecked(constant_factor, clamp, slope_factor);
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
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetDepthBias-depthBiasClamp-00790
        if clamp != 0.0 && !self.device().enabled_features().depth_bias_clamp {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`clamp` is not `0.0`",
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "depth_bias_clamp",
                )])]),
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_bias_unchecked(
        &mut self,
        constant_factor: f32,
        clamp: f32,
        slope_factor: f32,
    ) -> &mut Self {
        self.builder_state.depth_bias = Some(DepthBias {
            constant_factor,
            clamp,
            slope_factor,
        });
        self.add_command(
            "set_depth_bias",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_depth_bias(constant_factor, clamp, slope_factor);
            },
        );

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
    pub fn set_depth_bias_enable(&mut self, enable: bool) -> &mut Self {
        self.validate_set_depth_bias_enable(enable).unwrap();

        unsafe {
            self.set_depth_bias_enable_unchecked(enable);
        }

        self
    }

    fn validate_set_depth_bias_enable(&self, _enable: bool) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::DepthBiasEnable)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetDepthBiasEnable-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetDepthBiasEnable-None-04872
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state2)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`AutoCommandBufferBuilder::set_depth_bias_enable`",
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state2")]),
                ]),
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_bias_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.depth_bias_enable = Some(enable);
        self.add_command(
            "set_depth_bias_enable",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_depth_bias_enable(enable);
            },
        );

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
    ///   device extension is not enabled, panics if the start and end of `bounds` are not between
    ///   0.0 and 1.0 inclusive.
    pub fn set_depth_bounds(&mut self, bounds: RangeInclusive<f32>) -> &mut Self {
        self.validate_set_depth_bounds(bounds.clone()).unwrap();

        unsafe {
            self.set_depth_bounds_unchecked(bounds);
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
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
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
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "ext_depth_range_unrestricted",
                )])]),
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_bounds_unchecked(&mut self, bounds: RangeInclusive<f32>) -> &mut Self {
        self.builder_state.depth_bounds = Some(bounds.clone());
        self.add_command(
            "set_depth_bounds",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_depth_bounds(bounds.clone());
            },
        );

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
    pub fn set_depth_bounds_test_enable(&mut self, enable: bool) -> &mut Self {
        self.validate_set_depth_bounds_test_enable(enable).unwrap();

        unsafe {
            self.set_depth_bounds_test_enable_unchecked(enable);
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
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetDepthBoundsTestEnable-None-03349
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`AutoCommandBufferBuilder::set_depth_bounds_test_enable`",
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_bounds_test_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.depth_bounds_test_enable = Some(enable);
        self.add_command(
            "set_depth_bounds_test_enable",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_depth_bounds_test_enable(enable);
            },
        );

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
    pub fn set_depth_compare_op(&mut self, compare_op: CompareOp) -> &mut Self {
        self.validate_set_depth_compare_op(compare_op).unwrap();

        unsafe {
            self.set_depth_compare_op_unchecked(compare_op);
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
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetDepthCompareOp-None-03353
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`AutoCommandBufferBuilder::set_depth_compare_op`",
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_compare_op_unchecked(&mut self, compare_op: CompareOp) -> &mut Self {
        self.builder_state.depth_compare_op = Some(compare_op);
        self.add_command(
            "set_depth_compare_op",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_depth_compare_op(compare_op);
            },
        );

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
    pub fn set_depth_test_enable(&mut self, enable: bool) -> &mut Self {
        self.validate_set_depth_test_enable(enable).unwrap();

        unsafe {
            self.set_depth_test_enable_unchecked(enable);
        }

        self
    }

    fn validate_set_depth_test_enable(&self, _enable: bool) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::DepthTestEnable)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetDepthTestEnable-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetDepthTestEnable-None-03352
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`AutoCommandBufferBuilder::set_depth_test_enable`",
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_test_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.depth_test_enable = Some(enable);
        self.add_command(
            "set_depth_test_enable",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_depth_test_enable(enable);
            },
        );

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
    pub fn set_depth_write_enable(&mut self, enable: bool) -> &mut Self {
        self.validate_set_depth_write_enable(enable).unwrap();

        unsafe {
            self.set_depth_write_enable_unchecked(enable);
        }

        self
    }

    fn validate_set_depth_write_enable(&self, _enable: bool) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::DepthWriteEnable)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetDepthWriteEnable-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetDepthWriteEnable-None-03354
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`AutoCommandBufferBuilder::set_depth_write_enable`",
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_depth_write_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.depth_write_enable = Some(enable);
        self.add_command(
            "set_depth_write_enable",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_depth_write_enable(enable);
            },
        );

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
    pub fn set_discard_rectangle(
        &mut self,
        first_rectangle: u32,
        rectangles: SmallVec<[Scissor; 2]>,
    ) -> &mut Self {
        self.validate_set_discard_rectangle(first_rectangle, &rectangles)
            .unwrap();

        unsafe {
            self.set_discard_rectangle_unchecked(first_rectangle, rectangles);
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
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        if self.device().enabled_extensions().ext_discard_rectangles {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`AutoCommandBufferBuilder::set_discard_rectangle`",
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "ext_discard_rectangles",
                )])]),
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

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_discard_rectangle_unchecked(
        &mut self,
        first_rectangle: u32,
        rectangles: SmallVec<[Scissor; 2]>,
    ) -> &mut Self {
        for (num, rectangle) in rectangles.iter().enumerate() {
            let num = num as u32 + first_rectangle;
            self.builder_state.discard_rectangle.insert(num, *rectangle);
        }

        self.add_command(
            "set_discard_rectangle",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_discard_rectangle(first_rectangle, &rectangles);
            },
        );

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
    pub fn set_front_face(&mut self, face: FrontFace) -> &mut Self {
        self.validate_set_front_face(face).unwrap();

        unsafe {
            self.set_front_face_unchecked(face);
        }

        self
    }

    fn validate_set_front_face(&self, face: FrontFace) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::FrontFace)?;

        // VUID-vkCmdSetFrontFace-frontFace-parameter
        face.validate_device(self.device())?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetFrontFace-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetFrontFace-None-03383
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`AutoCommandBufferBuilder::set_front_face`",
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_front_face_unchecked(&mut self, face: FrontFace) -> &mut Self {
        self.builder_state.front_face = Some(face);
        self.add_command(
            "set_front_face",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_front_face(face);
            },
        );

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
    pub fn set_line_stipple(&mut self, factor: u32, pattern: u16) -> &mut Self {
        self.validate_set_line_stipple(factor, pattern).unwrap();

        unsafe {
            self.set_line_stipple_unchecked(factor, pattern);
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
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        if !self.device().enabled_extensions().ext_line_rasterization {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`AutoCommandBufferBuilder::set_line_stipple`",
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "ext_line_rasterization",
                )])]),
            });
        }

        // VUID-vkCmdSetLineStippleEXT-lineStippleFactor-02776
        if !(1..=256).contains(&factor) {
            return Err(SetDynamicStateError::FactorOutOfRange);
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_line_stipple_unchecked(&mut self, factor: u32, pattern: u16) -> &mut Self {
        self.builder_state.line_stipple = Some(LineStipple { factor, pattern });
        self.add_command(
            "set_line_stipple",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_line_stipple(factor, pattern);
            },
        );

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
        self.validate_set_line_width(line_width).unwrap();

        unsafe {
            self.set_line_width_unchecked(line_width);
        }

        self
    }

    fn validate_set_line_width(&self, line_width: f32) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::LineWidth)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetLineWidth-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetLineWidth-lineWidth-00788
        if !self.device().enabled_features().wide_lines && line_width != 1.0 {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`line_width` is not `1.0`",
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "wide_lines",
                )])]),
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_line_width_unchecked(&mut self, line_width: f32) -> &mut Self {
        self.builder_state.line_width = Some(line_width);
        self.add_command(
            "set_line_width",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_line_width(line_width);
            },
        );

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
    pub fn set_logic_op(&mut self, logic_op: LogicOp) -> &mut Self {
        self.validate_set_logic_op(logic_op).unwrap();

        unsafe {
            self.set_logic_op_unchecked(logic_op);
        }

        self
    }

    fn validate_set_logic_op(&self, logic_op: LogicOp) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::LogicOp)?;

        // VUID-vkCmdSetLogicOpEXT-logicOp-parameter
        logic_op.validate_device(self.device())?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetLogicOpEXT-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetLogicOpEXT-None-04867
        if !self
            .device()
            .enabled_features()
            .extended_dynamic_state2_logic_op
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`AutoCommandBufferBuilder::set_logic_op`",
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "extended_dynamic_state2_logic_op",
                )])]),
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_logic_op_unchecked(&mut self, logic_op: LogicOp) -> &mut Self {
        self.builder_state.logic_op = Some(logic_op);
        self.add_command(
            "set_logic_op",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_logic_op(logic_op);
            },
        );

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
    pub fn set_patch_control_points(&mut self, num: u32) -> &mut Self {
        self.validate_set_patch_control_points(num).unwrap();

        unsafe {
            self.set_patch_control_points_unchecked(num);
        }

        self
    }

    fn validate_set_patch_control_points(&self, num: u32) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::PatchControlPoints)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetPatchControlPointsEXT-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetPatchControlPointsEXT-None-04873
        if !self
            .device()
            .enabled_features()
            .extended_dynamic_state2_patch_control_points
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`AutoCommandBufferBuilder::set_patch_control_points`",
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "extended_dynamic_state2_patch_control_points",
                )])]),
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

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_patch_control_points_unchecked(&mut self, num: u32) -> &mut Self {
        self.builder_state.patch_control_points = Some(num);
        self.add_command(
            "set_patch_control_points",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_patch_control_points(num);
            },
        );

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
    pub fn set_primitive_restart_enable(&mut self, enable: bool) -> &mut Self {
        self.validate_set_primitive_restart_enable(enable).unwrap();

        unsafe {
            self.set_primitive_restart_enable_unchecked(enable);
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
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetPrimitiveRestartEnable-None-04866
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state2)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`AutoCommandBufferBuilder::set_primitive_restart_enable`",
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state2")]),
                ]),
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_primitive_restart_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.primitive_restart_enable = Some(enable);
        self.add_command(
            "set_primitive_restart_enable",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_primitive_restart_enable(enable);
            },
        );

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
    pub fn set_primitive_topology(&mut self, topology: PrimitiveTopology) -> &mut Self {
        self.validate_set_primitive_topology(topology).unwrap();

        unsafe {
            self.set_primitive_topology_unchecked(topology);
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
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetPrimitiveTopology-None-03347
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`AutoCommandBufferBuilder::set_primitive_topology`",
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
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
                        required_for: "this device is a portability subset device, and `topology` \
                            is `PrimitiveTopology::TriangleFan`",
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "triangle_fans",
                        )])]),
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
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "geometry_shader",
                        )])]),
                    });
                }
            }
            PrimitiveTopology::PatchList => {
                if !self.device().enabled_features().tessellation_shader {
                    return Err(SetDynamicStateError::RequirementNotMet {
                        required_for: "`topology` is `PrimitiveTopology::PatchList`",
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                            "tessellation_shader",
                        )])]),
                    });
                }
            }
            _ => (),
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_primitive_topology_unchecked(
        &mut self,
        topology: PrimitiveTopology,
    ) -> &mut Self {
        self.builder_state.primitive_topology = Some(topology);
        self.add_command(
            "set_primitive_topology",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_primitive_topology(topology);
            },
        );

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
    pub fn set_rasterizer_discard_enable(&mut self, enable: bool) -> &mut Self {
        self.validate_set_rasterizer_discard_enable(enable).unwrap();

        unsafe {
            self.set_rasterizer_discard_enable_unchecked(enable);
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
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetRasterizerDiscardEnable-None-04871
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state2)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`AutoCommandBufferBuilder::set_rasterizer_discard_enable`",
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state2")]),
                ]),
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_rasterizer_discard_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.rasterizer_discard_enable = Some(enable);
        self.add_command(
            "set_rasterizer_discard_enable",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_rasterizer_discard_enable(enable);
            },
        );

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
    pub fn set_scissor(
        &mut self,
        first_scissor: u32,
        scissors: SmallVec<[Scissor; 2]>,
    ) -> &mut Self {
        self.validate_set_scissor(first_scissor, &scissors).unwrap();

        unsafe {
            self.set_scissor_unchecked(first_scissor, scissors);
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
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
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
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "multi_viewport",
                    )])]),
                });
            }

            // VUID-vkCmdSetScissor-scissorCount-00594
            if scissors.len() > 1 {
                return Err(SetDynamicStateError::RequirementNotMet {
                    required_for: "`scissors.len()` is greater than `1`",
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "multi_viewport",
                    )])]),
                });
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_scissor_unchecked(
        &mut self,
        first_scissor: u32,
        scissors: SmallVec<[Scissor; 2]>,
    ) -> &mut Self {
        let scissors: SmallVec<[Scissor; 2]> = scissors.into_iter().collect();

        for (num, scissor) in scissors.iter().enumerate() {
            let num = num as u32 + first_scissor;
            self.builder_state.scissor.insert(num, *scissor);
        }

        self.add_command(
            "set_scissor",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_scissor(first_scissor, &scissors);
            },
        );

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
    pub fn set_scissor_with_count(&mut self, scissors: SmallVec<[Scissor; 2]>) -> &mut Self {
        self.validate_set_scissor_with_count(&scissors).unwrap();

        unsafe {
            self.set_scissor_with_count_unchecked(scissors);
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
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetScissorWithCount-None-03396
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`AutoCommandBufferBuilder::set_scissor_with_count`",
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
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
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "multi_viewport",
                )])]),
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_scissor_with_count_unchecked(
        &mut self,
        scissors: SmallVec<[Scissor; 2]>,
    ) -> &mut Self {
        self.builder_state.scissor_with_count = Some(scissors.clone());
        self.add_command(
            "set_scissor_with_count",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_scissor_with_count(&scissors);
            },
        );

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
        self.validate_set_stencil_compare_mask(faces, compare_mask)
            .unwrap();

        unsafe {
            self.set_stencil_compare_mask_unchecked(faces, compare_mask);
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
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_stencil_compare_mask_unchecked(
        &mut self,
        faces: StencilFaces,
        compare_mask: u32,
    ) -> &mut Self {
        let faces_vk = ash::vk::StencilFaceFlags::from(faces);

        if faces_vk.intersects(ash::vk::StencilFaceFlags::FRONT) {
            self.builder_state.stencil_compare_mask.front = Some(compare_mask);
        }

        if faces_vk.intersects(ash::vk::StencilFaceFlags::BACK) {
            self.builder_state.stencil_compare_mask.back = Some(compare_mask);
        }

        self.add_command(
            "set_stencil_compare_mask",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_stencil_compare_mask(faces, compare_mask);
            },
        );

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
            self.set_stencil_op_unchecked(faces, fail_op, pass_op, depth_fail_op, compare_op);
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
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetStencilOp-None-03351
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`AutoCommandBufferBuilder::set_stencil_op`",
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_stencil_op_unchecked(
        &mut self,
        faces: StencilFaces,
        fail_op: StencilOp,
        pass_op: StencilOp,
        depth_fail_op: StencilOp,
        compare_op: CompareOp,
    ) -> &mut Self {
        let faces_vk = ash::vk::StencilFaceFlags::from(faces);

        if faces_vk.intersects(ash::vk::StencilFaceFlags::FRONT) {
            self.builder_state.stencil_op.front = Some(StencilOps {
                fail_op,
                pass_op,
                depth_fail_op,
                compare_op,
            });
        }

        if faces_vk.intersects(ash::vk::StencilFaceFlags::BACK) {
            self.builder_state.stencil_op.back = Some(StencilOps {
                fail_op,
                pass_op,
                depth_fail_op,
                compare_op,
            });
        }

        self.add_command(
            "set_stencil_op",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_stencil_op(faces, fail_op, pass_op, depth_fail_op, compare_op);
            },
        );

        self
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
            self.set_stencil_reference_unchecked(faces, reference);
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
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_stencil_reference_unchecked(
        &mut self,
        faces: StencilFaces,
        reference: u32,
    ) -> &mut Self {
        let faces_vk = ash::vk::StencilFaceFlags::from(faces);

        if faces_vk.intersects(ash::vk::StencilFaceFlags::FRONT) {
            self.builder_state.stencil_reference.front = Some(reference);
        }

        if faces_vk.intersects(ash::vk::StencilFaceFlags::BACK) {
            self.builder_state.stencil_reference.back = Some(reference);
        }

        self.add_command(
            "set_stencil_reference",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_stencil_reference(faces, reference);
            },
        );

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
    pub fn set_stencil_test_enable(&mut self, enable: bool) -> &mut Self {
        self.validate_set_stencil_test_enable(enable).unwrap();

        unsafe {
            self.set_stencil_test_enable_unchecked(enable);
        }

        self
    }

    fn validate_set_stencil_test_enable(&self, _enable: bool) -> Result<(), SetDynamicStateError> {
        self.validate_pipeline_fixed_state(DynamicState::StencilTestEnable)?;

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdSetStencilTestEnable-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetStencilTestEnable-None-03350
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`AutoCommandBufferBuilder::set_stencil_test_enable`",
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_stencil_test_enable_unchecked(&mut self, enable: bool) -> &mut Self {
        self.builder_state.stencil_test_enable = Some(enable);
        self.add_command(
            "set_stencil_test_enable",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_stencil_test_enable(enable);
            },
        );

        self
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
            self.set_stencil_write_mask_unchecked(faces, write_mask);
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
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_stencil_write_mask_unchecked(
        &mut self,
        faces: StencilFaces,
        write_mask: u32,
    ) -> &mut Self {
        let faces_vk = ash::vk::StencilFaceFlags::from(faces);

        if faces_vk.intersects(ash::vk::StencilFaceFlags::FRONT) {
            self.builder_state.stencil_write_mask.front = Some(write_mask);
        }

        if faces_vk.intersects(ash::vk::StencilFaceFlags::BACK) {
            self.builder_state.stencil_write_mask.back = Some(write_mask);
        }

        self.add_command(
            "set_stencil_write_mask",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_stencil_write_mask(faces, write_mask);
            },
        );

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
    pub fn set_viewport(
        &mut self,
        first_viewport: u32,
        viewports: SmallVec<[Viewport; 2]>,
    ) -> &mut Self {
        self.validate_set_viewport(first_viewport, &viewports)
            .unwrap();

        unsafe {
            self.set_viewport_unchecked(first_viewport, viewports);
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
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
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
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "multi_viewport",
                    )])]),
                });
            }

            // VUID-vkCmdSetViewport-viewportCount-01225
            if viewports.len() > 1 {
                return Err(SetDynamicStateError::RequirementNotMet {
                    required_for: "`viewports.len()` is greater than `1`",
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "multi_viewport",
                    )])]),
                });
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_viewport_unchecked(
        &mut self,
        first_viewport: u32,
        viewports: SmallVec<[Viewport; 2]>,
    ) -> &mut Self {
        for (num, viewport) in viewports.iter().enumerate() {
            let num = num as u32 + first_viewport;
            self.builder_state.viewport.insert(num, viewport.clone());
        }

        self.add_command(
            "set_viewport",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_viewport(first_viewport, &viewports);
            },
        );

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
    pub fn set_viewport_with_count(&mut self, viewports: SmallVec<[Viewport; 2]>) -> &mut Self {
        self.validate_set_viewport_with_count(&viewports).unwrap();

        unsafe {
            self.set_viewport_with_count_unchecked(viewports);
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
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS)
        {
            return Err(SetDynamicStateError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdSetViewportWithCount-None-03393
        if !(self.device().api_version() >= Version::V1_3
            || self.device().enabled_features().extended_dynamic_state)
        {
            return Err(SetDynamicStateError::RequirementNotMet {
                required_for: "`AutoCommandBufferBuilder::set_viewport_with_count`",
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::Feature("extended_dynamic_state")]),
                ]),
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
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "multi_viewport",
                )])]),
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_viewport_with_count_unchecked(
        &mut self,
        viewports: SmallVec<[Viewport; 2]>,
    ) -> &mut Self {
        self.builder_state.viewport_with_count = Some(viewports.clone());
        self.add_command(
            "set_viewport",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.set_viewport_with_count(&viewports);
            },
        );

        self
    }
}

impl<A> UnsafeCommandBufferBuilder<A>
where
    A: CommandBufferAllocator,
{
    /// Calls `vkCmdSetBlendConstants` on the builder.
    #[inline]
    pub unsafe fn set_blend_constants(&mut self, constants: [f32; 4]) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_set_blend_constants)(self.handle(), &constants);

        self
    }

    /// Calls `vkCmdSetColorWriteEnableEXT` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    pub unsafe fn set_color_write_enable(&mut self, enables: &[bool]) -> &mut Self {
        let enables = enables
            .iter()
            .copied()
            .map(|v| v as ash::vk::Bool32)
            .collect::<SmallVec<[_; 4]>>();
        if enables.is_empty() {
            return self;
        }

        let fns = self.device().fns();
        (fns.ext_color_write_enable.cmd_set_color_write_enable_ext)(
            self.handle(),
            enables.len() as u32,
            enables.as_ptr(),
        );

        self
    }

    /// Calls `vkCmdSetCullModeEXT` on the builder.
    #[inline]
    pub unsafe fn set_cull_mode(&mut self, cull_mode: CullMode) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_cull_mode)(self.handle(), cull_mode.into());
        } else {
            (fns.ext_extended_dynamic_state.cmd_set_cull_mode_ext)(self.handle(), cull_mode.into());
        }

        self
    }

    /// Calls `vkCmdSetDepthBias` on the builder.
    #[inline]
    pub unsafe fn set_depth_bias(
        &mut self,
        constant_factor: f32,
        clamp: f32,
        slope_factor: f32,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_set_depth_bias)(self.handle(), constant_factor, clamp, slope_factor);

        self
    }

    /// Calls `vkCmdSetDepthBiasEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_depth_bias_enable(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_depth_bias_enable)(self.handle(), enable.into());
        } else {
            (fns.ext_extended_dynamic_state2
                .cmd_set_depth_bias_enable_ext)(self.handle(), enable.into());
        }

        self
    }

    /// Calls `vkCmdSetDepthBounds` on the builder.
    #[inline]
    pub unsafe fn set_depth_bounds(&mut self, bounds: RangeInclusive<f32>) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_set_depth_bounds)(self.handle(), *bounds.start(), *bounds.end());

        self
    }

    /// Calls `vkCmdSetDepthBoundsTestEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_depth_bounds_test_enable(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_depth_bounds_test_enable)(self.handle(), enable.into());
        } else {
            (fns.ext_extended_dynamic_state
                .cmd_set_depth_bounds_test_enable_ext)(self.handle(), enable.into());
        }

        self
    }

    /// Calls `vkCmdSetDepthCompareOpEXT` on the builder.
    #[inline]
    pub unsafe fn set_depth_compare_op(&mut self, compare_op: CompareOp) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_depth_compare_op)(self.handle(), compare_op.into());
        } else {
            (fns.ext_extended_dynamic_state.cmd_set_depth_compare_op_ext)(
                self.handle(),
                compare_op.into(),
            );
        }

        self
    }

    /// Calls `vkCmdSetDepthTestEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_depth_test_enable(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_depth_test_enable)(self.handle(), enable.into());
        } else {
            (fns.ext_extended_dynamic_state.cmd_set_depth_test_enable_ext)(
                self.handle(),
                enable.into(),
            );
        }

        self
    }

    /// Calls `vkCmdSetDepthWriteEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_depth_write_enable(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_depth_write_enable)(self.handle(), enable.into());
        } else {
            (fns.ext_extended_dynamic_state
                .cmd_set_depth_write_enable_ext)(self.handle(), enable.into());
        }

        self
    }

    /// Calls `vkCmdSetDiscardRectangleEXT` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    pub unsafe fn set_discard_rectangle(
        &mut self,
        first_rectangle: u32,
        rectangles: &[Scissor],
    ) -> &mut Self {
        let rectangles = rectangles
            .iter()
            .map(|v| v.into())
            .collect::<SmallVec<[_; 2]>>();
        if rectangles.is_empty() {
            return self;
        }

        let fns = self.device().fns();
        (fns.ext_discard_rectangles.cmd_set_discard_rectangle_ext)(
            self.handle(),
            first_rectangle,
            rectangles.len() as u32,
            rectangles.as_ptr(),
        );

        self
    }

    /// Calls `vkCmdSetFrontFaceEXT` on the builder.
    #[inline]
    pub unsafe fn set_front_face(&mut self, face: FrontFace) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_front_face)(self.handle(), face.into());
        } else {
            (fns.ext_extended_dynamic_state.cmd_set_front_face_ext)(self.handle(), face.into());
        }

        self
    }

    /// Calls `vkCmdSetLineStippleEXT` on the builder.
    #[inline]
    pub unsafe fn set_line_stipple(&mut self, factor: u32, pattern: u16) -> &mut Self {
        let fns = self.device().fns();
        (fns.ext_line_rasterization.cmd_set_line_stipple_ext)(self.handle(), factor, pattern);

        self
    }

    /// Calls `vkCmdSetLineWidth` on the builder.
    #[inline]
    pub unsafe fn set_line_width(&mut self, line_width: f32) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_set_line_width)(self.handle(), line_width);

        self
    }

    /// Calls `vkCmdSetLogicOpEXT` on the builder.
    #[inline]
    pub unsafe fn set_logic_op(&mut self, logic_op: LogicOp) -> &mut Self {
        let fns = self.device().fns();
        (fns.ext_extended_dynamic_state2.cmd_set_logic_op_ext)(self.handle(), logic_op.into());

        self
    }

    /// Calls `vkCmdSetPatchControlPointsEXT` on the builder.
    #[inline]
    pub unsafe fn set_patch_control_points(&mut self, num: u32) -> &mut Self {
        let fns = self.device().fns();
        (fns.ext_extended_dynamic_state2
            .cmd_set_patch_control_points_ext)(self.handle(), num);

        self
    }

    /// Calls `vkCmdSetPrimitiveRestartEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_primitive_restart_enable(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_primitive_restart_enable)(self.handle(), enable.into());
        } else {
            (fns.ext_extended_dynamic_state2
                .cmd_set_primitive_restart_enable_ext)(self.handle(), enable.into());
        }

        self
    }

    /// Calls `vkCmdSetPrimitiveTopologyEXT` on the builder.
    #[inline]
    pub unsafe fn set_primitive_topology(&mut self, topology: PrimitiveTopology) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_primitive_topology)(self.handle(), topology.into());
        } else {
            (fns.ext_extended_dynamic_state
                .cmd_set_primitive_topology_ext)(self.handle(), topology.into());
        }

        self
    }

    /// Calls `vkCmdSetRasterizerDiscardEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_rasterizer_discard_enable(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_rasterizer_discard_enable)(self.handle(), enable.into());
        } else {
            (fns.ext_extended_dynamic_state2
                .cmd_set_rasterizer_discard_enable_ext)(self.handle(), enable.into());
        }

        self
    }

    /// Calls `vkCmdSetStencilCompareMask` on the builder.
    #[inline]
    pub unsafe fn set_stencil_compare_mask(
        &mut self,
        face_mask: StencilFaces,
        compare_mask: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_set_stencil_compare_mask)(self.handle(), face_mask.into(), compare_mask);

        self
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
    ) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_stencil_op)(
                self.handle(),
                face_mask.into(),
                fail_op.into(),
                pass_op.into(),
                depth_fail_op.into(),
                compare_op.into(),
            );
        } else {
            (fns.ext_extended_dynamic_state.cmd_set_stencil_op_ext)(
                self.handle(),
                face_mask.into(),
                fail_op.into(),
                pass_op.into(),
                depth_fail_op.into(),
                compare_op.into(),
            );
        }

        self
    }

    /// Calls `vkCmdSetStencilReference` on the builder.
    #[inline]
    pub unsafe fn set_stencil_reference(
        &mut self,
        face_mask: StencilFaces,
        reference: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_set_stencil_reference)(self.handle(), face_mask.into(), reference);

        self
    }

    /// Calls `vkCmdSetStencilTestEnableEXT` on the builder.
    #[inline]
    pub unsafe fn set_stencil_test_enable(&mut self, enable: bool) -> &mut Self {
        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_stencil_test_enable)(self.handle(), enable.into());
        } else {
            (fns.ext_extended_dynamic_state
                .cmd_set_stencil_test_enable_ext)(self.handle(), enable.into());
        }

        self
    }

    /// Calls `vkCmdSetStencilWriteMask` on the builder.
    #[inline]
    pub unsafe fn set_stencil_write_mask(
        &mut self,
        face_mask: StencilFaces,
        write_mask: u32,
    ) -> &mut Self {
        let fns = self.device().fns();
        (fns.v1_0.cmd_set_stencil_write_mask)(self.handle(), face_mask.into(), write_mask);

        self
    }

    /// Calls `vkCmdSetScissor` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    pub unsafe fn set_scissor(&mut self, first_scissor: u32, scissors: &[Scissor]) -> &mut Self {
        let scissors = scissors
            .iter()
            .map(ash::vk::Rect2D::from)
            .collect::<SmallVec<[_; 2]>>();
        if scissors.is_empty() {
            return self;
        }

        let fns = self.device().fns();
        (fns.v1_0.cmd_set_scissor)(
            self.handle(),
            first_scissor,
            scissors.len() as u32,
            scissors.as_ptr(),
        );

        self
    }

    /// Calls `vkCmdSetScissorWithCountEXT` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    pub unsafe fn set_scissor_with_count(&mut self, scissors: &[Scissor]) -> &mut Self {
        let scissors = scissors
            .iter()
            .map(ash::vk::Rect2D::from)
            .collect::<SmallVec<[_; 2]>>();
        if scissors.is_empty() {
            return self;
        }

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_scissor_with_count)(
                self.handle(),
                scissors.len() as u32,
                scissors.as_ptr(),
            );
        } else {
            (fns.ext_extended_dynamic_state
                .cmd_set_scissor_with_count_ext)(
                self.handle(),
                scissors.len() as u32,
                scissors.as_ptr(),
            );
        }

        self
    }

    /// Calls `vkCmdSetViewport` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    pub unsafe fn set_viewport(
        &mut self,
        first_viewport: u32,
        viewports: &[Viewport],
    ) -> &mut Self {
        let viewports = viewports
            .iter()
            .map(|v| v.into())
            .collect::<SmallVec<[_; 2]>>();
        if viewports.is_empty() {
            return self;
        }

        let fns = self.device().fns();
        (fns.v1_0.cmd_set_viewport)(
            self.handle(),
            first_viewport,
            viewports.len() as u32,
            viewports.as_ptr(),
        );

        self
    }

    /// Calls `vkCmdSetViewportWithCountEXT` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    pub unsafe fn set_viewport_with_count(&mut self, viewports: &[Viewport]) -> &mut Self {
        let viewports = viewports
            .iter()
            .map(|v| v.into())
            .collect::<SmallVec<[_; 2]>>();
        if viewports.is_empty() {
            return self;
        }

        let fns = self.device().fns();

        if self.device().api_version() >= Version::V1_3 {
            (fns.v1_3.cmd_set_viewport_with_count)(
                self.handle(),
                viewports.len() as u32,
                viewports.as_ptr(),
            );
        } else {
            (fns.ext_extended_dynamic_state
                .cmd_set_viewport_with_count_ext)(
                self.handle(),
                viewports.len() as u32,
                viewports.as_ptr(),
            );
        }

        self
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(in super::super) enum SetDynamicStateError {
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
