// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{CommandBufferBuilder, DebugUtilsError};
use crate::{
    command_buffer::allocator::CommandBufferAllocator,
    device::{DeviceOwned, QueueFlags},
    instance::debug::DebugUtilsLabel,
    RequiresOneOf, VulkanObject,
};
use std::ffi::CString;

impl<L, A> CommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    /// Opens a command buffer debug label region.
    #[inline]
    pub fn begin_debug_utils_label(
        &mut self,
        label_info: DebugUtilsLabel,
    ) -> Result<&mut Self, DebugUtilsError> {
        self.validate_begin_debug_utils_label(&label_info)?;

        unsafe { Ok(self.begin_debug_utils_label_unchecked(label_info)) }
    }

    fn validate_begin_debug_utils_label(
        &self,
        _label_info: &DebugUtilsLabel,
    ) -> Result<(), DebugUtilsError> {
        if !self
            .device()
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(DebugUtilsError::RequirementNotMet {
                required_for: "`CommandBufferBuilder::begin_debug_utils_label`",
                requires_one_of: RequiresOneOf {
                    instance_extensions: &["ext_debug_utils"],
                    ..Default::default()
                },
            });
        }

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdBeginDebugUtilsLabelEXT-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(DebugUtilsError::NotSupportedByQueueFamily);
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn begin_debug_utils_label_unchecked(
        &mut self,
        label_info: DebugUtilsLabel,
    ) -> &mut Self {
        let DebugUtilsLabel {
            label_name,
            color,
            _ne: _,
        } = label_info;

        let label_name_vk = CString::new(label_name.as_str()).unwrap();
        let label_info = ash::vk::DebugUtilsLabelEXT {
            p_label_name: label_name_vk.as_ptr(),
            color,
            ..Default::default()
        };

        let fns = self.device().instance().fns();
        (fns.ext_debug_utils.cmd_begin_debug_utils_label_ext)(self.handle(), &label_info);

        self
    }

    /// Closes a command buffer debug label region.
    ///
    /// # Safety
    ///
    /// - When submitting the command buffer, there must be an outstanding command buffer label
    ///   region begun with `begin_debug_utils_label` in the queue, either within this command
    ///   buffer or a previously submitted one.
    #[inline]
    pub unsafe fn end_debug_utils_label(&mut self) -> Result<&mut Self, DebugUtilsError> {
        self.validate_end_debug_utils_label()?;

        Ok(self.end_debug_utils_label_unchecked())
    }

    fn validate_end_debug_utils_label(&self) -> Result<(), DebugUtilsError> {
        if !self
            .device()
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(DebugUtilsError::RequirementNotMet {
                required_for: "`CommandBufferBuilder::end_debug_utils_label`",
                requires_one_of: RequiresOneOf {
                    instance_extensions: &["ext_debug_utils"],
                    ..Default::default()
                },
            });
        }

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdEndDebugUtilsLabelEXT-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(DebugUtilsError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdEndDebugUtilsLabelEXT-commandBuffer-01912
        // TODO: not checked, so unsafe for now

        // VUID-vkCmdEndDebugUtilsLabelEXT-commandBuffer-01913
        // TODO: not checked, so unsafe for now

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn end_debug_utils_label_unchecked(&mut self) -> &mut Self {
        let fns = self.device().instance().fns();
        (fns.ext_debug_utils.cmd_end_debug_utils_label_ext)(self.handle());

        self
    }

    /// Inserts a command buffer debug label.
    #[inline]
    pub fn insert_debug_utils_label(
        &mut self,
        label_info: DebugUtilsLabel,
    ) -> Result<&mut Self, DebugUtilsError> {
        self.validate_insert_debug_utils_label(&label_info)?;

        unsafe { Ok(self.insert_debug_utils_label_unchecked(label_info)) }
    }

    fn validate_insert_debug_utils_label(
        &self,
        _label_info: &DebugUtilsLabel,
    ) -> Result<(), DebugUtilsError> {
        if !self
            .device()
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(DebugUtilsError::RequirementNotMet {
                required_for: "`CommandBufferBuilder::insert_debug_utils_label`",
                requires_one_of: RequiresOneOf {
                    instance_extensions: &["ext_debug_utils"],
                    ..Default::default()
                },
            });
        }

        let queue_family_properties = self.queue_family_properties();

        // VUID-vkCmdInsertDebugUtilsLabelEXT-commandBuffer-cmdpool
        if !queue_family_properties
            .queue_flags
            .intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE)
        {
            return Err(DebugUtilsError::NotSupportedByQueueFamily);
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn insert_debug_utils_label_unchecked(
        &mut self,
        label_info: DebugUtilsLabel,
    ) -> &mut Self {
        let DebugUtilsLabel {
            label_name,
            color,
            _ne: _,
        } = label_info;

        let label_name_vk = CString::new(label_name.as_str()).unwrap();
        let label_info = ash::vk::DebugUtilsLabelEXT {
            p_label_name: label_name_vk.as_ptr(),
            color,
            ..Default::default()
        };

        let fns = self.device().instance().fns();
        (fns.ext_debug_utils.cmd_insert_debug_utils_label_ext)(self.handle(), &label_info);

        self
    }
}
