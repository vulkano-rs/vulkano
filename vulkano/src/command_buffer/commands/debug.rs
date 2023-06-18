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
    instance::debug::DebugUtilsLabel,
    Requires, RequiresAllOf, RequiresOneOf, VulkanObject,
};
use std::{
    error::Error,
    ffi::CString,
    fmt::{Display, Error as FmtError, Formatter},
};

/// # Commands for debugging.
///
/// These commands all require the [`ext_debug_utils`] extension to be enabled on the instance.
///
/// [`ext_debug_utils`]: crate::instance::InstanceExtensions::ext_debug_utils
impl<L, A> AutoCommandBufferBuilder<L, A>
where
    A: CommandBufferAllocator,
{
    /// Opens a command buffer debug label region.
    pub fn begin_debug_utils_label(
        &mut self,
        label_info: DebugUtilsLabel,
    ) -> Result<&mut Self, DebugUtilsError> {
        self.validate_begin_debug_utils_label(&label_info)?;

        unsafe {
            self.begin_debug_utils_label_unchecked(label_info);
        }

        Ok(self)
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
                required_for: "`AutoCommandBufferBuilder::begin_debug_utils_label`",
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_debug_utils",
                )])]),
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
        self.add_command(
            "begin_debug_utils_label",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.begin_debug_utils_label(&label_info);
            },
        );

        self
    }

    /// Closes a command buffer debug label region.
    ///
    /// # Safety
    ///
    /// - When submitting the command buffer, there must be an outstanding command buffer label
    ///   region begun with `begin_debug_utils_label` in the queue, either within this command
    ///   buffer or a previously submitted one.
    pub unsafe fn end_debug_utils_label(&mut self) -> Result<&mut Self, DebugUtilsError> {
        self.validate_end_debug_utils_label()?;

        self.end_debug_utils_label_unchecked();

        Ok(self)
    }

    fn validate_end_debug_utils_label(&self) -> Result<(), DebugUtilsError> {
        if !self
            .device()
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(DebugUtilsError::RequirementNotMet {
                required_for: "`AutoCommandBufferBuilder::end_debug_utils_label`",
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_debug_utils",
                )])]),
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
        self.add_command(
            "end_debug_utils_label",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.end_debug_utils_label();
            },
        );

        self
    }

    /// Inserts a command buffer debug label.
    pub fn insert_debug_utils_label(
        &mut self,
        label_info: DebugUtilsLabel,
    ) -> Result<&mut Self, DebugUtilsError> {
        self.validate_insert_debug_utils_label(&label_info)?;

        unsafe {
            self.insert_debug_utils_label_unchecked(label_info);
        }

        Ok(self)
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
                required_for: "`AutoCommandBufferBuilder::insert_debug_utils_label`",
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_debug_utils",
                )])]),
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
        self.add_command(
            "insert_debug_utils_label",
            Default::default(),
            move |out: &mut UnsafeCommandBufferBuilder<A>| {
                out.insert_debug_utils_label(&label_info);
            },
        );

        self
    }
}

impl<A> UnsafeCommandBufferBuilder<A>
where
    A: CommandBufferAllocator,
{
    /// Calls `vkCmdBeginDebugUtilsLabelEXT` on the builder.
    ///
    /// # Safety
    /// The command pool that this command buffer was allocated from must support graphics or
    /// compute operations
    #[inline]
    pub unsafe fn begin_debug_utils_label(&mut self, label_info: &DebugUtilsLabel) -> &mut Self {
        let &DebugUtilsLabel {
            ref label_name,
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

    /// Calls `vkCmdEndDebugUtilsLabelEXT` on the builder.
    ///
    /// # Safety
    /// There must be an outstanding `vkCmdBeginDebugUtilsLabelEXT` command prior to the
    /// `vkQueueEndDebugUtilsLabelEXT` on the queue tha `CommandBuffer` is submitted to.
    #[inline]
    pub unsafe fn end_debug_utils_label(&mut self) -> &mut Self {
        let fns = self.device().instance().fns();
        (fns.ext_debug_utils.cmd_end_debug_utils_label_ext)(self.handle());

        self
    }

    /// Calls `vkCmdInsertDebugUtilsLabelEXT` on the builder.
    ///
    /// # Safety
    /// The command pool that this command buffer was allocated from must support graphics or
    /// compute operations
    #[inline]
    pub unsafe fn insert_debug_utils_label(&mut self, label_info: &DebugUtilsLabel) -> &mut Self {
        let &DebugUtilsLabel {
            ref label_name,
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

/// Error that can happen when recording a debug utils command.
#[derive(Clone, Debug)]
pub enum DebugUtilsError {
    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// The queue family doesn't allow this operation.
    NotSupportedByQueueFamily,
}

impl Error for DebugUtilsError {}

impl Display for DebugUtilsError {
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
            Self::NotSupportedByQueueFamily => {
                write!(f, "the queue family doesn't allow this operation")
            }
        }
    }
}
