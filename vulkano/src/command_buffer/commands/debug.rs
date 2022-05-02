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
    instance::debug::DebugUtilsLabel,
};
use std::{error, ffi::CString, fmt};

/// # Commands for debugging.
///
/// These commands all require the
/// [`ext_debug_utils`](crate::instance::InstanceExtensions::ext_debug_utils) to be enabled on the
/// instance.
impl<L, P> AutoCommandBufferBuilder<L, P> {
    /// Opens a command buffer debug label region.
    #[inline]
    pub fn begin_debug_utils_label(
        &mut self,
        mut label_info: DebugUtilsLabel,
    ) -> Result<&mut Self, DebugUtilsError> {
        self.validate_begin_debug_utils_label(&mut label_info)?;

        unsafe {
            self.inner.begin_debug_utils_label(label_info);
        }

        Ok(self)
    }

    fn validate_begin_debug_utils_label(
        &self,
        label_info: &mut DebugUtilsLabel,
    ) -> Result<(), DebugUtilsError> {
        if !self
            .device()
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(DebugUtilsError::ExtensionNotEnabled {
                extension: "ext_debug_utils",
                reason: "tried to record a debug utils command",
            });
        }

        // VUID-vkCmdBeginDebugUtilsLabelEXT-commandBuffer-cmdpool
        if !(self.queue_family().supports_graphics() || self.queue_family().supports_compute()) {
            return Err(DebugUtilsError::NotSupportedByQueueFamily);
        }

        Ok(())
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

        self.inner.end_debug_utils_label();

        Ok(self)
    }

    fn validate_end_debug_utils_label(&self) -> Result<(), DebugUtilsError> {
        if !self
            .device()
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(DebugUtilsError::ExtensionNotEnabled {
                extension: "ext_debug_utils",
                reason: "tried to record a debug utils command",
            });
        }

        // VUID-vkCmdEndDebugUtilsLabelEXT-commandBuffer-cmdpool
        if !(self.queue_family().supports_graphics() || self.queue_family().supports_compute()) {
            return Err(DebugUtilsError::NotSupportedByQueueFamily);
        }

        // VUID-vkCmdEndDebugUtilsLabelEXT-commandBuffer-01912
        // TODO: not checked, so unsafe for now

        // VUID-vkCmdEndDebugUtilsLabelEXT-commandBuffer-01913
        // TODO: not checked, so unsafe for now

        Ok(())
    }

    /// Inserts a command buffer debug label.
    #[inline]
    pub fn insert_debug_utils_label(
        &mut self,
        mut label_info: DebugUtilsLabel,
    ) -> Result<&mut Self, DebugUtilsError> {
        self.validate_insert_debug_utils_label(&mut label_info)?;

        unsafe {
            self.inner.insert_debug_utils_label(label_info);
        }

        Ok(self)
    }

    fn validate_insert_debug_utils_label(
        &self,
        label_info: &mut DebugUtilsLabel,
    ) -> Result<(), DebugUtilsError> {
        if !self
            .device()
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(DebugUtilsError::ExtensionNotEnabled {
                extension: "ext_debug_utils",
                reason: "tried to record a debug utils command",
            });
        }

        // VUID-vkCmdInsertDebugUtilsLabelEXT-commandBuffer-cmdpool
        if !(self.queue_family().supports_graphics() || self.queue_family().supports_compute()) {
            return Err(DebugUtilsError::NotSupportedByQueueFamily);
        }

        Ok(())
    }
}

impl SyncCommandBufferBuilder {
    /// Calls `vkCmdBeginDebugUtilsLabelEXT` on the builder.
    ///
    /// # Safety
    /// The command pool that this command buffer was allocated from must support graphics or
    /// compute operations
    #[inline]
    pub unsafe fn begin_debug_utils_label(&mut self, label_info: DebugUtilsLabel) {
        struct Cmd {
            label_info: DebugUtilsLabel,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "begin_debug_utils_label"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.begin_debug_utils_label(&self.label_info);
            }
        }

        self.commands.push(Box::new(Cmd { label_info }));
    }

    /// Calls `vkCmdEndDebugUtilsLabelEXT` on the builder.
    ///
    /// # Safety
    /// - The command pool that this command buffer was allocated from must support graphics or
    /// compute operations
    /// - There must be an outstanding `debug_marker_begin` command prior to the
    /// `debug_marker_end` on the queue.
    #[inline]
    pub unsafe fn end_debug_utils_label(&mut self) {
        struct Cmd {}

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "end_debug_utils_label"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.end_debug_utils_label();
            }
        }

        self.commands.push(Box::new(Cmd {}));
    }

    /// Calls `vkCmdInsertDebugUtilsLabelEXT` on the builder.
    ///
    /// # Safety
    /// The command pool that this command buffer was allocated from must support graphics or
    /// compute operations
    #[inline]
    pub unsafe fn insert_debug_utils_label(&mut self, label_info: DebugUtilsLabel) {
        struct Cmd {
            label_info: DebugUtilsLabel,
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "insert_debug_utils_label"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.insert_debug_utils_label(&self.label_info);
            }
        }

        self.commands.push(Box::new(Cmd { label_info }));
    }
}

impl UnsafeCommandBufferBuilder {
    /// Calls `vkCmdBeginDebugUtilsLabelEXT` on the builder.
    ///
    /// # Safety
    /// The command pool that this command buffer was allocated from must support graphics or
    /// compute operations
    #[inline]
    pub unsafe fn begin_debug_utils_label(&mut self, label_info: &DebugUtilsLabel) {
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

        let fns = self.device.instance().fns();
        fns.ext_debug_utils
            .cmd_begin_debug_utils_label_ext(self.handle, &label_info);
    }

    /// Calls `vkCmdEndDebugUtilsLabelEXT` on the builder.
    ///
    /// # Safety
    /// There must be an outstanding `vkCmdBeginDebugUtilsLabelEXT` command prior to the
    /// `vkQueueEndDebugUtilsLabelEXT` on the queue tha `CommandBuffer` is submitted to.
    #[inline]
    pub unsafe fn end_debug_utils_label(&mut self) {
        let fns = self.device.instance().fns();
        fns.ext_debug_utils
            .cmd_end_debug_utils_label_ext(self.handle);
    }

    /// Calls `vkCmdInsertDebugUtilsLabelEXT` on the builder.
    ///
    /// # Safety
    /// The command pool that this command buffer was allocated from must support graphics or
    /// compute operations
    #[inline]
    pub unsafe fn insert_debug_utils_label(&mut self, label_info: &DebugUtilsLabel) {
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

        let fns = self.device.instance().fns();
        fns.ext_debug_utils
            .cmd_insert_debug_utils_label_ext(self.handle, &label_info);
    }
}

/// Error that can happen when recording a debug utils command.
#[derive(Clone, Debug)]
pub enum DebugUtilsError {
    ExtensionNotEnabled {
        extension: &'static str,
        reason: &'static str,
    },

    /// The queue family doesn't allow this operation.
    NotSupportedByQueueFamily,
}

impl error::Error for DebugUtilsError {}

impl fmt::Display for DebugUtilsError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            Self::ExtensionNotEnabled { extension, reason } => {
                write!(f, "the extension {} must be enabled: {}", extension, reason)
            }
            Self::NotSupportedByQueueFamily => {
                write!(f, "the queue family doesn't allow this operation")
            }
        }
    }
}
