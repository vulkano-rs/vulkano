// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::command_buffer::{
    synced::{Command, SyncCommandBufferBuilder},
    sys::UnsafeCommandBufferBuilder,
    AutoCommandBufferBuilder, AutoCommandBufferBuilderContextError, DebugMarkerError,
};
use std::{error, ffi::CStr, fmt};

/// # Commands for debugging.
impl<L, P> AutoCommandBufferBuilder<L, P> {
    /// Open a command buffer debug label region.
    ///
    /// Note: you need to enable `VK_EXT_debug_utils` extension when creating an instance.
    #[inline]
    pub fn debug_marker_begin(
        &mut self,
        name: &'static CStr,
        color: [f32; 4],
    ) -> Result<&mut Self, DebugMarkerError> {
        if !self.queue_family().supports_graphics() && self.queue_family().supports_compute() {
            return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
        }

        check_debug_marker_color(color)?;

        unsafe {
            self.inner.debug_marker_begin(name.into(), color);
        }

        Ok(self)
    }

    /// Close a command buffer label region.
    ///
    /// Note: you need to open a command buffer label region first with `debug_marker_begin`.
    /// Note: you need to enable `VK_EXT_debug_utils` extension when creating an instance.
    #[inline]
    pub fn debug_marker_end(&mut self) -> Result<&mut Self, DebugMarkerError> {
        if !self.queue_family().supports_graphics() && self.queue_family().supports_compute() {
            return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
        }

        // TODO: validate that debug_marker_begin with same name was sent earlier

        unsafe {
            self.inner.debug_marker_end();
        }

        Ok(self)
    }

    /// Insert a label into a command buffer.
    ///
    /// Note: you need to enable `VK_EXT_debug_utils` extension when creating an instance.
    #[inline]
    pub fn debug_marker_insert(
        &mut self,
        name: &'static CStr,
        color: [f32; 4],
    ) -> Result<&mut Self, DebugMarkerError> {
        if !self.queue_family().supports_graphics() && self.queue_family().supports_compute() {
            return Err(AutoCommandBufferBuilderContextError::NotSupportedByQueueFamily.into());
        }

        check_debug_marker_color(color)?;

        unsafe {
            self.inner.debug_marker_insert(name.into(), color);
        }

        Ok(self)
    }
}

/// Checks whether the specified color is valid as debug marker color.
///
/// The color parameter must contain RGBA values in order, in the range 0.0 to 1.0.
fn check_debug_marker_color(color: [f32; 4]) -> Result<(), CheckColorError> {
    // The values contain RGBA values in order, in the range 0.0 to 1.0.
    if color.iter().any(|x| !(0f32..=1f32).contains(x)) {
        return Err(CheckColorError);
    }

    Ok(())
}

/// Error that can happen from `check_debug_marker_color`.
#[derive(Debug, Copy, Clone)]
pub struct CheckColorError;

impl error::Error for CheckColorError {}

impl fmt::Display for CheckColorError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                CheckColorError => "color parameter does contains values out of 0.0 to 1.0 range",
            }
        )
    }
}

impl SyncCommandBufferBuilder {
    /// Calls `vkCmdBeginDebugUtilsLabelEXT` on the builder.
    ///
    /// # Safety
    /// The command pool that this command buffer was allocated from must support graphics or
    /// compute operations
    #[inline]
    pub unsafe fn debug_marker_begin(&mut self, name: &'static CStr, color: [f32; 4]) {
        struct Cmd {
            name: &'static CStr,
            color: [f32; 4],
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "debug_marker_begin"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.debug_marker_begin(self.name, self.color);
            }
        }

        self.commands.push(Box::new(Cmd { name, color }));
    }

    /// Calls `vkCmdEndDebugUtilsLabelEXT` on the builder.
    ///
    /// # Safety
    /// - The command pool that this command buffer was allocated from must support graphics or
    /// compute operations
    /// - There must be an outstanding `debug_marker_begin` command prior to the
    /// `debug_marker_end` on the queue.
    #[inline]
    pub unsafe fn debug_marker_end(&mut self) {
        struct Cmd {}

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "debug_marker_end"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.debug_marker_end();
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
    pub unsafe fn debug_marker_insert(&mut self, name: &'static CStr, color: [f32; 4]) {
        struct Cmd {
            name: &'static CStr,
            color: [f32; 4],
        }

        impl Command for Cmd {
            fn name(&self) -> &'static str {
                "debug_marker_insert"
            }

            unsafe fn send(&self, out: &mut UnsafeCommandBufferBuilder) {
                out.debug_marker_insert(self.name, self.color);
            }
        }

        self.commands.push(Box::new(Cmd { name, color }));
    }
}

impl UnsafeCommandBufferBuilder {
    /// Calls `vkCmdBeginDebugUtilsLabelEXT` on the builder.
    ///
    /// # Safety
    /// The command pool that this command buffer was allocated from must support graphics or
    /// compute operations
    #[inline]
    pub unsafe fn debug_marker_begin(&mut self, name: &CStr, color: [f32; 4]) {
        let fns = self.device.instance().fns();
        let info = ash::vk::DebugUtilsLabelEXT {
            p_label_name: name.as_ptr(),
            color,
            ..Default::default()
        };
        fns.ext_debug_utils
            .cmd_begin_debug_utils_label_ext(self.handle, &info);
    }

    /// Calls `vkCmdEndDebugUtilsLabelEXT` on the builder.
    ///
    /// # Safety
    /// There must be an outstanding `vkCmdBeginDebugUtilsLabelEXT` command prior to the
    /// `vkQueueEndDebugUtilsLabelEXT` on the queue tha `CommandBuffer` is submitted to.
    #[inline]
    pub unsafe fn debug_marker_end(&mut self) {
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
    pub unsafe fn debug_marker_insert(&mut self, name: &CStr, color: [f32; 4]) {
        let fns = self.device.instance().fns();
        let info = ash::vk::DebugUtilsLabelEXT {
            p_label_name: name.as_ptr(),
            color,
            ..Default::default()
        };
        fns.ext_debug_utils
            .cmd_insert_debug_utils_label_ext(self.handle, &info);
    }
}
