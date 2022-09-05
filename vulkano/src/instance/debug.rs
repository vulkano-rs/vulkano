// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Debug messenger called by intermediate layers or by the driver.
//!
//! When working on an application, it is recommended to register a debug messenger. For example if
//! you enable the validation layers provided by the official Vulkan SDK, they will warn you about
//! invalid API usages or performance problems by calling this callback. The callback can also
//! be called by the driver or by whatever intermediate layer is activated.
//!
//! Note that the vulkano library can also emit messages to warn you about performance issues.
//! TODO: ^ that's not the case yet, need to choose whether we keep this idea
//!
//! # Example
//!
//! ```
//! # use vulkano::instance::Instance;
//! # use std::sync::Arc;
//! # let instance: Arc<Instance> = return;
//! use vulkano::instance::debug::{DebugUtilsMessenger, DebugUtilsMessengerCreateInfo};
//!
//! let _callback = unsafe {
//!     DebugUtilsMessenger::new(
//!         instance,
//!         DebugUtilsMessengerCreateInfo::user_callback(Arc::new(|msg| {
//!             println!("Debug callback: {:?}", msg.description);
//!         })),
//!     ).ok()
//! };
//! ```
//!
//! The type of `msg` in the callback is [`Message`].
//!
//! Note that you must keep the `_callback` object alive for as long as you want your callback to
//! be callable. If you don't store the return value of `DebugUtilsMessenger`'s constructor in a
//! variable, it will be immediately destroyed and your callback will not work.
//!

use super::Instance;
use crate::{macros::vulkan_bitflags, VulkanError, VulkanObject};
use std::{
    error::Error,
    ffi::{c_void, CStr},
    fmt,
    mem::MaybeUninit,
    panic::{catch_unwind, AssertUnwindSafe, RefUnwindSafe},
    ptr,
    sync::Arc,
};

pub(super) type UserCallback = Arc<dyn Fn(&Message) + RefUnwindSafe + Send + Sync>;

/// Registration of a callback called by validation layers.
///
/// The callback can be called as long as this object is alive.
#[must_use = "The DebugUtilsMessenger object must be kept alive for as long as you want your callback to be called"]
pub struct DebugUtilsMessenger {
    handle: ash::vk::DebugUtilsMessengerEXT,
    instance: Arc<Instance>,
    _user_callback: Box<UserCallback>,
}

impl DebugUtilsMessenger {
    /// Initializes a debug callback.
    ///
    /// # Panics
    ///
    /// - Panics if the `message_severity` or `message_type` members of `create_info` are empty.
    ///
    /// # Safety
    ///
    /// - `create_info.user_callback` must not make any calls to the Vulkan API.
    pub unsafe fn new(
        instance: Arc<Instance>,
        mut create_info: DebugUtilsMessengerCreateInfo,
    ) -> Result<Self, DebugUtilsMessengerCreationError> {
        Self::validate_create(&instance, &mut create_info)?;
        let (handle, user_callback) = Self::record_create(&instance, create_info)?;

        Ok(DebugUtilsMessenger {
            handle,
            instance,
            _user_callback: user_callback,
        })
    }

    fn validate_create(
        instance: &Instance,
        create_info: &mut DebugUtilsMessengerCreateInfo,
    ) -> Result<(), DebugUtilsMessengerCreationError> {
        let &mut DebugUtilsMessengerCreateInfo {
            message_type,
            message_severity,
            user_callback: _,
            _ne: _,
        } = create_info;

        if !instance.enabled_extensions().ext_debug_utils {
            return Err(DebugUtilsMessengerCreationError::ExtensionNotEnabled {
                extension: "ext_debug_utils",
                reason: "tried to create a DebugUtilsMessenger",
            });
        }

        // VUID-VkDebugUtilsMessengerCreateInfoEXT-messageSeverity-parameter
        // TODO: message_severity.validate()?;

        // VUID-VkDebugUtilsMessengerCreateInfoEXT-messageSeverity-requiredbitmask
        assert!(!message_severity.is_empty());

        // VUID-VkDebugUtilsMessengerCreateInfoEXT-messageType-parameter
        // TODO: message_type.validate()?;

        // VUID-VkDebugUtilsMessengerCreateInfoEXT-messageType-requiredbitmask
        assert!(!message_type.is_empty());

        // VUID-PFN_vkDebugUtilsMessengerCallbackEXT-None-04769
        // Can't be checked, creation is unsafe.

        Ok(())
    }

    unsafe fn record_create(
        instance: &Instance,
        create_info: DebugUtilsMessengerCreateInfo,
    ) -> Result<
        (ash::vk::DebugUtilsMessengerEXT, Box<UserCallback>),
        DebugUtilsMessengerCreationError,
    > {
        let DebugUtilsMessengerCreateInfo {
            message_severity,
            message_type,
            user_callback,
            _ne: _,
        } = create_info;

        // Note that we need to double-box the callback, because a `*const Fn()` is a fat pointer
        // that can't be cast to a `*const c_void`.
        let user_callback = Box::new(user_callback);

        let create_info = ash::vk::DebugUtilsMessengerCreateInfoEXT {
            flags: ash::vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
            message_severity: message_severity.into(),
            message_type: message_type.into(),
            pfn_user_callback: Some(trampoline),
            p_user_data: &*user_callback as &Arc<_> as *const Arc<_> as *const c_void as *mut _,
            ..Default::default()
        };

        let fns = instance.fns();

        let handle = {
            let mut output = MaybeUninit::uninit();
            (fns.ext_debug_utils.create_debug_utils_messenger_ext)(
                instance.internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok((handle, user_callback))
    }
}

impl Drop for DebugUtilsMessenger {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.instance.fns();
            (fns.ext_debug_utils.destroy_debug_utils_messenger_ext)(
                self.instance.internal_object(),
                self.handle,
                ptr::null(),
            );
        }
    }
}

impl fmt::Debug for DebugUtilsMessenger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let Self {
            handle,
            instance,
            _user_callback: _,
        } = self;

        f.debug_struct("DebugUtilsMessenger")
            .field("handle", handle)
            .field("instance", instance)
            .finish_non_exhaustive()
    }
}

pub(super) unsafe extern "system" fn trampoline(
    message_severity: ash::vk::DebugUtilsMessageSeverityFlagsEXT,
    message_types: ash::vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const ash::vk::DebugUtilsMessengerCallbackDataEXT,
    user_data: *mut c_void,
) -> ash::vk::Bool32 {
    // Since we box the closure, the type system doesn't detect that the `UnwindSafe`
    // bound is enforced. Therefore we enforce it manually.
    let _ = catch_unwind(AssertUnwindSafe(move || {
        let user_callback = user_data as *mut UserCallback as *const _;
        let user_callback: &UserCallback = &*user_callback;

        let layer_prefix = (*callback_data)
            .p_message_id_name
            .as_ref()
            .map(|msg_id_name| {
                CStr::from_ptr(msg_id_name)
                    .to_str()
                    .expect("debug callback message not utf-8")
            });

        let description = CStr::from_ptr((*callback_data).p_message)
            .to_str()
            .expect("debug callback message not utf-8");

        let message = Message {
            severity: message_severity.into(),
            ty: message_types.into(),
            layer_prefix,
            description,
        };

        user_callback(&message);
    }));

    ash::vk::FALSE
}

/// Error that can happen when creating a `DebugUtilsMessenger`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DebugUtilsMessengerCreationError {
    ExtensionNotEnabled {
        extension: &'static str,
        reason: &'static str,
    },
}

impl Error for DebugUtilsMessengerCreationError {}

impl fmt::Display for DebugUtilsMessengerCreationError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match self {
            Self::ExtensionNotEnabled { extension, reason } => {
                write!(f, "the extension {} must be enabled: {}", extension, reason)
            }
        }
    }
}

impl From<VulkanError> for DebugUtilsMessengerCreationError {
    #[inline]
    fn from(err: VulkanError) -> DebugUtilsMessengerCreationError {
        panic!("unexpected error: {:?}", err)
    }
}

/// Parameters to create a `DebugUtilsMessenger`.
#[derive(Clone)]
pub struct DebugUtilsMessengerCreateInfo {
    /// The message severity types that the callback should be called for.
    ///
    /// The value must not be empty.
    ///
    /// The default value is `MessageSeverity::errors_and_warnings()`.
    pub message_severity: DebugUtilsMessageSeverity,

    /// The message types that the callback should be called for.
    ///
    /// The value must not be empty.
    ///
    /// The default value is `MessageType::general()`.
    pub message_type: DebugUtilsMessageType,

    /// The closure that should be called.
    ///
    /// The closure must not make any calls to the Vulkan API.
    /// If the closure panics, the panic is caught and ignored.
    ///
    /// The callback is provided inside an `Arc` so that it can be shared across multiple
    /// messengers.
    pub user_callback: UserCallback,

    pub _ne: crate::NonExhaustive,
}

impl DebugUtilsMessengerCreateInfo {
    /// Returns a `DebugUtilsMessengerCreateInfo` with the specified `user_callback`.
    #[inline]
    pub fn user_callback(user_callback: UserCallback) -> Self {
        Self {
            message_severity: DebugUtilsMessageSeverity {
                error: true,
                warning: true,
                ..DebugUtilsMessageSeverity::empty()
            },
            message_type: DebugUtilsMessageType {
                general: true,
                ..DebugUtilsMessageType::empty()
            },
            user_callback,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl fmt::Debug for DebugUtilsMessengerCreateInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let Self {
            message_severity,
            message_type,
            user_callback: _,
            _ne: _,
        } = self;

        f.debug_struct("DebugUtilsMessengerCreateInfo")
            .field("message_severity", message_severity)
            .field("message_type", message_type)
            .finish_non_exhaustive()
    }
}

/// A message received by the callback.
pub struct Message<'a> {
    /// Severity of message.
    pub severity: DebugUtilsMessageSeverity,
    /// Type of message,
    pub ty: DebugUtilsMessageType,
    /// Prefix of the layer that reported this message or `None` if unknown.
    pub layer_prefix: Option<&'a str>,
    /// Description of the message.
    pub description: &'a str,
}

vulkan_bitflags! {
    /// Severity of message.
    #[non_exhaustive]
    DebugUtilsMessageSeverity = DebugUtilsMessageSeverityFlagsEXT(u32);

    /// An error that may cause undefined results, including an application crash.
    error = ERROR,

    /// An unexpected use.
    warning = WARNING,

    /// An informational message that may be handy when debugging an application.
    information = INFO,

    /// Diagnostic information from the loader and layers.
    verbose = VERBOSE,
}

vulkan_bitflags! {
    /// Type of message.
    #[non_exhaustive]
    DebugUtilsMessageType = DebugUtilsMessageTypeFlagsEXT(u32);

    /// Specifies that some general event has occurred.
    general = GENERAL,

    /// Specifies that something has occurred during validation against the vulkan specification
    validation = VALIDATION,

    /// Specifies a potentially non-optimal use of Vulkan
    performance = PERFORMANCE,
}

/// A label to associate with a span of work in a queue.
///
/// When debugging, labels can be useful to identify which queue, or where in a specific queue,
/// something happened.
#[derive(Clone, Debug)]
pub struct DebugUtilsLabel {
    /// The name of the label.
    ///
    /// The default value is empty.
    pub label_name: String,

    /// An RGBA color value that is associated with the label, with values in the range `0.0..=1.0`.
    ///
    /// If set to `[0.0; 4]`, the value is ignored.
    ///
    /// The default value is `[0.0; 4]`.
    pub color: [f32; 4],

    pub _ne: crate::NonExhaustive,
}

impl Default for DebugUtilsLabel {
    #[inline]
    fn default() -> Self {
        Self {
            label_name: String::new(),
            color: [0.0; 4],
            _ne: crate::NonExhaustive(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn ensure_sendable() {
        // It's useful to be able to initialize a DebugUtilsMessenger on one thread
        // and keep it alive on another thread.
        let instance = instance!();
        let callback = unsafe {
            DebugUtilsMessenger::new(
                instance,
                DebugUtilsMessengerCreateInfo {
                    message_severity: DebugUtilsMessageSeverity::empty(),
                    message_type: DebugUtilsMessageType {
                        general: true,
                        validation: true,
                        performance: true,
                        ..DebugUtilsMessageType::empty()
                    },
                    ..DebugUtilsMessengerCreateInfo::user_callback(Arc::new(|_| {}))
                },
            )
        };
        thread::spawn(move || {
            let _ = callback;
        });
    }
}
