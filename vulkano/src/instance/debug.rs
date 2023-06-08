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
//! # Examples
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

use super::{Instance, InstanceExtensions};
use crate::{
    macros::{vulkan_bitflags, vulkan_enum},
    RequiresOneOf, RuntimeError, ValidationError, Version, VulkanError, VulkanObject,
};
use std::{
    ffi::{c_void, CStr},
    fmt::{Debug, Error as FmtError, Formatter},
    mem::MaybeUninit,
    panic::{catch_unwind, AssertUnwindSafe, RefUnwindSafe},
    ptr,
    sync::Arc,
};

pub type UserCallback = Arc<dyn Fn(&Message<'_>) + RefUnwindSafe + Send + Sync>;

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
        create_info: DebugUtilsMessengerCreateInfo,
    ) -> Result<Self, VulkanError> {
        Self::validate_new(&instance, &create_info)?;

        Ok(Self::new_unchecked(instance, create_info)?)
    }

    fn validate_new(
        instance: &Instance,
        create_info: &DebugUtilsMessengerCreateInfo,
    ) -> Result<(), ValidationError> {
        if !instance.enabled_extensions().ext_debug_utils {
            return Err(ValidationError {
                requires_one_of: RequiresOneOf {
                    instance_extensions: &["ext_debug_utils"],
                    ..Default::default()
                },
                ..Default::default()
            });
        }

        create_info
            .validate(instance)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        instance: Arc<Instance>,
        create_info: DebugUtilsMessengerCreateInfo,
    ) -> Result<Self, RuntimeError> {
        let DebugUtilsMessengerCreateInfo {
            message_severity,
            message_type,
            user_callback,
            _ne: _,
        } = create_info;

        // Note that we need to double-box the callback, because a `*const Fn()` is a fat pointer
        // that can't be cast to a `*const c_void`.
        let user_callback = Box::new(user_callback);

        let create_info_vk = ash::vk::DebugUtilsMessengerCreateInfoEXT {
            flags: ash::vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
            message_severity: message_severity.into(),
            message_type: message_type.into(),
            pfn_user_callback: Some(trampoline),
            p_user_data: &*user_callback as &Arc<_> as *const Arc<_> as *const c_void as *mut _,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            (fns.ext_debug_utils.create_debug_utils_messenger_ext)(
                instance.handle(),
                &create_info_vk,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(RuntimeError::from)?;
            output.assume_init()
        };

        Ok(DebugUtilsMessenger {
            handle,
            instance,
            _user_callback: user_callback,
        })
    }
}

impl Drop for DebugUtilsMessenger {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.instance.fns();
            (fns.ext_debug_utils.destroy_debug_utils_messenger_ext)(
                self.instance.handle(),
                self.handle,
                ptr::null(),
            );
        }
    }
}

impl Debug for DebugUtilsMessenger {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
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
            message_severity: DebugUtilsMessageSeverity::ERROR | DebugUtilsMessageSeverity::WARNING,
            message_type: DebugUtilsMessageType::GENERAL,
            user_callback,
            _ne: crate::NonExhaustive(()),
        }
    }

    #[inline]
    pub(crate) fn validate(&self, instance: &Instance) -> Result<(), ValidationError> {
        self.validate_raw(instance.api_version(), instance.enabled_extensions())
    }

    pub(crate) fn validate_raw(
        &self,
        instance_api_version: Version,
        instance_extensions: &InstanceExtensions,
    ) -> Result<(), ValidationError> {
        let &DebugUtilsMessengerCreateInfo {
            message_severity,
            message_type,
            user_callback: _,
            _ne: _,
        } = self;

        message_severity
            .validate_instance_raw(instance_api_version, instance_extensions)
            .map_err(|err| ValidationError {
                context: "message_severity".into(),
                vuids: &["VUID-VkDebugUtilsMessengerCreateInfoEXT-messageSeverity-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        if message_severity.is_empty() {
            return Err(ValidationError {
                context: "message_severity".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkDebugUtilsMessengerCreateInfoEXT-messageSeverity-requiredbitmask"],
                ..Default::default()
            });
        }

        message_type
            .validate_instance_raw(instance_api_version, instance_extensions)
            .map_err(|err| ValidationError {
                context: "message_type".into(),
                vuids: &["VUID-VkDebugUtilsMessengerCreateInfoEXT-messageType-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        if message_type.is_empty() {
            return Err(ValidationError {
                context: "message_type".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkDebugUtilsMessengerCreateInfoEXT-messageType-requiredbitmask"],
                ..Default::default()
            });
        }

        // VUID-PFN_vkDebugUtilsMessengerCallbackEXT-None-04769
        // Can't be checked, creation is unsafe.

        Ok(())
    }
}

impl Debug for DebugUtilsMessengerCreateInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
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
    #[non_exhaustive]

    /// Severity of message.
    DebugUtilsMessageSeverity = DebugUtilsMessageSeverityFlagsEXT(u32);

    /// An error that may cause undefined results, including an application crash.
    ERROR = ERROR,

    /// An unexpected use.
    WARNING = WARNING,

    /// An informational message that may be handy when debugging an application.
    INFO = INFO,

    /// Diagnostic information from the loader and layers.
    VERBOSE = VERBOSE,
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Type of message.
    DebugUtilsMessageType = DebugUtilsMessageTypeFlagsEXT(u32);

    /// Specifies that some general event has occurred.
    GENERAL = GENERAL,

    /// Specifies that something has occurred during validation against the vulkan specification
    VALIDATION = VALIDATION,

    /// Specifies a potentially non-optimal use of Vulkan
    PERFORMANCE = PERFORMANCE,
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

vulkan_enum! {
    #[non_exhaustive]

    /// Features of the validation layer to enable.
    ValidationFeatureEnable = ValidationFeatureEnableEXT(i32);

    /// The validation layer will use shader programs running on the GPU to provide additional
    /// validation.
    ///
    /// This must not be used together with `DebugPrintf`.
    GpuAssisted = GPU_ASSISTED,

    /// The validation layer will reserve and use one descriptor set slot for its own use.
    /// The limit reported by
    /// [`max_bound_descriptor_sets`](crate::device::Properties::max_bound_descriptor_sets)
    /// will be reduced by 1.
    ///
    /// `GpuAssisted` must also be enabled.
    GpuAssistedReserveBindingSlot = GPU_ASSISTED_RESERVE_BINDING_SLOT,

    /// The validation layer will report recommendations that are not strictly errors,
    /// but that may be considered good Vulkan practice.
    BestPractices = BEST_PRACTICES,

    /// The validation layer will process `debugPrintfEXT` operations in shaders, and send them
    /// to the debug callback.
    ///
    /// This must not be used together with `GpuAssisted`.
    DebugPrintf = DEBUG_PRINTF,

    /// The validation layer will report errors relating to synchronization, such as data races and
    /// the use of synchronization primitives.
    SynchronizationValidation = SYNCHRONIZATION_VALIDATION,
}

vulkan_enum! {
    #[non_exhaustive]

    /// Features of the validation layer to disable.
    ValidationFeatureDisable = ValidationFeatureDisableEXT(i32);

    /// All validation is disabled.
    All = ALL,

    /// Shader validation is disabled.
    Shaders = SHADERS,

    /// Thread safety validation is disabled.
    ThreadSafety = THREAD_SAFETY,

    /// Stateless parameter validation is disabled.
    ApiParameters = API_PARAMETERS,

    /// Object lifetime validation is disabled.
    ObjectLifetimes = OBJECT_LIFETIMES,

    /// Core validation checks are disabled.
    ///
    /// This also disables shader validation and GPU-assisted validation.
    CoreChecks = CORE_CHECKS,

    /// Protection against duplicate non-dispatchable handles is disabled.
    UniqueHandles = UNIQUE_HANDLES,

    /// Results of shader validation will not be cached, and are validated from scratch each time.
    ShaderValidationCache = SHADER_VALIDATION_CACHE,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        instance::{InstanceCreateInfo, InstanceExtensions},
        VulkanLibrary,
    };
    use std::thread;

    #[test]
    fn ensure_sendable() {
        // It's useful to be able to initialize a DebugUtilsMessenger on one thread
        // and keep it alive on another thread.
        let instance = {
            let library = match VulkanLibrary::new() {
                Ok(x) => x,
                Err(_) => return,
            };

            match Instance::new(
                library,
                InstanceCreateInfo {
                    enabled_extensions: InstanceExtensions {
                        ext_debug_utils: true,
                        ..InstanceExtensions::empty()
                    },
                    ..Default::default()
                },
            ) {
                Ok(x) => x,
                Err(_) => return,
            }
        };

        let callback = unsafe {
            DebugUtilsMessenger::new(
                instance,
                DebugUtilsMessengerCreateInfo {
                    message_severity: DebugUtilsMessageSeverity::ERROR,
                    message_type: DebugUtilsMessageType::GENERAL
                        | DebugUtilsMessageType::VALIDATION
                        | DebugUtilsMessageType::PERFORMANCE,
                    ..DebugUtilsMessengerCreateInfo::user_callback(Arc::new(|_| {}))
                },
            )
        }
        .unwrap();
        thread::spawn(move || {
            drop(callback);
        });
    }
}
