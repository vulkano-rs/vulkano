// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! API entry point.
//!
//! The first thing to do before you start using Vulkan is to create an `Instance` object.
//!
//! For example:
//!
//! ```no_run
//! use vulkano::instance::Instance;
//! use vulkano::instance::InstanceExtensions;
//! use vulkano::Version;
//!
//! let instance = match Instance::new(Default::default()) {
//!     Ok(i) => i,
//!     Err(err) => panic!("Couldn't build instance: {:?}", err)
//! };
//! ```
//!
//! Creating an instance initializes everything and allows you to enumerate physical devices,
//! ie. all the Vulkan implementations that are available on the system.
//!
//! ```no_run
//! # use vulkano::instance::Instance;
//! # use vulkano::instance::InstanceExtensions;
//! # use vulkano::Version;
//! use vulkano::device::physical::PhysicalDevice;
//!
//! # let instance = Instance::new(Default::default()).unwrap();
//! for physical_device in PhysicalDevice::enumerate(&instance) {
//!     println!("Available device: {}", physical_device.properties().device_name);
//! }
//! ```
//!
//! # Enumerating physical devices and creating a device
//!
//! After you have created an instance, the next step is usually to enumerate the physical devices
//! that are available on the system with `PhysicalDevice::enumerate()` (see above).
//!
//! When choosing which physical device to use, keep in mind that physical devices may or may not
//! be able to draw to a certain surface (ie. to a window or a monitor), or may even not be able
//! to draw at all. See the `swapchain` module for more information about surfaces.
//!
//! Once you have chosen a physical device, you can create a `Device` object from it. See the
//! `device` module for more info.

use self::{
    debug::{DebugUtilsMessengerCreateInfo, UserCallback},
    loader::{FunctionPointers, Loader},
};
pub use self::{
    extensions::InstanceExtensions,
    layers::{layers_list, LayerProperties, LayersListError},
    loader::LoadingError,
};
use crate::{
    check_errors,
    device::physical::{init_physical_devices, PhysicalDeviceInfo},
    instance::debug::{trampoline, DebugUtilsMessageSeverity, DebugUtilsMessageType},
    Error, OomError, VulkanObject,
};
pub use crate::{
    extensions::{ExtensionRestriction, ExtensionRestrictionError, SupportedExtensionsError},
    fns::InstanceFunctions,
    version::Version,
};
use smallvec::SmallVec;
use std::{
    error,
    ffi::{c_void, CString},
    fmt,
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    ops::Deref,
    panic::{RefUnwindSafe, UnwindSafe},
    ptr,
    sync::Arc,
};

pub mod debug;
pub(crate) mod extensions;
mod layers;
pub mod loader;

/// An instance of a Vulkan context. This is the main object that should be created by an
/// application before everything else.
///
/// # Application and engine info
///
/// When you create an instance, you have the possibility to set information about your application
/// and its engine.
///
/// Providing this information allows for example the driver to let the user configure the driver's
/// behavior for your application alone through a control panel.
///
/// ```no_run
/// # #[macro_use] extern crate vulkano;
/// # fn main() {
/// use vulkano::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
/// use vulkano::Version;
///
/// let _instance = Instance::new(InstanceCreateInfo::application_from_cargo_toml()).unwrap();
/// # }
/// ```
///
/// # API versions
///
/// Both an `Instance` and a [`Device`](crate::device::Device) have a highest version of the Vulkan
/// API that they support. This places a limit on what Vulkan functions and features are available
/// to use when used on a particular instance or device. It is possible for the instance and the
/// device to support different versions. The supported version for an instance can be queried
/// before creation with
/// [`FunctionPointers::api_version`](crate::instance::loader::FunctionPointers::api_version),
/// while for a device it can be retrieved with
/// [`PhysicalDevice::api_version`](crate::device::physical::PhysicalDevice::api_version).
///
/// When creating an `Instance`, you have to specify a maximum API version that you will use.
/// This restricts the API version that is available for the instance and any devices created from
/// it. For example, if both instance and device potentially support Vulkan 1.2, but you specify
/// 1.1 as the maximum API version when creating the `Instance`, then you can only use Vulkan 1.1
/// functions, even though they could theoretically support a higher version. You can think of it
/// as a promise never to use any functionality from a higher version.
///
/// The maximum API version is not a _minimum_, so it is possible to set it to a higher version than
/// what the instance or device inherently support. The final API version that you are able to use
/// on an instance or device is the lower of the supported API version and the chosen maximum API
/// version of the `Instance`.
///
/// Due to a quirk in how the Vulkan 1.0 specification was written, if the instance only
/// supports Vulkan 1.0, then it is not possible to specify a maximum API version higher than 1.0.
/// Trying to create an `Instance` will return an `IncompatibleDriver` error. Consequently, it is
/// not possible to use a higher device API version with an instance that only supports 1.0.
///
/// # Extensions
///
/// When creating an `Instance`, you must provide a list of extensions that must be enabled on the
/// newly-created instance. Trying to enable an extension that is not supported by the system will
/// result in an error.
///
/// Contrary to OpenGL, it is not possible to use the features of an extension if it was not
/// explicitly enabled.
///
/// Extensions are especially important to take into account if you want to render images on the
/// screen, as the only way to do so is to use the `VK_KHR_surface` extension. More information
/// about this in the `swapchain` module.
///
/// For example, here is how we create an instance with the `VK_KHR_surface` and
/// `VK_KHR_android_surface` extensions enabled, which will allow us to render images to an
/// Android screen. You can compile and run this code on any system, but it is highly unlikely to
/// succeed on anything else than an Android-running device.
///
/// ```no_run
/// use vulkano::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
/// use vulkano::Version;
///
/// let extensions = InstanceExtensions {
///     khr_surface: true,
///     khr_android_surface: true,
///     .. InstanceExtensions::none()
/// };
///
/// let instance = match Instance::new(InstanceCreateInfo {
///     enabled_extensions: extensions,
///     ..Default::default()
/// }) {
///     Ok(i) => i,
///     Err(err) => panic!("Couldn't build instance: {:?}", err)
/// };
/// ```
///
/// # Layers
///
/// When creating an `Instance`, you have the possibility to pass a list of **layers** that will
/// be activated on the newly-created instance. The list of available layers can be retrieved by
/// calling [the `layers_list` function](crate::instance::layers_list).
///
/// A layer is a component that will hook and potentially modify the Vulkan function calls.
/// For example, activating a layer could add a frames-per-second counter on the screen, or it
/// could send information to a debugger that will debug your application.
///
/// > **Note**: From an application's point of view, layers "just exist". In practice, on Windows
/// > and Linux, layers can be installed by third party installers or by package managers and can
/// > also be activated by setting the value of the `VK_INSTANCE_LAYERS` environment variable
/// > before starting the program. See the documentation of the official Vulkan loader for these
/// > platforms.
///
/// > **Note**: In practice, the most common use of layers right now is for debugging purposes.
/// > To do so, you are encouraged to set the `VK_INSTANCE_LAYERS` environment variable on Windows
/// > or Linux instead of modifying the source code of your program. For example:
/// > `export VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_api_dump` on Linux if you installed the Vulkan SDK
/// > will print the list of raw Vulkan function calls.
///
/// ## Example
///
/// ```
/// # use std::sync::Arc;
/// # use std::error::Error;
/// # use vulkano::instance;
/// # use vulkano::instance::Instance;
/// # use vulkano::instance::InstanceCreateInfo;
/// # use vulkano::instance::InstanceExtensions;
/// # use vulkano::Version;
/// # fn test() -> Result<Arc<Instance>, Box<dyn Error>> {
/// // For the sake of the example, we activate all the layers that
/// // contain the word "foo" in their description.
/// let layers: Vec<_> = instance::layers_list()?
///     .filter(|l| l.description().contains("foo"))
///     .collect();
///
/// let instance = Instance::new(InstanceCreateInfo {
///     enabled_layers: layers.iter().map(|l| l.name().to_owned()).collect(),
///     ..Default::default()
/// })?;
/// # Ok(instance)
/// # }
/// ```
// TODO: mention that extensions must be supported by layers as well
pub struct Instance {
    handle: ash::vk::Instance,
    fns: InstanceFunctions,
    pub(crate) physical_device_infos: Vec<PhysicalDeviceInfo>,

    api_version: Version,
    enabled_extensions: InstanceExtensions,
    enabled_layers: Vec<String>,
    function_pointers: OwnedOrRef<FunctionPointers<Box<dyn Loader>>>,
    max_api_version: Version,
    user_callbacks: Vec<Box<UserCallback>>,
}

// TODO: fix the underlying cause instead
impl UnwindSafe for Instance {}
impl RefUnwindSafe for Instance {}

impl Instance {
    /// Creates a new `Instance`.
    ///
    /// # Panics
    ///
    /// - Panics if any version numbers in `create_info` contain a field too large to be converted
    ///   into a Vulkan version number.
    /// - Panics if `create_info.max_api_version` is not at least `V1_0`.
    pub fn new(create_info: InstanceCreateInfo) -> Result<Arc<Instance>, InstanceCreationError> {
        unsafe { Self::with_debug_utils_messengers(create_info, []) }
    }

    /// Creates a new `Instance` with debug messengers to use during the creation and destruction
    /// of the instance.
    ///
    /// The debug messengers are not used at any other time,
    /// [`DebugUtilsMessenger`](crate::instance::debug::DebugUtilsMessenger) should be used for
    /// that.
    ///
    /// If `debug_utils_messengers` is not empty, the `ext_debug_utils` extension must be set in
    /// `enabled_extensions`.
    ///
    /// # Panics
    ///
    /// - Panics if the `message_severity` or `message_type` members of any element of
    ///   `debug_utils_messengers` are empty.
    ///
    /// # Safety
    ///
    /// - The `user_callback` of each element of `debug_utils_messengers` must not make any calls
    ///   to the Vulkan API.
    pub unsafe fn with_debug_utils_messengers(
        create_info: InstanceCreateInfo,
        debug_utils_messengers: impl IntoIterator<Item = DebugUtilsMessengerCreateInfo>,
    ) -> Result<Arc<Instance>, InstanceCreationError> {
        let InstanceCreateInfo {
            application_name,
            application_version,
            enabled_extensions,
            enabled_layers,
            engine_name,
            engine_version,
            function_pointers,
            max_api_version,
            _ne: _,
        } = create_info;

        let function_pointers = if let Some(function_pointers) = function_pointers {
            OwnedOrRef::Owned(function_pointers)
        } else {
            OwnedOrRef::Ref(loader::auto_loader()?)
        };

        let (api_version, max_api_version) = {
            let api_version = function_pointers.api_version()?;
            let max_api_version = if let Some(max_api_version) = max_api_version {
                max_api_version
            } else if api_version < Version::V1_1 {
                api_version
            } else {
                Version::HEADER_VERSION
            };

            (std::cmp::min(max_api_version, api_version), max_api_version)
        };

        // VUID-VkApplicationInfo-apiVersion-04010
        assert!(max_api_version >= Version::V1_0);

        // Check if the extensions are correct
        enabled_extensions.check_requirements(
            &InstanceExtensions::supported_by_core_with_loader(&function_pointers)?,
            api_version,
        )?;

        // FIXME: check whether each layer is supported
        let enabled_layers_cstr: Vec<CString> = enabled_layers
            .iter()
            .map(|name| CString::new(name.clone()).unwrap())
            .collect();
        let enabled_layers_ptrs = enabled_layers_cstr
            .iter()
            .map(|layer| layer.as_ptr())
            .collect::<SmallVec<[_; 2]>>();

        let enabled_extensions_cstr: Vec<CString> = (&enabled_extensions).into();
        let enabled_extensions_ptrs = enabled_extensions_cstr
            .iter()
            .map(|extension| extension.as_ptr())
            .collect::<SmallVec<[_; 2]>>();

        let application_name_cstr = application_name.map(|name| CString::new(name).unwrap());
        let engine_name_cstr = engine_name.map(|name| CString::new(name).unwrap());
        let application_info = ash::vk::ApplicationInfo {
            p_application_name: application_name_cstr
                .as_ref()
                .map(|s| s.as_ptr())
                .unwrap_or(ptr::null()),
            application_version: application_version
                .try_into()
                .expect("Version out of range"),
            p_engine_name: engine_name_cstr
                .as_ref()
                .map(|s| s.as_ptr())
                .unwrap_or(ptr::null()),
            engine_version: engine_version.try_into().expect("Version out of range"),
            api_version: max_api_version.try_into().expect("Version out of range"),
            ..Default::default()
        };

        let mut create_info = ash::vk::InstanceCreateInfo {
            flags: ash::vk::InstanceCreateFlags::empty(),
            p_application_info: &application_info,
            enabled_layer_count: enabled_layers_ptrs.len() as u32,
            pp_enabled_layer_names: enabled_layers_ptrs.as_ptr(),
            enabled_extension_count: enabled_extensions_ptrs.len() as u32,
            pp_enabled_extension_names: enabled_extensions_ptrs.as_ptr(),
            ..Default::default()
        };

        // Handle debug messengers
        let debug_utils_messengers = debug_utils_messengers.into_iter();
        let mut debug_utils_messenger_create_infos =
            Vec::with_capacity(debug_utils_messengers.size_hint().0);
        let mut user_callbacks = Vec::with_capacity(debug_utils_messengers.size_hint().0);

        for create_info in debug_utils_messengers {
            let DebugUtilsMessengerCreateInfo {
                message_type,
                message_severity,
                user_callback,
                _ne: _,
            } = create_info;

            // VUID-VkInstanceCreateInfo-pNext-04926
            if !enabled_extensions.ext_debug_utils {
                return Err(InstanceCreationError::ExtensionNotEnabled {
                    extension: "ext_debug_utils",
                    reason: "debug_utils_messengers was not empty",
                });
            }

            // VUID-VkDebugUtilsMessengerCreateInfoEXT-messageSeverity-requiredbitmask
            assert!(message_severity != DebugUtilsMessageSeverity::none());

            // VUID-VkDebugUtilsMessengerCreateInfoEXT-messageType-requiredbitmask
            assert!(message_type != DebugUtilsMessageType::none());

            // VUID-PFN_vkDebugUtilsMessengerCallbackEXT-None-04769
            // Can't be checked, creation is unsafe.

            let user_callback = Box::new(user_callback);
            let create_info = ash::vk::DebugUtilsMessengerCreateInfoEXT {
                flags: ash::vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
                message_severity: message_severity.into(),
                message_type: message_type.into(),
                pfn_user_callback: Some(trampoline),
                p_user_data: &*user_callback as &Arc<_> as *const Arc<_> as *const c_void as *mut _,
                ..Default::default()
            };

            debug_utils_messenger_create_infos.push(create_info);
            user_callbacks.push(user_callback);
        }

        for i in 1..debug_utils_messenger_create_infos.len() {
            debug_utils_messenger_create_infos[i - 1].p_next =
                &debug_utils_messenger_create_infos[i] as *const _ as *const _;
        }

        if let Some(info) = debug_utils_messenger_create_infos.first() {
            create_info.p_next = info as *const _ as *const _;
        }

        // Creating the Vulkan instance.
        let handle = {
            let mut output = MaybeUninit::uninit();
            let fns = function_pointers.fns();
            check_errors(
                fns.v1_0
                    .create_instance(&create_info, ptr::null(), output.as_mut_ptr()),
            )?;
            output.assume_init()
        };

        // Loading the function pointers of the newly-created instance.
        let fns = {
            InstanceFunctions::load(|name| {
                function_pointers.get_instance_proc_addr(handle, name.as_ptr())
            })
        };

        let mut instance = Instance {
            handle,
            fns,
            physical_device_infos: Vec::new(),

            api_version,
            enabled_extensions,
            enabled_layers,
            function_pointers,
            max_api_version,
            user_callbacks,
        };

        // Enumerating all physical devices.
        instance.physical_device_infos = init_physical_devices(&instance)?;

        Ok(Arc::new(instance))
    }

    /// Returns the Vulkan version supported by the instance.
    ///
    /// This is the lower of the
    /// [driver's supported version](crate::instance::loader::FunctionPointers::api_version) and
    /// [`max_api_version`](Instance::max_api_version).
    #[inline]
    pub fn api_version(&self) -> Version {
        self.api_version
    }

    /// Returns the maximum Vulkan version that was specified when creating the instance.
    #[inline]
    pub fn max_api_version(&self) -> Version {
        self.max_api_version
    }

    /// Grants access to the Vulkan functions of the instance.
    #[inline]
    pub fn fns(&self) -> &InstanceFunctions {
        &self.fns
    }

    /// Returns the extensions that have been enabled on the instance.
    #[inline]
    pub fn enabled_extensions(&self) -> &InstanceExtensions {
        &self.enabled_extensions
    }

    /// Returns the layers that have been enabled on the instance.
    #[inline]
    pub fn enabled_layers(&self) -> &[String] {
        &self.enabled_layers
    }
}

impl Drop for Instance {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.fns.v1_0.destroy_instance(self.handle, ptr::null());
        }
    }
}

unsafe impl VulkanObject for Instance {
    type Object = ash::vk::Instance;

    #[inline]
    fn internal_object(&self) -> ash::vk::Instance {
        self.handle
    }
}

impl PartialEq for Instance {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

impl Eq for Instance {}

impl Hash for Instance {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
    }
}

impl fmt::Debug for Instance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let Self {
            handle,
            fns,
            physical_device_infos,
            api_version,
            enabled_extensions,
            enabled_layers,
            function_pointers,
            max_api_version,
            user_callbacks: _,
        } = self;

        f.debug_struct("Instance")
            .field("handle", handle)
            .field("fns", fns)
            .field("physical_device_infos", physical_device_infos)
            .field("api_version", api_version)
            .field("enabled_extensions", enabled_extensions)
            .field("enabled_layers", enabled_layers)
            .field("function_pointers", function_pointers)
            .field("max_api_version", max_api_version)
            .finish_non_exhaustive()
    }
}

/// Parameters to create a new `Instance`.
#[derive(Debug)]
pub struct InstanceCreateInfo {
    /// A string of your choice stating the name of your application.
    ///
    /// The default value is `None`.
    pub application_name: Option<String>,

    /// A version number of your choice specifying the version of your application.
    ///
    /// The default value is zero.
    pub application_version: Version,

    /// The extensions to enable on the instance.
    ///
    /// The default value is [`InstanceExtensions::none()`].
    pub enabled_extensions: InstanceExtensions,

    /// The layers to enable on the instance.
    ///
    /// The default value is empty.
    pub enabled_layers: Vec<String>,

    /// A string of your choice stating the name of the engine used to power the application.
    pub engine_name: Option<String>,

    /// A version number of your choice specifying the version of the engine used to power the
    /// application.
    ///
    /// The default value is zero.
    pub engine_version: Version,

    /// Function pointers loaded from a custom loader.
    ///
    /// You can use this if you want to load the Vulkan API explicitly, rather than using Vulkano's
    /// default.
    pub function_pointers: Option<FunctionPointers<Box<dyn Loader>>>,

    /// The highest Vulkan API version that the application will use with the instance.
    ///
    /// Usually, you will want to leave this at the default.
    ///
    /// The default value is [`Version::HEADER_VERSION`], but if the
    /// supported instance version is 1.0, then it will be 1.0.
    pub max_api_version: Option<Version>,

    pub _ne: crate::NonExhaustive,
}

impl Default for InstanceCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            application_name: None,
            application_version: Version::major_minor(0, 0),
            enabled_extensions: InstanceExtensions::none(),
            enabled_layers: Vec::new(),
            engine_name: None,
            engine_version: Version::major_minor(0, 0),
            function_pointers: None,
            max_api_version: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl InstanceCreateInfo {
    /// Returns an `InstanceCreateInfo` with the `application_name` and `application_version` set
    /// from information in your crate's Cargo.toml file.
    ///
    /// # Panics
    ///
    /// - Panics if the required environment variables are missing, which happens if the project
    ///   wasn't built by Cargo.
    #[inline]
    pub fn application_from_cargo_toml() -> Self {
        Self {
            application_name: Some(env!("CARGO_PKG_NAME").to_owned()),
            application_version: Version {
                major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
                minor: env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
                patch: env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
            },
            ..Default::default()
        }
    }
}

/// Error that can happen when creating an instance.
#[derive(Clone, Debug)]
pub enum InstanceCreationError {
    /// Failed to load the Vulkan shared library.
    LoadingError(LoadingError),
    /// Not enough memory.
    OomError(OomError),
    /// Failed to initialize for an implementation-specific reason.
    InitializationFailed,
    /// One of the requested layers is missing.
    LayerNotPresent,
    /// One of the requested extensions is not supported by the implementation.
    ExtensionNotPresent,
    /// The version requested is not supported by the implementation.
    IncompatibleDriver,
    /// A restriction for an extension was not met.
    ExtensionRestrictionNotMet(ExtensionRestrictionError),
    ExtensionNotEnabled {
        extension: &'static str,
        reason: &'static str,
    },
}

impl error::Error for InstanceCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Self::LoadingError(ref err) => Some(err),
            Self::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for InstanceCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Self::LoadingError(_) => {
                write!(fmt, "failed to load the Vulkan shared library")
            }
            Self::OomError(_) => write!(fmt, "not enough memory available"),
            Self::InitializationFailed => write!(fmt, "initialization failed"),
            Self::LayerNotPresent => write!(fmt, "layer not present"),
            Self::ExtensionNotPresent => write!(fmt, "extension not present"),
            Self::IncompatibleDriver => write!(fmt, "incompatible driver"),
            Self::ExtensionRestrictionNotMet(err) => err.fmt(fmt),
            Self::ExtensionNotEnabled { extension, reason } => write!(
                fmt,
                "the extension {} must be enabled: {}",
                extension, reason
            ),
        }
    }
}

impl From<OomError> for InstanceCreationError {
    #[inline]
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}

impl From<LoadingError> for InstanceCreationError {
    #[inline]
    fn from(err: LoadingError) -> Self {
        Self::LoadingError(err)
    }
}

impl From<ExtensionRestrictionError> for InstanceCreationError {
    #[inline]
    fn from(err: ExtensionRestrictionError) -> Self {
        Self::ExtensionRestrictionNotMet(err)
    }
}

impl From<Error> for InstanceCreationError {
    #[inline]
    fn from(err: Error) -> Self {
        match err {
            err @ Error::OutOfHostMemory => Self::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => Self::OomError(OomError::from(err)),
            Error::InitializationFailed => Self::InitializationFailed,
            Error::LayerNotPresent => Self::LayerNotPresent,
            Error::ExtensionNotPresent => Self::ExtensionNotPresent,
            Error::IncompatibleDriver => Self::IncompatibleDriver,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

// Same as Cow but less annoying.
#[derive(Debug)]
enum OwnedOrRef<T: 'static> {
    Owned(T),
    Ref(&'static T),
}

impl<T> Deref for OwnedOrRef<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        match *self {
            OwnedOrRef::Owned(ref v) => v,
            OwnedOrRef::Ref(v) => v,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::device::physical::PhysicalDevice;

    #[test]
    fn create_instance() {
        let _ = instance!();
    }

    #[test]
    fn queue_family_by_id() {
        let instance = instance!();

        let phys = match PhysicalDevice::enumerate(&instance).next() {
            Some(p) => p,
            None => return,
        };

        let queue_family = match phys.queue_families().next() {
            Some(q) => q,
            None => return,
        };

        let by_id = phys.queue_family_by_id(queue_family.id()).unwrap();
        assert_eq!(by_id.id(), queue_family.id());
    }
}
