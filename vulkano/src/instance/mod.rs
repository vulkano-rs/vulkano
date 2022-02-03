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
//! let instance = match Instance::start().build() {
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
//! # let instance = Instance::start().build().unwrap();
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

pub use self::extensions::InstanceExtensions;
pub use self::layers::layers_list;
pub use self::layers::LayerProperties;
pub use self::layers::LayersListError;
pub use self::loader::LoadingError;
use crate::check_errors;
use crate::device::physical::{init_physical_devices, PhysicalDeviceInfo};
pub use crate::extensions::{
    ExtensionRestriction, ExtensionRestrictionError, SupportedExtensionsError,
};
use crate::fns::InstanceFunctions;
use crate::instance::loader::FunctionPointers;
use crate::instance::loader::Loader;
pub use crate::version::Version;
use crate::Error;
use crate::OomError;
use crate::VulkanObject;
use smallvec::SmallVec;
use std::error;
use std::ffi::CString;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::ptr;
use std::slice;
use std::sync::Arc;

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
/// use vulkano::instance::{Instance, InstanceExtensions};
/// use vulkano::Version;
///
/// let _instance = Instance::start()
///     .application_from_cargo_toml()
///     .build().unwrap();
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
/// use vulkano::instance::Instance;
/// use vulkano::instance::InstanceExtensions;
/// use vulkano::Version;
///
/// let extensions = InstanceExtensions {
///     khr_surface: true,
///     khr_android_surface: true,
///     .. InstanceExtensions::none()
/// };
///
/// let instance = match Instance::start()
///     .enabled_extensions(extensions)
///     .build() {
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
/// # use vulkano::instance::InstanceExtensions;
/// # use vulkano::Version;
/// # fn test() -> Result<Arc<Instance>, Box<dyn Error>> {
/// // For the sake of the example, we activate all the layers that
/// // contain the word "foo" in their description.
/// let layers: Vec<_> = instance::layers_list()?
///     .filter(|l| l.description().contains("foo"))
///     .collect();
///
/// let layer_names = layers.iter()
///     .map(|l| l.name());
///
/// let instance = Instance::start()
///     .enabled_layers(layer_names)
///     .build()?;
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
    enabled_layers: Vec<CString>,
    function_pointers: OwnedOrRef<FunctionPointers<Box<dyn Loader + Send + Sync>>>,
    max_api_version: Version,
}

// TODO: fix the underlying cause instead
impl ::std::panic::UnwindSafe for Instance {}
impl ::std::panic::RefUnwindSafe for Instance {}

impl Instance {
    /// Starts constructing a new `Instance`.
    #[inline]
    pub fn start() -> InstanceBuilder {
        InstanceBuilder {
            application_name: None,
            application_version: Version::major_minor(0, 0),
            enabled_extensions: InstanceExtensions::none(),
            enabled_layers: Vec::new(),
            engine_name: None,
            engine_version: Version::major_minor(0, 0),
            function_pointers: None,
            max_api_version: None,
        }
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
    #[doc(hidden)]
    #[inline]
    pub fn enabled_layers(&self) -> slice::Iter<CString> {
        self.enabled_layers.iter()
    }
}

impl fmt::Debug for Instance {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan instance {:?}>", self.handle)
    }
}

unsafe impl VulkanObject for Instance {
    type Object = ash::vk::Instance;

    #[inline]
    fn internal_object(&self) -> ash::vk::Instance {
        self.handle
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

/// Used to construct a new `Instance`.
pub struct InstanceBuilder {
    application_name: Option<CString>,
    application_version: Version,
    enabled_extensions: InstanceExtensions,
    enabled_layers: Vec<CString>,
    engine_name: Option<CString>,
    engine_version: Version,
    function_pointers: Option<FunctionPointers<Box<dyn Loader + Send + Sync>>>,
    max_api_version: Option<Version>,
}

impl InstanceBuilder {
    /// Creates the `Instance`.
    pub fn build(self) -> Result<Arc<Instance>, InstanceCreationError> {
        let Self {
            application_name,
            application_version,
            enabled_extensions,
            enabled_layers,
            engine_name,
            engine_version,
            function_pointers,
            max_api_version,
        } = self;

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
                Version::V1_2 // TODO: Can this be extracted from vk.xml somehow?
            };

            (std::cmp::min(max_api_version, api_version), max_api_version)
        };

        // Check if the extensions are correct
        enabled_extensions.check_requirements(
            &InstanceExtensions::supported_by_core_with_loader(&function_pointers)?,
            api_version,
        )?;

        // FIXME: check whether each layer is supported
        let enabled_layers_ptrs = enabled_layers
            .iter()
            .map(|layer| layer.as_ptr())
            .collect::<SmallVec<[_; 2]>>();

        let enabled_extensions_list = Vec::<CString>::from(&enabled_extensions);
        let enabled_extensions_ptrs = enabled_extensions_list
            .iter()
            .map(|extension| extension.as_ptr())
            .collect::<SmallVec<[_; 2]>>();

        let application_info = ash::vk::ApplicationInfo {
            p_application_name: application_name
                .as_ref()
                .map(|s| s.as_ptr())
                .unwrap_or(ptr::null()),
            application_version: application_version
                .try_into()
                .expect("Version out of range"),
            p_engine_name: engine_name
                .as_ref()
                .map(|s| s.as_ptr())
                .unwrap_or(ptr::null()),
            engine_version: engine_version.try_into().expect("Version out of range"),
            api_version: max_api_version.try_into().expect("Version out of range"),
            ..Default::default()
        };

        let create_info = ash::vk::InstanceCreateInfo {
            flags: ash::vk::InstanceCreateFlags::empty(),
            p_application_info: &application_info,
            enabled_layer_count: enabled_layers_ptrs.len() as u32,
            pp_enabled_layer_names: enabled_layers_ptrs.as_ptr(),
            enabled_extension_count: enabled_extensions_ptrs.len() as u32,
            pp_enabled_extension_names: enabled_extensions_ptrs.as_ptr(),
            ..Default::default()
        };

        // Creating the Vulkan instance.
        let handle = unsafe {
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
        };

        // Enumerating all physical devices.
        instance.physical_device_infos = init_physical_devices(&instance)?;

        Ok(Arc::new(instance))
    }

    /// Sets the `application_name` and `application_version` from information in your
    /// crate's Cargo.toml file.
    ///
    /// # Panics
    ///
    /// - Panics if the required environment variables are missing, which happens if the project
    ///   wasn't built by Cargo.
    #[inline]
    pub fn application_from_cargo_toml(mut self) -> Self {
        self.application_name = Some(CString::new(env!("CARGO_PKG_NAME")).unwrap());
        self.application_version = Version {
            major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
            minor: env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
            patch: env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
        };
        self
    }

    /// A string of your choice stating the name of your application.
    ///
    /// # Panics
    ///
    /// - Panics if `name` contains a NUL character.
    #[inline]
    pub fn application_name(mut self, name: impl Into<Vec<u8>>) -> Self {
        self.application_name = Some(CString::new(name).unwrap());
        self
    }

    /// A version number of your choice specifying the version of your application.
    ///
    /// The default value is zero.
    ///
    /// # Panics
    ///
    /// - Panics if `version` contains a field too large to be converted into a Vulkan version
    ///   number.
    #[inline]
    pub fn application_version(mut self, version: Version) -> Self {
        assert!(u32::try_from(version).is_ok());
        self.application_version = version;
        self
    }

    /// A string of your choice stating the name of the engine used to power the application.
    ///
    /// # Panics
    ///
    /// - Panics if `name` contains a NUL character.
    #[inline]
    pub fn engine_name(mut self, name: impl Into<Vec<u8>>) -> Self {
        self.engine_name = Some(CString::new(name).unwrap());
        self
    }

    /// A version number of your choice specifying the version of the engine used to power the
    /// application.
    ///
    /// The default value is zero.
    ///
    /// # Panics
    ///
    /// - Panics if `version` contains a field too large to be converted into a Vulkan version
    ///   number.
    #[inline]
    pub fn engine_version(mut self, version: Version) -> Self {
        assert!(u32::try_from(version).is_ok());
        self.engine_version = version;
        self
    }

    /// The extensions to enable on the instance.
    ///
    /// The default value is [`InstanceExtensions::none()`].
    #[inline]
    pub fn enabled_extensions(mut self, extensions: InstanceExtensions) -> Self {
        self.enabled_extensions = extensions;
        self
    }

    /// The layers to enable on the instance.
    ///
    /// The default value is empty.
    ///
    /// # Panics
    ///
    /// - Panics if an element of `layers` contains a NUL character.
    #[inline]
    pub fn enabled_layers<L>(mut self, layers: L) -> Self
    where
        L: IntoIterator,
        L::Item: Into<Vec<u8>>,
    {
        self.enabled_layers = layers
            .into_iter()
            .map(|layer| CString::new(layer).unwrap())
            .collect();
        self
    }

    /// Function pointers loaded from a custom loader.
    ///
    /// You can use this if you want to load the Vulkan API explicitly, rather than using Vulkano's
    /// default.
    #[inline]
    pub fn function_pointers(
        mut self,
        function_pointers: FunctionPointers<Box<dyn Loader + Send + Sync>>,
    ) -> Self {
        self.function_pointers = Some(function_pointers);
        self
    }

    /// The highest Vulkan API version that the application will use with the instance.
    ///
    /// Usually, you will want to leave this at the default.
    ///
    /// The default value is the highest version currently supported by Vulkano, but if the
    /// supported instance version is 1.0, then it will be 1.0.
    ///
    /// # Panics
    ///
    /// - Panics if `version` is not at least `V1_0`.
    #[inline]
    pub fn max_api_version(mut self, version: Version) -> Self {
        assert!(version >= Version::V1_0);
        self.max_api_version = Some(version);
        self
    }
}

// Same as Cow but less annoying.
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
}

impl error::Error for InstanceCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            InstanceCreationError::LoadingError(ref err) => Some(err),
            InstanceCreationError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for InstanceCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            InstanceCreationError::LoadingError(_) => {
                write!(fmt, "failed to load the Vulkan shared library")
            }
            InstanceCreationError::OomError(_) => write!(fmt, "not enough memory available"),
            InstanceCreationError::InitializationFailed => write!(fmt, "initialization failed"),
            InstanceCreationError::LayerNotPresent => write!(fmt, "layer not present"),
            InstanceCreationError::ExtensionNotPresent => write!(fmt, "extension not present"),
            InstanceCreationError::IncompatibleDriver => write!(fmt, "incompatible driver"),
            InstanceCreationError::ExtensionRestrictionNotMet(err) => err.fmt(fmt),
        }
    }
}

impl From<OomError> for InstanceCreationError {
    #[inline]
    fn from(err: OomError) -> InstanceCreationError {
        InstanceCreationError::OomError(err)
    }
}

impl From<LoadingError> for InstanceCreationError {
    #[inline]
    fn from(err: LoadingError) -> InstanceCreationError {
        InstanceCreationError::LoadingError(err)
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
    fn from(err: Error) -> InstanceCreationError {
        match err {
            err @ Error::OutOfHostMemory => InstanceCreationError::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => InstanceCreationError::OomError(OomError::from(err)),
            Error::InitializationFailed => InstanceCreationError::InitializationFailed,
            Error::LayerNotPresent => InstanceCreationError::LayerNotPresent,
            Error::ExtensionNotPresent => InstanceCreationError::ExtensionNotPresent,
            Error::IncompatibleDriver => InstanceCreationError::IncompatibleDriver,
            _ => panic!("unexpected error: {:?}", err),
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
