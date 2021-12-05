// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::check_errors;
use crate::device::physical::{init_physical_devices, PhysicalDeviceInfo};
use crate::extensions::ExtensionRestrictionError;
use crate::fns::InstanceFunctions;
use crate::instance::loader;
use crate::instance::loader::FunctionPointers;
use crate::instance::loader::Loader;
use crate::instance::loader::LoadingError;
use crate::instance::InstanceExtensions;
use crate::Error;
use crate::OomError;
use crate::Version;
use crate::VulkanObject;
use smallvec::SmallVec;
use std::borrow::Cow;
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

/// An instance of a Vulkan context. This is the main object that should be created by an
/// application before everything else.
///
/// # Application info
///
/// When you create an instance, you have the possibility to pass an `ApplicationInfo` struct as
/// the first parameter. This struct contains various information about your application, most
/// notably its name and engine.
///
/// Passing such a structure allows for example the driver to let the user configure the driver's
/// behavior for your application alone through a control panel.
///
/// ```no_run
/// # #[macro_use] extern crate vulkano;
/// # fn main() {
/// use vulkano::instance::{Instance, InstanceExtensions};
/// use vulkano::Version;
///
/// // Builds an `ApplicationInfo` by looking at the content of the `Cargo.toml` file at
/// // compile-time.
/// let app_infos = app_info_from_cargo_toml!();
///
/// let _instance = Instance::new(Some(&app_infos), Version::V1_1, &InstanceExtensions::none(), None).unwrap();
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
/// However, due to a quirk in how the Vulkan 1.0 specification was written, if the instance only
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
/// let instance = match Instance::new(None, Version::V1_1, &extensions, None) {
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
/// let instance = Instance::new(None, Version::V1_1, &InstanceExtensions::none(), layer_names)?;
/// # Ok(instance)
/// # }
/// ```
// TODO: mention that extensions must be supported by layers as well
pub struct Instance {
    instance: ash::vk::Instance,
    //alloc: Option<Box<Alloc + Send + Sync>>,

    // The highest version that is supported for this instance.
    // This is the minimum of Instance::max_api_version and FunctionPointers::api_version.
    api_version: Version,

    // The highest allowed API version for instances and devices created from it.
    max_api_version: Version,

    pub(crate) physical_device_infos: Vec<PhysicalDeviceInfo>,
    fns: InstanceFunctions,
    extensions: InstanceExtensions,
    layers: SmallVec<[CString; 16]>,
    function_pointers: OwnedOrRef<FunctionPointers<Box<dyn Loader + Send + Sync>>>,
}

// TODO: fix the underlying cause instead
impl ::std::panic::UnwindSafe for Instance {}
impl ::std::panic::RefUnwindSafe for Instance {}

impl Instance {
    /// Initializes a new instance of Vulkan.
    ///
    /// See the documentation of `Instance` or of [the `instance` module](crate::instance) for more
    /// details.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use vulkano::instance::Instance;
    /// use vulkano::instance::InstanceExtensions;
    /// use vulkano::Version;
    ///
    /// let instance = match Instance::new(None, Version::V1_1, &InstanceExtensions::none(), None) {
    ///     Ok(i) => i,
    ///     Err(err) => panic!("Couldn't build instance: {:?}", err)
    /// };
    /// ```
    ///
    /// # Panic
    ///
    /// - Panics if the version numbers passed in `ApplicationInfo` are too large can't be
    ///   converted into a Vulkan version number.
    /// - Panics if the application name or engine name contain a null character.
    // TODO: add a test for these ^
    // TODO: if no allocator is specified by the user, use Rust's allocator instead of leaving
    //       the choice to Vulkan
    pub fn new<'a, L>(
        app_infos: Option<&ApplicationInfo>,
        max_api_version: Version,
        extensions: &InstanceExtensions,
        layers: L,
    ) -> Result<Arc<Instance>, InstanceCreationError>
    where
        L: IntoIterator<Item = &'a str>,
    {
        let layers = layers
            .into_iter()
            .map(|layer| CString::new(layer).unwrap())
            .collect::<SmallVec<[_; 16]>>();

        Instance::new_inner(
            app_infos,
            max_api_version,
            extensions,
            layers,
            OwnedOrRef::Ref(loader::auto_loader()?),
        )
    }

    /// Same as `new`, but allows specifying a loader where to load Vulkan from.
    pub fn with_loader<'a, L>(
        loader: FunctionPointers<Box<dyn Loader + Send + Sync>>,
        app_infos: Option<&ApplicationInfo>,
        max_api_version: Version,
        extensions: &InstanceExtensions,
        layers: L,
    ) -> Result<Arc<Instance>, InstanceCreationError>
    where
        L: IntoIterator<Item = &'a str>,
    {
        let layers = layers
            .into_iter()
            .map(|layer| CString::new(layer).unwrap())
            .collect::<SmallVec<[_; 16]>>();

        Instance::new_inner(
            app_infos,
            max_api_version,
            extensions,
            layers,
            OwnedOrRef::Owned(loader),
        )
    }

    fn new_inner(
        app_infos: Option<&ApplicationInfo>,
        max_api_version: Version,
        extensions: &InstanceExtensions,
        layers: SmallVec<[CString; 16]>,
        function_pointers: OwnedOrRef<FunctionPointers<Box<dyn Loader + Send + Sync>>>,
    ) -> Result<Arc<Instance>, InstanceCreationError> {
        let api_version = std::cmp::min(max_api_version, function_pointers.api_version()?);

        // Check if the extensions are correct
        extensions.check_requirements(
            &InstanceExtensions::supported_by_core_with_loader(&function_pointers)?,
            api_version,
        )?;

        // TODO: For now there are still buggy drivers that will segfault if you don't pass any
        //       appinfos. Therefore for now we ensure that it can't be `None`.
        let def = Default::default();
        let app_infos = match app_infos {
            Some(a) => Some(a),
            None => Some(&def),
        };

        // Building the CStrings from the `str`s within `app_infos`.
        // They need to be created ahead of time, since we pass pointers to them.
        let app_infos_strings = if let Some(app_infos) = app_infos {
            Some((
                app_infos
                    .application_name
                    .clone()
                    .map(|n| CString::new(n.as_bytes().to_owned()).unwrap()),
                app_infos
                    .engine_name
                    .clone()
                    .map(|n| CString::new(n.as_bytes().to_owned()).unwrap()),
            ))
        } else {
            None
        };

        // Building the `vk::ApplicationInfo` if required.
        let app_infos = if let Some(app_infos) = app_infos {
            Some(ash::vk::ApplicationInfo {
                p_application_name: app_infos_strings
                    .as_ref()
                    .unwrap()
                    .0
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(ptr::null()),
                application_version: app_infos
                    .application_version
                    .map(|v| v.try_into().expect("Version out of range"))
                    .unwrap_or(0),
                p_engine_name: app_infos_strings
                    .as_ref()
                    .unwrap()
                    .1
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(ptr::null()),
                engine_version: app_infos
                    .engine_version
                    .map(|v| v.try_into().expect("Version out of range"))
                    .unwrap_or(0),
                api_version: max_api_version.try_into().expect("Version out of range"),
                ..Default::default()
            })
        } else {
            None
        };

        // FIXME: check whether each layer is supported
        let layers_ptrs = layers
            .iter()
            .map(|layer| layer.as_ptr())
            .collect::<SmallVec<[_; 16]>>();

        let extensions_list: Vec<CString> = extensions.into();
        let extensions_ptrs = extensions_list
            .iter()
            .map(|extension| extension.as_ptr())
            .collect::<SmallVec<[_; 32]>>();

        // Creating the Vulkan instance.
        let instance = unsafe {
            let mut output = MaybeUninit::uninit();
            let infos = ash::vk::InstanceCreateInfo {
                flags: ash::vk::InstanceCreateFlags::empty(),
                p_application_info: if let Some(app) = app_infos.as_ref() {
                    app as *const _
                } else {
                    ptr::null()
                },
                enabled_layer_count: layers_ptrs.len() as u32,
                pp_enabled_layer_names: layers_ptrs.as_ptr(),
                enabled_extension_count: extensions_ptrs.len() as u32,
                pp_enabled_extension_names: extensions_ptrs.as_ptr(),
                ..Default::default()
            };

            let fns = function_pointers.fns();
            check_errors(
                fns.v1_0
                    .create_instance(&infos, ptr::null(), output.as_mut_ptr()),
            )?;
            output.assume_init()
        };

        // Loading the function pointers of the newly-created instance.
        let fns = {
            InstanceFunctions::load(|name| {
                function_pointers.get_instance_proc_addr(instance, name.as_ptr())
            })
        };

        let mut instance = Instance {
            instance,
            api_version,
            max_api_version,
            //alloc: None,
            physical_device_infos: Vec::new(),
            fns,
            extensions: extensions.clone(),
            layers,
            function_pointers,
        };

        // Enumerating all physical devices.
        instance.physical_device_infos = init_physical_devices(&instance)?;

        Ok(Arc::new(instance))
    }

    /*/// Same as `new`, but provides an allocator that will be used by the Vulkan library whenever
    /// it needs to allocate memory on the host.
    ///
    /// Note that this allocator can be overridden when you create a `Device`, a `MemoryPool`, etc.
    pub fn with_alloc(app_infos: Option<&ApplicationInfo>, alloc: Box<Alloc + Send + Sync>) -> Arc<Instance> {
        unimplemented!()
    }*/

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
        &self.extensions
    }

    /// Returns the layers that have been enabled on the instance.
    #[doc(hidden)]
    #[inline]
    pub fn enabled_layers(&self) -> slice::Iter<CString> {
        self.layers.iter()
    }
}

impl fmt::Debug for Instance {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan instance {:?}>", self.instance)
    }
}

unsafe impl VulkanObject for Instance {
    type Object = ash::vk::Instance;

    #[inline]
    fn internal_object(&self) -> ash::vk::Instance {
        self.instance
    }
}

impl Drop for Instance {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.fns.v1_0.destroy_instance(self.instance, ptr::null());
        }
    }
}

impl PartialEq for Instance {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.instance == other.instance
    }
}

impl Eq for Instance {}

impl Hash for Instance {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.instance.hash(state);
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

/// Information that can be given to the Vulkan driver so that it can identify your application.
// TODO: better documentation for struct and methods
#[derive(Debug, Clone)]
pub struct ApplicationInfo<'a> {
    /// Name of the application.
    pub application_name: Option<Cow<'a, str>>,
    /// An opaque number that contains the version number of the application.
    pub application_version: Option<Version>,
    /// Name of the engine used to power the application.
    pub engine_name: Option<Cow<'a, str>>,
    /// An opaque number that contains the version number of the engine.
    pub engine_version: Option<Version>,
}

impl<'a> ApplicationInfo<'a> {
    /// Builds an `ApplicationInfo` from the information gathered by Cargo.
    ///
    /// # Panic
    ///
    /// - Panics if the required environment variables are missing, which happens if the project
    ///   wasn't built by Cargo.
    ///
    #[deprecated(note = "Please use the `app_info_from_cargo_toml!` macro instead")]
    pub fn from_cargo_toml() -> ApplicationInfo<'a> {
        let version = Version {
            major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
            minor: env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
            patch: env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
        };

        let name = env!("CARGO_PKG_NAME");

        ApplicationInfo {
            application_name: Some(name.into()),
            application_version: Some(version),
            engine_name: None,
            engine_version: None,
        }
    }
}

/// Builds an `ApplicationInfo` from the information gathered by Cargo.
///
/// # Panic
///
/// - Panics if the required environment variables are missing, which happens if the project
///   wasn't built by Cargo.
///
#[macro_export]
macro_rules! app_info_from_cargo_toml {
    () => {{
        let version = $crate::instance::Version {
            major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
            minor: env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
            patch: env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
        };

        let name = env!("CARGO_PKG_NAME");

        $crate::instance::ApplicationInfo {
            application_name: Some(name.into()),
            application_version: Some(version),
            engine_name: None,
            engine_version: None,
        }
    }};
}

impl<'a> Default for ApplicationInfo<'a> {
    fn default() -> ApplicationInfo<'a> {
        ApplicationInfo {
            application_name: None,
            application_version: None,
            engine_name: None,
            engine_version: None,
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
