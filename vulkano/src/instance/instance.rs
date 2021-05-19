// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::check_errors;
use crate::instance::loader;
use crate::instance::loader::FunctionPointers;
use crate::instance::loader::Loader;
use crate::instance::loader::LoadingError;
use crate::instance::physical_device::{init_physical_devices, PhysicalDeviceInfos};
use crate::instance::{InstanceExtensions, RawInstanceExtensions};
use crate::vk;
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
/// let _instance = Instance::new(Some(&app_infos), Version::major_minor(1, 1), &InstanceExtensions::none(), None).unwrap();
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
/// [`PhysicalDevice::api_version`](crate::instance::PhysicalDevice::api_version).
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
/// let instance = match Instance::new(None, Version::major_minor(1, 1), &extensions, None) {
///     Ok(i) => i,
///     Err(err) => panic!("Couldn't build instance: {:?}", err)
/// };
/// ```
///
/// # Layers
///
/// When creating an `Instance`, you have the possibility to pass a list of **layers** that will
/// be activated on the newly-created instance. The list of available layers can be retrieved by
/// calling [the `layers_list` function](fn.layers_list.html).
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
/// let instance = Instance::new(None, Version::major_minor(1, 1), &InstanceExtensions::none(), layer_names)?;
/// # Ok(instance)
/// # }
/// ```
// TODO: mention that extensions must be supported by layers as well
pub struct Instance {
    instance: vk::Instance,
    //alloc: Option<Box<Alloc + Send + Sync>>,

    // The highest version that is supported for this instance.
    // This is the minimum of Instance::max_api_version and FunctionPointers::api_version.
    api_version: Version,

    // The highest allowed API version for instances and devices created from it.
    max_api_version: Version,

    pub(super) physical_devices: Vec<PhysicalDeviceInfos>,
    vk: vk::InstancePointers,
    extensions: RawInstanceExtensions,
    layers: SmallVec<[CString; 16]>,
    function_pointers: OwnedOrRef<FunctionPointers<Box<dyn Loader + Send + Sync>>>,
}

// TODO: fix the underlying cause instead
impl ::std::panic::UnwindSafe for Instance {}
impl ::std::panic::RefUnwindSafe for Instance {}

impl Instance {
    /// Initializes a new instance of Vulkan.
    ///
    /// See the documentation of `Instance` or of [the `instance` module](index.html) for more
    /// details.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use vulkano::instance::Instance;
    /// use vulkano::instance::InstanceExtensions;
    /// use vulkano::Version;
    ///
    /// let instance = match Instance::new(None, Version::major_minor(1, 1), &InstanceExtensions::none(), None) {
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
    pub fn new<'a, L, Ext>(
        app_infos: Option<&ApplicationInfo>,
        max_api_version: Version,
        extensions: Ext,
        layers: L,
    ) -> Result<Arc<Instance>, InstanceCreationError>
    where
        L: IntoIterator<Item = &'a str>,
        Ext: Into<RawInstanceExtensions>,
    {
        let layers = layers
            .into_iter()
            .map(|layer| CString::new(layer).unwrap())
            .collect::<SmallVec<[_; 16]>>();

        Instance::new_inner(
            app_infos,
            max_api_version,
            extensions.into(),
            layers,
            OwnedOrRef::Ref(loader::auto_loader()?),
        )
    }

    /// Same as `new`, but allows specifying a loader where to load Vulkan from.
    pub fn with_loader<'a, L, Ext>(
        loader: FunctionPointers<Box<dyn Loader + Send + Sync>>,
        app_infos: Option<&ApplicationInfo>,
        max_api_version: Version,
        extensions: Ext,
        layers: L,
    ) -> Result<Arc<Instance>, InstanceCreationError>
    where
        L: IntoIterator<Item = &'a str>,
        Ext: Into<RawInstanceExtensions>,
    {
        let layers = layers
            .into_iter()
            .map(|layer| CString::new(layer).unwrap())
            .collect::<SmallVec<[_; 16]>>();

        Instance::new_inner(
            app_infos,
            max_api_version,
            extensions.into(),
            layers,
            OwnedOrRef::Owned(loader),
        )
    }

    fn new_inner(
        app_infos: Option<&ApplicationInfo>,
        max_api_version: Version,
        extensions: RawInstanceExtensions,
        layers: SmallVec<[CString; 16]>,
        function_pointers: OwnedOrRef<FunctionPointers<Box<dyn Loader + Send + Sync>>>,
    ) -> Result<Arc<Instance>, InstanceCreationError> {
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

        let api_version = std::cmp::min(max_api_version, function_pointers.api_version()?);

        // Building the `vk::ApplicationInfo` if required.
        let app_infos = if let Some(app_infos) = app_infos {
            Some(vk::ApplicationInfo {
                sType: vk::STRUCTURE_TYPE_APPLICATION_INFO,
                pNext: ptr::null(),
                pApplicationName: app_infos_strings
                    .as_ref()
                    .unwrap()
                    .0
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(ptr::null()),
                applicationVersion: app_infos
                    .application_version
                    .map(|v| v.into_vulkan_version())
                    .unwrap_or(0),
                pEngineName: app_infos_strings
                    .as_ref()
                    .unwrap()
                    .1
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(ptr::null()),
                engineVersion: app_infos
                    .engine_version
                    .map(|v| v.into_vulkan_version())
                    .unwrap_or(0),
                apiVersion: max_api_version.into_vulkan_version(),
            })
        } else {
            None
        };

        // FIXME: check whether each layer is supported
        let layers_ptr = layers
            .iter()
            .map(|layer| layer.as_ptr())
            .collect::<SmallVec<[_; 16]>>();

        let extensions_list = extensions
            .iter()
            .map(|extension| extension.as_ptr())
            .collect::<SmallVec<[_; 32]>>();

        // Creating the Vulkan instance.
        let instance = unsafe {
            let mut output = MaybeUninit::uninit();
            let infos = vk::InstanceCreateInfo {
                sType: vk::STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                pApplicationInfo: if let Some(app) = app_infos.as_ref() {
                    app as *const _
                } else {
                    ptr::null()
                },
                enabledLayerCount: layers_ptr.len() as u32,
                ppEnabledLayerNames: layers_ptr.as_ptr(),
                enabledExtensionCount: extensions_list.len() as u32,
                ppEnabledExtensionNames: extensions_list.as_ptr(),
            };

            let entry_points = function_pointers.entry_points();
            check_errors(entry_points.CreateInstance(&infos, ptr::null(), output.as_mut_ptr()))?;
            output.assume_init()
        };

        // Loading the function pointers of the newly-created instance.
        let vk = {
            vk::InstancePointers::load(|name| {
                function_pointers.get_instance_proc_addr(instance, name.as_ptr())
            })
        };

        let mut instance = Instance {
            instance,
            api_version,
            max_api_version,
            //alloc: None,
            physical_devices: Vec::new(),
            vk,
            extensions,
            layers,
            function_pointers,
        };

        // Enumerating all physical devices.
        instance.physical_devices = init_physical_devices(&instance)?;

        Ok(Arc::new(instance))
    }

    /*/// Same as `new`, but provides an allocator that will be used by the Vulkan library whenever
    /// it needs to allocate memory on the host.
    ///
    /// Note that this allocator can be overridden when you create a `Device`, a `MemoryPool`, etc.
    pub fn with_alloc(app_infos: Option<&ApplicationInfo>, alloc: Box<Alloc + Send + Sync>) -> Arc<Instance> {
        unimplemented!()
    }*/

    /// Returns the Vulkan version supported by this `Instance`.
    ///
    /// This is the lower of the
    /// [driver's supported version](crate::instance::loader::FunctionPointers::api_version) and
    /// [`max_api_version`](Instance::max_api_version).
    #[inline]
    pub fn api_version(&self) -> Version {
        self.api_version
    }

    /// Returns the maximum Vulkan version that was specified when creating this `Instance`.
    #[inline]
    pub fn max_api_version(&self) -> Version {
        self.max_api_version
    }

    /// Grants access to the Vulkan functions of the instance.
    #[inline]
    pub fn pointers(&self) -> &vk::InstancePointers {
        &self.vk
    }

    /// Returns the list of extensions that have been loaded.
    ///
    /// This list is equal to what was passed to `Instance::new()`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use vulkano::instance::Instance;
    /// use vulkano::instance::InstanceExtensions;
    /// use vulkano::Version;
    ///
    /// let extensions = InstanceExtensions::supported_by_core().unwrap();
    /// let instance = Instance::new(None, Version::major_minor(1, 1), &extensions, None).unwrap();
    /// assert_eq!(instance.loaded_extensions(), extensions);
    /// ```
    #[inline]
    pub fn loaded_extensions(&self) -> InstanceExtensions {
        InstanceExtensions::from(&self.extensions)
    }

    #[inline]
    pub fn raw_loaded_extensions(&self) -> &RawInstanceExtensions {
        &self.extensions
    }

    /// Returns the list of layers requested when creating this instance.
    #[doc(hidden)]
    #[inline]
    pub fn loaded_layers(&self) -> slice::Iter<CString> {
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
    type Object = vk::Instance;

    const TYPE: vk::ObjectType = vk::OBJECT_TYPE_INSTANCE;

    #[inline]
    fn internal_object(&self) -> vk::Instance {
        self.instance
    }
}

impl Drop for Instance {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.vk.DestroyInstance(self.instance, ptr::null());
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
    /// One of the requested extensions is missing.
    ExtensionNotPresent,
    /// The version requested is not supported by the implementation.
    // TODO: more info about this once the question of the version has been resolved
    IncompatibleDriver,
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
        write!(
            fmt,
            "{}",
            match *self {
                InstanceCreationError::LoadingError(_) =>
                    "failed to load the Vulkan shared library",
                InstanceCreationError::OomError(_) => "not enough memory available",
                InstanceCreationError::InitializationFailed => "initialization failed",
                InstanceCreationError::LayerNotPresent => "layer not present",
                InstanceCreationError::ExtensionNotPresent => "extension not present",
                InstanceCreationError::IncompatibleDriver => "incompatible driver",
            }
        )
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
    use crate::instance;

    #[test]
    fn create_instance() {
        let _ = instance!();
    }

    #[test]
    fn queue_family_by_id() {
        let instance = instance!();

        let phys = match instance::PhysicalDevice::enumerate(&instance).next() {
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
