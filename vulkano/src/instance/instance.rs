// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use smallvec::SmallVec;
use std::borrow::Cow;
use std::error;
use std::ffi::CStr;
use std::ffi::CString;
use std::fmt;
use std::mem;
use std::ops::Deref;
use std::ptr;
use std::slice;
use std::sync::Arc;

use Error;
use OomError;
use VulkanObject;
use check_errors;
use instance::loader;
use instance::loader::FunctionPointers;
use instance::loader::Loader;
use instance::loader::LoadingError;
use vk;

use features::Features;
use instance::{InstanceExtensions, RawInstanceExtensions};
use version::Version;

/// An instance of a Vulkan context. This is the main object that should be created by an
/// application before everything else.
///
/// See the documentation of [the `instance` module](index.html) for an introduction about
/// Vulkan instances.
///
/// # Extensions and application infos
///
/// Please check the documentation of [the `instance` module](index.html).
///
/// # Layers
///
/// When creating an `Instance`, you have the possibility to pass a list of **layers** that will
/// be activated on the newly-created instance. The list of available layers can be retrieved by
/// calling [the `layers_list` function](fn.layers_list.html).
///
/// A layer is a component that will hook and potentially modify the Vulkan function calls.
/// For example, activating a layer could add a frames-per-second counter on the screen, or it
/// could send informations to a debugger that will debug your application.
///
/// > **Note**: From an application's point of view, layers "just exist". In practice, on Windows
/// > and Linux layers can be installed by third party installers or by package managers and can
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
/// ```ignore
/// // FIXME: this example doesn't run because of ownership problems ; Instance::new() needs a tweak
/// use vulkano::instance;
/// use vulkano::instance::Instance;
/// use vulkano::instance::InstanceExtensions;
///
/// // For the sake of the example, we activate all the layers that contain the word "foo" in their
/// // description.
/// let layers = instance::layers_list().unwrap()
///     .filter(|l| l.description().contains("foo"))
///     .map(|l| l.name());
///
/// let instance = Instance::new(None, &InstanceExtensions::none(), layers).unwrap();
/// ```
// TODO: mention that extensions must be supported by layers as well
pub struct Instance {
    instance: vk::Instance,
    //alloc: Option<Box<Alloc + Send + Sync>>,
    physical_devices: Vec<PhysicalDeviceInfos>,
    vk: vk::InstancePointers,
    extensions: InstanceExtensions,
    layers: SmallVec<[CString; 16]>,
    function_pointers: OwnedOrRef<FunctionPointers<Box<Loader + Send + Sync>>>,
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
    ///
    /// let instance = match Instance::new(None, &InstanceExtensions::none(), None) {
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
    pub fn new<'a, L, Ext>(app_infos: Option<&ApplicationInfo>, extensions: Ext, layers: L)
                           -> Result<Arc<Instance>, InstanceCreationError>
        where L: IntoIterator<Item = &'a &'a str>,
              Ext: Into<RawInstanceExtensions>
    {
        let layers = layers
            .into_iter()
            .map(|&layer| CString::new(layer).unwrap())
            .collect::<SmallVec<[_; 16]>>();

        Instance::new_inner(app_infos, extensions.into(), layers,
                            OwnedOrRef::Ref(loader::auto_loader()?))
    }

    /// Same as `new`, but allows specifying a loader where to load Vulkan from.
    pub fn with_loader<'a, L, Ext>(loader: FunctionPointers<Box<Loader + Send + Sync>>,
                                   app_infos: Option<&ApplicationInfo>, extensions: Ext,
                                   layers: L) -> Result<Arc<Instance>, InstanceCreationError>
        where L: IntoIterator<Item = &'a &'a str>,
              Ext: Into<RawInstanceExtensions>
    {
        let layers = layers
            .into_iter()
            .map(|&layer| CString::new(layer).unwrap())
            .collect::<SmallVec<[_; 16]>>();

        Instance::new_inner(app_infos, extensions.into(), layers,
                            OwnedOrRef::Owned(loader))
    }

    fn new_inner(app_infos: Option<&ApplicationInfo>, extensions: RawInstanceExtensions,
                 layers: SmallVec<[CString; 16]>,
                 function_pointers: OwnedOrRef<FunctionPointers<Box<Loader + Send + Sync>>>)
                 -> Result<Arc<Instance>, InstanceCreationError> {
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
            Some((app_infos
                      .application_name
                      .clone()
                      .map(|n| CString::new(n.as_bytes().to_owned()).unwrap()),
                  app_infos
                      .engine_name
                      .clone()
                      .map(|n| CString::new(n.as_bytes().to_owned()).unwrap())))
        } else {
            None
        };

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
                apiVersion: Version {
                    major: 1,
                    minor: 0,
                    patch: 0,
                }.into_vulkan_version(), // TODO:
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
            let mut output = mem::uninitialized();
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
            check_errors(entry_points.CreateInstance(&infos, ptr::null(), &mut output))?;
            output
        };

        // Loading the function pointers of the newly-created instance.
        let vk = {
            vk::InstancePointers::load(|name| unsafe {
                mem::transmute(function_pointers.get_instance_proc_addr(instance, name.as_ptr()))
            })
        };

        // Enumerating all physical devices.
        let physical_devices: Vec<vk::PhysicalDevice> = unsafe {
            let mut num = 0;
            check_errors(vk.EnumeratePhysicalDevices(instance, &mut num, ptr::null_mut()))?;

            let mut devices = Vec::with_capacity(num as usize);
            check_errors(vk.EnumeratePhysicalDevices(instance, &mut num, devices.as_mut_ptr()))?;
            devices.set_len(num as usize);
            devices
        };

        // TODO: should be Into
        let extensions: InstanceExtensions = (&extensions).into();

        // Getting the properties of all physical devices.
        // If possible, we use VK_KHR_get_physical_device_properties2.
        let physical_devices = if extensions.khr_get_physical_device_properties2 {
            Instance::init_physical_devices2(&vk, physical_devices, &extensions)
        } else {
            Instance::init_physical_devices(&vk, physical_devices)
        };

        Ok(Arc::new(Instance {
                        instance: instance,
                        //alloc: None,
                        physical_devices: physical_devices,
                        vk: vk,
                        extensions: extensions,
                        layers: layers,
                        function_pointers: function_pointers,
                    }))
    }

    /// Initialize all physical devices
    fn init_physical_devices(vk: &vk::InstancePointers, physical_devices: Vec<vk::PhysicalDevice>)
                             -> Vec<PhysicalDeviceInfos> {
        let mut output = Vec::with_capacity(physical_devices.len());

        for device in physical_devices.into_iter() {
            let properties: vk::PhysicalDeviceProperties = unsafe {
                let mut output = mem::uninitialized();
                vk.GetPhysicalDeviceProperties(device, &mut output);
                output
            };

            let queue_families = unsafe {
                let mut num = 0;
                vk.GetPhysicalDeviceQueueFamilyProperties(device, &mut num, ptr::null_mut());

                let mut families = Vec::with_capacity(num as usize);
                vk.GetPhysicalDeviceQueueFamilyProperties(device, &mut num, families.as_mut_ptr());
                families.set_len(num as usize);
                families
            };

            let memory: vk::PhysicalDeviceMemoryProperties = unsafe {
                let mut output = mem::uninitialized();
                vk.GetPhysicalDeviceMemoryProperties(device, &mut output);
                output
            };

            let available_features: vk::PhysicalDeviceFeatures = unsafe {
                let mut output = mem::uninitialized();
                vk.GetPhysicalDeviceFeatures(device, &mut output);
                output
            };

            output.push(PhysicalDeviceInfos {
                            device: device,
                            properties: properties,
                            memory: memory,
                            queue_families: queue_families,
                            available_features: Features::from_vulkan_features(available_features),
                        });
        }
        output
    }

    /// Initialize all physical devices, but use VK_KHR_get_physical_device_properties2
    /// TODO: Query extension-specific physical device properties, once a new instance extension is supported.
    fn init_physical_devices2(vk: &vk::InstancePointers,
                              physical_devices: Vec<vk::PhysicalDevice>,
                              extensions: &InstanceExtensions)
                              -> Vec<PhysicalDeviceInfos> {
        let mut output = Vec::with_capacity(physical_devices.len());

        for device in physical_devices.into_iter() {
            let properties: vk::PhysicalDeviceProperties = unsafe {
                let mut output = vk::PhysicalDeviceProperties2KHR {
                    sType: vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR,
                    pNext: ptr::null_mut(),
                    properties: mem::uninitialized(),
                };

                vk.GetPhysicalDeviceProperties2KHR(device, &mut output);
                output.properties
            };

            let queue_families = unsafe {
                let mut num = 0;
                vk.GetPhysicalDeviceQueueFamilyProperties2KHR(device, &mut num, ptr::null_mut());

                let mut families = (0 .. num)
                    .map(|_| {
                             vk::QueueFamilyProperties2KHR {
                                 sType: vk::STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2_KHR,
                                 pNext: ptr::null_mut(),
                                 queueFamilyProperties: mem::uninitialized(),
                             }
                         })
                    .collect::<Vec<_>>();

                vk.GetPhysicalDeviceQueueFamilyProperties2KHR(device,
                                                              &mut num,
                                                              families.as_mut_ptr());
                families
                    .into_iter()
                    .map(|family| family.queueFamilyProperties)
                    .collect()
            };

            let memory: vk::PhysicalDeviceMemoryProperties = unsafe {
                let mut output = vk::PhysicalDeviceMemoryProperties2KHR {
                    sType: vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR,
                    pNext: ptr::null_mut(),
                    memoryProperties: mem::uninitialized(),
                };
                vk.GetPhysicalDeviceMemoryProperties2KHR(device, &mut output);
                output.memoryProperties
            };

            let available_features: vk::PhysicalDeviceFeatures = unsafe {
                let mut output = vk::PhysicalDeviceFeatures2KHR {
                    sType: vk::STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR,
                    pNext: ptr::null_mut(),
                    features: mem::uninitialized(),
                };
                vk.GetPhysicalDeviceFeatures2KHR(device, &mut output);
                output.features
            };

            output.push(PhysicalDeviceInfos {
                            device: device,
                            properties: properties,
                            memory: memory,
                            queue_families: queue_families,
                            available_features: Features::from_vulkan_features(available_features),
                        });
        }
        output
    }

    /*/// Same as `new`, but provides an allocator that will be used by the Vulkan library whenever
    /// it needs to allocate memory on the host.
    ///
    /// Note that this allocator can be overriden when you create a `Device`, a `MemoryPool`, etc.
    pub fn with_alloc(app_infos: Option<&ApplicationInfo>, alloc: Box<Alloc + Send + Sync>) -> Arc<Instance> {
        unimplemented!()
    }*/

    /// Grants access to the Vulkan functions of the instance.
    #[inline]
    pub(crate) fn pointers(&self) -> &vk::InstancePointers {
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
    ///
    /// let extensions = InstanceExtensions::supported_by_core().unwrap();
    /// let instance = Instance::new(None, &extensions, None).unwrap();
    /// assert_eq!(instance.loaded_extensions(), &extensions);
    /// ```
    #[inline]
    pub fn loaded_extensions(&self) -> &InstanceExtensions {
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

impl<'a> Default for ApplicationInfo<'a> {
    fn default() -> ApplicationInfo<'a> {
        ApplicationInfo::from_cargo_toml()
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
    fn description(&self) -> &str {
        match *self {
            InstanceCreationError::LoadingError(_) => "failed to load the Vulkan shared library",
            InstanceCreationError::OomError(_) => "not enough memory available",
            InstanceCreationError::InitializationFailed => "initialization failed",
            InstanceCreationError::LayerNotPresent => "layer not present",
            InstanceCreationError::ExtensionNotPresent => "extension not present",
            InstanceCreationError::IncompatibleDriver => "incompatible driver",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
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
        write!(fmt, "{}", error::Error::description(self))
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

struct PhysicalDeviceInfos {
    device: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    queue_families: Vec<vk::QueueFamilyProperties>,
    memory: vk::PhysicalDeviceMemoryProperties,
    available_features: Features,
}

/// Represents one of the available devices on this machine.
///
/// This struct simply contains a pointer to an instance and a number representing the physical
/// device. You are therefore encouraged to pass this around by value instead of by reference.
///
/// # Example
///
/// ```no_run
/// # use vulkano::instance::Instance;
/// # use vulkano::instance::InstanceExtensions;
/// use vulkano::instance::PhysicalDevice;
///
/// # let instance = Instance::new(None, &InstanceExtensions::none(), None).unwrap();
/// for physical_device in PhysicalDevice::enumerate(&instance) {
///     print_infos(physical_device);
/// }
///
/// fn print_infos(dev: PhysicalDevice) {
///     println!("Name: {}", dev.name());
/// }
/// ```
#[derive(Debug, Copy, Clone)]
pub struct PhysicalDevice<'a> {
    instance: &'a Arc<Instance>,
    device: usize,
}

impl<'a> PhysicalDevice<'a> {
    /// Returns an iterator that enumerates the physical devices available.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use vulkano::instance::Instance;
    /// # use vulkano::instance::InstanceExtensions;
    /// use vulkano::instance::PhysicalDevice;
    ///
    /// # let instance = Instance::new(None, &InstanceExtensions::none(), None).unwrap();
    /// for physical_device in PhysicalDevice::enumerate(&instance) {
    ///     println!("Available device: {}", physical_device.name());
    /// }
    /// ```
    #[inline]
    pub fn enumerate(instance: &'a Arc<Instance>) -> PhysicalDevicesIter<'a> {
        PhysicalDevicesIter {
            instance: instance,
            current_id: 0,
        }
    }

    /// Returns a physical device from its index. Returns `None` if out of range.
    ///
    /// Indices range from 0 to the number of devices.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use vulkano::instance::Instance;
    /// use vulkano::instance::InstanceExtensions;
    /// use vulkano::instance::PhysicalDevice;
    ///
    /// let instance = Instance::new(None, &InstanceExtensions::none(), None).unwrap();
    /// let first_physical_device = PhysicalDevice::from_index(&instance, 0).unwrap();
    /// ```
    #[inline]
    pub fn from_index(instance: &'a Arc<Instance>, index: usize) -> Option<PhysicalDevice<'a>> {
        if instance.physical_devices.len() > index {
            Some(PhysicalDevice {
                     instance: instance,
                     device: index,
                 })
        } else {
            None
        }
    }

    /// Returns the instance corresponding to this physical device.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use vulkano::instance::PhysicalDevice;
    ///
    /// fn do_something(physical_device: PhysicalDevice) {
    ///     let _loaded_extensions = physical_device.instance().loaded_extensions();
    ///     // ...
    /// }
    /// ```
    #[inline]
    pub fn instance(&self) -> &'a Arc<Instance> {
        &self.instance
    }

    /// Returns the index of the physical device in the physical devices list.
    ///
    /// This index never changes and can be used later to retrieve a `PhysicalDevice` from an
    /// instance and an index.
    #[inline]
    pub fn index(&self) -> usize {
        self.device
    }

    /// Returns the human-readable name of the device.
    #[inline]
    pub fn name(&self) -> String {
        // FIXME: for some reason this panics if you use a `&str`
        unsafe {
            let val = self.infos().properties.deviceName;
            let val = CStr::from_ptr(val.as_ptr());
            val.to_str()
                .expect("physical device name contained non-UTF8 characters")
                .to_owned()
        }
    }

    /// Returns the type of the device.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use vulkano::instance::Instance;
    /// # use vulkano::instance::InstanceExtensions;
    /// use vulkano::instance::PhysicalDevice;
    ///
    /// # let instance = Instance::new(None, &InstanceExtensions::none(), None).unwrap();
    /// for physical_device in PhysicalDevice::enumerate(&instance) {
    ///     println!("Available device: {} (type: {:?})",
    ///               physical_device.name(), physical_device.ty());
    /// }
    /// ```
    #[inline]
    pub fn ty(&self) -> PhysicalDeviceType {
        match self.instance.physical_devices[self.device]
            .properties
            .deviceType {
            vk::PHYSICAL_DEVICE_TYPE_OTHER => PhysicalDeviceType::Other,
            vk::PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => PhysicalDeviceType::IntegratedGpu,
            vk::PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => PhysicalDeviceType::DiscreteGpu,
            vk::PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU => PhysicalDeviceType::VirtualGpu,
            vk::PHYSICAL_DEVICE_TYPE_CPU => PhysicalDeviceType::Cpu,
            _ => panic!("Unrecognized Vulkan device type"),
        }
    }

    /// Returns the version of Vulkan supported by this device.
    #[inline]
    pub fn api_version(&self) -> Version {
        let val = self.infos().properties.apiVersion;
        Version::from_vulkan_version(val)
    }

    /// Returns the Vulkan features that are supported by this physical device.
    #[inline]
    pub fn supported_features(&self) -> &'a Features {
        &self.infos().available_features
    }

    /// Builds an iterator that enumerates all the queue families on this physical device.
    #[inline]
    pub fn queue_families(&self) -> QueueFamiliesIter<'a> {
        QueueFamiliesIter {
            physical_device: *self,
            current_id: 0,
        }
    }

    /// Returns the queue family with the given index, or `None` if out of range.
    #[inline]
    pub fn queue_family_by_id(&self, id: u32) -> Option<QueueFamily<'a>> {
        if (id as usize) < self.infos().queue_families.len() {
            Some(QueueFamily {
                     physical_device: *self,
                     id: id,
                 })

        } else {
            None
        }
    }

    /// Builds an iterator that enumerates all the memory types on this physical device.
    #[inline]
    pub fn memory_types(&self) -> MemoryTypesIter<'a> {
        MemoryTypesIter {
            physical_device: *self,
            current_id: 0,
        }
    }

    /// Returns the memory type with the given index, or `None` if out of range.
    #[inline]
    pub fn memory_type_by_id(&self, id: u32) -> Option<MemoryType<'a>> {
        if id < self.infos().memory.memoryTypeCount {
            Some(MemoryType {
                     physical_device: *self,
                     id: id,
                 })

        } else {
            None
        }
    }

    /// Builds an iterator that enumerates all the memory heaps on this physical device.
    #[inline]
    pub fn memory_heaps(&self) -> MemoryHeapsIter<'a> {
        MemoryHeapsIter {
            physical_device: *self,
            current_id: 0,
        }
    }

    /// Returns the memory heap with the given index, or `None` if out of range.
    #[inline]
    pub fn memory_heap_by_id(&self, id: u32) -> Option<MemoryHeap<'a>> {
        if id < self.infos().memory.memoryHeapCount {
            Some(MemoryHeap {
                     physical_device: *self,
                     id: id,
                 })

        } else {
            None
        }
    }

    /// Gives access to the limits of the physical device.
    ///
    /// This function should be zero-cost in release mode. It only exists to not pollute the
    /// namespace of `PhysicalDevice` with all the limits-related getters.
    #[inline]
    pub fn limits(&self) -> Limits<'a> {
        Limits { device: *self }
    }

    /// Returns an opaque number representing the version of the driver of this device.
    ///
    /// The meaning of this number is implementation-specific. It can be used in bug reports, for
    /// example.
    #[inline]
    pub fn driver_version(&self) -> u32 {
        self.infos().properties.driverVersion
    }

    /// Returns the PCI ID of the device.
    #[inline]
    pub fn pci_device_id(&self) -> u32 {
        self.infos().properties.deviceID
    }

    /// Returns the PCI ID of the vendor.
    #[inline]
    pub fn pci_vendor_id(&self) -> u32 {
        self.infos().properties.vendorID
    }

    /// Returns a unique identifier for the device.
    ///
    /// Can be stored in a configuration file, so that you can retrieve the device again the next
    /// time the program is run.
    #[inline]
    pub fn uuid(&self) -> &[u8; 16] {
        // must be equal to vk::UUID_SIZE
        &self.infos().properties.pipelineCacheUUID
    }

    // Internal function to make it easier to get the infos of this device.
    #[inline]
    fn infos(&self) -> &'a PhysicalDeviceInfos {
        &self.instance.physical_devices[self.device]
    }
}

unsafe impl<'a> VulkanObject for PhysicalDevice<'a> {
    type Object = vk::PhysicalDevice;

    #[inline]
    fn internal_object(&self) -> vk::PhysicalDevice {
        self.infos().device
    }
}

/// Iterator for all the physical devices available on hardware.
#[derive(Debug, Clone)]
pub struct PhysicalDevicesIter<'a> {
    instance: &'a Arc<Instance>,
    current_id: usize,
}

impl<'a> Iterator for PhysicalDevicesIter<'a> {
    type Item = PhysicalDevice<'a>;

    #[inline]
    fn next(&mut self) -> Option<PhysicalDevice<'a>> {
        if self.current_id >= self.instance.physical_devices.len() {
            return None;
        }

        let dev = PhysicalDevice {
            instance: self.instance,
            device: self.current_id,
        };

        self.current_id += 1;
        Some(dev)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.instance.physical_devices.len() - self.current_id;
        (len, Some(len))
    }
}

impl<'a> ExactSizeIterator for PhysicalDevicesIter<'a> {
}

/// Type of a physical device.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum PhysicalDeviceType {
    /// The device is an integrated GPU.
    IntegratedGpu = 1,
    /// The device is a discrete GPU.
    DiscreteGpu = 2,
    /// The device is a virtual GPU.
    VirtualGpu = 3,
    /// The device is a CPU.
    Cpu = 4,
    /// The device is something else.
    Other = 0,
}

/// Represents a queue family in a physical device.
///
/// A queue family is group of one or multiple queues. All queues of one family have the same
/// characteristics.
#[derive(Debug, Copy, Clone)]
pub struct QueueFamily<'a> {
    physical_device: PhysicalDevice<'a>,
    id: u32,
}

impl<'a> QueueFamily<'a> {
    /// Returns the physical device associated to this queue family.
    #[inline]
    pub fn physical_device(&self) -> PhysicalDevice<'a> {
        self.physical_device
    }

    /// Returns the identifier of this queue family within the physical device.
    #[inline]
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Returns the number of queues that belong to this family.
    ///
    /// Guaranteed to be at least 1 (or else that family wouldn't exist).
    #[inline]
    pub fn queues_count(&self) -> usize {
        self.physical_device.infos().queue_families[self.id as usize].queueCount as usize
    }

    /// Returns true if queues of this family can execute graphics operations.
    #[inline]
    pub fn supports_graphics(&self) -> bool {
        (self.flags() & vk::QUEUE_GRAPHICS_BIT) != 0
    }

    /// Returns true if queues of this family can execute compute operations.
    #[inline]
    pub fn supports_compute(&self) -> bool {
        (self.flags() & vk::QUEUE_COMPUTE_BIT) != 0
    }

    /// Returns true if queues of this family can execute transfer operations.
    ///
    /// > **Note**: Queues that support graphics or compute operations also always support transfer
    /// > operations. As of writing this, this function will always return true. The purpose of
    /// > this function is to be future-proofed in case queues that don't support transfer
    /// > operations are ever added to Vulkan.
    #[inline]
    pub fn supports_transfers(&self) -> bool {
        (self.flags() & vk::QUEUE_TRANSFER_BIT) != 0 || self.supports_graphics() ||
            self.supports_compute()
    }

    /// Returns true if queues of this family can execute sparse resources binding operations.
    #[inline]
    pub fn supports_sparse_binding(&self) -> bool {
        (self.flags() & vk::QUEUE_SPARSE_BINDING_BIT) != 0
    }

    /// Internal utility function that returns the flags of this queue family.
    #[inline]
    fn flags(&self) -> u32 {
        self.physical_device.infos().queue_families[self.id as usize].queueFlags
    }
}

/// Iterator for all the queue families available on a physical device.
#[derive(Debug, Clone)]
pub struct QueueFamiliesIter<'a> {
    physical_device: PhysicalDevice<'a>,
    current_id: u32,
}

impl<'a> Iterator for QueueFamiliesIter<'a> {
    type Item = QueueFamily<'a>;

    #[inline]
    fn next(&mut self) -> Option<QueueFamily<'a>> {
        if self.current_id as usize >= self.physical_device.infos().queue_families.len() {
            return None;
        }

        let dev = QueueFamily {
            physical_device: self.physical_device,
            id: self.current_id,
        };

        self.current_id += 1;
        Some(dev)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.physical_device.infos().queue_families.len();
        let remain = len - self.current_id as usize;
        (remain, Some(remain))
    }
}

impl<'a> ExactSizeIterator for QueueFamiliesIter<'a> {
}

/// Represents a memory type in a physical device.
#[derive(Debug, Copy, Clone)]
pub struct MemoryType<'a> {
    physical_device: PhysicalDevice<'a>,
    id: u32,
}

impl<'a> MemoryType<'a> {
    /// Returns the physical device associated to this memory type.
    #[inline]
    pub fn physical_device(&self) -> PhysicalDevice<'a> {
        self.physical_device
    }

    /// Returns the identifier of this memory type within the physical device.
    #[inline]
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Returns the heap that corresponds to this memory type.
    #[inline]
    pub fn heap(&self) -> MemoryHeap<'a> {
        let heap_id = self.physical_device.infos().memory.memoryTypes[self.id as usize].heapIndex;
        MemoryHeap {
            physical_device: self.physical_device,
            id: heap_id,
        }
    }

    /// Returns true if the memory type is located on the device, which means that it's the most
    /// efficient for GPU accesses.
    #[inline]
    pub fn is_device_local(&self) -> bool {
        (self.flags() & vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0
    }

    /// Returns true if the memory type can be accessed by the host.
    #[inline]
    pub fn is_host_visible(&self) -> bool {
        (self.flags() & vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0
    }

    /// Returns true if modifications made by the host or the GPU on this memory type are
    /// instantaneously visible to the other party. False means that changes have to be flushed.
    ///
    /// You don't need to worry about this, as this library handles that for you.
    #[inline]
    pub fn is_host_coherent(&self) -> bool {
        (self.flags() & vk::MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0
    }

    /// Returns true if memory of this memory type is cached by the host. Host memory accesses to
    /// cached memory is faster than for uncached memory. However you are not guaranteed that it
    /// is coherent.
    #[inline]
    pub fn is_host_cached(&self) -> bool {
        (self.flags() & vk::MEMORY_PROPERTY_HOST_CACHED_BIT) != 0
    }

    /// Returns true if allocations made to this memory type is lazy.
    ///
    /// This means that no actual allocation is performed. Instead memory is automatically
    /// allocated by the Vulkan implementation.
    ///
    /// Memory of this type can only be used on images created with a certain flag. Memory of this
    /// type is never host-visible.
    #[inline]
    pub fn is_lazily_allocated(&self) -> bool {
        (self.flags() & vk::MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) != 0
    }

    /// Internal utility function that returns the flags of this queue family.
    #[inline]
    fn flags(&self) -> u32 {
        self.physical_device.infos().memory.memoryTypes[self.id as usize].propertyFlags
    }
}

/// Iterator for all the memory types available on a physical device.
#[derive(Debug, Clone)]
pub struct MemoryTypesIter<'a> {
    physical_device: PhysicalDevice<'a>,
    current_id: u32,
}

impl<'a> Iterator for MemoryTypesIter<'a> {
    type Item = MemoryType<'a>;

    #[inline]
    fn next(&mut self) -> Option<MemoryType<'a>> {
        if self.current_id >= self.physical_device.infos().memory.memoryTypeCount {
            return None;
        }

        let dev = MemoryType {
            physical_device: self.physical_device,
            id: self.current_id,
        };

        self.current_id += 1;
        Some(dev)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.physical_device.infos().memory.memoryTypeCount;
        let remain = (len - self.current_id) as usize;
        (remain, Some(remain))
    }
}

impl<'a> ExactSizeIterator for MemoryTypesIter<'a> {
}

/// Represents a memory heap in a physical device.
#[derive(Debug, Copy, Clone)]
pub struct MemoryHeap<'a> {
    physical_device: PhysicalDevice<'a>,
    id: u32,
}

impl<'a> MemoryHeap<'a> {
    /// Returns the physical device associated to this memory heap.
    #[inline]
    pub fn physical_device(&self) -> PhysicalDevice<'a> {
        self.physical_device
    }

    /// Returns the identifier of this memory heap within the physical device.
    #[inline]
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Returns the size in bytes on this heap.
    #[inline]
    pub fn size(&self) -> usize {
        self.physical_device.infos().memory.memoryHeaps[self.id as usize].size as usize
    }

    /// Returns true if the heap is local to the GPU.
    #[inline]
    pub fn is_device_local(&self) -> bool {
        let flags = self.physical_device.infos().memory.memoryHeaps[self.id as usize].flags;
        (flags & vk::MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0
    }
}

/// Iterator for all the memory heaps available on a physical device.
#[derive(Debug, Clone)]
pub struct MemoryHeapsIter<'a> {
    physical_device: PhysicalDevice<'a>,
    current_id: u32,
}

impl<'a> Iterator for MemoryHeapsIter<'a> {
    type Item = MemoryHeap<'a>;

    #[inline]
    fn next(&mut self) -> Option<MemoryHeap<'a>> {
        if self.current_id >= self.physical_device.infos().memory.memoryHeapCount {
            return None;
        }

        let dev = MemoryHeap {
            physical_device: self.physical_device,
            id: self.current_id,
        };

        self.current_id += 1;
        Some(dev)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.physical_device.infos().memory.memoryHeapCount;
        let remain = (len - self.current_id) as usize;
        (remain, Some(remain))
    }
}

impl<'a> ExactSizeIterator for MemoryHeapsIter<'a> {
}

/// Limits of a physical device.
pub struct Limits<'a> {
    device: PhysicalDevice<'a>,
}

macro_rules! limits_impl {
    ($($name:ident: $t:ty => $target:ident,)*) => (
        impl<'a> Limits<'a> {
            $(
                #[inline]
                pub fn $name(&self) -> $t {
                    self.device.infos().properties.limits.$target
                }
            )*
        }
    )
}

limits_impl!{
    max_image_dimension_1d: u32 => maxImageDimension1D,
    max_image_dimension_2d: u32 => maxImageDimension2D,
    max_image_dimension_3d: u32 => maxImageDimension3D,
    max_image_dimension_cube: u32 => maxImageDimensionCube,
    max_image_array_layers: u32 => maxImageArrayLayers,
    max_texel_buffer_elements: u32 => maxTexelBufferElements,
    max_uniform_buffer_range: u32 => maxUniformBufferRange,
    max_storage_buffer_range: u32 => maxStorageBufferRange,
    max_push_constants_size: u32 => maxPushConstantsSize,
    max_memory_allocation_count: u32 => maxMemoryAllocationCount,
    max_sampler_allocation_count: u32 => maxSamplerAllocationCount,
    buffer_image_granularity: u64 => bufferImageGranularity,
    sparse_address_space_size: u64 => sparseAddressSpaceSize,
    max_bound_descriptor_sets: u32 => maxBoundDescriptorSets,
    max_per_stage_descriptor_samplers: u32 => maxPerStageDescriptorSamplers,
    max_per_stage_descriptor_uniform_buffers: u32 => maxPerStageDescriptorUniformBuffers,
    max_per_stage_descriptor_storage_buffers: u32 => maxPerStageDescriptorStorageBuffers,
    max_per_stage_descriptor_sampled_images: u32 => maxPerStageDescriptorSampledImages,
    max_per_stage_descriptor_storage_images: u32 => maxPerStageDescriptorStorageImages,
    max_per_stage_descriptor_input_attachments: u32 => maxPerStageDescriptorInputAttachments,
    max_per_stage_resources: u32 => maxPerStageResources,
    max_descriptor_set_samplers: u32 => maxDescriptorSetSamplers,
    max_descriptor_set_uniform_buffers: u32 => maxDescriptorSetUniformBuffers,
    max_descriptor_set_uniform_buffers_dynamic: u32 => maxDescriptorSetUniformBuffersDynamic,
    max_descriptor_set_storage_buffers: u32 => maxDescriptorSetStorageBuffers,
    max_descriptor_set_storage_buffers_dynamic: u32 => maxDescriptorSetStorageBuffersDynamic,
    max_descriptor_set_sampled_images: u32 => maxDescriptorSetSampledImages,
    max_descriptor_set_storage_images: u32 => maxDescriptorSetStorageImages,
    max_descriptor_set_input_attachments: u32 => maxDescriptorSetInputAttachments,
    max_vertex_input_attributes: u32 => maxVertexInputAttributes,
    max_vertex_input_bindings: u32 => maxVertexInputBindings,
    max_vertex_input_attribute_offset: u32 => maxVertexInputAttributeOffset,
    max_vertex_input_binding_stride: u32 => maxVertexInputBindingStride,
    max_vertex_output_components: u32 => maxVertexOutputComponents,
    max_tessellation_generation_level: u32 => maxTessellationGenerationLevel,
    max_tessellation_patch_size: u32 => maxTessellationPatchSize,
    max_tessellation_control_per_vertex_input_components: u32 => maxTessellationControlPerVertexInputComponents,
    max_tessellation_control_per_vertex_output_components: u32 => maxTessellationControlPerVertexOutputComponents,
    max_tessellation_control_per_patch_output_components: u32 => maxTessellationControlPerPatchOutputComponents,
    max_tessellation_control_total_output_components: u32 => maxTessellationControlTotalOutputComponents,
    max_tessellation_evaluation_input_components: u32 => maxTessellationEvaluationInputComponents,
    max_tessellation_evaluation_output_components: u32 => maxTessellationEvaluationOutputComponents,
    max_geometry_shader_invocations: u32 => maxGeometryShaderInvocations,
    max_geometry_input_components: u32 => maxGeometryInputComponents,
    max_geometry_output_components: u32 => maxGeometryOutputComponents,
    max_geometry_output_vertices: u32 => maxGeometryOutputVertices,
    max_geometry_total_output_components: u32 => maxGeometryTotalOutputComponents,
    max_fragment_input_components: u32 => maxFragmentInputComponents,
    max_fragment_output_attachments: u32 => maxFragmentOutputAttachments,
    max_fragment_dual_src_attachments: u32 => maxFragmentDualSrcAttachments,
    max_fragment_combined_output_resources: u32 => maxFragmentCombinedOutputResources,
    max_compute_shared_memory_size: u32 => maxComputeSharedMemorySize,
    max_compute_work_group_count: [u32; 3] => maxComputeWorkGroupCount,
    max_compute_work_group_invocations: u32 => maxComputeWorkGroupInvocations,
    max_compute_work_group_size: [u32; 3] => maxComputeWorkGroupSize,
    sub_pixel_precision_bits: u32 => subPixelPrecisionBits,
    sub_texel_precision_bits: u32 => subTexelPrecisionBits,
    mipmap_precision_bits: u32 => mipmapPrecisionBits,
    max_draw_indexed_index_value: u32 => maxDrawIndexedIndexValue,
    max_draw_indirect_count: u32 => maxDrawIndirectCount,
    max_sampler_lod_bias: f32 => maxSamplerLodBias,
    max_sampler_anisotropy: f32 => maxSamplerAnisotropy,
    max_viewports: u32 => maxViewports,
    max_viewport_dimensions: [u32; 2] => maxViewportDimensions,
    viewport_bounds_range: [f32; 2] => viewportBoundsRange,
    viewport_sub_pixel_bits: u32 => viewportSubPixelBits,
    min_memory_map_alignment: usize => minMemoryMapAlignment,
    min_texel_buffer_offset_alignment: u64 => minTexelBufferOffsetAlignment,
    min_uniform_buffer_offset_alignment: u64 => minUniformBufferOffsetAlignment,
    min_storage_buffer_offset_alignment: u64 => minStorageBufferOffsetAlignment,
    min_texel_offset: i32 => minTexelOffset,
    max_texel_offset: u32 => maxTexelOffset,
    min_texel_gather_offset: i32 => minTexelGatherOffset,
    max_texel_gather_offset: u32 => maxTexelGatherOffset,
    min_interpolation_offset: f32 => minInterpolationOffset,
    max_interpolation_offset: f32 => maxInterpolationOffset,
    sub_pixel_interpolation_offset_bits: u32 => subPixelInterpolationOffsetBits,
    max_framebuffer_width: u32 => maxFramebufferWidth,
    max_framebuffer_height: u32 => maxFramebufferHeight,
    max_framebuffer_layers: u32 => maxFramebufferLayers,
    framebuffer_color_sample_counts: u32 => framebufferColorSampleCounts,      // FIXME: SampleCountFlag
    framebuffer_depth_sample_counts: u32 => framebufferDepthSampleCounts,      // FIXME: SampleCountFlag
    framebuffer_stencil_sample_counts: u32 => framebufferStencilSampleCounts,      // FIXME: SampleCountFlag
    framebuffer_no_attachments_sample_counts: u32 => framebufferNoAttachmentsSampleCounts,      // FIXME: SampleCountFlag
    max_color_attachments: u32 => maxColorAttachments,
    sampled_image_color_sample_counts: u32 => sampledImageColorSampleCounts,        // FIXME: SampleCountFlag
    sampled_image_integer_sample_counts: u32 => sampledImageIntegerSampleCounts,        // FIXME: SampleCountFlag
    sampled_image_depth_sample_counts: u32 => sampledImageDepthSampleCounts,        // FIXME: SampleCountFlag
    sampled_image_stencil_sample_counts: u32 => sampledImageStencilSampleCounts,        // FIXME: SampleCountFlag
    storage_image_sample_counts: u32 => storageImageSampleCounts,      // FIXME: SampleCountFlag
    max_sample_mask_words: u32 => maxSampleMaskWords,
    timestamp_compute_and_graphics: u32 => timestampComputeAndGraphics,        // TODO: these are booleans
    timestamp_period: f32 => timestampPeriod,
    max_clip_distances: u32 => maxClipDistances,
    max_cull_distances: u32 => maxCullDistances,
    max_combined_clip_and_cull_distances: u32 => maxCombinedClipAndCullDistances,
    discrete_queue_priorities: u32 => discreteQueuePriorities,
    point_size_range: [f32; 2] => pointSizeRange,
    line_width_range: [f32; 2] => lineWidthRange,
    point_size_granularity: f32 => pointSizeGranularity,
    line_width_granularity: f32 => lineWidthGranularity,
    strict_lines: u32 => strictLines,        // TODO: these are booleans
    standard_sample_locations: u32 => standardSampleLocations,        // TODO: these are booleans
    optimal_buffer_copy_offset_alignment: u64 => optimalBufferCopyOffsetAlignment,
    optimal_buffer_copy_row_pitch_alignment: u64 => optimalBufferCopyRowPitchAlignment,
    non_coherent_atom_size: u64 => nonCoherentAtomSize,
}

#[cfg(test)]
mod tests {
    use instance;

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
