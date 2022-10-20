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
//! The first thing to do after loading the Vulkan library is to create an `Instance` object.
//!
//! For example:
//!
//! ```no_run
//! use vulkano::{
//!     instance::{Instance, InstanceExtensions},
//!     Version, VulkanLibrary,
//! };
//!
//! let library = VulkanLibrary::new()
//!     .unwrap_or_else(|err| panic!("Couldn't load Vulkan library: {:?}", err));
//! let instance = Instance::new(library, Default::default())
//!     .unwrap_or_else(|err| panic!("Couldn't create instance: {:?}", err));
//! ```
//!
//! Creating an instance initializes everything and allows you to enumerate physical devices,
//! ie. all the Vulkan implementations that are available on the system.
//!
//! ```no_run
//! # use vulkano::{
//! #     instance::{Instance, InstanceExtensions},
//! #     Version, VulkanLibrary,
//! # };
//! use vulkano::device::physical::PhysicalDevice;
//!
//! # let library = VulkanLibrary::new().unwrap();
//! # let instance = Instance::new(library, Default::default()).unwrap();
//! for physical_device in instance.enumerate_physical_devices().unwrap() {
//!     println!("Available device: {}", physical_device.properties().device_name);
//! }
//! ```
//!
//! # Enumerating physical devices and creating a device
//!
//! After you have created an instance, the next step is usually to enumerate the physical devices
//! that are available on the system with `Instance::enumerate_physical_devices()` (see above).
//!
//! When choosing which physical device to use, keep in mind that physical devices may or may not
//! be able to draw to a certain surface (ie. to a window or a monitor), or may even not be able
//! to draw at all. See the `swapchain` module for more information about surfaces.
//!
//! Once you have chosen a physical device, you can create a `Device` object from it. See the
//! `device` module for more info.

use self::debug::{
    DebugUtilsMessengerCreateInfo, UserCallback, ValidationFeatureDisable, ValidationFeatureEnable,
};
pub use self::{extensions::InstanceExtensions, layers::LayerProperties};
use crate::{
    device::physical::PhysicalDevice, instance::debug::trampoline, OomError, RequiresOneOf,
    VulkanError, VulkanLibrary, VulkanObject,
};
pub use crate::{
    extensions::{ExtensionRestriction, ExtensionRestrictionError},
    fns::InstanceFunctions,
    version::Version,
};
use smallvec::SmallVec;
use std::{
    error::Error,
    ffi::{c_void, CString},
    fmt::{Debug, Display, Error as FmtError, Formatter},
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    panic::{RefUnwindSafe, UnwindSafe},
    ptr,
    sync::Arc,
};

pub mod debug;
pub(crate) mod extensions;
mod layers;

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
/// use vulkano::{
///     instance::{Instance, InstanceCreateInfo, InstanceExtensions},
///     Version, VulkanLibrary,
/// };
///
/// let library = VulkanLibrary::new().unwrap();
/// let _instance = Instance::new(
///     library,
///     InstanceCreateInfo::application_from_cargo_toml(),
/// ).unwrap();
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
/// [`VulkanLibrary::api_version`](crate::VulkanLibrary::api_version),
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
/// use vulkano::{
///     instance::{Instance, InstanceCreateInfo, InstanceExtensions},
///     Version, VulkanLibrary,
/// };
///
/// let library = VulkanLibrary::new()
///     .unwrap_or_else(|err| panic!("Couldn't load Vulkan library: {:?}", err));
///
/// let extensions = InstanceExtensions {
///     khr_surface: true,
///     khr_android_surface: true,
///     .. InstanceExtensions::empty()
/// };
///
/// let instance = Instance::new(
///     library,
///     InstanceCreateInfo {
///         enabled_extensions: extensions,
///         ..Default::default()
///     },
/// )
/// .unwrap_or_else(|err| panic!("Couldn't create instance: {:?}", err));
/// ```
///
/// # Layers
///
/// When creating an `Instance`, you have the possibility to pass a list of **layers** that will
/// be activated on the newly-created instance. The list of available layers can be retrieved by
/// calling the [`layer_properties`](crate::VulkanLibrary::layer_properties) method of
/// `VulkanLibrary`.
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
/// ## Examples
///
/// ```
/// # use std::{sync::Arc, error::Error};
/// # use vulkano::{
/// #     instance::{Instance, InstanceCreateInfo, InstanceExtensions},
/// #     Version, VulkanLibrary,
/// # };
/// # fn test() -> Result<Arc<Instance>, Box<dyn Error>> {
/// let library = VulkanLibrary::new()?;
///
/// // For the sake of the example, we activate all the layers that
/// // contain the word "foo" in their description.
/// let layers: Vec<_> = library.layer_properties()?
///     .filter(|l| l.description().contains("foo"))
///     .collect();
///
/// let instance = Instance::new(
///     library,
///     InstanceCreateInfo {
///         enabled_layers: layers.iter().map(|l| l.name().to_owned()).collect(),
///         ..Default::default()
///     },
/// )?;
/// # Ok(instance)
/// # }
/// ```
// TODO: mention that extensions must be supported by layers as well
pub struct Instance {
    handle: ash::vk::Instance,
    fns: InstanceFunctions,

    api_version: Version,
    enabled_extensions: InstanceExtensions,
    enabled_layers: Vec<String>,
    library: Arc<VulkanLibrary>,
    max_api_version: Version,
    _user_callbacks: Vec<Box<UserCallback>>,
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
    pub fn new(
        library: Arc<VulkanLibrary>,
        create_info: InstanceCreateInfo,
    ) -> Result<Arc<Instance>, InstanceCreationError> {
        unsafe { Self::with_debug_utils_messengers(library, create_info, []) }
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
        library: Arc<VulkanLibrary>,
        create_info: InstanceCreateInfo,
        debug_utils_messengers: impl IntoIterator<Item = DebugUtilsMessengerCreateInfo>,
    ) -> Result<Arc<Instance>, InstanceCreationError> {
        let InstanceCreateInfo {
            application_name,
            application_version,
            mut enabled_extensions,
            enabled_layers,
            engine_name,
            engine_version,
            max_api_version,
            enumerate_portability,
            enabled_validation_features,
            disabled_validation_features,
            _ne: _,
        } = create_info;

        let (api_version, max_api_version) = {
            let api_version = library.api_version();
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
        let supported_extensions =
            library.supported_extensions_with_layers(enabled_layers.iter().map(String::as_str))?;
        let mut flags = ash::vk::InstanceCreateFlags::empty();

        if enumerate_portability && supported_extensions.khr_portability_enumeration {
            enabled_extensions.khr_portability_enumeration = true;
            flags |= ash::vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        // Check if the extensions are correct
        enabled_extensions.check_requirements(&supported_extensions, api_version)?;

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

        let enable_validation_features_vk: SmallVec<[_; 5]> = enabled_validation_features
            .iter()
            .copied()
            .map(Into::into)
            .collect();
        let disable_validation_features_vk: SmallVec<[_; 8]> = disabled_validation_features
            .iter()
            .copied()
            .map(Into::into)
            .collect();

        let mut create_info_vk = ash::vk::InstanceCreateInfo {
            flags,
            p_application_info: &application_info,
            enabled_layer_count: enabled_layers_ptrs.len() as u32,
            pp_enabled_layer_names: enabled_layers_ptrs.as_ptr(),
            enabled_extension_count: enabled_extensions_ptrs.len() as u32,
            pp_enabled_extension_names: enabled_extensions_ptrs.as_ptr(),
            ..Default::default()
        };
        let mut validation_features_vk = None;

        if !enabled_validation_features.is_empty() || !disabled_validation_features.is_empty() {
            if !enabled_extensions.ext_validation_features {
                return Err(InstanceCreationError::RequirementNotMet {
                    required_for: "`enabled_validation_features` or `disabled_validation_features` are not empty",
                    requires_one_of: RequiresOneOf {
                        instance_extensions: &["ext_validation_features"],
                        ..Default::default()
                    },
                });
            }

            // VUID-VkValidationFeaturesEXT-pEnabledValidationFeatures-02967
            assert!(
                !enabled_validation_features
                    .contains(&ValidationFeatureEnable::GpuAssistedReserveBindingSlot)
                    || enabled_validation_features.contains(&ValidationFeatureEnable::GpuAssisted)
            );

            // VUID-VkValidationFeaturesEXT-pEnabledValidationFeatures-02968
            assert!(
                !(enabled_validation_features.contains(&ValidationFeatureEnable::DebugPrintf)
                    && enabled_validation_features.contains(&ValidationFeatureEnable::GpuAssisted))
            );

            let next = validation_features_vk.insert(ash::vk::ValidationFeaturesEXT {
                enabled_validation_feature_count: enable_validation_features_vk.len() as u32,
                p_enabled_validation_features: enable_validation_features_vk.as_ptr(),
                disabled_validation_feature_count: disable_validation_features_vk.len() as u32,
                p_disabled_validation_features: disable_validation_features_vk.as_ptr(),
                ..Default::default()
            });

            next.p_next = create_info_vk.p_next;
            create_info_vk.p_next = next as *const _ as *const _;
        }

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
                return Err(InstanceCreationError::RequirementNotMet {
                    required_for: "`debug_utils_messengers` is not empty",
                    requires_one_of: RequiresOneOf {
                        instance_extensions: &["ext_debug_utils"],
                        ..Default::default()
                    },
                });
            }

            // VUID-VkDebugUtilsMessengerCreateInfoEXT-messageSeverity-parameter
            // TODO: message_severity.validate_instance()?;

            // VUID-VkDebugUtilsMessengerCreateInfoEXT-messageSeverity-requiredbitmask
            assert!(!message_severity.is_empty());

            // VUID-VkDebugUtilsMessengerCreateInfoEXT-messageType-parameter
            // TODO: message_type.validate_instance()?;

            // VUID-VkDebugUtilsMessengerCreateInfoEXT-messageType-requiredbitmask
            assert!(!message_type.is_empty());

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
            create_info_vk.p_next = info as *const _ as *const _;
        }

        // Creating the Vulkan instance.
        let handle = {
            let mut output = MaybeUninit::uninit();
            let fns = library.fns();
            (fns.v1_0.create_instance)(&create_info_vk, ptr::null(), output.as_mut_ptr())
                .result()
                .map_err(VulkanError::from)?;
            output.assume_init()
        };

        // Loading the function pointers of the newly-created instance.
        let fns = {
            InstanceFunctions::load(|name| {
                library
                    .get_instance_proc_addr(handle, name.as_ptr())
                    .map_or(ptr::null(), |func| func as _)
            })
        };

        Ok(Arc::new(Instance {
            handle,
            fns,

            api_version,
            enabled_extensions,
            enabled_layers,
            library,
            max_api_version,
            _user_callbacks: user_callbacks,
        }))
    }

    /// Returns the Vulkan library used to create this instance.
    #[inline]
    pub fn library(&self) -> &Arc<VulkanLibrary> {
        &self.library
    }

    /// Returns the Vulkan version supported by the instance.
    ///
    /// This is the lower of the
    /// [driver's supported version](crate::VulkanLibrary::api_version) and
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

    /// Returns pointers to the raw Vulkan functions of the instance.
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

    /// Returns an iterator that enumerates the physical devices available.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use vulkano::{
    /// #     instance::{Instance, InstanceExtensions},
    /// #     Version, VulkanLibrary,
    /// # };
    ///
    /// # let library = VulkanLibrary::new().unwrap();
    /// # let instance = Instance::new(library, Default::default()).unwrap();
    /// for physical_device in instance.enumerate_physical_devices().unwrap() {
    ///     println!("Available device: {}", physical_device.properties().device_name);
    /// }
    /// ```
    pub fn enumerate_physical_devices(
        self: &Arc<Self>,
    ) -> Result<impl ExactSizeIterator<Item = Arc<PhysicalDevice>>, VulkanError> {
        let fns = self.fns();

        unsafe {
            let handles = loop {
                let mut count = 0;
                (fns.v1_0.enumerate_physical_devices)(self.handle, &mut count, ptr::null_mut())
                    .result()
                    .map_err(VulkanError::from)?;

                let mut handles = Vec::with_capacity(count as usize);
                let result = (fns.v1_0.enumerate_physical_devices)(
                    self.handle,
                    &mut count,
                    handles.as_mut_ptr(),
                );

                match result {
                    ash::vk::Result::SUCCESS => {
                        handles.set_len(count as usize);
                        break handles;
                    }
                    ash::vk::Result::INCOMPLETE => (),
                    err => return Err(VulkanError::from(err)),
                }
            };

            let physical_devices: SmallVec<[_; 4]> = handles
                .into_iter()
                .map(|handle| PhysicalDevice::from_handle(self.clone(), handle))
                .collect::<Result<_, _>>()?;

            Ok(physical_devices.into_iter())
        }
    }
}

impl Drop for Instance {
    #[inline]
    fn drop(&mut self) {
        let fns = self.fns();

        unsafe {
            (fns.v1_0.destroy_instance)(self.handle, ptr::null());
        }
    }
}

unsafe impl VulkanObject for Instance {
    type Handle = ash::vk::Instance;

    #[inline]
    fn handle(&self) -> Self::Handle {
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
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
    }
}

impl Debug for Instance {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        let Self {
            handle,
            fns,
            api_version,
            enabled_extensions,
            enabled_layers,
            library: function_pointers,
            max_api_version,
            _user_callbacks: _,
        } = self;

        f.debug_struct("Instance")
            .field("handle", handle)
            .field("fns", fns)
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
    /// The default value is [`InstanceExtensions::empty()`].
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

    /// The highest Vulkan API version that the application will use with the instance.
    ///
    /// Usually, you will want to leave this at the default.
    ///
    /// The default value is [`Version::HEADER_VERSION`], but if the
    /// supported instance version is 1.0, then it will be 1.0.
    pub max_api_version: Option<Version>,

    /// Enumerate devices that support `VK_KHR_portability_subset`.
    ///
    /// With this enabled, devices that use non-conformant vulkan implementations can be enumerated.
    /// (ex. MoltenVK)
    ///
    /// The default value is false.
    ///
    /// # Notes
    ///
    /// - If `true` and `khr_portability_enumeration` extension is not preset this field will be ignored
    ///   and the `ENUMERATE_PORTABILITY_KHR` flag will not be set.
    /// - If `true` and `khr_portability_enumeration` extension is present, `khr_portability_enumeration`
    ///   extension will automatically be enabled.
    pub enumerate_portability: bool,

    /// Features of the validation layer to enable.
    ///
    /// If not empty, the
    /// [`ext_validation_features`](crate::instance::InstanceExtensions::ext_validation_features)
    /// extension must be enabled on the instance.
    pub enabled_validation_features: Vec<ValidationFeatureEnable>,

    /// Features of the validation layer to disable.
    ///
    /// If not empty, the
    /// [`ext_validation_features`](crate::instance::InstanceExtensions::ext_validation_features)
    /// extension must be enabled on the instance.
    pub disabled_validation_features: Vec<ValidationFeatureDisable>,

    pub _ne: crate::NonExhaustive,
}

impl Default for InstanceCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            application_name: None,
            application_version: Version::major_minor(0, 0),
            enabled_extensions: InstanceExtensions::empty(),
            enabled_layers: Vec::new(),
            engine_name: None,
            engine_version: Version::major_minor(0, 0),
            max_api_version: None,
            enumerate_portability: false,
            enabled_validation_features: Vec::new(),
            disabled_validation_features: Vec::new(),
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

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },
}

impl Error for InstanceCreationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OomError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for InstanceCreationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => write!(f, "not enough memory available"),
            Self::InitializationFailed => write!(f, "initialization failed"),
            Self::LayerNotPresent => write!(f, "layer not present"),
            Self::ExtensionNotPresent => write!(f, "extension not present"),
            Self::IncompatibleDriver => write!(f, "incompatible driver"),
            Self::ExtensionRestrictionNotMet(err) => Display::fmt(err, f),
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
        }
    }
}

impl From<OomError> for InstanceCreationError {
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}

impl From<ExtensionRestrictionError> for InstanceCreationError {
    fn from(err: ExtensionRestrictionError) -> Self {
        Self::ExtensionRestrictionNotMet(err)
    }
}

impl From<VulkanError> for InstanceCreationError {
    fn from(err: VulkanError) -> Self {
        match err {
            err @ VulkanError::OutOfHostMemory => Self::OomError(OomError::from(err)),
            err @ VulkanError::OutOfDeviceMemory => Self::OomError(OomError::from(err)),
            VulkanError::InitializationFailed => Self::InitializationFailed,
            VulkanError::LayerNotPresent => Self::LayerNotPresent,
            VulkanError::ExtensionNotPresent => Self::ExtensionNotPresent,
            VulkanError::IncompatibleDriver => Self::IncompatibleDriver,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn create_instance() {
        let _ = instance!();
    }
}
