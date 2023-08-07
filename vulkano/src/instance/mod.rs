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
//!
//! # Portability subset devices and the `ENUMERATE_PORTABILITY` flag
//!
//! Certain devices, currently those on MacOS and iOS systems, do not fully conform to the Vulkan
//! specification. They are usable as normal devices, but they do not implement everything that
//! is required; some mandatory parts of Vulkan are missing. These are known as
//! "portability subset" devices.
//!
//! A portability subset device will advertise support for the
//! [`khr_portability_subset`](crate::device::DeviceExtensions::khr_portability_subset) device
//! extension. This extension must always be enabled when it is supported, and Vulkano will
//! automatically enable it when creating the device. When it is enabled, some parts of Vulkan that
//! are available in standard Vulkan will not be available by default, but they can be used by
//! enabling corresponding features when creating the device, if the device supports them.
//!
//! Because these devices are non-conformant, Vulkan programs that rely on full compliance may
//! not work (crash or have validation errors) when run on them, if they happen to use a part of
//! Vulkan that is missing from the non-conformant device. Therefore, Vulkan hides them from
//! the user by default when calling `enumerate_physical_devices` on the instance. If there are no
//! conformant devices on the system, `Instance::new` will return an `IncompatibleDriver` error.
//!
//! In order to enumerate portability subset devices, you must set the
//! [`InstanceCreateFlags::ENUMERATE_PORTABILITY`] flag when creating the instance. However, if you
//! do this, your program must be prepared to handle the non-conformant aspects of these devices,
//! and must enable the appropriate features when creating the `Device` if you intend to use them.

use self::debug::{
    DebugUtilsMessengerCallback, DebugUtilsMessengerCreateInfo, ValidationFeatureDisable,
    ValidationFeatureEnable,
};
pub use self::layers::LayerProperties;
use crate::{
    cache::WeakArcOnceCache,
    device::physical::{
        PhysicalDevice, PhysicalDeviceGroupProperties, PhysicalDeviceGroupPropertiesRaw,
    },
    instance::debug::trampoline,
    macros::{impl_id_counter, vulkan_bitflags},
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, VulkanError, VulkanLibrary,
    VulkanObject,
};
pub use crate::{fns::InstanceFunctions, version::Version};
use ash::vk::Handle;
use parking_lot::RwLock;
use smallvec::SmallVec;
use std::{
    borrow::Cow,
    ffi::{c_void, CString},
    fmt::{Debug, Error as FmtError, Formatter},
    mem::MaybeUninit,
    num::NonZeroU64,
    ops::Deref,
    panic::{RefUnwindSafe, UnwindSafe},
    ptr, slice,
    sync::Arc,
};

pub mod debug;
mod layers;

// Generated by build.rs
include!(concat!(env!("OUT_DIR"), "/instance_extensions.rs"));

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
    id: NonZeroU64,

    flags: InstanceCreateFlags,
    api_version: Version,
    enabled_extensions: InstanceExtensions,
    enabled_layers: Vec<String>,
    library: Arc<VulkanLibrary>,
    max_api_version: Version,
    _user_callbacks: Vec<Arc<DebugUtilsMessengerCallback>>,

    physical_devices: WeakArcOnceCache<ash::vk::PhysicalDevice, PhysicalDevice>,
    physical_device_groups: RwLock<(bool, Vec<PhysicalDeviceGroupPropertiesRaw>)>,
}

// TODO: fix the underlying cause instead
impl UnwindSafe for Instance {}
impl RefUnwindSafe for Instance {}

impl Instance {
    /// Creates a new `Instance`.
    #[inline]
    pub fn new(
        library: Arc<VulkanLibrary>,
        mut create_info: InstanceCreateInfo,
    ) -> Result<Arc<Instance>, Validated<VulkanError>> {
        create_info.max_api_version.get_or_insert_with(|| {
            let api_version = library.api_version();
            if api_version < Version::V1_1 {
                api_version
            } else {
                Version::HEADER_VERSION
            }
        });

        Self::validate_new(&library, &create_info)?;

        unsafe { Ok(Self::new_unchecked(library, create_info)?) }
    }

    fn validate_new(
        library: &VulkanLibrary,
        create_info: &InstanceCreateInfo,
    ) -> Result<(), Box<ValidationError>> {
        // VUID-vkCreateInstance-pCreateInfo-parameter
        create_info
            .validate(library)
            .map_err(|err| err.add_context("create_info"))?;

        let &InstanceCreateInfo {
            flags: _,
            application_name: _,
            application_version: _,
            engine_name: _,
            engine_version: _,
            max_api_version,
            ref enabled_layers,
            ref enabled_extensions,
            debug_utils_messengers: _,
            enabled_validation_features: _,
            disabled_validation_features: _,
            _ne,
        } = create_info;

        let api_version = std::cmp::min(max_api_version.unwrap_or_default(), library.api_version());
        let supported_extensions = library
            .supported_extensions_with_layers(enabled_layers.iter().map(String::as_str))
            .unwrap();

        enabled_extensions
            .check_requirements(&supported_extensions, api_version)
            .map_err(|err| ValidationError {
                context: "create_info.enabled_extensions".into(),
                problem: err.to_string().into(),
                vuids: &["VUID-vkCreateInstance-ppEnabledExtensionNames-01388"],
                ..Default::default()
            })?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        library: Arc<VulkanLibrary>,
        mut create_info: InstanceCreateInfo,
    ) -> Result<Arc<Instance>, VulkanError> {
        create_info.max_api_version.get_or_insert_with(|| {
            let api_version = library.api_version();
            if api_version < Version::V1_1 {
                api_version
            } else {
                Version::HEADER_VERSION
            }
        });
        create_info.enabled_extensions.enable_dependencies(
            std::cmp::min(
                create_info.max_api_version.unwrap_or_default(),
                library.api_version(),
            ),
            &library
                .supported_extensions_with_layers(
                    create_info.enabled_layers.iter().map(String::as_str),
                )
                .unwrap(),
        );

        let &InstanceCreateInfo {
            mut flags,
            ref application_name,
            application_version,
            ref engine_name,
            engine_version,
            max_api_version,
            ref enabled_layers,
            ref enabled_extensions,
            ref debug_utils_messengers,
            ref enabled_validation_features,
            ref disabled_validation_features,
            _ne: _,
        } = &create_info;

        let mut enabled_extensions = Cow::Borrowed(enabled_extensions);

        if flags.intersects(InstanceCreateFlags::ENUMERATE_PORTABILITY) {
            // VUID-VkInstanceCreateInfo-flags-06559
            if library
                .supported_extensions_with_layers(enabled_layers.iter().map(String::as_str))?
                .khr_portability_enumeration
            {
                enabled_extensions.to_mut().khr_portability_enumeration = true;
            } else {
                flags -= InstanceCreateFlags::ENUMERATE_PORTABILITY;
            }
        }

        let enabled_layers_vk: Vec<CString> = enabled_layers
            .iter()
            .map(|name| CString::new(name.clone()).unwrap())
            .collect();
        let enabled_layers_ptrs_vk = enabled_layers_vk
            .iter()
            .map(|layer| layer.as_ptr())
            .collect::<SmallVec<[_; 2]>>();

        let enabled_extensions_vk: Vec<CString> = enabled_extensions.as_ref().into();
        let enabled_extensions_ptrs_vk = enabled_extensions_vk
            .iter()
            .map(|extension| extension.as_ptr())
            .collect::<SmallVec<[_; 2]>>();

        let application_name_vk = application_name
            .as_ref()
            .map(|name| CString::new(name.clone()).unwrap());
        let engine_name_vk = engine_name
            .as_ref()
            .map(|name| CString::new(name.clone()).unwrap());
        let application_info_vk = ash::vk::ApplicationInfo {
            p_application_name: application_name_vk
                .as_ref()
                .map(|s| s.as_ptr())
                .unwrap_or(ptr::null()),
            application_version: application_version
                .try_into()
                .expect("Version out of range"),
            p_engine_name: engine_name_vk
                .as_ref()
                .map(|s| s.as_ptr())
                .unwrap_or(ptr::null()),
            engine_version: engine_version.try_into().expect("Version out of range"),
            api_version: max_api_version
                .unwrap()
                .try_into()
                .expect("Version out of range"),
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
            flags: flags.into(),
            p_application_info: &application_info_vk,
            enabled_layer_count: enabled_layers_ptrs_vk.len() as u32,
            pp_enabled_layer_names: enabled_layers_ptrs_vk.as_ptr(),
            enabled_extension_count: enabled_extensions_ptrs_vk.len() as u32,
            pp_enabled_extension_names: enabled_extensions_ptrs_vk.as_ptr(),
            ..Default::default()
        };
        let mut validation_features_vk = None;

        if !enabled_validation_features.is_empty() || !disabled_validation_features.is_empty() {
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

        let mut debug_utils_messenger_create_infos_vk: Vec<_> = debug_utils_messengers
            .iter()
            .map(|create_info| {
                let &DebugUtilsMessengerCreateInfo {
                    message_type,
                    message_severity,
                    ref user_callback,
                    _ne: _,
                } = create_info;

                ash::vk::DebugUtilsMessengerCreateInfoEXT {
                    flags: ash::vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
                    message_severity: message_severity.into(),
                    message_type: message_type.into(),
                    pfn_user_callback: Some(trampoline),
                    p_user_data: user_callback.as_ptr() as *const c_void as *mut _,
                    ..Default::default()
                }
            })
            .collect();

        for i in 1..debug_utils_messenger_create_infos_vk.len() {
            debug_utils_messenger_create_infos_vk[i - 1].p_next =
                &debug_utils_messenger_create_infos_vk[i] as *const _ as *const _;
        }

        if let Some(info) = debug_utils_messenger_create_infos_vk.first() {
            create_info_vk.p_next = info as *const _ as *const _;
        }

        let handle = {
            let mut output = MaybeUninit::uninit();
            let fns = library.fns();
            (fns.v1_0.create_instance)(&create_info_vk, ptr::null(), output.as_mut_ptr())
                .result()
                .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Self::from_handle(library, handle, create_info))
    }

    /// Creates a new `Instance` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `library`.
    /// - `create_info` must match the info used to create the object.
    pub unsafe fn from_handle(
        library: Arc<VulkanLibrary>,
        handle: ash::vk::Instance,
        mut create_info: InstanceCreateInfo,
    ) -> Arc<Self> {
        create_info.max_api_version.get_or_insert_with(|| {
            let api_version = library.api_version();
            if api_version < Version::V1_1 {
                api_version
            } else {
                Version::HEADER_VERSION
            }
        });

        let InstanceCreateInfo {
            flags,
            application_name: _,
            application_version: _,
            engine_name: _,
            engine_version: _,
            max_api_version,
            enabled_layers,
            enabled_extensions,
            debug_utils_messengers,
            enabled_validation_features: _,
            disabled_validation_features: _,
            _ne: _,
        } = create_info;

        let max_api_version = max_api_version.unwrap();
        let api_version = std::cmp::min(max_api_version, library.api_version());

        Arc::new(Instance {
            handle,
            fns: InstanceFunctions::load(|name| {
                library
                    .get_instance_proc_addr(handle, name.as_ptr())
                    .map_or(ptr::null(), |func| func as _)
            }),
            id: Self::next_id(),

            flags,
            api_version,
            enabled_extensions,
            enabled_layers,
            library,
            max_api_version,
            _user_callbacks: debug_utils_messengers
                .into_iter()
                .map(|m| m.user_callback)
                .collect(),

            physical_devices: WeakArcOnceCache::new(),
            physical_device_groups: RwLock::new((false, Vec::new())),
        })
    }

    /// Returns the Vulkan library used to create this instance.
    #[inline]
    pub fn library(&self) -> &Arc<VulkanLibrary> {
        &self.library
    }

    /// Returns the flags that the instance was created with.
    #[inline]
    pub fn flags(&self) -> InstanceCreateFlags {
        self.flags
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
    ///
    /// This includes both the extensions specified in [`InstanceCreateInfo::enabled_extensions`],
    /// and any extensions that are required by those extensions.
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
                .map(|handle| {
                    self.physical_devices.get_or_try_insert(handle, |&handle| {
                        PhysicalDevice::from_handle(self.clone(), handle)
                    })
                })
                .collect::<Result<_, _>>()?;

            Ok(physical_devices.into_iter())
        }
    }

    /// Returns an iterator that enumerates the groups of physical devices available. All
    /// physical devices in a group can be used to create a single logical device. They are
    /// guaranteed have the same [properties], and support the same [extensions] and [features].
    ///
    /// Every physical device will be returned exactly once;
    /// physical devices that are not part of any group will be returned as a group of size 1.
    ///
    /// The instance API version must be at least 1.1, or the [`khr_device_group_creation`]
    /// extension must be enabled on the instance.
    ///
    /// [properties]: PhysicalDevice::properties
    /// [extensions]: PhysicalDevice::supported_extensions
    /// [features]: PhysicalDevice::supported_features
    /// [`enumerate_physical_devices`]: Self::enumerate_physical_devices
    /// [`khr_device_group_creation`]: crate::instance::InstanceExtensions::khr_device_group_creation
    #[inline]
    pub fn enumerate_physical_device_groups(
        self: &Arc<Self>,
    ) -> Result<impl ExactSizeIterator<Item = PhysicalDeviceGroupProperties>, Validated<VulkanError>>
    {
        self.validate_enumerate_physical_device_groups()?;

        unsafe { Ok(self.enumerate_physical_device_groups_unchecked()?) }
    }

    fn validate_enumerate_physical_device_groups(&self) -> Result<(), Box<ValidationError>> {
        if !(self.api_version() >= Version::V1_1
            || self.enabled_extensions().khr_device_group_creation)
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                    RequiresAllOf(&[Requires::InstanceExtension("khr_device_group_creation")]),
                ]),
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn enumerate_physical_device_groups_unchecked(
        self: &Arc<Self>,
    ) -> Result<impl ExactSizeIterator<Item = PhysicalDeviceGroupProperties>, VulkanError> {
        let fns = self.fns();
        let enumerate_physical_device_groups = if self.api_version() >= Version::V1_1 {
            fns.v1_1.enumerate_physical_device_groups
        } else {
            fns.khr_device_group_creation
                .enumerate_physical_device_groups_khr
        };

        let properties_vk = loop {
            let mut count = 0;

            enumerate_physical_device_groups(self.handle, &mut count, ptr::null_mut())
                .result()
                .map_err(VulkanError::from)?;

            let mut properties = Vec::with_capacity(count as usize);
            let result =
                enumerate_physical_device_groups(self.handle, &mut count, properties.as_mut_ptr());

            match result {
                ash::vk::Result::SUCCESS => {
                    properties.set_len(count as usize);
                    break properties;
                }
                ash::vk::Result::INCOMPLETE => (),
                err => return Err(VulkanError::from(err)),
            }
        };

        let mut properties: SmallVec<[_; 4]> = SmallVec::with_capacity(properties_vk.len());
        let mut properties_raw: Vec<_> = Vec::with_capacity(properties_vk.len());

        for properties_vk in properties_vk {
            let &ash::vk::PhysicalDeviceGroupProperties {
                physical_device_count,
                physical_devices,
                subset_allocation,
                ..
            } = &properties_vk;

            properties.push(PhysicalDeviceGroupProperties {
                physical_devices: physical_devices[..physical_device_count as usize]
                    .iter()
                    .map(|&handle| {
                        self.physical_devices.get_or_try_insert(handle, |&handle| {
                            PhysicalDevice::from_handle(self.clone(), handle)
                        })
                    })
                    .collect::<Result<_, _>>()?,
                subset_allocation: subset_allocation != ash::vk::FALSE,
            });
            properties_raw.push(PhysicalDeviceGroupPropertiesRaw {
                physical_device_count,
                physical_devices,
                subset_allocation,
            });
        }

        *self.physical_device_groups.write() = (true, properties_raw);

        Ok(properties.into_iter())
    }

    /// Returns whether the given physical devices all belong to the same device group.
    ///
    /// Returns `false` if `physical_devices` is empty.
    pub fn is_same_device_group<'a>(
        self: &Arc<Self>,
        physical_devices: impl IntoIterator<Item = &'a PhysicalDevice>,
    ) -> bool {
        let mut physical_devices = physical_devices.into_iter();
        let first = match physical_devices.next() {
            Some(x) => x,
            None => return false,
        };

        if first.instance() != self {
            return false;
        }

        let lock = {
            let lock = self.physical_device_groups.read();

            if lock.0 {
                lock
            } else {
                drop(lock);

                if self.enumerate_physical_device_groups().is_err() {
                    return false;
                }

                self.physical_device_groups.read()
            }
        };

        lock.1
            .iter()
            // First find the group that contains the first physical device...
            .find_map(|properties_raw| {
                let group = &properties_raw.physical_devices
                    [..properties_raw.physical_device_count as usize];
                group.contains(&first.handle()).then_some(group)
            })
            // ...then check if all the remaining physical devices belong to that group too.
            .map_or(false, |group| {
                physical_devices.all(|physical_device| {
                    physical_device.instance() == self && group.contains(&physical_device.handle())
                })
            })
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

impl_id_counter!(Instance);

impl Debug for Instance {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        let Self {
            handle,
            fns,
            id,

            flags,
            api_version,
            enabled_extensions,
            enabled_layers,
            library,
            max_api_version,
            _user_callbacks: _,

            physical_devices: _,
            physical_device_groups: _,
        } = self;

        f.debug_struct("Instance")
            .field("handle", handle)
            .field("fns", fns)
            .field("id", id)
            .field("flags", flags)
            .field("api_version", api_version)
            .field("enabled_extensions", enabled_extensions)
            .field("enabled_layers", enabled_layers)
            .field("library", library)
            .field("max_api_version", max_api_version)
            .finish_non_exhaustive()
    }
}

/// Parameters to create a new `Instance`.
#[derive(Debug)]
pub struct InstanceCreateInfo {
    /// Additional properties of the instance.
    ///
    /// The default value is empty.
    pub flags: InstanceCreateFlags,

    /// A string of your choice stating the name of your application.
    ///
    /// The default value is `None`.
    pub application_name: Option<String>,

    /// A version number of your choice specifying the version of your application.
    ///
    /// The default value is zero.
    pub application_version: Version,

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

    /// The layers to enable on the instance.
    ///
    /// The default value is empty.
    pub enabled_layers: Vec<String>,

    /// The extensions to enable on the instance.
    ///
    /// You only need to enable the extensions that you need. If the extensions you specified
    /// require additional extensions to be enabled, they will be automatically enabled as well.
    ///
    /// The default value is [`InstanceExtensions::empty()`].
    pub enabled_extensions: InstanceExtensions,

    /// Creation parameters for debug messengers,
    /// to use during the creation and destruction of the instance.
    ///
    /// The debug messengers are not used at any other time,
    /// [`DebugUtilsMessenger`](crate::instance::debug::DebugUtilsMessenger) should be used for
    /// that.
    ///
    /// If this is not empty, the `ext_debug_utils` extension must be set in `enabled_extensions`.
    ///
    /// The default value is empty.
    pub debug_utils_messengers: Vec<DebugUtilsMessengerCreateInfo>,

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
            flags: InstanceCreateFlags::empty(),
            application_name: None,
            application_version: Version::major_minor(0, 0),
            engine_name: None,
            engine_version: Version::major_minor(0, 0),
            max_api_version: None,
            enabled_layers: Vec::new(),
            enabled_extensions: InstanceExtensions::empty(),
            debug_utils_messengers: Vec::new(),
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

    pub(crate) fn validate(&self, library: &VulkanLibrary) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            application_name: _,
            application_version: _,
            engine_name: _,
            engine_version: _,
            max_api_version,
            enabled_layers: _,
            ref enabled_extensions,
            ref debug_utils_messengers,
            ref enabled_validation_features,
            ref disabled_validation_features,
            _ne: _,
        } = self;

        let max_api_version = max_api_version.unwrap_or_default();
        let api_version = std::cmp::min(max_api_version, library.api_version());

        if max_api_version < Version::V1_0 {
            return Err(Box::new(ValidationError {
                context: "max_api_version".into(),
                problem: "is less than 1.0".into(),
                vuids: &["VUID-VkApplicationInfo-apiVersion-04010"],
                ..Default::default()
            }));
        }

        flags
            .validate_instance_raw(api_version, enabled_extensions)
            .map_err(|err| ValidationError {
                context: "flags".into(),
                vuids: &["VUID-VkInstanceCreateInfo-flags-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        if !debug_utils_messengers.is_empty() {
            if !enabled_extensions.ext_debug_utils {
                return Err(Box::new(ValidationError {
                    context: "debug_utils_messengers".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                        Requires::InstanceExtension("ext_debug_utils"),
                    ])]),
                    vuids: &["VUID-VkInstanceCreateInfo-pNext-04926"],
                }));
            }

            for (index, messenger_create_info) in debug_utils_messengers.iter().enumerate() {
                messenger_create_info
                    .validate_raw(api_version, enabled_extensions)
                    .map_err(|err| err.add_context(format!("debug_utils_messengers[{}]", index)))?;
            }
        }

        if !enabled_validation_features.is_empty() {
            if !enabled_extensions.ext_validation_features {
                return Err(Box::new(ValidationError {
                    context: "enabled_validation_features".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                        Requires::InstanceExtension("ext_validation_features"),
                    ])]),
                    ..Default::default()
                }));
            }

            for (index, enabled) in enabled_validation_features.iter().enumerate() {
                enabled
                    .validate_instance_raw(api_version, enabled_extensions)
                    .map_err(|err| ValidationError {
                        context: format!("enabled_validation_features[{}]", index).into(),
                        vuids: &[
                            "VUID-VkValidationFeaturesEXT-pEnabledValidationFeatures-parameter",
                        ],
                        ..ValidationError::from_requirement(err)
                    })?;
            }

            if enabled_validation_features
                .contains(&ValidationFeatureEnable::GpuAssistedReserveBindingSlot)
                && !enabled_validation_features.contains(&ValidationFeatureEnable::GpuAssisted)
            {
                return Err(Box::new(ValidationError {
                    context: "enabled_validation_features".into(),
                    problem: "contains `ValidationFeatureEnable::GpuAssistedReserveBindingSlot`, \
                        but does not also contain \
                        `ValidationFeatureEnable::GpuAssisted`"
                        .into(),
                    vuids: &["VUID-VkValidationFeaturesEXT-pEnabledValidationFeatures-02967"],
                    ..Default::default()
                }));
            }

            if enabled_validation_features.contains(&ValidationFeatureEnable::DebugPrintf)
                && enabled_validation_features.contains(&ValidationFeatureEnable::GpuAssisted)
            {
                return Err(Box::new(ValidationError {
                    context: "enabled_validation_features".into(),
                    problem: "contains both `ValidationFeatureEnable::DebugPrintf` and \
                        `ValidationFeatureEnable::GpuAssisted`"
                        .into(),
                    vuids: &["VUID-VkValidationFeaturesEXT-pEnabledValidationFeatures-02968"],
                    ..Default::default()
                }));
            }
        }

        if !disabled_validation_features.is_empty() {
            if !enabled_extensions.ext_validation_features {
                return Err(Box::new(ValidationError {
                    context: "disabled_validation_features".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                        Requires::InstanceExtension("ext_validation_features"),
                    ])]),
                    ..Default::default()
                }));
            }

            for (index, disabled) in disabled_validation_features.iter().enumerate() {
                disabled
                    .validate_instance_raw(api_version, enabled_extensions)
                    .map_err(|err| ValidationError {
                        context: format!("disabled_validation_features[{}]", index).into(),
                        vuids: &[
                            "VUID-VkValidationFeaturesEXT-pDisabledValidationFeatures-parameter",
                        ],
                        ..ValidationError::from_requirement(err)
                    })?;
            }
        }

        Ok(())
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of an instance.
    InstanceCreateFlags = InstanceCreateFlags(u32);

    /// Include [portability subset] devices when enumerating physical devices.
    ///
    /// If you enable this flag, you must ensure that your program is prepared to handle the
    /// non-conformant aspects of these devices.
    ///
    /// If this flag is not enabled, and there are no fully-conformant devices on the system, then
    /// [`Instance::new`] will return an `IncompatibleDriver` error.
    ///
    /// The default value is `false`.
    ///
    /// # Notes
    ///
    /// If this flag is enabled, and the [`khr_portability_enumeration`] extension is supported,
    /// it will be enabled automatically when creating the instance.
    /// If the extension is not supported, this flag will be ignored.
    ///
    /// [portability subset]: crate::instance#portability-subset-devices-and-the-enumerate_portability-flag
    /// [`khr_portability_enumeration`]: crate::instance::InstanceExtensions::khr_portability_enumeration
    ENUMERATE_PORTABILITY = ENUMERATE_PORTABILITY_KHR,
}

/// Implemented on objects that belong to a Vulkan instance.
///
/// # Safety
///
/// - `instance()` must return the correct instance.
pub unsafe trait InstanceOwned {
    /// Returns the instance that owns `self`.
    fn instance(&self) -> &Arc<Instance>;
}

unsafe impl<T> InstanceOwned for T
where
    T: Deref,
    T::Target: InstanceOwned,
{
    fn instance(&self) -> &Arc<Instance> {
        (**self).instance()
    }
}

/// Same as [`DebugWrapper`], but also prints the instance handle for disambiguation.
///
/// [`DebugWrapper`]: crate:: DebugWrapper
#[derive(PartialEq, Eq)]
#[repr(transparent)]
pub(crate) struct InstanceOwnedDebugWrapper<T>(pub(crate) T);

impl<T> InstanceOwnedDebugWrapper<T> {
    pub fn cast_slice_inner(slice: &[Self]) -> &[T] {
        // SAFETY: `InstanceOwnedDebugWrapper<T>` and `T` have the same layout.
        unsafe { slice::from_raw_parts(slice as *const _ as *const _, slice.len()) }
    }
}

impl<T> Debug for InstanceOwnedDebugWrapper<T>
where
    T: VulkanObject + InstanceOwned,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(
            f,
            "0x{:x} (instance: 0x{:x})",
            self.handle().as_raw(),
            self.instance().handle().as_raw(),
        )
    }
}

impl<T> Deref for InstanceOwnedDebugWrapper<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use crate::instance::InstanceExtensions;
    use std::ffi::CString;

    #[test]
    fn empty_extensions() {
        let i: Vec<CString> = (&InstanceExtensions::empty()).into();
        assert!(i.get(0).is_none());
    }

    #[test]
    fn into_iter() {
        let extensions = InstanceExtensions {
            khr_display: true,
            ..InstanceExtensions::empty()
        };
        for (name, enabled) in extensions {
            if name == "VK_KHR_display" {
                assert!(enabled);
            } else {
                assert!(!enabled);
            }
        }
    }

    #[test]
    fn create_instance() {
        let _ = instance!();
    }
}
