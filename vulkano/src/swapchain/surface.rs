// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{FullScreenExclusive, PresentGravityFlags, PresentScalingFlags, Win32Monitor};
use crate::{
    cache::OnceCache,
    device::physical::PhysicalDevice,
    display::{DisplayMode, DisplayPlaneAlpha},
    format::Format,
    image::ImageUsage,
    instance::{Instance, InstanceExtensions, InstanceOwned},
    macros::{impl_id_counter, vulkan_bitflags_enum, vulkan_enum},
    DebugWrapper, Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, VulkanError,
    VulkanObject,
};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use objc::{class, msg_send, runtime::Object, sel, sel_impl};
use raw_window_handle::{
    HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
};
use smallvec::SmallVec;
use std::{
    any::Any,
    error::Error,
    fmt::{Debug, Display, Error as FmtError, Formatter},
    mem::MaybeUninit,
    num::NonZeroU64,
    ptr,
    sync::Arc,
};

/// Represents a surface on the screen.
///
/// Creating a `Surface` is platform-specific.
pub struct Surface {
    handle: ash::vk::SurfaceKHR,
    instance: DebugWrapper<Arc<Instance>>,
    id: NonZeroU64,
    api: SurfaceApi,
    object: Option<Arc<dyn Any + Send + Sync>>,
    // FIXME: This field is never set.
    #[cfg(target_os = "ios")]
    metal_layer: IOSMetalLayer,

    // Data queried by the user at runtime, cached for faster lookups.
    // This is stored here rather than on `PhysicalDevice` to ensure that it's freed when the
    // `Surface` is destroyed.
    pub(crate) surface_formats:
        OnceCache<(ash::vk::PhysicalDevice, SurfaceInfo), Vec<(Format, ColorSpace)>>,
    pub(crate) surface_present_modes:
        OnceCache<(ash::vk::PhysicalDevice, SurfaceInfo), Vec<PresentMode>>,
    pub(crate) surface_support: OnceCache<(ash::vk::PhysicalDevice, u32), bool>,
}

impl Surface {
    /// Returns the instance extensions required to create a surface from a window of the given
    /// event loop.
    pub fn required_extensions(
        event_loop: &impl HasDisplayHandle,
    ) -> Result<InstanceExtensions, HandleError> {
        let mut extensions = InstanceExtensions {
            khr_surface: true,
            ..InstanceExtensions::empty()
        };
        match event_loop.display_handle()?.as_raw() {
            RawDisplayHandle::Android(_) => extensions.khr_android_surface = true,
            // FIXME: `mvk_macos_surface` and `mvk_ios_surface` are deprecated.
            RawDisplayHandle::AppKit(_) => extensions.mvk_macos_surface = true,
            RawDisplayHandle::UiKit(_) => extensions.mvk_ios_surface = true,
            RawDisplayHandle::Windows(_) => extensions.khr_win32_surface = true,
            RawDisplayHandle::Wayland(_) => extensions.khr_wayland_surface = true,
            RawDisplayHandle::Xcb(_) => extensions.khr_xcb_surface = true,
            RawDisplayHandle::Xlib(_) => extensions.khr_xlib_surface = true,
            _ => unimplemented!(),
        }

        Ok(extensions)
    }

    /// Creates a new `Surface` from the given `window`.
    pub fn from_window(
        instance: Arc<Instance>,
        window: Arc<impl HasWindowHandle + HasDisplayHandle + Any + Send + Sync>,
    ) -> Result<Arc<Self>, FromWindowError> {
        let mut surface = unsafe { Self::from_window_ref(instance, &*window) }?;
        Arc::get_mut(&mut surface).unwrap().object = Some(window);

        Ok(surface)
    }

    /// Creates a new `Surface` from the given `window` without ensuring that the window outlives
    /// the surface.
    ///
    /// # Safety
    ///
    /// - The given `window` must outlive the created surface.
    pub unsafe fn from_window_ref(
        instance: Arc<Instance>,
        window: &(impl HasWindowHandle + HasDisplayHandle),
    ) -> Result<Arc<Self>, FromWindowError> {
        let window_handle = window
            .window_handle()
            .map_err(FromWindowError::RetrieveHandle)?;
        let display_handle = window
            .display_handle()
            .map_err(FromWindowError::RetrieveHandle)?;

        match (window_handle.as_raw(), display_handle.as_raw()) {
            (RawWindowHandle::AndroidNdk(window), RawDisplayHandle::Android(_display)) => {
                Self::from_android(instance, window.a_native_window.as_ptr(), None)
            }
            #[cfg(target_os = "macos")]
            (RawWindowHandle::AppKit(window), RawDisplayHandle::AppKit(_display)) => {
                // Ensure the layer is `CAMetalLayer`.
                let layer = get_metal_layer_macos(window.ns_view.as_ptr());

                Self::from_mac_os(instance, layer as *const (), None)
            }
            #[cfg(target_os = "ios")]
            (RawWindowHandle::UiKit(window), RawDisplayHandle::UiKit(_display)) => {
                // Ensure the layer is `CAMetalLayer`.
                let layer = get_metal_layer_ios(window.ui_view.as_ptr());

                Self::from_ios(instance, layer, None)
            }
            (RawWindowHandle::Wayland(window), RawDisplayHandle::Wayland(display)) => {
                Self::from_wayland(
                    instance,
                    display.display.as_ptr(),
                    window.surface.as_ptr(),
                    None,
                )
            }
            (RawWindowHandle::Win32(window), RawDisplayHandle::Windows(_display)) => {
                Self::from_win32(
                    instance,
                    window.hinstance.unwrap().get() as *const (),
                    window.hwnd.get() as *const (),
                    None,
                )
            }
            (RawWindowHandle::Xcb(window), RawDisplayHandle::Xcb(display)) => Self::from_xcb(
                instance,
                display.connection.unwrap().as_ptr(),
                window.window.get(),
                None,
            ),
            (RawWindowHandle::Xlib(window), RawDisplayHandle::Xlib(display)) => Self::from_xlib(
                instance,
                display.display.unwrap().as_ptr(),
                window.window,
                None,
            ),
            _ => unimplemented!(
                "the window was created with a windowing API that is not supported \
                by Vulkan/Vulkano"
            ),
        }
        .map_err(FromWindowError::CreateSurface)
    }

    /// Creates a `Surface` from a raw handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `instance`.
    /// - `handle` must have been created using the function specified by `api`.
    /// - The window object that `handle` was created from must outlive the created `Surface`.
    ///   The `object` parameter can be used to ensure this.
    pub unsafe fn from_handle(
        instance: Arc<Instance>,
        handle: ash::vk::SurfaceKHR,
        api: SurfaceApi,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Self {
        Surface {
            handle,
            instance: DebugWrapper(instance),
            id: Self::next_id(),
            api,
            object,
            #[cfg(target_os = "ios")]
            metal_layer: IOSMetalLayer::new(std::ptr::null_mut(), std::ptr::null_mut()),
            surface_formats: OnceCache::new(),
            surface_present_modes: OnceCache::new(),
            surface_support: OnceCache::new(),
        }
    }

    /// Creates a `Surface` with no backing window or display.
    ///
    /// Presenting to a headless surface does nothing, so this is mostly useless in itself. However,
    /// it may be useful for testing, and it is available for future extensions to layer on top of.
    pub fn headless(
        instance: Arc<Instance>,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_headless(&instance)?;

        unsafe { Ok(Self::headless_unchecked(instance, object)?) }
    }

    fn validate_headless(instance: &Instance) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().ext_headless_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_headless_surface",
                )])]),
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn headless_unchecked(
        instance: Arc<Instance>,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info = ash::vk::HeadlessSurfaceCreateInfoEXT {
            flags: ash::vk::HeadlessSurfaceCreateFlagsEXT::empty(),
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            (fns.ext_headless_surface.create_headless_surface_ext)(
                instance.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Arc::new(Self::from_handle(
            instance,
            handle,
            SurfaceApi::Headless,
            object,
        )))
    }

    /// Creates a `Surface` from a `DisplayMode` and display plane.
    #[inline]
    pub fn from_display_plane(
        display_mode: Arc<DisplayMode>,
        create_info: DisplaySurfaceCreateInfo,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_display_plane(&display_mode, &create_info)?;

        unsafe {
            Ok(Self::from_display_plane_unchecked(
                display_mode,
                create_info,
            )?)
        }
    }

    fn validate_from_display_plane(
        display_mode: &DisplayMode,
        create_info: &DisplaySurfaceCreateInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !display_mode.instance().enabled_extensions().khr_display {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_display",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-vkCreateDisplayPlaneSurfaceKHR-pCreateInfo-parameter
        create_info
            .validate(display_mode.display().physical_device())
            .map_err(|err| err.add_context("create_info"))?;

        let &DisplaySurfaceCreateInfo {
            plane_index,
            plane_stack_index,
            transform,
            alpha_mode,
            global_alpha: _,
            image_extent: _,
            _ne: _,
        } = create_info;

        let display_plane_properties_raw = unsafe {
            display_mode
                .display()
                .physical_device()
                .display_plane_properties_raw()
                .map_err(|_err| {
                    Box::new(ValidationError {
                        problem: "`PhysicalDevice::display_plane_properties` \
                            returned an error"
                            .into(),
                        ..Default::default()
                    })
                })?
        };

        if display_mode.display().plane_reorder_possible() {
            if plane_stack_index as usize >= display_plane_properties_raw.len() {
                return Err(Box::new(ValidationError {
                    problem: "`plane_stack_index` is not less than the number of display planes \
                        on the physical device"
                        .into(),
                    vuids: &["VUID-VkDisplaySurfaceCreateInfoKHR-planeReorderPossible-01253"],
                    ..Default::default()
                }));
            }
        } else {
            if plane_stack_index
                != display_plane_properties_raw[plane_index as usize].current_stack_index
            {
                return Err(Box::new(ValidationError {
                    problem: "`display_mode.display().plane_reorder_possible()` is `false`, and \
                        `plane_stack_index` does not equal the `current_stack_index` value of the \
                        display plane properties"
                        .into(),
                    vuids: &["VUID-VkDisplaySurfaceCreateInfoKHR-planeReorderPossible-01253"],
                    ..Default::default()
                }));
            }
        }

        let display_plane_capabilities = unsafe {
            display_mode
                .display_plane_capabilities_unchecked(plane_index)
                .map_err(|_err| {
                    Box::new(ValidationError {
                        problem: "`DisplayMode::display_plane_capabilities` \
                            returned an error"
                            .into(),
                        ..Default::default()
                    })
                })?
        };

        if !display_plane_capabilities
            .supported_alpha
            .contains_enum(alpha_mode)
        {
            return Err(Box::new(ValidationError {
                problem: "the `supported_alpha` value of the display plane capabilities of \
                    `display_mode` for the provided plane index does not contain \
                    `create_info.alpha_mode`"
                    .into(),
                vuids: &["VUID-VkDisplaySurfaceCreateInfoKHR-alphaMode-01255"],
                ..Default::default()
            }));
        }

        if !display_mode
            .display()
            .supported_transforms()
            .contains_enum(transform)
        {
            return Err(Box::new(ValidationError {
                problem: "`display_mode.display().supported_transforms()` does not contain \
                    `create_info.transform`"
                    .into(),
                vuids: &["VUID-VkDisplaySurfaceCreateInfoKHR-transform-06740"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_display_plane_unchecked(
        display_mode: Arc<DisplayMode>,
        create_info: DisplaySurfaceCreateInfo,
    ) -> Result<Arc<Self>, VulkanError> {
        let &DisplaySurfaceCreateInfo {
            plane_index,
            plane_stack_index,
            transform,
            alpha_mode,
            global_alpha,
            image_extent,
            _ne: _,
        } = &create_info;

        let create_info_vk = ash::vk::DisplaySurfaceCreateInfoKHR {
            flags: ash::vk::DisplaySurfaceCreateFlagsKHR::empty(),
            display_mode: display_mode.handle(),
            plane_index,
            plane_stack_index,
            transform: transform.into(),
            alpha_mode: alpha_mode.into(),
            global_alpha,
            image_extent: ash::vk::Extent2D {
                width: image_extent[0],
                height: image_extent[1],
            },
            ..Default::default()
        };

        let instance = display_mode.instance();

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            (fns.khr_display.create_display_plane_surface_khr)(
                instance.handle(),
                &create_info_vk,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Arc::new(Self::from_handle(
            instance.clone(),
            handle,
            SurfaceApi::DisplayPlane,
            None,
        )))
    }

    /// Creates a `Surface` from an Android window.
    ///
    /// # Safety
    ///
    /// - `window` must be a valid Android `ANativeWindow` handle.
    /// - The object referred to by `window` must outlive the created `Surface`.
    ///   The `object` parameter can be used to ensure this.
    pub unsafe fn from_android<W>(
        instance: Arc<Instance>,
        window: *const W,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_android(&instance, window)?;

        Ok(Self::from_android_unchecked(instance, window, object)?)
    }

    fn validate_from_android<W>(
        instance: &Instance,
        _window: *const W,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().khr_android_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_android_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkAndroidSurfaceCreateInfoKHR-window-01248
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_android_unchecked<W>(
        instance: Arc<Instance>,
        window: *const W,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info = ash::vk::AndroidSurfaceCreateInfoKHR {
            flags: ash::vk::AndroidSurfaceCreateFlagsKHR::empty(),
            window: window as *mut _,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            (fns.khr_android_surface.create_android_surface_khr)(
                instance.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Arc::new(Self::from_handle(
            instance,
            handle,
            SurfaceApi::Android,
            object,
        )))
    }

    /// Creates a `Surface` from a DirectFB surface.
    ///
    /// # Safety
    ///
    /// - `dfb` must be a valid DirectFB `IDirectFB` handle.
    /// - `surface` must be a valid DirectFB `IDirectFBSurface` handle.
    /// - The object referred to by `dfb` and `surface` must outlive the created `Surface`.
    ///   The `object` parameter can be used to ensure this.
    pub unsafe fn from_directfb<D, S>(
        instance: Arc<Instance>,
        dfb: *const D,
        surface: *const S,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_directfb(&instance, dfb, surface)?;

        Ok(Self::from_directfb_unchecked(
            instance, dfb, surface, object,
        )?)
    }

    fn validate_from_directfb<D, S>(
        instance: &Instance,
        _dfb: *const D,
        _surface: *const S,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().ext_directfb_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_directfb_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkDirectFBSurfaceCreateInfoEXT-dfb-04117
        // Can't validate, therefore unsafe

        // VUID-VkDirectFBSurfaceCreateInfoEXT-surface-04118
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_directfb_unchecked<D, S>(
        instance: Arc<Instance>,
        dfb: *const D,
        surface: *const S,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info = ash::vk::DirectFBSurfaceCreateInfoEXT {
            flags: ash::vk::DirectFBSurfaceCreateFlagsEXT::empty(),
            dfb: dfb as *mut _,
            surface: surface as *mut _,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            (fns.ext_directfb_surface.create_direct_fb_surface_ext)(
                instance.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Arc::new(Self::from_handle(
            instance,
            handle,
            SurfaceApi::DirectFB,
            object,
        )))
    }

    /// Creates a `Surface` from an Fuchsia ImagePipe.
    ///
    /// # Safety
    ///
    /// - `image_pipe_handle` must be a valid Fuchsia `zx_handle_t` handle.
    /// - The object referred to by `image_pipe_handle` must outlive the created `Surface`.
    ///   The `object` parameter can be used to ensure this.
    pub unsafe fn from_fuchsia_image_pipe(
        instance: Arc<Instance>,
        image_pipe_handle: ash::vk::zx_handle_t,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_fuchsia_image_pipe(&instance, image_pipe_handle)?;

        Ok(Self::from_fuchsia_image_pipe_unchecked(
            instance,
            image_pipe_handle,
            object,
        )?)
    }

    fn validate_from_fuchsia_image_pipe(
        instance: &Instance,
        _image_pipe_handle: ash::vk::zx_handle_t,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().fuchsia_imagepipe_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "fuchsia_imagepipe_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkImagePipeSurfaceCreateInfoFUCHSIA-imagePipeHandle-04863
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_fuchsia_image_pipe_unchecked(
        instance: Arc<Instance>,
        image_pipe_handle: ash::vk::zx_handle_t,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info = ash::vk::ImagePipeSurfaceCreateInfoFUCHSIA {
            flags: ash::vk::ImagePipeSurfaceCreateFlagsFUCHSIA::empty(),
            image_pipe_handle,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            (fns.fuchsia_imagepipe_surface
                .create_image_pipe_surface_fuchsia)(
                instance.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Arc::new(Self::from_handle(
            instance,
            handle,
            SurfaceApi::FuchsiaImagePipe,
            object,
        )))
    }

    /// Creates a `Surface` from a Google Games Platform stream descriptor.
    ///
    /// # Safety
    ///
    /// - `stream_descriptor` must be a valid Google Games Platform `GgpStreamDescriptor` handle.
    /// - The object referred to by `stream_descriptor` must outlive the created `Surface`.
    ///   The `object` parameter can be used to ensure this.
    pub unsafe fn from_ggp_stream_descriptor(
        instance: Arc<Instance>,
        stream_descriptor: ash::vk::GgpStreamDescriptor,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_ggp_stream_descriptor(&instance, stream_descriptor)?;

        Ok(Self::from_ggp_stream_descriptor_unchecked(
            instance,
            stream_descriptor,
            object,
        )?)
    }

    fn validate_from_ggp_stream_descriptor(
        instance: &Instance,
        _stream_descriptor: ash::vk::GgpStreamDescriptor,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().ggp_stream_descriptor_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ggp_stream_descriptor_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkStreamDescriptorSurfaceCreateInfoGGP-streamDescriptor-02681
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_ggp_stream_descriptor_unchecked(
        instance: Arc<Instance>,
        stream_descriptor: ash::vk::GgpStreamDescriptor,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info = ash::vk::StreamDescriptorSurfaceCreateInfoGGP {
            flags: ash::vk::StreamDescriptorSurfaceCreateFlagsGGP::empty(),
            stream_descriptor,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            (fns.ggp_stream_descriptor_surface
                .create_stream_descriptor_surface_ggp)(
                instance.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Arc::new(Self::from_handle(
            instance,
            handle,
            SurfaceApi::GgpStreamDescriptor,
            object,
        )))
    }

    /// Creates a `Surface` from an iOS `UIView`.
    ///
    /// # Safety
    ///
    /// - `metal_layer` must be a valid `IOSMetalLayer` handle.
    /// - The object referred to by `metal_layer` must outlive the created `Surface`.
    ///   The `object` parameter can be used to ensure this.
    /// - The `UIView` must be backed by a `CALayer` instance of type `CAMetalLayer`.
    #[cfg(target_os = "ios")]
    pub unsafe fn from_ios(
        instance: Arc<Instance>,
        metal_layer: IOSMetalLayer,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_ios(&instance, &metal_layer)?;

        Ok(Self::from_ios_unchecked(instance, metal_layer, object)?)
    }

    #[cfg(target_os = "ios")]
    fn validate_from_ios(
        instance: &Instance,
        _metal_layer: &IOSMetalLayer,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().mvk_ios_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "mvk_ios_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkIOSSurfaceCreateInfoMVK-pView-04143
        // Can't validate, therefore unsafe

        // VUID-VkIOSSurfaceCreateInfoMVK-pView-01316
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg(target_os = "ios")]
    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_ios_unchecked(
        instance: Arc<Instance>,
        metal_layer: IOSMetalLayer,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info = ash::vk::IOSSurfaceCreateInfoMVK {
            flags: ash::vk::IOSSurfaceCreateFlagsMVK::empty(),
            p_view: metal_layer.render_layer.0 as *const _,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            (fns.mvk_ios_surface.create_ios_surface_mvk)(
                instance.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Arc::new(Self::from_handle(
            instance,
            handle,
            SurfaceApi::Ios,
            object,
        )))
    }

    /// Creates a `Surface` from a MacOS `NSView`.
    ///
    /// # Safety
    ///
    /// - `view` must be a valid `CAMetalLayer` or `NSView` handle.
    /// - The object referred to by `view` must outlive the created `Surface`.
    ///   The `object` parameter can be used to ensure this.
    /// - The `NSView` must be backed by a `CALayer` instance of type `CAMetalLayer`.
    #[cfg(target_os = "macos")]
    pub unsafe fn from_mac_os<V>(
        instance: Arc<Instance>,
        view: *const V,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_mac_os(&instance, view)?;

        Ok(Self::from_mac_os_unchecked(instance, view, object)?)
    }

    #[cfg(target_os = "macos")]
    fn validate_from_mac_os<V>(
        instance: &Instance,
        _view: *const V,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().mvk_macos_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "mvk_macos_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkMacOSSurfaceCreateInfoMVK-pView-04144
        // Can't validate, therefore unsafe

        // VUID-VkMacOSSurfaceCreateInfoMVK-pView-01317
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_mac_os_unchecked<V>(
        instance: Arc<Instance>,
        view: *const V,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info = ash::vk::MacOSSurfaceCreateInfoMVK {
            flags: ash::vk::MacOSSurfaceCreateFlagsMVK::empty(),
            p_view: view as *const _,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            (fns.mvk_macos_surface.create_mac_os_surface_mvk)(
                instance.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Arc::new(Self::from_handle(
            instance,
            handle,
            SurfaceApi::MacOs,
            object,
        )))
    }

    /// Creates a `Surface` from a Metal `CAMetalLayer`.
    ///
    /// # Safety
    ///
    /// - `layer` must be a valid Metal `CAMetalLayer` handle.
    /// - The object referred to by `layer` must outlive the created `Surface`.
    ///   The `object` parameter can be used to ensure this.
    pub unsafe fn from_metal<L>(
        instance: Arc<Instance>,
        layer: *const L,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_metal(&instance, layer)?;

        Ok(Self::from_metal_unchecked(instance, layer, object)?)
    }

    fn validate_from_metal<L>(
        instance: &Instance,
        _layer: *const L,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().ext_metal_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_metal_surface",
                )])]),
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_metal_unchecked<L>(
        instance: Arc<Instance>,
        layer: *const L,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info = ash::vk::MetalSurfaceCreateInfoEXT {
            flags: ash::vk::MetalSurfaceCreateFlagsEXT::empty(),
            p_layer: layer as *const _,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            (fns.ext_metal_surface.create_metal_surface_ext)(
                instance.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Arc::new(Self::from_handle(
            instance,
            handle,
            SurfaceApi::Metal,
            object,
        )))
    }

    /// Creates a `Surface` from a QNX Screen window.
    ///
    /// # Safety
    ///
    /// - `context` must be a valid QNX Screen `_screen_context` handle.
    /// - `window` must be a valid QNX Screen `_screen_window` handle.
    /// - The object referred to by `window` must outlive the created `Surface`.
    ///   The `object` parameter can be used to ensure this.
    pub unsafe fn from_qnx_screen<C, W>(
        instance: Arc<Instance>,
        context: *const C,
        window: *const W,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_qnx_screen(&instance, context, window)?;

        Ok(Self::from_qnx_screen_unchecked(
            instance, context, window, object,
        )?)
    }

    fn validate_from_qnx_screen<C, W>(
        instance: &Instance,
        _context: *const C,
        _window: *const W,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().qnx_screen_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "qnx_screen_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkScreenSurfaceCreateInfoQNX-context-04741
        // Can't validate, therefore unsafe

        // VUID-VkScreenSurfaceCreateInfoQNX-window-04742
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_qnx_screen_unchecked<C, W>(
        instance: Arc<Instance>,
        context: *const C,
        window: *const W,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info = ash::vk::ScreenSurfaceCreateInfoQNX {
            flags: ash::vk::ScreenSurfaceCreateFlagsQNX::empty(),
            context: context as *mut _,
            window: window as *mut _,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            (fns.qnx_screen_surface.create_screen_surface_qnx)(
                instance.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Arc::new(Self::from_handle(
            instance,
            handle,
            SurfaceApi::QnxScreen,
            object,
        )))
    }

    /// Creates a `Surface` from a `code:nn::code:vi::code:Layer`.
    ///
    /// # Safety
    ///
    /// - `window` must be a valid `nn::vi::NativeWindowHandle` handle.
    /// - The object referred to by `window` must outlive the created `Surface`.
    ///   The `object` parameter can be used to ensure this.
    pub unsafe fn from_vi<W>(
        instance: Arc<Instance>,
        window: *const W,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_vi(&instance, window)?;

        Ok(Self::from_vi_unchecked(instance, window, object)?)
    }

    fn validate_from_vi<W>(
        instance: &Instance,
        _window: *const W,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().nn_vi_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "nn_vi_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkViSurfaceCreateInfoNN-window-01318
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_vi_unchecked<W>(
        instance: Arc<Instance>,
        window: *const W,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info = ash::vk::ViSurfaceCreateInfoNN {
            flags: ash::vk::ViSurfaceCreateFlagsNN::empty(),
            window: window as *mut _,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            (fns.nn_vi_surface.create_vi_surface_nn)(
                instance.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Arc::new(Self::from_handle(
            instance,
            handle,
            SurfaceApi::Vi,
            object,
        )))
    }

    /// Creates a `Surface` from a Wayland window.
    ///
    /// The window's dimensions will be set to the size of the swapchain.
    ///
    /// # Safety
    ///
    /// - `display` must be a valid Wayland `wl_display` handle.
    /// - `surface` must be a valid Wayland `wl_surface` handle.
    /// - The objects referred to by `display` and `surface` must outlive the created `Surface`.
    ///   The `object` parameter can be used to ensure this.
    pub unsafe fn from_wayland<D, S>(
        instance: Arc<Instance>,
        display: *const D,
        surface: *const S,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_wayland(&instance, display, surface)?;

        Ok(Self::from_wayland_unchecked(
            instance, display, surface, object,
        )?)
    }

    fn validate_from_wayland<D, S>(
        instance: &Instance,
        _display: *const D,
        _surface: *const S,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().khr_wayland_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_wayland_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkWaylandSurfaceCreateInfoKHR-display-01304
        // Can't validate, therefore unsafe

        // VUID-VkWaylandSurfaceCreateInfoKHR-surface-01305
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_wayland_unchecked<D, S>(
        instance: Arc<Instance>,
        display: *const D,
        surface: *const S,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info = ash::vk::WaylandSurfaceCreateInfoKHR {
            flags: ash::vk::WaylandSurfaceCreateFlagsKHR::empty(),
            display: display as *mut _,
            surface: surface as *mut _,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            (fns.khr_wayland_surface.create_wayland_surface_khr)(
                instance.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Arc::new(Self::from_handle(
            instance,
            handle,
            SurfaceApi::Wayland,
            object,
        )))
    }

    /// Creates a `Surface` from a Win32 window.
    ///
    /// The surface's min, max and current extent will always match the window's dimensions.
    ///
    /// # Safety
    ///
    /// - `hinstance` must be a valid Win32 `HINSTANCE` handle.
    /// - `hwnd` must be a valid Win32 `HWND` handle.
    /// - The objects referred to by `hwnd` and `hinstance` must outlive the created `Surface`.
    ///   The `object` parameter can be used to ensure this.
    pub unsafe fn from_win32<I, W>(
        instance: Arc<Instance>,
        hinstance: *const I,
        hwnd: *const W,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_win32(&instance, hinstance, hwnd)?;

        Ok(Self::from_win32_unchecked(
            instance, hinstance, hwnd, object,
        )?)
    }

    fn validate_from_win32<I, W>(
        instance: &Instance,
        _hinstance: *const I,
        _hwnd: *const W,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().khr_win32_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_win32_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkWin32SurfaceCreateInfoKHR-hinstance-01307
        // Can't validate, therefore unsafe

        // VUID-VkWin32SurfaceCreateInfoKHR-hwnd-01308
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_win32_unchecked<I, W>(
        instance: Arc<Instance>,
        hinstance: *const I,
        hwnd: *const W,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info = ash::vk::Win32SurfaceCreateInfoKHR {
            flags: ash::vk::Win32SurfaceCreateFlagsKHR::empty(),
            hinstance: hinstance as *mut _,
            hwnd: hwnd as *mut _,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            (fns.khr_win32_surface.create_win32_surface_khr)(
                instance.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Arc::new(Self::from_handle(
            instance,
            handle,
            SurfaceApi::Win32,
            object,
        )))
    }

    /// Creates a `Surface` from an XCB window.
    ///
    /// The surface's min, max and current extent will always match the window's dimensions.
    ///
    /// # Safety
    ///
    /// - `connection` must be a valid X11 `xcb_connection_t` handle.
    /// - `window` must be a valid X11 `xcb_window_t` handle.
    /// - The objects referred to by `connection` and `window` must outlive the created `Surface`.
    ///   The `object` parameter can be used to ensure this.
    pub unsafe fn from_xcb<C>(
        instance: Arc<Instance>,
        connection: *const C,
        window: ash::vk::xcb_window_t,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_xcb(&instance, connection, window)?;

        Ok(Self::from_xcb_unchecked(
            instance, connection, window, object,
        )?)
    }

    fn validate_from_xcb<C>(
        instance: &Instance,
        _connection: *const C,
        _window: ash::vk::xcb_window_t,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().khr_xcb_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_xcb_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkXcbSurfaceCreateInfoKHR-connection-01310
        // Can't validate, therefore unsafe

        // VUID-VkXcbSurfaceCreateInfoKHR-window-01311
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_xcb_unchecked<C>(
        instance: Arc<Instance>,
        connection: *const C,
        window: ash::vk::xcb_window_t,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info = ash::vk::XcbSurfaceCreateInfoKHR {
            flags: ash::vk::XcbSurfaceCreateFlagsKHR::empty(),
            connection: connection as *mut _,
            window,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            (fns.khr_xcb_surface.create_xcb_surface_khr)(
                instance.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Arc::new(Self::from_handle(
            instance,
            handle,
            SurfaceApi::Xcb,
            object,
        )))
    }

    /// Creates a `Surface` from an Xlib window.
    ///
    /// The surface's min, max and current extent will always match the window's dimensions.
    ///
    /// # Safety
    ///
    /// - `display` must be a valid Xlib `Display` handle.
    /// - `window` must be a valid Xlib `Window` handle.
    /// - The objects referred to by `display` and `window` must outlive the created `Surface`.
    ///   The `object` parameter can be used to ensure this.
    pub unsafe fn from_xlib<D>(
        instance: Arc<Instance>,
        display: *const D,
        window: ash::vk::Window,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        Self::validate_from_xlib(&instance, display, window)?;

        Ok(Self::from_xlib_unchecked(
            instance, display, window, object,
        )?)
    }

    fn validate_from_xlib<D>(
        instance: &Instance,
        _display: *const D,
        _window: ash::vk::Window,
    ) -> Result<(), Box<ValidationError>> {
        if !instance.enabled_extensions().khr_xlib_surface {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "khr_xlib_surface",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-VkXlibSurfaceCreateInfoKHR-dpy-01313
        // Can't validate, therefore unsafe

        // VUID-VkXlibSurfaceCreateInfoKHR-window-01314
        // Can't validate, therefore unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn from_xlib_unchecked<D>(
        instance: Arc<Instance>,
        display: *const D,
        window: ash::vk::Window,
        object: Option<Arc<dyn Any + Send + Sync>>,
    ) -> Result<Arc<Self>, VulkanError> {
        let create_info = ash::vk::XlibSurfaceCreateInfoKHR {
            flags: ash::vk::XlibSurfaceCreateFlagsKHR::empty(),
            dpy: display as *mut _,
            window,
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = MaybeUninit::uninit();
            (fns.khr_xlib_surface.create_xlib_surface_khr)(
                instance.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Arc::new(Self::from_handle(
            instance,
            handle,
            SurfaceApi::Xlib,
            object,
        )))
    }

    /// Returns the instance this surface was created with.
    #[inline]
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    /// Returns the windowing API that was used to construct the surface.
    #[inline]
    pub fn api(&self) -> SurfaceApi {
        self.api
    }

    /// Returns a reference to the `object` parameter that was passed when creating the
    /// surface.
    #[inline]
    pub fn object(&self) -> Option<&Arc<dyn Any + Send + Sync>> {
        self.object.as_ref()
    }

    /// Resizes the sublayer bounds on iOS.
    /// It may not be necessary if original window size matches device's, but often it does not.
    /// Thus this should be called after a resize has occurred abd swapchain has been recreated.
    ///
    /// On iOS, we've created CAMetalLayer as a sublayer. However, when the view changes size,
    /// its sublayers are not automatically resized, and we must resize
    /// it here.
    #[cfg(target_os = "ios")]
    #[inline]
    pub unsafe fn update_ios_sublayer_on_resize(&self) {
        use core_graphics_types::geometry::CGRect;
        let class = class!(CAMetalLayer);
        let main_layer: *mut Object = self.metal_layer.main_layer.0;
        let bounds: CGRect = msg_send![main_layer, bounds];
        let render_layer: *mut Object = self.metal_layer.render_layer.0;
        let () = msg_send![render_layer, setFrame: bounds];
    }
}

impl Drop for Surface {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.instance.fns();
            (fns.khr_surface.destroy_surface_khr)(self.instance.handle(), self.handle, ptr::null());
        }
    }
}

unsafe impl VulkanObject for Surface {
    type Handle = ash::vk::SurfaceKHR;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl InstanceOwned for Surface {
    #[inline]
    fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }
}

impl Debug for Surface {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        let Self {
            handle,
            instance,
            id,
            api,
            object,

            surface_formats: _,
            surface_present_modes: _,
            surface_support: _,
        } = self;

        f.debug_struct("Surface")
            .field("handle", handle)
            .field("instance", instance)
            .field("id", id)
            .field("api", api)
            .field("object", object)
            .finish_non_exhaustive()
    }
}

impl_id_counter!(Surface);

/// Get sublayer from iOS main view (ui_view). The sublayer is created as `CAMetalLayer`.
#[cfg(target_os = "ios")]
unsafe fn get_metal_layer_ios(ui_view: *mut std::ffi::c_void) -> IOSMetalLayer {
    use core_graphics_types::{base::CGFloat, geometry::CGRect};

    let view: *mut Object = ui_view.cast();
    let main_layer: *mut Object = msg_send![view, layer];
    let class = class!(CAMetalLayer);
    let new_layer: *mut Object = msg_send![class, new];
    let frame: CGRect = msg_send![main_layer, bounds];
    let () = msg_send![new_layer, setFrame: frame];
    let () = msg_send![main_layer, addSublayer: new_layer];
    let screen: *mut Object = msg_send![class!(UIScreen), mainScreen];
    let scale_factor: CGFloat = msg_send![screen, nativeScale];
    let () = msg_send![view, setContentScaleFactor: scale_factor];
    IOSMetalLayer::new(view, new_layer)
}

/// Get (and set) `CAMetalLayer` to `ns_view`. This is necessary to be able to render on Mac.
#[cfg(target_os = "macos")]
unsafe fn get_metal_layer_macos(ns_view: *mut std::ffi::c_void) -> *mut Object {
    use core_graphics_types::base::CGFloat;
    use objc::runtime::YES;
    use objc::runtime::{BOOL, NO};

    let view: *mut Object = ns_view.cast();
    let main_layer: *mut Object = msg_send![view, layer];
    let class = class!(CAMetalLayer);
    let is_valid_layer: BOOL = msg_send![main_layer, isKindOfClass: class];
    if is_valid_layer == NO {
        let new_layer: *mut Object = msg_send![class, new];
        let () = msg_send![new_layer, setEdgeAntialiasingMask: 0];
        let () = msg_send![new_layer, setPresentsWithTransaction: false];
        let () = msg_send![new_layer, removeAllAnimations];
        let () = msg_send![view, setLayer: new_layer];
        let () = msg_send![view, setWantsLayer: YES];
        let window: *mut Object = msg_send![view, window];
        if !window.is_null() {
            let scale_factor: CGFloat = msg_send![window, backingScaleFactor];
            let () = msg_send![new_layer, setContentsScale: scale_factor];
        }
        new_layer
    } else {
        main_layer
    }
}

/// Parameters to create a surface from a display mode and plane.
#[derive(Clone, Debug)]
pub struct DisplaySurfaceCreateInfo {
    /// The index of the display plane in which the surface will appear.
    ///
    /// The default value is 0.
    pub plane_index: u32,

    /// The z-order of the plane.
    ///
    /// The default value is 0.
    pub plane_stack_index: u32,

    /// The transformation to apply to images when presented.
    ///
    /// The default value is [`SurfaceTransform::Identity`].
    pub transform: SurfaceTransform,

    /// How to do alpha blending on the surface when presenting new content.
    ///
    /// The default value is [`DisplayPlaneAlpha::Opaque`].
    pub alpha_mode: DisplayPlaneAlpha,

    /// If `alpha_mode` is `DisplayPlaneAlpha::Global`, specifies the global alpha value to use for
    /// all blending.
    ///
    /// The default value is 1.0.
    pub global_alpha: f32,

    /// The size of the presentable images that will be used.
    ///
    /// The default value is `[0; 2]`, which must be overridden.
    pub image_extent: [u32; 2],

    pub _ne: crate::NonExhaustive,
}

impl Default for DisplaySurfaceCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            plane_index: 0,
            plane_stack_index: 0,
            transform: SurfaceTransform::Identity,
            alpha_mode: DisplayPlaneAlpha::Opaque,
            global_alpha: 1.0,
            image_extent: [0; 2],
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl DisplaySurfaceCreateInfo {
    pub(crate) fn validate(
        &self,
        physical_device: &PhysicalDevice,
    ) -> Result<(), Box<ValidationError>> {
        let &Self {
            plane_index,
            plane_stack_index: _,
            transform,
            alpha_mode,
            global_alpha,
            image_extent,
            _ne: _,
        } = self;

        let properties = physical_device.properties();

        transform
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("transform")
                    .set_vuids(&["VUID-VkDisplaySurfaceCreateInfoKHR-transform-parameter"])
            })?;

        alpha_mode
            .validate_physical_device(physical_device)
            .map_err(|err| {
                err.add_context("alpha_mode")
                    .set_vuids(&["VUID-VkDisplaySurfaceCreateInfoKHR-alphaMode-parameter"])
            })?;

        let display_plane_properties_raw = unsafe {
            physical_device
                .display_plane_properties_raw()
                .map_err(|_err| {
                    Box::new(ValidationError {
                        problem: "`PhysicalDevice::display_plane_properties` \
                            returned an error"
                            .into(),
                        ..Default::default()
                    })
                })?
        };

        if plane_index as usize >= display_plane_properties_raw.len() {
            return Err(Box::new(ValidationError {
                problem: "`plane_index` is not less than the number of display planes on the \
                    physical device"
                    .into(),
                vuids: &["VUID-VkDisplaySurfaceCreateInfoKHR-planeIndex-01252"],
                ..Default::default()
            }));
        }

        if alpha_mode == DisplayPlaneAlpha::Global && !(0.0..=1.0).contains(&global_alpha) {
            return Err(Box::new(ValidationError {
                problem: "`alpha_mode` is `DisplayPlaneAlpha::Global`, but \
                    `global_alpha` is not between 0 and 1 inclusive"
                    .into(),
                vuids: &["VUID-VkDisplaySurfaceCreateInfoKHR-alphaMode-01254"],
                ..Default::default()
            }));
        }

        if image_extent[0] > properties.max_image_dimension2_d {
            return Err(Box::new(ValidationError {
                context: "image_extent[0]".into(),
                problem: "is greater than the `max_image_dimension2_d` device limit".into(),
                vuids: &["VUID-VkDisplaySurfaceCreateInfoKHR-width-01256"],
                ..Default::default()
            }));
        }

        if image_extent[1] > properties.max_image_dimension2_d {
            return Err(Box::new(ValidationError {
                context: "image_extent[1]".into(),
                problem: "is greater than the `max_image_dimension2_d` device limit".into(),
                vuids: &["VUID-VkDisplaySurfaceCreateInfoKHR-width-01256"],
                ..Default::default()
            }));
        }

        Ok(())
    }
}

/// The windowing API function that was used to construct a surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SurfaceApi {
    /// The surface was constructed using [`Surface::headless`] or the
    /// `vkCreateHeadlessSurfaceEXT` Vulkan API function.
    Headless,

    /// The surface was constructed using [`Surface::from_display_plane`] or the
    /// `vkCreateDisplayPlaneSurfaceKHR` Vulkan API function.
    DisplayPlane,

    /*
        Alphabetical order
    */
    /// The surface was constructed using [`Surface::from_android`] or the
    /// `vkCreateAndroidSurfaceKHR` Vulkan API function.
    Android,

    /// The surface was constructed using [`Surface::from_directfb`] or the
    /// `vkCreateDirectFBSurfaceEXT` Vulkan API function.
    DirectFB,

    /// The surface was constructed using [`Surface::from_fuchsia_image_pipe`] or the
    /// `vkCreateImagePipeSurfaceFUCHSIA` Vulkan API function.
    FuchsiaImagePipe,

    /// The surface was constructed using [`Surface::from_ggp_stream_descriptor`] or the
    /// `vkCreateStreamDescriptorSurfaceGGP` Vulkan API function.
    GgpStreamDescriptor,

    /// The surface was constructed using [`Surface::from_ios`] or the
    /// `vkCreateIOSSurfaceMVK` Vulkan API function.
    Ios,

    /// The surface was constructed using [`Surface::from_mac_os`] or the
    /// `vkCreateMacOSSurfaceMVK` Vulkan API function.
    MacOs,

    /// The surface was constructed using [`Surface::from_metal`] or the
    /// `vkCreateMetalSurfaceEXT` Vulkan API function.
    Metal,

    /// The surface was constructed using [`Surface::from_qnx_screen`] or the
    /// `vkCreateScreenSurfaceQNX` Vulkan API function.
    QnxScreen,

    /// The surface was constructed using [`Surface::from_vi`] or the
    /// `vkCreateViSurfaceNN` Vulkan API function.
    Vi,

    /// The surface was constructed using [`Surface::from_wayland`] or the
    /// `vkCreateWaylandSurfaceKHR` Vulkan API function.
    Wayland,

    /// The surface was constructed using [`Surface::from_win32`] or the
    /// `vkCreateWin32SurfaceKHR` Vulkan API function.
    Win32,

    /// The surface was constructed using [`Surface::from_xcb`] or the
    /// `vkCreateXcbSurfaceKHR` Vulkan API function.
    Xcb,

    /// The surface was constructed using [`Surface::from_xlib`] or the
    /// `vkCreateXlibSurfaceKHR` Vulkan API function.
    Xlib,
}

impl TryFrom<RawWindowHandle> for SurfaceApi {
    type Error = ();

    fn try_from(handle: RawWindowHandle) -> Result<Self, Self::Error> {
        match handle {
            RawWindowHandle::UiKit(_) => Ok(SurfaceApi::Ios),
            RawWindowHandle::AppKit(_) => Ok(SurfaceApi::MacOs),
            RawWindowHandle::Orbital(_) => Err(()),
            RawWindowHandle::Xlib(_) => Ok(SurfaceApi::Xlib),
            RawWindowHandle::Xcb(_) => Ok(SurfaceApi::Xcb),
            RawWindowHandle::Wayland(_) => Ok(SurfaceApi::Wayland),
            RawWindowHandle::Drm(_) => Err(()),
            RawWindowHandle::Gbm(_) => Err(()),
            RawWindowHandle::Win32(_) => Ok(SurfaceApi::Win32),
            RawWindowHandle::WinRt(_) => Err(()),
            RawWindowHandle::Web(_) => Err(()),
            RawWindowHandle::AndroidNdk(_) => Ok(SurfaceApi::Android),
            RawWindowHandle::Haiku(_) => Err(()),
            _ => Err(()),
        }
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// The mode of action when a swapchain image is presented.
    ///
    /// Swapchain images can be in one of three possible states:
    /// - Exactly one image is currently displayed on the screen.
    /// - Zero or more are acquired by the application, or available to be acquired.
    /// - Some may be held inside the presentation engine waiting to be displayed. The present mode
    ///   concerns the behaviour of this category, and by extension, which images are left over for
    ///   acquiring.
    ///
    /// The present mode affects what is commonly known as "vertical sync" or "vsync" for short.
    /// The `Immediate` mode is equivalent to disabling vertical sync, while the others enable
    /// vertical sync in various forms. An important aspect of the present modes is their potential
    /// *latency*: the time between when an image is presented, and when it actually appears on
    /// the display.
    ///
    /// Only `Fifo` is guaranteed to be supported on every device. For the others, you must call
    /// [`surface_present_modes`] to see if they are supported.
    ///
    /// [`surface_present_modes`]: crate::device::physical::PhysicalDevice::surface_present_modes
    PresentMode = PresentModeKHR(i32);

    /// The presentation engine holds only the currently displayed image. When presenting an image,
    /// the currently displayed image is immediately replaced with the presented image. The old
    /// image will be available for future acquire operations.
    ///
    /// This mode has the lowest latency of all present modes, but if the display is not in a
    /// vertical blanking period when the image is replaced, a tear will be visible.
    Immediate = IMMEDIATE,

    /// The presentation engine holds the currently displayed image, and optionally another in a
    /// waiting slot. The presentation engine waits until the next vertical blanking period, then
    /// removes any image from the waiting slot and displays it. Tearing will never be visible.
    /// When presenting an image, it is stored in the waiting slot. Any previous entry
    /// in the slot is discarded, and will be available for future acquire operations.
    ///
    /// Latency is relatively low with this mode, and will never be longer than the time between
    /// vertical blanking periods. However, if a previous image in the waiting slot is discarded,
    /// the work that went into producing that image was wasted.
    ///
    /// With two swapchain images, this mode behaves essentially identical to `Fifo`: once both
    /// images are held in the presentation engine, no images can be acquired until one is finished
    /// displaying. But with three or more swapchain images, any images beyond those two are always
    /// available to acquire.
    Mailbox = MAILBOX,

    /// The presentation engine holds the currently displayed image, and a queue of waiting images.
    /// When presenting an image, it is added to the tail of the queue, after previously presented
    /// images. The presentation engine waits until the next vertical blanking period, then removes
    /// an image from the head of the queue and displays it. Tearing will never be visible. Images
    /// become available for future acquire operations only after they have been displayed.
    ///
    /// This mode is guaranteed to be always supported. It is possible for all swapchain images to
    /// end up being held by the presentation engine, either being displayed or in the queue. When
    /// that happens, no images can be acquired until one is finished displaying. This can be used
    /// to limit the presentation rate to the display frame rate. Latency is bounded only by the
    /// number of images in the swapchain.
    ///
    /// This is the equivalent of OpenGL's `SwapInterval` with a value of 1.
    Fifo = FIFO,

    /// Similar to `Fifo`, but with the ability for images to "skip the queue" if presentation is
    /// lagging behind the display frame rate. If the queue is empty and a vertical blanking period
    /// has already passed since the previous image was displayed, then the currently displayed
    /// image is immediately replaced with the presented image, as in `Immediate`.
    ///
    /// This mode has high latency if images are presented faster than the display frame rate,
    /// as they will accumulate in the queue. But the latency is low if images are presented slower
    /// than the display frame rate. However, slower presentation can result in visible tearing.
    ///
    /// This is the equivalent of OpenGL's `SwapInterval` with a value of -1.
    FifoRelaxed = FIFO_RELAXED,

    /* TODO: enable
    // TODO: document
    SharedDemandRefresh = SHARED_DEMAND_REFRESH_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_shared_presentable_image)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    SharedContinuousRefresh = SHARED_CONTINUOUS_REFRESH_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_shared_presentable_image)]),
    ]),*/
}

vulkan_bitflags_enum! {
    #[non_exhaustive]

    /// A set of [`SurfaceTransform`] values.
    SurfaceTransforms,

    /// The presentation transform to apply when presenting a swapchain image to a surface.
    SurfaceTransform,

    = SurfaceTransformFlagsKHR(u32);

    /// Don't transform the image.
    IDENTITY, Identity = IDENTITY,

    /// Rotate 90 degrees.
    ROTATE_90, Rotate90 = ROTATE_90,

    /// Rotate 180 degrees.
    ROTATE_180, Rotate180 = ROTATE_180,

    /// Rotate 270 degrees.
    ROTATE_270, Rotate270 = ROTATE_270,

    /// Mirror the image horizontally.
    HORIZONTAL_MIRROR, HorizontalMirror = HORIZONTAL_MIRROR,

    /// Mirror the image horizontally and rotate 90 degrees.
    HORIZONTAL_MIRROR_ROTATE_90, HorizontalMirrorRotate90 = HORIZONTAL_MIRROR_ROTATE_90,

    /// Mirror the image horizontally and rotate 180 degrees.
    HORIZONTAL_MIRROR_ROTATE_180, HorizontalMirrorRotate180 = HORIZONTAL_MIRROR_ROTATE_180,

    /// Mirror the image horizontally and rotate 270 degrees.
    HORIZONTAL_MIRROR_ROTATE_270, HorizontalMirrorRotate270 = HORIZONTAL_MIRROR_ROTATE_270,

    /// Let the operating system or driver implementation choose.
    INHERIT, Inherit = INHERIT,
}

impl Default for SurfaceTransform {
    #[inline]
    fn default() -> SurfaceTransform {
        SurfaceTransform::Identity
    }
}

vulkan_bitflags_enum! {
    #[non_exhaustive]

    /// A set of [`CompositeAlpha`] values.
    CompositeAlphas,

    /// How the alpha values of the pixels of the window are treated.
    CompositeAlpha,

    = CompositeAlphaFlagsKHR(u32);

    /// The alpha channel of the image is ignored. All the pixels are considered as if they have a
    /// value of 1.0.
    OPAQUE, Opaque = OPAQUE,

    /// The alpha channel of the image is respected. The color channels are expected to have
    /// already been multiplied by the alpha value.
    PRE_MULTIPLIED, PreMultiplied = PRE_MULTIPLIED,

    /// The alpha channel of the image is respected. The color channels will be multiplied by the
    /// alpha value by the compositor before being added to what is behind.
    POST_MULTIPLIED, PostMultiplied = POST_MULTIPLIED,

    /// Let the operating system or driver implementation choose.
    INHERIT, Inherit = INHERIT,
}

vulkan_enum! {
    #[non_exhaustive]

    /// How the presentation engine should interpret the data.
    ///
    /// # A quick lesson about color spaces
    ///
    /// ## What is a color space?
    ///
    /// Each pixel of a monitor is made of three components: one red, one green, and one blue. In
    /// the past, computers would simply send to the monitor the intensity of each of the three
    /// components.
    ///
    /// This proved to be problematic, because depending on the brand of the monitor the colors
    /// would not exactly be the same. For example on some monitors, a value of `[1.0, 0.0, 0.0]`
    /// would be a
    /// bit more orange than on others.
    ///
    /// In order to standardize this, there exist what are called *color spaces*: sRGB, AdobeRGB,
    /// DCI-P3, scRGB, etc. When you manipulate RGB values in a specific color space, these values
    /// have a precise absolute meaning in terms of color, that is the same across all systems and
    /// monitors.
    ///
    /// > **Note**: Color spaces are orthogonal to concept of RGB. *RGB* only indicates what is the
    /// > representation of the data, but not how it is interpreted. You can think of this a bit
    /// > like text encoding. An *RGB* value is a like a byte, in other words it is the medium by
    /// > which values are communicated, and a *color space* is like a text encoding (eg. UTF-8),
    /// > in other words it is the way the value should be interpreted.
    ///
    /// The most commonly used color space today is sRGB. Most monitors today use this color space,
    /// and most images files are encoded in this color space.
    ///
    /// ## Pixel formats and linear vs non-linear
    ///
    /// In Vulkan all images have a specific format in which the data is stored. The data of an
    /// image consists of pixels in RGB but contains no information about the color space (or lack
    /// thereof) of these pixels. You are free to store them in whatever color space you want.
    ///
    /// But one big practical problem with color spaces is that they are sometimes not linear, and
    /// in particular the popular sRGB color space is not linear. In a non-linear color space, a
    /// value of `[0.6, 0.6, 0.6]` for example is **not** twice as bright as a value of `[0.3, 0.3,
    /// 0.3]`. This is problematic, because operations such as taking the average of two colors or
    /// calculating the lighting of a texture with a dot product are mathematically incorrect and
    /// will produce incorrect colors.
    ///
    /// > **Note**: If the texture format has an alpha component, it is not affected by the color
    /// > space and always behaves linearly.
    ///
    /// In order to solve this Vulkan also provides image formats with the `Srgb` suffix, which are
    /// expected to contain RGB data in the sRGB color space. When you sample an image with such a
    /// format from a shader, the implementation will automatically turn the pixel values into a
    /// linear color space that is suitable for linear operations (such as additions or
    /// multiplications). When you write to a framebuffer attachment with such a format, the
    /// implementation will automatically perform the opposite conversion. These conversions are
    /// most of the time performed by the hardware and incur no additional cost.
    ///
    /// ## Color space of the swapchain
    ///
    /// The color space that you specify when you create a swapchain is how the implementation will
    /// interpret the raw data inside of the image.
    ///
    /// > **Note**: The implementation can choose to send the data in the swapchain image directly
    /// > to the monitor, but it can also choose to write it in an intermediary buffer that is then
    /// > read by the operating system or windowing system. Therefore the color space that the
    /// > implementation supports is not necessarily the same as the one supported by the monitor.
    ///
    /// It is *your* job to ensure that the data in the swapchain image is in the color space
    /// that is specified here, otherwise colors will be incorrect. The implementation will never
    /// perform any additional automatic conversion after the colors have been written to the
    /// swapchain image.
    ///
    /// # How do I handle this correctly?
    ///
    /// The easiest way to handle color spaces in a cross-platform program is:
    ///
    /// - Always request the `SrgbNonLinear` color space when creating the swapchain.
    /// - Make sure that all your image files use the sRGB color space, and load them in images
    ///   whose format has the `Srgb` suffix. Only use non-sRGB image formats for intermediary
    ///   computations or to store non-color data.
    /// - Swapchain images should have a format with the `Srgb` suffix.
    ///
    /// > **Note**: Lots of developers are confused by color spaces. You can sometimes find articles
    /// > talking about gamma correction and suggestion to put your colors to the power 2.2 for
    /// > example. These are all hacks and you should use the sRGB pixel formats instead.
    ///
    /// If you follow these three rules, then everything should render the same way on all
    /// platforms.
    ///
    /// Additionally you can try detect whether the implementation supports any additional color
    /// space and perform a manual conversion to that color space from inside your shader.
    ColorSpace = ColorSpaceKHR(i32);

    // TODO: document
    SrgbNonLinear = SRGB_NONLINEAR,

    // TODO: document
    DisplayP3NonLinear = DISPLAY_P3_NONLINEAR_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    ExtendedSrgbLinear = EXTENDED_SRGB_LINEAR_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    ExtendedSrgbNonLinear = EXTENDED_SRGB_NONLINEAR_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    DisplayP3Linear = DISPLAY_P3_LINEAR_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    DciP3NonLinear = DCI_P3_NONLINEAR_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    Bt709Linear = BT709_LINEAR_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    Bt709NonLinear = BT709_NONLINEAR_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    Bt2020Linear = BT2020_LINEAR_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    Hdr10St2084 = HDR10_ST2084_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    DolbyVision = DOLBYVISION_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    Hdr10Hlg = HDR10_HLG_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    AdobeRgbLinear = ADOBERGB_LINEAR_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    AdobeRgbNonLinear = ADOBERGB_NONLINEAR_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    PassThrough = PASS_THROUGH_EXT
    RequiresOneOf([
        RequiresAllOf([InstanceExtension(ext_swapchain_colorspace)]),
    ]),

    // TODO: document
    DisplayNative = DISPLAY_NATIVE_AMD
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(amd_display_native_hdr)]),
    ]),
}

/// Parameters for [`PhysicalDevice::surface_capabilities`] and [`PhysicalDevice::surface_formats`].
///
/// [`PhysicalDevice::surface_capabilities`]: crate::device::physical::PhysicalDevice::surface_capabilities
/// [`PhysicalDevice::surface_formats`]: crate::device::physical::PhysicalDevice::surface_formats
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SurfaceInfo {
    /// If this is `Some`, the
    /// [`ext_surface_maintenance1`](crate::instance::InstanceExtensions::ext_surface_maintenance1)
    /// extension must be enabled on the instance.
    pub present_mode: Option<PresentMode>,

    /// If this is not [`FullScreenExclusive::Default`], the
    /// [`ext_full_screen_exclusive`](crate::device::DeviceExtensions::ext_full_screen_exclusive)
    /// extension must be supported by the physical device.
    pub full_screen_exclusive: FullScreenExclusive,

    /// If `full_screen_exclusive` is [`FullScreenExclusive::ApplicationControlled`], and the
    /// surface being queried is a Win32 surface, then this must be `Some`. Otherwise, it must be
    /// `None`.
    pub win32_monitor: Option<Win32Monitor>,

    pub _ne: crate::NonExhaustive,
}

impl Default for SurfaceInfo {
    #[inline]
    fn default() -> Self {
        Self {
            present_mode: None,
            full_screen_exclusive: FullScreenExclusive::Default,
            win32_monitor: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl SurfaceInfo {
    pub(crate) fn validate(
        &self,
        physical_device: &PhysicalDevice,
    ) -> Result<(), Box<ValidationError>> {
        let &Self {
            present_mode,
            full_screen_exclusive,
            win32_monitor: _,
            _ne: _,
        } = self;

        if let Some(present_mode) = present_mode {
            if !physical_device
                .instance()
                .enabled_extensions()
                .ext_surface_maintenance1
            {
                return Err(Box::new(ValidationError {
                    context: "present_mode".into(),
                    problem: "is `Some`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                        Requires::InstanceExtension("ext_surface_maintenance1"),
                    ])]),
                    ..Default::default()
                }));
            }

            present_mode
                .validate_physical_device(physical_device)
                .map_err(|err| {
                    err.add_context("present_mode")
                        .set_vuids(&["VUID-VkSurfacePresentModeEXT-presentMode-parameter"])
                })?;
        }

        if full_screen_exclusive != FullScreenExclusive::Default
            && !physical_device
                .supported_extensions()
                .ext_full_screen_exclusive
        {
            return Err(Box::new(ValidationError {
                context: "full_screen_exclusive".into(),
                problem: "is not `FullScreenExclusive::Default`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "ext_full_screen_exclusive",
                )])]),
                ..Default::default()
            }));
        }

        Ok(())
    }
}

#[cfg(target_os = "ios")]
struct LayerHandle(*mut Object);

#[cfg(target_os = "ios")]
unsafe impl Send for LayerHandle {}

#[cfg(target_os = "ios")]
unsafe impl Sync for LayerHandle {}

/// Represents the metal layer for IOS
#[cfg(target_os = "ios")]
pub struct IOSMetalLayer {
    main_layer: LayerHandle,
    render_layer: LayerHandle,
}

#[cfg(target_os = "ios")]
impl IOSMetalLayer {
    #[inline]
    pub fn new(main_layer: *mut Object, render_layer: *mut Object) -> Self {
        Self {
            main_layer: LayerHandle(main_layer),
            render_layer: LayerHandle(render_layer),
        }
    }
}

#[cfg(target_os = "ios")]
unsafe impl Send for IOSMetalLayer {}

#[cfg(target_os = "ios")]
unsafe impl Sync for IOSMetalLayer {}

/// The capabilities of a surface when used by a physical device.
///
/// You have to match these capabilities when you create a swapchain.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SurfaceCapabilities {
    /// Minimum number of images that must be present in the swapchain.
    pub min_image_count: u32,

    /// Maximum number of images that must be present in the swapchain, or `None` if there is no
    /// maximum value. Note that "no maximum" doesn't mean that you can set a very high value, as
    /// you may still get out of memory errors.
    pub max_image_count: Option<u32>,

    /// The current dimensions of the surface.
    ///
    /// `None` means that the surface's dimensions will depend on the dimensions of the swapchain.
    pub current_extent: Option<[u32; 2]>,

    /// Minimum width and height of a swapchain that uses this surface.
    pub min_image_extent: [u32; 2],

    /// Maximum width and height of a swapchain that uses this surface.
    pub max_image_extent: [u32; 2],

    /// Maximum number of image layers if you create an image array. The minimum is 1.
    pub max_image_array_layers: u32,

    /// List of transforms supported for the swapchain.
    pub supported_transforms: SurfaceTransforms,

    /// Current transform used by the surface.
    pub current_transform: SurfaceTransform,

    /// List of composite alpha modes supports for the swapchain.
    pub supported_composite_alpha: CompositeAlphas,

    /// List of image usages that are supported for images of the swapchain. Only
    /// the `color_attachment` usage is guaranteed to be supported.
    pub supported_usage_flags: ImageUsage,

    /// When [`SurfaceInfo::present_mode`] is provided,
    /// lists that present mode and any modes that are compatible with that present mode.
    ///
    /// If [`SurfaceInfo::present_mode`] was not provided, the value will be empty.
    pub compatible_present_modes: SmallVec<[PresentMode; PresentMode::COUNT]>,

    /// When [`SurfaceInfo::present_mode`] is provided,
    /// the supported present scaling modes for the queried present mode.
    ///
    /// If [`SurfaceInfo::present_mode`] was not provided, the value will be empty.
    pub supported_present_scaling: PresentScalingFlags,

    /// When [`SurfaceInfo::present_mode`] is provided,
    /// the supported present gravity modes, horizontally and vertically,
    /// for the queried present mode.
    ///
    /// If [`SurfaceInfo::present_mode`] was not provided, both values will be empty.
    pub supported_present_gravity: [PresentGravityFlags; 2],

    /// When [`SurfaceInfo::present_mode`] is provided,
    /// the smallest allowed extent for a swapchain, if it uses the queried present mode, and
    /// one of the scaling modes in `supported_present_scaling`.
    ///
    /// This is never greater than [`SurfaceCapabilities::min_image_extent`].
    ///
    /// `None` means that the surface's dimensions will depend on the dimensions of the swapchain.
    ///
    /// If [`SurfaceInfo::present_mode`] was not provided, this is will be equal to
    /// `min_image_extent`.
    pub min_scaled_image_extent: Option<[u32; 2]>,

    /// When [`SurfaceInfo::present_mode`] is provided,
    /// the largest allowed extent for a swapchain, if it uses the queried present mode, and
    /// one of the scaling modes in `supported_present_scaling`.
    ///
    /// This is never less than [`SurfaceCapabilities::max_image_extent`].
    ///
    /// `None` means that the surface's dimensions will depend on the dimensions of the swapchain.
    ///
    /// If [`SurfaceInfo::present_mode`] was not provided, this is will be equal to
    /// `max_image_extent`.
    pub max_scaled_image_extent: Option<[u32; 2]>,

    /// Whether creating a protected swapchain is supported.
    pub supports_protected: bool,

    /// Whether full-screen exclusivity is supported.
    pub full_screen_exclusive_supported: bool,
}

/// Error that can happen when creating a [`Surface`] from a window.
#[derive(Clone, Debug)]
pub enum FromWindowError {
    /// Retrieving the window or display handle failed.
    RetrieveHandle(HandleError),
    /// Creating the surface failed.
    CreateSurface(Validated<VulkanError>),
}

impl Error for FromWindowError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RetrieveHandle(err) => Some(err),
            Self::CreateSurface(err) => Some(err),
        }
    }
}

impl Display for FromWindowError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::RetrieveHandle(_) => {
                write!(f, "retrieving the window or display handle has failed")
            }
            Self::CreateSurface(_) => write!(f, "creating the surface has failed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        swapchain::Surface, Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError,
    };
    use std::ptr;

    #[test]
    fn khr_win32_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::from_win32(instance, ptr::null::<u8>(), ptr::null::<u8>(), None) } {
            Err(Validated::ValidationError(err))
                if matches!(
                    *err,
                    ValidationError {
                        requires_one_of: RequiresOneOf([RequiresAllOf([
                            Requires::InstanceExtension("khr_win32_surface")
                        ])]),
                        ..
                    }
                ) => {}
            _ => panic!(),
        }
    }

    #[test]
    fn khr_xcb_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::from_xcb(instance, ptr::null::<u8>(), 0, None) } {
            Err(Validated::ValidationError(err))
                if matches!(
                    *err,
                    ValidationError {
                        requires_one_of: RequiresOneOf([RequiresAllOf([
                            Requires::InstanceExtension("khr_xcb_surface")
                        ])]),
                        ..
                    }
                ) => {}
            _ => panic!(),
        }
    }

    #[test]
    fn khr_xlib_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::from_xlib(instance, ptr::null::<u8>(), 0, None) } {
            Err(Validated::ValidationError(err))
                if matches!(
                    *err,
                    ValidationError {
                        requires_one_of: RequiresOneOf([RequiresAllOf([
                            Requires::InstanceExtension("khr_xlib_surface")
                        ])]),
                        ..
                    }
                ) => {}
            _ => panic!(),
        }
    }

    #[test]
    fn khr_wayland_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::from_wayland(instance, ptr::null::<u8>(), ptr::null::<u8>(), None) }
        {
            Err(Validated::ValidationError(err))
                if matches!(
                    *err,
                    ValidationError {
                        requires_one_of: RequiresOneOf([RequiresAllOf([
                            Requires::InstanceExtension("khr_wayland_surface")
                        ])]),
                        ..
                    }
                ) => {}
            _ => panic!(),
        }
    }

    #[test]
    fn khr_android_surface_ext_missing() {
        let instance = instance!();
        match unsafe { Surface::from_android(instance, ptr::null::<u8>(), None) } {
            Err(Validated::ValidationError(err))
                if matches!(
                    *err,
                    ValidationError {
                        requires_one_of: RequiresOneOf([RequiresAllOf([
                            Requires::InstanceExtension("khr_android_surface")
                        ])]),
                        ..
                    }
                ) => {}
            _ => panic!(),
        }
    }
}
