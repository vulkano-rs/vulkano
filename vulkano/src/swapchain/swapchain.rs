// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::SupportedCompositeAlpha;
use super::SupportedSurfaceTransforms;
use crate::buffer::BufferAccess;
use crate::check_errors;
use crate::command_buffer::submit::SubmitAnyBuilder;
use crate::command_buffer::submit::SubmitPresentBuilder;
use crate::command_buffer::submit::SubmitPresentError;
use crate::command_buffer::submit::SubmitSemaphoresWaitBuilder;
use crate::device::physical::SurfaceInfo;
use crate::device::physical::SurfacePropertiesError;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::device::Queue;
use crate::format::Format;
use crate::image::swapchain::SwapchainImage;
use crate::image::sys::UnsafeImage;
use crate::image::ImageAccess;
use crate::image::ImageCreateFlags;
use crate::image::ImageDimensions;
use crate::image::ImageInner;
use crate::image::ImageLayout;
use crate::image::ImageTiling;
use crate::image::ImageType;
use crate::image::ImageUsage;
use crate::image::SampleCount;
use crate::swapchain::ColorSpace;
use crate::swapchain::CompositeAlpha;
use crate::swapchain::PresentMode;
use crate::swapchain::PresentRegion;
use crate::swapchain::Surface;
use crate::swapchain::SurfaceApi;
use crate::swapchain::SurfaceSwapchainLock;
use crate::swapchain::SurfaceTransform;
use crate::sync::semaphore::SemaphoreError;
use crate::sync::AccessCheckError;
use crate::sync::AccessError;
use crate::sync::AccessFlags;
use crate::sync::Fence;
use crate::sync::FlushError;
use crate::sync::GpuFuture;
use crate::sync::PipelineStages;
use crate::sync::Semaphore;
use crate::sync::Sharing;
use crate::Error;
use crate::OomError;
use crate::Success;
use crate::VulkanObject;
use smallvec::SmallVec;
use std::error;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

/// Contains the swapping system and the images that can be shown on a surface.
#[derive(Debug)]
pub struct Swapchain<W> {
    handle: ash::vk::SwapchainKHR,
    device: Arc<Device>,
    surface: Arc<Surface<W>>,

    min_image_count: u32,
    image_format: Format,
    image_color_space: ColorSpace,
    image_extent: [u32; 2],
    image_array_layers: u32,
    image_usage: ImageUsage,
    image_sharing: Sharing<SmallVec<[u32; 4]>>,
    pre_transform: SurfaceTransform,
    composite_alpha: CompositeAlpha,
    present_mode: PresentMode,
    clipped: bool,
    full_screen_exclusive: FullScreenExclusive,
    win32_monitor: Option<Win32Monitor>,

    // Whether full-screen exclusive is currently held.
    full_screen_exclusive_held: AtomicBool,

    // The images of this swapchain.
    images: Vec<ImageEntry>,

    // If true, that means we have tried to use this swapchain to recreate a new swapchain. The current
    // swapchain can no longer be used for anything except presenting already-acquired images.
    //
    // We use a `Mutex` instead of an `AtomicBool` because we want to keep that locked while
    // we acquire the image.
    retired: Mutex<bool>,
}

#[derive(Debug)]
struct ImageEntry {
    // The actual image.
    image: UnsafeImage,
    // If true, then the image is still in the undefined layout and must be transitioned.
    undefined_layout: AtomicBool,
}

impl<W> Swapchain<W> {
    /// Creates a new `Swapchain`.
    ///
    /// This function returns the swapchain plus a list of the images that belong to the
    /// swapchain. The order in which the images are returned is important for the
    /// `acquire_next_image` and `present` functions.
    ///
    /// # Panics
    ///
    /// - Panics if the device and the surface don't belong to the same instance.
    /// - Panics if `create_info.usage` is empty.
    ///
    // TODO: isn't it unsafe to take the surface through an Arc when it comes to vulkano-win?
    pub fn new(
        device: Arc<Device>,
        surface: Arc<Surface<W>>,
        mut create_info: SwapchainCreateInfo,
    ) -> Result<(Arc<Swapchain<W>>, Vec<Arc<SwapchainImage<W>>>), SwapchainCreationError> {
        assert_eq!(
            device.instance().internal_object(),
            surface.instance().internal_object()
        );

        if !device.enabled_extensions().khr_swapchain {
            return Err(SwapchainCreationError::ExtensionNotEnabled {
                extension: "khr_swapchain",
                reason: "created a new swapchain",
            });
        }

        Self::validate(&device, &surface, &mut create_info)?;

        // Checking that the surface doesn't already have a swapchain.
        if surface.flag().swap(true, Ordering::AcqRel) {
            return Err(SwapchainCreationError::SurfaceInUse);
        }

        let (handle, images) = unsafe {
            let (handle, image_handles) = Self::create(&device, &surface, &create_info, None)?;
            let images = Self::wrap_images(&device, image_handles, &create_info);
            (handle, images)
        };

        let SwapchainCreateInfo {
            min_image_count,
            image_format,
            image_color_space,
            image_extent,
            image_array_layers,
            image_usage,
            image_sharing,
            pre_transform,
            composite_alpha,
            present_mode,
            clipped,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = create_info;

        let swapchain = Arc::new(Swapchain {
            handle,
            device,
            surface,

            min_image_count,
            image_format: image_format.unwrap(),
            image_color_space,
            image_extent,
            image_array_layers,
            image_usage,
            image_sharing,
            pre_transform,
            composite_alpha,
            present_mode,
            clipped,
            full_screen_exclusive,
            win32_monitor,

            full_screen_exclusive_held: AtomicBool::new(false),
            images,
            retired: Mutex::new(false),
        });

        let swapchain_images = (0..swapchain.images.len())
            .map(|n| unsafe { SwapchainImage::from_raw(swapchain.clone(), n) })
            .collect::<Result<_, _>>()?;

        Ok((swapchain, swapchain_images))
    }

    /// Creates a new swapchain from this one.
    ///
    /// Use this when a swapchain has become invalidated, such as due to window resizes.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.usage` is empty.
    #[inline]
    pub fn recreate(
        self: &Arc<Self>,
        mut create_info: SwapchainCreateInfo,
    ) -> Result<(Arc<Swapchain<W>>, Vec<Arc<SwapchainImage<W>>>), SwapchainCreationError> {
        Self::validate(&self.device, &self.surface, &mut create_info)?;

        {
            let mut retired = self.retired.lock().unwrap();

            // The swapchain has already been used to create a new one.
            if *retired {
                return Err(SwapchainCreationError::SwapchainAlreadyRetired);
            } else {
                // According to the documentation of VkSwapchainCreateInfoKHR:
                //
                // > Upon calling vkCreateSwapchainKHR with a oldSwapchain that is not VK_NULL_HANDLE,
                // > any images not acquired by the application may be freed by the implementation,
                // > which may occur even if creation of the new swapchain fails.
                //
                // Therefore, we set retired to true and keep it to true even if the call to `vkCreateSwapchainKHR` below fails.
                *retired = true;
            }
        }

        let (handle, images) = unsafe {
            let (handle, image_handles) =
                Self::create(&self.device, &self.surface, &create_info, Some(self))?;
            let images = Self::wrap_images(&self.device, image_handles, &create_info);
            (handle, images)
        };

        let full_screen_exclusive_held =
            if self.full_screen_exclusive != FullScreenExclusive::ApplicationControlled {
                false
            } else {
                self.full_screen_exclusive_held.load(Ordering::SeqCst)
            };

        let SwapchainCreateInfo {
            min_image_count,
            image_format,
            image_color_space,
            image_extent,
            image_array_layers,
            image_usage,
            image_sharing,
            pre_transform,
            composite_alpha,
            present_mode,
            clipped,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = create_info;

        let swapchain = Arc::new(Swapchain {
            handle,
            device: self.device.clone(),
            surface: self.surface.clone(),

            min_image_count,
            image_format: image_format.unwrap(),
            image_color_space,
            image_extent,
            image_array_layers,
            image_usage,
            image_sharing,
            pre_transform,
            composite_alpha,
            present_mode,
            clipped,
            full_screen_exclusive,
            win32_monitor,

            full_screen_exclusive_held: AtomicBool::new(full_screen_exclusive_held),
            images,
            retired: Mutex::new(false),
        });

        let swapchain_images = (0..swapchain.images.len())
            .map(|n| unsafe { SwapchainImage::from_raw(swapchain.clone(), n) })
            .collect::<Result<_, _>>()?;

        Ok((swapchain, swapchain_images))
    }

    fn validate(
        device: &Device,
        surface: &Surface<W>,
        create_info: &mut SwapchainCreateInfo,
    ) -> Result<(), SwapchainCreationError> {
        let &mut SwapchainCreateInfo {
            min_image_count,
            ref mut image_format,
            image_color_space,
            ref mut image_extent,
            image_array_layers,
            image_usage,
            ref mut image_sharing,
            pre_transform,
            composite_alpha,
            present_mode,
            clipped,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = create_info;

        // VUID-VkSwapchainCreateInfoKHR-imageUsage-requiredbitmask
        assert!(image_usage != ImageUsage::none());

        if full_screen_exclusive != FullScreenExclusive::Default
            && !device.enabled_extensions().ext_full_screen_exclusive
        {
            return Err(SwapchainCreationError::ExtensionNotEnabled {
                extension: "ext_full_screen_exclusive",
                reason: "`full_screen_exclusive` was not `FullScreenExclusive::Default`",
            });
        }

        if surface.api() == SurfaceApi::Win32
            && full_screen_exclusive == FullScreenExclusive::ApplicationControlled
        {
            if win32_monitor.is_none() {
                return Err(SwapchainCreationError::Win32MonitorInvalid);
            }
        } else {
            if win32_monitor.is_some() {
                return Err(SwapchainCreationError::Win32MonitorInvalid);
            }
        }

        // VUID-VkSwapchainCreateInfoKHR-surface-01270
        *image_format = Some({
            let surface_formats = device.physical_device().surface_formats(
                &surface,
                SurfaceInfo {
                    full_screen_exclusive,
                    win32_monitor,
                    ..Default::default()
                },
            )?;

            if let Some(format) = image_format {
                // VUID-VkSwapchainCreateInfoKHR-imageFormat-01273
                if !surface_formats
                    .into_iter()
                    .any(|(f, c)| f == *format && c == image_color_space)
                {
                    return Err(SwapchainCreationError::FormatColorSpaceNotSupported);
                }
                *format
            } else {
                surface_formats
                    .into_iter()
                    .find_map(|(f, c)| {
                        (c == image_color_space
                            && [Format::R8G8B8A8_UNORM, Format::B8G8R8A8_UNORM].contains(&f))
                        .then(|| f)
                    })
                    .ok_or_else(|| SwapchainCreationError::FormatColorSpaceNotSupported)?
            }
        });

        let surface_capabilities = device.physical_device().surface_capabilities(
            &surface,
            SurfaceInfo {
                full_screen_exclusive,
                win32_monitor,
                ..Default::default()
            },
        )?;

        // VUID-VkSwapchainCreateInfoKHR-minImageCount-01272
        // VUID-VkSwapchainCreateInfoKHR-presentMode-02839
        if min_image_count < surface_capabilities.min_image_count
            || surface_capabilities
                .max_image_count
                .map_or(false, |c| min_image_count > c)
        {
            return Err(SwapchainCreationError::MinImageCountNotSupported {
                provided: min_image_count,
                min_supported: surface_capabilities.min_image_count,
                max_supported: surface_capabilities.max_image_count,
            });
        }

        if image_extent[0] == 0 || image_extent[1] == 0 {
            *image_extent = surface_capabilities.current_extent.unwrap();
        }

        // VUID-VkSwapchainCreateInfoKHR-imageExtent-01274
        if image_extent[0] < surface_capabilities.min_image_extent[0]
            || image_extent[1] < surface_capabilities.min_image_extent[1]
            || image_extent[0] > surface_capabilities.max_image_extent[0]
            || image_extent[1] > surface_capabilities.max_image_extent[1]
        {
            return Err(SwapchainCreationError::ImageExtentNotSupported {
                provided: *image_extent,
                min_supported: surface_capabilities.min_image_extent,
                max_supported: surface_capabilities.max_image_extent,
            });
        }

        // VUID-VkSwapchainCreateInfoKHR-imageExtent-01689
        // Shouldn't be possible with a properly behaving device
        assert!(image_extent[0] != 0 || image_extent[1] != 0);

        // VUID-VkSwapchainCreateInfoKHR-imageArrayLayers-01275
        if image_array_layers == 0
            || image_array_layers > surface_capabilities.max_image_array_layers
        {
            return Err(SwapchainCreationError::ImageArrayLayersNotSupported {
                provided: image_array_layers,
                max_supported: surface_capabilities.max_image_array_layers,
            });
        }

        // VUID-VkSwapchainCreateInfoKHR-presentMode-01427
        if (ash::vk::ImageUsageFlags::from(image_usage)
            & ash::vk::ImageUsageFlags::from(surface_capabilities.supported_usage_flags))
            != ash::vk::ImageUsageFlags::from(image_usage)
        {
            return Err(SwapchainCreationError::ImageUsageNotSupported {
                provided: image_usage,
                supported: surface_capabilities.supported_usage_flags,
            });
        }

        match image_sharing {
            Sharing::Exclusive => (),
            Sharing::Concurrent(ids) => {
                // VUID-VkSwapchainCreateInfoKHR-imageSharingMode-01278
                // VUID-VkSwapchainCreateInfoKHR-imageSharingMode-01428
                ids.sort_unstable();
                ids.dedup();
                assert!(ids.len() >= 2);

                for &id in ids.iter() {
                    // VUID-VkSwapchainCreateInfoKHR-imageSharingMode-01428
                    if device.physical_device().queue_family_by_id(id).is_none() {
                        return Err(SwapchainCreationError::ImageSharingInvalidQueueFamilyId {
                            id,
                        });
                    }
                }
            }
        };

        // VUID-VkSwapchainCreateInfoKHR-preTransform-01279
        if !surface_capabilities
            .supported_transforms
            .supports(pre_transform)
        {
            return Err(SwapchainCreationError::PreTransformNotSupported {
                provided: pre_transform,
                supported: surface_capabilities.supported_transforms,
            });
        }

        // VUID-VkSwapchainCreateInfoKHR-compositeAlpha-01280
        if !surface_capabilities
            .supported_composite_alpha
            .supports(composite_alpha)
        {
            return Err(SwapchainCreationError::CompositeAlphaNotSupported {
                provided: composite_alpha,
                supported: surface_capabilities.supported_composite_alpha,
            });
        }

        // VUID-VkSwapchainCreateInfoKHR-presentMode-01281
        if !device
            .physical_device()
            .surface_present_modes(&surface)?
            .any(|mode| mode == present_mode)
        {
            return Err(SwapchainCreationError::PresentModeNotSupported);
        }

        // VUID-VkSwapchainCreateInfoKHR-imageFormat-01778
        if device
            .physical_device()
            .image_format_properties(
                image_format.unwrap(),
                ImageType::Dim2d,
                ImageTiling::Optimal,
                image_usage,
                ImageCreateFlags::none(),
                None,
                None,
            )?
            .is_none()
        {
            return Err(SwapchainCreationError::ImageFormatPropertiesNotSupported);
        }

        Ok(())
    }

    unsafe fn create(
        device: &Device,
        surface: &Surface<W>,
        create_info: &SwapchainCreateInfo,
        old_swapchain: Option<&Swapchain<W>>,
    ) -> Result<(ash::vk::SwapchainKHR, Vec<ash::vk::Image>), SwapchainCreationError> {
        let &SwapchainCreateInfo {
            min_image_count,
            image_format,
            image_color_space,
            image_extent,
            image_array_layers,
            image_usage,
            ref image_sharing,
            pre_transform,
            composite_alpha,
            present_mode,
            clipped,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = create_info;

        let (image_sharing_mode, queue_family_index_count, p_queue_family_indices) =
            match image_sharing {
                Sharing::Exclusive => (ash::vk::SharingMode::EXCLUSIVE, 0, ptr::null()),
                Sharing::Concurrent(ref ids) => (
                    ash::vk::SharingMode::CONCURRENT,
                    ids.len() as u32,
                    ids.as_ptr(),
                ),
            };

        let mut surface_full_screen_exclusive_info =
            if full_screen_exclusive != FullScreenExclusive::Default {
                Some(ash::vk::SurfaceFullScreenExclusiveInfoEXT {
                    full_screen_exclusive: full_screen_exclusive.into(),
                    ..Default::default()
                })
            } else {
                None
            };

        let mut surface_full_screen_exclusive_win32_info =
            if let Some(Win32Monitor(hmonitor)) = win32_monitor {
                Some(ash::vk::SurfaceFullScreenExclusiveWin32InfoEXT {
                    hmonitor,
                    ..Default::default()
                })
            } else {
                None
            };

        let mut create_info = ash::vk::SwapchainCreateInfoKHR {
            flags: ash::vk::SwapchainCreateFlagsKHR::empty(),
            surface: surface.internal_object(),
            min_image_count,
            image_format: image_format.unwrap().into(),
            image_color_space: image_color_space.into(),
            image_extent: ash::vk::Extent2D {
                width: image_extent[0],
                height: image_extent[1],
            },
            image_array_layers,
            image_usage: image_usage.into(),
            image_sharing_mode,
            queue_family_index_count,
            p_queue_family_indices,
            pre_transform: pre_transform.into(),
            composite_alpha: composite_alpha.into(),
            present_mode: present_mode.into(),
            clipped: clipped as ash::vk::Bool32,
            old_swapchain: old_swapchain.map_or(ash::vk::SwapchainKHR::null(), |os| os.handle),
            ..Default::default()
        };

        if let Some(surface_full_screen_exclusive_info) =
            surface_full_screen_exclusive_info.as_mut()
        {
            surface_full_screen_exclusive_info.p_next = create_info.p_next as *mut _;
            create_info.p_next = surface_full_screen_exclusive_info as *const _ as *const _;
        }

        if let Some(surface_full_screen_exclusive_win32_info) =
            surface_full_screen_exclusive_win32_info.as_mut()
        {
            surface_full_screen_exclusive_win32_info.p_next = create_info.p_next as *mut _;
            create_info.p_next = surface_full_screen_exclusive_win32_info as *const _ as *const _;
        }

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.khr_swapchain.create_swapchain_khr(
                device.internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        let image_handles = {
            let fns = device.fns();
            let mut num = 0;
            check_errors(fns.khr_swapchain.get_swapchain_images_khr(
                device.internal_object(),
                handle,
                &mut num,
                ptr::null_mut(),
            ))?;

            let mut images = Vec::with_capacity(num as usize);
            check_errors(fns.khr_swapchain.get_swapchain_images_khr(
                device.internal_object(),
                handle,
                &mut num,
                images.as_mut_ptr(),
            ))?;
            images.set_len(num as usize);
            images
        };

        Ok((handle, image_handles))
    }

    unsafe fn wrap_images(
        device: &Arc<Device>,
        image_handles: Vec<ash::vk::Image>,
        create_info: &SwapchainCreateInfo,
    ) -> Vec<ImageEntry> {
        let &SwapchainCreateInfo {
            image_format,
            image_extent,
            image_array_layers,
            image_usage,
            ref image_sharing, // TODO: put this in the image too
            ..
        } = create_info;

        image_handles
            .into_iter()
            .map(|handle| {
                let dims = ImageDimensions::Dim2d {
                    width: image_extent[0],
                    height: image_extent[1],
                    array_layers: image_array_layers,
                };

                let img = unsafe {
                    UnsafeImage::from_raw(
                        device.clone(),
                        handle,
                        image_usage,
                        image_format.unwrap(),
                        ImageCreateFlags::none(),
                        dims,
                        SampleCount::Sample1,
                        1,
                    )
                };

                ImageEntry {
                    image: img,
                    undefined_layout: AtomicBool::new(true),
                }
            })
            .collect()
    }

    /// Returns the creation parameters of the swapchain.
    #[inline]
    pub fn create_info(&self) -> SwapchainCreateInfo {
        SwapchainCreateInfo {
            min_image_count: self.min_image_count,
            image_format: Some(self.image_format),
            image_color_space: self.image_color_space,
            image_extent: self.image_extent,
            image_array_layers: self.image_array_layers,
            image_usage: self.image_usage,
            image_sharing: self.image_sharing.clone(),
            pre_transform: self.pre_transform,
            composite_alpha: self.composite_alpha,
            present_mode: self.present_mode,
            clipped: self.clipped,
            full_screen_exclusive: self.full_screen_exclusive,
            win32_monitor: self.win32_monitor,
            _ne: crate::NonExhaustive(()),
        }
    }

    /// Returns the saved Surface, from the Swapchain creation.
    #[inline]
    pub fn surface(&self) -> &Arc<Surface<W>> {
        &self.surface
    }

    /// Returns of the images that belong to this swapchain.
    #[inline]
    pub fn raw_image(&self, offset: usize) -> Option<ImageInner> {
        self.images.get(offset).map(|i| ImageInner {
            image: &i.image,
            first_layer: 0,
            num_layers: self.image_array_layers as usize,
            first_mipmap_level: 0,
            num_mipmap_levels: 1,
        })
    }

    /// Returns the number of images of the swapchain.
    #[inline]
    pub fn image_count(&self) -> u32 {
        self.images.len() as u32
    }

    /// Returns the format of the images of the swapchain.
    #[inline]
    pub fn image_format(&self) -> Format {
        self.image_format
    }

    /// Returns the color space of the images of the swapchain.
    #[inline]
    pub fn image_color_space(&self) -> ColorSpace {
        self.image_color_space
    }

    /// Returns the extent of the images of the swapchain.
    #[inline]
    pub fn image_extent(&self) -> [u32; 2] {
        self.image_extent
    }

    /// Returns the number of array layers of the images of the swapchain.
    #[inline]
    pub fn image_array_layers(&self) -> u32 {
        self.image_array_layers
    }

    /// Returns the pre-transform that was passed when creating the swapchain.
    #[inline]
    pub fn pre_transform(&self) -> SurfaceTransform {
        self.pre_transform
    }

    /// Returns the alpha mode that was passed when creating the swapchain.
    #[inline]
    pub fn composite_alpha(&self) -> CompositeAlpha {
        self.composite_alpha
    }

    /// Returns the present mode that was passed when creating the swapchain.
    #[inline]
    pub fn present_mode(&self) -> PresentMode {
        self.present_mode
    }

    /// Returns the value of `clipped` that was passed when creating the swapchain.
    #[inline]
    pub fn clipped(&self) -> bool {
        self.clipped
    }

    /// Returns the value of 'full_screen_exclusive` that was passed when creating the swapchain.
    #[inline]
    pub fn full_screen_exclusive(&self) -> FullScreenExclusive {
        self.full_screen_exclusive
    }

    /// Acquires full-screen exclusivity.
    ///
    /// The swapchain must have been created with [`FullScreenExclusive::ApplicationControlled`],
    /// and must not already hold full-screen exclusivity. Full-screen exclusivity is held until
    /// either the `release_full_screen_exclusive` is called, or if any of the the other `Swapchain`
    /// functions return `FullScreenExclusiveLost`.
    pub fn acquire_full_screen_exclusive(&self) -> Result<(), FullScreenExclusiveError> {
        if self.full_screen_exclusive != FullScreenExclusive::ApplicationControlled {
            return Err(FullScreenExclusiveError::NotApplicationControlled);
        }

        if self.full_screen_exclusive_held.swap(true, Ordering::SeqCst) {
            return Err(FullScreenExclusiveError::DoubleAcquire);
        }

        unsafe {
            check_errors(
                self.device
                    .fns()
                    .ext_full_screen_exclusive
                    .acquire_full_screen_exclusive_mode_ext(
                        self.device.internal_object(),
                        self.handle,
                    ),
            )?;
        }

        Ok(())
    }

    /// Releases full-screen exclusivity.
    ///
    /// The swapchain must have been created with [`FullScreenExclusive::ApplicationControlled`],
    /// and must currently hold full-screen exclusivity.
    pub fn release_full_screen_exclusive(&self) -> Result<(), FullScreenExclusiveError> {
        if self.full_screen_exclusive != FullScreenExclusive::ApplicationControlled {
            return Err(FullScreenExclusiveError::NotApplicationControlled);
        }

        if !self
            .full_screen_exclusive_held
            .swap(false, Ordering::SeqCst)
        {
            return Err(FullScreenExclusiveError::DoubleRelease);
        }

        unsafe {
            check_errors(
                self.device
                    .fns()
                    .ext_full_screen_exclusive
                    .release_full_screen_exclusive_mode_ext(
                        self.device.internal_object(),
                        self.handle,
                    ),
            )?;
        }

        Ok(())
    }

    /// `FullScreenExclusive::AppControlled` is not the active full-screen exclusivity mode,
    /// then this function will always return false. If true is returned the swapchain
    /// is in `FullScreenExclusive::AppControlled` full-screen exclusivity mode and exclusivity
    /// is currently acquired.
    pub fn is_full_screen_exclusive(&self) -> bool {
        if self.full_screen_exclusive != FullScreenExclusive::ApplicationControlled {
            false
        } else {
            self.full_screen_exclusive_held.load(Ordering::SeqCst)
        }
    }

    // This method is necessary to allow `SwapchainImage`s to signal when they have been
    // transitioned out of their initial `undefined` image layout.
    //
    // See the `ImageAccess::layout_initialized` method documentation for more details.
    pub(crate) fn image_layout_initialized(&self, image_offset: usize) {
        let image_entry = self.images.get(image_offset);
        if let Some(ref image_entry) = image_entry {
            image_entry.undefined_layout.store(false, Ordering::SeqCst);
        }
    }

    pub(crate) fn is_image_layout_initialized(&self, image_offset: usize) -> bool {
        let image_entry = self.images.get(image_offset);
        if let Some(ref image_entry) = image_entry {
            !image_entry.undefined_layout.load(Ordering::SeqCst)
        } else {
            false
        }
    }
}

impl<W> Drop for Swapchain<W> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            fns.khr_swapchain.destroy_swapchain_khr(
                self.device.internal_object(),
                self.handle,
                ptr::null(),
            );
            self.surface.flag().store(false, Ordering::Release);
        }
    }
}

unsafe impl<W> VulkanObject for Swapchain<W> {
    type Object = ash::vk::SwapchainKHR;

    #[inline]
    fn internal_object(&self) -> ash::vk::SwapchainKHR {
        self.handle
    }
}

unsafe impl<W> DeviceOwned for Swapchain<W> {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl<W> PartialEq for Swapchain<W> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.device() == other.device()
    }
}

impl<W> Eq for Swapchain<W> {}

impl<W> Hash for Swapchain<W> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device().hash(state);
    }
}

/// Parameters to create a new `Swapchain`.
///
/// Many of the values here must be supported by the physical device.
/// [`PhysicalDevice`](crate::device::physical::PhysicalDevice) has several
/// methods to query what is supported.
#[derive(Clone, Debug)]
pub struct SwapchainCreateInfo {
    /// The minimum number of images that will be created.
    ///
    /// The implementation is allowed to create more than this number, but never less.
    ///
    /// The default value is `2`.
    pub min_image_count: u32,

    /// The format of the created images.
    ///
    /// If set to `None`, [`Format::R8G8B8A8_UNORM`] or [`Format::B8G8R8A8_UNORM`] will be selected,
    /// based on which is supported by the surface.
    ///
    /// The default value is `None`.
    pub image_format: Option<Format>,

    /// The color space of the created images.
    ///
    /// The default value is [`ColorSpace::SrgbNonLinear`].
    pub image_color_space: ColorSpace,

    /// The extent of the created images.
    ///
    /// If set to `None`, the value of
    /// [`SurfaceCapabilities::current_extent`](crate::swapchain::SurfaceCapabilities) will be used.
    ///
    /// The default value is `None`.
    pub image_extent: [u32; 2],

    /// The number of array layers of the created images.
    ///
    /// The default value is `1`.
    pub image_array_layers: u32,

    /// How the created images will be used.
    ///
    /// The default value is [`ImageUsage::none()`], which must be overridden.
    pub image_usage: ImageUsage,

    /// Whether the created images can be shared across multiple queues, or are limited to a single
    /// queue.
    ///
    /// The default value is [`Sharing::Exclusive`].
    pub image_sharing: Sharing<SmallVec<[u32; 4]>>,

    /// The transform that should be applied to an image before it is presented.
    ///
    /// The default value is [`SurfaceTransform::Identity`].
    pub pre_transform: SurfaceTransform,

    /// How alpha values of the pixels in the image are to be treated.
    ///
    /// The default value is [`CompositeAlpha::Opaque`].
    pub composite_alpha: CompositeAlpha,

    /// How the swapchain should behave when multiple images are waiting in the queue to be
    /// presented.
    ///
    /// The default is [`PresentMode::Fifo`].
    pub present_mode: PresentMode,

    /// Whether the implementation is allowed to discard rendering operations that affect regions of
    /// the surface which aren't visible. This is important to take into account if your fragment
    /// shader has side-effects or if you want to read back the content of the image afterwards.
    ///
    /// The default value is `true`.
    pub clipped: bool,

    /// How full-screen exclusivity is to be handled.
    ///
    /// If set to anything other than [`FullScreenExclusive::Default`], then the
    /// [`ext_full_screen_exclusive`](crate::device::DeviceExtensions::ext_full_screen_exclusive)
    /// extension must be enabled on the device.
    ///
    /// The default value is [`FullScreenExclusive::Default`].
    pub full_screen_exclusive: FullScreenExclusive,

    /// For Win32 surfaces, if `full_screen_exclusive` is
    /// [`FullScreenExclusive::ApplicationControlled`], this specifies the monitor on which
    /// full-screen exclusivity should be used.
    ///
    /// For this case, the value must be `Some`, and for all others it must be `None`.
    ///
    /// The default value is `None`.
    pub win32_monitor: Option<Win32Monitor>,

    pub _ne: crate::NonExhaustive,
}

impl Default for SwapchainCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            min_image_count: 2,
            image_format: None,
            image_color_space: ColorSpace::SrgbNonLinear,
            image_extent: [0, 0],
            image_array_layers: 1,
            image_usage: ImageUsage::none(),
            image_sharing: Sharing::Exclusive,
            pre_transform: SurfaceTransform::Identity,
            composite_alpha: CompositeAlpha::Opaque,
            present_mode: PresentMode::Fifo,
            clipped: true,
            full_screen_exclusive: FullScreenExclusive::Default,
            win32_monitor: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Error that can happen when creating a `Swapchain`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SwapchainCreationError {
    /// Not enough memory.
    OomError(OomError),

    /// The device was lost.
    DeviceLost,

    /// The surface was lost.
    SurfaceLost,

    /// The surface is already used by another swapchain.
    SurfaceInUse,

    /// The window is already in use by another API.
    NativeWindowInUse,

    ExtensionNotEnabled {
        extension: &'static str,
        reason: &'static str,
    },

    /// The provided `composite_alpha` is not supported by the surface for this device.
    CompositeAlphaNotSupported {
        provided: CompositeAlpha,
        supported: SupportedCompositeAlpha,
    },

    /// The provided `format` and `color_space` are not supported by the surface for this device.
    FormatColorSpaceNotSupported,

    /// The provided `image_array_layers` is greater than what is supported by the surface for this
    /// device.
    ImageArrayLayersNotSupported { provided: u32, max_supported: u32 },

    /// The provided `image_extent` is not within the range supported by the surface for this
    /// device.
    ImageExtentNotSupported {
        provided: [u32; 2],
        min_supported: [u32; 2],
        max_supported: [u32; 2],
    },

    /// The provided image parameters are not supported as queried from `image_format_properties`.
    ImageFormatPropertiesNotSupported,

    /// The provided `image_sharing` was set to `Concurrent`, but one of the specified queue family
    /// ids was not valid.
    ImageSharingInvalidQueueFamilyId { id: u32 },

    /// The provided `image_usage` has fields set that are not supported by the surface for this
    /// device.
    ImageUsageNotSupported {
        provided: ImageUsage,
        supported: ImageUsage,
    },

    /// The provided `min_image_count` is not within the range supported by the surface for this
    /// device.
    MinImageCountNotSupported {
        provided: u32,
        min_supported: u32,
        max_supported: Option<u32>,
    },

    /// The provided `present_mode` is not supported by the surface for this device.
    PresentModeNotSupported,

    /// The provided `pre_transform` is not supported by the surface for this device.
    PreTransformNotSupported {
        provided: SurfaceTransform,
        supported: SupportedSurfaceTransforms,
    },

    /// The swapchain has already been used to create a new one.
    SwapchainAlreadyRetired,

    /// The `win32_monitor` value was `Some` when it must be `None` or vice-versa.
    Win32MonitorInvalid,
}

impl error::Error for SwapchainCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Self::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for SwapchainCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            Self::OomError(_) => write!(fmt, "not enough memory available",),
            Self::DeviceLost => write!(fmt, "the device was lost",),
            Self::SurfaceLost => write!(fmt, "the surface was lost",),
            Self::SurfaceInUse => {
                write!(fmt, "the surface is already used by another swapchain",)
            }
            Self::NativeWindowInUse => {
                write!(fmt, "the window is already in use by another API")
            }

            Self::ExtensionNotEnabled { extension, reason } => write!(
                fmt,
                "the extension {} must be enabled: {}",
                extension, reason
            ),

            Self::CompositeAlphaNotSupported { .. } => write!(
                fmt,
                "the provided `composite_alpha` is not supported by the surface for this device",
            ),
            Self::FormatColorSpaceNotSupported => write!(
                fmt,
                "the provided `format` and `color_space` are not supported by the surface for this device",
            ),
            Self::ImageArrayLayersNotSupported { provided, max_supported } => write!(
                fmt,
                "the provided `image_array_layers` ({}) is greater than what is supported ({}) by the surface for this device",
                provided, max_supported,
            ),
            Self::ImageExtentNotSupported { provided, min_supported, max_supported } => write!(
                fmt,
                "the provided `min_image_count` ({:?}) is not within the range (min: {:?}, max: {:?}) supported by the surface for this device",
                provided, min_supported, max_supported,
            ),
            Self::ImageFormatPropertiesNotSupported => write!(
                fmt,
                "the provided image parameters are not supported as queried from `image_format_properties`",
            ),
            Self::ImageSharingInvalidQueueFamilyId { id } => write!(
                fmt,
                "the provided `image_sharing` was set to `Concurrent`, but one of the specified queue family ids ({}) was not valid",
                id,
            ),
            Self::ImageUsageNotSupported { .. } => write!(
                fmt,
                "the provided `image_usage` has fields set that are not supported by the surface for this device",
            ),
            Self::MinImageCountNotSupported { provided, min_supported, max_supported } => write!(
                fmt,
                "the provided `min_image_count` ({}) is not within the range (min: {}, max: {:?}) supported by the surface for this device",
                provided, min_supported, max_supported,
            ),
            Self::PresentModeNotSupported => write!(
                fmt,
                "the provided `present_mode` is not supported by the surface for this device",
            ),
            Self::PreTransformNotSupported { .. } => write!(
                fmt,
                "the provided `pre_transform` is not supported by the surface for this device",
            ),
            Self::SwapchainAlreadyRetired => write!(
                fmt,
                "the swapchain has already been used to create a new one",
            ),
            Self::Win32MonitorInvalid => write!(
                fmt,
                "the `win32_monitor` value was `Some` when it must be `None` or vice-versa",
            ),
        }
    }
}

impl From<Error> for SwapchainCreationError {
    #[inline]
    fn from(err: Error) -> SwapchainCreationError {
        match err {
            err @ Error::OutOfHostMemory => Self::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => Self::OomError(OomError::from(err)),
            Error::DeviceLost => Self::DeviceLost,
            Error::SurfaceLost => Self::SurfaceLost,
            Error::NativeWindowInUse => Self::NativeWindowInUse,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<OomError> for SwapchainCreationError {
    #[inline]
    fn from(err: OomError) -> SwapchainCreationError {
        Self::OomError(err)
    }
}

impl From<SurfacePropertiesError> for SwapchainCreationError {
    #[inline]
    fn from(err: SurfacePropertiesError) -> SwapchainCreationError {
        match err {
            SurfacePropertiesError::OomError(err) => Self::OomError(err),
            SurfacePropertiesError::SurfaceLost => Self::SurfaceLost,
            SurfacePropertiesError::NotSupported => unreachable!(),
        }
    }
}

/// The way full-screen exclusivity is handled.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(i32)]
#[non_exhaustive]
pub enum FullScreenExclusive {
    /// Indicates that the driver should determine the appropriate full-screen method
    /// by whatever means it deems appropriate.
    Default = ash::vk::FullScreenExclusiveEXT::DEFAULT.as_raw(),

    /// Indicates that the driver may use full-screen exclusive mechanisms when available.
    /// Such mechanisms may result in better performance and/or the availability of
    /// different presentation capabilities, but may require a more disruptive transition
    // during swapchain initialization, first presentation and/or destruction.
    Allowed = ash::vk::FullScreenExclusiveEXT::ALLOWED.as_raw(),

    /// Indicates that the driver should avoid using full-screen mechanisms which rely
    /// on disruptive transitions.
    Disallowed = ash::vk::FullScreenExclusiveEXT::DISALLOWED.as_raw(),

    /// Indicates the application will manage full-screen exclusive mode by using the
    /// [`Swapchain::acquire_full_screen_exclusive()`] and
    /// [`Swapchain::release_full_screen_exclusive()`] functions.
    ApplicationControlled = ash::vk::FullScreenExclusiveEXT::APPLICATION_CONTROLLED.as_raw(),
}

impl From<FullScreenExclusive> for ash::vk::FullScreenExclusiveEXT {
    #[inline]
    fn from(val: FullScreenExclusive) -> Self {
        Self::from_raw(val as i32)
    }
}

/// A wrapper around a Win32 monitor handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Win32Monitor(pub(crate) ash::vk::HMONITOR);

impl Win32Monitor {
    /// Wraps a Win32 monitor handle.
    ///
    /// # Safety
    ///
    /// - `hmonitor` must be a valid handle as returned by the Win32 API.
    #[inline]
    pub unsafe fn new<T>(hmonitor: *const T) -> Self {
        Self(hmonitor as _)
    }
}

// Winit's `MonitorHandle` is Send on Win32, so this seems safe.
unsafe impl Send for Win32Monitor {}
unsafe impl Sync for Win32Monitor {}

/// Error that can happen when calling `Swapchain::acquire_full_screen_exclusive` or
/// `Swapchain::release_full_screen_exclusive`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FullScreenExclusiveError {
    /// Not enough memory.
    OomError(OomError),

    /// Operation could not be completed for driver specific reasons.
    InitializationFailed,

    /// The surface is no longer accessible and must be recreated.
    SurfaceLost,

    /// Full-screen exclusivity is already acquired.
    DoubleAcquire,

    /// Full-screen exclusivity is not currently acquired.
    DoubleRelease,

    /// The swapchain is not in full-screen exclusive application controlled mode.
    NotApplicationControlled,
}

impl error::Error for FullScreenExclusiveError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            FullScreenExclusiveError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for FullScreenExclusiveError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                FullScreenExclusiveError::OomError(_) => "not enough memory",
                FullScreenExclusiveError::SurfaceLost => {
                    "the surface of this swapchain is no longer valid"
                }
                FullScreenExclusiveError::InitializationFailed => {
                    "operation could not be completed for driver specific reasons"
                }
                FullScreenExclusiveError::DoubleAcquire =>
                    "full-screen exclusivity is already acquired",
                FullScreenExclusiveError::DoubleRelease =>
                    "full-screen exclusivity is not acquired",
                FullScreenExclusiveError::NotApplicationControlled => {
                    "the swapchain is not in full-screen exclusive application controlled mode"
                }
            }
        )
    }
}

impl From<Error> for FullScreenExclusiveError {
    #[inline]
    fn from(err: Error) -> FullScreenExclusiveError {
        match err {
            err @ Error::OutOfHostMemory => FullScreenExclusiveError::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => {
                FullScreenExclusiveError::OomError(OomError::from(err))
            }
            Error::SurfaceLost => FullScreenExclusiveError::SurfaceLost,
            Error::InitializationFailed => FullScreenExclusiveError::InitializationFailed,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<OomError> for FullScreenExclusiveError {
    #[inline]
    fn from(err: OomError) -> FullScreenExclusiveError {
        FullScreenExclusiveError::OomError(err)
    }
}

/// Tries to take ownership of an image in order to draw on it.
///
/// The function returns the index of the image in the array of images that was returned
/// when creating the swapchain, plus a future that represents the moment when the image will
/// become available from the GPU (which may not be *immediately*).
///
/// If you try to draw on an image without acquiring it first, the execution will block. (TODO
/// behavior may change).
///
/// The second field in the tuple in the Ok result is a bool represent if the acquisition was
/// suboptimal. In this case the acquired image is still usable, but the swapchain should be
/// recreated as the Surface's properties no longer match the swapchain.
pub fn acquire_next_image<W>(
    swapchain: Arc<Swapchain<W>>,
    timeout: Option<Duration>,
) -> Result<(usize, bool, SwapchainAcquireFuture<W>), AcquireError> {
    let semaphore = Semaphore::from_pool(swapchain.device.clone())?;
    let fence = Fence::from_pool(swapchain.device.clone())?;

    let AcquiredImage { id, suboptimal } = {
        // Check that this is not an old swapchain. From specs:
        // > swapchain must not have been replaced by being passed as the
        // > VkSwapchainCreateInfoKHR::oldSwapchain value to vkCreateSwapchainKHR
        let retired = swapchain.retired.lock().unwrap();
        if *retired {
            return Err(AcquireError::OutOfDate);
        }

        let acquire_result =
            unsafe { acquire_next_image_raw(&swapchain, timeout, Some(&semaphore), Some(&fence)) };

        if let &Err(AcquireError::FullScreenExclusiveLost) = &acquire_result {
            swapchain
                .full_screen_exclusive_held
                .store(false, Ordering::SeqCst);
        }

        acquire_result?
    };

    Ok((
        id,
        suboptimal,
        SwapchainAcquireFuture {
            swapchain,
            semaphore: Some(semaphore),
            fence: Some(fence),
            image_id: id,
            finished: AtomicBool::new(false),
        },
    ))
}

/// Presents an image on the screen.
///
/// The parameter is the same index as what `acquire_next_image` returned. The image must
/// have been acquired first.
///
/// The actual behavior depends on the present mode that you passed when creating the
/// swapchain.
pub fn present<F, W>(
    swapchain: Arc<Swapchain<W>>,
    before: F,
    queue: Arc<Queue>,
    index: usize,
) -> PresentFuture<F, W>
where
    F: GpuFuture,
{
    assert!(index < swapchain.images.len());

    // TODO: restore this check with a dummy ImageAccess implementation
    /*let swapchain_image = me.images.lock().unwrap().get(index).unwrap().0.upgrade().unwrap();       // TODO: return error instead
    // Normally if `check_image_access` returns false we're supposed to call the `gpu_access`
    // function on the image instead. But since we know that this method on `SwapchainImage`
    // always returns false anyway (by design), we don't need to do it.
    assert!(before.check_image_access(&swapchain_image, ImageLayout::PresentSrc, true, &queue).is_ok());         // TODO: return error instead*/

    PresentFuture {
        previous: before,
        queue,
        swapchain,
        image_id: index,
        present_region: None,
        flushed: AtomicBool::new(false),
        finished: AtomicBool::new(false),
    }
}

/// Same as `swapchain::present`, except it allows specifying a present region.
/// Areas outside the present region may be ignored by Vulkan in order to optimize presentation.
///
/// This is just an optimization hint, as the Vulkan driver is free to ignore the given present region.
///
/// If `VK_KHR_incremental_present` is not enabled on the device, the parameter will be ignored.
pub fn present_incremental<F, W>(
    swapchain: Arc<Swapchain<W>>,
    before: F,
    queue: Arc<Queue>,
    index: usize,
    present_region: PresentRegion,
) -> PresentFuture<F, W>
where
    F: GpuFuture,
{
    assert!(index < swapchain.images.len());

    // TODO: restore this check with a dummy ImageAccess implementation
    /*let swapchain_image = me.images.lock().unwrap().get(index).unwrap().0.upgrade().unwrap();       // TODO: return error instead
    // Normally if `check_image_access` returns false we're supposed to call the `gpu_access`
    // function on the image instead. But since we know that this method on `SwapchainImage`
    // always returns false anyway (by design), we don't need to do it.
    assert!(before.check_image_access(&swapchain_image, ImageLayout::PresentSrc, true, &queue).is_ok());         // TODO: return error instead*/

    PresentFuture {
        previous: before,
        queue,
        swapchain,
        image_id: index,
        present_region: Some(present_region),
        flushed: AtomicBool::new(false),
        finished: AtomicBool::new(false),
    }
}

/// Represents the moment when the GPU will have access to a swapchain image.
#[must_use]
pub struct SwapchainAcquireFuture<W> {
    swapchain: Arc<Swapchain<W>>,
    image_id: usize,
    // Semaphore that is signalled when the acquire is complete. Empty if the acquire has already
    // happened.
    semaphore: Option<Semaphore>,
    // Fence that is signalled when the acquire is complete. Empty if the acquire has already
    // happened.
    fence: Option<Fence>,
    finished: AtomicBool,
}

impl<W> SwapchainAcquireFuture<W> {
    /// Returns the index of the image in the list of images returned when creating the swapchain.
    #[inline]
    pub fn image_id(&self) -> usize {
        self.image_id
    }

    /// Returns the corresponding swapchain.
    #[inline]
    pub fn swapchain(&self) -> &Arc<Swapchain<W>> {
        &self.swapchain
    }
}

unsafe impl<W> GpuFuture for SwapchainAcquireFuture<W> {
    #[inline]
    fn cleanup_finished(&mut self) {}

    #[inline]
    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError> {
        if let Some(ref semaphore) = self.semaphore {
            let mut sem = SubmitSemaphoresWaitBuilder::new();
            sem.add_wait_semaphore(&semaphore);
            Ok(SubmitAnyBuilder::SemaphoresWait(sem))
        } else {
            Ok(SubmitAnyBuilder::Empty)
        }
    }

    #[inline]
    fn flush(&self) -> Result<(), FlushError> {
        Ok(())
    }

    #[inline]
    unsafe fn signal_finished(&self) {
        self.finished.store(true, Ordering::SeqCst);
    }

    #[inline]
    fn queue_change_allowed(&self) -> bool {
        true
    }

    #[inline]
    fn queue(&self) -> Option<Arc<Queue>> {
        None
    }

    #[inline]
    fn check_buffer_access(
        &self,
        _: &dyn BufferAccess,
        _: bool,
        _: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        Err(AccessCheckError::Unknown)
    }

    #[inline]
    fn check_image_access(
        &self,
        image: &dyn ImageAccess,
        layout: ImageLayout,
        _: bool,
        _: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        let swapchain_image = self.swapchain.raw_image(self.image_id).unwrap();
        if swapchain_image.image.internal_object() != image.inner().image.internal_object() {
            return Err(AccessCheckError::Unknown);
        }

        if self.swapchain.images[self.image_id]
            .undefined_layout
            .load(Ordering::Relaxed)
            && layout != ImageLayout::Undefined
        {
            return Err(AccessCheckError::Denied(AccessError::ImageNotInitialized {
                requested: layout,
            }));
        }

        if layout != ImageLayout::Undefined && layout != ImageLayout::PresentSrc {
            return Err(AccessCheckError::Denied(
                AccessError::UnexpectedImageLayout {
                    allowed: ImageLayout::PresentSrc,
                    requested: layout,
                },
            ));
        }

        Ok(None)
    }
}

impl<W> Drop for SwapchainAcquireFuture<W> {
    fn drop(&mut self) {
        if let Some(ref fence) = self.fence {
            fence.wait(None).unwrap(); // TODO: handle error?
            self.semaphore = None;
        }

        // TODO: if this future is destroyed without being presented, then eventually acquiring
        // a new image will block forever ; difficulty: hard
    }
}

unsafe impl<W> DeviceOwned for SwapchainAcquireFuture<W> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.swapchain.device
    }
}

/// Error that can happen when calling `acquire_next_image`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum AcquireError {
    /// Not enough memory.
    OomError(OomError),

    /// The connection to the device has been lost.
    DeviceLost,

    /// The timeout of the function has been reached before an image was available.
    Timeout,

    /// The surface is no longer accessible and must be recreated.
    SurfaceLost,

    /// The swapchain has lost or doesn't have full-screen exclusivity possibly for
    /// implementation-specific reasons outside of the application’s control.
    FullScreenExclusiveLost,

    /// The surface has changed in a way that makes the swapchain unusable. You must query the
    /// surface's new properties and recreate a new swapchain if you want to continue drawing.
    OutOfDate,

    /// Error during semaphore creation
    SemaphoreError(SemaphoreError),
}

impl error::Error for AcquireError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            AcquireError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for AcquireError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                AcquireError::OomError(_) => "not enough memory",
                AcquireError::DeviceLost => "the connection to the device has been lost",
                AcquireError::Timeout => "no image is available for acquiring yet",
                AcquireError::SurfaceLost => "the surface of this swapchain is no longer valid",
                AcquireError::OutOfDate => "the swapchain needs to be recreated",
                AcquireError::FullScreenExclusiveLost => {
                    "the swapchain no longer has full-screen exclusivity"
                }
                AcquireError::SemaphoreError(_) => "error creating semaphore",
            }
        )
    }
}

impl From<SemaphoreError> for AcquireError {
    fn from(err: SemaphoreError) -> Self {
        AcquireError::SemaphoreError(err)
    }
}

impl From<OomError> for AcquireError {
    #[inline]
    fn from(err: OomError) -> AcquireError {
        AcquireError::OomError(err)
    }
}

impl From<Error> for AcquireError {
    #[inline]
    fn from(err: Error) -> AcquireError {
        match err {
            err @ Error::OutOfHostMemory => AcquireError::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => AcquireError::OomError(OomError::from(err)),
            Error::DeviceLost => AcquireError::DeviceLost,
            Error::SurfaceLost => AcquireError::SurfaceLost,
            Error::OutOfDate => AcquireError::OutOfDate,
            Error::FullScreenExclusiveLost => AcquireError::FullScreenExclusiveLost,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

/// Represents a swapchain image being presented on the screen.
#[must_use = "Dropping this object will immediately block the thread until the GPU has finished processing the submission"]
pub struct PresentFuture<P, W>
where
    P: GpuFuture,
{
    previous: P,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain<W>>,
    image_id: usize,
    present_region: Option<PresentRegion>,
    // True if `flush()` has been called on the future, which means that the present command has
    // been submitted.
    flushed: AtomicBool,
    // True if `signal_finished()` has been called on the future, which means that the future has
    // been submitted and has already been processed by the GPU.
    finished: AtomicBool,
}

impl<P, W> PresentFuture<P, W>
where
    P: GpuFuture,
{
    /// Returns the index of the image in the list of images returned when creating the swapchain.
    #[inline]
    pub fn image_id(&self) -> usize {
        self.image_id
    }

    /// Returns the corresponding swapchain.
    #[inline]
    pub fn swapchain(&self) -> &Arc<Swapchain<W>> {
        &self.swapchain
    }
}

unsafe impl<P, W> GpuFuture for PresentFuture<P, W>
where
    P: GpuFuture,
{
    #[inline]
    fn cleanup_finished(&mut self) {
        self.previous.cleanup_finished();
    }

    #[inline]
    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError> {
        if self.flushed.load(Ordering::SeqCst) {
            return Ok(SubmitAnyBuilder::Empty);
        }

        let queue = self.previous.queue().map(|q| q.clone());

        // TODO: if the swapchain image layout is not PRESENT, should add a transition command
        // buffer

        Ok(match self.previous.build_submission()? {
            SubmitAnyBuilder::Empty => {
                let mut builder = SubmitPresentBuilder::new();
                builder.add_swapchain(
                    &self.swapchain,
                    self.image_id as u32,
                    self.present_region.as_ref(),
                );
                SubmitAnyBuilder::QueuePresent(builder)
            }
            SubmitAnyBuilder::SemaphoresWait(sem) => {
                let mut builder: SubmitPresentBuilder = sem.into();
                builder.add_swapchain(
                    &self.swapchain,
                    self.image_id as u32,
                    self.present_region.as_ref(),
                );
                SubmitAnyBuilder::QueuePresent(builder)
            }
            SubmitAnyBuilder::CommandBuffer(cb) => {
                // submit the command buffer by flushing previous.
                // Since the implementation should remember being flushed it's safe to call build_submission multiple times
                self.previous.flush()?;

                let mut builder = SubmitPresentBuilder::new();
                builder.add_swapchain(
                    &self.swapchain,
                    self.image_id as u32,
                    self.present_region.as_ref(),
                );
                SubmitAnyBuilder::QueuePresent(builder)
            }
            SubmitAnyBuilder::BindSparse(cb) => {
                // submit the command buffer by flushing previous.
                // Since the implementation should remember being flushed it's safe to call build_submission multiple times
                self.previous.flush()?;

                let mut builder = SubmitPresentBuilder::new();
                builder.add_swapchain(
                    &self.swapchain,
                    self.image_id as u32,
                    self.present_region.as_ref(),
                );
                SubmitAnyBuilder::QueuePresent(builder)
            }
            SubmitAnyBuilder::QueuePresent(present) => {
                unimplemented!() // TODO:
                                 /*present.submit();
                                 let mut builder = SubmitPresentBuilder::new();
                                 builder.add_swapchain(self.command_buffer.inner(), self.image_id);
                                 SubmitAnyBuilder::CommandBuffer(builder)*/
            }
        })
    }

    #[inline]
    fn flush(&self) -> Result<(), FlushError> {
        unsafe {
            // If `flushed` already contains `true`, then `build_submission` will return `Empty`.

            let build_submission_result = self.build_submission();

            if let &Err(FlushError::FullScreenExclusiveLost) = &build_submission_result {
                self.swapchain
                    .full_screen_exclusive_held
                    .store(false, Ordering::SeqCst);
            }

            match build_submission_result? {
                SubmitAnyBuilder::Empty => {}
                SubmitAnyBuilder::QueuePresent(present) => {
                    let present_result = present.submit(&self.queue);

                    if let &Err(SubmitPresentError::FullScreenExclusiveLost) = &present_result {
                        self.swapchain
                            .full_screen_exclusive_held
                            .store(false, Ordering::SeqCst);
                    }

                    present_result?;
                }
                _ => unreachable!(),
            }

            self.flushed.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    #[inline]
    unsafe fn signal_finished(&self) {
        self.flushed.store(true, Ordering::SeqCst);
        self.finished.store(true, Ordering::SeqCst);
        self.previous.signal_finished();
    }

    #[inline]
    fn queue_change_allowed(&self) -> bool {
        false
    }

    #[inline]
    fn queue(&self) -> Option<Arc<Queue>> {
        debug_assert!(match self.previous.queue() {
            None => true,
            Some(q) => q.is_same(&self.queue),
        });

        Some(self.queue.clone())
    }

    #[inline]
    fn check_buffer_access(
        &self,
        buffer: &dyn BufferAccess,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        self.previous.check_buffer_access(buffer, exclusive, queue)
    }

    #[inline]
    fn check_image_access(
        &self,
        image: &dyn ImageAccess,
        layout: ImageLayout,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        let swapchain_image = self.swapchain.raw_image(self.image_id).unwrap();
        if swapchain_image.image.internal_object() == image.inner().image.internal_object() {
            // This future presents the swapchain image, which "unlocks" it. Therefore any attempt
            // to use this swapchain image afterwards shouldn't get granted automatic access.
            // Instead any attempt to access the image afterwards should get an authorization from
            // a later swapchain acquire future. Hence why we return `Unknown` here.
            Err(AccessCheckError::Unknown)
        } else {
            self.previous
                .check_image_access(image, layout, exclusive, queue)
        }
    }
}

unsafe impl<P, W> DeviceOwned for PresentFuture<P, W>
where
    P: GpuFuture,
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.queue.device()
    }
}

impl<P, W> Drop for PresentFuture<P, W>
where
    P: GpuFuture,
{
    fn drop(&mut self) {
        unsafe {
            if !*self.finished.get_mut() {
                match self.flush() {
                    Ok(()) => {
                        // Block until the queue finished.
                        self.queue().unwrap().wait().unwrap();
                        self.previous.signal_finished();
                    }
                    Err(_) => {
                        // In case of error we simply do nothing, as there's nothing to do
                        // anyway.
                    }
                }
            }
        }
    }
}

pub struct AcquiredImage {
    pub id: usize,
    pub suboptimal: bool,
}

/// Unsafe variant of `acquire_next_image`.
///
/// # Safety
///
/// - The semaphore and/or the fence must be kept alive until it is signaled.
/// - The swapchain must not have been replaced by being passed as the old swapchain when creating
///   a new one.
pub unsafe fn acquire_next_image_raw<W>(
    swapchain: &Swapchain<W>,
    timeout: Option<Duration>,
    semaphore: Option<&Semaphore>,
    fence: Option<&Fence>,
) -> Result<AcquiredImage, AcquireError> {
    let fns = swapchain.device.fns();

    let timeout_ns = if let Some(timeout) = timeout {
        timeout
            .as_secs()
            .saturating_mul(1_000_000_000)
            .saturating_add(timeout.subsec_nanos() as u64)
    } else {
        u64::MAX
    };

    let mut out = MaybeUninit::uninit();
    let r = check_errors(
        fns.khr_swapchain.acquire_next_image_khr(
            swapchain.device.internal_object(),
            swapchain.handle,
            timeout_ns,
            semaphore
                .map(|s| s.internal_object())
                .unwrap_or(ash::vk::Semaphore::null()),
            fence
                .map(|f| f.internal_object())
                .unwrap_or(ash::vk::Fence::null()),
            out.as_mut_ptr(),
        ),
    )?;

    let out = out.assume_init();
    let (id, suboptimal) = match r {
        Success::Success => (out as usize, false),
        Success::Suboptimal => (out as usize, true),
        Success::NotReady => return Err(AcquireError::Timeout),
        Success::Timeout => return Err(AcquireError::Timeout),
        s => panic!("unexpected success value: {:?}", s),
    };

    Ok(AcquiredImage { id, suboptimal })
}
