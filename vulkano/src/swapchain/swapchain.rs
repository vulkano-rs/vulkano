// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{
    ColorSpace, CompositeAlpha, PresentMode, SupportedCompositeAlpha, SupportedSurfaceTransforms,
    Surface, SurfaceTransform, SwapchainPresentInfo,
};
use crate::{
    buffer::sys::Buffer,
    device::{Device, DeviceOwned, Queue},
    format::Format,
    image::{
        sys::Image, ImageFormatInfo, ImageLayout, ImageTiling, ImageType, ImageUsage,
        SwapchainImage,
    },
    macros::vulkan_enum,
    swapchain::{PresentInfo, SurfaceApi, SurfaceInfo, SurfaceSwapchainLock},
    sync::{
        AccessCheckError, AccessError, AccessFlags, Fence, FenceError, FlushError, GpuFuture,
        PipelineStages, Semaphore, SemaphoreError, Sharing, SubmitAnyBuilder,
    },
    DeviceSize, OomError, RequirementNotMet, RequiresOneOf, VulkanError, VulkanObject,
};
use parking_lot::Mutex;
use smallvec::{smallvec, SmallVec};
use std::{
    error::Error,
    fmt::{Debug, Display, Error as FmtError, Formatter},
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    num::NonZeroU64,
    ops::Range,
    ptr,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

/// Contains the swapping system and the images that can be shown on a surface.
pub struct Swapchain {
    handle: ash::vk::SwapchainKHR,
    device: Arc<Device>,
    surface: Arc<Surface>,

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
    prev_present_id: AtomicU64,

    // Whether full-screen exclusive is currently held.
    full_screen_exclusive_held: AtomicBool,

    // The images of this swapchain.
    images: Vec<ImageEntry>,

    // If true, that means we have tried to use this swapchain to recreate a new swapchain. The
    // current swapchain can no longer be used for anything except presenting already-acquired
    // images.
    //
    // We use a `Mutex` instead of an `AtomicBool` because we want to keep that locked while
    // we acquire the image.
    retired: Mutex<bool>,
}

#[derive(Debug)]
struct ImageEntry {
    handle: ash::vk::Image,
    layout_initialized: AtomicBool,
}

impl Swapchain {
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
        surface: Arc<Surface>,
        mut create_info: SwapchainCreateInfo,
    ) -> Result<(Arc<Swapchain>, Vec<Arc<SwapchainImage>>), SwapchainCreationError> {
        Self::validate(&device, &surface, &mut create_info)?;

        // Checking that the surface doesn't already have a swapchain.
        if surface.flag().swap(true, Ordering::AcqRel) {
            return Err(SwapchainCreationError::SurfaceInUse);
        }

        let (handle, image_handles) =
            unsafe { Self::create(&device, &surface, &create_info, None)? };

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
            prev_present_id: Default::default(),

            full_screen_exclusive_held: AtomicBool::new(false),
            images: image_handles
                .iter()
                .map(|&handle| ImageEntry {
                    handle,
                    layout_initialized: AtomicBool::new(false),
                })
                .collect(),
            retired: Mutex::new(false),
        });

        let swapchain_images = image_handles
            .into_iter()
            .enumerate()
            .map(|(image_index, handle)| unsafe {
                SwapchainImage::from_handle(handle, swapchain.clone(), image_index as u32)
            })
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
    pub fn recreate(
        self: &Arc<Self>,
        mut create_info: SwapchainCreateInfo,
    ) -> Result<(Arc<Swapchain>, Vec<Arc<SwapchainImage>>), SwapchainCreationError> {
        Self::validate(&self.device, &self.surface, &mut create_info)?;

        {
            let mut retired = self.retired.lock();

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

        let (handle, image_handles) =
            unsafe { Self::create(&self.device, &self.surface, &create_info, Some(self))? };

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
            prev_present_id: Default::default(),

            full_screen_exclusive_held: AtomicBool::new(full_screen_exclusive_held),
            images: image_handles
                .iter()
                .map(|&handle| ImageEntry {
                    handle,
                    layout_initialized: AtomicBool::new(false),
                })
                .collect(),
            retired: Mutex::new(false),
        });

        let swapchain_images = image_handles
            .into_iter()
            .enumerate()
            .map(|(image_index, handle)| unsafe {
                SwapchainImage::from_handle(handle, swapchain.clone(), image_index as u32)
            })
            .collect::<Result<_, _>>()?;

        Ok((swapchain, swapchain_images))
    }

    fn validate(
        device: &Device,
        surface: &Surface,
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
            clipped: _,
            full_screen_exclusive,
            win32_monitor,
            _ne: _,
        } = create_info;

        if !device.enabled_extensions().khr_swapchain {
            return Err(SwapchainCreationError::RequirementNotMet {
                required_for: "`Swapchain`",
                requires_one_of: RequiresOneOf {
                    device_extensions: &["khr_swapchain"],
                    ..Default::default()
                },
            });
        }

        assert_eq!(device.instance(), surface.instance());

        // VUID-VkSwapchainCreateInfoKHR-imageColorSpace-parameter
        image_color_space.validate_device(device)?;

        // VUID-VkSwapchainCreateInfoKHR-imageUsage-parameter
        image_usage.validate_device(device)?;

        // VUID-VkSwapchainCreateInfoKHR-imageUsage-requiredbitmask
        assert!(!image_usage.is_empty());

        // VUID-VkSwapchainCreateInfoKHR-preTransform-parameter
        pre_transform.validate_device(device)?;

        // VUID-VkSwapchainCreateInfoKHR-compositeAlpha-parameter
        composite_alpha.validate_device(device)?;

        // VUID-VkSwapchainCreateInfoKHR-presentMode-parameter
        present_mode.validate_device(device)?;

        if full_screen_exclusive != FullScreenExclusive::Default {
            if !device.enabled_extensions().ext_full_screen_exclusive {
                return Err(SwapchainCreationError::RequirementNotMet {
                    required_for:
                        "`create_info.full_screen_exclusive` is not `FullScreenExclusive::Default`",
                    requires_one_of: RequiresOneOf {
                        device_extensions: &["ext_full_screen_exclusive"],
                        ..Default::default()
                    },
                });
            }

            // VUID-VkSurfaceFullScreenExclusiveInfoEXT-fullScreenExclusive-parameter
            full_screen_exclusive.validate_device(device)?;
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
        if !device
            .active_queue_family_indices()
            .iter()
            .copied()
            .any(|index| unsafe {
                // Use unchecked, because all validation has been done above.
                device
                    .physical_device()
                    .surface_support_unchecked(index, surface)
                    .unwrap_or_default()
            })
        {
            return Err(SwapchainCreationError::SurfaceNotSupported);
        }

        *image_format = Some({
            // Use unchecked, because all validation has been done above.
            let surface_formats = unsafe {
                device.physical_device().surface_formats_unchecked(
                    surface,
                    SurfaceInfo {
                        full_screen_exclusive,
                        win32_monitor,
                        ..Default::default()
                    },
                )?
            };

            if let Some(format) = image_format {
                // VUID-VkSwapchainCreateInfoKHR-imageFormat-parameter
                format.validate_device(device)?;

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
                        .then_some(f)
                    })
                    .ok_or(SwapchainCreationError::FormatColorSpaceNotSupported)?
            }
        });

        // Use unchecked, because all validation has been done above.
        let surface_capabilities = unsafe {
            device.physical_device().surface_capabilities_unchecked(
                surface,
                SurfaceInfo {
                    full_screen_exclusive,
                    win32_monitor,
                    ..Default::default()
                },
            )?
        };

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
        // On some platforms, dimensions of zero-length can occur by minimizing the surface.
        if image_extent.contains(&0) {
            return Err(SwapchainCreationError::ImageExtentZeroLengthDimensions);
        }

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
            Sharing::Concurrent(queue_family_indices) => {
                // VUID-VkSwapchainCreateInfoKHR-imageSharingMode-01278
                // VUID-VkSwapchainCreateInfoKHR-imageSharingMode-01428
                queue_family_indices.sort_unstable();
                queue_family_indices.dedup();
                assert!(queue_family_indices.len() >= 2);

                for &queue_family_index in queue_family_indices.iter() {
                    // VUID-VkSwapchainCreateInfoKHR-imageSharingMode-01428
                    if queue_family_index
                        >= device.physical_device().queue_family_properties().len() as u32
                    {
                        return Err(
                            SwapchainCreationError::ImageSharingQueueFamilyIndexOutOfRange {
                                queue_family_index,
                                queue_family_count: device
                                    .physical_device()
                                    .queue_family_properties()
                                    .len()
                                    as u32,
                            },
                        );
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
        // Use unchecked, because all validation has been done above.
        if !unsafe {
            device
                .physical_device()
                .surface_present_modes_unchecked(surface)?
        }
        .any(|mode| mode == present_mode)
        {
            return Err(SwapchainCreationError::PresentModeNotSupported);
        }

        // VUID-VkSwapchainCreateInfoKHR-imageFormat-01778
        // Use unchecked, because all validation has been done above.
        if unsafe {
            device
                .physical_device()
                .image_format_properties_unchecked(ImageFormatInfo {
                    format: *image_format,
                    image_type: ImageType::Dim2d,
                    tiling: ImageTiling::Optimal,
                    usage: image_usage,
                    ..Default::default()
                })?
        }
        .is_none()
        {
            return Err(SwapchainCreationError::ImageFormatPropertiesNotSupported);
        }

        Ok(())
    }

    unsafe fn create(
        device: &Device,
        surface: &Surface,
        create_info: &SwapchainCreateInfo,
        old_swapchain: Option<&Swapchain>,
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

        let mut info_vk = ash::vk::SwapchainCreateInfoKHR {
            flags: ash::vk::SwapchainCreateFlagsKHR::empty(),
            surface: surface.handle(),
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
        let mut surface_full_screen_exclusive_info_vk = None;
        let mut surface_full_screen_exclusive_win32_info_vk = None;

        if full_screen_exclusive != FullScreenExclusive::Default {
            let next = surface_full_screen_exclusive_info_vk.insert(
                ash::vk::SurfaceFullScreenExclusiveInfoEXT {
                    full_screen_exclusive: full_screen_exclusive.into(),
                    ..Default::default()
                },
            );

            next.p_next = info_vk.p_next as *mut _;
            info_vk.p_next = next as *const _ as *const _;
        }

        if let Some(Win32Monitor(hmonitor)) = win32_monitor {
            let next = surface_full_screen_exclusive_win32_info_vk.insert(
                ash::vk::SurfaceFullScreenExclusiveWin32InfoEXT {
                    hmonitor,
                    ..Default::default()
                },
            );

            next.p_next = info_vk.p_next as *mut _;
            info_vk.p_next = next as *const _ as *const _;
        }

        let fns = device.fns();

        let handle = {
            let mut output = MaybeUninit::uninit();
            (fns.khr_swapchain.create_swapchain_khr)(
                device.handle(),
                &info_vk,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        let image_handles = loop {
            let mut count = 0;
            (fns.khr_swapchain.get_swapchain_images_khr)(
                device.handle(),
                handle,
                &mut count,
                ptr::null_mut(),
            )
            .result()
            .map_err(VulkanError::from)?;

            let mut images = Vec::with_capacity(count as usize);
            let result = (fns.khr_swapchain.get_swapchain_images_khr)(
                device.handle(),
                handle,
                &mut count,
                images.as_mut_ptr(),
            );

            match result {
                ash::vk::Result::SUCCESS => {
                    images.set_len(count as usize);
                    break images;
                }
                ash::vk::Result::INCOMPLETE => (),
                err => return Err(VulkanError::from(err).into()),
            }
        };

        Ok((handle, image_handles))
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
    pub fn surface(&self) -> &Arc<Surface> {
        &self.surface
    }

    /// If `image` is one of the images of this swapchain, returns its index within the swapchain.
    #[inline]
    pub fn index_of_image(&self, image: &Image) -> Option<u32> {
        self.images
            .iter()
            .position(|entry| entry.handle == image.handle())
            .map(|i| i as u32)
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

    /// Returns the usage of the images of the swapchain.
    #[inline]
    pub fn image_usage(&self) -> ImageUsage {
        self.image_usage
    }

    /// Returns the sharing of the images of the swapchain.
    #[inline]
    pub fn image_sharing(&self) -> &Sharing<SmallVec<[u32; 4]>> {
        &self.image_sharing
    }

    #[inline]
    pub(crate) unsafe fn full_screen_exclusive_held(&self) -> &AtomicBool {
        &self.full_screen_exclusive_held
    }

    #[inline]
    pub(crate) unsafe fn try_claim_present_id(&self, present_id: NonZeroU64) -> bool {
        let present_id = u64::from(present_id);
        self.prev_present_id.fetch_max(present_id, Ordering::SeqCst) < present_id
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
            let fns = self.device.fns();
            (fns.ext_full_screen_exclusive
                .acquire_full_screen_exclusive_mode_ext)(
                self.device.handle(), self.handle
            )
            .result()
            .map_err(VulkanError::from)?;
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
            let fns = self.device.fns();
            (fns.ext_full_screen_exclusive
                .release_full_screen_exclusive_mode_ext)(
                self.device.handle(), self.handle
            )
            .result()
            .map_err(VulkanError::from)?;
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
    pub(crate) fn image_layout_initialized(&self, image_index: u32) {
        let image_entry = self.images.get(image_index as usize);
        if let Some(image_entry) = image_entry {
            image_entry
                .layout_initialized
                .store(true, Ordering::Relaxed);
        }
    }

    pub(crate) fn is_image_layout_initialized(&self, image_index: u32) -> bool {
        let image_entry = self.images.get(image_index as usize);
        if let Some(image_entry) = image_entry {
            image_entry.layout_initialized.load(Ordering::Relaxed)
        } else {
            false
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            (fns.khr_swapchain.destroy_swapchain_khr)(
                self.device.handle(),
                self.handle,
                ptr::null(),
            );
            self.surface.flag().store(false, Ordering::Release);
        }
    }
}

unsafe impl VulkanObject for Swapchain {
    type Handle = ash::vk::SwapchainKHR;

    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for Swapchain {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl PartialEq for Swapchain {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.device() == other.device()
    }
}

impl Eq for Swapchain {}

impl Hash for Swapchain {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device().hash(state);
    }
}

impl Debug for Swapchain {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        let Self {
            handle,
            device,
            surface,
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
            prev_present_id,
            full_screen_exclusive_held,
            images,
            retired,
        } = self;

        f.debug_struct("Swapchain")
            .field("handle", &handle)
            .field("device", &device.handle())
            .field("surface", &surface.handle())
            .field("min_image_count", &min_image_count)
            .field("image_format", &image_format)
            .field("image_color_space", &image_color_space)
            .field("image_extent", &image_extent)
            .field("image_array_layers", &image_array_layers)
            .field("image_usage", &image_usage)
            .field("image_sharing", &image_sharing)
            .field("pre_transform", &pre_transform)
            .field("composite_alpha", &composite_alpha)
            .field("present_mode", &present_mode)
            .field("clipped", &clipped)
            .field("full_screen_exclusive", &full_screen_exclusive)
            .field("win32_monitor", &win32_monitor)
            .field("prev_present_id", &prev_present_id)
            .field("full_screen_exclusive_held", &full_screen_exclusive_held)
            .field("images", &images)
            .field("retired", &retired)
            .finish()
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
    /// The default value is [`ImageUsage::empty()`], which must be overridden.
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
            image_usage: ImageUsage::empty(),
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

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
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

    /// The provided `image_extent` contained at least one dimension of zero length.
    /// This is prohibited by [VUID-VkSwapchainCreateInfoKHR-imageExtent-01689](https://khronos.org/registry/vulkan/specs/1.3-extensions/man/html/VkSwapchainCreateInfoKHR.html#VUID-VkSwapchainCreateInfoKHR-imageExtent-01689)
    /// which requires both the width and height be non-zero.
    ///
    /// This error is distinct from `ImageExtentNotSupported` because a surface's minimum supported
    /// length may not enforce this rule.
    ImageExtentZeroLengthDimensions,

    /// The provided image parameters are not supported as queried from `image_format_properties`.
    ImageFormatPropertiesNotSupported,

    /// The provided `image_sharing` was set to `Concurrent`, but one of the specified queue family
    /// indices was out of range.
    ImageSharingQueueFamilyIndexOutOfRange {
        queue_family_index: u32,
        queue_family_count: u32,
    },

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

    /// The provided `surface` is not supported by any of the device's queue families.
    SurfaceNotSupported,

    /// The swapchain has already been used to create a new one.
    SwapchainAlreadyRetired,

    /// The `win32_monitor` value was `Some` when it must be `None` or vice-versa.
    Win32MonitorInvalid,
}

impl Error for SwapchainCreationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OomError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for SwapchainCreationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => write!(f, "not enough memory available"),
            Self::DeviceLost => write!(f, "the device was lost"),
            Self::SurfaceLost => write!(f, "the surface was lost"),
            Self::SurfaceInUse => {
                write!(f, "the surface is already used by another swapchain")
            }
            Self::NativeWindowInUse => {
                write!(f, "the window is already in use by another API")
            }
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
            Self::CompositeAlphaNotSupported { .. } => write!(
                f,
                "the provided `composite_alpha` is not supported by the surface for this device",
            ),
            Self::FormatColorSpaceNotSupported => write!(
                f,
                "the provided `format` and `color_space` are not supported by the surface for this \
                device",
            ),
            Self::ImageArrayLayersNotSupported {
                provided,
                max_supported,
            } => write!(
                f,
                "the provided `image_array_layers` ({}) is greater than what is supported ({}) by \
                the surface for this device",
                provided, max_supported,
            ),
            Self::ImageExtentNotSupported {
                provided,
                min_supported,
                max_supported,
            } => write!(
                f,
                "the provided `image_extent` ({:?}) is not within the range (min: {:?}, max: {:?}) \
                supported by the surface for this device",
                provided, min_supported, max_supported,
            ),
            Self::ImageExtentZeroLengthDimensions => write!(
                f,
                "the provided `image_extent` contained at least one dimension of zero length",
            ),
            Self::ImageFormatPropertiesNotSupported => write!(
                f,
                "the provided image parameters are not supported as queried from \
                `image_format_properties`",
            ),
            Self::ImageSharingQueueFamilyIndexOutOfRange {
                queue_family_index,
                queue_family_count: _,
            } => write!(
                f,
                "the provided `image_sharing` was set to `Concurrent`, but one of the specified \
                queue family indices ({}) was out of range",
                queue_family_index,
            ),
            Self::ImageUsageNotSupported { .. } => write!(
                f,
                "the provided `image_usage` has fields set that are not supported by the surface \
                for this device",
            ),
            Self::MinImageCountNotSupported {
                provided,
                min_supported,
                max_supported,
            } => write!(
                f,
                "the provided `min_image_count` ({}) is not within the range (min: {}, max: {:?}) \
                supported by the surface for this device",
                provided, min_supported, max_supported,
            ),
            Self::PresentModeNotSupported => write!(
                f,
                "the provided `present_mode` is not supported by the surface for this device",
            ),
            Self::PreTransformNotSupported { .. } => write!(
                f,
                "the provided `pre_transform` is not supported by the surface for this device",
            ),
            Self::SurfaceNotSupported => write!(
                f,
                "the provided `surface` is not supported by any of the device's queue families",
            ),
            Self::SwapchainAlreadyRetired => {
                write!(f, "the swapchain has already been used to create a new one")
            }
            Self::Win32MonitorInvalid => write!(
                f,
                "the `win32_monitor` value was `Some` when it must be `None` or vice-versa",
            ),
        }
    }
}

impl From<VulkanError> for SwapchainCreationError {
    fn from(err: VulkanError) -> SwapchainCreationError {
        match err {
            err @ VulkanError::OutOfHostMemory => Self::OomError(OomError::from(err)),
            err @ VulkanError::OutOfDeviceMemory => Self::OomError(OomError::from(err)),
            VulkanError::DeviceLost => Self::DeviceLost,
            VulkanError::SurfaceLost => Self::SurfaceLost,
            VulkanError::NativeWindowInUse => Self::NativeWindowInUse,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<OomError> for SwapchainCreationError {
    fn from(err: OomError) -> SwapchainCreationError {
        Self::OomError(err)
    }
}

impl From<RequirementNotMet> for SwapchainCreationError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

vulkan_enum! {
    /// The way full-screen exclusivity is handled.
    #[non_exhaustive]
    FullScreenExclusive = FullScreenExclusiveEXT(i32);

    /// Indicates that the driver should determine the appropriate full-screen method
    /// by whatever means it deems appropriate.
    Default = DEFAULT,

    /// Indicates that the driver may use full-screen exclusive mechanisms when available.
    /// Such mechanisms may result in better performance and/or the availability of
    /// different presentation capabilities, but may require a more disruptive transition
    // during swapchain initialization, first presentation and/or destruction.
    Allowed = ALLOWED,

    /// Indicates that the driver should avoid using full-screen mechanisms which rely
    /// on disruptive transitions.
    Disallowed = DISALLOWED,

    /// Indicates the application will manage full-screen exclusive mode by using the
    /// [`Swapchain::acquire_full_screen_exclusive()`] and
    /// [`Swapchain::release_full_screen_exclusive()`] functions.
    ApplicationControlled = APPLICATION_CONTROLLED,
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

impl Error for FullScreenExclusiveError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            FullScreenExclusiveError::OomError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for FullScreenExclusiveError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(
            f,
            "{}",
            match self {
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

impl From<VulkanError> for FullScreenExclusiveError {
    fn from(err: VulkanError) -> FullScreenExclusiveError {
        match err {
            err @ VulkanError::OutOfHostMemory => {
                FullScreenExclusiveError::OomError(OomError::from(err))
            }
            err @ VulkanError::OutOfDeviceMemory => {
                FullScreenExclusiveError::OomError(OomError::from(err))
            }
            VulkanError::SurfaceLost => FullScreenExclusiveError::SurfaceLost,
            VulkanError::InitializationFailed => FullScreenExclusiveError::InitializationFailed,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<OomError> for FullScreenExclusiveError {
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
pub fn acquire_next_image(
    swapchain: Arc<Swapchain>,
    timeout: Option<Duration>,
) -> Result<(u32, bool, SwapchainAcquireFuture), AcquireError> {
    let semaphore = Arc::new(Semaphore::from_pool(swapchain.device.clone())?);
    let fence = Fence::from_pool(swapchain.device.clone())?;

    let AcquiredImage {
        image_index,
        suboptimal,
    } = {
        // Check that this is not an old swapchain. From specs:
        // > swapchain must not have been replaced by being passed as the
        // > VkSwapchainCreateInfoKHR::oldSwapchain value to vkCreateSwapchainKHR
        let retired = swapchain.retired.lock();
        if *retired {
            return Err(AcquireError::OutOfDate);
        }

        let acquire_result =
            unsafe { acquire_next_image_raw(&swapchain, timeout, Some(&semaphore), Some(&fence)) };

        if let &Err(AcquireError::FullScreenExclusiveModeLost) = &acquire_result {
            swapchain
                .full_screen_exclusive_held
                .store(false, Ordering::SeqCst);
        }

        acquire_result?
    };

    Ok((
        image_index,
        suboptimal,
        SwapchainAcquireFuture {
            swapchain,
            semaphore: Some(semaphore),
            fence: Some(fence),
            image_index,
            finished: AtomicBool::new(false),
        },
    ))
}

/// Presents an image on the screen.
///
/// The actual behavior depends on the present mode that you passed when creating the swapchain.
pub fn present<F>(
    before: F,
    queue: Arc<Queue>,
    swapchain_info: SwapchainPresentInfo,
) -> PresentFuture<F>
where
    F: GpuFuture,
{
    assert!(swapchain_info.image_index < swapchain_info.swapchain.image_count());

    // TODO: restore this check with a dummy ImageAccess implementation
    /*let swapchain_image = me.images.lock().unwrap().get(index).unwrap().0.upgrade().unwrap();       // TODO: return error instead
    // Normally if `check_image_access` returns false we're supposed to call the `gpu_access`
    // function on the image instead. But since we know that this method on `SwapchainImage`
    // always returns false anyway (by design), we don't need to do it.
    assert!(before.check_image_access(&swapchain_image, ImageLayout::PresentSrc, true, &queue).is_ok());         // TODO: return error instead*/

    PresentFuture {
        previous: before,
        queue,
        swapchain_info,
        flushed: AtomicBool::new(false),
        finished: AtomicBool::new(false),
    }
}

/// Wait for an image to be presented to the user. Must be used with a `present_id` given to
/// `present_with_id`.
///
/// Returns a bool to represent if the presentation was suboptimal. In this case the swapchain is
/// still usable, but the swapchain should be recreated as the Surface's properties no longer match
/// the swapchain.
pub fn wait_for_present(
    swapchain: Arc<Swapchain>,
    present_id: u64,
    timeout: Option<Duration>,
) -> Result<bool, PresentWaitError> {
    let retired = swapchain.retired.lock();

    // VUID-vkWaitForPresentKHR-swapchain-04997
    if *retired {
        return Err(PresentWaitError::OutOfDate);
    }

    if present_id == 0 {
        return Err(PresentWaitError::PresentIdZero);
    }

    // VUID-vkWaitForPresentKHR-presentWait-06234
    if !swapchain.device.enabled_features().present_wait {
        return Err(PresentWaitError::RequirementNotMet {
            required_for: "`wait_for_present`",
            requires_one_of: RequiresOneOf {
                features: &["present_wait"],
                ..Default::default()
            },
        });
    }

    let timeout_ns = timeout.map(|dur| dur.as_nanos() as u64).unwrap_or(0);

    let result = unsafe {
        (swapchain.device.fns().khr_present_wait.wait_for_present_khr)(
            swapchain.device.handle(),
            swapchain.handle,
            present_id,
            timeout_ns,
        )
    };

    match result {
        ash::vk::Result::SUCCESS => Ok(false),
        ash::vk::Result::SUBOPTIMAL_KHR => Ok(true),
        ash::vk::Result::TIMEOUT => Err(PresentWaitError::Timeout),
        err => {
            let err = VulkanError::from(err).into();

            if let PresentWaitError::FullScreenExclusiveModeLost = &err {
                swapchain
                    .full_screen_exclusive_held
                    .store(false, Ordering::SeqCst);
            }

            Err(err)
        }
    }
}

/// Represents the moment when the GPU will have access to a swapchain image.
#[must_use]
pub struct SwapchainAcquireFuture {
    swapchain: Arc<Swapchain>,
    image_index: u32,
    // Semaphore that is signalled when the acquire is complete. Empty if the acquire has already
    // happened.
    semaphore: Option<Arc<Semaphore>>,
    // Fence that is signalled when the acquire is complete. Empty if the acquire has already
    // happened.
    fence: Option<Fence>,
    finished: AtomicBool,
}

impl SwapchainAcquireFuture {
    /// Returns the index of the image in the list of images returned when creating the swapchain.
    pub fn image_index(&self) -> u32 {
        self.image_index
    }

    /// Returns the corresponding swapchain.
    pub fn swapchain(&self) -> &Arc<Swapchain> {
        &self.swapchain
    }
}

unsafe impl GpuFuture for SwapchainAcquireFuture {
    fn cleanup_finished(&mut self) {}

    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError> {
        if let Some(ref semaphore) = self.semaphore {
            let sem = smallvec![semaphore.clone()];
            Ok(SubmitAnyBuilder::SemaphoresWait(sem))
        } else {
            Ok(SubmitAnyBuilder::Empty)
        }
    }

    fn flush(&self) -> Result<(), FlushError> {
        Ok(())
    }

    unsafe fn signal_finished(&self) {
        self.finished.store(true, Ordering::SeqCst);
    }

    fn queue_change_allowed(&self) -> bool {
        true
    }

    fn queue(&self) -> Option<Arc<Queue>> {
        None
    }

    fn check_buffer_access(
        &self,
        _buffer: &Buffer,
        _range: Range<DeviceSize>,
        _exclusive: bool,
        _queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        Err(AccessCheckError::Unknown)
    }

    fn check_image_access(
        &self,
        image: &Image,
        _range: Range<DeviceSize>,
        _exclusive: bool,
        expected_layout: ImageLayout,
        _queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        if self.swapchain.index_of_image(image) != Some(self.image_index) {
            return Err(AccessCheckError::Unknown);
        }

        if !self.swapchain.images[self.image_index as usize]
            .layout_initialized
            .load(Ordering::Relaxed)
            && expected_layout != ImageLayout::Undefined
        {
            return Err(AccessCheckError::Denied(AccessError::ImageNotInitialized {
                requested: expected_layout,
            }));
        }

        if expected_layout != ImageLayout::Undefined && expected_layout != ImageLayout::PresentSrc {
            return Err(AccessCheckError::Denied(
                AccessError::UnexpectedImageLayout {
                    allowed: ImageLayout::PresentSrc,
                    requested: expected_layout,
                },
            ));
        }

        Ok(None)
    }

    #[inline]
    fn check_swapchain_image_acquired(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        before: bool,
    ) -> Result<(), AccessCheckError> {
        if before {
            Ok(())
        } else {
            if swapchain == self.swapchain.as_ref() && image_index == self.image_index {
                Ok(())
            } else {
                Err(AccessCheckError::Unknown)
            }
        }
    }
}

impl Drop for SwapchainAcquireFuture {
    fn drop(&mut self) {
        if let Some(ref fence) = self.fence {
            fence.wait(None).unwrap(); // TODO: handle error?
            self.semaphore = None;
        }

        // TODO: if this future is destroyed without being presented, then eventually acquiring
        // a new image will block forever ; difficulty: hard
    }
}

unsafe impl DeviceOwned for SwapchainAcquireFuture {
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
    FullScreenExclusiveModeLost,

    /// The surface has changed in a way that makes the swapchain unusable. You must query the
    /// surface's new properties and recreate a new swapchain if you want to continue drawing.
    OutOfDate,

    /// Error during fence creation.
    FenceError(FenceError),

    /// Error during semaphore creation.
    SemaphoreError(SemaphoreError),
}

impl Error for AcquireError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            AcquireError::OomError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for AcquireError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(
            f,
            "{}",
            match self {
                AcquireError::OomError(_) => "not enough memory",
                AcquireError::DeviceLost => "the connection to the device has been lost",
                AcquireError::Timeout => "no image is available for acquiring yet",
                AcquireError::SurfaceLost => "the surface of this swapchain is no longer valid",
                AcquireError::OutOfDate => "the swapchain needs to be recreated",
                AcquireError::FullScreenExclusiveModeLost => {
                    "the swapchain no longer has full-screen exclusivity"
                }
                AcquireError::FenceError(_) => "error creating fence",
                AcquireError::SemaphoreError(_) => "error creating semaphore",
            }
        )
    }
}

impl From<FenceError> for AcquireError {
    fn from(err: FenceError) -> Self {
        AcquireError::FenceError(err)
    }
}

impl From<SemaphoreError> for AcquireError {
    fn from(err: SemaphoreError) -> Self {
        AcquireError::SemaphoreError(err)
    }
}

impl From<OomError> for AcquireError {
    fn from(err: OomError) -> AcquireError {
        AcquireError::OomError(err)
    }
}

impl From<VulkanError> for AcquireError {
    fn from(err: VulkanError) -> AcquireError {
        match err {
            err @ VulkanError::OutOfHostMemory => AcquireError::OomError(OomError::from(err)),
            err @ VulkanError::OutOfDeviceMemory => AcquireError::OomError(OomError::from(err)),
            VulkanError::DeviceLost => AcquireError::DeviceLost,
            VulkanError::SurfaceLost => AcquireError::SurfaceLost,
            VulkanError::OutOfDate => AcquireError::OutOfDate,
            VulkanError::FullScreenExclusiveModeLost => AcquireError::FullScreenExclusiveModeLost,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

/// Error that can happen when calling `acquire_next_image`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum PresentWaitError {
    /// Not enough memory.
    OomError(OomError),

    /// The connection to the device has been lost.
    DeviceLost,

    /// The surface has changed in a way that makes the swapchain unusable. You must query the
    /// surface's new properties and recreate a new swapchain if you want to continue drawing.
    OutOfDate,

    /// The surface is no longer accessible and must be recreated.
    SurfaceLost,

    /// The swapchain has lost or doesn't have full-screen exclusivity possibly for
    /// implementation-specific reasons outside of the application’s control.
    FullScreenExclusiveModeLost,

    /// The timeout of the function has been reached before the present occured.
    Timeout,

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// Present id of zero is invalid.
    PresentIdZero,
}

impl Error for PresentWaitError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::OomError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for PresentWaitError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::OomError(e) => write!(f, "{}", e),
            Self::DeviceLost => write!(f, "the connection to the device has been lost"),
            Self::Timeout => write!(f, "no image is available for acquiring yet"),
            Self::SurfaceLost => write!(f, "the surface of this swapchain is no longer valid"),
            Self::OutOfDate => write!(f, "the swapchain needs to be recreated"),
            Self::FullScreenExclusiveModeLost => {
                write!(f, "the swapchain no longer has full-screen exclusivity")
            }
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
            Self::PresentIdZero => write!(f, "present id of zero is invalid"),
        }
    }
}

impl From<OomError> for PresentWaitError {
    fn from(err: OomError) -> PresentWaitError {
        Self::OomError(err)
    }
}

impl From<RequirementNotMet> for PresentWaitError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

impl From<VulkanError> for PresentWaitError {
    fn from(err: VulkanError) -> PresentWaitError {
        match err {
            err @ VulkanError::OutOfHostMemory => Self::OomError(OomError::from(err)),
            err @ VulkanError::OutOfDeviceMemory => Self::OomError(OomError::from(err)),
            VulkanError::DeviceLost => Self::DeviceLost,
            VulkanError::SurfaceLost => Self::SurfaceLost,
            VulkanError::OutOfDate => Self::OutOfDate,
            VulkanError::FullScreenExclusiveModeLost => Self::FullScreenExclusiveModeLost,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

/// Represents a swapchain image being presented on the screen.
#[must_use = "Dropping this object will immediately block the thread until the GPU has finished processing the submission"]
pub struct PresentFuture<P>
where
    P: GpuFuture,
{
    previous: P,
    queue: Arc<Queue>,
    swapchain_info: SwapchainPresentInfo,
    // True if `flush()` has been called on the future, which means that the present command has
    // been submitted.
    flushed: AtomicBool,
    // True if `signal_finished()` has been called on the future, which means that the future has
    // been submitted and has already been processed by the GPU.
    finished: AtomicBool,
}

impl<P> PresentFuture<P>
where
    P: GpuFuture,
{
    /// Returns the index of the image in the list of images returned when creating the swapchain.
    pub fn image_id(&self) -> u32 {
        self.swapchain_info.image_index
    }

    /// Returns the corresponding swapchain.
    pub fn swapchain(&self) -> &Arc<Swapchain> {
        &self.swapchain_info.swapchain
    }
}

unsafe impl<P> GpuFuture for PresentFuture<P>
where
    P: GpuFuture,
{
    fn cleanup_finished(&mut self) {
        self.previous.cleanup_finished();
    }

    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, FlushError> {
        if self.flushed.load(Ordering::SeqCst) {
            return Ok(SubmitAnyBuilder::Empty);
        }

        let mut swapchain_info = self.swapchain_info.clone();
        debug_assert!((swapchain_info.image_index as u32) < swapchain_info.swapchain.image_count());
        let device = swapchain_info.swapchain.device();

        if !device.enabled_features().present_id {
            swapchain_info.present_id = None;
        }

        if device.enabled_extensions().khr_incremental_present {
            for rectangle in &swapchain_info.present_regions {
                assert!(rectangle.is_compatible_with(swapchain_info.swapchain.as_ref()));
            }
        } else {
            swapchain_info.present_regions = Default::default();
        }

        let _queue = self.previous.queue();

        // TODO: if the swapchain image layout is not PRESENT, should add a transition command
        // buffer

        Ok(match self.previous.build_submission()? {
            SubmitAnyBuilder::Empty => SubmitAnyBuilder::QueuePresent(PresentInfo {
                swapchain_infos: vec![self.swapchain_info.clone()],
                ..Default::default()
            }),
            SubmitAnyBuilder::SemaphoresWait(semaphores) => {
                SubmitAnyBuilder::QueuePresent(PresentInfo {
                    wait_semaphores: semaphores.into_iter().collect(),
                    swapchain_infos: vec![self.swapchain_info.clone()],
                    ..Default::default()
                })
            }
            SubmitAnyBuilder::CommandBuffer(_, _) => {
                // submit the command buffer by flushing previous.
                // Since the implementation should remember being flushed it's safe to call build_submission multiple times
                self.previous.flush()?;

                SubmitAnyBuilder::QueuePresent(PresentInfo {
                    swapchain_infos: vec![self.swapchain_info.clone()],
                    ..Default::default()
                })
            }
            SubmitAnyBuilder::BindSparse(_, _) => {
                // submit the command buffer by flushing previous.
                // Since the implementation should remember being flushed it's safe to call build_submission multiple times
                self.previous.flush()?;

                SubmitAnyBuilder::QueuePresent(PresentInfo {
                    swapchain_infos: vec![self.swapchain_info.clone()],
                    ..Default::default()
                })
            }
            SubmitAnyBuilder::QueuePresent(mut present_info) => {
                present_info
                    .swapchain_infos
                    .push(self.swapchain_info.clone());

                SubmitAnyBuilder::QueuePresent(present_info)
            }
        })
    }

    fn flush(&self) -> Result<(), FlushError> {
        unsafe {
            // If `flushed` already contains `true`, then `build_submission` will return `Empty`.

            let build_submission_result = self.build_submission();
            self.flushed.store(true, Ordering::SeqCst);

            match build_submission_result? {
                SubmitAnyBuilder::Empty => Ok(()),
                SubmitAnyBuilder::QueuePresent(present_info) => {
                    // VUID-VkPresentIdKHR-presentIds-04999
                    for swapchain_info in &present_info.swapchain_infos {
                        if swapchain_info.present_id.map_or(false, |present_id| {
                            !swapchain_info.swapchain.try_claim_present_id(present_id)
                        }) {
                            return Err(FlushError::PresentIdLessThanOrEqual);
                        }
                    }

                    match self.previous.check_swapchain_image_acquired(
                        &self.swapchain_info.swapchain,
                        self.swapchain_info.image_index,
                        true,
                    ) {
                        Ok(_) => (),
                        Err(AccessCheckError::Unknown) => {
                            return Err(AccessError::SwapchainImageNotAcquired.into())
                        }
                        Err(AccessCheckError::Denied(e)) => return Err(e.into()),
                    }

                    Ok(self
                        .queue
                        .with(|mut q| q.present_unchecked(present_info))?
                        .map(|r| r.map(|_| ()))
                        .fold(Ok(()), Result::and)?)
                }
                _ => unreachable!(),
            }
        }
    }

    unsafe fn signal_finished(&self) {
        self.flushed.store(true, Ordering::SeqCst);
        self.finished.store(true, Ordering::SeqCst);
        self.previous.signal_finished();
    }

    fn queue_change_allowed(&self) -> bool {
        false
    }

    fn queue(&self) -> Option<Arc<Queue>> {
        debug_assert!(match self.previous.queue() {
            None => true,
            Some(q) => q == self.queue,
        });

        Some(self.queue.clone())
    }

    fn check_buffer_access(
        &self,
        buffer: &Buffer,
        range: Range<DeviceSize>,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        self.previous
            .check_buffer_access(buffer, range, exclusive, queue)
    }

    fn check_image_access(
        &self,
        image: &Image,
        range: Range<DeviceSize>,
        exclusive: bool,
        expected_layout: ImageLayout,
        queue: &Queue,
    ) -> Result<Option<(PipelineStages, AccessFlags)>, AccessCheckError> {
        if self.swapchain_info.swapchain.index_of_image(image)
            == Some(self.swapchain_info.image_index)
        {
            // This future presents the swapchain image, which "unlocks" it. Therefore any attempt
            // to use this swapchain image afterwards shouldn't get granted automatic access.
            // Instead any attempt to access the image afterwards should get an authorization from
            // a later swapchain acquire future. Hence why we return `Unknown` here.
            Err(AccessCheckError::Unknown)
        } else {
            self.previous
                .check_image_access(image, range, exclusive, expected_layout, queue)
        }
    }

    #[inline]
    fn check_swapchain_image_acquired(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        before: bool,
    ) -> Result<(), AccessCheckError> {
        if before {
            self.previous
                .check_swapchain_image_acquired(swapchain, image_index, false)
        } else if swapchain == self.swapchain_info.swapchain.as_ref()
            && image_index == self.swapchain_info.image_index
        {
            Err(AccessError::SwapchainImageNotAcquired.into())
        } else {
            self.previous
                .check_swapchain_image_acquired(swapchain, image_index, false)
        }
    }
}

unsafe impl<P> DeviceOwned for PresentFuture<P>
where
    P: GpuFuture,
{
    fn device(&self) -> &Arc<Device> {
        self.queue.device()
    }
}

impl<P> Drop for PresentFuture<P>
where
    P: GpuFuture,
{
    fn drop(&mut self) {
        unsafe {
            if !*self.flushed.get_mut() {
                // Flushing may fail, that's okay. We will still wait for the queue later, so any
                // previous futures that were flushed correctly will still be waited upon.
                self.flush().ok();
            }

            if !*self.finished.get_mut() {
                // Block until the queue finished.
                self.queue().unwrap().with(|mut q| q.wait_idle()).unwrap();
                self.previous.signal_finished();
            }
        }
    }
}

pub struct AcquiredImage {
    pub image_index: u32,
    pub suboptimal: bool,
}

/// Unsafe variant of `acquire_next_image`.
///
/// # Safety
///
/// - The semaphore and/or the fence must be kept alive until it is signaled.
/// - The swapchain must not have been replaced by being passed as the old swapchain when creating
///   a new one.
pub unsafe fn acquire_next_image_raw(
    swapchain: &Swapchain,
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
    let result = (fns.khr_swapchain.acquire_next_image_khr)(
        swapchain.device.handle(),
        swapchain.handle,
        timeout_ns,
        semaphore
            .map(|s| s.handle())
            .unwrap_or(ash::vk::Semaphore::null()),
        fence.map(|f| f.handle()).unwrap_or(ash::vk::Fence::null()),
        out.as_mut_ptr(),
    );

    let suboptimal = match result {
        ash::vk::Result::SUCCESS => false,
        ash::vk::Result::SUBOPTIMAL_KHR => true,
        ash::vk::Result::NOT_READY => return Err(AcquireError::Timeout),
        ash::vk::Result::TIMEOUT => return Err(AcquireError::Timeout),
        err => return Err(VulkanError::from(err).into()),
    };

    if let Some(semaphore) = semaphore {
        let mut state = semaphore.state();
        state.swapchain_acquire();
    }

    if let Some(fence) = fence {
        let mut state = fence.state();
        state.import_swapchain_acquire();
    }

    Ok(AcquiredImage {
        image_index: out.assume_init(),
        suboptimal,
    })
}
