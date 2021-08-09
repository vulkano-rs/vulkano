// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::BufferAccess;
use crate::check_errors;
use crate::command_buffer::submit::SubmitAnyBuilder;
use crate::command_buffer::submit::SubmitPresentBuilder;
use crate::command_buffer::submit::SubmitPresentError;
use crate::command_buffer::submit::SubmitSemaphoresWaitBuilder;
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
use crate::swapchain::CapabilitiesError;
use crate::swapchain::ColorSpace;
use crate::swapchain::CompositeAlpha;
use crate::swapchain::PresentMode;
use crate::swapchain::PresentRegion;
use crate::swapchain::Surface;
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
use crate::sync::SharingMode;
use crate::Error;
use crate::OomError;
use crate::Success;
use crate::VulkanObject;
use std::error;
use std::fmt;
use std::mem;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

/// The way fullscreen exclusivity is handled.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum FullscreenExclusive {
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
    /// Indicates the application will manage full-screen exclusive mode by using
    /// `Swapchain::acquire_fullscreen_exclusive()` and
    /// `Swapchain::release_fullscreen_exclusive()` functions.
    AppControlled = ash::vk::FullScreenExclusiveEXT::APPLICATION_CONTROLLED.as_raw(),
}

impl From<FullscreenExclusive> for ash::vk::FullScreenExclusiveEXT {
    #[inline]
    fn from(val: FullscreenExclusive) -> Self {
        Self::from_raw(val as i32)
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
        let stale = swapchain.stale.lock().unwrap();
        if *stale {
            return Err(AcquireError::OutOfDate);
        }

        let acquire_result =
            unsafe { acquire_next_image_raw(&swapchain, timeout, Some(&semaphore), Some(&fence)) };

        if let &Err(AcquireError::FullscreenExclusiveLost) = &acquire_result {
            swapchain
                .fullscreen_exclusive_held
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

/// Contains the swapping system and the images that can be shown on a surface.
pub struct Swapchain<W> {
    // The Vulkan device this swapchain was created with.
    device: Arc<Device>,
    // The surface, which we need to keep alive.
    surface: Arc<Surface<W>>,
    // The swapchain object.
    swapchain: ash::vk::SwapchainKHR,

    // The images of this swapchain.
    images: Vec<ImageEntry>,

    // If true, that means we have tried to use this swapchain to recreate a new swapchain. The current
    // swapchain can no longer be used for anything except presenting already-acquired images.
    //
    // We use a `Mutex` instead of an `AtomicBool` because we want to keep that locked while
    // we acquire the image.
    stale: Mutex<bool>,

    // Parameters passed to the constructor.
    num_images: u32,
    format: Format,
    color_space: ColorSpace,
    dimensions: [u32; 2],
    layers: u32,
    usage: ImageUsage,
    sharing_mode: SharingMode,
    transform: SurfaceTransform,
    composite_alpha: CompositeAlpha,
    present_mode: PresentMode,
    fullscreen_exclusive: FullscreenExclusive,
    fullscreen_exclusive_held: AtomicBool,
    clipped: bool,
}

struct ImageEntry {
    // The actual image.
    image: UnsafeImage,
    // If true, then the image is still in the undefined layout and must be transitioned.
    undefined_layout: AtomicBool,
}

impl<W> Swapchain<W> {
    /// Starts the process of building a new swapchain, using default values for the parameters.
    #[inline]
    pub fn start(device: Arc<Device>, surface: Arc<Surface<W>>) -> SwapchainBuilder<W> {
        SwapchainBuilder {
            device,
            surface,

            num_images: 2,
            format: None,
            color_space: ColorSpace::SrgbNonLinear,
            dimensions: None,
            layers: 1,
            usage: ImageUsage::none(),
            sharing_mode: SharingMode::Exclusive,
            transform: Default::default(),
            composite_alpha: CompositeAlpha::Opaque,
            present_mode: PresentMode::Fifo,
            fullscreen_exclusive: FullscreenExclusive::Default,
            clipped: true,

            old_swapchain: None,
        }
    }

    /// Starts building a new swapchain from an existing swapchain.
    ///
    /// Use this when a swapchain has become invalidated, such as due to window resizes.
    /// The builder is pre-filled with the parameters of the old one, except for `dimensions`,
    /// which is set to `None`.
    #[inline]
    pub fn recreate(self: &Arc<Self>) -> SwapchainBuilder<W> {
        SwapchainBuilder {
            device: self.device().clone(),
            surface: self.surface().clone(),

            num_images: self.images.len() as u32,
            format: Some(self.format),
            color_space: self.color_space,
            dimensions: None,
            layers: self.layers,
            usage: self.usage,
            sharing_mode: self.sharing_mode.clone(),
            transform: self.transform,
            composite_alpha: self.composite_alpha,
            present_mode: self.present_mode,
            fullscreen_exclusive: self.fullscreen_exclusive,
            clipped: self.clipped,

            old_swapchain: Some(self.clone()),
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
            num_layers: self.layers as usize,
            first_mipmap_level: 0,
            num_mipmap_levels: 1,
        })
    }

    /// Returns the number of images of the swapchain.
    #[inline]
    pub fn num_images(&self) -> u32 {
        self.images.len() as u32
    }

    /// Returns the format of the images of the swapchain.
    #[inline]
    pub fn format(&self) -> Format {
        self.format
    }

    /// Returns the dimensions of the images of the swapchain.
    #[inline]
    pub fn dimensions(&self) -> [u32; 2] {
        self.dimensions
    }

    /// Returns the number of layers of the images of the swapchain.
    #[inline]
    pub fn layers(&self) -> u32 {
        self.layers
    }

    /// Returns the transform that was passed when creating the swapchain.
    #[inline]
    pub fn transform(&self) -> SurfaceTransform {
        self.transform
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

    /// Returns the value of 'fullscreen_exclusive` that was passed when creating the swapchain.
    #[inline]
    pub fn fullscreen_exclusive(&self) -> FullscreenExclusive {
        self.fullscreen_exclusive
    }

    /// `FullscreenExclusive::AppControlled` must be the active fullscreen exclusivity mode.
    /// Acquire fullscreen exclusivity until either the `release_fullscreen_exclusive` is
    /// called, or if any of the the other `Swapchain` functions return `FullscreenExclusiveLost`.
    /// Requires: `FullscreenExclusive::AppControlled`
    pub fn acquire_fullscreen_exclusive(&self) -> Result<(), FullscreenExclusiveError> {
        if self.fullscreen_exclusive != FullscreenExclusive::AppControlled {
            return Err(FullscreenExclusiveError::NotAppControlled);
        }

        if self.fullscreen_exclusive_held.swap(true, Ordering::SeqCst) {
            return Err(FullscreenExclusiveError::DoubleAcquire);
        }

        unsafe {
            check_errors(
                self.device
                    .fns()
                    .ext_full_screen_exclusive
                    .acquire_full_screen_exclusive_mode_ext(
                        self.device.internal_object(),
                        self.swapchain,
                    ),
            )?;
        }

        Ok(())
    }

    /// `FullscreenExclusive::AppControlled` must be the active fullscreen exclusivity mode.
    /// Release fullscreen exclusivity.
    pub fn release_fullscreen_exclusive(&self) -> Result<(), FullscreenExclusiveError> {
        if self.fullscreen_exclusive != FullscreenExclusive::AppControlled {
            return Err(FullscreenExclusiveError::NotAppControlled);
        }

        if !self.fullscreen_exclusive_held.swap(false, Ordering::SeqCst) {
            return Err(FullscreenExclusiveError::DoubleRelease);
        }

        unsafe {
            check_errors(
                self.device
                    .fns()
                    .ext_full_screen_exclusive
                    .release_full_screen_exclusive_mode_ext(
                        self.device.internal_object(),
                        self.swapchain,
                    ),
            )?;
        }

        Ok(())
    }

    /// `FullscreenExclusive::AppControlled` is not the active fullscreen exclusivity mode,
    /// then this function will always return false. If true is returned the swapchain
    /// is in `FullscreenExclusive::AppControlled` fullscreen exclusivity mode and exclusivity
    /// is currently acquired.
    pub fn is_fullscreen_exclusive(&self) -> bool {
        if self.fullscreen_exclusive != FullscreenExclusive::AppControlled {
            false
        } else {
            self.fullscreen_exclusive_held.load(Ordering::SeqCst)
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

unsafe impl<W> VulkanObject for Swapchain<W> {
    type Object = ash::vk::SwapchainKHR;

    #[inline]
    fn internal_object(&self) -> ash::vk::SwapchainKHR {
        self.swapchain
    }
}

unsafe impl<W> DeviceOwned for Swapchain<W> {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl<W> fmt::Debug for Swapchain<W> {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan swapchain {:?}>", self.swapchain)
    }
}

impl<W> Drop for Swapchain<W> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            fns.khr_swapchain.destroy_swapchain_khr(
                self.device.internal_object(),
                self.swapchain,
                ptr::null(),
            );
            self.surface.flag().store(false, Ordering::Release);
        }
    }
}

/// Builder for a [`Swapchain`].
#[derive(Debug)]
pub struct SwapchainBuilder<W> {
    device: Arc<Device>,
    surface: Arc<Surface<W>>,
    old_swapchain: Option<Arc<Swapchain<W>>>,

    num_images: u32,
    format: Option<Format>, // None = use a default
    color_space: ColorSpace,
    dimensions: Option<[u32; 2]>,
    layers: u32,
    usage: ImageUsage,
    sharing_mode: SharingMode,
    transform: SurfaceTransform,
    composite_alpha: CompositeAlpha,
    present_mode: PresentMode,
    fullscreen_exclusive: FullscreenExclusive,
    clipped: bool,
}

impl<W> SwapchainBuilder<W> {
    /// Builds a new swapchain. Allocates images who content can be made visible on a surface.
    ///
    /// See also the `Surface::get_capabilities` function which returns the values that are
    /// supported by the implementation. All the parameters that you pass to the builder
    /// must be supported.
    ///
    /// This function returns the swapchain plus a list of the images that belong to the
    /// swapchain. The order in which the images are returned is important for the
    /// `acquire_next_image` and `present` functions.
    ///
    /// # Panic
    ///
    /// - Panics if the device and the surface don't belong to the same instance.
    /// - Panics if `usage` is empty.
    ///
    // TODO: isn't it unsafe to take the surface through an Arc when it comes to vulkano-win?
    pub fn build(
        self,
    ) -> Result<(Arc<Swapchain<W>>, Vec<Arc<SwapchainImage<W>>>), SwapchainCreationError> {
        let SwapchainBuilder {
            device,
            surface,
            old_swapchain,

            num_images,
            format,
            color_space,
            dimensions,
            layers,
            usage,
            sharing_mode,
            transform,
            composite_alpha,
            present_mode,
            fullscreen_exclusive,
            clipped,
        } = self;

        assert_eq!(
            device.instance().internal_object(),
            surface.instance().internal_object()
        );

        // Checking that the requested parameters match the capabilities.
        let capabilities = surface.capabilities(device.physical_device())?;
        if num_images < capabilities.min_image_count {
            return Err(SwapchainCreationError::UnsupportedMinImagesCount);
        }
        if let Some(c) = capabilities.max_image_count {
            if num_images > c {
                return Err(SwapchainCreationError::UnsupportedMaxImagesCount);
            }
        }

        let format = {
            if let Some(format) = format {
                if !capabilities
                    .supported_formats
                    .iter()
                    .any(|&(f, c)| f == format && c == color_space)
                {
                    return Err(SwapchainCreationError::UnsupportedFormat);
                }
                format
            } else {
                if let Some(format) = [Format::R8G8B8A8Unorm, Format::B8G8R8A8Unorm]
                    .iter()
                    .copied()
                    .find(|&format| {
                        capabilities
                            .supported_formats
                            .iter()
                            .any(|&(f, c)| f == format && c == color_space)
                    })
                {
                    format
                } else {
                    return Err(SwapchainCreationError::UnsupportedFormat);
                }
            }
        };

        let dimensions = if let Some(dimensions) = dimensions {
            if dimensions[0] < capabilities.min_image_extent[0] {
                return Err(SwapchainCreationError::UnsupportedDimensions);
            }
            if dimensions[1] < capabilities.min_image_extent[1] {
                return Err(SwapchainCreationError::UnsupportedDimensions);
            }
            if dimensions[0] > capabilities.max_image_extent[0] {
                return Err(SwapchainCreationError::UnsupportedDimensions);
            }
            if dimensions[1] > capabilities.max_image_extent[1] {
                return Err(SwapchainCreationError::UnsupportedDimensions);
            }
            dimensions
        } else {
            capabilities.current_extent.unwrap()
        };
        if layers < 1 || layers > capabilities.max_image_array_layers {
            return Err(SwapchainCreationError::UnsupportedArrayLayers);
        }
        if (ash::vk::ImageUsageFlags::from(usage)
            & ash::vk::ImageUsageFlags::from(capabilities.supported_usage_flags))
            != ash::vk::ImageUsageFlags::from(usage)
        {
            return Err(SwapchainCreationError::UnsupportedUsageFlags);
        }
        if !capabilities.supported_transforms.supports(transform) {
            return Err(SwapchainCreationError::UnsupportedSurfaceTransform);
        }
        if !capabilities
            .supported_composite_alpha
            .supports(composite_alpha)
        {
            return Err(SwapchainCreationError::UnsupportedCompositeAlpha);
        }
        if !capabilities.present_modes.supports(present_mode) {
            return Err(SwapchainCreationError::UnsupportedPresentMode);
        }

        let flags = ImageCreateFlags::none();

        // check that the physical device supports the swapchain image configuration
        match device.image_format_properties(
            format,
            ImageType::Dim2d,
            ImageTiling::Optimal,
            usage,
            flags,
        ) {
            Ok(_) => (),
            Err(e) => {
                eprintln!("{}", e);
                return Err(SwapchainCreationError::UnsupportedImageConfiguration);
            }
        }

        // If we recreate a swapchain, make sure that the surface is the same.
        if let Some(ref sc) = old_swapchain {
            if surface.internal_object() != sc.surface.internal_object() {
                return Err(SwapchainCreationError::OldSwapchainSurfaceMismatch);
            }
        } else {
            // Checking that the surface doesn't already have a swapchain.
            let has_already = surface.flag().swap(true, Ordering::AcqRel);
            if has_already {
                return Err(SwapchainCreationError::SurfaceInUse);
            }
        }

        if !device.enabled_extensions().khr_swapchain {
            return Err(SwapchainCreationError::MissingExtensionKHRSwapchain);
        }

        let mut surface_full_screen_exclusive_info = None;

        // TODO: VK_EXT_FULL_SCREEN_EXCLUSIVE requires these extensions, so they should always
        // be enabled if it is. A separate check here is unnecessary; this should be checked at
        // device creation.
        if device.enabled_extensions().ext_full_screen_exclusive
            && surface
                .instance()
                .enabled_extensions()
                .khr_get_physical_device_properties2
            && surface
                .instance()
                .enabled_extensions()
                .khr_get_surface_capabilities2
        {
            surface_full_screen_exclusive_info = Some(ash::vk::SurfaceFullScreenExclusiveInfoEXT {
                full_screen_exclusive: fullscreen_exclusive.into(),
                ..Default::default()
            });
        }

        let p_next = match surface_full_screen_exclusive_info.as_ref() {
            Some(some) => unsafe { mem::transmute(some as *const _) },
            None => ptr::null(),
        };

        // Required by the specs.
        assert_ne!(usage, ImageUsage::none());

        if let Some(ref old_swapchain) = old_swapchain {
            let mut stale = old_swapchain.stale.lock().unwrap();

            // The swapchain has already been used to create a new one.
            if *stale {
                return Err(SwapchainCreationError::OldSwapchainAlreadyUsed);
            } else {
                // According to the documentation of VkSwapchainCreateInfoKHR:
                //
                // > Upon calling vkCreateSwapchainKHR with a oldSwapchain that is not VK_NULL_HANDLE,
                // > any images not acquired by the application may be freed by the implementation,
                // > which may occur even if creation of the new swapchain fails.
                //
                // Therefore, we set stale to true and keep it to true even if the call to `vkCreateSwapchainKHR` below fails.
                *stale = true;
            }
        }

        let fns = device.fns();

        let swapchain = unsafe {
            let (sh_mode, sh_count, sh_indices) = match sharing_mode {
                SharingMode::Exclusive => (ash::vk::SharingMode::EXCLUSIVE, 0, ptr::null()),
                SharingMode::Concurrent(ref ids) => (
                    ash::vk::SharingMode::CONCURRENT,
                    ids.len() as u32,
                    ids.as_ptr(),
                ),
            };

            let infos = ash::vk::SwapchainCreateInfoKHR {
                p_next,
                flags: ash::vk::SwapchainCreateFlagsKHR::empty(),
                surface: surface.internal_object(),
                min_image_count: num_images,
                image_format: format.into(),
                image_color_space: color_space.into(),
                image_extent: ash::vk::Extent2D {
                    width: dimensions[0],
                    height: dimensions[1],
                },
                image_array_layers: layers,
                image_usage: usage.into(),
                image_sharing_mode: sh_mode,
                queue_family_index_count: sh_count,
                p_queue_family_indices: sh_indices,
                pre_transform: transform.into(),
                composite_alpha: composite_alpha.into(),
                present_mode: present_mode.into(),
                clipped: if clipped {
                    ash::vk::TRUE
                } else {
                    ash::vk::FALSE
                },
                old_swapchain: if let Some(ref old_swapchain) = old_swapchain {
                    old_swapchain.swapchain
                } else {
                    ash::vk::SwapchainKHR::null()
                },
                ..Default::default()
            };

            let mut output = MaybeUninit::uninit();
            check_errors(fns.khr_swapchain.create_swapchain_khr(
                device.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        let image_handles = unsafe {
            let mut num = 0;
            check_errors(fns.khr_swapchain.get_swapchain_images_khr(
                device.internal_object(),
                swapchain,
                &mut num,
                ptr::null_mut(),
            ))?;

            let mut images = Vec::with_capacity(num as usize);
            check_errors(fns.khr_swapchain.get_swapchain_images_khr(
                device.internal_object(),
                swapchain,
                &mut num,
                images.as_mut_ptr(),
            ))?;
            images.set_len(num as usize);
            images
        };

        let images = image_handles
            .into_iter()
            .map(|image| unsafe {
                let dims = ImageDimensions::Dim2d {
                    width: dimensions[0],
                    height: dimensions[1],
                    array_layers: layers,
                };

                let img = UnsafeImage::from_raw(
                    device.clone(),
                    image,
                    usage,
                    format,
                    flags,
                    dims,
                    SampleCount::Sample1,
                    1,
                );

                ImageEntry {
                    image: img,
                    undefined_layout: AtomicBool::new(true),
                }
            })
            .collect::<Vec<_>>();

        let fullscreen_exclusive_held = old_swapchain
            .as_ref()
            .map(|old_swapchain| {
                if old_swapchain.fullscreen_exclusive != FullscreenExclusive::AppControlled {
                    false
                } else {
                    old_swapchain
                        .fullscreen_exclusive_held
                        .load(Ordering::SeqCst)
                }
            })
            .unwrap_or(false);

        let swapchain = Arc::new(Swapchain {
            device: device.clone(),
            surface: surface.clone(),
            swapchain,
            images,
            stale: Mutex::new(false),
            num_images,
            format,
            color_space,
            dimensions,
            layers,
            usage: usage.clone(),
            sharing_mode,
            transform,
            composite_alpha,
            present_mode,
            fullscreen_exclusive,
            fullscreen_exclusive_held: AtomicBool::new(fullscreen_exclusive_held),
            clipped,
        });

        let swapchain_images = unsafe {
            let mut swapchain_images = Vec::with_capacity(swapchain.images.len());
            for n in 0..swapchain.images.len() {
                swapchain_images.push(SwapchainImage::from_raw(swapchain.clone(), n)?);
            }
            swapchain_images
        };

        Ok((swapchain, swapchain_images))
    }

    /// Sets the number of images that will be created.
    ///
    /// The default is 2.
    #[inline]
    pub fn num_images(mut self, num_images: u32) -> Self {
        self.num_images = num_images;
        self
    }

    /// Sets the pixel format that will be used for the images.
    ///
    /// The default is either `R8G8B8A8Unorm` or `B8G8R8A8Unorm`, whichever is supported.
    #[inline]
    pub fn format(mut self, format: Format) -> Self {
        self.format = Some(format);
        self
    }

    /// Sets the color space that will be used for the images.
    ///
    /// The default is `SrgbNonLinear`.
    #[inline]
    pub fn color_space(mut self, color_space: ColorSpace) -> Self {
        self.color_space = color_space;
        self
    }

    /// Sets the dimensions of the images.
    ///
    /// The default is `None`, which means the value of
    /// [`Capabilities::current_extent`](crate::swapchain::Capabilities::current_extent) will be
    /// used. Setting this will override it with a custom `Some` value.
    #[inline]
    pub fn dimensions(mut self, dimensions: [u32; 2]) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Sets the number of layers for each image.
    ///
    /// The default is 1.
    #[inline]
    pub fn layers(mut self, layers: u32) -> Self {
        self.layers = layers;
        self
    }

    /// Sets how the images will be used.
    ///
    /// The default is `ImageUsage::none()`.
    #[inline]
    pub fn usage(mut self, usage: ImageUsage) -> Self {
        self.usage = usage;
        self
    }

    /// Sets the sharing mode of the images.
    ///
    /// The default is `Exclusive`.
    #[inline]
    pub fn sharing_mode<S>(mut self, sharing_mode: S) -> Self
    where
        S: Into<SharingMode>,
    {
        self.sharing_mode = sharing_mode.into();
        self
    }

    /// Sets the transform that is to be applied to the surface.
    ///
    /// The default is `Identity`.
    #[inline]
    pub fn transform(mut self, transform: SurfaceTransform) -> Self {
        self.transform = transform;
        self
    }

    /// Sets how alpha values of the pixels in the image are to be treated.
    ///
    /// The default is `Opaque`.
    #[inline]
    pub fn composite_alpha(mut self, composite_alpha: CompositeAlpha) -> Self {
        self.composite_alpha = composite_alpha;
        self
    }

    /// Sets the present mode for the swapchain.
    ///
    /// The default is `Fifo`.
    #[inline]
    pub fn present_mode(mut self, present_mode: PresentMode) -> Self {
        self.present_mode = present_mode;
        self
    }

    /// Sets how fullscreen exclusivity is to be handled.
    ///
    /// The default is `Default`.
    #[inline]
    pub fn fullscreen_exclusive(mut self, fullscreen_exclusive: FullscreenExclusive) -> Self {
        self.fullscreen_exclusive = fullscreen_exclusive;
        self
    }

    /// Sets whether the implementation is allowed to discard rendering operations that affect
    /// regions of the surface which aren't visible. This is important to take into account if
    /// your fragment shader has side-effects or if you want to read back the content of the image
    /// afterwards.
    ///
    /// The default is `true`.
    #[inline]
    pub fn clipped(mut self, clipped: bool) -> Self {
        self.clipped = clipped;
        self
    }
}

/// Error that can happen when creation a swapchain.
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
    /// The `VK_KHR_swapchain` extension was not enabled.
    MissingExtensionKHRSwapchain,
    /// The `VK_EXT_full_screen_exclusive` extension was not enabled.
    MissingExtensionExtFullScreenExclusive,
    /// Surface mismatch between old and new swapchain.
    OldSwapchainSurfaceMismatch,
    /// The old swapchain has already been used to recreate another one.
    OldSwapchainAlreadyUsed,
    /// The requested number of swapchain images is not supported by the surface.
    UnsupportedMinImagesCount,
    /// The requested number of swapchain images is not supported by the surface.
    UnsupportedMaxImagesCount,
    /// The requested image format is not supported by the surface.
    UnsupportedFormat,
    /// The requested dimensions are not supported by the surface.
    UnsupportedDimensions,
    /// The requested array layers count is not supported by the surface.
    UnsupportedArrayLayers,
    /// The requested image usage is not supported by the surface.
    UnsupportedUsageFlags,
    /// The requested surface transform is not supported by the surface.
    UnsupportedSurfaceTransform,
    /// The requested composite alpha is not supported by the surface.
    UnsupportedCompositeAlpha,
    /// The requested present mode is not supported by the surface.
    UnsupportedPresentMode,
    /// The image configuration is not supported by the physical device.
    UnsupportedImageConfiguration,
}

impl error::Error for SwapchainCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            SwapchainCreationError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for SwapchainCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                SwapchainCreationError::OomError(_) => "not enough memory available",
                SwapchainCreationError::DeviceLost => "the device was lost",
                SwapchainCreationError::SurfaceLost => "the surface was lost",
                SwapchainCreationError::SurfaceInUse => {
                    "the surface is already used by another swapchain"
                }
                SwapchainCreationError::NativeWindowInUse => {
                    "the window is already in use by another API"
                }
                SwapchainCreationError::MissingExtensionKHRSwapchain => {
                    "the `VK_KHR_swapchain` extension was not enabled"
                }
                SwapchainCreationError::MissingExtensionExtFullScreenExclusive => {
                    "the `VK_EXT_full_screen_exclusive` extension was not enabled"
                }
                SwapchainCreationError::OldSwapchainSurfaceMismatch => {
                    "surface mismatch between old and new swapchain"
                }
                SwapchainCreationError::OldSwapchainAlreadyUsed => {
                    "old swapchain has already been used to recreate a new one"
                }
                SwapchainCreationError::UnsupportedMinImagesCount => {
                    "the requested number of swapchain images is not supported by the surface"
                }
                SwapchainCreationError::UnsupportedMaxImagesCount => {
                    "the requested number of swapchain images is not supported by the surface"
                }
                SwapchainCreationError::UnsupportedFormat => {
                    "the requested image format is not supported by the surface"
                }
                SwapchainCreationError::UnsupportedDimensions => {
                    "the requested dimensions are not supported by the surface"
                }
                SwapchainCreationError::UnsupportedArrayLayers => {
                    "the requested array layers count is not supported by the surface"
                }
                SwapchainCreationError::UnsupportedUsageFlags => {
                    "the requested image usage is not supported by the surface"
                }
                SwapchainCreationError::UnsupportedSurfaceTransform => {
                    "the requested surface transform is not supported by the surface"
                }
                SwapchainCreationError::UnsupportedCompositeAlpha => {
                    "the requested composite alpha is not supported by the surface"
                }
                SwapchainCreationError::UnsupportedPresentMode => {
                    "the requested present mode is not supported by the surface"
                }
                SwapchainCreationError::UnsupportedImageConfiguration => {
                    "the requested image configuration is not supported by the physical device"
                }
            }
        )
    }
}

impl From<Error> for SwapchainCreationError {
    #[inline]
    fn from(err: Error) -> SwapchainCreationError {
        match err {
            err @ Error::OutOfHostMemory => SwapchainCreationError::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => SwapchainCreationError::OomError(OomError::from(err)),
            Error::DeviceLost => SwapchainCreationError::DeviceLost,
            Error::SurfaceLost => SwapchainCreationError::SurfaceLost,
            Error::NativeWindowInUse => SwapchainCreationError::NativeWindowInUse,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<OomError> for SwapchainCreationError {
    #[inline]
    fn from(err: OomError) -> SwapchainCreationError {
        SwapchainCreationError::OomError(err)
    }
}

impl From<CapabilitiesError> for SwapchainCreationError {
    #[inline]
    fn from(err: CapabilitiesError) -> SwapchainCreationError {
        match err {
            CapabilitiesError::OomError(err) => SwapchainCreationError::OomError(err),
            CapabilitiesError::SurfaceLost => SwapchainCreationError::SurfaceLost,
        }
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

unsafe impl<W> DeviceOwned for SwapchainAcquireFuture<W> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.swapchain.device
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

/// Error that can happen when calling `Swapchain::acquire_fullscreen_exclusive` or `Swapchain::release_fullscreen_exclusive`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum FullscreenExclusiveError {
    /// Not enough memory.
    OomError(OomError),

    /// Operation could not be completed for driver specific reasons.
    InitializationFailed,

    /// The surface is no longer accessible and must be recreated.
    SurfaceLost,

    /// Fullscreen exclusivity is already acquired.
    DoubleAcquire,

    /// Fullscreen exclusivity is not current acquired.
    DoubleRelease,

    /// Swapchain is not in fullscreen exclusive app controlled mode
    NotAppControlled,
}

impl From<Error> for FullscreenExclusiveError {
    #[inline]
    fn from(err: Error) -> FullscreenExclusiveError {
        match err {
            err @ Error::OutOfHostMemory => FullscreenExclusiveError::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => {
                FullscreenExclusiveError::OomError(OomError::from(err))
            }
            Error::SurfaceLost => FullscreenExclusiveError::SurfaceLost,
            Error::InitializationFailed => FullscreenExclusiveError::InitializationFailed,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl From<OomError> for FullscreenExclusiveError {
    #[inline]
    fn from(err: OomError) -> FullscreenExclusiveError {
        FullscreenExclusiveError::OomError(err)
    }
}

impl error::Error for FullscreenExclusiveError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            FullscreenExclusiveError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for FullscreenExclusiveError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                FullscreenExclusiveError::OomError(_) => "not enough memory",
                FullscreenExclusiveError::SurfaceLost => {
                    "the surface of this swapchain is no longer valid"
                }
                FullscreenExclusiveError::InitializationFailed => {
                    "operation could not be completed for driver specific reasons"
                }
                FullscreenExclusiveError::DoubleAcquire =>
                    "fullscreen exclusivity is already acquired",
                FullscreenExclusiveError::DoubleRelease => "fullscreen exclusivity is not acquired",
                FullscreenExclusiveError::NotAppControlled => {
                    "swapchain is not in fullscreen exclusive app controlled mode"
                }
            }
        )
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

    /// The swapchain has lost or doesn't have fullscreen exclusivity possibly for
    /// implementation-specific reasons outside of the application’s control.
    FullscreenExclusiveLost,

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
                AcquireError::FullscreenExclusiveLost => {
                    "the swapchain no longer has fullscreen exclusivity"
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
            Error::FullscreenExclusiveLost => AcquireError::FullscreenExclusiveLost,
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

            if let &Err(FlushError::FullscreenExclusiveLost) = &build_submission_result {
                self.swapchain
                    .fullscreen_exclusive_held
                    .store(false, Ordering::SeqCst);
            }

            match build_submission_result? {
                SubmitAnyBuilder::Empty => {}
                SubmitAnyBuilder::QueuePresent(present) => {
                    let present_result = present.submit(&self.queue);

                    if let &Err(SubmitPresentError::FullscreenExclusiveLost) = &present_result {
                        self.swapchain
                            .fullscreen_exclusive_held
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
            swapchain.swapchain,
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
