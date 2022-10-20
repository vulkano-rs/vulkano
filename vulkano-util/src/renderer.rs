// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{context::VulkanoContext, window::WindowDescriptor};
use ahash::HashMap;
use std::sync::Arc;
use vulkano::{
    device::{Device, Queue},
    format::Format,
    image::{
        view::ImageView, ImageAccess, ImageUsage, ImageViewAbstract, StorageImage, SwapchainImage,
    },
    swapchain::{
        self, AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
        SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
};
use vulkano_win::create_surface_from_winit;
use winit::window::Window;

/// Swapchain Image View. Your final render target typically.
pub type SwapchainImageView = Arc<ImageView<SwapchainImage>>;
/// Multipurpose image view
pub type DeviceImageView = Arc<ImageView<StorageImage>>;

/// Most common image format
pub const DEFAULT_IMAGE_FORMAT: Format = Format::R8G8B8A8_UNORM;

/// A window renderer struct holding the winit window surface and functionality for organizing your
/// render between frames.
///
/// Begin rendering with [`VulkanoWindowRenderer::acquire`] and finish with
/// [`VulkanoWindowRenderer::present`]. Between those, you should execute your command buffers.
///
/// The intended usage of this struct is through [`crate::window::VulkanoWindows`].
pub struct VulkanoWindowRenderer {
    surface: Arc<Surface>,
    graphics_queue: Arc<Queue>,
    compute_queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    final_views: Vec<SwapchainImageView>,
    /// Additional image views that you can add which are resized with the window.
    /// Use associated functions to get access to these.
    additional_image_views: HashMap<usize, DeviceImageView>,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    image_index: u32,
    present_mode: vulkano::swapchain::PresentMode,
}

impl VulkanoWindowRenderer {
    /// Creates a new [`VulkanoWindowRenderer`] which is used to orchestrate your rendering with
    /// Vulkano. Pass [`WindowDescriptor`] and optionally a function modifying the
    /// [`SwapchainCreateInfo`](vulkano::swapchain::SwapchainCreateInfo) parameters.
    pub fn new(
        vulkano_context: &VulkanoContext,
        window: winit::window::Window,
        descriptor: &WindowDescriptor,
        swapchain_create_info_modify: fn(&mut SwapchainCreateInfo),
    ) -> VulkanoWindowRenderer {
        // Create rendering surface from window
        let surface =
            create_surface_from_winit(Arc::new(window), vulkano_context.instance().clone())
                .unwrap();

        // Create swap chain & frame(s) to which we'll render
        let (swap_chain, final_views) = Self::create_swapchain(
            vulkano_context.device().clone(),
            surface.clone(),
            descriptor,
            swapchain_create_info_modify,
        );

        let previous_frame_end = Some(sync::now(vulkano_context.device().clone()).boxed());

        VulkanoWindowRenderer {
            surface,
            graphics_queue: vulkano_context.graphics_queue().clone(),
            compute_queue: vulkano_context.compute_queue().clone(),
            swapchain: swap_chain,
            final_views,
            additional_image_views: HashMap::default(),
            recreate_swapchain: false,
            previous_frame_end,
            image_index: 0,
            present_mode: descriptor.present_mode,
        }
    }

    /// Creates the swapchain and its images based on [`WindowDescriptor`]. The swapchain creation
    /// can be modified with the `swapchain_create_info_modify` function passed as an input.
    fn create_swapchain(
        device: Arc<Device>,
        surface: Arc<Surface>,
        window_descriptor: &WindowDescriptor,
        swapchain_create_info_modify: fn(&mut SwapchainCreateInfo),
    ) -> (Arc<Swapchain>, Vec<SwapchainImageView>) {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let image_format = Some(
            device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );
        let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
        let image_extent = window.inner_size().into();
        let (swapchain, images) = Swapchain::new(device, surface, {
            let mut create_info = SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count,
                image_format,
                image_extent,
                image_usage: ImageUsage {
                    color_attachment: true,
                    ..ImageUsage::empty()
                },
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            };
            // Get present mode from window descriptor
            create_info.present_mode = window_descriptor.present_mode;
            swapchain_create_info_modify(&mut create_info);
            create_info
        })
        .unwrap();
        let images = images
            .into_iter()
            .map(|image| ImageView::new_default(image).unwrap())
            .collect::<Vec<_>>();
        (swapchain, images)
    }

    /// Set window renderer present mode. This triggers a swapchain recreation.
    #[inline]
    pub fn set_present_mode(&mut self, present_mode: vulkano::swapchain::PresentMode) {
        if self.present_mode != present_mode {
            self.present_mode = present_mode;
            self.recreate_swapchain = true;
        }
    }

    /// Return swapchain image format.
    #[inline]
    pub fn swapchain_format(&self) -> Format {
        self.final_views[self.image_index as usize]
            .format()
            .unwrap()
    }

    /// Returns the index of last swapchain image that is the next render target.
    #[inline]
    pub fn image_index(&self) -> u32 {
        self.image_index
    }

    /// Graphics queue of this window. You also can access this through [`VulkanoContext`].
    #[inline]
    pub fn graphics_queue(&self) -> Arc<Queue> {
        self.graphics_queue.clone()
    }

    /// Compute queue of this window. You can also access this through [`VulkanoContext`].
    #[inline]
    pub fn compute_queue(&self) -> Arc<Queue> {
        self.compute_queue.clone()
    }

    /// Render target surface.
    #[inline]
    pub fn surface(&self) -> Arc<Surface> {
        self.surface.clone()
    }

    /// Winit window (you can manipulate window through this).
    #[inline]
    pub fn window(&self) -> &Window {
        self.surface.object().unwrap().downcast_ref().unwrap()
    }

    /// Size of the physical window.
    #[inline]
    pub fn window_size(&self) -> [f32; 2] {
        let size = self.window().inner_size();
        [size.width as f32, size.height as f32]
    }

    /// Size of the final swapchain image (surface).
    #[inline]
    pub fn swapchain_image_size(&self) -> [u32; 2] {
        self.final_views[0].image().dimensions().width_height()
    }

    /// Return the current swapchain image view.
    #[inline]
    pub fn swapchain_image_view(&self) -> SwapchainImageView {
        self.final_views[self.image_index as usize].clone()
    }

    /// Return scale factor accounted window size.
    #[inline]
    pub fn resolution(&self) -> [f32; 2] {
        let size = self.window().inner_size();
        let scale_factor = self.window().scale_factor();
        [
            (size.width as f64 / scale_factor) as f32,
            (size.height as f64 / scale_factor) as f32,
        ]
    }

    #[inline]
    pub fn aspect_ratio(&self) -> f32 {
        let dims = self.window_size();
        dims[0] / dims[1]
    }

    /// Resize swapchain and camera view images at the beginning of next frame based on window
    /// dimensions.
    #[inline]
    pub fn resize(&mut self) {
        self.recreate_swapchain = true;
    }

    /// Add interim image view that resizes with window.
    #[inline]
    pub fn add_additional_image_view(&mut self, key: usize, format: Format, usage: ImageUsage) {
        let size = self.swapchain_image_size();
        let image = StorageImage::general_purpose_image_view(
            self.graphics_queue.clone(),
            size,
            format,
            usage,
        )
        .unwrap();
        self.additional_image_views.insert(key, image);
    }

    /// Get additional image view by key.
    #[inline]
    pub fn get_additional_image_view(&mut self, key: usize) -> DeviceImageView {
        self.additional_image_views.get(&key).unwrap().clone()
    }

    /// Remove additional image by key.
    #[inline]
    pub fn remove_additional_image_view(&mut self, key: usize) {
        self.additional_image_views.remove(&key);
    }

    /// Begin your rendering by calling `acquire`.
    /// Returns a [`GpuFuture`](vulkano::sync::GpuFuture) representing the time after which the
    /// swapchain image has been acquired and previous frame ended.
    /// Execute your command buffers after calling this function and finish rendering by calling
    /// [`VulkanoWindowRenderer::present`].
    #[inline]
    pub fn acquire(&mut self) -> std::result::Result<Box<dyn GpuFuture>, AcquireError> {
        // Recreate swap chain if needed (when resizing of window occurs or swapchain is outdated)
        // Also resize render views if needed
        if self.recreate_swapchain {
            self.recreate_swapchain_and_views();
        }

        // Acquire next image in the swapchain
        let (image_index, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    return Err(AcquireError::OutOfDate);
                }
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };
        if suboptimal {
            self.recreate_swapchain = true;
        }
        // Update our image index
        self.image_index = image_index;

        let future = self.previous_frame_end.take().unwrap().join(acquire_future);

        Ok(future.boxed())
    }

    /// Finishes rendering by presenting the swapchain. Pass your last future as an input to this
    /// function.
    ///
    /// Depending on your implementation, you may want to wait on your future. For example, a
    /// compute shader dispatch using an image that's being later drawn should probably be waited
    /// on.
    #[inline]
    pub fn present(&mut self, after_future: Box<dyn GpuFuture>, wait_future: bool) {
        let future = after_future
            .then_swapchain_present(
                self.graphics_queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.swapchain.clone(),
                    self.image_index,
                ),
            )
            .then_signal_fence_and_flush();
        match future {
            Ok(mut future) => {
                if wait_future {
                    match future.wait(None) {
                        Ok(x) => x,
                        Err(err) => println!("{:?}", err),
                    }
                    // wait allows you to organize resource waiting yourself.
                } else {
                    future.cleanup_finished();
                }

                self.previous_frame_end = Some(future.boxed());
            }
            Err(FlushError::OutOfDate) => {
                self.recreate_swapchain = true;
                self.previous_frame_end =
                    Some(sync::now(self.graphics_queue.device().clone()).boxed());
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                self.previous_frame_end =
                    Some(sync::now(self.graphics_queue.device().clone()).boxed());
            }
        }
    }

    /// Recreates swapchain images and image views which follow the window size.
    fn recreate_swapchain_and_views(&mut self) {
        let dimensions: [u32; 2] = self.window().inner_size().into();
        let (new_swapchain, new_images) = match self.swapchain.recreate(SwapchainCreateInfo {
            image_extent: dimensions,
            // Use present mode from current state
            present_mode: self.present_mode,
            ..self.swapchain.create_info()
        }) {
            Ok(r) => r,
            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
        };

        self.swapchain = new_swapchain;
        let new_images = new_images
            .into_iter()
            .map(|image| ImageView::new_default(image).unwrap())
            .collect::<Vec<_>>();
        self.final_views = new_images;
        // Resize images that follow swapchain size
        let resizable_views = self
            .additional_image_views
            .iter()
            .map(|c| *c.0)
            .collect::<Vec<usize>>();
        for i in resizable_views {
            let format = self.get_additional_image_view(i).format().unwrap();
            let usage = *self.get_additional_image_view(i).usage();
            self.remove_additional_image_view(i);
            self.add_additional_image_view(i, format, usage);
        }
        #[cfg(target_os = "ios")]
        unsafe {
            self.surface.update_ios_sublayer_on_resize();
        }
        self.recreate_swapchain = false;
    }
}
