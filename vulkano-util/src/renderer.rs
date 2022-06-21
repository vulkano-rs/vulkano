// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::collections::HashMap;
use std::sync::Arc;

use crate::context::VulkanoContext;
use crate::window::WindowDescriptor;
use vulkano::device::Device;
use vulkano::image::{ImageUsage, StorageImage, SwapchainImage};
use vulkano::{
    device::Queue,
    format::Format,
    image::{view::ImageView, ImageAccess, ImageViewAbstract},
    swapchain,
    swapchain::{AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError},
    sync,
    sync::{FlushError, GpuFuture},
};
use vulkano_win::create_surface_from_winit;
use winit::window::Window;

/// Final render target to which you draw and then display on the window
pub type SwapchainImageView = Arc<ImageView<SwapchainImage<Window>>>;
/// Multipurpose image view
pub type DeviceImageView = Arc<ImageView<StorageImage>>;

/// Most common image format
pub const DEFAULT_IMAGE_FORMAT: Format = Format::R8G8B8A8_UNORM;

/// A window renderer struct holding the winit window surface and functionality for organizing your render
/// between frames.
///
/// Begin rendering with [`VulkanoWindowRenderer::start_frame`] and finish with [`VulkanoWindowRenderer::finish_frame`].
/// Between those, you should execute your command buffers.
///
/// The intended usage of this struct is through [`crate::window::VulkanoWindows`].
pub struct VulkanoWindowRenderer {
    surface: Arc<Surface<Window>>,
    graphics_queue: Arc<Queue>,
    compute_queue: Arc<Queue>,
    swap_chain: Arc<Swapchain<Window>>,
    final_views: Vec<SwapchainImageView>,
    /// Additional image views that you can add which are resized with the window.
    /// Use associated functions to get access to these.
    additional_image_views: HashMap<usize, DeviceImageView>,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    image_index: usize,
}

unsafe impl Sync for VulkanoWindowRenderer {}

unsafe impl Send for VulkanoWindowRenderer {}

impl VulkanoWindowRenderer {
    /// Creates a new [`VulkanoWindowRenderer`] which is used to orchestrate your rendering with Vulkano.
    /// Pass [`WindowDescriptor`] and optionally a function modifying the [`SwapchainCreateInfo`](vulkano::swapchain::SwapchainCreateInfo) parameters.
    pub(crate) fn new(
        vulkano_context: &VulkanoContext,
        window: winit::window::Window,
        descriptor: &WindowDescriptor,
        swapchain_create_info_modify: fn(&mut SwapchainCreateInfo),
    ) -> VulkanoWindowRenderer {
        // Create rendering surface from window
        let surface = create_surface_from_winit(window, vulkano_context.instance()).unwrap();

        // Create swap chain & frame(s) to which we'll render
        let (swap_chain, final_views) = Self::create_swap_chain(
            vulkano_context.device(),
            surface.clone(),
            descriptor,
            swapchain_create_info_modify,
        );

        let previous_frame_end = Some(sync::now(vulkano_context.device()).boxed());

        VulkanoWindowRenderer {
            surface,
            graphics_queue: vulkano_context.graphics_queue(),
            compute_queue: vulkano_context.compute_queue(),
            swap_chain,
            final_views,
            additional_image_views: HashMap::default(),
            recreate_swapchain: false,
            previous_frame_end,
            image_index: 0,
        }
    }

    /// Creates the swapchain and its images based on [`WindowDescriptor`]. The swapchain creation
    /// can be modified with the `swapchain_create_info_modify` function passed as an input.
    fn create_swap_chain(
        device: Arc<Device>,
        surface: Arc<Surface<Window>>,
        window_descriptor: &WindowDescriptor,
        swapchain_create_info_modify: fn(&mut SwapchainCreateInfo),
    ) -> (Arc<Swapchain<Window>>, Vec<SwapchainImageView>) {
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
        let image_extent = surface.window().inner_size().into();
        let (swapchain, images) = Swapchain::new(device, surface, {
            let mut create_info = SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count,
                image_format,
                image_extent,
                image_usage: ImageUsage::color_attachment(),
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

    /// Return swapchain image format
    pub fn swapchain_format(&self) -> Format {
        self.final_views[self.image_index].format().unwrap()
    }

    /// Returns the index of last swapchain image that is the next render target
    pub fn image_index(&self) -> usize {
        self.image_index
    }

    /// Graphics queue of this window. You also can access this through [`VulkanoContext`]
    pub fn graphics_queue(&self) -> Arc<Queue> {
        self.graphics_queue.clone()
    }

    /// Compute queue of this window. You can also access this through [`VulkanoContext`]
    pub fn compute_queue(&self) -> Arc<Queue> {
        self.compute_queue.clone()
    }

    /// Render target surface
    pub fn surface(&self) -> Arc<Surface<Window>> {
        self.surface.clone()
    }

    /// Winit window (you can manipulate window through this).
    pub fn window(&self) -> &Window {
        self.surface.window()
    }

    /// Size of the physical window
    pub fn window_size(&self) -> [f32; 2] {
        let size = self.window().inner_size();
        [size.width as f32, size.height as f32]
    }

    /// Size of the final swapchain image (surface)
    pub fn swapchain_image_size(&self) -> [u32; 2] {
        self.final_views[0].image().dimensions().width_height()
    }

    /// Return the current swapchain image view
    pub fn swapchain_image_view(&self) -> SwapchainImageView {
        self.final_views[self.image_index].clone()
    }

    /// Return scale factor accounted window size
    pub fn resolution(&self) -> [f32; 2] {
        let size = self.window().inner_size();
        let scale_factor = self.window().scale_factor();
        [
            (size.width as f64 / scale_factor) as f32,
            (size.height as f64 / scale_factor) as f32,
        ]
    }

    pub fn aspect_ratio(&self) -> f32 {
        let dims = self.window_size();
        dims[0] / dims[1]
    }

    /// Resize swapchain and camera view images at the beginning of next frame
    pub fn resize(&mut self) {
        self.recreate_swapchain = true;
    }

    /// Add interim image view that resizes with window
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

    /// Get additional image view by key
    pub fn get_additional_image_view(&mut self, key: usize) -> DeviceImageView {
        self.additional_image_views.get(&key).unwrap().clone()
    }

    /// Remove additional image by key
    pub fn remove_additional_image_view(&mut self, key: usize) {
        self.additional_image_views.remove(&key);
    }

    /// Begin your rendering by calling `start_frame`.
    /// Returns a [`GpuFuture`](vulkano::sync::future::GpuFuture) representing the time after which the swapchain image has been acquired
    /// and previous frame ended.
    /// Execute your command buffers after calling this function and finish rendering by calling [`VulkanoWindowRenderer::finish_frame`].
    pub fn start_frame(&mut self) -> std::result::Result<Box<dyn GpuFuture>, AcquireError> {
        // Recreate swap chain if needed (when resizing of window occurs or swapchain is outdated)
        // Also resize render views if needed
        if self.recreate_swapchain {
            self.recreate_swapchain_and_views();
        }

        // Acquire next image in the swapchain
        let (image_num, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.swap_chain.clone(), None) {
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
        self.image_index = image_num;

        let future = self.previous_frame_end.take().unwrap().join(acquire_future);

        Ok(future.boxed())
    }

    /// Finishes rendering by presenting the swapchain. Pass your last future as an input to this function.
    pub fn finish_frame(&mut self, after_future: Box<dyn GpuFuture>) {
        let future = after_future
            .then_swapchain_present(
                self.graphics_queue.clone(),
                self.swap_chain.clone(),
                self.image_index,
            )
            .then_signal_fence_and_flush();
        match future {
            Ok(future) => {
                // Prevent OutOfMemory error on Nvidia :(
                // https://github.com/vulkano-rs/vulkano/issues/627.
                // Maybe there's some way to prevent this with synchronization...
                match future.wait(None) {
                    Ok(x) => x,
                    Err(err) => println!("{:?}", err),
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

    /// Recreates swapchain images and image views which follow the window size
    fn recreate_swapchain_and_views(&mut self) {
        let dimensions: [u32; 2] = self.window().inner_size().into();
        let (new_swapchain, new_images) = match self.swap_chain.recreate(SwapchainCreateInfo {
            image_extent: dimensions,
            ..self.swap_chain.create_info()
        }) {
            Ok(r) => r,
            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
        };

        self.swap_chain = new_swapchain;
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
            let usage = self.get_additional_image_view(i).usage().clone();
            self.remove_additional_image_view(i);
            self.add_additional_image_view(i, format, usage);
        }
        self.recreate_swapchain = false;
    }
}
