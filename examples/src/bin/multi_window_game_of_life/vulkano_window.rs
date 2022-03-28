// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::vulkano_context::{DeviceImageView, FinalImageView, VulkanoContext};
use std::{collections::HashMap, sync::Arc};
use vulkano::{
    device::{DeviceOwned, Queue},
    format::Format,
    image::{
        view::ImageView, ImageAccess, ImageCreateFlags, ImageDimensions, ImageUsage,
        ImageViewAbstract, StorageImage,
    },
    swapchain,
    swapchain::{
        AcquireError, PresentMode, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    },
    sync,
    sync::{FlushError, GpuFuture},
};
use vulkano_win::create_surface_from_winit;
use winit::window::Window;

pub struct VulkanoWindow {
    surface: Arc<Surface<Window>>,
    graphics_queue: Arc<Queue>,
    swapchain: Arc<Swapchain<Window>>,
    final_views: Vec<FinalImageView>,
    /// Image view that is to be rendered with our pipeline.
    /// (bool refers to whether it should get resized with swapchain resize)
    image_views: HashMap<usize, (DeviceImageView, bool)>,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    image_index: usize,
}

unsafe impl Sync for VulkanoWindow {}

unsafe impl Send for VulkanoWindow {}

impl VulkanoWindow {
    /// Creates new `VulkanoWindow` that is used to orchestrate rendering on the window's swapchain.
    /// Takes ownership of winit window
    pub fn new(
        vulkano_context: &VulkanoContext,
        window: winit::window::Window,
        vsync: bool,
    ) -> VulkanoWindow {
        // Create rendering surface from window
        let surface = create_surface_from_winit(window, vulkano_context.instance()).unwrap();
        // Create swap chain & frame(s) to which we'll render
        let (swapchain, final_views) = vulkano_context.create_swapchain(
            surface.clone(),
            if vsync {
                PresentMode::Fifo
            } else {
                PresentMode::Immediate
            },
        );

        let previous_frame_end = Some(sync::now(vulkano_context.device()).boxed());

        VulkanoWindow {
            surface,
            graphics_queue: vulkano_context.graphics_queue(),
            swapchain,
            final_views,
            image_views: HashMap::default(),
            recreate_swapchain: false,
            previous_frame_end,
            image_index: 0,
        }
    }

    /// Return swapchain image format
    pub fn swapchain_format(&self) -> Format {
        self.final_views[self.image_index].format().unwrap()
    }

    /// Return default image format for images  
    pub fn default_image_format(&self) -> Format {
        Format::R8G8B8A8_UNORM
    }

    /// Returns the index of last swapchain image that is the next render target
    pub fn image_index(&self) -> usize {
        self.image_index
    }

    /// Graphics queue of this window
    pub fn graphics_queue(&self) -> Arc<Queue> {
        self.graphics_queue.clone()
    }

    /// Render target surface
    pub fn surface(&self) -> Arc<Surface<Window>> {
        self.surface.clone()
    }

    /// Winit window (you can manipulate window through this)
    pub fn window(&self) -> &Window {
        self.surface.window()
    }

    pub fn window_size(&self) -> [u32; 2] {
        let size = self.window().inner_size();
        [size.width, size.height]
    }

    /// Size of the final swapchain image (surface)
    pub fn final_image_size(&self) -> [u32; 2] {
        self.final_views[0].image().dimensions().width_height()
    }

    /// Return final image which can be used as a render pipeline target
    pub fn final_image(&self) -> FinalImageView {
        self.final_views[self.image_index].clone()
    }

    /// Return scale factor accounted window size
    pub fn resolution(&self) -> [u32; 2] {
        let size = self.window().inner_size();
        let scale_factor = self.window().scale_factor();
        [
            (size.width as f64 / scale_factor) as u32,
            (size.height as f64 / scale_factor) as u32,
        ]
    }

    pub fn aspect_ratio(&self) -> f32 {
        let dims = self.window_size();
        dims[0] as f32 / dims[1] as f32
    }

    /// Resize swapchain and camera view images at the beginning of next frame
    pub fn resize(&mut self) {
        self.recreate_swapchain = true;
    }

    /// Add an image view that can be used to render e.g. camera views or other views using
    /// the render pipeline. Not giving a view size ensures the image view follows swapchain (window).
    pub fn add_image_target(&mut self, key: usize, view_size: Option<[u32; 2]>, format: Format) {
        let size = if let Some(s) = view_size {
            s
        } else {
            self.final_image_size()
        };
        let image = create_device_image(self.graphics_queue.clone(), size, format);
        self.image_views.insert(key, (image, view_size.is_none()));
    }

    /// Get interim image view by key
    pub fn get_image_target(&mut self, key: usize) -> DeviceImageView {
        self.image_views.get(&key).unwrap().clone().0
    }

    /// Get interim image view by key
    pub fn has_image_target(&mut self, key: usize) -> bool {
        self.image_views.get(&key).is_some()
    }

    pub fn remove_image_target(&mut self, key: usize) {
        self.image_views.remove(&key);
    }

    /*================
    RENDERING
    =================*/

    /// Acquires next swapchain image and increments image index
    /// This is the first to call in render orchestration.
    /// Returns a gpu future representing the time after which the swapchain image has been acquired
    /// and previous frame ended.
    /// After this, execute command buffers and return future from them to `finish_frame`.
    pub fn start_frame(&mut self) -> std::result::Result<Box<dyn GpuFuture>, AcquireError> {
        // Recreate swap chain if needed (when resizing of window occurs or swapchain is outdated)
        // Also resize render views if needed
        if self.recreate_swapchain {
            self.recreate_swapchain_and_views();
        }

        // Acquire next image in the swapchain
        let (image_num, suboptimal, acquire_future) =
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
        self.image_index = image_num;

        let future = self.previous_frame_end.take().unwrap().join(acquire_future);

        Ok(future.boxed())
    }

    /// Finishes render by presenting the swapchain
    pub fn finish_frame(&mut self, after_future: Box<dyn GpuFuture>) {
        let future = after_future
            .then_swapchain_present(
                self.graphics_queue.clone(),
                self.swapchain.clone(),
                self.image_index,
            )
            .then_signal_fence_and_flush();
        match future {
            Ok(future) => {
                // A hack to prevent OutOfMemory error on Nvidia :(
                // https://github.com/vulkano-rs/vulkano/issues/627
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

    /// Recreates swapchain images and image views that should follow swap chain image size
    fn recreate_swapchain_and_views(&mut self) {
        let (new_swapchain, new_images) = match self.swapchain.recreate(SwapchainCreateInfo {
            image_extent: self.window().inner_size().into(),
            ..self.swapchain.create_info()
        }) {
            Ok(r) => r,
            Err(e @ SwapchainCreationError::ImageExtentNotSupported { .. }) => {
                println!("{}", e);
                return;
            }
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
            .image_views
            .iter()
            .filter(|(_, (_img, follow_swapchain))| *follow_swapchain)
            .map(|c| *c.0)
            .collect::<Vec<usize>>();
        for i in resizable_views {
            let format = self.get_image_target(i).format().unwrap();
            self.remove_image_target(i);
            self.add_image_target(i, None, format);
        }
        self.recreate_swapchain = false;
    }
}

/// Creates a default storage image
pub fn create_device_image(queue: Arc<Queue>, size: [u32; 2], format: Format) -> DeviceImageView {
    let dims = ImageDimensions::Dim2d {
        width: size[0],
        height: size[1],
        array_layers: 1,
    };
    let flags = ImageCreateFlags::none();
    ImageView::new_default(
        StorageImage::with_usage(
            queue.device().clone(),
            dims,
            format,
            ImageUsage {
                sampled: true,
                storage: true,
                color_attachment: true,
                transfer_destination: true,
                ..ImageUsage::none()
            },
            flags,
            Some(queue.family()),
        )
        .unwrap(),
    )
    .unwrap()
}

/// Creates a device image with usage flags
pub fn create_device_image_with_usage(
    queue: Arc<Queue>,
    size: [u32; 2],
    format: Format,
    usage: ImageUsage,
) -> DeviceImageView {
    let dims = ImageDimensions::Dim2d {
        width: size[0],
        height: size[1],
        array_layers: 1,
    };
    let flags = ImageCreateFlags::none();
    ImageView::new_default(
        StorageImage::with_usage(
            queue.device().clone(),
            dims,
            format,
            usage,
            flags,
            Some(queue.family()),
        )
        .unwrap(),
    )
    .unwrap()
}
