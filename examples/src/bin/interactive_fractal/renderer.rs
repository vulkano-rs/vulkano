// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::place_over_frame::RenderPassPlaceOverFrame;

use vulkano_win::VkSurfaceBuild;

use std::collections::HashMap;
use std::sync::Arc;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{
    AttachmentImage, ImageAccess, ImageUsage, ImageViewAbstract, SampleCount, SwapchainImage,
};
use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::swapchain::{
    AcquireError, ColorSpace, FullscreenExclusive, PresentMode, Surface, SurfaceTransform,
    Swapchain, SwapchainCreationError,
};
use vulkano::sync::{FlushError, GpuFuture};
use vulkano::{swapchain, sync, Version};
use winit::event_loop::EventLoop;
use winit::window::{Fullscreen, Window, WindowBuilder};

/// Final render target (swapchain image)
pub type FinalImageView = Arc<ImageView<SwapchainImage<Window>>>;
/// Other intermediate render targets
pub type InterimImageView = Arc<ImageView<AttachmentImage>>;

/// A simple struct to organize renderpasses.
/// You could add more here. E.g. the `frame_system`
/// from the deferred examples...
pub struct RenderPasses {
    pub place_over_frame: RenderPassPlaceOverFrame,
}

#[derive(Debug, Copy, Clone)]
pub struct RenderOptions {
    pub title: &'static str,
    pub window_size: [u32; 2],
    pub v_sync: bool,
}

impl Default for RenderOptions {
    fn default() -> Self {
        RenderOptions {
            title: "App",
            window_size: [1920, 1080],
            v_sync: false,
        }
    }
}

pub struct Renderer {
    _instance: Arc<Instance>,
    device: Arc<Device>,
    surface: Arc<Surface<Window>>,
    queue: Arc<Queue>,
    swap_chain: Arc<Swapchain<Window>>,
    image_index: usize,
    final_views: Vec<FinalImageView>,
    /// Image view that is to be rendered with our pipeline.
    /// (bool refers to whether it should get resized with swapchain resize)
    interim_image_views: HashMap<usize, (InterimImageView, bool)>,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    render_passes: RenderPasses,
    is_fullscreen: bool,
}

impl Renderer {
    /// Creates a new GPU renderer for window with given parameters
    pub fn new(event_loop: &EventLoop<()>, opts: RenderOptions) -> Self {
        println!("Creating renderer for window size {:?}", opts.window_size);
        // Add instance extensions based on needs
        let instance_extensions = InstanceExtensions {
            ..vulkano_win::required_extensions()
        };
        // Create instance
        let _instance = Instance::new(None, Version::V1_2, &instance_extensions, None)
            .expect("Failed to create instance");

        // Get desired device
        let physical_device = PhysicalDevice::enumerate(&_instance)
            .min_by_key(|p| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
            })
            .unwrap();
        println!("Using device {}", physical_device.properties().device_name);

        // Create rendering surface along with window
        let surface = WindowBuilder::new()
            .with_inner_size(winit::dpi::LogicalSize::new(
                opts.window_size[0],
                opts.window_size[1],
            ))
            .with_title(opts.title)
            .build_vk_surface(event_loop, _instance.clone())
            .unwrap();
        println!("Window scale factor {}", surface.window().scale_factor());

        // Create device
        let (device, queue) = Self::create_device(physical_device, surface.clone());
        // Create swap chain & frame(s) to which we'll render
        let (swap_chain, final_images) = Self::create_swap_chain(
            surface.clone(),
            physical_device,
            device.clone(),
            queue.clone(),
            if opts.v_sync {
                PresentMode::Fifo
            } else {
                PresentMode::Immediate
            },
        );
        let previous_frame_end = Some(sync::now(device.clone()).boxed());
        let is_fullscreen = swap_chain.surface().window().fullscreen().is_some();
        let image_format = final_images.first().unwrap().format();
        let render_passes = RenderPasses {
            place_over_frame: RenderPassPlaceOverFrame::new(queue.clone(), image_format),
        };

        Renderer {
            _instance,
            device,
            surface,
            queue,
            swap_chain,
            image_index: 0,
            final_views: final_images,
            interim_image_views: HashMap::new(),
            previous_frame_end,
            recreate_swapchain: false,
            render_passes,
            is_fullscreen,
        }
    }

    /// Creates vulkan device with required queue families and required extensions
    fn create_device(
        physical: PhysicalDevice,
        surface: Arc<Surface<Window>>,
    ) -> (Arc<Device>, Arc<Queue>) {
        let queue_family = physical
            .queue_families()
            .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
            .unwrap();

        // Add device extensions based on needs,
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };

        // Add device features
        let features = Features {
            fill_mode_non_solid: true,
            ..Features::none()
        };
        let (device, mut queues) = {
            Device::new(
                physical,
                &features,
                &physical.required_extensions().union(&device_extensions),
                [(queue_family, 0.5)].iter().cloned(),
            )
            .unwrap()
        };
        (device, queues.next().unwrap())
    }

    /// Creates swapchain and swapchain images
    fn create_swap_chain(
        surface: Arc<Surface<Window>>,
        physical: PhysicalDevice,
        device: Arc<Device>,
        queue: Arc<Queue>,
        present_mode: PresentMode,
    ) -> (Arc<Swapchain<Window>>, Vec<FinalImageView>) {
        let caps = surface.capabilities(physical).unwrap();
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();
        let (swap_chain, images) = Swapchain::start(device, surface)
            .num_images(caps.min_image_count)
            .format(format)
            .dimensions(dimensions)
            .usage(ImageUsage::color_attachment())
            .sharing_mode(&queue)
            .composite_alpha(alpha)
            .transform(SurfaceTransform::Identity)
            .present_mode(present_mode)
            .fullscreen_exclusive(FullscreenExclusive::Default)
            .clipped(true)
            .color_space(ColorSpace::SrgbNonLinear)
            .layers(1)
            .build()
            .unwrap();
        let images = images
            .into_iter()
            .map(|image| ImageView::new(image).unwrap())
            .collect::<Vec<_>>();
        (swap_chain, images)
    }

    /// Return default image format for images (swapchain format may differ)
    pub fn image_format(&self) -> Format {
        Format::R8G8B8A8_UNORM
    }

    /// Return swapchain image format
    #[allow(unused)]
    pub fn swapchain_format(&self) -> Format {
        self.final_views[self.image_index].format()
    }

    /// Returns the index of last swapchain image that is the next render target
    /// All camera views will render onto their image at the same index
    #[allow(unused)]
    pub fn image_index(&self) -> usize {
        self.image_index
    }

    /// Access device
    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }

    /// Access rendering queue
    pub fn queue(&self) -> Arc<Queue> {
        self.queue.clone()
    }

    /// Render target surface
    #[allow(unused)]
    pub fn surface(&self) -> Arc<Surface<Window>> {
        self.surface.clone()
    }

    /// Winit window
    pub fn window(&self) -> &Window {
        self.surface.window()
    }

    /// Winit window size
    #[allow(unused)]
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
    #[allow(unused)]
    pub fn resolution(&self) -> [u32; 2] {
        let size = self.window().inner_size();
        let scale_factor = self.window().scale_factor();
        [
            (size.width as f64 / scale_factor) as u32,
            (size.height as f64 / scale_factor) as u32,
        ]
    }

    /// Add interim image view that can be used as a render target.
    pub fn add_interim_image_view(
        &mut self,
        key: usize,
        view_size: Option<[u32; 2]>,
        format: Format,
    ) {
        let image = ImageView::new(
            AttachmentImage::multisampled_with_usage(
                self.device(),
                if view_size.is_some() {
                    view_size.unwrap()
                } else {
                    self.final_image_size()
                },
                SampleCount::Sample1,
                format,
                ImageUsage {
                    sampled: true,
                    // So we can use the image as an input attachment
                    input_attachment: true,
                    // So we can write to the image in e.g. a compute shader
                    storage: true,
                    ..ImageUsage::none()
                },
            )
            .unwrap(),
        )
        .unwrap();
        self.interim_image_views
            .insert(key, (image.clone(), !view_size.is_some()));
    }

    /// Get interim image view by key
    pub fn get_interim_image_view(&mut self, key: usize) -> InterimImageView {
        self.interim_image_views.get(&key).unwrap().clone().0
    }

    /// Remove an interim image view from the renderer
    pub fn remove_interim_image_view(&mut self, key: usize) {
        self.interim_image_views.remove(&key);
    }

    /// Toggles fullscreen view
    pub fn toggle_fullscreen(&mut self) {
        self.is_fullscreen = !self.is_fullscreen;
        self.window().set_fullscreen(if self.is_fullscreen {
            Some(Fullscreen::Borderless(self.window().current_monitor()))
        } else {
            None
        });
    }

    /// Resize swapchain and camera view images
    pub fn resize(&mut self) {
        self.recreate_swapchain = true
    }

    /*================
    RENDERING
    =================*/

    /// Acquires next swapchain image and increments image index
    /// This is the first to call in render orchestration.
    /// Returns a gpu future representing the time after which the swapchain image has been acquired
    /// and previous frame ended.
    /// After this, execute command buffers and return future from them to `finish_frame`.
    pub(crate) fn start_frame(&mut self) -> Result<Box<dyn GpuFuture>, AcquireError> {
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

    /// Finishes render by presenting the swapchain
    pub(crate) fn finish_frame(&mut self, after_future: Box<dyn GpuFuture>) {
        let future = after_future
            .then_swapchain_present(
                self.queue.clone(),
                self.swap_chain.clone(),
                self.image_index,
            )
            .then_signal_fence_and_flush();
        match future {
            Ok(future) => {
                // Prevent OutOfMemory error on Nvidia :(
                // https://github.com/vulkano-rs/vulkano/issues/627
                match future.wait(None) {
                    Ok(x) => x,
                    Err(err) => println!("{:?}", err),
                }
                self.previous_frame_end = Some(future.boxed());
            }
            Err(FlushError::OutOfDate) => {
                self.recreate_swapchain = true;
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
        }
    }

    /// Swapchain is recreated when resized. Interim image views that should follow swapchain
    /// are also recreated
    fn recreate_swapchain_and_views(&mut self) {
        let dimensions: [u32; 2] = self.window().inner_size().into();
        let (new_swapchain, new_images) =
            match self.swap_chain.recreate().dimensions(dimensions).build() {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => {
                    println!(
                        "{}",
                        SwapchainCreationError::UnsupportedDimensions.to_string()
                    );
                    return;
                }
                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
            };

        self.swap_chain = new_swapchain;
        let new_images = new_images
            .into_iter()
            .map(|image| ImageView::new(image).unwrap())
            .collect::<Vec<_>>();
        self.final_views = new_images;
        // Resize images that follow swapchain size
        let resizable_views = self
            .interim_image_views
            .iter()
            .filter(|(_, (_img, follow_swapchain))| *follow_swapchain)
            .map(|c| *c.0)
            .collect::<Vec<usize>>();
        for i in resizable_views {
            self.remove_interim_image_view(i);
            self.add_interim_image_view(i, None, self.image_format());
        }
        self.recreate_swapchain = false;
    }
}

/// Between `start_frame` and `end_frame` use this pipeline to fill framebuffer with your interim image
pub fn image_over_frame_renderpass<F>(
    renderer: &mut Renderer,
    before_pipeline_future: F,
    image: InterimImageView,
) -> Box<dyn GpuFuture>
where
    F: GpuFuture + 'static,
{
    renderer
        .render_passes
        .place_over_frame
        .render(before_pipeline_future, image, renderer.final_image())
        .then_signal_fence_and_flush()
        .unwrap()
        .boxed()
}
