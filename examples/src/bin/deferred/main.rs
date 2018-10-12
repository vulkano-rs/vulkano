// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.


// Welcome to the deferred lighting example!
// 
// The idea behind deferred lighting is to render the scene in two steps.
// 
// First you draw all the objects of the scene. But instead of calculating the color they will
// have on the screen, you output their characteristics such as their diffuse color and their
// normals, and write this to images.
// 
// After all the objects are drawn, you should obtain several images that contain the
// characteristics of each pixel.
// 
// Then you apply lighting to the scene. In other words you draw to the final image by taking
// these intermediate images and the various lights of the scene as input.
// 
// This technique allows you to apply tons of light sources to a scene, which would be too
// expensive otherwise. It has some drawbacks, which are the fact that transparent objects must be
// drawn after the lighting, and that the whole process consumes more memory.

extern crate cgmath;
#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shader_derive;
extern crate winit;
extern crate vulkano_win;

use vulkano_win::VkSurfaceBuild;

use vulkano::device::Device;
use vulkano::instance::Instance;
use vulkano::swapchain;
use vulkano::swapchain::PresentMode;
use vulkano::swapchain::SurfaceTransform;
use vulkano::swapchain::Swapchain;
use vulkano::swapchain::AcquireError;
use vulkano::swapchain::SwapchainCreationError;
use vulkano::sync::now;
use vulkano::sync::GpuFuture;

use cgmath::Matrix4;
use cgmath::SquareMatrix;
use cgmath::Vector3;

mod frame;
mod triangle_draw_system;

fn main() {
    // Basic initialization. See the triangle example if you want more details about this.
    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None).expect("failed to create Vulkan instance")
    };
    let physical = vulkano::instance::PhysicalDevice::enumerate(&instance)
                            .next().expect("no device available");

    let mut events_loop = winit::EventsLoop::new();
    let surface = winit::WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();

    let queue_family = physical.queue_families().find(|&q| {
        q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
    }).expect("couldn't find a graphical queue family");

    let (device, mut queues) = {
        let device_ext = vulkano::device::DeviceExtensions {
            khr_swapchain: true,
            .. vulkano::device::DeviceExtensions::none()
        };

        Device::new(physical, physical.supported_features(), &device_ext,
                    [(queue_family, 0.5)].iter().cloned()).expect("failed to create device")
    };

    let queue = queues.next().unwrap();

    let mut dimensions;

    let (mut swapchain, mut images) = {
        let caps = surface.capabilities(physical)
                         .expect("failed to get surface capabilities");
        dimensions = caps.current_extent.unwrap_or([1024, 768]);
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format,
                       dimensions, 1, caps.supported_usage_flags, &queue,
                       SurfaceTransform::Identity, alpha, PresentMode::Fifo, true,
                       None).expect("failed to create swapchain")
    };


    // Here is the basic initialization for the deferred system.
    let mut frame_system = frame::FrameSystem::new(queue.clone(), swapchain.format());
    let triangle_draw_system = triangle_draw_system::TriangleDrawSystem::new(queue.clone(),
                                                                             frame_system.deferred_subpass());

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Box::new(now(device.clone())) as Box<GpuFuture>;

    loop {
        previous_frame_end.cleanup_finished();

        if recreate_swapchain {
            dimensions = surface.capabilities(physical)
                        .expect("failed to get surface capabilities")
                        .current_extent.unwrap();
            
            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => {
                    continue;
                }
                Err(err) => panic!("{:?}", err)
            };

            swapchain = new_swapchain;
            images = new_images;
            recreate_swapchain = false;
        }

        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(),
                                                                              None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                continue;
            }
            Err(err) => panic!("{:?}", err)
        };

        let future = previous_frame_end.join(acquire_future);
        let mut frame = frame_system.frame(future, images[image_num].clone(), Matrix4::identity());
        let mut after_future = None;
        while let Some(pass) = frame.next_pass() {
            match pass {
                frame::Pass::Deferred(mut draw_pass) => {
                    let cb = triangle_draw_system.draw(draw_pass.viewport_dimensions());
                    draw_pass.execute(cb);
                }
                frame::Pass::Lighting(mut lighting) => {
                    lighting.ambient_light([0.1, 0.1, 0.1]);
                    lighting.directional_light(Vector3::new(0.2, -0.1, -0.7), [0.6, 0.6, 0.6]);
                    lighting.point_light(Vector3::new(0.5, -0.5, -0.1), [1.0, 0.0, 0.0]);
                    lighting.point_light(Vector3::new(-0.9, 0.2, -0.15), [0.0, 1.0, 0.0]);
                    lighting.point_light(Vector3::new(0.0, 0.5, -0.05), [0.0, 0.0, 1.0]);
                }
                frame::Pass::Finished(af) => {
                    after_future = Some(af);
                }
            }
        }

        let future = after_future.unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                previous_frame_end = Box::new(future) as Box<_>;
            }
            Err(vulkano::sync::FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Box::new(vulkano::sync::now(device.clone())) as Box<_>;
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame_end = Box::new(vulkano::sync::now(device.clone())) as Box<_>;
            }
        }

        let mut done = false;
        events_loop.poll_events(|ev| {
            match ev {
                winit::Event::WindowEvent { event: winit::WindowEvent::CloseRequested, .. } => done = true,
                _ => ()
            }
        });
        if done { return; }
    }
}
