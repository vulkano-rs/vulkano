// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// Welcome to the ray tracing example!
//
// While real-time rendering has traditionally been using rasterization to render
// primitives, advances in computing power of graphics cards have enabled ray tracing,
// the tracing of paths of light throughout a scene to be performed in real time.
// This example demonstrates simple ray tracing.

extern crate vulkano;
extern crate vulkano_shaders;
extern crate vulkano_win;
extern crate winit;

use vulkano::buffer::{BufferUsage, ImmutableBuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::RayTracingPipeline;
use vulkano::swapchain;
use vulkano::swapchain::{
    AcquireError, ColorSpace, FullscreenExclusive, PresentMode, SurfaceTransform, Swapchain,
    SwapchainCreationError,
};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};

use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

use std::sync::Arc;

fn main() {
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, &required_extensions, None).unwrap();
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    println!(
        "Using device: {} (type: {:?})",
        physical.name(),
        physical.ty()
    );

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let queue_family = physical
        .queue_families()
        .find(|&q| {
            // TODO: use QUEUE_TRANSFER_BIT?
            q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
        })
        .unwrap();

    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        khr_get_memory_requirements2: true,
        nv_ray_tracing: true,
        ..DeviceExtensions::none()
    };
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        assert!(caps.supported_usage_flags.storage);
        let usage = caps.supported_usage_flags;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();
        let mode = if caps.present_modes.mailbox {
            PresentMode::Mailbox
        } else if caps.present_modes.immediate {
            PresentMode::Immediate
        } else {
            PresentMode::Fifo
        };

        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count,
            format,
            dimensions,
            1,
            usage,
            &queue,
            SurfaceTransform::Identity,
            alpha,
            mode,
            FullscreenExclusive::Default,
            true,
            ColorSpace::SrgbNonLinear,
        )
        .unwrap()
    };

    mod rs {
        vulkano_shaders::shader! {
            ty: "ray_generation",
            src: "#version 460 core
#extension GL_NV_ray_tracing : enable

layout(set = 0, binding = 0, rgba8) uniform image2D result;
layout(location = 0) rayPayloadNV vec4 payload;

void main() {
    ivec2 coord = ivec2(gl_LaunchIDNV);
    const vec2 pixelCenter = coord + vec2(0.5);
    const vec2 inUV = pixelCenter / vec2(gl_LaunchSizeNV.xy);

    payload = vec4(inUV.x, inUV.y, 1.0f, 1.0f);
    imageStore(result, coord, payload);
}
"
        }
    }
    let rs = rs::Shader::load(device.clone()).unwrap();

    // We set a limit to the recursion of a ray so that the shader does not run infinitely
    let max_recursion_depth = 5;

    let pipeline = Arc::new(
        RayTracingPipeline::nv(max_recursion_depth)
        // We need at least one ray generation shader to describe where rays go
        // and to store the result of their path tracing
        .raygen_shader(rs.main_entry_point(), ())
        .build(device.clone())
        .unwrap(),
    );

    let layout = pipeline.layout().descriptor_set_layout(0).unwrap();
    let mut sets = images
        .iter()
        .map(|image| {
            Arc::new(
                PersistentDescriptorSet::start(layout.clone())
                    .add_image(image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            )
        })
        .collect::<Vec<_>>();

    // TODO: Auto-generate a shader binding table buffer and return 4 views
    // TODO: Return 1 buffer with 4 slices instead of 4 different buffers
    // TODO: `miss_shader_binding_table`, `hit_shader_binding_table`,
    //       `callable_shader_binding_table` should empty buffers (size 0 and no handles)
    //       if there are no handles
    let group_handles = pipeline.group_handles(queue.clone());
    let group_handle_size = device.physical_device().shader_group_handle_size() as usize;

    let (raygen_shader_binding_table, raygen_buffer_future) = ImmutableBuffer::from_iter(
        group_handles[0..group_handle_size].iter().copied(),
        BufferUsage::ray_tracing(),
        queue.clone(),
    )
    .unwrap();
    let (miss_shader_binding_table, miss_buffer_future) = ImmutableBuffer::from_iter(
        (0..0).map(|_| 5u8),
        BufferUsage::ray_tracing(),
        queue.clone(),
    )
    .unwrap();
    let (hit_shader_binding_table, hit_buffer_future) = ImmutableBuffer::from_iter(
        (0..0).map(|_| 5u8),
        BufferUsage::ray_tracing(),
        queue.clone(),
    )
    .unwrap();
    let (callable_shader_binding_table, callable_buffer_future) = ImmutableBuffer::from_iter(
        (0..0).map(|_| 5u8),
        BufferUsage::ray_tracing(),
        queue.clone(),
    )
    .unwrap();

    raygen_buffer_future
        .join(miss_buffer_future)
        .join(hit_buffer_future)
        .join(callable_buffer_future)
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();


    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            recreate_swapchain = true;
        }
        Event::RedrawEventsCleared => {
            previous_frame_end.as_mut().unwrap().cleanup_finished();

            if recreate_swapchain {
                let dimensions: [u32; 2] = surface.window().inner_size().into();
                let (new_swapchain, new_images) =
                    match swapchain.recreate_with_dimensions(dimensions) {
                        Ok(r) => r,
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                    };

                swapchain = new_swapchain;
                let layout = pipeline.layout().descriptor_set_layout(0).unwrap();
                sets = new_images
                    .iter()
                    .map(|image| {
                        Arc::new(
                            PersistentDescriptorSet::start(layout.clone())
                                .add_image(image.clone())
                                .unwrap()
                                .build()
                                .unwrap(),
                        )
                    })
                    .collect::<Vec<_>>();
                recreate_swapchain = false;
            }

            let (image_num, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            let future = previous_frame_end
                .take()
                .unwrap()
                .join(acquire_future)
                .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => {
                    previous_frame_end = Some(Box::new(future) as Box<_>);
                }
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                }
                Err(e) => {
                    println!("Failed to flush future: {:?}", e);
                    previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                }
            }
        }
        _ => (),
    });
}
