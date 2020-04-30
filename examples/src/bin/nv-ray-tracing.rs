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

use vulkano::acceleration_structure::{AabbPositions, AccelerationStructure};
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
use std::time::Instant;

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
        khr_storage_buffer_storage_class: true,
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

    // We now create a buffer that will store the AABBs for custom intersections.
    let (aabb_buffer, aabb_buffer_future) = {
        ImmutableBuffer::from_iter(
            [
                AabbPositions {
                    min: [-2.0, -0.5, -2.5],
                    max: [-1.0, 0.5, -1.5],
                },
                AabbPositions {
                    min: [-0.5, -0.5, -2.5],
                    max: [0.5, 0.5, -1.5],
                },
                AabbPositions {
                    min: [1.0, -0.5, -2.5],
                    max: [2.0, 0.5, -1.5],
                },
            ]
            .iter()
            .cloned(),
            BufferUsage {
                storage_buffer: true,
                ..BufferUsage::none()
            },
            queue.clone(),
        )
        .unwrap()
    };

    aabb_buffer_future
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    mod rs {
        vulkano_shaders::shader! {
            ty: "ray_generation",
            src: "#version 460 core
#extension GL_NV_ray_tracing : enable

layout(set = 0, binding = 0, rgba8) uniform image2D result;
layout(set = 0, binding = 1) uniform accelerationStructureNV scene;

struct payload_t {
    vec3 color;
    uint recursion_depth;
};

layout(location = 0) rayPayloadNV payload_t payload;

void main() {
    ivec2 coord = ivec2(gl_LaunchIDNV);
    const vec2 pixelCenter = coord + vec2(0.5);
    const vec2 inUV = pixelCenter / vec2(gl_LaunchSizeNV.xy);

    float aspect = float(gl_LaunchSizeNV.x) / gl_LaunchSizeNV.y;
    vec3 lower_left_corner = vec3(-aspect, 1.0, -1.0);
    vec3 horizontal = vec3(2.0 * aspect, 0.0, 0.0);
    vec3 vertical = vec3(0.0, -2.0, 0.0);
    vec3 origin = vec3(0.0, 0.0, 0.0);
    vec3 direction = normalize(lower_left_corner + inUV.x * horizontal + inUV.y * vertical);

    payload.recursion_depth = 0;
    traceNV(scene, gl_RayFlagsOpaqueNV, 0xFF, 0, 0, 0, origin, 0.001, direction, 1000.0, 0);
    imageStore(result, coord, vec4(payload.color, 1.0));
}
"
        }
    }
    let rs = rs::Shader::load(device.clone()).unwrap();

    mod ms {
        vulkano_shaders::shader! {
            ty: "miss",
            src: "#version 460 core
#extension GL_NV_ray_tracing : enable

struct payload_t {
    vec3 color;
    uint recursion_depth;
};

layout(location = 0) rayPayloadInNV payload_t payload;

void main() {
    vec3 unit_direction = normalize(gl_WorldRayDirectionNV);
    float t = 0.5 * (unit_direction.y + 1.0);
    payload.color = mix(vec3(0.5, 0.7, 1.0), vec3(1.0, 1.0, 1.0), t);
}
"
        }
    }
    let ms = ms::Shader::load(device.clone()).unwrap();

    mod cs {
        vulkano_shaders::shader! {
            ty: "closest_hit",
            src: "#version 460 core
#extension GL_NV_ray_tracing : enable

layout(set = 0, binding = 1) uniform accelerationStructureNV scene;

struct payload_t {
    vec3 color;
    uint recursion_depth;
};

layout(location = 0) rayPayloadInNV payload_t payload;

struct hit_record_t {
    vec3 position;
    vec3 normal;
};
hitAttributeNV hit_record_t hit_record;

void main() {
    payload.recursion_depth++;
    if (payload.recursion_depth < 15) {
        vec3 target = reflect(gl_WorldRayDirectionNV, hit_record.normal);
        vec3 origin = hit_record.position + 0.001 * hit_record.normal;
        traceNV(scene, gl_RayFlagsOpaqueNV, 0xFF, 0, 0, 0, origin, 0.001, target, 1000.0, 0);
        payload.color *= 0.9f;
    }
}
"
        }
    }
    let cs = cs::Shader::load(device.clone()).unwrap();

    mod is {
        vulkano_shaders::shader! {
            ty: "intersection",
            src: "#version 460 core
#extension GL_NV_ray_tracing : enable

struct Aabb {
    float min_x;
    float min_y;
    float min_z;
    float max_x;
    float max_y;
    float max_z;
};

layout(set = 0, binding = 2) readonly buffer AabbArray { Aabb[] aabbs; };

layout(push_constant) uniform PushConstants {
    float time;
} push_constants;

struct hit_record_t {
    vec3 position;
    vec3 normal;
};
hitAttributeNV hit_record_t hit_record;

const uint MAX_STEPS = 100;
const uint MAX_REFINE_STEPS = 4;
const float MIN_DISTANCE = 1e-6f;
const float MIN_STEP_SIZE = 0.5f;
float CUBE_SIDES = 0.45f;
float SPHERE_RADIUS = 0.55f;
float GRAD_STEP = 0.01f;

float sphere_sdf(vec3 point, float radius) {
    return length(point) - radius;
}

float box_sdf(vec3 point, vec3 sides) {
    vec3 q = abs(point) - sides;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

vec3 twist_point(vec3 point, float amount) {
    float c = cos(amount * point.y);
    float s = sin(amount * point.y);
    mat2 m = mat2(c, -s, s, c);
    return vec3(m * point.xz, point.y);
}

float signed_distance_function(vec3 point, vec3 size) {
    vec3 twisted = twist_point(point, sin(push_constants.time) * (2 * gl_PrimitiveID - 1));
    return max(box_sdf(twisted, size * CUBE_SIDES), -sphere_sdf(point, min(size.x, min(size.y, size.z)) * SPHERE_RADIUS));
}

void main() {
    vec3 aabb_min = vec3(aabbs[gl_PrimitiveID].min_x, aabbs[gl_PrimitiveID].min_y, aabbs[gl_PrimitiveID].min_z);
    vec3 aabb_max = vec3(aabbs[gl_PrimitiveID].max_x, aabbs[gl_PrimitiveID].max_y, aabbs[gl_PrimitiveID].max_z);
    vec3 center = (aabb_min + aabb_max) * 0.5;
    vec3 size = aabb_max - aabb_min;

    float t = gl_RayTminNV;  // TODO: start stepping at bounding box
    float t_max = gl_RayTmaxNV;  // TODO: stop stepping when out of the bb
    float distance = gl_RayTmaxNV;
    float step_size = 0.0f;
    // Ray march
    for (uint i = 0; i < MAX_STEPS; ++i) {
        distance = signed_distance_function(gl_ObjectRayOriginNV + gl_ObjectRayDirectionNV * t - center, size);
        // Ray has marched close enough to object, register a hit here
        if (distance < MIN_DISTANCE) {
            break;
        }
        // Not close enough, step forward
        step_size = min(abs(distance), MIN_STEP_SIZE);
        t += step_size;
        // Ray has marched too far without any hits, register no hits
        if (t > t_max) {
            break;
        }
    }
    // Ray has hit something
    if (distance < MIN_DISTANCE) {
        // refine the value for t by stepping back and taking smaller steps
        t -= step_size;
        for (uint i = 0; i < MAX_REFINE_STEPS; ++i) {
            step_size *= 0.5;
            distance = signed_distance_function(gl_ObjectRayOriginNV + gl_ObjectRayDirectionNV * t - center, size);
            if (distance >= MIN_DISTANCE) {
               t += step_size;
            }
        }
        vec3 dx = vec3(GRAD_STEP, 0.0, 0.0);
        vec3 dy = vec3(0.0, GRAD_STEP, 0.0);
        vec3 dz = vec3(0.0, 0.0, GRAD_STEP);
        hit_record.position = gl_ObjectRayOriginNV + gl_ObjectRayDirectionNV * t - center;
        hit_record.normal = normalize(vec3(signed_distance_function(hit_record.position + dx, size) - signed_distance_function(hit_record.position - dx, size),
                                           signed_distance_function(hit_record.position + dy, size) - signed_distance_function(hit_record.position - dy, size),
                                           signed_distance_function(hit_record.position + dz, size) - signed_distance_function(hit_record.position - dz, size)));
        reportIntersectionNV(t, 0);
    }
}
"
        }
    }
    let is = is::Shader::load(device.clone()).unwrap();

    // We set a limit to the recursion of a ray so that the shader does not run infinitely
    let max_recursion_depth = 15;

    let pipeline = Arc::new(
        RayTracingPipeline::nv(max_recursion_depth)
        // We need at least one ray generation shader to describe where rays go
        // and to store the result of their path tracing
        .raygen_shader(rs.main_entry_point(), ())
        .miss_shader(ms.main_entry_point(), ())
        .group(RayTracingPipeline::group().closest_hit_shader(cs.main_entry_point(), ()).intersection_shader(is.main_entry_point(), ()))
        .build(device.clone())
        .unwrap(),
    );

    // We create an acceleration structure allow traversal of the scene by rays
    // There can be any number of acceleration structures grouped under a top
    // level structure.
    let acceleration_structure = Arc::new(
        AccelerationStructure::nv()
            .add_aabbs(aabb_buffer.clone())
            .unwrap()
            .build(device.clone(), queue.clone())
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
                    .add_acceleration_structure(acceleration_structure.clone())
                    .unwrap()
                    .add_buffer(aabb_buffer.clone())
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
        group_handles[group_handle_size..2 * group_handle_size].iter().copied(),
        BufferUsage::ray_tracing(),
        queue.clone(),
    )
    .unwrap();
    let (hit_shader_binding_table, hit_buffer_future) = ImmutableBuffer::from_iter(
        group_handles[2 * group_handle_size..3 * group_handle_size]
            .iter()
            .copied(),
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

    let time_start = Instant::now();
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
                                .add_acceleration_structure(acceleration_structure.clone())
                                .unwrap()
                                .add_buffer(aabb_buffer.clone())
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

            let command_buffer =
                AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
                    .unwrap()
                    .build_acceleration_structure(acceleration_structure.as_ref())
                    .unwrap()
                    .trace_rays(
                        pipeline.clone(),
                        raygen_shader_binding_table.clone(),
                        miss_shader_binding_table.clone(),
                        hit_shader_binding_table.clone(),
                        callable_shader_binding_table.clone(),
                        [swapchain.dimensions()[0], swapchain.dimensions()[1], 1],
                        sets[image_num].clone(),
                        is::ty::PushConstants {
                            time: {
                                let elapsed = time_start.elapsed();
                                elapsed.as_secs() as f32 + elapsed.subsec_nanos() as f32 / 1e9
                            },
                        },
                    )
                    .unwrap()
                    .build()
                    .unwrap();

            let future = previous_frame_end
                .take()
                .unwrap()
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
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
