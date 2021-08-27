// Copyright (c) 2020 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// This example demonstrates how to initialize immutable buffers.

use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::instance::{Instance, InstanceExtensions};
use vulkano::pipeline::{ComputePipeline, PipelineBindPoint};
use vulkano::sync;
use vulkano::sync::GpuFuture;
use vulkano::Version;

fn main() {
    // The most part of this example is exactly the same as `basic-compute-shader`. You should read the
    // `basic-compute-shader` example if you haven't done so yet.

    let instance = Instance::new(None, Version::V1_1, &InstanceExtensions::none(), None).unwrap();

    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::none()
    };
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_compute())
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type
    );

    let (device, mut queues) = Device::new(
        physical_device,
        &Features::none(),
        &physical_device
            .required_extensions()
            .union(&device_extensions),
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let pipeline = Arc::new({
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                src: "
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) restrict buffer Data {
    uint data[];
} data;

layout(set = 0, binding = 1) readonly restrict buffer ImmutableData {
    uint data;
} immutable_data;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    data.data[idx] *= 12;
    data.data[idx] += immutable_data.data;
}"
            }
        }
        let shader = cs::Shader::load(device.clone()).unwrap();
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &(), None).unwrap()
    });

    let data_buffer = {
        let data_iter = (0..65536u32).map(|n| n);
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, data_iter)
            .unwrap()
    };

    // Create immutable buffer and initialize it
    let immutable_data_buffer = {
        // uninitialized(), uninitialized_array() and raw() return two things: the buffer,
        // and a special access that should be used for the initial upload to the buffer.
        let (immutable_data_buffer, immutable_data_buffer_init) = unsafe {
            ImmutableBuffer::<u32>::uninitialized(device.clone(), BufferUsage::all()).unwrap()
        };

        // Build command buffer which initialize our buffer.
        let mut builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Initializing a immutable buffer is done by coping data to
        // ImmutableBufferInitialization which is returned by a function we use to create buffer.
        // We can use copy_buffer(), fill_buffer() and some other functions that copies data to
        // buffer also.
        builder
            .update_buffer(immutable_data_buffer_init, &3)
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        // Once a buffer is initialized, we no longer need ImmutableBufferInitialization.
        // So we return only the buffer.
        immutable_data_buffer
    };

    let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
    let mut set_builder = PersistentDescriptorSet::start(layout.clone(), None).unwrap();

    set_builder
        .add_buffer(data_buffer.clone())
        .unwrap()
        // Now you can just add immutable buffer like other buffers.
        .add_buffer(immutable_data_buffer.clone())
        .unwrap();

    let set = Arc::new(
        set_builder
            .build()
            .unwrap()
    );

    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .bind_pipeline_compute(pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            set.clone(),
        )
        .dispatch([1024, 1, 1])
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let data_buffer_content = data_buffer.read().unwrap();
    for n in 0..65536u32 {
        assert_eq!(data_buffer_content[n as usize], n * 12 + 3);
    }
}
