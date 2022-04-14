// Copyright (c) 2020 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// This example demonstrates how to initialize immutable buffers.

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    instance::{Instance, VulkanLibrary},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::{self, GpuFuture},
};

fn main() {
    // The most part of this example is exactly the same as `basic-compute-shader`. You should read the
    // `basic-compute-shader` example if you haven't done so yet.
    let entry = VulkanLibrary::default();
    let instance = Instance::new(entry, Default::default()).unwrap();

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
        DeviceCreateInfo {
            enabled_extensions: physical_device
                .required_extensions()
                .union(&device_extensions),
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let pipeline = {
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
        let shader = cs::load(device.clone()).unwrap();
        ComputePipeline::new(
            device.clone(),
            shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .unwrap()
    };

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
            .update_buffer(immutable_data_buffer_init, &3u32)
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

    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, data_buffer.clone()),
            // Now you can just add immutable buffer like other buffers.
            WriteDescriptorSet::buffer(1, immutable_data_buffer.clone()),
        ],
    )
    .unwrap();

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
