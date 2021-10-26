// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// This example demonstrates how to use dynamic uniform buffers.
//
// Dynamic uniform and storage buffers store buffer data for different
// calls in one large buffer. Each draw or dispatch call can specify an
// offset into the buffer to read object data from, without having to
// rebind descriptor sets.

use std::mem;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::{DescriptorSet, PersistentDescriptorSet};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::instance::{Instance, InstanceExtensions};
use vulkano::pipeline::{ComputePipeline, PipelineBindPoint};
use vulkano::sync;
use vulkano::sync::GpuFuture;
use vulkano::Version;

fn main() {
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

    mod shader {
        vulkano_shaders::shader! {
            ty: "compute",
            src: "
                #version 450

                layout(local_size_x = 12) in;

                // Uniform Buffer Object
                layout(set = 0, binding = 0) uniform readonly InData {
                    uint data;
                } ubo;

                // Output Buffer
                layout(set = 0, binding = 1) buffer OutData {
                    uint data[];
                } data;

                // Toy shader that only runs for the index specified in `ubo`.
                void main() {
                    uint index = gl_GlobalInvocationID.x;
                    if(index == ubo.data) {
                        data.data[index] = index;
                    }
                }
                "
        }
    }

    let shader = shader::Shader::load(device.clone()).unwrap();
    let pipeline = ComputePipeline::new(
        device.clone(),
        &shader.main_entry_point(),
        &(),
        None,
        |set_descs| {
            set_descs[0].set_buffer_dynamic(0);
        },
    )
    .unwrap();

    // Declare input buffer.
    // Data in a dynamic buffer **MUST** be aligned to min_uniform_buffer_offset_align
    // or min_storage_buffer_offset_align, depending on the type of buffer.
    let data: Vec<u8> = vec![3, 11, 7];
    let min_dynamic_align = device
        .physical_device()
        .properties()
        .min_uniform_buffer_offset_alignment as usize;
    println!(
        "Minimum uniform buffer offset alignment: {}",
        min_dynamic_align
    );
    println!("Input: {:?}", data);
    // Round size up to the next multiple of align.
    let align = (mem::size_of::<u32>() + min_dynamic_align - 1) & !(min_dynamic_align - 1);
    let aligned_data = {
        let mut aligned_data = Vec::with_capacity(align * data.len());
        for i in 0..data.len() {
            let bytes = data[i].to_ne_bytes();
            // Fill up the buffer with data
            for bi in 0..bytes.len() {
                aligned_data.push(bytes[bi]);
            }
            // Zero out any padding needed for alignment
            for _ in 0..align - bytes.len() {
                aligned_data.push(0);
            }
        }
        aligned_data
    };

    let input_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        aligned_data.into_iter(),
    )
    .unwrap();

    let output_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        (0..12).map(|_| 0u32),
    )
    .unwrap();

    let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
    let mut set_builder = PersistentDescriptorSet::start(layout.clone());

    set_builder
        .add_buffer(input_buffer.clone())
        .unwrap()
        .add_buffer(output_buffer.clone())
        .unwrap();

    let set = set_builder.build().unwrap();

    // Build the command buffer, using different offsets for each call.
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
            set.clone().offsets([0 * align as u32]),
        )
        .dispatch([12, 1, 1])
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            set.clone().offsets([1 * align as u32]),
        )
        .dispatch([12, 1, 1])
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            set.clone().offsets([2 * align as u32]),
        )
        .dispatch([12, 1, 1])
        .unwrap();
    let command_buffer = builder.build().unwrap();

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let output_content = output_buffer.read().unwrap();
    println!(
        "Output: {:?}",
        output_content.iter().cloned().collect::<Vec<u32>>()
    );
}
