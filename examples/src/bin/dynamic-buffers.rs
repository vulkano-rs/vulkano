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
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::pipeline::layout::PipelineLayout;
use vulkano::pipeline::shader::EntryPointAbstract;
use vulkano::pipeline::ComputePipeline;
use vulkano::pipeline::ComputePipelineAbstract;
use vulkano::sync;
use vulkano::sync::GpuFuture;
use vulkano::Version;

fn main() {
    let instance = Instance::new(None, Version::V1_1, &InstanceExtensions::none(), None).unwrap();

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_compute())
        .unwrap();
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &DeviceExtensions {
            khr_storage_buffer_storage_class: true,
            ..DeviceExtensions::none()
        },
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

    // For Graphics pipelines, use the `with_auto_layout` method
    // instead of the `build` method to specify dynamic buffers.
    // `with_auto_layout` will automatically handle tweaking the
    // pipeline.
    let pipeline = Arc::new(
        ComputePipeline::with_pipeline_layout(
            device.clone(),
            &shader.main_entry_point(),
            &(),
            {
                let mut layout_desc = shader.main_entry_point().layout_desc().clone();
                layout_desc.tweak(vec![(0, 0)]); // The dynamic uniform buffer is at set 0, descriptor 0
                Arc::new(PipelineLayout::new(device.clone(), layout_desc).unwrap())
            },
            None,
        )
        .unwrap(),
    );

    // Declare input buffer.
    // Data in a dynamic buffer **MUST** be aligned to min_uniform_buffer_offset_align
    // or min_storage_buffer_offset_align, depending on the type of buffer.
    let data: Vec<u8> = vec![3, 11, 7];
    let min_dynamic_align = device
        .physical_device()
        .properties()
        .min_uniform_buffer_offset_alignment
        .unwrap() as usize;
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

    let layout = pipeline.layout().descriptor_set_layout(0).unwrap();
    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(input_buffer.clone())
            .unwrap()
            .add_buffer(output_buffer.clone())
            .unwrap()
            .build()
            .unwrap(),
    );

    // Build the command buffer, using different offsets for each call.
    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        .dispatch(
            [12, 1, 1],
            pipeline.clone(),
            set.clone(),
            (),
            vec![0 * align as u32],
        )
        .unwrap()
        .dispatch(
            [12, 1, 1],
            pipeline.clone(),
            set.clone(),
            (),
            vec![1 * align as u32],
        )
        .unwrap()
        .dispatch(
            [12, 1, 1],
            pipeline.clone(),
            set.clone(),
            (),
            vec![2 * align as u32],
        )
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
