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

use std::{iter::repeat, mem::size_of};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, layout::DescriptorType, DescriptorSet,
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
        QueueFlags,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::{self, GpuFuture},
    DeviceSize, VulkanLibrary,
};

fn main() {
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            // Enable enumerating devices that use non-conformant vulkan implementations. (ex. MoltenVK)
            enumerate_portability: true,
            ..Default::default()
        },
    )
    .unwrap();

    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::empty()
    };
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
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
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
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
                layout(set = 0, binding = 0) uniform InData {
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

    let shader = shader::load(device.clone()).unwrap();
    let pipeline = ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").unwrap(),
        &(),
        None,
        |layout_create_infos| {
            let binding = layout_create_infos[0].bindings.get_mut(&0).unwrap();
            binding.descriptor_type = DescriptorType::UniformBufferDynamic;
        },
    )
    .unwrap();

    let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

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
    let align = (size_of::<u32>() + min_dynamic_align - 1) & !(min_dynamic_align - 1);
    let aligned_data = {
        let mut aligned_data = Vec::with_capacity(align * data.len());
        for elem in data {
            let bytes = elem.to_ne_bytes();
            // Fill up the buffer with data
            for b in bytes {
                aligned_data.push(b);
            }
            // Zero out any padding needed for alignment
            aligned_data.extend(repeat(0).take(align - bytes.len()));
        }
        aligned_data
    };

    let input_buffer = CpuAccessibleBuffer::from_iter(
        &memory_allocator,
        BufferUsage::UNIFORM_BUFFER,
        false,
        aligned_data.into_iter(),
    )
    .unwrap();

    let output_buffer = CpuAccessibleBuffer::from_iter(
        &memory_allocator,
        BufferUsage::STORAGE_BUFFER,
        false,
        (0..12).map(|_| 0u32),
    )
    .unwrap();

    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout.clone(),
        [
            // When writing to the dynamic buffer binding, the range of the buffer that the shader
            // will access must also be provided. We specify the size of the `InData` struct here.
            // When dynamic offsets are provided later, they get added to the start and end of
            // this range.
            WriteDescriptorSet::buffer_with_range(
                0,
                input_buffer,
                0..size_of::<shader::ty::InData>() as DeviceSize,
            ),
            WriteDescriptorSet::buffer(1, output_buffer.clone()),
        ],
    )
    .unwrap();

    // Build the command buffer, using different offsets for each call.
    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    #[allow(clippy::erasing_op, clippy::identity_op)]
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
            set.offsets([2 * align as u32]),
        )
        .dispatch([12, 1, 1])
        .unwrap();
    let command_buffer = builder.build().unwrap();

    let future = sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();

    let output_content = output_buffer.read().unwrap();
    println!("Output: {:?}", &*output_content);
}
