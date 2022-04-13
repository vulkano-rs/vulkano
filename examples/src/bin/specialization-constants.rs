// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// TODO: Give a paragraph about what specialization are and what problems they solve

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    instance::Instance,
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::{self, GpuFuture},
};

fn main() {
    let entry = Instance::entry();
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

    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            src: "
                #version 450

                layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
                layout(constant_id = 0) const int multiple = 64;
                layout(constant_id = 1) const float addend = 64;
                layout(constant_id = 2) const bool enable = true;
                const vec2 foo = vec2(0, 0); // TODO: How do I hit Instruction::SpecConstantComposite

                layout(set = 0, binding = 0) buffer Data {
                    uint data[];
                } data;

                void main() {
                    uint idx = gl_GlobalInvocationID.x;
                    if (enable) {
                        data.data[idx] *= multiple;
                        data.data[idx] += uint(addend);
                    }
                }
            "
        }
    }

    let shader = cs::load(device.clone()).unwrap();

    let spec_consts = cs::SpecializationConstants {
        enable: 1,
        multiple: 1,
        addend: 1.0,
    };
    let pipeline = ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").unwrap(),
        &spec_consts,
        None,
        |_| {},
    )
    .unwrap();

    let data_buffer = {
        let data_iter = (0..65536u32).map(|n| n);
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, data_iter)
            .unwrap()
    };

    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    let set = PersistentDescriptorSet::new(
        layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())],
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
        assert_eq!(data_buffer_content[n as usize], n * 1 + 1);
    }
    println!("Success");
}
