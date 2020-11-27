// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// TODO: Give a paragraph about what specialization are and what problems they solve

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::pipeline::ComputePipeline;
use vulkano::sync;
use vulkano::sync::GpuFuture;

use std::sync::Arc;

fn main() {
    let instance = Instance::new(None, &InstanceExtensions::none(), None).unwrap();
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

    let shader = cs::Shader::load(device.clone()).unwrap();

    let spec_consts = cs::SpecializationConstants {
        enable: 1,
        multiple: 1,
        addend: 1.0,
    };
    let pipeline = Arc::new(
        ComputePipeline::new(
            device.clone(),
            &shader.main_entry_point(),
            &spec_consts,
            None,
        )
        .unwrap(),
    );

    let data_buffer = {
        let data_iter = (0..65536u32).map(|n| n);
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, data_iter)
            .unwrap()
    };

    let layout = pipeline.layout().descriptor_set_layout(0).unwrap();
    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(data_buffer.clone())
            .unwrap()
            .build()
            .unwrap(),
    );

    let mut builder =
        AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap();
    builder
        .dispatch([1024, 1, 1], pipeline.clone(), set.clone(), ())
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
