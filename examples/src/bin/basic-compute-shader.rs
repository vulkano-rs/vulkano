// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// This example demonstrates how to use the compute capabilities of Vulkan.
//
// While graphics cards have traditionally been used for graphical operations, over time they have
// been more or more used for general-purpose operations as well. This is called "General-Purpose
// GPU", or *GPGPU*. This is what this example demonstrates.

use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::instance::{Instance, InstanceExtensions};
use vulkano::pipeline::ComputePipeline;
use vulkano::pipeline::ComputePipelineAbstract;
use vulkano::sync;
use vulkano::sync::GpuFuture;
use vulkano::Version;

fn main() {
    // As with other examples, the first step is to create an instance.
    let instance = Instance::new(None, Version::V1_1, &InstanceExtensions::none(), None).unwrap();

    // Choose which physical device to use.
    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::none()
    };
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            // The Vulkan specs guarantee that a compliant implementation must provide at least one queue
            // that supports compute operations.
            p.queue_families()
                .find(|&q| q.supports_compute())
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type.unwrap() {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name.as_ref().unwrap(),
        physical_device.properties().device_type.unwrap()
    );

    // Now initializing the device.
    let (device, mut queues) = Device::new(
        physical_device,
        &Features::none(),
        &physical_device
            .required_extensions()
            .union(&device_extensions),
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();

    // Since we can request multiple queues, the `queues` variable is in fact an iterator. In this
    // example we use only one queue, so we just retrieve the first and only element of the
    // iterator and throw it away.
    let queue = queues.next().unwrap();

    // Now let's get to the actual example.
    //
    // What we are going to do is very basic: we are going to fill a buffer with 64k integers
    // and ask the GPU to multiply each of them by 12.
    //
    // GPUs are very good at parallel computations (SIMD-like operations), and thus will do this
    // much more quickly than a CPU would do. While a CPU would typically multiply them one by one
    // or four by four, a GPU will do it by groups of 32 or 64.
    //
    // Note however that in a real-life situation for such a simple operation the cost of
    // accessing memory usually outweighs the benefits of a faster calculation. Since both the CPU
    // and the GPU will need to access data, there is no other choice but to transfer the data
    // through the slow PCI express bus.

    // We need to create the compute pipeline that describes our operation.
    //
    // If you are familiar with graphics pipeline, the principle is the same except that compute
    // pipelines are much simpler to create.
    let pipeline = Arc::new({
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                src: "
                    #version 450

                    layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

                    layout(set = 0, binding = 0) buffer Data {
                        uint data[];
                    } data;

                    void main() {
                        uint idx = gl_GlobalInvocationID.x;
                        data.data[idx] *= 12;
                    }
                "
            }
        }
        let shader = cs::Shader::load(device.clone()).unwrap();
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &(), None).unwrap()
    });

    // We start by creating the buffer that will store the data.
    let data_buffer = {
        // Iterator that produces the data.
        let data_iter = (0..65536u32).map(|n| n);
        // Builds the buffer and fills it with this iterator.
        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::none()
            },
            false,
            data_iter,
        )
        .unwrap()
    };

    // In order to let the shader access the buffer, we need to build a *descriptor set* that
    // contains the buffer.
    //
    // The resources that we bind to the descriptor set must match the resources expected by the
    // pipeline which we pass as the first parameter.
    //
    // If you want to run the pipeline on multiple different buffers, you need to create multiple
    // descriptor sets that each contain the buffer you want to run the shader on.
    let layout = pipeline.layout().descriptor_set_layout(0).unwrap();
    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(data_buffer.clone())
            .unwrap()
            .build()
            .unwrap(),
    );

    // In order to execute our operation, we have to build a command buffer.
    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();
    builder
        // The command buffer only does one thing: execute the compute pipeline.
        // This is called a *dispatch* operation.
        //
        // Note that we clone the pipeline and the set. Since they are both wrapped around an
        // `Arc`, this only clones the `Arc` and not the whole pipeline or set (which aren't
        // cloneable anyway). In this example we would avoid cloning them since this is the last
        // time we use them, but in a real code you would probably need to clone them.
        .dispatch([1024, 1, 1], pipeline.clone(), set.clone(), (), vec![])
        .unwrap();
    // Finish building the command buffer by calling `build`.
    let command_buffer = builder.build().unwrap();

    // Let's execute this command buffer now.
    // To do so, we TODO: this is a bit clumsy, probably needs a shortcut
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        // This line instructs the GPU to signal a *fence* once the command buffer has finished
        // execution. A fence is a Vulkan object that allows the CPU to know when the GPU has
        // reached a certain point.
        // We need to signal a fence here because below we want to block the CPU until the GPU has
        // reached that point in the execution.
        .then_signal_fence_and_flush()
        .unwrap();

    // Blocks execution until the GPU has finished the operation. This method only exists on the
    // future that corresponds to a signalled fence. In other words, this method wouldn't be
    // available if we didn't call `.then_signal_fence_and_flush()` earlier.
    // The `None` parameter is an optional timeout.
    //
    // Note however that dropping the `future` variable (with `drop(future)` for example) would
    // block execution as well, and this would be the case even if we didn't call
    // `.then_signal_fence_and_flush()`.
    // Therefore the actual point of calling `.then_signal_fence_and_flush()` and `.wait()` is to
    // make things more explicit. In the future, if the Rust language gets linear types vulkano may
    // get modified so that only fence-signalled futures can get destroyed like this.
    future.wait(None).unwrap();

    // Now that the GPU is done, the content of the buffer should have been modified. Let's
    // check it out.
    // The call to `read()` would return an error if the buffer was still in use by the GPU.
    let data_buffer_content = data_buffer.read().unwrap();
    for n in 0..65536u32 {
        assert_eq!(data_buffer_content[n as usize], n * 12);
    }
}
