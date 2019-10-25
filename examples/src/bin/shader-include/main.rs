// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// This example demonstrates how to use the standard and relative include directives within
// shader source code. The boilerplate is taken from the "basic-compute-shader.rs" example, where
// most of the boilerplate is explained.

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::GpuFuture;
use vulkano::sync;

use std::sync::Arc;

fn main() {
   let instance = Instance::new(None, &InstanceExtensions::none(), None).unwrap();
   let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
   let queue_family = physical.queue_families().find(|&q| q.supports_compute()).unwrap();
   let (device, mut queues) = Device::new(physical, physical.supported_features(),
       &DeviceExtensions::none(), [(queue_family, 0.5)].iter().cloned()).unwrap();
   let queue = queues.next().unwrap();

   println!("Device initialized");

   let pipeline = Arc::new({
       mod cs {
           vulkano_shaders::shader!{
               ty: "compute",
               // We declare what directories to search for when using the `#include <...>`
               // syntax. Specified directories have descending priorities based on their order.
               include: [ "src/bin/shader-include/standard-shaders" ],
               src: "
#version 450
// Substitutes this line with the contents of the file `common.glsl` found in one of the standard
// `include` directories specified above.
// Note, that relative inclusion (`#include \"...\"`), although it falls back to standard
// inclusion, should not be used for **embedded** shader source, as it may be misleading and/or
// confusing.
#include <common.glsl>

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data {
   uint data[];
} data;

void main() {
   uint idx = gl_GlobalInvocationID.x;
   data.data[idx] = multiply_by_12(data.data[idx]);
}"
           }
       }
       let shader = cs::Shader::load(device.clone()).unwrap();
       ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap()
   });

   let data_buffer = {
       let data_iter = (0 .. 65536u32).map(|n| n);
       CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), data_iter).unwrap()
   };
   let set = Arc::new(PersistentDescriptorSet::start(pipeline.clone(), 0)
       .add_buffer(data_buffer.clone()).unwrap()
       .build().unwrap()
   );
   let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
       .dispatch([1024, 1, 1], pipeline.clone(), set.clone(), ()).unwrap()
       .build().unwrap();
   let future = sync::now(device.clone())
       .then_execute(queue.clone(), command_buffer).unwrap()
       .then_signal_fence_and_flush().unwrap();

   future.wait(None).unwrap();

   let data_buffer_content = data_buffer.read().unwrap();
   for n in 0 .. 65536u32 {
       assert_eq!(data_buffer_content[n as usize], n * 12);
   }
}
