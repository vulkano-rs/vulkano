// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

#[macro_use]
extern crate vulkano;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer;
use vulkano::command_buffer::PrimaryCommandBufferBuilder;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::descriptor::descriptor_set::DescriptorPool;
use vulkano::format::Format;
use vulkano::image::sys::Dimensions;
use vulkano::image::StorageImage;
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::pipeline::ComputePipeline;

use std::time::Duration;

fn main() {
    // Vulkano Compute Shader Example Code
    //
    // This example code will demonstrate using the compute functions of the Vulkano api. This
    // example is completely headless and has no windowing support/requirements.
    //
    // Contrived problem definition:
    // We will start with a simple data buffer (packed_input) filled with 32bpp RGBA values. This
    // should be similar to many common image processing problems, but simplified to show the
    // concepts only. From the packed_input we will compute a regular vulkan texture object
    // (uimage2d) that can be used in rendering logic which will not be shown here. Sometimes
    // compute applications require the opposite (texture -> buffer) logic and so we'll use a second
    // compute shader to re-create the original data buffer as a second step.
    //
    // For simplification, we're not using real images and will instead algorithmically generate and
    // verify the color values.

    // Image metadata:
    let image_width: u32 = 2;
    let image_height: u32 = 2;

    // Buffer metadata:
    let bytes_per_pixel: u32 = 4;
    let byte_len = (bytes_per_pixel * image_width * image_height) as usize;

    // Workgroup metadata:
    let dispatch_count = image_width * image_height;

    // Create a vulkan compute queue
    let instance = Instance::new(None,
        &InstanceExtensions::none(),
        None).expect("failed to create Vulkan instance");
    let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");
    let queue = physical.queue_families().find(|q| {
        q.supports_compute()
    }).expect("couldn't find a compute queue family");
    let (device, mut queues) = Device::new(&physical,
        physical.supported_features(),
        &DeviceExtensions::none(),
        [(queue, 0.5)].iter().cloned()).expect("failed to create device");
    let descriptor_pool = DescriptorPool::new(&device);
    let queue = queues.next().unwrap();

    // Create an image to store the pixel values we compute (this will be the output of step 1 and
    // the input to step 2)
    let image_buffer = StorageImage::new(&device, Dimensions::Dim2d {
            width: image_width,
            height: image_height
        }, Format::R8G8B8A8Uint, Some(queue.family())).unwrap();

    {
        ////////////////////////////////////////////////////////////////////////////////////////////
        // The first step (buffer->image) is contained within the bounds of this block            //
        ////////////////////////////////////////////////////////////////////////////////////////////

        // Define a shader and pipeline for computing images from buffers
        mod shader_mod {
            #![allow(warnings)]
            include!(concat!(env!("OUT_DIR"), "/shaders/src/bin/compute_image_cs.glsl"));
        }
        mod pipeline_mod {
            #![allow(warnings)]
            pipeline_layout!{
                set0: {
                    packed_input: StorageBuffer<[u8]>
                },
                set1: {
                    output_image: StorageImage
                }
            }
        }

        // Create and fill an input buffer: The values will 0, 1, 2, ..(etc) for our contrived
        // example instead of real RGBA values.
        let packed_input = CpuAccessibleBuffer::<[u8]>::from_iter(&device,
            &BufferUsage::all(),
            Some(queue.family()),
            0..byte_len as u8).expect("failed to create buffer");

        // Instantiate our shader/pipeline/command objects
        let shader = shader_mod::Shader::load(&device).expect("failed to create shader module");
        let pipeline = pipeline_mod::CustomPipeline::new(&device).unwrap();
        let set0 = pipeline_mod::set0::Set::new(&descriptor_pool,
            &pipeline,
            &pipeline_mod::set0::Descriptors {
                packed_input: &packed_input
            });
        let set1 = pipeline_mod::set1::Set::new(&descriptor_pool,
            &pipeline,
            &pipeline_mod::set1::Descriptors {
                output_image: &image_buffer
            });
        let pipeline = ComputePipeline::new(&device,
            &pipeline,
            &shader.main_entry_point(),
            &()).unwrap();
        let command = PrimaryCommandBufferBuilder::new(&device, queue.family())
            .dispatch(&pipeline, (&set0, &set1), [dispatch_count, 1, 1], &())
            .build();

        // Submit this command buffer, which causes computation to begin
        command_buffer::submit(&command, &queue).unwrap();
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Note: At this point you would do lots of wonderful things with image_buffer. Oh the things //
    // you could do.                                                                              //
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Just so there is no hand-waving we're going to use a new buffer to store the final result
    let packed_output = CpuAccessibleBuffer::<[u8]>::uninitialized_array(&device,
        byte_len,
        &BufferUsage::all(),
        Some(queue.family())).expect("failed to create buffer");

    {
        // A second shader and pipeline for computing buffers from images
        mod shader_mod {
            #![allow(warnings)]
            include!(concat!(env!("OUT_DIR"), "/shaders/src/bin/compute_buffer_cs.glsl"));
        }
        mod pipeline_mod {
            #![allow(warnings)]
            pipeline_layout!{
                set0: {
                    input_image: StorageImage
                },
                set1: {
                    packed_output: StorageBuffer<[u8]>
                }
            }
        }

        // Instantiate our shader/pipeline/command objects
        let shader = shader_mod::Shader::load(&device).expect("failed to create shader module");	
        let pipeline = pipeline_mod::CustomPipeline::new(&device).unwrap();
        let set0 = pipeline_mod::set0::Set::new(&descriptor_pool,
            &pipeline,
            &pipeline_mod::set0::Descriptors {
                input_image: &image_buffer
            });
        let set1 = pipeline_mod::set1::Set::new(&descriptor_pool,
            &pipeline,
            &pipeline_mod::set1::Descriptors {
                packed_output: &packed_output
            });
        let pipeline = ComputePipeline::new(&device,
            &pipeline,
            &shader.main_entry_point(),
            &()).unwrap();
        let command = PrimaryCommandBufferBuilder::new(&device, queue.family())
            .dispatch(&pipeline, (&set0, &set1), [dispatch_count, 1, 1], &())
            .build();

        // Submit this command buffer, which causes computation to begin
        command_buffer::submit(&command, &queue).unwrap();
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Note: At this point packed_output has been filled with the same values we started from:    //
    // 0, 1, 2, 3...                                                                              //
    ////////////////////////////////////////////////////////////////////////////////////////////////

    let mapping = packed_output.read(Duration::new(0, 0)).unwrap();
    for i in 0..byte_len {
        println!("packed_output[{}] = {}", i, mapping[i]);
        assert!(i == mapping[i] as usize);
    }
}
