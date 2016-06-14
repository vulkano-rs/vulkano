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
use vulkano::image::attachment::AttachmentImage;
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::pipeline::ComputePipeline;

use std::time::Duration;

fn main() {
    // Vulkano Compute Shader Example Code
    //
    // This example code will demonstrate using the compute functions of the Vulkano api. This example is completely headless
    // and has no windowing support/requirements.
    //
    // Contrived problem definition:
    // We will start with a simple data buffer (packed_input) filled with 4bpp RGBA values. This should be similar to many
    // common image processing problems, but simplified to show the concepts only. From the input buffer we will create a
    // regular vulkan texture object (image2d) that can be used in rendering logic which will not be shown here. Sometimes
    // compute applications require the opposite (texture -> buffer) logic and so we'll use a second compute shader to
    // re-create the original data buffer as a second step.
    //
    // For simplification, we're not using real images and will instead algorithmically generate and verify the color values. 

    // Image metadata:
    let image_width: u32 = 2;
    let image_height: u32 = 2;

    // Buffer metadata:
    let bytes_per_pixel: u32 = 4;
    let byte_len = (bytes_per_pixel * image_width * image_height) as usize;

    // Workgroup metadata:
    let dispatch_count = image_width * image_height;

    // Create vulkan objects in the same way as the other examples
    let instance = Instance::new(None, &InstanceExtensions::none(), vec![&"VK_LAYER_LUNARG_standard_validation"]).expect("failed to create Vulkan instance");
    let physical = PhysicalDevice::enumerate(&instance).next().expect("no device available");
    let queue = physical.queue_families().find(|q| {
        q.supports_compute()
    }).expect("couldn't find a compute queue family");
    let (device, mut queues) = Device::new(&physical, physical.supported_features(), &DeviceExtensions::none(), [(queue, 0.5)].iter().cloned()).expect("failed to create device");
    let descriptor_pool = DescriptorPool::new(&device);
    let queue = queues.next().unwrap();

    // Create an image to store the pixel values we compute
    let image_buffer = AttachmentImage::storage(&device, [image_width, image_height], Format::R8G8B8A8Unorm).unwrap();

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // The first step (buffer->image) is contained within the bounds of this big block comment and the next.     //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Use macros to include our buffer-to-image compute shader implementation (it has been compiled to spir-v by the build.rs logic)
    mod compute_image {
        #![allow(warnings)]
        include!(concat!(env!("OUT_DIR"), "/shaders/src/bin/compute_image.glsl"));
    }

    // Use macros to include our pipeline layout implementation
    mod image_pipeline {
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

    // Create and fill an input buffer: The values will 0, 1, 2, ..(etc) for our contrived example instead of real RGBA values.
    let packed_input = {
        let buffer = CpuAccessibleBuffer::<[u8]>::array(&device, byte_len, &BufferUsage::all(), Some(queue.family())).expect("failed to create buffer");
        {
            let mut mapping = buffer.write(Duration::new(0, 0)).unwrap();
            for i in 0..byte_len {
                mapping[i] = i as u8;
            }
        }

        buffer
    };

    // Instantiate our pipeline/shader/command buffer objects
    let image_pipeline_layout = image_pipeline::CustomPipeline::new(&device).unwrap();
    let image_compute_shader = compute_image::Shader::load(&device).expect("failed to create shader module");
    let image_set_input = image_pipeline::set0::Set::new(&descriptor_pool, &image_pipeline_layout, &image_pipeline::set0::Descriptors {
        packed_input: &packed_input
    });
    let image_set_output = image_pipeline::set1::Set::new(&descriptor_pool, &image_pipeline_layout, &image_pipeline::set1::Descriptors {
        output_image: &image_buffer
    });
    let image_compute_pipeline = ComputePipeline::new(&device, &image_pipeline_layout, &image_compute_shader.main_entry_point(), &()).unwrap();
    let image_command = PrimaryCommandBufferBuilder::new(&device, queue.family())
        .dispatch(&image_compute_pipeline, (&image_set_input, &image_set_output), [dispatch_count, 1, 1], &())
        .build();

    // Submit this command buffer, which causes computation to begin
    command_buffer::submit(&image_command, &queue).unwrap();

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Note: At this point you would do lots of wonderful things with image_buffer. Oh the things you could do.  //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Use macros to include our image-to-buffer compute shader implementation
    mod compute_buffer {
        #![allow(warnings)]
        include!(concat!(env!("OUT_DIR"), "/shaders/src/bin/compute_buffer.glsl"));
    }

    // More macro use - The pipeline layout for this operation is purposefully different from the first bit; but you could
    // re-use them if the signatures matched.
    mod buffer_pipeline {
        #![allow(warnings)]
        pipeline_layout!{
            set0: {
                packed_output: StorageBuffer<[u8]>
            },
            set1: {
                input_image: StorageImage
            }
        }
    }

    // Just so there is no hand-waving we're going to use a new buffer to store the final result - but you could re-use packed_input for this
    let final_buffer = CpuAccessibleBuffer::<[u8]>::array(&device, byte_len, &BufferUsage::all(), Some(queue.family())).expect("failed to create buffer");

    // Instantiate our pipeline/shader/command buffer objects
    let buffer_pipeline_layout = buffer_pipeline::CustomPipeline::new(&device).unwrap();
    let buffer_compute_shader = compute_buffer::Shader::load(&device).expect("failed to create shader module");	
    let buffer_set_ouput = buffer_pipeline::set0::Set::new(&descriptor_pool, &buffer_pipeline_layout, &buffer_pipeline::set0::Descriptors {
        packed_output: &final_buffer
    });
    let buffer_set_input = buffer_pipeline::set1::Set::new(&descriptor_pool, &buffer_pipeline_layout, &buffer_pipeline::set1::Descriptors {
        input_image: &image_buffer
    });
    let buffer_compute_pipeline = ComputePipeline::new(&device, &buffer_pipeline_layout, &buffer_compute_shader.main_entry_point(), &()).unwrap();
    let buffer_command = PrimaryCommandBufferBuilder::new(&device, queue.family())
        .dispatch(&buffer_compute_pipeline, (&buffer_set_ouput, &buffer_set_input), [dispatch_count, 1, 1], &())
        .build();

    // Submit this command buffer, which causes computation to begin
    command_buffer::submit(&buffer_command, &queue).unwrap();

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Note: At this point final_buffer has been filled with the same values we started from: 0, 1, 2, 3...      //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    let mapping = final_buffer.read(Duration::new(0, 0)).unwrap();
    for i in 0..byte_len {
        println!("final_buffer[{}] = {}", i, mapping[i]);
    }
}