// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Multisampling anti-aliasing example, using a render pass resolve.
//!
//! # Introduction to multisampling
//!
//! When you draw an object on an image, this object occupies a certain set of pixels. Each pixel
//! of the image is either fully covered by the object, or not covered at all. There is no such
//! thing as a pixel that is half-covered by the object that you're drawing. What this means is
//! that you will sometimes see a "staircase effect" at the border of your object, also called
//! aliasing.
//!
//! The root cause of aliasing is that the resolution of the image is not high enough. If you
//! increase the size of the image you're drawing to, this effect will still exist but will be
//! much less visible.
//!
//! In order to decrease aliasing, some games and programs use what we call "Super-Sampling Anti
//! Aliasing" (SSAA). For example instead of drawing to an image of size 1024x1024, you draw to an
//! image of size 4096x4096. Then at the end, you scale down your image to 1024x1024 by merging
//! nearby pixels. Since the intermediate image is 4 times larger than the destination, this would
//! be x4 SSAA.
//!
//! However this technique is very expensive in terms of GPU power. The fragment shader and all
//! its calculations has to run four times more often.
//!
//! So instead of SSAA, a common alternative is MSAA (MultiSampling Anti Aliasing). The base
//! principle is more or less the same: you draw to an image of a larger dimension, and then at
//! the end you scale it down to the final size. The difference is that the fragment shader is
//! only run once per pixel of the final size, and its value is duplicated to fill to all the
//! pixels of the intermediate image that are covered by the object.
//!
//! For example, let's say that you use x4 MSAA, you draw to an intermediate image of size
//! 4096x4096, and your object covers the whole image. With MSAA, the fragment shader will only
//! be 1,048,576 times (1024 * 1024), compared to 16,777,216 times (4096 * 4096) with 4x SSAA.
//! Then the output of each fragment shader invocation is copied in each of the four pixels of the
//! intermediate image that correspond to each pixel of the final image.
//!
//! Now, let's say that your object doesn't cover the whole image. In this situation, only the
//! pixels of the intermediate image that are covered by the object will receive the output of the
//! fragment shader.
//!
//! Because of the way it works, this technique requires direct support from the hardware,
//! contrary to SSAA which can be done on any machine.
//!
//! # Multisampled images
//!
//! Using MSAA with Vulkan is done by creating a regular image, but with a number of samples per
//! pixel different from 1. For example if you want to use 4x MSAA, you should create an image with
//! 4 samples per pixel. Internally this image will have 4 times as many pixels as its dimensions
//! would normally require, but this is handled transparently for you. Drawing to a multisampled
//! image is exactly the same as drawing to a regular image.
//!
//! However multisampled images have some restrictions, for example you can't show them on the
//! screen (swapchain images are always single-sampled), and you can't copy them into a buffer.
//! Therefore when you have finished drawing, you have to blit your multisampled image to a
//! non-multisampled image. This operation is not a regular blit (blitting a multisampled image is
//! an error), instead it is called *resolving* the image.
//!

use std::sync::Arc;
use image::ImageBuffer;
use image::Rgba;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer, DynamicState};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, Subpass};
use vulkano::image::{AttachmentImage, Dimensions, StorageImage};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::sync::GpuFuture;
use vulkano::format::ClearValue;

fn main() {
    // The usual Vulkan initialization.
    let instance = Instance::new(None, &InstanceExtensions::none(), None).unwrap();
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    let queue_family = physical.queue_families().find(|&q| q.supports_graphics()).unwrap();
    let (device, mut queues) = Device::new(physical, physical.supported_features(),
        &DeviceExtensions::none(), [(queue_family, 0.5)].iter().cloned()).unwrap();
    let queue = queues.next().unwrap();

    // Creating our intermediate multisampled image.
    //
    // As explained in the introduction, we pass the same dimensions and format as for the final
    // image. But we also pass the number of samples-per-pixel, which is 4 here.
    let intermediary = AttachmentImage::transient_multisampled(device.clone(),
        [1024, 1024], 4, Format::R8G8B8A8Unorm).unwrap();

    // This is the final image that will receive the anti-aliased triangle.
    let image = StorageImage::new(device.clone(), Dimensions::Dim2d { width: 1024, height: 1024 },
        Format::R8G8B8A8Unorm, Some(queue.family())).unwrap();

    // In this example, we are going to perform the *resolve* (ie. turning a multisampled image
    // into a non-multisampled one) as part of the render pass. This is the preferred method of
    // doing so, as it the advantage that the Vulkan implementation doesn't have to write the
    // content of the multisampled image back to memory at the end.
    let render_pass = Arc::new(vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            // The first framebuffer attachment is the intermediary image.
            intermediary: {
                load: Clear,
                store: DontCare,
                format: Format::R8G8B8A8Unorm,
                samples: 4,     // This has to match the image definition.
            },
            // The second framebuffer attachment is the final image.
            color: {
                load: DontCare,
                store: Store,
                format: Format::R8G8B8A8Unorm,
                samples: 1,     // Same here, this has to match.
            }
        },
        pass: {
            // When drawing, we have only one output which is the intermediary image.
            color: [intermediary],
            depth_stencil: {},
            // The `resolve` array here must contain either zero entry (if you don't use
            // multisampling), or one entry per color attachment. At the end of the pass, each
            // color attachment will be *resolved* into the given image. In other words, here, at
            // the end of the pass, the `intermediary` attachment will be copied to the attachment
            // named `color`.
            resolve: [color],
        }
    ).unwrap());

    // Creating the framebuffer, the calls to `add` match the list of attachments in order.
    let framebuffer = Arc::new(Framebuffer::start(render_pass.clone())
        .add(intermediary.clone()).unwrap()
        .add(image.clone()).unwrap()
        .build().unwrap());

    // Here is the "end" of the multisampling example, as starting from here everything is the same
    // as in any other example.
    // The pipeline, vertex buffer, and command buffer are created in exactly the same way as
    // without multisampling.
    // At the end of the example, we copy the content of `image` (ie. the final image) to a buffer,
    // then read the content of that buffer and save it to a PNG file.

    mod vs {
        vulkano_shaders::shader!{
            ty: "vertex",
            src: "
#version 450

layout(location = 0) in vec2 position;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}"
        }
    }

    mod fs {
        vulkano_shaders::shader!{
            ty: "fragment",
            src: "
#version 450

layout(location = 0) out vec4 f_color;

void main() {
    f_color = vec4(1.0, 0.0, 0.0, 1.0);
}"
        }
    }

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    #[derive(Default, Copy, Clone)]
    struct Vertex {
        position: [f32; 2],
    }
    vulkano::impl_vertex!(Vertex, position);

    let vertex1 = Vertex { position: [-0.5, -0.5] };
    let vertex2 = Vertex { position: [ 0.0,  0.5] };
    let vertex3 = Vertex { position: [ 0.5, -0.25] };
    let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
        vec![vertex1, vertex2, vertex3].into_iter()).unwrap();

    let pipeline = Arc::new(GraphicsPipeline::start()
        .vertex_input_single_buffer::<Vertex>()
        .vertex_shader(vs.main_entry_point(), ())
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(fs.main_entry_point(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap());

    let dynamic_state = DynamicState {
        viewports: Some(vec![Viewport {
            origin: [0.0, 0.0],
            dimensions: [1024.0, 1024.0],
            depth_range: 0.0 .. 1.0,
        }]),
        .. DynamicState::none()
    };

    let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(),
        (0 .. 1024 * 1024 * 4).map(|_| 0u8)).unwrap();

    let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
        .begin_render_pass(framebuffer.clone(), false, vec![[0.0, 0.0, 1.0, 1.0].into(), ClearValue::None])
        .unwrap()

        .draw(pipeline.clone(), &dynamic_state, vertex_buffer.clone(), (), ())
        .unwrap()

        .end_render_pass()
        .unwrap()

        .copy_image_to_buffer(image.clone(), buf.clone())
        .unwrap()

        .build()
        .unwrap();

    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap().wait(None).unwrap();

    let buffer_content = buf.read().unwrap();
    let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
    image.save("triangle.png").unwrap();
}
