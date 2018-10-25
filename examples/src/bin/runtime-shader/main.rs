// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.
//
// This example demonstrates one way of preparing data structures and loading
// SPIRV shaders from external source (file system).
//
// Note that you will need to do all correctness checking by yourself.
//
// vert.glsl and frag.glsl must be built by yourself.
// One way of building them is to build Khronos' glslang and use
// glslangValidator tool:
// $ glslangValidator vert.glsl -V -S vert -o vert.spv
// $ glslangValidator frag.glsl -V -S frag -o frag.spv
// Vulkano uses glslangValidator to build your shaders internally.
#[macro_use]
extern crate vulkano;
extern crate vulkano_win;
extern crate winit;

use vulkano as vk;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::cpu_access::CpuAccessibleBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::DynamicState;
use vulkano::descriptor::descriptor::DescriptorDesc;
use vulkano::descriptor::descriptor::ShaderStages;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDesc;
use vulkano::descriptor::pipeline_layout::PipelineLayoutDescPcRange;
use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::SwapchainImage;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::shader::{GraphicsShaderType, ShaderInterfaceDef, ShaderInterfaceDefEntry, ShaderModule};
use vulkano::pipeline::vertex::SingleBufferDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError};
use vulkano::swapchain;
use vulkano::sync::GpuFuture;
use vulkano::sync;

use vulkano_win::VkSurfaceBuild;

use winit::Window;

use std::borrow::Cow;
use std::ffi::CStr;
use std::fs::File;
use std::io::Read;
use std::sync::Arc;

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 3],
}

impl_vertex!(Vertex, position, color);

fn main() {
    let instance = vk::instance::Instance::new(
        None,
        &vulkano_win::required_extensions(),
        None,
    ).expect("no instance with surface extension");
    let physical = vk::instance::PhysicalDevice::enumerate(&instance)
        .next()
        .expect("no graphics device");
    let mut events_loop = winit::EventsLoop::new();
    let surface = winit::WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();
    let window = surface.window();
    let (device, mut queues) = {
        let graphical_queue_family = physical
            .queue_families()
            .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
            .expect("couldn't find a graphic queue family");
        let device_ext = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };
        Device::new(
            physical.clone(),
            physical.supported_features(),
            &device_ext,
            [(graphical_queue_family, 0.5)].iter().cloned(),
        ).expect("failed to create device")
    };
    let graphics_queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface
            .capabilities(device.physical_device())
            .expect("failure to get surface capabilities");
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions = caps.current_extent.unwrap_or([1024, 768]);
        let usage = caps.supported_usage_flags;

        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count,
            format,
            dimensions,
            1,
            usage,
            &graphics_queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Fifo,
            true,
            None,
        ).expect("failed to create swapchain")
    };

    let render_pass = Arc::new(
        single_pass_renderpass!(
            device.clone(), attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        ).unwrap(),
    );

    let vs = {
        let mut f = File::open("src/bin/runtime-shader/vert.spv")
            .expect("Can't find file src/bin/runtime-shader/vert.spv This example needs to be run from the root of the example crate.");
        let mut v = vec![];
        f.read_to_end(&mut v).unwrap();
        // Create a ShaderModule on a device the same Shader::load does it.
        // NOTE: You will have to verify correctness of the data by yourself!
        unsafe { ShaderModule::new(device.clone(), &v) }.unwrap()
    };

    let fs = {
        let mut f = File::open("src/bin/runtime-shader/frag.spv")
            .expect("Can't find file src/bin/runtime-shader/frag.spv");
        let mut v = vec![];
        f.read_to_end(&mut v).unwrap();
        unsafe { ShaderModule::new(device.clone(), &v) }.unwrap()
    };

    // This structure will tell Vulkan how input entries of our vertex shader
    // look like.
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    struct VertInput;
    unsafe impl ShaderInterfaceDef for VertInput {
        type Iter = VertInputIter;

        fn elements(&self) -> VertInputIter {
            VertInputIter(0)
        }
    }
    #[derive(Debug, Copy, Clone)]
    struct VertInputIter(u16);
    impl Iterator for VertInputIter {
        type Item = ShaderInterfaceDefEntry;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            // There are things to consider when giving out entries:
            // * There must be only one entry per one location, you can't have
            //   `color' and `position' entries both at 0..1 locations.  They also
            //   should not overlap.
            // * Format of each element must be no larger than 128 bits.
            if self.0 == 0 {
                self.0 += 1;
                return Some(ShaderInterfaceDefEntry {
                    location: 1..2,
                    format: format::Format::R32G32B32Sfloat,
                    name: Some(Cow::Borrowed("color"))
                })
            }
            if self.0 == 1 {
                self.0 += 1;
                return Some(ShaderInterfaceDefEntry {
                    location: 0..1,
                    format: format::Format::R32G32Sfloat,
                    name: Some(Cow::Borrowed("position"))
                })
            }
            None
        }
        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            // We must return exact number of entries left in iterator.
            let len = (2 - self.0) as usize;
            (len, Some(len))
        }
    }
    impl ExactSizeIterator for VertInputIter {
    }
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    struct VertOutput;
    unsafe impl ShaderInterfaceDef for VertOutput {
        type Iter = VertOutputIter;

        fn elements(&self) -> VertOutputIter {
            VertOutputIter(0)
        }
    }
    // This structure will tell Vulkan how output entries (those passed to next
    // stage) of our vertex shader look like.
    #[derive(Debug, Copy, Clone)]
    struct VertOutputIter(u16);
    impl Iterator for VertOutputIter {
        type Item = ShaderInterfaceDefEntry;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            if self.0 == 0 {
                self.0 += 1;
                return Some(ShaderInterfaceDefEntry {
                    location: 0..1,
                    format: format::Format::R32G32B32Sfloat,
                    name: Some(Cow::Borrowed("v_color"))
                })
            }
            None
        }
        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = (1 - self.0) as usize;
            (len, Some(len))
        }
    }
    impl ExactSizeIterator for VertOutputIter {
    }
    // This structure describes layout of this stage.
    #[derive(Debug, Copy, Clone)]
    struct VertLayout(ShaderStages);
    unsafe impl PipelineLayoutDesc for VertLayout {
        // Number of descriptor sets it takes.
        fn num_sets(&self) -> usize { 0 }
        // Number of entries (bindings) in each set.
        fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
            match set { _ => None, }
        }
        // Descriptor descriptions.
        fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
            match (set, binding) { _ => None, }
        }
        // Number of push constants ranges (think: number of push constants).
        fn num_push_constants_ranges(&self) -> usize { 0 }
        // Each push constant range in memory.
        fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
            if num != 0 || 0 == 0 { return None; }
            Some(PipelineLayoutDescPcRange { offset: 0,
                                             size: 0,
                                             stages: ShaderStages::all() })
        }
    }

    // Same as with our vertex shader, but for fragment one instead.
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    struct FragInput;
    unsafe impl ShaderInterfaceDef for FragInput {
        type Iter = FragInputIter;

        fn elements(&self) -> FragInputIter {
            FragInputIter(0)
        }
    }
    #[derive(Debug, Copy, Clone)]
    struct FragInputIter(u16);
    impl Iterator for FragInputIter {
        type Item = ShaderInterfaceDefEntry;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            if self.0 == 0 {
                self.0 += 1;
                return Some(ShaderInterfaceDefEntry {
                    location: 0..1,
                    format: format::Format::R32G32B32Sfloat,
                    name: Some(Cow::Borrowed("v_color"))
                })
            }
            None
        }
        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = (1 - self.0) as usize;
            (len, Some(len))
        }
    }
    impl ExactSizeIterator for FragInputIter {
    }
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    struct FragOutput;
    unsafe impl ShaderInterfaceDef for FragOutput {
        type Iter = FragOutputIter;

        fn elements(&self) -> FragOutputIter {
            FragOutputIter(0)
        }
    }
    #[derive(Debug, Copy, Clone)]
    struct FragOutputIter(u16);
    impl Iterator for FragOutputIter {
        type Item = ShaderInterfaceDefEntry;

        #[inline]
        fn next(&mut self) -> Option<Self::Item> {
            // Note that color fragment color entry will be determined
            // automatically by Vulkano.
            if self.0 == 0 {
                self.0 += 1;
                return Some(ShaderInterfaceDefEntry {
                    location: 0..1,
                    format: format::Format::R32G32B32A32Sfloat,
                    name: Some(Cow::Borrowed("f_color"))
                })
            }
            None
        }
        #[inline]
        fn size_hint(&self) -> (usize, Option<usize>) {
            let len = (1 - self.0) as usize;
            (len, Some(len))
        }
    }
    impl ExactSizeIterator for FragOutputIter {
    }
    // Layout same as with vertex shader.
    #[derive(Debug, Copy, Clone)]
    struct FragLayout(ShaderStages);
    unsafe impl PipelineLayoutDesc for FragLayout {
        fn num_sets(&self) -> usize { 0 }
        fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
            match set { _ => None, }
        }
        fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
            match (set, binding) { _ => None, }
        }
        fn num_push_constants_ranges(&self) -> usize { 0 }
        fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
            if num != 0 || 0 == 0 { return None; }
            Some(PipelineLayoutDescPcRange { offset: 0,
                                             size: 0,
                                             stages: ShaderStages::all() })
        }
    }

    // NOTE: ShaderModule::*_shader_entry_point calls do not do any error
    // checking and you have to verify correctness of what you are doing by
    // yourself.
    //
    // You must be extra careful to specify correct entry point, or program will
    // crash at runtime outside of rust and you will get NO meaningful error
    // information!
    let vert_main = unsafe { vs.graphics_entry_point(
        CStr::from_bytes_with_nul_unchecked(b"main\0"),
        VertInput,
        VertOutput,
        VertLayout(ShaderStages { vertex: true, ..ShaderStages::none() }),
        GraphicsShaderType::Vertex
    ) };

    let frag_main = unsafe { fs.graphics_entry_point(
        CStr::from_bytes_with_nul_unchecked(b"main\0"),
        FragInput,
        FragOutput,
        FragLayout(ShaderStages { fragment: true, ..ShaderStages::none() }),
        GraphicsShaderType::Fragment
    ) };

    let graphics_pipeline = Arc::new(
        GraphicsPipeline::start()
            .vertex_input(SingleBufferDefinition::<Vertex>::new())
            .vertex_shader(vert_main, ())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(frag_main, ())
            .cull_mode_front()
            .front_face_counter_clockwise()
            .depth_stencil_disabled()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap(),
    );

    let mut recreate_swapchain = false;

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        [
            Vertex { position: [-1.0,  1.0], color: [1.0, 0.0, 0.0] },
            Vertex { position: [ 0.0, -1.0], color: [0.0, 1.0, 0.0] },
            Vertex { position: [ 1.0,  1.0], color: [0.0, 0.0, 1.0] },
        ].iter().cloned()
    ).expect("failed to create vertex buffer");

    // NOTE: We don't create any descriptor sets in this example, but you should
    // note that passing wrong types, providing sets at wrong indexes will cause
    // descriptor set builder to return Err!

    let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None };
    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);
    let mut previous_frame_end = Box::new(sync::now(device.clone())) as Box<GpuFuture>;

    loop {
        previous_frame_end.cleanup_finished();

        if recreate_swapchain {
            // Get the new dimensions for the viewport/framebuffers.
            let dimensions = if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
                [dimensions.0, dimensions.1]
            } else {
                return;
            };

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => continue,
                Err(err) => panic!("{:?}", err)
            };

            swapchain = new_swapchain;
            framebuffers = window_size_dependent_setup(&new_images, render_pass.clone(), &mut dynamic_state);

            recreate_swapchain = false;
        }

        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                continue;
            },
            Err(err) => panic!("{:?}", err)
        };

        let command_buffer = AutoCommandBufferBuilder::new(
                device.clone(),
                graphics_queue.family(),
            ).unwrap()
            .begin_render_pass(
                framebuffers[image_num].clone(),
                false,
                vec![[0.0, 0.0, 0.0, 1.0].into()],
            ).unwrap()
            .draw(
                graphics_pipeline.clone(),
                &dynamic_state,
                vertex_buffer.clone(),
                (),
                (),
            ).unwrap()
            .end_render_pass().unwrap()
            .build().unwrap();

        let future = previous_frame_end.join(acquire_future)
            .then_execute(graphics_queue.clone(), command_buffer).unwrap()
            .then_swapchain_present(graphics_queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                previous_frame_end = Box::new(future) as Box<_>;
            }
            Err(vulkano::sync::FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Box::new(vulkano::sync::now(device.clone())) as Box<_>;
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame_end = Box::new(vulkano::sync::now(device.clone())) as Box<_>;
            }
        }

        let mut done = false;
        events_loop.poll_events(|ev| {
            match ev {
                winit::Event::WindowEvent { event: winit::WindowEvent::CloseRequested, .. } => done = true,
                winit::Event::WindowEvent { event: winit::WindowEvent::Resized(_), .. } => recreate_swapchain = true,
                _ => ()
            }
        });
        if done { return; }
    }
}

/// This method is called once during initialization then again whenever the window is resized
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState
) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0 .. 1.0,
    };
    dynamic_state.viewports = Some(vec!(viewport));

    images.iter().map(|image| {
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .build().unwrap()
        ) as Arc<FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>()
}
