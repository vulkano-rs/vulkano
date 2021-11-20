extern crate glium;

use std::{
    sync::{Arc, Barrier},
    time::Instant,
};

use glium::glutin::{self, platform::unix::HeadlessContextExt};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{
        submit::SubmitCommandBufferBuilder, AutoCommandBufferBuilder, CommandBufferUsage,
        SubpassContents,
    },
    descriptor_set::PersistentDescriptorSet,
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceExtensions, Queue,
    },
    format::Format,
    image::{view::ImageView, ImageCreateFlags, ImageUsage, StorageImage, SwapchainImage},
    instance::{debug::DebugCallback, Instance, InstanceExtensions},
    pipeline::{
        color_blend::ColorBlendState,
        input_assembly::{InputAssemblyState, PrimitiveTopology},
        viewport::{Scissor, Viewport, ViewportState},
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, RenderPass, Subpass},
    sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode},
    swapchain::{AcquireError, Swapchain, SwapchainCreationError},
    sync::{now, FlushError, GpuFuture, PipelineStages, Semaphore},
    Version,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

fn main() {
    let event_loop_gl = glutin::event_loop::EventLoop::new();
    // For some reason, this must be created before the vulkan window
    let hrb = glutin::ContextBuilder::new()
        .with_gl_debug_flag(true)
        .with_gl(glutin::GlRequest::Latest)
        .build_surfaceless(&event_loop_gl)
        .unwrap();

    let (
        device,
        _instance,
        mut swapchain,
        surface,
        event_loop,
        mut viewport,
        queue,
        render_pass,
        mut framebuffers,
        sampler,
        pipeline,
        vertex_buffer,
    ) = vk_setup();

    let image = StorageImage::new_with_exportable_fd(
        device.clone(),
        vulkano::image::ImageDimensions::Dim2d {
            width: 200,
            height: 200,
            array_layers: 1,
        },
        Format::R16G16B16A16_UNORM,
        ImageUsage {
            sampled: true,
            transfer_source: true,
            transfer_destination: true,
            ..ImageUsage::none()
        },
        ImageCreateFlags {
            mutable_format: true,
            ..ImageCreateFlags::none()
        },
        [queue.family()],
    )
    .unwrap();

    let image_fd = image.export_posix_fd().unwrap();

    let image_view = ImageView::new(image.clone()).unwrap();

    let barrier = Arc::new(Barrier::new(2));
    let barrier_2 = Arc::new(Barrier::new(2));

    let acquire_sem = Arc::new(Semaphore::alloc_with_exportable_fd(device.clone()).unwrap());
    let release_sem = Arc::new(Semaphore::alloc_with_exportable_fd(device.clone()).unwrap());

    let acquire_fd = acquire_sem.export_opaque_fd().unwrap();
    let release_fd = release_sem.export_opaque_fd().unwrap();

    let barrier_clone = barrier.clone();
    let barrier_2_clone = barrier_2.clone();
    build_display(hrb, move |gl_display| {
        let gl_tex = unsafe {
            glium::texture::Texture2d::new_from_fd(
                gl_display.as_ref(),
                glium::texture::UncompressedFloatFormat::U16U16U16U16,
                glium::texture::MipmapsOption::NoMipmap,
                glium::texture::Dimensions::Texture2d {
                    width: 200,
                    height: 200,
                },
                glium::texture::ImportParameters {
                    dedicated_memory: true,
                    size: image.mem_size(),
                    offset: 0,
                    tiling: glium::texture::ExternalTilingMode::Optimal,
                },
                image_fd,
            )
        }
        .unwrap();

        let gl_acquire_sem = unsafe {
            glium::semaphore::Semaphore::new_from_fd(gl_display.as_ref(), acquire_fd).unwrap()
        };

        let gl_release_sem = unsafe {
            glium::semaphore::Semaphore::new_from_fd(gl_display.as_ref(), release_fd).unwrap()
        };

        let rotation_start = Instant::now();

        loop {
            barrier_clone.wait();
            gl_acquire_sem
                .wait_textures(Some(&[(&gl_tex, glium::semaphore::TextureLayout::General)]));

            gl_display.get_context().flush();

            let elapsed = rotation_start.elapsed();
            let rotation = elapsed.as_nanos() as f64 / 2_000_000_000.0;

            use glium::Surface;
            {
                let mut fb = gl_tex.as_surface();

                fb.clear_color(
                    0.0,
                    (((rotation as f32).sin() + 1.) / 2.).powf(2.2),
                    0.0,
                    1.0,
                );
            }
            gl_release_sem
                .signal_textures(Some(&[(&gl_tex, glium::semaphore::TextureLayout::General)]));
            barrier_2_clone.wait();

            gl_display.get_context().finish();

            gl_display.get_context().assert_no_error(Some("err"));
        }
    });

    let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();

    let mut set_builder = PersistentDescriptorSet::start(layout.clone());

    set_builder
        .add_sampled_image(image_view, sampler.clone())
        .unwrap();

    let set = set_builder.build().unwrap();

    let mut recreate_swapchain = false;
    let mut previous_frame_end: Option<Box<dyn GpuFuture>> = Some(Box::new(now(device.clone())));

    event_loop.run(move |event, _, control_flow| {
        unsafe {
            let mut builder = SubmitCommandBufferBuilder::new();
            builder.add_signal_semaphore(&acquire_sem);
            builder.submit(&queue).unwrap();
        };

        barrier.wait();
        barrier_2.wait();

        unsafe {
            let mut builder = SubmitCommandBufferBuilder::new();
            builder.add_wait_semaphore(
                &release_sem,
                PipelineStages {
                    all_commands: true,
                    ..PipelineStages::none()
                },
            );
            builder.submit(&queue).unwrap();
        };

        previous_frame_end.as_mut().unwrap().cleanup_finished();

        if recreate_swapchain {
            let dimensions: [u32; 2] = surface.window().inner_size().into();
            let (new_swapchain, new_images) =
                match swapchain.recreate().dimensions(dimensions).build() {
                    Ok(r) => r,
                    Err(SwapchainCreationError::UnsupportedDimensions) => return,
                    Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                };

            swapchain = new_swapchain;
            framebuffers =
                window_size_dependent_setup(&new_images, render_pass.clone(), &mut viewport);
            recreate_swapchain = false;
        }

        let (image_num, suboptimal, acquire_future) =
            match vulkano::swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };

        if suboptimal {
            recreate_swapchain = true;
        }

        let clear_values = vec![[0.0, 0.0, 1.0, 1.0].into()];
        let mut builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        builder
            .begin_render_pass(
                framebuffers[image_num].clone(),
                SubpassContents::Inline,
                clear_values,
            )
            .unwrap()
            .set_viewport(0, [viewport.clone()])
            .bind_pipeline_graphics(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap()
            .end_render_pass()
            .unwrap();
        let command_buffer = builder.build().unwrap();

        let future = previous_frame_end.take().unwrap().join(acquire_future);

        // let future = then_signal_ext_semaphore(future, sem.clone(), queue.clone());

        let future = // then_wait_ext_semaphore(future, sem.clone(), queue.clone())
            future.then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                future.wait(None).unwrap();
                previous_frame_end = Some(future.boxed());
            }
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Some(vulkano::sync::now(device.clone()).boxed());
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                previous_frame_end = Some(vulkano::sync::now(device.clone()).boxed());
            }
        };

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }

            _ => (),
        };
    });
}

#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 2],
}
vulkano::impl_vertex!(Vertex, position);

fn vk_setup() -> (
    Arc<vulkano::device::Device>,
    Arc<vulkano::instance::Instance>,
    Arc<Swapchain<winit::window::Window>>,
    Arc<vulkano::swapchain::Surface<winit::window::Window>>,
    winit::event_loop::EventLoop<()>,
    vulkano::pipeline::viewport::Viewport,
    Arc<Queue>,
    Arc<RenderPass>,
    Vec<Arc<Framebuffer>>,
    Arc<vulkano::sampler::Sampler>,
    Arc<GraphicsPipeline>,
    Arc<CpuAccessibleBuffer<[Vertex]>>,
) {
    let required_extensions = vulkano_win::required_extensions();

    let instance = Instance::new(
        None,
        Version::V1_2,
        &(InstanceExtensions {
            khr_get_physical_device_properties2: true,
            khr_external_memory_capabilities: true,
            khr_external_semaphore_capabilities: true,
            khr_external_fence_capabilities: true,
            ext_debug_utils: true,
            ..InstanceExtensions::none()
        }
        .union(&required_extensions)),
        vec!["VK_LAYER_KHRONOS_validation"],
    )
    .unwrap();

    let _debug_callback = DebugCallback::errors_and_warnings(&instance, |msg| {
        println!(
            "{} {:?} {:?}: {}",
            msg.layer_prefix.unwrap_or("unknown"),
            msg.ty,
            msg.severity,
            msg.description
        );
    })
    .unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_ext = DeviceExtensions {
        khr_external_semaphore: true,
        khr_external_semaphore_fd: true,
        khr_external_memory: true,
        khr_external_memory_fd: true,
        khr_external_fence: true,
        khr_external_fence_fd: true,
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_ext))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
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
        physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
        physical_device,
        physical_device.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let (swapchain, images) = {
        let caps = surface.capabilities(physical_device).unwrap();
        let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();

        Swapchain::start(device.clone(), surface.clone())
            .num_images(caps.min_image_count)
            .format(format)
            .dimensions(dimensions)
            .usage(ImageUsage::color_attachment())
            .sharing_mode(&queue)
            .composite_alpha(composite_alpha)
            .build()
            .unwrap()
    };

    let vertex_buffer = CpuAccessibleBuffer::<[Vertex]>::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        [
            Vertex {
                position: [-0.5, -0.5],
            },
            Vertex {
                position: [-0.5, 0.5],
            },
            Vertex {
                position: [0.5, -0.5],
            },
            Vertex {
                position: [0.5, 0.5],
            },
        ]
        .iter()
        .cloned(),
    )
    .unwrap();

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    let render_pass = vulkano::single_pass_renderpass!(device.clone(),
        attachments: {
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
    )
    .unwrap();

    let sampler = Sampler::new(
        device.clone(),
        Filter::Linear,
        Filter::Linear,
        MipmapMode::Nearest,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        0.0,
        1.0,
        0.0,
        0.0,
    )
    .unwrap();

    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

    let pipeline = GraphicsPipeline::start()
        .vertex_input_single_buffer::<Vertex>()
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::TriangleStrip))
        .viewport_state(ViewportState::FixedScissor {
            scissors: (0..1).map(|_| Scissor::irrelevant()).collect(),
            viewport_count_dynamic: false,
        })
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .color_blend_state(ColorBlendState::new(1).blend_alpha())
        .render_pass(subpass)
        .build(device.clone())
        .unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };
    let framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);

    (
        device,
        instance,
        swapchain,
        surface,
        event_loop,
        viewport,
        queue,
        render_pass,
        framebuffers,
        sampler,
        pipeline,
        vertex_buffer,
    )
}

fn build_display<F>(ctx: glutin::Context<glutin::NotCurrent>, f: F)
where
    F: FnOnce(Box<dyn glium::backend::Facade>),
    F: Send + 'static,
{
    std::thread::spawn(move || {
        let display = Box::new(
            glium::HeadlessRenderer::with_debug(ctx, glium::debug::DebugCallbackBehavior::PrintAll)
                .unwrap(),
        );

        f(display);
    });
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    use vulkano::image::ImageAccess;
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    images
        .iter()
        .map(|image| -> Arc<Framebuffer> {
            let view = ImageView::new(image.clone()).unwrap();

            Framebuffer::start(render_pass.clone())
                .add(view)
                .unwrap()
                .build()
                .unwrap()
        })
        .collect::<Vec<_>>()
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450
layout(location = 0) in vec2 position;
layout(location = 0) out vec2 tex_coords;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    tex_coords = position + vec2(0.5);
}"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450
layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;
layout(set = 0, binding = 0) uniform sampler2D tex;
void main() {
    f_color = texture(tex, tex_coords);
}"
    }
}
