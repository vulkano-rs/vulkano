fn main() {
    #[cfg(target_os = "linux")]
    linux::main();
    #[cfg(not(target_os = "linux"))]
    println!("Not Implemented");
}

// TODO: Can this be demonstrated for other platforms as well?
#[cfg(target_os = "linux")]
mod linux {
    use bytemuck::{Pod, Zeroable};
    use glium::glutin::{self, platform::unix::HeadlessContextExt};
    use std::{
        sync::{Arc, Barrier},
        time::Instant,
    };
    use vulkano::{
        buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess},
        command_buffer::{
            AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SemaphoreSubmitInfo,
            SubmitInfo, SubpassContents,
        },
        descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
        device::{
            physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
            QueueCreateInfo,
        },
        format::Format,
        image::{view::ImageView, ImageCreateFlags, ImageUsage, StorageImage, SwapchainImage},
        impl_vertex,
        instance::{
            debug::{DebugUtilsMessenger, DebugUtilsMessengerCreateInfo},
            Instance, InstanceCreateInfo, InstanceExtensions,
        },
        pipeline::{
            graphics::{
                color_blend::ColorBlendState,
                input_assembly::{InputAssemblyState, PrimitiveTopology},
                vertex_input::BuffersDefinition,
                viewport::{Scissor, Viewport, ViewportState},
            },
            GraphicsPipeline, Pipeline, PipelineBindPoint,
        },
        render_pass::{Framebuffer, RenderPass, Subpass},
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        swapchain::{
            AcquireError, Swapchain, SwapchainAbstract, SwapchainCreateInfo,
            SwapchainCreationError, SwapchainPresentInfo,
        },
        sync::{
            now, ExternalSemaphoreHandleType, ExternalSemaphoreHandleTypes, FlushError, GpuFuture,
            Semaphore, SemaphoreCreateInfo,
        },
        VulkanLibrary,
    };
    use vulkano_win::VkSurfaceBuild;
    use winit::{
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::{Window, WindowBuilder},
    };

    pub fn main() {
        let event_loop = EventLoop::new();
        // For some reason, this must be created before the vulkan window
        let hrb = glutin::ContextBuilder::new()
            .with_gl_debug_flag(true)
            .with_gl(glutin::GlRequest::Latest)
            .build_surfaceless(&event_loop)
            .unwrap();

        let hrb_vk = glutin::ContextBuilder::new()
            .with_gl_debug_flag(true)
            .with_gl(glutin::GlRequest::Latest)
            .build_surfaceless(&event_loop)
            .unwrap();

        let display = glium::HeadlessRenderer::with_debug(
            hrb_vk,
            glium::debug::DebugCallbackBehavior::PrintAll,
        )
        .unwrap(); // Used for checking device and driver UUIDs
        let (
            device,
            _instance,
            mut swapchain,
            surface,
            mut viewport,
            queue,
            render_pass,
            mut framebuffers,
            sampler,
            pipeline,
            vertex_buffer,
        ) = vk_setup(display, &event_loop);

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
                transfer_src: true,
                transfer_dst: true,
                ..ImageUsage::empty()
            },
            ImageCreateFlags {
                mutable_format: true,
                ..ImageCreateFlags::empty()
            },
            [queue.queue_family_index()],
        )
        .unwrap();

        let image_fd = image.export_posix_fd().unwrap();

        let image_view = ImageView::new_default(image.clone()).unwrap();

        let barrier = Arc::new(Barrier::new(2));
        let barrier_2 = Arc::new(Barrier::new(2));

        let acquire_sem = Arc::new(
            Semaphore::new(
                device.clone(),
                SemaphoreCreateInfo {
                    export_handle_types: ExternalSemaphoreHandleTypes {
                        opaque_fd: true,
                        ..ExternalSemaphoreHandleTypes::empty()
                    },
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let release_sem = Arc::new(
            Semaphore::new(
                device.clone(),
                SemaphoreCreateInfo {
                    export_handle_types: ExternalSemaphoreHandleTypes {
                        opaque_fd: true,
                        ..ExternalSemaphoreHandleTypes::empty()
                    },
                    ..Default::default()
                },
            )
            .unwrap(),
        );

        let acquire_fd = unsafe {
            acquire_sem
                .export_fd(ExternalSemaphoreHandleType::OpaqueFd)
                .unwrap()
        };
        let release_fd = unsafe {
            release_sem
                .export_fd(ExternalSemaphoreHandleType::OpaqueFd)
                .unwrap()
        };

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

        let layout = pipeline.layout().set_layouts().get(0).unwrap();

        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::image_view_sampler(
                0, image_view, sampler,
            )],
        )
        .unwrap();

        let mut recreate_swapchain = false;
        let mut previous_frame_end: Option<Box<dyn GpuFuture>> =
            Some(Box::new(now(device.clone())));

        event_loop.run(move |event, _, control_flow| {
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
                Event::RedrawEventsCleared => {
                    queue
                        .with(|mut q| unsafe {
                            q.submit_unchecked(
                                [SubmitInfo {
                                    signal_semaphores: vec![SemaphoreSubmitInfo::semaphore(
                                        acquire_sem.clone(),
                                    )],
                                    ..Default::default()
                                }],
                                None,
                            )
                        })
                        .unwrap();

                    barrier.wait();
                    barrier_2.wait();

                    queue
                        .with(|mut q| unsafe {
                            q.submit_unchecked(
                                [SubmitInfo {
                                    wait_semaphores: vec![SemaphoreSubmitInfo::semaphore(
                                        release_sem.clone(),
                                    )],
                                    ..Default::default()
                                }],
                                None,
                            )
                        })
                        .unwrap();

                    previous_frame_end.as_mut().unwrap().cleanup_finished();

                    if recreate_swapchain {
                        let (new_swapchain, new_images) =
                            match swapchain.recreate(SwapchainCreateInfo {
                                image_extent: surface.window().inner_size().into(),
                                ..swapchain.create_info()
                            }) {
                                Ok(r) => r,
                                Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => {
                                    return
                                }
                                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                            };

                        swapchain = new_swapchain;
                        framebuffers = window_size_dependent_setup(
                            &new_images,
                            render_pass.clone(),
                            &mut viewport,
                        );
                        recreate_swapchain = false;
                    }

                    let (image_index, suboptimal, acquire_future) =
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

                    let mut builder = AutoCommandBufferBuilder::primary(
                        device.clone(),
                        queue.queue_family_index(),
                        CommandBufferUsage::OneTimeSubmit,
                    )
                    .unwrap();
                    builder
                        .begin_render_pass(
                            RenderPassBeginInfo {
                                clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                                ..RenderPassBeginInfo::framebuffer(
                                    framebuffers[image_index as usize].clone(),
                                )
                            },
                            SubpassContents::Inline,
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

                    let future = future
                        .then_execute(queue.clone(), command_buffer)
                        .unwrap()
                        .then_swapchain_present(
                            queue.clone(),
                            SwapchainPresentInfo::swapchain_image_index(
                                swapchain.clone(),
                                image_index,
                            ),
                        )
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
                }

                _ => (),
            };
        });
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
    struct Vertex {
        position: [f32; 2],
    }
    impl_vertex!(Vertex, position);

    #[allow(clippy::type_complexity)]
    fn vk_setup(
        display: glium::HeadlessRenderer,
        event_loop: &EventLoop<()>,
    ) -> (
        Arc<vulkano::device::Device>,
        Arc<vulkano::instance::Instance>,
        Arc<Swapchain<winit::window::Window>>,
        Arc<vulkano::swapchain::Surface<winit::window::Window>>,
        vulkano::pipeline::graphics::viewport::Viewport,
        Arc<Queue>,
        Arc<RenderPass>,
        Vec<Arc<Framebuffer>>,
        Arc<vulkano::sampler::Sampler>,
        Arc<GraphicsPipeline>,
        Arc<CpuAccessibleBuffer<[Vertex]>>,
    ) {
        let library = VulkanLibrary::new().unwrap();
        let required_extensions = vulkano_win::required_extensions(&library);
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: InstanceExtensions {
                    khr_get_physical_device_properties2: true,
                    khr_external_memory_capabilities: true,
                    khr_external_semaphore_capabilities: true,
                    khr_external_fence_capabilities: true,
                    ext_debug_utils: true,

                    ..InstanceExtensions::empty()
                }
                .union(&required_extensions),

                // Enable enumerating devices that use non-conformant vulkan implementations. (ex. MoltenVK)
                enumerate_portability: true,

                ..Default::default()
            },
        )
        .unwrap();

        let _debug_callback = unsafe {
            DebugUtilsMessenger::new(
                instance.clone(),
                DebugUtilsMessengerCreateInfo::user_callback(Arc::new(|msg| {
                    println!(
                        "{} {:?} {:?}: {}",
                        msg.layer_prefix.unwrap_or("unknown"),
                        msg.ty,
                        msg.severity,
                        msg.description
                    );
                })),
            )
            .unwrap()
        };

        let surface = WindowBuilder::new()
            .build_vk_surface(event_loop, instance.clone())
            .unwrap();

        let device_extensions = DeviceExtensions {
            khr_external_semaphore: true,
            khr_external_semaphore_fd: true,
            khr_external_memory: true,
            khr_external_memory_fd: true,
            khr_external_fence: true,
            khr_external_fence_fd: true,
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.graphics
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .filter(|(p, _)| p.properties().driver_uuid.unwrap() == display.driver_uuid().unwrap())
            .filter(|(p, _)| {
                display
                    .device_uuids()
                    .unwrap()
                    .contains(&p.properties().device_uuid.unwrap())
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
            physical_device.properties().device_type,
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

        let (swapchain, images) = {
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();
            let image_format = Some(
                device
                    .physical_device()
                    .surface_formats(&surface, Default::default())
                    .unwrap()[0]
                    .0,
            );

            Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count,
                    image_format,
                    image_extent: surface.window().inner_size().into(),
                    image_usage: ImageUsage {
                        color_attachment: true,
                        ..ImageUsage::empty()
                    },
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let vertices = [
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
        ];
        let vertex_buffer = CpuAccessibleBuffer::<[Vertex]>::from_iter(
            device.clone(),
            BufferUsage {
                vertex_buffer: true,
                ..BufferUsage::empty()
            },
            false,
            vertices,
        )
        .unwrap();

        let vs = vs::load(device.clone()).unwrap();
        let fs = fs::load(device.clone()).unwrap();

        let render_pass = vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.image_format(),
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
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .input_assembly_state(
                InputAssemblyState::new().topology(PrimitiveTopology::TriangleStrip),
            )
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
                glium::HeadlessRenderer::with_debug(
                    ctx,
                    glium::debug::DebugCallbackBehavior::PrintAll,
                )
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
        use vulkano::{image::ImageAccess, render_pass::FramebufferCreateInfo};
        let dimensions = images[0].dimensions().width_height();
        viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

        images
            .iter()
            .map(|image| -> Arc<Framebuffer> {
                let view = ImageView::new_default(image.clone()).unwrap();

                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                )
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
}
