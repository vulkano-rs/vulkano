fn main() -> Result<(), winit::error::EventLoopError> {
    #[cfg(target_os = "linux")]
    {
        linux::main()
    }
    #[cfg(not(target_os = "linux"))]
    {
        Ok(println!("Not Implemented"))
    }
}

// TODO: Can this be demonstrated for other platforms as well?
#[cfg(target_os = "linux")]
mod linux {
    use glium::glutin::{self, platform::unix::HeadlessContextExt};
    use std::{
        sync::{Arc, Barrier},
        time::Instant,
    };
    use vulkano::{
        buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
        command_buffer::{
            allocator::StandardCommandBufferAllocator, CommandBufferBeginInfo, CommandBufferLevel,
            CommandBufferUsage, RecordingCommandBuffer, RenderPassBeginInfo, SemaphoreSubmitInfo,
            SubmitInfo,
        },
        descriptor_set::{
            allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
        },
        device::{
            physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
            QueueCreateInfo, QueueFlags,
        },
        format::Format,
        image::{
            sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
            sys::RawImage,
            view::ImageView,
            Image, ImageCreateFlags, ImageCreateInfo, ImageType, ImageUsage,
        },
        instance::{
            debug::{
                DebugUtilsMessenger, DebugUtilsMessengerCallback, DebugUtilsMessengerCreateInfo,
            },
            Instance, InstanceCreateFlags, InstanceCreateInfo, InstanceExtensions,
        },
        memory::{
            allocator::{
                AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
            },
            DedicatedAllocation, DeviceMemory, ExternalMemoryHandleType, ExternalMemoryHandleTypes,
            MemoryAllocateInfo, ResourceMemory,
        },
        pipeline::{
            graphics::{
                color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState},
                input_assembly::{InputAssemblyState, PrimitiveTopology},
                multisample::MultisampleState,
                rasterization::RasterizationState,
                vertex_input::{Vertex, VertexDefinition},
                viewport::{Viewport, ViewportState},
                GraphicsPipelineCreateInfo,
            },
            layout::PipelineDescriptorSetLayoutCreateInfo,
            DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
            PipelineShaderStageCreateInfo,
        },
        render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
        swapchain::{
            acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
        },
        sync::{
            now,
            semaphore::{
                ExternalSemaphoreHandleType, ExternalSemaphoreHandleTypes, Semaphore,
                SemaphoreCreateInfo,
            },
            GpuFuture,
        },
        Validated, VulkanError, VulkanLibrary,
    };
    use winit::{
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::{Window, WindowBuilder},
    };

    pub fn main() -> Result<(), winit::error::EventLoopError> {
        let event_loop_gl = winit_glium::event_loop::EventLoop::new();
        // For some reason, this must be created before the vulkan window
        let hrb = glutin::ContextBuilder::new()
            .with_gl_debug_flag(true)
            .with_gl(glutin::GlRequest::Latest)
            .build_surfaceless(&event_loop_gl)
            .unwrap();

        let hrb_vk = glutin::ContextBuilder::new()
            .with_gl_debug_flag(true)
            .with_gl(glutin::GlRequest::Latest)
            .build_surfaceless(&event_loop_gl)
            .unwrap();

        // Used for checking device and driver UUIDs.
        let display = glium::HeadlessRenderer::with_debug(
            hrb_vk,
            glium::debug::DebugCallbackBehavior::PrintAll,
        )
        .unwrap();

        let event_loop = EventLoop::new().unwrap();
        let (
            device,
            _instance,
            mut swapchain,
            window,
            mut viewport,
            queue,
            render_pass,
            mut framebuffers,
            sampler,
            pipeline,
            memory_allocator,
            vertex_buffer,
        ) = vk_setup(display, &event_loop);

        let raw_image = RawImage::new(
            device.clone(),
            ImageCreateInfo {
                flags: ImageCreateFlags::MUTABLE_FORMAT,
                image_type: ImageType::Dim2d,
                format: Format::R16G16B16A16_UNORM,
                extent: [200, 200, 1],
                usage: ImageUsage::TRANSFER_SRC | ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                external_memory_handle_types: ExternalMemoryHandleTypes::OPAQUE_FD,
                ..Default::default()
            },
        )
        .unwrap();

        let image_requirements = raw_image.memory_requirements()[0];

        let image_memory = DeviceMemory::allocate(
            device.clone(),
            MemoryAllocateInfo {
                allocation_size: image_requirements.layout.size(),
                memory_type_index: memory_allocator
                    .find_memory_type_index(
                        image_requirements.memory_type_bits,
                        MemoryTypeFilter::PREFER_DEVICE,
                    )
                    .unwrap(),
                dedicated_allocation: Some(DedicatedAllocation::Image(&raw_image)),
                export_handle_types: ExternalMemoryHandleTypes::OPAQUE_FD,
                ..Default::default()
            },
        )
        .unwrap();

        let allocation_size = image_memory.allocation_size();
        let image_fd = image_memory
            .export_fd(ExternalMemoryHandleType::OpaqueFd)
            .unwrap();

        // SAFETY: we just created this raw image and hasn't bound any memory to it.
        let image = Arc::new(unsafe {
            raw_image
                .bind_memory([ResourceMemory::new_dedicated(image_memory)])
                .map_err(|(err, _, _)| err)
                .unwrap()
        });

        let image_view = ImageView::new_default(image).unwrap();

        let barrier = Arc::new(Barrier::new(2));
        let barrier_2 = Arc::new(Barrier::new(2));

        let acquire_sem = Arc::new(
            Semaphore::new(
                device.clone(),
                SemaphoreCreateInfo {
                    export_handle_types: ExternalSemaphoreHandleTypes::OPAQUE_FD,
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let release_sem = Arc::new(
            Semaphore::new(
                device.clone(),
                SemaphoreCreateInfo {
                    export_handle_types: ExternalSemaphoreHandleTypes::OPAQUE_FD,
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
                        size: allocation_size,
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

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let layout = &pipeline.layout().set_layouts()[0];

        let set = DescriptorSet::new(
            descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::sampler(0, sampler),
                WriteDescriptorSet::image_view(1, image_view),
            ],
            [],
        )
        .unwrap();

        let mut recreate_swapchain = false;
        let mut previous_frame_end: Option<Box<dyn GpuFuture>> =
            Some(Box::new(now(device.clone())));

        event_loop.run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    elwt.exit();
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(_),
                    ..
                } => {
                    recreate_swapchain = true;
                }
                Event::WindowEvent {
                    event: WindowEvent::RedrawRequested,
                    ..
                } => {
                    queue
                        .with(|mut q| unsafe {
                            q.submit(
                                &[SubmitInfo {
                                    signal_semaphores: vec![SemaphoreSubmitInfo::new(
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
                            q.submit(
                                &[SubmitInfo {
                                    wait_semaphores: vec![SemaphoreSubmitInfo::new(
                                        release_sem.clone(),
                                    )],
                                    ..Default::default()
                                }],
                                None,
                            )
                        })
                        .unwrap();

                    let image_extent: [u32; 2] = window.inner_size().into();

                    if image_extent.contains(&0) {
                        return;
                    }

                    previous_frame_end.as_mut().unwrap().cleanup_finished();

                    if recreate_swapchain {
                        let (new_swapchain, new_images) = swapchain
                            .recreate(SwapchainCreateInfo {
                                image_extent,
                                ..swapchain.create_info()
                            })
                            .expect("failed to recreate swapchain");

                        swapchain = new_swapchain;
                        framebuffers = window_size_dependent_setup(
                            &new_images,
                            render_pass.clone(),
                            &mut viewport,
                        );
                        recreate_swapchain = false;
                    }

                    let (image_index, suboptimal, acquire_future) = match acquire_next_image(
                        swapchain.clone(),
                        None,
                    )
                    .map_err(Validated::unwrap)
                    {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };

                    if suboptimal {
                        recreate_swapchain = true;
                    }

                    let mut builder = RecordingCommandBuffer::new(
                        command_buffer_allocator.clone(),
                        queue.queue_family_index(),
                        CommandBufferLevel::Primary,
                        CommandBufferBeginInfo {
                            usage: CommandBufferUsage::OneTimeSubmit,
                            ..Default::default()
                        },
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
                            Default::default(),
                        )
                        .unwrap()
                        .set_viewport(0, [viewport.clone()].into_iter().collect())
                        .unwrap()
                        .bind_pipeline_graphics(pipeline.clone())
                        .unwrap()
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            pipeline.layout().clone(),
                            0,
                            set.clone(),
                        )
                        .unwrap()
                        .bind_vertex_buffers(0, vertex_buffer.clone())
                        .unwrap();

                    unsafe {
                        builder.draw(vertex_buffer.len() as u32, 1, 0, 0).unwrap();
                    }

                    builder.end_render_pass(Default::default()).unwrap();

                    let command_buffer = builder.end().unwrap();
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

                    match future.map_err(Validated::unwrap) {
                        Ok(future) => {
                            future.wait(None).unwrap();
                            previous_frame_end = Some(future.boxed());
                        }
                        Err(VulkanError::OutOfDate) => {
                            recreate_swapchain = true;
                            previous_frame_end = Some(now(device.clone()).boxed());
                        }
                        Err(e) => {
                            println!("failed to flush future: {e}");
                            previous_frame_end = Some(now(device.clone()).boxed());
                        }
                    };
                }
                Event::AboutToWait => window.request_redraw(),
                _ => (),
            };
        })
    }

    #[derive(BufferContents, Vertex)]
    #[repr(C)]
    struct MyVertex {
        #[format(R32G32_SFLOAT)]
        position: [f32; 2],
    }

    #[allow(clippy::type_complexity)]
    fn vk_setup(
        display: glium::HeadlessRenderer,
        event_loop: &EventLoop<()>,
    ) -> (
        Arc<Device>,
        Arc<Instance>,
        Arc<Swapchain>,
        Arc<Window>,
        Viewport,
        Arc<Queue>,
        Arc<RenderPass>,
        Vec<Arc<Framebuffer>>,
        Arc<Sampler>,
        Arc<GraphicsPipeline>,
        Arc<StandardMemoryAllocator>,
        Subbuffer<[MyVertex]>,
    ) {
        let library = VulkanLibrary::new().unwrap();
        let required_extensions = Surface::required_extensions(&event_loop).unwrap();
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: InstanceExtensions {
                    khr_get_physical_device_properties2: true,
                    khr_external_memory_capabilities: true,
                    khr_external_semaphore_capabilities: true,
                    khr_external_fence_capabilities: true,
                    ext_debug_utils: true,
                    ..required_extensions
                },
                ..Default::default()
            },
        )
        .unwrap();

        let _debug_callback = unsafe {
            DebugUtilsMessenger::new(
                instance.clone(),
                DebugUtilsMessengerCreateInfo::user_callback(DebugUtilsMessengerCallback::new(
                    |message_severity, message_type, callback_data| {
                        println!(
                            "{} {:?} {:?}: {}",
                            callback_data.message_id_name.unwrap_or("unknown"),
                            message_type,
                            message_severity,
                            callback_data.message,
                        );
                    },
                )),
            )
            .unwrap()
        };

        let window = Arc::new(WindowBuilder::new().build(event_loop).unwrap());
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

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
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
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
            let image_format = device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0;

            Swapchain::new(
                device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let vertices = [
            MyVertex {
                position: [-0.5, -0.5],
            },
            MyVertex {
                position: [-0.5, 0.5],
            },
            MyVertex {
                position: [0.5, -0.5],
            },
            MyVertex {
                position: [0.5, 0.5],
            },
        ];
        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();

        let render_pass = vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
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

        let pipeline = {
            let vs = vs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = fs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();
            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState {
                        topology: PrimitiveTopology::TriangleStrip,
                        ..Default::default()
                    }),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend::alpha()),
                            ..Default::default()
                        },
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        let mut viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [0.0, 0.0],
            depth_range: 0.0..=1.0,
        };
        let framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);

        (
            device,
            instance,
            swapchain,
            window,
            viewport,
            queue,
            render_pass,
            framebuffers,
            sampler,
            pipeline,
            memory_allocator,
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
        images: &[Arc<Image>],
        render_pass: Arc<RenderPass>,
        viewport: &mut Viewport,
    ) -> Vec<Arc<Framebuffer>> {
        let extent = images[0].extent();
        viewport.extent = [extent[0] as f32, extent[1] as f32];

        images
            .iter()
            .map(|image| {
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
            src: r"
                #version 450
                layout(location = 0) in vec2 position;
                layout(location = 0) out vec2 tex_coords;
                void main() {
                    gl_Position = vec4(position, 0.0, 1.0);
                    tex_coords = position + vec2(0.5);
                }
            ",
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: r"
                #version 450
                layout(location = 0) in vec2 tex_coords;
                layout(location = 0) out vec4 f_color;

                layout(set = 0, binding = 0) uniform sampler s;
                layout(set = 0, binding = 1) uniform texture2D tex;

                void main() {
                    f_color = texture(sampler2D(tex, s), tex_coords);
                }
            ",
        }
    }
}
