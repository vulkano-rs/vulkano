// This example showcases how you can most effectively update a resource asynchronously, such that
// your rendering or any other tasks can use the resource without any latency at the same time as
// it's being updated.
//
// There are two kinds of resources that are updated asynchronously here:
//
// - A uniform buffer, which needs to be updated every frame.
// - A large texture, which needs to be updated partially at the request of the user.
//
// For the first, since the data needs to be updated every frame, we have to use one buffer per
// frame in flight. The swapchain most commonly has multiple images that are all processed at the
// same time, therefore writing the same buffer during each frame in flight would result in one of
// two things: either you would have to synchronize the writes from the host and reads from the
// device such that only one of the images in the swapchain is actually processed at any point in
// time (bad), or a race condition (bad). Therefore we are left with no choice but to use a
// different buffer for each frame in flight. This is best suited to very small pieces of data that
// change rapidly, and where the data of one frame doesn't depend on data from a previous one.
//
// For the second, since this texture is rather large, we can't afford to overwrite the entire
// texture every time a part of it needs to be updated. Also, we don't need as many textures as
// there are frames in flight since the texture doesn't need to be updated every frame, but we
// still need at least two textures. That way we can write one of the textures at the same time as
// reading the other, swapping them after the write is done such that the newly updated one is read
// and the now out-of-date one can be written to next time, known as *eventual consistency*.
//
// In an eventually consistent system, a number of *replicas* are used, all of which represent the
// same data but their consistency is not strict. A replica might be out-of-date for some time
// before *reaching convergence*, hence becoming consistent, eventually.

use glam::f32::Mat4;
use rand::Rng;
use std::{
    error::Error,
    hint,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc, Arc,
    },
    thread,
    time::{SystemTime, UNIX_EPOCH},
};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, BufferImageCopy, ClearColorImageInfo,
        CommandBufferBeginInfo, CommandBufferLevel, CommandBufferUsage, CopyBufferToImageInfo,
        RecordingCommandBuffer, RenderPassBeginInfo,
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
        sampler::{Sampler, SamplerCreateInfo},
        view::ImageView,
        Image, ImageCreateInfo, ImageType, ImageUsage,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
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
    sync::{self, GpuFuture},
    Validated, VulkanError, VulkanLibrary,
};
use winit::{
    event::{ElementState, Event, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::WindowBuilder,
};

const TRANSFER_GRANULARITY: u32 = 4096;

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();

    let library = VulkanLibrary::new().unwrap();
    let required_extensions = Surface::required_extensions(&event_loop).unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .unwrap();

    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
    let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };
    let (physical_device, graphics_family_index) = instance
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

    // Since we are going to be updating the texture on a separate thread asynchronously from the
    // execution of graphics commands, it would make sense to also do the transfer on a dedicated
    // transfer queue, if such a queue family exists. That way, the graphics queue is not blocked
    // during the transfers either and the two tasks are truly asynchronous.
    //
    // For this, we need to find the queue family with the fewest queue flags set, since if the
    // queue fmaily has more flags than `TRANSFER | SPARSE_BINDING`, that means it is not dedicated
    // to transfer operations.
    let transfer_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .filter(|(_, q)| {
            q.queue_flags.intersects(QueueFlags::TRANSFER)
                // Queue familes dedicated to transfers are not required to support partial 
                // transfers of images, reported by a mininum granularity of [0, 0, 0]. If you need 
                // to do partial transfers of images like we do in this example, you therefore have 
                // to make sure the queue family supports that.
                && q.min_image_transfer_granularity != [0; 3]
                // Unlike queue familes for graphics and/or compute, queue familes dedicated to
                // transfers don't have to support image transfers of arbitrary granularity.
                // Therefore, if you are going to use one, you have to either make sure the
                // granularity is granular enough for your needs, or you have to align your
                // transfer offsets and extents to this granularity. Our minimum granularity is
                // 4096 which should be more than coarse enough so we just check that it is.
                && q.min_image_transfer_granularity[0..2]
                    .iter()
                    .all(|&g| TRANSFER_GRANULARITY % g == 0)
        })
        .min_by_key(|(_, q)| q.queue_flags.count())
        .unwrap()
        .0 as u32;

    let (device, mut queues) = {
        let mut queue_create_infos = vec![QueueCreateInfo {
            queue_family_index: graphics_family_index,
            ..Default::default()
        }];

        // It's possible that the physical device doesn't have any queue familes supporting
        // transfers other than the graphics and/or compute queue family. In that case we must make
        // sure we don't request the same queue family twice.
        if transfer_family_index != graphics_family_index {
            queue_create_infos.push(QueueCreateInfo {
                queue_family_index: transfer_family_index,
                ..Default::default()
            });
        }

        Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos,
                ..Default::default()
            },
        )
        .unwrap()
    };

    let graphics_queue = queues.next().unwrap();

    // If we didn't get a dedicated transfer queue, fall back to the graphics queue for transfers.
    let transfer_queue = queues.next().unwrap_or_else(|| graphics_queue.clone());

    println!(
        "Using queue family {graphics_family_index} for graphics and queue family \
        {transfer_family_index} for transfers",
    );

    let (mut swapchain, images) = {
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

    #[derive(BufferContents, Vertex)]
    #[repr(C)]
    struct MyVertex {
        #[format(R32G32_SFLOAT)]
        position: [f32; 2],
    }

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

    // Create a pool of uniform buffers, one per frame in flight. This way we always have an
    // available buffer to write during each frame while reusing them as much as possible.
    let uniform_buffers = (0..swapchain.image_count())
        .map(|_| {
            Buffer::new_sized(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::UNIFORM_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    // Create two textures, where at any point in time one is used exclusively for reading and one
    // is used exclusively for writing, swapping the two after each update.
    let textures = [(); 2].map(|_| {
        Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R8G8B8A8_UNORM,
                extent: [TRANSFER_GRANULARITY * 2, TRANSFER_GRANULARITY * 2, 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap()
    });

    // The index of the currently most up-to-date texture. The worker thread swaps the index after
    // every finished write, which is always done to the, at that point in time, unused texture.
    let current_texture_index = Arc::new(AtomicBool::new(false));

    // Current generation, used to notify the worker thread of when a texture is no longer read.
    let current_generation = Arc::new(AtomicU64::new(0));

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));

    // Initialize the textures.
    {
        let mut builder = RecordingCommandBuffer::new(
            command_buffer_allocator.clone(),
            graphics_queue.queue_family_index(),
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                ..Default::default()
            },
        )
        .unwrap();
        for texture in &textures {
            builder
                .clear_color_image(ClearColorImageInfo::image(texture.clone()))
                .unwrap();
        }
        let command_buffer = builder.end().unwrap();

        // This waits for the queue to become idle, which is fine for startup initializations.
        let _ = command_buffer.execute(graphics_queue.clone()).unwrap();
    }

    // Start the worker thread.
    let (channel, receiver) = mpsc::channel();
    run_worker(
        receiver,
        transfer_queue,
        textures.clone(),
        current_texture_index.clone(),
        current_generation.clone(),
        swapchain.image_count(),
        memory_allocator,
        command_buffer_allocator.clone(),
    );

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: r"
                #version 450

                layout(location = 0) in vec2 position;
                layout(location = 0) out vec2 tex_coords;

                layout(set = 0, binding = 0) uniform Data {
                    mat4 transform;
                };

                void main() {
                    gl_Position = vec4(transform * vec4(position, 0.0, 1.0));
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

                layout(set = 1, binding = 0) uniform sampler s;
                layout(set = 1, binding = 1) uniform texture2D tex;

                void main() {
                    f_color = texture(sampler2D(tex, s), tex_coords);
                }
            ",
        }
    }

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
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
                    ColorBlendAttachmentState::default(),
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
    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);

    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));

    // A byproduct of always using the same set of uniform buffers is that we can also create one
    // descriptor set for each, reusing them in the same way as the buffers.
    let uniform_buffer_sets = uniform_buffers
        .iter()
        .map(|buffer| {
            DescriptorSet::new(
                descriptor_set_allocator.clone(),
                pipeline.layout().set_layouts()[0].clone(),
                [WriteDescriptorSet::buffer(0, buffer.clone())],
                [],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    // Create the descriptor sets for sampling the textures.
    let sampler = Sampler::new(device.clone(), SamplerCreateInfo::simple_repeat_linear()).unwrap();
    let sampler_sets = textures.map(|texture| {
        DescriptorSet::new(
            descriptor_set_allocator.clone(),
            pipeline.layout().set_layouts()[1].clone(),
            [
                WriteDescriptorSet::sampler(0, sampler.clone()),
                WriteDescriptorSet::image_view(1, ImageView::new_default(texture).unwrap()),
            ],
            [],
        )
        .unwrap()
    });

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    println!("\nPress space to update part of the texture");

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
                event:
                    WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                logical_key: Key::Named(NamedKey::Space),
                                state: ElementState::Released,
                                ..
                            },
                        ..
                    },
                ..
            } => {
                channel.send(()).unwrap();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                let image_extent: [u32; 2] = window.inner_size().into();

                if image_extent.contains(&0) {
                    return;
                }

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

                let (image_index, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None).map_err(Validated::unwrap) {
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
                    graphics_queue.queue_family_index(),
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
                            clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
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
                        (
                            // Bind the uniform buffer designated for this frame.
                            uniform_buffer_sets[image_index as usize].clone(),
                            // Bind the currenly most up-to-date texture.
                            sampler_sets[current_texture_index.load(Ordering::Acquire) as usize]
                                .clone(),
                        ),
                    )
                    .unwrap()
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .unwrap();

                unsafe {
                    builder.draw(vertex_buffer.len() as u32, 1, 0, 0).unwrap();
                }

                builder.end_render_pass(Default::default()).unwrap();

                let command_buffer = builder.end().unwrap();
                acquire_future.wait(None).unwrap();
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                // Write to the uniform buffer designated for this frame. This must happen after
                // waiting for the acquire future and cleaning up, otherwise the buffer is still
                // going to be marked as in use by the device.
                *uniform_buffers[image_index as usize].write().unwrap() = vs::Data {
                    transform: {
                        const DURATION: f64 = 5.0;

                        let elapsed = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs_f64();
                        let remainder = elapsed.rem_euclid(DURATION);
                        let delta = (remainder / DURATION) as f32;
                        let angle = delta * std::f32::consts::PI * 2.0;

                        Mat4::from_rotation_z(angle).to_cols_array_2d()
                    },
                };

                // Increment the generation, signalling that the previous frame has finished. This
                // must be done after waiting on the acquire future, otherwise the oldest frame
                // would still be in flight.
                //
                // NOTE: We are relying on the fact that this thread is the only one doing stores.
                current_generation.fetch_add(1, Ordering::Release);

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(graphics_queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        graphics_queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                    )
                    .then_signal_fence_and_flush();

                match future.map_err(Validated::unwrap) {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(VulkanError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        // previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            Event::AboutToWait => window.request_redraw(),
            _ => (),
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn run_worker(
    channel: mpsc::Receiver<()>,
    transfer_queue: Arc<Queue>,
    textures: [Arc<Image>; 2],
    current_texture_index: Arc<AtomicBool>,
    current_generation: Arc<AtomicU64>,
    swapchain_image_count: u32,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
) {
    thread::spawn(move || {
        const CORNER_OFFSETS: [[u32; 3]; 4] = [
            [0, 0, 0],
            [TRANSFER_GRANULARITY, 0, 0],
            [TRANSFER_GRANULARITY, TRANSFER_GRANULARITY, 0],
            [0, TRANSFER_GRANULARITY, 0],
        ];

        // We are going to be updating one of 4 corners of the texture at any point in time. For
        // that, we will use a staging buffer and initiate a copy. However, since our texture is
        // eventually consistent and there are 2 replicas, that means that every time we update one
        // of the replicas the other replica is going to be behind by one update. Therefore we
        // actually need 2 staging buffers as well: one for the update that happened to the
        // currently up-to-date texture (at `current_index`) and one for the update that is about
        // to happen to the currently out-of-date texture (at `!current_index`), so that we can
        // apply both the current and the upcoming update to the out-of-date texture. Then the
        // out-of-date texture is the current up-to-date texture and vice-versa, cycle repeating.
        let staging_buffers = [(); 2].map(|_| {
            Buffer::from_iter(
                memory_allocator.clone(),
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                (0..TRANSFER_GRANULARITY * TRANSFER_GRANULARITY).map(|_| [0u8; 4]),
            )
            .unwrap()
        });

        let mut current_corner = 0;
        let mut rng = rand::thread_rng();
        let mut last_generation = 0;

        // The worker thread is awakened by sending a signal through the channel. In a real program
        // you would likely send some actual data over the channel, instructing the worker what to
        // do, but our work is hard-coded.
        while let Ok(()) = channel.recv() {
            let current_index = current_texture_index.load(Ordering::Acquire);

            // We simulate some work for the worker to indulge in. In a real program this would
            // likely be some kind of I/O, for example reading from disk (think loading the next
            // level in a level-based game, loading the next chunk of terrain in an open-world
            // game, etc.) or downloading images or other data from the internet.
            //
            // NOTE: The size of these textures is exceedingly large on purpose, so that you can
            // feel that the update is in fact asynchronous due to the latency of the updates while
            // the rendering continues without any.
            let color = [rng.gen(), rng.gen(), rng.gen(), u8::MAX];
            for texel in &mut *staging_buffers[!current_index as usize].write().unwrap() {
                *texel = color;
            }

            // Write to the texture that's currently not in use for rendering.
            let texture = textures[!current_index as usize].clone();

            let mut builder = RecordingCommandBuffer::new(
                command_buffer_allocator.clone(),
                transfer_queue.queue_family_index(),
                CommandBufferLevel::Primary,
                CommandBufferBeginInfo {
                    usage: CommandBufferUsage::OneTimeSubmit,
                    ..Default::default()
                },
            )
            .unwrap();
            builder
                .copy_buffer_to_image(CopyBufferToImageInfo {
                    regions: [BufferImageCopy {
                        image_subresource: texture.subresource_layers(),
                        image_offset: CORNER_OFFSETS[current_corner % 4],
                        image_extent: [TRANSFER_GRANULARITY, TRANSFER_GRANULARITY, 1],
                        ..Default::default()
                    }]
                    .into(),
                    ..CopyBufferToImageInfo::buffer_image(
                        staging_buffers[current_index as usize].clone(),
                        texture.clone(),
                    )
                })
                .unwrap()
                .copy_buffer_to_image(CopyBufferToImageInfo {
                    regions: [BufferImageCopy {
                        image_subresource: texture.subresource_layers(),
                        image_offset: CORNER_OFFSETS[(current_corner + 1) % 4],
                        image_extent: [TRANSFER_GRANULARITY, TRANSFER_GRANULARITY, 1],
                        ..Default::default()
                    }]
                    .into(),
                    ..CopyBufferToImageInfo::buffer_image(
                        staging_buffers[!current_index as usize].clone(),
                        texture,
                    )
                })
                .unwrap();
            let command_buffer = builder.end().unwrap();

            // We swap the texture index to use after a write, but there is no guarantee that other
            // tasks have actually moved on to using the new texture. What could happen then, if
            // the writes being done are quicker than rendering a frame (or any other task reading
            // the same resource), is the following:
            //
            // 1. Task A starts reading texture 0
            // 2. Task B writes texture 1, swapping the index
            // 3. Task B writes texture 0, swapping the index
            // 4. Task A stops reading texture 0
            //
            // This is known as the A/B/A problem. In this case it results in a race condition,
            // since task A (rendering, in our case) is still reading texture 0 while task B (our
            // worker) has already started writing the very same texture.
            //
            // The most common way to solve this issue is using *generations*, also known as
            // *epochs*. A generation is simply a monotonically increasing integer. What exactly
            // one generation represents depends on the application. In our case, one generation
            // passed represents one frame that finished rendering. Knowing this, we can keep track
            // of the generation at the time of swapping the texture index, and ensure that any
            // further write only happens after a generation was reached which makes it impossible
            // for any readers to be stuck on the old index. Here we are simply spinning.
            //
            // NOTE: You could also use the thread for other things in the meantime. Since frames
            // are typically very short though, it would make no sense to do that in this case.
            while current_generation.load(Ordering::Acquire) - last_generation
                < swapchain_image_count as u64
            {
                hint::spin_loop();
            }

            // Execute the transfer, blocking the thread until it finishes.
            //
            // NOTE: You could also use the thread for other things in the meantime.
            command_buffer
                .execute(transfer_queue.clone())
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();

            // Remember the latest generation.
            last_generation = current_generation.load(Ordering::Acquire);

            // Swap the texture used for rendering to the newly updated one.
            //
            // NOTE: We are relying on the fact that this thread is the only one doing stores.
            current_texture_index.store(!current_index, Ordering::Release);

            current_corner += 1;
        }
    });
}

/// This function is called once during initialization, then again whenever the window is resized.
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
