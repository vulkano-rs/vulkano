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
    alloc::Layout,
    error::Error,
    slice,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread,
    time::{SystemTime, UNIX_EPOCH},
};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        sys::RawRecordingCommandBuffer, BufferImageCopy, ClearColorImageInfo,
        CopyBufferToImageInfo, RenderPassBeginInfo,
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
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
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
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    sync::Sharing,
    DeviceSize, Validated, VulkanError, VulkanLibrary,
};
use vulkano_taskgraph::{
    graph::{CompileInfo, ExecuteError, TaskGraph},
    resource::{AccessType, Flight, HostAccessType, ImageLayoutType, Resources},
    resource_map, Id, QueueFamilyType, Task, TaskContext, TaskResult,
};
use winit::{
    event::{ElementState, Event, KeyEvent, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::WindowBuilder,
};

const TRANSFER_GRANULARITY: u32 = 4096;
const MAX_FRAMES_IN_FLIGHT: u32 = 2;

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
    // queue family has more flags than `TRANSFER | SPARSE_BINDING`, that means it is not dedicated
    // to transfer operations.
    let transfer_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .filter(|(_, q)| {
            q.queue_flags.intersects(QueueFlags::TRANSFER)
                // Queue families dedicated to transfers are not required to support partial 
                // transfers of images, reported by a minimum granularity of [0, 0, 0]. If you need 
                // to do partial transfers of images like we do in this example, you therefore have 
                // to make sure the queue family supports that.
                && q.min_image_transfer_granularity != [0; 3]
                // Unlike queue families for graphics and/or compute, queue families dedicated to
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

        // It's possible that the physical device doesn't have any queue families supporting
        // transfers other than the graphics and/or compute queue family. In that case we must make
        // sure we don't request the same queue family twice.
        if transfer_family_index != graphics_family_index {
            queue_create_infos.push(QueueCreateInfo {
                queue_family_index: transfer_family_index,
                ..Default::default()
            });
        } else {
            let queue_family_properties =
                &physical_device.queue_family_properties()[graphics_family_index as usize];

            // Even if we can't get an async transfer queue family, it's still better to use
            // different queues on the same queue family. This way, at least the threads on the
            // host don't have lock the same queue when submitting.
            if queue_family_properties.queue_count > 1 {
                queue_create_infos[0].queues.push(0.5);
            }
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

    let resources = Resources::new(device.clone(), Default::default());

    let graphics_flight_id = resources.create_flight(MAX_FRAMES_IN_FLIGHT).unwrap();
    let transfer_flight_id = resources.create_flight(1).unwrap();

    let swapchain_format = device
        .physical_device()
        .surface_formats(&surface, Default::default())
        .unwrap()[0]
        .0;
    let mut swapchain_id = {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        resources
            .create_swapchain(
                graphics_flight_id,
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(3),
                    image_format: swapchain_format,
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
    let vertex_buffer_id = resources
        .create_buffer(
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            DeviceLayout::from_layout(Layout::for_value(&vertices)).unwrap(),
        )
        .unwrap();

    // Create a pool of uniform buffers, one per frame in flight. This way we always have an
    // available buffer to write during each frame while reusing them as much as possible.
    let uniform_buffer_ids = [(); MAX_FRAMES_IN_FLIGHT as usize].map(|_| {
        resources
            .create_buffer(
                BufferCreateInfo {
                    usage: BufferUsage::UNIFORM_BUFFER,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::from_layout(Layout::new::<vs::Data>()).unwrap(),
            )
            .unwrap()
    });

    let texture_create_info = ImageCreateInfo {
        image_type: ImageType::Dim2d,
        format: Format::R8G8B8A8_UNORM,
        extent: [TRANSFER_GRANULARITY * 2, TRANSFER_GRANULARITY * 2, 1],
        usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
        sharing: if graphics_family_index != transfer_family_index {
            Sharing::Concurrent(
                [graphics_family_index, transfer_family_index]
                    .into_iter()
                    .collect(),
            )
        } else {
            Sharing::Exclusive
        },
        ..Default::default()
    };

    // Create two textures, where at any point in time one is used exclusively for reading and one
    // is used exclusively for writing, swapping the two after each update.
    let texture_ids = [(); 2].map(|_| {
        resources
            .create_image(texture_create_info.clone(), AllocationCreateInfo::default())
            .unwrap()
    });

    // The index of the currently most up-to-date texture. The worker thread swaps the index after
    // every finished write, which is always done to the, at that point in time, unused texture.
    let current_texture_index = Arc::new(AtomicBool::new(false));

    // Initialize the resources.
    unsafe {
        vulkano_taskgraph::execute(
            graphics_queue.clone(),
            resources.clone(),
            graphics_flight_id,
            |cbf, tcx| {
                tcx.write_buffer::<[MyVertex]>(vertex_buffer_id, ..)?
                    .copy_from_slice(&vertices);

                for &texture_id in &texture_ids {
                    let texture = tcx.image(texture_id)?.image();
                    cbf.clear_color_image(&ClearColorImageInfo::image(texture.clone()))?;
                }

                Ok(())
            },
            [(vertex_buffer_id, HostAccessType::Write)],
            [],
            [
                (
                    texture_ids[0],
                    AccessType::ClearTransferWrite,
                    ImageLayoutType::Optimal,
                ),
                (
                    texture_ids[1],
                    AccessType::ClearTransferWrite,
                    ImageLayoutType::Optimal,
                ),
            ],
        )
    }
    .unwrap();

    // Start the worker thread.
    let (channel, receiver) = mpsc::channel();
    run_worker(
        receiver,
        transfer_queue,
        resources.clone(),
        graphics_flight_id,
        transfer_flight_id,
        &texture_create_info,
        texture_ids,
        current_texture_index.clone(),
    );

    let render_pass = vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                format: swapchain_format,
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
    let framebuffers =
        window_size_dependent_setup(&resources, swapchain_id, &render_pass, &mut viewport);

    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));

    // A byproduct of always using the same set of uniform buffers is that we can also create one
    // descriptor set for each, reusing them in the same way as the buffers.
    let uniform_buffer_sets = uniform_buffer_ids.map(|buffer_id| {
        let buffer_state = resources.buffer(buffer_id).unwrap();
        let buffer = buffer_state.buffer();

        DescriptorSet::new(
            descriptor_set_allocator.clone(),
            pipeline.layout().set_layouts()[0].clone(),
            [WriteDescriptorSet::buffer(0, buffer.clone().into())],
            [],
        )
        .unwrap()
    });

    // Create the descriptor sets for sampling the textures.
    let sampler = Sampler::new(device.clone(), SamplerCreateInfo::simple_repeat_linear()).unwrap();
    let sampler_sets = texture_ids.map(|texture_id| {
        let texture_state = resources.image(texture_id).unwrap();
        let texture = texture_state.image();

        DescriptorSet::new(
            descriptor_set_allocator.clone(),
            pipeline.layout().set_layouts()[1].clone(),
            [
                WriteDescriptorSet::sampler(0, sampler.clone()),
                WriteDescriptorSet::image_view(1, ImageView::new_default(texture.clone()).unwrap()),
            ],
            [],
        )
        .unwrap()
    });

    let mut rcx = RenderContext {
        viewport,
        framebuffers,
    };

    let mut task_graph = TaskGraph::new(resources.clone(), 1, 4);

    let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo::default());
    let virtual_texture_id = task_graph.add_image(&texture_create_info);
    let virtual_uniform_buffer_id = task_graph.add_buffer(&BufferCreateInfo::default());

    task_graph.add_host_buffer_access(virtual_uniform_buffer_id, HostAccessType::Write);

    task_graph
        .create_task_node(
            "Render",
            QueueFamilyType::Graphics,
            RenderTask {
                swapchain_id: virtual_swapchain_id,
                vertex_buffer_id,
                current_texture_index: current_texture_index.clone(),
                pipeline: pipeline.clone(),
                uniform_buffer_id: virtual_uniform_buffer_id,
                uniform_buffer_sets: uniform_buffer_sets.clone(),
                sampler_sets: sampler_sets.clone(),
            },
        )
        .image_access(
            virtual_swapchain_id.current_image_id(),
            AccessType::ColorAttachmentWrite,
            ImageLayoutType::Optimal,
        )
        .buffer_access(vertex_buffer_id, AccessType::VertexAttributeRead)
        .image_access(
            virtual_texture_id,
            AccessType::FragmentShaderSampledRead,
            ImageLayoutType::Optimal,
        )
        .buffer_access(
            virtual_uniform_buffer_id,
            AccessType::VertexShaderUniformRead,
        );

    let task_graph = unsafe {
        task_graph.compile(CompileInfo {
            queues: vec![graphics_queue.clone()],
            present_queue: Some(graphics_queue.clone()),
            flight_id: graphics_flight_id,
            ..Default::default()
        })
    }
    .unwrap();

    let mut recreate_swapchain = false;

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

                let flight = resources.flight(graphics_flight_id).unwrap();

                if recreate_swapchain {
                    swapchain_id = resources
                        .recreate_swapchain(swapchain_id, |create_info| SwapchainCreateInfo {
                            image_extent,
                            ..create_info
                        })
                        .expect("failed to recreate swapchain");

                    flight.destroy_objects(rcx.framebuffers.drain(..));

                    rcx.framebuffers = window_size_dependent_setup(
                        &resources,
                        swapchain_id,
                        &render_pass,
                        &mut rcx.viewport,
                    );

                    recreate_swapchain = false;
                }

                let frame_index = flight.current_frame_index();
                let texture_index = current_texture_index.load(Ordering::Relaxed);

                let resource_map = resource_map!(
                    &task_graph,
                    virtual_swapchain_id => swapchain_id,
                    virtual_texture_id => texture_ids[texture_index as usize],
                    virtual_uniform_buffer_id => uniform_buffer_ids[frame_index as usize],
                )
                .unwrap();

                flight.wait(None).unwrap();

                match unsafe {
                    task_graph.execute(resource_map, &rcx, || window.pre_present_notify())
                } {
                    Ok(()) => {}
                    Err(ExecuteError::Swapchain {
                        error: Validated::Error(VulkanError::OutOfDate),
                        ..
                    }) => {
                        recreate_swapchain = true;
                    }
                    Err(e) => {
                        panic!("failed to execute next frame: {e:?}");
                    }
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::LoopExiting => {
                let flight = resources.flight(graphics_flight_id).unwrap();
                flight.destroy_object(pipeline.clone());
                flight.destroy_objects(rcx.framebuffers.drain(..));
                flight.destroy_objects(uniform_buffer_sets.clone());
                flight.destroy_objects(sampler_sets.clone());
            }
            _ => (),
        }
    })
}

#[derive(Clone, Copy, BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

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

struct RenderContext {
    viewport: Viewport,
    framebuffers: Vec<Arc<Framebuffer>>,
}

struct RenderTask {
    swapchain_id: Id<Swapchain>,
    vertex_buffer_id: Id<Buffer>,
    current_texture_index: Arc<AtomicBool>,
    pipeline: Arc<GraphicsPipeline>,
    uniform_buffer_id: Id<Buffer>,
    uniform_buffer_sets: [Arc<DescriptorSet>; MAX_FRAMES_IN_FLIGHT as usize],
    sampler_sets: [Arc<DescriptorSet>; 2],
}

impl Task for RenderTask {
    type World = RenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RawRecordingCommandBuffer,
        tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        let frame_index = tcx.current_frame_index();
        let swapchain_state = tcx.swapchain(self.swapchain_id)?;
        let image_index = swapchain_state.current_image_index().unwrap();
        let vertex_buffer = Subbuffer::from(tcx.buffer(self.vertex_buffer_id)?.buffer().clone());

        // Write to the uniform buffer designated for this frame.
        *tcx.write_buffer(self.uniform_buffer_id, ..)? = vs::Data {
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

        cbf.begin_render_pass(
            &RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(rcx.framebuffers[image_index as usize].clone())
            },
            &Default::default(),
        )?
        .set_viewport(0, slice::from_ref(&rcx.viewport))?
        .bind_pipeline_graphics(&self.pipeline)?
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            self.pipeline.layout(),
            0,
            &[
                // Bind the uniform buffer designated for this frame.
                self.uniform_buffer_sets[frame_index as usize]
                    .clone()
                    .into(),
                // Bind the currently most up-to-date texture.
                self.sampler_sets[self.current_texture_index.load(Ordering::Relaxed) as usize]
                    .clone()
                    .into(),
            ],
        )?
        .bind_vertex_buffers(0, slice::from_ref(&vertex_buffer))?;

        let vertex_count = vertex_buffer.reinterpret_ref::<[MyVertex]>().len();
        unsafe { cbf.draw(vertex_count as u32, 1, 0, 0) }?;

        cbf.end_render_pass(&Default::default())?;

        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn run_worker(
    channel: mpsc::Receiver<()>,
    transfer_queue: Arc<Queue>,
    resources: Arc<Resources>,
    graphics_flight_id: Id<Flight>,
    transfer_flight_id: Id<Flight>,
    texture_create_info: &ImageCreateInfo,
    texture_ids: [Id<Image>; 2],
    current_texture_index: Arc<AtomicBool>,
) {
    // We are going to be updating one of 4 corners of the texture at any point in time. For that,
    // we will use a staging buffer and initiate a copy. However, since our texture is eventually
    // consistent and there are 2 replicas, that means that every time we update one of the
    // replicas the other replica is going to be behind by one update. Therefore we actually need 2
    // staging buffers as well: one for the update that happened to the currently up-to-date
    // texture (at `current_index`) and one for the update that is about to happen to the currently
    // out-of-date texture (at `!current_index`), so that we can apply both the current and the
    // upcoming update to the out-of-date texture. Then the out-of-date texture is the current
    // up-to-date texture and vice-versa, cycle repeating.
    let staging_buffer_ids = [(); 2].map(|_| {
        resources
            .create_buffer(
                BufferCreateInfo {
                    usage: BufferUsage::TRANSFER_SRC,
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_HOST
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::from_size_alignment(
                    TRANSFER_GRANULARITY as DeviceSize * TRANSFER_GRANULARITY as DeviceSize * 4,
                    1,
                )
                .unwrap(),
            )
            .unwrap()
    });

    let mut task_graph = TaskGraph::new(resources.clone(), 1, 3);

    let virtual_front_staging_buffer_id = task_graph.add_buffer(&BufferCreateInfo::default());
    let virtual_back_staging_buffer_id = task_graph.add_buffer(&BufferCreateInfo::default());
    let virtual_texture_id = task_graph.add_image(texture_create_info);

    task_graph.add_host_buffer_access(virtual_front_staging_buffer_id, HostAccessType::Write);

    task_graph
        .create_task_node(
            "Image Upload",
            QueueFamilyType::Transfer,
            UploadTask {
                front_staging_buffer_id: virtual_front_staging_buffer_id,
                back_staging_buffer_id: virtual_back_staging_buffer_id,
                texture_id: virtual_texture_id,
            },
        )
        .buffer_access(
            virtual_front_staging_buffer_id,
            AccessType::CopyTransferRead,
        )
        .buffer_access(virtual_back_staging_buffer_id, AccessType::CopyTransferRead)
        .image_access(
            virtual_texture_id,
            AccessType::CopyTransferWrite,
            ImageLayoutType::Optimal,
        );

    let task_graph = unsafe {
        task_graph.compile(CompileInfo {
            queues: vec![transfer_queue],
            flight_id: transfer_flight_id,
            ..Default::default()
        })
    }
    .unwrap();

    thread::spawn(move || {
        let mut current_corner = 0;
        let mut last_frame = 0;

        // The worker thread is awakened by sending a signal through the channel. In a real program
        // you would likely send some actual data over the channel, instructing the worker what to
        // do, but our work is hard-coded.
        while let Ok(()) = channel.recv() {
            let graphics_flight = resources.flight(graphics_flight_id).unwrap();

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
            // This is known as the A/B/A problem. In this case it results in a data race, since
            // task A (rendering, in our case) is still reading texture 0 while task B (our worker)
            // has already started writing the very same texture.
            //
            // To solve this issue, we keep track of the frame counter before swapping the texture
            // index and ensure that any further write only happens after a frame was reached which
            // makes it impossible for any readers to be stuck on the old index, by waiting on the
            // frame to finish on the rendering thread.
            graphics_flight.wait_for_frame(last_frame, None).unwrap();

            let current_index = current_texture_index.load(Ordering::Relaxed);

            let resource_map = resource_map!(
                &task_graph,
                virtual_front_staging_buffer_id => staging_buffer_ids[current_index as usize],
                virtual_back_staging_buffer_id => staging_buffer_ids[!current_index as usize],
                // Write to the texture that's currently not in use for rendering.
                virtual_texture_id => texture_ids[!current_index as usize],
            )
            .unwrap();

            unsafe { task_graph.execute(resource_map, &current_corner, || {}) }.unwrap();

            // Block the thread until the transfer finishes.
            resources
                .flight(transfer_flight_id)
                .unwrap()
                .wait(None)
                .unwrap();

            last_frame = graphics_flight.current_frame();

            // Swap the texture used for rendering to the newly updated one.
            //
            // NOTE: We are relying on the fact that this thread is the only one doing stores.
            current_texture_index.store(!current_index, Ordering::Relaxed);

            current_corner += 1;
        }
    });
}

struct UploadTask {
    front_staging_buffer_id: Id<Buffer>,
    back_staging_buffer_id: Id<Buffer>,
    texture_id: Id<Image>,
}

impl Task for UploadTask {
    type World = usize;

    unsafe fn execute(
        &self,
        cbf: &mut RawRecordingCommandBuffer,
        tcx: &mut TaskContext<'_>,
        &current_corner: &Self::World,
    ) -> TaskResult {
        const CORNER_OFFSETS: [[u32; 3]; 4] = [
            [0, 0, 0],
            [TRANSFER_GRANULARITY, 0, 0],
            [TRANSFER_GRANULARITY, TRANSFER_GRANULARITY, 0],
            [0, TRANSFER_GRANULARITY, 0],
        ];

        let mut rng = rand::thread_rng();

        // We simulate some work for the worker to indulge in. In a real program this would likely
        // be some kind of I/O, for example reading from disk (think loading the next level in a
        // level-based game, loading the next chunk of terrain in an open-world game, etc.) or
        // downloading images or other data from the internet.
        //
        // NOTE: The size of these textures is exceedingly large on purpose, so that you can feel
        // that the update is in fact asynchronous due to the latency of the updates while the
        // rendering continues without any.
        let color = [rng.gen(), rng.gen(), rng.gen(), u8::MAX];
        tcx.write_buffer::<[_]>(self.front_staging_buffer_id, ..)?
            .fill(color);

        let texture = tcx.image(self.texture_id)?.image();

        cbf.copy_buffer_to_image(&CopyBufferToImageInfo {
            regions: [BufferImageCopy {
                image_subresource: texture.subresource_layers(),
                image_offset: CORNER_OFFSETS[current_corner % 4],
                image_extent: [TRANSFER_GRANULARITY, TRANSFER_GRANULARITY, 1],
                ..Default::default()
            }]
            .into(),
            ..CopyBufferToImageInfo::buffer_image(
                tcx.buffer(self.front_staging_buffer_id)?
                    .buffer()
                    .clone()
                    .into(),
                texture.clone(),
            )
        })?;

        if current_corner > 0 {
            cbf.copy_buffer_to_image(&CopyBufferToImageInfo {
                regions: [BufferImageCopy {
                    image_subresource: texture.subresource_layers(),
                    image_offset: CORNER_OFFSETS[(current_corner - 1) % 4],
                    image_extent: [TRANSFER_GRANULARITY, TRANSFER_GRANULARITY, 1],
                    ..Default::default()
                }]
                .into(),
                ..CopyBufferToImageInfo::buffer_image(
                    tcx.buffer(self.back_staging_buffer_id)?
                        .buffer()
                        .clone()
                        .into(),
                    texture.clone(),
                )
            })?;
        }

        Ok(())
    }
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    resources: &Resources,
    swapchain_id: Id<Swapchain>,
    render_pass: &Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let swapchain_state = resources.swapchain(swapchain_id).unwrap();
    let images = swapchain_state.images();
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
