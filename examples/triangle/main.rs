// Welcome to the triangle example!
//
// This is the only example that is entirely detailed. All the other examples avoid code
// duplication by using helper functions.
//
// This example assumes that you are already more or less familiar with graphics programming and
// that you want to learn Vulkan. This means that for example it won't go into details about what a
// vertex or a shader is.

use std::{error::Error, slice, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo, QueueFlags,
    },
    image::ImageUsage,
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::Subpass,
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    VulkanError, VulkanLibrary,
};
use vulkano_taskgraph::{
    command_buffer::RecordingCommandBuffer,
    graph::{AttachmentInfo, CompileInfo, ExecutableTaskGraph, ExecuteError, TaskGraph},
    resource::{AccessTypes, Flight, HostAccessType, ImageLayoutType, Resources},
    resource_map, ClearValues, Id, QueueFamilyType, Task, TaskContext,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

const MAX_FRAMES_IN_FLIGHT: u32 = 2;
const MIN_SWAPCHAIN_IMAGES: u32 = MAX_FRAMES_IN_FLIGHT + 1;

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    resources: Arc<Resources>,
    flight_id: Id<Flight>,
    rcx: Option<RenderContext>,
}

struct RenderContext {
    window: Arc<Window>,
    swapchain_id: Id<Swapchain>,
    viewport: Viewport,
    recreate_swapchain: bool,
    task_graph: ExecutableTaskGraph<Self>,
    virtual_swapchain_id: Id<Swapchain>,
}

struct TriangleTask {
    pipeline: Option<Arc<GraphicsPipeline>>,
    vertex_buffer_id: Id<Buffer>,
    swapchain_id: Id<Swapchain>,
}

impl TriangleTask {
    fn new(app: &mut App, swapchain_id: Id<Swapchain>) -> Self {
        // We now create a buffer that will store the shape of our triangle.
        let vertices = [
            MyVertex {
                position: [-0.5, -0.25],
            },
            MyVertex {
                position: [0.0, 0.5],
            },
            MyVertex {
                position: [0.25, -0.1],
            },
        ];

        let vertex_buffer_id = app
            .resources
            .create_buffer(
                &BufferCreateInfo {
                    usage: BufferUsage::VERTEX_BUFFER,
                    ..Default::default()
                },
                &AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                DeviceLayout::for_value(vertices.as_slice()).unwrap(),
            )
            .unwrap();

        app.resources
            .flight(app.flight_id)
            .unwrap()
            .wait(None)
            .unwrap();

        unsafe {
            vulkano_taskgraph::execute(
                &app.queue,
                &app.resources,
                app.flight_id,
                |_cbf, tcx| {
                    tcx.write_buffer::<[MyVertex]>(vertex_buffer_id, ..)?
                        .copy_from_slice(&vertices);

                    Ok(())
                },
                [(vertex_buffer_id, HostAccessType::Write)],
                [],
                [],
            )
        }
        .unwrap();

        Self {
            pipeline: None,
            vertex_buffer_id,
            swapchain_id,
        }
    }

    pub fn create_pipeline(&mut self, app: &App, subpass: &Subpass) {
        // The next step is to create the shaders.
        //
        // The raw shader creation API provided by the vulkano library is unsafe for various
        // reasons, so The `shader!` macro provides a way to generate a Rust module from GLSL
        // source - in the example below, the source is provided as a string input directly to the
        // shader, but a path to a source file can be provided as well. Note that the user must
        // specify the type of shader (e.g. "vertex", "fragment", etc.) using the `ty` option of
        // the macro.
        //
        // The items generated by the `shader!` macro include a `load` function which loads the
        // shader using an input logical device. The module also includes type definitions for
        // layout structures defined in the shader source, for example uniforms and push constants.
        //
        // A more detailed overview of what the `shader!` macro generates can be found in the
        // vulkano-shaders crate docs. You can view them at https://docs.rs/vulkano-shaders/
        mod vs {
            vulkano_shaders::shader! {
                ty: "vertex",
                src: r"
                    #version 450

                    layout(location = 0) in vec2 position;

                    void main() {
                        gl_Position = vec4(position, 0.0, 1.0);
                    }
                ",
            }
        }

        mod fs {
            vulkano_shaders::shader! {
                ty: "fragment",
                src: r"
                    #version 450

                    layout(location = 0) out vec4 frag_color;

                    void main() {
                        frag_color = vec4(251.0 / 255.0, 113.0 / 255.0, 133.0 / 255.0, 1.0);
                    }
                ",
            }
        }

        // Before we draw, we have to create what is called a **pipeline**. A pipeline describes
        // how a GPU operation is to be performed. It is similar to an OpenGL program, but it also
        // contains many settings for customization, all baked into a single object. For drawing,
        // we create a **graphics** pipeline, but there are also other types of pipeline.
        let pipeline = {
            // First, we load the shaders that the pipeline will use: the vertex shader and the
            // fragment shader.
            //
            // A Vulkan shader can in theory contain multiple entry points, so we have to specify
            // which one.
            let vs = vs::load(&app.device).unwrap().entry_point("main").unwrap();
            let fs = fs::load(&app.device).unwrap().entry_point("main").unwrap();

            // Automatically generate a vertex input state from the vertex shader's input
            // interface, that takes a single vertex buffer containing `Vertex` structs.
            let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();

            // Make a list of the shader stages that the pipeline will have.
            let stages = [
                PipelineShaderStageCreateInfo::new(&vs),
                PipelineShaderStageCreateInfo::new(&fs),
            ];

            // We must now create a **pipeline layout** object, which describes the locations and
            // types of descriptor sets and push constants used by the shaders in the pipeline.
            //
            // Multiple pipelines can share a common layout object, which is more efficient. The
            // shaders in a pipeline must use a subset of the resources described in its pipeline
            // layout, but the pipeline layout is allowed to contain resources that are not present
            // in the shaders; they can be used by shaders in other pipelines that share the same
            // layout. Thus, it is a good idea to design shaders so that many pipelines have common
            // resource locations, which allows them to share pipeline layouts.
            //
            // Since we only have one pipeline in this example, and thus one pipeline layout, we
            // automatically generate the layout from the resources used in the shaders. In a real
            // application, you would specify this information manually so that you can re-use one
            // layout in multiple pipelines.
            let layout = PipelineLayout::from_stages(&app.device, &stages).unwrap();

            // Finally, create the pipeline.
            GraphicsPipeline::new(
                &app.device,
                None,
                &GraphicsPipelineCreateInfo {
                    stages: &stages,
                    // How vertex data is read from the vertex buffers into the vertex shader.
                    vertex_input_state: Some(&vertex_input_state),
                    // How vertices are arranged into primitive shapes. The default primitive shape
                    // is a triangle.
                    input_assembly_state: Some(&InputAssemblyState::default()),
                    // How primitives are transformed and clipped to fit the framebuffer. We use a
                    // resizable viewport, set to draw over the entire window.
                    viewport_state: Some(&ViewportState::default()),
                    // How polygons are culled and converted into a raster of pixels. The default
                    // value does not perform any culling.
                    rasterization_state: Some(&RasterizationState::default()),
                    // How multiple fragment shader samples are converted to a single pixel value.
                    // The default value does not perform any multisampling.
                    multisample_state: Some(&MultisampleState::default()),
                    // How pixel values are combined with the values already present in the
                    // framebuffer. The default value overwrites the old value with the new one,
                    // without any blending.
                    color_blend_state: Some(&ColorBlendState {
                        attachments: &[ColorBlendAttachmentState::default()],
                        ..Default::default()
                    }),
                    // Dynamic states allows us to specify parts of the pipeline settings when
                    // recording the command buffer, before we perform drawing. Here, we specify
                    // that the viewport should be dynamic.
                    dynamic_state: &[DynamicState::Viewport],
                    // We have to indicate which subpass of which render pass this pipeline is
                    // going to be used in. The pipeline will only be usable from this particular
                    // subpass.
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::new(&layout)
                },
            )
            .unwrap()
        };

        self.pipeline = Some(pipeline);
    }
}

impl Task for TriangleTask {
    type World = RenderContext;

    fn clear_values(&self, clear_values: &mut ClearValues<'_>, _world: &Self::World) {
        clear_values.set(
            self.swapchain_id.current_image_id(),
            [2.0 / 255.0, 6.0 / 255.0, 24.0 / 255.0, 1.0],
        );
    }

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> vulkano_taskgraph::TaskResult {
        cbf.set_viewport(0, slice::from_ref(&rcx.viewport))?;
        cbf.bind_pipeline_graphics(self.pipeline.as_ref().unwrap())?;
        cbf.bind_vertex_buffers(0, &[self.vertex_buffer_id], &[0], &[], &[])?;

        unsafe { cbf.draw(3, 1, 0, 0) }?;

        Ok(())
    }
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let library = unsafe { VulkanLibrary::new() }.unwrap();

        // The first step of any Vulkan program is to create an instance.
        //
        // When we create an instance, we have to pass a list of extensions that we want to enable.
        //
        // All the window-drawing functionalities are part of non-core extensions that we need to
        // enable manually. To do so, we ask `Surface` for the list of extensions required to draw
        // to a window.
        let required_extensions = Surface::required_extensions(event_loop).unwrap();

        // Now creating the instance.
        let instance = Instance::new(
            &library,
            &InstanceCreateInfo {
                // Enable enumerating devices that use non-conformant Vulkan implementations.
                // (e.g. MoltenVK)
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: &required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        // Choose device extensions that we're going to use. In order to present images to a
        // surface, we need a `Swapchain`, which is provided by the `khr_swapchain` extension.
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        // We then choose which physical device to use. First, we enumerate all the available
        // physical devices, then apply filters to narrow them down to those that can support our
        // needs.
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| {
                // Some devices may not support the extensions or features that your application,
                // or report properties and limits that are not sufficient for your application.
                // These should be filtered out here.
                p.supported_extensions().contains(&device_extensions)
            })
            .filter_map(|p| {
                // For each physical device, we try to find a suitable queue family that will
                // execute our draw commands.
                //
                // Devices can provide multiple queues to run commands in parallel (for example a
                // draw queue and a compute queue), similar to CPU threads. This is
                // something you have to have to manage manually in Vulkan. Queues
                // of the same type belong to the same queue family.
                //
                // Here, we look for a single queue family that is suitable for our purposes. In a
                // real-world application, you may want to use a separate dedicated transfer queue
                // to handle data transfers in parallel with graphics operations.
                // You may also need a separate queue for compute operations, if
                // your application uses those.
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        // We select a queue family that supports graphics operations. When drawing
                        // to a window surface, as we do in this example, we also need to check
                        // that queues in this queue family are capable of presenting images to the
                        // surface.
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, event_loop).unwrap()
                    })
                    // The code here searches for the first queue family that is suitable. If none
                    // is found, `None` is returned to `filter_map`, which
                    // disqualifies this physical device.
                    .map(|i| (p, i as u32))
            })
            // All the physical devices that pass the filters above are suitable for the
            // application. However, not every device is equal, some are preferred over others.
            // Now, we assign each physical device a score, and pick the device with the lowest
            // ("best") score.
            //
            // In this example, we simply select the best-scoring device to use in the application.
            // In a real-world setting, you may want to use the best-scoring device only as a
            // "default" or "recommended" device, and let the user choose the device themself.
            .min_by_key(|(p, _)| {
                // We assign a lower score to device types that are likely to be faster/better.
                match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                    _ => 5,
                }
            })
            .expect("no suitable physical device found");

        // Some little debug infos.
        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        // Now initializing the device. This is probably the most important object of Vulkan.
        //
        // An iterator of created queues is returned by the function alongside the device.
        let (device, mut queues) = Device::new(
            // Which physical device to connect to.
            &physical_device,
            &DeviceCreateInfo {
                // A list of optional features and extensions that our program needs to work
                // correctly. Some parts of the Vulkan specs are optional and must be enabled
                // manually at device creation. In this example the only thing we are going to need
                // is the `khr_swapchain` extension that allows us to draw to a window.
                enabled_extensions: &device_extensions,

                // The list of queues that we are going to use. Here we only use one queue, from
                // the previously chosen queue family.
                queue_create_infos: &[QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],

                ..Default::default()
            },
        )
        .unwrap();

        // Since we can request multiple queues, the `queues` variable is in fact an iterator. We
        // only use one queue in this example, so we just retrieve the first and only element of
        // the iterator.
        let queue = queues.next().unwrap();

        let resources = Resources::new(&device, &Default::default()).unwrap();

        let flight_id = resources.create_flight(MAX_FRAMES_IN_FLIGHT).unwrap();

        let rcx = None;

        App {
            instance,
            device,
            queue,
            resources,
            flight_id,
            rcx,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // The objective of this example is to draw a triangle on a window. To do so, we first need
        // to create the window. We use the `WindowBuilder` from the `winit` crate to do that here.
        //
        // Before we can render to a window, we must first create a `vulkano::swapchain::Surface`
        // object from it, which represents the drawable surface of a window. For that we must wrap
        // the `winit::window::Window` in an `Arc`.
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );
        let surface = Surface::from_window(&self.instance, &window).unwrap();
        let window_size = window.inner_size();

        // Before we can draw on the surface, we have to create what is called a swapchain.
        // Creating a swapchain allocates the color buffers that will contain the image that will
        // ultimately be visible on the screen. These images are returned alongside the swapchain.
        let swapchain_format;
        let swapchain_id = {
            // Querying the capabilities of the surface. When we create the swapchain we can only
            // pass values that are allowed by the capabilities.
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, &Default::default())
                .unwrap();

            // Choosing the internal format that the images will have.
            (swapchain_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, &Default::default())
                .unwrap()[0];

            // Please take a look at the docs for the meaning of the parameters we didn't mention.
            self.resources
                .create_swapchain(
                    &surface,
                    &SwapchainCreateInfo {
                        // Some drivers report an `min_image_count` of 1, but fullscreen mode
                        // requires at least 2. Therefore we must ensure the count is at least 2,
                        // otherwise the program would crash when entering fullscreen mode on those
                        // drivers.
                        min_image_count: surface_capabilities
                            .min_image_count
                            .max(MIN_SWAPCHAIN_IMAGES),
                        image_format: swapchain_format,

                        // The size of the window, only used to initially setup the swapchain.
                        //
                        // NOTE:
                        // On some drivers the swapchain extent is specified by
                        // `surface_capabilities.current_extent` and the swapchain size must use
                        // this extent. This extent is always the same as the window size.
                        //
                        // However, other drivers don't specify a value, i.e.
                        // `surface_capabilities.current_extent` is `None`. These drivers will allow
                        // anything, but the only sensible value is the window size.
                        //
                        // Both of these cases need the swapchain to use the window size, so we just
                        // use that.
                        image_extent: window_size.into(),
                        image_usage: ImageUsage::COLOR_ATTACHMENT,

                        // The alpha mode indicates how the alpha value of the final image will
                        // behave. For example, you can choose whether the window will be opaque or
                        // transparent.
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

        // Dynamic viewports allow us to recreate just the viewport when the window is resized.
        // Otherwise we would have to recreate the whole pipeline.
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let mut task_graph = TaskGraph::new(&self.resources);

        let virtual_swapchain_id = task_graph.add_swapchain(&SwapchainCreateInfo {
            image_format: swapchain_format,
            ..Default::default()
        });

        let virtual_framebuffer_id = task_graph.add_framebuffer();

        let triangle_node_id = task_graph
            .create_task_node(
                "Triangle",
                QueueFamilyType::Graphics,
                TriangleTask::new(self, virtual_swapchain_id),
            )
            .framebuffer(virtual_framebuffer_id)
            .color_attachment(
                virtual_swapchain_id.current_image_id(),
                AccessTypes::COLOR_ATTACHMENT_READ | AccessTypes::COLOR_ATTACHMENT_WRITE,
                ImageLayoutType::Optimal,
                &AttachmentInfo {
                    clear: true,
                    ..Default::default()
                },
            )
            .build();

        let mut task_graph = unsafe {
            task_graph.compile(&CompileInfo {
                queues: &[&self.queue],
                present_queue: Some(&self.queue),
                flight_id: self.flight_id,
                ..Default::default()
            })
        }
        .unwrap();

        let triangle_node = task_graph.task_node_mut(triangle_node_id).unwrap();
        let subpass = triangle_node.subpass().unwrap().clone();
        triangle_node
            .task_mut()
            .downcast_mut::<TriangleTask>()
            .unwrap()
            .create_pipeline(self, &subpass);

        // In some situations, the swapchain will become invalid by itself. This includes for
        // example when the window is resized (as the images of the swapchain will no longer match
        // the window's) or, on Android, when the application went to the background and goes back
        // to the foreground.
        //
        // In this situation, acquiring a swapchain image or presenting it will return an error.
        // Rendering to an image of that swapchain will not produce any error, but may or may not
        // work. To continue rendering, we need to recreate the swapchain by creating a new
        // swapchain. Here, we remember that we need to do this for the next loop iteration.
        let recreate_swapchain = false;

        self.rcx = Some(RenderContext {
            window,
            swapchain_id,
            viewport,
            recreate_swapchain,
            task_graph,
            virtual_swapchain_id,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let rcx = self.rcx.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                rcx.recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => {
                let window_size = rcx.window.inner_size();

                // Do not draw the frame when the screen size is zero. On Windows, this can occur
                // when minimizing the application.
                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                let flight = self.resources.flight(self.flight_id).unwrap();

                // Whenever the window resizes we need to recreate everything dependent on the
                // window size. In this example that includes the swapchain, the framebuffers and
                // the dynamic state viewport.
                if rcx.recreate_swapchain {
                    // Use the new dimensions of the window.

                    rcx.swapchain_id = self
                        .resources
                        .recreate_swapchain(rcx.swapchain_id, |create_info| SwapchainCreateInfo {
                            image_extent: window_size.into(),
                            ..*create_info
                        })
                        .expect("failed to recreate swapchain");

                    rcx.viewport.extent = window_size.into();

                    rcx.recreate_swapchain = false;
                }

                flight.wait(None).unwrap();

                let resource_map =
                    resource_map!(&rcx.task_graph, rcx.virtual_swapchain_id => rcx.swapchain_id)
                        .unwrap();

                match unsafe {
                    rcx.task_graph
                        .execute(resource_map, rcx, || rcx.window.pre_present_notify())
                } {
                    Ok(()) => {}
                    Err(ExecuteError::Swapchain {
                        error: VulkanError::OutOfDate,
                        ..
                    }) => {
                        rcx.recreate_swapchain = true;
                    }
                    Err(e) => {
                        panic!("failed to execute next frame: {e:?}");
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}

// We use `#[repr(C)]` here to force rustc to use a defined layout for our data, as the default
// representation has *no guarantees*.
#[derive(Clone, Copy, BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}
