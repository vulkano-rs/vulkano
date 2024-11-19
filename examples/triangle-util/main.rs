// Welcome to the triangle-util example!
//
// This is almost exactly the same as the triangle example, except that it uses utility functions
// to make life easier.
//
// This example assumes that you are already more or less familiar with graphics programming and
// that you want to learn Vulkan. This means that for example it won't go into details about what a
// vertex or a shader is.

use std::{error::Error, sync::Arc, time::Duration};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    device::DeviceOwnedVulkanObject,
    image::view::ImageView,
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::{
        graphics::{
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        ray_tracing::{
            RayTracingPipeline, RayTracingPipelineCreateInfo, RayTracingShaderGroupCreateInfo,
            ShaderBindingTable,
        },
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::GpuFuture,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::VulkanoWindows,
};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::WindowId,
};

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}

struct App {
    context: VulkanoContext,
    windows: VulkanoWindows,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    vertex_buffer: Subbuffer<[MyVertex]>,
    rcx: Option<RenderContext>,
}

struct RenderContext {
    pipeline: Arc<RayTracingPipeline>,
}

mod raygen {
    vulkano_shaders::shader! {
        ty: "raygen",
        path: "raytrace.rgen",
        vulkan_version: "1.2"
    }
}

mod closest_hit {
    vulkano_shaders::shader! {
        ty: "closesthit",
        path: "raytrace.rchit",
        vulkan_version: "1.2"
    }
}

mod miss {
    vulkano_shaders::shader! {
        ty: "miss",
        path: "raytrace.rmiss",
        vulkan_version: "1.2"
    }
}

impl App {
    fn new(_event_loop: &EventLoop<()>) -> Self {
        let context = VulkanoContext::new(VulkanoConfig::default());

        // Manages any windows and their rendering.
        let windows = VulkanoWindows::default();

        // Some little debug infos.
        println!(
            "Using device: {} (type: {:?})",
            context.device().physical_device().properties().device_name,
            context.device().physical_device().properties().device_type,
        );

        // Before we can start creating and recording command buffers, we need a way of allocating
        // them. Vulkano provides a command buffer allocator, which manages raw Vulkan command
        // pools underneath and provides a safe interface for them.
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device().clone(),
            Default::default(),
        ));

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
        let vertex_buffer = Buffer::from_iter(
            context.memory_allocator().clone(),
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

        App {
            context,
            windows,
            command_buffer_allocator,
            vertex_buffer,
            rcx: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(primary_window_id) = self.windows.primary_window_id() {
            self.windows.remove_renderer(primary_window_id);
        }

        self.windows
            .create_window(event_loop, &self.context, &Default::default(), |_| {});
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();
        let window_size = window_renderer.window().inner_size();

        // Before we draw, we have to create what is called a **pipeline**. A pipeline describes
        // how a GPU operation is to be performed. It is similar to an OpenGL program, but it also
        // contains many settings for customization, all baked into a single object. For drawing,
        // we create a **graphics** pipeline, but there are also other types of pipeline.
        let pipeline = {
            let raygen = raygen::load(self.context.device().clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let closest_hit = closest_hit::load(self.context.device().clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let miss = miss::load(self.context.device().clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            // Make a list of the shader stages that the pipeline will have.
            let stages = [
                PipelineShaderStageCreateInfo::new(raygen),
                PipelineShaderStageCreateInfo::new(miss),
                PipelineShaderStageCreateInfo::new(closest_hit),
            ];

            let groups = [
                RayTracingShaderGroupCreateInfo {
                    // Raygen
                    general_shader: Some(0),
                    ..Default::default()
                },
                RayTracingShaderGroupCreateInfo {
                    // Miss
                    general_shader: Some(1),
                    ..Default::default()
                },
                RayTracingShaderGroupCreateInfo {
                    // Closest Hit
                    group_type: ash::vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP,
                    closest_hit_shader: Some(2),
                    ..Default::default()
                },
            ];

            let layout = PipelineLayout::new(
                self.context.device().clone(),
                // Since we only have one pipeline in this example, and thus one pipeline layout,
                // we automatically generate the creation info for it from the resources used in the
                // shaders. In a real application, you would specify this information manually so that
                // you can re-use one layout in multiple pipelines.
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(self.context.device().clone())
                    .unwrap(),
            )
            .unwrap();

            RayTracingPipeline::new(
                self.context.device().clone(),
                None,
                RayTracingPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    groups: groups.into_iter().collect(),
                    max_pipeline_ray_recursion_depth: 1,

                    ..RayTracingPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };
        pipeline
            .set_debug_utils_object_name("Ray Tracing Pipeline".into())
            .unwrap();

        let shader_binding_table =
            ShaderBindingTable::new(self.context.memory_allocator().clone(), &pipeline, 1, 1, 0)
                .unwrap();

        // In the `window_event` handler below we are going to submit commands to the GPU.
        // Submitting a command produces an object that implements the `GpuFuture` trait, which
        // holds the resources for as long as they are in use by the GPU.

        self.rcx = Some(RenderContext { pipeline });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();
        let rcx = self.rcx.as_mut().unwrap();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                window_renderer.resize();
            }
            WindowEvent::RedrawRequested => {
                let window_size = window_renderer.window().inner_size();

                // Do not draw the frame when the screen size is zero. On Windows, this can
                // occur when minimizing the application.
                if window_size.width == 0 || window_size.height == 0 {
                    return;
                }

                // Begin rendering by acquiring the gpu future from the window renderer.
                let previous_frame_end = window_renderer
                    .acquire(Some(Duration::from_millis(1000)), |_swapchain_images| {})
                    .unwrap();

                // In order to draw, we have to record a *command buffer*. The command buffer
                // object holds the list of commands that are going to be executed.
                //
                // Recording a command buffer is an expensive operation (usually a few hundred
                // microseconds), but it is known to be a hot path in the driver and is expected to
                // be optimized.
                //
                // Note that we have to pass a queue family when we create the command buffer. The
                // command buffer will only be executable on that given queue family.
                let mut builder = AutoCommandBufferBuilder::primary(
                    self.command_buffer_allocator.clone(),
                    self.context.graphics_queue().queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                // We add a draw command.
                unsafe { builder.draw(self.vertex_buffer.len() as u32, 1, 0, 0) }.unwrap();

                // Finish recording the command buffer by calling `end`.
                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .then_execute(self.context.graphics_queue().clone(), command_buffer)
                    .unwrap()
                    .boxed();

                // The color output is now expected to contain our triangle. But in order to show
                // it on the screen, we have to *present* the image by calling `present` on the
                // window renderer.
                //
                // This function does not actually present the image immediately. Instead it
                // submits a present command at the end of the queue. This means that it will only
                // be presented once the GPU has finished executing the command buffer that draws
                // the triangle.
                window_renderer.present(future, false);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let window_renderer = self.windows.get_primary_renderer_mut().unwrap();
        window_renderer.window().request_redraw();
    }
}

// We use `#[repr(C)]` here to force rustc to use a defined layout for our data, as the default
// representation has *no guarantees*.
#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}
