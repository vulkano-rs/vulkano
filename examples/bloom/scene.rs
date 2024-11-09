use crate::{App, RenderContext};
use std::{alloc::Layout, slice, sync::Arc};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::RenderPassBeginInfo,
    format::Format,
    image::{
        view::{ImageView, ImageViewCreateInfo},
        Image, ImageAspects, ImageSubresourceRange, ImageUsage,
    },
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryTypeFilter},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::ViewportState,
            GraphicsPipelineCreateInfo,
        },
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
};
use vulkano_taskgraph::{
    command_buffer::RecordingCommandBuffer,
    resource::{HostAccessType, Resources},
    Id, Task, TaskContext, TaskResult,
};

pub struct SceneTask {
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    framebuffer: Arc<Framebuffer>,
    vertex_buffer_id: Id<Buffer>,
}

impl SceneTask {
    pub fn new(
        app: &App,
        pipeline_layout: &Arc<PipelineLayout>,
        bloom_image_id: Id<Image>,
    ) -> Self {
        let render_pass = vulkano::single_pass_renderpass!(
            app.device.clone(),
            attachments: {
                color: {
                    format: Format::R32_UINT,
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
            let vs = vs::load(app.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = fs::load(app.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let vertex_input_state = MyVertex::per_vertex().definition(&vs).unwrap();
            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];
            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            GraphicsPipeline::new(
                app.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(pipeline_layout.clone())
                },
            )
            .unwrap()
        };

        let framebuffer = window_size_dependent_setup(&app.resources, bloom_image_id, &render_pass);

        let vertices = [
            MyVertex {
                position: [-0.5, 0.5],
            },
            MyVertex {
                position: [0.5, 0.5],
            },
            MyVertex {
                position: [0.0, -0.5],
            },
        ];
        let vertex_buffer_id = app
            .resources
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

        SceneTask {
            render_pass,
            pipeline,
            framebuffer,
            vertex_buffer_id,
        }
    }

    pub fn handle_resize(&mut self, resources: &Resources, bloom_image_id: Id<Image>) {
        self.framebuffer =
            window_size_dependent_setup(resources, bloom_image_id, &self.render_pass);
    }
}

impl Task for SceneTask {
    type World = RenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        _tcx: &mut TaskContext<'_>,
        rcx: &Self::World,
    ) -> TaskResult {
        cbf.as_raw().begin_render_pass(
            &RenderPassBeginInfo {
                clear_values: vec![Some([0u32; 4].into())],
                ..RenderPassBeginInfo::framebuffer(self.framebuffer.clone())
            },
            &Default::default(),
        )?;
        cbf.set_viewport(0, slice::from_ref(&rcx.viewport))?;
        cbf.bind_pipeline_graphics(&self.pipeline)?;
        cbf.bind_vertex_buffers(0, &[self.vertex_buffer_id], &[0], &[], &[])?;

        unsafe { cbf.draw(3, 1, 0, 0) }?;

        cbf.as_raw().end_render_pass(&Default::default())?;

        cbf.destroy_object(self.framebuffer.clone());

        Ok(())
    }
}

#[derive(Clone, Copy, BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

fn window_size_dependent_setup(
    resources: &Resources,
    bloom_image_id: Id<Image>,
    render_pass: &Arc<RenderPass>,
) -> Arc<Framebuffer> {
    let image_state = resources.image(bloom_image_id).unwrap();
    let image = image_state.image();
    let view = ImageView::new(
        image.clone(),
        ImageViewCreateInfo {
            format: Format::R32_UINT,
            subresource_range: ImageSubresourceRange {
                aspects: ImageAspects::COLOR,
                mip_levels: 0..1,
                array_layers: 0..1,
            },
            usage: ImageUsage::COLOR_ATTACHMENT,
            ..Default::default()
        },
    )
    .unwrap();

    Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![view],
            ..Default::default()
        },
    )
    .unwrap()
}

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
            #include <shared_exponent.glsl>

            layout(location = 0) out uint f_color;

            void main() {
                f_color = convertToSharedExponent(vec3(2.0, 0.0, 0.0));
            }
        ",
        include: ["."],
    }
}
