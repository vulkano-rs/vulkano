use std::sync::Arc;
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, CommandBufferInheritanceInfo,
        CommandBufferUsage, RecordingCommandBuffer, SecondaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::Queue,
    image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode},
        view::ImageView,
    },
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::VertexInputState,
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::Subpass,
};

/// A subpass pipeline that fills a quad over frame.
pub struct PixelsDrawPipeline {
    gfx_queue: Arc<Queue>,
    subpass: Subpass,
    pipeline: Arc<GraphicsPipeline>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl PixelsDrawPipeline {
    pub fn new(
        gfx_queue: Arc<Queue>,
        subpass: Subpass,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> PixelsDrawPipeline {
        let pipeline = {
            let device = gfx_queue.device();
            let vs = vs::load(device.clone())
                .expect("failed to create shader module")
                .entry_point("main")
                .expect("shader entry point not found");
            let fs = fs::load(device.clone())
                .expect("failed to create shader module")
                .entry_point("main")
                .expect("shader entry point not found");
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

            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(VertexInputState::default()),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.clone().into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        PixelsDrawPipeline {
            gfx_queue,
            subpass,
            pipeline,
            command_buffer_allocator,
            descriptor_set_allocator,
        }
    }

    fn create_descriptor_set(&self, image: Arc<ImageView>) -> Arc<DescriptorSet> {
        let layout = &self.pipeline.layout().set_layouts()[0];
        let sampler = Sampler::new(
            self.gfx_queue.device().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mipmap_mode: SamplerMipmapMode::Linear,
                ..Default::default()
            },
        )
        .unwrap();

        DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            layout.clone(),
            [
                WriteDescriptorSet::sampler(0, sampler),
                WriteDescriptorSet::image_view(1, image),
            ],
            [],
        )
        .unwrap()
    }

    /// Draws input `image` over a quad of size -1.0 to 1.0.
    pub fn draw(
        &self,
        viewport_dimensions: [u32; 2],
        image: Arc<ImageView>,
    ) -> Arc<SecondaryAutoCommandBuffer> {
        let mut builder = RecordingCommandBuffer::secondary(
            self.command_buffer_allocator.clone(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.subpass.clone().into()),
                ..Default::default()
            },
        )
        .unwrap();

        builder
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                self.create_descriptor_set(image),
            )
            .unwrap();

        unsafe {
            builder.draw(6, 1, 0, 0).unwrap();
        }

        builder.end().unwrap()
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            const vec2[6] POSITIONS = {
                vec2(-1.0, -1.0),
                vec2( 1.0,  1.0),
                vec2(-1.0,  1.0),
                vec2(-1.0, -1.0),
                vec2( 1.0, -1.0),
                vec2( 1.0,  1.0),
            };

            const vec2[6] TEX_COORDS = {
                vec2(0.0, 1.0),
                vec2(1.0, 0.0),
                vec2(0.0, 0.0),
                vec2(0.0, 1.0),
                vec2(1.0, 1.0),
                vec2(1.0, 0.0),
            };

            layout(location = 0) out vec2 f_tex_coords;

            void main() {
                gl_Position = vec4(POSITIONS[gl_VertexIndex], 0.0, 1.0);
                f_tex_coords = TEX_COORDS[gl_VertexIndex];
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450
            layout(location = 0) in vec2 v_tex_coords;

            layout(location = 0) out vec4 f_color;

            layout(set = 0, binding = 0) uniform sampler s;
            layout(set = 0, binding = 1) uniform texture2D tex;

            void main() {
                f_color = texture(sampler2D(tex, s), v_tex_coords);
            }
        ",
    }
}
