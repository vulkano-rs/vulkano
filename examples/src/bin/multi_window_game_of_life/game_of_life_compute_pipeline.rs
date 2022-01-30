// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::vulkano_context::DeviceImageView;
use crate::vulkano_window::create_device_image;
use cgmath::Vector2;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::command_buffer::{PrimaryAutoCommandBuffer, PrimaryCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::ImageAccess;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::sync::GpuFuture;

pub struct GameOfLifeComputePipeline {
    compute_queue: Arc<Queue>,
    compute_life_pipeline: Arc<ComputePipeline>,
    life_in: Arc<CpuAccessibleBuffer<[u32]>>,
    life_out: Arc<CpuAccessibleBuffer<[u32]>>,
    image: DeviceImageView,
}

impl GameOfLifeComputePipeline {
    pub fn new(compute_queue: Arc<Queue>, size: [u32; 2]) -> GameOfLifeComputePipeline {
        let life_in = CpuAccessibleBuffer::from_iter(
            compute_queue.device().clone(),
            BufferUsage::all(),
            false,
            vec![0; (size[0] * size[1]) as usize],
        )
        .unwrap();
        let life_out = CpuAccessibleBuffer::from_iter(
            compute_queue.device().clone(),
            BufferUsage::all(),
            false,
            vec![0; (size[0] * size[1]) as usize],
        )
        .unwrap();

        let compute_life_pipeline = {
            let shader = compute_life_cs::load(compute_queue.device().clone()).unwrap();
            ComputePipeline::new(
                compute_queue.device().clone(),
                shader.entry_point("main").unwrap(),
                &(),
                None,
                |_| {},
            )
            .unwrap()
        };

        let image = create_device_image(compute_queue.clone(), size, Format::R8G8B8A8_UNORM);
        GameOfLifeComputePipeline {
            compute_queue,
            compute_life_pipeline,
            life_in,
            life_out,
            image,
        }
    }

    pub fn color_image(&self) -> DeviceImageView {
        self.image.clone()
    }

    pub fn draw_life(&mut self, pos: Vector2<i32>) {
        let mut life_in = self.life_in.write().unwrap();
        let size = self.image.image().dimensions().width_height();
        if pos.y < 0 || pos.y >= size[1] as i32 || pos.x < 0 || pos.x >= size[0] as i32 {
            return;
        }
        let index = (pos.y * size[0] as i32 + pos.x) as usize;
        life_in[index] = 1;
    }

    pub fn compute(
        &mut self,
        before_future: Box<dyn GpuFuture>,
        life_color: [f32; 4],
        dead_color: [f32; 4],
    ) -> Box<dyn GpuFuture> {
        let mut builder = AutoCommandBufferBuilder::primary(
            self.compute_queue.device().clone(),
            self.compute_queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        self.dispatch(&mut builder, life_color, dead_color, 0);
        self.dispatch(&mut builder, life_color, dead_color, 1);

        let command_buffer = builder.build().unwrap();
        let finished = before_future
            .then_execute(self.compute_queue.clone(), command_buffer)
            .unwrap();
        let after_pipeline = finished.then_signal_fence_and_flush().unwrap().boxed();

        // Swap input and output
        std::mem::swap(&mut self.life_in, &mut self.life_out);

        after_pipeline
    }

    fn dispatch(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        life_color: [f32; 4],
        dead_color: [f32; 4],
        step: i32,
    ) {
        // Resize image if needed
        let img_dims = self.image.image().dimensions().width_height();
        let pipeline_layout = self.compute_life_pipeline.layout();
        let desc_layout = pipeline_layout.descriptor_set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            desc_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, self.image.clone()),
                WriteDescriptorSet::buffer(1, self.life_in.clone()),
                WriteDescriptorSet::buffer(2, self.life_out.clone()),
            ],
        )
        .unwrap();

        let push_constants = compute_life_cs::ty::PushConstants {
            life_color,
            dead_color,
            step,
        };
        builder
            .bind_pipeline_compute(self.compute_life_pipeline.clone())
            .bind_descriptor_sets(PipelineBindPoint::Compute, pipeline_layout.clone(), 0, set)
            .push_constants(pipeline_layout.clone(), 0, push_constants)
            .dispatch([img_dims[0] / 8, img_dims[1] / 8, 1])
            .unwrap();
    }
}

mod compute_life_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;
layout(set = 0, binding = 1) buffer LifeInBuffer { uint life_in[]; };
layout(set = 0, binding = 2) buffer LifeOutBuffer { uint life_out[]; };

layout(push_constant) uniform PushConstants {
    vec4 life_color;
    vec4 dead_color;
    int step;
} push_constants;

int get_index(ivec2 pos) {
    ivec2 dims = ivec2(imageSize(img));
    return pos.y * dims.x + pos.x;
}

void compute_life() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    int index = get_index(pos);
    life_out[index] = life_in[index];
}

void compute_color() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    int index = get_index(pos);
    if (life_out[index] == 1) {
        imageStore(img, pos, push_constants.life_color);
    } else {
        imageStore(img, pos, push_constants.dead_color);
    }
}

void main() {
    if (push_constants.step == 0) {
        compute_life();
    } else {
        compute_color();
    }
}"
    }
}
