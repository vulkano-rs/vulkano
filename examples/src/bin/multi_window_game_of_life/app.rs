// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    game_of_life::GameOfLifeComputePipeline, render_pass::RenderPassPlaceOverFrame, SCALING,
    WINDOW2_HEIGHT, WINDOW2_WIDTH, WINDOW_HEIGHT, WINDOW_WIDTH,
};
use std::{collections::HashMap, sync::Arc};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::{device::Queue, format::Format};
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use vulkano_util::window::{VulkanoWindows, WindowDescriptor};
use winit::{event_loop::EventLoop, window::WindowId};

pub struct RenderPipeline {
    pub compute: GameOfLifeComputePipeline,
    pub place_over_frame: RenderPassPlaceOverFrame,
}

impl RenderPipeline {
    pub fn new(
        compute_queue: Arc<Queue>,
        gfx_queue: Arc<Queue>,
        size: [u32; 2],
        swapchain_format: Format,
    ) -> RenderPipeline {
        let memory_allocator = StandardMemoryAllocator::new_default(gfx_queue.device().clone());
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            gfx_queue.device().clone(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            gfx_queue.device().clone(),
        ));

        RenderPipeline {
            compute: GameOfLifeComputePipeline::new(
                compute_queue,
                &memory_allocator,
                command_buffer_allocator.clone(),
                descriptor_set_allocator.clone(),
                size,
            ),
            place_over_frame: RenderPassPlaceOverFrame::new(
                gfx_queue,
                &memory_allocator,
                command_buffer_allocator,
                descriptor_set_allocator,
                swapchain_format,
            ),
        }
    }
}

pub struct App {
    pub context: VulkanoContext,
    pub windows: VulkanoWindows,
    pub pipelines: HashMap<WindowId, RenderPipeline>,
}

impl App {
    pub fn open(&mut self, event_loop: &EventLoop<()>) {
        // Create windows & pipelines
        let id1 = self.windows.create_window(
            event_loop,
            &self.context,
            &WindowDescriptor {
                width: WINDOW_WIDTH,
                height: WINDOW_HEIGHT,
                title: "Game of Life Primary".to_string(),
                ..Default::default()
            },
            |_| {},
        );
        let id2 = self.windows.create_window(
            event_loop,
            &self.context,
            &WindowDescriptor {
                width: WINDOW2_WIDTH,
                height: WINDOW2_HEIGHT,
                title: "Game of Life Secondary".to_string(),
                ..Default::default()
            },
            |_| {},
        );
        self.pipelines.insert(
            id1,
            RenderPipeline::new(
                // Use same queue.. for synchronization
                self.context.graphics_queue().clone(),
                self.context.graphics_queue().clone(),
                [
                    (WINDOW_WIDTH / SCALING) as u32,
                    (WINDOW_HEIGHT / SCALING) as u32,
                ],
                self.windows
                    .get_primary_renderer()
                    .unwrap()
                    .swapchain_format(),
            ),
        );
        self.pipelines.insert(
            id2,
            RenderPipeline::new(
                self.context.graphics_queue().clone(),
                self.context.graphics_queue().clone(),
                [
                    (WINDOW2_WIDTH / SCALING) as u32,
                    (WINDOW2_HEIGHT / SCALING) as u32,
                ],
                self.windows.get_renderer(id2).unwrap().swapchain_format(),
            ),
        );
    }
}

impl Default for App {
    fn default() -> Self {
        App {
            context: VulkanoContext::new(VulkanoConfig::default()),
            windows: VulkanoWindows::default(),
            pipelines: HashMap::new(),
        }
    }
}
