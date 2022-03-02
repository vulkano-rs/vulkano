// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    game_of_life::GameOfLifeComputePipeline, render_pass::RenderPassPlaceOverFrame,
    vulkano_config::VulkanoConfig, vulkano_context::VulkanoContext, vulkano_window::VulkanoWindow,
    SCALING, WINDOW2_HEIGHT, WINDOW2_WIDTH, WINDOW_HEIGHT, WINDOW_WIDTH,
};
use std::{collections::HashMap, sync::Arc};
use vulkano::{device::Queue, format::Format};
use winit::{
    event_loop::EventLoop,
    window::{WindowBuilder, WindowId},
};

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
        RenderPipeline {
            compute: GameOfLifeComputePipeline::new(compute_queue, size),
            place_over_frame: RenderPassPlaceOverFrame::new(gfx_queue, swapchain_format),
        }
    }
}

pub struct App {
    pub context: VulkanoContext,
    pub windows: HashMap<WindowId, VulkanoWindow>,
    pub pipelines: HashMap<WindowId, RenderPipeline>,
    pub primary_window_id: WindowId,
}

impl App {
    pub fn open(&mut self, event_loop: &EventLoop<()>) {
        // Create windows & pipelines
        let winit_window_primary_builder = WindowBuilder::new()
            .with_inner_size(winit::dpi::LogicalSize::new(
                WINDOW_WIDTH as f32,
                WINDOW_HEIGHT as f32,
            ))
            .with_title("Game of Life Primary");
        let winit_window_secondary_builder = WindowBuilder::new()
            .with_inner_size(winit::dpi::LogicalSize::new(
                WINDOW2_WIDTH as f32,
                WINDOW2_HEIGHT as f32,
            ))
            .with_title("Game of Life Secondary");
        let winit_window_primary = winit_window_primary_builder.build(&event_loop).unwrap();
        let winit_window_secondary = winit_window_secondary_builder.build(&event_loop).unwrap();
        let window_primary = VulkanoWindow::new(&self.context, winit_window_primary, false);
        let window_secondary = VulkanoWindow::new(&self.context, winit_window_secondary, false);
        self.pipelines.insert(
            window_primary.window().id(),
            RenderPipeline::new(
                // Use same queue.. for synchronization
                self.context.graphics_queue(),
                self.context.graphics_queue(),
                [WINDOW_WIDTH / SCALING, WINDOW_HEIGHT / SCALING],
                window_primary.swapchain_format(),
            ),
        );
        self.pipelines.insert(
            window_secondary.window().id(),
            RenderPipeline::new(
                self.context.graphics_queue(),
                self.context.graphics_queue(),
                [WINDOW2_WIDTH / SCALING, WINDOW2_HEIGHT / SCALING],
                window_secondary.swapchain_format(),
            ),
        );
        self.primary_window_id = window_primary.window().id();
        self.windows
            .insert(window_primary.window().id(), window_primary);
        self.windows
            .insert(window_secondary.window().id(), window_secondary);
    }
}

impl Default for App {
    fn default() -> Self {
        App {
            context: VulkanoContext::new(&VulkanoConfig::default()),
            windows: HashMap::new(),
            pipelines: HashMap::new(),
            primary_window_id: unsafe { WindowId::dummy() },
        }
    }
}
