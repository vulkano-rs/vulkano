use crate::game_of_life_compute_pipeline::GameOfLifeComputePipeline;
use crate::render_pass::RenderPassPlaceOverFrame;
use crate::vulkano_config::VulkanoConfig;
use crate::vulkano_context::VulkanoContext;
use crate::vulkano_window::VulkanoWindow;
use std::collections::HashMap;
use std::sync::Arc;
use vulkano::device::Queue;
use vulkano::format::Format;
use winit::event_loop::EventLoop;
use winit::window::{WindowBuilder, WindowId};

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
            .with_inner_size(winit::dpi::LogicalSize::new(1920.0, 1080.0))
            .with_title("Game of Life Primary");
        let winit_window_secondary_builder = WindowBuilder::new()
            .with_inner_size(winit::dpi::LogicalSize::new(512.0, 512.0))
            .with_title("Game of Life Secondary");
        let winit_window_primary = winit_window_primary_builder.build(&event_loop).unwrap();
        let winit_window_secondary = winit_window_secondary_builder.build(&event_loop).unwrap();
        let window_primary = VulkanoWindow::new(&self.context, winit_window_primary, false);
        let window_secondary = VulkanoWindow::new(&self.context, winit_window_secondary, false);
        self.pipelines.insert(
            window_primary.window().id(),
            RenderPipeline::new(
                self.context.compute_queue(),
                self.context.graphics_queue(),
                [1920, 1080],
                window_primary.swapchain_format(),
            ),
        );
        self.pipelines.insert(
            window_secondary.window().id(),
            RenderPipeline::new(
                self.context.compute_queue(),
                self.context.graphics_queue(),
                [512, 512],
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
