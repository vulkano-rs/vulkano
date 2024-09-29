use crate::{
    game_of_life::GameOfLifeComputePipeline, render_pass::RenderPassPlaceOverFrame, SCALING,
    WINDOW2_HEIGHT, WINDOW2_WIDTH, WINDOW_HEIGHT, WINDOW_WIDTH,
};
use glam::Vec2;
use std::{collections::HashMap, sync::Arc, time::Instant};
use vulkano::{
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::Queue,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    renderer::VulkanoWindowRenderer,
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{application::ApplicationHandler, window::WindowId};

pub struct RenderPipeline {
    pub compute: GameOfLifeComputePipeline,
    pub place_over_frame: RenderPassPlaceOverFrame,
}

impl RenderPipeline {
    pub fn new(
        app: &App,
        compute_queue: Arc<Queue>,
        gfx_queue: Arc<Queue>,
        size: [u32; 2],
        window_renderer: &VulkanoWindowRenderer,
    ) -> RenderPipeline {
        RenderPipeline {
            compute: GameOfLifeComputePipeline::new(app, compute_queue, size),
            place_over_frame: RenderPassPlaceOverFrame::new(app, gfx_queue, window_renderer),
        }
    }
}

pub struct App {
    pub context: VulkanoContext,
    pub windows: VulkanoWindows,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    pub pipelines: HashMap<WindowId, RenderPipeline>,
    pub time: Instant,
    pub cursor_pos: Vec2,
    pub mouse_is_pressed_w1: bool,
    pub mouse_is_pressed_w2: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        // Create windows & pipelines.
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
                self,
                // Use same queue.. for synchronization.
                self.context.graphics_queue().clone(),
                self.context.graphics_queue().clone(),
                [
                    (WINDOW_WIDTH / SCALING) as u32,
                    (WINDOW_HEIGHT / SCALING) as u32,
                ],
                self.windows.get_primary_renderer().unwrap(),
            ),
        );
        self.pipelines.insert(
            id2,
            RenderPipeline::new(
                self,
                self.context.graphics_queue().clone(),
                self.context.graphics_queue().clone(),
                [
                    (WINDOW2_WIDTH / SCALING) as u32,
                    (WINDOW2_HEIGHT / SCALING) as u32,
                ],
                self.windows.get_renderer(id2).unwrap(),
            ),
        );

        event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    }
    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: WindowId,
        event: winit::event::WindowEvent,
    ) {
        if super::process_event(event, window_id, self) {
            event_loop.exit();
            return;
        }

        // Draw life on windows if mouse is down.
        super::draw_life(self);

        // Compute life & render 60fps.
        if (Instant::now() - self.time).as_secs_f64() > 1.0 / 60.0 {
            super::compute_then_render_per_window(self);
            self.time = Instant::now();
        }
    }
    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        for (_, renderer) in self.windows.iter() {
            renderer.window().request_redraw();
        }
    }
}

impl Default for App {
    fn default() -> Self {
        let context = VulkanoContext::new(VulkanoConfig::default());
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            context.device().clone(),
            StandardCommandBufferAllocatorCreateInfo {
                secondary_buffer_count: 32,
                ..Default::default()
            },
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            context.device().clone(),
            Default::default(),
        ));

        // Time & inputs...
        let time = Instant::now();
        let cursor_pos = Vec2::ZERO;

        // An extremely crude way to handle input state... but works for this example.
        let mouse_is_pressed_w1 = false;
        let mouse_is_pressed_w2 = false;

        App {
            context,
            windows: VulkanoWindows::default(),
            command_buffer_allocator,
            descriptor_set_allocator,
            pipelines: HashMap::new(),
            time,
            cursor_pos,
            mouse_is_pressed_w1,
            mouse_is_pressed_w2,
        }
    }
}
