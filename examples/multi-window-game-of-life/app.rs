use crate::{
    game_of_life::GameOfLifeComputePipeline, render_pass::RenderPassPlaceOverFrame, SCALING,
    WINDOW2_HEIGHT, WINDOW2_WIDTH, WINDOW_HEIGHT, WINDOW_WIDTH,
};
use std::{collections::HashMap, sync::Arc};
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
use winit::{event_loop::EventLoop, window::WindowId};

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
}

impl App {
    pub fn open(&mut self, event_loop: &EventLoop<()>) {
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

        App {
            context,
            windows: VulkanoWindows::default(),
            command_buffer_allocator,
            descriptor_set_allocator,
            pipelines: HashMap::new(),
        }
    }
}
