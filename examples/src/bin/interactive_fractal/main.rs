// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::app::FractalApp;
use vulkano::image::ImageUsage;
use vulkano::swapchain::PresentMode;
use vulkano::sync::GpuFuture;
use vulkano_util::context::{VulkanoConfig, VulkanoContext};
use vulkano_util::renderer::{VulkanoWindowRenderer, DEFAULT_IMAGE_FORMAT};
use vulkano_util::window::{VulkanoWindows, WindowDescriptor};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
};

mod app;
mod fractal_compute_pipeline;
mod pixels_draw_pipeline;
mod place_over_frame;

/// This is an example demonstrating an application with some more non-trivial functionality.
/// It should get you more up to speed with how you can use Vulkano.
/// It contains
/// - Compute pipeline to calculate Mandelbrot and Julia fractals writing them to an image target
/// - Graphics pipeline to draw the fractal image over a quad that covers the whole screen
/// - Renderpass rendering that image over swapchain image
/// - An organized Renderer with functionality good enough to copy to other projects
/// - Simple FractalApp to handle runtime state
/// - Simple Input system to interact with the application
fn main() {
    // Create event loop
    let mut event_loop = EventLoop::new();
    let context = VulkanoContext::new(VulkanoConfig::default());
    let mut windows = VulkanoWindows::default();
    windows.create_window(
        &event_loop,
        &context,
        &WindowDescriptor {
            title: "Fractal".to_string(),
            present_mode: PresentMode::Immediate,
            ..Default::default()
        },
        |_| {},
    );

    // Add our render target image onto which we'll be rendering our fractals.
    let render_target_id = 0;
    let primary_window_renderer = windows.get_primary_renderer_mut().unwrap();
    // Make sure the image usage is correct (based on your pipeline).
    primary_window_renderer.add_additional_image_view(
        render_target_id,
        DEFAULT_IMAGE_FORMAT,
        ImageUsage {
            sampled: true,
            storage: true,
            color_attachment: true,
            transfer_dst: true,
            ..ImageUsage::none()
        },
    );

    // Create app to hold the logic of our fractal explorer
    let gfx_queue = context.graphics_queue();
    // We intend to eventually render on our swapchain, thus we use that format when creating the app here.
    let mut app = FractalApp::new(gfx_queue, primary_window_renderer.swapchain_format());
    app.print_guide();

    // Basic loop for our runtime
    // 1. Handle events
    // 2. Update state based on events
    // 3. Compute & Render
    // 4. Reset input state
    // 5. Update time & title
    loop {
        if !handle_events(&mut event_loop, primary_window_renderer, &mut app) {
            break;
        }

        match primary_window_renderer.window_size() {
            [w, h] => {
                // Skip this frame when minimized
                if w == 0.0 || h == 0.0 {
                    continue;
                }
            }
        }

        app.update_state_after_inputs(primary_window_renderer);
        compute_then_render(primary_window_renderer, &mut app, render_target_id);
        app.reset_input_state();
        app.update_time();
        primary_window_renderer.window().set_title(&format!(
            "{} fps: {:.2} dt: {:.2}, Max Iterations: {}",
            if app.is_julia { "Julia" } else { "Mandelbrot" },
            app.avg_fps(),
            app.dt(),
            app.max_iters
        ));
    }
}

/// Handle events and return `bool` if we should quit
fn handle_events(
    event_loop: &mut EventLoop<()>,
    renderer: &mut VulkanoWindowRenderer,
    app: &mut FractalApp,
) -> bool {
    let mut is_running = true;
    event_loop.run_return(|event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        match &event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => is_running = false,
                WindowEvent::Resized(..) | WindowEvent::ScaleFactorChanged { .. } => {
                    renderer.resize()
                }
                _ => (),
            },
            Event::MainEventsCleared => *control_flow = ControlFlow::Exit,
            _ => (),
        }
        // Pass event for app to handle our inputs
        app.handle_input(renderer.window_size(), &event);
    });
    is_running && app.is_running()
}

/// Orchestrate rendering here
fn compute_then_render(
    renderer: &mut VulkanoWindowRenderer,
    app: &mut FractalApp,
    target_image_id: usize,
) {
    // Start frame
    let before_pipeline_future = match renderer.start_frame() {
        Err(e) => {
            println!("{}", e.to_string());
            return;
        }
        Ok(future) => future,
    };
    // Retrieve target image
    let image = renderer.get_additional_image_view(target_image_id);
    // Compute our fractal (writes to target image). Join future with `before_pipeline_future`.
    let after_compute = app.compute(image.clone()).join(before_pipeline_future);
    // Render image over frame. Input previous future. Draw on swapchain image
    let after_renderpass_future =
        app.place_over_frame
            .render(after_compute, image, renderer.swapchain_image_view());
    // Finish frame (which presents the view). Input last future
    renderer.finish_frame(after_renderpass_future);
}
