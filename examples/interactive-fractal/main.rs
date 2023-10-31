// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// This is an example demonstrating an application with some more non-trivial functionality.
// It should get you more up to speed with how you can use Vulkano.
//
// It contains:
//
// - A compute pipeline to calculate Mandelbrot and Julia fractals writing them to an image.
// - A graphics pipeline to draw the fractal image over a quad that covers the whole screen.
// - A renderpass rendering that image on the swapchain image.
// - An organized renderer with functionality good enough to copy to other projects.
// - A simple `FractalApp` to handle runtime state.
// - A simple `InputState` to interact with the application.

use crate::app::FractalApp;
use std::error::Error;
use vulkano::{image::ImageUsage, swapchain::PresentMode, sync::GpuFuture};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    renderer::{VulkanoWindowRenderer, DEFAULT_IMAGE_FORMAT},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

mod app;
mod fractal_compute_pipeline;
mod pixels_draw_pipeline;
mod place_over_frame;

fn main() -> Result<(), impl Error> {
    // Create the event loop.
    let event_loop = EventLoop::new().unwrap();
    let context = VulkanoContext::new(VulkanoConfig::default());
    let mut windows = VulkanoWindows::default();
    let _id = windows.create_window(
        &event_loop,
        &context,
        &WindowDescriptor {
            title: "Fractal".to_string(),
            present_mode: PresentMode::Fifo,
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
        ImageUsage::SAMPLED | ImageUsage::STORAGE | ImageUsage::TRANSFER_DST,
    );

    // Create app to hold the logic of our fractal explorer.
    let gfx_queue = context.graphics_queue();

    // We intend to eventually render on our swapchain, thus we use that format when creating the
    // app here.
    let mut app = FractalApp::new(
        gfx_queue.clone(),
        primary_window_renderer.swapchain_format(),
    );
    app.print_guide();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        let renderer = windows.get_primary_renderer_mut().unwrap();

        if process_event(renderer, &event, &mut app, render_target_id) {
            elwt.exit();
            return;
        }

        // Pass event for the app to handle our inputs.
        app.handle_input(renderer.window_size(), &event);
    })
}

/// Processes a single event for an event loop.
/// Returns true only if the window is to be closed.
pub fn process_event(
    renderer: &mut VulkanoWindowRenderer,
    event: &Event<()>,
    app: &mut FractalApp,
    render_target_id: usize,
) -> bool {
    match &event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            return true;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(..) | WindowEvent::ScaleFactorChanged { .. },
            ..
        } => renderer.resize(),
        Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            ..
        } => 'redraw: {
            // Tasks for redrawing:
            // 1. Update state based on events
            // 2. Compute & Render
            // 3. Reset input state
            // 4. Update time & title

            // The rendering part goes here:
            match renderer.window_size() {
                [w, h] => {
                    // Skip this frame when minimized.
                    if w == 0.0 || h == 0.0 {
                        break 'redraw;
                    }
                }
            }
            app.update_state_after_inputs(renderer);
            compute_then_render(renderer, app, render_target_id);
            app.reset_input_state();
            app.update_time();
            renderer.window().set_title(&format!(
                "{} fps: {:.2} dt: {:.2}, Max Iterations: {}",
                if app.is_julia { "Julia" } else { "Mandelbrot" },
                app.avg_fps(),
                app.dt(),
                app.max_iters
            ));
        }
        Event::AboutToWait => renderer.window().request_redraw(),
        _ => (),
    }
    !app.is_running()
}

/// Orchestrates rendering.
fn compute_then_render(
    renderer: &mut VulkanoWindowRenderer,
    app: &mut FractalApp,
    target_image_id: usize,
) {
    // Start the frame.
    let before_pipeline_future = match renderer.acquire() {
        Err(e) => {
            println!("{e}");
            return;
        }
        Ok(future) => future,
    };

    // Retrieve the target image.
    let image = renderer.get_additional_image_view(target_image_id);

    // Compute our fractal (writes to target image). Join future with `before_pipeline_future`.
    let after_compute = app.compute(image.clone()).join(before_pipeline_future);

    // Render the image over the swapchain image, inputting the previous future.
    let after_renderpass_future =
        app.place_over_frame
            .render(after_compute, image, renderer.swapchain_image_view());

    // Finish the frame (which presents the view), inputting the last future. Wait for the future
    // so resources are not in use when we render.
    renderer.present(after_renderpass_future, true);
}
