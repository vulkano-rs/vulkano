// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.
mod app;
mod game_of_life_compute_pipeline;
mod pixels_draw;
mod render_pass;
mod vulkano_config;
mod vulkano_context;
mod vulkano_window;

use crate::app::{App, RenderPipeline};
use crate::vulkano_window::VulkanoWindow;
use cgmath::Vector2;
use time::Instant;
use vulkano::image::ImageAccess;
use vulkano::sync::GpuFuture;
use winit::event::{ElementState, MouseButton};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    platform::run_return::EventLoopExtRunReturn,
};

/*
A multi windowed game of life application
 */

fn main() {
    // Create event loop
    let mut event_loop = EventLoop::new();
    // Create app with vulkano context
    let mut app = App::default();
    app.open(&event_loop);

    // Time & inputs...
    let mut time = Instant::now();
    let mut cursor_pos = Vector2::new(0.0, 0.0);

    loop {
        if !handle_events(&mut event_loop, &mut app, &mut cursor_pos) {
            break;
        }
        // Render 60fps
        if (Instant::now() - time).as_seconds_f64() > 1.0 / 60.0 {
            compute_then_render_per_window(&mut app);
            time = Instant::now();
        }
    }
}

/// Handle events and return `bool` if we should quit
fn handle_events(
    event_loop: &mut EventLoop<()>,
    app: &mut App,
    cursor_pos: &mut Vector2<f32>,
) -> bool {
    let mut is_running = true;
    event_loop.run_return(|event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        match &event {
            Event::WindowEvent {
                event, window_id, ..
            } => match event {
                WindowEvent::CloseRequested => {
                    if *window_id == app.primary_window_id {
                        is_running = false;
                    } else {
                        // Destroy window by removing it...
                        app.windows.remove(window_id);
                        app.pipelines.remove(window_id);
                    }
                }
                // Resize window and its images...
                WindowEvent::Resized(..) | WindowEvent::ScaleFactorChanged { .. } => {
                    let vulkano_window = app.windows.get_mut(window_id).unwrap();
                    vulkano_window.resize();
                }
                WindowEvent::CursorMoved { position, .. } => {
                    *cursor_pos = Vector2::new(position.x as f32, position.y as f32)
                }
                // Mouse button event
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == &MouseButton::Left && state == &ElementState::Pressed {
                        let window_size = app.windows.get(window_id).unwrap().window_size();
                        let compute_pipeline =
                            &mut app.pipelines.get_mut(window_id).unwrap().compute;
                        let normalized_pos = Vector2::new(
                            (cursor_pos.x / window_size[0] as f32).clamp(0.0, 1.0),
                            (cursor_pos.y / window_size[1] as f32).clamp(0.0, 1.0),
                        );
                        let image_size = compute_pipeline
                            .color_image()
                            .image()
                            .dimensions()
                            .width_height();
                        compute_pipeline.draw_life(Vector2::new(
                            (image_size[0] as f32 * normalized_pos.x) as i32,
                            (image_size[1] as f32 * normalized_pos.y) as i32,
                        ))
                    }
                }
                _ => (),
            },
            Event::MainEventsCleared => *control_flow = ControlFlow::Exit,
            _ => (),
        }
    });
    is_running
}

/// Compute and render per window
fn compute_then_render_per_window(app: &mut App) {
    for (window_id, vulkano_window) in app.windows.iter_mut() {
        let pipeline = app.pipelines.get_mut(window_id).unwrap();
        if *window_id == app.primary_window_id {
            compute_then_render(vulkano_window, pipeline, [1.0, 0.0, 0.0, 1.0], [0.0; 4]);
        } else {
            compute_then_render(vulkano_window, pipeline, [0.0, 0.0, 0.0, 1.0], [1.0; 4]);
        }
    }
}

/// Compute game of life, then display result on target image
fn compute_then_render(
    vulkano_window: &mut VulkanoWindow,
    pipeline: &mut RenderPipeline,
    life_color: [f32; 4],
    dead_color: [f32; 4],
) {
    // Start frame
    let before_pipeline_future = match vulkano_window.start_frame() {
        Err(e) => {
            println!("{}", e.to_string());
            return;
        }
        Ok(future) => future,
    };

    // Compute
    let after_compute = pipeline.compute.compute_life();
    let after_color = pipeline
        .compute
        .compute_color(life_color, dead_color)
        .join(after_compute)
        .join(before_pipeline_future);
    let color_image = pipeline.compute.color_image();

    // Render
    let target_image = vulkano_window.final_image();
    let after_render = pipeline
        .place_over_frame
        .render(after_color, color_image, target_image);

    // Finish frame
    vulkano_window.finish_frame(after_render);
}
