// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    fractal_compute_pipeline::FractalComputePipeline, place_over_frame::RenderPassPlaceOverFrame,
};
use cgmath::Vector2;
use std::{sync::Arc, time::Instant};
use vulkano::{
    command_buffer::allocator::{
        StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::Queue,
    image::view::ImageView,
    memory::allocator::StandardMemoryAllocator,
    sync::GpuFuture,
};
use vulkano_util::{renderer::VulkanoWindowRenderer, window::WindowDescriptor};
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    keyboard::{Key, NamedKey},
    window::Fullscreen,
};

const MAX_ITERS_INIT: u32 = 200;
const MOVE_SPEED: f32 = 0.5;

/// App for exploring Julia and Mandelbrot fractals.
pub struct FractalApp {
    /// Pipeline that computes Mandelbrot & Julia fractals and writes them to an image.
    fractal_pipeline: FractalComputePipeline,
    /// Our render pipeline (pass).
    pub place_over_frame: RenderPassPlaceOverFrame,
    /// Toggle that flips between Julia and Mandelbrot.
    pub is_julia: bool,
    /// Toggle that stops the movement on Julia.
    is_c_paused: bool,
    /// C is a constant input to Julia escape time algorithm (mouse position).
    c: Vector2<f32>,
    /// Our zoom level.
    scale: Vector2<f32>,
    /// Our translation on the complex plane.
    translation: Vector2<f32>,
    /// How long the escape time algorithm should run (higher = less performance, more accurate
    /// image).
    pub max_iters: u32,
    /// Time tracking, useful for frame independent movement.
    time: Instant,
    dt: f32,
    dt_sum: f32,
    frame_count: f32,
    avg_fps: f32,
    /// Input state to handle mouse positions, continuous movement etc.
    input_state: InputState,
}

impl FractalApp {
    pub fn new(gfx_queue: Arc<Queue>, image_format: vulkano::format::Format) -> FractalApp {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(
            gfx_queue.device().clone(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            gfx_queue.device().clone(),
            StandardCommandBufferAllocatorCreateInfo {
                secondary_buffer_count: 32,
                ..Default::default()
            },
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            gfx_queue.device().clone(),
            Default::default(),
        ));

        FractalApp {
            fractal_pipeline: FractalComputePipeline::new(
                gfx_queue.clone(),
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                descriptor_set_allocator.clone(),
            ),
            place_over_frame: RenderPassPlaceOverFrame::new(
                gfx_queue,
                memory_allocator.clone(),
                command_buffer_allocator,
                descriptor_set_allocator,
                image_format,
            ),
            is_julia: false,
            is_c_paused: false,
            c: Vector2::new(0.0, 0.0),
            scale: Vector2::new(4.0, 4.0),
            translation: Vector2::new(0.0, 0.0),
            max_iters: MAX_ITERS_INIT,
            time: Instant::now(),
            dt: 0.0,
            dt_sum: 0.0,
            frame_count: 0.0,
            avg_fps: 0.0,
            input_state: InputState::new(),
        }
    }

    pub fn print_guide(&self) {
        println!(
            "\
Usage:
    WASD: Pan view
    Scroll: Zoom in/out
    Space: Toggle between Mandelbrot and Julia
    Enter: Randomize color palette
    Equals/Minus: Increase/Decrease max iterations
    F: Toggle full-screen
    Right mouse: Stop movement in Julia (mouse position determines c)
    Esc: Quit\
            ",
        );
    }

    /// Runs our compute pipeline and return a future of when the compute is finished.
    pub fn compute(&self, image_target: Arc<ImageView>) -> Box<dyn GpuFuture> {
        self.fractal_pipeline.compute(
            image_target,
            self.c,
            self.scale,
            self.translation,
            self.max_iters,
            self.is_julia,
        )
    }

    /// Returns whether the app should quit. (Happens on when pressing ESC.)
    pub fn is_running(&self) -> bool {
        !self.input_state.should_quit
    }

    /// Returns the average FPS.
    pub fn avg_fps(&self) -> f32 {
        self.avg_fps
    }

    /// Returns the delta time in milliseconds.
    pub fn dt(&self) -> f32 {
        self.dt * 1000.0
    }

    /// Updates times and dt at the end of each frame.
    pub fn update_time(&mut self) {
        // Each second, update average fps & reset frame count & dt sum.
        if self.dt_sum > 1.0 {
            self.avg_fps = self.frame_count / self.dt_sum;
            self.frame_count = 0.0;
            self.dt_sum = 0.0;
        }
        self.dt = self.time.elapsed().as_secs_f32();
        self.dt_sum += self.dt;
        self.frame_count += 1.0;
        self.time = Instant::now();
    }

    /// Updates app state based on input state.
    pub fn update_state_after_inputs(&mut self, renderer: &mut VulkanoWindowRenderer) {
        // Zoom in or out.
        if self.input_state.scroll_delta > 0. {
            self.scale /= 1.05;
        } else if self.input_state.scroll_delta < 0. {
            self.scale *= 1.05;
        }

        // Move speed scaled by zoom level.
        let move_speed = MOVE_SPEED * self.dt * self.scale.x;

        // Panning.
        if self.input_state.pan_up {
            self.translation += Vector2::new(0.0, move_speed);
        }
        if self.input_state.pan_down {
            self.translation += Vector2::new(0.0, -move_speed);
        }
        if self.input_state.pan_right {
            self.translation += Vector2::new(move_speed, 0.0);
        }
        if self.input_state.pan_left {
            self.translation += Vector2::new(-move_speed, 0.0);
        }

        // Toggle between Julia and Mandelbrot.
        if self.input_state.toggle_julia {
            self.is_julia = !self.is_julia;
        }

        // Toggle c.
        if self.input_state.toggle_c {
            self.is_c_paused = !self.is_c_paused;
        }

        // Update c.
        if !self.is_c_paused {
            // Scale normalized mouse pos between -1.0 and 1.0.
            let mouse_pos = self.input_state.normalized_mouse_pos() * 2.0 - Vector2::new(1.0, 1.0);
            // Scale by our zoom (scale) level so when zooming in the movement on Julia is not so
            // drastic.
            self.c = mouse_pos * self.scale.x;
        }

        // Update how many iterations we have.
        if self.input_state.increase_iterations {
            self.max_iters += 1;
        }
        if self.input_state.decrease_iterations {
            if self.max_iters as i32 - 1 <= 0 {
                self.max_iters = 0;
            } else {
                self.max_iters -= 1;
            }
        }

        // Randomize our palette.
        if self.input_state.randomize_palette {
            self.fractal_pipeline.randomize_palette();
        }

        // Toggle full-screen.
        if self.input_state.toggle_full_screen {
            let is_full_screen = renderer.window().fullscreen().is_some();
            renderer.window().set_fullscreen(if !is_full_screen {
                Some(Fullscreen::Borderless(renderer.window().current_monitor()))
            } else {
                None
            });
        }
    }

    /// Update input state.
    pub fn handle_input(&mut self, window_size: [f32; 2], event: &Event<()>) {
        self.input_state.handle_input(window_size, event);
    }

    /// Reset input state at the end of the frame.
    pub fn reset_input_state(&mut self) {
        self.input_state.reset()
    }
}

fn state_is_pressed(state: ElementState) -> bool {
    match state {
        ElementState::Pressed => true,
        ElementState::Released => false,
    }
}

/// Just a very simple input state (mappings). Winit only has `Pressed` and `Released` events, thus
/// continuous movement needs toggles. Panning is one of those things where continuous movement
/// feels better.
struct InputState {
    pub window_size: [f32; 2],
    pub pan_up: bool,
    pub pan_down: bool,
    pub pan_right: bool,
    pub pan_left: bool,
    pub increase_iterations: bool,
    pub decrease_iterations: bool,
    pub randomize_palette: bool,
    pub toggle_full_screen: bool,
    pub toggle_julia: bool,
    pub toggle_c: bool,
    pub should_quit: bool,
    pub scroll_delta: f32,
    pub mouse_pos: Vector2<f32>,
}

impl InputState {
    fn new() -> InputState {
        InputState {
            window_size: [
                WindowDescriptor::default().width,
                WindowDescriptor::default().height,
            ],
            pan_up: false,
            pan_down: false,
            pan_right: false,
            pan_left: false,
            increase_iterations: false,
            decrease_iterations: false,
            randomize_palette: false,
            toggle_full_screen: false,
            toggle_julia: false,
            toggle_c: false,
            should_quit: false,
            scroll_delta: 0.0,
            mouse_pos: Vector2::new(0.0, 0.0),
        }
    }

    fn normalized_mouse_pos(&self) -> Vector2<f32> {
        Vector2::new(
            (self.mouse_pos.x / self.window_size[0]).clamp(0.0, 1.0),
            (self.mouse_pos.y / self.window_size[1]).clamp(0.0, 1.0),
        )
    }

    /// Resets values that should be reset. All incremental mappings and toggles should be reset.
    fn reset(&mut self) {
        *self = InputState {
            scroll_delta: 0.0,
            toggle_full_screen: false,
            toggle_julia: false,
            toggle_c: false,
            randomize_palette: false,
            increase_iterations: false,
            decrease_iterations: false,
            ..*self
        }
    }

    fn handle_input(&mut self, window_size: [f32; 2], event: &Event<()>) {
        self.window_size = window_size;
        if let Event::WindowEvent { event, .. } = event {
            match event {
                WindowEvent::KeyboardInput { event, .. } => self.on_keyboard_event(event),
                WindowEvent::MouseInput { state, button, .. } => {
                    self.on_mouse_click_event(*state, *button)
                }
                WindowEvent::CursorMoved { position, .. } => self.on_cursor_moved_event(position),
                WindowEvent::MouseWheel { delta, .. } => self.on_mouse_wheel_event(delta),
                _ => {}
            }
        }
    }

    /// Matches keyboard events to our defined inputs.
    fn on_keyboard_event(&mut self, event: &KeyEvent) {
        match event.logical_key.as_ref() {
            Key::Named(NamedKey::Escape) => self.should_quit = state_is_pressed(event.state),
            Key::Character("w") => self.pan_up = state_is_pressed(event.state),
            Key::Character("a") => self.pan_left = state_is_pressed(event.state),
            Key::Character("s") => self.pan_down = state_is_pressed(event.state),
            Key::Character("d") => self.pan_right = state_is_pressed(event.state),
            Key::Character("f") => self.toggle_full_screen = state_is_pressed(event.state),
            Key::Named(NamedKey::Enter) => self.randomize_palette = state_is_pressed(event.state),
            Key::Character("=") => self.increase_iterations = state_is_pressed(event.state),
            Key::Character("-") => self.decrease_iterations = state_is_pressed(event.state),
            Key::Named(NamedKey::Space) => self.toggle_julia = state_is_pressed(event.state),
            _ => (),
        }
    }

    /// Updates mouse scroll delta.
    fn on_mouse_wheel_event(&mut self, delta: &MouseScrollDelta) {
        let change = match delta {
            MouseScrollDelta::LineDelta(_x, y) => *y,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
        };
        self.scroll_delta += change;
    }

    /// Update mouse position
    fn on_cursor_moved_event(&mut self, pos: &PhysicalPosition<f64>) {
        self.mouse_pos = Vector2::new(pos.x as f32, pos.y as f32);
    }

    /// Update toggle julia state (if right mouse is clicked)
    fn on_mouse_click_event(&mut self, state: ElementState, mouse_btn: winit::event::MouseButton) {
        if mouse_btn == MouseButton::Right {
            self.toggle_c = state_is_pressed(state)
        }
    }
}
