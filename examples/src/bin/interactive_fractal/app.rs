// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    fractal_compute_pipeline::FractalComputePipeline,
    renderer::{InterimImageView, RenderOptions, Renderer},
};
use cgmath::Vector2;
use std::time::Instant;
use vulkano::sync::GpuFuture;
use winit::{
    dpi::PhysicalPosition,
    event::{
        ElementState, Event, KeyboardInput, MouseButton, MouseScrollDelta, VirtualKeyCode,
        WindowEvent,
    },
};

const MAX_ITERS_INIT: u32 = 200;
const MOVE_SPEED: f32 = 0.5;

/// App for exploring Julia and Mandelbrot fractals
pub struct FractalApp {
    /// Pipeline that computes Mandelbrot & Julia fractals and writes them to an image
    fractal_pipeline: FractalComputePipeline,
    /// Toggle that flips between julia and mandelbrot
    pub is_julia: bool,
    /// Togglet thats stops the movement on Julia
    is_c_paused: bool,
    /// C is a constant input to Julia escape time algorithm (mouse position).
    c: Vector2<f32>,
    /// Our zoom level
    scale: Vector2<f32>,
    /// Our translation on the complex plane
    translation: Vector2<f32>,
    /// How far should the escape time algorithm run (higher = less performance, more accurate image)
    pub max_iters: u32,
    /// Time tracking, useful for frame independent movement
    time: Instant,
    dt: f32,
    dt_sum: f32,
    frame_count: f32,
    avg_fps: f32,
    /// Input state to handle mouse positions, continuous movement etc.
    input_state: InputState,
}

impl FractalApp {
    pub fn new(renderer: &Renderer) -> FractalApp {
        FractalApp {
            fractal_pipeline: FractalComputePipeline::new(renderer.queue()),
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
        "
        );
    }

    /// Run our compute pipeline and return a future of when the compute is finished
    pub fn compute(&mut self, image_target: InterimImageView) -> Box<dyn GpuFuture> {
        self.fractal_pipeline.compute(
            image_target,
            self.c,
            self.scale,
            self.translation,
            self.max_iters,
            self.is_julia,
        )
    }

    /// Should the app quit? (on esc)
    pub fn is_running(&self) -> bool {
        !self.input_state.should_quit
    }

    /// Return average fps
    pub fn avg_fps(&self) -> f32 {
        self.avg_fps
    }

    /// Delta time in milliseconds
    pub fn dt(&self) -> f32 {
        self.dt * 1000.0
    }

    /// Update times and dt at the end of each frame
    pub fn update_time(&mut self) {
        // Each second, update average fps & reset frame count & dt sum
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

    /// Updates app state based on input state
    pub fn update_state_after_inputs(&mut self, renderer: &mut Renderer) {
        // Zoom in or out
        if self.input_state.scroll_delta > 0. {
            self.scale /= 1.05;
        } else if self.input_state.scroll_delta < 0. {
            self.scale *= 1.05;
        }
        // Move speed scaled by zoom level
        let move_speed = MOVE_SPEED * self.dt * self.scale.x;
        // Panning
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
        // Toggle between julia and mandelbrot
        if self.input_state.toggle_julia {
            self.is_julia = !self.is_julia;
        }
        // Toggle c
        if self.input_state.toggle_c {
            self.is_c_paused = !self.is_c_paused;
        }
        // Update c
        if !self.is_c_paused {
            // Scale normalized mouse pos between -1.0 and 1.0;
            let mouse_pos = self.input_state.normalized_mouse_pos() * 2.0 - Vector2::new(1.0, 1.0);
            // Scale by our zoom (scale) level so when zooming in the movement on julia is not so drastic
            self.c = mouse_pos * self.scale.x;
        }
        // Update how many iterations we have
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
        // Randomize our palette
        if self.input_state.randomize_palette {
            self.fractal_pipeline.randomize_palette();
        }
        // Toggle full-screen
        if self.input_state.toggle_full_screen {
            renderer.toggle_full_screen()
        }
    }

    /// Update input state
    pub fn handle_input(&mut self, window_size: [u32; 2], event: &Event<()>) {
        self.input_state.handle_input(window_size, event);
    }

    /// reset input state at the end of frame
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

/// Just a very simple input state (mappings).
/// Winit only has Pressed and Released events, thus continuous movement needs toggles.
/// Panning is one of those where continuous movement feels better.
struct InputState {
    pub window_size: [u32; 2],
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
            window_size: RenderOptions::default().window_size,
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
            (self.mouse_pos.x / self.window_size[0] as f32).clamp(0.0, 1.0),
            (self.mouse_pos.y / self.window_size[1] as f32).clamp(0.0, 1.0),
        )
    }

    // Resets values that should be reset. All incremental mappings and toggles should be reset.
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

    fn handle_input(&mut self, window_size: [u32; 2], event: &Event<()>) {
        self.window_size = window_size;
        if let winit::event::Event::WindowEvent { event, .. } = event {
            match event {
                WindowEvent::KeyboardInput { input, .. } => self.on_keyboard_event(input),
                WindowEvent::MouseInput { state, button, .. } => {
                    self.on_mouse_click_event(*state, *button)
                }
                WindowEvent::CursorMoved { position, .. } => self.on_cursor_moved_event(position),
                WindowEvent::MouseWheel { delta, .. } => self.on_mouse_wheel_event(delta),
                _ => {}
            }
        }
    }

    /// Match keyboard event to our defined inputs
    fn on_keyboard_event(&mut self, input: &KeyboardInput) {
        if let Some(key_code) = input.virtual_keycode {
            match key_code {
                VirtualKeyCode::Escape => self.should_quit = state_is_pressed(input.state),
                VirtualKeyCode::W => self.pan_up = state_is_pressed(input.state),
                VirtualKeyCode::A => self.pan_left = state_is_pressed(input.state),
                VirtualKeyCode::S => self.pan_down = state_is_pressed(input.state),
                VirtualKeyCode::D => self.pan_right = state_is_pressed(input.state),
                VirtualKeyCode::F => self.toggle_full_screen = state_is_pressed(input.state),
                VirtualKeyCode::Return => self.randomize_palette = state_is_pressed(input.state),
                VirtualKeyCode::Equals => self.increase_iterations = state_is_pressed(input.state),
                VirtualKeyCode::Minus => self.decrease_iterations = state_is_pressed(input.state),
                VirtualKeyCode::Space => self.toggle_julia = state_is_pressed(input.state),
                _ => (),
            }
        }
    }

    /// Update mouse scroll delta
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
        match mouse_btn {
            MouseButton::Right => self.toggle_c = state_is_pressed(state),
            _ => (),
        };
    }
}
