// Mostly a copy from https://github.com/bevyengine/bevy/tree/main/crates/bevy_window, modified to fit Vulkano.

// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::context::VulkanoContext;
use crate::renderer::VulkanoWindowRenderer;
use std::collections::HashMap;
use vulkano::swapchain::{PresentMode, SwapchainCreateInfo};
use winit::dpi::LogicalSize;

/// A struct organizing windows and their corresponding renderers. This makes it easy to handle multiple windows.
///
/// ## Example
///```
/// use vulkano_util::context::{VulkanoConfig, VulkanoContext};
/// use winit::event_loop::EventLoop;
/// use vulkano_util::window::VulkanoWindows;
///
/// #[test]
/// fn test() {
///    let context = VulkanoContext::new(VulkanoConfig::default());
///    let event_loop = EventLoop::new();
///    let mut vulkano_windows = VulkanoWindows::default();
///    vulkano_windows.create_window(&event_loop, &context, &Default::default(), Default::default());
///    vulkano_windows.create_window(&event_loop, &context, &Default::default(), Default::default());
/// // You should now have two windows
/// }
/// ```
#[derive(Default)]
pub struct VulkanoWindows {
    windows: HashMap<winit::window::WindowId, VulkanoWindowRenderer>,
    primary: Option<winit::window::WindowId>,
}

impl VulkanoWindows {
    pub fn create_window(
        &mut self,
        event_loop: &winit::event_loop::EventLoopWindowTarget<()>,
        vulkano_context: &VulkanoContext,
        window_descriptor: &WindowDescriptor,
        swapchain_create_info_overriders: SwapchainCreateInfo,
    ) {
        #[cfg(target_os = "windows")]
        let mut winit_window_builder = {
            use winit::platform::windows::WindowBuilderExtWindows;
            winit::window::WindowBuilder::new().with_drag_and_drop(false)
        };

        #[cfg(not(target_os = "windows"))]
        let mut winit_window_builder = winit::window::WindowBuilder::new();

        winit_window_builder = match window_descriptor.mode {
            WindowMode::BorderlessFullscreen => winit_window_builder.with_fullscreen(Some(
                winit::window::Fullscreen::Borderless(event_loop.primary_monitor()),
            )),
            WindowMode::Fullscreen => {
                winit_window_builder.with_fullscreen(Some(winit::window::Fullscreen::Exclusive(
                    get_best_videomode(&event_loop.primary_monitor().unwrap()),
                )))
            }
            WindowMode::SizedFullscreen => winit_window_builder.with_fullscreen(Some(
                winit::window::Fullscreen::Exclusive(get_fitting_videomode(
                    &event_loop.primary_monitor().unwrap(),
                    window_descriptor.width as u32,
                    window_descriptor.height as u32,
                )),
            )),
            _ => {
                let WindowDescriptor {
                    width,
                    height,
                    position,
                    scale_factor_override,
                    ..
                } = window_descriptor;

                if let Some(position) = position {
                    if let Some(sf) = scale_factor_override {
                        winit_window_builder = winit_window_builder.with_position(
                            winit::dpi::LogicalPosition::new(
                                position[0] as f64,
                                position[1] as f64,
                            )
                            .to_physical::<f64>(*sf),
                        );
                    } else {
                        winit_window_builder =
                            winit_window_builder.with_position(winit::dpi::LogicalPosition::new(
                                position[0] as f64,
                                position[1] as f64,
                            ));
                    }
                }
                if let Some(sf) = scale_factor_override {
                    winit_window_builder.with_inner_size(
                        winit::dpi::LogicalSize::new(*width, *height).to_physical::<f64>(*sf),
                    )
                } else {
                    winit_window_builder
                        .with_inner_size(winit::dpi::LogicalSize::new(*width, *height))
                }
            }
            .with_resizable(window_descriptor.resizable)
            .with_decorations(window_descriptor.decorations)
            .with_transparent(window_descriptor.transparent),
        };

        let constraints = window_descriptor.resize_constraints.check_constraints();
        let min_inner_size = LogicalSize {
            width: constraints.min_width,
            height: constraints.min_height,
        };
        let max_inner_size = LogicalSize {
            width: constraints.max_width,
            height: constraints.max_height,
        };

        let winit_window_builder =
            if constraints.max_width.is_finite() && constraints.max_height.is_finite() {
                winit_window_builder
                    .with_min_inner_size(min_inner_size)
                    .with_max_inner_size(max_inner_size)
            } else {
                winit_window_builder.with_min_inner_size(min_inner_size)
            };

        #[allow(unused_mut)]
        let mut winit_window_builder = winit_window_builder.with_title(&window_descriptor.title);

        let winit_window = winit_window_builder.build(event_loop).unwrap();

        if window_descriptor.cursor_locked {
            match winit_window.set_cursor_grab(true) {
                Ok(_) => {}
                Err(winit::error::ExternalError::NotSupported(_)) => {}
                Err(err) => Err(err).unwrap(),
            }
        }

        winit_window.set_cursor_visible(window_descriptor.cursor_visible);

        if self.primary.is_none() {
            self.primary = Some(winit_window.id());
        }

        self.windows.insert(
            winit_window.id(),
            VulkanoWindowRenderer::new(
                vulkano_context,
                winit_window,
                window_descriptor,
                swapchain_create_info_overriders,
            ),
        );
    }

    /// Get a mutable reference to the primary window's renderer
    pub fn get_primary_renderer_mut(&mut self) -> Option<&mut VulkanoWindowRenderer> {
        if self.primary.is_some() {
            self.get_renderer_mut(self.primary.unwrap())
        } else {
            None
        }
    }

    /// Get a reference to the primary window's renderer
    pub fn get_primary_renderer(&self) -> Option<&VulkanoWindowRenderer> {
        if self.primary.is_some() {
            self.get_renderer(self.primary.unwrap())
        } else {
            None
        }
    }

    /// Get a reference to the primary winit window
    pub fn get_primary_window(&self) -> Option<&winit::window::Window> {
        if self.primary.is_some() {
            self.get_window(self.primary.unwrap())
        } else {
            None
        }
    }

    /// Get a mutable reference to the renderer by winit window id
    pub fn get_renderer_mut(
        &mut self,
        id: winit::window::WindowId,
    ) -> Option<&mut VulkanoWindowRenderer> {
        self.windows.get_mut(&id).and_then(|v| Some(v))
    }

    /// Get a reference to the renderer by winit window id
    pub fn get_renderer(&self, id: winit::window::WindowId) -> Option<&VulkanoWindowRenderer> {
        self.windows.get(&id).and_then(|v| Some(v))
    }

    /// Get a reference to the winit window by winit window id
    pub fn get_window(&self, id: winit::window::WindowId) -> Option<&winit::window::Window> {
        self.windows
            .get(&id)
            .and_then(|v_window| Some(v_window.window()))
    }
}

pub fn get_fitting_videomode(
    monitor: &winit::monitor::MonitorHandle,
    width: u32,
    height: u32,
) -> winit::monitor::VideoMode {
    let mut modes = monitor.video_modes().collect::<Vec<_>>();

    fn abs_diff(a: u32, b: u32) -> u32 {
        if a > b {
            return a - b;
        }
        b - a
    }

    modes.sort_by(|a, b| {
        use std::cmp::Ordering::*;
        match abs_diff(a.size().width, width).cmp(&abs_diff(b.size().width, width)) {
            Equal => {
                match abs_diff(a.size().height, height).cmp(&abs_diff(b.size().height, height)) {
                    Equal => b.refresh_rate().cmp(&a.refresh_rate()),
                    default => default,
                }
            }
            default => default,
        }
    });

    modes.first().unwrap().clone()
}

pub fn get_best_videomode(monitor: &winit::monitor::MonitorHandle) -> winit::monitor::VideoMode {
    let mut modes = monitor.video_modes().collect::<Vec<_>>();
    modes.sort_by(|a, b| {
        use std::cmp::Ordering::*;
        match b.size().width.cmp(&a.size().width) {
            Equal => match b.size().height.cmp(&a.size().height) {
                Equal => b.refresh_rate().cmp(&a.refresh_rate()),
                default => default,
            },
            default => default,
        }
    });

    modes.first().unwrap().clone()
}

/// Defines the way a window is displayed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowMode {
    /// Creates a window that uses the given size.
    Windowed,
    /// Creates a borderless window that uses the full size of the screen.
    BorderlessFullscreen,
    /// Creates a fullscreen window that will render at desktop resolution.
    ///
    /// The app will use the closest supported size from the given size and scale it to fit the screen.
    SizedFullscreen,
    /// Creates a fullscreen window that uses the maximum supported size.
    Fullscreen,
}

/// Describes the information needed for creating a window.
#[derive(Debug, Clone)]
pub struct WindowDescriptor {
    /// The requested logical width of the window's client area.
    ///
    /// May vary from the physical width due to different pixel density on different monitors.
    pub width: f32,
    /// The requested logical height of the window's client area.
    ///
    /// May vary from the physical height due to different pixel density on different monitors.
    pub height: f32,
    /// The position on the screen that the window will be centered at.
    ///
    /// If set to `None`, some platform-specific position will be chosen.
    pub position: Option<[f32; 2]>,
    /// Sets minimum and maximum resize limits.
    pub resize_constraints: WindowResizeConstraints,
    /// Overrides the window's ratio of physical pixels to logical pixels.
    ///
    /// If there are some scaling problems on X11 try to set this option to `Some(1.0)`.
    pub scale_factor_override: Option<f64>,
    /// Sets the title that displays on the window top bar, on the system task bar and other OS specific places.
    pub title: String,
    /// The window's [`PresentMode`].
    ///
    /// Used to select whether or not VSync is used
    pub present_mode: PresentMode,
    /// Sets whether the window is resizable.
    pub resizable: bool,
    /// Sets whether the window should have borders and bars.
    pub decorations: bool,
    /// Sets whether the cursor is visible when the window has focus.
    pub cursor_visible: bool,
    /// Sets whether the window locks the cursor inside its borders when the window has focus.
    pub cursor_locked: bool,
    /// Sets the [`WindowMode`](crate::WindowMode).
    pub mode: WindowMode,
    /// Sets whether the background of the window should be transparent.
    pub transparent: bool,
}

impl Default for WindowDescriptor {
    fn default() -> Self {
        WindowDescriptor {
            title: "Vulkano App".to_string(),
            width: 1280.,
            height: 720.,
            position: None,
            resize_constraints: WindowResizeConstraints::default(),
            scale_factor_override: None,
            present_mode: PresentMode::Fifo,
            resizable: true,
            decorations: true,
            cursor_locked: false,
            cursor_visible: true,
            mode: WindowMode::Windowed,
            transparent: false,
        }
    }
}

/// The size limits on a window.
///
/// These values are measured in logical pixels, so the user's
/// scale factor does affect the size limits on the window.
/// Please note that if the window is resizable, then when the window is
/// maximized it may have a size outside of these limits. The functionality
/// required to disable maximizing is not yet exposed by winit.
#[derive(Debug, Clone, Copy)]
pub struct WindowResizeConstraints {
    pub min_width: f32,
    pub min_height: f32,
    pub max_width: f32,
    pub max_height: f32,
}

impl Default for WindowResizeConstraints {
    fn default() -> Self {
        Self {
            min_width: 180.,
            min_height: 120.,
            max_width: f32::INFINITY,
            max_height: f32::INFINITY,
        }
    }
}

impl WindowResizeConstraints {
    #[must_use]
    pub fn check_constraints(&self) -> Self {
        let WindowResizeConstraints {
            mut min_width,
            mut min_height,
            mut max_width,
            mut max_height,
        } = self;
        min_width = min_width.max(1.);
        min_height = min_height.max(1.);
        if max_width < min_width {
            println!(
                "The given maximum width {} is smaller than the minimum width {}",
                max_width, min_width
            );
            max_width = min_width;
        }
        if max_height < min_height {
            println!(
                "The given maximum height {} is smaller than the minimum height {}",
                max_height, min_height
            );
            max_height = min_height;
        }
        WindowResizeConstraints {
            min_width,
            min_height,
            max_width,
            max_height,
        }
    }
}
