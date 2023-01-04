//! # Cargo features
//!
//! | Feature              | Description                                                                |
//! |----------------------|----------------------------------------------------------------------------|
//! | `raw_window_handle_` | Include support for the [`raw_window_handle`] library. Enabled by default. |
//! | `winit_`             | Include support for the [`winit`] library. Enabled by default.             |
//!
//! [`raw_window_handle`]: https://crates.io/crates/raw_window_handle
//! [`winit`]: https://crates.io/crates/winit

#![doc(html_logo_url = "https://raw.githubusercontent.com/vulkano-rs/vulkano/master/logo.png")]
#![allow(clippy::missing_safety_doc)]
#![warn(rust_2018_idioms, rust_2021_compatibility)]

#[cfg(feature = "raw-window-handle_")]
mod raw_window_handle;
#[cfg(feature = "raw-window-handle_")]
pub use crate::raw_window_handle::*;

#[cfg(feature = "winit_")]
mod winit;
#[cfg(feature = "winit_")]
pub use crate::winit::*;
