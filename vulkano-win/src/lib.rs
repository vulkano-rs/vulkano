//! # Cargo features
//!
//! | Feature             | Description                                                                |
//! |---------------------|----------------------------------------------------------------------------|
//! | `raw_window_handle` | Include support for the [`raw_window_handle`] library. Enabled by default. |
//! | `winit`             | Include support for the [`winit`] library. Enabled by default.             |
//!
//! [`raw_window_handle`]: https://crates.io/crates/raw_window_handle
//! [`winit`]: https://crates.io/crates/winit

#![deprecated(
    since = "0.34.0",
    note = "vulkano-win is deprecated, use `Surface::required_extensions` and \
    `Surface::from_window` instead"
)]
#![doc(html_logo_url = "https://raw.githubusercontent.com/vulkano-rs/vulkano/master/logo.png")]
#![allow(clippy::missing_safety_doc)]
#![warn(rust_2018_idioms, rust_2021_compatibility)]

#[cfg(feature = "raw-window-handle")]
mod raw_window_handle;
#[cfg(feature = "raw-window-handle")]
pub use crate::raw_window_handle::*;

#[cfg(feature = "winit")]
mod winit;
#[cfg(feature = "winit")]
pub use crate::winit::*;

#[cfg(feature = "raw-window-handle_")]
#[deprecated(
    since = "0.33.0",
    note = "the `raw-window-handle_` feature is deprecated, use `raw-window-handle` instead"
)]
mod raw_window_handle;
#[cfg(feature = "raw-window-handle_")]
pub use crate::raw_window_handle::*;

#[cfg(feature = "winit_")]
#[deprecated(
    since = "0.33.0",
    note = "the `winit_` feature is deprecated, use `winit` instead"
)]
mod winit;
#[cfg(feature = "winit_")]
pub use crate::winit::*;
