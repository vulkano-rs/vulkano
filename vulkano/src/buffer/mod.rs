// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Location in memory that contains data.
//!
//! A Vulkan buffer is very similar to a buffer that you would use in programming languages in
//! general, in the sense that it is a location in memory that contains data. The difference
//! between a Vulkan buffer and a regular buffer is that the content of a Vulkan buffer is
//! accessible from the GPU.
//!
//! # High-level wrappers
//!
//! The low level implementation of a buffer is `UnsafeBuffer`. However, the vulkano library
//! provides high-level wrappers around that type that are specialized depending on the way you
//! are going to use it:
//!
//! - `CpuAccessBuffer` designates a buffer located in RAM and whose content can be directly
//!   written by your application.
//! - `DeviceLocalBuffer` designates a buffer located in video memory and whose content can't be
//!   written by your application. Accessing this buffer from the GPU is usually faster than the
//!   `CpuAccessBuffer`.
//! - `ImmutableBuffer` designates a buffer in video memory and whose content can only be
//!   written once. Compared to `DeviceLocalBuffer`, this buffer requires less processing on the
//!   CPU because we don't need to keep track of the reads and writes.
//!
//! If you have data that is modified at every single frame, you are encouraged to use a
//! `CpuAccessibleBuffer`. If you have data that is very rarely modified, you are encouraged to
//! use an `ImmutableBuffer` or a `DeviceLocalBuffer` instead.
//!
//! If you just want to get started, you can use the `CpuAccessibleBuffer` everywhere, as it is
//! the most flexible type of buffer.
//!
//! # Buffers usage
//!
//! When you create a buffer object, you have to specify its *usage*. In other words, you have to
//! specify the way it is going to be used. Trying to use a buffer in a way that wasn't specified
//! when you created it will result in an error.
//!
//! You can use buffers for the following purposes:
//!
//! - Can contain arbitrary data that can be transferred from/to other buffers and images.
//! - Can be read and modified from a shader.
//! - Can be used as a source of vertices and indices.
//! - Can be used as a source of list of models for draw indirect commands.
//!
//! Accessing a buffer from a shader can be done in the following ways:
//!
//! - As a uniform buffer. Uniform buffers are read-only.
//! - As a storage buffer. Storage buffers can be read and written.
//! - As a uniform texel buffer. Contrary to a uniform buffer, the data is interpreted by the
//!   GPU and can be for example normalized.
//! - As a storage texel buffer. Additionnally, some data formats can be modified with atomic
//!   operations.
//!
//! Using uniform/storage texel buffers requires creating a *buffer view*. See the `view` module
//! for how to create a buffer view.
//!

pub use self::cpu_access::CpuAccessibleBuffer;
pub use self::device_local::DeviceLocalBuffer;
pub use self::immutable::ImmutableBuffer;
pub use self::slice::BufferSlice;
pub use self::sys::BufferCreationError;
pub use self::sys::Usage as BufferUsage;
pub use self::traits::Buffer;
pub use self::traits::BufferInner;
pub use self::traits::TypedBuffer;
pub use self::traits::TrackedBuffer;
pub use self::view::BufferView;
pub use self::view::BufferViewRef;

pub mod cpu_access;
pub mod device_local;
pub mod immutable;
pub mod sys;
pub mod view;

mod slice;
mod traits;
