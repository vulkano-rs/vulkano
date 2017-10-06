// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! # Vertex sources definition
//!
//! When you create a graphics pipeline object, you need to pass an object which indicates the
//! layout of the vertex buffer(s) that will serve as input for the vertex shader. This is done
//! by passing an implementation of the `VertexDefinition` trait.
//!
//! In addition to this, the object that you pass when you create the graphics pipeline must also
//! implement the `VertexSource` trait. This trait has a template parameter which corresponds to the
//! list of vertex buffers.
//!
//! The vulkano library provides some structs that already implement these traits.
//! The most common situation is a single vertex buffer and no instancing, in which case you can
//! pass a `SingleBufferDefinition` when you create the pipeline.
//!
//! # Implementing `Vertex`
//!
//! The implementations of the `VertexDefinition` trait that are provided by vulkano (like
//! `SingleBufferDefinition`) require you to use a buffer whose content is `[V]` where `V`
//! implements the `Vertex` trait.
//!
//! The `Vertex` trait is unsafe, but can be implemented on a struct with the `impl_vertex!`
//! macro.
//!
//! # Example
//!
//! ```ignore       // TODO:
//! # #[macro_use] extern crate vulkano
//! # fn main() {
//! # use std::sync::Arc;
//! # use vulkano::device::Device;
//! # use vulkano::device::Queue;
//! use vulkano::buffer::BufferAccess;
//! use vulkano::buffer::BufferUsage;
//! use vulkano::memory::HostVisible;
//! use vulkano::pipeline::vertex::;
//! # let device: Arc<Device> = return;
//! # let queue: Arc<Queue> = return;
//!
//! struct Vertex {
//!     position: [f32; 2]
//! }
//!
//! impl_vertex!(Vertex, position);
//!
//! let usage = BufferUsage {
//!     vertex_buffer: true,
//!     .. BufferUsage::none()
//! };
//!
//! let vertex_buffer = BufferAccess::<[Vertex], _>::array(&device, 128, &usage, HostVisible, &queue)
//!                                                     .expect("failed to create buffer");
//!
//! // TODO: finish example
//! # }
//! ```

pub use self::bufferless::BufferlessDefinition;
pub use self::bufferless::BufferlessVertices;
pub use self::definition::AttributeInfo;
pub use self::definition::IncompatibleVertexDefinitionError;
pub use self::definition::InputRate;
pub use self::definition::VertexDefinition;
pub use self::definition::VertexSource;
pub use self::impl_vertex::VertexMember;
pub use self::one_one::OneVertexOneInstanceDefinition;
pub use self::single::SingleBufferDefinition;
pub use self::two::TwoBuffersDefinition;
pub use self::vertex::Vertex;
pub use self::vertex::VertexMemberInfo;
pub use self::vertex::VertexMemberTy;

mod definition;
mod impl_vertex;
mod one_one;
mod single;
mod two;
mod vertex;
mod bufferless;
