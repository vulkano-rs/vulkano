// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
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
pub use self::buffers::BuffersDefinition;
pub use self::definition::IncompatibleVertexDefinitionError;
pub use self::definition::VertexDefinition;
pub use self::definition::VertexSource;
pub use self::impl_vertex::VertexMember;
pub use self::vertex::Vertex;
pub use self::vertex::VertexMemberInfo;
pub use self::vertex::VertexMemberTy;
use crate::buffer::BufferAccess;
use crate::format::Format;
use fnv::FnvHashMap;
use std::convert::TryInto;

mod bufferless;
mod buffers;
mod definition;
mod impl_vertex;
mod vertex;

/// A description of the vertex input of a graphics pipeline.
#[derive(Clone, Debug, Default)]
pub struct VertexInput {
    bindings: FnvHashMap<u32, VertexInputBinding>,
    attributes: FnvHashMap<u32, VertexInputAttribute>,
}

impl VertexInput {
    /// Constructs a new `VertexInput` from the given bindings and attributes.
    ///
    /// # Panics
    ///
    /// Panics if any element of `attributes` refers to a binding number that is not provided in
    /// `bindings`.
    #[inline]
    pub fn new(
        bindings: impl IntoIterator<Item = (u32, VertexInputBinding)>,
        attributes: impl IntoIterator<Item = (u32, VertexInputAttribute)>,
    ) -> VertexInput {
        let bindings: FnvHashMap<_, _> = bindings.into_iter().collect();
        let attributes: FnvHashMap<_, _> = attributes.into_iter().collect();

        assert!(attributes
            .iter()
            .all(|(_, attr)| bindings.contains_key(&attr.binding)));

        VertexInput {
            bindings,
            attributes,
        }
    }

    /// Constructs a new empty `VertexInput`.
    #[inline]
    pub fn empty() -> VertexInput {
        VertexInput {
            bindings: Default::default(),
            attributes: Default::default(),
        }
    }

    /// Returns an iterator of the binding numbers and their descriptions.
    #[inline]
    pub fn bindings(&self) -> impl ExactSizeIterator<Item = (u32, &VertexInputBinding)> {
        self.bindings.iter().map(|(&key, val)| (key, val))
    }

    /// Returns an iterator of the attribute numbers and their descriptions.
    #[inline]
    pub fn attributes(&self) -> impl ExactSizeIterator<Item = (u32, &VertexInputAttribute)> {
        self.attributes.iter().map(|(&key, val)| (key, val))
    }

    /// Given an iterator of vertex buffers and their binding numbers, returns the maximum number
    /// of vertices and instances that can be drawn with them.
    ///
    /// # Panics
    ///
    /// Panics if the binding number of a provided vertex buffer does not exist in `self`.
    pub fn max_vertices_instances<'a>(
        &self,
        buffers: impl IntoIterator<Item = (u32, &'a dyn BufferAccess)>,
    ) -> (u32, u32) {
        let buffers = buffers.into_iter();
        let mut max_vertices = u32::MAX;
        let mut max_instances = u32::MAX;

        for (binding, buffer) in buffers {
            let binding_desc = &self.bindings[&binding];
            let mut num_elements = (buffer.size() / binding_desc.stride as usize)
                .try_into()
                .unwrap_or(u32::MAX);

            match binding_desc.input_rate {
                VertexInputRate::Vertex => {
                    max_vertices = max_vertices.min(num_elements);
                }
                VertexInputRate::Instance { divisor } => {
                    if divisor == 0 {
                        // A divisor of 0 means the same instance data is used for all instances,
                        // so we can draw any number of instances from a single element.
                        // The buffer must contain at least one element though.
                        if num_elements != 0 {
                            num_elements = u32::MAX;
                        }
                    } else {
                        // If divisor is 2, we use only half the amount of data from the source buffer,
                        // so the number of instances that can be drawn is twice as large.
                        num_elements = num_elements.saturating_mul(divisor);
                    }

                    max_instances = max_instances.min(num_elements);
                }
            };
        }

        (max_vertices, max_instances)
    }
}

/// Describes a single vertex buffer binding in a graphics pipeline.
#[derive(Clone, Debug)]
pub struct VertexInputBinding {
    /// The size of each element in the vertex buffer.
    pub stride: u32,
    /// How often the vertex input should advance to the next element.
    pub input_rate: VertexInputRate,
}

/// Describes how each vertex shader input attribute should be read from a vertex buffer.
///
/// An attribute can contain a maximum of four data elements, described by a particular `Format`.
/// For shader inputs that are larger than this, such as matrices or arrays, multiple separate
/// attributes should be used, with increasing offsets.
///
/// For example, to pass a `mat4` to the shader using sixteen `f32` as input, you would include four
/// attributes with `Format::R32G32B32A32Sfloat`, using the offset of the matrix from the start of
/// the vertex buffer element, plus 0, 16, 32, 48 respectively.
#[derive(Clone, Copy, Debug)]
pub struct VertexInputAttribute {
    /// The vertex buffer binding number that this attribute should take its data from.
    pub binding: u32,
    /// The size and type of the vertex data.
    pub format: Format,
    /// Number of bytes between the start of a vertex buffer element and the location of attribute.
    pub offset: u32,
}

/// How the vertex source should be unrolled.
#[derive(Clone, Copy, Debug)]
pub enum VertexInputRate {
    /// Each element of the source corresponds to a vertex.
    Vertex,

    /// Each element of the source corresponds to an instance.
    ///
    /// `divisor` indicates how many consecutive instances will use the same instance buffer data.
    /// This value must be 1, unless the
    /// [`vertex_attribute_instance_rate_divisor`](crate::device::Features::vertex_attribute_instance_rate_divisor)
    /// feature has been enabled on the device.
    ///
    /// `divisor` can be 0 if the
    /// [`vertex_attribute_instance_rate_zero_divisor`](crate::device::Features::vertex_attribute_instance_rate_zero_divisor)
    /// feature is also enabled. This means that every vertex will use the same vertex and instance
    /// data.
    Instance { divisor: u32 },
}

impl From<VertexInputRate> for ash::vk::VertexInputRate {
    #[inline]
    fn from(val: VertexInputRate) -> Self {
        match val {
            VertexInputRate::Vertex => ash::vk::VertexInputRate::VERTEX,
            VertexInputRate::Instance { .. } => ash::vk::VertexInputRate::INSTANCE,
        }
    }
}
