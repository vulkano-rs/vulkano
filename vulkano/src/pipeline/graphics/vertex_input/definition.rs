// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Definition for creating a [`VertexInputState`] based on a [`ShaderInterface`].
//!
//! # Implementing `Vertex`
//!
//! The implementations of the `VertexDefinition` trait that are provided by vulkano require you to
//! use a buffer whose content is `[V]` where `V` implements the `Vertex` trait.
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

use crate::format::Format;
use crate::pipeline::graphics::vertex_input::vertex::VertexMemberInfo;
use crate::pipeline::graphics::vertex_input::Vertex;
use crate::pipeline::graphics::vertex_input::VertexInputAttributeDescription;
use crate::pipeline::graphics::vertex_input::VertexInputBindingDescription;
use crate::pipeline::graphics::vertex_input::VertexInputRate;
use crate::pipeline::graphics::vertex_input::VertexInputState;
use crate::pipeline::graphics::vertex_input::VertexMemberTy;
use crate::shader::ShaderInterface;
use crate::DeviceSize;
use std::error;
use std::fmt;
use std::mem;

/// Trait for types that can create a [`VertexInputState`] from a [`ShaderInterface`].
pub unsafe trait VertexDefinition {
    /// Builds the vertex definition to use to link this definition to a vertex shader's input
    /// interface.
    // TODO: remove error return, move checks to GraphicsPipelineBuilder
    fn definition(
        &self,
        interface: &ShaderInterface,
    ) -> Result<VertexInputState, IncompatibleVertexDefinitionError>;
}

unsafe impl VertexDefinition for VertexInputState {
    fn definition(
        &self,
        interface: &ShaderInterface,
    ) -> Result<VertexInputState, IncompatibleVertexDefinitionError> {
        Ok(self.clone())
    }
}

/// Error that can happen when the vertex definition doesn't match the input of the vertex shader.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IncompatibleVertexDefinitionError {
    /// An attribute of the vertex shader is missing in the vertex source.
    MissingAttribute {
        /// Name of the missing attribute.
        attribute: String,
    },

    /// The format of an attribute does not match.
    FormatMismatch {
        /// Name of the attribute.
        attribute: String,
        /// The format in the vertex shader.
        shader: (Format, usize),
        /// The format in the vertex definition.
        definition: (VertexMemberTy, usize),
    },
}

impl error::Error for IncompatibleVertexDefinitionError {}

impl fmt::Display for IncompatibleVertexDefinitionError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            IncompatibleVertexDefinitionError::MissingAttribute { .. } => {
                write!(fmt, "an attribute is missing",)
            }
            IncompatibleVertexDefinitionError::FormatMismatch { .. } => {
                write!(fmt, "the format of an attribute does not match")
            }
        }
    }
}

/// A vertex definition for any number of vertex and instance buffers.
#[derive(Clone, Debug, Default)]
pub struct BuffersDefinition(Vec<VertexBuffer>);

#[derive(Clone, Copy)]
struct VertexBuffer {
    info_fn: fn(&str) -> Option<VertexMemberInfo>,
    stride: u32,
    input_rate: VertexInputRate,
}

impl std::fmt::Debug for VertexBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VertexBuffer")
            .field("stride", &self.stride)
            .field("input_rate", &self.input_rate)
            .finish()
    }
}

impl From<VertexBuffer> for VertexInputBindingDescription {
    #[inline]
    fn from(val: VertexBuffer) -> Self {
        Self {
            stride: val.stride,
            input_rate: val.input_rate,
        }
    }
}

impl BuffersDefinition {
    /// Constructs a new definition.
    pub fn new() -> Self {
        BuffersDefinition(Vec::new())
    }

    /// Adds a new vertex buffer containing elements of type `V` to the definition.
    pub fn vertex<V: Vertex>(mut self) -> Self {
        self.0.push(VertexBuffer {
            info_fn: V::member,
            stride: mem::size_of::<V>() as u32,
            input_rate: VertexInputRate::Vertex,
        });
        self
    }

    /// Adds a new instance buffer containing elements of type `V` to the definition.
    pub fn instance<V: Vertex>(mut self) -> Self {
        self.0.push(VertexBuffer {
            info_fn: V::member,
            stride: mem::size_of::<V>() as u32,
            input_rate: VertexInputRate::Instance { divisor: 1 },
        });
        self
    }

    /// Adds a new instance buffer containing elements of type `V` to the definition, with the
    /// specified input rate divisor.
    ///
    /// This requires the
    /// [`vertex_attribute_instance_rate_divisor`](crate::device::Features::vertex_attribute_instance_rate_divisor)
    /// feature has been enabled on the device, unless `divisor` is 1.
    ///
    /// `divisor` can be 0 if the
    /// [`vertex_attribute_instance_rate_zero_divisor`](crate::device::Features::vertex_attribute_instance_rate_zero_divisor)
    /// feature is also enabled. This means that every vertex will use the same vertex and instance
    /// data.
    pub fn instance_with_divisor<V: Vertex>(mut self, divisor: u32) -> Self {
        self.0.push(VertexBuffer {
            info_fn: V::member,
            stride: mem::size_of::<V>() as u32,
            input_rate: VertexInputRate::Instance { divisor },
        });
        self
    }
}

unsafe impl VertexDefinition for BuffersDefinition {
    fn definition(
        &self,
        interface: &ShaderInterface,
    ) -> Result<VertexInputState, IncompatibleVertexDefinitionError> {
        let bindings = self
            .0
            .iter()
            .enumerate()
            .map(|(binding, &buffer)| (binding as u32, buffer.into()));
        let mut attributes: Vec<(u32, VertexInputAttributeDescription)> = Vec::new();

        for element in interface.elements() {
            let name = element.name.as_ref().unwrap();

            let (infos, binding) = self
                .0
                .iter()
                .enumerate()
                .find_map(|(binding, buffer)| {
                    (buffer.info_fn)(name).map(|infos| (infos, binding as u32))
                })
                .ok_or_else(||
                    // TODO: move this check to GraphicsPipelineBuilder
                    IncompatibleVertexDefinitionError::MissingAttribute {
                        attribute: name.clone().into_owned(),
                    })?;

            if !infos.ty.matches(
                infos.array_size,
                element.format,
                element.location.end - element.location.start,
            ) {
                // TODO: move this check to GraphicsPipelineBuilder
                return Err(IncompatibleVertexDefinitionError::FormatMismatch {
                    attribute: name.clone().into_owned(),
                    shader: (
                        element.format,
                        (element.location.end - element.location.start) as usize,
                    ),
                    definition: (infos.ty, infos.array_size),
                });
            }

            let mut offset = infos.offset as DeviceSize;
            for location in element.location.clone() {
                attributes.push((
                    location,
                    VertexInputAttributeDescription {
                        binding,
                        format: element.format,
                        offset: offset as u32,
                    },
                ));
                offset += element.format.size().unwrap();
            }
        }

        Ok(VertexInputState::new()
            .bindings(bindings)
            .attributes(attributes))
    }
}
