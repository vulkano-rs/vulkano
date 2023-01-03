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
//! # Examples
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
//!     ..BufferUsage::empty()
//! };
//!
//! let vertex_buffer = BufferAccess::<[Vertex], _>::array(&device, 128, &usage, HostVisible, &queue)
//!     .expect("failed to create buffer");
//!
//! // TODO: finish example
//! # }
//! ```

use super::{
    VertexBufferDescription, VertexInputAttributeDescription, VertexInputBindingDescription,
};
use crate::{
    pipeline::graphics::vertex_input::{VertexInputState, VertexMemberInfo},
    shader::{ShaderInterface, ShaderInterfaceEntryType},
    DeviceSize,
};
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
};

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
        _interface: &ShaderInterface,
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
        shader: ShaderInterfaceEntryType,
        /// The format in the vertex definition.
        definition: VertexMemberInfo,
    },
}

impl Error for IncompatibleVertexDefinitionError {}

impl Display for IncompatibleVertexDefinitionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            IncompatibleVertexDefinitionError::MissingAttribute { .. } => {
                write!(f, "an attribute is missing")
            }
            IncompatibleVertexDefinitionError::FormatMismatch { .. } => {
                write!(f, "the format of an attribute does not match")
            }
        }
    }
}

unsafe impl VertexDefinition for &[VertexBufferDescription] {
    #[inline]
    fn definition(
        &self,
        interface: &ShaderInterface,
    ) -> Result<VertexInputState, IncompatibleVertexDefinitionError> {
        let bindings = self.iter().enumerate().map(|(binding, buffer)| {
            (
                binding as u32,
                VertexInputBindingDescription {
                    stride: buffer.stride,
                    input_rate: buffer.input_rate,
                },
            )
        });
        let mut attributes: Vec<(u32, VertexInputAttributeDescription)> = Vec::new();

        for element in interface.elements() {
            let name = element.name.as_ref().unwrap().clone().into_owned();

            let (infos, binding) = self
                .iter()
                .enumerate()
                .find_map(|(binding, buffer)| {
                    buffer
                        .members
                        .get(&name)
                        .map(|infos| (infos.clone(), binding as u32))
                })
                .ok_or_else(||
                    // TODO: move this check to GraphicsPipelineBuilder
                    IncompatibleVertexDefinitionError::MissingAttribute {
                        attribute: name.clone(),
                    })?;

            // TODO: ShaderInterfaceEntryType does not properly support 64bit.
            //       Once it does the below logic around num_elements and num_locations
            //       might have to be updated.
            if infos.num_components() != element.ty.num_components
                || infos.num_elements != element.ty.num_locations()
            {
                return Err(IncompatibleVertexDefinitionError::FormatMismatch {
                    attribute: name,
                    shader: element.ty,
                    definition: infos,
                });
            }

            let mut offset = infos.offset as DeviceSize;
            let block_size = infos.format.block_size().unwrap();
            // Double precision formats can exceed a single location.
            // R64B64G64A64_SFLOAT requires two locations, so we need to adapt how we bind
            let location_range = if block_size > 16 {
                (element.location..element.location + 2 * element.ty.num_locations()).step_by(2)
            } else {
                (element.location..element.location + element.ty.num_locations()).step_by(1)
            };

            for location in location_range {
                attributes.push((
                    location,
                    VertexInputAttributeDescription {
                        binding,
                        format: infos.format,
                        offset: offset as u32,
                    },
                ));
                offset += block_size;
            }
        }

        Ok(VertexInputState::new()
            .bindings(bindings)
            .attributes(attributes))
    }
}

unsafe impl<const N: usize> VertexDefinition for [VertexBufferDescription; N] {
    #[inline]
    fn definition(
        &self,
        interface: &ShaderInterface,
    ) -> Result<VertexInputState, IncompatibleVertexDefinitionError> {
        self.as_slice().definition(interface)
    }
}

unsafe impl VertexDefinition for Vec<VertexBufferDescription> {
    #[inline]
    fn definition(
        &self,
        interface: &ShaderInterface,
    ) -> Result<VertexInputState, IncompatibleVertexDefinitionError> {
        self.as_slice().definition(interface)
    }
}

unsafe impl VertexDefinition for VertexBufferDescription {
    #[inline]
    fn definition(
        &self,
        interface: &ShaderInterface,
    ) -> Result<VertexInputState, IncompatibleVertexDefinitionError> {
        std::slice::from_ref(self).definition(interface)
    }
}
