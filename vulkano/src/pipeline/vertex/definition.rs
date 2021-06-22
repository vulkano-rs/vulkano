// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::BufferAccess;
use crate::format::Format;
use crate::pipeline::shader::ShaderInterface;
use crate::pipeline::vertex::VertexMemberTy;
use crate::SafeDeref;
use std::error;
use std::fmt;
use std::sync::Arc;

/// Trait for types that describe the definition of the vertex input used by a graphics pipeline.
pub unsafe trait VertexDefinition:
    VertexSource<Vec<Arc<dyn BufferAccess + Send + Sync>>>
{
    /// Builds the vertex definition to use to link this definition to a vertex shader's input
    /// interface.
    ///
    /// Returns a list of vertex input binding descriptions.
    fn definition(
        &self,
        interface: &ShaderInterface,
    ) -> Result<Vec<VertexInputBinding>, IncompatibleVertexDefinitionError>;
}

unsafe impl<T> VertexDefinition for T
where
    T: SafeDeref,
    T::Target: VertexDefinition,
{
    #[inline]
    fn definition(
        &self,
        interface: &ShaderInterface,
    ) -> Result<Vec<VertexInputBinding>, IncompatibleVertexDefinitionError> {
        (**self).definition(interface)
    }
}

/// How the vertex source should be unrolled.
#[derive(Copy, Clone, Debug)]
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

/// Information about a single attribute within a vertex buffer element.
/// TODO: change that API
#[derive(Copy, Clone, Debug)]
pub struct AttributeInfo {
    /// Number of bytes between the start of an element and the location of attribute.
    pub offset: u32,
    /// VertexMember type of the attribute.
    pub format: Format,
}

/// Describes a vertex input binding that serves as input to a graphics pipeline.
#[derive(Clone, Debug)]
pub struct VertexInputBinding {
    /// The input attributes that are to be taken from this binding.
    pub attributes: Vec<VertexInputAttribute>,
    /// The size of each element in the vertex buffer.
    pub stride: u32,
    /// How often the vertex input should advance to the next element.
    pub input_rate: VertexInputRate,
}

/// Describes a vertex input attribute that is read from an input binding in a graphics pipeline.
#[derive(Clone, Copy, Debug)]
pub struct VertexInputAttribute {
    /// The location in the shader interface that this attribute is to be bound to.
    pub location: u32,
    /// VertexMember type of the attribute.
    pub format: Format,
    /// Number of bytes between the start of an element and the location of attribute.
    pub offset: u32,
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
        write!(
            fmt,
            "{}",
            match *self {
                IncompatibleVertexDefinitionError::MissingAttribute { .. } =>
                    "an attribute is missing",
                IncompatibleVertexDefinitionError::FormatMismatch { .. } => {
                    "the format of an attribute does not match"
                }
            }
        )
    }
}

/// Extension trait of `VertexDefinition`. The `L` parameter is an acceptable vertex source for this
/// vertex definition.
pub unsafe trait VertexSource<L> {
    /// Checks and returns the list of buffers with offsets, number of vertices and number of instances.
    // TODO: return error if problem
    // TODO: better than a Vec
    // TODO: return a struct instead
    fn decode(&self, list: L) -> (Vec<Box<dyn BufferAccess + Send + Sync>>, usize, usize);
}

unsafe impl<L, T> VertexSource<L> for T
where
    T: SafeDeref,
    T::Target: VertexSource<L>,
{
    #[inline]
    fn decode(&self, list: L) -> (Vec<Box<dyn BufferAccess + Send + Sync>>, usize, usize) {
        (**self).decode(list)
    }
}
