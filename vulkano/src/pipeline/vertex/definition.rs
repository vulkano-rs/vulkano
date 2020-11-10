// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::sync::Arc;

use buffer::BufferAccess;
use format::Format;
use pipeline::vertex::VertexMemberTy;
use vk;
use SafeDeref;

/// Trait for types that describe the definition of the vertex input used by a graphics pipeline.
pub unsafe trait VertexDefinition<I>:
    VertexSource<Vec<Arc<dyn BufferAccess + Send + Sync>>>
{
    /// Iterator that returns the offset, the stride (in bytes) and input rate of each buffer.
    type BuffersIter: ExactSizeIterator<Item = (u32, usize, InputRate)>;
    /// Iterator that returns the attribute location, buffer id, and infos.
    type AttribsIter: ExactSizeIterator<Item = (u32, u32, AttributeInfo)>;

    /// Builds the vertex definition to use to link this definition to a vertex shader's input
    /// interface.
    fn definition(
        &self,
        interface: &I,
    ) -> Result<(Self::BuffersIter, Self::AttribsIter), IncompatibleVertexDefinitionError>;
}

unsafe impl<I, T> VertexDefinition<I> for T
where
    T: SafeDeref,
    T::Target: VertexDefinition<I>,
{
    type BuffersIter = <T::Target as VertexDefinition<I>>::BuffersIter;
    type AttribsIter = <T::Target as VertexDefinition<I>>::AttribsIter;

    #[inline]
    fn definition(
        &self,
        interface: &I,
    ) -> Result<(Self::BuffersIter, Self::AttribsIter), IncompatibleVertexDefinitionError> {
        (**self).definition(interface)
    }
}

/// How the vertex source should be unrolled.
#[derive(Copy, Clone, Debug)]
#[repr(u32)]
pub enum InputRate {
    /// Each element of the source corresponds to a vertex.
    Vertex = vk::VERTEX_INPUT_RATE_VERTEX,
    /// Each element of the source corresponds to an instance.
    Instance = vk::VERTEX_INPUT_RATE_INSTANCE,
}

/// Information about a single attribute within a vertex.
/// TODO: change that API
#[derive(Copy, Clone, Debug)]
pub struct AttributeInfo {
    /// Number of bytes between the start of a vertex and the location of attribute.
    pub offset: usize,
    /// VertexMember type of the attribute.
    pub format: Format,
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
    fn decode(&self, L) -> (Vec<Box<dyn BufferAccess + Send + Sync>>, usize, usize);
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
