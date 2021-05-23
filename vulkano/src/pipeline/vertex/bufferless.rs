// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::BufferAccess;
use crate::pipeline::shader::ShaderInterface;
use crate::pipeline::vertex::AttributeInfo;
use crate::pipeline::vertex::IncompatibleVertexDefinitionError;
use crate::pipeline::vertex::InputRate;
use crate::pipeline::vertex::VertexDefinition;
use crate::pipeline::vertex::VertexSource;
use std::iter;

/// Implementation of `VertexDefinition` for drawing with no buffers at all.
///
/// This is only useful if your shaders come up with vertex data on their own, e.g. by inspecting
/// `gl_VertexIndex`
#[derive(Copy, Clone)]
pub struct BufferlessDefinition;

/// Value to be passed as the vertex source for bufferless draw commands.
///
/// Note that the concrete type of the graphics pipeline using `BufferlessDefinition` must be
/// visible to the command buffer builder for this to be usable.
#[derive(Copy, Clone)]
pub struct BufferlessVertices {
    pub vertices: usize,
    pub instances: usize,
}

unsafe impl VertexSource<BufferlessVertices> for BufferlessDefinition {
    fn decode(
        &self,
        n: BufferlessVertices,
    ) -> (
        Vec<Box<dyn BufferAccess + Sync + Send + 'static>>,
        usize,
        usize,
    ) {
        (Vec::new(), n.vertices, n.instances)
    }
}

unsafe impl<T> VertexSource<Vec<T>> for BufferlessDefinition {
    fn decode<'l>(
        &self,
        _: Vec<T>,
    ) -> (
        Vec<Box<dyn BufferAccess + Sync + Send + 'static>>,
        usize,
        usize,
    ) {
        panic!("bufferless drawing should not be supplied with buffers")
    }
}

unsafe impl VertexDefinition for BufferlessDefinition {
    type BuffersIter = iter::Empty<(u32, usize, InputRate)>;
    type AttribsIter = iter::Empty<(u32, u32, AttributeInfo)>;
    fn definition(
        &self,
        _: &ShaderInterface,
    ) -> Result<(Self::BuffersIter, Self::AttribsIter), IncompatibleVertexDefinitionError> {
        Ok((iter::empty(), iter::empty()))
    }
}
