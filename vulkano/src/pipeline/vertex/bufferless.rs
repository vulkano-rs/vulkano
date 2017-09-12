// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::iter;

use pipeline::vertex::AttributeInfo;
use pipeline::vertex::IncompatibleVertexDefinitionError;
use pipeline::vertex::InputRate;
use pipeline::vertex::VertexDefinition;
use pipeline::vertex::VertexSource;
use buffer::BufferAccess;

/// Implementation of `VertexDefinition` for drawing with no buffers at all.
///
/// This is only useful if your shaders come up with vertex data on their own, e.g. by inspecting
/// `gl_VertexIndex`
pub struct BufferlessDefinition;

unsafe impl VertexSource<usize> for BufferlessDefinition {
    fn decode(&self, n: usize) -> (Vec<Box<BufferAccess + Sync + Send + 'static>>, usize, usize) {
        (Vec::new(), n, 1)
    }
}

unsafe impl<T> VertexSource<Vec<T>> for BufferlessDefinition {
    fn decode<'l>(&self, _: Vec<T>) -> (Vec<Box<BufferAccess + Sync + Send + 'static>>, usize, usize) {
        panic!("bufferless drawing should not be supplied with buffers")
    }
}

unsafe impl<I> VertexDefinition<I> for BufferlessDefinition {
    type BuffersIter = iter::Empty<(u32, usize, InputRate)>;
    type AttribsIter = iter::Empty<(u32, u32, AttributeInfo)>;
    fn definition(&self, _: &I) -> Result<(Self::BuffersIter, Self::AttribsIter), IncompatibleVertexDefinitionError> {
        Ok((iter::empty(), iter::empty()))
    }
}
