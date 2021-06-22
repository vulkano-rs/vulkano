// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::VertexMemberInfo;
use crate::buffer::BufferAccess;
use crate::pipeline::shader::ShaderInterface;
use crate::pipeline::vertex::IncompatibleVertexDefinitionError;
use crate::pipeline::vertex::InputRate;
use crate::pipeline::vertex::Vertex;
use crate::pipeline::vertex::VertexDefinition;
use crate::pipeline::vertex::VertexInputAttribute;
use crate::pipeline::vertex::VertexInputBinding;
use crate::pipeline::vertex::VertexSource;
use std::mem;
use std::sync::Arc;

/// A vertex definition for any number of vertex and instance buffers.
#[derive(Clone, Default)]
pub struct BuffersDefinition(Vec<VertexBuffer>);

#[derive(Clone, Copy)]
struct VertexBuffer {
    info_fn: fn(&str) -> Option<VertexMemberInfo>,
    stride: u32,
    input_rate: InputRate,
}

impl From<VertexBuffer> for VertexInputBinding {
    #[inline]
    fn from(val: VertexBuffer) -> Self {
        Self {
            attributes: Vec::new(),
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

    /// Adds a new buffer to the definition.
    pub fn push<V: Vertex>(mut self, input_rate: InputRate) -> Self {
        self.0.push(VertexBuffer {
            info_fn: V::member,
            stride: mem::size_of::<V>() as u32,
            input_rate,
        });
        self
    }
}

unsafe impl VertexDefinition for BuffersDefinition {
    fn definition(
        &self,
        interface: &ShaderInterface,
    ) -> Result<Vec<VertexInputBinding>, IncompatibleVertexDefinitionError> {
        let mut bindings: Vec<VertexInputBinding> =
            self.0.iter().map(|&buffer| buffer.into()).collect();

        for element in interface.elements() {
            let name = element.name.as_ref().unwrap();

            let (infos, binding) = self
                .0
                .iter()
                .enumerate()
                .find_map(|(binding, buffer)| (buffer.info_fn)(name).map(|infos| (infos, binding)))
                .ok_or_else(|| IncompatibleVertexDefinitionError::MissingAttribute {
                    attribute: name.clone().into_owned(),
                })?;
            let attributes = &mut bindings[binding].attributes;

            if !infos.ty.matches(
                infos.array_size,
                element.format,
                element.location.end - element.location.start,
            ) {
                return Err(IncompatibleVertexDefinitionError::FormatMismatch {
                    attribute: name.clone().into_owned(),
                    shader: (
                        element.format,
                        (element.location.end - element.location.start) as usize,
                    ),
                    definition: (infos.ty, infos.array_size),
                });
            }

            let mut offset = infos.offset;
            for location in element.location.clone() {
                attributes.push(VertexInputAttribute {
                    location,
                    format: element.format,
                    offset: offset as u32,
                });
                offset += element.format.size().unwrap();
            }
        }

        Ok(bindings)
    }
}

unsafe impl VertexSource<Vec<Arc<dyn BufferAccess + Send + Sync>>> for BuffersDefinition {
    #[inline]
    fn decode(
        &self,
        source: Vec<Arc<dyn BufferAccess + Send + Sync>>,
    ) -> (Vec<Box<dyn BufferAccess + Send + Sync>>, usize, usize) {
        let result = source
            .into_iter()
            .map(|source| Box::new(source) as Box<_>)
            .collect::<Vec<_>>();
        let (vertices, instances) = self.vertices_instances(&result);
        (result, vertices, instances)
    }
}

unsafe impl<B> VertexSource<B> for BuffersDefinition
where
    B: BufferAccess + Send + Sync + 'static,
{
    #[inline]
    fn decode(&self, source: B) -> (Vec<Box<dyn BufferAccess + Send + Sync>>, usize, usize) {
        let result = vec![Box::new(source) as Box<_>];
        let (vertices, instances) = self.vertices_instances(&result);
        (result, vertices, instances)
    }
}

unsafe impl<B1, B2> VertexSource<(B1, B2)> for BuffersDefinition
where
    B1: BufferAccess + Send + Sync + 'static,
    B2: BufferAccess + Send + Sync + 'static,
{
    #[inline]
    fn decode(&self, source: (B1, B2)) -> (Vec<Box<dyn BufferAccess + Send + Sync>>, usize, usize) {
        let result = vec![Box::new(source.0) as Box<_>, Box::new(source.1) as Box<_>];
        let (vertices, instances) = self.vertices_instances(&result);
        (result, vertices, instances)
    }
}

impl BuffersDefinition {
    fn vertices_instances(&self, source: &[Box<dyn BufferAccess + Send + Sync>]) -> (usize, usize) {
        assert_eq!(source.len(), self.0.len());
        let mut vertices = None;
        let mut instances = None;

        for (buffer, source) in self.0.iter().zip(source) {
            let len = source.size() / buffer.stride as usize;
            let count = match buffer.input_rate {
                InputRate::Vertex => &mut vertices,
                InputRate::Instance => &mut instances,
            };

            if let Some(count) = count {
                *count = std::cmp::min(*count, len);
            } else {
                *count = Some(len);
            }
        }

        // TODO: find some way to let the user specify these when drawing
        (vertices.unwrap_or(1), instances.unwrap_or(1))
    }
}
