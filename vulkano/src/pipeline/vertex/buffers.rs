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
use crate::pipeline::vertex::Vertex;
use crate::pipeline::vertex::VertexDefinition;
use crate::pipeline::vertex::VertexInput;
use crate::pipeline::vertex::VertexInputAttribute;
use crate::pipeline::vertex::VertexInputBinding;
use crate::pipeline::vertex::VertexInputRate;
use crate::pipeline::vertex::VertexSource;
use crate::DeviceSize;
use std::mem;
use std::sync::Arc;

/// A vertex definition for any number of vertex and instance buffers.
#[derive(Clone, Default)]
pub struct BuffersDefinition(Vec<VertexBuffer>);

#[derive(Clone, Copy)]
struct VertexBuffer {
    info_fn: fn(&str) -> Option<VertexMemberInfo>,
    stride: u32,
    input_rate: VertexInputRate,
}

impl From<VertexBuffer> for VertexInputBinding {
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
    ) -> Result<VertexInput, IncompatibleVertexDefinitionError> {
        let bindings = self
            .0
            .iter()
            .enumerate()
            .map(|(binding, &buffer)| (binding as u32, buffer.into()));
        let mut attributes: Vec<(u32, VertexInputAttribute)> = Vec::new();

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
                    VertexInputAttribute {
                        binding,
                        format: element.format,
                        offset: offset as u32,
                    },
                ));
                offset += element.format.size().unwrap();
            }
        }

        Ok(VertexInput::new(bindings, attributes))
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
            let items = (source.size() / buffer.stride as DeviceSize) as usize;
            let (items, count) = match buffer.input_rate {
                VertexInputRate::Vertex => (items, &mut vertices),
                VertexInputRate::Instance { divisor } => (
                    if divisor == 0 {
                        // A divisor of 0 means the same instance data is used for all instances,
                        // so we can draw any number of instances from a single element.
                        // The buffer must contain at least one element though.
                        if items == 0 {
                            0
                        } else {
                            usize::MAX
                        }
                    } else {
                        // If divisor is 2, we use only half the amount of data from the source buffer,
                        // so the number of instances that can be drawn is twice as large.
                        items * divisor as usize
                    },
                    &mut instances,
                ),
            };

            if let Some(count) = count {
                *count = std::cmp::min(*count, items);
            } else {
                *count = Some(items);
            }
        }

        // TODO: find some way to let the user specify these when drawing
        (vertices.unwrap_or(1), instances.unwrap_or(1))
    }
}
