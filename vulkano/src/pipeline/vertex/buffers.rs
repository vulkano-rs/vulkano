// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::VertexMemberInfo;
use crate::pipeline::shader::ShaderInterface;
use crate::pipeline::vertex::IncompatibleVertexDefinitionError;
use crate::pipeline::vertex::Vertex;
use crate::pipeline::vertex::VertexDefinition;
use crate::pipeline::vertex::VertexInput;
use crate::pipeline::vertex::VertexInputAttribute;
use crate::pipeline::vertex::VertexInputBinding;
use crate::pipeline::vertex::VertexInputRate;
use crate::DeviceSize;
use std::mem;

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
