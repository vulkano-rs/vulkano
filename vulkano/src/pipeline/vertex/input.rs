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
use crate::format::Format;
use crate::pipeline::shader::ShaderInterface;
use crate::pipeline::vertex::Vertex;
use crate::pipeline::vertex::VertexMemberTy;
use crate::SafeDeref;
use std::error;
use std::fmt;
use std::mem;
use std::sync::Arc;

/// A definition for the vertex input of a graphics pipeline.
#[derive(Clone, Default)]
pub struct VertexInput(Vec<VertexBuffer>);

impl VertexInput {
    /// Constructs a new definition.
    pub fn new() -> Self {
        VertexInput(Vec::new())
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

    /// Returns a list of vertex input bindings that link this definition to a vertex shader's input
    /// interface.
    pub fn definition(
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

    fn vertices_instances(&self, source: &[Box<dyn BufferAccess + Send + Sync>]) -> (usize, usize) {
        assert_eq!(source.len(), self.0.len());
        let mut vertices = None;
        let mut instances = None;

        for (buffer, source) in self.0.iter().zip(source) {
            let items = source.size() / buffer.stride as usize;
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
            attributes: Vec::new(),
            stride: val.stride,
            input_rate: val.input_rate,
        }
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

/// Extension trait of `VertexInput`. The `L` parameter is an acceptable vertex source for this
/// vertex input.
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

unsafe impl VertexSource<Vec<Arc<dyn BufferAccess + Send + Sync>>> for VertexInput {
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

unsafe impl<B> VertexSource<B> for VertexInput
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

unsafe impl<B1, B2> VertexSource<(B1, B2)> for VertexInput
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

/// Value to be passed as the vertex source for bufferless draw commands.
#[derive(Copy, Clone)]
pub struct BufferlessVertices {
    pub vertices: usize,
    pub instances: usize,
}

unsafe impl VertexSource<BufferlessVertices> for VertexInput {
    fn decode(
        &self,
        n: BufferlessVertices,
    ) -> (
        Vec<Box<dyn BufferAccess + Sync + Send + 'static>>,
        usize,
        usize,
    ) {
        assert!(self.0.is_empty());
        (Vec::new(), n.vertices, n.instances)
    }
}
