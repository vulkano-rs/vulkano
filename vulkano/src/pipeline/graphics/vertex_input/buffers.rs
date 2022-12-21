// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::VertexMemberInfo;
use crate::{
    pipeline::graphics::vertex_input::{
        IncompatibleVertexDefinitionError, Vertex, VertexDefinition,
        VertexInputAttributeDescription, VertexInputBindingDescription, VertexInputRate,
        VertexInputState,
    },
    shader::ShaderInterface,
    DeviceSize,
};
use std::{
    fmt::{Debug, Error as FmtError, Formatter},
    mem,
};

/// A vertex definition for any number of vertex and instance buffers.
#[derive(Clone, Debug, Default)]
pub struct BuffersDefinition(Vec<VertexBuffer>);

#[derive(Clone, Copy)]
struct VertexBuffer {
    info_fn: fn(&str) -> Option<VertexMemberInfo>,
    stride: u32,
    input_rate: VertexInputRate,
}

impl Debug for VertexBuffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.debug_struct("VertexBuffer")
            .field("stride", &self.stride)
            .field("input_rate", &self.input_rate)
            .finish()
    }
}

impl From<VertexBuffer> for VertexInputBindingDescription {
    fn from(val: VertexBuffer) -> Self {
        Self {
            stride: val.stride,
            input_rate: val.input_rate,
        }
    }
}

impl BuffersDefinition {
    /// Constructs a new definition.
    #[inline]
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
    /// This requires the [`vertex_attribute_instance_rate_divisor`] feature has been enabled on
    /// the device, unless `divisor` is 1.
    ///
    /// `divisor` can be 0 if the [`vertex_attribute_instance_rate_zero_divisor`] feature is also
    /// enabled. This means that every vertex will use the same vertex and instance data.
    ///
    /// [`vertex_attribute_instance_rate_divisor`]: crate::device::Features::vertex_attribute_instance_rate_divisor
    /// [`vertex_attribute_instance_rate_zero_divisor`]: crate::device::Features::vertex_attribute_instance_rate_zero_divisor
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
    #[inline]
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

            // TODO: ShaderInterfaceEntryType does not properly support 64bit.
            //       Once it does the below logic around num_elements and num_locations
            //       might have to be updated.
            if infos.num_components() != element.ty.num_components
                || infos.num_elements != element.ty.num_locations()
            {
                return Err(IncompatibleVertexDefinitionError::FormatMismatch {
                    attribute: name.clone().into_owned(),
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
