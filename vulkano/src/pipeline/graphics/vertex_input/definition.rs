// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{
    VertexBufferDescription, VertexInputAttributeDescription, VertexInputBindingDescription,
};
use crate::{
    pipeline::graphics::vertex_input::VertexInputState, shader::ShaderInterface, DeviceSize,
    ValidationError,
};

/// Trait for types that can create a [`VertexInputState`] from a [`ShaderInterface`].
pub unsafe trait VertexDefinition {
    /// Builds the `VertexInputState` for the provided `interface`.
    fn definition(
        &self,
        interface: &ShaderInterface,
    ) -> Result<VertexInputState, Box<ValidationError>>;
}

unsafe impl VertexDefinition for &[VertexBufferDescription] {
    #[inline]
    fn definition(
        &self,
        interface: &ShaderInterface,
    ) -> Result<VertexInputState, Box<ValidationError>> {
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
            let name = element.name.as_ref().unwrap();

            let (infos, binding) = self
                .iter()
                .enumerate()
                .find_map(|(binding, buffer)| {
                    buffer
                        .members
                        .get(name.as_ref())
                        .map(|infos| (infos.clone(), binding as u32))
                })
                .ok_or_else(|| {
                    Box::new(ValidationError {
                        problem: format!(
                            "the shader interface contains a variable named \"{}\", \
                        but no such attribute exists in the vertex definition",
                            name,
                        )
                        .into(),
                        ..Default::default()
                    })
                })?;

            // TODO: ShaderInterfaceEntryType does not properly support 64bit.
            //       Once it does the below logic around num_elements and num_locations
            //       might have to be updated.
            if infos.num_components() != element.ty.num_components
                || infos.num_elements != element.ty.num_locations()
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "for the variable \"{}\", the number of locations and components \
                        required by the shader don't match the number of locations and components \
                        of the type provided in the vertex definition",
                        name,
                    )
                    .into(),
                    ..Default::default()
                }));
            }

            let mut offset = infos.offset as DeviceSize;
            let block_size = infos.format.block_size();
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
    ) -> Result<VertexInputState, Box<ValidationError>> {
        self.as_slice().definition(interface)
    }
}

unsafe impl VertexDefinition for Vec<VertexBufferDescription> {
    #[inline]
    fn definition(
        &self,
        interface: &ShaderInterface,
    ) -> Result<VertexInputState, Box<ValidationError>> {
        self.as_slice().definition(interface)
    }
}

unsafe impl VertexDefinition for VertexBufferDescription {
    #[inline]
    fn definition(
        &self,
        interface: &ShaderInterface,
    ) -> Result<VertexInputState, Box<ValidationError>> {
        std::slice::from_ref(self).definition(interface)
    }
}
