// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Configures how data from vertex buffers is read into vertex shader input locations.
//!
//! The vertex input stage is the stage where data is read from a buffer and fed into the vertex
//! shader. After each invocation of the vertex shader, the pipeline then proceeds to the input
//! assembly stage.
//!
//! # Input locations and components
//!
//! Input data is assigned per shader input location. Locations are set by adding the `location`
//! layout qualifier to an input variable in GLSL. A single location contains four data elements,
//! named "components", which are each 32 bits in size. These correspond to the `x`, `y`, `z` and
//! `w` (or equivalently `r`, `g`, `b`, `a`) components of a `vec4` inside the shader.
//! A component can contain at most one value, and data types that are smaller than 32 bits will
//! still take up a whole component, so a single `i8vec4` variable will still take up all four
//! components in a location, even if not all bits are actually used.
//!
//! A variable may take up fewer than four components. For example, a single `float` takes up only
//! one component, a `vec2` takes up two, and so on. Using the `component` layout qualifier in GLSL,
//! it is possible to fit multiple variables into a single four-component location slot, as long
//! as the components of each variable don't overlap.
//!
//! If the input variable is an array, then it takes up a series of consecutive locations. Each
//! element of the array always starts at a new location, regardless of whether there is still room
//! in the previous one. So, for example, an array of three `vec2` takes three locations, since
//! `vec2` alone needs one location. An array can be decorated with the `component` qualifier as
//! well; this is equivalent to applying the qualifier to every element of the array. If elements do
//! not use all components in their locations, those free components can be filled with additional
//! variables, just like for non-array types.
//!
//! Matrices are laid out as if they were an array of column vectors. Thus, a `mat4x3` is laid out
//! as an array of four `vec3`s, `mat2x4` as two `vec4`s. As with individual vectors, each column of
//! the matrix uses up as many components of its location as there are rows in the matrix, and the
//! remaining components are available for additional variables as described above. However, it is
//! not possible to use the `component` qualifier on a matrix.
//!
//! If a 64-bit value is to be passed to a shader, it will take up two adjacent components. Vectors
//! of 64-bit values are correspondingly twice as large: `dvec2` takes up all four components of a
//! location, `dvec4` takes two full locations, while `dvec3` takes one full location and the first
//! two components of the next. An array or matrix of a 64-bit type is made up of multiple adjacent
//! 64-bit elements, just like for smaller types: each new element starts at a fresh location.
//!
//! # Input attributes
//!
//! An input attribute is a mapping between data in a vertex buffer and the locations and components
//! of the vertex shader.
//!
//! Input attributes are assigned on a per-location basis; it is not possible to assign attributes
//! to individual components. Instead, each attribute specifies up to four values to be read from
//! the vertex buffer at once, which are then mapped to the four components of the given location.
//! Like the texels in an image, each attribute's data format in a vertex buffer is described by a
//! [`Format`]. The input data doesn't have to be an actual color, the format simply describes the
//! type, size and layout of the data for the four input components. For example,
//! `Format::R32G32B32A32_SFLOAT` will read four `f32` values from the vertex buffer and assigns
//! them to the four components of the attribute's location.
//!
//! It is possible to specify a `Format` that contains less than four components. In this case, the
//! missing components are given default values: the first three components default to 0, while the
//! fourth defaults to 1. This means that you can, for example, store only the `x`, `y` and `z`
//! components of a vertex position in a vertex buffer, and have the vertex input state
//! automatically set the `w` value to 1 for you. An exception to this are 64-bit values: these do
//! *not* receive default values, meaning that components that are missing from the format are
//! assigned no value and must not be used in the shader at all.
//!
//! When matching attribute formats to shader input types, the following rules apply:
//! - Signed integers in the shader must have an attribute format with a `SINT` type.
//! - Unsigned integers in the shader must have an attribute format with a `UINT` type.
//! - Floating point values in the shader must have an attribute format with a type other than
//!   `SINT` or `UINT`. This includes `SFLOAT`, `UFLOAT` and `SRGB`, but also `SNORM`, `UNORM`,
//!   `SSCALED` and `USCALED`.
//! - 64-bit values in the shader must have a 64-bit attribute format.
//! - 32-bit and smaller values in the shader must have a 32-bit or smaller attribute format, but
//!   the exact number of bits doesn't matter. For example, `Format::R8G8B8A8_UNORM` can be used
//!   with a `vec4` in the shader.
//!
//! # Input bindings
//!
//! An input binding is a definition of a Vulkan buffer that contains the actual data from which
//! each input attribute is to be read. The buffer itself is referred to as a "vertex buffer", and
//! is set during drawing with the
//! [`bind_vertex_buffers`](crate::command_buffer::AutoCommandBufferBuilder::bind_vertex_buffers)
//! command.
//!
//! The data in a vertex buffer is typically arranged into an array, where each array element
//! contains the data for a single vertex shader invocation. When deciding which element read from
//! the vertex buffer for a given vertex and instance number, each binding has an "input rate".
//! If the input rate is `Vertex`, then the vertex input state advances to the next element of that
//! buffer each time a new vertex number is processed. Likewise, if the input rate is `Instance`,
//! it advances to the next element for each new instance number. Different bindings can have
//! different input rates, and it's also possible to have multiple bindings with the same input
//! rate.

pub use self::buffers::BuffersDefinition;
pub use self::collection::VertexBuffersCollection;
pub use self::definition::IncompatibleVertexDefinitionError;
pub use self::definition::VertexDefinition;
pub use self::impl_vertex::VertexMember;
pub use self::vertex::Vertex;
pub use self::vertex::VertexMemberInfo;
pub use self::vertex::VertexMemberTy;
use crate::device::Device;
use crate::format::Format;
use crate::pipeline::graphics::GraphicsPipelineCreationError;
use crate::pipeline::DynamicState;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::ptr;

mod buffers;
mod collection;
mod definition;
mod impl_vertex;
mod vertex;

/// The state in a graphics pipeline describing how the vertex input stage should behave.
#[derive(Clone, Debug, Default)]
pub struct VertexInputState {
    /// A description of the vertex buffers that the vertex input stage will read from.
    pub bindings: HashMap<u32, VertexInputBindingDescription>,

    /// Describes, for each shader input location, the mapping between elements in a vertex buffer
    /// and the components of that location in the shader.
    pub attributes: HashMap<u32, VertexInputAttributeDescription>,
}

impl VertexInputState {
    /// Constructs a new `VertexInputState` with no bindings or attributes.
    #[inline]
    pub fn new() -> VertexInputState {
        VertexInputState {
            bindings: Default::default(),
            attributes: Default::default(),
        }
    }

    /// Adds a single binding.
    #[inline]
    pub fn binding(mut self, binding: u32, description: VertexInputBindingDescription) -> Self {
        self.bindings.insert(binding, description);
        self
    }

    /// Sets all bindings.
    #[inline]
    pub fn bindings(
        mut self,
        bindings: impl IntoIterator<Item = (u32, VertexInputBindingDescription)>,
    ) -> Self {
        self.bindings = bindings.into_iter().collect();
        self
    }

    /// Adds a single attribute.
    #[inline]
    pub fn attribute(
        mut self,
        location: u32,
        description: VertexInputAttributeDescription,
    ) -> Self {
        self.attributes.insert(location, description);
        self
    }

    /// Sets all attributes.
    #[inline]
    pub fn attributes(
        mut self,
        attributes: impl IntoIterator<Item = (u32, VertexInputAttributeDescription)>,
    ) -> Self {
        self.attributes = attributes.into_iter().collect();
        self
    }

    pub(crate) fn to_vulkan(
        &self,
        device: &Device,
        dynamic_state_modes: &mut HashMap<DynamicState, bool>,
        binding_descriptions: &[ash::vk::VertexInputBindingDescription],
        attribute_descriptions: &[ash::vk::VertexInputAttributeDescription],
        binding_divisor_state: Option<&ash::vk::PipelineVertexInputDivisorStateCreateInfoEXT>,
    ) -> ash::vk::PipelineVertexInputStateCreateInfo {
        dynamic_state_modes.insert(DynamicState::VertexInput, false);

        ash::vk::PipelineVertexInputStateCreateInfo {
            p_next: if let Some(next) = binding_divisor_state {
                next as *const _ as *const _
            } else {
                ptr::null()
            },
            flags: ash::vk::PipelineVertexInputStateCreateFlags::empty(),
            vertex_binding_description_count: binding_descriptions.len() as u32,
            p_vertex_binding_descriptions: binding_descriptions.as_ptr(),
            vertex_attribute_description_count: attribute_descriptions.len() as u32,
            p_vertex_attribute_descriptions: attribute_descriptions.as_ptr(),
            ..Default::default()
        }
    }

    pub(crate) fn to_vulkan_bindings(
        &self,
        device: &Device,
    ) -> Result<SmallVec<[ash::vk::VertexInputBindingDescription; 8]>, GraphicsPipelineCreationError>
    {
        let binding_descriptions: SmallVec<[_; 8]> = self
            .bindings
            .iter()
            .map(|(&binding, binding_desc)| {
                if binding
                    >= device
                        .physical_device()
                        .properties()
                        .max_vertex_input_bindings
                {
                    return Err(
                        GraphicsPipelineCreationError::MaxVertexInputBindingsExceeded {
                            max: device
                                .physical_device()
                                .properties()
                                .max_vertex_input_bindings,
                            obtained: binding,
                        },
                    );
                }

                if binding_desc.stride
                    > device
                        .physical_device()
                        .properties()
                        .max_vertex_input_binding_stride
                {
                    return Err(
                        GraphicsPipelineCreationError::MaxVertexInputBindingStrideExceeded {
                            binding,
                            max: device
                                .physical_device()
                                .properties()
                                .max_vertex_input_binding_stride,
                            obtained: binding_desc.stride,
                        },
                    );
                }

                Ok(ash::vk::VertexInputBindingDescription {
                    binding,
                    stride: binding_desc.stride,
                    input_rate: binding_desc.input_rate.into(),
                })
            })
            .collect::<Result<_, _>>()?;

        if binding_descriptions.len()
            > device
                .physical_device()
                .properties()
                .max_vertex_input_bindings as usize
        {
            return Err(
                GraphicsPipelineCreationError::MaxVertexInputBindingsExceeded {
                    max: device
                        .physical_device()
                        .properties()
                        .max_vertex_input_bindings,
                    obtained: binding_descriptions.len() as u32,
                },
            );
        }

        Ok(binding_descriptions)
    }

    pub(crate) fn to_vulkan_attributes(
        &self,
        device: &Device,
    ) -> Result<
        SmallVec<[ash::vk::VertexInputAttributeDescription; 8]>,
        GraphicsPipelineCreationError,
    > {
        let attribute_descriptions: SmallVec<[_; 8]> = self
            .attributes
            .iter()
            .map(|(&location, attribute_desc)| {
                if !self.bindings.contains_key(&attribute_desc.binding) {
                    return Err(
                        GraphicsPipelineCreationError::VertexInputAttributeInvalidBinding {
                            location,
                            binding: attribute_desc.binding,
                        },
                    );
                }

                if attribute_desc.offset
                    > device
                        .physical_device()
                        .properties()
                        .max_vertex_input_attribute_offset
                {
                    return Err(
                        GraphicsPipelineCreationError::MaxVertexInputAttributeOffsetExceeded {
                            max: device
                                .physical_device()
                                .properties()
                                .max_vertex_input_attribute_offset,
                            obtained: attribute_desc.offset,
                        },
                    );
                }

                let format_features = device
                    .physical_device()
                    .format_properties(attribute_desc.format)
                    .buffer_features;

                if !format_features.vertex_buffer {
                    return Err(
                        GraphicsPipelineCreationError::VertexInputAttributeUnsupportedFormat {
                            location,
                            format: attribute_desc.format,
                        },
                    );
                }

                Ok(ash::vk::VertexInputAttributeDescription {
                    location,
                    binding: attribute_desc.binding,
                    format: attribute_desc.format.into(),
                    offset: attribute_desc.offset,
                })
            })
            .collect::<Result<_, _>>()?;

        if attribute_descriptions.len()
            > device
                .physical_device()
                .properties()
                .max_vertex_input_attributes as usize
        {
            return Err(
                GraphicsPipelineCreationError::MaxVertexInputAttributesExceeded {
                    max: device
                        .physical_device()
                        .properties()
                        .max_vertex_input_attributes,
                    obtained: attribute_descriptions.len(),
                },
            );
        }

        Ok(attribute_descriptions)
    }

    pub(crate) fn to_vulkan_binding_divisor_state(
        &self,
        binding_divisor_descriptions: &[ash::vk::VertexInputBindingDivisorDescriptionEXT],
    ) -> Option<ash::vk::PipelineVertexInputDivisorStateCreateInfoEXT> {
        if !binding_divisor_descriptions.is_empty() {
            Some(ash::vk::PipelineVertexInputDivisorStateCreateInfoEXT {
                vertex_binding_divisor_count: binding_divisor_descriptions.len() as u32,
                p_vertex_binding_divisors: binding_divisor_descriptions.as_ptr(),
                ..Default::default()
            })
        } else {
            None
        }
    }

    pub(crate) fn to_vulkan_binding_divisors(
        &self,
        device: &Device,
    ) -> Result<
        SmallVec<[ash::vk::VertexInputBindingDivisorDescriptionEXT; 8]>,
        GraphicsPipelineCreationError,
    > {
        self.bindings
            .iter()
            .filter_map(|(&binding, binding_desc)| match binding_desc.input_rate {
                VertexInputRate::Instance { divisor } if divisor != 1 => Some((binding, divisor)),
                _ => None,
            })
            .map(|(binding, divisor)| {
                if !device
                    .enabled_features()
                    .vertex_attribute_instance_rate_divisor
                {
                    return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                        feature: "vertex_attribute_instance_rate_divisor",
                        reason: "VertexInputRate::Instance::divisor was not 1",
                    });
                }

                if divisor == 0
                    && !device
                        .enabled_features()
                        .vertex_attribute_instance_rate_zero_divisor
                {
                    return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                        feature: "vertex_attribute_instance_rate_zero_divisor",
                        reason: "VertexInputRate::Instance::divisor was 0",
                    });
                }

                if divisor
                    > device
                        .physical_device()
                        .properties()
                        .max_vertex_attrib_divisor
                        .unwrap()
                {
                    return Err(
                        GraphicsPipelineCreationError::MaxVertexAttribDivisorExceeded {
                            binding,
                            max: device
                                .physical_device()
                                .properties()
                                .max_vertex_attrib_divisor
                                .unwrap(),
                            obtained: divisor,
                        },
                    );
                }

                Ok(ash::vk::VertexInputBindingDivisorDescriptionEXT { binding, divisor })
            })
            .collect()
    }
}

/// Describes a single vertex buffer binding.
#[derive(Clone, Debug)]
pub struct VertexInputBindingDescription {
    /// The number of bytes from the start of one element in the vertex buffer to the start of the
    /// next element. This can be simply the size of the data in each element, but larger strides
    /// are possible.
    pub stride: u32,

    /// How often the vertex input should advance to the next element.
    pub input_rate: VertexInputRate,
}

/// Describes a single vertex buffer attribute mapping.
#[derive(Clone, Copy, Debug)]
pub struct VertexInputAttributeDescription {
    /// The vertex buffer binding number that this attribute should take its data from.
    pub binding: u32,

    /// The size and type of the vertex data.
    pub format: Format,

    /// Number of bytes between the start of a vertex buffer element and the location of attribute.
    pub offset: u32,
}

/// How the vertex source should be unrolled.
#[derive(Clone, Copy, Debug)]
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
