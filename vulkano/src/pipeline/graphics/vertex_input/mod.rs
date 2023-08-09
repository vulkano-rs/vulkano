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

#[allow(deprecated)]
pub use self::{
    buffers::BuffersDefinition,
    collection::VertexBuffersCollection,
    definition::VertexDefinition,
    impl_vertex::VertexMember,
    vertex::{Vertex, VertexBufferDescription, VertexMemberInfo},
};
use crate::{
    device::Device,
    format::{Format, FormatFeatures},
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError,
};
use ahash::HashMap;

mod buffers;
mod collection;
mod definition;
mod impl_vertex;
mod vertex;

/// The state in a graphics pipeline describing how the vertex input stage should behave.
#[derive(Clone, Debug)]
pub struct VertexInputState {
    /// A description of the vertex buffers that the vertex input stage will read from.
    pub bindings: HashMap<u32, VertexInputBindingDescription>,

    /// Describes, for each shader input location, the mapping between elements in a vertex buffer
    /// and the components of that location in the shader.
    pub attributes: HashMap<u32, VertexInputAttributeDescription>,

    pub _ne: crate::NonExhaustive,
}

impl VertexInputState {
    /// Constructs a new `VertexInputState` with no bindings or attributes.
    #[inline]
    pub fn new() -> VertexInputState {
        VertexInputState {
            bindings: Default::default(),
            attributes: Default::default(),
            _ne: crate::NonExhaustive(()),
        }
    }

    /// Adds a single binding.
    #[inline]
    pub fn binding(mut self, binding: u32, description: VertexInputBindingDescription) -> Self {
        self.bindings.insert(binding, description);
        self
    }

    /// Sets all bindings.
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
    pub fn attributes(
        mut self,
        attributes: impl IntoIterator<Item = (u32, VertexInputAttributeDescription)>,
    ) -> Self {
        self.attributes = attributes.into_iter().collect();
        self
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let Self {
            bindings,
            attributes,
            _ne: _,
        } = self;

        let properties = device.physical_device().properties();

        if bindings.len() > properties.max_vertex_input_bindings as usize {
            return Err(Box::new(ValidationError {
                context: "bindings".into(),
                problem: "the length exceeds the `max_vertex_input_bindings` limit".into(),
                vuids: &[
                    "VUID-VkPipelineVertexInputStateCreateInfo-vertexBindingDescriptionCount-00613",
                ],
                ..Default::default()
            }));
        }

        // VUID-VkPipelineVertexInputStateCreateInfo-pVertexBindingDescriptions-00616
        // Ensured by HashMap.

        for (&binding, binding_desc) in bindings {
            binding_desc
                .validate(device)
                .map_err(|err| err.add_context(format!("bindings[{}]", binding)))?;
        }

        if attributes.len() > properties.max_vertex_input_attributes as usize {
            return Err(Box::new(ValidationError {
                context: "attributes".into(),
                problem: "the length exceeds the `max_vertex_input_attributes` limit".into(),
                vuids: &[
                    "VUID-VkPipelineVertexInputStateCreateInfo-vertexAttributeDescriptionCount-00614",
                ],
                ..Default::default()
            }));
        }

        for (&location, attribute_desc) in attributes {
            attribute_desc
                .validate(device)
                .map_err(|err| err.add_context(format!("attributes[{}]", location)))?;

            let &VertexInputAttributeDescription {
                binding,
                format,
                offset,
            } = attribute_desc;

            if location > properties.max_vertex_input_attributes {
                return Err(Box::new(ValidationError {
                    context: "attributes".into(),
                    problem: format!(
                        "the location {} exceeds the `max_vertex_input_attributes` limit",
                        location
                    )
                    .into(),
                    vuids: &["VUID-VkVertexInputAttributeDescription-location-00620"],
                    ..Default::default()
                }));
            }

            let binding_desc = bindings.get(&binding).ok_or_else(|| {
                Box::new(ValidationError {
                    problem: format!(
                        "`attributes[{}].binding` is not present in `bindings`",
                        binding
                    )
                    .into(),
                    vuids: &["VUID-VkPipelineVertexInputStateCreateInfo-binding-00615"],
                    ..Default::default()
                })
            })?;

            if device.enabled_extensions().khr_portability_subset
                && !device
                    .enabled_features()
                    .vertex_attribute_access_beyond_stride
                && offset as DeviceSize + format.block_size() > binding_desc.stride as DeviceSize
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "this device is a portability subset device, and \
                        `attributes[{0}].offset + attributes[{0}].format.block_size()` \
                        is greater than `bindings[attributes[{0}]].stride`",
                        location,
                    )
                    .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "vertex_attribute_access_beyond_stride",
                    )])]),
                    vuids: &["VUID-VkVertexInputAttributeDescription-vertexAttributeAccessBeyondStride-04457"],
                    ..Default::default()
                }));
            }
        }

        // When a format exceeds a single location (e.g. R64B64G64_SFLOAT),
        // the location following it needs to be empty.
        let unassigned_locations = attributes
            .iter()
            .filter(|&(_, attribute_desc)| attribute_desc.format.block_size() > 16)
            .map(|(location, _)| location + 1);

        for location in unassigned_locations {
            if !attributes.get(&location).is_none() {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`attributes[{}].format` takes up two locations, but \
                        `attributes` also contains a description for location {}",
                        location - 1, location,
                    )
                    .into(),
                    vuids: &["VUID-VkPipelineVertexInputStateCreateInfo-pVertexAttributeDescriptions-00617"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }
}

impl Default for VertexInputState {
    #[inline]
    fn default() -> Self {
        Self {
            bindings: HashMap::default(),
            attributes: HashMap::default(),
            _ne: crate::NonExhaustive(()),
        }
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

impl VertexInputBindingDescription {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self { stride, input_rate } = self;

        let properties = device.physical_device().properties();

        if stride > properties.max_vertex_input_binding_stride {
            return Err(Box::new(ValidationError {
                context: "stride".into(),
                problem: "exceeds the `max_vertex_input_binding_stride` limit".into(),
                vuids: &["VUID-VkVertexInputBindingDescription-stride-00619"],
                ..Default::default()
            }));
        }

        if device.enabled_extensions().khr_portability_subset
            && (stride == 0
                || stride
                    % properties
                        .min_vertex_input_binding_stride_alignment
                        .unwrap()
                    != 0)
        {
            return Err(Box::new(ValidationError {
                problem: "this device is a portability subset device, and \
                    `stride` is not a multiple of, and at least as large as, the \
                    `min_vertex_input_binding_stride_alignment` limit"
                    .into(),
                vuids: &["VUID-VkVertexInputBindingDescription-stride-04456"],
                ..Default::default()
            }));
        }

        match input_rate {
            VertexInputRate::Instance { divisor } if divisor != 1 => {
                if !device
                    .enabled_features()
                    .vertex_attribute_instance_rate_divisor
                {
                    return Err(Box::new(ValidationError {
                        context: "input_rate".into(),
                        problem: "is `VertexInputRate::Instance`, and \
                            its `divisor` value is not 1".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                                "vertex_attribute_instance_rate_divisor",
                            )])]),
                        vuids: &["VUID-VkVertexInputBindingDivisorDescriptionEXT-vertexAttributeInstanceRateDivisor-02229"],
                    }));
                }

                if divisor == 0
                    && !device
                        .enabled_features()
                        .vertex_attribute_instance_rate_zero_divisor
                {
                    return Err(Box::new(ValidationError {
                        context: "input_rate".into(),
                        problem: "is `VertexInputRate::Instance`, and \
                            its `divisor` value is 0".into(),
                        requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                                "vertex_attribute_instance_rate_zero_divisor",
                            )])]),
                        vuids: &["VUID-VkVertexInputBindingDivisorDescriptionEXT-vertexAttributeInstanceRateZeroDivisor-02228"],
                    }));
                }

                if divisor > properties.max_vertex_attrib_divisor.unwrap() {
                    return Err(Box::new(ValidationError {
                        context: "input_rate".into(),
                        problem: "is `VertexInputRate::Instance`, and \
                            its `divisor` value exceeds the `max_vertex_attrib_divisor` limit"
                            .into(),
                        vuids: &["VUID-VkVertexInputBindingDivisorDescriptionEXT-divisor-01870"],
                        ..Default::default()
                    }));
                }
            }
            _ => (),
        }

        Ok(())
    }
}

/// Describes a single vertex buffer attribute mapping.
#[derive(Clone, Copy, Debug)]
pub struct VertexInputAttributeDescription {
    /// The vertex buffer binding number that this attribute should take its data from.
    pub binding: u32,

    /// The size and type of the vertex data.
    pub format: Format,

    /// Number of bytes between the start of a vertex buffer element and the location of attribute.
    ///
    /// On [portability subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, if the sum of `offset + format.block_size()` is greater than the `stride` of
    /// `binding`, the
    /// [`vertex_attribute_access_beyond_stride`](crate::device::Features::vertex_attribute_access_beyond_stride)
    /// feature must be enabled on the device.
    pub offset: u32,
}

impl VertexInputAttributeDescription {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            binding,
            format,
            offset,
        } = self;

        let properties = device.physical_device().properties();

        format.validate_device(device).map_err(|err| {
            err.add_context("format")
                .set_vuids(&["VUID-VkVertexInputAttributeDescription-format-parameter"])
        })?;

        if binding > properties.max_vertex_input_bindings {
            return Err(Box::new(ValidationError {
                context: "binding".into(),
                problem: "exceeds the `max_vertex_input_bindings` limit".into(),
                vuids: &["VUID-VkVertexInputAttributeDescription-binding-00621"],
                ..Default::default()
            }));
        }

        if offset > properties.max_vertex_input_attribute_offset {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "exceeds the `max_vertex_input_attribute_offset` limit".into(),
                vuids: &["VUID-VkVertexInputAttributeDescription-offset-00622"],
                ..Default::default()
            }));
        }

        let format_features = unsafe {
            device
                .physical_device()
                .format_properties_unchecked(format)
                .buffer_features
        };

        if !format_features.intersects(FormatFeatures::VERTEX_BUFFER) {
            return Err(Box::new(ValidationError {
                context: "format".into(),
                problem: "the format features do not include `FormatFeatures::VERTEX_BUFFER`"
                    .into(),
                vuids: &["VUID-VkVertexInputAttributeDescription-format-00623"],
                ..Default::default()
            }));
        }

        Ok(())
    }
}

/// How the vertex source should be unrolled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VertexInputRate {
    /// Each element of the source corresponds to a vertex.
    Vertex,

    /// Each element of the source corresponds to an instance.
    ///
    /// `divisor` indicates how many consecutive instances will use the same instance buffer data.
    /// This value must be 1, unless the [`vertex_attribute_instance_rate_divisor`] feature has
    /// been enabled on the device.
    ///
    /// `divisor` can be 0 if the [`vertex_attribute_instance_rate_zero_divisor`] feature is also
    /// enabled. This means that every vertex will use the same vertex and instance data.
    ///
    /// [`vertex_attribute_instance_rate_divisor`]: crate::device::Features::vertex_attribute_instance_rate_divisor
    /// [`vertex_attribute_instance_rate_zero_divisor`]: crate::device::Features::vertex_attribute_instance_rate_zero_divisor
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
