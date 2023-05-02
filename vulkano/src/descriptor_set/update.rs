// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorType};
use crate::{
    buffer::{view::BufferView, BufferUsage, Subbuffer},
    device::DeviceOwned,
    image::{
        view::ImageViewType, ImageAspects, ImageLayout, ImageType, ImageUsage, ImageViewAbstract,
    },
    sampler::{Sampler, SamplerImageViewIncompatibleError},
    DeviceSize, RequiresOneOf, VulkanObject,
};
use smallvec::SmallVec;
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    ops::Range,
    ptr,
    sync::Arc,
};

/// Represents a single write operation to the binding of a descriptor set.
///
/// `WriteDescriptorSet` specifies the binding number and target array index, and includes one or
/// more resources of a given type that need to be written to that location. Two constructors are
/// provided for each resource type:
/// - The basic constructor variant writes a single element to array index 0. It is intended for
///   non-arrayed bindings, where `descriptor_count` in the descriptor set layout is 1.
/// - The `_array` variant writes several elements and allows specifying the target array index.
///   At least one element must be provided; a panic results if the provided iterator is empty.
#[derive(Clone, Debug)]
pub struct WriteDescriptorSet {
    binding: u32,
    first_array_element: u32,
    pub(crate) elements: WriteDescriptorSetElements, // public so that the image layouts can be changed
}

impl WriteDescriptorSet {
    /// Write an empty element to array element 0.
    ///
    /// This is used for push descriptors in combination with `Sampler` descriptors that have
    /// immutable samplers in the layout. The Vulkan spec requires these elements to be explicitly
    /// written, but since there is no data to write, a dummy write is provided instead.
    ///
    /// For regular descriptor sets, the data for such descriptors is automatically valid, and dummy
    /// writes are not allowed.
    #[inline]
    pub fn none(binding: u32) -> Self {
        Self::none_array(binding, 0, 1)
    }

    /// Write a number of consecutive empty elements.
    ///
    /// See [`none`](Self::none) for more information.
    #[inline]
    pub fn none_array(binding: u32, first_array_element: u32, num_elements: u32) -> Self {
        assert!(num_elements != 0);
        Self {
            binding,
            first_array_element,
            elements: WriteDescriptorSetElements::None(num_elements),
        }
    }

    /// Write a single buffer to array element 0, with the bound range covering the whole buffer.
    ///
    /// For dynamic buffer bindings, this will bind the whole buffer, and only a dynamic offset
    /// of zero will be valid, which is probably not what you want.
    /// Use [`buffer_with_range`](Self::buffer_with_range) instead.
    #[inline]
    pub fn buffer(binding: u32, buffer: Subbuffer<impl ?Sized>) -> Self {
        let range = 0..buffer.size();
        Self::buffer_with_range_array(
            binding,
            0,
            [DescriptorBufferInfo {
                buffer: buffer.into_bytes(),
                range,
            }],
        )
    }

    /// Write a number of consecutive buffer elements.
    ///
    /// See [`buffer`](Self::buffer) for more information.
    #[inline]
    pub fn buffer_array(
        binding: u32,
        first_array_element: u32,
        elements: impl IntoIterator<Item = Subbuffer<impl ?Sized>>,
    ) -> Self {
        Self::buffer_with_range_array(
            binding,
            first_array_element,
            elements.into_iter().map(|buffer| {
                let range = 0..buffer.size();
                DescriptorBufferInfo {
                    buffer: buffer.into_bytes(),
                    range,
                }
            }),
        )
    }

    /// Write a single buffer to array element 0, specifying the range of the buffer to be bound.
    #[inline]
    pub fn buffer_with_range(binding: u32, buffer_info: DescriptorBufferInfo) -> Self {
        Self::buffer_with_range_array(binding, 0, [buffer_info])
    }

    /// Write a number of consecutive buffer elements, specifying the ranges of the buffers to be
    /// bound.
    ///
    /// See [`buffer_with_range`](Self::buffer_with_range) for more information.
    pub fn buffer_with_range_array(
        binding: u32,
        first_array_element: u32,
        elements: impl IntoIterator<Item = DescriptorBufferInfo>,
    ) -> Self {
        let elements: SmallVec<_> = elements.into_iter().collect();
        assert!(!elements.is_empty());

        Self {
            binding,
            first_array_element,
            elements: WriteDescriptorSetElements::Buffer(elements),
        }
    }

    /// Write a single buffer view to array element 0.
    #[inline]
    pub fn buffer_view(binding: u32, buffer_view: Arc<BufferView>) -> Self {
        Self::buffer_view_array(binding, 0, [buffer_view])
    }

    /// Write a number of consecutive buffer view elements.
    pub fn buffer_view_array(
        binding: u32,
        first_array_element: u32,
        elements: impl IntoIterator<Item = Arc<BufferView>>,
    ) -> Self {
        let elements: SmallVec<_> = elements.into_iter().collect();
        assert!(!elements.is_empty());

        Self {
            binding,
            first_array_element,
            elements: WriteDescriptorSetElements::BufferView(elements),
        }
    }

    /// Write a single image view to array element 0, using the `Undefined` image layout,
    /// which will be automatically replaced with an appropriate default layout.
    #[inline]
    pub fn image_view(binding: u32, image_view: Arc<dyn ImageViewAbstract>) -> Self {
        Self::image_view_with_layout_array(
            binding,
            0,
            [DescriptorImageViewInfo {
                image_view,
                image_layout: ImageLayout::Undefined,
            }],
        )
    }

    /// Write a number of consecutive image view elements, using the `Undefined` image layout,
    /// which will be automatically replaced with an appropriate default layout.
    #[inline]
    pub fn image_view_array(
        binding: u32,
        first_array_element: u32,
        elements: impl IntoIterator<Item = Arc<dyn ImageViewAbstract>>,
    ) -> Self {
        Self::image_view_with_layout_array(
            binding,
            first_array_element,
            elements
                .into_iter()
                .map(|image_view| DescriptorImageViewInfo {
                    image_view,
                    image_layout: ImageLayout::Undefined,
                }),
        )
    }

    /// Write a single image view to array element 0, specifying the layout of the image to be
    /// bound.
    #[inline]
    pub fn image_view_with_layout(binding: u32, image_view_info: DescriptorImageViewInfo) -> Self {
        Self::image_view_with_layout_array(binding, 0, [image_view_info])
    }

    /// Write a number of consecutive image view elements, specifying the layouts of the images to
    /// be bound.
    pub fn image_view_with_layout_array(
        binding: u32,
        first_array_element: u32,
        elements: impl IntoIterator<Item = DescriptorImageViewInfo>,
    ) -> Self {
        let elements: SmallVec<_> = elements.into_iter().collect();
        assert!(!elements.is_empty());
        Self {
            binding,
            first_array_element,
            elements: WriteDescriptorSetElements::ImageView(elements),
        }
    }

    /// Write a single image view and sampler to array element 0,
    /// using the `Undefined` image layout, which will be automatically replaced with
    /// an appropriate default layout.
    #[inline]
    pub fn image_view_sampler(
        binding: u32,
        image_view: Arc<dyn ImageViewAbstract>,
        sampler: Arc<Sampler>,
    ) -> Self {
        Self::image_view_with_layout_sampler_array(
            binding,
            0,
            [(
                DescriptorImageViewInfo {
                    image_view,
                    image_layout: ImageLayout::Undefined,
                },
                sampler,
            )],
        )
    }

    /// Write a number of consecutive image view and sampler elements,
    /// using the `Undefined` image layout, which will be automatically replaced with
    /// an appropriate default layout.
    #[inline]
    pub fn image_view_sampler_array(
        binding: u32,
        first_array_element: u32,
        elements: impl IntoIterator<Item = (Arc<dyn ImageViewAbstract>, Arc<Sampler>)>,
    ) -> Self {
        Self::image_view_with_layout_sampler_array(
            binding,
            first_array_element,
            elements.into_iter().map(|(image_view, sampler)| {
                (
                    DescriptorImageViewInfo {
                        image_view,
                        image_layout: ImageLayout::Undefined,
                    },
                    sampler,
                )
            }),
        )
    }

    /// Write a single image view and sampler to array element 0, specifying the layout of the
    /// image to be bound.
    #[inline]
    pub fn image_view_with_layout_sampler(
        binding: u32,
        image_view_info: DescriptorImageViewInfo,
        sampler: Arc<Sampler>,
    ) -> Self {
        Self::image_view_with_layout_sampler_array(binding, 0, [(image_view_info, sampler)])
    }

    /// Write a number of consecutive image view and sampler elements, specifying the layout of the
    /// image to be bound.
    pub fn image_view_with_layout_sampler_array(
        binding: u32,
        first_array_element: u32,
        elements: impl IntoIterator<Item = (DescriptorImageViewInfo, Arc<Sampler>)>,
    ) -> Self {
        let elements: SmallVec<_> = elements.into_iter().collect();
        assert!(!elements.is_empty());

        Self {
            binding,
            first_array_element,
            elements: WriteDescriptorSetElements::ImageViewSampler(elements),
        }
    }

    /// Write a single sampler to array element 0.
    #[inline]
    pub fn sampler(binding: u32, sampler: Arc<Sampler>) -> Self {
        Self::sampler_array(binding, 0, [sampler])
    }

    /// Write a number of consecutive sampler elements.
    pub fn sampler_array(
        binding: u32,
        first_array_element: u32,
        elements: impl IntoIterator<Item = Arc<Sampler>>,
    ) -> Self {
        let elements: SmallVec<_> = elements.into_iter().collect();
        assert!(!elements.is_empty());

        Self {
            binding,
            first_array_element,
            elements: WriteDescriptorSetElements::Sampler(elements),
        }
    }

    /// Returns the binding number that is updated by this descriptor write.
    #[inline]
    pub fn binding(&self) -> u32 {
        self.binding
    }

    /// Returns the first array element in the binding that is updated by this descriptor write.
    #[inline]
    pub fn first_array_element(&self) -> u32 {
        self.first_array_element
    }

    /// Returns a reference to the elements held by this descriptor write.
    #[inline]
    pub fn elements(&self) -> &WriteDescriptorSetElements {
        &self.elements
    }

    pub(crate) fn to_vulkan_info(&self, descriptor_type: DescriptorType) -> DescriptorWriteInfo {
        match &self.elements {
            WriteDescriptorSetElements::None(num_elements) => {
                debug_assert!(matches!(descriptor_type, DescriptorType::Sampler));
                DescriptorWriteInfo::Image(
                    std::iter::repeat_with(|| ash::vk::DescriptorImageInfo {
                        sampler: ash::vk::Sampler::null(),
                        image_view: ash::vk::ImageView::null(),
                        image_layout: ash::vk::ImageLayout::UNDEFINED,
                    })
                    .take(*num_elements as usize)
                    .collect(),
                )
            }
            WriteDescriptorSetElements::Buffer(elements) => {
                debug_assert!(matches!(
                    descriptor_type,
                    DescriptorType::UniformBuffer
                        | DescriptorType::StorageBuffer
                        | DescriptorType::UniformBufferDynamic
                        | DescriptorType::StorageBufferDynamic
                ));
                DescriptorWriteInfo::Buffer(
                    elements
                        .iter()
                        .map(|buffer_info| {
                            let DescriptorBufferInfo { buffer, range } = buffer_info;

                            debug_assert!(!range.is_empty());
                            debug_assert!(range.end <= buffer.buffer().size());

                            ash::vk::DescriptorBufferInfo {
                                buffer: buffer.buffer().handle(),
                                offset: buffer.offset() + range.start,
                                range: range.end - range.start,
                            }
                        })
                        .collect(),
                )
            }
            WriteDescriptorSetElements::BufferView(elements) => {
                debug_assert!(matches!(
                    descriptor_type,
                    DescriptorType::UniformTexelBuffer | DescriptorType::StorageTexelBuffer
                ));
                DescriptorWriteInfo::BufferView(
                    elements
                        .iter()
                        .map(|buffer_view| buffer_view.handle())
                        .collect(),
                )
            }
            WriteDescriptorSetElements::ImageView(elements) => {
                // Note: combined image sampler can occur with immutable samplers
                debug_assert!(matches!(
                    descriptor_type,
                    DescriptorType::CombinedImageSampler
                        | DescriptorType::SampledImage
                        | DescriptorType::StorageImage
                        | DescriptorType::InputAttachment
                ));
                DescriptorWriteInfo::Image(
                    elements
                        .iter()
                        .map(|image_view_info| {
                            let &DescriptorImageViewInfo {
                                ref image_view,
                                image_layout,
                            } = image_view_info;

                            ash::vk::DescriptorImageInfo {
                                sampler: ash::vk::Sampler::null(),
                                image_view: image_view.handle(),
                                image_layout: image_layout.into(),
                            }
                        })
                        .collect(),
                )
            }
            WriteDescriptorSetElements::ImageViewSampler(elements) => {
                debug_assert!(matches!(
                    descriptor_type,
                    DescriptorType::CombinedImageSampler
                ));
                DescriptorWriteInfo::Image(
                    elements
                        .iter()
                        .map(|(image_view_info, sampler)| {
                            let &DescriptorImageViewInfo {
                                ref image_view,
                                image_layout,
                            } = image_view_info;

                            ash::vk::DescriptorImageInfo {
                                sampler: sampler.handle(),
                                image_view: image_view.handle(),
                                image_layout: image_layout.into(),
                            }
                        })
                        .collect(),
                )
            }
            WriteDescriptorSetElements::Sampler(elements) => {
                debug_assert!(matches!(descriptor_type, DescriptorType::Sampler));
                DescriptorWriteInfo::Image(
                    elements
                        .iter()
                        .map(|sampler| ash::vk::DescriptorImageInfo {
                            sampler: sampler.handle(),
                            image_view: ash::vk::ImageView::null(),
                            image_layout: ash::vk::ImageLayout::UNDEFINED,
                        })
                        .collect(),
                )
            }
        }
    }

    pub(crate) fn to_vulkan(
        &self,
        dst_set: ash::vk::DescriptorSet,
        descriptor_type: DescriptorType,
    ) -> ash::vk::WriteDescriptorSet {
        ash::vk::WriteDescriptorSet {
            dst_set,
            dst_binding: self.binding,
            dst_array_element: self.first_array_element,
            descriptor_count: 0,
            descriptor_type: descriptor_type.into(),
            p_image_info: ptr::null(),
            p_buffer_info: ptr::null(),
            p_texel_buffer_view: ptr::null(),
            ..Default::default()
        }
    }
}

/// The elements held by a `WriteDescriptorSet`.
#[derive(Clone, Debug)]
pub enum WriteDescriptorSetElements {
    None(u32),
    Buffer(SmallVec<[DescriptorBufferInfo; 1]>),
    BufferView(SmallVec<[Arc<BufferView>; 1]>),
    ImageView(SmallVec<[DescriptorImageViewInfo; 1]>),
    ImageViewSampler(SmallVec<[(DescriptorImageViewInfo, Arc<Sampler>); 1]>),
    Sampler(SmallVec<[Arc<Sampler>; 1]>),
}

impl WriteDescriptorSetElements {
    /// Returns the number of elements.
    #[inline]
    pub fn len(&self) -> u32 {
        match self {
            Self::None(num_elements) => *num_elements,
            Self::Buffer(elements) => elements.len() as u32,
            Self::BufferView(elements) => elements.len() as u32,
            Self::ImageView(elements) => elements.len() as u32,
            Self::ImageViewSampler(elements) => elements.len() as u32,
            Self::Sampler(elements) => elements.len() as u32,
        }
    }
}

/// Parameters to write a buffer reference to a descriptor.
#[derive(Clone, Debug)]
pub struct DescriptorBufferInfo {
    /// The buffer to write to the descriptor.
    pub buffer: Subbuffer<[u8]>,

    /// The slice of bytes in `buffer` that will be made available to the shader.
    /// `range` must not be outside the range `buffer`.
    ///
    /// For dynamic buffer bindings, `range` specifies the slice that is to be bound if the
    /// dynamic offset were zero. When binding the descriptor set, the effective value of `range`
    /// shifts forward by the offset that was provided. For example, if `range` is specified as
    /// `0..8` when writing the descriptor set, and then when binding the descriptor set the
    /// offset `16` is used, then the range of `buffer` that will actually be bound is `16..24`.
    pub range: Range<DeviceSize>,
}

/// Parameters to write an image view reference to a descriptor.
#[derive(Clone, Debug)]
pub struct DescriptorImageViewInfo {
    /// The image view to write to the descriptor.
    pub image_view: Arc<dyn ImageViewAbstract>,

    /// The layout that the image is expected to be in when it's accessed in the shader.
    ///
    /// Only certain layouts are allowed, depending on the type of descriptor.
    ///
    /// For `SampledImage`, `CombinedImageSampler` and `InputAttachment`:
    /// - `General`
    /// - `ShaderReadOnlyOptimal`
    /// - `DepthStencilReadOnlyOptimal`
    /// - `DepthReadOnlyStencilAttachmentOptimal`
    /// - `DepthAttachmentStencilReadOnlyOptimal`
    ///
    /// For `StorageImage`:
    /// - `General`
    ///
    /// If the `Undefined` layout is provided, then it will be automatically replaced with
    /// `General` for `StorageImage` descriptors, and with `ShaderReadOnlyOptimal` for any other
    /// descriptor type.
    pub image_layout: ImageLayout,
}

#[derive(Clone, Debug)]
pub(crate) enum DescriptorWriteInfo {
    Image(SmallVec<[ash::vk::DescriptorImageInfo; 1]>),
    Buffer(SmallVec<[ash::vk::DescriptorBufferInfo; 1]>),
    BufferView(SmallVec<[ash::vk::BufferView; 1]>),
}

pub(crate) fn set_descriptor_write_image_layouts(
    write: &mut WriteDescriptorSet,
    layout: &DescriptorSetLayout,
) {
    let default_layout = if let Some(layout_binding) = layout.bindings().get(&write.binding()) {
        match layout_binding.descriptor_type {
            DescriptorType::CombinedImageSampler
            | DescriptorType::SampledImage
            | DescriptorType::InputAttachment => ImageLayout::ShaderReadOnlyOptimal,
            DescriptorType::StorageImage => ImageLayout::General,
            _ => return,
        }
    } else {
        return;
    };

    match &mut write.elements {
        WriteDescriptorSetElements::ImageView(elements) => {
            for image_view_info in elements {
                let DescriptorImageViewInfo {
                    image_view: _,
                    image_layout,
                } = image_view_info;

                if *image_layout == ImageLayout::Undefined {
                    *image_layout = default_layout;
                }
            }
        }
        WriteDescriptorSetElements::ImageViewSampler(elements) => {
            for (image_view_info, _sampler) in elements {
                let DescriptorImageViewInfo {
                    image_view: _,
                    image_layout,
                } = image_view_info;

                if *image_layout == ImageLayout::Undefined {
                    *image_layout = default_layout;
                }
            }
        }
        _ => (),
    }
}

pub(crate) fn validate_descriptor_write<'a>(
    write: &WriteDescriptorSet,
    layout: &'a DescriptorSetLayout,
    variable_descriptor_count: u32,
) -> Result<&'a DescriptorSetLayoutBinding, DescriptorSetUpdateError> {
    fn provided_element_type(elements: &WriteDescriptorSetElements) -> &'static str {
        match elements {
            WriteDescriptorSetElements::None(_) => "none",
            WriteDescriptorSetElements::Buffer(_) => "buffer",
            WriteDescriptorSetElements::BufferView(_) => "buffer_view",
            WriteDescriptorSetElements::ImageView(_) => "image_view",
            WriteDescriptorSetElements::ImageViewSampler(_) => "image_view_sampler",
            WriteDescriptorSetElements::Sampler(_) => "sampler",
        }
    }

    let device = layout.device();

    let layout_binding = match layout.bindings().get(&write.binding()) {
        Some(binding) => binding,
        None => {
            return Err(DescriptorSetUpdateError::InvalidBinding {
                binding: write.binding(),
            })
        }
    };

    let max_descriptor_count = if layout_binding.variable_descriptor_count {
        variable_descriptor_count
    } else {
        layout_binding.descriptor_count
    };

    let binding = write.binding();
    let elements = write.elements();
    let num_elements = elements.len();
    debug_assert!(num_elements != 0);

    let descriptor_range_start = write.first_array_element();
    let descriptor_range_end = descriptor_range_start + num_elements;

    if descriptor_range_end > max_descriptor_count {
        return Err(DescriptorSetUpdateError::ArrayIndexOutOfBounds {
            binding,
            available_count: max_descriptor_count,
            written_count: descriptor_range_end,
        });
    }

    match layout_binding.descriptor_type {
        DescriptorType::Sampler => {
            if layout_binding.immutable_samplers.is_empty() {
                let elements = if let WriteDescriptorSetElements::Sampler(elements) = elements {
                    elements
                } else {
                    return Err(DescriptorSetUpdateError::IncompatibleElementType {
                        binding,
                        provided_element_type: provided_element_type(elements),
                        allowed_element_types: &["sampler"],
                    });
                };

                for (index, sampler) in elements.iter().enumerate() {
                    assert_eq!(device, sampler.device());

                    if sampler.sampler_ycbcr_conversion().is_some() {
                        return Err(DescriptorSetUpdateError::SamplerHasSamplerYcbcrConversion {
                            binding: write.binding(),
                            index: descriptor_range_start + index as u32,
                        });
                    }
                }
            } else if layout.push_descriptor() {
                // For push descriptors, we must write a dummy element.
                if let WriteDescriptorSetElements::None(_) = elements {
                    // Do nothing
                } else {
                    return Err(DescriptorSetUpdateError::IncompatibleElementType {
                        binding,
                        provided_element_type: provided_element_type(elements),
                        allowed_element_types: &["none"],
                    });
                }
            } else {
                // For regular descriptors, no element must be written.
                return Err(DescriptorSetUpdateError::IncompatibleElementType {
                    binding,
                    provided_element_type: provided_element_type(elements),
                    allowed_element_types: &[],
                });
            }
        }

        DescriptorType::CombinedImageSampler => {
            if layout_binding.immutable_samplers.is_empty() {
                let elements =
                    if let WriteDescriptorSetElements::ImageViewSampler(elements) = elements {
                        elements
                    } else {
                        return Err(DescriptorSetUpdateError::IncompatibleElementType {
                            binding,
                            provided_element_type: provided_element_type(elements),
                            allowed_element_types: &["image_view_sampler"],
                        });
                    };

                for (index, (image_view_info, sampler)) in elements.iter().enumerate() {
                    let &DescriptorImageViewInfo {
                        ref image_view,
                        image_layout,
                    } = image_view_info;

                    assert_eq!(device, image_view.device());
                    assert_eq!(device, sampler.device());

                    // VUID-VkWriteDescriptorSet-descriptorType-00337
                    if !image_view.usage().intersects(ImageUsage::SAMPLED) {
                        return Err(DescriptorSetUpdateError::MissingUsage {
                            binding: write.binding(),
                            index: descriptor_range_start + index as u32,
                            usage: "sampled",
                        });
                    }

                    // VUID-VkDescriptorImageInfo-imageView-00343
                    if matches!(
                        image_view.view_type(),
                        ImageViewType::Dim2d | ImageViewType::Dim2dArray
                    ) && image_view.image().inner().image.dimensions().image_type()
                        == ImageType::Dim3d
                    {
                        return Err(DescriptorSetUpdateError::ImageView2dFrom3d {
                            binding: write.binding(),
                            index: descriptor_range_start + index as u32,
                        });
                    }

                    // VUID-VkDescriptorImageInfo-imageView-01976
                    if image_view
                        .subresource_range()
                        .aspects
                        .contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
                    {
                        return Err(DescriptorSetUpdateError::ImageViewDepthAndStencil {
                            binding: write.binding(),
                            index: descriptor_range_start + index as u32,
                        });
                    }

                    // VUID-VkWriteDescriptorSet-descriptorType-04150
                    if !matches!(
                        image_layout,
                        ImageLayout::DepthStencilReadOnlyOptimal
                            | ImageLayout::ShaderReadOnlyOptimal
                            | ImageLayout::General
                            | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                            | ImageLayout::DepthAttachmentStencilReadOnlyOptimal,
                    ) {
                        return Err(DescriptorSetUpdateError::ImageLayoutInvalid {
                            binding: write.binding(),
                            index: descriptor_range_start + index as u32,
                        });
                    }

                    // VUID-VkDescriptorImageInfo-mutableComparisonSamplers-04450
                    if device.enabled_extensions().khr_portability_subset
                        && !device.enabled_features().mutable_comparison_samplers
                        && sampler.compare().is_some()
                    {
                        return Err(DescriptorSetUpdateError::RequirementNotMet {
                            binding: write.binding(),
                            index: descriptor_range_start + index as u32,
                            required_for: "this device is a portability subset device, and \
                                `sampler.compare()` is `Some`",
                            requires_one_of: RequiresOneOf {
                                features: &["mutable_comparison_samplers"],
                                ..Default::default()
                            },
                        });
                    }

                    if image_view.sampler_ycbcr_conversion().is_some() {
                        return Err(
                            DescriptorSetUpdateError::ImageViewHasSamplerYcbcrConversion {
                                binding: write.binding(),
                                index: descriptor_range_start + index as u32,
                            },
                        );
                    }

                    if sampler.sampler_ycbcr_conversion().is_some() {
                        return Err(DescriptorSetUpdateError::SamplerHasSamplerYcbcrConversion {
                            binding: write.binding(),
                            index: descriptor_range_start + index as u32,
                        });
                    }

                    if let Err(error) = sampler.check_can_sample(image_view.as_ref()) {
                        return Err(DescriptorSetUpdateError::ImageViewIncompatibleSampler {
                            binding: write.binding(),
                            index: descriptor_range_start + index as u32,
                            error,
                        });
                    }
                }
            } else {
                let elements = if let WriteDescriptorSetElements::ImageView(elements) = elements {
                    elements
                } else {
                    return Err(DescriptorSetUpdateError::IncompatibleElementType {
                        binding,
                        provided_element_type: provided_element_type(elements),
                        allowed_element_types: &["image_view"],
                    });
                };

                let immutable_samplers = &layout_binding.immutable_samplers
                    [descriptor_range_start as usize..descriptor_range_end as usize];

                for (index, (image_view_info, sampler)) in
                    elements.iter().zip(immutable_samplers).enumerate()
                {
                    let &DescriptorImageViewInfo {
                        ref image_view,
                        image_layout,
                    } = image_view_info;

                    assert_eq!(device, image_view.device());

                    // VUID-VkWriteDescriptorSet-descriptorType-00337
                    if !image_view.usage().intersects(ImageUsage::SAMPLED) {
                        return Err(DescriptorSetUpdateError::MissingUsage {
                            binding: write.binding(),
                            index: descriptor_range_start + index as u32,
                            usage: "sampled",
                        });
                    }

                    // VUID-VkDescriptorImageInfo-imageView-00343
                    if matches!(
                        image_view.view_type(),
                        ImageViewType::Dim2d | ImageViewType::Dim2dArray
                    ) && image_view.image().inner().image.dimensions().image_type()
                        == ImageType::Dim3d
                    {
                        return Err(DescriptorSetUpdateError::ImageView2dFrom3d {
                            binding: write.binding(),
                            index: descriptor_range_start + index as u32,
                        });
                    }

                    // VUID-VkDescriptorImageInfo-imageView-01976
                    if image_view
                        .subresource_range()
                        .aspects
                        .contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
                    {
                        return Err(DescriptorSetUpdateError::ImageViewDepthAndStencil {
                            binding: write.binding(),
                            index: descriptor_range_start + index as u32,
                        });
                    }

                    // VUID-VkWriteDescriptorSet-descriptorType-04150
                    if !matches!(
                        image_layout,
                        ImageLayout::DepthStencilReadOnlyOptimal
                            | ImageLayout::ShaderReadOnlyOptimal
                            | ImageLayout::General
                            | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                            | ImageLayout::DepthAttachmentStencilReadOnlyOptimal,
                    ) {
                        return Err(DescriptorSetUpdateError::ImageLayoutInvalid {
                            binding: write.binding(),
                            index: descriptor_range_start + index as u32,
                        });
                    }

                    if let Err(error) = sampler.check_can_sample(image_view.as_ref()) {
                        return Err(DescriptorSetUpdateError::ImageViewIncompatibleSampler {
                            binding: write.binding(),
                            index: descriptor_range_start + index as u32,
                            error,
                        });
                    }
                }
            }
        }

        DescriptorType::SampledImage => {
            let elements = if let WriteDescriptorSetElements::ImageView(elements) = elements {
                elements
            } else {
                return Err(DescriptorSetUpdateError::IncompatibleElementType {
                    binding,
                    provided_element_type: provided_element_type(elements),
                    allowed_element_types: &["image_view"],
                });
            };

            for (index, image_view_info) in elements.iter().enumerate() {
                let &DescriptorImageViewInfo {
                    ref image_view,
                    image_layout,
                } = image_view_info;

                assert_eq!(device, image_view.device());

                // VUID-VkWriteDescriptorSet-descriptorType-00337
                if !image_view.usage().intersects(ImageUsage::SAMPLED) {
                    return Err(DescriptorSetUpdateError::MissingUsage {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                        usage: "sampled",
                    });
                }

                // VUID-VkDescriptorImageInfo-imageView-00343
                if matches!(
                    image_view.view_type(),
                    ImageViewType::Dim2d | ImageViewType::Dim2dArray
                ) && image_view.image().inner().image.dimensions().image_type()
                    == ImageType::Dim3d
                {
                    return Err(DescriptorSetUpdateError::ImageView2dFrom3d {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                    });
                }

                // VUID-VkDescriptorImageInfo-imageView-01976
                if image_view
                    .subresource_range()
                    .aspects
                    .contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
                {
                    return Err(DescriptorSetUpdateError::ImageViewDepthAndStencil {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                    });
                }

                // VUID-VkWriteDescriptorSet-descriptorType-04149
                if !matches!(
                    image_layout,
                    ImageLayout::DepthStencilReadOnlyOptimal
                        | ImageLayout::ShaderReadOnlyOptimal
                        | ImageLayout::General
                        | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                        | ImageLayout::DepthAttachmentStencilReadOnlyOptimal,
                ) {
                    return Err(DescriptorSetUpdateError::ImageLayoutInvalid {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                    });
                }

                // VUID-VkWriteDescriptorSet-descriptorType-01946
                if image_view.sampler_ycbcr_conversion().is_some() {
                    return Err(
                        DescriptorSetUpdateError::ImageViewHasSamplerYcbcrConversion {
                            binding: write.binding(),
                            index: descriptor_range_start + index as u32,
                        },
                    );
                }
            }
        }

        DescriptorType::StorageImage => {
            let elements = if let WriteDescriptorSetElements::ImageView(elements) = elements {
                elements
            } else {
                return Err(DescriptorSetUpdateError::IncompatibleElementType {
                    binding,
                    provided_element_type: provided_element_type(elements),
                    allowed_element_types: &["image_view"],
                });
            };

            for (index, image_view_info) in elements.iter().enumerate() {
                let &DescriptorImageViewInfo {
                    ref image_view,
                    image_layout,
                } = image_view_info;

                assert_eq!(device, image_view.device());

                // VUID-VkWriteDescriptorSet-descriptorType-00339
                if !image_view.usage().intersects(ImageUsage::STORAGE) {
                    return Err(DescriptorSetUpdateError::MissingUsage {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                        usage: "storage",
                    });
                }

                // VUID-VkDescriptorImageInfo-imageView-00343
                if matches!(
                    image_view.view_type(),
                    ImageViewType::Dim2d | ImageViewType::Dim2dArray
                ) && image_view.image().inner().image.dimensions().image_type()
                    == ImageType::Dim3d
                {
                    return Err(DescriptorSetUpdateError::ImageView2dFrom3d {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                    });
                }

                // VUID-VkDescriptorImageInfo-imageView-01976
                if image_view
                    .subresource_range()
                    .aspects
                    .contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
                {
                    return Err(DescriptorSetUpdateError::ImageViewDepthAndStencil {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                    });
                }

                // VUID-VkWriteDescriptorSet-descriptorType-04152
                if !matches!(image_layout, ImageLayout::General) {
                    return Err(DescriptorSetUpdateError::ImageLayoutInvalid {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                    });
                }

                // VUID-VkWriteDescriptorSet-descriptorType-00336
                if !image_view.component_mapping().is_identity() {
                    return Err(DescriptorSetUpdateError::ImageViewNotIdentitySwizzled {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                    });
                }

                // VUID??
                if image_view.sampler_ycbcr_conversion().is_some() {
                    return Err(
                        DescriptorSetUpdateError::ImageViewHasSamplerYcbcrConversion {
                            binding: write.binding(),
                            index: descriptor_range_start + index as u32,
                        },
                    );
                }
            }
        }

        DescriptorType::UniformTexelBuffer => {
            let elements = if let WriteDescriptorSetElements::BufferView(elements) = elements {
                elements
            } else {
                return Err(DescriptorSetUpdateError::IncompatibleElementType {
                    binding,
                    provided_element_type: provided_element_type(elements),
                    allowed_element_types: &["buffer_view"],
                });
            };

            for (index, buffer_view) in elements.iter().enumerate() {
                assert_eq!(device, buffer_view.device());

                if !buffer_view
                    .buffer()
                    .buffer()
                    .usage()
                    .intersects(BufferUsage::UNIFORM_TEXEL_BUFFER)
                {
                    return Err(DescriptorSetUpdateError::MissingUsage {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                        usage: "uniform_texel_buffer",
                    });
                }
            }
        }

        DescriptorType::StorageTexelBuffer => {
            let elements = if let WriteDescriptorSetElements::BufferView(elements) = elements {
                elements
            } else {
                return Err(DescriptorSetUpdateError::IncompatibleElementType {
                    binding,
                    provided_element_type: provided_element_type(elements),
                    allowed_element_types: &["buffer_view"],
                });
            };

            for (index, buffer_view) in elements.iter().enumerate() {
                assert_eq!(device, buffer_view.device());

                // TODO: storage_texel_buffer_atomic
                if !buffer_view
                    .buffer()
                    .buffer()
                    .usage()
                    .intersects(BufferUsage::STORAGE_TEXEL_BUFFER)
                {
                    return Err(DescriptorSetUpdateError::MissingUsage {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                        usage: "storage_texel_buffer",
                    });
                }
            }
        }

        DescriptorType::UniformBuffer | DescriptorType::UniformBufferDynamic => {
            let elements = if let WriteDescriptorSetElements::Buffer(elements) = elements {
                elements
            } else {
                return Err(DescriptorSetUpdateError::IncompatibleElementType {
                    binding,
                    provided_element_type: provided_element_type(elements),
                    allowed_element_types: &["buffer"],
                });
            };

            for (index, buffer_info) in elements.iter().enumerate() {
                let DescriptorBufferInfo { buffer, range } = buffer_info;

                assert_eq!(device, buffer.device());

                if !buffer
                    .buffer()
                    .usage()
                    .intersects(BufferUsage::UNIFORM_BUFFER)
                {
                    return Err(DescriptorSetUpdateError::MissingUsage {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                        usage: "uniform_buffer",
                    });
                }

                assert!(!range.is_empty());

                if range.end > buffer.size() {
                    return Err(DescriptorSetUpdateError::RangeOutOfBufferBounds {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                        range_end: range.end,
                        buffer_size: buffer.size(),
                    });
                }
            }
        }

        DescriptorType::StorageBuffer | DescriptorType::StorageBufferDynamic => {
            let elements = if let WriteDescriptorSetElements::Buffer(elements) = elements {
                elements
            } else {
                return Err(DescriptorSetUpdateError::IncompatibleElementType {
                    binding,
                    provided_element_type: provided_element_type(elements),
                    allowed_element_types: &["buffer"],
                });
            };

            for (index, buffer_info) in elements.iter().enumerate() {
                let DescriptorBufferInfo { buffer, range } = buffer_info;

                assert_eq!(device, buffer.device());

                if !buffer
                    .buffer()
                    .usage()
                    .intersects(BufferUsage::STORAGE_BUFFER)
                {
                    return Err(DescriptorSetUpdateError::MissingUsage {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                        usage: "storage_buffer",
                    });
                }

                assert!(!range.is_empty());

                if range.end > buffer.size() {
                    return Err(DescriptorSetUpdateError::RangeOutOfBufferBounds {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                        range_end: range.end,
                        buffer_size: buffer.size(),
                    });
                }
            }
        }

        DescriptorType::InputAttachment => {
            let elements = if let WriteDescriptorSetElements::ImageView(elements) = elements {
                elements
            } else {
                return Err(DescriptorSetUpdateError::IncompatibleElementType {
                    binding,
                    provided_element_type: provided_element_type(elements),
                    allowed_element_types: &["image_view"],
                });
            };

            for (index, image_view_info) in elements.iter().enumerate() {
                let &DescriptorImageViewInfo {
                    ref image_view,
                    image_layout,
                } = image_view_info;

                assert_eq!(device, image_view.device());

                // VUID-VkWriteDescriptorSet-descriptorType-00338
                if !image_view.usage().intersects(ImageUsage::INPUT_ATTACHMENT) {
                    return Err(DescriptorSetUpdateError::MissingUsage {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                        usage: "input_attachment",
                    });
                }

                // VUID-VkDescriptorImageInfo-imageView-00343
                if matches!(
                    image_view.view_type(),
                    ImageViewType::Dim2d | ImageViewType::Dim2dArray
                ) && image_view.image().inner().image.dimensions().image_type()
                    == ImageType::Dim3d
                {
                    return Err(DescriptorSetUpdateError::ImageView2dFrom3d {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                    });
                }

                // VUID-VkDescriptorImageInfo-imageView-01976
                if image_view
                    .subresource_range()
                    .aspects
                    .contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
                {
                    return Err(DescriptorSetUpdateError::ImageViewDepthAndStencil {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                    });
                }

                // VUID-VkWriteDescriptorSet-descriptorType-04151
                if !matches!(
                    image_layout,
                    ImageLayout::DepthStencilReadOnlyOptimal
                        | ImageLayout::ShaderReadOnlyOptimal
                        | ImageLayout::General
                        | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                        | ImageLayout::DepthAttachmentStencilReadOnlyOptimal,
                ) {
                    return Err(DescriptorSetUpdateError::ImageLayoutInvalid {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                    });
                }

                // VUID-VkWriteDescriptorSet-descriptorType-00336
                if !image_view.component_mapping().is_identity() {
                    return Err(DescriptorSetUpdateError::ImageViewNotIdentitySwizzled {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                    });
                }

                // VUID??
                if image_view.sampler_ycbcr_conversion().is_some() {
                    return Err(
                        DescriptorSetUpdateError::ImageViewHasSamplerYcbcrConversion {
                            binding: write.binding(),
                            index: descriptor_range_start + index as u32,
                        },
                    );
                }

                // VUID??
                if image_view.view_type().is_arrayed() {
                    return Err(DescriptorSetUpdateError::ImageViewIsArrayed {
                        binding: write.binding(),
                        index: descriptor_range_start + index as u32,
                    });
                }
            }
        }
    }

    Ok(layout_binding)
}

#[derive(Clone, Copy, Debug)]
pub enum DescriptorSetUpdateError {
    RequirementNotMet {
        binding: u32,
        index: u32,
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// Tried to write more elements than were available in a binding.
    ArrayIndexOutOfBounds {
        /// Binding that is affected.
        binding: u32,
        /// Number of available descriptors in the binding.
        available_count: u32,
        /// The number of descriptors that were in the update.
        written_count: u32,
    },

    ImageLayoutInvalid {
        binding: u32,
        index: u32,
    },

    /// Tried to write an image view with a 2D type and a 3D underlying image.
    ImageView2dFrom3d {
        binding: u32,
        index: u32,
    },

    /// Tried to write an image view that has both the `depth` and `stencil` aspects.
    ImageViewDepthAndStencil {
        binding: u32,
        index: u32,
    },

    /// Tried to write an image view with an attached sampler YCbCr conversion to a binding that
    /// does not support it.
    ImageViewHasSamplerYcbcrConversion {
        binding: u32,
        index: u32,
    },

    /// Tried to write an image view of an arrayed type to a descriptor type that does not support
    /// it.
    ImageViewIsArrayed {
        binding: u32,
        index: u32,
    },

    /// Tried to write an image view that was not compatible with the sampler that was provided as
    /// part of the update or immutably in the layout.
    ImageViewIncompatibleSampler {
        binding: u32,
        index: u32,
        error: SamplerImageViewIncompatibleError,
    },

    /// Tried to write an image view to a descriptor type that requires it to be identity swizzled,
    /// but it was not.
    ImageViewNotIdentitySwizzled {
        binding: u32,
        index: u32,
    },

    /// Tried to write an element type that was not compatible with the descriptor type in the
    /// layout.
    IncompatibleElementType {
        binding: u32,
        provided_element_type: &'static str,
        allowed_element_types: &'static [&'static str],
    },

    /// Tried to write to a nonexistent binding.
    InvalidBinding {
        binding: u32,
    },

    /// A resource was missing a usage flag that was required.
    MissingUsage {
        binding: u32,
        index: u32,
        usage: &'static str,
    },

    /// The end of the provided `range` for a buffer is larger than the size of the buffer.
    RangeOutOfBufferBounds {
        binding: u32,
        index: u32,
        range_end: DeviceSize,
        buffer_size: DeviceSize,
    },

    /// Tried to write a sampler that has an attached sampler YCbCr conversion.
    SamplerHasSamplerYcbcrConversion {
        binding: u32,
        index: u32,
    },
}

impl Error for DescriptorSetUpdateError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ImageViewIncompatibleSampler { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl Display for DescriptorSetUpdateError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::RequirementNotMet {
                binding,
                index,
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement on binding {} index {} was not met for: {}; requires one of: {}",
                binding, index, required_for, requires_one_of,
            ),

            Self::ArrayIndexOutOfBounds {
                binding,
                available_count,
                written_count,
            } => write!(
                f,
                "tried to write up to element {} to binding {}, but only {} descriptors are \
                available",
                written_count, binding, available_count,
            ),
            Self::ImageLayoutInvalid { binding, index } => write!(
                f,
                "tried to write an image view to binding {} index {} with an image layout that is \
                not valid for that descriptor type",
                binding, index,
            ),
            Self::ImageView2dFrom3d { binding, index } => write!(
                f,
                "tried to write an image view to binding {} index {} with a 2D type and a 3D \
                underlying image",
                binding, index,
            ),
            Self::ImageViewDepthAndStencil { binding, index } => write!(
                f,
                "tried to write an image view to binding {} index {} that has both the `depth` and \
                `stencil` aspects",
                binding, index,
            ),
            Self::ImageViewHasSamplerYcbcrConversion { binding, index } => write!(
                f,
                "tried to write an image view to binding {} index {} with an attached sampler \
                YCbCr conversion to binding that does not support it",
                binding, index,
            ),
            Self::ImageViewIsArrayed { binding, index } => write!(
                f,
                "tried to write an image view of an arrayed type to binding {} index {}, but this \
                binding has a descriptor type that does not support arrayed image views",
                binding, index,
            ),
            Self::ImageViewIncompatibleSampler { binding, index, .. } => write!(
                f,
                "tried to write an image view to binding {} index {}, that was not compatible with \
                the sampler that was provided as part of the update or immutably in the layout",
                binding, index,
            ),
            Self::ImageViewNotIdentitySwizzled { binding, index } => write!(
                f,
                "tried to write an image view with non-identity swizzling to binding {} index {}, \
                but this binding has a descriptor type that requires it to be identity swizzled",
                binding, index,
            ),
            Self::IncompatibleElementType {
                binding,
                provided_element_type,
                allowed_element_types,
            } => write!(
                f,
                "tried to write a resource to binding {} whose type ({}) was not one of the \
                resource types allowed for the descriptor type (",
                binding, provided_element_type,
            )
            .and_then(|_| {
                let mut first = true;

                for elem_type in *allowed_element_types {
                    if first {
                        write!(f, "{}", elem_type)?;
                        first = false;
                    } else {
                        write!(f, ", {}", elem_type)?;
                    }
                }

                Ok(())
            })
            .and_then(|_| write!(f, ") that can be bound to this buffer")),
            Self::InvalidBinding { binding } => {
                write!(f, "tried to write to a nonexistent binding {}", binding,)
            }
            Self::MissingUsage {
                binding,
                index,
                usage,
            } => write!(
                f,
                "tried to write a resource to binding {} index {} that did not have the required \
                usage {} enabled",
                binding, index, usage,
            ),
            Self::RangeOutOfBufferBounds {
                binding,
                index,
                range_end,
                buffer_size,
            } => write!(
                f,
                "the end of the provided `range` for the buffer at binding {} index {} ({:?}) is
                larger than the size of the buffer ({})",
                binding, index, range_end, buffer_size,
            ),
            Self::SamplerHasSamplerYcbcrConversion { binding, index } => write!(
                f,
                "tried to write a sampler to binding {} index {} that has an attached sampler \
                YCbCr conversion",
                binding, index,
            ),
        }
    }
}
