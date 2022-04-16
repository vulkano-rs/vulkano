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
    buffer::{view::BufferViewAbstract, BufferAccess, BufferInner},
    device::DeviceOwned,
    image::{view::ImageViewType, ImageType, ImageViewAbstract},
    sampler::{Sampler, SamplerImageViewIncompatibleError},
    DeviceSize, VulkanObject,
};
use smallvec::SmallVec;
use std::{ptr, sync::Arc};

/// Represents a single write operation to the binding of a descriptor set.
///
/// `WriteDescriptorSet` specifies the binding number and target array index, and includes one or
/// more resources of a given type that need to be written to that location. Two constructors are
/// provided for each resource type:
/// - The basic constructor variant writes a single element to array index 0. It is intended for
///   non-arrayed bindings, where `descriptor_count` in the descriptor set layout is 1.
/// - The `_array` variant writes several elements and allows specifying the target array index.
///   At least one element must be provided; a panic results if the provided iterator is empty.
pub struct WriteDescriptorSet {
    binding: u32,
    first_array_element: u32,
    elements: WriteDescriptorSetElements,
}

impl WriteDescriptorSet {
    /// Write an empty element to array element 0.
    ///
    /// See `none_array` for more information.
    #[inline]
    pub fn none(binding: u32) -> Self {
        Self::none_array(binding, 0, 1)
    }

    /// Write a number of consecutive empty elements.
    ///
    /// This is used for push descriptors in combination with `Sampler` descriptors that have
    /// immutable samplers in the layout. The Vulkan spec requires these elements to be explicitly
    /// written, but since there is no data to write, a dummy write is provided instead.
    ///
    /// For regular descriptor sets, the data for such descriptors is automatically valid, and dummy
    /// writes are not allowed.
    #[inline]
    pub fn none_array(binding: u32, first_array_element: u32, num_elements: u32) -> Self {
        assert!(num_elements != 0);
        Self {
            binding,
            first_array_element,
            elements: WriteDescriptorSetElements::None(num_elements),
        }
    }

    /// Write a single buffer to array element 0.
    #[inline]
    pub fn buffer(binding: u32, buffer: Arc<dyn BufferAccess>) -> Self {
        Self::buffer_array(binding, 0, [buffer])
    }

    /// Write a number of consecutive buffer elements.
    #[inline]
    pub fn buffer_array(
        binding: u32,
        first_array_element: u32,
        elements: impl IntoIterator<Item = Arc<dyn BufferAccess>>,
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
    pub fn buffer_view(binding: u32, buffer_view: Arc<dyn BufferViewAbstract>) -> Self {
        Self::buffer_view_array(binding, 0, [buffer_view])
    }

    /// Write a number of consecutive buffer view elements.
    #[inline]
    pub fn buffer_view_array(
        binding: u32,
        first_array_element: u32,
        elements: impl IntoIterator<Item = Arc<dyn BufferViewAbstract>>,
    ) -> Self {
        let elements: SmallVec<_> = elements.into_iter().collect();
        assert!(!elements.is_empty());
        Self {
            binding,
            first_array_element,
            elements: WriteDescriptorSetElements::BufferView(elements),
        }
    }

    /// Write a single image view to array element 0.
    #[inline]
    pub fn image_view(binding: u32, image_view: Arc<dyn ImageViewAbstract>) -> Self {
        Self::image_view_array(binding, 0, [image_view])
    }

    /// Write a number of consecutive image view elements.
    #[inline]
    pub fn image_view_array(
        binding: u32,
        first_array_element: u32,
        elements: impl IntoIterator<Item = Arc<dyn ImageViewAbstract>>,
    ) -> Self {
        let elements: SmallVec<_> = elements.into_iter().collect();
        assert!(!elements.is_empty());
        Self {
            binding,
            first_array_element,
            elements: WriteDescriptorSetElements::ImageView(elements),
        }
    }

    /// Write a single image view and sampler to array element 0.
    #[inline]
    pub fn image_view_sampler(
        binding: u32,
        image_view: Arc<dyn ImageViewAbstract>,
        sampler: Arc<Sampler>,
    ) -> Self {
        Self::image_view_sampler_array(binding, 0, [(image_view, sampler)])
    }

    /// Write a number of consecutive image view and sampler elements.
    #[inline]
    pub fn image_view_sampler_array(
        binding: u32,
        first_array_element: u32,
        elements: impl IntoIterator<Item = (Arc<dyn ImageViewAbstract>, Arc<Sampler>)>,
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
    #[inline]
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
                        .map(|buffer| {
                            let size = buffer.size();
                            let BufferInner { buffer, offset } = buffer.inner();

                            debug_assert_eq!(
                                offset
                                    % buffer
                                        .device()
                                        .physical_device()
                                        .properties()
                                        .min_storage_buffer_offset_alignment,
                                0
                            );
                            debug_assert!(
                                size <= buffer
                                    .device()
                                    .physical_device()
                                    .properties()
                                    .max_storage_buffer_range
                                    as DeviceSize
                            );
                            ash::vk::DescriptorBufferInfo {
                                buffer: buffer.internal_object(),
                                offset,
                                range: size,
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
                        .map(|buffer_view| buffer_view.internal_object())
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
                        .map(|image_view| {
                            let layouts = image_view.image().descriptor_layouts().expect(
                                "descriptor_layouts must return Some when used in an image view",
                            );
                            ash::vk::DescriptorImageInfo {
                                sampler: ash::vk::Sampler::null(),
                                image_view: image_view.internal_object(),
                                image_layout: layouts.layout_for(descriptor_type).into(),
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
                        .map(|(image_view, sampler)| {
                            let layouts = image_view.image().descriptor_layouts().expect(
                                "descriptor_layouts must return Some when used in an image view",
                            );
                            ash::vk::DescriptorImageInfo {
                                sampler: sampler.internal_object(),
                                image_view: image_view.internal_object(),
                                image_layout: layouts.layout_for(descriptor_type).into(),
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
                            sampler: sampler.internal_object(),
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
pub enum WriteDescriptorSetElements {
    None(u32),
    Buffer(SmallVec<[Arc<dyn BufferAccess>; 1]>),
    BufferView(SmallVec<[Arc<dyn BufferViewAbstract>; 1]>),
    ImageView(SmallVec<[Arc<dyn ImageViewAbstract>; 1]>),
    ImageViewSampler(SmallVec<[(Arc<dyn ImageViewAbstract>, Arc<Sampler>); 1]>),
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

#[derive(Clone, Debug)]
pub(crate) enum DescriptorWriteInfo {
    Image(SmallVec<[ash::vk::DescriptorImageInfo; 1]>),
    Buffer(SmallVec<[ash::vk::DescriptorBufferInfo; 1]>),
    BufferView(SmallVec<[ash::vk::BufferView; 1]>),
}

impl DescriptorWriteInfo {
    fn set_info(&self, write: &mut ash::vk::WriteDescriptorSet) {
        match self {
            DescriptorWriteInfo::Image(info) => {
                write.descriptor_count = info.len() as u32;
                write.p_image_info = info.as_ptr();
            }
            DescriptorWriteInfo::Buffer(info) => {
                write.descriptor_count = info.len() as u32;
                write.p_buffer_info = info.as_ptr();
            }
            DescriptorWriteInfo::BufferView(info) => {
                write.descriptor_count = info.len() as u32;
                write.p_texel_buffer_view = info.as_ptr();
            }
        }

        debug_assert!(write.descriptor_count != 0);
    }
}

pub(crate) fn check_descriptor_write<'a>(
    write: &WriteDescriptorSet,
    layout: &'a DescriptorSetLayout,
    variable_descriptor_count: u32,
) -> Result<&'a DescriptorSetLayoutBinding, DescriptorSetUpdateError> {
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

    let elements = write.elements();
    let num_elements = elements.len();
    debug_assert!(num_elements != 0);

    let descriptor_range_start = write.first_array_element();
    let descriptor_range_end = descriptor_range_start + num_elements;

    if descriptor_range_end > max_descriptor_count {
        return Err(DescriptorSetUpdateError::ArrayIndexOutOfBounds {
            binding: write.binding(),
            available_count: max_descriptor_count,
            written_count: descriptor_range_end,
        });
    }

    match elements {
        WriteDescriptorSetElements::None(num_elements) => match layout_binding.descriptor_type {
            DescriptorType::Sampler
                if layout.push_descriptor() && !layout_binding.immutable_samplers.is_empty() => {}
            _ => {
                return Err(DescriptorSetUpdateError::IncompatibleDescriptorType {
                    binding: write.binding(),
                })
            }
        },
        WriteDescriptorSetElements::Buffer(elements) => {
            match layout_binding.descriptor_type {
                DescriptorType::StorageBuffer | DescriptorType::StorageBufferDynamic => {
                    for (index, buffer) in elements.iter().enumerate() {
                        assert_eq!(
                            buffer.device().internal_object(),
                            layout.device().internal_object(),
                        );

                        if !buffer.inner().buffer.usage().storage_buffer {
                            return Err(DescriptorSetUpdateError::MissingUsage {
                                binding: write.binding(),
                                index: descriptor_range_start + index as u32,
                                usage: "storage_buffer",
                            });
                        }
                    }
                }
                DescriptorType::UniformBuffer | DescriptorType::UniformBufferDynamic => {
                    for (index, buffer) in elements.iter().enumerate() {
                        assert_eq!(
                            buffer.device().internal_object(),
                            layout.device().internal_object(),
                        );

                        if !buffer.inner().buffer.usage().uniform_buffer {
                            return Err(DescriptorSetUpdateError::MissingUsage {
                                binding: write.binding(),
                                index: descriptor_range_start + index as u32,
                                usage: "uniform_buffer",
                            });
                        }
                    }
                }
                _ => {
                    return Err(DescriptorSetUpdateError::IncompatibleDescriptorType {
                        binding: write.binding(),
                    })
                }
            }

            // Note that the buffer content is not checked. This is technically not unsafe as
            // long as the data in the buffer has no invalid memory representation (ie. no
            // bool, no enum, no pointer, no str) and as long as the robust buffer access
            // feature is enabled.
            // TODO: this is not checked ^

            // TODO: eventually shouldn't be an assert ; for now robust_buffer_access is always
            //       enabled so this assert should never fail in practice, but we put it anyway
            //       in case we forget to adjust this code
            assert!(layout.device().enabled_features().robust_buffer_access);
        }
        WriteDescriptorSetElements::BufferView(elements) => {
            match layout_binding.descriptor_type {
                DescriptorType::StorageTexelBuffer => {
                    for (index, buffer_view) in elements.iter().enumerate() {
                        assert_eq!(
                            buffer_view.device().internal_object(),
                            layout.device().internal_object(),
                        );

                        // TODO: storage_texel_buffer_atomic
                        if !buffer_view
                            .buffer()
                            .inner()
                            .buffer
                            .usage()
                            .storage_texel_buffer
                        {
                            return Err(DescriptorSetUpdateError::MissingUsage {
                                binding: write.binding(),
                                index: descriptor_range_start + index as u32,
                                usage: "storage_texel_buffer",
                            });
                        }
                    }
                }
                DescriptorType::UniformTexelBuffer => {
                    for (index, buffer_view) in elements.iter().enumerate() {
                        assert_eq!(
                            buffer_view.device().internal_object(),
                            layout.device().internal_object(),
                        );

                        if !buffer_view
                            .buffer()
                            .inner()
                            .buffer
                            .usage()
                            .uniform_texel_buffer
                        {
                            return Err(DescriptorSetUpdateError::MissingUsage {
                                binding: write.binding(),
                                index: descriptor_range_start + index as u32,
                                usage: "uniform_texel_buffer",
                            });
                        }
                    }
                }
                _ => {
                    return Err(DescriptorSetUpdateError::IncompatibleDescriptorType {
                        binding: write.binding(),
                    })
                }
            }
        }
        WriteDescriptorSetElements::ImageView(elements) => match layout_binding.descriptor_type {
            DescriptorType::CombinedImageSampler
                if !layout_binding.immutable_samplers.is_empty() =>
            {
                let immutable_samplers = &layout_binding.immutable_samplers
                    [descriptor_range_start as usize..descriptor_range_end as usize];

                for (index, (image_view, sampler)) in
                    elements.iter().zip(immutable_samplers).enumerate()
                {
                    assert_eq!(
                        image_view.device().internal_object(),
                        layout.device().internal_object(),
                    );

                    // VUID-VkWriteDescriptorSet-descriptorType-00337
                    if !image_view.usage().sampled {
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
                    if image_view.subresource_range().aspects.depth
                        && image_view.subresource_range().aspects.stencil
                    {
                        return Err(DescriptorSetUpdateError::ImageViewDepthAndStencil {
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
            DescriptorType::SampledImage => {
                for (index, image_view) in elements.iter().enumerate() {
                    assert_eq!(
                        image_view.device().internal_object(),
                        layout.device().internal_object(),
                    );

                    // VUID-VkWriteDescriptorSet-descriptorType-00337
                    if !image_view.usage().sampled {
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
                    if image_view.subresource_range().aspects.depth
                        && image_view.subresource_range().aspects.stencil
                    {
                        return Err(DescriptorSetUpdateError::ImageViewDepthAndStencil {
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
                for (index, image_view) in elements.iter().enumerate() {
                    assert_eq!(
                        image_view.device().internal_object(),
                        layout.device().internal_object(),
                    );

                    // VUID-VkWriteDescriptorSet-descriptorType-00339
                    if !image_view.usage().storage {
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
                    if image_view.subresource_range().aspects.depth
                        && image_view.subresource_range().aspects.stencil
                    {
                        return Err(DescriptorSetUpdateError::ImageViewDepthAndStencil {
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
            DescriptorType::InputAttachment => {
                for (index, image_view) in elements.iter().enumerate() {
                    assert_eq!(
                        image_view.device().internal_object(),
                        layout.device().internal_object(),
                    );

                    // VUID-VkWriteDescriptorSet-descriptorType-00338
                    if !image_view.usage().input_attachment {
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
                    if image_view.subresource_range().aspects.depth
                        && image_view.subresource_range().aspects.stencil
                    {
                        return Err(DescriptorSetUpdateError::ImageViewDepthAndStencil {
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
            _ => {
                return Err(DescriptorSetUpdateError::IncompatibleDescriptorType {
                    binding: write.binding(),
                })
            }
        },
        WriteDescriptorSetElements::ImageViewSampler(elements) => match layout_binding
            .descriptor_type
        {
            DescriptorType::CombinedImageSampler => {
                if !layout_binding.immutable_samplers.is_empty() {
                    return Err(DescriptorSetUpdateError::SamplerIsImmutable {
                        binding: write.binding(),
                    });
                }

                for (index, (image_view, sampler)) in elements.iter().enumerate() {
                    assert_eq!(
                        image_view.device().internal_object(),
                        layout.device().internal_object(),
                    );
                    assert_eq!(
                        sampler.device().internal_object(),
                        layout.device().internal_object(),
                    );

                    // VUID-VkWriteDescriptorSet-descriptorType-00337
                    if !image_view.usage().sampled {
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
                    if image_view.subresource_range().aspects.depth
                        && image_view.subresource_range().aspects.stencil
                    {
                        return Err(DescriptorSetUpdateError::ImageViewDepthAndStencil {
                            binding: write.binding(),
                            index: descriptor_range_start + index as u32,
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
            }
            _ => {
                return Err(DescriptorSetUpdateError::IncompatibleDescriptorType {
                    binding: write.binding(),
                })
            }
        },
        WriteDescriptorSetElements::Sampler(elements) => match layout_binding.descriptor_type {
            DescriptorType::Sampler => {
                if !layout_binding.immutable_samplers.is_empty() {
                    return Err(DescriptorSetUpdateError::SamplerIsImmutable {
                        binding: write.binding(),
                    });
                }

                for (index, sampler) in elements.iter().enumerate() {
                    assert_eq!(
                        sampler.device().internal_object(),
                        layout.device().internal_object(),
                    );

                    if sampler.sampler_ycbcr_conversion().is_some() {
                        return Err(DescriptorSetUpdateError::SamplerHasSamplerYcbcrConversion {
                            binding: write.binding(),
                            index: descriptor_range_start + index as u32,
                        });
                    }
                }
            }
            _ => {
                return Err(DescriptorSetUpdateError::IncompatibleDescriptorType {
                    binding: write.binding(),
                })
            }
        },
    }

    Ok(layout_binding)
}

#[derive(Clone, Copy, Debug)]
pub enum DescriptorSetUpdateError {
    /// Tried to write more elements than were available in a binding.
    ArrayIndexOutOfBounds {
        /// Binding that is affected.
        binding: u32,
        /// Number of available descriptors in the binding.
        available_count: u32,
        /// The number of descriptors that were in the update.
        written_count: u32,
    },

    /// Tried to write an image view with a 2D type and a 3D underlying image.
    ImageView2dFrom3d { binding: u32, index: u32 },

    /// Tried to write an image view that has both the `depth` and `stencil` aspects.
    ImageViewDepthAndStencil { binding: u32, index: u32 },

    /// Tried to write an image view with an attached sampler YCbCr conversion to a binding that
    /// does not support it.
    ImageViewHasSamplerYcbcrConversion { binding: u32, index: u32 },

    /// Tried to write an image view of an arrayed type to a descriptor type that does not support
    /// it.
    ImageViewIsArrayed { binding: u32, index: u32 },

    /// Tried to write an image view that was not compatible with the sampler that was provided as
    /// part of the update or immutably in the layout.
    ImageViewIncompatibleSampler {
        binding: u32,
        index: u32,
        error: SamplerImageViewIncompatibleError,
    },

    /// Tried to write an image view to a descriptor type that requires it to be identity swizzled,
    /// but it was not.
    ImageViewNotIdentitySwizzled { binding: u32, index: u32 },

    /// Tried to write an element type that was not compatible with the descriptor type in the
    /// layout.
    IncompatibleDescriptorType { binding: u32 },

    /// Tried to write to a nonexistent binding.
    InvalidBinding { binding: u32 },

    /// A resource was missing a usage flag that was required.
    MissingUsage {
        binding: u32,
        index: u32,
        usage: &'static str,
    },

    /// Tried to write a sampler that has an attached sampler YCbCr conversion.
    SamplerHasSamplerYcbcrConversion { binding: u32, index: u32 },

    /// Tried to write a sampler to a binding with immutable samplers.
    SamplerIsImmutable { binding: u32 },
}

impl std::error::Error for DescriptorSetUpdateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ImageViewIncompatibleSampler { error, .. } => Some(error),
            _ => None,
        }
    }
}

impl std::fmt::Display for DescriptorSetUpdateError {
    #[inline]
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        match self {
            Self::ArrayIndexOutOfBounds {
                binding,
                available_count,
                written_count,
            } => write!(
                fmt,
                "tried to write up to element {} to binding {}, but only {} descriptors are available",
                written_count, binding, available_count,
            ),
            Self::ImageView2dFrom3d { binding, index } => write!(
                fmt,
                "tried to write an image view to binding {} index {} with a 2D type and a 3D underlying image",
                binding, index,
            ),
            Self::ImageViewDepthAndStencil { binding, index } => write!(
                fmt,
                "tried to write an image view to binding {} index {} that has both the `depth` and `stencil` aspects",
                binding, index,
            ),
            Self::ImageViewHasSamplerYcbcrConversion { binding, index } => write!(
                fmt,
                "tried to write an image view to binding {} index {} with an attached sampler YCbCr conversion to binding that does not support it",
                binding, index,
            ),
            Self::ImageViewIsArrayed { binding, index } => write!(
                fmt,
                "tried to write an image view of an arrayed type to binding {} index {}, but this binding has a descriptor type that does not support arrayed image views",
                binding, index,
            ),
            Self::ImageViewIncompatibleSampler { binding, index, .. } => write!(
                fmt,
                "tried to write an image view to binding {} index {}, that was not compatible with the sampler that was provided as part of the update or immutably in the layout",
                binding, index,
            ),
            Self::ImageViewNotIdentitySwizzled { binding, index } => write!(
                fmt,
                "tried to write an image view with non-identity swizzling to binding {} index {}, but this binding has a descriptor type that requires it to be identity swizzled",
                binding, index,
            ),
            Self::IncompatibleDescriptorType { binding } => write!(
                fmt,
                "tried to write a resource to binding {} whose type was not compatible with the descriptor type",
                binding,
            ),
            Self::InvalidBinding { binding } => write!(
                fmt,
                "tried to write to a nonexistent binding {}",
                binding,
            ),
            Self::MissingUsage {
                binding,
                index,
                usage,
            } => write!(
                fmt,
                "tried to write a resource to binding {} index {} that did not have the required usage {} enabled",
                binding, index, usage,
            ),
            Self::SamplerHasSamplerYcbcrConversion { binding, index } => write!(
                fmt,
                "tried to write a sampler to binding {} index {} that has an attached sampler YCbCr conversion",
                binding, index,
            ),
            Self::SamplerIsImmutable { binding } => write!(
                fmt,
                "tried to write a sampler to binding {}, which already contains immutable samplers in the descriptor set layout",
                binding,
            ),
        }
    }
}
