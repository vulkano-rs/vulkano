// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::descriptor_set::layout::DescriptorType;

/// Layout of an image.
///
/// > **Note**: In vulkano, image layouts are mostly a low-level detail. You can ignore them,
/// > unless you use an unsafe function that states in its documentation that you must take care of
/// > an image's layout.
///
/// In the Vulkan API, each mipmap level of each array layer is in one of the layouts of this enum.
///
/// Unless you use some sort of high-level shortcut function, an image always starts in either
/// the `Undefined` or the `Preinitialized` layout.
/// Before you can use an image for a given purpose, you must ensure that the image in question is
/// in the layout required for that purpose. For example if you want to write data to an image, you
/// must first transition the image to the `TransferDstOptimal` layout. The `General` layout can
/// also be used as a general-purpose fit-all layout, but using it will result in slower operations.
///
/// Transitioning between layouts can only be done through a GPU-side operation that is part of
/// a command buffer.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(i32)]
pub enum ImageLayout {
    Undefined = ash::vk::ImageLayout::UNDEFINED.as_raw(),
    General = ash::vk::ImageLayout::GENERAL.as_raw(),
    ColorAttachmentOptimal = ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL.as_raw(),
    DepthStencilAttachmentOptimal = ash::vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL.as_raw(),
    DepthStencilReadOnlyOptimal = ash::vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL.as_raw(),
    ShaderReadOnlyOptimal = ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL.as_raw(),
    TransferSrcOptimal = ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL.as_raw(),
    TransferDstOptimal = ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL.as_raw(),
    Preinitialized = ash::vk::ImageLayout::PREINITIALIZED.as_raw(),
    PresentSrc = ash::vk::ImageLayout::PRESENT_SRC_KHR.as_raw(),
}

impl From<ImageLayout> for ash::vk::ImageLayout {
    #[inline]
    fn from(val: ImageLayout) -> Self {
        Self::from_raw(val as i32)
    }
}

/// The set of layouts to use for an image when used in descriptor of various kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageDescriptorLayouts {
    /// The image layout to use in a descriptor as a storage image.
    pub storage_image: ImageLayout,
    /// The image layout to use in a descriptor as a combined image sampler.
    pub combined_image_sampler: ImageLayout,
    /// The image layout to use in a descriptor as a sampled image.
    pub sampled_image: ImageLayout,
    /// The image layout to use in a descriptor as an input attachment.
    pub input_attachment: ImageLayout,
}

impl ImageDescriptorLayouts {
    /// Returns the layout for the given descriptor type. Panics if `descriptor_type` is not an
    /// image descriptor type.
    #[inline]
    pub fn layout_for(&self, descriptor_type: DescriptorType) -> ImageLayout {
        match descriptor_type {
            DescriptorType::CombinedImageSampler => self.combined_image_sampler,
            DescriptorType::SampledImage => self.sampled_image,
            DescriptorType::StorageImage => self.storage_image,
            DescriptorType::InputAttachment => self.input_attachment,
            _ => panic!("{:?} is not an image descriptor type", descriptor_type),
        }
    }
}
