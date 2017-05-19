// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use vk;

/// Layout of an image.
///
/// > **Note**: In vulkano, image layouts are mostly a low-level detail. You can ignore them,
/// > unless you use an unsafe function that states in its documentation that you must take care of
/// > an image's layout.
///
/// In the Vulkan API, each mipmap level of each array layer is in one of the layouts of this enum.
///
/// Unless you use some short of high-level shortcut function, an image always starts in either
/// the `Undefined` or the `Preinitialized` layout.
/// Before you can use an image for a given purpose, you must ensure that the image in question is
/// in the layout required for that purpose. For example if you want to write data to an image, you
/// must first transition the image to the `TransferDstOptimal` layout. The `General` layout can
/// also be used as a general-purpose fit-all layout, but using it will result in slower operations.
///
/// Transitionning between layouts can only be done through a GPU-side operation that is part of
/// a command buffer.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum ImageLayout {
    Undefined = vk::IMAGE_LAYOUT_UNDEFINED,
    General = vk::IMAGE_LAYOUT_GENERAL,
    ColorAttachmentOptimal = vk::IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    DepthStencilAttachmentOptimal = vk::IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    DepthStencilReadOnlyOptimal = vk::IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
    ShaderReadOnlyOptimal = vk::IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    TransferSrcOptimal = vk::IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    TransferDstOptimal = vk::IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    Preinitialized = vk::IMAGE_LAYOUT_PREINITIALIZED,
    PresentSrc = vk::IMAGE_LAYOUT_PRESENT_SRC_KHR,
}
