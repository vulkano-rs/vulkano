// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use vk;

/// Describes how an image is going to be used. This is **not** an optimization.
///
/// If you try to use an image in a way that you didn't declare, a panic will happen.
///
/// If `transient_attachment` is true, then only `color_attachment`, `depth_stencil_attachment`
/// and `input_attachment` can be true as well. The rest must be false or an error will be returned
/// when creating the image.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ImageUsage {
    /// Can be used a source for transfers. Includes blits.
    pub transfer_source: bool,

    /// Can be used a destination for transfers. Includes blits.
    pub transfer_dest: bool,

    /// Can be sampled from a shader.
    pub sampled: bool,

    /// Can be used as an image storage in a shader.
    pub storage: bool,

    /// Can be attached as a color attachment to a framebuffer.
    pub color_attachment: bool,

    /// Can be attached as a depth, stencil or depth-stencil attachment to a framebuffer.
    pub depth_stencil_attachment: bool,

    /// Indicates that this image will only ever be used as a temporary framebuffer attachment.
    /// As soon as you leave a render pass, the content of transient images becomes undefined.
    ///
    /// This is a hint to the Vulkan implementation that it may not need allocate any memory for
    /// this image if the image can live entirely in some cache.
    pub transient_attachment: bool,

    /// Can be used as an input attachment. In other words, you can draw to it in a subpass then
    /// read from it in a following pass.
    pub input_attachment: bool,
}

impl ImageUsage {
    /// Builds a `ImageUsage` with all values set to true. Note that using the returned value will
    /// produce an error because of `transient_attachment` being true.
    #[inline]
    pub fn all() -> ImageUsage {
        ImageUsage {
            transfer_source: true,
            transfer_dest: true,
            sampled: true,
            storage: true,
            color_attachment: true,
            depth_stencil_attachment: true,
            transient_attachment: true,
            input_attachment: true,
        }
    }

    /// Builds a `ImageUsage` with all values set to false. Useful as a default value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use vulkano::image::ImageUsage as ImageUsage;
    ///
    /// let _usage = ImageUsage {
    ///     transfer_dest: true,
    ///     sampled: true,
    ///     .. ImageUsage::none()
    /// };
    /// ```
    #[inline]
    pub fn none() -> ImageUsage {
        ImageUsage {
            transfer_source: false,
            transfer_dest: false,
            sampled: false,
            storage: false,
            color_attachment: false,
            depth_stencil_attachment: false,
            transient_attachment: false,
            input_attachment: false,
        }
    }

    #[doc(hidden)]
    #[inline]
    pub fn to_usage_bits(&self) -> vk::ImageUsageFlagBits {
        let mut result = 0;
        if self.transfer_source { result |= vk::IMAGE_USAGE_TRANSFER_SRC_BIT; }
        if self.transfer_dest { result |= vk::IMAGE_USAGE_TRANSFER_DST_BIT; }
        if self.sampled { result |= vk::IMAGE_USAGE_SAMPLED_BIT; }
        if self.storage { result |= vk::IMAGE_USAGE_STORAGE_BIT; }
        if self.color_attachment { result |= vk::IMAGE_USAGE_COLOR_ATTACHMENT_BIT; }
        if self.depth_stencil_attachment { result |= vk::IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT; }
        if self.transient_attachment { result |= vk::IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT; }
        if self.input_attachment { result |= vk::IMAGE_USAGE_INPUT_ATTACHMENT_BIT; }
        result
    }

    #[inline]
    #[doc(hidden)]
    pub fn from_bits(val: u32) -> ImageUsage {
        ImageUsage {
            transfer_source: (val & vk::IMAGE_USAGE_TRANSFER_SRC_BIT) != 0,
            transfer_dest: (val & vk::IMAGE_USAGE_TRANSFER_DST_BIT) != 0,
            sampled: (val & vk::IMAGE_USAGE_SAMPLED_BIT) != 0,
            storage: (val & vk::IMAGE_USAGE_STORAGE_BIT) != 0,
            color_attachment: (val & vk::IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0,
            depth_stencil_attachment: (val & vk::IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0,
            transient_attachment: (val & vk::IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) != 0,
            input_attachment: (val & vk::IMAGE_USAGE_INPUT_ATTACHMENT_BIT) != 0,
        }
    }
}
