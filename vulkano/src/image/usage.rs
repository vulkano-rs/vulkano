// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ops::BitOr;

/// Describes how an image is going to be used. This is **not** just an optimization.
///
/// If you try to use an image in a way that you didn't declare, a panic will happen.
///
/// If `transient_attachment` is true, then only `color_attachment`, `depth_stencil_attachment`
/// and `input_attachment` can be true as well. The rest must be false or an error will be returned
/// when creating the image.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ImageUsage {
    /// Can be used as a source for transfers. Includes blits.
    pub transfer_source: bool,

    /// Can be used as a destination for transfers. Includes blits.
    pub transfer_destination: bool,

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
            transfer_destination: true,
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
    ///     transfer_destination: true,
    ///     sampled: true,
    ///     .. ImageUsage::none()
    /// };
    /// ```
    #[inline]
    pub fn none() -> ImageUsage {
        ImageUsage {
            transfer_source: false,
            transfer_destination: false,
            sampled: false,
            storage: false,
            color_attachment: false,
            depth_stencil_attachment: false,
            transient_attachment: false,
            input_attachment: false,
        }
    }

    /// Builds a ImageUsage with color_attachment set to true and the rest to false.
    #[inline]
    pub fn color_attachment() -> ImageUsage {
        ImageUsage {
            transfer_source: false,
            transfer_destination: false,
            sampled: false,
            storage: false,
            color_attachment: true,
            depth_stencil_attachment: false,
            transient_attachment: false,
            input_attachment: false,
        }
    }

    /// Builds a ImageUsage with depth_stencil_attachment set to true and the rest to false.
    #[inline]
    pub fn depth_stencil_attachment() -> ImageUsage {
        ImageUsage {
            transfer_source: false,
            transfer_destination: false,
            sampled: false,
            storage: false,
            color_attachment: false,
            depth_stencil_attachment: true,
            transient_attachment: false,
            input_attachment: false,
        }
    }

    /// Builds a ImageUsage with color_attachment and transient_attachment set to true and the rest to false.
    #[inline]
    pub fn transient_color_attachment() -> ImageUsage {
        ImageUsage {
            transfer_source: false,
            transfer_destination: false,
            sampled: false,
            storage: false,
            color_attachment: true,
            depth_stencil_attachment: false,
            transient_attachment: true,
            input_attachment: false,
        }
    }

    /// Builds a ImageUsage with depth_stencil_attachment and transient_attachment set to true and the rest to false.
    #[inline]
    pub fn transient_depth_stencil_attachment() -> ImageUsage {
        ImageUsage {
            transfer_source: false,
            transfer_destination: false,
            sampled: false,
            storage: false,
            color_attachment: false,
            depth_stencil_attachment: true,
            transient_attachment: true,
            input_attachment: false,
        }
    }
}

impl From<ImageUsage> for ash::vk::ImageUsageFlags {
    #[inline]
    fn from(val: ImageUsage) -> Self {
        let mut result = ash::vk::ImageUsageFlags::empty();
        if val.transfer_source {
            result |= ash::vk::ImageUsageFlags::TRANSFER_SRC;
        }
        if val.transfer_destination {
            result |= ash::vk::ImageUsageFlags::TRANSFER_DST;
        }
        if val.sampled {
            result |= ash::vk::ImageUsageFlags::SAMPLED;
        }
        if val.storage {
            result |= ash::vk::ImageUsageFlags::STORAGE;
        }
        if val.color_attachment {
            result |= ash::vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        if val.depth_stencil_attachment {
            result |= ash::vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        }
        if val.transient_attachment {
            result |= ash::vk::ImageUsageFlags::TRANSIENT_ATTACHMENT;
        }
        if val.input_attachment {
            result |= ash::vk::ImageUsageFlags::INPUT_ATTACHMENT;
        }
        result
    }
}

impl From<ash::vk::ImageUsageFlags> for ImageUsage {
    #[inline]
    fn from(val: ash::vk::ImageUsageFlags) -> ImageUsage {
        ImageUsage {
            transfer_source: !(val & ash::vk::ImageUsageFlags::TRANSFER_SRC).is_empty(),
            transfer_destination: !(val & ash::vk::ImageUsageFlags::TRANSFER_DST).is_empty(),
            sampled: !(val & ash::vk::ImageUsageFlags::SAMPLED).is_empty(),
            storage: !(val & ash::vk::ImageUsageFlags::STORAGE).is_empty(),
            color_attachment: !(val & ash::vk::ImageUsageFlags::COLOR_ATTACHMENT).is_empty(),
            depth_stencil_attachment: !(val & ash::vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .is_empty(),
            transient_attachment: !(val & ash::vk::ImageUsageFlags::TRANSIENT_ATTACHMENT)
                .is_empty(),
            input_attachment: !(val & ash::vk::ImageUsageFlags::INPUT_ATTACHMENT).is_empty(),
        }
    }
}

impl BitOr for ImageUsage {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        ImageUsage {
            transfer_source: self.transfer_source || rhs.transfer_source,
            transfer_destination: self.transfer_destination || rhs.transfer_destination,
            sampled: self.sampled || rhs.sampled,
            storage: self.storage || rhs.storage,
            color_attachment: self.color_attachment || rhs.color_attachment,
            depth_stencil_attachment: self.depth_stencil_attachment || rhs.depth_stencil_attachment,
            transient_attachment: self.transient_attachment || rhs.transient_attachment,
            input_attachment: self.input_attachment || rhs.input_attachment,
        }
    }
}
