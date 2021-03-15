// Copyright (c) 2020 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ops::BitOr;
use crate::vk;

/// Describes how an aspect of the image that be used to query Vulkan.  This is **not** just a suggestion.
/// Check out VkImageAspectFlagBits in the Vulkan spec.
///
/// If you specify an aspect of the image that doesn't exist (for example, depth for a YUV image), a panic
/// will happen.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ImageAspect {
    pub color: bool,
    pub depth: bool,
    pub stencil: bool,
    pub metadata: bool,
    pub plane0: bool,
    pub plane1: bool,
    pub plane2: bool,
    pub memory_plane0: bool,
    pub memory_plane1: bool,
    pub memory_plane2: bool,
}

impl ImageAspect {
    /// Builds a `ImageAspect` with all values set to false. Useful as a default value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use vulkano::image::ImageAspect as ImageAspect;
    ///
    /// let _aspect = ImageAspect {
    ///     color: true,
    ///     depth: true,
    ///     .. ImageAspect::none()
    /// };
    /// ```
    #[inline]
    pub fn none() -> ImageAspect {
        ImageAspect {
            color: false,
            depth: false,
            stencil: false,
            metadata: false,
            plane0: false,
            plane1: false,
            plane2: false,
            memory_plane0: false,
            memory_plane1: false,
            memory_plane2: false,
        }
    }

    #[inline]
    pub(crate) fn to_aspect_bits(&self) -> vk::ImageAspectFlagBits {
        let mut result = 0;
        if self.color {
            result |= vk::IMAGE_ASPECT_COLOR_BIT;
        }
        if self.depth {
            result |= vk::IMAGE_ASPECT_DEPTH_BIT;
        }
        if self.stencil {
            result |= vk::IMAGE_ASPECT_STENCIL_BIT;
        }
        if self.metadata {
            result |= vk::IMAGE_ASPECT_METADATA_BIT;
        }
        if self.plane0 {
            result |= vk::IMAGE_ASPECT_PLANE_0_BIT;
        }
        if self.plane1 {
            result |= vk::IMAGE_ASPECT_PLANE_1_BIT;
        }
        if self.plane2 {
            result |= vk::IMAGE_ASPECT_PLANE_2_BIT;
        }
        if self.memory_plane0 {
            result |= vk::IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT;
        }
        if self.memory_plane1 {
            result |= vk::IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT;
        }
        if self.memory_plane2 {
            result |= vk::IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT
        }
        result
    }

    pub(crate) fn from_bits(val: u32) -> ImageAspect {
        ImageAspect {
            color: (val & vk::IMAGE_ASPECT_COLOR_BIT) != 0,
            depth: (val & vk::IMAGE_ASPECT_DEPTH_BIT) != 0,
            stencil: (val & vk::IMAGE_ASPECT_STENCIL_BIT) != 0,
            metadata: (val & vk::IMAGE_ASPECT_METADATA_BIT) != 0,
            plane0: (val & vk::IMAGE_ASPECT_PLANE_0_BIT) != 0,
            plane1: (val & vk::IMAGE_ASPECT_PLANE_1_BIT) != 0,
            plane2: (val & vk::IMAGE_ASPECT_PLANE_2_BIT) != 0,
            memory_plane0: (val & vk::IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT) != 0,
            memory_plane1: (val & vk::IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT) != 0,
            memory_plane2: (val & vk::IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT) != 0,
        }
    }
}

impl BitOr for ImageAspect {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        ImageAspect {
            color: self.color || rhs.color,
            depth: self.depth || rhs.depth,
            stencil: self.stencil || rhs.stencil,
            metadata: self.metadata || rhs.metadata,
            plane0: self.plane0 || rhs.plane0,
            plane1: self.plane1 || rhs.plane1,
            plane2: self.plane2 || rhs.plane2,
            memory_plane0: self.memory_plane0 || rhs.memory_plane0,
            memory_plane1: self.memory_plane1 || rhs.memory_plane1,
            memory_plane2: self.memory_plane2 || rhs.memory_plane2,
        }
    }
}
