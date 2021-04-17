// Copyright (c) 2020 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::vk;
use std::ops::BitOr;

/// An individual data type within an image.
///
/// Most images have only the `Color` aspect, but some may have several.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ImageAspect {
    Color = vk::IMAGE_ASPECT_COLOR_BIT,
    Depth = vk::IMAGE_ASPECT_DEPTH_BIT,
    Stencil = vk::IMAGE_ASPECT_STENCIL_BIT,
    Metadata = vk::IMAGE_ASPECT_METADATA_BIT,
    Plane0 = vk::IMAGE_ASPECT_PLANE_0_BIT,
    Plane1 = vk::IMAGE_ASPECT_PLANE_1_BIT,
    Plane2 = vk::IMAGE_ASPECT_PLANE_2_BIT,
    MemoryPlane0 = vk::IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT,
    MemoryPlane1 = vk::IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT,
    MemoryPlane2 = vk::IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT,
}

impl From<ImageAspect> for vk::ImageAspectFlags {
    #[inline]
    fn from(value: ImageAspect) -> vk::ImageAspectFlags {
        value as u32
    }
}

/// A mask specifying one or more `ImageAspect`s.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ImageAspects {
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

impl ImageAspects {
    /// Builds an `ImageAspect` with all values set to false. Useful as a default value.
    #[inline]
    pub const fn none() -> ImageAspects {
        ImageAspects {
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
}

impl BitOr for ImageAspects {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        ImageAspects {
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

impl From<ImageAspects> for vk::ImageAspectFlags {
    #[inline]
    fn from(value: ImageAspects) -> vk::ImageAspectFlags {
        let mut result = 0;
        if value.color {
            result |= vk::IMAGE_ASPECT_COLOR_BIT;
        }
        if value.depth {
            result |= vk::IMAGE_ASPECT_DEPTH_BIT;
        }
        if value.stencil {
            result |= vk::IMAGE_ASPECT_STENCIL_BIT;
        }
        if value.metadata {
            result |= vk::IMAGE_ASPECT_METADATA_BIT;
        }
        if value.plane0 {
            result |= vk::IMAGE_ASPECT_PLANE_0_BIT;
        }
        if value.plane1 {
            result |= vk::IMAGE_ASPECT_PLANE_1_BIT;
        }
        if value.plane2 {
            result |= vk::IMAGE_ASPECT_PLANE_2_BIT;
        }
        if value.memory_plane0 {
            result |= vk::IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT;
        }
        if value.memory_plane1 {
            result |= vk::IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT;
        }
        if value.memory_plane2 {
            result |= vk::IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT
        }
        result
    }
}

impl From<vk::ImageAspectFlags> for ImageAspects {
    #[inline]
    fn from(val: vk::ImageAspectFlags) -> ImageAspects {
        ImageAspects {
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
