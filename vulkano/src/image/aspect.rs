// Copyright (c) 2020 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ops::BitOr;

/// An individual data type within an image.
///
/// Most images have only the `Color` aspect, but some may have several.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u32)]
pub enum ImageAspect {
    Color = ash::vk::ImageAspectFlags::COLOR.as_raw(),
    Depth = ash::vk::ImageAspectFlags::DEPTH.as_raw(),
    Stencil = ash::vk::ImageAspectFlags::STENCIL.as_raw(),
    Metadata = ash::vk::ImageAspectFlags::METADATA.as_raw(),
    Plane0 = ash::vk::ImageAspectFlags::PLANE_0.as_raw(),
    Plane1 = ash::vk::ImageAspectFlags::PLANE_1.as_raw(),
    Plane2 = ash::vk::ImageAspectFlags::PLANE_2.as_raw(),
    MemoryPlane0 = ash::vk::ImageAspectFlags::MEMORY_PLANE_0_EXT.as_raw(),
    MemoryPlane1 = ash::vk::ImageAspectFlags::MEMORY_PLANE_1_EXT.as_raw(),
    MemoryPlane2 = ash::vk::ImageAspectFlags::MEMORY_PLANE_2_EXT.as_raw(),
}

impl From<ImageAspect> for ash::vk::ImageAspectFlags {
    #[inline]
    fn from(val: ImageAspect) -> Self {
        Self::from_raw(val as u32)
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

    pub const fn contains(&self, other: &Self) -> bool {
        let Self {
            color,
            depth,
            stencil,
            metadata,
            plane0,
            plane1,
            plane2,
            memory_plane0,
            memory_plane1,
            memory_plane2,
        } = *self;

        (color || !other.color)
            && (depth || !other.depth)
            && (stencil || !other.stencil)
            && (metadata || !other.metadata)
            && (plane0 || !other.plane0)
            && (plane1 || !other.plane1)
            && (plane2 || !other.plane2)
            && (memory_plane0 || !other.memory_plane0)
            && (memory_plane1 || !other.memory_plane1)
            && (memory_plane2 || !other.memory_plane2)
    }

    pub fn iter(&self) -> impl Iterator<Item = ImageAspect> {
        let Self {
            color,
            depth,
            stencil,
            metadata,
            plane0,
            plane1,
            plane2,
            memory_plane0,
            memory_plane1,
            memory_plane2,
        } = *self;

        [
            color.then(|| ImageAspect::Color),
            depth.then(|| ImageAspect::Depth),
            stencil.then(|| ImageAspect::Stencil),
            metadata.then(|| ImageAspect::Metadata),
            plane0.then(|| ImageAspect::Plane0),
            plane1.then(|| ImageAspect::Plane1),
            plane2.then(|| ImageAspect::Plane2),
            memory_plane0.then(|| ImageAspect::MemoryPlane0),
            memory_plane1.then(|| ImageAspect::MemoryPlane1),
            memory_plane2.then(|| ImageAspect::MemoryPlane2),
        ]
        .into_iter()
        .flatten()
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

impl From<ImageAspects> for ash::vk::ImageAspectFlags {
    #[inline]
    fn from(value: ImageAspects) -> ash::vk::ImageAspectFlags {
        let mut result = ash::vk::ImageAspectFlags::empty();
        if value.color {
            result |= ash::vk::ImageAspectFlags::COLOR;
        }
        if value.depth {
            result |= ash::vk::ImageAspectFlags::DEPTH;
        }
        if value.stencil {
            result |= ash::vk::ImageAspectFlags::STENCIL;
        }
        if value.metadata {
            result |= ash::vk::ImageAspectFlags::METADATA;
        }
        if value.plane0 {
            result |= ash::vk::ImageAspectFlags::PLANE_0;
        }
        if value.plane1 {
            result |= ash::vk::ImageAspectFlags::PLANE_1;
        }
        if value.plane2 {
            result |= ash::vk::ImageAspectFlags::PLANE_2;
        }
        if value.memory_plane0 {
            result |= ash::vk::ImageAspectFlags::MEMORY_PLANE_0_EXT;
        }
        if value.memory_plane1 {
            result |= ash::vk::ImageAspectFlags::MEMORY_PLANE_1_EXT;
        }
        if value.memory_plane2 {
            result |= ash::vk::ImageAspectFlags::MEMORY_PLANE_2_EXT
        }
        result
    }
}

impl From<ash::vk::ImageAspectFlags> for ImageAspects {
    #[inline]
    fn from(val: ash::vk::ImageAspectFlags) -> ImageAspects {
        ImageAspects {
            color: !(val & ash::vk::ImageAspectFlags::COLOR).is_empty(),
            depth: !(val & ash::vk::ImageAspectFlags::DEPTH).is_empty(),
            stencil: !(val & ash::vk::ImageAspectFlags::STENCIL).is_empty(),
            metadata: !(val & ash::vk::ImageAspectFlags::METADATA).is_empty(),
            plane0: !(val & ash::vk::ImageAspectFlags::PLANE_0).is_empty(),
            plane1: !(val & ash::vk::ImageAspectFlags::PLANE_1).is_empty(),
            plane2: !(val & ash::vk::ImageAspectFlags::PLANE_2).is_empty(),
            memory_plane0: !(val & ash::vk::ImageAspectFlags::MEMORY_PLANE_0_EXT).is_empty(),
            memory_plane1: !(val & ash::vk::ImageAspectFlags::MEMORY_PLANE_1_EXT).is_empty(),
            memory_plane2: !(val & ash::vk::ImageAspectFlags::MEMORY_PLANE_2_EXT).is_empty(),
        }
    }
}
