// Copyright (c) 2020 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::macros::{vulkan_bitflags, vulkan_enum};

vulkan_enum! {
    /// An individual data type within an image.
    ///
    /// Most images have only the `Color` aspect, but some may have several.
    #[non_exhaustive]
    ImageAspect = ImageAspectFlags(u32);

    // TODO: document
    Color = COLOR,

    // TODO: document
    Depth = DEPTH,

    // TODO: document
    Stencil = STENCIL,

    // TODO: document
    Metadata = METADATA,

    // TODO: document
    Plane0 = PLANE_0 {
        api_version: V1_1,
        device_extensions: [khr_sampler_ycbcr_conversion],
    },

    // TODO: document
    Plane1 = PLANE_1 {
        api_version: V1_1,
        device_extensions: [khr_sampler_ycbcr_conversion],
    },

    // TODO: document
    Plane2 = PLANE_2 {
        api_version: V1_1,
        device_extensions: [khr_sampler_ycbcr_conversion],
    },

    // TODO: document
    MemoryPlane0 = MEMORY_PLANE_0_EXT {
        device_extensions: [ext_image_drm_format_modifier],
    },

    // TODO: document
    MemoryPlane1 = MEMORY_PLANE_1_EXT {
        device_extensions: [ext_image_drm_format_modifier],
    },

    // TODO: document
    MemoryPlane2 = MEMORY_PLANE_2_EXT {
        device_extensions: [ext_image_drm_format_modifier],
    },
}

vulkan_bitflags! {
    /// A mask specifying one or more `ImageAspect`s.
    #[non_exhaustive]
    ImageAspects = ImageAspectFlags(u32);

    // TODO: document
    COLOR = COLOR,

    // TODO: document
    DEPTH = DEPTH,

    // TODO: document
    STENCIL = STENCIL,

    // TODO: document
    METADATA = METADATA,

    // TODO: document
    PLANE_0 = PLANE_0 {
        api_version: V1_1,
        device_extensions: [khr_sampler_ycbcr_conversion],
    },

    // TODO: document
    PLANE_1 = PLANE_1 {
        api_version: V1_1,
        device_extensions: [khr_sampler_ycbcr_conversion],
    },

    // TODO: document
    PLANE_2 = PLANE_2 {
        api_version: V1_1,
        device_extensions: [khr_sampler_ycbcr_conversion],
    },

    // TODO: document
    MEMORY_PLANE_0 = MEMORY_PLANE_0_EXT {
        device_extensions: [ext_image_drm_format_modifier],
    },

    // TODO: document
    MEMORY_PLANE_1 = MEMORY_PLANE_1_EXT {
        device_extensions: [ext_image_drm_format_modifier],
    },

    // TODO: document
    MEMORY_PLANE_2 = MEMORY_PLANE_2_EXT {
        device_extensions: [ext_image_drm_format_modifier],
    },
}

impl ImageAspects {
    #[inline]
    pub fn iter(self) -> impl Iterator<Item = ImageAspect> {
        [
            self.intersects(ImageAspects::COLOR)
                .then_some(ImageAspect::Color),
            self.intersects(ImageAspects::DEPTH)
                .then_some(ImageAspect::Depth),
            self.intersects(ImageAspects::STENCIL)
                .then_some(ImageAspect::Stencil),
            self.intersects(ImageAspects::METADATA)
                .then_some(ImageAspect::Metadata),
            self.intersects(ImageAspects::PLANE_0)
                .then_some(ImageAspect::Plane0),
            self.intersects(ImageAspects::PLANE_1)
                .then_some(ImageAspect::Plane1),
            self.intersects(ImageAspects::PLANE_2)
                .then_some(ImageAspect::Plane2),
            self.intersects(ImageAspects::MEMORY_PLANE_0)
                .then_some(ImageAspect::MemoryPlane0),
            self.intersects(ImageAspects::MEMORY_PLANE_1)
                .then_some(ImageAspect::MemoryPlane1),
            self.intersects(ImageAspects::MEMORY_PLANE_2)
                .then_some(ImageAspect::MemoryPlane2),
        ]
        .into_iter()
        .flatten()
    }
}

impl From<ImageAspect> for ImageAspects {
    #[inline]
    fn from(aspect: ImageAspect) -> Self {
        let mut result = Self::empty();

        match aspect {
            ImageAspect::Color => result |= ImageAspects::COLOR,
            ImageAspect::Depth => result |= ImageAspects::DEPTH,
            ImageAspect::Stencil => result |= ImageAspects::STENCIL,
            ImageAspect::Metadata => result |= ImageAspects::METADATA,
            ImageAspect::Plane0 => result |= ImageAspects::PLANE_0,
            ImageAspect::Plane1 => result |= ImageAspects::PLANE_1,
            ImageAspect::Plane2 => result |= ImageAspects::PLANE_2,
            ImageAspect::MemoryPlane0 => result |= ImageAspects::MEMORY_PLANE_0,
            ImageAspect::MemoryPlane1 => result |= ImageAspects::MEMORY_PLANE_1,
            ImageAspect::MemoryPlane2 => result |= ImageAspects::MEMORY_PLANE_2,
        }

        result
    }
}

impl FromIterator<ImageAspect> for ImageAspects {
    fn from_iter<T: IntoIterator<Item = ImageAspect>>(iter: T) -> Self {
        let mut result = Self::empty();

        for aspect in iter {
            match aspect {
                ImageAspect::Color => result |= ImageAspects::COLOR,
                ImageAspect::Depth => result |= ImageAspects::DEPTH,
                ImageAspect::Stencil => result |= ImageAspects::STENCIL,
                ImageAspect::Metadata => result |= ImageAspects::METADATA,
                ImageAspect::Plane0 => result |= ImageAspects::PLANE_0,
                ImageAspect::Plane1 => result |= ImageAspects::PLANE_1,
                ImageAspect::Plane2 => result |= ImageAspects::PLANE_2,
                ImageAspect::MemoryPlane0 => result |= ImageAspects::MEMORY_PLANE_0,
                ImageAspect::MemoryPlane1 => result |= ImageAspects::MEMORY_PLANE_1,
                ImageAspect::MemoryPlane2 => result |= ImageAspects::MEMORY_PLANE_2,
            }
        }

        result
    }
}
