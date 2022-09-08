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
    color = COLOR,

    // TODO: document
    depth = DEPTH,

    // TODO: document
    stencil = STENCIL,

    // TODO: document
    metadata = METADATA,

    // TODO: document
    plane0 = PLANE_0 {
        api_version: V1_1,
        device_extensions: [khr_sampler_ycbcr_conversion],
    },

    // TODO: document
    plane1 = PLANE_1 {
        api_version: V1_1,
        device_extensions: [khr_sampler_ycbcr_conversion],
    },

    // TODO: document
    plane2 = PLANE_2 {
        api_version: V1_1,
        device_extensions: [khr_sampler_ycbcr_conversion],
    },

    // TODO: document
    memory_plane0 = MEMORY_PLANE_0_EXT {
        device_extensions: [ext_image_drm_format_modifier],
    },

    // TODO: document
    memory_plane1 = MEMORY_PLANE_1_EXT {
        device_extensions: [ext_image_drm_format_modifier],
    },

    // TODO: document
    memory_plane2 = MEMORY_PLANE_2_EXT {
        device_extensions: [ext_image_drm_format_modifier],
    },
}

impl ImageAspects {
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
            _ne: _,
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

impl From<ImageAspect> for ImageAspects {
    fn from(aspect: ImageAspect) -> Self {
        let mut result = Self::empty();

        match aspect {
            ImageAspect::Color => result.color = true,
            ImageAspect::Depth => result.depth = true,
            ImageAspect::Stencil => result.stencil = true,
            ImageAspect::Metadata => result.metadata = true,
            ImageAspect::Plane0 => result.plane0 = true,
            ImageAspect::Plane1 => result.plane1 = true,
            ImageAspect::Plane2 => result.plane2 = true,
            ImageAspect::MemoryPlane0 => result.memory_plane0 = true,
            ImageAspect::MemoryPlane1 => result.memory_plane1 = true,
            ImageAspect::MemoryPlane2 => result.memory_plane2 = true,
        }

        result
    }
}

impl FromIterator<ImageAspect> for ImageAspects {
    fn from_iter<T: IntoIterator<Item = ImageAspect>>(iter: T) -> Self {
        let mut result = Self::empty();

        for aspect in iter {
            match aspect {
                ImageAspect::Color => result.color = true,
                ImageAspect::Depth => result.depth = true,
                ImageAspect::Stencil => result.stencil = true,
                ImageAspect::Metadata => result.metadata = true,
                ImageAspect::Plane0 => result.plane0 = true,
                ImageAspect::Plane1 => result.plane1 = true,
                ImageAspect::Plane2 => result.plane2 = true,
                ImageAspect::MemoryPlane0 => result.memory_plane0 = true,
                ImageAspect::MemoryPlane1 => result.memory_plane1 = true,
                ImageAspect::MemoryPlane2 => result.memory_plane2 = true,
            }
        }

        result
    }
}
