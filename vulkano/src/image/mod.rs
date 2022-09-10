// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Image storage (1D, 2D, 3D, arrays, etc.) and image views.
//!
//! An *image* is a region of memory whose purpose is to store multi-dimensional data. Its
//! most common use is to store a 2D array of color pixels (in other words an *image* in
//! everyday language), but it can also be used to store arbitrary data.
//!
//! The advantage of using an image compared to a buffer is that the memory layout is optimized
//! for locality. When reading a specific pixel of an image, reading the nearby pixels is really
//! fast. Most implementations have hardware dedicated to reading from images if you access them
//! through a sampler.
//!
//! # Properties of an image
//!
//! # Images and image views
//!
//! There is a distinction between *images* and *image views*. As its name suggests, an image
//! view describes how the GPU must interpret the image.
//!
//! Transfer and memory operations operate on images themselves, while reading/writing an image
//! operates on image views. You can create multiple image views from the same image.
//!
//! # High-level wrappers
//!
//! In the vulkano library, an image is any object that implements the [`ImageAccess`] trait. You
//! can create a view by wrapping them in an [`ImageView`](crate::image::view::ImageView).
//!
//! Since the `ImageAccess` trait is low-level, you are encouraged to not implement it yourself but
//! instead use one of the provided implementations that are specialized depending on the way you
//! are going to use the image:
//!
//! - An `AttachmentImage` can be used when you want to draw to an image.
//! - An `ImmutableImage` stores data which never need be changed after the initial upload,
//!   like a texture.
//!
//! # Low-level information
//!
//! To be written.
//!

pub use self::{
    aspect::{ImageAspect, ImageAspects},
    attachment::AttachmentImage,
    immutable::ImmutableImage,
    layout::{ImageDescriptorLayouts, ImageLayout},
    storage::StorageImage,
    swapchain::SwapchainImage,
    sys::ImageCreationError,
    traits::{ImageAccess, ImageInner},
    usage::ImageUsage,
    view::{ImageViewAbstract, ImageViewType},
};
use crate::{
    format::Format,
    macros::{vulkan_bitflags, vulkan_enum},
    memory::{ExternalMemoryHandleType, ExternalMemoryProperties},
    DeviceSize,
};
use std::{cmp, ops::Range};

mod aspect;
pub mod attachment; // TODO: make private
pub mod immutable; // TODO: make private
mod layout;
mod storage;
pub mod swapchain; // TODO: make private
pub mod sys;
pub mod traits;
mod usage;
pub mod view;

vulkan_enum! {
    // TODO: document
    #[non_exhaustive]
    SampleCount = SampleCountFlags(u32);

    // TODO: document
    Sample1 = TYPE_1,

    // TODO: document
    Sample2 = TYPE_2,

    // TODO: document
    Sample4 = TYPE_4,

    // TODO: document
    Sample8 = TYPE_8,

    // TODO: document
    Sample16 = TYPE_16,

    // TODO: document
    Sample32 = TYPE_32,

    // TODO: document
    Sample64 = TYPE_64,
}

impl TryFrom<u32> for SampleCount {
    type Error = ();

    #[inline]
    fn try_from(val: u32) -> Result<Self, Self::Error> {
        match val {
            1 => Ok(Self::Sample1),
            2 => Ok(Self::Sample2),
            4 => Ok(Self::Sample4),
            8 => Ok(Self::Sample8),
            16 => Ok(Self::Sample16),
            32 => Ok(Self::Sample32),
            64 => Ok(Self::Sample64),
            _ => Err(()),
        }
    }
}

vulkan_bitflags! {
    /// Specifies a set of [`SampleCount`] values.
    #[non_exhaustive]
    SampleCounts = SampleCountFlags(u32);

    /// 1 sample per pixel.
    sample1 = TYPE_1,

    /// 2 samples per pixel.
    sample2 = TYPE_2,

    /// 4 samples per pixel.
    sample4 = TYPE_4,

    /// 8 samples per pixel.
    sample8 = TYPE_8,

    /// 16 samples per pixel.
    sample16 = TYPE_16,

    /// 32 samples per pixel.
    sample32 = TYPE_32,

    /// 64 samples per pixel.
    sample64 = TYPE_64,
}

impl SampleCounts {
    /// Returns true if `self` has the `sample_count` value set.
    #[inline]
    pub const fn contains_count(&self, sample_count: SampleCount) -> bool {
        match sample_count {
            SampleCount::Sample1 => self.sample1,
            SampleCount::Sample2 => self.sample2,
            SampleCount::Sample4 => self.sample4,
            SampleCount::Sample8 => self.sample8,
            SampleCount::Sample16 => self.sample16,
            SampleCount::Sample32 => self.sample32,
            SampleCount::Sample64 => self.sample64,
        }
    }

    /// Returns the maximum sample count supported by `self`.
    #[inline]
    pub const fn max_count(&self) -> SampleCount {
        match self {
            Self { sample64: true, .. } => SampleCount::Sample64,
            Self { sample32: true, .. } => SampleCount::Sample32,
            Self { sample16: true, .. } => SampleCount::Sample16,
            Self { sample8: true, .. } => SampleCount::Sample8,
            Self { sample4: true, .. } => SampleCount::Sample4,
            Self { sample2: true, .. } => SampleCount::Sample2,
            _ => SampleCount::Sample1,
        }
    }
}

/// Specifies how many mipmaps must be allocated.
///
/// Note that at least one mipmap must be allocated, to store the main level of the image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MipmapsCount {
    /// Allocates the number of mipmaps required to store all the mipmaps of the image where each
    /// mipmap is half the dimensions of the previous level. Guaranteed to be always supported.
    ///
    /// Note that this is not necessarily the maximum number of mipmaps, as the Vulkan
    /// implementation may report that it supports a greater value.
    Log2,

    /// Allocate one mipmap (ie. just the main level). Always supported.
    One,

    /// Allocate the given number of mipmaps. May result in an error if the value is out of range
    /// of what the implementation supports.
    Specific(u32),
}

impl From<u32> for MipmapsCount {
    #[inline]
    fn from(num: u32) -> MipmapsCount {
        MipmapsCount::Specific(num)
    }
}

vulkan_bitflags! {
    /// Flags that can be set when creating a new image.
    #[non_exhaustive]
    ImageCreateFlags = ImageCreateFlags(u32);

    /// The image will be backed by sparsely bound memory.
    ///
    /// Requires the [`sparse_binding`](crate::device::Features::sparse_binding) feature to be
    /// enabled.
    sparse_binding = SPARSE_BINDING,

    /// The image is allowed to be only partially resident in memory, not all parts of the image
    /// must be backed by memory.
    ///
    /// Requires the `sparse_binding` flag, and depending on the image dimensions, either the
    /// [`sparse_residency_image2_d`](crate::device::Features::sparse_residency_image2_d) or the
    /// [`sparse_residency_image3_d`](crate::device::Features::sparse_residency_image3_d) feature to
    /// be enabled. For a multisampled image, this also requires the appropriate sparse residency
    /// feature for the number of samples to be enabled.
    sparse_residency = SPARSE_RESIDENCY,

    /// The image can be backed by memory that is shared (aliased) with other images.
    ///
    /// Requires the `sparse_binding` flag and the
    /// [`sparse_residency_aliased`](crate::device::Features::sparse_residency_aliased) feature to
    /// be enabled.
    sparse_aliased = SPARSE_ALIASED,

    /// For non-multi-planar formats, an image view wrapping this image can have a different format.
    ///
    /// For multi-planar formats, an image view wrapping this image can be created from a single
    /// plane of the image.
    mutable_format = MUTABLE_FORMAT,

    /// For 2D images, allows creation of an image view of type `Cube` or `CubeArray`.
    cube_compatible = CUBE_COMPATIBLE,

    /// For 3D images, allows creation of an image view of type `Dim2d` or `Dim2dArray`.
    array_2d_compatible = TYPE_2D_ARRAY_COMPATIBLE {
        api_version: V1_1,
        device_extensions: [khr_maintenance1],
    },

    /// For images with a compressed format, allows creation of an image view with an uncompressed
    /// format, where each texel in the view will correspond to a compressed texel block in the
    /// image.
    ///
    /// Requires `mutable_format`.
    block_texel_view_compatible = BLOCK_TEXEL_VIEW_COMPATIBLE {
        api_version: V1_1,
        device_extensions: [khr_maintenance1],
    },
}

vulkan_enum! {
    // TODO: document
    #[non_exhaustive]
    ImageType = ImageType(i32);

    // TODO: document
    Dim1d = TYPE_1D,

    // TODO: document
    Dim2d = TYPE_2D,

    // TODO: document
    Dim3d = TYPE_3D,
}

vulkan_enum! {
    // TODO: document
    #[non_exhaustive]
    ImageTiling = ImageTiling(i32);

    // TODO: document
    Optimal = OPTIMAL,

    // TODO: document
    Linear = LINEAR,

    /*
    // TODO: document
    DrmFormatModifier = DRM_FORMAT_MODIFIER_EXT {
        device_extensions: [ext_image_drm_format_modifier],
    },
     */
}

/// The dimensions of an image.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ImageDimensions {
    Dim1d {
        width: u32,
        array_layers: u32,
    },
    Dim2d {
        width: u32,
        height: u32,
        array_layers: u32,
    },
    Dim3d {
        width: u32,
        height: u32,
        depth: u32,
    },
}

impl ImageDimensions {
    #[inline]
    pub fn width(&self) -> u32 {
        match *self {
            ImageDimensions::Dim1d { width, .. } => width,
            ImageDimensions::Dim2d { width, .. } => width,
            ImageDimensions::Dim3d { width, .. } => width,
        }
    }

    #[inline]
    pub fn height(&self) -> u32 {
        match *self {
            ImageDimensions::Dim1d { .. } => 1,
            ImageDimensions::Dim2d { height, .. } => height,
            ImageDimensions::Dim3d { height, .. } => height,
        }
    }

    #[inline]
    pub fn width_height(&self) -> [u32; 2] {
        [self.width(), self.height()]
    }

    #[inline]
    pub fn depth(&self) -> u32 {
        match *self {
            ImageDimensions::Dim1d { .. } => 1,
            ImageDimensions::Dim2d { .. } => 1,
            ImageDimensions::Dim3d { depth, .. } => depth,
        }
    }

    #[inline]
    pub fn width_height_depth(&self) -> [u32; 3] {
        [self.width(), self.height(), self.depth()]
    }

    #[inline]
    pub fn array_layers(&self) -> u32 {
        match *self {
            ImageDimensions::Dim1d { array_layers, .. } => array_layers,
            ImageDimensions::Dim2d { array_layers, .. } => array_layers,
            ImageDimensions::Dim3d { .. } => 1,
        }
    }

    /// Returns the total number of texels for an image of these dimensions.
    #[inline]
    pub fn num_texels(&self) -> u32 {
        self.width() * self.height() * self.depth() * self.array_layers()
    }

    #[inline]
    pub fn image_type(&self) -> ImageType {
        match *self {
            ImageDimensions::Dim1d { .. } => ImageType::Dim1d,
            ImageDimensions::Dim2d { .. } => ImageType::Dim2d,
            ImageDimensions::Dim3d { .. } => ImageType::Dim3d,
        }
    }

    /// Returns the maximum number of mipmap levels for these image dimensions.
    ///
    /// The returned value is always at least 1.
    ///
    /// # Example
    ///
    /// ```
    /// use vulkano::image::ImageDimensions;
    ///
    /// let dims = ImageDimensions::Dim2d {
    ///     width: 32,
    ///     height: 50,
    ///     array_layers: 1,
    /// };
    ///
    /// assert_eq!(dims.max_mip_levels(), 6);
    /// ```
    ///
    #[inline]
    pub fn max_mip_levels(&self) -> u32 {
        // This calculates `log2(max(width, height, depth)) + 1` using fast integer operations.
        let max = match *self {
            ImageDimensions::Dim1d { width, .. } => width,
            ImageDimensions::Dim2d { width, height, .. } => width | height,
            ImageDimensions::Dim3d {
                width,
                height,
                depth,
            } => width | height | depth,
        };
        32 - max.leading_zeros()
    }

    /// Returns the dimensions of the `level`th mipmap level. If `level` is 0, then the dimensions
    /// are left unchanged.
    ///
    /// Returns `None` if `level` is superior or equal to `max_mip_levels()`.
    ///
    /// # Example
    ///
    /// ```
    /// use vulkano::image::ImageDimensions;
    ///
    /// let dims = ImageDimensions::Dim2d {
    ///     width: 963,
    ///     height: 256,
    ///     array_layers: 1,
    /// };
    ///
    /// assert_eq!(dims.mip_level_dimensions(0), Some(dims));
    /// assert_eq!(dims.mip_level_dimensions(1), Some(ImageDimensions::Dim2d {
    ///     width: 481,
    ///     height: 128,
    ///     array_layers: 1,
    /// }));
    /// assert_eq!(dims.mip_level_dimensions(6), Some(ImageDimensions::Dim2d {
    ///     width: 15,
    ///     height: 4,
    ///     array_layers: 1,
    /// }));
    /// assert_eq!(dims.mip_level_dimensions(9), Some(ImageDimensions::Dim2d {
    ///     width: 1,
    ///     height: 1,
    ///     array_layers: 1,
    /// }));
    /// assert_eq!(dims.mip_level_dimensions(11), None);
    /// ```
    ///
    /// # Panic
    ///
    /// In debug mode, Panics if `width`, `height` or `depth` is equal to 0. In release, returns
    /// an unspecified value.
    ///
    pub fn mip_level_dimensions(&self, level: u32) -> Option<ImageDimensions> {
        if level == 0 {
            return Some(*self);
        }

        if level >= self.max_mip_levels() {
            return None;
        }

        Some(match *self {
            ImageDimensions::Dim1d {
                width,
                array_layers,
            } => {
                debug_assert_ne!(width, 0);
                ImageDimensions::Dim1d {
                    array_layers,
                    width: cmp::max(1, width >> level),
                }
            }

            ImageDimensions::Dim2d {
                width,
                height,
                array_layers,
            } => {
                debug_assert_ne!(width, 0);
                debug_assert_ne!(height, 0);
                ImageDimensions::Dim2d {
                    width: cmp::max(1, width >> level),
                    height: cmp::max(1, height >> level),
                    array_layers,
                }
            }

            ImageDimensions::Dim3d {
                width,
                height,
                depth,
            } => {
                debug_assert_ne!(width, 0);
                debug_assert_ne!(height, 0);
                ImageDimensions::Dim3d {
                    width: cmp::max(1, width >> level),
                    height: cmp::max(1, height >> level),
                    depth: cmp::max(1, depth >> level),
                }
            }
        })
    }
}

/// One or more subresources of an image, spanning a single mip level, that should be accessed by a
/// command.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageSubresourceLayers {
    /// Selects the aspects that will be included.
    ///
    /// The value must not be empty, and must not include any of the `memory_plane` aspects.
    /// The `color` aspect cannot be selected together any of with the `plane` aspects.
    pub aspects: ImageAspects,

    /// Selects mip level that will be included.
    pub mip_level: u32,

    /// Selects the range of array layers that will be included.
    ///
    /// The range must not be empty.
    pub array_layers: Range<u32>,
}

impl From<ImageSubresourceLayers> for ash::vk::ImageSubresourceLayers {
    fn from(val: ImageSubresourceLayers) -> Self {
        Self {
            aspect_mask: val.aspects.into(),
            mip_level: val.mip_level,
            base_array_layer: val.array_layers.start,
            layer_count: val.array_layers.end - val.array_layers.start,
        }
    }
}

/// One or more subresources of an image that should be accessed by a command.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageSubresourceRange {
    /// Selects the aspects that will be included.
    ///
    /// The value must not be empty, and must not include any of the `memory_plane` aspects.
    /// The `color` aspect cannot be selected together any of with the `plane` aspects.
    pub aspects: ImageAspects,

    /// Selects the range of the mip levels that will be included.
    ///
    /// The range must not be empty.
    pub mip_levels: Range<u32>,

    /// Selects the range of array layers that will be included.
    ///
    /// The range must not be empty.
    pub array_layers: Range<u32>,
}

impl From<ImageSubresourceRange> for ash::vk::ImageSubresourceRange {
    #[inline]
    fn from(val: ImageSubresourceRange) -> Self {
        Self {
            aspect_mask: val.aspects.into(),
            base_mip_level: val.mip_levels.start,
            level_count: val.mip_levels.end - val.mip_levels.start,
            base_array_layer: val.array_layers.start,
            layer_count: val.array_layers.end - val.array_layers.start,
        }
    }
}

impl From<ImageSubresourceLayers> for ImageSubresourceRange {
    #[inline]
    fn from(val: ImageSubresourceLayers) -> Self {
        Self {
            aspects: val.aspects,
            mip_levels: val.mip_level..val.mip_level + 1,
            array_layers: val.array_layers,
        }
    }
}

/// The image configuration to query in
/// [`PhysicalDevice::image_format_properties`](crate::device::physical::PhysicalDevice::image_format_properties).
#[derive(Clone, Debug)]
pub struct ImageFormatInfo {
    /// The `format` that the image will have.
    ///
    /// The default value is `None`, which must be overridden.
    pub format: Option<Format>,

    /// The dimension type that the image will have.
    ///
    /// The default value is [`ImageType::Dim2d`].
    pub image_type: ImageType,

    /// The `tiling` that the image will have.
    ///
    /// The default value is [`ImageTiling::Optimal`].
    pub tiling: ImageTiling,

    /// The `usage` that the image will have.
    ///
    /// The default value is [`ImageUsage::empty()`], which must be overridden.
    pub usage: ImageUsage,

    /// An external memory handle type that will be imported to or exported from the image.
    ///
    /// This is needed to retrieve the
    /// [`external_memory_properties`](ImageFormatProperties::external_memory_properties) value,
    /// and the physical device API version must be at least 1.1 or the
    /// [`khr_external_memory_capabilities`](crate::instance::InstanceExtensions::khr_external_memory_capabilities)
    /// extension must be enabled on the instance.
    ///
    /// The default value is `None`.
    pub external_memory_handle_type: Option<ExternalMemoryHandleType>,

    /// The image view type that will be created from the image.
    ///
    /// This is needed to retrieve the
    /// [`filter_cubic`](ImageFormatProperties::filter_cubic) and
    /// [`filter_cubic_minmax`](ImageFormatProperties::filter_cubic_minmax) values, and the
    /// [`ext_filter_cubic`](crate::device::DeviceExtensions::ext_filter_cubic) extension must be
    /// supported on the physical device.
    ///
    /// The default value is `None`.
    pub image_view_type: Option<ImageViewType>,

    /// The `mutable_format` that the image will have.
    ///
    /// The default value is `false`.
    pub mutable_format: bool,

    /// The `cube_compatible` that the image will have.
    ///
    /// The default value is `false`.
    pub cube_compatible: bool,

    /// The `array_2d_compatible` that the image will have.
    ///
    /// The default value is `false`.
    pub array_2d_compatible: bool,

    /// The `block_texel_view_compatible` that the image will have.
    ///
    /// The default value is `false`.
    pub block_texel_view_compatible: bool,

    pub _ne: crate::NonExhaustive,
}

impl Default for ImageFormatInfo {
    #[inline]
    fn default() -> Self {
        Self {
            format: None,
            image_type: ImageType::Dim2d,
            tiling: ImageTiling::Optimal,
            usage: ImageUsage::empty(),
            external_memory_handle_type: None,
            image_view_type: None,
            mutable_format: false,
            cube_compatible: false,
            array_2d_compatible: false,
            block_texel_view_compatible: false,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// The properties that are supported by a physical device for images of a certain type.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ImageFormatProperties {
    /// The maximum dimensions.
    pub max_extent: [u32; 3],

    /// The maximum number of mipmap levels.
    pub max_mip_levels: u32,

    /// The maximum number of array layers.
    pub max_array_layers: u32,

    /// The supported sample counts.
    pub sample_counts: SampleCounts,

    /// The maximum total size of an image, in bytes. This is guaranteed to be at least
    /// 0x80000000.
    pub max_resource_size: DeviceSize,

    /// The properties for external memory.
    /// This will be [`ExternalMemoryProperties::default()`] if `external_handle_type` was `None`.
    pub external_memory_properties: ExternalMemoryProperties,

    /// When querying with an image view type, whether such image views support sampling with
    /// a [`Cubic`](crate::sampler::Filter::Cubic) `mag_filter` or `min_filter`.
    pub filter_cubic: bool,

    /// When querying with an image view type, whether such image views support sampling with
    /// a [`Cubic`](crate::sampler::Filter::Cubic) `mag_filter` or `min_filter`, and with a
    /// [`Min`](crate::sampler::SamplerReductionMode::Min) or
    /// [`Max`](crate::sampler::SamplerReductionMode::Max) `reduction_mode`.
    pub filter_cubic_minmax: bool,
}

impl From<ash::vk::ImageFormatProperties> for ImageFormatProperties {
    fn from(props: ash::vk::ImageFormatProperties) -> Self {
        Self {
            max_extent: [
                props.max_extent.width,
                props.max_extent.height,
                props.max_extent.depth,
            ],
            max_mip_levels: props.max_mip_levels,
            max_array_layers: props.max_array_layers,
            sample_counts: props.sample_counts.into(),
            max_resource_size: props.max_resource_size,
            external_memory_properties: Default::default(),
            filter_cubic: false,
            filter_cubic_minmax: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        format::Format,
        image::{ImageAccess, ImageDimensions, ImmutableImage, MipmapsCount},
    };

    #[test]
    fn max_mip_levels() {
        let dims = ImageDimensions::Dim2d {
            width: 2,
            height: 1,
            array_layers: 1,
        };
        assert_eq!(dims.max_mip_levels(), 2);

        let dims = ImageDimensions::Dim2d {
            width: 2,
            height: 3,
            array_layers: 1,
        };
        assert_eq!(dims.max_mip_levels(), 2);

        let dims = ImageDimensions::Dim2d {
            width: 512,
            height: 512,
            array_layers: 1,
        };
        assert_eq!(dims.max_mip_levels(), 10);
    }

    #[test]
    fn mip_level_dimensions() {
        let dims = ImageDimensions::Dim2d {
            width: 283,
            height: 175,
            array_layers: 1,
        };
        assert_eq!(dims.mip_level_dimensions(0), Some(dims));
        assert_eq!(
            dims.mip_level_dimensions(1),
            Some(ImageDimensions::Dim2d {
                width: 141,
                height: 87,
                array_layers: 1,
            })
        );
        assert_eq!(
            dims.mip_level_dimensions(2),
            Some(ImageDimensions::Dim2d {
                width: 70,
                height: 43,
                array_layers: 1,
            })
        );
        assert_eq!(
            dims.mip_level_dimensions(3),
            Some(ImageDimensions::Dim2d {
                width: 35,
                height: 21,
                array_layers: 1,
            })
        );

        assert_eq!(
            dims.mip_level_dimensions(4),
            Some(ImageDimensions::Dim2d {
                width: 17,
                height: 10,
                array_layers: 1,
            })
        );
        assert_eq!(
            dims.mip_level_dimensions(5),
            Some(ImageDimensions::Dim2d {
                width: 8,
                height: 5,
                array_layers: 1,
            })
        );
        assert_eq!(
            dims.mip_level_dimensions(6),
            Some(ImageDimensions::Dim2d {
                width: 4,
                height: 2,
                array_layers: 1,
            })
        );
        assert_eq!(
            dims.mip_level_dimensions(7),
            Some(ImageDimensions::Dim2d {
                width: 2,
                height: 1,
                array_layers: 1,
            })
        );
        assert_eq!(
            dims.mip_level_dimensions(8),
            Some(ImageDimensions::Dim2d {
                width: 1,
                height: 1,
                array_layers: 1,
            })
        );
        assert_eq!(dims.mip_level_dimensions(9), None);
    }

    #[test]
    fn mipmap_working_immutable_image() {
        let (_device, queue) = gfx_dev_and_queue!();

        let dimensions = ImageDimensions::Dim2d {
            width: 512,
            height: 512,
            array_layers: 1,
        };
        {
            let mut vec = Vec::new();

            vec.resize(512 * 512, 0u8);

            let (image, _) = ImmutableImage::from_iter(
                vec.into_iter(),
                dimensions,
                MipmapsCount::One,
                Format::R8_UNORM,
                queue.clone(),
            )
            .unwrap();
            assert_eq!(image.mip_levels(), 1);
        }
        {
            let mut vec = Vec::new();

            vec.resize(512 * 512, 0u8);

            let (image, _) = ImmutableImage::from_iter(
                vec.into_iter(),
                dimensions,
                MipmapsCount::Log2,
                Format::R8_UNORM,
                queue,
            )
            .unwrap();
            assert_eq!(image.mip_levels(), 10);
        }
    }
}
