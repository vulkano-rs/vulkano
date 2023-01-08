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
    sys::ImageError,
    traits::{ImageAccess, ImageInner},
    usage::ImageUsage,
    view::{ImageViewAbstract, ImageViewType},
};
use crate::{
    format::Format,
    macros::{vulkan_bitflags, vulkan_bitflags_enum, vulkan_enum},
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

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags that can be set when creating a new image.
    ImageCreateFlags = ImageCreateFlags(u32);

    /* TODO: enable
    /// The image will be backed by sparse memory binding (through queue commands) instead of
    /// regular binding (through [`bind_memory`]).
    ///
    /// The [`sparse_binding`] feature must be enabled on the device.
    ///
    /// [`bind_memory`]: sys::RawImage::bind_memory
    /// [`sparse_binding`]: crate::device::Features::sparse_binding
    SPARSE_BINDING = SPARSE_BINDING,*/

    /* TODO: enable
    /// The image can be used without being fully resident in memory at the time of use.
    ///
    /// This requires the `sparse_binding` flag as well.
    ///
    /// Depending on the image dimensions, either the [`sparse_residency_image2_d`] or the
    /// [`sparse_residency_image3_d`] feature must be enabled on the device.
    /// For a multisampled image, the one of the features [`sparse_residency2_samples`],
    /// [`sparse_residency4_samples`], [`sparse_residency8_samples`] or
    /// [`sparse_residency16_samples`], corresponding to the sample count of the image, must
    /// be enabled on the device.
    ///
    /// [`sparse_binding`]: crate::device::Features::sparse_binding
    /// [`sparse_residency_image2_d`]: crate::device::Features::sparse_residency_image2_d
    /// [`sparse_residency_image2_3`]: crate::device::Features::sparse_residency_image3_d
    /// [`sparse_residency2_samples`]: crate::device::Features::sparse_residency2_samples
    /// [`sparse_residency4_samples`]: crate::device::Features::sparse_residency4_samples
    /// [`sparse_residency8_samples`]: crate::device::Features::sparse_residency8_samples
    /// [`sparse_residency16_samples`]: crate::device::Features::sparse_residency16_samples
    SPARSE_RESIDENCY = SPARSE_RESIDENCY,*/

    /* TODO: enable
    /// The buffer's memory can alias with another image or a different part of the same image.
    ///
    /// This requires the `sparse_binding` flag as well.
    ///
    /// The [`sparse_residency_aliased`] feature must be enabled on the device.
    ///
    /// [`sparse_residency_aliased`]: crate::device::Features::sparse_residency_aliased
    SPARSE_ALIASED = SPARSE_ALIASED,*/

    /// For non-multi-planar formats, whether an image view wrapping the image can have a
    /// different format.
    ///
    /// For multi-planar formats, whether an image view wrapping the image can be created from a
    /// single plane of the image.
    MUTABLE_FORMAT = MUTABLE_FORMAT,

    /// For 2D images, whether an image view of type [`ImageViewType::Cube`] or
    /// [`ImageViewType::CubeArray`] can be created from the image.
    ///
    /// [`ImageViewType::Cube`]: crate::image::view::ImageViewType::Cube
    /// [`ImageViewType::CubeArray`]: crate::image::view::ImageViewType::CubeArray
    CUBE_COMPATIBLE = CUBE_COMPATIBLE,

    /* TODO: enable
    // TODO: document
    ALIAS = ALIAS {
        api_version: V1_1,
        device_extensions: [khr_bind_memory2],
    },*/

    /* TODO: enable
    // TODO: document
    SPLIT_INSTANCE_BIND_REGIONS = SPLIT_INSTANCE_BIND_REGIONS {
        api_version: V1_1,
        device_extensions: [khr_device_group],
    },*/

    /// For 3D images, whether an image view of type [`ImageViewType::Dim2d`] or
    /// [`ImageViewType::Dim2dArray`] can be created from the image.
    ///
    /// On [portability subset] devices, the [`image_view2_d_on3_d_image`] feature must be enabled
    /// on the device.
    ///
    /// [`ImageViewType::Dim2d`]: crate::image::view::ImageViewType::Dim2d
    /// [`ImageViewType::Dim2dArray`]: crate::image::view::ImageViewType::Dim2dArray
    /// [portability subset]: crate::instance#portability-subset-devices-and-the-enumerate_portability-flag
    /// [`image_view2_d_on3_d_image`]: crate::device::Features::image_view2_d_on3_d_image
    ARRAY_2D_COMPATIBLE = TYPE_2D_ARRAY_COMPATIBLE {
        api_version: V1_1,
        device_extensions: [khr_maintenance1],
    },

    /// For images with a compressed format, whether an image view with an uncompressed
    /// format can be created from the image, where each texel in the view will correspond to a
    /// compressed texel block in the image.
    ///
    /// Requires `mutable_format`.
    BLOCK_TEXEL_VIEW_COMPATIBLE = BLOCK_TEXEL_VIEW_COMPATIBLE {
        api_version: V1_1,
        device_extensions: [khr_maintenance2],
    },

    /* TODO: enable
    // TODO: document
    EXTENDED_USAGE = EXTENDED_USAGE {
        api_version: V1_1,
        device_extensions: [khr_maintenance2],
    },*/

    /* TODO: enable
    // TODO: document
    PROTECTED = PROTECTED {
        api_version: V1_1,
    },*/

    /// For images with a multi-planar format, whether each plane will have its memory bound
    /// separately, rather than having a single memory binding for the whole image.
    DISJOINT = DISJOINT {
        api_version: V1_1,
        device_extensions: [khr_sampler_ycbcr_conversion],
    },

    /* TODO: enable
    // TODO: document
    CORNER_SAMPLED = CORNER_SAMPLED_NV {
        device_extensions: [nv_corner_sampled_image],
    },*/

    /* TODO: enable
    // TODO: document
    SAMPLE_LOCATIONS_COMPATIBLE_DEPTH = SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_EXT {
        device_extensions: [ext_sample_locations],
    },*/

    /* TODO: enable
    // TODO: document
    SUBSAMPLED = SUBSAMPLED_EXT {
        device_extensions: [ext_fragment_density_map],
    },*/

    /* TODO: enable
    // TODO: document
    MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED = MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_EXT {
        device_extensions: [ext_multisampled_render_to_single_sampled],
    },*/

    /* TODO: enable
    // TODO: document
    TYPE_2D_VIEW_COMPATIBLE = TYPE_2D_VIEW_COMPATIBLE_EXT {
        device_extensions: [ext_image_2d_view_of_3d],
    },*/

    /* TODO: enable
    // TODO: document
    FRAGMENT_DENSITY_MAP_OFFSET = FRAGMENT_DENSITY_MAP_OFFSET_QCOM {
        device_extensions: [qcom_fragment_density_map_offset],
    },*/
}

vulkan_bitflags_enum! {
    #[non_exhaustive]

    /// A set of [`SampleCount`] values.
    SampleCounts impl {
        /// Returns the maximum sample count in `self`.
        #[inline]
        pub const fn max_count(self) -> SampleCount {
            if self.intersects(SampleCounts::SAMPLE_64) {
                SampleCount::Sample64
            } else if self.intersects(SampleCounts::SAMPLE_32) {
                SampleCount::Sample32
            } else if self.intersects(SampleCounts::SAMPLE_16) {
                SampleCount::Sample16
            } else if self.intersects(SampleCounts::SAMPLE_8) {
                SampleCount::Sample8
            } else if self.intersects(SampleCounts::SAMPLE_4) {
                SampleCount::Sample4
            } else if self.intersects(SampleCounts::SAMPLE_2) {
                SampleCount::Sample2
            } else {
                SampleCount::Sample1
            }
        }
    },

    /// The number of samples per texel of an image.
    SampleCount,

    = SampleCountFlags(u32);

    /// 1 sample per texel.
    SAMPLE_1, Sample1 = TYPE_1,

    /// 2 samples per texel.
    SAMPLE_2, Sample2 = TYPE_2,

    /// 4 samples per texel.
    SAMPLE_4, Sample4 = TYPE_4,

    /// 8 samples per texel.
    SAMPLE_8, Sample8 = TYPE_8,

    /// 16 samples per texel.
    SAMPLE_16, Sample16 = TYPE_16,

    /// 32 samples per texel.
    SAMPLE_32, Sample32 = TYPE_32,

    /// 64 samples per texel.
    SAMPLE_64, Sample64 = TYPE_64,
}

impl From<SampleCount> for u32 {
    #[inline]
    fn from(value: SampleCount) -> Self {
        value as u32
    }
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

vulkan_enum! {
    #[non_exhaustive]

    // TODO: document
    ImageType = ImageType(i32);

    // TODO: document
    Dim1d = TYPE_1D,

    // TODO: document
    Dim2d = TYPE_2D,

    // TODO: document
    Dim3d = TYPE_3D,
}

vulkan_enum! {
    #[non_exhaustive]

    // TODO: document
    ImageTiling = ImageTiling(i32);

    // TODO: document
    Optimal = OPTIMAL,

    // TODO: document
    Linear = LINEAR,

    /* TODO: enable
    // TODO: document
    DrmFormatModifier = DRM_FORMAT_MODIFIER_EXT {
        device_extensions: [ext_image_drm_format_modifier],
    },*/
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
    /// # Examples
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
    /// # Examples
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
    /// # Panics
    ///
    /// - In debug mode, panics if `width`, `height` or `depth` is equal to 0. In release, returns
    ///   an unspecified value.
    #[inline]
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

impl ImageSubresourceLayers {
    /// Returns an `ImageSubresourceLayers` from the given image parameters, covering the first
    /// mip level of the image. All aspects of the image are selected, or `plane0` if the image
    /// is multi-planar.
    #[inline]
    pub fn from_parameters(format: Format, array_layers: u32) -> Self {
        Self {
            aspects: {
                let aspects = format.aspects();

                if aspects.intersects(ImageAspects::PLANE_0) {
                    ImageAspects::PLANE_0
                } else {
                    aspects
                }
            },
            mip_level: 0,
            array_layers: 0..array_layers,
        }
    }
}

impl From<ImageSubresourceLayers> for ash::vk::ImageSubresourceLayers {
    #[inline]
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

impl ImageSubresourceRange {
    /// Returns an `ImageSubresourceRange` from the given image parameters, covering the whole
    /// image. If the image is multi-planar, only the `color` aspect is selected.
    #[inline]
    pub fn from_parameters(format: Format, mip_levels: u32, array_layers: u32) -> Self {
        Self {
            aspects: format.aspects()
                - (ImageAspects::PLANE_0 | ImageAspects::PLANE_1 | ImageAspects::PLANE_2),
            mip_levels: 0..mip_levels,
            array_layers: 0..array_layers,
        }
    }
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

/// Describes the memory layout of an image.
///
/// The address of a texel at `(x, y, z, layer)` is `layer * array_pitch + z * depth_pitch +
/// y * row_pitch + x * size_of_each_texel + offset`. `size_of_each_texel` must be determined
/// depending on the format. The same formula applies for compressed formats, except that the
/// coordinates must be in number of blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubresourceLayout {
    /// The number of bytes from the start of the memory where the subresource begins.
    pub offset: DeviceSize,

    /// The size in bytes in the subresource. It includes any extra memory that is required based on
    /// `row_pitch`.
    pub size: DeviceSize,

    /// The number of bytes between adjacent rows of texels.
    pub row_pitch: DeviceSize,

    /// The number of bytes between adjacent array layers.
    ///
    /// This value is undefined for images with only one array layer.
    pub array_pitch: DeviceSize,

    /// The number of bytes between adjacent depth slices.
    ///
    /// This value is undefined for images that are not three-dimensional.
    pub depth_pitch: DeviceSize,
}

/// The image configuration to query in
/// [`PhysicalDevice::image_format_properties`](crate::device::physical::PhysicalDevice::image_format_properties).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImageFormatInfo {
    /// The `flags` that the image will have.
    ///
    /// The default value is [`ImageCreateFlags::empty()`].
    pub flags: ImageCreateFlags,

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

    /// The `stencil_usage` that the image will have.
    ///
    /// If `stencil_usage` is empty or if `format` does not have both a depth and a stencil aspect,
    /// then it is automatically set to equal `usage`.
    ///
    /// If after this, `stencil_usage` does not equal `usage`,
    /// then the physical device API version must be at least 1.2, or the
    /// [`ext_separate_stencil_usage`](crate::device::DeviceExtensions::ext_separate_stencil_usage)
    /// extension must be supported by the physical device.
    ///
    /// The default value is [`ImageUsage::empty()`].
    pub stencil_usage: ImageUsage,

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

    pub _ne: crate::NonExhaustive,
}

impl Default for ImageFormatInfo {
    #[inline]
    fn default() -> Self {
        Self {
            flags: ImageCreateFlags::empty(),
            format: None,
            image_type: ImageType::Dim2d,
            tiling: ImageTiling::Optimal,
            usage: ImageUsage::empty(),
            stencil_usage: ImageUsage::empty(),
            external_memory_handle_type: None,
            image_view_type: None,
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
    #[inline]
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

/// The image configuration to query in
/// [`PhysicalDevice::sparse_image_format_properties`](crate::device::physical::PhysicalDevice::sparse_image_format_properties).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SparseImageFormatInfo {
    /// The `format` that the image will have.
    ///
    /// The default value is `None`, which must be overridden.
    pub format: Option<Format>,

    /// The dimension type that the image will have.
    ///
    /// The default value is [`ImageType::Dim2d`].
    pub image_type: ImageType,

    /// The `samples` that the image will have.
    ///
    /// The default value is `SampleCount::Sample1`.
    pub samples: SampleCount,

    /// The `usage` that the image will have.
    ///
    /// The default value is [`ImageUsage::empty()`], which must be overridden.
    pub usage: ImageUsage,

    /// The `tiling` that the image will have.
    ///
    /// The default value is [`ImageTiling::Optimal`].
    pub tiling: ImageTiling,

    pub _ne: crate::NonExhaustive,
}

impl Default for SparseImageFormatInfo {
    #[inline]
    fn default() -> Self {
        Self {
            format: None,
            image_type: ImageType::Dim2d,
            samples: SampleCount::Sample1,
            usage: ImageUsage::empty(),
            tiling: ImageTiling::Optimal,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// The properties that are supported by a physical device for sparse images of a certain type.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SparseImageFormatProperties {
    /// The aspects of the image that the properties apply to.
    pub aspects: ImageAspects,

    /// The size of the sparse image block, in texels or compressed texel blocks.
    ///
    /// If `flags.nonstandard_block_size` is set, then these values do not match the standard
    /// sparse block dimensions for the given format.
    pub image_granularity: [u32; 3],

    /// Additional information about the sparse image.
    pub flags: SparseImageFormatFlags,
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying information about a sparse resource.
    SparseImageFormatFlags = SparseImageFormatFlags(u32);

    /// The image uses a single mip tail region for all array layers, instead of one mip tail region
    /// per array layer.
    SINGLE_MIPTAIL = SINGLE_MIPTAIL,

    /// The image's mip tail region begins with the first mip level whose dimensions are not an
    /// integer multiple of the corresponding sparse image block dimensions.
    ALIGNED_MIP_SIZE = ALIGNED_MIP_SIZE,

    /// The image uses non-standard sparse image block dimensions.
    NONSTANDARD_BLOCK_SIZE = NONSTANDARD_BLOCK_SIZE,
}

/// Requirements for binding memory to a sparse image.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SparseImageMemoryRequirements {
    /// The properties of the image format.
    pub format_properties: SparseImageFormatProperties,

    /// The first mip level at which image subresources are included in the mip tail region.
    pub image_mip_tail_first_lod: u32,

    /// The size in bytes of the mip tail region. This value is guaranteed to be a multiple of the
    /// sparse block size in bytes.
    ///
    /// If `format_properties.flags.single_miptail` is set, then this is the size of the whole
    /// mip tail. Otherwise it is the size of the mip tail of a single array layer.
    pub image_mip_tail_size: DeviceSize,

    /// The memory offset that must be used to bind the mip tail region.
    pub image_mip_tail_offset: DeviceSize,

    /// If `format_properties.flags.single_miptail` is not set, specifies the stride between
    /// the mip tail regions of each array layer.
    pub image_mip_tail_stride: Option<DeviceSize>,
}

#[cfg(test)]
mod tests {
    use crate::{
        command_buffer::{
            allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        },
        format::Format,
        image::{ImageAccess, ImageDimensions, ImmutableImage, MipmapsCount},
        memory::allocator::StandardMemoryAllocator,
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
        let (device, queue) = gfx_dev_and_queue!();

        let cb_allocator = StandardCommandBufferAllocator::new(device.clone(), Default::default());
        let mut cbb = AutoCommandBufferBuilder::primary(
            &cb_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let memory_allocator = StandardMemoryAllocator::new_default(device);
        let dimensions = ImageDimensions::Dim2d {
            width: 512,
            height: 512,
            array_layers: 1,
        };
        {
            let mut vec = Vec::new();

            vec.resize(512 * 512, 0u8);

            let image = ImmutableImage::from_iter(
                &memory_allocator,
                vec.into_iter(),
                dimensions,
                MipmapsCount::One,
                Format::R8_UNORM,
                &mut cbb,
            )
            .unwrap();
            assert_eq!(image.mip_levels(), 1);
        }
        {
            let mut vec = Vec::new();

            vec.resize(512 * 512, 0u8);

            let image = ImmutableImage::from_iter(
                &memory_allocator,
                vec.into_iter(),
                dimensions,
                MipmapsCount::Log2,
                Format::R8_UNORM,
                &mut cbb,
            )
            .unwrap();
            assert_eq!(image.mip_levels(), 10);
        }
    }
}
