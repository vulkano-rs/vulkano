// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Image views.
//!
//! This module contains types related to image views. An image view wraps around
//! an image and describes how the GPU should interpret the data. It is needed when an image is
//! to be used in a shader descriptor or as a framebuffer attachment.

use crate::device::physical::FormatFeatures;
use crate::device::{Device, DeviceOwned};
use crate::format::Format;
use crate::image::{ImageAccess, ImageDimensions, ImageTiling};
use crate::sampler::ComponentMapping;
use crate::OomError;
use crate::VulkanObject;
use crate::{check_errors, Error};
use std::error;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem::MaybeUninit;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;

/// A wrapper around an image that makes it available to shaders or framebuffers.
pub struct ImageView<I>
where
    I: ImageAccess,
{
    handle: ash::vk::ImageView,
    image: Arc<I>,

    array_layers: Range<u32>,
    component_mapping: ComponentMapping,
    format: Format,
    format_features: FormatFeatures,
    ty: ImageViewType,
}

impl<I> ImageView<I>
where
    I: ImageAccess,
{
    /// Creates a default `ImageView`. Equivalent to `ImageView::start(image).build()`.
    #[inline]
    pub fn new(image: Arc<I>) -> Result<Arc<ImageView<I>>, ImageViewCreationError> {
        Self::start(image).build()
    }

    /// Begins building an `ImageView`.
    pub fn start(image: Arc<I>) -> ImageViewBuilder<I> {
        let ty = match image.dimensions() {
            ImageDimensions::Dim1d {
                array_layers: 1, ..
            } => ImageViewType::Dim1d,
            ImageDimensions::Dim1d { .. } => ImageViewType::Dim1dArray,
            ImageDimensions::Dim2d {
                array_layers: 1, ..
            } => ImageViewType::Dim2d,
            ImageDimensions::Dim2d { .. } => ImageViewType::Dim2dArray,
            ImageDimensions::Dim3d { .. } => ImageViewType::Dim3d,
        };
        let mip_levels = 0..image.mip_levels();
        let array_layers = 0..image.dimensions().array_layers();

        ImageViewBuilder {
            array_layers,
            component_mapping: ComponentMapping::default(),
            format: image.format(),
            mip_levels,
            ty,

            image,
        }
    }

    /// Returns the wrapped image that this image view was created from.
    pub fn image(&self) -> &Arc<I> {
        &self.image
    }
}

unsafe impl<I> VulkanObject for ImageView<I>
where
    I: ImageAccess,
{
    type Object = ash::vk::ImageView;

    #[inline]
    fn internal_object(&self) -> ash::vk::ImageView {
        self.handle
    }
}

unsafe impl<I> DeviceOwned for ImageView<I>
where
    I: ImageAccess,
{
    fn device(&self) -> &Arc<Device> {
        self.image.inner().image.device()
    }
}

impl<I> fmt::Debug for ImageView<I>
where
    I: ImageAccess,
{
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan image view {:?}>", self.handle)
    }
}

impl<I> Drop for ImageView<I>
where
    I: ImageAccess,
{
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let device = self.device();
            let fns = device.fns();
            fns.v1_0
                .destroy_image_view(device.internal_object(), self.handle, ptr::null());
        }
    }
}

impl<I> PartialEq for ImageView<I>
where
    I: ImageAccess,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle && self.device() == other.device()
    }
}

impl<I> Eq for ImageView<I> where I: ImageAccess {}

impl<I> Hash for ImageView<I>
where
    I: ImageAccess,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
        self.device().hash(state);
    }
}

#[derive(Debug)]
pub struct ImageViewBuilder<I> {
    array_layers: Range<u32>,
    component_mapping: ComponentMapping,
    format: Format,
    mip_levels: Range<u32>,
    ty: ImageViewType,

    image: Arc<I>,
}

impl<I> ImageViewBuilder<I>
where
    I: ImageAccess,
{
    /// Builds the `ImageView`.
    pub fn build(self) -> Result<Arc<ImageView<I>>, ImageViewCreationError> {
        let dimensions = self.image.dimensions();
        let image_inner = self.image.inner().image;
        let image_flags = image_inner.flags();
        let image_format = image_inner.format();
        let image_usage = image_inner.usage();

        // TODO: Let user choose
        let aspects = image_format.aspects();

        if self.mip_levels.end <= self.mip_levels.start
            || self.mip_levels.end > image_inner.mip_levels()
        {
            return Err(ImageViewCreationError::MipLevelsOutOfRange);
        }

        if self.array_layers.end <= self.array_layers.start
            || self.array_layers.end > dimensions.array_layers()
        {
            return Err(ImageViewCreationError::ArrayLayersOutOfRange);
        }

        if !(image_usage.sampled
            || image_usage.storage
            || image_usage.color_attachment
            || image_usage.depth_stencil_attachment
            || image_usage.input_attachment
            || image_usage.transient_attachment)
        {
            return Err(ImageViewCreationError::InvalidImageUsage);
        }

        // Check for compatibility with the image
        match (
            self.ty,
            self.image.dimensions(),
            self.array_layers.end - self.array_layers.start,
            self.mip_levels.end - self.mip_levels.start,
        ) {
            (ImageViewType::Dim1d, ImageDimensions::Dim1d { .. }, 1, _) => (),
            (ImageViewType::Dim1dArray, ImageDimensions::Dim1d { .. }, _, _) => (),
            (ImageViewType::Dim2d, ImageDimensions::Dim2d { .. }, 1, _) => (),
            (ImageViewType::Dim2dArray, ImageDimensions::Dim2d { .. }, _, _) => (),
            (ImageViewType::Cube, ImageDimensions::Dim2d { .. }, 6, _)
                if image_flags.cube_compatible =>
            {
                ()
            }
            (ImageViewType::CubeArray, ImageDimensions::Dim2d { .. }, n, _)
                if image_flags.cube_compatible && n % 6 == 0 =>
            {
                ()
            }
            (ImageViewType::Dim3d, ImageDimensions::Dim3d { .. }, 1, _) => (),
            (ImageViewType::Dim2d, ImageDimensions::Dim3d { .. }, 1, 1)
                if image_flags.array_2d_compatible =>
            {
                ()
            }
            (ImageViewType::Dim2dArray, ImageDimensions::Dim3d { .. }, _, 1)
                if image_flags.array_2d_compatible =>
            {
                ()
            }
            _ => return Err(ImageViewCreationError::IncompatibleType),
        }

        if image_format.ycbcr_chroma_sampling().is_some() {
            unimplemented!()
        }

        if image_flags.block_texel_view_compatible {
            if self.format.compatibility() != image_format.compatibility()
                || self.format.block_size() != image_format.block_size()
            {
                return Err(ImageViewCreationError::IncompatibleFormat);
            }

            if self.array_layers.end - self.array_layers.start != 1 {
                return Err(ImageViewCreationError::ArrayLayersOutOfRange);
            }

            if self.mip_levels.end - self.mip_levels.start != 1 {
                return Err(ImageViewCreationError::MipLevelsOutOfRange);
            }

            if self.format.compression().is_none() && self.ty == ImageViewType::Dim3d {
                return Err(ImageViewCreationError::IncompatibleType);
            }
        } else if image_flags.mutable_format {
            if image_format.planes().is_empty() {
                if self.format != image_format {
                    return Err(ImageViewCreationError::IncompatibleFormat);
                }
            } else {
                // TODO: VUID-VkImageViewCreateInfo-image-01586
                // If image was created with the VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT flag, if the
                // format of the image is a multi-planar format, and if subresourceRange.aspectMask
                // is one of VK_IMAGE_ASPECT_PLANE_0_BIT, VK_IMAGE_ASPECT_PLANE_1_BIT, or
                // VK_IMAGE_ASPECT_PLANE_2_BIT, then format must be compatible with the VkFormat for
                // the plane of the image format indicated by subresourceRange.aspectMask, as
                // defined in Compatible formats of planes of multi-planar formats

                // TODO: VUID-VkImageViewCreateInfo-image-01762
                // If image was not created with the VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT flag, or if
                // the format of the image is a multi-planar format and if
                // subresourceRange.aspectMask is VK_IMAGE_ASPECT_COLOR_BIT, format must be
                // identical to the format used to create image
            }
        } else if self.format != image_format {
            return Err(ImageViewCreationError::IncompatibleFormat);
        }

        let format_features = if self.format != image_format {
            if !(image_flags.mutable_format && image_format.planes().is_empty()) {
                return Err(ImageViewCreationError::IncompatibleFormat);
            } else if self.format.compatibility() != image_format.compatibility() {
                if !image_flags.block_texel_view_compatible {
                    return Err(ImageViewCreationError::IncompatibleFormat);
                } else if self.format.block_size() != image_format.block_size() {
                    return Err(ImageViewCreationError::IncompatibleFormat);
                }
            }

            let format_properties = image_inner
                .device()
                .physical_device()
                .format_properties(self.format);

            match image_inner.tiling() {
                ImageTiling::Optimal => format_properties.optimal_tiling_features,
                ImageTiling::Linear => format_properties.linear_tiling_features,
            }
        } else {
            *image_inner.format_features()
        };

        let create_info = ash::vk::ImageViewCreateInfo {
            flags: ash::vk::ImageViewCreateFlags::empty(),
            image: image_inner.internal_object(),
            view_type: self.ty.into(),
            format: image_format.into(),
            components: self.component_mapping.into(),
            subresource_range: ash::vk::ImageSubresourceRange {
                aspect_mask: aspects.into(),
                base_mip_level: self.mip_levels.start,
                level_count: self.mip_levels.end - self.mip_levels.start,
                base_array_layer: self.array_layers.start,
                layer_count: self.array_layers.end - self.array_layers.start,
            },
            ..Default::default()
        };

        let handle = unsafe {
            let fns = image_inner.device().fns();
            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_image_view(
                image_inner.device().internal_object(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(Arc::new(ImageView {
            handle,
            image: self.image,

            array_layers: self.array_layers,
            component_mapping: self.component_mapping,
            format: self.format,
            format_features,
            ty: self.ty,
        }))
    }

    /// Sets the image view type.
    ///
    /// The view type must be compatible with the dimensions of the image and the selected array
    /// layers.
    ///
    /// The default value is determined from the image, based on its dimensions and number of
    /// layers.
    #[inline]
    pub fn ty(mut self, ty: ImageViewType) -> Self {
        self.ty = ty;
        self
    }

    /// Sets the format of the image view.
    ///
    /// If this is set to a format that is different from the image, the image must be created with
    /// the `mutable_format` flag.
    ///
    /// The default value is the format of the image.
    #[inline]
    pub fn format(mut self, format: Format) -> Self {
        self.format = format;
        self
    }

    /// Sets how to map components of each pixel.
    ///
    /// The default value is [`ComponentMapping::identity()`].
    #[inline]
    pub fn component_mapping(mut self, component_mapping: ComponentMapping) -> Self {
        self.component_mapping = component_mapping;
        self
    }

    /// Sets the range of mipmap levels that the view should cover.
    ///
    /// The default value is the full range of mipmaps present in the image.
    #[inline]
    pub fn mip_levels(mut self, mip_levels: Range<u32>) -> Self {
        self.mip_levels = mip_levels;
        self
    }

    /// Sets the range of array layers that the view should cover.
    ///
    /// The default value is the full range of array layers present in the image.
    #[inline]
    pub fn array_layers(mut self, array_layers: Range<u32>) -> Self {
        self.array_layers = array_layers;
        self
    }
}

/// Error that can happen when creating an image view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageViewCreationError {
    /// Allocating memory failed.
    OomError(OomError),

    /// The specified range of array layers was out of range for the image.
    ArrayLayersOutOfRange,

    /// The format requires a sampler YCbCr conversion, but none was provided.
    FormatRequiresSamplerYcbcrConversion { format: Format },

    /// The specified range of mipmap levels was out of range for the image.
    MipLevelsOutOfRange,

    /// The requested format was not compatible with the image.
    IncompatibleFormat,

    /// The requested [`ImageViewType`] was not compatible with the image, or with the specified ranges of array layers and mipmap levels.
    IncompatibleType,

    /// The image was not created with
    /// [one of the required usages](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#valid-imageview-imageusage)
    /// for image views.
    InvalidImageUsage,

    /// Sampler YCbCr conversion was enabled, but `component_mapping` was not the identity mapping.
    SamplerYcbcrConversionComponentMappingNotIdentity { component_mapping: ComponentMapping },
}

impl error::Error for ImageViewCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            ImageViewCreationError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for ImageViewCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                Self::OomError(err) => "allocating memory failed",
                Self::ArrayLayersOutOfRange => "array layers are out of range",
                Self::FormatRequiresSamplerYcbcrConversion { .. } => "the format requires a sampler YCbCr conversion, but none was provided",
                Self::MipLevelsOutOfRange => "mipmap levels are out of range",
                Self::IncompatibleFormat => "format is not compatible with image",
                Self::IncompatibleType =>
                    "image view type is not compatible with image, array layers or mipmap levels",
                Self::InvalidImageUsage =>
                    "the usage of the image is not compatible with image views",
                Self::SamplerYcbcrConversionComponentMappingNotIdentity { .. } => "sampler YCbCr conversion was enabled, but `component_mapping` was not the identity mapping",
            }
        )
    }
}

impl From<OomError> for ImageViewCreationError {
    #[inline]
    fn from(err: OomError) -> ImageViewCreationError {
        ImageViewCreationError::OomError(err)
    }
}

impl From<Error> for ImageViewCreationError {
    #[inline]
    fn from(err: Error) -> ImageViewCreationError {
        match err {
            err @ Error::OutOfHostMemory => OomError::from(err).into(),
            err @ Error::OutOfDeviceMemory => OomError::from(err).into(),
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

/// The geometry type of an image view.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ImageViewType {
    Dim1d = ash::vk::ImageViewType::TYPE_1D.as_raw(),
    Dim1dArray = ash::vk::ImageViewType::TYPE_1D_ARRAY.as_raw(),
    Dim2d = ash::vk::ImageViewType::TYPE_2D.as_raw(),
    Dim2dArray = ash::vk::ImageViewType::TYPE_2D_ARRAY.as_raw(),
    Dim3d = ash::vk::ImageViewType::TYPE_3D.as_raw(),
    Cube = ash::vk::ImageViewType::CUBE.as_raw(),
    CubeArray = ash::vk::ImageViewType::CUBE_ARRAY.as_raw(),
}

impl ImageViewType {
    #[inline]
    pub fn is_arrayed(&self) -> bool {
        match self {
            Self::Dim1d | Self::Dim2d | Self::Dim3d | Self::Cube => false,
            Self::Dim1dArray | Self::Dim2dArray | Self::CubeArray => true,
        }
    }
}

impl From<ImageViewType> for ash::vk::ImageViewType {
    fn from(val: ImageViewType) -> Self {
        Self::from_raw(val as i32)
    }
}

/// Trait for types that represent the GPU can access an image view.
pub unsafe trait ImageViewAbstract:
    VulkanObject<Object = ash::vk::ImageView> + DeviceOwned + Send + Sync
{
    /// Returns the wrapped image that this image view was created from.
    fn image(&self) -> Arc<dyn ImageAccess>;

    /// Returns the range of array layers of the wrapped image that this view exposes.
    fn array_layers(&self) -> Range<u32>;

    /// Returns the component mapping of this view.
    fn component_mapping(&self) -> ComponentMapping;

    /// Returns the format of this view. This can be different from the parent's format.
    fn format(&self) -> Format;

    /// Returns the features supported by the image view's format.
    fn format_features(&self) -> &FormatFeatures;

    /// Returns the [`ImageViewType`] of this image view.
    fn ty(&self) -> ImageViewType;
}

unsafe impl<I> ImageViewAbstract for ImageView<I>
where
    I: ImageAccess + 'static,
{
    #[inline]
    fn image(&self) -> Arc<dyn ImageAccess> {
        self.image.clone() as Arc<_>
    }

    #[inline]
    fn array_layers(&self) -> Range<u32> {
        self.array_layers.clone()
    }

    #[inline]
    fn component_mapping(&self) -> ComponentMapping {
        self.component_mapping
    }

    #[inline]
    fn format(&self) -> Format {
        self.format
    }

    #[inline]
    fn format_features(&self) -> &FormatFeatures {
        &self.format_features
    }

    #[inline]
    fn ty(&self) -> ImageViewType {
        self.ty
    }
}

impl PartialEq for dyn ImageViewAbstract {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.internal_object() == other.internal_object() && self.device() == other.device()
    }
}

impl Eq for dyn ImageViewAbstract {}

impl Hash for dyn ImageViewAbstract {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.internal_object().hash(state);
        self.device().hash(state);
    }
}
