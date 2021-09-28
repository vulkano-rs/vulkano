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

use crate::check_errors;
use crate::device::Device;
use crate::format::Format;
use crate::image::sys::UnsafeImage;
use crate::image::ImageAccess;
use crate::image::ImageDimensions;
use crate::memory::DeviceMemoryAllocError;
use crate::sampler::Sampler;
use crate::OomError;
use crate::SafeDeref;
use crate::VulkanObject;
use std::error;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem::MaybeUninit;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;

/// A safe image view that checks for validity and keeps its attached image alive.
pub struct ImageView<I>
where
    I: ImageAccess,
{
    inner: UnsafeImageView,
    image: I,

    array_layers: Range<u32>,
    component_mapping: ComponentMapping,
    format: Format,
    ty: ImageViewType,
}

impl<I> ImageView<I>
where
    I: ImageAccess,
{
    /// Creates a default `ImageView`. Equivalent to `ImageView::start(image).build()`.
    #[inline]
    pub fn new(image: I) -> Result<Arc<ImageView<I>>, ImageViewCreationError> {
        Self::start(image).build()
    }

    /// Begins building an `ImageView`.
    pub fn start(image: I) -> ImageViewBuilder<I> {
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
        let mipmap_levels = 0..image.mipmap_levels();
        let array_layers = 0..image.dimensions().array_layers();

        ImageViewBuilder {
            array_layers,
            component_mapping: ComponentMapping::default(),
            format: image.format(),
            mipmap_levels,
            ty,

            image,
        }
    }

    /// Returns the wrapped image that this image view was created from.
    pub fn image(&self) -> &I {
        &self.image
    }
}

#[derive(Debug)]
pub struct ImageViewBuilder<I> {
    array_layers: Range<u32>,
    component_mapping: ComponentMapping,
    format: Format,
    mipmap_levels: Range<u32>,
    ty: ImageViewType,

    image: I,
}

impl<I> ImageViewBuilder<I>
where
    I: ImageAccess,
{
    /// Sets the image view type.
    ///
    /// By default, this is determined from the image, based on its dimensions and number of layers.
    /// The value of `ty` must be compatible with the dimensions of the image and the selected
    /// array layers.
    #[inline]
    pub fn with_type(mut self, ty: ImageViewType) -> Self {
        self.ty = ty;
        self
    }

    /// Sets how to map components of each pixel.
    ///
    /// By default, this is the identity mapping, with every component mapped directly.
    #[inline]
    pub fn with_component_mapping(mut self, component_mapping: ComponentMapping) -> Self {
        self.component_mapping = component_mapping;
        self
    }

    /// Sets the format of the image view.
    ///
    /// By default, this is the format of the image. Using a different format requires enabling the
    /// `mutable_format` flag on the image.
    #[inline]
    pub fn with_format(mut self, format: Format) -> Self {
        self.format = format;
        self
    }

    /// Sets the range of mipmap levels that the view should cover.
    ///
    /// By default, this is the full range of mipmaps present in the image.
    #[inline]
    pub fn with_mipmap_levels(mut self, mipmap_levels: Range<u32>) -> Self {
        self.mipmap_levels = mipmap_levels;
        self
    }

    /// Sets the range of array layers that the view should cover.
    ///
    /// By default, this is the full range of array layers present in the image.
    #[inline]
    pub fn with_array_layers(mut self, array_layers: Range<u32>) -> Self {
        self.array_layers = array_layers;
        self
    }

    /// Builds the `ImageView`.
    pub fn build(self) -> Result<Arc<ImageView<I>>, ImageViewCreationError> {
        let dimensions = self.image.dimensions();
        let image_inner = self.image.inner().image;
        let image_flags = image_inner.flags();
        let image_format = image_inner.format();
        let image_usage = image_inner.usage();

        if self.mipmap_levels.end <= self.mipmap_levels.start
            || self.mipmap_levels.end > image_inner.mipmap_levels()
        {
            return Err(ImageViewCreationError::MipMapLevelsOutOfRange);
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
            self.mipmap_levels.end - self.mipmap_levels.start,
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

        if image_format.requires_sampler_ycbcr_conversion() {
            unimplemented!()
        }

        if image_flags.block_texel_view_compatible {
            if self.format.compatibility() != image_format.compatibility()
                || self.format.size() != image_format.size()
            {
                return Err(ImageViewCreationError::IncompatibleFormat);
            }

            if self.array_layers.end - self.array_layers.start != 1 {
                return Err(ImageViewCreationError::ArrayLayersOutOfRange);
            }

            if self.mipmap_levels.end - self.mipmap_levels.start != 1 {
                return Err(ImageViewCreationError::MipMapLevelsOutOfRange);
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

        if self.format != image_format {
            if !(image_flags.mutable_format && image_format.planes().is_empty()) {
                return Err(ImageViewCreationError::IncompatibleFormat);
            } else if self.format.compatibility() != image_format.compatibility() {
                if !image_flags.block_texel_view_compatible {
                    return Err(ImageViewCreationError::IncompatibleFormat);
                } else if self.format.size() != image_format.size() {
                    return Err(ImageViewCreationError::IncompatibleFormat);
                }
            }
        }

        let inner = unsafe {
            UnsafeImageView::new(
                image_inner,
                self.ty,
                self.component_mapping,
                self.mipmap_levels,
                self.array_layers.clone(),
            )?
        };

        Ok(Arc::new(ImageView {
            inner,
            image: self.image,

            array_layers: self.array_layers,
            component_mapping: self.component_mapping,
            format: self.format,
            ty: self.ty,
        }))
    }
}

/// Error that can happen when creating an image view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageViewCreationError {
    /// Allocating memory failed.
    AllocError(DeviceMemoryAllocError),
    /// The specified range of array layers was out of range for the image.
    ArrayLayersOutOfRange,
    /// The specified range of mipmap levels was out of range for the image.
    MipMapLevelsOutOfRange,
    /// The requested format was not compatible with the image.
    IncompatibleFormat,
    /// The requested [`ImageViewType`] was not compatible with the image, or with the specified ranges of array layers and mipmap levels.
    IncompatibleType,
    /// The image was not created with
    /// [one of the required usages](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#valid-imageview-imageusage)
    /// for image views.
    InvalidImageUsage,
}

impl error::Error for ImageViewCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            ImageViewCreationError::AllocError(ref err) => Some(err),
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
                ImageViewCreationError::AllocError(err) => "allocating memory failed",
                ImageViewCreationError::ArrayLayersOutOfRange => "array layers are out of range",
                ImageViewCreationError::MipMapLevelsOutOfRange => "mipmap levels are out of range",
                ImageViewCreationError::IncompatibleFormat => "format is not compatible with image",
                ImageViewCreationError::IncompatibleType =>
                    "image view type is not compatible with image, array layers or mipmap levels",
                ImageViewCreationError::InvalidImageUsage =>
                    "the usage of the image is not compatible with image views",
            }
        )
    }
}

impl From<OomError> for ImageViewCreationError {
    #[inline]
    fn from(err: OomError) -> ImageViewCreationError {
        ImageViewCreationError::AllocError(DeviceMemoryAllocError::OomError(err))
    }
}

/// A low-level wrapper around a `vkImageView`.
pub struct UnsafeImageView {
    view: ash::vk::ImageView,
    device: Arc<Device>,
}

impl UnsafeImageView {
    /// Creates a new view from an image.
    ///
    /// # Safety
    /// - The returned `UnsafeImageView` must not outlive `image`.
    /// - `image` must have a usage that is compatible with image views.
    /// - `ty` must be compatible with the dimensions and flags of the image.
    /// - `mipmap_levels` must not be empty, must be within the range of levels of the image, and be compatible with the requested `ty`.
    /// - `array_layers` must not be empty, must be within the range of layers of the image, and be compatible with the requested `ty`.
    ///
    /// # Panics
    /// Panics if the image is a YcbCr image, since the Vulkano API is not yet flexible enough to
    /// specify the aspect of image.
    pub unsafe fn new(
        image: &UnsafeImage,
        ty: ImageViewType,
        component_mapping: ComponentMapping,
        mipmap_levels: Range<u32>,
        array_layers: Range<u32>,
    ) -> Result<UnsafeImageView, OomError> {
        let fns = image.device().fns();

        debug_assert!(mipmap_levels.end > mipmap_levels.start);
        debug_assert!(mipmap_levels.end <= image.mipmap_levels());
        debug_assert!(array_layers.end > array_layers.start);
        debug_assert!(array_layers.end <= image.dimensions().array_layers());

        if image.format().requires_sampler_ycbcr_conversion() {
            unimplemented!();
        }

        // TODO: Let user choose
        let aspects = image.format().aspects();

        let view = {
            let infos = ash::vk::ImageViewCreateInfo {
                flags: ash::vk::ImageViewCreateFlags::empty(),
                image: image.internal_object(),
                view_type: ty.into(),
                format: image.format().into(),
                components: component_mapping.into(),
                subresource_range: ash::vk::ImageSubresourceRange {
                    aspect_mask: aspects.into(),
                    base_mip_level: mipmap_levels.start,
                    level_count: mipmap_levels.end - mipmap_levels.start,
                    base_array_layer: array_layers.start,
                    layer_count: array_layers.end - array_layers.start,
                },
                ..Default::default()
            };

            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_image_view(
                image.device().internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(UnsafeImageView {
            view,
            device: image.device().clone(),
        })
    }
}

unsafe impl VulkanObject for UnsafeImageView {
    type Object = ash::vk::ImageView;

    #[inline]
    fn internal_object(&self) -> ash::vk::ImageView {
        self.view
    }
}

impl fmt::Debug for UnsafeImageView {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan image view {:?}>", self.view)
    }
}

impl Drop for UnsafeImageView {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            fns.v1_0
                .destroy_image_view(self.device.internal_object(), self.view, ptr::null());
        }
    }
}

impl PartialEq for UnsafeImageView {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.view == other.view && self.device == other.device
    }
}

impl Eq for UnsafeImageView {}

impl Hash for UnsafeImageView {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.view.hash(state);
        self.device.hash(state);
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

/// Specifies how the components of an image must be mapped.
///
/// When creating an image view, it is possible to ask the implementation to modify the value
/// returned when accessing a given component from within a shader.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ComponentMapping {
    /// First component.
    pub r: ComponentSwizzle,
    /// Second component.
    pub g: ComponentSwizzle,
    /// Third component.
    pub b: ComponentSwizzle,
    /// Fourth component.
    pub a: ComponentSwizzle,
}

impl ComponentMapping {
    /// Returns `true` if the component mapping is identity swizzled,
    /// meaning that all the members are `Identity`.
    ///
    /// Certain operations require views that are identity swizzled, and will return an error
    /// otherwise. For example, attaching a view to a framebuffer is only possible if the view is
    /// identity swizzled.
    #[inline]
    pub fn is_identity(&self) -> bool {
        self.r == ComponentSwizzle::Identity
            && self.g == ComponentSwizzle::Identity
            && self.b == ComponentSwizzle::Identity
            && self.a == ComponentSwizzle::Identity
    }
}

impl From<ComponentMapping> for ash::vk::ComponentMapping {
    #[inline]
    fn from(value: ComponentMapping) -> Self {
        Self {
            r: value.r.into(),
            g: value.g.into(),
            b: value.b.into(),
            a: value.a.into(),
        }
    }
}

/// Describes the value that an individual component must return when being accessed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ComponentSwizzle {
    /// Returns the value that this component should normally have.
    ///
    /// This is the `Default` value.
    Identity = ash::vk::ComponentSwizzle::IDENTITY.as_raw(),
    /// Always return zero.
    Zero = ash::vk::ComponentSwizzle::ZERO.as_raw(),
    /// Always return one.
    One = ash::vk::ComponentSwizzle::ONE.as_raw(),
    /// Returns the value of the first component.
    Red = ash::vk::ComponentSwizzle::R.as_raw(),
    /// Returns the value of the second component.
    Green = ash::vk::ComponentSwizzle::G.as_raw(),
    /// Returns the value of the third component.
    Blue = ash::vk::ComponentSwizzle::B.as_raw(),
    /// Returns the value of the fourth component.
    Alpha = ash::vk::ComponentSwizzle::A.as_raw(),
}

impl From<ComponentSwizzle> for ash::vk::ComponentSwizzle {
    #[inline]
    fn from(val: ComponentSwizzle) -> Self {
        Self::from_raw(val as i32)
    }
}

impl Default for ComponentSwizzle {
    #[inline]
    fn default() -> ComponentSwizzle {
        ComponentSwizzle::Identity
    }
}

/// Trait for types that represent the GPU can access an image view.
pub unsafe trait ImageViewAbstract: Send + Sync {
    /// Returns the wrapped image that this image view was created from.
    fn image(&self) -> &dyn ImageAccess;

    /// Returns the inner unsafe image view object used by this image view.
    fn inner(&self) -> &UnsafeImageView;

    /// Returns the range of array layers of the wrapped image that this view exposes.
    fn array_layers(&self) -> Range<u32>;

    /// Returns the format of this view. This can be different from the parent's format.
    fn format(&self) -> Format;

    /// Returns the component mapping of this view.
    fn component_mapping(&self) -> ComponentMapping;

    /// Returns the [`ImageViewType`] of this image view.
    fn ty(&self) -> ImageViewType;

    /// Returns true if the given sampler can be used with this image view.
    ///
    /// This method should check whether the sampler's configuration can be used with the format
    /// of the view.
    // TODO: return a Result and propagate it when binding to a descriptor set
    fn can_be_sampled(&self, _sampler: &Sampler) -> bool {
        true /* FIXME */
    }
}

unsafe impl<I> ImageViewAbstract for ImageView<I>
where
    I: ImageAccess,
{
    #[inline]
    fn image(&self) -> &dyn ImageAccess {
        &self.image
    }

    #[inline]
    fn inner(&self) -> &UnsafeImageView {
        &self.inner
    }

    #[inline]
    fn array_layers(&self) -> Range<u32> {
        self.array_layers.clone()
    }

    #[inline]
    fn format(&self) -> Format {
        // TODO: remove this default impl
        self.format
    }

    #[inline]
    fn component_mapping(&self) -> ComponentMapping {
        self.component_mapping
    }

    #[inline]
    fn ty(&self) -> ImageViewType {
        self.ty
    }
}

unsafe impl<T> ImageViewAbstract for T
where
    T: SafeDeref + Send + Sync,
    T::Target: ImageViewAbstract,
{
    #[inline]
    fn image(&self) -> &dyn ImageAccess {
        (**self).image()
    }

    #[inline]
    fn inner(&self) -> &UnsafeImageView {
        (**self).inner()
    }

    #[inline]
    fn array_layers(&self) -> Range<u32> {
        (**self).array_layers()
    }

    #[inline]
    fn format(&self) -> Format {
        (**self).format()
    }

    #[inline]
    fn component_mapping(&self) -> ComponentMapping {
        (**self).component_mapping()
    }

    #[inline]
    fn ty(&self) -> ImageViewType {
        (**self).ty()
    }

    #[inline]
    fn can_be_sampled(&self, sampler: &Sampler) -> bool {
        (**self).can_be_sampled(sampler)
    }
}

impl PartialEq for dyn ImageViewAbstract {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner()
    }
}

impl Eq for dyn ImageViewAbstract {}

impl Hash for dyn ImageViewAbstract {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
    }
}
