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
use crate::format::FormatTy;
use crate::image::sys::UnsafeImage;
use crate::image::ImageAccess;
use crate::image::ImageDimensions;
use crate::memory::DeviceMemoryAllocError;
use crate::sampler::Sampler;
use crate::vk;
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
    image: I,
    inner: UnsafeImageView,
    format: Format,

    ty: ImageViewType,
    component_mapping: ComponentMapping,
    array_layers: Range<u32>,
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
            image,
            ty,
            component_mapping: ComponentMapping::default(),
            mipmap_levels,
            array_layers,
        }
    }

    /// Returns the wrapped image that this image view was created from.
    pub fn image(&self) -> &I {
        &self.image
    }
}

#[derive(Debug)]
pub struct ImageViewBuilder<I> {
    image: I,
    ty: ImageViewType,
    component_mapping: ComponentMapping,
    mipmap_levels: Range<u32>,
    array_layers: Range<u32>,
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
        let format = self.image.format();
        let image_inner = self.image.inner().image;
        let usage = image_inner.usage();
        let flags = image_inner.flags();

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

        if !(usage.sampled
            || usage.storage
            || usage.color_attachment
            || usage.depth_stencil_attachment
            || usage.input_attachment
            || usage.transient_attachment)
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
            (ImageViewType::Cubemap, ImageDimensions::Dim2d { .. }, 6, _)
                if flags.cube_compatible =>
            {
                ()
            }
            (ImageViewType::CubemapArray, ImageDimensions::Dim2d { .. }, n, _)
                if flags.cube_compatible && n % 6 == 0 =>
            {
                ()
            }
            (ImageViewType::Dim3d, ImageDimensions::Dim3d { .. }, 1, _) => (),
            (ImageViewType::Dim2d, ImageDimensions::Dim3d { .. }, 1, 1)
                if flags.array_2d_compatible =>
            {
                ()
            }
            (ImageViewType::Dim2dArray, ImageDimensions::Dim3d { .. }, _, 1)
                if flags.array_2d_compatible =>
            {
                ()
            }
            _ => return Err(ImageViewCreationError::IncompatibleType),
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
            image: self.image,
            inner,
            format,

            ty: self.ty,
            component_mapping: self.component_mapping,
            array_layers: self.array_layers,
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
    view: vk::ImageView,
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
        let vk = image.device().pointers();

        debug_assert!(mipmap_levels.end > mipmap_levels.start);
        debug_assert!(mipmap_levels.end <= image.mipmap_levels());
        debug_assert!(array_layers.end > array_layers.start);
        debug_assert!(array_layers.end <= image.dimensions().array_layers());

        if image.format().ty() == FormatTy::Ycbcr {
            unimplemented!();
        }

        // TODO: Let user choose
        let aspects = image.format().aspects();

        let view = {
            let infos = vk::ImageViewCreateInfo {
                sType: vk::STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0, // reserved
                image: image.internal_object(),
                viewType: ty.into(),
                format: image.format() as u32,
                components: component_mapping.into(),
                subresourceRange: vk::ImageSubresourceRange {
                    aspectMask: aspects.into(),
                    baseMipLevel: mipmap_levels.start,
                    levelCount: mipmap_levels.end - mipmap_levels.start,
                    baseArrayLayer: array_layers.start,
                    layerCount: array_layers.end - array_layers.start,
                },
            };

            let mut output = MaybeUninit::uninit();
            check_errors(vk.CreateImageView(
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
    type Object = vk::ImageView;

    const TYPE: vk::ObjectType = vk::OBJECT_TYPE_IMAGE_VIEW;

    #[inline]
    fn internal_object(&self) -> vk::ImageView {
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
            let vk = self.device.pointers();
            vk.DestroyImageView(self.device.internal_object(), self.view, ptr::null());
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
pub enum ImageViewType {
    Dim1d,
    Dim1dArray,
    Dim2d,
    Dim2dArray,
    Dim3d,
    Cubemap,
    CubemapArray,
}

impl From<ImageViewType> for vk::ImageViewType {
    fn from(image_view_type: ImageViewType) -> Self {
        match image_view_type {
            ImageViewType::Dim1d => vk::IMAGE_VIEW_TYPE_1D,
            ImageViewType::Dim1dArray => vk::IMAGE_VIEW_TYPE_1D_ARRAY,
            ImageViewType::Dim2d => vk::IMAGE_VIEW_TYPE_2D,
            ImageViewType::Dim2dArray => vk::IMAGE_VIEW_TYPE_2D_ARRAY,
            ImageViewType::Dim3d => vk::IMAGE_VIEW_TYPE_3D,
            ImageViewType::Cubemap => vk::IMAGE_VIEW_TYPE_CUBE,
            ImageViewType::CubemapArray => vk::IMAGE_VIEW_TYPE_CUBE_ARRAY,
        }
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

impl From<ComponentMapping> for vk::ComponentMapping {
    #[inline]
    fn from(value: ComponentMapping) -> Self {
        Self {
            r: value.r as u32,
            g: value.g as u32,
            b: value.b as u32,
            a: value.a as u32,
        }
    }
}

/// Describes the value that an individual component must return when being accessed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ComponentSwizzle {
    /// Returns the value that this component should normally have.
    ///
    /// This is the `Default` value.
    Identity = vk::COMPONENT_SWIZZLE_IDENTITY,
    /// Always return zero.
    Zero = vk::COMPONENT_SWIZZLE_ZERO,
    /// Always return one.
    One = vk::COMPONENT_SWIZZLE_ONE,
    /// Returns the value of the first component.
    Red = vk::COMPONENT_SWIZZLE_R,
    /// Returns the value of the second component.
    Green = vk::COMPONENT_SWIZZLE_G,
    /// Returns the value of the third component.
    Blue = vk::COMPONENT_SWIZZLE_B,
    /// Returns the value of the fourth component.
    Alpha = vk::COMPONENT_SWIZZLE_A,
}

impl Default for ComponentSwizzle {
    #[inline]
    fn default() -> ComponentSwizzle {
        ComponentSwizzle::Identity
    }
}

/// Trait for types that represent the GPU can access an image view.
pub unsafe trait ImageViewAbstract {
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
    T: SafeDeref,
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

impl PartialEq for dyn ImageViewAbstract + Send + Sync {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner()
    }
}

impl Eq for dyn ImageViewAbstract + Send + Sync {}

impl Hash for dyn ImageViewAbstract + Send + Sync {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
    }
}
