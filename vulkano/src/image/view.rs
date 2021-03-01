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

use std::error;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem::MaybeUninit;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;

use device::Device;
use format::Format;
use format::FormatTy;
use image::sys::UnsafeImage;
use image::ImageAccess;
use image::ImageDimensions;
use memory::DeviceMemoryAllocError;
use sampler::Sampler;

use check_errors;
use vk;
use OomError;
use SafeDeref;
use VulkanObject;

/// A safe image view that checks for validity and keeps its attached image alive.
pub struct ImageView<I>
where
    I: ImageAccess,
{
    image: I,
    inner: UnsafeImageView,

    array_layers: Range<u32>,
    format: Format,
    identity_swizzle: bool,
    ty: ImageViewType,
}

impl<I> ImageView<I>
where
    I: ImageAccess,
{
    /// Creates a new image view spanning all mipmap levels and array layers in the image.
    ///
    /// The view type is automatically determined from the image, based on its dimensions and
    /// number of layers.
    #[inline]
    pub fn new(image: I) -> Result<Arc<ImageView<I>>, ImageViewCreationError> {
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
        Self::with_type(image, ty)
    }

    /// Crates a new image view with a custom type.
    pub fn with_type(
        image: I,
        ty: ImageViewType,
    ) -> Result<Arc<ImageView<I>>, ImageViewCreationError> {
        let mipmap_levels = 0..image.mipmap_levels();
        let array_layers = 0..image.dimensions().array_layers();
        Self::with_type_ranges(image, ty, mipmap_levels, array_layers)
    }

    /// Creates a new image view with a custom type and ranges of mipmap levels and array layers.
    pub fn with_type_ranges(
        image: I,
        ty: ImageViewType,
        mipmap_levels: Range<u32>,
        array_layers: Range<u32>,
    ) -> Result<Arc<ImageView<I>>, ImageViewCreationError> {
        let dimensions = image.dimensions();
        let format = image.format();
        let image_inner = image.inner().image;
        let usage = image_inner.usage();
        let flags = image_inner.flags();

        if mipmap_levels.end <= mipmap_levels.start
            || mipmap_levels.end > image_inner.mipmap_levels()
        {
            return Err(ImageViewCreationError::MipMapLevelsOutOfRange);
        }

        if array_layers.end <= array_layers.start || array_layers.end > dimensions.array_layers() {
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
            ty,
            image.dimensions(),
            array_layers.end - array_layers.start,
            mipmap_levels.end - mipmap_levels.start,
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

        let inner =
            unsafe { UnsafeImageView::new(image_inner, ty, mipmap_levels, array_layers.clone())? };

        Ok(Arc::new(ImageView {
            image,
            inner,
            array_layers,
            format,
            identity_swizzle: true, // FIXME:
            ty,
        }))
    }

    /// Returns the wrapped image that this image view was created from.
    fn image(&self) -> &I {
        &self.image
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
    fn cause(&self) -> Option<&dyn error::Error> {
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
                ImageViewCreationError::ArrayLayersOutOfRange => "array layers are out of ramge",
                ImageViewCreationError::MipMapLevelsOutOfRange => "mipmap levels are out of ramge",
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
        mipmap_levels: Range<u32>,
        array_layers: Range<u32>,
    ) -> Result<UnsafeImageView, OomError> {
        let vk = image.device().pointers();

        debug_assert!(mipmap_levels.end > mipmap_levels.start);
        debug_assert!(mipmap_levels.end <= image.mipmap_levels());
        debug_assert!(array_layers.end > array_layers.start);
        debug_assert!(array_layers.end <= image.dimensions().array_layers());

        let aspect_mask = match image.format().ty() {
            FormatTy::Float | FormatTy::Uint | FormatTy::Sint | FormatTy::Compressed => {
                vk::IMAGE_ASPECT_COLOR_BIT
            }
            FormatTy::Depth => vk::IMAGE_ASPECT_DEPTH_BIT,
            FormatTy::Stencil => vk::IMAGE_ASPECT_STENCIL_BIT,
            FormatTy::DepthStencil => vk::IMAGE_ASPECT_DEPTH_BIT | vk::IMAGE_ASPECT_STENCIL_BIT,
            // Not yet supported --> would require changes to ImmutableImage API :-)
            FormatTy::Ycbcr => unimplemented!(),
        };

        let view = {
            let infos = vk::ImageViewCreateInfo {
                sType: vk::STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0, // reserved
                image: image.internal_object(),
                viewType: ty.into(),
                format: image.format() as u32,
                components: vk::ComponentMapping {
                    r: 0,
                    g: 0,
                    b: 0,
                    a: 0,
                }, // FIXME:
                subresourceRange: vk::ImageSubresourceRange {
                    aspectMask: aspect_mask,
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

    /// Returns true if the view doesn't use components swizzling.
    ///
    /// Must be true when the view is used as a framebuffer attachment or TODO: I don't remember
    /// the other thing.
    fn identity_swizzle(&self) -> bool;

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
    fn identity_swizzle(&self) -> bool {
        self.identity_swizzle
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
    fn identity_swizzle(&self) -> bool {
        (**self).identity_swizzle()
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
