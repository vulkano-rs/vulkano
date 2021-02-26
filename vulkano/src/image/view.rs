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
//! This module contains wrappers around the Vulkan image view types.

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
use image::ImageLayout;
use sampler::Sampler;

use check_errors;
use vk;
use OomError;
use SafeDeref;
use VulkanObject;

pub struct UnsafeImageView {
    view: vk::ImageView,
    device: Arc<Device>,
    usage: vk::ImageUsageFlagBits,
    identity_swizzle: bool,
    format: Format,
}

impl UnsafeImageView {
    /// See the docs of new().
    pub unsafe fn raw(
        image: &UnsafeImage,
        ty: ImageViewType,
        mipmap_levels: Range<u32>,
        array_layers: Range<u32>,
    ) -> Result<UnsafeImageView, OomError> {
        let vk = image.device().pointers();

        assert!(mipmap_levels.end > mipmap_levels.start);
        assert!(mipmap_levels.end <= image.mipmap_levels());
        assert!(array_layers.end > array_layers.start);
        assert!(array_layers.end <= image.dimensions().array_layers());
        assert!(
            (
                vk::IMAGE_USAGE_SAMPLED_BIT
                    | vk::IMAGE_USAGE_STORAGE_BIT
                    | vk::IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                    | vk::IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                    | vk::IMAGE_USAGE_INPUT_ATTACHMENT_BIT
                    | vk::IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT
                // TODO | vk::IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR
                // TODO | vk::IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT
            ) & image.usage().to_usage_bits()
                != 0
        );

        let aspect_mask = match image.format().ty() {
            FormatTy::Float | FormatTy::Uint | FormatTy::Sint | FormatTy::Compressed => {
                vk::IMAGE_ASPECT_COLOR_BIT
            }
            FormatTy::Depth => vk::IMAGE_ASPECT_DEPTH_BIT,
            FormatTy::Stencil => vk::IMAGE_ASPECT_STENCIL_BIT,
            FormatTy::DepthStencil => vk::IMAGE_ASPECT_DEPTH_BIT | vk::IMAGE_ASPECT_STENCIL_BIT,
            // Not yet supported --> would require changes to ImmutableImage API :-)
            FormatTy::Ycbcr => panic!(),
        };

        let view_type = match (
            image.dimensions(),
            image.create_flags().cube_compatible,
            ty,
            array_layers.end - array_layers.start,
        ) {
            (ImageDimensions::Dim1d { .. }, _, ImageViewType::Dim1d, 1) => vk::IMAGE_VIEW_TYPE_1D,
            (ImageDimensions::Dim1d { .. }, _, ImageViewType::Dim1dArray, _) => {
                vk::IMAGE_VIEW_TYPE_1D_ARRAY
            }
            (ImageDimensions::Dim2d { .. }, _, ImageViewType::Dim2d, 1) => vk::IMAGE_VIEW_TYPE_2D,
            (ImageDimensions::Dim2d { .. }, _, ImageViewType::Dim2dArray, _) => {
                vk::IMAGE_VIEW_TYPE_2D_ARRAY
            }
            (ImageDimensions::Dim2d { .. }, true, ImageViewType::Cubemap, 6) => {
                vk::IMAGE_VIEW_TYPE_CUBE
            }
            (ImageDimensions::Dim2d { .. }, true, ImageViewType::CubemapArray, n) => {
                assert_eq!(n % 6, 0);
                vk::IMAGE_VIEW_TYPE_CUBE_ARRAY
            }
            (ImageDimensions::Dim3d { .. }, _, ImageViewType::Dim3d, _) => vk::IMAGE_VIEW_TYPE_3D,
            _ => panic!(),
        };

        let view = {
            let infos = vk::ImageViewCreateInfo {
                sType: vk::STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0, // reserved
                image: image.internal_object(),
                viewType: view_type,
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
            usage: image.usage().to_usage_bits(),
            identity_swizzle: true, // FIXME:
            format: image.format(),
        })
    }

    /// Creates a new view from an image.
    ///
    /// Note that you must create the view with identity swizzling if you want to use this view
    /// as a framebuffer attachment.
    ///
    /// # Panic
    ///
    /// - Panics if `mipmap_levels` or `array_layers` is out of range of the image.
    /// - Panics if the view types doesn't match the dimensions of the image (for example a 2D
    ///   view from a 3D image).
    /// - Panics if trying to create a cubemap with a number of array layers different from 6.
    /// - Panics if trying to create a cubemap array with a number of array layers not a multiple
    ///   of 6.
    /// - Panics if the device or host ran out of memory.
    /// - Panics if the image is a YcbCr image, since the Vulkano API is not yet flexible enough to
    ///   specify the aspect of image.
    #[inline]
    pub unsafe fn new(
        image: &UnsafeImage,
        ty: ImageViewType,
        mipmap_levels: Range<u32>,
        array_layers: Range<u32>,
    ) -> UnsafeImageView {
        UnsafeImageView::raw(image, ty, mipmap_levels, array_layers).unwrap()
    }

    #[inline]
    pub fn format(&self) -> Format {
        self.format
    }

    #[inline]
    pub fn usage_transfer_source(&self) -> bool {
        (self.usage & vk::IMAGE_USAGE_TRANSFER_SRC_BIT) != 0
    }

    #[inline]
    pub fn usage_transfer_destination(&self) -> bool {
        (self.usage & vk::IMAGE_USAGE_TRANSFER_DST_BIT) != 0
    }

    #[inline]
    pub fn usage_sampled(&self) -> bool {
        (self.usage & vk::IMAGE_USAGE_SAMPLED_BIT) != 0
    }

    #[inline]
    pub fn usage_storage(&self) -> bool {
        (self.usage & vk::IMAGE_USAGE_STORAGE_BIT) != 0
    }

    #[inline]
    pub fn usage_color_attachment(&self) -> bool {
        (self.usage & vk::IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0
    }

    #[inline]
    pub fn usage_depth_stencil_attachment(&self) -> bool {
        (self.usage & vk::IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0
    }

    #[inline]
    pub fn usage_transient_attachment(&self) -> bool {
        (self.usage & vk::IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) != 0
    }

    #[inline]
    pub fn usage_input_attachment(&self) -> bool {
        (self.usage & vk::IMAGE_USAGE_INPUT_ATTACHMENT_BIT) != 0
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

/// The geometry type of an image view, along with its dimensions.
///
/// This is essentially a combination of [`ImageViewType`] and [`ImageDimensions`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ImageViewDimensions {
    Dim1d {
        width: u32,
    },
    Dim1dArray {
        width: u32,
        array_layers: u32,
    },
    Dim2d {
        width: u32,
        height: u32,
    },
    Dim2dArray {
        width: u32,
        height: u32,
        array_layers: u32,
    },
    Dim3d {
        width: u32,
        height: u32,
        depth: u32,
    },
    Cubemap {
        size: u32,
    },
    CubemapArray {
        size: u32,
        array_layers: u32,
    },
}

impl ImageViewDimensions {
    #[inline]
    pub fn width(&self) -> u32 {
        match *self {
            ImageViewDimensions::Dim1d { width } => width,
            ImageViewDimensions::Dim1dArray { width, .. } => width,
            ImageViewDimensions::Dim2d { width, .. } => width,
            ImageViewDimensions::Dim2dArray { width, .. } => width,
            ImageViewDimensions::Dim3d { width, .. } => width,
            ImageViewDimensions::Cubemap { size } => size,
            ImageViewDimensions::CubemapArray { size, .. } => size,
        }
    }

    #[inline]
    pub fn height(&self) -> u32 {
        match *self {
            ImageViewDimensions::Dim1d { .. } => 1,
            ImageViewDimensions::Dim1dArray { .. } => 1,
            ImageViewDimensions::Dim2d { height, .. } => height,
            ImageViewDimensions::Dim2dArray { height, .. } => height,
            ImageViewDimensions::Dim3d { height, .. } => height,
            ImageViewDimensions::Cubemap { size } => size,
            ImageViewDimensions::CubemapArray { size, .. } => size,
        }
    }

    #[inline]
    pub fn width_height(&self) -> [u32; 2] {
        [self.width(), self.height()]
    }

    #[inline]
    pub fn depth(&self) -> u32 {
        match *self {
            ImageViewDimensions::Dim1d { .. } => 1,
            ImageViewDimensions::Dim1dArray { .. } => 1,
            ImageViewDimensions::Dim2d { .. } => 1,
            ImageViewDimensions::Dim2dArray { .. } => 1,
            ImageViewDimensions::Dim3d { depth, .. } => depth,
            ImageViewDimensions::Cubemap { .. } => 1,
            ImageViewDimensions::CubemapArray { .. } => 1,
        }
    }

    #[inline]
    pub fn width_height_depth(&self) -> [u32; 3] {
        [self.width(), self.height(), self.depth()]
    }

    #[inline]
    pub fn array_layers(&self) -> u32 {
        match *self {
            ImageViewDimensions::Dim1d { .. } => 1,
            ImageViewDimensions::Dim1dArray { array_layers, .. } => array_layers,
            ImageViewDimensions::Dim2d { .. } => 1,
            ImageViewDimensions::Dim2dArray { array_layers, .. } => array_layers,
            ImageViewDimensions::Dim3d { .. } => 1,
            ImageViewDimensions::Cubemap { .. } => 1,
            ImageViewDimensions::CubemapArray { array_layers, .. } => array_layers,
        }
    }

    #[inline]
    pub fn array_layers_with_cube(&self) -> u32 {
        match *self {
            ImageViewDimensions::Dim1d { .. } => 1,
            ImageViewDimensions::Dim1dArray { array_layers, .. } => array_layers,
            ImageViewDimensions::Dim2d { .. } => 1,
            ImageViewDimensions::Dim2dArray { array_layers, .. } => array_layers,
            ImageViewDimensions::Dim3d { .. } => 1,
            ImageViewDimensions::Cubemap { .. } => 6,
            ImageViewDimensions::CubemapArray { array_layers, .. } => array_layers * 6,
        }
    }

    /// Builds the corresponding `ImageDimensions`.
    #[inline]
    pub fn to_image_dimensions(&self) -> ImageDimensions {
        match *self {
            ImageViewDimensions::Dim1d { width } => ImageDimensions::Dim1d {
                width,
                array_layers: 1,
            },
            ImageViewDimensions::Dim1dArray {
                width,
                array_layers,
            } => ImageDimensions::Dim1d {
                width,
                array_layers,
            },
            ImageViewDimensions::Dim2d { width, height } => ImageDimensions::Dim2d {
                width,
                height,
                array_layers: 1,
            },
            ImageViewDimensions::Dim2dArray {
                width,
                height,
                array_layers,
            } => ImageDimensions::Dim2d {
                width,
                height,
                array_layers,
            },
            ImageViewDimensions::Dim3d {
                width,
                height,
                depth,
            } => ImageDimensions::Dim3d {
                width,
                height,
                depth,
            },
            ImageViewDimensions::Cubemap { size } => ImageDimensions::Dim2d {
                width: size,
                height: size,
                array_layers: 6,
            },
            ImageViewDimensions::CubemapArray { size, array_layers } => ImageDimensions::Dim2d {
                width: size,
                height: size,
                array_layers: array_layers * 6,
            },
        }
    }

    /// Builds the corresponding `ImageViewType`.
    #[inline]
    pub fn to_image_view_type(&self) -> ImageViewType {
        match *self {
            ImageViewDimensions::Dim1d { .. } => ImageViewType::Dim1d,
            ImageViewDimensions::Dim1dArray { .. } => ImageViewType::Dim1dArray,
            ImageViewDimensions::Dim2d { .. } => ImageViewType::Dim2d,
            ImageViewDimensions::Dim2dArray { .. } => ImageViewType::Dim2dArray,
            ImageViewDimensions::Dim3d { .. } => ImageViewType::Dim3d,
            ImageViewDimensions::Cubemap { .. } => ImageViewType::Cubemap,
            ImageViewDimensions::CubemapArray { .. } => ImageViewType::CubemapArray,
        }
    }

    /// Returns the total number of texels for an image of these dimensions.
    #[inline]
    pub fn num_texels(&self) -> u32 {
        self.width() * self.height() * self.depth() * self.array_layers_with_cube()
    }
}

/// Trait for types that represent the GPU can access an image view.
pub unsafe trait ImageViewAccess {
    fn parent(&self) -> &dyn ImageAccess;

    /// Returns the dimensions of the image view.
    fn dimensions(&self) -> ImageViewDimensions;

    /// Returns the inner unsafe image view object used by this image view.
    fn inner(&self) -> &UnsafeImageView;

    /// Returns the format of this view. This can be different from the parent's format.
    #[inline]
    fn format(&self) -> Format {
        // TODO: remove this default impl
        self.inner().format()
    }

    #[inline]
    fn samples(&self) -> u32 {
        self.parent().samples()
    }

    /// Returns the image layout to use in a descriptor with the given subresource.
    fn descriptor_set_storage_image_layout(&self) -> ImageLayout;
    /// Returns the image layout to use in a descriptor with the given subresource.
    fn descriptor_set_combined_image_sampler_layout(&self) -> ImageLayout;
    /// Returns the image layout to use in a descriptor with the given subresource.
    fn descriptor_set_sampled_image_layout(&self) -> ImageLayout;
    /// Returns the image layout to use in a descriptor with the given subresource.
    fn descriptor_set_input_attachment_layout(&self) -> ImageLayout;

    /// Returns true if the view doesn't use components swizzling.
    ///
    /// Must be true when the view is used as a framebuffer attachment or TODO: I don't remember
    /// the other thing.
    fn identity_swizzle(&self) -> bool;

    /// Returns true if the given sampler can be used with this image view.
    ///
    /// This method should check whether the sampler's configuration can be used with the format
    /// of the view.
    // TODO: return a Result and propagate it when binding to a descriptor set
    fn can_be_sampled(&self, _sampler: &Sampler) -> bool {
        true /* FIXME */
    }

    //fn usable_as_render_pass_attachment(&self, ???) -> Result<(), ???>;
}

unsafe impl<T> ImageViewAccess for T
where
    T: SafeDeref,
    T::Target: ImageViewAccess,
{
    #[inline]
    fn parent(&self) -> &dyn ImageAccess {
        (**self).parent()
    }

    #[inline]
    fn inner(&self) -> &UnsafeImageView {
        (**self).inner()
    }

    #[inline]
    fn dimensions(&self) -> ImageViewDimensions {
        (**self).dimensions()
    }

    #[inline]
    fn descriptor_set_storage_image_layout(&self) -> ImageLayout {
        (**self).descriptor_set_storage_image_layout()
    }
    #[inline]
    fn descriptor_set_combined_image_sampler_layout(&self) -> ImageLayout {
        (**self).descriptor_set_combined_image_sampler_layout()
    }
    #[inline]
    fn descriptor_set_sampled_image_layout(&self) -> ImageLayout {
        (**self).descriptor_set_sampled_image_layout()
    }
    #[inline]
    fn descriptor_set_input_attachment_layout(&self) -> ImageLayout {
        (**self).descriptor_set_input_attachment_layout()
    }

    #[inline]
    fn identity_swizzle(&self) -> bool {
        (**self).identity_swizzle()
    }

    #[inline]
    fn can_be_sampled(&self, sampler: &Sampler) -> bool {
        (**self).can_be_sampled(sampler)
    }
}

impl PartialEq for dyn ImageViewAccess + Send + Sync {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner()
    }
}

impl Eq for dyn ImageViewAccess + Send + Sync {}

impl Hash for dyn ImageViewAccess + Send + Sync {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
    }
}
