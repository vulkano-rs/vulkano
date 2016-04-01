// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Low-level implementation of images and images views.
//! 
//! This module contains low-level wrappers around the Vulkan image and image view types. All
//! other image or image view types of this library, and all custom image or image view types
//! that you create must wrap around the types in this module.

use std::error;
use std::fmt;
use std::mem;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;
use smallvec::SmallVec;

use device::Device;
use format::Format;
use format::FormatTy;
use image::MipmapsCount;
use memory::DeviceMemory;
use memory::MemoryRequirements;
use sync::Sharing;

use Error;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// A storage for pixels or arbitrary data.
///
/// # Safety
///
/// This type is not just unsafe but very unsafe. Don't use it directly.
///
/// - You must manually bind memory to the image with `bind_memory`. The memory must respect the
///   requirements returned by `new`.
/// - The memory that you bind to the image must be manually kept alive.
/// - The queue family ownership must be manually enforced.
/// - The usage must be manually enforced.
/// - The image layout must be manually enforced and transitionned.
///
pub struct UnsafeImage {
    image: vk::Image,
    device: Arc<Device>,
    usage: vk::ImageUsageFlagBits,
    format: Format,

    dimensions: Dimensions,
    samples: u32,
    mipmaps: u32,

    // Features that are supported for this particular format.
    format_features: vk::FormatFeatureFlagBits,

    // `vkDestroyImage` is called only if `needs_destruction` is true.
    needs_destruction: bool,
}

impl UnsafeImage {
    /// Creates a new image and allocates memory for it.
    ///
    /// # Panic
    ///
    /// - Panicks if one of the dimensions is 0.
    /// - Panicks if the number of mipmaps is 0.
    /// - Panicks if the number of samples is 0.
    ///
    pub unsafe fn new<'a, Mi, I>(device: &Arc<Device>, usage: &Usage, format: Format,
                                 dimensions: Dimensions, num_samples: u32, mipmaps: Mi,
                                 sharing: Sharing<I>, linear_tiling: bool,
                                 preinitialized_layout: bool)
                                 -> Result<(UnsafeImage, MemoryRequirements), ImageCreationError>
        where Mi: Into<MipmapsCount>, I: Iterator<Item = u32>
    {
        // TODO: doesn't check that the proper features are enabled

        let vk = device.pointers();
        let vk_i = device.instance().pointers();

        // Preprocessing parameters.
        let sharing = sharing.into();

        // Checking if image usage conforms to what is supported.
        let format_features = {
            let physical_device = device.physical_device().internal_object();

            let mut output = mem::uninitialized();
            vk_i.GetPhysicalDeviceFormatProperties(physical_device, format as u32, &mut output);

            let features = if linear_tiling {
                output.linearTilingFeatures
            } else {
                output.optimalTilingFeatures
            };

            if features == 0 {
                return Err(ImageCreationError::FormatNotSupported);
            }

            if usage.sampled && (features & vk::FORMAT_FEATURE_SAMPLED_IMAGE_BIT == 0) {
                return Err(ImageCreationError::UnsupportedUsage);
            }
            if usage.storage && (features & vk::FORMAT_FEATURE_STORAGE_IMAGE_BIT == 0) {
                return Err(ImageCreationError::UnsupportedUsage);
            }
            if usage.color_attachment && (features & vk::FORMAT_FEATURE_COLOR_ATTACHMENT_BIT == 0) {
                return Err(ImageCreationError::UnsupportedUsage);
            }
            if usage.depth_stencil_attachment && (features & vk::FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT == 0) {
                return Err(ImageCreationError::UnsupportedUsage);
            }
            if usage.input_attachment && (features & (vk::FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | vk::FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) == 0) {
                return Err(ImageCreationError::UnsupportedUsage);
            }

            features
        };

        // If `transient_attachment` is true, then only `color_attachment`,
        // `depth_stencil_attachment` and `input_attachment` can be true as well.
        if usage.transient_attachment {
            let u = Usage {
                color_attachment: false,
                depth_stencil_attachment: false,
                input_attachment: false,
                .. usage.clone()
            };

            if u != Usage::none() {
                return Err(ImageCreationError::UnsupportedUsage);
            }
        }

        // This function is going to perform various checks and write to `capabilities_error` in
        // case of error.
        //
        // If `capabilities_error` is not `None` after the checks are finished, the function will
        // check for additional image capabilities (section 31.4 of the specs).
        let mut capabilities_error = None;

        // Compute the maximum number of mipmaps.
        // TODO: only compte if necessary?
        let max_mipmaps = {
            let smallest_dim: u32 = match dimensions {
                Dimensions::Dim1d { width } | Dimensions::Dim1dArray { width, .. } => width,
                Dimensions::Dim2d { width, height } | Dimensions::Dim2dArray { width, height, .. } => {
                    if width < height { width } else { height }
                },
                Dimensions::Dim3d { width, height, depth } => {
                    if width < height {
                        if depth < width { depth } else { width }
                    } else {
                        if depth < height { depth } else { height }
                    }
                },
            };

            32 - smallest_dim.leading_zeros()
        };

        // Compute the number of mipmaps.
        let mipmaps = match mipmaps.into() {
            MipmapsCount::Specific(num) => {
                if num < 1 {
                    return Err(ImageCreationError::InvalidMipmapsCount {
                        obtained: num, valid_range: 1 .. max_mipmaps + 1
                    });
                } else if num > max_mipmaps {
                    capabilities_error = Some(ImageCreationError::InvalidMipmapsCount {
                        obtained: num, valid_range: 1 .. max_mipmaps + 1
                    });
                }

                num
            },
            MipmapsCount::Log2 => max_mipmaps,
            MipmapsCount::One => 1,
        };

        // Checking whether the number of samples is supported.
        if num_samples == 0 {
            return Err(ImageCreationError::UnsupportedSamplesCount { obtained: num_samples });

        } else if !num_samples.is_power_of_two() {
            return Err(ImageCreationError::UnsupportedSamplesCount { obtained: num_samples });

        } else {
            let mut supported_samples = 0x7f;       // all bits up to VK_SAMPLE_COUNT_64_BIT

            if usage.sampled {
                match format.ty() {
                    FormatTy::Float | FormatTy::Compressed => {
                        supported_samples &= device.physical_device().limits()
                                                   .sampled_image_color_sample_counts();
                    },
                    FormatTy::Uint | FormatTy::Sint => {
                        supported_samples &= device.physical_device().limits()
                                                   .sampled_image_integer_sample_counts();
                    },
                    FormatTy::Depth => {
                        supported_samples &= device.physical_device().limits()
                                                   .sampled_image_depth_sample_counts();
                    },
                    FormatTy::Stencil => {
                        supported_samples &= device.physical_device().limits()
                                                   .sampled_image_stencil_sample_counts();
                    },
                    FormatTy::DepthStencil => {
                        supported_samples &= device.physical_device().limits()
                                                   .sampled_image_depth_sample_counts();
                        supported_samples &= device.physical_device().limits()
                                                   .sampled_image_stencil_sample_counts();
                    },
                }
            }

            if usage.storage {
                supported_samples &= device.physical_device().limits()
                                           .storage_image_sample_counts();
            }

            if usage.color_attachment || usage.depth_stencil_attachment || usage.input_attachment ||
               usage.transient_attachment
            {
                match format.ty() {
                    FormatTy::Float | FormatTy::Compressed | FormatTy::Uint | FormatTy::Sint => {
                        supported_samples &= device.physical_device().limits()
                                                   .framebuffer_color_sample_counts();
                    },
                    FormatTy::Depth => {
                        supported_samples &= device.physical_device().limits()
                                                   .framebuffer_depth_sample_counts();
                    },
                    FormatTy::Stencil => {
                        supported_samples &= device.physical_device().limits()
                                                   .framebuffer_stencil_sample_counts();
                    },
                    FormatTy::DepthStencil => {
                        supported_samples &= device.physical_device().limits()
                                                   .framebuffer_depth_sample_counts();
                        supported_samples &= device.physical_device().limits()
                                                   .framebuffer_stencil_sample_counts();
                    },
                }
            }

            if (num_samples & supported_samples) == 0 {
                let err = ImageCreationError::UnsupportedSamplesCount { obtained: num_samples };
                capabilities_error = Some(err);
            }
        }

        // If the `shaderStorageImageMultisample` feature is not enabled and we have
        // `usage_storage` set to true, then the number of samples must be 1.
        if usage.storage && num_samples > 1 {
            if !device.enabled_features().shader_storage_image_multisample {
                return Err(ImageCreationError::ShaderStorageImageMultisampleFeatureNotEnabled);
            }
        }

        // Decoding the dimensions.
        let (ty, extent, array_layers) = match dimensions {
            Dimensions::Dim1d { width } => {
                if width == 0 {
                    return Err(ImageCreationError::UnsupportedDimensions { dimensions: dimensions });
                }
                let extent = vk::Extent3D { width: width, height: 1, depth: 1 };
                (vk::IMAGE_TYPE_1D, extent, 1)
            },
            Dimensions::Dim1dArray { width, array_layers } => {
                if width == 0 || array_layers == 0 {
                    return Err(ImageCreationError::UnsupportedDimensions { dimensions: dimensions });
                }
                let extent = vk::Extent3D { width: width, height: 1, depth: 1 };
                (vk::IMAGE_TYPE_1D, extent, array_layers)
            },
            Dimensions::Dim2d { width, height } => {
                if width == 0 || height == 0 {
                    return Err(ImageCreationError::UnsupportedDimensions { dimensions: dimensions });
                }
                let extent = vk::Extent3D { width: width, height: height, depth: 1 };
                (vk::IMAGE_TYPE_2D, extent, 1)
            },
            Dimensions::Dim2dArray { width, height, array_layers } => {
                if width == 0 || height == 0 || array_layers == 0 {
                    return Err(ImageCreationError::UnsupportedDimensions { dimensions: dimensions });
                }
                let extent = vk::Extent3D { width: width, height: height, depth: 1 };
                (vk::IMAGE_TYPE_2D, extent, array_layers)
            },
            Dimensions::Dim3d { width, height, depth } => {
                if width == 0 || height == 0 || depth == 0 {
                    return Err(ImageCreationError::UnsupportedDimensions { dimensions: dimensions });
                }
                let extent = vk::Extent3D { width: width, height: height, depth: depth };
                (vk::IMAGE_TYPE_3D, extent, 1)
            },
        };

        // Checking the dimensions against the limits.
        if array_layers > device.physical_device().limits().max_image_array_layers() {
            let err = ImageCreationError::UnsupportedDimensions { dimensions: dimensions };
            capabilities_error = Some(err);
        }
        match ty {
            vk::IMAGE_TYPE_1D => {
                if extent.width > device.physical_device().limits().max_image_dimension_1d() {
                    let err = ImageCreationError::UnsupportedDimensions { dimensions: dimensions };
                    capabilities_error = Some(err);
                }
            },
            vk::IMAGE_TYPE_2D => {
                let limit = device.physical_device().limits().max_image_dimension_2d();
                if extent.width > limit || extent.height > limit {
                    let err = ImageCreationError::UnsupportedDimensions { dimensions: dimensions };
                    capabilities_error = Some(err);
                }
            },
            vk::IMAGE_TYPE_3D => {
                let limit = device.physical_device().limits().max_image_dimension_3d();
                if extent.width > limit || extent.height > limit || extent.depth > limit {
                    let err = ImageCreationError::UnsupportedDimensions { dimensions: dimensions };
                    capabilities_error = Some(err);
                }
            },
            _ => unreachable!()
        };

        let usage = usage.to_usage_bits();

        // Now that all checks have been performed, if any of the check failed we query the Vulkan
        // implementation for additional image capabilities.
        if let Some(capabilities_error) = capabilities_error {
            let tiling = if linear_tiling {
                vk::IMAGE_TILING_LINEAR
            } else {
                vk::IMAGE_TILING_OPTIMAL
            };

            let mut output = mem::uninitialized();
            let physical_device = device.physical_device().internal_object();
            let r = vk_i.GetPhysicalDeviceImageFormatProperties(physical_device, format as u32, ty,
                                                                tiling, usage, 0 /* TODO */,
                                                                &mut output);

            match check_errors(r) {
                Ok(_) => (),
                Err(Error::FormatNotSupported) => return Err(ImageCreationError::FormatNotSupported),
                Err(err) => return Err(err.into()),
            }

            if extent.width > output.maxExtent.width || extent.height > output.maxExtent.height ||
               extent.depth > output.maxExtent.depth || mipmaps > output.maxMipLevels ||
               array_layers > output.maxArrayLayers || (num_samples & output.sampleCounts) == 0
            {
                return Err(capabilities_error);
            }
        }

        // Everything now ok. Creating the image.
        let image = {
            let (sh_mode, sh_indices) = match sharing {
                Sharing::Exclusive => (vk::SHARING_MODE_EXCLUSIVE, SmallVec::<[u32; 8]>::new()),
                Sharing::Concurrent(ids) => (vk::SHARING_MODE_CONCURRENT, ids.collect()),
            };

            let infos = vk::ImageCreateInfo {
                sType: vk::STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,                               // TODO:
                imageType: ty,
                format: format as u32,
                extent: extent,
                mipLevels: mipmaps,
                arrayLayers: array_layers,
                samples: num_samples,
                tiling: if linear_tiling {
                    vk::IMAGE_TILING_LINEAR
                } else {
                    vk::IMAGE_TILING_OPTIMAL
                },
                usage: usage,
                sharingMode: sh_mode,
                queueFamilyIndexCount: sh_indices.len() as u32,
                pQueueFamilyIndices: sh_indices.as_ptr(),
                initialLayout: if preinitialized_layout {
                    vk::IMAGE_LAYOUT_PREINITIALIZED
                } else {
                    vk::IMAGE_LAYOUT_UNDEFINED
                },
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateImage(device.internal_object(), &infos,
                                             ptr::null(), &mut output)));
            output
        };

        let mem_reqs: vk::MemoryRequirements = {
            let mut output = mem::uninitialized();
            vk.GetImageMemoryRequirements(device.internal_object(), image, &mut output);
            debug_assert!(output.memoryTypeBits != 0);
            output
        };

        let image = UnsafeImage {
            device: device.clone(),
            image: image,
            usage: usage,
            format: format,
            dimensions: dimensions,
            samples: num_samples,
            mipmaps: mipmaps,
            format_features: format_features,
            needs_destruction: true,
        };

        Ok((image, mem_reqs.into()))
    }

    /// Creates an image from a raw handle. The image won't be destroyed.
    ///
    /// This function is for example used at the swapchain's initialization.
    pub unsafe fn from_raw(device: &Arc<Device>, handle: u64, usage: u32, format: Format,
                           dimensions: Dimensions, samples: u32, mipmaps: u32)
                           -> UnsafeImage
    {
        let vk_i = device.instance().pointers();
        let physical_device = device.physical_device().internal_object();

        let mut output = mem::uninitialized();
        vk_i.GetPhysicalDeviceFormatProperties(physical_device, format as u32, &mut output);

        // TODO: check that usage is correct in regard to `output`?

        UnsafeImage {
            device: device.clone(),
            image: handle,
            usage: usage,
            format: format,
            dimensions: dimensions,
            samples: samples,
            mipmaps: mipmaps,
            format_features: output.optimalTilingFeatures,
            needs_destruction: false,       // TODO: pass as parameter
        }
    }

    pub unsafe fn bind_memory(&self, memory: &DeviceMemory, offset: usize)
                              -> Result<(), OomError>
    {
        let vk = self.device.pointers();

        // We check for correctness in debug mode.
        debug_assert!({
            let mut mem_reqs = mem::uninitialized();
            vk.GetImageMemoryRequirements(self.device.internal_object(), self.image,
                                          &mut mem_reqs);
            mem_reqs.size <= (memory.size() - offset) as u64 &&
            (offset as u64 % mem_reqs.alignment) == 0 &&
            mem_reqs.memoryTypeBits & (1 << memory.memory_type().id()) != 0
        });

        try!(check_errors(vk.BindImageMemory(self.device.internal_object(), self.image,
                                             memory.internal_object(), offset as vk::DeviceSize)));
        Ok(())
    }

    #[inline]
    pub fn format(&self) -> Format {
        self.format
    }

    #[inline]
    pub fn mipmap_levels(&self) -> u32 {
        self.mipmaps
    }

    #[inline]
    pub fn dimensions(&self) -> Dimensions {
        self.dimensions
    }

    #[inline]
    pub fn samples(&self) -> u32 {
        self.samples
    }

    /// Returns true if the image can be used as a source for blits.
    #[inline]
    pub fn supports_blit_source(&self) -> bool {
        (self.format_features & vk::FORMAT_FEATURE_BLIT_SRC_BIT) != 0
    }

    /// Returns true if the image can be used as a destination for blits.
    #[inline]
    pub fn supports_blit_destination(&self) -> bool {
        (self.format_features & vk::FORMAT_FEATURE_BLIT_DST_BIT) != 0
    }
}

unsafe impl VulkanObject for UnsafeImage {
    type Object = vk::Image;

    #[inline]
    fn internal_object(&self) -> vk::Image {
        self.image
    }
}

impl Drop for UnsafeImage {
    #[inline]
    fn drop(&mut self) {
        if !self.needs_destruction {
            return;
        }

        unsafe {
            let vk = self.device.pointers();
            vk.DestroyImage(self.device.internal_object(), self.image, ptr::null());
        }
    }
}

/// Error that can happen when creating an instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageCreationError {
    /// Not enough memory.
    OomError(OomError),
    /// A wrong number of mipmaps was provided.
    InvalidMipmapsCount { obtained: u32, valid_range: Range<u32> },
    /// The requeted number of samples is not supported, or is 0.
    UnsupportedSamplesCount { obtained: u32 },
    /// The dimensions are too large, or one of the dimensions is 0.
    UnsupportedDimensions { dimensions: Dimensions },
    /// The requested format is not supported by the Vulkan implementation.
    FormatNotSupported,
    /// The format is supported, but at least one of the requested usages is not supported.
    UnsupportedUsage,
    /// The `shader_storage_image_multisample` feature must be enabled to create such an image.
    ShaderStorageImageMultisampleFeatureNotEnabled,
}

impl error::Error for ImageCreationError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            ImageCreationError::OomError(_) => "not enough memory available",
            ImageCreationError::InvalidMipmapsCount { .. } => "a wrong number of mipmaps was \
                                                               provided",
            ImageCreationError::UnsupportedSamplesCount { .. } => "the requeted number of samples \
                                                                   is not supported, or is 0",
            ImageCreationError::UnsupportedDimensions { .. } => "the dimensions are too large, or \
                                                                 one of the dimensions is 0",
            ImageCreationError::FormatNotSupported => "the requested format is not supported by \
                                                       the Vulkan implementation",
            ImageCreationError::UnsupportedUsage => "the format is supported, but at least one \
                                                     of the requested usages is not supported",
            ImageCreationError::ShaderStorageImageMultisampleFeatureNotEnabled => {
                "the `shader_storage_image_multisample` feature must be enabled to create such \
                 an image"
            },
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            ImageCreationError::OomError(ref err) => Some(err),
            _ => None
        }
    }
}

impl fmt::Display for ImageCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<OomError> for ImageCreationError {
    #[inline]
    fn from(err: OomError) -> ImageCreationError {
        ImageCreationError::OomError(err)
    }
}

impl From<Error> for ImageCreationError {
    #[inline]
    fn from(err: Error) -> ImageCreationError {
        match err {
            err @ Error::OutOfHostMemory => ImageCreationError::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => ImageCreationError::OomError(OomError::from(err)),
            _ => panic!("unexpected error: {:?}", err)
        }
    }
}

pub struct UnsafeImageView {
    view: vk::ImageView,
    device: Arc<Device>,
    usage: vk::ImageUsageFlagBits,
    identity_swizzle: bool,
    format: Format,
}

impl UnsafeImageView {
    /// Creates a new view from an image.
    ///
    /// Note that you must create the view with identity swizzling if you want to use this view
    /// as a framebuffer attachment.
    pub unsafe fn new(image: &UnsafeImage, mipmap_levels: Range<u32>, array_layers: Range<u32>)
                      -> Result<UnsafeImageView, OomError>
    {
        let vk = image.device.pointers();

        assert!(mipmap_levels.end > mipmap_levels.start);
        assert!(mipmap_levels.end <= image.mipmaps);
        assert!(array_layers.end > array_layers.start);
        assert!(array_layers.end <= image.dimensions.array_layers());

        let aspect_mask = match image.format.ty() {
            FormatTy::Float | FormatTy::Uint | FormatTy::Sint | FormatTy::Compressed => {
                vk::IMAGE_ASPECT_COLOR_BIT
            },
            FormatTy::Depth => vk::IMAGE_ASPECT_DEPTH_BIT,
            FormatTy::Stencil => vk::IMAGE_ASPECT_STENCIL_BIT,
            FormatTy::DepthStencil => vk::IMAGE_ASPECT_DEPTH_BIT | vk::IMAGE_ASPECT_STENCIL_BIT,
        };

        let view = {
            let infos = vk::ImageViewCreateInfo {
                sType: vk::STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                image: image.internal_object(),
                viewType: vk::IMAGE_VIEW_TYPE_2D,     // FIXME:
                format: image.format as u32,
                components: vk::ComponentMapping { r: 0, g: 0, b: 0, a: 0 },     // FIXME:
                subresourceRange: vk::ImageSubresourceRange {
                    aspectMask: aspect_mask,
                    baseMipLevel: mipmap_levels.start,
                    levelCount: mipmap_levels.end - mipmap_levels.start,
                    baseArrayLayer: array_layers.start,
                    layerCount: array_layers.end - array_layers.start,
                },
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateImageView(image.device.internal_object(), &infos,
                                                 ptr::null(), &mut output)));
            output
        };

        Ok(UnsafeImageView {
            view: view,
            device: image.device.clone(),
            usage: image.usage,
            identity_swizzle: true,     // FIXME:
            format: image.format,
        })
    }

    #[inline]
    pub fn format(&self) -> Format {
        self.format
    }

    #[inline]
    pub fn usage_transfer_src(&self) -> bool {
        (self.usage & vk::IMAGE_USAGE_TRANSFER_SRC_BIT) != 0
    }

    #[inline]
    pub fn usage_transfer_dest(&self) -> bool {
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

    #[inline]
    fn internal_object(&self) -> vk::ImageView {
        self.view
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Dimensions {
    Dim1d { width: u32 },
    Dim1dArray { width: u32, array_layers: u32 },
    Dim2d { width: u32, height: u32 },
    Dim2dArray { width: u32, height: u32, array_layers: u32 },
    Dim3d { width: u32, height: u32, depth: u32 }
}

impl Dimensions {
    #[inline]
    pub fn width(&self) -> u32 {
        match *self {
            Dimensions::Dim1d { width } => width,
            Dimensions::Dim1dArray { width, .. } => width,
            Dimensions::Dim2d { width, .. } => width,
            Dimensions::Dim2dArray { width, .. } => width,
            Dimensions::Dim3d { width, .. }  => width,
        }
    }

    #[inline]
    pub fn height(&self) -> u32 {
        match *self {
            Dimensions::Dim1d { .. } => 1,
            Dimensions::Dim1dArray { .. } => 1,
            Dimensions::Dim2d { height, .. } => height,
            Dimensions::Dim2dArray { height, .. } => height,
            Dimensions::Dim3d { height, .. }  => height,
        }
    }

    #[inline]
    pub fn width_height(&self) -> [u32; 2] {
        [self.width(), self.height()]
    }

    #[inline]
    pub fn array_layers(&self) -> u32 {
        match *self {
            Dimensions::Dim1d { .. } => 1,
            Dimensions::Dim1dArray { array_layers, .. } => array_layers,
            Dimensions::Dim2d { .. } => 1,
            Dimensions::Dim2dArray { array_layers, .. } => array_layers,
            Dimensions::Dim3d { .. }  => 1,
        }
    }
}

/// Describes how an image is going to be used. This is **not** an optimization.
///
/// If you try to use an image in a way that you didn't declare, a panic will happen.
///
/// If `transient_attachment` is true, then only `color_attachment`, `depth_stencil_attachment`
/// and `input_attachment` can be true as well. The rest must be false or an error will be returned
/// when creating the image.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Usage {
    /// Can be used a source for transfers. Includes blits.
    pub transfer_source: bool,

    /// Can be used a destination for transfers. Includes blits.
    pub transfer_dest: bool,

    /// Can be sampled from a shader.
    pub sampled: bool,

    /// Can be used as an image storage in a shader.
    pub storage: bool,

    /// Can be attached as a color attachment to a framebuffer.
    pub color_attachment: bool,

    /// Can be attached as a depth, stencil or depth-stencil attachment to a framebuffer.
    pub depth_stencil_attachment: bool,

    /// Indicates that this image will only ever be used as a temporary framebuffer attachment.
    /// As soon as you leave a render pass, the content of transient images becomes undefined.
    ///
    /// This is a hint to the Vulkan implementation that it may not need allocate any memory for
    /// this image if the image can live entirely in some cache.
    pub transient_attachment: bool,

    /// Can be used as an input attachment. In other words, you can draw to it in a subpass then
    /// read from it in a following pass.
    pub input_attachment: bool,
}

impl Usage {
    /// Builds a `Usage` with all values set to true. Note that using the returned value will
    /// produce an error because of `transient_attachment` being true.
    #[inline]
    pub fn all() -> Usage {
        Usage {
            transfer_source: true,
            transfer_dest: true,
            sampled: true,
            storage: true,
            color_attachment: true,
            depth_stencil_attachment: true,
            transient_attachment: true,
            input_attachment: true,
        }
    }

    /// Builds a `Usage` with all values set to false. Useful as a default value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use vulkano::image::Usage as ImageUsage;
    ///
    /// let _usage = ImageUsage {
    ///     transfer_dest: true,
    ///     sampled: true,
    ///     .. ImageUsage::none()
    /// };
    /// ```
    #[inline]
    pub fn none() -> Usage {
        Usage {
            transfer_source: false,
            transfer_dest: false,
            sampled: false,
            storage: false,
            color_attachment: false,
            depth_stencil_attachment: false,
            transient_attachment: false,
            input_attachment: false,
        }
    }

    #[doc(hidden)]
    #[inline]
    pub fn to_usage_bits(&self) -> vk::ImageUsageFlagBits {
        let mut result = 0;
        if self.transfer_source { result |= vk::IMAGE_USAGE_TRANSFER_SRC_BIT; }
        if self.transfer_dest { result |= vk::IMAGE_USAGE_TRANSFER_DST_BIT; }
        if self.sampled { result |= vk::IMAGE_USAGE_SAMPLED_BIT; }
        if self.storage { result |= vk::IMAGE_USAGE_STORAGE_BIT; }
        if self.color_attachment { result |= vk::IMAGE_USAGE_COLOR_ATTACHMENT_BIT; }
        if self.depth_stencil_attachment { result |= vk::IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT; }
        if self.transient_attachment { result |= vk::IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT; }
        if self.input_attachment { result |= vk::IMAGE_USAGE_INPUT_ATTACHMENT_BIT; }
        result
    }

    #[inline]
    #[doc(hidden)]
    pub fn from_bits(val: u32) -> Usage {
        Usage {
            transfer_source: (val & vk::IMAGE_USAGE_TRANSFER_SRC_BIT) != 0,
            transfer_dest: (val & vk::IMAGE_USAGE_TRANSFER_DST_BIT) != 0,
            sampled: (val & vk::IMAGE_USAGE_SAMPLED_BIT) != 0,
            storage: (val & vk::IMAGE_USAGE_STORAGE_BIT) != 0,
            color_attachment: (val & vk::IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0,
            depth_stencil_attachment: (val & vk::IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0,
            transient_attachment: (val & vk::IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) != 0,
            input_attachment: (val & vk::IMAGE_USAGE_INPUT_ATTACHMENT_BIT) != 0,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum Layout {
    Undefined = vk::IMAGE_LAYOUT_UNDEFINED,
    General = vk::IMAGE_LAYOUT_GENERAL,
    ColorAttachmentOptimal = vk::IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    DepthStencilAttachmentOptimal = vk::IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    DepthStencilReadOnlyOptimal = vk::IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
    ShaderReadOnlyOptimal = vk::IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    TransferSrcOptimal = vk::IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    TransferDstOptimal = vk::IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    Preinitialized = vk::IMAGE_LAYOUT_PREINITIALIZED,
    PresentSrc = vk::IMAGE_LAYOUT_PRESENT_SRC_KHR,
}

#[cfg(test)]
mod tests {
    use std::iter::Empty;
    use std::u32;

    use super::Dimensions;
    use super::ImageCreationError;
    use super::UnsafeImage;
    use super::Usage;

    use format::Format;
    use sync::Sharing;

    #[test]
    fn create() {
        let (device, _) = gfx_dev_and_queue!();

        let usage = Usage {
            sampled: true,
            .. Usage::none()
        };

        let (_img, _) = unsafe {
            UnsafeImage::new(&device, &usage, Format::R8G8B8A8Unorm,
                             Dimensions::Dim2d { width: 32, height: 32 }, 1, 1,
                             Sharing::Exclusive::<Empty<_>>, false, false)
        }.unwrap();
    }

    #[test]
    fn zero_sample() {
        let (device, _) = gfx_dev_and_queue!();

        let usage = Usage {
            sampled: true,
            .. Usage::none()
        };

        let res = unsafe {
            UnsafeImage::new(&device, &usage, Format::R8G8B8A8Unorm,
                             Dimensions::Dim2d { width: 32, height: 32 }, 0, 1,
                             Sharing::Exclusive::<Empty<_>>, false, false)
        };

        match res {
            Err(ImageCreationError::UnsupportedSamplesCount { .. }) => (),
            _ => panic!()
        };
    }

    #[test]
    fn non_po2_sample() {
        let (device, _) = gfx_dev_and_queue!();

        let usage = Usage {
            sampled: true,
            .. Usage::none()
        };

        let res = unsafe {
            UnsafeImage::new(&device, &usage, Format::R8G8B8A8Unorm,
                             Dimensions::Dim2d { width: 32, height: 32 }, 5, 1,
                             Sharing::Exclusive::<Empty<_>>, false, false)
        };

        match res {
            Err(ImageCreationError::UnsupportedSamplesCount { .. }) => (),
            _ => panic!()
        };
    }

    #[test]
    fn zero_mipmap() {
        let (device, _) = gfx_dev_and_queue!();

        let usage = Usage {
            sampled: true,
            .. Usage::none()
        };

        let res = unsafe {
            UnsafeImage::new(&device, &usage, Format::R8G8B8A8Unorm,
                             Dimensions::Dim2d { width: 32, height: 32 }, 1, 0,
                             Sharing::Exclusive::<Empty<_>>, false, false)
        };

        match res {
            Err(ImageCreationError::InvalidMipmapsCount { .. }) => (),
            _ => panic!()
        };
    }

    #[test]
    #[ignore]       // TODO: AMD card seems to support a u32::MAX number of mipmaps
    fn mipmaps_too_high() {
        let (device, _) = gfx_dev_and_queue!();

        let usage = Usage {
            sampled: true,
            .. Usage::none()
        };

        let res = unsafe {
            UnsafeImage::new(&device, &usage, Format::R8G8B8A8Unorm,
                             Dimensions::Dim2d { width: 32, height: 32 }, 1, u32::MAX,
                             Sharing::Exclusive::<Empty<_>>, false, false)
        };

        match res {
            Err(ImageCreationError::InvalidMipmapsCount { obtained, valid_range }) => {
                assert_eq!(obtained, u32::MAX);
                assert_eq!(valid_range.start, 1);
            },
            _ => panic!()
        };
    }

    #[test]
    fn shader_storage_image_multisample() {
        let (device, _) = gfx_dev_and_queue!();

        let usage = Usage {
            storage: true,
            .. Usage::none()
        };

        let res = unsafe {
            UnsafeImage::new(&device, &usage, Format::R8G8B8A8Unorm,
                             Dimensions::Dim2d { width: 32, height: 32 }, 2, 1,
                             Sharing::Exclusive::<Empty<_>>, false, false)
        };

        match res {
            Err(ImageCreationError::ShaderStorageImageMultisampleFeatureNotEnabled) => (),
            Err(ImageCreationError::UnsupportedSamplesCount { .. }) => (), // unlikely but possible
            _ => panic!()
        };
    }

    #[test]
    fn compressed_not_color_attachment() {
        let (device, _) = gfx_dev_and_queue!();

        let usage = Usage {
            color_attachment: true,
            .. Usage::none()
        };

        let res = unsafe {
            UnsafeImage::new(&device, &usage, Format::ASTC_5x4UnormBlock,
                             Dimensions::Dim2d { width: 32, height: 32 }, 1, u32::MAX,
                             Sharing::Exclusive::<Empty<_>>, false, false)
        };

        match res {
            Err(ImageCreationError::FormatNotSupported) => (),
            Err(ImageCreationError::UnsupportedUsage) => (),
            _ => panic!()
        };
    }

    #[test]
    fn transient_forbidden_with_some_usages() {
        let (device, _) = gfx_dev_and_queue!();

        let usage = Usage {
            transient_attachment: true,
            sampled: true,
            .. Usage::none()
        };

        let res = unsafe {
            UnsafeImage::new(&device, &usage, Format::R8G8B8A8Unorm,
                             Dimensions::Dim2d { width: 32, height: 32 }, 1, 1,
                             Sharing::Exclusive::<Empty<_>>, false, false)
        };

        match res {
            Err(ImageCreationError::UnsupportedUsage) => (),
            _ => panic!()
        };
    }
}
