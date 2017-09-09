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

use smallvec::SmallVec;
use std::error;
use std::fmt;
use std::mem;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;

use device::Device;
use format::Format;
use format::FormatTy;
use image::ImageDimensions;
use image::ImageUsage;
use image::MipmapsCount;
use image::ViewType;
use memory::DeviceMemory;
use memory::DeviceMemoryAllocError;
use memory::MemoryRequirements;
use sync::Sharing;

use Error;
use OomError;
use VulkanObject;
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

    dimensions: ImageDimensions,
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
    /// - Panics if one of the dimensions is 0.
    /// - Panics if the number of mipmaps is 0.
    /// - Panics if the number of samples is 0.
    ///
    #[inline]
    pub unsafe fn new<'a, Mi, I>(device: Arc<Device>, usage: ImageUsage, format: Format,
                                 dimensions: ImageDimensions, num_samples: u32, mipmaps: Mi,
                                 sharing: Sharing<I>, linear_tiling: bool,
                                 preinitialized_layout: bool)
                                 -> Result<(UnsafeImage, MemoryRequirements), ImageCreationError>
        where Mi: Into<MipmapsCount>,
              I: Iterator<Item = u32>
    {
        let sharing = match sharing {
            Sharing::Exclusive => (vk::SHARING_MODE_EXCLUSIVE, SmallVec::<[u32; 8]>::new()),
            Sharing::Concurrent(ids) => (vk::SHARING_MODE_CONCURRENT, ids.collect()),
        };

        UnsafeImage::new_impl(device,
                              usage,
                              format,
                              dimensions,
                              num_samples,
                              mipmaps.into(),
                              sharing,
                              linear_tiling,
                              preinitialized_layout)
    }

    // Non-templated version to avoid inlining and improve compile times.
    unsafe fn new_impl(device: Arc<Device>, usage: ImageUsage, format: Format,
                       dimensions: ImageDimensions, num_samples: u32, mipmaps: MipmapsCount,
                       (sh_mode, sh_indices): (vk::SharingMode, SmallVec<[u32; 8]>),
                       linear_tiling: bool, preinitialized_layout: bool)
                       -> Result<(UnsafeImage, MemoryRequirements), ImageCreationError> {
        // TODO: doesn't check that the proper features are enabled

        let vk = device.pointers();
        let vk_i = device.instance().pointers();

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
            if usage.depth_stencil_attachment &&
                (features & vk::FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT == 0)
            {
                return Err(ImageCreationError::UnsupportedUsage);
            }
            if usage.input_attachment &&
                (features &
                     (vk::FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                          vk::FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) == 0)
            {
                return Err(ImageCreationError::UnsupportedUsage);
            }
            if device.loaded_extensions().khr_maintenance1 {
                if usage.transfer_source &&
                    (features & vk::FORMAT_FEATURE_TRANSFER_SRC_BIT_KHR == 0)
                {
                    return Err(ImageCreationError::UnsupportedUsage);
                }
                if usage.transfer_destination &&
                    (features & vk::FORMAT_FEATURE_TRANSFER_DST_BIT_KHR == 0)
                {
                    return Err(ImageCreationError::UnsupportedUsage);
                }
            }

            features
        };

        // If `transient_attachment` is true, then only `color_attachment`,
        // `depth_stencil_attachment` and `input_attachment` can be true as well.
        if usage.transient_attachment {
            let u = ImageUsage {
                transient_attachment: false,
                color_attachment: false,
                depth_stencil_attachment: false,
                input_attachment: false,
                ..usage.clone()
            };

            if u != ImageUsage::none() {
                return Err(ImageCreationError::UnsupportedUsage);
            }
        }

        // This function is going to perform various checks and write to `capabilities_error` in
        // case of error.
        //
        // If `capabilities_error` is not `None` after the checks are finished, the function will
        // check for additional image capabilities (section 31.4 of the specs).
        let mut capabilities_error = None;

        // Compute the number of mipmaps.
        let mipmaps = match mipmaps.into() {
            MipmapsCount::Specific(num) => {
                let max_mipmaps = dimensions.max_mipmaps();
                debug_assert!(max_mipmaps >= 1);
                if num < 1 {
                    return Err(ImageCreationError::InvalidMipmapsCount {
                                   obtained: num,
                                   valid_range: 1 .. max_mipmaps + 1,
                               });
                } else if num > max_mipmaps {
                    capabilities_error = Some(ImageCreationError::InvalidMipmapsCount {
                                                  obtained: num,
                                                  valid_range: 1 .. max_mipmaps + 1,
                                              });
                }

                num
            },
            MipmapsCount::Log2 => dimensions.max_mipmaps(),
            MipmapsCount::One => 1,
        };

        // Checking whether the number of samples is supported.
        if num_samples == 0 {
            return Err(ImageCreationError::UnsupportedSamplesCount { obtained: num_samples });

        } else if !num_samples.is_power_of_two() {
            return Err(ImageCreationError::UnsupportedSamplesCount { obtained: num_samples });

        } else {
            let mut supported_samples = 0x7f; // all bits up to VK_SAMPLE_COUNT_64_BIT

            if usage.sampled {
                match format.ty() {
                    FormatTy::Float | FormatTy::Compressed => {
                        supported_samples &= device
                            .physical_device()
                            .limits()
                            .sampled_image_color_sample_counts();
                    },
                    FormatTy::Uint | FormatTy::Sint => {
                        supported_samples &= device
                            .physical_device()
                            .limits()
                            .sampled_image_integer_sample_counts();
                    },
                    FormatTy::Depth => {
                        supported_samples &= device
                            .physical_device()
                            .limits()
                            .sampled_image_depth_sample_counts();
                    },
                    FormatTy::Stencil => {
                        supported_samples &= device
                            .physical_device()
                            .limits()
                            .sampled_image_stencil_sample_counts();
                    },
                    FormatTy::DepthStencil => {
                        supported_samples &= device
                            .physical_device()
                            .limits()
                            .sampled_image_depth_sample_counts();
                        supported_samples &= device
                            .physical_device()
                            .limits()
                            .sampled_image_stencil_sample_counts();
                    },
                }
            }

            if usage.storage {
                supported_samples &= device
                    .physical_device()
                    .limits()
                    .storage_image_sample_counts();
            }

            if usage.color_attachment || usage.depth_stencil_attachment ||
                usage.input_attachment || usage.transient_attachment
            {
                match format.ty() {
                    FormatTy::Float | FormatTy::Compressed | FormatTy::Uint | FormatTy::Sint => {
                        supported_samples &= device
                            .physical_device()
                            .limits()
                            .framebuffer_color_sample_counts();
                    },
                    FormatTy::Depth => {
                        supported_samples &= device
                            .physical_device()
                            .limits()
                            .framebuffer_depth_sample_counts();
                    },
                    FormatTy::Stencil => {
                        supported_samples &= device
                            .physical_device()
                            .limits()
                            .framebuffer_stencil_sample_counts();
                    },
                    FormatTy::DepthStencil => {
                        supported_samples &= device
                            .physical_device()
                            .limits()
                            .framebuffer_depth_sample_counts();
                        supported_samples &= device
                            .physical_device()
                            .limits()
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
        let (ty, extent, array_layers, flags) = match dimensions {
            ImageDimensions::Dim1d {
                width,
                array_layers,
            } => {
                if width == 0 || array_layers == 0 {
                    return Err(ImageCreationError::UnsupportedDimensions {
                                   dimensions: dimensions,
                               });
                }
                let extent = vk::Extent3D {
                    width: width,
                    height: 1,
                    depth: 1,
                };
                (vk::IMAGE_TYPE_1D, extent, array_layers, 0)
            },
            ImageDimensions::Dim2d {
                width,
                height,
                array_layers,
                cubemap_compatible,
            } => {
                if width == 0 || height == 0 || array_layers == 0 {
                    return Err(ImageCreationError::UnsupportedDimensions {
                                   dimensions: dimensions,
                               });
                }
                if cubemap_compatible && width != height {
                    return Err(ImageCreationError::UnsupportedDimensions {
                                   dimensions: dimensions,
                               });
                }
                let extent = vk::Extent3D {
                    width: width,
                    height: height,
                    depth: 1,
                };
                let flags = if cubemap_compatible {
                    vk::IMAGE_CREATE_CUBE_COMPATIBLE_BIT
                } else {
                    0
                };
                (vk::IMAGE_TYPE_2D, extent, array_layers, flags)
            },
            ImageDimensions::Dim3d {
                width,
                height,
                depth,
            } => {
                if width == 0 || height == 0 || depth == 0 {
                    return Err(ImageCreationError::UnsupportedDimensions {
                                   dimensions: dimensions,
                               });
                }
                let extent = vk::Extent3D {
                    width: width,
                    height: height,
                    depth: depth,
                };
                (vk::IMAGE_TYPE_3D, extent, 1, 0)
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

                if (flags & vk::IMAGE_CREATE_CUBE_COMPATIBLE_BIT) != 0 {
                    let limit = device.physical_device().limits().max_image_dimension_cube();
                    debug_assert_eq!(extent.width, extent.height); // checked above
                    if extent.width > limit {
                        let err =
                            ImageCreationError::UnsupportedDimensions { dimensions: dimensions };
                        capabilities_error = Some(err);
                    }
                }
            },
            vk::IMAGE_TYPE_3D => {
                let limit = device.physical_device().limits().max_image_dimension_3d();
                if extent.width > limit || extent.height > limit || extent.depth > limit {
                    let err = ImageCreationError::UnsupportedDimensions { dimensions: dimensions };
                    capabilities_error = Some(err);
                }
            },
            _ => unreachable!(),
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
            let r = vk_i.GetPhysicalDeviceImageFormatProperties(physical_device,
                                                                format as u32,
                                                                ty,
                                                                tiling,
                                                                usage,
                                                                0, /* TODO */
                                                                &mut output);

            match check_errors(r) {
                Ok(_) => (),
                Err(Error::FormatNotSupported) =>
                    return Err(ImageCreationError::FormatNotSupported),
                Err(err) => return Err(err.into()),
            }

            if extent.width > output.maxExtent.width || extent.height > output.maxExtent.height ||
                extent.depth > output.maxExtent.depth ||
                mipmaps > output.maxMipLevels ||
                array_layers > output.maxArrayLayers ||
                (num_samples & output.sampleCounts) == 0
            {
                return Err(capabilities_error);
            }
        }

        // Everything now ok. Creating the image.
        let image = {
            let infos = vk::ImageCreateInfo {
                sType: vk::STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                pNext: ptr::null(),
                flags: flags,
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
            check_errors(vk.CreateImage(device.internal_object(),
                                        &infos,
                                        ptr::null(),
                                        &mut output))?;
            output
        };

        let mem_reqs = if device.loaded_extensions().khr_get_memory_requirements2 {
            let infos = vk::ImageMemoryRequirementsInfo2KHR {
                sType: vk::STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2_KHR,
                pNext: ptr::null_mut(),
                image: image,
            };

            let mut output2 = if device.loaded_extensions().khr_dedicated_allocation {
                Some(vk::MemoryDedicatedRequirementsKHR {
                         sType: vk::STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR,
                         pNext: ptr::null(),
                         prefersDedicatedAllocation: mem::uninitialized(),
                         requiresDedicatedAllocation: mem::uninitialized(),
                     })
            } else {
                None
            };

            let mut output = vk::MemoryRequirements2KHR {
                sType: vk::STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR,
                pNext: output2
                    .as_mut()
                    .map(|o| o as *mut vk::MemoryDedicatedRequirementsKHR)
                    .unwrap_or(ptr::null_mut()) as *mut _,
                memoryRequirements: mem::uninitialized(),
            };

            vk.GetImageMemoryRequirements2KHR(device.internal_object(), &infos, &mut output);
            debug_assert!(output.memoryRequirements.memoryTypeBits != 0);

            let mut out = MemoryRequirements::from_vulkan_reqs(output.memoryRequirements);
            if let Some(output2) = output2 {
                debug_assert_eq!(output2.requiresDedicatedAllocation, 0);
                out.prefer_dedicated = output2.prefersDedicatedAllocation != 0;
            }
            out

        } else {
            let mut output: vk::MemoryRequirements = mem::uninitialized();
            vk.GetImageMemoryRequirements(device.internal_object(), image, &mut output);
            debug_assert!(output.memoryTypeBits != 0);
            MemoryRequirements::from_vulkan_reqs(output)
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

        Ok((image, mem_reqs))
    }

    /// Creates an image from a raw handle. The image won't be destroyed.
    ///
    /// This function is for example used at the swapchain's initialization.
    pub unsafe fn from_raw(device: Arc<Device>, handle: u64, usage: u32, format: Format,
                           dimensions: ImageDimensions, samples: u32, mipmaps: u32)
                           -> UnsafeImage {
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
            needs_destruction: false, // TODO: pass as parameter
        }
    }

    pub unsafe fn bind_memory(&self, memory: &DeviceMemory, offset: usize) -> Result<(), OomError> {
        let vk = self.device.pointers();

        // We check for correctness in debug mode.
        debug_assert!({
                          let mut mem_reqs = mem::uninitialized();
                          vk.GetImageMemoryRequirements(self.device.internal_object(),
                                                        self.image,
                                                        &mut mem_reqs);
                          mem_reqs.size <= (memory.size() - offset) as u64 &&
                              (offset as u64 % mem_reqs.alignment) == 0 &&
                              mem_reqs.memoryTypeBits & (1 << memory.memory_type().id()) != 0
                      });

        check_errors(vk.BindImageMemory(self.device.internal_object(),
                                        self.image,
                                        memory.internal_object(),
                                        offset as vk::DeviceSize))?;
        Ok(())
    }

    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
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
    pub fn dimensions(&self) -> ImageDimensions {
        self.dimensions
    }

    #[inline]
    pub fn samples(&self) -> u32 {
        self.samples
    }

    /// Returns a key unique to each `UnsafeImage`. Can be used for the `conflicts_key` method.
    #[inline]
    pub fn key(&self) -> u64 {
        self.image
    }

    /// Queries the layout of an image in memory. Only valid for images with linear tiling.
    ///
    /// This function is only valid for images with a color format. See the other similar functions
    /// for the other aspects.
    ///
    /// The layout is invariant for each image. However it is not cached, as this would waste
    /// memory in the case of non-linear-tiling images. You are encouraged to store the layout
    /// somewhere in order to avoid calling this semi-expensive function at every single memory
    /// access.
    ///
    /// Note that while Vulkan allows querying the array layers other than 0, it is redundant as
    /// you can easily calculate the position of any layer.
    ///
    /// # Panic
    ///
    /// - Panics if the mipmap level is out of range.
    ///
    /// # Safety
    ///
    /// - The image must *not* have a depth, stencil or depth-stencil format.
    /// - The image must have been created with linear tiling.
    ///
    #[inline]
    pub unsafe fn color_linear_layout(&self, mip_level: u32) -> LinearLayout {
        self.linear_layout_impl(mip_level, vk::IMAGE_ASPECT_COLOR_BIT)
    }

    /// Same as `color_linear_layout`, except that it retreives the depth component of the image.
    ///
    /// # Panic
    ///
    /// - Panics if the mipmap level is out of range.
    ///
    /// # Safety
    ///
    /// - The image must have a depth or depth-stencil format.
    /// - The image must have been created with linear tiling.
    ///
    #[inline]
    pub unsafe fn depth_linear_layout(&self, mip_level: u32) -> LinearLayout {
        self.linear_layout_impl(mip_level, vk::IMAGE_ASPECT_DEPTH_BIT)
    }

    /// Same as `color_linear_layout`, except that it retreives the stencil component of the image.
    ///
    /// # Panic
    ///
    /// - Panics if the mipmap level is out of range.
    ///
    /// # Safety
    ///
    /// - The image must have a stencil or depth-stencil format.
    /// - The image must have been created with linear tiling.
    ///
    #[inline]
    pub unsafe fn stencil_linear_layout(&self, mip_level: u32) -> LinearLayout {
        self.linear_layout_impl(mip_level, vk::IMAGE_ASPECT_STENCIL_BIT)
    }

    // Implementation of the `*_layout` functions.
    unsafe fn linear_layout_impl(&self, mip_level: u32, aspect: u32) -> LinearLayout {
        let vk = self.device.pointers();

        assert!(mip_level < self.mipmaps);

        let subresource = vk::ImageSubresource {
            aspectMask: aspect,
            mipLevel: mip_level,
            arrayLayer: 0,
        };

        let mut out = mem::uninitialized();
        vk.GetImageSubresourceLayout(self.device.internal_object(),
                                     self.image,
                                     &subresource,
                                     &mut out);

        LinearLayout {
            offset: out.offset as usize,
            size: out.size as usize,
            row_pitch: out.rowPitch as usize,
            array_pitch: out.arrayPitch as usize,
            depth_pitch: out.depthPitch as usize,
        }
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

    /// Returns true if the image can be sampled with a linear filtering.
    #[inline]
    pub fn supports_linear_filtering(&self) -> bool {
        (self.format_features & vk::FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0
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

unsafe impl VulkanObject for UnsafeImage {
    type Object = vk::Image;

    #[inline]
    fn internal_object(&self) -> vk::Image {
        self.image
    }
}

impl fmt::Debug for UnsafeImage {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan image {:?}>", self.image)
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
    /// Allocating memory failed.
    AllocError(DeviceMemoryAllocError),
    /// A wrong number of mipmaps was provided.
    InvalidMipmapsCount {
        obtained: u32,
        valid_range: Range<u32>,
    },
    /// The requeted number of samples is not supported, or is 0.
    UnsupportedSamplesCount { obtained: u32 },
    /// The dimensions are too large, or one of the dimensions is 0.
    UnsupportedDimensions { dimensions: ImageDimensions },
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
            ImageCreationError::AllocError(_) => "allocating memory failed",
            ImageCreationError::InvalidMipmapsCount { .. } =>
                "a wrong number of mipmaps was provided",
            ImageCreationError::UnsupportedSamplesCount { .. } =>
                "the requeted number of samples is not supported, or is 0",
            ImageCreationError::UnsupportedDimensions { .. } =>
                "the dimensions are too large, or one of the dimensions is 0",
            ImageCreationError::FormatNotSupported =>
                "the requested format is not supported by the Vulkan implementation",
            ImageCreationError::UnsupportedUsage =>
                "the format is supported, but at least one of the requested usages is not \
                 supported",
            ImageCreationError::ShaderStorageImageMultisampleFeatureNotEnabled => {
                "the `shader_storage_image_multisample` feature must be enabled to create such \
                 an image"
            },
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            ImageCreationError::AllocError(ref err) => Some(err),
            _ => None,
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
        ImageCreationError::AllocError(DeviceMemoryAllocError::OomError(err))
    }
}

impl From<DeviceMemoryAllocError> for ImageCreationError {
    #[inline]
    fn from(err: DeviceMemoryAllocError) -> ImageCreationError {
        ImageCreationError::AllocError(err)
    }
}

impl From<Error> for ImageCreationError {
    #[inline]
    fn from(err: Error) -> ImageCreationError {
        match err {
            err @ Error::OutOfHostMemory => ImageCreationError::AllocError(err.into()),
            err @ Error::OutOfDeviceMemory => ImageCreationError::AllocError(err.into()),
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

/// Describes the memory layout of an image with linear tiling.
///
/// Obtained by calling `*_linear_layout` on the image.
///
/// The address of a texel at `(x, y, z, layer)` is `layer * array_pitch + z * depth_pitch +
/// y * row_pitch + x * size_of_each_texel + offset`. `size_of_each_texel` must be determined
/// depending on the format. The same formula applies for compressed formats, except that the
/// coordinates must be in number of blocks.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LinearLayout {
    /// Number of bytes from the start of the memory and the start of the queried subresource.
    pub offset: usize,
    /// Total number of bytes for the queried subresource. Can be used for a safety check.
    pub size: usize,
    /// Number of bytes between two texels or two blocks in adjacent rows.
    pub row_pitch: usize,
    /// Number of bytes between two texels or two blocks in adjacent array layers. This value is
    /// undefined for images with only one array layer.
    pub array_pitch: usize,
    /// Number of bytes between two texels or two blocks in adjacent depth layers. This value is
    /// undefined for images that are not three-dimensional.
    pub depth_pitch: usize,
}

pub struct UnsafeImageView {
    view: vk::ImageView,
    device: Arc<Device>,
    usage: vk::ImageUsageFlagBits,
    identity_swizzle: bool,
    format: Format,
}

impl UnsafeImageView {
    /// See the docs of new().
    pub unsafe fn raw(image: &UnsafeImage, ty: ViewType, mipmap_levels: Range<u32>,
                      array_layers: Range<u32>)
                      -> Result<UnsafeImageView, OomError> {
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

        let view_type = match (image.dimensions(), ty, array_layers.end - array_layers.start) {
            (ImageDimensions::Dim1d { .. }, ViewType::Dim1d, 1) => vk::IMAGE_VIEW_TYPE_1D,
            (ImageDimensions::Dim1d { .. }, ViewType::Dim1dArray, _) =>
                vk::IMAGE_VIEW_TYPE_1D_ARRAY,
            (ImageDimensions::Dim2d { .. }, ViewType::Dim2d, 1) => vk::IMAGE_VIEW_TYPE_2D,
            (ImageDimensions::Dim2d { .. }, ViewType::Dim2dArray, _) =>
                vk::IMAGE_VIEW_TYPE_2D_ARRAY,
            (ImageDimensions::Dim2d { cubemap_compatible, .. }, ViewType::Cubemap, n)
                if cubemap_compatible => {
                assert_eq!(n, 6);
                vk::IMAGE_VIEW_TYPE_CUBE
            },
            (ImageDimensions::Dim2d { cubemap_compatible, .. }, ViewType::CubemapArray, n)
                if cubemap_compatible => {
                assert_eq!(n % 6, 0);
                vk::IMAGE_VIEW_TYPE_CUBE_ARRAY
            },
            (ImageDimensions::Dim3d { .. }, ViewType::Dim3d, _) => vk::IMAGE_VIEW_TYPE_3D,
            _ => panic!(),
        };

        let view = {
            let infos = vk::ImageViewCreateInfo {
                sType: vk::STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0, // reserved
                image: image.internal_object(),
                viewType: view_type,
                format: image.format as u32,
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

            let mut output = mem::uninitialized();
            check_errors(vk.CreateImageView(image.device.internal_object(),
                                            &infos,
                                            ptr::null(),
                                            &mut output))?;
            output
        };

        Ok(UnsafeImageView {
               view: view,
               device: image.device.clone(),
               usage: image.usage,
               identity_swizzle: true, // FIXME:
               format: image.format,
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
    ///
    #[inline]
    pub unsafe fn new(image: &UnsafeImage, ty: ViewType, mipmap_levels: Range<u32>,
                      array_layers: Range<u32>)
                      -> UnsafeImageView {
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

#[cfg(test)]
mod tests {
    use std::iter::Empty;
    use std::u32;

    use super::ImageCreationError;
    use super::ImageUsage;
    use super::UnsafeImage;

    use format::Format;
    use image::ImageDimensions;
    use sync::Sharing;

    #[test]
    fn create_sampled() {
        let (device, _) = gfx_dev_and_queue!();

        let usage = ImageUsage {
            sampled: true,
            ..ImageUsage::none()
        };

        let (_img, _) = unsafe {
            UnsafeImage::new(device,
                             usage,
                             Format::R8G8B8A8Unorm,
                             ImageDimensions::Dim2d {
                                 width: 32,
                                 height: 32,
                                 array_layers: 1,
                                 cubemap_compatible: false,
                             },
                             1,
                             1,
                             Sharing::Exclusive::<Empty<_>>,
                             false,
                             false)
        }.unwrap();
    }

    #[test]
    fn create_transient() {
        let (device, _) = gfx_dev_and_queue!();

        let usage = ImageUsage {
            transient_attachment: true,
            color_attachment: true,
            ..ImageUsage::none()
        };

        let (_img, _) = unsafe {
            UnsafeImage::new(device,
                             usage,
                             Format::R8G8B8A8Unorm,
                             ImageDimensions::Dim2d {
                                 width: 32,
                                 height: 32,
                                 array_layers: 1,
                                 cubemap_compatible: false,
                             },
                             1,
                             1,
                             Sharing::Exclusive::<Empty<_>>,
                             false,
                             false)
        }.unwrap();
    }

    #[test]
    fn zero_sample() {
        let (device, _) = gfx_dev_and_queue!();

        let usage = ImageUsage {
            sampled: true,
            ..ImageUsage::none()
        };

        let res = unsafe {
            UnsafeImage::new(device,
                             usage,
                             Format::R8G8B8A8Unorm,
                             ImageDimensions::Dim2d {
                                 width: 32,
                                 height: 32,
                                 array_layers: 1,
                                 cubemap_compatible: false,
                             },
                             0,
                             1,
                             Sharing::Exclusive::<Empty<_>>,
                             false,
                             false)
        };

        match res {
            Err(ImageCreationError::UnsupportedSamplesCount { .. }) => (),
            _ => panic!(),
        };
    }

    #[test]
    fn non_po2_sample() {
        let (device, _) = gfx_dev_and_queue!();

        let usage = ImageUsage {
            sampled: true,
            ..ImageUsage::none()
        };

        let res = unsafe {
            UnsafeImage::new(device,
                             usage,
                             Format::R8G8B8A8Unorm,
                             ImageDimensions::Dim2d {
                                 width: 32,
                                 height: 32,
                                 array_layers: 1,
                                 cubemap_compatible: false,
                             },
                             5,
                             1,
                             Sharing::Exclusive::<Empty<_>>,
                             false,
                             false)
        };

        match res {
            Err(ImageCreationError::UnsupportedSamplesCount { .. }) => (),
            _ => panic!(),
        };
    }

    #[test]
    fn zero_mipmap() {
        let (device, _) = gfx_dev_and_queue!();

        let usage = ImageUsage {
            sampled: true,
            ..ImageUsage::none()
        };

        let res = unsafe {
            UnsafeImage::new(device,
                             usage,
                             Format::R8G8B8A8Unorm,
                             ImageDimensions::Dim2d {
                                 width: 32,
                                 height: 32,
                                 array_layers: 1,
                                 cubemap_compatible: false,
                             },
                             1,
                             0,
                             Sharing::Exclusive::<Empty<_>>,
                             false,
                             false)
        };

        match res {
            Err(ImageCreationError::InvalidMipmapsCount { .. }) => (),
            _ => panic!(),
        };
    }

    #[test]
    #[ignore] // TODO: AMD card seems to support a u32::MAX number of mipmaps
    fn mipmaps_too_high() {
        let (device, _) = gfx_dev_and_queue!();

        let usage = ImageUsage {
            sampled: true,
            ..ImageUsage::none()
        };

        let res = unsafe {
            UnsafeImage::new(device,
                             usage,
                             Format::R8G8B8A8Unorm,
                             ImageDimensions::Dim2d {
                                 width: 32,
                                 height: 32,
                                 array_layers: 1,
                                 cubemap_compatible: false,
                             },
                             1,
                             u32::MAX,
                             Sharing::Exclusive::<Empty<_>>,
                             false,
                             false)
        };

        match res {
            Err(ImageCreationError::InvalidMipmapsCount {
                    obtained,
                    valid_range,
                }) => {
                assert_eq!(obtained, u32::MAX);
                assert_eq!(valid_range.start, 1);
            },
            _ => panic!(),
        };
    }

    #[test]
    fn shader_storage_image_multisample() {
        let (device, _) = gfx_dev_and_queue!();

        let usage = ImageUsage {
            storage: true,
            ..ImageUsage::none()
        };

        let res = unsafe {
            UnsafeImage::new(device,
                             usage,
                             Format::R8G8B8A8Unorm,
                             ImageDimensions::Dim2d {
                                 width: 32,
                                 height: 32,
                                 array_layers: 1,
                                 cubemap_compatible: false,
                             },
                             2,
                             1,
                             Sharing::Exclusive::<Empty<_>>,
                             false,
                             false)
        };

        match res {
            Err(ImageCreationError::ShaderStorageImageMultisampleFeatureNotEnabled) => (),
            Err(ImageCreationError::UnsupportedSamplesCount { .. }) => (), // unlikely but possible
            _ => panic!(),
        };
    }

    #[test]
    fn compressed_not_color_attachment() {
        let (device, _) = gfx_dev_and_queue!();

        let usage = ImageUsage {
            color_attachment: true,
            ..ImageUsage::none()
        };

        let res = unsafe {
            UnsafeImage::new(device,
                             usage,
                             Format::ASTC_5x4UnormBlock,
                             ImageDimensions::Dim2d {
                                 width: 32,
                                 height: 32,
                                 array_layers: 1,
                                 cubemap_compatible: false,
                             },
                             1,
                             u32::MAX,
                             Sharing::Exclusive::<Empty<_>>,
                             false,
                             false)
        };

        match res {
            Err(ImageCreationError::FormatNotSupported) => (),
            Err(ImageCreationError::UnsupportedUsage) => (),
            _ => panic!(),
        };
    }

    #[test]
    fn transient_forbidden_with_some_usages() {
        let (device, _) = gfx_dev_and_queue!();

        let usage = ImageUsage {
            transient_attachment: true,
            sampled: true,
            ..ImageUsage::none()
        };

        let res = unsafe {
            UnsafeImage::new(device,
                             usage,
                             Format::R8G8B8A8Unorm,
                             ImageDimensions::Dim2d {
                                 width: 32,
                                 height: 32,
                                 array_layers: 1,
                                 cubemap_compatible: false,
                             },
                             1,
                             1,
                             Sharing::Exclusive::<Empty<_>>,
                             false,
                             false)
        };

        match res {
            Err(ImageCreationError::UnsupportedUsage) => (),
            _ => panic!(),
        };
    }

    #[test]
    fn cubecompatible_dims_mismatch() {
        let (device, _) = gfx_dev_and_queue!();

        let usage = ImageUsage {
            sampled: true,
            ..ImageUsage::none()
        };

        let res = unsafe {
            UnsafeImage::new(device,
                             usage,
                             Format::R8G8B8A8Unorm,
                             ImageDimensions::Dim2d {
                                 width: 32,
                                 height: 64,
                                 array_layers: 1,
                                 cubemap_compatible: true,
                             },
                             1,
                             1,
                             Sharing::Exclusive::<Empty<_>>,
                             false,
                             false)
        };

        match res {
            Err(ImageCreationError::UnsupportedDimensions { .. }) => (),
            _ => panic!(),
        };
    }
}
