// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Low-level implementation of images.
//!
//! This module contains low-level wrappers around the Vulkan image types. All
//! other image types of this library, and all custom image types
//! that you create must wrap around the types in this module.

use crate::check_errors;
use crate::device::Device;
use crate::format::CompressionType;
use crate::format::Format;
use crate::format::FormatFeatures;
use crate::format::NumericType;
use crate::image::ImageAspect;
use crate::image::ImageCreateFlags;
use crate::image::ImageDimensions;
use crate::image::ImageUsage;
use crate::image::MipmapsCount;
use crate::image::SampleCount;
use crate::memory::DeviceMemory;
use crate::memory::DeviceMemoryAllocError;
use crate::memory::MemoryRequirements;
use crate::sync::Sharing;
use crate::DeviceSize;
use crate::Error;
use crate::OomError;
use crate::Version;
use crate::VulkanObject;
use ash::vk::Handle;
use smallvec::SmallVec;
use std::error;
use std::fmt;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem::MaybeUninit;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;

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
/// - The image layout must be manually enforced and transitioned.
///
pub struct UnsafeImage {
    image: ash::vk::Image,
    device: Arc<Device>,
    usage: ImageUsage,
    format: Format,
    flags: ImageCreateFlags,

    dimensions: ImageDimensions,
    samples: SampleCount,
    mipmaps: u32,

    // Features that are supported for this particular format.
    format_features: FormatFeatures,

    // `vkDestroyImage` is called only if `needs_destruction` is true.
    needs_destruction: bool,
    preinitialized_layout: bool,
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
    pub unsafe fn new<'a, Mi, I>(
        device: Arc<Device>,
        usage: ImageUsage,
        format: Format,
        flags: ImageCreateFlags,
        dimensions: ImageDimensions,
        num_samples: SampleCount,
        mipmaps: Mi,
        sharing: Sharing<I>,
        linear_tiling: bool,
        preinitialized_layout: bool,
    ) -> Result<(UnsafeImage, MemoryRequirements), ImageCreationError>
    where
        Mi: Into<MipmapsCount>,
        I: IntoIterator<Item = u32>,
    {
        let sharing = match sharing {
            Sharing::Exclusive => (ash::vk::SharingMode::EXCLUSIVE, SmallVec::<[u32; 8]>::new()),
            Sharing::Concurrent(ids) => {
                (ash::vk::SharingMode::CONCURRENT, ids.into_iter().collect())
            }
        };

        UnsafeImage::new_impl(
            device,
            usage,
            format,
            flags,
            dimensions,
            num_samples,
            mipmaps.into(),
            sharing,
            linear_tiling,
            preinitialized_layout,
            None,
        )
    }

    #[cfg(any(
        target_os = "linux",
        target_os = "dragonflybsd",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "android"
    ))]
    #[inline]
    pub unsafe fn new_with_exportable_fd<'a, Mi, I>(
        device: Arc<Device>,
        usage: ImageUsage,
        format: Format,
        flags: ImageCreateFlags,
        dimensions: ImageDimensions,
        num_samples: SampleCount,
        mipmaps: Mi,
        sharing: Sharing<I>,
        linear_tiling: bool,
        preinitialized_layout: bool,
    ) -> Result<(UnsafeImage, MemoryRequirements), ImageCreationError>
    where
        Mi: Into<MipmapsCount>,
        I: IntoIterator<Item = u32>,
    {
        let sharing = match sharing {
            Sharing::Exclusive => (ash::vk::SharingMode::EXCLUSIVE, SmallVec::<[u32; 8]>::new()),
            Sharing::Concurrent(ids) => {
                (ash::vk::SharingMode::CONCURRENT, ids.into_iter().collect())
            }
        };

        UnsafeImage::new_impl(
            device,
            usage,
            format,
            flags,
            dimensions,
            num_samples,
            mipmaps.into(),
            sharing,
            linear_tiling,
            preinitialized_layout,
            Some(crate::memory::ExternalMemoryHandleType::posix()),
        )
    }

    // Non-templated version to avoid inlining and improve compile times.
    unsafe fn new_impl(
        device: Arc<Device>,
        usage: ImageUsage,
        format: Format,
        flags: ImageCreateFlags,
        dimensions: ImageDimensions,
        num_samples: SampleCount,
        mipmaps: MipmapsCount,
        (sh_mode, sh_indices): (ash::vk::SharingMode, SmallVec<[u32; 8]>),
        linear_tiling: bool,
        preinitialized_layout: bool,
        external_mem_handle_type: Option<crate::memory::ExternalMemoryHandleType>,
    ) -> Result<(UnsafeImage, MemoryRequirements), ImageCreationError> {
        // TODO: doesn't check that the proper features are enabled

        if flags.sparse_binding || flags.sparse_residency || flags.sparse_aliased {
            unimplemented!();
        }

        let fns = device.fns();
        let fns_i = device.instance().fns();

        // Checking if image usage conforms to what is supported.
        let format_features = {
            let format_properties = format.properties(device.physical_device());

            let features = if linear_tiling {
                format_properties.linear_tiling_features
            } else {
                format_properties.optimal_tiling_features
            };

            if features == FormatFeatures::default() {
                return Err(ImageCreationError::FormatNotSupported);
            }

            if usage.sampled && !features.sampled_image {
                return Err(ImageCreationError::UnsupportedUsage);
            }
            if usage.storage && !features.storage_image {
                return Err(ImageCreationError::UnsupportedUsage);
            }
            if usage.color_attachment && !features.color_attachment {
                return Err(ImageCreationError::UnsupportedUsage);
            }
            if usage.depth_stencil_attachment && !features.depth_stencil_attachment {
                return Err(ImageCreationError::UnsupportedUsage);
            }
            if usage.input_attachment
                && !(features.color_attachment || features.depth_stencil_attachment)
            {
                return Err(ImageCreationError::UnsupportedUsage);
            }
            if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance1
            {
                if usage.transfer_source && !features.transfer_src {
                    return Err(ImageCreationError::UnsupportedUsage);
                }
                if usage.transfer_destination && !features.transfer_dst {
                    return Err(ImageCreationError::UnsupportedUsage);
                }
            }

            features
        };

        //  VUID-VkImageCreateInfo-usage-requiredbitmask: usage must not be 0
        if usage == ImageUsage::none() {
            return Err(ImageCreationError::UnsupportedUsage);
        }

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
                        valid_range: 1..max_mipmaps + 1,
                    });
                } else if num > max_mipmaps {
                    capabilities_error = Some(ImageCreationError::InvalidMipmapsCount {
                        obtained: num,
                        valid_range: 1..max_mipmaps + 1,
                    });
                }

                num
            }
            MipmapsCount::Log2 => dimensions.max_mipmaps(),
            MipmapsCount::One => 1,
        };

        // Checking whether the number of samples is supported.
        let mut supported_samples = ash::vk::SampleCountFlags::from_raw(0x7f); // all bits up to VK_SAMPLE_COUNT_64_BIT

        if usage.sampled {
            let aspects = format.aspects();

            if format.requires_sampler_ycbcr_conversion() {
                supported_samples &= ash::vk::SampleCountFlags::TYPE_1;
            } else if let Some(numeric_type) = format.type_color() {
                match numeric_type {
                    NumericType::UINT | NumericType::SINT => {
                        supported_samples &= device
                            .physical_device()
                            .properties()
                            .sampled_image_integer_sample_counts
                            .into();
                    }
                    NumericType::SFLOAT
                    | NumericType::UFLOAT
                    | NumericType::SNORM
                    | NumericType::UNORM
                    | NumericType::SSCALED
                    | NumericType::USCALED
                    | NumericType::SRGB => {
                        supported_samples &= device
                            .physical_device()
                            .properties()
                            .sampled_image_color_sample_counts
                            .into();
                    }
                }
            } else {
                if aspects.depth {
                    supported_samples &= device
                        .physical_device()
                        .properties()
                        .sampled_image_depth_sample_counts
                        .into();
                }

                if aspects.stencil {
                    supported_samples &= device
                        .physical_device()
                        .properties()
                        .sampled_image_stencil_sample_counts
                        .into();
                }
            }

            if usage.storage {
                supported_samples &= device
                    .physical_device()
                    .properties()
                    .storage_image_sample_counts
                    .into();
            }

            if usage.color_attachment
                || usage.depth_stencil_attachment
                || usage.input_attachment
                || usage.transient_attachment
            {
                if format.requires_sampler_ycbcr_conversion() {
                    /*
                     * It's generally not possible to use a Ycbcr image as a framebuffer color
                     * attachment.
                     */
                    return Err(ImageCreationError::UnsupportedUsage);
                } else if format.type_color().is_some() {
                    supported_samples &= device
                        .physical_device()
                        .properties()
                        .framebuffer_color_sample_counts
                        .into();
                } else {
                    if aspects.depth || aspects.stencil {
                        if aspects.depth {
                            supported_samples &= device
                                .physical_device()
                                .properties()
                                .framebuffer_depth_sample_counts
                                .into();
                        }

                        if aspects.stencil {
                            supported_samples &= device
                                .physical_device()
                                .properties()
                                .framebuffer_stencil_sample_counts
                                .into();
                        }
                    }
                }
            }

            if (ash::vk::SampleCountFlags::from(num_samples) & supported_samples).is_empty() {
                let err = ImageCreationError::UnsupportedSamplesCount {
                    obtained: num_samples,
                };
                capabilities_error = Some(err);
            }
        }

        // If the `shaderStorageImageMultisample` feature is not enabled and we have
        // `usage_storage` set to true, then the number of samples must be 1.
        if usage.storage && num_samples as u32 > 1 {
            if !device.enabled_features().shader_storage_image_multisample {
                return Err(ImageCreationError::ShaderStorageImageMultisampleFeatureNotEnabled);
            }
        }

        // Decoding the dimensions.
        let (ty, extent, array_layers) = match dimensions {
            ImageDimensions::Dim1d {
                width,
                array_layers,
            } => {
                if width == 0 || array_layers == 0 {
                    return Err(ImageCreationError::UnsupportedDimensions { dimensions });
                }
                let extent = ash::vk::Extent3D {
                    width,
                    height: 1,
                    depth: 1,
                };
                (ash::vk::ImageType::TYPE_1D, extent, array_layers)
            }
            ImageDimensions::Dim2d {
                width,
                height,
                array_layers,
            } => {
                if width == 0 || height == 0 || array_layers == 0 {
                    return Err(ImageCreationError::UnsupportedDimensions { dimensions });
                }
                let extent = ash::vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                };
                (ash::vk::ImageType::TYPE_2D, extent, array_layers)
            }
            ImageDimensions::Dim3d {
                width,
                height,
                depth,
            } => {
                if width == 0 || height == 0 || depth == 0 {
                    return Err(ImageCreationError::UnsupportedDimensions { dimensions });
                }
                let extent = ash::vk::Extent3D {
                    width,
                    height,
                    depth,
                };
                (ash::vk::ImageType::TYPE_3D, extent, 1)
            }
        };

        // Checking flags requirements.
        if flags.cube_compatible {
            if !(ty == ash::vk::ImageType::TYPE_2D
                && extent.width == extent.height
                && array_layers >= 6)
            {
                return Err(ImageCreationError::CreationFlagRequirementsNotMet);
            }
        }

        if flags.array_2d_compatible {
            if !(ty == ash::vk::ImageType::TYPE_3D) {
                return Err(ImageCreationError::CreationFlagRequirementsNotMet);
            }
        }

        if flags.block_texel_view_compatible {
            if !flags.mutable_format {
                return Err(ImageCreationError::CreationFlagRequirementsNotMet);
            }

            if !matches!(
                format.compression(),
                Some(
                    CompressionType::BC1
                        | CompressionType::BC2
                        | CompressionType::BC3
                        | CompressionType::BC4
                        | CompressionType::BC5
                        | CompressionType::BC6H
                        | CompressionType::BC7
                        | CompressionType::ETC2
                        | CompressionType::ASTC
                )
            ) {
                return Err(ImageCreationError::CreationFlagRequirementsNotMet);
            }
        }

        // Checking the dimensions against the limits.
        if array_layers > device.physical_device().properties().max_image_array_layers {
            let err = ImageCreationError::UnsupportedDimensions { dimensions };
            capabilities_error = Some(err);
        }

        match ty {
            ash::vk::ImageType::TYPE_1D => {
                if extent.width > device.physical_device().properties().max_image_dimension1_d {
                    let err = ImageCreationError::UnsupportedDimensions { dimensions };
                    capabilities_error = Some(err);
                }
            }
            ash::vk::ImageType::TYPE_2D => {
                let limit = device.physical_device().properties().max_image_dimension2_d;
                if extent.width > limit || extent.height > limit {
                    let err = ImageCreationError::UnsupportedDimensions { dimensions };
                    capabilities_error = Some(err);
                }

                if flags.cube_compatible {
                    let limit = device
                        .physical_device()
                        .properties()
                        .max_image_dimension_cube;
                    if extent.width > limit {
                        let err = ImageCreationError::UnsupportedDimensions { dimensions };
                        capabilities_error = Some(err);
                    }
                }
            }
            ash::vk::ImageType::TYPE_3D => {
                let limit = device.physical_device().properties().max_image_dimension3_d;
                if extent.width > limit || extent.height > limit || extent.depth > limit {
                    let err = ImageCreationError::UnsupportedDimensions { dimensions };
                    capabilities_error = Some(err);
                }
            }
            _ => unreachable!(),
        };

        let usage_bits = usage.into();

        // Now that all checks have been performed, if any of the check failed we query the Vulkan
        // implementation for additional image capabilities.
        if let Some(capabilities_error) = capabilities_error {
            let tiling = if linear_tiling {
                ash::vk::ImageTiling::LINEAR
            } else {
                ash::vk::ImageTiling::OPTIMAL
            };

            let mut output = MaybeUninit::uninit();
            let physical_device = device.physical_device().internal_object();
            let r = fns_i.v1_0.get_physical_device_image_format_properties(
                physical_device,
                format.into(),
                ty,
                tiling,
                usage_bits,
                ash::vk::ImageCreateFlags::empty(), /* TODO */
                output.as_mut_ptr(),
            );

            match check_errors(r) {
                Ok(_) => (),
                Err(Error::FormatNotSupported) => {
                    return Err(ImageCreationError::FormatNotSupported)
                }
                Err(err) => return Err(err.into()),
            }

            let output = output.assume_init();

            if extent.width > output.max_extent.width
                || extent.height > output.max_extent.height
                || extent.depth > output.max_extent.depth
                || mipmaps > output.max_mip_levels
                || array_layers > output.max_array_layers
                || (ash::vk::SampleCountFlags::from(num_samples) & output.sample_counts).is_empty()
            {
                return Err(capabilities_error);
            }
        }

        // Everything now ok. Creating the image.

        let image = {
            let mut infos_builder = ash::vk::ImageCreateInfo::builder()
                .flags(flags.into())
                .image_type(ty)
                .format(format.into())
                .extent(extent)
                .mip_levels(mipmaps)
                .array_layers(array_layers)
                .samples(num_samples.into())
                .tiling({
                    if linear_tiling {
                        ash::vk::ImageTiling::LINEAR
                    } else {
                        ash::vk::ImageTiling::OPTIMAL
                    }
                })
                .usage(usage_bits)
                .sharing_mode(sh_mode)
                .queue_family_indices(&sh_indices)
                .initial_layout({
                    if preinitialized_layout {
                        ash::vk::ImageLayout::PREINITIALIZED
                    } else {
                        ash::vk::ImageLayout::UNDEFINED
                    }
                });

            let mut ext_image_create_info;
            if let Some(mem_types) = external_mem_handle_type {
                ext_image_create_info = ash::vk::ExternalMemoryImageCreateInfo {
                    handle_types: mem_types.into(),
                    ..Default::default()
                };

                infos_builder = infos_builder.push_next(&mut ext_image_create_info);
            }

            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_image(
                device.internal_object(),
                &infos_builder.build(),
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        let mem_reqs = if device.api_version() >= Version::V1_1
            || device.enabled_extensions().khr_get_memory_requirements2
        {
            let infos = ash::vk::ImageMemoryRequirementsInfo2 {
                image,
                ..Default::default()
            };

            let mut output2 = if device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_dedicated_allocation
            {
                Some(ash::vk::MemoryDedicatedRequirements::default())
            } else {
                None
            };

            let mut output = ash::vk::MemoryRequirements2 {
                p_next: output2
                    .as_mut()
                    .map(|o| o as *mut _)
                    .unwrap_or(ptr::null_mut()) as *mut _,
                ..Default::default()
            };

            if device.api_version() >= Version::V1_1 {
                fns.v1_1.get_image_memory_requirements2(
                    device.internal_object(),
                    &infos,
                    &mut output,
                );
            } else {
                fns.khr_get_memory_requirements2
                    .get_image_memory_requirements2_khr(
                        device.internal_object(),
                        &infos,
                        &mut output,
                    );
            }

            debug_assert!(output.memory_requirements.memory_type_bits != 0);

            let mut out = MemoryRequirements::from(output.memory_requirements);
            if let Some(output2) = output2 {
                out.prefer_dedicated = output2.prefers_dedicated_allocation != 0;
            }
            out
        } else {
            let mut output: MaybeUninit<ash::vk::MemoryRequirements> = MaybeUninit::uninit();
            fns.v1_0.get_image_memory_requirements(
                device.internal_object(),
                image,
                output.as_mut_ptr(),
            );
            let output = output.assume_init();
            debug_assert!(output.memory_type_bits != 0);
            MemoryRequirements::from(output)
        };

        let image = UnsafeImage {
            device: device.clone(),
            image,
            usage,
            format,
            flags,
            dimensions,
            samples: num_samples,
            mipmaps,
            format_features,
            needs_destruction: true,
            preinitialized_layout,
        };

        Ok((image, mem_reqs))
    }

    /// Creates an image from a raw handle. The image won't be destroyed.
    ///
    /// This function is for example used at the swapchain's initialization.
    pub unsafe fn from_raw(
        device: Arc<Device>,
        handle: ash::vk::Image,
        usage: ImageUsage,
        format: Format,
        flags: ImageCreateFlags,
        dimensions: ImageDimensions,
        samples: SampleCount,
        mipmaps: u32,
    ) -> UnsafeImage {
        let format_properties = format.properties(device.physical_device());

        // TODO: check that usage is correct in regard to `output`?

        UnsafeImage {
            device: device.clone(),
            image: handle,
            usage,
            format,
            flags,
            dimensions,
            samples,
            mipmaps,
            format_features: format_properties.optimal_tiling_features,
            needs_destruction: false,     // TODO: pass as parameter
            preinitialized_layout: false, // TODO: Maybe this should be passed in?
        }
    }

    pub unsafe fn bind_memory(
        &self,
        memory: &DeviceMemory,
        offset: DeviceSize,
    ) -> Result<(), OomError> {
        let fns = self.device.fns();

        // We check for correctness in debug mode.
        debug_assert!({
            let mut mem_reqs = MaybeUninit::uninit();
            fns.v1_0.get_image_memory_requirements(
                self.device.internal_object(),
                self.image,
                mem_reqs.as_mut_ptr(),
            );

            let mem_reqs = mem_reqs.assume_init();
            mem_reqs.size <= memory.size() - offset
                && offset % mem_reqs.alignment == 0
                && mem_reqs.memory_type_bits & (1 << memory.memory_type().id()) != 0
        });

        check_errors(fns.v1_0.bind_image_memory(
            self.device.internal_object(),
            self.image,
            memory.internal_object(),
            offset,
        ))?;
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

    pub fn create_flags(&self) -> ImageCreateFlags {
        self.flags
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
    pub fn samples(&self) -> SampleCount {
        self.samples
    }

    /// Returns a key unique to each `UnsafeImage`. Can be used for the `conflicts_key` method.
    #[inline]
    pub fn key(&self) -> u64 {
        self.image.as_raw()
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
        self.linear_layout_impl(mip_level, ImageAspect::Color)
    }

    /// Same as `color_linear_layout`, except that it retrieves the depth component of the image.
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
        self.linear_layout_impl(mip_level, ImageAspect::Depth)
    }

    /// Same as `color_linear_layout`, except that it retrieves the stencil component of the image.
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
        self.linear_layout_impl(mip_level, ImageAspect::Stencil)
    }

    /// Same as `color_linear_layout`, except that it retrieves layout for the requested YCbCr
    /// component too if the format is a YCbCr format.
    ///
    /// # Panic
    ///
    /// - Panics if plane aspect is out of range.
    /// - Panics if the aspect is not a color or planar aspect.
    /// - Panics if the number of mipmaps is not 1.
    #[inline]
    pub unsafe fn multiplane_color_layout(&self, aspect: ImageAspect) -> LinearLayout {
        // This function only supports color and planar aspects currently.
        assert!(matches!(
            aspect,
            ImageAspect::Color | ImageAspect::Plane0 | ImageAspect::Plane1 | ImageAspect::Plane2
        ));
        assert!(self.mipmaps == 1);

        if matches!(
            aspect,
            ImageAspect::Plane0 | ImageAspect::Plane1 | ImageAspect::Plane2
        ) {
            debug_assert!(self.format.requires_sampler_ycbcr_conversion());
        }

        self.linear_layout_impl(0, aspect)
    }

    // Implementation of the `*_layout` functions.
    unsafe fn linear_layout_impl(&self, mip_level: u32, aspect: ImageAspect) -> LinearLayout {
        let fns = self.device.fns();

        assert!(mip_level < self.mipmaps);

        let subresource = ash::vk::ImageSubresource {
            aspect_mask: ash::vk::ImageAspectFlags::from(aspect),
            mip_level: mip_level,
            array_layer: 0,
        };

        let mut out = MaybeUninit::uninit();
        fns.v1_0.get_image_subresource_layout(
            self.device.internal_object(),
            self.image,
            &subresource,
            out.as_mut_ptr(),
        );

        let out = out.assume_init();
        LinearLayout {
            offset: out.offset,
            size: out.size,
            row_pitch: out.row_pitch,
            array_pitch: out.array_pitch,
            depth_pitch: out.depth_pitch,
        }
    }

    /// Returns the flags the image was created with.
    #[inline]
    pub fn flags(&self) -> ImageCreateFlags {
        self.flags
    }

    /// Returns the features supported by the image's format.
    #[inline]
    pub fn format_features(&self) -> FormatFeatures {
        self.format_features
    }

    /// Returns the usage the image was created with.
    #[inline]
    pub fn usage(&self) -> ImageUsage {
        self.usage
    }

    #[inline]
    pub fn preinitialized_layout(&self) -> bool {
        self.preinitialized_layout
    }
}

unsafe impl VulkanObject for UnsafeImage {
    type Object = ash::vk::Image;

    #[inline]
    fn internal_object(&self) -> ash::vk::Image {
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
            let fns = self.device.fns();
            fns.v1_0
                .destroy_image(self.device.internal_object(), self.image, ptr::null());
        }
    }
}

impl PartialEq for UnsafeImage {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.image == other.image && self.device == other.device
    }
}

impl Eq for UnsafeImage {}

impl Hash for UnsafeImage {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.image.hash(state);
        self.device.hash(state);
    }
}

/// Error that can happen when creating an instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageCreationError {
    /// Allocating memory failed.
    AllocError(DeviceMemoryAllocError),
    /// The specified creation flags have requirements (e.g. specific dimension) that were not met.
    CreationFlagRequirementsNotMet,
    /// The format is supported, but at least one of the requested usages is not supported.
    FormatNotSupported,
    /// A wrong number of mipmaps was provided.
    InvalidMipmapsCount {
        obtained: u32,
        valid_range: Range<u32>,
    },
    /// The requested number of samples is not supported, or is 0.
    UnsupportedSamplesCount { obtained: SampleCount },
    /// The dimensions are too large, or one of the dimensions is 0.
    UnsupportedDimensions { dimensions: ImageDimensions },
    /// The requested format is not supported by the Vulkan implementation.
    UnsupportedUsage,
    /// The `shader_storage_image_multisample` feature must be enabled to create such an image.
    ShaderStorageImageMultisampleFeatureNotEnabled,
}

impl error::Error for ImageCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            ImageCreationError::AllocError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for ImageCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                ImageCreationError::AllocError(_) => "allocating memory failed",
                ImageCreationError::CreationFlagRequirementsNotMet => {
                    "the requested creation flags have additional requirements that were not met"
                }
                ImageCreationError::FormatNotSupported => {
                    "the requested format is not supported by the Vulkan implementation"
                }
                ImageCreationError::InvalidMipmapsCount { .. } => {
                    "a wrong number of mipmaps was provided"
                }
                ImageCreationError::UnsupportedSamplesCount { .. } => {
                    "the requested number of samples is not supported, or is 0"
                }
                ImageCreationError::UnsupportedDimensions { .. } => {
                    "the dimensions are too large, or one of the dimensions is 0"
                }
                ImageCreationError::UnsupportedUsage => {
                    "the format is supported, but at least one of the requested usages is not \
                 supported"
                }
                ImageCreationError::ShaderStorageImageMultisampleFeatureNotEnabled => {
                    "the `shader_storage_image_multisample` feature must be enabled to create such \
                 an image"
                }
            }
        )
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
    pub offset: DeviceSize,
    /// Total number of bytes for the queried subresource. Can be used for a safety check.
    pub size: DeviceSize,
    /// Number of bytes between two texels or two blocks in adjacent rows.
    pub row_pitch: DeviceSize,
    /// Number of bytes between two texels or two blocks in adjacent array layers. This value is
    /// undefined for images with only one array layer.
    pub array_pitch: DeviceSize,
    /// Number of bytes between two texels or two blocks in adjacent depth layers. This value is
    /// undefined for images that are not three-dimensional.
    pub depth_pitch: DeviceSize,
}

#[cfg(test)]
mod tests {
    use super::ImageCreateFlags;
    use super::ImageCreationError;
    use super::ImageUsage;
    use super::UnsafeImage;
    use crate::format::Format;
    use crate::image::ImageDimensions;
    use crate::image::SampleCount;
    use crate::sync::Sharing;
    use std::iter::Empty;
    use std::u32;

    #[test]
    fn create_sampled() {
        let (device, _) = gfx_dev_and_queue!();

        let usage = ImageUsage {
            sampled: true,
            ..ImageUsage::none()
        };

        let (_img, _) = unsafe {
            UnsafeImage::new(
                device,
                usage,
                Format::R8G8B8A8_UNORM,
                ImageCreateFlags::none(),
                ImageDimensions::Dim2d {
                    width: 32,
                    height: 32,
                    array_layers: 1,
                },
                SampleCount::Sample1,
                1,
                Sharing::Exclusive::<Empty<_>>,
                false,
                false,
            )
        }
        .unwrap();
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
            UnsafeImage::new(
                device,
                usage,
                Format::R8G8B8A8_UNORM,
                ImageCreateFlags::none(),
                ImageDimensions::Dim2d {
                    width: 32,
                    height: 32,
                    array_layers: 1,
                },
                SampleCount::Sample1,
                1,
                Sharing::Exclusive::<Empty<_>>,
                false,
                false,
            )
        }
        .unwrap();
    }

    #[test]
    fn zero_mipmap() {
        let (device, _) = gfx_dev_and_queue!();

        let usage = ImageUsage {
            sampled: true,
            ..ImageUsage::none()
        };

        let res = unsafe {
            UnsafeImage::new(
                device,
                usage,
                Format::R8G8B8A8_UNORM,
                ImageCreateFlags::none(),
                ImageDimensions::Dim2d {
                    width: 32,
                    height: 32,
                    array_layers: 1,
                },
                SampleCount::Sample1,
                0,
                Sharing::Exclusive::<Empty<_>>,
                false,
                false,
            )
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
            UnsafeImage::new(
                device,
                usage,
                Format::R8G8B8A8_UNORM,
                ImageCreateFlags::none(),
                ImageDimensions::Dim2d {
                    width: 32,
                    height: 32,
                    array_layers: 1,
                },
                SampleCount::Sample1,
                u32::MAX,
                Sharing::Exclusive::<Empty<_>>,
                false,
                false,
            )
        };

        match res {
            Err(ImageCreationError::InvalidMipmapsCount {
                obtained,
                valid_range,
            }) => {
                assert_eq!(obtained, u32::MAX);
                assert_eq!(valid_range.start, 1);
            }
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
            UnsafeImage::new(
                device,
                usage,
                Format::R8G8B8A8_UNORM,
                ImageCreateFlags::none(),
                ImageDimensions::Dim2d {
                    width: 32,
                    height: 32,
                    array_layers: 1,
                },
                SampleCount::Sample2,
                1,
                Sharing::Exclusive::<Empty<_>>,
                false,
                false,
            )
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
            UnsafeImage::new(
                device,
                usage,
                Format::ASTC_5x4_UNORM_BLOCK,
                ImageCreateFlags::none(),
                ImageDimensions::Dim2d {
                    width: 32,
                    height: 32,
                    array_layers: 1,
                },
                SampleCount::Sample1,
                u32::MAX,
                Sharing::Exclusive::<Empty<_>>,
                false,
                false,
            )
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
            UnsafeImage::new(
                device,
                usage,
                Format::R8G8B8A8_UNORM,
                ImageCreateFlags::none(),
                ImageDimensions::Dim2d {
                    width: 32,
                    height: 32,
                    array_layers: 1,
                },
                SampleCount::Sample1,
                1,
                Sharing::Exclusive::<Empty<_>>,
                false,
                false,
            )
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
            UnsafeImage::new(
                device,
                usage,
                Format::R8G8B8A8_UNORM,
                ImageCreateFlags {
                    cube_compatible: true,
                    ..ImageCreateFlags::none()
                },
                ImageDimensions::Dim2d {
                    width: 32,
                    height: 64,
                    array_layers: 1,
                },
                SampleCount::Sample1,
                1,
                Sharing::Exclusive::<Empty<_>>,
                false,
                false,
            )
        };

        match res {
            Err(ImageCreationError::CreationFlagRequirementsNotMet) => (),
            _ => panic!(),
        };
    }
}
