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
//!
//! See also [the parent module-level documentation] for more information about images.
//!
//! [the parent module-level documentation]: super

use super::{
    Image, ImageAspect, ImageAspects, ImageCreateFlags, ImageLayout, ImageMemory,
    ImageSubresourceLayers, ImageSubresourceRange, ImageTiling, ImageUsage, SampleCount,
    SparseImageMemoryRequirements, SubresourceLayout,
};
use crate::{
    cache::OnceCache,
    device::{Device, DeviceOwned},
    format::{ChromaSampling, Format, FormatFeatures},
    image::{
        max_mip_levels, ImageDrmFormatModifierInfo, ImageFormatInfo, ImageFormatProperties,
        ImageType, SparseImageFormatProperties,
    },
    instance::InstanceOwnedDebugWrapper,
    macros::impl_id_counter,
    memory::{
        allocator::{AllocationType, DeviceLayout, MemoryAlloc},
        is_aligned, DedicatedTo, ExternalMemoryHandleTypes, MemoryPropertyFlags,
        MemoryRequirements,
    },
    sync::Sharing,
    Requires, RequiresAllOf, RequiresOneOf, RuntimeError, ValidationError, Version, VulkanError,
    VulkanObject,
};
use smallvec::{smallvec, SmallVec};
use std::{mem::MaybeUninit, num::NonZeroU64, ptr, sync::Arc};

/// A raw image, with no memory backing it.
///
/// This is the basic image type, a direct translation of a `VkImage` object, but it is mostly
/// useless in this form. After creating a raw image, you must call `bind_memory` to make a
/// complete image object.
///
/// See also [the parent module-level documentation] for more information about images.
///
/// [the parent module-level documentation]: super
#[derive(Debug)]
pub struct RawImage {
    handle: ash::vk::Image,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZeroU64,

    flags: ImageCreateFlags,
    image_type: ImageType,
    format: Option<Format>,
    format_features: FormatFeatures,
    extent: [u32; 3],
    array_layers: u32,
    mip_levels: u32,
    samples: SampleCount,
    tiling: ImageTiling,
    usage: ImageUsage,
    stencil_usage: ImageUsage,
    sharing: Sharing<SmallVec<[u32; 4]>>,
    initial_layout: ImageLayout,
    drm_format_modifier: Option<(u64, u32)>,
    external_memory_handle_types: ExternalMemoryHandleTypes,

    memory_requirements: SmallVec<[MemoryRequirements; 4]>,
    needs_destruction: bool, // `vkDestroyImage` is called only if true.
    subresource_layout: OnceCache<(ImageAspect, u32, u32), SubresourceLayout>,
}

impl RawImage {
    /// Creates a new `RawImage`.
    #[inline]
    pub fn new(device: Arc<Device>, create_info: ImageCreateInfo) -> Result<RawImage, VulkanError> {
        Self::validate_new(&device, &create_info)?;

        unsafe { Ok(RawImage::new_unchecked(device, create_info)?) }
    }

    fn validate_new(device: &Device, create_info: &ImageCreateInfo) -> Result<(), ValidationError> {
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        create_info: ImageCreateInfo,
    ) -> Result<Self, RuntimeError> {
        let &ImageCreateInfo {
            flags,
            image_type,
            format,
            extent,
            array_layers,
            mip_levels,
            samples,
            tiling,
            usage,
            mut stencil_usage,
            ref sharing,
            initial_layout,
            external_memory_handle_types,
            _ne: _,
            ref drm_format_modifiers,
            ref drm_format_modifier_plane_layouts,
        } = &create_info;

        let aspects = format.map_or_else(Default::default, |format| format.aspects());

        let has_separate_stencil_usage = if stencil_usage.is_empty()
            || !aspects.contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
        {
            stencil_usage = usage;
            false
        } else {
            stencil_usage == usage
        };

        let (sharing_mode, queue_family_index_count, p_queue_family_indices) = match sharing {
            Sharing::Exclusive => (ash::vk::SharingMode::EXCLUSIVE, 0, &[] as _),
            Sharing::Concurrent(queue_family_indices) => (
                ash::vk::SharingMode::CONCURRENT,
                queue_family_indices.len() as u32,
                queue_family_indices.as_ptr(),
            ),
        };

        let mut info_vk = ash::vk::ImageCreateInfo {
            flags: flags.into(),
            image_type: image_type.into(),
            format: format.map(Into::into).unwrap_or_default(),
            extent: ash::vk::Extent3D {
                width: extent[0],
                height: extent[1],
                depth: extent[2],
            },
            mip_levels,
            array_layers,
            samples: samples.into(),
            tiling: tiling.into(),
            usage: usage.into(),
            sharing_mode,
            queue_family_index_count,
            p_queue_family_indices,
            initial_layout: initial_layout.into(),
            ..Default::default()
        };
        let mut drm_format_modifier_explicit_info_vk = None;
        let drm_format_modifier_plane_layouts_vk: SmallVec<[_; 4]>;
        let mut drm_format_modifier_list_info_vk = None;
        let mut external_memory_info_vk = None;
        let mut stencil_usage_info_vk = None;

        #[allow(clippy::comparison_chain)]
        if drm_format_modifiers.len() == 1 {
            drm_format_modifier_plane_layouts_vk = drm_format_modifier_plane_layouts
                .iter()
                .map(|subresource_layout| {
                    let &SubresourceLayout {
                        offset,
                        size,
                        row_pitch,
                        array_pitch,
                        depth_pitch,
                    } = subresource_layout;

                    ash::vk::SubresourceLayout {
                        offset,
                        size,
                        row_pitch,
                        array_pitch: array_pitch.unwrap_or(0),
                        depth_pitch: depth_pitch.unwrap_or(0),
                    }
                })
                .collect();

            let next = drm_format_modifier_explicit_info_vk.insert(
                ash::vk::ImageDrmFormatModifierExplicitCreateInfoEXT {
                    drm_format_modifier: drm_format_modifiers[0],
                    drm_format_modifier_plane_count: drm_format_modifier_plane_layouts_vk.len()
                        as u32,
                    p_plane_layouts: drm_format_modifier_plane_layouts_vk.as_ptr(),
                    ..Default::default()
                },
            );

            next.p_next = info_vk.p_next;
            info_vk.p_next = next as *const _ as *const _;
        } else if drm_format_modifiers.len() > 1 {
            let next = drm_format_modifier_list_info_vk.insert(
                ash::vk::ImageDrmFormatModifierListCreateInfoEXT {
                    drm_format_modifier_count: drm_format_modifiers.len() as u32,
                    p_drm_format_modifiers: drm_format_modifiers.as_ptr(),
                    ..Default::default()
                },
            );

            next.p_next = info_vk.p_next;
            info_vk.p_next = next as *const _ as *const _;
        }

        if !external_memory_handle_types.is_empty() {
            let next = external_memory_info_vk.insert(ash::vk::ExternalMemoryImageCreateInfo {
                handle_types: external_memory_handle_types.into(),
                ..Default::default()
            });

            next.p_next = info_vk.p_next;
            info_vk.p_next = next as *const _ as *const _;
        }

        if has_separate_stencil_usage {
            let next = stencil_usage_info_vk.insert(ash::vk::ImageStencilUsageCreateInfo {
                stencil_usage: stencil_usage.into(),
                ..Default::default()
            });

            next.p_next = info_vk.p_next;
            info_vk.p_next = next as *const _ as *const _;
        }

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_image)(device.handle(), &info_vk, ptr::null(), output.as_mut_ptr())
                .result()
                .map_err(RuntimeError::from)?;
            output.assume_init()
        };

        Self::from_handle(device, handle, create_info)
    }

    /// Creates a new `RawImage` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `handle` must refer to an image that has not yet had memory bound to it.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::Image,
        create_info: ImageCreateInfo,
    ) -> Result<Self, RuntimeError> {
        Self::from_handle_with_destruction(device, handle, create_info, true)
    }

    pub(super) unsafe fn from_handle_with_destruction(
        device: Arc<Device>,
        handle: ash::vk::Image,
        create_info: ImageCreateInfo,
        needs_destruction: bool,
    ) -> Result<Self, RuntimeError> {
        let ImageCreateInfo {
            flags,
            image_type,
            format,
            extent,
            array_layers,
            mip_levels,
            samples,
            tiling,
            usage,
            mut stencil_usage,
            sharing,
            initial_layout,
            external_memory_handle_types,
            _ne: _,
            drm_format_modifiers: _,
            drm_format_modifier_plane_layouts: _,
        } = create_info;

        let format = format.unwrap();
        let format_properties =
            unsafe { device.physical_device().format_properties_unchecked(format) };

        let drm_format_modifier = if tiling == ImageTiling::DrmFormatModifier {
            let drm_format_modifier = Self::get_drm_format_modifier_properties(&device, handle)?;
            let drm_format_modifier_plane_count = format_properties
                .drm_format_modifier_properties
                .iter()
                .find(|properties| properties.drm_format_modifier == drm_format_modifier)
                .expect("couldn't get the DRM format modifier plane count for the image")
                .drm_format_modifier_plane_count;
            Some((drm_format_modifier, drm_format_modifier_plane_count))
        } else {
            None
        };

        let format_features = {
            let drm_format_modifiers: SmallVec<[_; 1]> =
                drm_format_modifier.map_or_else(Default::default, |(m, _)| smallvec![m]);
            format_properties.format_features(tiling, &drm_format_modifiers)
        };

        let format_aspects = format.aspects();

        if stencil_usage.is_empty()
            || !format_aspects.contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
        {
            stencil_usage = usage;
        }

        let memory_requirements = if needs_destruction {
            if flags.intersects(ImageCreateFlags::DISJOINT) {
                // VUID-VkImageMemoryRequirementsInfo2-image-01589
                // VUID-VkImageMemoryRequirementsInfo2-image-02279
                let plane_count = drm_format_modifier.map_or_else(
                    || format.planes().len(),
                    |(_, plane_count)| plane_count as usize,
                );

                (0..plane_count)
                    .map(|plane| {
                        Self::get_memory_requirements(&device, handle, Some((plane, tiling)))
                    })
                    .collect()
            } else {
                // VUID-VkImageMemoryRequirementsInfo2-image-01590
                smallvec![Self::get_memory_requirements(&device, handle, None)]
            }
        } else {
            smallvec![]
        };

        Ok(RawImage {
            handle,
            device: InstanceOwnedDebugWrapper(device),
            id: Self::next_id(),

            flags,
            image_type,
            format: Some(format),
            format_features,
            extent,
            array_layers,
            mip_levels,
            initial_layout,
            samples,
            tiling,
            usage,
            stencil_usage,
            sharing,
            drm_format_modifier,
            external_memory_handle_types,

            memory_requirements,
            needs_destruction,
            subresource_layout: OnceCache::new(),
        })
    }

    unsafe fn get_memory_requirements(
        device: &Device,
        handle: ash::vk::Image,
        plane_tiling: Option<(usize, ImageTiling)>,
    ) -> MemoryRequirements {
        let mut info_vk = ash::vk::ImageMemoryRequirementsInfo2 {
            image: handle,
            ..Default::default()
        };
        let mut plane_info_vk = None;

        if let Some((plane, tiling)) = plane_tiling {
            debug_assert!(
                device.api_version() >= Version::V1_1
                    || device.enabled_extensions().khr_get_memory_requirements2
            );

            let plane_aspect = match tiling {
                // VUID-VkImagePlaneMemoryRequirementsInfo-planeAspect-02281
                ImageTiling::Optimal | ImageTiling::Linear => {
                    debug_assert!(
                        device.api_version() >= Version::V1_1
                            || device.enabled_extensions().khr_sampler_ycbcr_conversion
                    );
                    match plane {
                        0 => ash::vk::ImageAspectFlags::PLANE_0,
                        1 => ash::vk::ImageAspectFlags::PLANE_1,
                        2 => ash::vk::ImageAspectFlags::PLANE_2,
                        _ => unreachable!(),
                    }
                }
                // VUID-VkImagePlaneMemoryRequirementsInfo-planeAspect-02282
                ImageTiling::DrmFormatModifier => {
                    debug_assert!(device.enabled_extensions().ext_image_drm_format_modifier);
                    match plane {
                        0 => ash::vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
                        1 => ash::vk::ImageAspectFlags::MEMORY_PLANE_1_EXT,
                        2 => ash::vk::ImageAspectFlags::MEMORY_PLANE_2_EXT,
                        3 => ash::vk::ImageAspectFlags::MEMORY_PLANE_3_EXT,
                        _ => unreachable!(),
                    }
                }
            };

            let next = plane_info_vk.insert(ash::vk::ImagePlaneMemoryRequirementsInfo {
                plane_aspect,
                ..Default::default()
            });

            next.p_next = info_vk.p_next;
            info_vk.p_next = next as *mut _ as *mut _;
        }

        let mut memory_requirements2_vk = ash::vk::MemoryRequirements2::default();
        let mut memory_dedicated_requirements_vk = None;

        if device.api_version() >= Version::V1_1
            || device.enabled_extensions().khr_dedicated_allocation
        {
            debug_assert!(
                device.api_version() >= Version::V1_1
                    || device.enabled_extensions().khr_get_memory_requirements2
            );

            let next = memory_dedicated_requirements_vk
                .insert(ash::vk::MemoryDedicatedRequirements::default());

            next.p_next = memory_requirements2_vk.p_next;
            memory_requirements2_vk.p_next = next as *mut _ as *mut _;
        }

        unsafe {
            let fns = device.fns();

            if device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_get_memory_requirements2
            {
                if device.api_version() >= Version::V1_1 {
                    (fns.v1_1.get_image_memory_requirements2)(
                        device.handle(),
                        &info_vk,
                        &mut memory_requirements2_vk,
                    );
                } else {
                    (fns.khr_get_memory_requirements2
                        .get_image_memory_requirements2_khr)(
                        device.handle(),
                        &info_vk,
                        &mut memory_requirements2_vk,
                    );
                }
            } else {
                (fns.v1_0.get_image_memory_requirements)(
                    device.handle(),
                    handle,
                    &mut memory_requirements2_vk.memory_requirements,
                );
            }
        }

        MemoryRequirements {
            layout: DeviceLayout::from_size_alignment(
                memory_requirements2_vk.memory_requirements.size,
                memory_requirements2_vk.memory_requirements.alignment,
            )
            .unwrap(),
            memory_type_bits: memory_requirements2_vk.memory_requirements.memory_type_bits,
            prefers_dedicated_allocation: memory_dedicated_requirements_vk
                .map_or(false, |dreqs| dreqs.prefers_dedicated_allocation != 0),
            requires_dedicated_allocation: memory_dedicated_requirements_vk
                .map_or(false, |dreqs| dreqs.requires_dedicated_allocation != 0),
        }
    }

    #[allow(dead_code)] // Remove when sparse memory is implemented
    fn get_sparse_memory_requirements(&self) -> Vec<SparseImageMemoryRequirements> {
        let device = &self.device;

        unsafe {
            let fns = self.device.fns();

            if device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_get_memory_requirements2
            {
                let info2 = ash::vk::ImageSparseMemoryRequirementsInfo2 {
                    image: self.handle,
                    ..Default::default()
                };

                let mut count = 0;

                if device.api_version() >= Version::V1_1 {
                    (fns.v1_1.get_image_sparse_memory_requirements2)(
                        device.handle(),
                        &info2,
                        &mut count,
                        ptr::null_mut(),
                    );
                } else {
                    (fns.khr_get_memory_requirements2
                        .get_image_sparse_memory_requirements2_khr)(
                        device.handle(),
                        &info2,
                        &mut count,
                        ptr::null_mut(),
                    );
                }

                let mut sparse_image_memory_requirements2 =
                    vec![ash::vk::SparseImageMemoryRequirements2::default(); count as usize];

                if device.api_version() >= Version::V1_1 {
                    (fns.v1_1.get_image_sparse_memory_requirements2)(
                        self.device.handle(),
                        &info2,
                        &mut count,
                        sparse_image_memory_requirements2.as_mut_ptr(),
                    );
                } else {
                    (fns.khr_get_memory_requirements2
                        .get_image_sparse_memory_requirements2_khr)(
                        self.device.handle(),
                        &info2,
                        &mut count,
                        sparse_image_memory_requirements2.as_mut_ptr(),
                    );
                }

                sparse_image_memory_requirements2.set_len(count as usize);

                sparse_image_memory_requirements2
                    .into_iter()
                    .map(
                        |sparse_image_memory_requirements2| SparseImageMemoryRequirements {
                            format_properties: SparseImageFormatProperties {
                                aspects: sparse_image_memory_requirements2
                                    .memory_requirements
                                    .format_properties
                                    .aspect_mask
                                    .into(),
                                image_granularity: [
                                    sparse_image_memory_requirements2
                                        .memory_requirements
                                        .format_properties
                                        .image_granularity
                                        .width,
                                    sparse_image_memory_requirements2
                                        .memory_requirements
                                        .format_properties
                                        .image_granularity
                                        .height,
                                    sparse_image_memory_requirements2
                                        .memory_requirements
                                        .format_properties
                                        .image_granularity
                                        .depth,
                                ],
                                flags: sparse_image_memory_requirements2
                                    .memory_requirements
                                    .format_properties
                                    .flags
                                    .into(),
                            },
                            image_mip_tail_first_lod: sparse_image_memory_requirements2
                                .memory_requirements
                                .image_mip_tail_first_lod,
                            image_mip_tail_size: sparse_image_memory_requirements2
                                .memory_requirements
                                .image_mip_tail_size,
                            image_mip_tail_offset: sparse_image_memory_requirements2
                                .memory_requirements
                                .image_mip_tail_offset,
                            image_mip_tail_stride: (!sparse_image_memory_requirements2
                                .memory_requirements
                                .format_properties
                                .flags
                                .intersects(ash::vk::SparseImageFormatFlags::SINGLE_MIPTAIL))
                            .then_some(
                                sparse_image_memory_requirements2
                                    .memory_requirements
                                    .image_mip_tail_stride,
                            ),
                        },
                    )
                    .collect()
            } else {
                let mut count = 0;

                (fns.v1_0.get_image_sparse_memory_requirements)(
                    device.handle(),
                    self.handle,
                    &mut count,
                    ptr::null_mut(),
                );

                let mut sparse_image_memory_requirements =
                    vec![ash::vk::SparseImageMemoryRequirements::default(); count as usize];

                (fns.v1_0.get_image_sparse_memory_requirements)(
                    device.handle(),
                    self.handle,
                    &mut count,
                    sparse_image_memory_requirements.as_mut_ptr(),
                );

                sparse_image_memory_requirements.set_len(count as usize);

                sparse_image_memory_requirements
                    .into_iter()
                    .map(
                        |sparse_image_memory_requirements| SparseImageMemoryRequirements {
                            format_properties: SparseImageFormatProperties {
                                aspects: sparse_image_memory_requirements
                                    .format_properties
                                    .aspect_mask
                                    .into(),
                                image_granularity: [
                                    sparse_image_memory_requirements
                                        .format_properties
                                        .image_granularity
                                        .width,
                                    sparse_image_memory_requirements
                                        .format_properties
                                        .image_granularity
                                        .height,
                                    sparse_image_memory_requirements
                                        .format_properties
                                        .image_granularity
                                        .depth,
                                ],
                                flags: sparse_image_memory_requirements
                                    .format_properties
                                    .flags
                                    .into(),
                            },
                            image_mip_tail_first_lod: sparse_image_memory_requirements
                                .image_mip_tail_first_lod,
                            image_mip_tail_size: sparse_image_memory_requirements
                                .image_mip_tail_size,
                            image_mip_tail_offset: sparse_image_memory_requirements
                                .image_mip_tail_offset,
                            image_mip_tail_stride: (!sparse_image_memory_requirements
                                .format_properties
                                .flags
                                .intersects(ash::vk::SparseImageFormatFlags::SINGLE_MIPTAIL))
                            .then_some(sparse_image_memory_requirements.image_mip_tail_stride),
                        },
                    )
                    .collect()
            }
        }
    }

    unsafe fn get_drm_format_modifier_properties(
        device: &Device,
        handle: ash::vk::Image,
    ) -> Result<u64, RuntimeError> {
        let mut properties_vk = ash::vk::ImageDrmFormatModifierPropertiesEXT::default();

        let fns = device.fns();
        (fns.ext_image_drm_format_modifier
            .get_image_drm_format_modifier_properties_ext)(
            device.handle(),
            handle,
            &mut properties_vk,
        )
        .result()
        .map_err(RuntimeError::from)?;

        Ok(properties_vk.drm_format_modifier)
    }

    /// Binds device memory to this image.
    ///
    /// - If `self.flags()` does not contain `ImageCreateFlags::DISJOINT`,
    ///   then `allocations` must contain exactly one element.
    /// - If `self.flags()` contains `ImageCreateFlags::DISJOINT`, and
    ///   `self.tiling()` is `ImageTiling::Linear` or `ImageTiling::Optimal`, then
    ///   `allocations` must contain exactly `self.format().unwrap().planes().len()` elements.
    /// - If `self.flags()` contains `ImageCreateFlags::DISJOINT`, and
    ///   `self.tiling()` is `ImageTiling::DrmFormatModifier`, then
    ///   `allocations` must contain exactly `self.drm_format_modifier().unwrap().1` elements.
    pub fn bind_memory(
        self,
        allocations: impl IntoIterator<Item = MemoryAlloc>,
    ) -> Result<
        Image,
        (
            VulkanError,
            RawImage,
            impl ExactSizeIterator<Item = MemoryAlloc>,
        ),
    > {
        let allocations: SmallVec<[_; 4]> = allocations.into_iter().collect();

        if let Err(err) = self.validate_bind_memory(&allocations) {
            return Err((err.into(), self, allocations.into_iter()));
        }

        unsafe { self.bind_memory_unchecked(allocations) }.map_err(|(err, image, allocations)| {
            (
                err.into(),
                image,
                allocations
                    .into_iter()
                    .collect::<SmallVec<[_; 4]>>()
                    .into_iter(),
            )
        })
    }

    fn validate_bind_memory(&self, allocations: &[MemoryAlloc]) -> Result<(), ValidationError> {
        if self.flags.intersects(ImageCreateFlags::DISJOINT) {
            match self.tiling {
                ImageTiling::Optimal | ImageTiling::Linear => {
                    if allocations.len() != self.format.unwrap().planes().len() {
                        return Err(ValidationError {
                            problem: "`self.flags()` contains `ImageCreateFlags::DISJOINT`, and \
                                `self.tiling()` is `ImageTiling::Optimal` or \
                                `ImageTiling::Linear`, but \
                                the length of `allocations` does not equal \
                                the number of planes in the format of the image"
                                .into(),
                            ..Default::default()
                        });
                    }
                }
                ImageTiling::DrmFormatModifier => {
                    if allocations.len() != self.drm_format_modifier.unwrap().1 as usize {
                        return Err(ValidationError {
                            problem: "`self.flags()` contains `ImageCreateFlags::DISJOINT`, and \
                                `self.tiling()` is `ImageTiling::DrmFormatModifier`, but \
                                the length of `allocations` does not equal \
                                the number of memory planes of the DRM format modifier of the \
                                image"
                                .into(),
                            ..Default::default()
                        });
                    }
                }
            }
        } else {
            if allocations.len() != 1 {
                return Err(ValidationError {
                    problem: "`self.flags()` does not contain `ImageCreateFlags::DISJOINT`, but \
                        the length of `allocations` is not 1"
                        .into(),
                    ..Default::default()
                });
            }
        }

        for (index, (allocation, memory_requirements)) in (allocations.iter())
            .zip(self.memory_requirements.iter())
            .enumerate()
        {
            if allocation.allocation_type() == AllocationType::Linear {
                return Err(ValidationError {
                    problem: format!(
                        "`allocations[{}].allocation_type()` is `AllocationType::Linear`",
                        index
                    )
                    .into(),
                    ..Default::default()
                });
            }

            let memory = allocation.device_memory();
            let memory_offset = allocation.offset();
            let memory_type = &self
                .device
                .physical_device()
                .memory_properties()
                .memory_types[memory.memory_type_index() as usize];

            // VUID-VkBindImageMemoryInfo-commonparent
            assert_eq!(self.device(), memory.device());

            // VUID-VkBindImageMemoryInfo-image-07460
            // Ensured by taking ownership of `RawImage`.

            // VUID-VkBindImageMemoryInfo-image-01045
            // Currently ensured by not having sparse binding flags, but this needs to be checked
            // once those are enabled.

            // VUID-VkBindImageMemoryInfo-memoryOffset-01046
            // Assume that `allocation` was created correctly.

            if memory_requirements.memory_type_bits & (1 << memory.memory_type_index()) == 0 {
                return Err(ValidationError {
                    problem: format!(
                        "`allocation[{}].device_memory().memory_type_index()` is not a bit set in \
                        `self.memory_requirements().memory_type_bits`",
                        index
                    )
                    .into(),
                    vuids: &[
                        "VUID-VkBindImageMemoryInfo-pNext-01615",
                        "VUID-VkBindImageMemoryInfo-pNext-01619",
                    ],
                    ..Default::default()
                });
            }

            if !is_aligned(memory_offset, memory_requirements.layout.alignment()) {
                return Err(ValidationError {
                    problem: format!(
                        "`allocations[{}].offset()` is not aligned according to \
                        `self.memory_requirements().layout.alignment()`",
                        index
                    )
                    .into(),
                    vuids: &[
                        "VUID-VkBindImageMemoryInfo-pNext-01616",
                        "VUID-VkBindImageMemoryInfo-pNext-01620",
                    ],
                    ..Default::default()
                });
            }

            if allocation.size() < memory_requirements.layout.size() {
                return Err(ValidationError {
                    problem: format!(
                        "`allocations[{}].size()` is less than \
                        `self.memory_requirements().layout.size()`",
                        index
                    )
                    .into(),
                    vuids: &[
                        "VUID-VkBindImageMemoryInfo-pNext-01617",
                        "VUID-VkBindImageMemoryInfo-pNext-01621",
                    ],
                    ..Default::default()
                });
            }

            if let Some(dedicated_to) = memory.dedicated_to() {
                match dedicated_to {
                    DedicatedTo::Image(id) if id == self.id => {}
                    _ => {
                        return Err(ValidationError {
                            problem: format!(
                                "`allocations[{}].device_memory()` is a dedicated allocation, but \
                                it is not dedicated to this image",
                                index
                            )
                            .into(),
                            vuids: &["VUID-VkBindImageMemoryInfo-memory-02628"],
                            ..Default::default()
                        });
                    }
                }
                debug_assert!(memory_offset == 0); // This should be ensured by the allocator
            } else {
                if memory_requirements.requires_dedicated_allocation {
                    return Err(ValidationError {
                        problem: format!(
                            "`self.memory_requirements().requires_dedicated_allocation` is \
                            `true`, but `allocations[{}].device_memory()` is not a \
                            dedicated allocation",
                            index
                        )
                        .into(),
                        vuids: &["VUID-VkBindImageMemoryInfo-image-01445"],
                        ..Default::default()
                    });
                }
            }

            if memory_type
                .property_flags
                .intersects(MemoryPropertyFlags::PROTECTED)
            {
                return Err(ValidationError {
                    problem: format!(
                        "the `property_flags` of the memory type of \
                        `allocations[{}].device_memory()` contains \
                        `MemoryPropertyFlags::PROTECTED`",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkBindImageMemoryInfo-None-01901"],
                    ..Default::default()
                });
            }

            if !memory.export_handle_types().is_empty()
                && !memory
                    .export_handle_types()
                    .intersects(self.external_memory_handle_types)
            {
                return Err(ValidationError {
                    problem: format!(
                        "`allocations[{}].device_memory().export_handle_types()` is not empty, \
                        but it does not share at least one memory type with \
                        `self.external_memory_handle_types()`",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkBindImageMemoryInfo-memory-02728"],
                    ..Default::default()
                });
            }

            if let Some(handle_type) = memory.imported_handle_type() {
                if !ExternalMemoryHandleTypes::from(handle_type)
                    .intersects(self.external_memory_handle_types)
                {
                    return Err(ValidationError {
                        problem: format!(
                            "`allocations[{}].device_memory()` is imported, but \
                            `self.external_memory_handle_types()` does not contain the imported \
                            handle type",
                            index
                        )
                        .into(),
                        vuids: &["VUID-VkBindImageMemoryInfo-memory-02989"],
                        ..Default::default()
                    });
                }
            }
        }

        Ok(())
    }

    /// # Safety
    ///
    /// - If `self.flags()` does not contain `ImageCreateFlags::DISJOINT`,
    ///   then `allocations` must contain exactly one element.
    /// - If `self.flags()` contains `ImageCreateFlags::DISJOINT`, and
    ///   `self.tiling()` is `ImageTiling::Linear` or `ImageTiling::Optimal`, then
    ///   `allocations` must contain exactly `self.format().unwrap().planes().len()` elements.
    /// - If `self.flags()` contains `ImageCreateFlags::DISJOINT`, and
    ///   `self.tiling()` is `ImageTiling::DrmFormatModifier`, then
    ///   `allocations` must contain exactly `self.drm_format_modifier().unwrap().1` elements.
    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_memory_unchecked(
        self,
        allocations: impl IntoIterator<Item = MemoryAlloc>,
    ) -> Result<
        Image,
        (
            RuntimeError,
            RawImage,
            impl ExactSizeIterator<Item = MemoryAlloc>,
        ),
    > {
        let allocations: SmallVec<[_; 4]> = allocations.into_iter().collect();
        let fns = self.device.fns();

        let result = if self.device.api_version() >= Version::V1_1
            || self.device.enabled_extensions().khr_bind_memory2
        {
            let mut infos_vk: SmallVec<[_; 4]> = SmallVec::with_capacity(3);
            let mut plane_infos_vk: SmallVec<[_; 4]> = SmallVec::with_capacity(3);

            if self.flags.intersects(ImageCreateFlags::DISJOINT) {
                for (plane, allocation) in allocations.iter().enumerate() {
                    let memory = allocation.device_memory();
                    let memory_offset = allocation.offset();
                    let plane_aspect = match self.tiling {
                        // VUID-VkBindImagePlaneMemoryInfo-planeAspect-02283
                        ImageTiling::Optimal | ImageTiling::Linear => {
                            debug_assert_eq!(
                                allocations.len(),
                                self.format.unwrap().planes().len()
                            );
                            match plane {
                                0 => ash::vk::ImageAspectFlags::PLANE_0,
                                1 => ash::vk::ImageAspectFlags::PLANE_1,
                                2 => ash::vk::ImageAspectFlags::PLANE_2,
                                _ => unreachable!(),
                            }
                        }
                        // VUID-VkBindImagePlaneMemoryInfo-planeAspect-02284
                        ImageTiling::DrmFormatModifier => {
                            debug_assert_eq!(
                                allocations.len(),
                                self.drm_format_modifier.unwrap().1 as usize
                            );
                            match plane {
                                0 => ash::vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
                                1 => ash::vk::ImageAspectFlags::MEMORY_PLANE_1_EXT,
                                2 => ash::vk::ImageAspectFlags::MEMORY_PLANE_2_EXT,
                                3 => ash::vk::ImageAspectFlags::MEMORY_PLANE_3_EXT,
                                _ => unreachable!(),
                            }
                        }
                    };

                    infos_vk.push(ash::vk::BindImageMemoryInfo {
                        image: self.handle,
                        memory: memory.handle(),
                        memory_offset,
                        ..Default::default()
                    });
                    // VUID-VkBindImageMemoryInfo-pNext-01618
                    plane_infos_vk.push(ash::vk::BindImagePlaneMemoryInfo {
                        plane_aspect,
                        ..Default::default()
                    });
                }
            } else {
                debug_assert_eq!(allocations.len(), 1);

                let allocation = &allocations[0];
                let memory = allocation.device_memory();
                let memory_offset = allocation.offset();

                infos_vk.push(ash::vk::BindImageMemoryInfo {
                    image: self.handle,
                    memory: memory.handle(),
                    memory_offset,
                    ..Default::default()
                });
            };

            for (info_vk, plane_info_vk) in (infos_vk.iter_mut()).zip(plane_infos_vk.iter_mut()) {
                info_vk.p_next = plane_info_vk as *mut _ as *mut _;
            }

            if self.device.api_version() >= Version::V1_1 {
                (fns.v1_1.bind_image_memory2)(
                    self.device.handle(),
                    infos_vk.len() as u32,
                    infos_vk.as_ptr(),
                )
            } else {
                (fns.khr_bind_memory2.bind_image_memory2_khr)(
                    self.device.handle(),
                    infos_vk.len() as u32,
                    infos_vk.as_ptr(),
                )
            }
        } else {
            debug_assert_eq!(allocations.len(), 1);

            let allocation = &allocations[0];
            let memory = allocation.device_memory();
            let memory_offset = allocation.offset();

            (fns.v1_0.bind_image_memory)(
                self.device.handle(),
                self.handle,
                memory.handle(),
                memory_offset,
            )
        }
        .result();

        if let Err(err) = result {
            return Err((RuntimeError::from(err), self, allocations.into_iter()));
        }

        Ok(Image::from_raw(
            self,
            ImageMemory::Normal(allocations),
            ImageLayout::General,
        ))
    }

    /// Returns the memory requirements for this image.
    ///
    /// - If the image is a swapchain image, this returns a slice with a length of 0.
    /// - If `self.flags().disjoint` is not set, this returns a slice with a length of 1.
    /// - If `self.flags().disjoint` is set, this returns a slice with a length equal to
    ///   `self.format().unwrap().planes().len()`.
    #[inline]
    pub fn memory_requirements(&self) -> &[MemoryRequirements] {
        &self.memory_requirements
    }

    /// Returns the flags the image was created with.
    #[inline]
    pub fn flags(&self) -> ImageCreateFlags {
        self.flags
    }

    /// Returns the image type of the image.
    #[inline]
    pub fn image_type(&self) -> ImageType {
        self.image_type
    }

    /// Returns the image's format.
    #[inline]
    pub fn format(&self) -> Option<Format> {
        self.format
    }

    /// Returns the features supported by the image's format.
    #[inline]
    pub fn format_features(&self) -> FormatFeatures {
        self.format_features
    }

    /// Returns the extent of the image.
    #[inline]
    pub fn extent(&self) -> [u32; 3] {
        self.extent
    }

    /// Returns the number of array layers in the image.
    #[inline]
    pub fn array_layers(&self) -> u32 {
        self.array_layers
    }

    /// Returns the number of mip levels in the image.
    #[inline]
    pub fn mip_levels(&self) -> u32 {
        self.mip_levels
    }

    /// Returns the initial layout of the image.
    #[inline]
    pub fn initial_layout(&self) -> ImageLayout {
        self.initial_layout
    }

    /// Returns the number of samples for the image.
    #[inline]
    pub fn samples(&self) -> SampleCount {
        self.samples
    }

    /// Returns the tiling of the image.
    #[inline]
    pub fn tiling(&self) -> ImageTiling {
        self.tiling
    }

    /// Returns the usage the image was created with.
    #[inline]
    pub fn usage(&self) -> ImageUsage {
        self.usage
    }

    /// Returns the stencil usage the image was created with.
    #[inline]
    pub fn stencil_usage(&self) -> ImageUsage {
        self.stencil_usage
    }

    /// Returns the sharing the image was created with.
    #[inline]
    pub fn sharing(&self) -> &Sharing<SmallVec<[u32; 4]>> {
        &self.sharing
    }

    /// If `self.tiling()` is `ImageTiling::DrmFormatModifier`, returns the DRM format modifier
    /// of the image, and the number of memory planes.
    /// This was either provided in [`ImageCreateInfo::drm_format_modifiers`], or if
    /// multiple modifiers were provided, selected from the list by the Vulkan implementation.
    #[inline]
    pub fn drm_format_modifier(&self) -> Option<(u64, u32)> {
        self.drm_format_modifier
    }

    /// Returns the external memory handle types that are supported with this image.
    #[inline]
    pub fn external_memory_handle_types(&self) -> ExternalMemoryHandleTypes {
        self.external_memory_handle_types
    }

    /// Returns an `ImageSubresourceLayers` covering the first mip level of the image. All aspects
    /// of the image are selected, or `plane0` if the image is multi-planar.
    #[inline]
    pub fn subresource_layers(&self) -> ImageSubresourceLayers {
        ImageSubresourceLayers {
            aspects: {
                let aspects = self.format.unwrap().aspects();

                if aspects.intersects(ImageAspects::PLANE_0) {
                    ImageAspects::PLANE_0
                } else {
                    aspects
                }
            },
            mip_level: 0,
            array_layers: 0..self.array_layers,
        }
    }

    /// Returns an `ImageSubresourceRange` covering the whole image. If the image is multi-planar,
    /// only the `color` aspect is selected.
    #[inline]
    pub fn subresource_range(&self) -> ImageSubresourceRange {
        ImageSubresourceRange {
            aspects: self.format.unwrap().aspects()
                - (ImageAspects::PLANE_0 | ImageAspects::PLANE_1 | ImageAspects::PLANE_2),
            mip_levels: 0..self.mip_levels,
            array_layers: 0..self.array_layers,
        }
    }

    /// Queries the memory layout of a single subresource of the image.
    ///
    /// Only images with linear tiling are supported, if they do not have a format with both a
    /// depth and a stencil format. Images with optimal tiling have an opaque image layout that is
    /// not suitable for direct memory accesses, and likewise for combined depth/stencil formats.
    /// Multi-planar formats are supported, but you must specify one of the planes as the `aspect`,
    /// not [`ImageAspect::Color`].
    ///
    /// The results of this function are cached, so that future calls with the same arguments
    /// do not need to make a call to the Vulkan API again.
    pub fn subresource_layout(
        &self,
        aspect: ImageAspect,
        mip_level: u32,
        array_layer: u32,
    ) -> Result<SubresourceLayout, ValidationError> {
        self.validate_subresource_layout(aspect, mip_level, array_layer)?;

        unsafe { Ok(self.subresource_layout_unchecked(aspect, mip_level, array_layer)) }
    }

    fn validate_subresource_layout(
        &self,
        aspect: ImageAspect,
        mip_level: u32,
        array_layer: u32,
    ) -> Result<(), ValidationError> {
        aspect
            .validate_device(&self.device)
            .map_err(|err| ValidationError {
                context: "aspect".into(),
                vuids: &["VUID-VkImageSubresource-aspectMask-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        // VUID-VkImageSubresource-aspectMask-requiredbitmask
        // VUID-vkGetImageSubresourceLayout-aspectMask-00997
        // Ensured by use of enum `ImageAspect`.

        if !matches!(
            self.tiling,
            ImageTiling::Linear | ImageTiling::DrmFormatModifier
        ) {
            return Err(ValidationError {
                context: "self.tiling()".into(),
                problem: "is not `ImageTiling::Linear` or `ImageTiling::DrmFormatModifier`".into(),
                vuids: &["VUID-vkGetImageSubresourceLayout-image-02270"],
                ..Default::default()
            });
        }

        if mip_level >= self.mip_levels {
            return Err(ValidationError {
                context: "mip_level".into(),
                problem: "is greater than the number of mip levels in the image".into(),
                vuids: &["VUID-vkGetImageSubresourceLayout-mipLevel-01716"],
                ..Default::default()
            });
        }

        if array_layer >= self.array_layers {
            return Err(ValidationError {
                context: "array_layer".into(),
                problem: "is greater than the number of array layers in the image".into(),
                vuids: &["VUID-vkGetImageSubresourceLayout-arrayLayer-01717"],
                ..Default::default()
            });
        }

        let format = self.format.unwrap();
        let format_aspects = format.aspects();

        if let Some((_, drm_format_modifier_plane_count)) = self.drm_format_modifier {
            match drm_format_modifier_plane_count {
                1 => {
                    if !matches!(aspect, ImageAspect::MemoryPlane0) {
                        return Err(ValidationError {
                            problem: "the image has a DRM format modifier with 1 memory plane, \
                                but `aspect` is not `ImageAspect::MemoryPlane0`"
                                .into(),
                            vuids: &["VUID-vkGetImageSubresourceLayout-tiling-02271"],
                            ..Default::default()
                        });
                    }
                }
                2 => {
                    if !matches!(
                        aspect,
                        ImageAspect::MemoryPlane0 | ImageAspect::MemoryPlane1
                    ) {
                        return Err(ValidationError {
                            problem: "the image has a DRM format modifier with 2 memory planes, \
                                but `aspect` is not `ImageAspect::MemoryPlane0` or \
                                `ImageAspect::MemoryPlane1`"
                                .into(),
                            vuids: &["VUID-vkGetImageSubresourceLayout-tiling-02271"],
                            ..Default::default()
                        });
                    }
                }
                3 => {
                    if !matches!(
                        aspect,
                        ImageAspect::MemoryPlane0
                            | ImageAspect::MemoryPlane1
                            | ImageAspect::MemoryPlane2
                    ) {
                        return Err(ValidationError {
                            problem: "the image has a DRM format modifier with 3 memory planes, \
                                but `aspect` is not `ImageAspect::MemoryPlane0`, \
                                `ImageAspect::MemoryPlane1` or `ImageAspect::MemoryPlane2`"
                                .into(),
                            vuids: &["VUID-vkGetImageSubresourceLayout-tiling-02271"],
                            ..Default::default()
                        });
                    }
                }
                4 => {
                    if !matches!(
                        aspect,
                        ImageAspect::MemoryPlane0
                            | ImageAspect::MemoryPlane1
                            | ImageAspect::MemoryPlane2
                            | ImageAspect::MemoryPlane3
                    ) {
                        return Err(ValidationError {
                            problem: "the image has a DRM format modifier with 4 memory planes, \
                                but `aspect` is not `ImageAspect::MemoryPlane0`, \
                                `ImageAspect::MemoryPlane1`, `ImageAspect::MemoryPlane2` or \
                                `ImageAspect::MemoryPlane3`"
                                .into(),
                            vuids: &["VUID-vkGetImageSubresourceLayout-tiling-02271"],
                            ..Default::default()
                        });
                    }
                }
                _ => unreachable!("image has more than 4 memory planes??"),
            }
        } else if format_aspects.contains(ImageAspects::DEPTH | ImageAspects::STENCIL) {
            // Follows from the combination of these three VUIDs. See:
            // https://github.com/KhronosGroup/Vulkan-Docs/issues/1942
            return Err(ValidationError {
                context: "self.format()".into(),
                problem: "has both a depth and a stencil aspect".into(),
                vuids: &[
                    "VUID-vkGetImageSubresourceLayout-aspectMask-00997",
                    "VUID-vkGetImageSubresourceLayout-format-04462",
                    "VUID-vkGetImageSubresourceLayout-format-04463",
                ],
                ..Default::default()
            });
        } else if format_aspects.intersects(ImageAspects::DEPTH) {
            if aspect != ImageAspect::Depth {
                return Err(ValidationError {
                    problem: "`self.format()` is a depth format, but \
                        `aspect` is not `ImageAspect::Depth`"
                        .into(),
                    vuids: &["VUID-vkGetImageSubresourceLayout-format-04462"],
                    ..Default::default()
                });
            }
        } else if format_aspects.intersects(ImageAspects::STENCIL) {
            if aspect != ImageAspect::Stencil {
                return Err(ValidationError {
                    problem: "`self.format()` is a stencil format, but \
                        `aspect` is not `ImageAspect::Stencil`"
                        .into(),
                    vuids: &["VUID-vkGetImageSubresourceLayout-format-04463"],
                    ..Default::default()
                });
            }
        } else if format_aspects.intersects(ImageAspects::COLOR) {
            if format.planes().is_empty() {
                if aspect != ImageAspect::Color {
                    return Err(ValidationError {
                        problem: "`self.format()` is a color format with a single plane, but \
                        `aspect` is not `ImageAspect::Color`"
                            .into(),
                        vuids: &["VUID-vkGetImageSubresourceLayout-format-08886"],
                        ..Default::default()
                    });
                }
            } else if format.planes().len() == 2 {
                if !matches!(aspect, ImageAspect::Plane0 | ImageAspect::Plane1) {
                    return Err(ValidationError {
                        problem: "`self.format()` is a color format with two planes, but \
                            `aspect` is not `ImageAspect::Plane0` or `ImageAspect::Plane1`"
                            .into(),
                        vuids: &["VUID-vkGetImageSubresourceLayout-tiling-08717"],
                        ..Default::default()
                    });
                }
            } else if format.planes().len() == 3 {
                if !matches!(
                    aspect,
                    ImageAspect::Plane0 | ImageAspect::Plane1 | ImageAspect::Plane2
                ) {
                    return Err(ValidationError {
                        problem: "`self.format()` is a color format with three planes, but \
                            `aspect` is not `ImageAspect::Plane0`, `ImageAspect::Plane1` or \
                            `ImageAspect::Plane2`"
                            .into(),
                        vuids: &["VUID-vkGetImageSubresourceLayout-tiling-08717"],
                        ..Default::default()
                    });
                }
            }
        }

        // TODO:  VUID-vkGetImageSubresourceLayout-tiling-02271
        //if self.tiling == ImageTiling::DrmFormatModifier {
        // Only one-plane image importing is possible for now.
        //}

        // VUID-vkGetImageSubresourceLayout-format-08886
        // VUID-vkGetImageSubresourceLayout-format-04462
        // VUID-vkGetImageSubresourceLayout-format-04463
        // VUID-vkGetImageSubresourceLayout-format-04464
        // VUID-vkGetImageSubresourceLayout-format-01581
        // VUID-vkGetImageSubresourceLayout-format-01582
        if !format_aspects.contains(aspect.into()) {
            return Err(ValidationError {
                context: "array_layer".into(),
                problem: "is greater than the number of array layers in the image".into(),
                vuids: &["VUID-vkGetImageSubresourceLayout-arrayLayer-01717"],
                ..Default::default()
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn subresource_layout_unchecked(
        &self,
        aspect: ImageAspect,
        mip_level: u32,
        array_layer: u32,
    ) -> SubresourceLayout {
        self.subresource_layout.get_or_insert(
            (aspect, mip_level, array_layer),
            |&(aspect, mip_level, array_layer)| {
                let fns = self.device.fns();

                let subresource = ash::vk::ImageSubresource {
                    aspect_mask: aspect.into(),
                    mip_level,
                    array_layer,
                };

                let mut output = MaybeUninit::uninit();
                (fns.v1_0.get_image_subresource_layout)(
                    self.device.handle(),
                    self.handle,
                    &subresource,
                    output.as_mut_ptr(),
                );
                let output = output.assume_init();

                SubresourceLayout {
                    offset: output.offset,
                    size: output.size,
                    row_pitch: output.row_pitch,
                    array_pitch: (self.array_layers > 1).then_some(output.array_pitch),
                    depth_pitch: matches!(self.image_type, ImageType::Dim3d)
                        .then_some(output.depth_pitch),
                }
            },
        )
    }
}

impl Drop for RawImage {
    #[inline]
    fn drop(&mut self) {
        if !self.needs_destruction {
            return;
        }

        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.destroy_image)(self.device.handle(), self.handle, ptr::null());
        }
    }
}

unsafe impl VulkanObject for RawImage {
    type Handle = ash::vk::Image;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for RawImage {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(RawImage);

/// Parameters to create a new `Image`.
#[derive(Clone, Debug)]
pub struct ImageCreateInfo {
    /// Flags to enable.
    ///
    /// The default value is [`ImageCreateFlags::empty()`].
    pub flags: ImageCreateFlags,

    /// The basic image dimensionality to create the image with.
    ///
    /// The default value is `ImageType::Dim2d`.
    pub image_type: ImageType,

    /// The format used to store the image data.
    ///
    /// The default value is `None`, which must be overridden.
    pub format: Option<Format>,

    /// The width, height and depth of the image.
    ///
    /// If `image_type` is `ImageType::Dim2d`, then the depth must be 1.
    /// If `image_type` is `ImageType::Dim1d`, then the height and depth must be 1.
    ///
    /// The default value is `[0; 3]`, which must be overridden.
    pub extent: [u32; 3],

    /// The number of array layers to create the image with.
    ///
    /// On [portability subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, if `samples` is not [`SampleCount::Sample1`] and `array_layers` is not 1,
    /// the [`multisample_array_image`](crate::device::Features::multisample_array_image)
    /// feature must be enabled on the device.
    ///
    /// The default value is `1`.
    pub array_layers: u32,

    /// The number of mip levels to create the image with.
    ///
    /// The default value is `1`.
    pub mip_levels: u32,

    /// The number of samples per texel that the image should use.
    ///
    /// On [portability subset](crate::instance#portability-subset-devices-and-the-enumerate_portability-flag)
    /// devices, if `samples` is not [`SampleCount::Sample1`] and `array_layers` is not 1,
    /// the [`multisample_array_image`](crate::device::Features::multisample_array_image)
    /// feature must be enabled on the device.
    ///
    /// The default value is [`SampleCount::Sample1`].
    pub samples: SampleCount,

    /// The memory arrangement of the texel blocks.
    ///
    /// The default value is [`ImageTiling::Optimal`].
    pub tiling: ImageTiling,

    /// How the image is going to be used.
    ///
    /// The default value is [`ImageUsage::empty()`], which must be overridden.
    pub usage: ImageUsage,

    /// How the stencil aspect of the image is going to be used, if any.
    ///
    /// If `stencil_usage` is empty or if `format` does not have both a depth and a stencil aspect,
    /// then it is automatically set to equal `usage`.
    ///
    /// If after this, `stencil_usage` does not equal `usage`,
    /// then the device API version must be at least 1.2, or the
    /// [`ext_separate_stencil_usage`](crate::device::DeviceExtensions::ext_separate_stencil_usage)
    /// extension must be enabled on the device.
    ///
    /// The default value is [`ImageUsage::empty()`].
    pub stencil_usage: ImageUsage,

    /// Whether the image can be shared across multiple queues, or is limited to a single queue.
    ///
    /// The default value is [`Sharing::Exclusive`].
    pub sharing: Sharing<SmallVec<[u32; 4]>>,

    /// The image layout that the image will have when it is created.
    ///
    /// The default value is [`ImageLayout::Undefined`].
    pub initial_layout: ImageLayout,

    /// The external memory handle types that are going to be used with the image.
    ///
    /// If any of the fields in this value are set, the device must either support API version 1.1
    /// or the [`khr_external_memory`](crate::device::DeviceExtensions::khr_external_memory)
    /// extension must be enabled, and `initial_layout` must be set to
    /// [`ImageLayout::Undefined`].
    ///
    /// The default value is [`ExternalMemoryHandleTypes::empty()`].
    pub external_memory_handle_types: ExternalMemoryHandleTypes,

    /// The Linux DRM format modifiers that the image should be created with.
    ///
    /// If this is not empty, then the
    /// [`ext_image_drm_format_modifier`](crate::device::DeviceExtensions::ext_image_drm_format_modifier)
    /// extension must be enabled on the device.
    ///
    /// The default value is empty.
    pub drm_format_modifiers: SmallVec<[u64; 1]>,

    /// If `drm_format_modifiers` contains one element, provides the subresource layouts of each
    /// memory plane of the image. The number of elements must equal
    /// [`DrmFormatModifierProperties::drm_format_modifier_plane_count`], and the following
    /// additional requirements apply to each element:
    /// - [`SubresourceLayout::size`] must always be 0.
    /// - If `array_layers` is 1, then [`SubresourceLayout::array_pitch`] must be `None`.
    /// - If `image_type` is not [`ImageType::Dim3d`] or `extent[2]` is 1, then
    ///   [`SubresourceLayout::depth_pitch`] must be `None`.
    ///
    /// If `drm_format_modifiers` does not contain one element, then
    /// this must be empty.
    ///
    /// [`DrmFormatModifierProperties::drm_format_modifier_plane_count`]: crate::format::DrmFormatModifierProperties::drm_format_modifier_plane_count
    pub drm_format_modifier_plane_layouts: SmallVec<[SubresourceLayout; 4]>,

    pub _ne: crate::NonExhaustive,
}

impl Default for ImageCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            flags: ImageCreateFlags::empty(),
            image_type: ImageType::Dim2d,
            format: None,
            extent: [0; 3],
            array_layers: 1,
            mip_levels: 1,
            samples: SampleCount::Sample1,
            tiling: ImageTiling::Optimal,
            usage: ImageUsage::empty(),
            stencil_usage: ImageUsage::empty(),
            sharing: Sharing::Exclusive,
            initial_layout: ImageLayout::Undefined,
            external_memory_handle_types: ExternalMemoryHandleTypes::empty(),
            drm_format_modifiers: SmallVec::new(),
            drm_format_modifier_plane_layouts: SmallVec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl ImageCreateInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            flags,
            image_type,
            format,
            extent,
            array_layers,
            mip_levels,
            samples,
            tiling,
            usage,
            mut stencil_usage,
            ref sharing,
            initial_layout,
            external_memory_handle_types,
            _ne: _,
            ref drm_format_modifiers,
            ref drm_format_modifier_plane_layouts,
        } = self;

        let physical_device = device.physical_device();
        let device_properties = physical_device.properties();

        flags
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "flags".into(),
                vuids: &["VUID-VkImageCreateInfo-flags-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        // Can be None for "external formats" but Vulkano doesn't support that yet
        let format = format.ok_or(ValidationError {
            context: "format".into(),
            problem: "is `None`".into(),
            ..Default::default()
        })?;

        format
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "format".into(),
                vuids: &["VUID-VkImageCreateInfo-format-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        samples
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "samples".into(),
                vuids: &["VUID-VkImageCreateInfo-samples-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        tiling
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "tiling".into(),
                vuids: &["VUID-VkImageCreateInfo-tiling-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        usage
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "usage".into(),
                vuids: &["VUID-VkImageCreateInfo-usage-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        if usage.is_empty() {
            return Err(ValidationError {
                context: "usage".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkImageCreateInfo-usage-requiredbitmask"],
                ..Default::default()
            });
        }

        let format_properties = unsafe { physical_device.format_properties_unchecked(format) };
        let image_create_drm_format_modifiers = &drm_format_modifiers;
        let image_create_maybe_linear = match tiling {
            ImageTiling::Linear => true,
            ImageTiling::Optimal => false,
            ImageTiling::DrmFormatModifier => {
                const DRM_FORMAT_MOD_LINEAR: u64 = 0;
                image_create_drm_format_modifiers.contains(&DRM_FORMAT_MOD_LINEAR)
            }
        };
        let image_create_format_features =
            format_properties.format_features(tiling, drm_format_modifiers);

        let aspects = format.aspects();

        let has_separate_stencil_usage = if aspects
            .contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
            && !stencil_usage.is_empty()
        {
            stencil_usage == usage
        } else {
            stencil_usage = usage;
            false
        };

        if has_separate_stencil_usage {
            if !(device.api_version() >= Version::V1_2
                || device.enabled_extensions().ext_separate_stencil_usage)
            {
                return Err(ValidationError {
                    problem: "`format` has both a depth and a stencil aspect, and `stencil_usage` \
                        is not empty or equal to `usage`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_2)]),
                        RequiresAllOf(&[Requires::DeviceExtension("ext_separate_stencil_usage")]),
                    ]),
                    ..Default::default()
                });
            }

            stencil_usage
                .validate_device(device)
                .map_err(|err| ValidationError {
                    context: "stencil_usage".into(),
                    vuids: &["VUID-VkImageStencilUsageCreateInfo-stencilUsage-parameter"],
                    ..ValidationError::from_requirement(err)
                })?;

            if stencil_usage.is_empty() {
                return Err(ValidationError {
                    context: "stencil_usage".into(),
                    problem: "is empty".into(),
                    vuids: &["VUID-VkImageStencilUsageCreateInfo-usage-requiredbitmask"],
                    ..Default::default()
                });
            }
        }

        initial_layout
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "initial_layout".into(),
                vuids: &["VUID-VkImageCreateInfo-initialLayout-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        if !matches!(
            initial_layout,
            ImageLayout::Undefined | ImageLayout::Preinitialized
        ) {
            return Err(ValidationError {
                context: "initial_layout".into(),
                problem: "is not `ImageLayout::Undefined` or `ImageLayout::Preinitialized`".into(),
                vuids: &["VUID-VkImageCreateInfo-initialLayout-00993"],
                ..Default::default()
            });
        }

        if flags.intersects(ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE)
            && !flags.intersects(ImageCreateFlags::MUTABLE_FORMAT)
        {
            return Err(ValidationError {
                context: "flags".into(),
                problem: "contains `ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE`, but does not \
                    contain `ImageCreateFlags::MUTABLE_FORMAT`"
                    .into(),
                vuids: &["VUID-VkImageCreateInfo-flags-01573"],
                ..Default::default()
            });
        }

        match image_type {
            ImageType::Dim1d => {
                if extent[1] != 1 {
                    return Err(ValidationError {
                        problem: "`image_type` is `ImageType::Dim1d`, but `extent[1]` is not 1"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-imageType-00956"],
                        ..Default::default()
                    });
                }

                if extent[2] != 1 {
                    return Err(ValidationError {
                        problem: "`image_type` is `ImageType::Dim1d`, but `extent[2]` is not 1"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-imageType-00956"],
                        ..Default::default()
                    });
                }
            }
            ImageType::Dim2d => {
                if extent[2] != 1 {
                    return Err(ValidationError {
                        problem: "`image_type` is `ImageType::Dim2d`, but `extent[2]` is not 1"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-imageType-00957"],
                        ..Default::default()
                    });
                }
            }
            ImageType::Dim3d => {
                if array_layers != 1 {
                    return Err(ValidationError {
                        problem: "`image_type` is `ImageType::Dim3d`, but `array_layers` is not 1"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-imageType-00961"],
                        ..Default::default()
                    });
                }
            }
        }

        if extent[0] == 0 {
            return Err(ValidationError {
                context: "extent[0]".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkImageCreateInfo-extent-00944"],
                ..Default::default()
            });
        }

        if extent[1] == 0 {
            return Err(ValidationError {
                context: "extent[1]".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkImageCreateInfo-extent-00945"],
                ..Default::default()
            });
        }

        if extent[2] == 0 {
            return Err(ValidationError {
                context: "extent[2]".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkImageCreateInfo-extent-00946"],
                ..Default::default()
            });
        }

        if array_layers == 0 {
            return Err(ValidationError {
                context: "array_layers".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkImageCreateInfo-arrayLayers-00948"],
                ..Default::default()
            });
        }

        if mip_levels == 0 {
            return Err(ValidationError {
                context: "mip_levels".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkImageCreateInfo-mipLevels-00947"],
                ..Default::default()
            });
        }

        let max_mip_levels = max_mip_levels(extent);
        debug_assert!(max_mip_levels >= 1);

        if mip_levels > max_mip_levels {
            return Err(ValidationError {
                problem: "`mip_levels` is greater than the maximum allowed number of mip levels \
                    for `dimensions`"
                    .into(),
                vuids: &["VUID-VkImageCreateInfo-mipLevels-00958"],
                ..Default::default()
            });
        }

        if samples != SampleCount::Sample1 {
            if image_type != ImageType::Dim2d {
                return Err(ValidationError {
                    problem: "`samples` is not `samples != SampleCount::Sample1`, but \
                        `image_type` is not `ImageType::Dim2d`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-samples-02257"],
                    ..Default::default()
                });
            }

            if flags.intersects(ImageCreateFlags::CUBE_COMPATIBLE) {
                return Err(ValidationError {
                    problem: "`samples` is not `samples != SampleCount::Sample1`, but \
                        `flags` contains `ImageCreateFlags::CUBE_COMPATIBLE`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-samples-02257"],
                    ..Default::default()
                });
            }

            if mip_levels != 1 {
                return Err(ValidationError {
                    problem: "`samples` is not `samples != SampleCount::Sample1`, but \
                        `mip_levels` is not 1"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-samples-02257"],
                    ..Default::default()
                });
            }

            if image_create_maybe_linear {
                return Err(ValidationError {
                    problem: "`samples` is not `samples != SampleCount::Sample1`, but \
                        the image may have linear tiling"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-samples-02257"],
                    ..Default::default()
                });
            }

            if device.enabled_extensions().khr_portability_subset
                && !device.enabled_features().multisample_array_image
                && array_layers != 1
            {
                return Err(ValidationError {
                    problem: "this device is a portability subset device, \
                        `samples` is not `SampleCount::Sample1`, and \
                        `array_layers` is greater than 1"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "multisample_array_image",
                    )])]),
                    vuids: &["VUID-VkImageCreateInfo-multisampleArrayImage-04460"],
                    ..Default::default()
                });
            }
        }

        // Check limits for YCbCr formats
        if let Some(chroma_sampling) = format.ycbcr_chroma_sampling() {
            if mip_levels != 1 {
                return Err(ValidationError {
                    problem: "`format` is a YCbCr format, but \
                        `mip_levels` is not 1"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-format-06410"],
                    ..Default::default()
                });
            }

            if samples != SampleCount::Sample1 {
                return Err(ValidationError {
                    problem: "`format` is a YCbCr format, but \
                        `samples` is not `SampleCount::Sample1`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-format-06411"],
                    ..Default::default()
                });
            }

            if image_type != ImageType::Dim2d {
                return Err(ValidationError {
                    problem: "`format` is a YCbCr format, but \
                        `image_type` is not `ImageType::Dim2d`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-format-06412"],
                    ..Default::default()
                });
            }

            if array_layers > 1 && !device.enabled_features().ycbcr_image_arrays {
                return Err(ValidationError {
                    problem: "`format` is is a YCbCr format, and \
                        `array_layers` is greater than 1"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "ycbcr_image_arrays",
                    )])]),
                    vuids: &["VUID-VkImageCreateInfo-format-06413"],
                    ..Default::default()
                });
            }

            match chroma_sampling {
                ChromaSampling::Mode444 => (),
                ChromaSampling::Mode422 => {
                    if extent[0] % 2 != 0 {
                        return Err(ValidationError {
                            problem: "`format` is a YCbCr format with horizontal \
                                chroma subsampling, but \
                                `extent[0]` is not \
                                a multiple of 2"
                                .into(),
                            vuids: &["VUID-VkImageCreateInfo-format-04712"],
                            ..Default::default()
                        });
                    }
                }
                ChromaSampling::Mode420 => {
                    if !(extent[0] % 2 == 0 && extent[1] % 2 == 0) {
                        return Err(ValidationError {
                            problem: "`format` is a YCbCr format with horizontal and vertical \
                                chroma subsampling, but \
                                `extent[0]` and `extent[1]` are not both \
                                a multiple of 2"
                                .into(),
                            vuids: &[
                                "VUID-VkImageCreateInfo-format-04712",
                                "VUID-VkImageCreateInfo-format-04713",
                            ],
                            ..Default::default()
                        });
                    }
                }
            }
        }

        /* Check usage requirements */

        let combined_usage = usage | stencil_usage;

        if combined_usage.intersects(ImageUsage::STORAGE)
            && samples != SampleCount::Sample1
            && !device.enabled_features().shader_storage_image_multisample
        {
            return Err(ValidationError {
                problem: "`usage` or `stencil_usage` contains \
                        `ImageUsage::STORAGE`, but `samples` is not `SampleCount::Sample1`"
                    .into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "shader_storage_image_multisample",
                )])]),
                vuids: &[
                    "VUID-VkImageCreateInfo-usage-00968",
                    "VUID-VkImageCreateInfo-format-02538",
                ],
                ..Default::default()
            });
        }

        if usage.intersects(ImageUsage::TRANSIENT_ATTACHMENT) {
            if !usage.intersects(
                ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT,
            ) {
                return Err(ValidationError {
                    context: "usage".into(),
                    problem: "contains `ImageUsage::TRANSIENT_ATTACHMENT`, but does not also \
                        contain one of `ImageUsage::COLOR_ATTACHMENT`, \
                        `ImageUsage::DEPTH_STENCIL_ATTACHMENT` or `ImageUsage::INPUT_ATTACHMENT`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-usage-00966"],
                    ..Default::default()
                });
            }

            if !(usage
                - (ImageUsage::TRANSIENT_ATTACHMENT
                    | ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    | ImageUsage::INPUT_ATTACHMENT))
                .is_empty()
            {
                return Err(ValidationError {
                    context: "usage".into(),
                    problem: "contains `ImageUsage::TRANSIENT_ATTACHMENT`, but also contains \
                        usages other than `ImageUsage::COLOR_ATTACHMENT`, \
                        `ImageUsage::DEPTH_STENCIL_ATTACHMENT` or `ImageUsage::INPUT_ATTACHMENT`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-usage-00963"],
                    ..Default::default()
                });
            }
        }

        if combined_usage.intersects(
            ImageUsage::COLOR_ATTACHMENT
                | ImageUsage::DEPTH_STENCIL_ATTACHMENT
                | ImageUsage::INPUT_ATTACHMENT
                | ImageUsage::TRANSIENT_ATTACHMENT,
        ) {
            if extent[0] > device_properties.max_framebuffer_width {
                return Err(ValidationError {
                    problem: "`usage` or `stencil_usage` contains \
                        `ImageUsage::COLOR_ATTACHMENT`, \
                        `ImageUsage::DEPTH_STENCIL_ATTACHMENT`, \
                        `ImageUsage::INPUT_ATTACHMENT`, or \
                        `ImageUsage::TRANSIENT_ATTACHMENT`, but \
                        `extent[0]` exceeds the `max_framebuffer_width` limit"
                        .into(),
                    vuids: &[
                        "VUID-VkImageCreateInfo-usage-00964",
                        "VUID-VkImageCreateInfo-Format-02536",
                    ],
                    ..Default::default()
                });
            }

            if extent[1] > device_properties.max_framebuffer_height {
                return Err(ValidationError {
                    problem: "`usage` or `stencil_usage` contains \
                        `ImageUsage::COLOR_ATTACHMENT`, \
                        `ImageUsage::DEPTH_STENCIL_ATTACHMENT`, \
                        `ImageUsage::INPUT_ATTACHMENT`, or \
                        `ImageUsage::TRANSIENT_ATTACHMENT`, but \
                        `extent[1]` exceeds the `max_framebuffer_height` limit"
                        .into(),
                    vuids: &[
                        "VUID-VkImageCreateInfo-usage-00965",
                        "VUID-VkImageCreateInfo-format-02537",
                    ],
                    ..Default::default()
                });
            }
        }

        if has_separate_stencil_usage {
            if usage.intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
                && !stencil_usage.intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
            {
                return Err(ValidationError {
                    problem: "`format` has both a depth and a stencil aspect, and \
                        `stencil_usage` is not empty or equal to `usage`, and \
                        `usage` contains `ImageUsage::DEPTH_STENCIL_ATTACHMENT`, but \
                        `stencil_usage` does not also contain \
                        `ImageUsage::DEPTH_STENCIL_ATTACHMENT`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-format-02795"],
                    ..Default::default()
                });
            }

            if !usage.intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
                && stencil_usage.intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
            {
                return Err(ValidationError {
                    problem: "`format` has both a depth and a stencil aspect, and \
                        `stencil_usage` is not empty or equal to `usage`, and \
                        `usage` does not contain `ImageUsage::DEPTH_STENCIL_ATTACHMENT`, but \
                        `stencil_usage` does contain `ImageUsage::DEPTH_STENCIL_ATTACHMENT`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-format-02796"],
                    ..Default::default()
                });
            }

            if usage.intersects(ImageUsage::TRANSIENT_ATTACHMENT)
                && !stencil_usage.intersects(ImageUsage::TRANSIENT_ATTACHMENT)
            {
                return Err(ValidationError {
                    problem: "`format` has both a depth and a stencil aspect, and \
                        `stencil_usage` is not empty or equal to `usage`, and \
                        `usage` contains `ImageUsage::TRANSIENT_ATTACHMENT`, but \
                        `stencil_usage` does not also contain \
                        `ImageUsage::TRANSIENT_ATTACHMENT`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-format-02797"],
                    ..Default::default()
                });
            }

            if !usage.intersects(ImageUsage::TRANSIENT_ATTACHMENT)
                && stencil_usage.intersects(ImageUsage::TRANSIENT_ATTACHMENT)
            {
                return Err(ValidationError {
                    problem: "`format` has both a depth and a stencil aspect, and \
                        `stencil_usage` is not empty or equal to `usage`, and \
                        `usage` does not contain `ImageUsage::TRANSIENT_ATTACHMENT`, but \
                        `stencil_usage` does contain \
                        `ImageUsage::TRANSIENT_ATTACHMENT`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-format-02798"],
                    ..Default::default()
                });
            }

            if stencil_usage.intersects(ImageUsage::TRANSIENT_ATTACHMENT)
                && !(stencil_usage
                    - (ImageUsage::TRANSIENT_ATTACHMENT
                        | ImageUsage::DEPTH_STENCIL_ATTACHMENT
                        | ImageUsage::INPUT_ATTACHMENT))
                    .is_empty()
            {
                return Err(ValidationError {
                    problem: "`format` has both a depth and a stencil aspect, and \
                        `stencil_usage` is not empty or equal to `usage`, and \
                        `stencil_usage contains `ImageUsage::TRANSIENT_ATTACHMENT`, but \
                        also contains usages other than `ImageUsage::DEPTH_STENCIL_ATTACHMENT` or \
                        `ImageUsage::INPUT_ATTACHMENT`"
                        .into(),
                    vuids: &["VUID-VkImageStencilUsageCreateInfo-stencilUsage-02539"],
                    ..Default::default()
                });
            }
        }

        /* Check flags requirements */

        if flags.intersects(ImageCreateFlags::CUBE_COMPATIBLE) {
            if image_type != ImageType::Dim2d {
                return Err(ValidationError {
                    problem: "`flags` contains `ImageCreateFlags::CUBE_COMPATIBLE`, but \
                        `image_type` is not `ImageType::Dim2d`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-flags-00949"],
                    ..Default::default()
                });
            }

            if extent[0] != extent[1] {
                return Err(ValidationError {
                    problem: "`flags` contains `ImageCreateFlags::CUBE_COMPATIBLE`, but \
                        `extent[0]` does not equal `extent[1]`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-imageType-00954"],
                    ..Default::default()
                });
            }

            if array_layers < 6 {
                return Err(ValidationError {
                    problem: "`flags` contains `ImageCreateFlags::CUBE_COMPATIBLE`, but \
                        `array_layers` is less than 6"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-imageType-00954"],
                    ..Default::default()
                });
            }
        }

        if flags.intersects(ImageCreateFlags::DIM2D_ARRAY_COMPATIBLE) {
            if image_type != ImageType::Dim3d {
                return Err(ValidationError {
                    problem: "`flags` contains `ImageCreateFlags::DIM2D_ARRAY_COMPATIBLE`, but \
                        `image_type` is not `ImageType::Dim3d`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-flags-00950"],
                    ..Default::default()
                });
            }

            if device.enabled_extensions().khr_portability_subset
                && !device.enabled_features().image_view2_d_on3_d_image
            {
                return Err(ValidationError {
                    problem: "this device is a portability subset device, and \
                        `flags` contains `ImageCreateFlags::DIM2D_ARRAY_COMPATIBLE`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                        "image_view2_d_on3_d_image",
                    )])]),
                    vuids: &["VUID-VkImageCreateInfo-imageView2DOn3DImage-04459"],
                    ..Default::default()
                });
            }
        }

        if flags.intersects(ImageCreateFlags::DIM2D_VIEW_COMPATIBLE) {
            if image_type != ImageType::Dim3d {
                return Err(ValidationError {
                    problem: "`flags` contains `ImageCreateFlags::DIM2D_VIEW_COMPATIBLE`, but \
                        `image_type` is not `ImageType::Dim3d`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-flags-07755"],
                    ..Default::default()
                });
            }
        }

        if flags.intersects(ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE)
            && format.compression().is_none()
        {
            return Err(ValidationError {
                problem: "`flags` contains `ImageCreateFlags::BLOCK_TEXEL_VIEW_COMPATIBLE`, \
                        but `format` is not a compressed format"
                    .into(),
                vuids: &["VUID-VkImageCreateInfo-flags-01572"],
                ..Default::default()
            });
        }

        if flags.intersects(ImageCreateFlags::DISJOINT) {
            if format.planes().len() < 2 {
                return Err(ValidationError {
                    problem: "`flags` contains `ImageCreateFlags::DISJOINT`, but `format` \
                            is not a multi-planat format"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-format-01577"],
                    ..Default::default()
                });
            }

            if !image_create_format_features.intersects(FormatFeatures::DISJOINT) {
                return Err(ValidationError {
                    problem: "`flags` contains `ImageCreateFlags::DISJOINT`, but the \
                        format features of `format` do not include \
                        `FormatFeatures::DISJOINT`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-imageCreateFormatFeatures-02260"],
                    ..Default::default()
                });
            }
        }

        /* Check sharing mode and queue families */

        match sharing {
            Sharing::Exclusive => (),
            Sharing::Concurrent(queue_family_indices) => {
                if queue_family_indices.len() < 2 {
                    return Err(ValidationError {
                        context: "sharing".into(),
                        problem: "is `Sharing::Concurrent`, but contains less than 2 elements"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-sharingMode-00942"],
                        ..Default::default()
                    });
                }

                let queue_family_count =
                    device.physical_device().queue_family_properties().len() as u32;

                for (index, &queue_family_index) in queue_family_indices.iter().enumerate() {
                    if queue_family_indices[..index].contains(&queue_family_index) {
                        return Err(ValidationError {
                            context: "queue_family_indices".into(),
                            problem: format!(
                                "the queue family index in the list at index {} is contained in \
                                the list more than once",
                                index,
                            )
                            .into(),
                            vuids: &["VUID-VkImageCreateInfo-sharingMode-01420"],
                            ..Default::default()
                        });
                    }

                    if queue_family_index >= queue_family_count {
                        return Err(ValidationError {
                            context: format!("sharing[{}]", index).into(),
                            problem: "is not less than the number of queue families in the device"
                                .into(),
                            vuids: &["VUID-VkImageCreateInfo-sharingMode-01420"],
                            ..Default::default()
                        });
                    }
                }
            }
        }

        /* External memory handles */

        if !external_memory_handle_types.is_empty() {
            if !(device.api_version() >= Version::V1_1
                || device.enabled_extensions().khr_external_memory)
            {
                return Err(ValidationError {
                    context: "external_memory_handle_types".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_external_memory")]),
                    ]),
                    ..Default::default()
                });
            }

            external_memory_handle_types
                .validate_device(device)
                .map_err(|err| ValidationError {
                    context: "external_memory_handle_types".into(),
                    vuids: &["VUID-VkExternalMemoryImageCreateInfo-handleTypes-parameter"],
                    ..ValidationError::from_requirement(err)
                })?;

            if initial_layout != ImageLayout::Undefined {
                return Err(ValidationError {
                    problem: "`external_memory_handle_types` is not empty, but \
                        `initial_layout` is not `ImageLayout::Undefined`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-pNext-01443"],
                    ..Default::default()
                });
            }
        }

        if !drm_format_modifiers.is_empty() {
            // This implicitly checks for the enabled extension too,
            // so no need to check that separately.
            if tiling != ImageTiling::DrmFormatModifier {
                return Err(ValidationError {
                    problem: "`drm_format_modifiers` is not empty, but \
                        `tiling` is not `ImageTiling::DrmFormatModifier`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-pNext-02262"],
                    ..Default::default()
                });
            }

            if flags.intersects(ImageCreateFlags::MUTABLE_FORMAT) {
                return Err(ValidationError {
                    problem: "`tiling` is `ImageTiling::DrmFormatModifier`, but \
                        `flags` contains `ImageCreateFlags::MUTABLE_FORMAT`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-tiling-02353"],
                    ..Default::default()
                });
            }

            if drm_format_modifiers.len() == 1 {
                let drm_format_modifier = drm_format_modifiers[0];
                let drm_format_modifier_properties = format_properties
                    .drm_format_modifier_properties
                    .iter()
                    .find(|properties| properties.drm_format_modifier == drm_format_modifier)
                    .ok_or(ValidationError {
                        problem: "`drm_format_modifiers` has a length of 1, but \
                            `drm_format_modifiers[0]` is not one of the modifiers in \
                            `FormatProperties::drm_format_properties`, as returned by \
                            `PhysicalDevice::format_properties` for `format`".into(),
                        vuids: &["VUID-VkImageDrmFormatModifierExplicitCreateInfoEXT-drmFormatModifierPlaneCount-02265"],
                        ..Default::default()
                    })?;

                if drm_format_modifier_plane_layouts.len()
                    != drm_format_modifier_properties.drm_format_modifier_plane_count as usize
                {
                    return Err(ValidationError {
                        problem: "`drm_format_modifiers` has a length of 1, but the length of \
                            `drm_format_modifiers_plane_layouts` does not \
                            equal `DrmFormatModifierProperties::drm_format_modifier_plane_count` \
                            for `drm_format_modifiers[0]`, as returned by \
                            `PhysicalDevice::format_properties` for `format`".into(),
                        vuids: &["VUID-VkImageDrmFormatModifierExplicitCreateInfoEXT-drmFormatModifierPlaneCount-02265"],
                        ..Default::default()
                    });
                }

                for (index, subresource_layout) in
                    drm_format_modifier_plane_layouts.iter().enumerate()
                {
                    let &SubresourceLayout {
                        offset: _,
                        size,
                        row_pitch: _,
                        array_pitch,
                        depth_pitch,
                    } = subresource_layout;

                    if size != 0 {
                        return Err(ValidationError {
                            context: format!("drm_format_modifier_plane_layouts[{}].size", index)
                                .into(),
                            problem: "is not zero".into(),
                            vuids: &[
                                "VUID-VkImageDrmFormatModifierExplicitCreateInfoEXT-size-02267",
                            ],
                            ..Default::default()
                        });
                    }

                    if array_layers == 1 && array_pitch.is_some() {
                        return Err(ValidationError {
                            problem: format!(
                                "`array_layers` is 1, but \
                                `drm_format_modifier_plane_layouts[{}].array_pitch` is `Some`",
                                index
                            )
                            .into(),
                            vuids: &["VUID-VkImageDrmFormatModifierExplicitCreateInfoEXT-arrayPitch-02268"],
                            ..Default::default()
                        });
                    }

                    if extent[2] == 1 && depth_pitch.is_some() {
                        return Err(ValidationError {
                            problem: format!(
                                "`extent[2]` is 1, but \
                                `drm_format_modifier_plane_layouts[{}].depth_pitch` is `Some`",
                                index
                            )
                            .into(),
                            vuids: &["VUID-VkImageDrmFormatModifierExplicitCreateInfoEXT-depthPitch-02269"],
                            ..Default::default()
                        });
                    }
                }
            } else {
                if !drm_format_modifier_plane_layouts.is_empty() {
                    return Err(ValidationError {
                        problem: "`drm_format_modifiers` does not contain one element, but \
                            `drm_format_modifier_plane_layouts` is not empty"
                            .into(),
                        ..Default::default()
                    });
                }
            }
        } else {
            if tiling == ImageTiling::DrmFormatModifier {
                return Err(ValidationError {
                    problem: "`tiling` is `ImageTiling::DrmFormatModifier`, but \
                        `drm_format_modifiers` is `None`"
                        .into(),
                    vuids: &["VUID-VkImageCreateInfo-tiling-02261"],
                    ..Default::default()
                });
            }

            if !drm_format_modifier_plane_layouts.is_empty() {
                return Err(ValidationError {
                    problem: "`drm_format_modifiers` does not contain one element, but \
                        `drm_format_modifier_plane_layouts` is not empty"
                        .into(),
                    ..Default::default()
                });
            }
        }

        /*
            Some device limits can be exceeded, but only for particular image configurations, which
            must be queried with `image_format_properties`. See:
            https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap48.html#capabilities-image
            First, we check if this is the case, then query the device if so.
        */

        fn iter_or_none<T>(iter: impl IntoIterator<Item = T>) -> impl Iterator<Item = Option<T>> {
            let mut iter = iter.into_iter();
            [iter.next()].into_iter().chain(iter.map(Some))
        }

        for drm_format_modifier in iter_or_none(drm_format_modifiers.iter().copied()) {
            for external_memory_handle_type in iter_or_none(external_memory_handle_types) {
                let image_format_properties = unsafe {
                    device
                        .physical_device()
                        .image_format_properties_unchecked(ImageFormatInfo {
                            flags,
                            format: Some(format),
                            image_type,
                            tiling,
                            usage,
                            stencil_usage,
                            external_memory_handle_type,
                            drm_format_modifier_info: drm_format_modifier.map(
                                |drm_format_modifier| ImageDrmFormatModifierInfo {
                                    drm_format_modifier,
                                    sharing: sharing.clone(),
                                    ..Default::default()
                                },
                            ),
                            ..Default::default()
                        })
                        .map_err(|_err| ValidationError {
                            context: "PhysicalDevice::image_format_properties".into(),
                            problem: "returned an error".into(),
                            ..Default::default()
                        })?
                };

                let image_format_properties = image_format_properties.ok_or(ValidationError {
                    problem: "the combination of parameters of this image is not \
                        supported by the physical device, as returned by \
                        `PhysicalDevice::image_format_properties`"
                        .into(),
                    vuids: &[
                        "VUID-VkImageDrmFormatModifierExplicitCreateInfoEXT-drmFormatModifier-02264",
                        "VUID-VkImageDrmFormatModifierListCreateInfoEXT-pDrmFormatModifiers-02263",
                    ],
                    ..Default::default()
                })?;

                let ImageFormatProperties {
                    max_extent,
                    max_mip_levels,
                    max_array_layers,
                    sample_counts,
                    max_resource_size: _,
                    external_memory_properties,
                    filter_cubic: _,
                    filter_cubic_minmax: _,
                } = image_format_properties;

                if extent[0] > max_extent[0] {
                    return Err(ValidationError {
                        problem: "`extent[0]` exceeds `max_extent[0]` for \
                            the combination of parameters of this image, as returned by \
                            `PhysicalDevice::image_format_properties`"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-extent-02252"],
                        ..Default::default()
                    });
                }

                if extent[1] > max_extent[1] {
                    return Err(ValidationError {
                        problem: "`extent[1]` exceeds `max_extent[1]` for \
                            the combination of parameters of this image, as returned by \
                            `PhysicalDevice::image_format_properties`"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-extent-02253"],
                        ..Default::default()
                    });
                }

                if extent[2] > max_extent[2] {
                    return Err(ValidationError {
                        problem: "`extent[2]` exceeds `max_extent[2]` for \
                            the combination of parameters of this image, as returned by \
                            `PhysicalDevice::image_format_properties`"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-extent-02254"],
                        ..Default::default()
                    });
                }

                if mip_levels > max_mip_levels {
                    return Err(ValidationError {
                        problem: "`mip_levels` exceeds `max_mip_levels` for \
                            the combination of parameters of this image, as returned by \
                            `PhysicalDevice::image_format_properties`"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-mipLevels-02255"],
                        ..Default::default()
                    });
                }

                if array_layers > max_array_layers {
                    return Err(ValidationError {
                        problem: "`array_layers` exceeds `max_array_layers` for \
                            the combination of parameters of this image, as returned by \
                            `PhysicalDevice::image_format_properties`"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-arrayLayers-02256"],
                        ..Default::default()
                    });
                }

                if !sample_counts.contains_enum(samples) {
                    return Err(ValidationError {
                        problem: "`samples` is not present in the `sample_counts` value for \
                            the combination of parameters of this image, as returned by \
                            `PhysicalDevice::image_format_properties`"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-samples-02258"],
                        ..Default::default()
                    });
                }

                if !external_memory_properties
                    .compatible_handle_types
                    .contains(external_memory_handle_types)
                {
                    return Err(ValidationError {
                        problem: "`external_memory_handle_types` is not a subset of the \
                            `external_memory_properties.compatible_handle_types` value for \
                            the combination of parameters of this image, as returned by \
                            `PhysicalDevice::image_format_properties`"
                            .into(),
                        vuids: &["VUID-VkImageCreateInfo-pNext-00990"],
                        ..Default::default()
                    });
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{ImageCreateInfo, ImageUsage, RawImage};
    use crate::{
        format::Format,
        image::{
            ImageAspect, ImageAspects, ImageCreateFlags, ImageSubresourceRange, ImageType,
            SampleCount, SubresourceRangeIterator,
        },
        DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError, VulkanError,
    };
    use smallvec::SmallVec;

    #[test]
    fn create_sampled() {
        let (device, _) = gfx_dev_and_queue!();

        let _ = RawImage::new(
            device,
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Some(Format::R8G8B8A8_UNORM),
                extent: [32, 32, 1],
                usage: ImageUsage::SAMPLED,
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn create_transient() {
        let (device, _) = gfx_dev_and_queue!();

        let _ = RawImage::new(
            device,
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Some(Format::R8G8B8A8_UNORM),
                extent: [32, 32, 1],
                usage: ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::COLOR_ATTACHMENT,
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn zero_mipmap() {
        let (device, _) = gfx_dev_and_queue!();

        assert!(matches!(
            RawImage::new(
                device,
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Some(Format::R8G8B8A8_UNORM),
                    extent: [32, 32, 1],
                    mip_levels: 0,
                    usage: ImageUsage::SAMPLED,
                    ..Default::default()
                },
            ),
            Err(VulkanError::ValidationError(_))
        ),);
    }

    #[test]
    fn mipmaps_too_high() {
        let (device, _) = gfx_dev_and_queue!();

        let res = RawImage::new(
            device,
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Some(Format::R8G8B8A8_UNORM),
                extent: [32, 32, 1],
                mip_levels: u32::MAX,
                usage: ImageUsage::SAMPLED,
                ..Default::default()
            },
        );

        match res {
            Err(VulkanError::ValidationError(_)) => (),
            _ => panic!(),
        };
    }

    #[test]
    fn shader_storage_image_multisample() {
        let (device, _) = gfx_dev_and_queue!();

        let res = RawImage::new(
            device,
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Some(Format::R8G8B8A8_UNORM),
                extent: [32, 32, 1],
                samples: SampleCount::Sample2,
                usage: ImageUsage::STORAGE,
                ..Default::default()
            },
        );

        match res {
            Err(VulkanError::ValidationError(ValidationError {
                requires_one_of:
                    RequiresOneOf(
                        [RequiresAllOf([Requires::Feature("shader_storage_image_multisample")])],
                    ),
                ..
            })) => (),
            Err(VulkanError::ValidationError(_)) => (), // unlikely but possible
            _ => panic!(),
        };
    }

    #[test]
    fn compressed_not_color_attachment() {
        let (device, _) = gfx_dev_and_queue!();

        let res = RawImage::new(
            device,
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Some(Format::ASTC_5x4_UNORM_BLOCK),
                extent: [32, 32, 1],
                usage: ImageUsage::COLOR_ATTACHMENT,
                ..Default::default()
            },
        );

        match res {
            Err(VulkanError::ValidationError(_)) => (),
            _ => panic!(),
        };
    }

    #[test]
    fn transient_forbidden_with_some_usages() {
        let (device, _) = gfx_dev_and_queue!();

        assert!(matches!(
            RawImage::new(
                device,
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Some(Format::R8G8B8A8_UNORM),
                    extent: [32, 32, 1],
                    usage: ImageUsage::TRANSIENT_ATTACHMENT | ImageUsage::SAMPLED,
                    ..Default::default()
                },
            ),
            Err(VulkanError::ValidationError(_))
        ))
    }

    #[test]
    fn cubecompatible_dims_mismatch() {
        let (device, _) = gfx_dev_and_queue!();

        let res = RawImage::new(
            device,
            ImageCreateInfo {
                flags: ImageCreateFlags::CUBE_COMPATIBLE,
                image_type: ImageType::Dim2d,
                format: Some(Format::R8G8B8A8_UNORM),
                extent: [32, 64, 1],
                usage: ImageUsage::SAMPLED,
                ..Default::default()
            },
        );

        match res {
            Err(VulkanError::ValidationError(_)) => (),
            _ => panic!(),
        };
    }

    #[test]
    #[allow(clippy::erasing_op, clippy::identity_op)]
    fn subresource_range_iterator() {
        // A fictitious set of aspects that no real image would actually ever have.
        let image_aspect_list: SmallVec<[ImageAspect; 4]> = (ImageAspects::COLOR
            | ImageAspects::DEPTH
            | ImageAspects::STENCIL
            | ImageAspects::PLANE_0)
            .into_iter()
            .collect();
        let image_mip_levels = 6;
        let image_array_layers = 8;

        let mip = image_array_layers as DeviceSize;
        let asp = mip * image_mip_levels as DeviceSize;

        // Whole image
        let mut iter = SubresourceRangeIterator::new(
            ImageSubresourceRange {
                aspects: ImageAspects::COLOR
                    | ImageAspects::DEPTH
                    | ImageAspects::STENCIL
                    | ImageAspects::PLANE_0,
                mip_levels: 0..6,
                array_layers: 0..8,
            },
            &image_aspect_list,
            asp,
            image_mip_levels,
            mip,
            image_array_layers,
        );

        assert_eq!(iter.next(), Some(0 * asp..4 * asp));
        assert_eq!(iter.next(), None);

        // Only some aspects
        let mut iter = SubresourceRangeIterator::new(
            ImageSubresourceRange {
                aspects: ImageAspects::COLOR | ImageAspects::DEPTH | ImageAspects::PLANE_0,
                mip_levels: 0..6,
                array_layers: 0..8,
            },
            &image_aspect_list,
            asp,
            image_mip_levels,
            mip,
            image_array_layers,
        );

        assert_eq!(iter.next(), Some(0 * asp..2 * asp));
        assert_eq!(iter.next(), Some(3 * asp..4 * asp));
        assert_eq!(iter.next(), None);

        // Two aspects, and only some of the mip levels
        let mut iter = SubresourceRangeIterator::new(
            ImageSubresourceRange {
                aspects: ImageAspects::DEPTH | ImageAspects::STENCIL,
                mip_levels: 2..4,
                array_layers: 0..8,
            },
            &image_aspect_list,
            asp,
            image_mip_levels,
            mip,
            image_array_layers,
        );
        assert_eq!(iter.next(), Some(1 * asp + 2 * mip..1 * asp + 4 * mip));
        assert_eq!(iter.next(), Some(2 * asp + 2 * mip..2 * asp + 4 * mip));
        assert_eq!(iter.next(), None);

        // One aspect, one mip level, only some of the array layers
        let mut iter = SubresourceRangeIterator::new(
            ImageSubresourceRange {
                aspects: ImageAspects::COLOR,

                mip_levels: 0..1,
                array_layers: 2..4,
            },
            &image_aspect_list,
            asp,
            image_mip_levels,
            mip,
            image_array_layers,
        );

        assert_eq!(
            iter.next(),
            Some(0 * asp + 0 * mip + 2..0 * asp + 0 * mip + 4)
        );
        assert_eq!(iter.next(), None);

        // Two aspects, two mip levels, only some of the array layers
        let mut iter = SubresourceRangeIterator::new(
            ImageSubresourceRange {
                aspects: ImageAspects::DEPTH | ImageAspects::STENCIL,
                mip_levels: 2..4,
                array_layers: 6..8,
            },
            &image_aspect_list,
            asp,
            image_mip_levels,
            mip,
            image_array_layers,
        );
        assert_eq!(
            iter.next(),
            Some(1 * asp + 2 * mip + 6..1 * asp + 2 * mip + 8)
        );
        assert_eq!(
            iter.next(),
            Some(1 * asp + 3 * mip + 6..1 * asp + 3 * mip + 8)
        );
        assert_eq!(
            iter.next(),
            Some(2 * asp + 2 * mip + 6..2 * asp + 2 * mip + 8)
        );
        assert_eq!(
            iter.next(),
            Some(2 * asp + 3 * mip + 6..2 * asp + 3 * mip + 8)
        );
        assert_eq!(iter.next(), None);
    }
}
