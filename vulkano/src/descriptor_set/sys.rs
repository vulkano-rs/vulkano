// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Low-level descriptor set.

use crate::{
    descriptor_set::{
        layout::DescriptorSetLayout,
        update::{DescriptorWriteInfo, WriteDescriptorSet},
    },
    device::DeviceOwned,
    macros::impl_id_counter,
    VulkanObject,
};
use smallvec::SmallVec;
use std::{
    fmt::{Debug, Error as FmtError, Formatter},
    num::NonZeroU64,
    ptr,
};

/// Low-level descriptor set.
///
/// Contrary to most other objects in this library, this one doesn't free itself automatically and
/// doesn't hold the pool or the device it is associated to.
/// Instead it is an object meant to be used with the [`DescriptorPool`].
///
/// [`DescriptorPool`]: super::pool::DescriptorPool
pub struct UnsafeDescriptorSet {
    handle: ash::vk::DescriptorSet,
    id: NonZeroU64,
}

impl UnsafeDescriptorSet {
    pub(crate) fn new(handle: ash::vk::DescriptorSet) -> Self {
        Self {
            handle,
            id: Self::next_id(),
        }
    }

    /// Modifies a descriptor set. Doesn't check that the writes or copies are correct, and
    /// doesn't check whether the descriptor set is in use.
    ///
    /// # Safety
    ///
    /// - The `Device` must be the device the pool of this set was created with.
    /// - Doesn't verify that the things you write in the descriptor set match its layout.
    /// - Doesn't keep the resources alive. You have to do that yourself.
    /// - Updating a descriptor set obeys synchronization rules that aren't checked here. Once a
    ///   command buffer contains a pointer/reference to a descriptor set, it is illegal to write
    ///   to it.
    pub unsafe fn write<'a>(
        &mut self,
        layout: &DescriptorSetLayout,
        writes: impl IntoIterator<Item = &'a WriteDescriptorSet>,
    ) {
        struct PerDescriptorWrite {
            acceleration_structures: ash::vk::WriteDescriptorSetAccelerationStructureKHR,
        }

        let writes_iter = writes.into_iter();
        let (lower_size_bound, _) = writes_iter.size_hint();
        let mut infos_vk: SmallVec<[_; 8]> = SmallVec::with_capacity(lower_size_bound);
        let mut writes_vk: SmallVec<[_; 8]> = SmallVec::with_capacity(lower_size_bound);
        let mut per_writes_vk: SmallVec<[_; 8]> = SmallVec::with_capacity(lower_size_bound);

        for write in writes_iter {
            let layout_binding = &layout.bindings()[&write.binding()];

            infos_vk.push(write.to_vulkan_info(layout_binding.descriptor_type));
            writes_vk.push(write.to_vulkan(
                ash::vk::DescriptorSet::null(),
                layout_binding.descriptor_type,
            ));
            per_writes_vk.push(PerDescriptorWrite {
                acceleration_structures: Default::default(),
            });
        }

        // It is forbidden to call `vkUpdateDescriptorSets` with 0 writes, so we need to perform
        // this emptiness check.
        if writes_vk.is_empty() {
            return;
        }

        for ((info_vk, write_vk), per_write_vk) in infos_vk
            .iter()
            .zip(writes_vk.iter_mut())
            .zip(per_writes_vk.iter_mut())
        {
            match info_vk {
                DescriptorWriteInfo::Image(info) => {
                    write_vk.descriptor_count = info.len() as u32;
                    write_vk.p_image_info = info.as_ptr();
                }
                DescriptorWriteInfo::Buffer(info) => {
                    write_vk.descriptor_count = info.len() as u32;
                    write_vk.p_buffer_info = info.as_ptr();
                }
                DescriptorWriteInfo::BufferView(info) => {
                    write_vk.descriptor_count = info.len() as u32;
                    write_vk.p_texel_buffer_view = info.as_ptr();
                }
                DescriptorWriteInfo::AccelerationStructure(info) => {
                    write_vk.descriptor_count = info.len() as u32;
                    write_vk.p_next = &per_write_vk.acceleration_structures as *const _ as _;
                    per_write_vk
                        .acceleration_structures
                        .acceleration_structure_count = write_vk.descriptor_count;
                    per_write_vk
                        .acceleration_structures
                        .p_acceleration_structures = info.as_ptr();
                }
            }

            debug_assert!(write_vk.descriptor_count != 0);
        }

        let fns = layout.device().fns();

        (fns.v1_0.update_descriptor_sets)(
            layout.device().handle(),
            writes_vk.len() as u32,
            writes_vk.as_ptr(),
            0,
            ptr::null(),
        );
    }

    // TODO: add copying from other descriptor sets
    //       add a `copy` method that just takes a copy, and an `update` method that takes both
    //       writes and copies and that actually performs the operation
}

unsafe impl VulkanObject for UnsafeDescriptorSet {
    type Handle = ash::vk::DescriptorSet;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

impl Debug for UnsafeDescriptorSet {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(f, "<Vulkan descriptor set {:?}>", self.handle)
    }
}

impl_id_counter!(UnsafeDescriptorSet);
