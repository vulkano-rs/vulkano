// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Low-level descriptor set.

use super::CopyDescriptorSet;
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
use std::{fmt::Debug, num::NonZeroU64};

/// Low-level descriptor set.
///
/// Contrary to most other objects in this library, this one doesn't free itself automatically and
/// doesn't hold the pool or the device it is associated to.
/// Instead it is an object meant to be used with the [`DescriptorPool`].
///
/// [`DescriptorPool`]: super::pool::DescriptorPool
#[derive(Debug)]
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
    pub unsafe fn update<'a>(
        &mut self,
        layout: &DescriptorSetLayout,
        descriptor_writes: impl IntoIterator<Item = &'a WriteDescriptorSet>,
        descriptor_copies: impl IntoIterator<Item = &'a CopyDescriptorSet>,
    ) {
        struct PerDescriptorWrite {
            write_info: DescriptorWriteInfo,
            acceleration_structures: ash::vk::WriteDescriptorSetAccelerationStructureKHR,
            inline_uniform_block: ash::vk::WriteDescriptorSetInlineUniformBlock,
        }

        let writes_iter = descriptor_writes.into_iter();
        let (lower_size_bound, _) = writes_iter.size_hint();
        let mut writes_vk: SmallVec<[_; 8]> = SmallVec::with_capacity(lower_size_bound);
        let mut per_writes_vk: SmallVec<[_; 8]> = SmallVec::with_capacity(lower_size_bound);

        for write in writes_iter {
            let layout_binding = &layout.bindings()[&write.binding()];
            writes_vk.push(write.to_vulkan(self.handle, layout_binding.descriptor_type));
            per_writes_vk.push(PerDescriptorWrite {
                write_info: write.to_vulkan_info(layout_binding.descriptor_type),
                acceleration_structures: Default::default(),
                inline_uniform_block: Default::default(),
            });
        }

        // It is forbidden to call `vkUpdateDescriptorSets` with 0 writes, so we need to perform
        // this emptiness check.
        if writes_vk.is_empty() {
            return;
        }

        for (write_vk, per_write_vk) in writes_vk.iter_mut().zip(per_writes_vk.iter_mut()) {
            match &mut per_write_vk.write_info {
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
                DescriptorWriteInfo::InlineUniformBlock(data) => {
                    write_vk.descriptor_count = data.len() as u32;
                    write_vk.p_next = &per_write_vk.inline_uniform_block as *const _ as _;
                    per_write_vk.inline_uniform_block.data_size = write_vk.descriptor_count;
                    per_write_vk.inline_uniform_block.p_data = data.as_ptr() as *const _;
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

        let copies_iter = descriptor_copies.into_iter();
        let (lower_size_bound, _) = copies_iter.size_hint();
        let mut copies_vk: SmallVec<[_; 8]> = SmallVec::with_capacity(lower_size_bound);

        for copy in copies_iter {
            let &CopyDescriptorSet {
                ref src_set,
                src_binding,
                src_first_array_element,
                dst_binding,
                dst_first_array_element,
                descriptor_count,
                _ne: _,
            } = copy;

            copies_vk.push(ash::vk::CopyDescriptorSet {
                src_set: src_set.inner().handle(),
                src_binding,
                src_array_element: src_first_array_element,
                dst_set: self.handle,
                dst_binding,
                dst_array_element: dst_first_array_element,
                descriptor_count,
                ..Default::default()
            });
        }

        let fns = layout.device().fns();
        (fns.v1_0.update_descriptor_sets)(
            layout.device().handle(),
            writes_vk.len() as u32,
            writes_vk.as_ptr(),
            copies_vk.len() as u32,
            copies_vk.as_ptr(),
        );
    }
}

unsafe impl VulkanObject for UnsafeDescriptorSet {
    type Handle = ash::vk::DescriptorSet;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

impl_id_counter!(UnsafeDescriptorSet);
