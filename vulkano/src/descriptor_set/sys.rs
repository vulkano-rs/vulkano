// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Low-level descriptor set.

use crate::descriptor_set::layout::DescriptorSetLayout;
use crate::descriptor_set::update::{DescriptorWriteInfo, WriteDescriptorSet};
use crate::device::DeviceOwned;
use crate::VulkanObject;
use smallvec::SmallVec;
use std::fmt;
use std::ptr;

/// Low-level descriptor set.
///
/// Contrary to most other objects in this library, this one doesn't free itself automatically and
/// doesn't hold the pool or the device it is associated to.
/// Instead it is an object meant to be used with the `UnsafeDescriptorPool`.
pub struct UnsafeDescriptorSet {
    handle: ash::vk::DescriptorSet,
}

impl UnsafeDescriptorSet {
    pub(crate) fn new(handle: ash::vk::DescriptorSet) -> Self {
        Self { handle }
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
    ///
    pub unsafe fn write<'a>(
        &mut self,
        layout: &DescriptorSetLayout,
        writes: impl IntoIterator<Item = &'a WriteDescriptorSet>,
    ) {
        let (infos, mut writes): (SmallVec<[_; 8]>, SmallVec<[_; 8]>) = writes
            .into_iter()
            .map(|write| {
                let descriptor_type = layout.descriptor(write.binding()).unwrap().ty;

                (
                    write.to_vulkan_info(descriptor_type),
                    write.to_vulkan(self.handle, descriptor_type),
                )
            })
            .unzip();

        // It is forbidden to call `vkUpdateDescriptorSets` with 0 writes, so we need to perform
        // this emptiness check.
        if writes.is_empty() {
            return;
        }

        // Set the info pointers separately.
        for (info, write) in infos.iter().zip(writes.iter_mut()) {
            match info {
                DescriptorWriteInfo::Image(info) => {
                    write.descriptor_count = info.len() as u32;
                    write.p_image_info = info.as_ptr();
                }
                DescriptorWriteInfo::Buffer(info) => {
                    write.descriptor_count = info.len() as u32;
                    write.p_buffer_info = info.as_ptr();
                }
                DescriptorWriteInfo::BufferView(info) => {
                    write.descriptor_count = info.len() as u32;
                    write.p_texel_buffer_view = info.as_ptr();
                }
            }

            debug_assert!(write.descriptor_count != 0);
        }

        let fns = layout.device().fns();

        fns.v1_0.update_descriptor_sets(
            layout.device().internal_object(),
            writes.len() as u32,
            writes.as_ptr(),
            0,
            ptr::null(),
        );
    }

    // TODO: add copying from other descriptor sets
    //       add a `copy` method that just takes a copy, and an `update` method that takes both
    //       writes and copies and that actually performs the operation
}

unsafe impl VulkanObject for UnsafeDescriptorSet {
    type Object = ash::vk::DescriptorSet;

    #[inline]
    fn internal_object(&self) -> ash::vk::DescriptorSet {
        self.handle
    }
}

impl fmt::Debug for UnsafeDescriptorSet {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan descriptor set {:?}>", self.handle)
    }
}
