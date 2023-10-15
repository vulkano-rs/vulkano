// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Low-level descriptor set.

use super::{
    allocator::{DescriptorSetAlloc, DescriptorSetAllocator, StandardDescriptorSetAlloc},
    pool::DescriptorPool,
    CopyDescriptorSet,
};
use crate::{
    descriptor_set::{
        layout::DescriptorSetLayout,
        update::{DescriptorWriteInfo, WriteDescriptorSet},
    },
    device::{Device, DeviceOwned},
    Validated, ValidationError, VulkanError, VulkanObject,
};
use smallvec::SmallVec;
use std::{fmt::Debug, hash::Hash, sync::Arc};

/// Low-level descriptor set.
///
/// This descriptor set does not keep track of synchronization,
/// nor does it store any information on what resources have been written to each descriptor.
#[derive(Debug)]
pub struct UnsafeDescriptorSet<P = StandardDescriptorSetAlloc> {
    alloc: P,
}

impl UnsafeDescriptorSet {
    /// Allocates a new descriptor set and returns it.
    #[inline]
    pub fn new<A>(
        allocator: &A,
        layout: &Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
    ) -> Result<UnsafeDescriptorSet<A::Alloc>, Validated<VulkanError>>
    where
        A: DescriptorSetAllocator + ?Sized,
    {
        Ok(UnsafeDescriptorSet {
            alloc: allocator.allocate(layout, variable_descriptor_count)?,
        })
    }
}

impl<P> UnsafeDescriptorSet<P>
where
    P: DescriptorSetAlloc,
{
    /// Returns the allocation of this descriptor set.
    #[inline]
    pub fn alloc(&self) -> &P {
        &self.alloc
    }

    /// Returns the descriptor pool that the descriptor set was allocated from.
    #[inline]
    pub fn pool(&self) -> &DescriptorPool {
        self.alloc.pool()
    }

    /// Returns the layout of this descriptor set.
    #[inline]
    pub fn layout(&self) -> &Arc<DescriptorSetLayout> {
        self.alloc.inner().layout()
    }

    /// Returns the variable descriptor count that this descriptor set was allocated with.
    #[inline]
    pub fn variable_descriptor_count(&self) -> u32 {
        self.alloc.inner().variable_descriptor_count()
    }

    /// Updates the descriptor set with new values.
    ///
    /// # Safety
    ///
    /// - The resources in `descriptor_writes` and `descriptor_copies` must be kept alive for as
    ///   long as `self` is in use.
    /// - The descriptor set must not be in use by the device,
    ///   or be recorded to a command buffer as part of a bind command.
    #[inline]
    pub unsafe fn update(
        &mut self,
        descriptor_writes: &[WriteDescriptorSet],
        descriptor_copies: &[CopyDescriptorSet],
    ) -> Result<(), Box<ValidationError>> {
        self.validate_update(descriptor_writes, descriptor_copies)?;

        self.update_unchecked(descriptor_writes, descriptor_copies);
        Ok(())
    }

    fn validate_update(
        &self,
        descriptor_writes: &[WriteDescriptorSet],
        descriptor_copies: &[CopyDescriptorSet],
    ) -> Result<(), Box<ValidationError>> {
        for (index, write) in descriptor_writes.iter().enumerate() {
            write
                .validate(self.layout(), self.variable_descriptor_count())
                .map_err(|err| err.add_context(format!("descriptor_writes[{}]", index)))?;
        }

        for (index, copy) in descriptor_copies.iter().enumerate() {
            copy.validate(self)
                .map_err(|err| err.add_context(format!("descriptor_copies[{}]", index)))?;
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn update_unchecked(
        &mut self,
        descriptor_writes: &[WriteDescriptorSet],
        descriptor_copies: &[CopyDescriptorSet],
    ) {
        struct PerDescriptorWrite {
            write_info: DescriptorWriteInfo,
            acceleration_structures: ash::vk::WriteDescriptorSetAccelerationStructureKHR,
            inline_uniform_block: ash::vk::WriteDescriptorSetInlineUniformBlock,
        }

        let mut writes_vk: SmallVec<[_; 8]> = SmallVec::with_capacity(descriptor_writes.len());
        let mut per_writes_vk: SmallVec<[_; 8]> = SmallVec::with_capacity(descriptor_writes.len());

        for write in descriptor_writes {
            let layout_binding = &self.layout().bindings()[&write.binding()];
            writes_vk.push(write.to_vulkan(self.handle(), layout_binding.descriptor_type));
            per_writes_vk.push(PerDescriptorWrite {
                write_info: write.to_vulkan_info(layout_binding.descriptor_type),
                acceleration_structures: Default::default(),
                inline_uniform_block: Default::default(),
            });
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

        let mut copies_vk: SmallVec<[_; 8]> = SmallVec::with_capacity(descriptor_copies.len());

        for copy in descriptor_copies {
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
                src_set: src_set.handle(),
                src_binding,
                src_array_element: src_first_array_element,
                dst_set: self.handle(),
                dst_binding,
                dst_array_element: dst_first_array_element,
                descriptor_count,
                ..Default::default()
            });
        }

        let fns = self.device().fns();
        (fns.v1_0.update_descriptor_sets)(
            self.device().handle(),
            writes_vk.len() as u32,
            writes_vk.as_ptr(),
            copies_vk.len() as u32,
            copies_vk.as_ptr(),
        );
    }
}

unsafe impl<P> VulkanObject for UnsafeDescriptorSet<P>
where
    P: DescriptorSetAlloc,
{
    type Handle = ash::vk::DescriptorSet;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.alloc.inner().handle()
    }
}

unsafe impl<P> DeviceOwned for UnsafeDescriptorSet<P>
where
    P: DescriptorSetAlloc,
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.alloc.inner().device()
    }
}

impl<P> PartialEq for UnsafeDescriptorSet<P>
where
    P: DescriptorSetAlloc,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.alloc.inner() == other.alloc.inner()
    }
}

impl<P> Eq for UnsafeDescriptorSet<P> where P: DescriptorSetAlloc {}

impl<P> Hash for UnsafeDescriptorSet<P>
where
    P: DescriptorSetAlloc,
{
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.alloc.inner().hash(state);
    }
}
