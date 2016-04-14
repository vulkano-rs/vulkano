// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::mem;
use std::ptr;
use std::sync::Arc;
use smallvec::SmallVec;

use check_errors;
use OomError;
use VulkanObject;
use VulkanPointers;
use vk;

use descriptor::descriptor::DescriptorDesc;
use device::Device;

/// Describes to the Vulkan implementation the layout of all descriptors within a descriptor set.
///
/// Despite its name, this type is technically not unsafe. However it serves the same purpose
/// in the API as other types whose names start with `Unsafe`.
pub struct UnsafeDescriptorSetLayout {
    layout: vk::DescriptorSetLayout,
    device: Arc<Device>,
}

impl UnsafeDescriptorSetLayout {
    /// Builds a new `UnsafeDescriptorSetLayout` with the given descriptors.
    pub fn raw<I>(device: &Arc<Device>, descriptors: I)
                  -> Result<UnsafeDescriptorSetLayout, OomError>
        where I: IntoIterator<Item = DescriptorDesc>
    {
        let vk = device.pointers();

        let bindings = descriptors.into_iter().map(|desc| {
            vk::DescriptorSetLayoutBinding {
                binding: desc.binding,
                descriptorType: desc.ty.vk_enum(),
                descriptorCount: desc.array_count,
                stageFlags: desc.stages.into(),
                pImmutableSamplers: ptr::null(),        // FIXME: not yet implemented
            }
        }).collect::<SmallVec<[_; 32]>>();

        let layout = unsafe {
            let infos = vk::DescriptorSetLayoutCreateInfo {
                sType: vk::STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                bindingCount: bindings.len() as u32,
                pBindings: bindings.as_ptr(),
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateDescriptorSetLayout(device.internal_object(), &infos,
                                                           ptr::null(), &mut output)));
            output
        };

        Ok(UnsafeDescriptorSetLayout {
            layout: layout,
            device: device.clone(),
        })
    }
    
    /// Builds a new `UnsafeDescriptorSetLayout` with the given descriptors.
    ///
    /// # Panic
    ///
    /// - Panicks if the device or host ran out of memory.
    ///
    #[inline]
    pub fn new<I>(device: &Arc<Device>, descriptors: I) -> Arc<UnsafeDescriptorSetLayout>
        where I: IntoIterator<Item = DescriptorDesc>
    {
        Arc::new(UnsafeDescriptorSetLayout::raw(device, descriptors).unwrap())
    }

    /// Returns the device used to create this layout.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl VulkanObject for UnsafeDescriptorSetLayout {
    type Object = vk::DescriptorSetLayout;

    #[inline]
    fn internal_object(&self) -> vk::DescriptorSetLayout {
        self.layout
    }
}

impl Drop for UnsafeDescriptorSetLayout {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyDescriptorSetLayout(self.device.internal_object(), self.layout,
                                          ptr::null());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter;
    use descriptor::descriptor_set::unsafe_layout::UnsafeDescriptorSetLayout;

    #[test]
    fn empty() {
        let (device, _) = gfx_dev_and_queue!();
        let _layout = UnsafeDescriptorSetLayout::new(&device, iter::empty());
    }
}
