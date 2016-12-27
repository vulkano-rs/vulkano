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
use descriptor::descriptor_set::DescriptorsCount;
use device::Device;

/// Describes to the Vulkan implementation the layout of all descriptors within a descriptor set.
///
/// Despite its name, this type is technically not unsafe. However it serves the same purpose
/// in the API as other types whose names start with `Unsafe`.
pub struct UnsafeDescriptorSetLayout {
    // The layout.
    layout: vk::DescriptorSetLayout,
    // The device this layout belongs to.
    device: Arc<Device>,
    // Number of descriptors.
    descriptors_count: DescriptorsCount,
}

impl UnsafeDescriptorSetLayout {
    /// Builds a new `UnsafeDescriptorSetLayout` with the given descriptors.
    ///
    /// The descriptors must be passed in the order of the bindings. In order words, descriptor
    /// at bind point 0 first, then descriptor at bind point 1, and so on. If a binding must remain
    /// empty, you can make the iterator yield `None` for an element.
    pub fn new<I>(device: Arc<Device>, descriptors: I) -> Result<UnsafeDescriptorSetLayout, OomError>
        where I: IntoIterator<Item = Option<DescriptorDesc>>
    {
        let mut descriptors_count = DescriptorsCount::zero();

        let bindings = descriptors.into_iter()
            .enumerate()
            .filter_map(|(binding, desc)| {
                let desc = match desc {
                    Some(d) => d,
                    None => return None,
                };

                // FIXME: it is not legal to pass eg. the TESSELLATION_SHADER bit when the device
                //        doesn't have tess shaders enabled

                let ty = desc.ty.ty().unwrap();     // TODO: shouldn't panic
                descriptors_count.add_one(ty);

                Some(vk::DescriptorSetLayoutBinding {
                    binding: binding as u32,
                    descriptorType: ty as u32,
                    descriptorCount: desc.array_count,
                    stageFlags: desc.stages.into(),
                    pImmutableSamplers: ptr::null(), // FIXME: not yet implemented
                })
            })
            .collect::<SmallVec<[_; 32]>>();

        // Note that it seems legal to have no descriptor at all in the set.

        let layout = unsafe {
            let infos = vk::DescriptorSetLayoutCreateInfo {
                sType: vk::STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0, // reserved
                bindingCount: bindings.len() as u32,
                pBindings: bindings.as_ptr(),
            };

            let mut output = mem::uninitialized();
            let vk = device.pointers();
            try!(check_errors(vk.CreateDescriptorSetLayout(device.internal_object(), &infos, ptr::null(), &mut output)));
            output
        };

        Ok(UnsafeDescriptorSetLayout {
            layout: layout,
            device: device,
            descriptors_count: descriptors_count,
        })
    }

    /// Returns the device used to create this layout.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the number of descriptors of each type.
    #[inline]
    pub fn descriptors_count(&self) -> &DescriptorsCount {
        &self.descriptors_count
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
            vk.DestroyDescriptorSetLayout(self.device.internal_object(), self.layout, ptr::null());
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
        let _layout = UnsafeDescriptorSetLayout::new(device, iter::empty());
    }
}
