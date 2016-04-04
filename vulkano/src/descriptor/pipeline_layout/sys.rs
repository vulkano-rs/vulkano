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

use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use device::Device;

/// Low-level struct that represents the layout of the resources available to your shaders.
// TODO: push constants.
pub struct UnsafePipelineLayout {
    device: Arc<Device>,
    layout: vk::PipelineLayout,
    layouts: SmallVec<[Arc<UnsafeDescriptorSetLayout>; 16]>,
}

impl UnsafePipelineLayout {
    /// Creates a new `UnsafePipelineLayout`.
    // TODO: is this function unsafe?
    #[inline]
    pub unsafe fn new<'a, I>(device: &Arc<Device>, layouts: I)
                             -> Result<UnsafePipelineLayout, OomError>
        where I: IntoIterator<Item = &'a Arc<UnsafeDescriptorSetLayout>>
    {
        UnsafePipelineLayout::new_inner(device, layouts.into_iter().map(|e| e.clone()).collect())
    }

    // TODO: is this function unsafe?
    unsafe fn new_inner(device: &Arc<Device>,
                        layouts: SmallVec<[Arc<UnsafeDescriptorSetLayout>; 16]>)
                        -> Result<UnsafePipelineLayout, OomError>
    {
        let vk = device.pointers();

        // FIXME: check that they belong to the right device
        let layouts_ids = layouts.iter().map(|l| l.internal_object())
                                 .collect::<SmallVec<[_; 16]>>();

        let layout = {
            let infos = vk::PipelineLayoutCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                setLayoutCount: layouts_ids.len() as u32,
                pSetLayouts: layouts_ids.as_ptr(),
                pushConstantRangeCount: 0,      // TODO: unimplemented
                pPushConstantRanges: ptr::null(),    // TODO: unimplemented
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreatePipelineLayout(device.internal_object(), &infos,
                                                      ptr::null(), &mut output)));
            output
        };

        Ok(UnsafePipelineLayout {
            device: device.clone(),
            layout: layout,
            layouts: layouts,
        })
    }

    /// Returns the `UnsafeDescriptorSetLayout` object of the specified set index.
    ///
    /// Returns `None` if out of range.
    #[inline]
    pub fn descriptor_set_layout(&self, index: usize) -> Option<&Arc<UnsafeDescriptorSetLayout>> {
        self.layouts.get(index)
    }
}

unsafe impl VulkanObject for UnsafePipelineLayout {
    type Object = vk::PipelineLayout;

    #[inline]
    fn internal_object(&self) -> vk::PipelineLayout {
        self.layout
    }
}

impl Drop for UnsafePipelineLayout {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyPipelineLayout(self.device.internal_object(), self.layout, ptr::null());
        }
    }
}
