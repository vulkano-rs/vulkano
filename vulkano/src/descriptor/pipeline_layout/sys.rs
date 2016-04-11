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

use descriptor::descriptor::ShaderStages;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use device::Device;

/// Low-level struct that represents the layout of the resources available to your shaders.
pub struct UnsafePipelineLayout {
    device: Arc<Device>,
    layout: vk::PipelineLayout,
    layouts: SmallVec<[Arc<UnsafeDescriptorSetLayout>; 16]>,
}

impl UnsafePipelineLayout {
    /// Creates a new `UnsafePipelineLayout`.
    ///
    /// # Panic
    ///
    /// Panicks if one of the `UnsafeDescriptorSetLayout` was not created with `device`.
    #[inline]
    pub fn new<'a, I, P>(device: &Arc<Device>, layouts: I, push_constants: P)
                         -> Result<UnsafePipelineLayout, OomError>
        where I: IntoIterator<Item = &'a Arc<UnsafeDescriptorSetLayout>>,
              P: IntoIterator<Item = (usize, usize, ShaderStages)>,
    {
        UnsafePipelineLayout::new_inner(device, layouts.into_iter().map(|e| e.clone()).collect(),
                                        push_constants.into_iter().collect())
    }

    /// Same as `new` but won't be inlined.
    fn new_inner(device: &Arc<Device>, layouts: SmallVec<[Arc<UnsafeDescriptorSetLayout>; 16]>,
                 push_constants: SmallVec<[(usize, usize, ShaderStages); 8]>)
                 -> Result<UnsafePipelineLayout, OomError>
    {
        let vk = device.pointers();

        let layouts_ids = layouts.iter().map(|l| {
                                    assert_eq!(&**l.device() as *const Device,
                                               &**device as *const Device);
                                    l.internal_object()
                                 }).collect::<SmallVec<[_; 16]>>();

        let push_constants = push_constants.iter().map(|pc| {
            // TODO: error
            assert!(pc.2 != ShaderStages::none());
            assert!(pc.1 > 0);
            assert!((pc.1 % 4) == 0);
            assert!(pc.0 + pc.1 <=
                    device.physical_device().limits().max_push_constants_size() as usize);

            vk::PushConstantRange {
                stageFlags: pc.2.into(),
                offset: pc.0 as u32,
                size: pc.1 as u32,
            }
        }).collect::<SmallVec<[_; 8]>>();

        let layout = unsafe {
            let infos = vk::PipelineLayoutCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                setLayoutCount: layouts_ids.len() as u32,
                pSetLayouts: layouts_ids.as_ptr(),
                pushConstantRangeCount: push_constants.len() as u32,
                pPushConstantRanges: push_constants.as_ptr(),
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

    /// Returns the device used to create this pipeline layout.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
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

#[cfg(test)]
mod tests {
    use std::iter;
    use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
    use descriptor::pipeline_layout::sys::UnsafePipelineLayout;

    #[test]
    fn empty() {
        let (device, _) = gfx_dev_and_queue!();
        let _layout = UnsafePipelineLayout::new(&device, iter::empty(), iter::empty());
    }

    #[test]
    #[should_panic]
    fn wrong_device_panic() {
        let (device1, _) = gfx_dev_and_queue!();
        let (device2, _) = gfx_dev_and_queue!();

        let set = match UnsafeDescriptorSetLayout::new(&device1, iter::empty()) {
            Ok(s) => s,
            Err(_) => return
        };

        UnsafePipelineLayout::new(&device2, Some(&set), iter::empty());
    }
}
