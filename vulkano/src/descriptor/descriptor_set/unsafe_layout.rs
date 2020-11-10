// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use smallvec::SmallVec;
use std::fmt;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

use check_errors;
use vk;
use OomError;
use VulkanObject;

use descriptor::descriptor::DescriptorDesc;
use descriptor::descriptor_set::DescriptorSetDesc;
use descriptor::descriptor_set::DescriptorsCount;
use device::Device;
use device::DeviceOwned;

/// Describes to the Vulkan implementation the layout of all descriptors within a descriptor set.
///
/// Despite its name, this type is technically not unsafe. However it serves the same purpose
/// in the API as other types whose names start with `Unsafe`. Using the same naming scheme avoids
/// confusions.
pub struct UnsafeDescriptorSetLayout {
    // The layout.
    layout: vk::DescriptorSetLayout,
    // The device this layout belongs to.
    device: Arc<Device>,
    // Descriptors.
    descriptors: SmallVec<[Option<DescriptorDesc>; 32]>,
    // Number of descriptors.
    descriptors_count: DescriptorsCount,
}

impl UnsafeDescriptorSetLayout {
    /// Builds a new `UnsafeDescriptorSetLayout` with the given descriptors.
    ///
    /// The descriptors must be passed in the order of the bindings. In order words, descriptor
    /// at bind point 0 first, then descriptor at bind point 1, and so on. If a binding must remain
    /// empty, you can make the iterator yield `None` for an element.
    pub fn new<I>(
        device: Arc<Device>,
        descriptors: I,
    ) -> Result<UnsafeDescriptorSetLayout, OomError>
    where
        I: IntoIterator<Item = Option<DescriptorDesc>>,
    {
        let descriptors = descriptors.into_iter().collect::<SmallVec<[_; 32]>>();
        let mut descriptors_count = DescriptorsCount::zero();

        let bindings = descriptors
            .iter()
            .enumerate()
            .filter_map(|(binding, desc)| {
                let desc = match desc {
                    Some(d) => d,
                    None => return None,
                };

                // FIXME: it is not legal to pass eg. the TESSELLATION_SHADER bit when the device
                //        doesn't have tess shaders enabled

                let ty = desc.ty.ty().unwrap(); // TODO: shouldn't panic
                descriptors_count.add_one(ty);

                Some(vk::DescriptorSetLayoutBinding {
                    binding: binding as u32,
                    descriptorType: ty as u32,
                    descriptorCount: desc.array_count,
                    stageFlags: desc.stages.into_vulkan_bits(),
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

            let mut output = MaybeUninit::uninit();
            let vk = device.pointers();
            check_errors(vk.CreateDescriptorSetLayout(
                device.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(UnsafeDescriptorSetLayout {
            layout: layout,
            device: device,
            descriptors: descriptors,
            descriptors_count: descriptors_count,
        })
    }

    /// Returns the number of descriptors of each type.
    #[inline]
    pub fn descriptors_count(&self) -> &DescriptorsCount {
        &self.descriptors_count
    }
}

unsafe impl DescriptorSetDesc for UnsafeDescriptorSetLayout {
    #[inline]
    fn num_bindings(&self) -> usize {
        self.descriptors.len()
    }
    #[inline]
    fn descriptor(&self, binding: usize) -> Option<DescriptorDesc> {
        self.descriptors.get(binding).cloned().unwrap_or(None)
    }
}

unsafe impl DeviceOwned for UnsafeDescriptorSetLayout {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl fmt::Debug for UnsafeDescriptorSetLayout {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt.debug_struct("UnsafeDescriptorSetLayout")
            .field("raw", &self.layout)
            .field("device", &self.device)
            .finish()
    }
}

unsafe impl VulkanObject for UnsafeDescriptorSetLayout {
    type Object = vk::DescriptorSetLayout;

    const TYPE: vk::ObjectType = vk::OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT;

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
    use descriptor::descriptor::DescriptorBufferDesc;
    use descriptor::descriptor::DescriptorDesc;
    use descriptor::descriptor::DescriptorDescTy;
    use descriptor::descriptor::ShaderStages;
    use descriptor::descriptor_set::DescriptorsCount;
    use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
    use std::iter;

    #[test]
    fn empty() {
        let (device, _) = gfx_dev_and_queue!();
        let _layout = UnsafeDescriptorSetLayout::new(device, iter::empty());
    }

    #[test]
    fn basic_create() {
        let (device, _) = gfx_dev_and_queue!();

        let layout = DescriptorDesc {
            ty: DescriptorDescTy::Buffer(DescriptorBufferDesc {
                dynamic: Some(false),
                storage: false,
            }),
            array_count: 1,
            stages: ShaderStages::all_graphics(),
            readonly: true,
        };

        let sl = UnsafeDescriptorSetLayout::new(device.clone(), iter::once(Some(layout))).unwrap();

        assert_eq!(
            sl.descriptors_count(),
            &DescriptorsCount {
                uniform_buffer: 1,
                ..DescriptorsCount::zero()
            }
        );
    }
}
