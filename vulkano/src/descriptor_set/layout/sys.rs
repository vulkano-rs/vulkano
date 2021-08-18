// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::check_errors;
use crate::descriptor_set::layout::DescriptorDesc;
use crate::descriptor_set::layout::DescriptorSetDesc;
use crate::descriptor_set::pool::DescriptorsCount;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::OomError;
use crate::VulkanObject;
use smallvec::SmallVec;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

/// Describes to the Vulkan implementation the layout of all descriptors within a descriptor set.
#[derive(Debug)]
pub struct DescriptorSetLayout {
    // The layout.
    handle: ash::vk::DescriptorSetLayout,
    // The device this layout belongs to.
    device: Arc<Device>,
    // Descriptors.
    desc: DescriptorSetDesc,
    // Number of descriptors.
    descriptors_count: DescriptorsCount,
}

impl DescriptorSetLayout {
    /// Builds a new `DescriptorSetLayout` with the given descriptors.
    ///
    /// The descriptors must be passed in the order of the bindings. In order words, descriptor
    /// at bind point 0 first, then descriptor at bind point 1, and so on. If a binding must remain
    /// empty, you can make the iterator yield `None` for an element.
    pub fn new<D>(device: Arc<Device>, desc: D) -> Result<DescriptorSetLayout, OomError>
    where
        D: Into<DescriptorSetDesc>,
    {
        let desc = desc.into();
        let mut descriptors_count = DescriptorsCount::zero();
        let mut variable_descriptor_count = false;

        let bindings = desc
            .bindings()
            .iter()
            .enumerate()
            .filter_map(|(binding, desc)| {
                let desc = match desc {
                    Some(d) => d,
                    None => return None,
                };

                // FIXME: it is not legal to pass eg. the TESSELLATION_SHADER bit when the device
                //        doesn't have tess shaders enabled

                let ty = desc.ty.ty();

                let array_count = if desc.variable_count {
                    variable_descriptor_count = true;

                    if desc.array_count == 0 {
                        1
                    } else {
                        desc.array_count
                    }
                } else {
                    desc.array_count
                };

                descriptors_count.add_num(ty, array_count);

                Some(ash::vk::DescriptorSetLayoutBinding {
                    binding: binding as u32,
                    descriptor_type: ty.into(),
                    descriptor_count: array_count,
                    stage_flags: desc.stages.into(),
                    p_immutable_samplers: ptr::null(), // FIXME: not yet implemented
                })
            })
            .collect::<SmallVec<[_; 32]>>();

        // Note that it seems legal to have no descriptor at all in the set.

        let handle = unsafe {
            if variable_descriptor_count {
                // TODO: Check vulkan version & features

                let mut flags = vec![ash::vk::DescriptorBindingFlags::empty(); bindings.len()];
                *flags.last_mut().unwrap() = ash::vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT | ash::vk::DescriptorBindingFlags::PARTIALLY_BOUND;

                let binding_flags = ash::vk::DescriptorSetLayoutBindingFlagsCreateInfo {
                    binding_count: bindings.len() as u32,
                    p_binding_flags: flags.as_ptr(),
                    ..Default::default()
                };

                let infos = ash::vk::DescriptorSetLayoutCreateInfo {
                    flags: ash::vk::DescriptorSetLayoutCreateFlags::empty(),
                    binding_count: bindings.len() as u32,
                    p_bindings: bindings.as_ptr(),
                    p_next: &binding_flags as *const _ as *const _,
                    ..Default::default()
                };

                let mut output = MaybeUninit::uninit();
                let fns = device.fns();

                check_errors(fns.v1_0.create_descriptor_set_layout(
                    device.internal_object(),
                    &infos,
                    ptr::null(),
                    output.as_mut_ptr(),
                ))?;
                
                output.assume_init()
            } else {
                let infos = ash::vk::DescriptorSetLayoutCreateInfo {
                    flags: ash::vk::DescriptorSetLayoutCreateFlags::empty(),
                    binding_count: bindings.len() as u32,
                    p_bindings: bindings.as_ptr(),
                    ..Default::default()
                };

                let mut output = MaybeUninit::uninit();
                let fns = device.fns();

                check_errors(fns.v1_0.create_descriptor_set_layout(
                    device.internal_object(),
                    &infos,
                    ptr::null(),
                    output.as_mut_ptr(),
                ))?;
                
                output.assume_init()
            }
        };

        Ok(DescriptorSetLayout {
            handle,
            device,
            desc,
            descriptors_count,
        })
    }

    pub(crate) fn desc(&self) -> &DescriptorSetDesc {
        &self.desc
    }

    /// Returns the number of descriptors of each type.
    #[inline]
    pub fn descriptors_count(&self) -> &DescriptorsCount {
        &self.descriptors_count
    }

    /// Returns the number of binding slots in the set.
    #[inline]
    pub fn num_bindings(&self) -> usize {
        self.desc.bindings().len()
    }

    /// Returns a description of a descriptor, or `None` if out of range.
    #[inline]
    pub fn descriptor(&self, binding: usize) -> Option<DescriptorDesc> {
        self.desc.bindings().get(binding).cloned().unwrap_or(None)
    }
}

unsafe impl DeviceOwned for DescriptorSetLayout {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl VulkanObject for DescriptorSetLayout {
    type Object = ash::vk::DescriptorSetLayout;

    #[inline]
    fn internal_object(&self) -> ash::vk::DescriptorSetLayout {
        self.handle
    }
}

impl Drop for DescriptorSetLayout {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            fns.v1_0.destroy_descriptor_set_layout(
                self.device.internal_object(),
                self.handle,
                ptr::null(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::descriptor_set::layout::DescriptorBufferDesc;
    use crate::descriptor_set::layout::DescriptorDesc;
    use crate::descriptor_set::layout::DescriptorDescTy;
    use crate::descriptor_set::layout::DescriptorSetDesc;
    use crate::descriptor_set::layout::DescriptorSetLayout;
    use crate::descriptor_set::pool::DescriptorsCount;
    use crate::pipeline::shader::ShaderStages;
    use std::iter;

    #[test]
    fn empty() {
        let (device, _) = gfx_dev_and_queue!();
        let _layout = DescriptorSetLayout::new(device, DescriptorSetDesc::empty());
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

        let sl = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetDesc::new(iter::once(Some(layout))),
        )
        .unwrap();

        assert_eq!(
            sl.descriptors_count(),
            &DescriptorsCount {
                uniform_buffer: 1,
                ..DescriptorsCount::zero()
            }
        );
    }
}
