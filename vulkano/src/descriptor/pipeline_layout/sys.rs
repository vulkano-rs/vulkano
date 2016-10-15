// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::mem;
use std::ptr;
use std::sync::Arc;
use smallvec::SmallVec;

use check_errors;
use Error;
use OomError;
use VulkanObject;
use VulkanPointers;
use vk;

use descriptor::descriptor::ShaderStages;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use descriptor::pipeline_layout::PipelineLayoutDesc;
use device::Device;

/// Low-level struct that represents the layout of the resources available to your shaders.
///
/// Despite its name, this type is technically not unsafe. However it serves the same purpose
/// in the API as other types whose names start with `Unsafe`.
pub struct PipelineLayout<L = Box<PipelineLayoutDesc + Send + Sync>> {
    device: Arc<Device>,
    layout: vk::PipelineLayout,
    layouts: SmallVec<[Arc<UnsafeDescriptorSetLayout>; 16]>,
    desc: L,
}

impl<L> PipelineLayout<L> where L: PipelineLayoutDesc {
    /// Creates a new `PipelineLayout`.
    #[inline]
    pub fn new(device: &Arc<Device>, desc: L)
               -> Result<PipelineLayout<L>, UnsafePipelineLayoutCreationError>
    {
        let vk = device.pointers();
        let limits = device.physical_device().limits();

        let layouts = {
            let mut layouts: SmallVec<[_; 16]> = SmallVec::new();
            for num in 0 .. desc.num_sets() {
                layouts.push(match desc.provided_set_layout(num) {
                    Some(l) => l,
                    None => {
                        let sets_iter = 0 .. desc.num_bindings_in_set(num).unwrap_or(0);
                        let desc_iter = sets_iter.filter_map(|d| desc.descriptor(num, d));
                        Arc::new(try!(UnsafeDescriptorSetLayout::raw(device.clone(), desc_iter)))
                    },
                });
            }
            layouts
        };

        let layouts_ids = layouts.iter().map(|l| {
                                    assert_eq!(&**l.device() as *const Device,
                                               &**device as *const Device);
                                    l.internal_object()
                                 }).collect::<SmallVec<[_; 16]>>();

        // FIXME: must also check per-descriptor-type limits (eg. max uniform buffer descriptors)

        if layouts_ids.len() > limits.max_bound_descriptor_sets() as usize {
            return Err(UnsafePipelineLayoutCreationError::MaxDescriptorSetsLimitExceeded);
        }

        let push_constants = {
            let mut out: SmallVec<[_; 8]> = SmallVec::new();

            for pc_id in 0 .. desc.num_push_constants_ranges() {
                let (offset, size, shader_stages) = match desc.push_constants_range(pc_id) {
                    Some(o) => o,
                    None => continue,
                };

                if shader_stages == ShaderStages::none() || size == 0 || (size % 4) != 0 {
                    return Err(UnsafePipelineLayoutCreationError::InvalidPushConstant);
                }

                if offset + size > limits.max_push_constants_size() as usize {
                    return Err(UnsafePipelineLayoutCreationError::MaxPushConstantsSizeExceeded);
                }

                out.push(vk::PushConstantRange {
                    stageFlags: shader_stages.into(),
                    offset: offset as u32,
                    size: size as u32,
                });
            }

            out
        };

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

        Ok(PipelineLayout {
            device: device.clone(),
            layout: layout,
            layouts: layouts,
            desc: desc,
        })
    }

    /// Returns the `UnsafeDescriptorSetLayout` object of the specified set index.
    ///
    /// Returns `None` if out of range or if the set is empty for this index.
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

unsafe impl<L> VulkanObject for PipelineLayout<L> {
    type Object = vk::PipelineLayout;

    #[inline]
    fn internal_object(&self) -> vk::PipelineLayout {
        self.layout
    }
}

impl<L> Drop for PipelineLayout<L> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyPipelineLayout(self.device.internal_object(), self.layout, ptr::null());
        }
    }
}

/// Error that can happen when creating an instance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnsafePipelineLayoutCreationError {
    /// Not enough memory.
    OomError(OomError),
    /// The maximum number of descriptor sets has been exceeded.
    MaxDescriptorSetsLimitExceeded,
    /// The maximum size of push constants has been exceeded.
    MaxPushConstantsSizeExceeded,
    /// One of the push constants range didn't obey the rules. The list of stages must not be
    /// empty, the size must not be 0, and the size must be a multiple or 4.
    InvalidPushConstant,
}

impl error::Error for UnsafePipelineLayoutCreationError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            UnsafePipelineLayoutCreationError::OomError(_) => {
                "not enough memory available"
            },
            UnsafePipelineLayoutCreationError::MaxDescriptorSetsLimitExceeded => {
                "the maximum number of descriptor sets has been exceeded"
            },
            UnsafePipelineLayoutCreationError::MaxPushConstantsSizeExceeded => {
                "the maximum size of push constants has been exceeded"
            },
            UnsafePipelineLayoutCreationError::InvalidPushConstant => {
                "one of the push constants range didn't obey the rules"
            },
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            UnsafePipelineLayoutCreationError::OomError(ref err) => Some(err),
            _ => None
        }
    }
}

impl fmt::Display for UnsafePipelineLayoutCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<OomError> for UnsafePipelineLayoutCreationError {
    #[inline]
    fn from(err: OomError) -> UnsafePipelineLayoutCreationError {
        UnsafePipelineLayoutCreationError::OomError(err)
    }
}

impl From<Error> for UnsafePipelineLayoutCreationError {
    #[inline]
    fn from(err: Error) -> UnsafePipelineLayoutCreationError {
        match err {
            err @ Error::OutOfHostMemory => {
                UnsafePipelineLayoutCreationError::OomError(OomError::from(err))
            },
            err @ Error::OutOfDeviceMemory => {
                UnsafePipelineLayoutCreationError::OomError(OomError::from(err))
            },
            _ => panic!("unexpected error: {:?}", err)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter;
    use std::sync::Arc;
    use descriptor::descriptor::ShaderStages;
    use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
    use descriptor::pipeline_layout::sys::PipelineLayout;
    use descriptor::pipeline_layout::sys::UnsafePipelineLayoutCreationError;

    #[test]
    fn empty() {
        let (device, _) = gfx_dev_and_queue!();
        let _layout = PipelineLayout::new(&device, iter::empty(), iter::empty()).unwrap();
    }

    #[test]
    #[should_panic]
    fn wrong_device_panic() {
        let (device1, _) = gfx_dev_and_queue!();
        let (device2, _) = gfx_dev_and_queue!();

        let set = match UnsafeDescriptorSetLayout::raw(device1, iter::empty()) {
            Ok(s) => Arc::new(s),
            Err(_) => return
        };

        let _ = PipelineLayout::new(&device2, Some(&set), iter::empty());
    }

    #[test]
    fn invalid_push_constant_stages() {
        let (device, _) = gfx_dev_and_queue!();

        let push_constant = (0, 8, ShaderStages::none());

        match PipelineLayout::new(&device, iter::empty(), Some(push_constant)) {
            Err(UnsafePipelineLayoutCreationError::InvalidPushConstant) => (),
            _ => panic!()
        }
    }

    #[test]
    fn invalid_push_constant_size1() {
        let (device, _) = gfx_dev_and_queue!();

        let push_constant = (0, 0, ShaderStages::all_graphics());

        match PipelineLayout::new(&device, iter::empty(), Some(push_constant)) {
            Err(UnsafePipelineLayoutCreationError::InvalidPushConstant) => (),
            _ => panic!()
        }
    }

    #[test]
    fn invalid_push_constant_size2() {
        let (device, _) = gfx_dev_and_queue!();

        let push_constant = (0, 11, ShaderStages::all_graphics());

        match PipelineLayout::new(&device, iter::empty(), Some(push_constant)) {
            Err(UnsafePipelineLayoutCreationError::InvalidPushConstant) => (),
            _ => panic!()
        }
    }
}
