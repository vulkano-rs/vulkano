// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::check_errors;
use crate::descriptor_set::descriptor::ShaderStages;
use crate::descriptor_set::UnsafeDescriptorSetLayout;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::pipeline::layout::PipelineLayoutDesc;
use crate::pipeline::layout::PipelineLayoutDescPcRange;
use crate::pipeline::layout::PipelineLayoutLimitsError;
use crate::Error;
use crate::OomError;
use crate::VulkanObject;
use smallvec::SmallVec;
use std::error;
use std::fmt;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

/// Wrapper around the `PipelineLayout` Vulkan object. Describes to the Vulkan implementation the
/// descriptor sets and push constants available to your shaders.
pub struct PipelineLayout {
    device: Arc<Device>,
    layout: ash::vk::PipelineLayout,
    descriptor_set_layouts: SmallVec<[Arc<UnsafeDescriptorSetLayout>; 16]>,
    desc: PipelineLayoutDesc,
}

impl PipelineLayout {
    /// Creates a new `PipelineLayout`.
    #[inline]
    pub fn new(
        device: Arc<Device>,
        desc: PipelineLayoutDesc,
    ) -> Result<PipelineLayout, PipelineLayoutCreationError> {
        let fns = device.fns();

        desc.check_against_limits(&device)?;

        // Building the list of `UnsafeDescriptorSetLayout` objects.
        let descriptor_set_layouts = {
            let mut layouts: SmallVec<[_; 16]> = SmallVec::new();
            for set in desc.descriptor_sets() {
                layouts.push({
                    Arc::new(UnsafeDescriptorSetLayout::new(
                        device.clone(),
                        set.iter().map(|s| s.clone()),
                    )?)
                });
            }
            layouts
        };

        // Grab the list of `vkDescriptorSetLayout` objects from `layouts`.
        let layouts_ids = descriptor_set_layouts
            .iter()
            .map(|l| l.internal_object())
            .collect::<SmallVec<[_; 16]>>();

        // Builds a list of `vkPushConstantRange` that describe the push constants.
        let push_constants = {
            let mut out: SmallVec<[_; 8]> = SmallVec::new();

            for &PipelineLayoutDescPcRange {
                offset,
                size,
                stages,
            } in desc.push_constants()
            {
                if stages == ShaderStages::none() || size == 0 || (size % 4) != 0 {
                    return Err(PipelineLayoutCreationError::InvalidPushConstant);
                }

                out.push(ash::vk::PushConstantRange {
                    stage_flags: stages.into(),
                    offset: offset as u32,
                    size: size as u32,
                });
            }

            out
        };

        // Each bit of `stageFlags` must only be present in a single push constants range.
        // We check that with a debug_assert because it's supposed to be enforced by the
        // `PipelineLayoutDesc`.
        debug_assert!({
            let mut stages = ash::vk::ShaderStageFlags::empty();
            let mut outcome = true;
            for pc in push_constants.iter() {
                if !(stages & pc.stage_flags).is_empty() {
                    outcome = false;
                    break;
                }
                stages &= pc.stage_flags;
            }
            outcome
        });

        // FIXME: it is not legal to pass eg. the TESSELLATION_SHADER bit when the device doesn't
        //        have tess shaders enabled

        // Build the final object.
        let layout = unsafe {
            let infos = ash::vk::PipelineLayoutCreateInfo {
                flags: ash::vk::PipelineLayoutCreateFlags::empty(),
                set_layout_count: layouts_ids.len() as u32,
                p_set_layouts: layouts_ids.as_ptr(),
                push_constant_range_count: push_constants.len() as u32,
                p_push_constant_ranges: push_constants.as_ptr(),
                ..Default::default()
            };

            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_pipeline_layout(
                device.internal_object(),
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        Ok(PipelineLayout {
            device: device.clone(),
            layout,
            descriptor_set_layouts,
            desc,
        })
    }
}

impl PipelineLayout {
    /// Returns the description of the pipeline layout.
    #[inline]
    pub fn desc(&self) -> &PipelineLayoutDesc {
        &self.desc
    }

    /// Returns the `UnsafeDescriptorSetLayout` object of the specified set index.
    ///
    /// Returns `None` if out of range or if the set is empty for this index.
    #[inline]
    pub fn descriptor_set_layout(&self, index: usize) -> Option<&Arc<UnsafeDescriptorSetLayout>> {
        self.descriptor_set_layouts.get(index)
    }
}

unsafe impl DeviceOwned for PipelineLayout {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl VulkanObject for PipelineLayout {
    type Object = ash::vk::PipelineLayout;

    fn internal_object(&self) -> Self::Object {
        self.layout
    }
}

impl fmt::Debug for PipelineLayout {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt.debug_struct("PipelineLayout")
            .field("raw", &self.layout)
            .field("device", &self.device)
            .field("desc", &self.desc)
            .finish()
    }
}

impl Drop for PipelineLayout {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            fns.v1_0.destroy_pipeline_layout(
                self.device.internal_object(),
                self.layout,
                ptr::null(),
            );
        }
    }
}

/// Error that can happen when creating a pipeline layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PipelineLayoutCreationError {
    /// Not enough memory.
    OomError(OomError),
    /// The pipeline layout description doesn't fulfill the limit requirements.
    LimitsError(PipelineLayoutLimitsError),
    /// One of the push constants range didn't obey the rules. The list of stages must not be
    /// empty, the size must not be 0, and the size must be a multiple or 4.
    InvalidPushConstant,
}

impl error::Error for PipelineLayoutCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            PipelineLayoutCreationError::OomError(ref err) => Some(err),
            PipelineLayoutCreationError::LimitsError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for PipelineLayoutCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                PipelineLayoutCreationError::OomError(_) => "not enough memory available",
                PipelineLayoutCreationError::LimitsError(_) => {
                    "the pipeline layout description doesn't fulfill the limit requirements"
                }
                PipelineLayoutCreationError::InvalidPushConstant => {
                    "one of the push constants range didn't obey the rules"
                }
            }
        )
    }
}

impl From<OomError> for PipelineLayoutCreationError {
    #[inline]
    fn from(err: OomError) -> PipelineLayoutCreationError {
        PipelineLayoutCreationError::OomError(err)
    }
}

impl From<PipelineLayoutLimitsError> for PipelineLayoutCreationError {
    #[inline]
    fn from(err: PipelineLayoutLimitsError) -> PipelineLayoutCreationError {
        PipelineLayoutCreationError::LimitsError(err)
    }
}

impl From<Error> for PipelineLayoutCreationError {
    #[inline]
    fn from(err: Error) -> PipelineLayoutCreationError {
        match err {
            err @ Error::OutOfHostMemory => {
                PipelineLayoutCreationError::OomError(OomError::from(err))
            }
            err @ Error::OutOfDeviceMemory => {
                PipelineLayoutCreationError::OomError(OomError::from(err))
            }
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

/* TODO: restore
#[cfg(test)]
mod tests {
    use std::iter;
    use std::sync::Arc;
    use descriptor::descriptor::ShaderStages;
    use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
    use descriptor::pipeline_layout::sys::PipelineLayout;
    use descriptor::pipeline_layout::sys::PipelineLayoutCreationError;

    #[test]
    fn empty() {
        let (device, _) = gfx_dev_and_queue!();
        let _layout = PipelineLayout::new(&device, iter::empty(), iter::empty()).unwrap();
    }

    #[test]
    fn wrong_device_panic() {
        let (device1, _) = gfx_dev_and_queue!();
        let (device2, _) = gfx_dev_and_queue!();

        let set = match UnsafeDescriptorSetLayout::raw(device1, iter::empty()) {
            Ok(s) => Arc::new(s),
            Err(_) => return
        };

        assert_should_panic!({
            let _ = PipelineLayout::new(&device2, Some(&set), iter::empty());
        });
    }

    #[test]
    fn invalid_push_constant_stages() {
        let (device, _) = gfx_dev_and_queue!();

        let push_constant = (0, 8, ShaderStages::none());

        match PipelineLayout::new(&device, iter::empty(), Some(push_constant)) {
            Err(PipelineLayoutCreationError::InvalidPushConstant) => (),
            _ => panic!()
        }
    }

    #[test]
    fn invalid_push_constant_size1() {
        let (device, _) = gfx_dev_and_queue!();

        let push_constant = (0, 0, ShaderStages::all_graphics());

        match PipelineLayout::new(&device, iter::empty(), Some(push_constant)) {
            Err(PipelineLayoutCreationError::InvalidPushConstant) => (),
            _ => panic!()
        }
    }

    #[test]
    fn invalid_push_constant_size2() {
        let (device, _) = gfx_dev_and_queue!();

        let push_constant = (0, 11, ShaderStages::all_graphics());

        match PipelineLayout::new(&device, iter::empty(), Some(push_constant)) {
            Err(PipelineLayoutCreationError::InvalidPushConstant) => (),
            _ => panic!()
        }
    }
}*/
