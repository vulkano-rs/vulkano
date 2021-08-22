// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::limits_check;
use crate::check_errors;
use crate::descriptor_set::layout::DescriptorSetCompatibilityError;
use crate::descriptor_set::layout::DescriptorSetDesc;
use crate::descriptor_set::layout::DescriptorSetLayout;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::pipeline::layout::PipelineLayoutLimitsError;
use crate::pipeline::shader::ShaderStages;
use crate::Error;
use crate::OomError;
use crate::VulkanObject;
use smallvec::SmallVec;
use std::cmp;
use std::error;
use std::fmt;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

/// Wrapper around the `PipelineLayout` Vulkan object. Describes to the Vulkan implementation the
/// descriptor sets and push constants available to your shaders.
pub struct PipelineLayout {
    handle: ash::vk::PipelineLayout,
    device: Arc<Device>,
    descriptor_set_layouts: SmallVec<[Arc<DescriptorSetLayout>; 16]>,
    push_constant_ranges: SmallVec<[PipelineLayoutPcRange; 8]>,
}

impl PipelineLayout {
    /// Creates a new `PipelineLayout`.
    #[inline]
    pub fn new<D, P>(
        device: Arc<Device>,
        descriptor_set_layouts: D,
        push_constant_ranges: P,
    ) -> Result<PipelineLayout, PipelineLayoutCreationError>
    where
        D: IntoIterator<Item = Arc<DescriptorSetLayout>>,
        P: IntoIterator<Item = PipelineLayoutPcRange>,
    {
        let fns = device.fns();
        let descriptor_set_layouts: SmallVec<[Arc<DescriptorSetLayout>; 16]> =
            descriptor_set_layouts.into_iter().collect();
        let push_constant_ranges: SmallVec<[PipelineLayoutPcRange; 8]> =
            push_constant_ranges.into_iter().collect();

        // Check for overlapping stages
        for (a_id, a) in push_constant_ranges.iter().enumerate() {
            for b in push_constant_ranges.iter().skip(a_id + 1) {
                if a.stages.intersects(&b.stages) {
                    return Err(PipelineLayoutCreationError::PushConstantsConflict {
                        first_range: *a,
                        second_range: *b,
                    });
                }
            }
        }

        // Check against device limits
        limits_check::check_desc_against_limits(
            device.physical_device().properties(),
            &descriptor_set_layouts,
            &push_constant_ranges,
        )?;

        // Grab the list of `vkDescriptorSetLayout` objects from `layouts`.
        let layouts_ids = descriptor_set_layouts
            .iter()
            .map(|l| l.internal_object())
            .collect::<SmallVec<[_; 16]>>();

        // Builds a list of `vkPushConstantRange` that describe the push constants.
        let push_constants = {
            let mut out: SmallVec<[_; 8]> = SmallVec::new();

            for &PipelineLayoutPcRange {
                offset,
                size,
                stages,
            } in &push_constant_ranges
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
        let handle = unsafe {
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
            handle,
            device: device.clone(),
            descriptor_set_layouts,
            push_constant_ranges,
        })
    }
}

impl PipelineLayout {
    /// Returns the descriptor set layouts this pipeline layout was created from.
    #[inline]
    pub fn descriptor_set_layouts(&self) -> &[Arc<DescriptorSetLayout>] {
        &self.descriptor_set_layouts
    }

    /// Returns a slice containing the push constant ranges this pipeline layout was created from.
    #[inline]
    pub fn push_constant_ranges(&self) -> &[PipelineLayoutPcRange] {
        &self.push_constant_ranges
    }

    /// Makes sure that `self` is a superset of the provided descriptor set layouts and push
    /// constant ranges. Returns an `Err` if this is not the case.
    pub fn ensure_compatible_with_shader(
        &self,
        descriptor_set_layout_descs: &[DescriptorSetDesc],
        push_constant_range: &Option<PipelineLayoutPcRange>,
    ) -> Result<(), PipelineLayoutSupersetError> {
        // Ewwwwwww
        let empty = DescriptorSetDesc::empty();
        let num_sets = cmp::max(
            self.descriptor_set_layouts.len(),
            descriptor_set_layout_descs.len(),
        );

        for set_num in 0..num_sets {
            let first = self
                .descriptor_set_layouts
                .get(set_num)
                .map(|set| set.desc())
                .unwrap_or_else(|| &empty);
            let second = descriptor_set_layout_descs
                .get(set_num)
                .unwrap_or_else(|| &empty);

            if let Err(error) = first.ensure_compatible_with_shader(second) {
                return Err(PipelineLayoutSupersetError::DescriptorSet {
                    error,
                    set_num: set_num as u32,
                });
            }
        }

        // FIXME: check push constants
        if let Some(range) = push_constant_range {
            for own_range in self.push_constant_ranges.as_ref().into_iter() {
                if range.stages.intersects(&own_range.stages) &&       // check if it shares any stages
                    (range.offset < own_range.offset || // our range must start before and end after the given range
                        own_range.offset + own_range.size < range.offset + range.size)
                {
                    return Err(PipelineLayoutSupersetError::PushConstantRange {
                        first_range: *own_range,
                        second_range: *range,
                    });
                }
            }
        }

        Ok(())
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
        self.handle
    }
}

impl fmt::Debug for PipelineLayout {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt.debug_struct("PipelineLayout")
            .field("raw", &self.handle)
            .field("device", &self.device)
            .field("descriptor_set_layouts", &self.descriptor_set_layouts)
            .field("push_constant_ranges", &self.push_constant_ranges)
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
                self.handle,
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
    /// Conflict between different push constants ranges.
    PushConstantsConflict {
        first_range: PipelineLayoutPcRange,
        second_range: PipelineLayoutPcRange,
    },
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
                PipelineLayoutCreationError::PushConstantsConflict { .. } => {
                    "conflict between different push constants ranges"
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

/// Error when checking whether a pipeline layout is a superset of another one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PipelineLayoutSupersetError {
    DescriptorSet {
        error: DescriptorSetCompatibilityError,
        set_num: u32,
    },
    PushConstantRange {
        first_range: PipelineLayoutPcRange,
        second_range: PipelineLayoutPcRange,
    },
}

impl error::Error for PipelineLayoutSupersetError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            PipelineLayoutSupersetError::DescriptorSet { ref error, .. } => Some(error),
            ref error @ PipelineLayoutSupersetError::PushConstantRange { .. } => Some(error),
        }
    }
}

impl fmt::Display for PipelineLayoutSupersetError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            PipelineLayoutSupersetError::DescriptorSet { .. } => {
                write!(fmt, "the descriptor set was not a superset of the other")
            }
            PipelineLayoutSupersetError::PushConstantRange {
                first_range,
                second_range,
            } => {
                writeln!(
                    fmt,
                    "our range did not completely encompass the other range"
                )?;
                writeln!(fmt, "    our stages: {:?}", first_range.stages)?;
                writeln!(
                    fmt,
                    "    our range: {} - {}",
                    first_range.offset,
                    first_range.offset + first_range.size
                )?;
                writeln!(fmt, "    other stages: {:?}", second_range.stages)?;
                write!(
                    fmt,
                    "    other range: {} - {}",
                    second_range.offset,
                    second_range.offset + second_range.size
                )
            }
        }
    }
}

/// Description of a range of the push constants of a pipeline layout.
// TODO: should contain the layout as well
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct PipelineLayoutPcRange {
    /// Offset in bytes from the start of the push constants to this range.
    pub offset: usize,
    /// Size in bytes of the range.
    pub size: usize,
    /// The stages which can access this range. Note that the same shader stage can't access two
    /// different ranges.
    pub stages: ShaderStages,
}

/* TODO: restore
#[cfg(test)]
mod tests {
    use std::iter;
    use std::sync::Arc;
    use descriptor::descriptor::ShaderStages;
    use descriptor::descriptor_set::DescriptorSetLayout;
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

        let set = match DescriptorSetLayout::raw(device1, iter::empty()) {
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
