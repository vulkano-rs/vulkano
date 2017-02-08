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
use std::ptr;
use std::sync::Arc;
use smallvec::SmallVec;

use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use descriptor::descriptor_set::DescriptorSetsCollection;
use descriptor::pipeline_layout::PipelineLayoutAbstract;
use descriptor::pipeline_layout::PipelineLayoutSetsCompatible;
use device::Device;
use device::DeviceOwned;
use VulkanObject;
use VulkanPointers;
use vk;

/// Command that binds descriptor sets to the command buffer.
pub struct CmdBindDescriptorSets<S, P> {
    // The raw Vulkan enum representing the kind of pipeline.
    pipeline_ty: vk::PipelineBindPoint,
    // The raw pipeline object to bind.
    raw_pipeline_layout: vk::PipelineLayout,
    // The raw sets to bind. Array where each element is a tuple of the first set to bind and the
    // sets to bind.
    raw_sets: SmallVec<[(u32, SmallVec<[vk::DescriptorSet; 8]>); 4]>,
    // The device of the pipeline object, so that we can compare it with the command buffer's
    // device.
    device: Arc<Device>,
    // The sets to bind. Unused, but we need to keep them alive.
    sets: S,
    // The pipeline layout. Unused, but we need to keep it alive.
    pipeline_layout: P,
}

impl<S, P> CmdBindDescriptorSets<S, P>
    where P: PipelineLayoutAbstract, S: DescriptorSetsCollection
{
    /// Builds the command.
    ///
    /// If `graphics` is true, the sets will be bound to the graphics slot. If false, they will be
    /// bound to the compute slot.
    ///
    /// Returns an error if the sets are not compatible with the pipeline layout.
    #[inline]
    pub fn new(graphics: bool, pipeline_layout: P, sets: S)
               -> Result<CmdBindDescriptorSets<S, P>, CmdBindDescriptorSetsError> 
    {
        if !PipelineLayoutSetsCompatible::is_compatible(pipeline_layout.desc(), &sets) {
            return Err(CmdBindDescriptorSetsError::IncompatibleSets);
        }

        let raw_pipeline_layout = pipeline_layout.sys().internal_object();
        let device = pipeline_layout.device().clone();

        let raw_sets = {
            let mut raw_sets: SmallVec<[(u32, SmallVec<[_; 8]>); 4]> = SmallVec::new();
            let mut add_new = true;
            for set_num in 0 .. sets.num_sets() {
                let set = match sets.descriptor_set(set_num) {
                    Some(set) => set.internal_object(),
                    None => { add_new = true; continue; },
                };
                
                if add_new {
                    let mut v = SmallVec::new(); v.push(set);
                    raw_sets.push((set_num as u32, v));
                    add_new = false;
                } else {
                    raw_sets.last_mut().unwrap().1.push(set);
                }
            }
            raw_sets
        };

        Ok(CmdBindDescriptorSets {
            raw_pipeline_layout: raw_pipeline_layout,
            raw_sets: raw_sets,
            pipeline_ty: if graphics { vk::PIPELINE_BIND_POINT_GRAPHICS }
                         else { vk::PIPELINE_BIND_POINT_COMPUTE },
            device: device,
            sets: sets,
            pipeline_layout: pipeline_layout,
        })
    }
}

unsafe impl<S, Pl> DeviceOwned for CmdBindDescriptorSets<S, Pl>
    where Pl: DeviceOwned
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.pipeline_layout.device()
    }
}

unsafe impl<'a, P, Pl, S> AddCommand<&'a CmdBindDescriptorSets<S, Pl>> for UnsafeCommandBufferBuilder<P>
    where P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdBindDescriptorSets<S, Pl>) -> Self::Out {
        unsafe {
            let vk = self.device().pointers();
            let cmd = self.internal_object();

            for &(first_set, ref sets) in command.raw_sets.iter() {
                vk.CmdBindDescriptorSets(cmd, command.pipeline_ty, command.raw_pipeline_layout,
                                         first_set, sets.len() as u32, sets.as_ptr(),
                                         0, ptr::null());        // TODO: dynamic offset not supported
            }
        }

        self
    }
}

/// Error that can happen when creating a `CmdBindDescriptorSets`.
#[derive(Debug, Copy, Clone)]
pub enum CmdBindDescriptorSetsError {
    /// The sets are not compatible with the pipeline layout.
    // TODO: inner error
    IncompatibleSets,
}

impl error::Error for CmdBindDescriptorSetsError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CmdBindDescriptorSetsError::IncompatibleSets => {
                "the sets are not compatible with the pipeline layout"
            },
        }
    }
}

impl fmt::Display for CmdBindDescriptorSetsError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
