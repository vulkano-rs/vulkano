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

use command_buffer::RawCommandBufferPrototype;
use command_buffer::CommandsList;
use command_buffer::CommandsListSink;
use descriptor::descriptor_set::TrackedDescriptorSetsCollection;
use descriptor::pipeline_layout::PipelineLayoutRef;
use descriptor::pipeline_layout::PipelineLayoutSetsCompatible;
use device::Device;
use VulkanObject;
use VulkanPointers;
use vk;

/// Wraps around a commands list and adds at the end of it a command that binds descriptor sets.
pub struct CmdBindDescriptorSets<L, S, P> where L: CommandsList {
    // Parent commands list.
    previous: L,
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

impl<L, S, P> CmdBindDescriptorSets<L, S, P>
    where L: CommandsList, S: TrackedDescriptorSetsCollection, P: PipelineLayoutRef
{
    /// Builds the command.
    ///
    /// If `graphics` is true, the sets will be bound to the graphics slot. If false, they will be
    /// bound to the compute slot.
    ///
    /// Returns an error if the sets are not compatible with the pipeline layout.
    #[inline]
    pub fn new(previous: L, graphics: bool, pipeline_layout: P, sets: S)
               -> Result<CmdBindDescriptorSets<L, S, P>, CmdBindDescriptorSetsError> 
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
            previous: previous,
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

unsafe impl<L, S, P> CommandsList for CmdBindDescriptorSets<L, S, P>
    where L: CommandsList, S: TrackedDescriptorSetsCollection
{
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        self.previous.append(builder);

        assert_eq!(self.device.internal_object(), builder.device().internal_object());

        self.sets.add_transition(builder);

        builder.add_command(Box::new(move |raw: &mut RawCommandBufferPrototype| {
            unsafe {
                let vk = raw.device.pointers();
                let cmd = raw.command_buffer.clone().take().unwrap();

                for &(first_set, ref sets) in self.raw_sets.iter() {
                    vk.CmdBindDescriptorSets(cmd, self.pipeline_ty, self.raw_pipeline_layout,
                                            first_set, sets.len() as u32, sets.as_ptr(),
                                            0, ptr::null());        // TODO: dynamic offset not supported
                }
            }
        }));
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
