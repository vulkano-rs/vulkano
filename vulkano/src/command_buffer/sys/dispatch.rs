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
use std::sync::Arc;

use command_buffer::pool::CommandPool;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use command_buffer::sys::bind_pipeline::ComputePipelineBindCommand;
use command_buffer::sys::bind_sets::DescriptorSetsBindCommand;
use command_buffer::sys::bind_sets::DescriptorSetsBindError;
use descriptor::descriptor_set::DescriptorSetsCollection;
use descriptor::pipeline_layout::PipelineLayout;
use descriptor::pipeline_layout::PipelineLayoutSetsCompatible;
use descriptor::pipeline_layout::PipelineLayoutPushConstantsCompatible;
use pipeline::ComputePipeline;

use VulkanPointers;

/// Prototype for a command that draws.
pub struct DispatchCommand {
    bind_pipeline: ComputePipelineBindCommand,
    bind_descriptor_sets: DescriptorSetsBindCommand,

    dimensions: [u32; 3],
}

impl DispatchCommand {
    /// Builds a new command that executes a compute shader.
    ///
    /// # Panic
    ///
    /// - Panicks if some of the parameters were not created with the same device.
    ///
    pub fn new<S, P, Pl>(pipeline: &Arc<ComputePipeline<Pl>>, dimensions: [u32; 3],
                         sets: S, push_constants: &P) -> Result<DispatchCommand, DispatchError>
        where Pl: PipelineLayout + PipelineLayoutSetsCompatible<S> +
                  PipelineLayoutPushConstantsCompatible<P>,
              S: DescriptorSetsCollection,
              P: Copy,
    {
        let device = pipeline.layout().inner_pipeline_layout().device();
        let max_work_group_count = device.physical_device().limits().max_compute_work_group_count();

        if dimensions[0] >= max_work_group_count[0] || dimensions[1] >= max_work_group_count[1] ||
           dimensions[2] >= max_work_group_count[2]
        {
            return Err(DispatchError::MaxWorkGroupCountExceeded);
        }

        Ok(DispatchCommand {
            bind_pipeline: ComputePipelineBindCommand::new(pipeline),
            bind_descriptor_sets: try!({
                let pipeline_layout = pipeline.layout();
                DescriptorSetsBindCommand::new(true, &**pipeline_layout, sets, push_constants)
            }),
            dimensions: dimensions,
        })
    }

    /// Submits the command to the command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if the various parameters were not created with the same device as the
    ///   command buffer.
    /// - Panicks if the command buffer is inside a render pass.
    /// - Panicks if the queue family does not support compute operations.
    ///
    pub fn submit<P>(&mut self, mut cb: UnsafeCommandBufferBuilder<P>)
                     -> UnsafeCommandBufferBuilder<P>
        where P: CommandPool
    {
        unsafe {
            // Various checks.
            assert!(!cb.within_render_pass);
            assert!(cb.pool().queue_family().supports_compute());

            // Binding stuff.
            cb = self.bind_pipeline.submit(cb);
            cb = self.bind_descriptor_sets.submit(cb);

            // Dispatching.
            {
                let _pool_lock = cb.pool().lock();
                let vk = cb.device.pointers();
                let cmd = cb.cmd.clone().unwrap();
                vk.CmdDispatch(cmd, self.dimensions[0], self.dimensions[1], self.dimensions[2]);
            }

            cb
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DispatchError {
    MaxWorkGroupCountExceeded,
    DescriptorSetsBindError(DescriptorSetsBindError),
}

impl error::Error for DispatchError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            DispatchError::MaxWorkGroupCountExceeded => {
                "the maximum number of work groups has been exceeded"
            },
            DispatchError::DescriptorSetsBindError(_) => {
                "error when binding the descriptor sets and push constants"
            },
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            DispatchError::DescriptorSetsBindError(ref err) => Some(err),
            _ => None
        }
    }
}

impl From<DescriptorSetsBindError> for DispatchError {
    #[inline]
    fn from(err: DescriptorSetsBindError) -> DispatchError {
        DispatchError::DescriptorSetsBindError(err)
    }
}

impl fmt::Display for DispatchError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
