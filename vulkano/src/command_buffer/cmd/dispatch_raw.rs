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

use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use VulkanObject;
use VulkanPointers;

/// Command that executes a compute shader.
pub struct CmdDispatchRaw {
    dimensions: [u32; 3],
}

impl CmdDispatchRaw {
    /// Builds a new command that executes a compute shader.
    ///
    /// The command will use the descriptor sets, push constants, and pipeline currently bound.
    #[inline]
    pub unsafe fn new(dimensions: [u32; 3]) -> Result<CmdDispatchRaw, CmdDispatchRawError> {
        // FIXME: check dimensions limits

        Ok(CmdDispatchRaw {
            dimensions: dimensions,
        })
    }
}

unsafe impl<'a, P> AddCommand<&'a CmdDispatchRaw> for UnsafeCommandBufferBuilder<P>
    where P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdDispatchRaw) -> Self::Out {
        unsafe {
            let vk = self.device().pointers();
            let cmd = self.internal_object();
            vk.CmdDispatch(cmd, command.dimensions[0], command.dimensions[1],
                           command.dimensions[2]);
        }

        self
    }
}

/// Error that can happen when creating a `CmdDispatch`.
#[derive(Debug, Copy, Clone)]
pub enum CmdDispatchRawError {
    /// The dispatch dimensions are larger than the hardware limits.
    DimensionsTooLarge,
}

impl error::Error for CmdDispatchRawError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CmdDispatchRawError::DimensionsTooLarge => {
                "the dispatch dimensions are larger than the hardware limits"
            },
        }
    }
}

impl fmt::Display for CmdDispatchRawError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
