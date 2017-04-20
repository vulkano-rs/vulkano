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

use command_buffer::CommandAddError;
use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use device::Device;
use device::DeviceOwned;
use VulkanObject;
use VulkanPointers;

/// Command that executes a compute shader.
///
/// > **Note**: Unless you are writing a custom implementation of a command buffer, you are
/// > encouraged to ignore this struct and use a `CmdDispatch` instead.
pub struct CmdDispatchRaw {
    dimensions: [u32; 3],
    device: Arc<Device>,
}

impl CmdDispatchRaw {
    /// Builds a new command that executes a compute shader.
    ///
    /// The command will use the descriptor sets, push constants, and pipeline currently bound.
    ///
    /// This function checks whether the dimensions are supported by the device. It returns an
    /// error if they are not.
    ///
    /// # Safety
    ///
    /// While building the command is always safe, care must be taken when it is added to a command
    /// buffer. A correct combination of compute pipeline, descriptor set and push constants must
    /// have been bound beforehand.
    ///
    #[inline]
    pub unsafe fn new(device: Arc<Device>, dimensions: [u32; 3])
                      -> Result<CmdDispatchRaw, CmdDispatchRawError>
    {
        let max_dims = device.physical_device().limits().max_compute_work_group_count();

        if dimensions[0] > max_dims[0] || dimensions[1] > max_dims[1] ||
           dimensions[2] > max_dims[2]
        {
            return Err(CmdDispatchRawError::DimensionsTooLarge);
        }

        Ok(CmdDispatchRaw {
            dimensions: dimensions,
            device: device,
        })
    }

    /// Builds a new command that executes a compute shader.
    ///
    /// The command will use the descriptor sets, push constants, and pipeline currently bound.
    ///
    /// Contrary to `new`, this function doesn't check whether the dimensions are supported by the
    /// device. It always succeeds.
    ///
    /// # Safety
    ///
    /// See the documentation of `new`. Contrary to `new`, the dimensions are not checked by
    /// this function. It is illegal to build a command with dimensions that are not supported by
    /// the device.
    ///
    #[inline]
    pub unsafe fn unchecked_dimensions(device: Arc<Device>, dimensions: [u32; 3])
                                       -> Result<CmdDispatchRaw, CmdDispatchRawError>
    {
        Ok(CmdDispatchRaw {
            dimensions: dimensions,
            device: device,
        })
    }
}

unsafe impl DeviceOwned for CmdDispatchRaw {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl<'a, P> AddCommand<&'a CmdDispatchRaw> for UnsafeCommandBufferBuilder<P>
    where P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdDispatchRaw) -> Result<Self::Out, CommandAddError> {
        unsafe {
            let vk = self.device().pointers();
            let cmd = self.internal_object();
            vk.CmdDispatch(cmd, command.dimensions[0], command.dimensions[1],
                           command.dimensions[2]);
        }

        Ok(self)
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

#[cfg(test)]
mod tests {
    use command_buffer::commands_raw::CmdDispatchRaw;
    use command_buffer::commands_raw::CmdDispatchRawError;

    #[test]
    fn basic_create() {
        let (device, _) = gfx_dev_and_queue!();

        // Min required supported dimensions is 65535.
        match unsafe { CmdDispatchRaw::new(device, [128, 128, 128]) } {
            Ok(_) => (),
            _ => panic!()
        }
    }

    #[test]
    fn limit_checked() {
        let (device, _) = gfx_dev_and_queue!();

        let limit = device.physical_device().limits().max_compute_work_group_count();
        let x = match limit[0].checked_add(2) {
            None => return,
            Some(x) => x,
        };

        match unsafe { CmdDispatchRaw::new(device, [x, limit[1], limit[2]]) } {
            Err(CmdDispatchRawError::DimensionsTooLarge) => (),
            _ => panic!()
        }
    }
}
