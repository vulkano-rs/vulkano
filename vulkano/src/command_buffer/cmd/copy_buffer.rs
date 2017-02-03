// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::cmp;
use std::error;
use std::fmt;

use buffer::Buffer;
use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use device::DeviceOwned;
use VulkanObject;
use VulkanPointers;
use vk;

/// Command that copies from a buffer to another.
pub struct CmdCopyBuffer<S, D> {
    source: S,
    source_raw: vk::Buffer,
    destination: D,
    destination_raw: vk::Buffer,
    src_offset: vk::DeviceSize,
    dst_offset: vk::DeviceSize,
    size: vk::DeviceSize,
}

impl<S, D> CmdCopyBuffer<S, D>
    where S: Buffer, D: Buffer
{
    /// Builds a new command.
    ///
    /// This command will copy from the source to the destination. If their size is not equal, then
    /// the amount of data copied is equal to the smallest of the two.
    ///
    /// # Panic
    ///
    /// - Panics if the source and destination were not created with the same device.
    // FIXME: type safety
    pub fn new(source: S, destination: D)
               -> Result<CmdCopyBuffer<S, D>, CmdCopyBufferError>
    {
        // TODO:
        //assert!(previous.is_outside_render_pass());     // TODO: error
        assert_eq!(source.inner().buffer.device().internal_object(),
                   destination.inner().buffer.device().internal_object());

        let (source_raw, src_offset) = {
            let inner = source.inner();
            if !inner.buffer.usage_transfer_src() {
                return Err(CmdCopyBufferError::SourceMissingTransferUsage);
            }
            (inner.buffer.internal_object(), inner.offset)
        };

        let (destination_raw, dst_offset) = {
            let inner = destination.inner();
            if !inner.buffer.usage_transfer_dest() {
                return Err(CmdCopyBufferError::DestinationMissingTransferUsage);
            }
            (inner.buffer.internal_object(), inner.offset)
        };

        let size = cmp::min(source.size(), destination.size());

        if source.conflicts_buffer(0, size, &destination, 0, size) {
            return Err(CmdCopyBufferError::OverlappingRanges);
        } else {
            debug_assert!(!destination.conflicts_buffer(0, size, &source, 0, size));
        }

        Ok(CmdCopyBuffer {
            source: source,
            source_raw: source_raw,
            destination: destination,
            destination_raw: destination_raw,
            src_offset: src_offset as u64,
            dst_offset: dst_offset as u64,
            size: size as u64,
        })
    }
}

unsafe impl<'a, P, S, D> AddCommand<&'a CmdCopyBuffer<S, D>> for UnsafeCommandBufferBuilder<P>
    where S: Buffer,
          D: Buffer,
          P: CommandPool,
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdCopyBuffer<S, D>) -> Self::Out {
        unsafe {
            let vk = self.device().pointers();
            let cmd = self.internal_object();

            let region = vk::BufferCopy {
                srcOffset: command.src_offset,
                dstOffset: command.dst_offset,
                size: command.size,
            };

            vk.CmdCopyBuffer(cmd, command.source_raw, command.destination_raw, 1, &region);
        }

        self
    }
}

/// Error that can happen when creating a `CmdCopyBuffer`.
#[derive(Debug, Copy, Clone)]
pub enum CmdCopyBufferError {
    /// The source buffer is missing the transfer source usage.
    SourceMissingTransferUsage,
    /// The destination buffer is missing the transfer destination usage.
    DestinationMissingTransferUsage,
    /// The source and destination are overlapping.
    OverlappingRanges,
}

impl error::Error for CmdCopyBufferError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            CmdCopyBufferError::SourceMissingTransferUsage => {
                "the source buffer is missing the transfer source usage"
            },
            CmdCopyBufferError::DestinationMissingTransferUsage => {
                "the destination buffer is missing the transfer destination usage"
            },
            CmdCopyBufferError::OverlappingRanges => {
                "the source and destination are overlapping"
            },
        }
    }
}

impl fmt::Display for CmdCopyBufferError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
