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
use smallvec::SmallVec;

use buffer::Buffer;
use command_buffer::pool::CommandPool;
use command_buffer::sys::CommandPrototype;
use command_buffer::sys::KeepAlive;
use command_buffer::sys::UnsafeCommandBufferBuilder;

use VulkanObject;
use VulkanPointers;
use vk;

/// Prototype for a command that copies data between buffers.
pub struct BufferCopyCommand {
    keep_alive_src: Arc<KeepAlive + 'static>,
    keep_alive_dest: Arc<KeepAlive + 'static>,
    device: vk::Device,
    internal_src: vk::Buffer,
    internal_dest: vk::Buffer,
    regions: SmallVec<[vk::BufferCopy; 4]>,
}

impl BufferCopyCommand {
    /// Adds a command that copies regions between a source and a destination buffer. Does not
    /// check the type of the content.
    ///
    /// Regions whose size is 0 are automatically ignored. If no region was passed or if all
    /// regions have a size of 0, then no command is added to the command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if the two the buffers were not allocated with the same device.
    ///
    pub fn new<Bs, Bd, I>(src: &Arc<Bs>, dest: &Arc<Bd>, regions: I)
                          -> Result<BufferCopyCommand, BufferCopyError>
        where Bs: Buffer + Send + Sync + 'static,
              Bd: Buffer + Send + Sync + 'static,
              I: IntoIterator<Item = BufferCopyRegion>
    {
        // Various checks.
        assert_eq!(src.inner_buffer().device().internal_object(),
                   dest.inner_buffer().device().internal_object());
        if !src.inner_buffer().usage_transfer_src() ||
           !dest.inner_buffer().usage_transfer_dest()
        {
            return Err(BufferCopyError::WrongUsageFlag);
        }

        // Building the list of regions.
        let regions: SmallVec<[_; 4]> = {
            let mut res = SmallVec::new();
            for region in regions.into_iter() {
                if region.source_offset + region.size > src.size() {
                    return Err(BufferCopyError::OutOfRange);
                }
                if region.destination_offset + region.size > dest.size() {
                    return Err(BufferCopyError::OutOfRange);
                }
                if region.size == 0 { continue; }

                res.push(vk::BufferCopy {
                    srcOffset: region.source_offset as vk::DeviceSize,
                    dstOffset: region.destination_offset as vk::DeviceSize,
                    size: region.size as vk::DeviceSize,
                });
            }
            res
        };

        // Checking for overlaps.
        for r1 in 0 .. regions.len() {
            for r2 in (r1 + 1) .. regions.len() {
                let r1 = &regions[r1];
                let r2 = &regions[r2];

                if r1.srcOffset <= r2.srcOffset && r1.srcOffset + r1.size >= r2.srcOffset {
                    return Err(BufferCopyError::OverlappingRegions);
                }
                if r2.srcOffset <= r1.srcOffset && r2.srcOffset + r2.size >= r1.srcOffset {
                    return Err(BufferCopyError::OverlappingRegions);
                }
                if r1.dstOffset <= r2.dstOffset && r1.dstOffset + r1.size >= r2.dstOffset {
                    return Err(BufferCopyError::OverlappingRegions);
                }
                if r2.dstOffset <= r1.dstOffset && r2.dstOffset + r2.size >= r1.dstOffset {
                    return Err(BufferCopyError::OverlappingRegions);
                }

                if src.inner_buffer().internal_object() == dest.inner_buffer().internal_object() {
                    if r1.srcOffset <= r2.dstOffset && r1.srcOffset + r1.size >= r2.dstOffset {
                        return Err(BufferCopyError::OverlappingRegions);
                    }
                    if r2.srcOffset <= r1.dstOffset && r2.srcOffset + r2.size >= r1.dstOffset {
                        return Err(BufferCopyError::OverlappingRegions);
                    }
                }
            }
        }

        Ok(BufferCopyCommand {
            internal_src: src.inner_buffer().internal_object(),
            internal_dest: dest.inner_buffer().internal_object(),
            device: src.inner_buffer().device().internal_object(),
            keep_alive_src: src.clone(),
            keep_alive_dest: dest.clone(),
            regions: regions,
        })
    }

    /// Submits the command to the command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if the command buffer is within a render pass.
    /// - Panicks if the buffers were not allocated with the same device as the command buffer.
    ///
    pub fn submit<P>(&mut self, mut cb: UnsafeCommandBufferBuilder<P>)
                     -> UnsafeCommandBufferBuilder<P>
        where P: CommandPool
    {
        unsafe {
            let _pool_lock = cb.pool().lock();

            // Various checks.
            assert!(!cb.within_render_pass);
            assert_eq!(self.device, cb.device().internal_object());

            // Vulkan requires that the number of regions must always be >= 1.
            if self.regions.is_empty() { return cb; }

            cb.keep_alive.push(self.keep_alive_src.clone());
            cb.keep_alive.push(self.keep_alive_dest.clone());

            {
                let vk = cb.device.pointers();
                let cmd = cb.cmd.clone().unwrap();
                vk.CmdCopyBuffer(cmd, self.internal_src, self.internal_dest,
                                 self.regions.len() as u32, self.regions.as_ptr());
            }

            cb
        }
    }
}

unsafe impl CommandPrototype for BufferCopyCommand {
    unsafe fn submit<P>(&mut self, cb: UnsafeCommandBufferBuilder<P>)
                        -> UnsafeCommandBufferBuilder<P>
        where P: CommandPool
    {
        self.submit(cb)
    }
}

/// A copy between two buffers.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct BufferCopyRegion {
    /// Offset of the first byte to read from the source buffer.
    pub source_offset: usize,
    /// Offset of the first byte to write to the destination buffer.
    pub destination_offset: usize,
    /// Size in bytes of the copy.
    pub size: usize,
}

error_ty!{BufferCopyError => "Error that can happen when copying between buffers.",
    OutOfRange => "one of regions is out of range of the buffer",
    WrongUsageFlag => "one of the buffers doesn't have the correct usage flag",
    OverlappingRegions => "some regions are overlapping",
}
