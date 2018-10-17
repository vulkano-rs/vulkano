// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ops::Range;
use std::collections::HashMap;

use bio::data_structures::interval_tree::IntervalTree;
use bio::utils::Interval;
use sync::AccessError;
use vk::CommandBuffer;

#[derive(Debug)]
pub enum GpuAccessType {
    Exclusive,
    NonExclusive
}

use self::GpuAccessType::Exclusive;

#[derive(Debug)]
pub struct GpuAccess {
    trees: HashMap<CommandBuffer, IntervalTree<usize, GpuAccessType>>
}

impl GpuAccess {
    pub fn new() -> Self {
        GpuAccess {
            trees: HashMap::new()
        }
    }

    #[inline]
    pub fn can_cpu_lock(&self, access: GpuAccessType, range: Range<usize>) -> bool {
        // Find all ranges that overlap with the range we're looking to lock
        for (_, tree) in &self.trees {
            for entry in tree.find(range.clone()) {
                // Can't exclusively lock the range if it's already locked by
                // someone else
                if let Exclusive = access {
                    return false;
                }

                // Can't lock the range if it's exclusively locked by someone else
                if let Exclusive = entry.data() {
                    return false;
                }
            }
        }

        return true;
    }

    #[inline]
    pub fn try_gpu_lock(&mut self, command_buffer: CommandBuffer, access: GpuAccessType, range: Range<usize>) -> Result<(), AccessError> {
        let mut range_already_locked = false;

        // Find all ranges that overlap with the range we're looking to lock
        for (_, tree) in &self.trees {
            for entry in tree.find(range.clone()) {
                // Can't exclusively lock the range if it's already locked by
                // someone else
                if let Exclusive = access {
                    return Err(AccessError::AlreadyInUse);
                }

                let interval = entry.interval();

                // Can't lock the range if it's exclusively locked by someone
                // else
                if let Exclusive = entry.data() {
                    return Err(AccessError::AlreadyInUse);
                }

                // Check to see if we've already non-exclusively locked this
                // exact range so we don't insert the same range to the tree
                // twice
                if interval == &Interval::from(&range) {
                    debug_assert!(!range_already_locked);
                    range_already_locked = true;
                }
            }
        }

        // Didn't find this range already locked, so go ahead and lock
        if !range_already_locked {
            let tree = self.trees.entry(command_buffer).or_insert(IntervalTree::new());
            tree.insert(range, access);
        }

        Ok(())
    }

    #[inline]
    pub fn gpu_unlock(&mut self, command_buffer: CommandBuffer) {
        self.trees.remove(&command_buffer);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use buffer::BufferUsage;
    use buffer::BufferAccess;
    use buffer::cpu_access::CpuAccessibleBuffer;
    use command_buffer::AutoCommandBufferBuilder;
    use command_buffer::CommandBuffer;
    use command_buffer::CommandBufferExecError;

    #[test]
    fn write_to_overlapping_ranges_forbidden() {
        let (device, queue) = gfx_dev_and_queue!();

        let buffer1: Arc<CpuAccessibleBuffer<[u8]>> = unsafe {
            CpuAccessibleBuffer::uninitialized_array(device.clone(),
                                                     1000,
                                                     BufferUsage::all()).unwrap()
        };
        let buffer2: Arc<CpuAccessibleBuffer<[u8]>> = unsafe {
            CpuAccessibleBuffer::uninitialized_array(device.clone(),
                                                     1000,
                                                     BufferUsage::all()).unwrap()
        };

        let source = buffer1.clone()
                            .into_buffer_slice()
                            .slice(0..100)
                            .unwrap();

        let destination = buffer2.clone()
                                 .into_buffer_slice()
                                 .slice(0..100)
                                 .unwrap();

        let transfer_future = AutoCommandBufferBuilder::new(device.clone(), queue.family())
            .unwrap()
            .copy_buffer(source.clone(), destination)
            .unwrap()
            .build()
            .unwrap()
            .execute(queue.clone())
            .unwrap();

        let destination = buffer2.clone()
                                 .into_buffer_slice()
                                 .slice(50..150)
                                 .unwrap();

        let transfer_future = AutoCommandBufferBuilder::new(device.clone(), queue.family())
            .unwrap()
            .copy_buffer(source, destination)
            .unwrap()
            .build()
            .unwrap()
            .execute(queue.clone());

        match transfer_future {
            Err(CommandBufferExecError::AccessError { command_name, command_param, .. }) => {
                assert_eq!(command_name, "vkCmdCopyBuffer");
                assert_eq!(command_param, "destination");
            },
            Err(e) => panic!("{:?}", e),
            Ok(_) => panic!()
        }
    }

    #[test]
    fn write_to_non_overlapping_ranges_allowed() {
        let (device, queue) = gfx_dev_and_queue!();

        let buffer1: Arc<CpuAccessibleBuffer<[u8]>> = unsafe {
            CpuAccessibleBuffer::uninitialized_array(device.clone(),
                                                     1000,
                                                     BufferUsage::all()).unwrap()
        };
        let buffer2: Arc<CpuAccessibleBuffer<[u8]>> = unsafe {
            CpuAccessibleBuffer::uninitialized_array(device.clone(),
                                                     1000,
                                                     BufferUsage::all()).unwrap()
        };

        let source = buffer1.clone()
                            .into_buffer_slice()
                            .slice(0..100)
                            .unwrap();

        let destination = buffer2.clone()
                                 .into_buffer_slice()
                                 .slice(0..100)
                                 .unwrap();

        let transfer_future = AutoCommandBufferBuilder::new(device.clone(), queue.family())
            .unwrap()
            .copy_buffer(source.clone(), destination)
            .unwrap()
            .build()
            .unwrap()
            .execute(queue.clone())
            .unwrap();

        let destination = buffer2.clone()
                                 .into_buffer_slice()
                                 .slice(100..150)
                                 .unwrap();

        let transfer_future = AutoCommandBufferBuilder::new(device.clone(), queue.family())
            .unwrap()
            .copy_buffer(source, destination)
            .unwrap()
            .build()
            .unwrap()
            .execute(queue.clone())
            .unwrap();
    }
}
