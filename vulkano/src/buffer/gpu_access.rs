// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use smallvec::SmallVec;
use std::ops::Range;

use sync::AccessError;

#[derive(Debug)]
pub struct GpuAccessRange {
    range: Range<usize>,
    locks: u32,
    exclusive: bool
}

#[derive(Debug)]
pub struct GpuAccess {
    ranges: SmallVec<[GpuAccessRange; 4]>
}

fn ranges_intersect(a: &Range<usize>, b: &Range<usize>) -> bool {
    a.start < b.end && a.end > b.start
}

// TODO: to make loops here faster, we could use an interval tree to do so,
// which is a data structure for efficiently finding overlapping ranges
impl GpuAccess {
    pub fn new() -> Self {
        GpuAccess {
            ranges: smallvec![]
        }
    }

    #[inline]
    pub fn can_lock(&self, exclusive: bool, range: Range<usize>) -> bool {
        // Find all ranges that intersect with the range we're looking to lock
        for access in self.ranges
            .iter()
            .filter(|x| ranges_intersect(&x.range, &range)) {
            // Can't exclusively lock the range if it's already locked by
            // someone else
            if exclusive {
                return false;
            }

            // Can't lock the range if it's exclusively locked by someone else
            if access.exclusive {
                return false;
            }
        }

        return true;
    }

    #[inline]
    pub fn try_lock(&mut self, exclusive: bool, range: Range<usize>) -> Result<(), AccessError> {
        // If there are no other locks yet, we can go ahead and lock
        if self.ranges.is_empty() {
            self.ranges.push(GpuAccessRange {
                range,
                locks: 1,
                exclusive
            });

            return Ok(());
        }

        let mut range_already_locked = false;

        // Find all ranges that intersect with the range we're looking to lock
        for access in self.ranges
            .iter_mut()
            .filter(|x| ranges_intersect(&x.range, &range)) {
            // Can't exclusively lock the range if it's already locked by
            // someone else
            if exclusive {
                return Err(AccessError::AlreadyInUse);
            }

            // Can't lock the range if it's exclusively locked by someone else
            if access.exclusive {
                return Err(AccessError::AlreadyInUse);
            }

            // If this is the same range we're trying to lock, increase the lock
            // count
            if access.range == range {
                debug_assert!(!range_already_locked);
                range_already_locked = true;

                access.locks += 1;
            }
        }

        // Didn't find this range already locked, so go ahead and lock
        if !range_already_locked {
            self.ranges.push(GpuAccessRange {
                range,
                locks: 1,
                exclusive
            });
        }

        Ok(())
    }

    #[inline]
    pub unsafe fn increase_lock(&mut self, range: Range<usize>) {
        // When a lock is increased, it's possible that the range is not exactly
        // the same that was used to create the lock. This can happen, for
        // example when using a single copy_buffer command to copy both vertex
        // and index data into the buffer. This will result in a single lock
        // containing the both vertex and index buffer ranges.

        // Later, when draw_indexed is invoked, bind_index_buffer is called, and
        // it will try to increase the lock with its range, but that range is in
        // fact a subrange of the range that was used to create the lock.

        // Hence, we will look for an intersecting range and increase the lock
        // for that instead.

        let mut intersecting_ranges = 0;

        for access in self.ranges
            .iter_mut()
            .filter(|x| ranges_intersect(&x.range, &range)) {
            debug_assert!(access.locks >= 1);
            access.locks += 1;

            intersecting_ranges += 1;
        }

        assert!(intersecting_ranges > 0, "Tried to increase lock for a buffer range that is not locked");

        // If multiple intersecting ranges are found, I think that means that an
        // invalid range was provided here?

        assert!(intersecting_ranges == 1);
    }

    #[inline]
    pub unsafe fn unlock(&mut self, range: Range<usize>) {
        let mut intersecting_ranges = 0;

        for (i, access) in self.ranges
            .iter_mut()
            .enumerate()
            .filter(|(_, x)| ranges_intersect(&x.range, &range)) {
            assert!(access.locks >= 1);
            access.locks -= 1;

            intersecting_ranges += 1;
        }

        assert!(intersecting_ranges > 0, "Tried to unlock a buffer range that isn't locked");

        // If multiple intersecting ranges are found, I think that means that an
        // invalid range was provided here?

        assert!(intersecting_ranges == 1);

        self.ranges.retain(|x| x.locks > 0);
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
