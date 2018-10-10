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

impl GpuAccess {
    pub fn new() -> Self {
        GpuAccess {
            ranges: smallvec![]
        }
    }

    #[inline]
    pub fn can_lock(&self, exclusive: bool, range: Range<usize>) -> bool {
        // Find all ranges that intersect with the range we're looking to lock
        // TODO: if we need to make this loop faster, we can use an interval
        // tree to do so, which is a data structure for efficiently finding
        // overlapping ranges
        for access in self.ranges
            .iter()
            .take_while(|x| x.range.start <= range.end && x.range.end <= range.start) {
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
        // TODO: if we need to make this loop faster, we can use an interval
        // tree to do so, which is a data structure for efficiently finding
        // overlapping ranges
        for access in self.ranges
            .iter_mut()
            .take_while(|x| x.range.start <= range.end && x.range.end <= range.start) {
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
        let access = self.ranges
            .iter_mut()
            .find(|x| x.range == range)
            .expect("Tried to increase lock for a buffer range that is not locked");

        debug_assert!(access.locks >= 1);
        access.locks += 1;
    }

    #[inline]
    pub unsafe fn unlock(&mut self, range: Range<usize>) {
        let (i, _) = self.ranges
            .iter_mut()
            .enumerate()
            .find(|(_, x)| x.range == range)
            .expect("Tried to unlock a buffer range that isn't locked");

        assert!(self.ranges[i].locks >= 1);
        self.ranges[i].locks -= 1;

        if self.ranges[i].locks == 0 {
            self.ranges.remove(i);
        }
    }
}
