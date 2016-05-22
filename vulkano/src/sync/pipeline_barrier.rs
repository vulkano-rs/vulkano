// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use smallvec::SmallVec;
use vk;

/// Represents the state of a pipeline barrier that can be inserted in a command buffer.
pub struct PipelineBarrier {
    src_stages: vk::PipelineStageFlagBits,
    dest_stages: vk::PipelineStageFlagBits,
    flags: vk::DependencyFlags,
    memory_barriers: SmallVec<[vk::MemoryBarrier; 2]>,
    buffer_memory_barriers: SmallVec<[vk::BufferMemoryBarrier; 8]>,
    image_memory_barriers: SmallVec<[vk::ImageMemoryBarrier; 8]>,
}

impl PipelineBarrier {
    /// Builds a new `PipelineBarrier`.
    #[inline]
    pub fn new() -> PipelineBarrier {
        PipelineBarrier {
            src_stages: 0,
            dest_stages: 0,
            flags: vk::DEPENDENCY_BY_REGION_BIT,
            memory_barriers: SmallVec::new(),
            buffer_memory_barriers: SmallVec::new(),
            image_memory_barriers: SmallVec::new(),
        }
    }

    /// Sets that the pipeline barrier is not by region.
    #[inline]
    pub fn set_not_by_region(&mut self) {
        self.flags &= !vk::DEPENDENCY_BY_REGION_BIT;
    }

    // TODO: method to add the barrier to a command buffer
}
