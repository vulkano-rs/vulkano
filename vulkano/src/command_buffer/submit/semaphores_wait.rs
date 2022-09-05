// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    command_buffer::submit::{SubmitCommandBufferBuilder, SubmitPresentBuilder},
    sync::{PipelineStages, Semaphore},
};
use smallvec::SmallVec;

/// Prototype for a submission that waits on semaphores.
///
/// This prototype can't actually be submitted because it doesn't correspond to anything in Vulkan.
/// However you can convert it into another builder prototype through the `Into` trait.
#[derive(Debug)]
pub struct SubmitSemaphoresWaitBuilder<'a> {
    semaphores: SmallVec<[&'a Semaphore; 8]>,
}

impl<'a> SubmitSemaphoresWaitBuilder<'a> {
    /// Builds a new empty `SubmitSemaphoresWaitBuilder`.
    #[inline]
    pub fn new() -> SubmitSemaphoresWaitBuilder<'a> {
        SubmitSemaphoresWaitBuilder {
            semaphores: SmallVec::new(),
        }
    }

    /// Adds an operation that waits on a semaphore.
    ///
    /// The semaphore must be signaled by a previous submission.
    #[inline]
    pub unsafe fn add_wait_semaphore(&mut self, semaphore: &'a Semaphore) {
        self.semaphores.push(semaphore);
    }

    /// Merges this builder with another builder.
    #[inline]
    pub fn merge(&mut self, mut other: SubmitSemaphoresWaitBuilder<'a>) {
        self.semaphores.extend(other.semaphores.drain(..));
    }
}

impl<'a> From<SubmitSemaphoresWaitBuilder<'a>> for SubmitCommandBufferBuilder<'a> {
    #[inline]
    fn from(mut val: SubmitSemaphoresWaitBuilder<'a>) -> Self {
        unsafe {
            let mut builder = SubmitCommandBufferBuilder::new();
            for sem in val.semaphores.drain(..) {
                builder.add_wait_semaphore(
                    sem,
                    PipelineStages {
                        // TODO: correct stages ; hard
                        all_commands: true,
                        ..PipelineStages::empty()
                    },
                );
            }
            builder
        }
    }
}

impl<'a> From<SubmitSemaphoresWaitBuilder<'a>> for SubmitPresentBuilder<'a> {
    #[inline]
    fn from(mut val: SubmitSemaphoresWaitBuilder<'a>) -> Self {
        unsafe {
            let mut builder = SubmitPresentBuilder::new();
            for sem in val.semaphores.drain(..) {
                builder.add_wait_semaphore(sem);
            }
            builder
        }
    }
}
