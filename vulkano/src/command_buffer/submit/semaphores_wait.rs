// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use smallvec::SmallVec;

use command_buffer::submit::SubmitCommandBufferBuilder;
use command_buffer::submit::SubmitPresentBuilder;
use sync::PipelineStages;
use sync::Semaphore;

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
        SubmitSemaphoresWaitBuilder { semaphores: SmallVec::new() }
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

impl<'a> Into<SubmitCommandBufferBuilder<'a>> for SubmitSemaphoresWaitBuilder<'a> {
    #[inline]
    fn into(mut self) -> SubmitCommandBufferBuilder<'a> {
        unsafe {
            let mut builder = SubmitCommandBufferBuilder::new();
            for sem in self.semaphores.drain(..) {
                builder.add_wait_semaphore(sem,
                                           PipelineStages {
                                               // TODO: correct stages ; hard
                                               all_commands: true,
                                               ..PipelineStages::none()
                                           });
            }
            builder
        }
    }
}

impl<'a> Into<SubmitPresentBuilder<'a>> for SubmitSemaphoresWaitBuilder<'a> {
    #[inline]
    fn into(mut self) -> SubmitPresentBuilder<'a> {
        unsafe {
            let mut builder = SubmitPresentBuilder::new();
            for sem in self.semaphores.drain(..) {
                builder.add_wait_semaphore(sem);
            }
            builder
        }
    }
}
