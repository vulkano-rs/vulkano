// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Low-level builders that allow submitting an operation to a queue.
//!
//! In order to submit an operation to the GPU, you must use one of the builder structs of this
//! module. These structs are low-level and unsafe, and are mostly used to implement other parts
//! of vulkano, so you are encouraged to not use them directly.

pub use self::bind_sparse::SubmitBindSparseBatchBuilder;
pub use self::bind_sparse::SubmitBindSparseBufferBindBuilder;
pub use self::bind_sparse::SubmitBindSparseBuilder;
pub use self::bind_sparse::SubmitBindSparseError;
pub use self::bind_sparse::SubmitBindSparseImageBindBuilder;
pub use self::bind_sparse::SubmitBindSparseImageOpaqueBindBuilder;
pub use self::queue_present::SubmitPresentBuilder;
pub use self::queue_present::SubmitPresentError;
pub use self::queue_submit::SubmitCommandBufferBuilder;
pub use self::queue_submit::SubmitCommandBufferError;
pub use self::semaphores_wait::SubmitSemaphoresWaitBuilder;

mod bind_sparse;
mod queue_present;
mod queue_submit;
mod semaphores_wait;

/// Contains all the possible submission builders.
#[derive(Debug)]
pub enum SubmitAnyBuilder<'a> {
    Empty,
    SemaphoresWait(SubmitSemaphoresWaitBuilder<'a>),
    CommandBuffer(SubmitCommandBufferBuilder<'a>),
    QueuePresent(SubmitPresentBuilder<'a>),
    BindSparse(SubmitBindSparseBuilder<'a>),
}

impl<'a> SubmitAnyBuilder<'a> {
    /// Returns true if equal to `SubmitAnyBuilder::Empty`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            &SubmitAnyBuilder::Empty => true,
            _ => false,
        }
    }
}
