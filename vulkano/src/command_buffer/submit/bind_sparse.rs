// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    buffer::BufferAccess,
    device::Queue,
    image::ImageAccess,
    memory::{
        BindSparseInfo, DeviceMemory, SparseBufferMemoryBind, SparseImageMemoryBind,
        SparseImageOpaqueMemoryBind,
    },
    sync::{Fence, Semaphore},
    DeviceSize, OomError, VulkanError,
};
use smallvec::SmallVec;
use std::{
    error::Error,
    fmt::{Debug, Display, Error as FmtError, Formatter},
    sync::Arc,
};

// TODO: correctly implement Debug on all the structs of this module

/// Prototype for a submission that binds sparse memory.
// TODO: example here
#[derive(Debug)]
pub struct SubmitBindSparseBuilder {
    bind_infos: SmallVec<[BindSparseInfo; 1]>,
    fence: Option<Arc<Fence>>,
}

impl SubmitBindSparseBuilder {
    /// Builds a new empty `SubmitBindSparseBuilder`.
    #[inline]
    pub fn new() -> Self {
        SubmitBindSparseBuilder {
            bind_infos: SmallVec::new(),
            fence: None,
        }
    }

    /// Adds a batch to the command.
    ///
    /// Batches start execution in order, but can finish in a different order. In other words any
    /// wait semaphore added to a batch will apply to further batches as well, but when a semaphore
    /// is signalled, it does **not** mean that previous batches have been completed.
    #[inline]
    pub fn add(&mut self, builder: SubmitBindSparseBatchBuilder) {
        self.bind_infos.push(builder.bind_info);
    }

    /// Returns true if this builder will signal a fence when submitted.
    ///
    /// # Example
    ///
    /// ```
    /// use vulkano::command_buffer::submit::SubmitBindSparseBuilder;
    /// use vulkano::sync::Fence;
    /// # use std::sync::Arc;
    /// # let device: std::sync::Arc<vulkano::device::Device> = return;
    ///
    /// unsafe {
    ///     let fence = Arc::new(Fence::from_pool(device.clone()).unwrap());
    ///
    ///     let mut builder = SubmitBindSparseBuilder::new();
    ///     assert!(!builder.has_fence());
    ///     builder.set_fence_signal(fence);
    ///     assert!(builder.has_fence());
    /// }
    /// ```
    #[inline]
    pub fn has_fence(&self) -> bool {
        self.fence.is_some()
    }

    /// Adds an operation that signals a fence after this submission ends.
    ///
    /// # Example
    ///
    /// ```
    /// use std::time::Duration;
    /// use vulkano::command_buffer::submit::SubmitBindSparseBuilder;
    /// use vulkano::sync::Fence;
    /// # use std::sync::Arc;
    /// # let device: std::sync::Arc<vulkano::device::Device> = return;
    /// # let queue: std::sync::Arc<vulkano::device::Queue> = return;
    ///
    /// unsafe {
    ///     let fence = Arc::new(Fence::from_pool(device.clone()).unwrap());
    ///
    ///     let mut builder = SubmitBindSparseBuilder::new();
    ///     builder.set_fence_signal(fence.clone());
    ///
    ///     builder.submit(&queue).unwrap();
    ///
    ///     // We must not destroy the fence before it is signaled.
    ///     fence.wait(None).unwrap();
    /// }
    /// ```
    ///
    /// # Safety
    ///
    /// - The fence must not be signaled at the time when you call `submit()`.
    ///
    /// - If you use the fence for multiple submissions, only one at a time must be executed by the
    ///   GPU. In other words, you must submit one, wait for the fence to be signaled, then reset
    ///   the fence, and then only submit the second.
    ///
    /// - If you submit this builder, the fence must be kept alive until it is signaled by the GPU.
    ///   Destroying the fence earlier is an undefined behavior.
    ///
    /// - The fence, buffers, images, and semaphores must all belong to the same device.
    ///
    #[inline]
    pub unsafe fn set_fence_signal(&mut self, fence: Arc<Fence>) {
        self.fence = Some(fence);
    }

    /// Attempts to merge this builder with another one.
    ///
    /// If both builders have a fence already set, then this function will return `other` as an
    /// error.
    #[inline]
    pub fn merge(&mut self, other: SubmitBindSparseBuilder) -> Result<(), SubmitBindSparseBuilder> {
        if self.fence.is_some() && other.fence.is_some() {
            return Err(other);
        }

        self.bind_infos.extend(other.bind_infos.into_iter());
        Ok(())
    }

    /// Submits the command. Calls `vkQueueBindSparse`.
    pub fn submit(self, queue: &Queue) -> Result<(), SubmitBindSparseError> {
        unsafe {
            debug_assert!(
                queue.device().physical_device().queue_family_properties()
                    [queue.queue_family_index() as usize]
                    .queue_flags
                    .sparse_binding
            );

            let mut queue_guard = queue.lock();
            Ok(queue_guard.bind_sparse_unchecked(self.bind_infos, self.fence)?)
        }
    }
}

/// A single batch of a sparse bind operation.
pub struct SubmitBindSparseBatchBuilder {
    bind_info: BindSparseInfo,
}

impl SubmitBindSparseBatchBuilder {
    /// Builds a new empty `SubmitBindSparseBatchBuilder`.
    #[inline]
    pub fn new() -> Self {
        SubmitBindSparseBatchBuilder {
            bind_info: Default::default(),
        }
    }

    /// Adds an operation that binds memory to a buffer.
    pub fn add_buffer(&mut self, cmd: SubmitBindSparseBufferBindBuilder) {
        self.bind_info
            .buffer_binds
            .push((cmd.buffer, cmd.binds.into_iter().collect()));
    }

    /// Adds an operation that binds memory to an opaque image.
    pub fn add_image_opaque(&mut self, cmd: SubmitBindSparseImageOpaqueBindBuilder) {
        self.bind_info
            .image_opaque_binds
            .push((cmd.image, cmd.binds.into_iter().collect()));
    }

    /// Adds an operation that binds memory to an image.
    pub fn add_image(&mut self, cmd: SubmitBindSparseImageBindBuilder) {
        self.bind_info
            .image_binds
            .push((cmd.image, cmd.binds.into_iter().collect()));
    }

    /// Adds a semaphore to be waited upon before the sparse binding is executed.
    ///
    /// # Safety
    ///
    /// - If you submit this builder, the semaphore must be kept alive until you are guaranteed
    ///   that the GPU has at least started executing the operation.
    ///
    /// - If you submit this builder, no other queue must be waiting on these semaphores. In other
    ///   words, each semaphore signal can only correspond to one semaphore wait.
    ///
    /// - If you submit this builder, the semaphores must be signaled when the queue execution
    ///   reaches this submission, or there must be one or more submissions in queues that are
    ///   going to signal these semaphores. In other words, you must not block the queue with
    ///   semaphores that can't get signaled.
    ///
    /// - The fence, buffers, images, and semaphores must all belong to the same device.
    ///
    #[inline]
    pub unsafe fn add_wait_semaphore(&mut self, semaphore: Arc<Semaphore>) {
        self.bind_info.wait_semaphores.push(semaphore);
    }

    /// Returns the number of semaphores to signal.
    ///
    /// In other words, this is the number of times `add_signal_semaphore` has been called.
    #[inline]
    pub fn num_signal_semaphores(&self) -> usize {
        self.bind_info.signal_semaphores.len()
    }

    /// Adds a semaphore that is going to be signaled at the end of the submission.
    ///
    /// # Safety
    ///
    /// - If you submit this builder, the semaphore must be kept alive until you are guaranteed
    ///   that the GPU has finished executing this submission.
    ///
    /// - The semaphore must be in the unsignaled state when queue execution reaches this
    ///   submission.
    ///
    /// - The fence, buffers, images, and semaphores must all belong to the same device.
    ///
    #[inline]
    pub unsafe fn add_signal_semaphore(&mut self, semaphore: Arc<Semaphore>) {
        self.bind_info.signal_semaphores.push(semaphore);
    }
}

pub struct SubmitBindSparseBufferBindBuilder {
    buffer: Arc<dyn BufferAccess>,
    binds: SmallVec<[SparseBufferMemoryBind; 1]>,
}

impl SubmitBindSparseBufferBindBuilder {
    ///
    /// # Safety
    ///
    /// - `buffer` must be a buffer with sparse binding enabled.
    pub unsafe fn new(buffer: Arc<dyn BufferAccess>) -> Self {
        SubmitBindSparseBufferBindBuilder {
            buffer,
            binds: SmallVec::new(),
        }
    }

    pub unsafe fn add_bind(
        &mut self,
        resource_offset: DeviceSize,
        size: DeviceSize,
        memory: Arc<DeviceMemory>,
        memory_offset: DeviceSize,
    ) {
        self.binds.push(SparseBufferMemoryBind {
            resource_offset,
            size,
            memory: Some((memory, memory_offset)),
        });
    }

    pub unsafe fn add_unbind(&mut self, resource_offset: DeviceSize, size: DeviceSize) {
        self.binds.push(SparseBufferMemoryBind {
            resource_offset,
            size,
            memory: None,
        });
    }
}

pub struct SubmitBindSparseImageOpaqueBindBuilder {
    image: Arc<dyn ImageAccess>,
    binds: SmallVec<[SparseImageOpaqueMemoryBind; 1]>,
}

impl SubmitBindSparseImageOpaqueBindBuilder {
    ///
    /// # Safety
    ///
    /// - `image` must be an image with sparse binding enabled.
    pub unsafe fn new(image: Arc<dyn ImageAccess>) -> Self {
        SubmitBindSparseImageOpaqueBindBuilder {
            image,
            binds: SmallVec::new(),
        }
    }

    pub unsafe fn add_bind(
        &mut self,
        resource_offset: DeviceSize,
        size: DeviceSize,
        memory: Arc<DeviceMemory>,
        memory_offset: DeviceSize,
        metadata: bool,
    ) {
        self.binds.push(SparseImageOpaqueMemoryBind {
            resource_offset,
            size,
            memory: Some((memory, memory_offset)),
            metadata,
        });
    }

    pub unsafe fn add_unbind(&mut self, resource_offset: DeviceSize, size: DeviceSize) {
        self.binds.push(SparseImageOpaqueMemoryBind {
            resource_offset,
            size,
            memory: None,
            metadata: false,
        });
    }
}

pub struct SubmitBindSparseImageBindBuilder {
    image: Arc<dyn ImageAccess>,
    binds: SmallVec<[SparseImageMemoryBind; 1]>,
}

impl SubmitBindSparseImageBindBuilder {
    ///
    /// # Safety
    ///
    /// - `image` must be an image with sparse binding enabled.
    pub unsafe fn new(image: Arc<dyn ImageAccess>) -> Self {
        SubmitBindSparseImageBindBuilder {
            image,
            binds: SmallVec::new(),
        }
    }

    // TODO: finish
}

/// Error that can happen when submitting the present prototype.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SubmitBindSparseError {
    /// Not enough memory.
    OomError(OomError),

    /// The connection to the device has been lost.
    DeviceLost,
}

impl Error for SubmitBindSparseError {
    #[inline]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match *self {
            SubmitBindSparseError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl Display for SubmitBindSparseError {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        write!(
            f,
            "{}",
            match *self {
                SubmitBindSparseError::OomError(_) => "not enough memory",
                SubmitBindSparseError::DeviceLost => "the connection to the device has been lost",
            }
        )
    }
}

impl From<VulkanError> for SubmitBindSparseError {
    #[inline]
    fn from(err: VulkanError) -> SubmitBindSparseError {
        match err {
            err @ VulkanError::OutOfHostMemory => {
                SubmitBindSparseError::OomError(OomError::from(err))
            }
            err @ VulkanError::OutOfDeviceMemory => {
                SubmitBindSparseError::OomError(OomError::from(err))
            }
            VulkanError::DeviceLost => SubmitBindSparseError::DeviceLost,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}
