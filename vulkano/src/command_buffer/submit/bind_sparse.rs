// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::sys::UnsafeBuffer;
use crate::check_errors;
use crate::device::Queue;
use crate::image::sys::UnsafeImage;
use crate::memory::DeviceMemory;
use crate::sync::Fence;
use crate::sync::Semaphore;
use crate::DeviceSize;
use crate::Error;
use crate::OomError;
use crate::SynchronizedVulkanObject;
use crate::VulkanObject;
use smallvec::SmallVec;
use std::error;
use std::fmt;
use std::marker::PhantomData;

// TODO: correctly implement Debug on all the structs of this module

/// Prototype for a submission that binds sparse memory.
// TODO: example here
pub struct SubmitBindSparseBuilder<'a> {
    infos: SmallVec<[SubmitBindSparseBatchBuilder<'a>; 1]>,
    fence: ash::vk::Fence,
}

impl<'a> SubmitBindSparseBuilder<'a> {
    /// Builds a new empty `SubmitBindSparseBuilder`.
    #[inline]
    pub fn new() -> SubmitBindSparseBuilder<'a> {
        SubmitBindSparseBuilder {
            infos: SmallVec::new(),
            fence: ash::vk::Fence::null(),
        }
    }

    /// Adds a batch to the command.
    ///
    /// Batches start execution in order, but can finish in a different order. In other words any
    /// wait semaphore added to a batch will apply to further batches as well, but when a semaphore
    /// is signalled, it does **not** mean that previous batches have been completed.
    #[inline]
    pub fn add(&mut self, builder: SubmitBindSparseBatchBuilder<'a>) {
        self.infos.push(builder);
    }

    /// Returns true if this builder will signal a fence when submitted.
    ///
    /// # Example
    ///
    /// ```
    /// use vulkano::command_buffer::submit::SubmitBindSparseBuilder;
    /// use vulkano::sync::Fence;
    /// # let device: std::sync::Arc<vulkano::device::Device> = return;
    ///
    /// unsafe {
    ///     let fence = Fence::from_pool(device.clone()).unwrap();
    ///
    ///     let mut builder = SubmitBindSparseBuilder::new();
    ///     assert!(!builder.has_fence());
    ///     builder.set_fence_signal(&fence);
    ///     assert!(builder.has_fence());
    /// }
    /// ```
    #[inline]
    pub fn has_fence(&self) -> bool {
        self.fence != ash::vk::Fence::null()
    }

    /// Adds an operation that signals a fence after this submission ends.
    ///
    /// # Example
    ///
    /// ```
    /// use std::time::Duration;
    /// use vulkano::command_buffer::submit::SubmitBindSparseBuilder;
    /// use vulkano::sync::Fence;
    /// # let device: std::sync::Arc<vulkano::device::Device> = return;
    /// # let queue: std::sync::Arc<vulkano::device::Queue> = return;
    ///
    /// unsafe {
    ///     let fence = Fence::from_pool(device.clone()).unwrap();
    ///
    ///     let mut builder = SubmitBindSparseBuilder::new();
    ///     builder.set_fence_signal(&fence);
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
    pub unsafe fn set_fence_signal(&mut self, fence: &'a Fence) {
        self.fence = fence.internal_object();
    }

    /// Attempts to merge this builder with another one.
    ///
    /// If both builders have a fence already set, then this function will return `other` as an
    /// error.
    #[inline]
    pub fn merge(
        &mut self,
        other: SubmitBindSparseBuilder<'a>,
    ) -> Result<(), SubmitBindSparseBuilder<'a>> {
        if self.fence != ash::vk::Fence::null() && other.fence != ash::vk::Fence::null() {
            return Err(other);
        }

        self.infos.extend(other.infos.into_iter());
        Ok(())
    }

    /// Submits the command. Calls `vkQueueBindSparse`.
    pub fn submit(self, queue: &Queue) -> Result<(), SubmitBindSparseError> {
        unsafe {
            debug_assert!(queue.family().supports_sparse_binding());

            let fns = queue.device().fns();
            let queue = queue.internal_object_guard();

            // We start by storing all the `VkSparseBufferMemoryBindInfo`s of the whole command
            // in the same collection.
            let buffer_binds_storage: SmallVec<[_; 4]> = self
                .infos
                .iter()
                .flat_map(|infos| infos.buffer_binds.iter())
                .map(|buf_bind| ash::vk::SparseBufferMemoryBindInfo {
                    buffer: buf_bind.buffer,
                    bind_count: buf_bind.binds.len() as u32,
                    p_binds: buf_bind.binds.as_ptr(),
                })
                .collect();

            // Same for all the `VkSparseImageOpaqueMemoryBindInfo`s.
            let image_opaque_binds_storage: SmallVec<[_; 4]> = self
                .infos
                .iter()
                .flat_map(|infos| infos.image_opaque_binds.iter())
                .map(|img_bind| ash::vk::SparseImageOpaqueMemoryBindInfo {
                    image: img_bind.image,
                    bind_count: img_bind.binds.len() as u32,
                    p_binds: img_bind.binds.as_ptr(),
                })
                .collect();

            // And finally the `VkSparseImageMemoryBindInfo`s.
            let image_binds_storage: SmallVec<[_; 4]> = self
                .infos
                .iter()
                .flat_map(|infos| infos.image_binds.iter())
                .map(|img_bind| ash::vk::SparseImageMemoryBindInfo {
                    image: img_bind.image,
                    bind_count: img_bind.binds.len() as u32,
                    p_binds: img_bind.binds.as_ptr(),
                })
                .collect();

            // Now building the collection of `VkBindSparseInfo`s.
            let bs_infos = {
                let mut bs_infos: SmallVec<[_; 4]> = SmallVec::new();

                // Since we stores all the bind infos contiguously, we keep track of the current
                // offset within these containers.
                let mut next_buffer_bind = 0;
                let mut next_image_opaque_bind = 0;
                let mut next_image_bind = 0;

                for builder in self.infos.iter() {
                    bs_infos.push(ash::vk::BindSparseInfo {
                        wait_semaphore_count: builder.wait_semaphores.len() as u32,
                        p_wait_semaphores: builder.wait_semaphores.as_ptr(),
                        buffer_bind_count: builder.buffer_binds.len() as u32,
                        p_buffer_binds: if next_buffer_bind != 0 {
                            // We need that `if` because `.as_ptr().offset(0)` is technically UB.
                            buffer_binds_storage.as_ptr().offset(next_buffer_bind)
                        } else {
                            buffer_binds_storage.as_ptr()
                        },
                        image_opaque_bind_count: builder.image_opaque_binds.len() as u32,
                        p_image_opaque_binds: if next_image_opaque_bind != 0 {
                            // We need that `if` because `.as_ptr().offset(0)` is technically UB.
                            image_opaque_binds_storage
                                .as_ptr()
                                .offset(next_image_opaque_bind)
                        } else {
                            image_opaque_binds_storage.as_ptr()
                        },
                        image_bind_count: builder.image_binds.len() as u32,
                        p_image_binds: if next_image_bind != 0 {
                            // We need that `if` because `.as_ptr().offset(0)` is technically UB.
                            image_binds_storage.as_ptr().offset(next_image_bind)
                        } else {
                            image_binds_storage.as_ptr()
                        },
                        signal_semaphore_count: builder.signal_semaphores.len() as u32,
                        p_signal_semaphores: builder.signal_semaphores.as_ptr(),
                        ..Default::default()
                    });

                    next_buffer_bind += builder.buffer_binds.len() as isize;
                    next_image_opaque_bind += builder.image_opaque_binds.len() as isize;
                    next_image_bind += builder.image_binds.len() as isize;
                }

                // If these assertions fail, then there's something wrong in the code above.
                debug_assert_eq!(next_buffer_bind as usize, buffer_binds_storage.len());
                debug_assert_eq!(
                    next_image_opaque_bind as usize,
                    image_opaque_binds_storage.len()
                );
                debug_assert_eq!(next_image_bind as usize, image_binds_storage.len());

                bs_infos
            };

            // Finally executing the command.
            check_errors(fns.v1_0.queue_bind_sparse(
                *queue,
                bs_infos.len() as u32,
                bs_infos.as_ptr(),
                self.fence,
            ))?;
            Ok(())
        }
    }
}

impl<'a> fmt::Debug for SubmitBindSparseBuilder<'a> {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Bind sparse operation>")
    }
}

/// A single batch of a sparse bind operation.
pub struct SubmitBindSparseBatchBuilder<'a> {
    wait_semaphores: SmallVec<[ash::vk::Semaphore; 8]>,
    buffer_binds: SmallVec<[SubmitBindSparseBufferBindBuilder<'a>; 2]>,
    image_opaque_binds: SmallVec<[SubmitBindSparseImageOpaqueBindBuilder<'a>; 2]>,
    image_binds: SmallVec<[SubmitBindSparseImageBindBuilder<'a>; 2]>,
    signal_semaphores: SmallVec<[ash::vk::Semaphore; 8]>,
    marker: PhantomData<&'a ()>,
}

impl<'a> SubmitBindSparseBatchBuilder<'a> {
    /// Builds a new empty `SubmitBindSparseBatchBuilder`.
    #[inline]
    pub fn new() -> SubmitBindSparseBatchBuilder<'a> {
        SubmitBindSparseBatchBuilder {
            wait_semaphores: SmallVec::new(),
            buffer_binds: SmallVec::new(),
            image_opaque_binds: SmallVec::new(),
            image_binds: SmallVec::new(),
            signal_semaphores: SmallVec::new(),
            marker: PhantomData,
        }
    }

    /// Adds an operation that binds memory to a buffer.
    pub fn add_buffer(&mut self, cmd: SubmitBindSparseBufferBindBuilder<'a>) {
        self.buffer_binds.push(cmd);
    }

    /// Adds an operation that binds memory to an opaque image.
    pub fn add_image_opaque(&mut self, cmd: SubmitBindSparseImageOpaqueBindBuilder<'a>) {
        self.image_opaque_binds.push(cmd);
    }

    /// Adds an operation that binds memory to an image.
    pub fn add_image(&mut self, cmd: SubmitBindSparseImageBindBuilder<'a>) {
        self.image_binds.push(cmd);
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
    pub unsafe fn add_wait_semaphore(&mut self, semaphore: &'a Semaphore) {
        self.wait_semaphores.push(semaphore.internal_object());
    }

    /// Returns the number of semaphores to signal.
    ///
    /// In other words, this is the number of times `add_signal_semaphore` has been called.
    #[inline]
    pub fn num_signal_semaphores(&self) -> usize {
        self.signal_semaphores.len()
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
    pub unsafe fn add_signal_semaphore(&mut self, semaphore: &'a Semaphore) {
        self.signal_semaphores.push(semaphore.internal_object());
    }
}

pub struct SubmitBindSparseBufferBindBuilder<'a> {
    buffer: ash::vk::Buffer,
    binds: SmallVec<[ash::vk::SparseMemoryBind; 1]>,
    marker: PhantomData<&'a ()>,
}

impl<'a> SubmitBindSparseBufferBindBuilder<'a> {
    ///
    /// # Safety
    ///
    /// - `buffer` must be a buffer with sparse binding enabled.
    pub unsafe fn new(buffer: &'a UnsafeBuffer) -> SubmitBindSparseBufferBindBuilder {
        SubmitBindSparseBufferBindBuilder {
            buffer: buffer.internal_object(),
            binds: SmallVec::new(),
            marker: PhantomData,
        }
    }

    pub unsafe fn add_bind(
        &mut self,
        offset: DeviceSize,
        size: DeviceSize,
        memory: &DeviceMemory,
        memory_offset: DeviceSize,
    ) {
        self.binds.push(ash::vk::SparseMemoryBind {
            resource_offset: offset,
            size,
            memory: memory.internal_object(),
            memory_offset,
            flags: ash::vk::SparseMemoryBindFlags::empty(), // Flags are only relevant for images.
        });
    }

    pub unsafe fn add_unbind(&mut self, offset: DeviceSize, size: DeviceSize) {
        self.binds.push(ash::vk::SparseMemoryBind {
            resource_offset: offset,
            size,
            memory: ash::vk::DeviceMemory::null(),
            memory_offset: 0,
            flags: ash::vk::SparseMemoryBindFlags::empty(),
        });
    }
}

pub struct SubmitBindSparseImageOpaqueBindBuilder<'a> {
    image: ash::vk::Image,
    binds: SmallVec<[ash::vk::SparseMemoryBind; 1]>,
    marker: PhantomData<&'a ()>,
}

impl<'a> SubmitBindSparseImageOpaqueBindBuilder<'a> {
    ///
    /// # Safety
    ///
    /// - `image` must be an image with sparse binding enabled.
    pub unsafe fn new(image: &'a UnsafeImage) -> SubmitBindSparseImageOpaqueBindBuilder {
        SubmitBindSparseImageOpaqueBindBuilder {
            image: image.internal_object(),
            binds: SmallVec::new(),
            marker: PhantomData,
        }
    }

    pub unsafe fn add_bind(
        &mut self,
        offset: DeviceSize,
        size: DeviceSize,
        memory: &DeviceMemory,
        memory_offset: DeviceSize,
        bind_metadata: bool,
    ) {
        self.binds.push(ash::vk::SparseMemoryBind {
            resource_offset: offset,
            size,
            memory: memory.internal_object(),
            memory_offset,
            flags: if bind_metadata {
                ash::vk::SparseMemoryBindFlags::METADATA
            } else {
                ash::vk::SparseMemoryBindFlags::empty()
            },
        });
    }

    pub unsafe fn add_unbind(&mut self, offset: DeviceSize, size: DeviceSize) {
        self.binds.push(ash::vk::SparseMemoryBind {
            resource_offset: offset,
            size,
            memory: ash::vk::DeviceMemory::null(),
            memory_offset: 0,
            flags: ash::vk::SparseMemoryBindFlags::empty(), // TODO: is that relevant?
        });
    }
}

pub struct SubmitBindSparseImageBindBuilder<'a> {
    image: ash::vk::Image,
    binds: SmallVec<[ash::vk::SparseImageMemoryBind; 1]>,
    marker: PhantomData<&'a ()>,
}

impl<'a> SubmitBindSparseImageBindBuilder<'a> {
    ///
    /// # Safety
    ///
    /// - `image` must be an image with sparse binding enabled.
    pub unsafe fn new(image: &'a UnsafeImage) -> SubmitBindSparseImageBindBuilder {
        SubmitBindSparseImageBindBuilder {
            image: image.internal_object(),
            binds: SmallVec::new(),
            marker: PhantomData,
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

impl error::Error for SubmitBindSparseError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            SubmitBindSparseError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for SubmitBindSparseError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                SubmitBindSparseError::OomError(_) => "not enough memory",
                SubmitBindSparseError::DeviceLost => "the connection to the device has been lost",
            }
        )
    }
}

impl From<Error> for SubmitBindSparseError {
    #[inline]
    fn from(err: Error) -> SubmitBindSparseError {
        match err {
            err @ Error::OutOfHostMemory => SubmitBindSparseError::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => SubmitBindSparseError::OomError(OomError::from(err)),
            Error::DeviceLost => SubmitBindSparseError::DeviceLost,
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}
