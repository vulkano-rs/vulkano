// FIXME:
#![allow(unused)]
#![forbid(unsafe_op_in_unsafe_fn)]

use concurrent_slotmap::SlotId;
use resource::{BufferRange, BufferState, DeathRow, ImageState, Resources, SwapchainState};
use std::{
    any::{Any, TypeId},
    cell::Cell,
    cmp,
    error::Error,
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{Deref, DerefMut, Range, RangeBounds},
    thread,
};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferMemory, Subbuffer},
    command_buffer::sys::{RawCommandBuffer, RawRecordingCommandBuffer},
    image::Image,
    memory::{
        allocator::{align_down, align_up},
        DeviceAlignment, MappedMemoryRange, ResourceMemory,
    },
    swapchain::Swapchain,
    DeviceSize, ValidationError, VulkanError,
};

pub mod resource;

/// A task represents a unit of work to be recorded to a command buffer.
pub trait Task: Any + Send + Sync {
    type World: ?Sized;

    // Potentially TODO:
    // fn update(&mut self, ...) {}

    /// Executes the task, which should record its commands using the provided context.
    ///
    /// # Safety
    ///
    /// - Every subresource in the [task's input/output interface] must not be written to
    ///   concurrently in any other tasks during execution on the device.
    /// - Every subresource in the task's input/output interface, if it's a [host access], must not
    ///   be written to concurrently in any other tasks during execution on the host.
    /// - Every subresource in the task's input interface, if it's an [image access], must have had
    ///   its layout transitioned to the layout specified in the interface.
    /// - Every subresource in the task's input interface, if the resource's [sharing mode] is
    ///   exclusive, must be currently owned by the queue family the task is executing on.
    unsafe fn execute(&self, tcx: &mut TaskContext<'_>, world: &Self::World) -> TaskResult;
}

impl<W: ?Sized + 'static> dyn Task<World = W> {
    /// Returns `true` if `self` is of type `T`.
    #[inline]
    pub fn is<T: Task<World = W>>(&self) -> bool {
        self.type_id() == TypeId::of::<T>()
    }

    /// Returns a reference to the inner value if it is of type `T`, or returns `None` otherwise.
    #[inline]
    pub fn downcast_ref<T: Task<World = W>>(&self) -> Option<&T> {
        if self.is::<T>() {
            // SAFETY: We just checked that the type is correct.
            Some(unsafe { self.downcast_unchecked_ref() })
        } else {
            None
        }
    }

    /// Returns a reference to the inner value if it is of type `T`, or returns `None` otherwise.
    #[inline]
    pub fn downcast_mut<T: Task<World = W>>(&mut self) -> Option<&mut T> {
        if self.is::<T>() {
            // SAFETY: We just checked that the type is correct.
            Some(unsafe { self.downcast_unchecked_mut() })
        } else {
            None
        }
    }

    /// Returns a reference to the inner value without checking if it is of type `T`.
    ///
    /// # Safety
    ///
    /// `self` must be of type `T`.
    #[inline]
    pub unsafe fn downcast_unchecked_ref<T: Task<World = W>>(&self) -> &T {
        // SAFETY: The caller must guarantee that the type is correct.
        unsafe { &*<*const dyn Task<World = W>>::cast::<T>(self) }
    }

    /// Returns a reference to the inner value without checking if it is of type `T`.
    ///
    /// # Safety
    ///
    /// `self` must be of type `T`.
    #[inline]
    pub unsafe fn downcast_unchecked_mut<T: Task<World = W>>(&mut self) -> &mut T {
        // SAFETY: The caller must guarantee that the type is correct.
        unsafe { &mut *<*mut dyn Task<World = W>>::cast::<T>(self) }
    }
}

impl<W: ?Sized> fmt::Debug for dyn Task<World = W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Task").finish_non_exhaustive()
    }
}

/// The context of a task.
///
/// This gives you access to the current command buffer, resources, as well as resource cleanup.
pub struct TaskContext<'a> {
    resources: &'a Resources,
    death_row: Cell<Option<&'a mut DeathRow>>,
    current_command_buffer: Cell<Option<&'a mut RawRecordingCommandBuffer>>,
    command_buffers: Cell<Option<&'a mut Vec<RawCommandBuffer>>>,
}

impl<'a> TaskContext<'a> {
    /// Returns the current raw command buffer for the task.
    ///
    /// While this method is safe, using the command buffer isn't. You must guarantee that any
    /// subresources you use while recording commands are either accounted for in the [task's
    /// input/output interface], or that those subresources don't require any synchronization
    /// (including layout transitions and queue family ownership transfers), or that no other task
    /// is accessing the subresources at the same time without appropriate synchronization.
    ///
    /// # Panics
    ///
    /// - Panics if called more than once.
    // TODO: We could alternatively to ^ pass two parameters to `Task::execute`.
    #[inline]
    pub fn raw_command_buffer(&self) -> &'a mut RawRecordingCommandBuffer {
        self.current_command_buffer
            .take()
            .expect("`TaskContext::raw_command_buffer` can only be called once")
    }

    /// Pushes a command buffer into the list of command buffers to be executed on the queue.
    ///
    /// All command buffers will be executed in the order in which they are pushed after the task
    /// has finished execution. That means in particular, that commands recorded by the task will
    /// start execution before execution of any pushed command buffers starts.
    ///
    /// # Safety
    ///
    /// The same safety preconditions apply as outlined in the [`raw_command_buffer`] method. Since
    /// the command buffer will be executed on the same queue right after the current command
    /// buffer, without any added synchronization, it must be safe to do so. The given command
    /// buffer must not do any accesses not accounted for in the [task's input/output interface],
    /// or ensure that such accesses are appropriately synchronized.
    ///
    /// [`raw_command_buffer`]: Self::raw_command_buffer
    #[inline]
    pub unsafe fn push_command_buffer(&self, command_buffer: RawCommandBuffer) {
        let vec = self.command_buffers.take().unwrap();
        vec.push(command_buffer);
        self.command_buffers.set(Some(vec));
    }

    /// Extends the list of command buffers to be executed on the queue.
    ///
    /// This function behaves identically to the [`push_command_buffer`] method, except that it
    /// pushes all command buffers from the given iterator in order.
    ///
    /// # Safety
    ///
    /// See the [`push_command_buffer`] method for the safety preconditions.
    ///
    /// [`push_command_buffer`]: Self::push_command_buffer
    #[inline]
    pub unsafe fn extend_command_buffers(
        &self,
        command_buffers: impl IntoIterator<Item = RawCommandBuffer>,
    ) {
        let vec = self.command_buffers.take().unwrap();
        vec.extend(command_buffers);
        self.command_buffers.set(Some(vec));
    }

    /// Returns the buffer corresponding to `id`, or returns an error if it isn't present.
    #[inline]
    pub fn buffer(&self, id: Id<Buffer>) -> TaskResult<&'a BufferState> {
        // SAFETY: Ensured by the caller of `Task::execute`.
        Ok(unsafe { self.resources.buffer_unprotected(id) }?)
    }

    /// Returns the image corresponding to `id`, or returns an error if it isn't present.
    #[inline]
    pub fn image(&self, id: Id<Image>) -> TaskResult<&'a ImageState> {
        // SAFETY: Ensured by the caller of `Task::execute`.
        Ok(unsafe { self.resources.image_unprotected(id) }?)
    }

    /// Returns the swapchain corresponding to `id`, or returns an error if it isn't present.
    #[inline]
    pub fn swapchain(&self, id: Id<Swapchain>) -> TaskResult<&'a SwapchainState> {
        // SAFETY: Ensured by the caller of `Task::execute`.
        Ok(unsafe { self.resources.swapchain_unprotected(id) }?)
    }

    /// Returns the `Resources` collection.
    #[inline]
    pub fn resources(&self) -> &'a Resources {
        self.resources
    }

    /// Tries to get read access to a portion of the buffer corresponding to `id`.
    ///
    /// If host read access of the portion of the buffer is not accounted for in the [task's
    /// input/output interface], this method will return an error.
    ///
    /// If the memory backing the buffer is not [host-coherent], then this method will check a
    /// range that is potentially larger than the given range, because the range given to
    /// [`invalidate_range`] must be aligned to the [`non_coherent_atom_size`]. This means that for
    /// example if your Vulkan implementation reports an atom size of 64, and you tried to put 2
    /// subbuffers of size 32 in the same buffer, one at offset 0 and one at offset 32, while the
    /// buffer is backed by non-coherent memory, then invalidating one subbuffer would also
    /// invalidate the other subbuffer. This can lead to data races and is therefore not allowed.
    /// What you should do in that case is ensure that each subbuffer is aligned to the
    /// non-coherent atom size, so in this case one would be at offset 0 and the other at offset
    /// 64.
    ///
    /// If the memory backing the buffer is not managed by vulkano (i.e. the buffer was created
    /// by [`RawBuffer::assume_bound`]), then it can't be read using this method and an error will
    /// be returned.
    ///
    /// # Panics
    ///
    /// - Panics if the alignment of `T` is greater than 64.
    /// - Panics if [`Subbuffer::slice`] with the given `range` panics.
    /// - Panics if [`Subbuffer::reinterpret`] to the given `T` panics.
    ///
    /// [host-coherent]: vulkano::memory::MemoryPropertyFlags::HOST_COHERENT
    /// [`invalidate_range`]: vulkano::memory::ResourceMemory::invalidate_range
    /// [`non_coherent_atom_size`]: vulkano::device::DeviceProperties::non_coherent_atom_size
    /// [`RawBuffer::assume_bound`]: vulkano::buffer::sys::RawBuffer::assume_bound
    pub fn read_buffer<T: BufferContents + ?Sized>(
        &self,
        id: Id<Buffer>,
        range: impl RangeBounds<DeviceSize>,
    ) -> TaskResult<BufferReadGuard<'_, T>> {
        #[cold]
        unsafe fn invalidate_subbuffer(
            tcx: &TaskContext<'_>,
            subbuffer: &Subbuffer<[u8]>,
            allocation: &ResourceMemory,
            atom_size: DeviceAlignment,
        ) -> TaskResult {
            // This works because the memory allocator must align allocations to the non-coherent
            // atom size when the memory is host-visible but not host-coherent.
            let start = align_down(subbuffer.offset(), atom_size);
            let end = cmp::min(
                align_up(subbuffer.offset() + subbuffer.size(), atom_size),
                allocation.size(),
            );
            let range = Range { start, end };

            tcx.validate_read_buffer(subbuffer.buffer(), range.clone())?;

            let memory_range = MappedMemoryRange {
                offset: range.start,
                size: range.end - range.start,
                _ne: crate::NE,
            };

            // SAFETY:
            // - We checked that the task has read access to the subbuffer above.
            // - The caller must guarantee that the subbuffer falls within the mapped range of
            //   memory.
            // - We ensure that memory mappings are always aligned to the non-coherent atom size for
            //   non-host-coherent memory, therefore the subbuffer's range aligned to the
            //   non-coherent atom size must fall within the mapped range of the memory.
            unsafe { allocation.invalidate_range_unchecked(memory_range) }
                .map_err(HostAccessError::Invalidate)?;

            Ok(())
        }

        assert!(T::LAYOUT.alignment().as_devicesize() <= 64);

        let buffer = self.buffer(id)?.buffer();
        let subbuffer = Subbuffer::from(buffer.clone())
            .slice(range)
            .reinterpret::<T>();

        let allocation = match buffer.memory() {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => {
                todo!("`TaskContext::read_buffer` doesn't support sparse binding yet")
            }
            BufferMemory::External => {
                return Err(TaskError::HostAccess(HostAccessError::Unmanaged))
            }
            _ => unreachable!(),
        };

        let mapped_slice = subbuffer.mapped_slice().map_err(|err| match err {
            vulkano::sync::HostAccessError::NotHostMapped => HostAccessError::NotHostMapped,
            vulkano::sync::HostAccessError::OutOfMappedRange => HostAccessError::OutOfMappedRange,
            _ => unreachable!(),
        })?;

        let atom_size = allocation.atom_size();

        if let Some(atom_size) = atom_size {
            // SAFETY:
            // `subbuffer.mapped_slice()` didn't return an error, which means that the subbuffer
            // falls within the mapped range of the memory.
            unsafe { invalidate_subbuffer(self, subbuffer.as_bytes(), allocation, atom_size) }?;
        } else {
            let range = subbuffer.offset()..subbuffer.offset() + subbuffer.size();
            self.validate_write_buffer(buffer, range)?;
        }

        // SAFETY: We checked that the task has read access to the subbuffer above, which also
        // includes the guarantee that no other tasks can be writing the subbuffer on neither the
        // host nor the device. The same task cannot obtain another `BufferWriteGuard` to the
        // subbuffer because `TaskContext::write_buffer` requires a mutable reference.
        let data = unsafe { &*T::ptr_from_slice(mapped_slice) };

        Ok(BufferReadGuard { data })
    }

    fn validate_read_buffer(&self, _buffer: &Buffer, _range: BufferRange) -> TaskResult {
        todo!()
    }

    /// Gets read access to a portion of the buffer corresponding to `id` without checking if this
    /// access is accounted for in the [task's input/output interface].
    ///
    /// This method doesn't do any host cache control. If the memory backing the buffer is not
    /// [host-coherent], you must call [`invalidate_range`] in order for any device writes to be
    /// visible to the host, and must not forget that such flushes must be aligned to the
    /// [`non_coherent_atom_size`] and hence the aligned range must be accounted for in the task's
    /// input/output interface.
    ///
    /// If the memory backing the buffer is not managed by vulkano (i.e. the buffer was created
    /// by [`RawBuffer::assume_bound`]), then it can't be read using this method and an error will
    /// be returned.
    ///
    /// # Safety
    ///
    /// This access must be accounted for in the task's input/output interface.
    ///
    /// # Panics
    ///
    /// - Panics if the alignment of `T` is greater than 64.
    /// - Panics if [`Subbuffer::slice`] with the given `range` panics.
    /// - Panics if [`Subbuffer::reinterpret`] to the given `T` panics.
    ///
    /// [host-coherent]: vulkano::memory::MemoryPropertyFlags::HOST_COHERENT
    /// [`invalidate_range`]: vulkano::memory::ResourceMemory::invalidate_range
    /// [`non_coherent_atom_size`]: vulkano::device::DeviceProperties::non_coherent_atom_size
    /// [`RawBuffer::assume_bound`]: vulkano::buffer::sys::RawBuffer::assume_bound
    pub unsafe fn read_buffer_unchecked<T: BufferContents + ?Sized>(
        &self,
        id: Id<Buffer>,
        range: impl RangeBounds<DeviceSize>,
    ) -> TaskResult<&T> {
        assert!(T::LAYOUT.alignment().as_devicesize() <= 64);

        let buffer = self.buffer(id)?.buffer();
        let subbuffer = Subbuffer::from(buffer.clone())
            .slice(range)
            .reinterpret::<T>();

        match buffer.memory() {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => {
                todo!("`TaskContext::read_buffer_unchecked` doesn't support sparse binding yet");
            }
            BufferMemory::External => {
                return Err(TaskError::HostAccess(HostAccessError::Unmanaged));
            }
            _ => unreachable!(),
        };

        let mapped_slice = subbuffer.mapped_slice().map_err(|err| match err {
            vulkano::sync::HostAccessError::NotHostMapped => HostAccessError::NotHostMapped,
            vulkano::sync::HostAccessError::OutOfMappedRange => HostAccessError::OutOfMappedRange,
            _ => unreachable!(),
        })?;

        // SAFETY: The caller must ensure that access to the data is synchronized.
        let data = unsafe { &*T::ptr_from_slice(mapped_slice) };

        Ok(data)
    }

    /// Tries to get write access to a portion of the buffer corresponding to `id`.
    ///
    /// If host write access of the portion of the buffer is not accounted for in the [task's
    /// input/output interface], this method will return an error.
    ///
    /// If the memory backing the buffer is not [host-coherent], then this method will check a
    /// range that is potentially larger than the given range, because the range given to
    /// [`flush_range`] must be aligned to the [`non_coherent_atom_size`]. This means that for
    /// example if your Vulkan implementation reports an atom size of 64, and you tried to put 2
    /// subbuffers of size 32 in the same buffer, one at offset 0 and one at offset 32, while the
    /// buffer is backed by non-coherent memory, then invalidating one subbuffer would also
    /// invalidate the other subbuffer. This can lead to data races and is therefore not allowed.
    /// What you should do in that case is ensure that each subbuffer is aligned to the
    /// non-coherent atom size, so in this case one would be at offset 0 and the other at offset
    /// 64.
    ///
    /// If the memory backing the buffer is not managed by vulkano (i.e. the buffer was created
    /// by [`RawBuffer::assume_bound`]), then it can't be written using this method and an error
    /// will be returned.
    ///
    /// # Panics
    ///
    /// - Panics if the alignment of `T` is greater than 64.
    /// - Panics if [`Subbuffer::slice`] with the given `range` panics.
    /// - Panics if [`Subbuffer::reinterpret`] to the given `T` panics.
    ///
    /// [host-coherent]: vulkano::memory::MemoryPropertyFlags::HOST_COHERENT
    /// [`flush_range`]: vulkano::memory::ResourceMemory::flush_range
    /// [`non_coherent_atom_size`]: vulkano::device::DeviceProperties::non_coherent_atom_size
    /// [`RawBuffer::assume_bound`]: vulkano::buffer::sys::RawBuffer::assume_bound
    pub fn write_buffer<T: BufferContents + ?Sized>(
        &mut self,
        id: Id<Buffer>,
        range: impl RangeBounds<DeviceSize>,
    ) -> TaskResult<BufferWriteGuard<'_, T>> {
        #[cold]
        unsafe fn invalidate_subbuffer(
            tcx: &TaskContext<'_>,
            subbuffer: &Subbuffer<[u8]>,
            allocation: &ResourceMemory,
            atom_size: DeviceAlignment,
        ) -> TaskResult {
            // This works because the memory allocator must align allocations to the non-coherent
            // atom size when the memory is host-visible but not host-coherent.
            let start = align_down(subbuffer.offset(), atom_size);
            let end = cmp::min(
                align_up(subbuffer.offset() + subbuffer.size(), atom_size),
                allocation.size(),
            );
            let range = Range { start, end };

            tcx.validate_write_buffer(subbuffer.buffer(), range.clone())?;

            let memory_range = MappedMemoryRange {
                offset: range.start,
                size: range.end - range.start,
                _ne: crate::NE,
            };

            // SAFETY:
            // - We checked that the task has write access to the subbuffer above.
            // - The caller must guarantee that the subbuffer falls within the mapped range of
            //   memory.
            // - We ensure that memory mappings are always aligned to the non-coherent atom size for
            //   non-host-coherent memory, therefore the subbuffer's range aligned to the
            //   non-coherent atom size must fall within the mapped range of the memory.
            unsafe { allocation.invalidate_range_unchecked(memory_range) }
                .map_err(HostAccessError::Invalidate)?;

            Ok(())
        }

        assert!(T::LAYOUT.alignment().as_devicesize() <= 64);

        let buffer = self.buffer(id)?.buffer();
        let subbuffer = Subbuffer::from(buffer.clone())
            .slice(range)
            .reinterpret::<T>();

        let allocation = match buffer.memory() {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => {
                todo!("`TaskContext::write_buffer` doesn't support sparse binding yet");
            }
            BufferMemory::External => {
                return Err(TaskError::HostAccess(HostAccessError::Unmanaged));
            }
            _ => unreachable!(),
        };

        let mapped_slice = subbuffer.mapped_slice().map_err(|err| match err {
            vulkano::sync::HostAccessError::NotHostMapped => HostAccessError::NotHostMapped,
            vulkano::sync::HostAccessError::OutOfMappedRange => HostAccessError::OutOfMappedRange,
            _ => unreachable!(),
        })?;

        let atom_size = allocation.atom_size();

        if let Some(atom_size) = atom_size {
            // SAFETY:
            // `subbuffer.mapped_slice()` didn't return an error, which means that the subbuffer
            // falls within the mapped range of the memory.
            unsafe { invalidate_subbuffer(self, subbuffer.as_bytes(), allocation, atom_size) }?;
        } else {
            let range = subbuffer.offset()..subbuffer.offset() + subbuffer.size();
            self.validate_write_buffer(buffer, range)?;
        }

        // SAFETY: We checked that the task has write access to the subbuffer above, which also
        // includes the guarantee that no other tasks can be accessing the subbuffer on neither the
        // host nor the device. The same task cannot obtain another `BufferWriteGuard` to the
        // subbuffer because `TaskContext::write_buffer` requires a mutable reference.
        let data = unsafe { &mut *T::ptr_from_slice(mapped_slice) };

        Ok(BufferWriteGuard {
            subbuffer: subbuffer.into_bytes(),
            data,
            atom_size,
        })
    }

    fn validate_write_buffer(&self, _buffer: &Buffer, _range: BufferRange) -> TaskResult {
        todo!()
    }

    /// Gets write access to a portion of the buffer corresponding to `id` without checking if this
    /// access is accounted for in the [task's input/output interface].
    ///
    /// This method doesn't do any host cache control. If the memory backing the buffer is not
    /// [host-coherent], you must call [`flush_range`] in order for any writes to be available to
    /// the host memory domain, and must not forget that such flushes must be aligned to the
    /// [`non_coherent_atom_size`] and hence the aligned range must be accounted for in the task's
    /// input/output interface.
    ///
    /// If the memory backing the buffer is not managed by vulkano (i.e. the buffer was created
    /// by [`RawBuffer::assume_bound`]), then it can't be written using this method and an error
    /// will be returned.
    ///
    /// # Safety
    ///
    /// This access must be accounted for in the task's input/output interface.
    ///
    /// # Panics
    ///
    /// - Panics if the alignment of `T` is greater than 64.
    /// - Panics if [`Subbuffer::slice`] with the given `range` panics.
    /// - Panics if [`Subbuffer::reinterpret`] to the given `T` panics.
    ///
    /// [host-coherent]: vulkano::memory::MemoryPropertyFlags::HOST_COHERENT
    /// [`flush_range`]: vulkano::memory::ResourceMemory::flush_range
    /// [`non_coherent_atom_size`]: vulkano::device::DeviceProperties::non_coherent_atom_size
    /// [`RawBuffer::assume_bound`]: vulkano::buffer::sys::RawBuffer::assume_bound
    pub unsafe fn write_buffer_unchecked<T: BufferContents + ?Sized>(
        &mut self,
        id: Id<Buffer>,
        range: impl RangeBounds<DeviceSize>,
    ) -> TaskResult<&mut T> {
        assert!(T::LAYOUT.alignment().as_devicesize() <= 64);

        let buffer = self.buffer(id)?.buffer();
        let subbuffer = Subbuffer::from(buffer.clone())
            .slice(range)
            .reinterpret::<T>();

        match buffer.memory() {
            BufferMemory::Normal(a) => a,
            BufferMemory::Sparse => {
                todo!("`TaskContext::write_buffer_unchecked` doesn't support sparse binding yet");
            }
            BufferMemory::External => {
                return Err(TaskError::HostAccess(HostAccessError::Unmanaged));
            }
            _ => unreachable!(),
        };

        let mapped_slice = subbuffer.mapped_slice().map_err(|err| match err {
            vulkano::sync::HostAccessError::NotHostMapped => HostAccessError::NotHostMapped,
            vulkano::sync::HostAccessError::OutOfMappedRange => HostAccessError::OutOfMappedRange,
            _ => unreachable!(),
        })?;

        // SAFETY: The caller must ensure that access to the data is synchronized.
        let data = unsafe { &mut *T::ptr_from_slice(mapped_slice) };

        Ok(data)
    }

    /// Queues the destruction of the buffer corresponding to `id` after the destruction of the
    /// command buffer(s) for this task.
    // FIXME: unsafe
    #[inline]
    pub unsafe fn destroy_buffer(&self, id: Id<Buffer>) -> TaskResult {
        let state = unsafe { self.resources.remove_buffer(id) }?;
        let death_row = self.death_row.take().unwrap();
        // FIXME:
        death_row.push(state.buffer().clone());
        self.death_row.set(Some(death_row));

        Ok(())
    }

    /// Queues the destruction of the image corresponding to `id` after the destruction of the
    /// command buffer(s) for this task.
    // FIXME: unsafe
    #[inline]
    pub unsafe fn destroy_image(&self, id: Id<Image>) -> TaskResult {
        let state = unsafe { self.resources.remove_image(id) }?;
        let death_row = self.death_row.take().unwrap();
        // FIXME:
        death_row.push(state.image().clone());
        self.death_row.set(Some(death_row));

        Ok(())
    }

    /// Queues the destruction of the swapchain corresponding to `id` after the destruction of the
    /// command buffer(s) for this task.
    // FIXME: unsafe
    #[inline]
    pub unsafe fn destroy_swapchain(&self, id: Id<Swapchain>) -> TaskResult {
        let state = unsafe { self.resources.remove_swapchain(id) }?;
        let death_row = self.death_row.take().unwrap();
        // FIXME:
        death_row.push(state.swapchain().clone());
        self.death_row.set(Some(death_row));

        Ok(())
    }
}

/// Allows you to read a subbuffer from the host.
///
/// This type is created by the [`read_buffer`] method on [`TaskContext`].
///
/// [`read_buffer`]: TaskContext::read_buffer
// NOTE(Marc): This type doesn't actually do anything, but exists for forward-compatibility.
#[derive(Debug)]
pub struct BufferReadGuard<'a, T: ?Sized> {
    data: &'a T,
}

impl<T: ?Sized> Deref for BufferReadGuard<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.data
    }
}

/// Allows you to write a subbuffer from the host.
///
/// This type is created by the [`write_buffer`] method on [`TaskContext`].
///
/// [`write_buffer`]: TaskContext::write_buffer
pub struct BufferWriteGuard<'a, T: ?Sized> {
    subbuffer: Subbuffer<[u8]>,
    data: &'a mut T,
    atom_size: Option<DeviceAlignment>,
}

impl<T: ?Sized> Deref for BufferWriteGuard<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<T: ?Sized> DerefMut for BufferWriteGuard<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
}

impl<T: ?Sized> Drop for BufferWriteGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
        #[cold]
        fn flush_subbuffer(subbuffer: &Subbuffer<[u8]>, atom_size: DeviceAlignment) {
            let allocation = match subbuffer.buffer().memory() {
                BufferMemory::Normal(a) => a,
                _ => unreachable!(),
            };

            let memory_range = MappedMemoryRange {
                offset: align_down(subbuffer.offset(), atom_size),
                size: cmp::min(
                    align_up(subbuffer.offset() + subbuffer.size(), atom_size),
                    allocation.size(),
                ) - subbuffer.offset(),
                _ne: crate::NE,
            };

            // SAFETY: `TaskContext::write_buffer` ensures that the task has write access to this
            // subbuffer aligned to the non-coherent atom size.
            if let Err(err) = unsafe { allocation.flush_range_unchecked(memory_range) } {
                if !thread::panicking() {
                    panic!("failed to flush buffer write: {err:?}");
                }
            }
        }

        if let Some(atom_size) = self.atom_size {
            flush_subbuffer(&self.subbuffer, atom_size);
        }
    }
}

/// The type of result returned by a task.
pub type TaskResult<T = (), E = TaskError> = ::std::result::Result<T, E>;

/// Error that can happen inside a task.
#[derive(Debug)]
pub enum TaskError {
    InvalidSlot(InvalidSlotError),
    HostAccess(HostAccessError),
    ValidationError(Box<ValidationError>),
}

impl From<InvalidSlotError> for TaskError {
    fn from(err: InvalidSlotError) -> Self {
        Self::InvalidSlot(err)
    }
}

impl From<HostAccessError> for TaskError {
    fn from(err: HostAccessError) -> Self {
        Self::HostAccess(err)
    }
}

impl From<Box<ValidationError>> for TaskError {
    fn from(err: Box<ValidationError>) -> Self {
        Self::ValidationError(err)
    }
}

impl fmt::Display for TaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::InvalidSlot(_) => "invalid slot",
            Self::HostAccess(_) => "a host access error occurred",
            Self::ValidationError(_) => "a validation error occurred",
        };

        f.write_str(msg)
    }
}

impl Error for TaskError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidSlot(err) => Some(err),
            Self::HostAccess(err) => Some(err),
            Self::ValidationError(err) => Some(err),
        }
    }
}

/// Error that can happen when trying to retrieve a Vulkan object or state by [`Id`].
#[derive(Debug)]
pub struct InvalidSlotError {
    slot: SlotId,
}

impl InvalidSlotError {
    fn new<O>(id: Id<O>) -> Self {
        InvalidSlotError { slot: id.slot }
    }
}

impl fmt::Display for InvalidSlotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let &InvalidSlotError { slot } = self;
        let object_type = match slot.tag() {
            0 => ObjectType::Buffer,
            1 => ObjectType::Image,
            2 => ObjectType::Swapchain,
            3 => ObjectType::Flight,
            _ => unreachable!(),
        };

        write!(f, "invalid slot for object type {object_type:?}: {slot:?}")
    }
}

impl Error for InvalidSlotError {}

/// Error that can happen when attempting to read or write a resource from the host.
#[derive(Debug)]
pub enum HostAccessError {
    Invalidate(VulkanError),
    Unmanaged,
    NotHostMapped,
    OutOfMappedRange,
}

impl fmt::Display for HostAccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::Invalidate(_) => "invalidating the device memory failed",
            Self::Unmanaged => "the resource is not managed by vulkano",
            Self::NotHostMapped => "the device memory is not current host-mapped",
            Self::OutOfMappedRange => {
                "the requested range is not within the currently mapped range of device memory"
            }
        };

        f.write_str(msg)
    }
}

impl Error for HostAccessError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Invalidate(err) => Some(err),
            _ => None,
        }
    }
}

/// Specifies the type of queue family that a task can be executed on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QueueFamilyType {
    /// Picks a queue family that supports graphics and transfer operations.
    Graphics,

    /// Picks a queue family that supports compute and transfer operations.
    Compute,

    /// Picks a queue family that supports transfer operations.
    Transfer,

    // TODO:
    // VideoDecode,

    // TODO:
    // VideoEncode,
    /// Picks the queue family of the given index. You should generally avoid this and use one of
    /// the other variants, so that the task graph compiler can pick the most optimal queue family
    /// indices that still satisfy the supported operations that the tasks require (and also, it's
    /// more convenient that way, as there's less to think about). Nevertheless, you may want to
    /// use this if you're looking for some very specific outcome.
    Specific { index: u32 },
}

/// This ID type is used throughout the crate to refer to Vulkan objects such as resource objects
/// and their synchronization state, synchronization object state, and other state.
///
/// The type parameter denotes the type of object or state being referred to.
///
/// Note that this ID **is not** globally unique. It is unique in the scope of a logical device.
#[repr(transparent)]
pub struct Id<T> {
    slot: SlotId,
    marker: PhantomData<fn() -> T>,
}

impl<T> Id<T> {
    fn new(slot: SlotId) -> Self {
        Id {
            slot,
            marker: PhantomData,
        }
    }
}

impl<T> Clone for Id<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Id<T> {}

impl<T> fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Id")
            .field("generation", &self.slot.generation())
            .field("index", &self.slot.index())
            .finish()
    }
}

impl<T> PartialEq for Id<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.slot == other.slot
    }
}

impl<T> Eq for Id<T> {}

impl<T> Hash for Id<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.slot.hash(state);
    }
}

impl<T> PartialOrd for Id<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Id<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.slot.cmp(&other.slot)
    }
}

/// A reference to some Vulkan object or state.
///
/// When you use [`Id`] to retrieve something, you can get back a `Ref` with the same type
/// parameter, which you can then dereference to get at the underlying data denoted by the type
/// parameter.
pub struct Ref<'a, T>(concurrent_slotmap::Ref<'a, T>);

impl<T> Deref for Ref<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: fmt::Debug> fmt::Debug for Ref<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

#[derive(Debug, Clone, Copy)]
enum ObjectType {
    Buffer = 0,
    Image = 1,
    Swapchain = 2,
    Flight = 3,
}

// SAFETY: ZSTs can always be safely produced out of thin air, barring any safety invariants they
// might impose, which in the case of `NonExhaustive` are none.
const NE: vulkano::NonExhaustive =
    unsafe { ::std::mem::transmute::<(), ::vulkano::NonExhaustive>(()) };
