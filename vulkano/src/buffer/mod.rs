// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Location in memory that contains data.
//!
//! A Vulkan buffer is very similar to a buffer that you would use in programming languages in
//! general, in the sense that it is a location in memory that contains data. The difference
//! between a Vulkan buffer and a regular buffer is that the content of a Vulkan buffer is
//! accessible from the GPU.
//!
//! Vulkano does not perform any specific marshalling of buffer data. The representation of the buffer in
//! memory is identical between the CPU and GPU. Because the Rust compiler is allowed to reorder struct
//! fields at will by default when using `#[repr(Rust)]`, it is advised to mark each struct requiring
//! imput assembly as `#[repr(C)]`. This forces Rust to follow the standard C procedure. Each element is
//! laid out in memory in the order of declaration and aligned to a multiple of their alignment.
//!
//! # Various kinds of buffers
//!
//! The low level implementation of a buffer is [`UnsafeBuffer`](crate::buffer::sys::UnsafeBuffer).
//! This type makes it possible to use all the features that Vulkan is capable of, but as its name
//! tells it is unsafe to use.
//!
//! Instead you are encouraged to use one of the high-level wrappers that vulkano provides. Which
//! wrapper to use depends on the way you are going to use the buffer:
//!
//! - A [`DeviceLocalBuffer`](crate::buffer::device_local::DeviceLocalBuffer) designates a buffer
//!   usually located in video memory and whose content can't be directly accessed by your
//!   application. Accessing this buffer from the GPU is generally faster compared to accessing a
//!   CPU-accessible buffer.
//! - An [`ImmutableBuffer`](crate::buffer::immutable::ImmutableBuffer) designates a buffer in video
//!   memory and whose content can only be written at creation. Compared to `DeviceLocalBuffer`,
//!   this buffer requires less CPU processing because we don't need to keep track of the reads
//!   and writes.
//! - A [`CpuBufferPool`](crate::buffer::cpu_pool::CpuBufferPool) is a ring buffer that can be used to
//!   transfer data between the CPU and the GPU at a high rate.
//! - A [`CpuAccessibleBuffer`](crate::buffer::cpu_access::CpuAccessibleBuffer) is a simple buffer that
//!   can be used to prototype. It may be removed from vulkano in the far future.
//!
//! Here is a quick way to choose which buffer to use. Do you often need to read or write
//! the content of the buffer? If so, use a `CpuBufferPool`. Otherwise, do you need to be able to
//! modify the content of the buffer after its initialization? If so, use a `DeviceLocalBuffer`.
//! If no to both questions, use an `ImmutableBuffer`.
//!
//! When deciding how your buffer is going to be used, don't forget that sometimes the best
//! solution is to manipulate multiple buffers instead. For example if you need to update a buffer's
//! content only from time to time, it may be a good idea to simply recreate a new `ImmutableBuffer`
//! every time.
//! Another example: if a buffer is under constant access by the GPU but you need to
//! read its content on the CPU from time to time, it may be a good idea to use a
//! `DeviceLocalBuffer` as the main buffer and a `CpuBufferPool` for when you need to read it.
//! Then whenever you need to read the main buffer, ask the GPU to copy from the device-local
//! buffer to the CPU buffer pool, and read the CPU buffer pool instead.
//!
//! # Buffers usage
//!
//! When you create a buffer object, you have to specify its *usage*. In other words, you have to
//! specify the way it is going to be used. Trying to use a buffer in a way that wasn't specified
//! when you created it will result in a runtime error.
//!
//! You can use buffers for the following purposes:
//!
//! - Can contain arbitrary data that can be transferred from/to other buffers and images.
//! - Can be read and modified from a shader.
//! - Can be used as a source of vertices and indices.
//! - Can be used as a source of list of models for draw indirect commands.
//!
//! Accessing a buffer from a shader can be done in the following ways:
//!
//! - As a uniform buffer. Uniform buffers are read-only.
//! - As a storage buffer. Storage buffers can be read and written.
//! - As a uniform texel buffer. Contrary to a uniform buffer, the data is interpreted by the
//!   GPU and can be for example normalized.
//! - As a storage texel buffer. Additionally, some data formats can be modified with atomic
//!   operations.
//!
//! Using uniform/storage texel buffers requires creating a *buffer view*. See the `view` module
//! for how to create a buffer view.
//!

pub use self::{
    cpu_access::CpuAccessibleBuffer,
    cpu_pool::CpuBufferPool,
    device_local::DeviceLocalBuffer,
    immutable::ImmutableBuffer,
    slice::BufferSlice,
    sys::{BufferCreationError, SparseLevel},
    traits::{
        BufferAccess, BufferAccessObject, BufferDeviceAddressError, BufferInner, TypedBufferAccess,
    },
    usage::BufferUsage,
};
use crate::{
    memory::{ExternalMemoryHandleType, ExternalMemoryProperties},
    DeviceSize,
};
use bytemuck::{
    bytes_of, cast_slice, try_cast_slice, try_cast_slice_mut, try_from_bytes, try_from_bytes_mut,
    Pod, PodCastError,
};
use std::mem::size_of;

pub mod cpu_access;
pub mod cpu_pool;
pub mod device_local;
pub mod immutable;
pub mod sys;
pub mod view;

mod slice;
mod traits;
mod usage;

/// Trait for types of data that can be put in a buffer. These can be safely transmuted to and from
/// a slice of bytes.
pub unsafe trait BufferContents: Send + Sync + 'static {
    /// Converts an immutable reference to `Self` to an immutable byte slice.
    fn as_bytes(&self) -> &[u8];

    /// Converts an immutable byte slice into an immutable reference to `Self`.
    fn from_bytes(bytes: &[u8]) -> Result<&Self, PodCastError>;

    /// Converts a mutable byte slice into a mutable reference to `Self`.
    fn from_bytes_mut(bytes: &mut [u8]) -> Result<&mut Self, PodCastError>;

    /// Returns the size of an element of the type.
    fn size_of_element() -> DeviceSize;
}

unsafe impl<T> BufferContents for T
where
    T: Pod + Send + Sync,
{
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        bytes_of(self)
    }

    #[inline]
    fn from_bytes(bytes: &[u8]) -> Result<&T, PodCastError> {
        try_from_bytes(bytes)
    }

    #[inline]
    fn from_bytes_mut(bytes: &mut [u8]) -> Result<&mut T, PodCastError> {
        try_from_bytes_mut(bytes)
    }

    #[inline]
    fn size_of_element() -> DeviceSize {
        1
    }
}

unsafe impl<T> BufferContents for [T]
where
    T: Pod + Send + Sync,
{
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        cast_slice(self)
    }

    #[inline]
    fn from_bytes(bytes: &[u8]) -> Result<&[T], PodCastError> {
        try_cast_slice(bytes)
    }

    #[inline]
    fn from_bytes_mut(bytes: &mut [u8]) -> Result<&mut [T], PodCastError> {
        try_cast_slice_mut(bytes)
    }

    #[inline]
    fn size_of_element() -> DeviceSize {
        size_of::<T>() as DeviceSize
    }
}

/// The buffer configuration to query in
/// [`PhysicalDevice::external_buffer_properties`](crate::device::physical::PhysicalDevice::external_buffer_properties).
#[derive(Clone, Debug)]
pub struct ExternalBufferInfo {
    /// The external handle type that will be used with the buffer.
    pub handle_type: ExternalMemoryHandleType,

    /// The usage that the buffer will have.
    pub usage: BufferUsage,

    /// The sparse binding parameters that will be used.
    pub sparse: Option<SparseLevel>,

    pub _ne: crate::NonExhaustive,
}

impl ExternalBufferInfo {
    /// Returns an `ExternalBufferInfo` with the specified `handle_type`.
    #[inline]
    pub fn handle_type(handle_type: ExternalMemoryHandleType) -> Self {
        Self {
            handle_type,
            usage: BufferUsage::none(),
            sparse: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// The external memory properties supported for buffers with a given configuration.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ExternalBufferProperties {
    /// The properties for external memory.
    pub external_memory_properties: ExternalMemoryProperties,
}
