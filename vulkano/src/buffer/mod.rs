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
//! Vulkano does not perform any specific marshalling of buffer data. The representation of the
//! buffer in memory is identical between the CPU and GPU. Because the Rust compiler is allowed to
//! reorder struct fields at will by default when using `#[repr(Rust)]`, it is advised to mark each
//! struct requiring imput assembly as `#[repr(C)]`. This forces Rust to follow the standard C
//! procedure. Each element is laid out in memory in the order of declaration and aligned to a
//! multiple of their alignment.
//!
//! # Various kinds of buffers
//!
//! The low level implementation of a buffer is [`RawBuffer`](crate::buffer::sys::RawBuffer).
//! This type makes it possible to use all the features that Vulkan is capable of.
//!
//! Instead you are encouraged to use one of the high-level wrappers that vulkano provides. Which
//! wrapper to use depends on the way you are going to use the buffer:
//!
//! - A [`DeviceLocalBuffer`] designates a buffer usually located in video memory and whose content
//!   can't be directly accessed by your application. Accessing this buffer from the GPU is
//!   generally faster compared to accessing a CPU-accessible buffer.
//! - A [`CpuBufferAllocator`] can be used to transfer data between the CPU and the GPU at a high
//!   rate.
//! - A [`CpuAccessibleBuffer`] is a simple buffer that can be used to prototype.
//!
//! Here is a quick way to choose which buffer to use. Do you often need to read or write the
//! content of the buffer? If so, use a `CpuBufferAllocator`. Otherwise, do you need to have access
//! to the buffer on the CPU? Then use `CpuAccessibleBuffer`. Otherwise, use a `DeviceLocalBuffer`.
//!
//! Another example: if a buffer is under constant access by the GPU but you need to read its
//! content on the CPU from time to time, it may be a good idea to use a `DeviceLocalBuffer` as the
//! main buffer and a `CpuAccessibleBuffer` for when you need to read it. Then whenever you need to
//! read the main buffer, ask the GPU to copy from the device-local buffer to the CPU-accessible
//! buffer, and read the CPU-accessible buffer instead.
//!
//! # Buffer usage
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
//! - As a uniform texel buffer. Contrary to a uniform buffer, the data is interpreted by the GPU
//!   and can be for example normalized.
//! - As a storage texel buffer. Additionally, some data formats can be modified with atomic
//!   operations.
//!
//! Using uniform/storage texel buffers requires creating a *buffer view*. See the `view` module
//! for how to create a buffer view.
//!
//! [`CpuBufferAllocator`]: allocator::CpuBufferAllocator

pub use self::{
    cpu_access::CpuAccessibleBuffer,
    device_local::DeviceLocalBuffer,
    slice::BufferSlice,
    sys::BufferError,
    traits::{
        BufferAccess, BufferAccessObject, BufferDeviceAddressError, BufferInner, TypedBufferAccess,
    },
    usage::BufferUsage,
};
use crate::{
    macros::vulkan_bitflags,
    memory::{ExternalMemoryHandleType, ExternalMemoryProperties},
    DeviceSize,
};
use bytemuck::{
    bytes_of, bytes_of_mut, cast_slice, cast_slice_mut, try_cast_slice, try_cast_slice_mut,
    try_from_bytes, try_from_bytes_mut, Pod, PodCastError,
};
use std::mem::size_of;

pub mod allocator;
pub mod cpu_access;
pub mod device_local;
pub mod sys;
pub mod view;

mod slice;
mod traits;
mod usage;

vulkan_bitflags! {
    /// Flags to be set when creating a buffer.
    #[non_exhaustive]
    BufferCreateFlags = BufferCreateFlags(u32);

    /*
    /// The buffer will be backed by sparse memory binding (through queue commands) instead of
    /// regular binding (through [`bind_memory`]).
    ///
    /// The [`sparse_binding`] feature must be enabled on the device.
    ///
    /// [`bind_memory`]: sys::RawBuffer::bind_memory
    /// [`sparse_binding`]: crate::device::Features::sparse_binding
    sparse_binding = SPARSE_BINDING,

    /// The buffer can be used without being fully resident in memory at the time of use.
    ///
    /// This requires the `sparse_binding` flag as well.
    ///
    /// The [`sparse_residency_buffer`] feature must be enabled on the device.
    ///
    /// [`sparse_residency_buffer`]: crate::device::Features::sparse_residency_buffer
    sparse_residency = SPARSE_RESIDENCY,

    /// The buffer's memory can alias with another buffer or a different part of the same buffer.
    ///
    /// This requires the `sparse_binding` flag as well.
    ///
    /// The [`sparse_residency_aliased`] feature must be enabled on the device.
    ///
    /// [`sparse_residency_aliased`]: crate::device::Features::sparse_residency_aliased
    sparse_aliased = SPARSE_ALIASED,

    /// The buffer is protected, and can only be used in combination with protected memory and other
    /// protected objects.
    ///
    /// The device API version must be at least 1.1.
    protected = PROTECTED {
        api_version: V1_1,
    },

    /// The buffer's device address can be saved and reused on a subsequent run.
    ///
    /// The device API version must be at least 1.2, or either the [`khr_buffer_device_address`] or
    /// [`ext_buffer_device_address`] extension must be enabled on the device.
    device_address_capture_replay = DEVICE_ADDRESS_CAPTURE_REPLAY {
        api_version: V1_2,
        device_extensions: [khr_buffer_device_address, ext_buffer_device_address],
    },
     */
}

/// Trait for types of data that can be put in a buffer. These can be safely transmuted to and from
/// a slice of bytes.
pub unsafe trait BufferContents: Send + Sync + 'static {
    /// Converts an immutable reference to `Self` to an immutable byte slice.
    fn as_bytes(&self) -> &[u8];

    /// Converts a mutable reference to `Self` to a mutable byte slice.
    fn as_bytes_mut(&mut self) -> &mut [u8];

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
    fn as_bytes(&self) -> &[u8] {
        bytes_of(self)
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        bytes_of_mut(self)
    }

    fn from_bytes(bytes: &[u8]) -> Result<&T, PodCastError> {
        try_from_bytes(bytes)
    }

    fn from_bytes_mut(bytes: &mut [u8]) -> Result<&mut T, PodCastError> {
        try_from_bytes_mut(bytes)
    }

    fn size_of_element() -> DeviceSize {
        1
    }
}

unsafe impl<T> BufferContents for [T]
where
    T: Pod + Send + Sync,
{
    fn as_bytes(&self) -> &[u8] {
        cast_slice(self)
    }

    fn as_bytes_mut(&mut self) -> &mut [u8] {
        cast_slice_mut(self)
    }

    fn from_bytes(bytes: &[u8]) -> Result<&[T], PodCastError> {
        try_cast_slice(bytes)
    }

    fn from_bytes_mut(bytes: &mut [u8]) -> Result<&mut [T], PodCastError> {
        try_cast_slice_mut(bytes)
    }

    fn size_of_element() -> DeviceSize {
        size_of::<T>() as DeviceSize
    }
}

/// The buffer configuration to query in
/// [`PhysicalDevice::external_buffer_properties`](crate::device::physical::PhysicalDevice::external_buffer_properties).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExternalBufferInfo {
    /// The external handle type that will be used with the buffer.
    pub handle_type: ExternalMemoryHandleType,

    /// The usage that the buffer will have.
    pub usage: BufferUsage,

    /// The sparse binding parameters that will be used.
    pub sparse: Option<BufferCreateFlags>,

    pub _ne: crate::NonExhaustive,
}

impl ExternalBufferInfo {
    /// Returns an `ExternalBufferInfo` with the specified `handle_type`.
    #[inline]
    pub fn handle_type(handle_type: ExternalMemoryHandleType) -> Self {
        Self {
            handle_type,
            usage: BufferUsage::empty(),
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
