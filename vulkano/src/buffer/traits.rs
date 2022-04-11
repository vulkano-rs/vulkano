// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{sys::UnsafeBuffer, BufferContents, BufferSlice, BufferUsage};
use crate::{device::DeviceOwned, DeviceSize, SafeDeref, VulkanObject};
use std::{
    error, fmt,
    hash::{Hash, Hasher},
    num::NonZeroU64,
    ops::Range,
    sync::Arc,
};

/// Trait for objects that represent a way for the GPU to have access to a buffer or a slice of a
/// buffer.
///
/// See also `TypedBufferAccess`.
pub unsafe trait BufferAccess: DeviceOwned + Send + Sync {
    /// Returns the inner information about this buffer.
    fn inner(&self) -> BufferInner;

    /// Returns the size of the buffer in bytes.
    fn size(&self) -> DeviceSize;

    /// Returns the usage the buffer was created with.
    #[inline]
    fn usage(&self) -> &BufferUsage {
        self.inner().buffer.usage()
    }

    /// Returns a `BufferSlice` covering the whole buffer.
    #[inline]
    fn into_buffer_slice(self: &Arc<Self>) -> Arc<BufferSlice<Self::Content, Self>>
    where
        Self: Sized + TypedBufferAccess,
    {
        BufferSlice::from_typed_buffer_access(self.clone())
    }

    /// Returns a `BufferSlice` for a subrange of elements in the buffer. Returns `None` if
    /// out of range.
    ///
    /// This method can be used when you want to perform an operation on some part of the buffer
    /// and not on the whole buffer.
    #[inline]
    fn slice<T>(self: &Arc<Self>, range: Range<DeviceSize>) -> Option<Arc<BufferSlice<[T], Self>>>
    where
        Self: Sized + TypedBufferAccess<Content = [T]>,
    {
        BufferSlice::slice(&self.into_buffer_slice(), range)
    }

    /// Returns a `BufferSlice` for a single element in the buffer. Returns `None` if out of range.
    ///
    /// This method can be used when you want to perform an operation on a specific element of the
    /// buffer and not on the whole buffer.
    #[inline]
    fn index<T>(self: &Arc<Self>, index: DeviceSize) -> Option<Arc<BufferSlice<T, Self>>>
    where
        Self: Sized + TypedBufferAccess<Content = [T]>,
    {
        BufferSlice::index(&self.into_buffer_slice(), index)
    }

    /// Returns a key that uniquely identifies the buffer. Two buffers or images that potentially
    /// overlap in memory must return the same key.
    ///
    /// The key is shared amongst all buffers and images, which means that you can make several
    /// different buffer objects share the same memory, or make some buffer objects share memory
    /// with images, as long as they return the same key.
    ///
    /// Since it is possible to accidentally return the same key for memory ranges that don't
    /// overlap, the `conflicts_buffer` or `conflicts_image` function should always be called to
    /// verify whether they actually overlap.
    fn conflict_key(&self) -> (u64, u64);

    /// Gets the device address for this buffer.
    ///
    /// # Safety
    ///
    /// No lock checking or waiting is performed. This is nevertheless still safe because the
    /// returned value isn't directly dereferencable. Unsafe code is required to dereference the
    /// value in a shader.
    fn raw_device_address(&self) -> Result<NonZeroU64, BufferDeviceAddressError> {
        let inner = self.inner();
        let device = self.device();

        // VUID-vkGetBufferDeviceAddress-bufferDeviceAddress-03324
        if !device.enabled_features().buffer_device_address {
            return Err(BufferDeviceAddressError::FeatureNotEnabled);
        }

        // VUID-VkBufferDeviceAddressInfo-buffer-02601
        if !inner.buffer.usage().device_address {
            return Err(BufferDeviceAddressError::BufferMissingUsage);
        }

        unsafe {
            let info = ash::vk::BufferDeviceAddressInfo {
                buffer: inner.buffer.internal_object(),
                ..Default::default()
            };
            let ptr = device
                .fns()
                .ext_buffer_device_address
                .get_buffer_device_address_ext(device.internal_object(), &info);

            if ptr == 0 {
                panic!("got null ptr from a valid GetBufferDeviceAddressEXT call");
            }

            Ok(NonZeroU64::new_unchecked(ptr + inner.offset))
        }
    }
}

pub trait BufferAccessObject {
    fn as_buffer_access_object(&self) -> Arc<dyn BufferAccess>;
}

impl BufferAccessObject for Arc<dyn BufferAccess> {
    fn as_buffer_access_object(&self) -> Arc<dyn BufferAccess> {
        self.clone()
    }
}

/// Inner information about a buffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferInner<'a> {
    /// The underlying buffer object.
    pub buffer: &'a Arc<UnsafeBuffer>,
    /// The offset in bytes from the start of the underlying buffer object to the start of the
    /// buffer we're describing.
    pub offset: DeviceSize,
}

unsafe impl<T> BufferAccess for T
where
    T: SafeDeref + Send + Sync,
    T::Target: BufferAccess,
{
    #[inline]
    fn inner(&self) -> BufferInner {
        (**self).inner()
    }

    #[inline]
    fn size(&self) -> DeviceSize {
        (**self).size()
    }

    #[inline]
    fn conflict_key(&self) -> (u64, u64) {
        (**self).conflict_key()
    }
}

/// Extension trait for `BufferAccess`. Indicates the type of the content of the buffer.
pub unsafe trait TypedBufferAccess: BufferAccess {
    /// The type of the content.
    type Content: BufferContents + ?Sized;

    /// Returns the length of the buffer in number of elements.
    ///
    /// This method can only be called for buffers whose type is known to be an array.
    #[inline]
    fn len(&self) -> DeviceSize {
        self.size() / Self::Content::size_of_element()
    }
}

unsafe impl<T> TypedBufferAccess for T
where
    T: SafeDeref + Send + Sync,
    T::Target: TypedBufferAccess,
{
    type Content = <T::Target as TypedBufferAccess>::Content;
}

impl fmt::Debug for dyn BufferAccess {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("dyn BufferAccess")
            .field("inner", &self.inner())
            .finish()
    }
}

impl PartialEq for dyn BufferAccess {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner() && self.size() == other.size()
    }
}

impl Eq for dyn BufferAccess {}

impl Hash for dyn BufferAccess {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
        self.size().hash(state);
    }
}

/// Error that can happen when querying the device address of a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferDeviceAddressError {
    BufferMissingUsage,
    FeatureNotEnabled,
}

impl error::Error for BufferDeviceAddressError {}

impl fmt::Display for BufferDeviceAddressError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BufferMissingUsage => write!(
                fmt,
                "the device address usage flag was not set on this buffer",
            ),
            Self::FeatureNotEnabled => write!(
                fmt,
                "the buffer_device_address feature was not enabled on the device",
            ),
        }
    }
}
