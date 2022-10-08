// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{sys::UnsafeBuffer, BufferContents, BufferSlice, BufferUsage};
use crate::{device::DeviceOwned, DeviceSize, RequiresOneOf, SafeDeref, VulkanObject};
use std::{
    error::Error,
    fmt::{Debug, Display, Error as FmtError, Formatter},
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
    fn inner(&self) -> BufferInner<'_>;

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
    fn index<T>(self: &Arc<Self>, index: DeviceSize) -> Option<Arc<BufferSlice<T, Self>>>
    where
        Self: Sized + TypedBufferAccess<Content = [T]>,
    {
        BufferSlice::index(&self.into_buffer_slice(), index)
    }

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
            return Err(BufferDeviceAddressError::RequirementNotMet {
                required_for: "`raw_device_address`",
                requires_one_of: RequiresOneOf {
                    features: &["buffer_device_address"],
                    ..Default::default()
                },
            });
        }

        // VUID-VkBufferDeviceAddressInfo-buffer-02601
        if !inner.buffer.usage().shader_device_address {
            return Err(BufferDeviceAddressError::BufferMissingUsage);
        }

        unsafe {
            let info = ash::vk::BufferDeviceAddressInfo {
                buffer: inner.buffer.internal_object(),
                ..Default::default()
            };
            let fns = device.fns();
            let f = if device.enabled_extensions().ext_buffer_device_address {
                fns.ext_buffer_device_address.get_buffer_device_address_ext
            } else {
                fns.khr_buffer_device_address.get_buffer_device_address_khr
            };
            let ptr = f(device.internal_object(), &info);

            if ptr == 0 {
                panic!("got null ptr from a valid GetBufferDeviceAddress call");
            }

            Ok(NonZeroU64::new_unchecked(ptr + inner.offset))
        }
    }
}

pub trait BufferAccessObject {
    fn as_buffer_access_object(&self) -> Arc<dyn BufferAccess>;
}

impl BufferAccessObject for Arc<dyn BufferAccess> {
    #[inline]
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
    fn inner(&self) -> BufferInner<'_> {
        (**self).inner()
    }

    fn size(&self) -> DeviceSize {
        (**self).size()
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

impl Debug for dyn BufferAccess {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
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
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
        self.size().hash(state);
    }
}

/// Error that can happen when querying the device address of a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferDeviceAddressError {
    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    BufferMissingUsage,
}

impl Error for BufferDeviceAddressError {}

impl Display for BufferDeviceAddressError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
            Self::BufferMissingUsage => write!(
                f,
                "the device address usage flag was not set on this buffer",
            ),
        }
    }
}
