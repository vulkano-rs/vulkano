// Copyright (c) 2020 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ops::BitOr;
use vk;

/// Describes the handle type used for Vulkan external memory apis.  This is **not** just a
/// suggestion.  Check out vkExternalMemoryHandleTypeFlagBits in the Vulkan spec.
///
/// If you specify an handle type that doesnt make sense (for example, using a dma-buf handle type
/// on Windows) when using this handle, a panic will happen.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct ExternalMemoryHandleType {
    pub opaque_fd: bool,
    pub opaque_win32: bool,
    pub opaque_win32_kmt: bool,
    pub d3d11_texture: bool,
    pub d3d11_texture_kmt: bool,
    pub d3d12_heap: bool,
    pub d3d12_resource: bool,
    pub dma_buf: bool,
    pub android_hardware_buffer: bool,
    pub host_allocation: bool,
    pub host_mapped_foreign_memory: bool,
}

impl ExternalMemoryHandleType {
    /// Builds a `ExternalMemoryHandleType` with all values set to false. Useful as a default value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use vulkano::memory::ExternalMemoryHandleType as ExternalMemoryHandleType;
    ///
    /// let _handle_type = ExternalMemoryHandleType {
    ///     opaque_fd: true,
    ///     .. ExternalMemoryHandleType::none()
    /// };
    /// ```
    #[inline]
    pub fn none() -> ExternalMemoryHandleType {
        ExternalMemoryHandleType {
            opaque_fd: false,
            opaque_win32: false,
            opaque_win32_kmt: false,
            d3d11_texture: false,
            d3d11_texture_kmt: false,
            d3d12_heap: false,
            d3d12_resource: false,
            dma_buf: false,
            android_hardware_buffer: false,
            host_allocation: false,
            host_mapped_foreign_memory: false,
        }
    }

    /// Builds an `ExternalMemoryHandleType` for a posix file descriptor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use vulkano::memory::ExternalMemoryHandleType as ExternalMemoryHandleType;
    ///
    /// let _handle_type = ExternalMemoryHandleType::posix();
    /// ```
    #[inline]
    pub fn posix() -> ExternalMemoryHandleType {
        ExternalMemoryHandleType {
            opaque_fd: true,
            ..ExternalMemoryHandleType::none()
        }
    }

    #[inline]
    pub(crate) fn to_bits(&self) -> vk::ExternalMemoryHandleTypeFlagBits {
        let mut result = 0;
        if self.opaque_fd {
            result |= vk::EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        }
        if self.opaque_win32 {
            result |= vk::EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        }
        if self.opaque_win32_kmt {
            result |= vk::EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
        }
        if self.d3d11_texture {
            result |= vk::EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
        }
        if self.d3d11_texture_kmt {
            result |= vk::EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT;
        }
        if self.d3d12_heap {
            result |= vk::EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT;
        }
        if self.d3d12_resource {
            result |= vk::EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT;
        }
        if self.dma_buf {
            result |= vk::EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        }
        if self.android_hardware_buffer {
            result |= vk::EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
        }
        if self.host_allocation {
            result |= vk::EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        }
        if self.host_mapped_foreign_memory {
            result |= vk::EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT
        }
        result
    }

    pub(crate) fn from_bits(val: u32) -> ExternalMemoryHandleType {
        ExternalMemoryHandleType {
            opaque_fd: (val & vk::EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) != 0,
            opaque_win32: (val & vk::EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT) != 0,
            opaque_win32_kmt: (val & vk::EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) != 0,
            d3d11_texture: (val & vk::EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT) != 0,
            d3d11_texture_kmt: (val & vk::EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT) != 0,
            d3d12_heap: (val & vk::EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT) != 0,
            d3d12_resource: (val & vk::EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT) != 0,
            dma_buf: (val & vk::EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) != 0,
            android_hardware_buffer: (val
                & vk::EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID)
                != 0,
            host_allocation: (val & vk::EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT) != 0,
            host_mapped_foreign_memory: (val
                & vk::EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT)
                != 0,
        }
    }
}

impl BitOr for ExternalMemoryHandleType {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        ExternalMemoryHandleType {
            opaque_fd: self.opaque_fd || rhs.opaque_fd,
            opaque_win32: self.opaque_win32 || rhs.opaque_win32,
            opaque_win32_kmt: self.opaque_win32_kmt || rhs.opaque_win32_kmt,
            d3d11_texture: self.d3d11_texture || rhs.d3d11_texture,
            d3d11_texture_kmt: self.d3d11_texture_kmt || rhs.d3d11_texture_kmt,
            d3d12_heap: self.d3d12_heap || rhs.d3d12_heap,
            d3d12_resource: self.d3d12_resource || rhs.d3d12_resource,
            dma_buf: self.dma_buf || rhs.dma_buf,
            android_hardware_buffer: self.android_hardware_buffer || rhs.android_hardware_buffer,
            host_allocation: self.host_allocation || rhs.host_allocation,
            host_mapped_foreign_memory: self.host_mapped_foreign_memory
                || rhs.host_mapped_foreign_memory,
        }
    }
}
