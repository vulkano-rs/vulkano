// Copyright (c) 2020 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ops::BitOr;

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
}

impl From<ExternalMemoryHandleType> for ash::vk::ExternalMemoryHandleTypeFlags {
    #[inline]
    fn from(val: ExternalMemoryHandleType) -> Self {
        let mut result = ash::vk::ExternalMemoryHandleTypeFlags::empty();
        if val.opaque_fd {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;
        }
        if val.opaque_win32 {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32;
        }
        if val.opaque_win32_kmt {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KMT;
        }
        if val.d3d11_texture {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE;
        }
        if val.d3d11_texture_kmt {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE_KMT;
        }
        if val.d3d12_heap {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::D3D12_HEAP;
        }
        if val.d3d12_resource {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::D3D12_RESOURCE;
        }
        if val.dma_buf {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
        }
        if val.android_hardware_buffer {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::ANDROID_HARDWARE_BUFFER_ANDROID;
        }
        if val.host_allocation {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        }
        if val.host_mapped_foreign_memory {
            result |= ash::vk::ExternalMemoryHandleTypeFlags::HOST_MAPPED_FOREIGN_MEMORY_EXT
        }
        result
    }
}

impl From<ash::vk::ExternalMemoryHandleTypeFlags> for ExternalMemoryHandleType {
    fn from(val: ash::vk::ExternalMemoryHandleTypeFlags) -> Self {
        ExternalMemoryHandleType {
            opaque_fd: !(val & ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD).is_empty(),
            opaque_win32: !(val & ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32).is_empty(),
            opaque_win32_kmt: !(val & ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KMT)
                .is_empty(),
            d3d11_texture: !(val & ash::vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE)
                .is_empty(),
            d3d11_texture_kmt: !(val & ash::vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE_KMT)
                .is_empty(),
            d3d12_heap: !(val & ash::vk::ExternalMemoryHandleTypeFlags::D3D12_HEAP).is_empty(),
            d3d12_resource: !(val & ash::vk::ExternalMemoryHandleTypeFlags::D3D12_RESOURCE)
                .is_empty(),
            dma_buf: !(val & ash::vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT).is_empty(),
            android_hardware_buffer: !(val
                & ash::vk::ExternalMemoryHandleTypeFlags::ANDROID_HARDWARE_BUFFER_ANDROID)
                .is_empty(),
            host_allocation: !(val & ash::vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
                .is_empty(),
            host_mapped_foreign_memory: !(val
                & ash::vk::ExternalMemoryHandleTypeFlags::HOST_MAPPED_FOREIGN_MEMORY_EXT)
                .is_empty(),
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
