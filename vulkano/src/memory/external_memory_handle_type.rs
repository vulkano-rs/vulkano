// Copyright (c) 2020 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ops::BitOr;

/// Describes a handle type used for Vulkan external memory apis.  This is **not** just a
/// suggestion.  Check out vkExternalMemoryHandleTypeFlagBits in the Vulkan spec.
///
/// If you specify an handle type that doesnt make sense (for example, using a dma-buf handle type
/// on Windows) when using this handle, a panic will happen.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ExternalMemoryHandleType {
    OpaqueFd = ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD.as_raw(),
    OpaqueWin32 = ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32.as_raw(),
    OpaqueWin32Kmt = ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KMT.as_raw(),
    D3D11Texture = ash::vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE.as_raw(),
    D3D11TextureKmt = ash::vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE_KMT.as_raw(),
    D3D12Heap = ash::vk::ExternalMemoryHandleTypeFlags::D3D12_HEAP.as_raw(),
    D3D12Resource = ash::vk::ExternalMemoryHandleTypeFlags::D3D12_RESOURCE.as_raw(),
    DmaBuf = ash::vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT.as_raw(),
    AndroidHardwareBuffer =
        ash::vk::ExternalMemoryHandleTypeFlags::ANDROID_HARDWARE_BUFFER_ANDROID.as_raw(),
    HostAllocation = ash::vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT.as_raw(),
    HostMappedForeignMemory =
        ash::vk::ExternalMemoryHandleTypeFlags::HOST_MAPPED_FOREIGN_MEMORY_EXT.as_raw(),
}

impl From<ExternalMemoryHandleType> for ash::vk::ExternalMemoryHandleTypeFlags {
    fn from(val: ExternalMemoryHandleType) -> Self {
        Self::from_raw(val as u32)
    }
}

/// A mask of multiple handle types.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ExternalMemoryHandleTypes {
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

impl ExternalMemoryHandleTypes {
    /// Builds a `ExternalMemoryHandleTypes` with all values set to false. Useful as a default value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use vulkano::memory::ExternalMemoryHandleTypes as ExternalMemoryHandleTypes;
    ///
    /// let _handle_type = ExternalMemoryHandleTypes {
    ///     opaque_fd: true,
    ///     .. ExternalMemoryHandleTypes::none()
    /// };
    /// ```
    #[inline]
    pub fn none() -> Self {
        ExternalMemoryHandleTypes {
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

    /// Builds an `ExternalMemoryHandleTypes` for a posix file descriptor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use vulkano::memory::ExternalMemoryHandleTypes as ExternalMemoryHandleTypes;
    ///
    /// let _handle_type = ExternalMemoryHandleTypes::posix();
    /// ```
    #[inline]
    pub fn posix() -> ExternalMemoryHandleTypes {
        ExternalMemoryHandleTypes {
            opaque_fd: true,
            ..ExternalMemoryHandleTypes::none()
        }
    }

    /// Returns whether any of the fields are set.
    #[inline]
    pub fn is_empty(&self) -> bool {
        let ExternalMemoryHandleTypes {
            opaque_fd,
            opaque_win32,
            opaque_win32_kmt,
            d3d11_texture,
            d3d11_texture_kmt,
            d3d12_heap,
            d3d12_resource,
            dma_buf,
            android_hardware_buffer,
            host_allocation,
            host_mapped_foreign_memory,
        } = *self;

        !(opaque_fd
            || opaque_win32
            || opaque_win32_kmt
            || d3d11_texture
            || d3d11_texture_kmt
            || d3d12_heap
            || d3d12_resource
            || dma_buf
            || android_hardware_buffer
            || host_allocation
            || host_mapped_foreign_memory)
    }

    /// Returns an iterator of `ExternalMemoryHandleType` enum values, representing the fields that
    /// are set in `self`.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = ExternalMemoryHandleType> {
        let ExternalMemoryHandleTypes {
            opaque_fd,
            opaque_win32,
            opaque_win32_kmt,
            d3d11_texture,
            d3d11_texture_kmt,
            d3d12_heap,
            d3d12_resource,
            dma_buf,
            android_hardware_buffer,
            host_allocation,
            host_mapped_foreign_memory,
        } = *self;

        [
            opaque_fd.then(|| ExternalMemoryHandleType::OpaqueFd),
            opaque_win32.then(|| ExternalMemoryHandleType::OpaqueWin32),
            opaque_win32_kmt.then(|| ExternalMemoryHandleType::OpaqueWin32Kmt),
            d3d11_texture.then(|| ExternalMemoryHandleType::D3D11Texture),
            d3d11_texture_kmt.then(|| ExternalMemoryHandleType::D3D11TextureKmt),
            d3d12_heap.then(|| ExternalMemoryHandleType::D3D12Heap),
            d3d12_resource.then(|| ExternalMemoryHandleType::D3D12Resource),
            dma_buf.then(|| ExternalMemoryHandleType::DmaBuf),
            android_hardware_buffer.then(|| ExternalMemoryHandleType::AndroidHardwareBuffer),
            host_allocation.then(|| ExternalMemoryHandleType::HostAllocation),
            host_mapped_foreign_memory.then(|| ExternalMemoryHandleType::HostMappedForeignMemory),
        ]
        .into_iter()
        .flatten()
    }
}

impl From<ExternalMemoryHandleTypes> for ash::vk::ExternalMemoryHandleTypeFlags {
    #[inline]
    fn from(val: ExternalMemoryHandleTypes) -> Self {
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

impl From<ash::vk::ExternalMemoryHandleTypeFlags> for ExternalMemoryHandleTypes {
    fn from(val: ash::vk::ExternalMemoryHandleTypeFlags) -> Self {
        ExternalMemoryHandleTypes {
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

impl BitOr for ExternalMemoryHandleTypes {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        ExternalMemoryHandleTypes {
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
