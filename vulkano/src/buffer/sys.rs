// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::mem;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;
use smallvec::SmallVec;

use device::Device;
use memory::DeviceMemory;
use memory::MemoryRequirements;
use sync::Sharing;

use check_errors;
use OomError;
use VulkanObject;
use VulkanPointers;
use vk;

/// Data storage in a GPU-accessible location.
///
/// # Safety
///
///
pub struct UnsafeBuffer {
    buffer: vk::Buffer,
    device: Arc<Device>,
    size: usize,
    usage: vk::BufferUsageFlags,
}

impl UnsafeBuffer {
    /// Creates a new buffer of the given size.
    ///
    /// See the module's documentation for information about safety.
    pub unsafe fn new<'a, I>(device: &Arc<Device>, size: usize, usage: &Usage, sharing: Sharing<I>)
                             -> Result<(UnsafeBuffer, MemoryRequirements), OomError>
        where I: Iterator<Item = u32>
    {
        let vk = device.pointers();

        let usage = usage.to_usage_bits();

        let buffer = {
            let (sh_mode, sh_indices) = match sharing {
                Sharing::Exclusive => (vk::SHARING_MODE_EXCLUSIVE, SmallVec::<[u32; 8]>::new()),
                Sharing::Concurrent(ids) => (vk::SHARING_MODE_CONCURRENT, ids.collect()),
            };

            let infos = vk::BufferCreateInfo {
                sType: vk::STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,       // TODO: sparse resources binding
                size: size as u64,
                usage: usage,
                sharingMode: sh_mode,
                queueFamilyIndexCount: sh_indices.len() as u32,
                pQueueFamilyIndices: sh_indices.as_ptr(),
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateBuffer(device.internal_object(), &infos,
                                              ptr::null(), &mut output)));
            output
        };

        let mem_reqs: vk::MemoryRequirements = {
            let mut output = mem::uninitialized();
            vk.GetBufferMemoryRequirements(device.internal_object(), buffer, &mut output);
            debug_assert!(output.size >= size as u64);
            debug_assert!(output.memoryTypeBits != 0);
            output
        };

        let obj = UnsafeBuffer {
            buffer: buffer,
            device: device.clone(),
            size: size as usize,
            usage: usage,
        };

        Ok((obj, mem_reqs.into()))
    }

    pub unsafe fn bind_memory(&self, memory: &DeviceMemory, range: Range<usize>)
                              -> Result<(), OomError>
    {
        let vk = self.device.pointers();
        try!(check_errors(vk.BindBufferMemory(self.device.internal_object(), self.buffer,
                                              memory.internal_object(),
                                              range.start as vk::DeviceSize)));
        Ok(())
    }

    /// Returns the device used to create this buffer.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the size of the buffer in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn usage_transfer_src(&self) -> bool {
        (self.usage & vk::BUFFER_USAGE_TRANSFER_SRC_BIT) != 0
    }

    #[inline]
    pub fn usage_transfer_dest(&self) -> bool {
        (self.usage & vk::BUFFER_USAGE_TRANSFER_DST_BIT) != 0
    }

    #[inline]
    pub fn usage_uniform_texel_buffer(&self) -> bool {
        (self.usage & vk::BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT) != 0
    }

    #[inline]
    pub fn usage_storage_texel_buffer(&self) -> bool {
        (self.usage & vk::BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT) != 0
    }

    #[inline]
    pub fn usage_uniform_buffer(&self) -> bool {
        (self.usage & vk::BUFFER_USAGE_UNIFORM_BUFFER_BIT) != 0
    }

    #[inline]
    pub fn usage_storage_buffer(&self) -> bool {
        (self.usage & vk::BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0
    }

    #[inline]
    pub fn usage_index_buffer(&self) -> bool {
        (self.usage & vk::BUFFER_USAGE_INDEX_BUFFER_BIT) != 0
    }

    #[inline]
    pub fn usage_vertex_buffer(&self) -> bool {
        (self.usage & vk::BUFFER_USAGE_VERTEX_BUFFER_BIT) != 0
    }

    #[inline]
    pub fn usage_indirect_buffer(&self) -> bool {
        (self.usage & vk::BUFFER_USAGE_INDIRECT_BUFFER_BIT) != 0
    }
}

unsafe impl VulkanObject for UnsafeBuffer {
    type Object = vk::Buffer;

    #[inline]
    fn internal_object(&self) -> vk::Buffer {
        self.buffer
    }
}

impl Drop for UnsafeBuffer {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyBuffer(self.device.internal_object(), self.buffer, ptr::null());
        }
    }
}

/// Describes how a buffer is going to be used. This is **not** an optimization.
///
/// If you try to use a buffer in a way that you didn't declare, a panic will happen.
///
/// Some methods are provided to build `Usage` structs for some common situations. However
/// there is no restriction in the combination of usages that can be enabled.
#[derive(Debug, Copy, Clone)]
pub struct Usage {
    pub transfer_source: bool,
    pub transfer_dest: bool,
    pub uniform_texel_buffer: bool,
    pub storage_texel_buffer: bool,
    pub uniform_buffer: bool,
    pub storage_buffer: bool,
    pub index_buffer: bool,
    pub vertex_buffer: bool,
    pub indirect_buffer: bool,
}

impl Usage {
    /// Builds a `Usage` with all values set to false.
    #[inline]
    pub fn none() -> Usage {
        Usage {
            transfer_source: false,
            transfer_dest: false,
            uniform_texel_buffer: false,
            storage_texel_buffer: false,
            uniform_buffer: false,
            storage_buffer: false,
            index_buffer: false,
            vertex_buffer: false,
            indirect_buffer: false,
        }
    }

    /// Builds a `Usage` with all values set to true. Can be used for quick prototyping.
    #[inline]
    pub fn all() -> Usage {
        Usage {
            transfer_source: true,
            transfer_dest: true,
            uniform_texel_buffer: true,
            storage_texel_buffer: true,
            uniform_buffer: true,
            storage_buffer: true,
            index_buffer: true,
            vertex_buffer: true,
            indirect_buffer: true,
        }
    }

    /// Builds a `Usage` with `transfer_source` set to true and the rest to false.
    #[inline]
    pub fn transfer_source() -> Usage {
        Usage {
            transfer_source: true,
            .. Usage::none()
        }
    }

    /// Builds a `Usage` with `vertex_buffer` set to true and the rest to false.
    #[inline]
    pub fn vertex_buffer() -> Usage {
        Usage {
            vertex_buffer: true,
            .. Usage::none()
        }
    }

    /// Builds a `Usage` with `vertex_buffer` and `transfer_dest` set to true and the rest to false.
    #[inline]
    pub fn vertex_buffer_transfer_dest() -> Usage {
        Usage {
            vertex_buffer: true,
            transfer_dest: true,
            .. Usage::none()
        }
    }

    /// Builds a `Usage` with `index_buffer` set to true and the rest to false.
    #[inline]
    pub fn index_buffer() -> Usage {
        Usage {
            index_buffer: true,
            .. Usage::none()
        }
    }

    /// Builds a `Usage` with `index_buffer` and `transfer_dest` set to true and the rest to false.
    #[inline]
    pub fn index_buffer_transfer_dest() -> Usage {
        Usage {
            index_buffer: true,
            transfer_dest: true,
            .. Usage::none()
        }
    }

    /// Builds a `Usage` with `uniform_buffer` set to true and the rest to false.
    #[inline]
    pub fn uniform_buffer() -> Usage {
        Usage {
            uniform_buffer: true,
            .. Usage::none()
        }
    }

    /// Builds a `Usage` with `uniform_buffer` and `transfer_dest` set to true and the rest
    /// to false.
    #[inline]
    pub fn uniform_buffer_transfer_dest() -> Usage {
        Usage {
            uniform_buffer: true,
            transfer_dest: true,
            .. Usage::none()
        }
    }

    #[inline]
    #[doc(hidden)]
    // TODO: shouldn't be public (it's just temporarily public)
    pub fn to_usage_bits(&self) -> vk::BufferUsageFlagBits {
        let mut result = 0;
        if self.transfer_source { result |= vk::BUFFER_USAGE_TRANSFER_SRC_BIT; }
        if self.transfer_dest { result |= vk::BUFFER_USAGE_TRANSFER_DST_BIT; }
        if self.uniform_texel_buffer { result |= vk::BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT; }
        if self.storage_texel_buffer { result |= vk::BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT; }
        if self.uniform_buffer { result |= vk::BUFFER_USAGE_UNIFORM_BUFFER_BIT; }
        if self.storage_buffer { result |= vk::BUFFER_USAGE_STORAGE_BUFFER_BIT; }
        if self.index_buffer { result |= vk::BUFFER_USAGE_INDEX_BUFFER_BIT; }
        if self.vertex_buffer { result |= vk::BUFFER_USAGE_VERTEX_BUFFER_BIT; }
        if self.indirect_buffer { result |= vk::BUFFER_USAGE_INDIRECT_BUFFER_BIT; }
        result
    }
}
