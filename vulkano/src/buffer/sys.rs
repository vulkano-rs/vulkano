// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Low level implementation of buffers.
//! 
//! Wraps directly around Vulkan buffers, with the exceptions of a few safety checks.
//! 
//! The `UnsafeBuffer` type is the lowest-level buffer object provided by this library. It is used
//! internally by the higher-level buffer types. You are strongly encouraged to have excellent
//! knowledge of the Vulkan specs if you want to use an `UnsafeBuffer`.
//! 
//! Here is what you must take care of when you use an `UnsafeBuffer`:
//! 
//! - Synchronization, ie. avoid reading and writing simultaneously to the same buffer.
//! - Memory aliasing considerations. If you use the same memory to back multiple resources, you
//!   must ensure that they are not used together and must enable some additional flags.
//! - Binding memory correctly and only once. If you use sparse binding, respect the rules of
//!   sparse binding.
//! - Type safety.

use std::error;
use std::fmt;
use std::mem;
use std::ptr;
use std::sync::Arc;
use smallvec::SmallVec;

use device::Device;
use memory::DeviceMemory;
use memory::MemoryRequirements;
use sync::Sharing;

use check_errors;
use Error;
use OomError;
use VulkanObject;
use VulkanPointers;
use vk;

/// Data storage in a GPU-accessible location.
#[derive(Debug)]
pub struct UnsafeBuffer {
    buffer: vk::Buffer,
    device: Arc<Device>,
    size: usize,
    usage: vk::BufferUsageFlags,
    exclusive_sharing_mode: bool,
}

impl UnsafeBuffer {
    /// Creates a new buffer of the given size.
    ///
    /// See the module's documentation for information about safety.
    ///
    /// # Panic
    ///
    /// Panicks if `sparse.sparse` is false and `sparse.sparse_residency` or
    /// `sparse.sparse_aliased` is true.
    ///
    pub unsafe fn new<'a, I>(device: &Arc<Device>, size: usize, usage: &Usage, sharing: Sharing<I>,
                             sparse: SparseLevel)
                             -> Result<(UnsafeBuffer, MemoryRequirements), BufferCreationError>
        where I: Iterator<Item = u32>
    {
        let vk = device.pointers();

        let usage_bits = usage.to_usage_bits();

        // Checking sparse features.
        assert!(sparse.sparse || !sparse.sparse_residency, "Can't enable sparse residency without \
                                                            enabling sparse binding as well");
        assert!(sparse.sparse || !sparse.sparse_aliased, "Can't enable sparse aliasing without \
                                                          enabling sparse binding as well");
        if sparse.sparse && !device.enabled_features().sparse_binding {
            return Err(BufferCreationError::SparseBindingFeatureNotEnabled);
        }
        if sparse.sparse_residency && !device.enabled_features().sparse_residency_buffer {
            return Err(BufferCreationError::SparseResidencyBufferFeatureNotEnabled);
        }
        if sparse.sparse_aliased && !device.enabled_features().sparse_residency_aliased {
            return Err(BufferCreationError::SparseResidencyAliasedFeatureNotEnabled);
        }

        let (buffer, exclusive_sharing_mode) = {
            let (sh_mode, sh_indices) = match sharing {
                Sharing::Exclusive => (vk::SHARING_MODE_EXCLUSIVE, SmallVec::<[u32; 8]>::new()),
                Sharing::Concurrent(ids) => (vk::SHARING_MODE_CONCURRENT, ids.collect()),
            };

            let exclusive_sharing_mode = sh_mode == vk::SHARING_MODE_EXCLUSIVE;

            let infos = vk::BufferCreateInfo {
                sType: vk::STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                pNext: ptr::null(),
                flags: sparse.to_flags(),
                size: size as u64,
                usage: usage_bits,
                sharingMode: sh_mode,
                queueFamilyIndexCount: sh_indices.len() as u32,
                pQueueFamilyIndices: sh_indices.as_ptr(),
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateBuffer(device.internal_object(), &infos,
                                              ptr::null(), &mut output)));
            (output, exclusive_sharing_mode)
        };

        let mem_reqs = {
            #[inline] fn align(val: usize, al: usize) -> usize { al * (1 + (val - 1) / al) }

            let mut output: vk::MemoryRequirements = mem::uninitialized();
            vk.GetBufferMemoryRequirements(device.internal_object(), buffer, &mut output);
            debug_assert!(output.size >= size as u64);
            debug_assert!(output.memoryTypeBits != 0);

            let mut output: MemoryRequirements = output.into();

            // We have to manually enforce some additional requirements for some buffer types.
            let limits = device.physical_device().limits();
            if usage.uniform_texel_buffer || usage.storage_texel_buffer {
                output.alignment = align(output.alignment,
                                         limits.min_texel_buffer_offset_alignment() as usize);
            }

            if usage.storage_buffer {
                output.alignment = align(output.alignment,
                                         limits.min_storage_buffer_offset_alignment() as usize);
            }

            if usage.uniform_buffer {
                output.alignment = align(output.alignment,
                                         limits.min_uniform_buffer_offset_alignment() as usize);
            }

            output
        };

        let obj = UnsafeBuffer {
            buffer: buffer,
            device: device.clone(),
            size: size as usize,
            usage: usage_bits,
            exclusive_sharing_mode: exclusive_sharing_mode,
        };

        Ok((obj, mem_reqs))
    }

    pub unsafe fn bind_memory(&self, memory: &DeviceMemory, offset: usize)
                              -> Result<(), OomError>
    {
        let vk = self.device.pointers();

        // We check for correctness in debug mode.
        debug_assert!({
            let mut mem_reqs = mem::uninitialized();
            vk.GetBufferMemoryRequirements(self.device.internal_object(), self.buffer,
                                           &mut mem_reqs);
            mem_reqs.size <= (memory.size() - offset) as u64 &&
            (offset as u64 % mem_reqs.alignment) == 0 &&
            mem_reqs.memoryTypeBits & (1 << memory.memory_type().id()) != 0
        });

        // Check for alignment correctness.
        {
            let limits = self.device().physical_device().limits();
            if self.usage_uniform_texel_buffer() || self.usage_storage_texel_buffer() {
                debug_assert!(offset % limits.min_texel_buffer_offset_alignment() as usize == 0);
            }
            if self.usage_storage_buffer() {
                debug_assert!(offset % limits.min_storage_buffer_offset_alignment() as usize == 0);
            }
            if self.usage_uniform_buffer() {
                debug_assert!(offset % limits.min_uniform_buffer_offset_alignment() as usize == 0);
            }
        }

        try!(check_errors(vk.BindBufferMemory(self.device.internal_object(), self.buffer,
                                              memory.internal_object(), offset as vk::DeviceSize)));
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

    /// If true, this buffer was created with the exclusive sharing mode. If false, it was created
    /// with the concurrent sharing mode.
    #[inline]
    pub fn exclusive_sharing_mode(&self) -> bool {
        self.exclusive_sharing_mode
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

#[derive(Debug, Copy, Clone)]
pub struct SparseLevel {
    pub sparse: bool,
    pub sparse_residency: bool,
    pub sparse_aliased: bool,
}

impl SparseLevel {
    #[inline]
    pub fn none() -> SparseLevel {
        SparseLevel {
            sparse: false,
            sparse_residency: false,
            sparse_aliased: false,
        }
    }

    #[inline]
    fn to_flags(&self) -> vk::BufferCreateFlagBits {
        let mut result = 0;
        if self.sparse { result |= vk::BUFFER_CREATE_SPARSE_BINDING_BIT; }
        if self.sparse_residency { result |= vk::BUFFER_CREATE_SPARSE_RESIDENCY_BIT; }
        if self.sparse_aliased { result |= vk::BUFFER_CREATE_SPARSE_ALIASED_BIT; }
        result
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

    /// Builds a `Usage` with `indirect_buffer` set to true and the rest to false.
    #[inline]
    pub fn indirect_buffer() -> Usage {
        Usage {
            indirect_buffer: true,
            .. Usage::none()
        }
    }

    /// Builds a `Usage` with `indirect_buffer` and `transfer_dest` set to true and the rest
    /// to false.
    #[inline]
    pub fn indirect_buffer_transfer_dest() -> Usage {
        Usage {
            indirect_buffer: true,
            transfer_dest: true,
            .. Usage::none()
        }
    }

    #[inline]
    fn to_usage_bits(&self) -> vk::BufferUsageFlagBits {
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

/// Error that can happen when creating a buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BufferCreationError {
    /// Not enough memory.
    OomError(OomError),
    /// Sparse binding was requested but the corresponding feature wasn't enabled.
    SparseBindingFeatureNotEnabled,
    /// Sparse residency was requested but the corresponding feature wasn't enabled.
    SparseResidencyBufferFeatureNotEnabled,
    /// Sparse aliasing was requested but the corresponding feature wasn't enabled.
    SparseResidencyAliasedFeatureNotEnabled,
}

impl error::Error for BufferCreationError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            BufferCreationError::OomError(_) => "not enough memory available",
            BufferCreationError::SparseBindingFeatureNotEnabled => {
                "sparse binding was requested but the corresponding feature wasn't enabled"
            },
            BufferCreationError::SparseResidencyBufferFeatureNotEnabled => {
                "sparse residency was requested but the corresponding feature wasn't enabled"
            },
            BufferCreationError::SparseResidencyAliasedFeatureNotEnabled => {
                "sparse aliasing was requested but the corresponding feature wasn't enabled"
            },
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            BufferCreationError::OomError(ref err) => Some(err),
            _ => None
        }
    }
}

impl fmt::Display for BufferCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<OomError> for BufferCreationError {
    #[inline]
    fn from(err: OomError) -> BufferCreationError {
        BufferCreationError::OomError(err)
    }
}

impl From<Error> for BufferCreationError {
    #[inline]
    fn from(err: Error) -> BufferCreationError {
        match err {
            err @ Error::OutOfHostMemory => BufferCreationError::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => BufferCreationError::OomError(OomError::from(err)),
            _ => panic!("unexpected error: {:?}", err)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter::Empty;

    use super::BufferCreationError;
    use super::SparseLevel;
    use super::UnsafeBuffer;
    use super::Usage;

    use device::Device;
    use sync::Sharing;

    #[test]
    fn create() {
        let (device, _) = gfx_dev_and_queue!();
        let (buf, reqs) = unsafe {
            UnsafeBuffer::new(&device, 128, &Usage::all(), Sharing::Exclusive::<Empty<_>>,
                              SparseLevel::none())
        }.unwrap();

        assert!(reqs.size >= 128);
        assert_eq!(buf.size(), 128);
        assert_eq!(&**buf.device() as *const Device, &*device as *const Device);
    }

    #[test]
    #[should_panic = "Can't enable sparse residency without enabling sparse binding as well"]
    fn panic_wrong_sparse_residency() {
        let (device, _) = gfx_dev_and_queue!();
        let sparse = SparseLevel { sparse: false, sparse_residency: true, sparse_aliased: false };
        let _ = unsafe {
            UnsafeBuffer::new(&device, 128, &Usage::all(), Sharing::Exclusive::<Empty<_>>,
                              sparse)
        };
    }

    #[test]
    #[should_panic = "Can't enable sparse aliasing without enabling sparse binding as well"]
    fn panic_wrong_sparse_aliased() {
        let (device, _) = gfx_dev_and_queue!();
        let sparse = SparseLevel { sparse: false, sparse_residency: false, sparse_aliased: true };
        let _ = unsafe {
            UnsafeBuffer::new(&device, 128, &Usage::all(), Sharing::Exclusive::<Empty<_>>,
                              sparse)
        };
    }

    #[test]
    fn missing_feature_sparse_binding() {
        let (device, _) = gfx_dev_and_queue!();
        let sparse = SparseLevel { sparse: true, sparse_residency: false, sparse_aliased: false };
        unsafe {
            match UnsafeBuffer::new(&device, 128, &Usage::all(), Sharing::Exclusive::<Empty<_>>,
                                    sparse)
            {
                Err(BufferCreationError::SparseBindingFeatureNotEnabled) => (),
                _ => panic!()
            }
        };
    }

    #[test]
    fn missing_feature_sparse_residency() {
        let (device, _) = gfx_dev_and_queue!(sparse_binding);
        let sparse = SparseLevel { sparse: true, sparse_residency: true, sparse_aliased: false };
        unsafe {
            match UnsafeBuffer::new(&device, 128, &Usage::all(), Sharing::Exclusive::<Empty<_>>,
                                    sparse)
            {
                Err(BufferCreationError::SparseResidencyBufferFeatureNotEnabled) => (),
                _ => panic!()
            }
        };
    }

    #[test]
    fn missing_feature_sparse_aliased() {
        let (device, _) = gfx_dev_and_queue!(sparse_binding);
        let sparse = SparseLevel { sparse: true, sparse_residency: false, sparse_aliased: true };
        unsafe {
            match UnsafeBuffer::new(&device, 128, &Usage::all(), Sharing::Exclusive::<Empty<_>>,
                                    sparse)
            {
                Err(BufferCreationError::SparseResidencyAliasedFeatureNotEnabled) => (),
                _ => panic!()
            }
        };
    }
}
