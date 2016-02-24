//! Location in memory that contains data.
//!
//! All buffers are guaranteed to be accessible from the GPU.
//! 
//! # Strong typing
//! 
//! All buffers take a template parameter that indicates their content.
//! 
//! # Memory
//! 
//! Creating a buffer requires passing an object that will be used by this library to provide
//! memory to the buffer.
//! 
//! All accesses to the memory are done through the `Buffer` object.
//! 
//! TODO: proof read this section
//!
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::sync::Arc;

use device::Device;
use device::Queue;
use memory::CpuAccessible;
use memory::CpuWriteAccessible;
use memory::ChunkProperties;
use memory::ChunkRange;
use memory::MemorySource;
use memory::MemorySourceChunk;
use sync::Fence;
use sync::Resource;
use sync::Semaphore;
use sync::SharingMode;

use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

pub unsafe trait AbstractBuffer: Resource + ::VulkanObjectU64 {
    /// Returns the size of the buffer in bytes.
    fn size(&self) -> usize;

    /// Instructs the resource that it is going to be used by the GPU soon in the future. The
    /// function should block if the memory is currently being accessed by the CPU.
    ///
    /// `write` indicates whether the GPU will write to the memory. If `false`, then it will only
    /// be written.
    ///
    /// `queue` is the queue where the command buffer that accesses the memory will be submitted.
    /// If the `gpu_access` function submits something to that queue, it will thus be submitted
    /// beforehand. This behavior can be used for example to submit sparse binding commands.
    ///
    /// `fence` is a fence that will be signaled when this GPU access will stop. It should be
    /// waited upon whenever the user wants to read this memory from the CPU. If `requires_fence`
    /// returned false, then this value will be `None`.
    ///
    /// `semaphore` is a semaphore that will be signaled when this GPU access will stop. This value
    /// is intended to be returned later, in a follow-up call to `gpu_access`. If
    /// `requires_semaphore` returned false, then this value will be `None`.
    ///
    /// The function can return a semaphore which will be waited up by the GPU before the
    /// work starts.
    unsafe fn gpu_access(&self, write: bool, offset: usize, size: usize, queue: &mut Queue,
                         fence: Option<Arc<Fence>>, semaphore: Option<Arc<Semaphore>>)
                         -> Option<Arc<Semaphore>>;

    /// True if the buffer can be used as a source for buffer transfers.
    fn usage_transfer_src(&self) -> bool;
    /// True if the buffer can be used as a destination for buffer transfers.
    fn usage_transfer_dest(&self) -> bool;
    /// True if the buffer can be used as
    fn usage_uniform_texel_buffer(&self) -> bool;
    /// True if the buffer can be used as
    fn usage_storage_texel_buffer(&self) -> bool;
    /// True if the buffer can be used as
    fn usage_uniform_buffer(&self) -> bool;
    /// True if the buffer can be used as
    fn usage_storage_buffer(&self) -> bool;
    /// True if the buffer can be used as a source for index data.
    fn usage_index_buffer(&self) -> bool;
    /// True if the buffer can be used as a source for vertex data.
    fn usage_vertex_buffer(&self) -> bool;
    /// True if the buffer can be used as an indirect buffer.
    fn usage_indirect_buffer(&self) -> bool;
}

pub struct Buffer<T: ?Sized, M> {
    marker: PhantomData<T>,
    inner: Inner<M>,
}

struct Inner<M> {
    device: Arc<Device>,
    memory: M,
    buffer: vk::Buffer,
    size: usize,
    usage: vk::BufferUsageFlags,
    sharing: SharingMode,
}

impl<T, M> Buffer<T, M> where M: MemorySourceChunk {
    /// Creates a new buffer.
    pub fn new<S, Sh>(device: &Arc<Device>, usage: &Usage, memory: S, sharing: Sh)
                      -> Result<Arc<Buffer<T, M>>, OomError>
        where S: MemorySource<Chunk = M>, Sh: Into<SharingMode>
    {
        unsafe {
            Buffer::raw(device, mem::size_of::<T>(), usage, memory, sharing)
        }
    }
}

impl<T, M> Buffer<[T], M> where M: MemorySourceChunk {
    /// Creates a new buffer with a number of elements.
    pub fn array<S, Sh>(device: &Arc<Device>, len: usize, usage: &Usage, memory: S, sharing: Sh)
                        -> Result<Arc<Buffer<[T], M>>, OomError>
        where S: MemorySource<Chunk = M>, Sh: Into<SharingMode>
    {
        unsafe {
            Buffer::raw(device, len * mem::size_of::<T>(), usage, memory, sharing)
        }
    }
}

impl<T: ?Sized, M> Buffer<T, M> where M: MemorySourceChunk {
    /// Creates a new buffer of the given size without checking whether the type is correct.
    ///
    /// # Safety
    ///
    /// - Type safety is not checked.
    ///
    pub unsafe fn raw<S, Sh>(device: &Arc<Device>, size: usize, usage: &Usage, memory: S,
                             sharing: Sh) -> Result<Arc<Buffer<T, M>>, OomError>
        where S: MemorySource<Chunk = M>, Sh: Into<SharingMode>
    {
        let vk = device.pointers();

        let usage = usage.to_usage_bits();
        let sharing = sharing.into();

        assert!(!memory.is_sparse());       // not implemented

        let buffer = {
            let (sh_mode, sh_count, sh_indices) = match sharing {
                SharingMode::Exclusive(id) => (vk::SHARING_MODE_EXCLUSIVE, 0, ptr::null()),
                SharingMode::Concurrent(ref ids) => (vk::SHARING_MODE_CONCURRENT, ids.len() as u32,
                                                     ids.as_ptr()),
            };

            let infos = vk::BufferCreateInfo {
                sType: vk::STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,       // TODO: sparse resources binding
                size: size as u64,
                usage: usage,
                sharingMode: sh_mode,
                queueFamilyIndexCount: sh_count,
                pQueueFamilyIndices: sh_indices,
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateBuffer(device.internal_object(), &infos,
                                              ptr::null(), &mut output)));
            output
        };

        let mem_reqs: vk::MemoryRequirements = {
            let mut output = mem::uninitialized();
            vk.GetBufferMemoryRequirements(device.internal_object(), buffer, &mut output);
            output
        };

        let memory = memory.allocate(device, mem_reqs.size as usize, mem_reqs.alignment as usize,
                                     mem_reqs.memoryTypeBits)
                           .expect("failed to allocate");     // TODO: use try!() instead

        match memory.properties() {
            ChunkProperties::Regular { memory, offset, .. } => {
                try!(check_errors(vk.BindBufferMemory(device.internal_object(), buffer,
                                                      memory.internal_object(),
                                                      offset as vk::DeviceSize)));
            },
            _ => unimplemented!()
        }

        Ok(Arc::new(Buffer {
            marker: PhantomData,
            inner: Inner {
                device: device.clone(),
                memory: memory,
                buffer: buffer,
                size: size as usize,
                usage: usage,
                sharing: sharing,
            }
        }))
    }
}

impl<T: ?Sized, M> Buffer<T, M> {
    /// Returns the device used to create this buffer.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.inner.device
    }

    /// Returns the size of the buffer in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.inner.size
    }
}

impl<T, M> Buffer<[T], M> {
    /// Returns the number of elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.size() / mem::size_of::<T>()
    }
}

unsafe impl<T: ?Sized, M> Resource for Buffer<T, M> where M: MemorySourceChunk {
    #[inline]
    fn requires_fence(&self) -> bool {
        self.inner.memory.requires_fence()
    }

    #[inline]
    fn requires_semaphore(&self) -> bool {
        self.inner.memory.requires_semaphore()
    }

    #[inline]
    fn sharing_mode(&self) -> &SharingMode {
        &self.inner.sharing
    }
}

unsafe impl<T: ?Sized, M> AbstractBuffer for Buffer<T, M> where M: MemorySourceChunk {
    #[inline]
    fn size(&self) -> usize {
        self.inner.size
    }

    #[inline]
    unsafe fn gpu_access(&self, write: bool, offset: usize, size: usize, queue: &mut Queue,
                         fence: Option<Arc<Fence>>, semaphore: Option<Arc<Semaphore>>)
                         -> Option<Arc<Semaphore>>
    {
        self.inner.memory.gpu_access(write, ChunkRange::Range { offset: offset, size: size },
                                     queue, fence, semaphore)
    }

    #[inline]
    fn usage_transfer_src(&self) -> bool {
        (self.inner.usage & vk::BUFFER_USAGE_TRANSFER_SRC_BIT) != 0
    }

    #[inline]
    fn usage_transfer_dest(&self) -> bool {
        (self.inner.usage & vk::BUFFER_USAGE_TRANSFER_DST_BIT) != 0
    }

    #[inline]
    fn usage_uniform_texel_buffer(&self) -> bool {
        (self.inner.usage & vk::BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT) != 0
    }

    #[inline]
    fn usage_storage_texel_buffer(&self) -> bool {
        (self.inner.usage & vk::BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT) != 0
    }

    #[inline]
    fn usage_uniform_buffer(&self) -> bool {
        (self.inner.usage & vk::BUFFER_USAGE_UNIFORM_BUFFER_BIT) != 0
    }

    #[inline]
    fn usage_storage_buffer(&self) -> bool {
        (self.inner.usage & vk::BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0
    }

    #[inline]
    fn usage_index_buffer(&self) -> bool {
        (self.inner.usage & vk::BUFFER_USAGE_INDEX_BUFFER_BIT) != 0
    }

    #[inline]
    fn usage_vertex_buffer(&self) -> bool {
        (self.inner.usage & vk::BUFFER_USAGE_VERTEX_BUFFER_BIT) != 0
    }

    #[inline]
    fn usage_indirect_buffer(&self) -> bool {
        (self.inner.usage & vk::BUFFER_USAGE_INDIRECT_BUFFER_BIT) != 0
    }
}

impl<'a, T: ?Sized, M> Buffer<T, M> where M: CpuAccessible<'a, T> {
    /// Gives a read access to the content of the buffer.
    ///
    /// If the buffer is in use by the GPU, blocks until it is available.
    #[inline]
    pub fn read(&'a self, timeout_ns: u64) -> M::Read {
        self.inner.memory.read(timeout_ns)
    }

    /// Tries to give a read access to the content of the buffer.
    ///
    /// If the buffer is in use by the GPU, returns `None`.
    #[inline]
    pub fn try_read(&'a self) -> Option<M::Read> {
        self.inner.memory.try_read()
    }
}

impl<'a, T: ?Sized, M> Buffer<T, M> where M: CpuWriteAccessible<'a, T> {
    /// Gives a write access to the content of the buffer.
    ///
    /// If the buffer is in use by the GPU, blocks until it is available.
    #[inline]
    pub fn write(&'a self, timeout_ns: u64) -> M::Write {
        self.inner.memory.write(timeout_ns)
    }

    /// Tries to give a write access to the content of the buffer.
    ///
    /// If the buffer is in use by the GPU, returns `None`.
    #[inline]
    pub fn try_write(&'a self) -> Option<M::Write> {
        self.inner.memory.try_write()
    }
}

unsafe impl<'a, T: ?Sized, M> CpuAccessible<'a, T> for Buffer<T, M>
    where M: CpuAccessible<'a, T>
{
    type Read = M::Read;

    #[inline]
    fn read(&'a self, timeout_ns: u64) -> M::Read {
        self.read(timeout_ns)
    }

    #[inline]
    fn try_read(&'a self) -> Option<M::Read> {
        self.try_read()
    }
}

unsafe impl<'a, T: ?Sized, M> CpuWriteAccessible<'a, T> for Buffer<T, M>
    where M: CpuWriteAccessible<'a, T>
{
    type Write = M::Write;

    #[inline]
    fn write(&'a self, timeout_ns: u64) -> M::Write {
        self.write(timeout_ns)
    }

    #[inline]
    fn try_write(&'a self) -> Option<M::Write> {
        self.try_write()
    }
}

unsafe impl<T: ?Sized, M> VulkanObject for Buffer<T, M> {
    type Object = vk::Buffer;

    #[inline]
    fn internal_object(&self) -> vk::Buffer {
        self.inner.buffer
    }
}

impl<T: ?Sized, M> Drop for Buffer<T, M> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.inner.device.pointers();
            vk.DestroyBuffer(self.inner.device.internal_object(), self.inner.buffer, ptr::null());
        }
    }
}

/// Describes how a buffer is going to be used. This is **not** an optimization.
///
/// If you try to use a buffer in a way that you didn't declare, a panic will happen.
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

/// A subpart of a buffer.
///
/// This object doesn't correspond to any Vulkan object. It exists for the programmer's
/// convenience.
#[derive(Clone)]
pub struct BufferSlice<T: ?Sized> {
    marker: PhantomData<T>,
    resource: Arc<AbstractBuffer>,
    offset: usize,
    size: usize,
}

impl<T: ?Sized> BufferSlice<T> {
    /// Returns the buffer that this slice belongs to.
    pub fn buffer(&self) -> &Arc<AbstractBuffer> {
        &self.resource
    }

    /// Returns the offset of that slice within the buffer.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the size of that slice in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }
}

impl<T> BufferSlice<[T]> {
    /// Returns the number of elements in this slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.size() / mem::size_of::<T>()
    }
}

impl<'a, T: ?Sized + 'static, M: 'static> From<&'a Arc<Buffer<T, M>>> for BufferSlice<T>
    where M: MemorySourceChunk
{
    #[inline]
    fn from(r: &'a Arc<Buffer<T, M>>) -> BufferSlice<T> {
        BufferSlice {
            marker: PhantomData,
            resource: r.clone(),
            offset: 0,
            size: r.inner.size,
        }
    }
}

impl<T> From<BufferSlice<T>> for BufferSlice<[T]> {
    #[inline]
    fn from(r: BufferSlice<T>) -> BufferSlice<[T]> {
        BufferSlice {
            marker: PhantomData,
            resource: r.resource,
            offset: r.offset,
            size: r.size,
        }
    }
}

/// Represents a way for the GPU to interpret buffer data.
///
/// Note that a buffer view is only required for some operations. For example using a buffer as a
/// uniform buffer doesn't require creating a `BufferView`.
pub struct BufferView<T: ?Sized, M> {
    buffer: Arc<Buffer<T, M>>,
}

#[cfg(test)]
mod tests {
    use std::mem;

    use buffer::Usage;
    use buffer::Buffer;
    use memory::DeviceLocal;

    #[test]
    fn create() {
        let (device, queue) = gfx_dev_and_queue!();

        let _ = Buffer::<[i8; 16], _>::new(&device, &Usage::all(), DeviceLocal, &queue).unwrap();
    }

    #[test]
    fn array_len() {
        let (device, queue) = gfx_dev_and_queue!();

        let b = Buffer::<[i16], _>::array(&device, 12, &Usage::all(),
                                          DeviceLocal, &queue).unwrap();
        assert_eq!(b.len(), 12);
        assert_eq!(b.size(), 12 * mem::size_of::<i16>());
    }
}
