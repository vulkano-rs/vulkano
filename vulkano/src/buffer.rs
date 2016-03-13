//! Location in memory that contains data.
//!
//! All buffers are guaranteed to be accessible from the GPU.
//!
//! The `Buffer` struct has two template parameters:
//!
//! - `T` is the type of data that is contained in the buffer. It can be a struct
//!   (eg. `Foo`), an array (eg. `[u16; 1024]` or `[Foo; 1024]`), or an unsized array (eg. `[u16]`).
//!
//! - `M` is the object that provides memory and handles synchronization for the buffer.
//!   If the `CpuAccessible` and/or `CpuWriteAccessible` traits are implemented on `M`, then you
//!   can access the buffer's content from your program.
//!
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
use std::error;
use std::fmt;
use std::mem;
use std::ptr;
use std::sync::Arc;

use device::Device;
use device::Queue;
use format::Data as FormatData;
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

use Error;
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
    unsafe fn gpu_access(&self, write: bool, offset: usize, size: usize, queue: &Arc<Queue>,
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

/// Data storage in a GPU-accessible location.
///
/// See the module's documentation for more info.
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
    ///
    /// - `usage` indicates how the buffer is going to be used. Using the buffer in a way that
    ///   wasn't declared here will result in an error or a panic.
    /// - `memory` indicates how the memory backing the buffer will be allocated.
    /// - `sharing` indicates which queue family or queue families are going to use the buffer.
    ///   Just like `usage`, using the buffer in a different queue family will result in an error
    ///   or panic.
    ///
    /// This function is suitable when the type of buffer is `Sized`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use vulkano::device::Device;
    /// # use vulkano::device::Queue;
    /// use vulkano::buffer::Buffer;
    /// use vulkano::buffer::Usage as BufferUsage;
    /// use vulkano::memory::HostVisible;
    /// # let device: Device = unsafe { ::std::mem::uninitialized() };
    /// # let queue: Queue = unsafe { ::std::mem::uninitialized() };
    /// struct Data {
    ///     matrix: [[f32; 4]; 4],
    ///     color: [f32; 3],
    /// }
    ///
    /// let usage = BufferUsage {
    ///     transfer_dest: true,
    ///     uniform_buffer: true,
    ///     .. BufferUsage::none()
    /// };
    ///
    /// let _buffer: Buffer<Data, _> = Buffer::new(&device, &usage, HostVisible, &queue).unwrap();
    /// ```
    ///
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
    /// Creates a new buffer with a number of elements known at runtime.
    ///
    /// See `new` for more information about the parameters.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use vulkano::device::Device;
    /// # use vulkano::device::Queue;
    /// use vulkano::buffer::Buffer;
    /// use vulkano::buffer::Usage as BufferUsage;
    /// use vulkano::memory::HostVisible;
    /// # let device: Device = unsafe { ::std::mem::uninitialized() };
    /// # let queue: Queue = unsafe { ::std::mem::uninitialized() };
    /// let usage = BufferUsage {
    ///     index_buffer: true,
    ///     .. BufferUsage::none()
    /// };
    ///
    /// let buffer: Buffer<[u16], _> = Buffer::array(&device, 1024, &usage,
    ///                                              HostVisible, &queue).unwrap();
    /// assert_eq!(buffer.len(), 1024);
    /// ```
    ///
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
    /// Creates a new buffer of the given size without checking whether there is enough memory
    /// to hold the type.
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
    unsafe fn gpu_access(&self, write: bool, offset: usize, size: usize, queue: &Arc<Queue>,
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
    /// If the buffer is in use by the GPU, blocks until it is available or until the timeout
    /// has elapsed.
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
    /// If the buffer is in use by the GPU, blocks until it is available or until the timeout
    /// has elapsed.
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
///
/// # Example
///
/// ```no_run
/// # use vulkano::device::Device;
/// # use vulkano::device::Queue;
/// use vulkano::buffer::Buffer;
/// use vulkano::buffer::BufferSlice;
/// use vulkano::buffer::Usage as BufferUsage;
/// use vulkano::memory::HostVisible;
/// # let device: Device = unsafe { ::std::mem::uninitialized() };
/// # let queue: Queue = unsafe { ::std::mem::uninitialized() };
/// let usage = BufferUsage {
///     index_buffer: true,
///     .. BufferUsage::none()
/// };
///
/// let buffer: Buffer<[u16], _> = Buffer::array(&device, 1024, &usage,
///                                              HostVisible, &queue).unwrap();
///
/// let slice = BufferSlice::from(&buffer);
/// TODO: add slice.slice() or something to show that it's useful
/// ```
///
#[derive(Clone)]
pub struct BufferSlice<'a, T: ?Sized, O: ?Sized + 'a, M: 'a> {
    marker: PhantomData<T>,
    resource: &'a Arc<Buffer<O, M>>,
    offset: usize,
    size: usize,
}

impl<'a, T: ?Sized, O: ?Sized, M> BufferSlice<'a, T, O, M> {
    /// Returns the buffer that this slice belongs to.
    pub fn buffer(&self) -> &Arc<Buffer<O, M>> {
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

impl<'a, T, O: ?Sized, M> BufferSlice<'a, [T], O, M> {
    /// Returns the number of elements in this slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.size() / mem::size_of::<T>()
    }
}

impl<'a, T: ?Sized, M> From<&'a Arc<Buffer<T, M>>> for BufferSlice<'a, T, T, M>
    where M: MemorySourceChunk
{
    #[inline]
    fn from(r: &'a Arc<Buffer<T, M>>) -> BufferSlice<'a, T, T, M> {
        BufferSlice {
            marker: PhantomData,
            resource: r,
            offset: 0,
            size: r.inner.size,
        }
    }
}

impl<'a, T, O: ?Sized, M> From<BufferSlice<'a, T, O, M>> for BufferSlice<'a, [T], O, M> {
    #[inline]
    fn from(r: BufferSlice<'a, T, O, M>) -> BufferSlice<'a, [T], O, M> {
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
pub struct BufferView<T, O: ?Sized, M> {
    view: vk::BufferView,
    buffer: Arc<Buffer<O, M>>,
    marker: PhantomData<T>,
}

impl<T, O: ?Sized, M> BufferView<T, O, M> {
    /// Builds a new buffer view.
    ///
    /// The format of the view will be automatically determined by the `T` parameter.
    ///
    /// The buffer must have been created with either the `uniform_texel_buffer` or
    /// the `storage_texel_buffer` usage or an error will occur.
    ///
    pub fn new<'a, S>(buffer: S) -> Result<Arc<BufferView<T, O, M>>, BufferViewCreationError>
        where S: Into<BufferSlice<'a, [T], O, M>>, T: FormatData, M: MemorySourceChunk + 'static,
              O: 'static
    {
        let buffer = buffer.into();
        let device = buffer.resource.device();
        let format = T::ty();

        if !buffer.buffer().usage_uniform_texel_buffer() &&
           !buffer.buffer().usage_storage_texel_buffer()
        {
            return Err(BufferViewCreationError::WrongBufferUsage);
        }

        // TODO: check that format is supported? or check only when the view is used?

        let infos = vk::BufferViewCreateInfo {
            sType: vk::STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,   // reserved,
            buffer: buffer.resource.internal_object(),
            format: format as u32,
            offset: buffer.offset as u64,
            range: buffer.size as u64,
        };

        let view = unsafe {
            let vk = device.pointers();
            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateBufferView(device.internal_object(), &infos,
                                                  ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(BufferView {
            view: view,
            buffer: buffer.resource.clone(),
            marker: PhantomData,
        }))
    }
}

unsafe impl<T, O: ?Sized, M> VulkanObject for BufferView<T, O, M> {
    type Object = vk::BufferView;

    #[inline]
    fn internal_object(&self) -> vk::BufferView {
        self.view
    }
}

impl<T, O: ?Sized, M> Drop for BufferView<T, O, M> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.buffer.device().pointers();
            vk.DestroyBufferView(self.buffer.inner.device.internal_object(), self.view,
                                 ptr::null());
        }
    }
}

/// Error that can happen when creating a buffer view.
#[derive(Debug, Copy, Clone)]
pub enum BufferViewCreationError {
    /// Out of memory.
    OomError(OomError),

    /// The buffer was not creating with one of the `storage_texel_buffer` or
    /// `uniform_texel_buffer` usages.
    WrongBufferUsage,
}

impl error::Error for BufferViewCreationError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            BufferViewCreationError::OomError(_) => "out of memory when creating buffer view",
            BufferViewCreationError::WrongBufferUsage => "the buffer is missing correct usage \
                                                          flags",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            BufferViewCreationError::OomError(ref err) => Some(err),
            _ => None,
        }
    }
}

impl fmt::Display for BufferViewCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

impl From<OomError> for BufferViewCreationError {
    #[inline]
    fn from(err: OomError) -> BufferViewCreationError {
        BufferViewCreationError::OomError(err)
    }
}

impl From<Error> for BufferViewCreationError {
    #[inline]
    fn from(err: Error) -> BufferViewCreationError {
        OomError::from(err).into()
    }
}

#[cfg(test)]
mod tests {
    use std::mem;

    use buffer::Usage;
    use buffer::Buffer;
    use buffer::BufferView;
    use buffer::BufferViewCreationError;
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

    #[test]
    fn view_create() {
        let (device, queue) = gfx_dev_and_queue!();

        let buffer = Buffer::<[i8], _>::array(&device, 128, &Usage::all(), DeviceLocal,
                                              &queue).unwrap();
        let _ = BufferView::new(&buffer).unwrap();
    }

    #[test]
    fn view_wrong_usage() {
        let (device, queue) = gfx_dev_and_queue!();

        let buffer = Buffer::<[i8], _>::array(&device, 128, &Usage::none(), DeviceLocal,
                                              &queue).unwrap();

        match BufferView::new(&buffer) {
            BufferViewCreationError::WrongBufferUsage => (),
            _ => panic!()
        }
    }
}
