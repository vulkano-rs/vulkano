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

use format::Data as FormatData;

use Error;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

pub use self::traits::Buffer;
pub use self::traits::TypedBuffer;
pub use self::unsafe_buffer::Usage;

pub mod cpu_access;
pub mod immutable;
pub mod staging;
pub mod traits;
pub mod unsafe_buffer;

/// A subpart of a buffer.
///
/// This object doesn't correspond to any Vulkan object. It exists for the programmer's
/// convenience.
///
/// # Example
///
/// TODO: example
///
#[derive(Clone)]
pub struct BufferSlice<'a, T: ?Sized, B: 'a> {
    marker: PhantomData<T>,
    resource: &'a Arc<B>,
    offset: usize,
    size: usize,
}

impl<'a, T: ?Sized, B: 'a> BufferSlice<'a, T, B> {
    /// Returns the buffer that this slice belongs to.
    pub fn buffer(&self) -> &'a Arc<B> {
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

impl<'a, T, B: 'a> BufferSlice<'a, [T], B> {
    /// Returns the number of elements in this slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.size() / mem::size_of::<T>()
    }
}

impl<'a, T: ?Sized, B: 'a> From<&'a Arc<B>> for BufferSlice<'a, T, B>
    where B: TypedBuffer<Content = T>, T: 'static
{
    #[inline]
    fn from(r: &'a Arc<B>) -> BufferSlice<'a, T, B> {
        BufferSlice {
            marker: PhantomData,
            resource: r,
            offset: 0,
            size: r.size(),
        }
    }
}

impl<'a, T, B: 'a> From<BufferSlice<'a, T, B>> for BufferSlice<'a, [T], B>
    where T: 'static
{
    #[inline]
    fn from(r: BufferSlice<'a, T, B>) -> BufferSlice<'a, [T], B> {
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
pub struct BufferView<T, B> where B: Buffer {
    view: vk::BufferView,
    buffer: Arc<B>,
    marker: PhantomData<T>,
}

impl<T, B> BufferView<T, B> where B: TypedBuffer {
    /// Builds a new buffer view.
    ///
    /// The format of the view will be automatically determined by the `T` parameter.
    ///
    /// The buffer must have been created with either the `uniform_texel_buffer` or
    /// the `storage_texel_buffer` usage or an error will occur.
    ///
    // FIXME: how to handle the fact that eg. `u8` can be either Unorm or Uint?
    pub fn new<'a, S>(buffer: S) -> Result<Arc<BufferView<T, B>>, BufferViewCreationError>
        where S: Into<BufferSlice<'a, [T], B>>, B: 'static, T: FormatData + 'static
    {
        let buffer = buffer.into();
        let device = buffer.resource.inner_buffer().device();
        let format = T::ty();

        if !buffer.buffer().inner_buffer().usage_uniform_texel_buffer() &&
           !buffer.buffer().inner_buffer().usage_storage_texel_buffer()
        {
            return Err(BufferViewCreationError::WrongBufferUsage);
        }

        // TODO: check that format is supported? or check only when the view is used?

        let infos = vk::BufferViewCreateInfo {
            sType: vk::STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,   // reserved,
            buffer: buffer.resource.inner_buffer().internal_object(),
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

unsafe impl<T, B> VulkanObject for BufferView<T, B> where B: Buffer {
    type Object = vk::BufferView;

    #[inline]
    fn internal_object(&self) -> vk::BufferView {
        self.view
    }
}

impl<T, B> Drop for BufferView<T, B> where B: Buffer {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.buffer.inner_buffer().device().pointers();
            vk.DestroyBufferView(self.buffer.inner_buffer().device().internal_object(), self.view,
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
            Err(BufferViewCreationError::WrongBufferUsage) => (),
            _ => panic!()
        }
    }
}
