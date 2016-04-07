use std::marker::PhantomData;
use std::error;
use std::fmt;
use std::mem;
use std::ptr;
use std::sync::Arc;

use buffer::Buffer;
use buffer::BufferSlice;
use buffer::TypedBuffer;
use format::Data as FormatData;

use Error;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

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
