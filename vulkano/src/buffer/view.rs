use std::marker::PhantomData;
use std::error;
use std::fmt;
use std::mem;
use std::ptr;
use std::sync::Arc;

use buffer::Buffer;
use buffer::BufferSlice;
use format::StrongStorage;

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
pub struct BufferView<F, B> where B: Buffer {
    view: vk::BufferView,
    buffer: Arc<B>,
    marker: PhantomData<F>,
}

impl<F, B> BufferView<F, B> where B: Buffer {
    /// Builds a new buffer view.
    ///
    /// The format of the view will be automatically determined by the `T` parameter.
    ///
    /// The buffer must have been created with either the `uniform_texel_buffer` or
    /// the `storage_texel_buffer` usage or an error will occur.
    pub fn new<'a, S>(buffer: S, format: F)
                      -> Result<Arc<BufferView<F, B>>, BufferViewCreationError>
        where S: Into<BufferSlice<'a, [F::Pixel], B>>, B: 'static, F: StrongStorage + 'static
    {
        let buffer = buffer.into();
        let device = buffer.resource.inner_buffer().device();
        let format = format.format();

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

unsafe impl<F, B> VulkanObject for BufferView<F, B> where B: Buffer {
    type Object = vk::BufferView;

    #[inline]
    fn internal_object(&self) -> vk::BufferView {
        self.view
    }
}

impl<F, B> Drop for BufferView<F, B> where B: Buffer {
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
    use buffer::Buffer;
    use buffer::BufferView;
    use buffer::sys::Usage;
    use buffer::view::BufferViewCreationError;
    use buffer::immutable::ImmutableBuffer;
    use format;

    #[test]
    fn create_uniform() {
        let (device, queue) = gfx_dev_and_queue!();

        let usage = Usage {
            uniform_texel_buffer: true,
            .. Usage::none()
        };

        let buffer = ImmutableBuffer::<[i8]>::array(&device, 128, &usage,
                                                    Some(queue.family())).unwrap();
        let _ = BufferView::new(&buffer, format::R8Sscaled).unwrap();
    }

    #[test]
    fn create_storage() {
        let (device, queue) = gfx_dev_and_queue!();

        let usage = Usage {
            storage_texel_buffer: true,
            .. Usage::none()
        };

        let buffer = ImmutableBuffer::<[i8]>::array(&device, 128, &usage,
                                                    Some(queue.family())).unwrap();
        let _ = BufferView::new(&buffer, format::R8Sscaled).unwrap();
    }

    /*#[test]
    fn wrong_usage() {
        let (device, queue) = gfx_dev_and_queue!();

        let buffer = Buffer::<[i8], _>::array(&device, 128, &Usage::none(), DeviceLocal,
                                              &queue).unwrap();

        match BufferView::new(&buffer) {
            Err(BufferViewCreationError::WrongBufferUsage) => (),
            _ => panic!()
        }
    }*/
}
