use std::iter::Empty;
use std::marker::PhantomData;
use std::mem;
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;

use buffer::traits::Buffer;
use buffer::traits::TypedBuffer;
use buffer::unsafe_buffer::UnsafeBuffer;
use buffer::unsafe_buffer::Usage;
use command_buffer::Submission;
use device::Device;
use memory::DeviceMemory;
use memory::MappedDeviceMemory;
use sync::Sharing;

use OomError;

pub struct StagingBuffer<T: ?Sized> {
    // Inner content.
    inner: UnsafeBuffer,

    memory: MappedDeviceMemory,

    // Queue family that has exclusive ownership of this buffer.
    owner_queue_family: Mutex<Option<u32>>,     // TODO: could be atomic

    latest_submission: Mutex<Option<Arc<Submission>>>,

    marker: PhantomData<*const T>,
}

impl<T: ?Sized> StagingBuffer<T> {
    pub fn with_data(device: &Arc<Device>, data: &T) -> Result<Arc<StagingBuffer<T>>, OomError> {
        let usage = Usage {
            transfer_source: true,
            .. Usage::none()
        };

        let (buffer, mem_reqs) = unsafe {
            try!(UnsafeBuffer::new(device, mem::size_of_val(data), &usage,
                                   Sharing::Exclusive::<Empty<u32>>))
        };

        let mem_ty = device.physical_device().memory_types()
                           .filter(|t| (mem_reqs.memory_type_bits & (1 << t.id())) != 0)
                           .filter(|t| t.is_host_visible())
                           .next().unwrap();    // Vk specs guarantee that this can't fail

        // note: alignment doesn't need to be checked because allocating memory is guaranteed to
        //       fulfill any alignment requirement

        let mem = try!(DeviceMemory::alloc_and_map(device, &mem_ty, mem_reqs.size));

        unsafe { try!(buffer.bind_memory(mem.memory(), 0 .. mem_reqs.size)) };

        // FIXME: write data in memory

        Ok(Arc::new(StagingBuffer {
            inner: buffer,
            memory: mem,
            owner_queue_family: Mutex::new(None),
            latest_submission: Mutex::new(None),
            marker: PhantomData,
        }))
    }
}

unsafe impl<T: ?Sized> Buffer for StagingBuffer<T> {
    #[inline]
    fn inner_buffer(&self) -> &UnsafeBuffer {
        &self.inner
    }

    fn needs_fence(&self, _: bool, _: Range<usize>) -> Option<bool> {
        Some(true)
    }

    unsafe fn gpu_access(&self, _: bool, _: Range<usize>, submission: &Arc<Submission>)
                         -> Vec<Arc<Submission>>
    {
        {
            let mut owner_queue_family = self.owner_queue_family.lock().unwrap();
            match (&mut *owner_queue_family, submission.queue().family().id()) {
                (mine @ &mut None, id) => *mine = Some(id),
                (&mut Some(my_id), q_id) if my_id == q_id => (),
                (&mut Some(my_id), q_id) => panic!(),       // TODO: use a slow path instead?
            }
        }

        let dependency = {
            let mut latest_submission = self.latest_submission.lock().unwrap();
            mem::replace(&mut *latest_submission, Some(submission.clone()))
        };

        if let Some(dependency) = dependency {
            vec![dependency]
        } else {
            vec![]
        }
    }
}

unsafe impl<T: ?Sized + 'static> TypedBuffer for StagingBuffer<T> {
    type Content = T;
}
