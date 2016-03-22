use std::iter::Empty;
use std::marker::PhantomData;
use std::mem;
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;

use buffer::traits::Buffer;
use buffer::unsafe_buffer::UnsafeBuffer;
use buffer::unsafe_buffer::Usage;
use command_buffer::Submission;
use device::Device;
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

        /*try!(DeviceMemory::alloc_and_map(device, , ))
    pub fn alloc_and_map(device: &Arc<Device>, memory_type: &MemoryType, size: usize)
                         -> Result<MappedDeviceMemory, OomError>*/

        let (buffer, mem_reqs) = unsafe {
            try!(UnsafeBuffer::new(device, mem::size_of_val(data), &usage,
                                   Sharing::Exclusive::<Empty<u32>>))
        };

        Ok(Arc::new(StagingBuffer {
            inner: buffer,
            memory: unimplemented!(),
            owner_queue_family: Mutex::new(None),
            latest_submission: Mutex::new(None),
            marker: PhantomData,
        }))
    }
}

unsafe impl<T: ?Sized> Buffer for StagingBuffer<T> {
    #[inline]
    fn inner(&self) -> &UnsafeBuffer {
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
