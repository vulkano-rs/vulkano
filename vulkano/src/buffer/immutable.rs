use std::marker::PhantomData;
use std::mem;
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use smallvec::SmallVec;

use buffer::traits::Buffer;
use buffer::unsafe_buffer::UnsafeBuffer;
use buffer::unsafe_buffer::Usage;
use command_buffer::Submission;
use device::Device;
use instance::QueueFamily;
use memory::DeviceMemory;
use memory::MappedDeviceMemory;
use sync::Sharing;

use OomError;

pub struct ImmutableBuffer<T: ?Sized> {
    // Inner content.
    inner: UnsafeBuffer,

    memory: MappedDeviceMemory,

    // Queue families allowed to access this buffer.
    queue_families: SmallVec<[u32; 4]>,

    latest_write_submission: Mutex<Option<Arc<Submission>>>,

    started_reading: AtomicBool,

    marker: PhantomData<*const T>,
}

impl<T> ImmutableBuffer<T> {
    #[inline]
    pub fn new<'a, I>(device: &Arc<Device>, usage: &Usage, queue_families: I)
                      -> Result<Arc<ImmutableBuffer<T>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        unsafe {
            ImmutableBuffer::raw(device, mem::size_of::<T>(), usage, queue_families)
        }
    }
}

impl<T> ImmutableBuffer<[T]> {
    #[inline]
    pub fn array<'a, I>(device: &Arc<Device>, len: usize, usage: &Usage, queue_families: I)
                      -> Result<Arc<ImmutableBuffer<T>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        unsafe {
            ImmutableBuffer::raw(device, len * mem::size_of::<T>(), usage, queue_families)
        }
    }
}

impl<T: ?Sized> ImmutableBuffer<T> {
    pub unsafe fn raw<'a, I>(device: &Arc<Device>, size: usize, usage: &Usage, queue_families: I)
                             -> Result<Arc<ImmutableBuffer<T>>, OomError>
        where I: IntoIterator<Item = QueueFamily<'a>>
    {
        let queue_families = queue_families.into_iter().map(|f| f.id())
                                           .collect::<SmallVec<[u32; 4]>>();

        let (buffer, mem_reqs) = {
            let sharing = if queue_families.len() >= 2 {
                Sharing::Concurrent(queue_families.iter().cloned())
            } else {
                Sharing::Exclusive
            };

            try!(UnsafeBuffer::new(device, size, &usage, sharing))
        };

        let mem_ty = {
            let device_local = device.physical_device().memory_types()
                                     .filter(|t| (mem_reqs.memory_type_bits & (1 << t.id())) != 0)
                                     .filter(|t| t.is_device_local());
            let any = device.physical_device().memory_types()
                            .filter(|t| (mem_reqs.memory_type_bits & (1 << t.id())) != 0);
            device_local.chain(any).next().unwrap()
        };

        // note: alignment doesn't need to be checked because allocating memory is guaranteed to
        //       fulfill any alignment requirement

        let mem = try!(DeviceMemory::alloc_and_map(device, &mem_ty, mem_reqs.size));

        try!(buffer.bind_memory(mem.memory(), 0 .. mem_reqs.size));

        Ok(Arc::new(ImmutableBuffer {
            inner: buffer,
            memory: mem,
            queue_families: queue_families,
            latest_write_submission: Mutex::new(None),
            started_reading: AtomicBool::new(false),
            marker: PhantomData,
        }))
    }
}

unsafe impl<T: ?Sized> Buffer for ImmutableBuffer<T> {
    #[inline]
    fn inner_buffer(&self) -> &UnsafeBuffer {
        &self.inner
    }

    fn needs_fence(&self, _: bool, _: Range<usize>) -> Option<bool> {
        Some(true)
    }

    unsafe fn gpu_access(&self, write: bool, _: Range<usize>, submission: &Arc<Submission>)
                         -> Vec<Arc<Submission>>
    {
        let queue_id = submission.queue().family().id();
        if self.queue_families.iter().find(|&&id| id == queue_id).is_none() {
            panic!()
        }

        if write {
            assert!(self.started_reading.load(Ordering::AcqRel) == false);
        }

        let dependency = {
            let mut latest_submission = self.latest_write_submission.lock().unwrap();

            if write {
                mem::replace(&mut *latest_submission, Some(submission.clone()))
            } else {
                latest_submission.clone()
            }
        };

        if write {
            assert!(self.started_reading.load(Ordering::AcqRel) == false);
        } else {        
            self.started_reading.store(true, Ordering::AcqRel);
        }

        if let Some(dependency) = dependency {
            vec![dependency]
        } else {
            vec![]
        }
    }
}
