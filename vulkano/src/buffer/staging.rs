// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::iter::Empty;
use std::marker::PhantomData;
use std::mem;
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Weak;

use buffer::sys::UnsafeBuffer;
use buffer::sys::Usage;
use buffer::traits::AccessRange;
use buffer::traits::Buffer;
use buffer::traits::GpuAccessResult;
use buffer::traits::TypedBuffer;
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

    latest_submission: Mutex<Option<Weak<Submission>>>,     // TODO: can use `Weak::new()` once it's stabilized

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
        unsafe { try!(buffer.bind_memory(mem.memory(), 0)) };

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

    #[inline]
    fn blocks(&self, _: Range<usize>) -> Vec<usize> {
        vec![0]
    }

    #[inline]
    fn block_memory_range(&self, _: usize) -> Range<usize> {
        0 .. self.size()
    }

    fn needs_fence(&self, _: bool, _: Range<usize>) -> Option<bool> {
        Some(true)
    }

    #[inline]
    fn host_accesses(&self, _: usize) -> bool {
        true
    }

    unsafe fn gpu_access(&self, _: &mut Iterator<Item = AccessRange>, submission: &Arc<Submission>)
                         -> GpuAccessResult
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
            mem::replace(&mut *latest_submission, Some(Arc::downgrade(submission)))
        };
        let dependency = dependency.and_then(|d| d.upgrade());

        GpuAccessResult {
            dependencies: if let Some(dependency) = dependency {
                vec![dependency]
            } else {
                vec![]
            },
            additional_wait_semaphore: None,
            additional_signal_semaphore: None,
        }
    }
}

unsafe impl<T: ?Sized + 'static> TypedBuffer for StagingBuffer<T> {
    type Content = T;
}
