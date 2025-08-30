//! Resource state manipulation facilities.

use super::{BufferAccess, ImageAccess, Resources};
use crate::{collector, Id};
use parking_lot::{Mutex, RwLock};
use smallvec::SmallVec;
use std::{
    num::NonZero,
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc, Weak,
    },
    time::Duration,
};
use vulkano::{
    buffer::Buffer,
    device::DeviceOwned,
    image::Image,
    swapchain::Swapchain,
    sync::{
        fence::{Fence, FenceCreateFlags, FenceCreateInfo},
        semaphore::Semaphore,
    },
    VulkanError,
};

#[derive(Debug)]
pub struct BufferState {
    buffer: Arc<Buffer>,
    last_access: Mutex<BufferAccess>,
}

impl BufferState {
    pub(super) fn new(buffer: Arc<Buffer>) -> Self {
        BufferState {
            buffer,
            last_access: Mutex::new(BufferAccess::NONE),
        }
    }

    /// Returns the buffer.
    #[inline]
    #[must_use]
    pub fn buffer(&self) -> &Arc<Buffer> {
        &self.buffer
    }

    /// Returns the last access that was performed on the buffer.
    #[inline]
    pub fn access(&self) -> BufferAccess {
        *self.last_access.lock()
    }

    /// Sets the last access that was performed on the buffer.
    ///
    /// # Safety
    ///
    /// - `access` must constitute the correct access that was last performed on the buffer.
    #[inline]
    pub unsafe fn set_access(&self, access: BufferAccess) {
        *self.last_access.lock() = access;
    }
}

#[derive(Debug)]
pub struct ImageState {
    image: Arc<Image>,
    last_access: Mutex<ImageAccess>,
}

impl ImageState {
    pub(super) fn new(image: Arc<Image>) -> Self {
        ImageState {
            image,
            last_access: Mutex::new(ImageAccess::NONE),
        }
    }

    /// Returns the image.
    #[inline]
    #[must_use]
    pub fn image(&self) -> &Arc<Image> {
        &self.image
    }

    /// Returns the last access that was performed on the image.
    #[inline]
    pub fn access(&self) -> ImageAccess {
        *self.last_access.lock()
    }

    /// Sets the last access that was performed on the image.
    ///
    /// # Safety
    ///
    /// - `access` must constitute the correct access that was last performed on the image.
    #[inline]
    pub unsafe fn set_access(&self, access: ImageAccess) {
        *self.last_access.lock() = access;
    }
}

// FIXME: imported/exported semaphores
#[derive(Debug)]
pub struct SwapchainState {
    swapchain: Arc<Swapchain>,
    images: SmallVec<[Arc<Image>; 3]>,
    pub(crate) semaphores: Arc<[SwapchainSemaphoreState]>,
    flight_id: Id<Flight>,
    pub(crate) current_image_index: AtomicU32,
    last_access: Mutex<ImageAccess>,
}

#[derive(Debug)]
pub(crate) struct SwapchainSemaphoreState {
    pub(crate) image_available_semaphore: Semaphore,
    pub(crate) pre_present_complete_semaphore: Semaphore,
    pub(crate) tasks_complete_semaphore: Semaphore,
}

impl SwapchainState {
    pub(super) fn new(
        swapchain: Arc<Swapchain>,
        images: Vec<Arc<Image>>,
        resources: &Resources,
        flight_id: Id<Flight>,
    ) -> Result<Self, VulkanError> {
        let frames_in_flight = resources
            .storage
            .flight_protected(flight_id, &resources.pin())
            .unwrap()
            .frame_count();

        let semaphores = (0..frames_in_flight)
            .map(|_| {
                Ok(SwapchainSemaphoreState {
                    // SAFETY: The parameters are valid.
                    image_available_semaphore: unsafe {
                        Semaphore::new_unchecked(resources.device(), &Default::default())
                    }?,
                    // SAFETY: The parameters are valid.
                    pre_present_complete_semaphore: unsafe {
                        Semaphore::new_unchecked(resources.device(), &Default::default())
                    }?,
                    // SAFETY: The parameters are valid.
                    tasks_complete_semaphore: unsafe {
                        Semaphore::new_unchecked(resources.device(), &Default::default())
                    }?,
                })
            })
            .collect::<Result<_, VulkanError>>()?;

        Ok(SwapchainState {
            swapchain,
            images: images.into(),
            semaphores,
            flight_id,
            current_image_index: AtomicU32::new(u32::MAX),
            last_access: Mutex::new(ImageAccess::NONE),
        })
    }

    pub(super) unsafe fn with_old_state(
        swapchain: Arc<Swapchain>,
        images: Vec<Arc<Image>>,
        old_state: &Self,
    ) -> Self {
        SwapchainState {
            swapchain,
            images: images.into(),
            semaphores: old_state.semaphores.clone(),
            flight_id: old_state.flight_id,
            current_image_index: AtomicU32::new(u32::MAX),
            last_access: Mutex::new(ImageAccess::NONE),
        }
    }

    /// Returns the swapchain.
    #[inline]
    #[must_use]
    pub fn swapchain(&self) -> &Arc<Swapchain> {
        &self.swapchain
    }

    /// Returns the images comprising the swapchain.
    #[inline]
    #[must_use]
    pub fn images(&self) -> &[Arc<Image>] {
        &self.images
    }

    /// Returns the ID of the [flight] which owns this swapchain.
    #[inline]
    #[must_use]
    pub fn flight_id(&self) -> Id<Flight> {
        self.flight_id
    }

    /// Returns the image index that's acquired in the current frame, or returns `None` if no image
    /// index is acquired.
    #[inline]
    #[must_use]
    pub fn current_image_index(&self) -> Option<u32> {
        let index = self.current_image_index.load(Ordering::Relaxed);

        if index == u32::MAX {
            None
        } else {
            Some(index)
        }
    }

    pub(crate) fn current_image(&self) -> &Arc<Image> {
        &self.images[self.current_image_index.load(Ordering::Relaxed) as usize]
    }

    pub(crate) fn access(&self) -> ImageAccess {
        *self.last_access.lock()
    }

    pub(crate) unsafe fn set_access(&self, access: ImageAccess) {
        *self.last_access.lock() = access;
    }
}

// FIXME: imported/exported fences
#[derive(Debug)]
pub struct Flight {
    // HACK: We need this in order to collect garbage.
    resources: Weak<Resources>,
    pub(super) frame_count: NonZero<u32>,
    biased_started_frame: AtomicU64,
    current_frame: AtomicU64,
    biased_complete_frame: AtomicU64,
    pub(super) fences: SmallVec<[RwLock<Fence>; 3]>,
    pub(super) garbage_queue: collector::LocalQueue,
    pub(crate) state: Mutex<()>,
}

impl Flight {
    pub(super) fn new(
        resources: &Arc<Resources>,
        frame_count: NonZero<u32>,
    ) -> Result<Self, VulkanError> {
        let fences = (0..frame_count.get())
            .map(|_| {
                // SAFETY: The parameters are valid.
                unsafe {
                    Fence::new_unchecked(
                        resources.device(),
                        &FenceCreateInfo {
                            flags: FenceCreateFlags::SIGNALED,
                            ..Default::default()
                        },
                    )
                }
                .map(RwLock::new)
            })
            .collect::<Result<_, VulkanError>>()?;

        Ok(Flight {
            resources: Arc::downgrade(resources),
            frame_count,
            biased_started_frame: AtomicU64::new(u64::from(frame_count.get())),
            current_frame: AtomicU64::new(0),
            biased_complete_frame: AtomicU64::new(u64::from(frame_count.get())),
            fences,
            garbage_queue: resources.garbage_queue().register_local(),
            state: Mutex::new(()),
        })
    }

    /// Returns the number of [frames] in this [flight].
    #[inline]
    #[must_use]
    pub fn frame_count(&self) -> u32 {
        self.frame_count.get()
    }

    /// Returns the latest started frame stored at a bias of `frame_count + 1`. That means that if
    /// this value reaches `n + frame_count + 1` then frame `n` has started execution. This starts
    /// out at `frame_count` because no frame has started execution yet when a flight is created.
    pub(crate) fn biased_started_frame(&self) -> u64 {
        self.biased_started_frame.load(Ordering::Relaxed)
    }

    /// Returns the current frame counter value. This always starts out at `0` and increases by `1`
    /// after every successful [task graph execution].
    ///
    /// [task graph execution]: crate::graph::ExecutableTaskGraph::execute
    #[inline]
    #[must_use]
    pub fn current_frame(&self) -> u64 {
        self.current_frame.load(Ordering::Relaxed)
    }

    /// Returns the latest complete frame stored at a bias of `frame_count + 1`. That means that if
    /// this value reaches `n + frame_count + 1` then frame `n` has been waited on. This starts out
    /// at `frame_count` because there is nothing to wait for in the first ever `frame_count`
    /// frames.
    fn biased_complete_frame(&self) -> u64 {
        self.biased_complete_frame.load(Ordering::Relaxed)
    }

    /// Returns the index of the current [frame] in [flight].
    #[inline]
    #[must_use]
    pub fn current_frame_index(&self) -> u32 {
        (self.current_frame() % NonZero::<u64>::from(self.frame_count)) as u32
    }

    pub(crate) fn current_fence(&self) -> &RwLock<Fence> {
        &self.fences[self.current_frame_index() as usize]
    }

    pub(crate) fn garbage_queue(&self) -> &collector::LocalQueue {
        &self.garbage_queue
    }

    /// Waits for the oldest [frame] in [flight] to finish.
    ///
    /// This is equivalent to [`Fence::wait`] on the fence corresponding to the current frame
    /// index, but unlike that method, this method additionally collects outstanding garbage.
    pub fn wait(&self, timeout: Option<Duration>) -> Result<(), VulkanError> {
        self.wait_for_biased_frame(self.current_frame() + 1, timeout)
    }

    /// Waits for the given [frame] to finish.
    ///
    /// `frame` must be a past (not including current) frame previously obtained using
    /// [`current_frame`] on `self`.
    ///
    /// This is equivalent to [`Fence::wait`] on the fence corresponding to the frame index `frame
    /// % frame_count`, but unlike that method, this method additionally collects outstanding
    /// garbage.
    ///
    /// # Panics
    ///
    /// - Panics if `frame` is greater than the current frame.
    ///
    /// [`current_frame`]: Self::current_frame
    pub fn wait_for_frame(&self, frame: u64, timeout: Option<Duration>) -> Result<(), VulkanError> {
        assert!(frame < self.current_frame());

        let biased_frame = frame + u64::from(self.frame_count()) + 1;

        self.wait_for_biased_frame(biased_frame, timeout)
    }

    /// Waits for all [frames] in [flight] to finish.
    ///
    /// This is equivalent to [`Fence::wait`] on the fence corresponding to the previous frame
    /// index, but unlike that method, this method additionally collects outstanding garbage.
    pub fn wait_idle(&self) -> Result<(), VulkanError> {
        let biased_frame = self.current_frame() + u64::from(self.frame_count());

        self.wait_for_biased_frame(biased_frame, None)
    }

    fn wait_for_biased_frame(
        &self,
        biased_frame: u64,
        timeout: Option<Duration>,
    ) -> Result<(), VulkanError> {
        if self.is_biased_frame_complete(biased_frame) {
            return Ok(());
        }

        // The `frame_count` bias cancels out under the modulus; however, the `1` bias doesn't, so
        // we subtract it to get the same index as would be computed using the unbiased `frame`.
        self.fences[((biased_frame - 1) % NonZero::<u64>::from(self.frame_count)) as usize]
            .read()
            .wait(timeout)?;

        // SAFETY: We waited for the frame.
        if unsafe { self.update_biased_complete_frame(biased_frame) }.is_ok() {
            let resources = self.resources.upgrade().unwrap();
            let guard = &resources.pin();

            unsafe { self.garbage_queue.collect(&resources, self, guard) };
            unsafe { resources.garbage_queue().collect(&resources, guard) };
        }

        Ok(())
    }

    pub(super) unsafe fn update_biased_complete_frame(
        &self,
        biased_frame: u64,
    ) -> Result<u64, u64> {
        self.biased_complete_frame.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |biased_complete_frame| (biased_complete_frame < biased_frame).then_some(biased_frame),
        )
    }

    pub(crate) fn is_oldest_frame_complete(&self) -> bool {
        self.biased_complete_frame() > self.current_frame()
    }

    pub(crate) fn is_biased_frame_complete(&self, biased_frame: u64) -> bool {
        self.biased_complete_frame() >= biased_frame
    }

    pub(crate) unsafe fn start_next_frame(&self) {
        self.biased_started_frame.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) unsafe fn undo_start_next_frame(&self) {
        self.biased_started_frame.fetch_sub(1, Ordering::Relaxed);
    }

    pub(crate) unsafe fn next_frame(&self) {
        self.current_frame.fetch_add(1, Ordering::Relaxed);
    }
}
