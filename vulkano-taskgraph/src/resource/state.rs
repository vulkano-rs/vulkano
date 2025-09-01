//! Resource state manipulation facilities.

use super::{BufferAccess, ImageAccess, Resources};
use crate::{
    collector::{self, DeferredBatch},
    Id,
};
use ash::vk;
use parking_lot::{Mutex, MutexGuard, RwLock};
use smallvec::SmallVec;
use std::{
    cell::UnsafeCell,
    collections::VecDeque,
    error::Error,
    fmt,
    num::NonZero,
    sync::{
        atomic::{
            AtomicU32, AtomicU64,
            Ordering::{Acquire, Relaxed, Release},
        },
        Arc, Weak,
    },
    time::Duration,
};
use vulkano::{
    buffer::Buffer,
    device::{Device, DeviceOwned},
    image::Image,
    swapchain::{AcquireNextImageInfo, AcquiredImage, Swapchain},
    sync::{
        fence::{Fence, FenceCreateFlags, FenceCreateInfo},
        semaphore::{Semaphore, SemaphoreCreateInfo},
    },
    Validated, VulkanError, VulkanObject,
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

#[derive(Debug)]
pub struct SwapchainState {
    swapchain: Arc<Swapchain>,
    images: SmallVec<[Arc<Image>; 3]>,
    generation: u32,
    sync_state: Arc<SwapchainLock>,
    current_image_index: AtomicU32,
    last_access: Mutex<ImageAccess>,
}

/// Swapchains have Super Duper Uber Special<sup>TM</sup> synchronization needs. Namely, a
/// swapchain can't be used again after it has been recreated. This lock prevents that by assigning
/// each swapchain in the chain of swapchain (re)creations a *generation*, which is just a
/// monotonically increasing integer. When a swapchain is created, it gets a generation of 0, and
/// every subsequent recreation gets a generation one higher than the previous. The lock stores the
/// current generation and only successfully locks when the generation of the swapchain matches the
/// current generation. The lock can be locked for use in a task graph execution or for use in a
/// recreation. This ensures that a swapchain can't be used in more than one task graph execution
/// at a time since that would be UB, and that it can't be used in an execution and recreation at
/// the same time. The latter, coupled with the fact that a swapchain can't be used after being
/// recreated, allows us to remove a swapchain from which an image has never been acquired
/// immediately.
///
/// The swapchain can also be removed, which is similar to being locked except that this state is
/// final. Like with locking, removal is mutually exclusive and ensures that the swapchain can't be
/// used in a task graph execution when removed, which allows us to schedule the destruction of the
/// swapchain, any of its old swapchains, and sync objects after a device wait for idle.
struct SwapchainLock {
    state: AtomicU32,
    data: UnsafeCell<SwapchainSyncState>,
}

#[derive(Debug)]
struct SwapchainSyncState {
    device: Arc<Device>,
    // HACK: We need this in order to collect old swapchains.
    resources: Weak<Resources>,
    current_acquire_semaphore: Option<Semaphore>,
    current_acquire_fence: Option<Fence>,
    current_pre_present_semaphore: Option<Semaphore>,
    current_present_semaphore: Option<Semaphore>,
    present_queue: VecDeque<SwapchainPresentOperation>,
    garbage_queue: VecDeque<SwapchainGarbage>,
    semaphore_pool: Vec<Semaphore>,
    fence_pool: Vec<Fence>,
}

#[derive(Debug)]
struct SwapchainPresentOperation {
    generation: u32,
    acquire_semaphore: Option<Semaphore>,
    image_index: u32,
    pre_present_semaphore: Option<Semaphore>,
    present_semaphore: Option<Semaphore>,
    cleanup_fence: Option<Fence>,
}

#[derive(Debug, Default)]
struct SwapchainGarbage {
    generation: u32,
    swapchains: Vec<Id<Swapchain>>,
    semaphores: Vec<Semaphore>,
}

impl SwapchainState {
    pub(super) fn new(
        swapchain: Arc<Swapchain>,
        images: Vec<Arc<Image>>,
        resources: &Arc<Resources>,
    ) -> Self {
        SwapchainState {
            swapchain,
            images: images.into(),
            generation: 0,
            sync_state: Arc::new(SwapchainLock::new(SwapchainSyncState {
                device: resources.device().clone(),
                resources: Arc::downgrade(resources),
                current_acquire_semaphore: None,
                current_acquire_fence: None,
                current_pre_present_semaphore: None,
                current_present_semaphore: None,
                present_queue: VecDeque::new(),
                garbage_queue: VecDeque::new(),
                semaphore_pool: Vec::new(),
                fence_pool: Vec::new(),
            })),
            current_image_index: AtomicU32::new(u32::MAX),
            last_access: Mutex::new(ImageAccess::NONE),
        }
    }

    pub(super) unsafe fn with_old_state(
        swapchain: Arc<Swapchain>,
        images: Vec<Arc<Image>>,
        old_state: &Self,
    ) -> Self {
        SwapchainState {
            swapchain,
            images: images.into(),
            generation: old_state.generation.wrapping_add(1),
            sync_state: old_state.sync_state.clone(),
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

    pub(crate) fn try_execute(&self) -> Result<(), SwapchainLockError> {
        self.sync_state.try_execute(self.generation)
    }

    pub(super) fn try_recreate(&self) -> Result<SwapchainLockGuard<'_>, SwapchainLockError> {
        self.sync_state.try_recreate(self.generation)?;

        Ok(SwapchainLockGuard { state: self })
    }

    pub(crate) unsafe fn unlock(&self) {
        // SAFETY: The caller must ensure that the swapchain has been locked.
        unsafe { self.sync_state.unlock() };
    }

    pub(crate) unsafe fn acquire_next_image(&self) -> Result<(), VulkanError> {
        // SAFETY: The caller must ensure that the swapchain has been locked for execution, which
        // ensures correct synchronization. We also don't create additional references.
        let sync_state = unsafe { self.sync_state.get_mut_unchecked() };

        let semaphore = sync_state.allocate_semaphore()?;
        let fence = sync_state.allocate_fence()?;

        let res = unsafe {
            self.swapchain().acquire_next_image(&AcquireNextImageInfo {
                semaphore: Some(&semaphore),
                fence: Some(&fence),
                ..Default::default()
            })
        };

        // This should not panic because our swapchain lock prevents using a swapchain after it has
        // been recreated. However, this will panic if the user circumvents that by calling
        // `Swapchain::recreate` themself. The user is not supposed to do that.
        match res.map_err(Validated::unwrap) {
            Ok(AcquiredImage { image_index, .. }) => {
                assert!(sync_state.current_acquire_semaphore.is_none());
                assert!(sync_state.current_acquire_fence.is_none());
                sync_state.current_acquire_semaphore = Some(semaphore);
                sync_state.current_acquire_fence = Some(fence);

                self.current_image_index.store(image_index, Relaxed);

                Ok(())
            }
            Err(err) => {
                sync_state.deallocate_semaphore(semaphore);
                sync_state.deallocate_fence(fence);

                Err(err)
            }
        }
    }

    pub(crate) unsafe fn current_acquire_semaphore(&self) -> Option<vk::Semaphore> {
        // SAFETY: The caller must ensure that the swapchain has been locked for execution, which
        // ensures correct synchronization. We also don't create additional references.
        let sync_state = unsafe { self.sync_state.get_mut_unchecked() };

        let semaphore = sync_state.current_acquire_semaphore.as_ref()?;

        Some(semaphore.handle())
    }

    pub(crate) unsafe fn init_pre_present_semaphore(&self) -> Result<vk::Semaphore, VulkanError> {
        // SAFETY: The caller must ensure that the swapchain has been locked for execution, which
        // ensures correct synchronization. We also don't create additional references.
        let sync_state = unsafe { self.sync_state.get_mut_unchecked() };

        let semaphore = sync_state.allocate_semaphore()?;
        let handle = semaphore.handle();

        assert!(sync_state.current_pre_present_semaphore.is_none());
        sync_state.current_pre_present_semaphore = Some(semaphore);

        Ok(handle)
    }

    pub(crate) unsafe fn current_pre_present_semaphore(&self) -> Option<vk::Semaphore> {
        // SAFETY: The caller must ensure that the swapchain has been locked for execution, which
        // ensures correct synchronization. We also don't create additional references.
        let sync_state = unsafe { self.sync_state.get_mut_unchecked() };

        let semaphore = sync_state.current_pre_present_semaphore.as_ref()?;

        Some(semaphore.handle())
    }

    pub(crate) unsafe fn init_present_semaphore(&self) -> Result<vk::Semaphore, VulkanError> {
        // SAFETY: The caller must ensure that the swapchain has been locked for execution, which
        // ensures correct synchronization. We also don't create additional references.
        let sync_state = unsafe { self.sync_state.get_mut_unchecked() };

        let semaphore = sync_state.allocate_semaphore()?;
        let handle = semaphore.handle();

        assert!(sync_state.current_present_semaphore.is_none());
        sync_state.current_present_semaphore = Some(semaphore);

        Ok(handle)
    }

    pub(crate) unsafe fn current_present_semaphore(&self) -> Option<vk::Semaphore> {
        // SAFETY: The caller must ensure that the swapchain has been locked for execution, which
        // ensures correct synchronization. We also don't create additional references.
        let sync_state = unsafe { self.sync_state.get_mut_unchecked() };

        let semaphore = sync_state.current_present_semaphore.as_ref()?;

        Some(semaphore.handle())
    }

    pub(crate) unsafe fn handle_presentation(
        &self,
        result: vk::Result,
        deferred_batch: &mut DeferredBatch<'_>,
    ) {
        // SAFETY: The caller must ensure that the swapchain has been locked for execution, which
        // ensures correct synchronization. We also don't create additional references.
        let sync_state = unsafe { self.sync_state.get_mut_unchecked() };

        let generation = self.generation;
        let image_index = self.current_image_index().unwrap();

        // The acquire fence isn't set if an image index has already been acquired but not presented
        // in the previous task graph execution.
        if let Some(acquire_fence) = sync_state.current_acquire_fence.take() {
            sync_state.associate_acquire_fence_with_present_operation(
                generation,
                image_index,
                acquire_fence,
            );
        }

        // The acquire semaphore isn't set if an image index has already been acquired but not
        // presented in the previous task graph execution and the semaphore has already been waited
        // on.
        let acquire_semaphore = sync_state.current_acquire_semaphore.take();

        // The pre-present semaphore isn't set if the task graph presents on the same queue as it
        // renders.
        let pre_present_semaphore = sync_state.current_pre_present_semaphore.take();

        let present_semaphore = sync_state.current_present_semaphore.take().unwrap();

        // In case of these error codes, the semaphore wait operation is not executed.
        if matches!(
            result,
            vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        ) {
            // We could reuse the acquire and pre-present semaphores but that's way too much added
            // complexity for no reason.
            if let Some(semaphore) = acquire_semaphore {
                deferred_batch.destroy_object(semaphore);
            }

            if let Some(semaphore) = pre_present_semaphore {
                deferred_batch.destroy_object(semaphore);
            }

            // We can't reuse the semaphore since it would be left in a signaled state. But the fact
            // that the wait operation isn't executed is also a blessing in that we can simply
            // destroy the semaphore normally.
            deferred_batch.destroy_object(present_semaphore);
        } else {
            self.current_image_index.store(u32::MAX, Relaxed);

            sync_state
                .present_queue
                .push_back(SwapchainPresentOperation {
                    generation,
                    acquire_semaphore,
                    image_index,
                    pre_present_semaphore,
                    present_semaphore: Some(present_semaphore),
                    cleanup_fence: None,
                });
        }
    }

    pub(crate) unsafe fn collect(&self) -> Result<(), VulkanError> {
        // SAFETY: The caller must ensure that the swapchain has been locked for execution, which
        // ensures correct synchronization. We also don't create additional references.
        let sync_state = unsafe { self.sync_state.get_mut_unchecked() };

        let mut present_index = 0;

        while present_index < sync_state.present_queue.len() {
            let present_operation = &sync_state.present_queue[present_index];

            let Some(fence) = &present_operation.cleanup_fence else {
                // The present operation doesn't have an associated fence yet. When the same image
                // index as this presentation operation's is acquired next time, that acquire's
                // fence will be associated with this present operation. However, there is no
                // guarantee that an image index is ever reacquired, so it's possible that a present
                // operation never gets a fence associated with it. That's why we step over such
                // present operations in case there are present operations with associated fences
                // following.
                present_index += 1;
                continue;
            };

            if !fence.is_signaled()? {
                // We can't be certain that the present operation is done yet. Unlike when it comes
                // to acquires, we do know the order of presents because we're in control of them.
                // The order the Presentation Engine gets the presents is the same as that of our
                // `present_queue`. Therefore, we know that if one present operation in our queue
                // isn't done, the following ones aren't done either.
                break;
            }

            let present_operation = sync_state.present_queue.remove(present_index).unwrap();

            let cleanup_fence = present_operation.cleanup_fence.unwrap();

            // SAFETY: We checked that the fence is signaled above. The caller must ensure that the
            // swapchain has been locked for execution, which ensures correct synchronization.
            unsafe { cleanup_fence.reset_unchecked() }?;

            sync_state.deallocate_fence(cleanup_fence);

            if let Some(semaphore) = present_operation.present_semaphore {
                sync_state.deallocate_semaphore(semaphore);

                // The pre-present semaphore isn't set if the task graph presents on the same queue
                // as it renders.
                if let Some(semaphore) = present_operation.pre_present_semaphore {
                    sync_state.deallocate_semaphore(semaphore);
                }

                // The acquire semaphore isn't set if an image index has already been acquired but
                // not presented in the task graph execution before the present operation and the
                // semaphore has already been waited on.
                if let Some(semaphore) = present_operation.acquire_semaphore {
                    sync_state.deallocate_semaphore(semaphore);
                }
            } else {
                assert!(present_operation.pre_present_semaphore.is_none());

                // The acquire semaphore is set if this dummy present operation is used to clean up
                // the acquire semaphore and fence when a swapchain is recreated while still having
                // an acquired image. In this case, the acquire semaphore is left in a signaled
                // state, so we can't reuse it.
                let _ = present_operation.acquire_semaphore;

                // If there is no present semaphore then this is a dummy present operation used only
                // to clean up the fence and potentially acquire semaphore, so we can't allow it to
                // clean up anything else.
                continue;
            }

            // Clean up old swapchains and their semaphores if there are any associated with the
            // swapchain whose fence we successfully waited on.
            if sync_state
                .garbage_queue
                .front()
                .is_some_and(|garbage| garbage.generation == present_operation.generation)
            {
                let garbage = sync_state.garbage_queue.pop_front().unwrap();
                let resources = sync_state.resources.upgrade().unwrap();

                for swapchain_id in garbage.swapchains {
                    unsafe { resources.remove_invalidated_swapchain_unchecked(swapchain_id) };
                }

                for semaphore in garbage.semaphores {
                    sync_state.deallocate_semaphore(semaphore);
                }
            }
        }

        Ok(())
    }

    pub(crate) unsafe fn handle_execution_failure(&self, sync_stage: SwapchainSyncStage) {
        // SAFETY: The caller must ensure that the swapchain has been locked for execution, which
        // ensures correct synchronization. We also don't create additional references.
        let sync_state = unsafe { self.sync_state.get_mut_unchecked() };

        let generation = self.generation;
        let image_index = self.current_image_index().unwrap();
        let acquire_fence = sync_state.current_acquire_fence.take().unwrap();

        sync_state.associate_acquire_fence_with_present_operation(
            generation,
            image_index,
            acquire_fence,
        );

        // If the acquire semaphore wasn't waited on, we have to keep it and wait on it in the next
        // task graph execution. Otherwise, we don't need an acquire semaphore at all in the next
        // task graph execution, so we recycle it but keep the current image index. This way, the
        // next task graph execution is going to use the same image index without waiting on an
        // acquire semaphore.
        //
        // If we ended on a signal operation, we can't reuse the semaphore as it would be left in a
        // signaled state, so we destroy it.
        match sync_stage {
            SwapchainSyncStage::SignalAcquire => {}
            SwapchainSyncStage::WaitAcquire => {
                let acquire_semaphore = sync_state.current_acquire_semaphore.take().unwrap();
                sync_state.deallocate_semaphore(acquire_semaphore);
            }
            SwapchainSyncStage::SignalPrePresent => {
                let acquire_semaphore = sync_state.current_acquire_semaphore.take().unwrap();
                sync_state.deallocate_semaphore(acquire_semaphore);

                sync_state.current_pre_present_semaphore.take().unwrap();
            }
            SwapchainSyncStage::WaitPrePresent => {
                let acquire_semaphore = sync_state.current_acquire_semaphore.take().unwrap();
                sync_state.deallocate_semaphore(acquire_semaphore);

                let semaphore = sync_state.current_pre_present_semaphore.take().unwrap();
                sync_state.deallocate_semaphore(semaphore);
            }
            SwapchainSyncStage::SignalPresent => {
                let acquire_semaphore = sync_state.current_acquire_semaphore.take().unwrap();
                sync_state.deallocate_semaphore(acquire_semaphore);

                // The pre-present semaphore isn't set if the task graph presents on the same queue
                // as it renders.
                if let Some(semaphore) = sync_state.current_pre_present_semaphore.take() {
                    sync_state.deallocate_semaphore(semaphore);
                }

                sync_state.current_present_semaphore.take().unwrap();
            }
        }

        // Since we haven't successfully executed the submission(s) with these semaphore operations,
        // these semaphores are still unsignaled and can be recycled.
        if sync_stage < SwapchainSyncStage::SignalPrePresent {
            if let Some(semaphore) = sync_state.current_pre_present_semaphore.take() {
                sync_state.deallocate_semaphore(semaphore);
            }

            if sync_stage < SwapchainSyncStage::SignalPresent {
                if let Some(semaphore) = sync_state.current_present_semaphore.take() {
                    sync_state.deallocate_semaphore(semaphore);
                }
            }
        }
    }

    pub(super) unsafe fn handle_recreation(&self, swapchain_id: Id<Swapchain>) {
        // SAFETY: The caller must ensure that the swapchain has been locked for recreation, which
        // ensures correct synchronization. We also don't create additional references.
        let sync_state = unsafe { self.sync_state.get_mut_unchecked() };

        let generation = self.generation;
        let new_generation = generation.wrapping_add(1);

        if let Some(acquire_semaphore) = sync_state.current_acquire_semaphore.take() {
            // An image is still acquired and the acquire semaphore and fence haven't been waited
            // on. We have to do something with them, so we add a dummy present operation just to
            // clean them up. Normally, the acquire fence would be used to clean up the semaphores
            // of the previous presentation of the image, in which case all of the semaphores would
            // have been waited on; however, in this case, the acquire fence is abused to determine
            // when the semaphore of the same acquire can be destroyed. We can't reuse the acquire
            // semaphore since it would be left in a signaled state.
            let image_index = self.current_image_index().unwrap();
            let acquire_fence = sync_state.current_acquire_fence.take().unwrap();
            sync_state
                .present_queue
                .push_back(SwapchainPresentOperation {
                    generation,
                    acquire_semaphore: Some(acquire_semaphore),
                    image_index,
                    pre_present_semaphore: None,
                    present_semaphore: None,
                    cleanup_fence: Some(acquire_fence),
                });
        }

        let has_present_operations = sync_state
            .present_queue
            .back()
            .is_some_and(|present_operation| present_operation.generation == generation);

        let resources = sync_state.resources.upgrade().unwrap();
        let guard = &resources.pin();

        if has_present_operations {
            // We can't remove the old swapchain until we know that there are no pending present
            // operations, so we only invalidate it for now.
            resources.invalidate_swapchain(swapchain_id, guard).unwrap();

            let has_garbage = sync_state
                .garbage_queue
                .back()
                .is_some_and(|garbage| garbage.generation == generation);
            let has_cleanup_fence = sync_state
                .present_queue
                .iter()
                .rev()
                .take_while(|present_operation| present_operation.generation == generation)
                .filter(|present_operation| present_operation.present_semaphore.is_some())
                .any(|present_operation| present_operation.cleanup_fence.is_some());

            // If the old swapchain doesn't have any fence to check for cleanup, the garbage
            // associated with the old swapchain cannot be cleaned up until the new swapchain's
            // garbage can be cleaned up. We therefore combine the old and new garbage and associate
            // it with the new swapchain. Otherwise, we add new garbage.
            if !has_garbage || has_cleanup_fence {
                sync_state
                    .garbage_queue
                    .push_back(SwapchainGarbage::default());
            }

            let garbage = sync_state.garbage_queue.back_mut().unwrap();
            garbage.generation = new_generation;
            garbage.swapchains.push(swapchain_id);

            let mut present_index = sync_state.present_queue.len();

            // Remove all present operations without a cleanup fence since they would have no way of
            // getting cleaned up and add them to the garbage associated with the new swapchain.
            while present_index > 0 {
                present_index -= 1;

                let present_operation = &sync_state.present_queue[present_index];

                if present_operation.generation != generation {
                    break;
                }

                if present_operation.cleanup_fence.is_none() {
                    let present_operation = sync_state.present_queue.remove(present_index).unwrap();

                    // The acquire semaphore isn't set if an image index has already been acquired
                    // but not presented in the task graph execution before the present operation
                    // and the semaphore has already been waited on.
                    if let Some(semaphore) = present_operation.acquire_semaphore {
                        garbage.semaphores.push(semaphore);
                    }

                    // The pre-present semaphore isn't set if the task graph presents on the same
                    // queue as it renders.
                    if let Some(semaphore) = present_operation.pre_present_semaphore {
                        garbage.semaphores.push(semaphore);
                    }

                    let semaphore = present_operation.present_semaphore.unwrap();
                    garbage.semaphores.push(semaphore);
                }
            }
        } else {
            // Since the swapchain must have been locked for recreation, which ensures that it can't
            // also be used in a task graph execution or used again in the future, and there are no
            // existing present operations, there are no current uses or possible future uses of the
            // swapchain, so we know it is sound to remove the swapchain immediately.
            resources
                .storage
                .swapchains
                .remove(swapchain_id.erase(), guard)
                .unwrap();
        }
    }

    #[track_caller]
    pub(super) unsafe fn remove(&self, swapchain_id: Id<Swapchain>) {
        self.sync_state
            .try_remove(self.generation)
            .expect("failed to remove the swapchain");

        // SAFETY: We removed the swapchain above, which ensures correct synchronization. We also
        // don't create additional references.
        let sync_state = unsafe { self.sync_state.get_mut_unchecked() };

        let mut garbage = super::SwapchainGarbage::default();

        for queued_garbage in sync_state.garbage_queue.drain(..) {
            garbage.swapchains.extend(queued_garbage.swapchains);
            garbage.semaphores.extend(queued_garbage.semaphores);
        }

        let generation = self.generation;

        if let Some(acquire_semaphore) = sync_state.current_acquire_semaphore.take() {
            // An image is still acquired and the acquire semaphore and fence haven't been waited
            // on. We have to do something with them, so we add a dummy present operation just to
            // clean them up.
            let image_index = self.current_image_index().unwrap();
            let acquire_fence = sync_state.current_acquire_fence.take().unwrap();
            sync_state
                .present_queue
                .push_back(SwapchainPresentOperation {
                    generation,
                    acquire_semaphore: Some(acquire_semaphore),
                    image_index,
                    pre_present_semaphore: None,
                    present_semaphore: None,
                    cleanup_fence: Some(acquire_fence),
                });
        }

        let has_present_operations = sync_state
            .present_queue
            .back()
            .is_some_and(|present_operation| present_operation.generation == generation);

        for present_operation in sync_state.present_queue.drain(..) {
            if let Some(semaphore) = present_operation.acquire_semaphore {
                garbage.semaphores.push(semaphore);
            }

            if let Some(semaphore) = present_operation.pre_present_semaphore {
                garbage.semaphores.push(semaphore);
            }

            if let Some(semaphore) = present_operation.present_semaphore {
                garbage.semaphores.push(semaphore);
            }

            if let Some(fence) = present_operation.cleanup_fence {
                garbage.fences.push(fence);
            }
        }

        let resources = sync_state.resources.upgrade().unwrap();
        let guard = &resources.pin();

        if has_present_operations {
            // We can't remove the old swapchain until we know that there are no pending present
            // operations, so we only invalidate it for now.
            resources.invalidate_swapchain(swapchain_id, guard).unwrap();

            garbage.swapchains.push(swapchain_id);
        } else {
            // Since we removed the swapchain above, which ensures that it can't also be used in a
            // task graph execution or used again in the future, and there are no existing present
            // operations, there are no current uses or possible future uses of the swapchain, so we
            // know it is sound to remove the swapchain immediately.
            resources
                .storage
                .swapchains
                .remove(swapchain_id.erase(), guard)
                .unwrap();
        }

        if garbage.swapchains.is_empty() {
            assert!(garbage.semaphores.is_empty());
            assert!(garbage.fences.is_empty());
        } else {
            resources.storage.swapchain_garbage.lock().push(garbage);
        }
    }

    /// Returns the image index that's acquired in the current frame, or returns `None` if no image
    /// index is acquired.
    #[inline]
    #[must_use]
    pub fn current_image_index(&self) -> Option<u32> {
        let index = self.current_image_index.load(Relaxed);

        if index == u32::MAX {
            None
        } else {
            Some(index)
        }
    }

    pub(crate) fn current_image(&self) -> &Arc<Image> {
        &self.images[self.current_image_index.load(Relaxed) as usize]
    }

    pub(crate) fn is_image_acquired(&self) -> bool {
        self.current_image_index.load(Relaxed) != u32::MAX
    }

    pub(crate) fn access(&self) -> ImageAccess {
        *self.last_access.lock()
    }

    pub(crate) unsafe fn set_access(&self, access: ImageAccess) {
        *self.last_access.lock() = access;
    }
}

impl SwapchainSyncState {
    fn associate_acquire_fence_with_present_operation(
        &mut self,
        generation: u32,
        image_index: u32,
        acquire_fence: Fence,
    ) {
        if let Some(present_operation) = self
            .present_queue
            .iter_mut()
            .rev()
            .take_while(|present_operation| present_operation.generation == generation)
            .find(|present_operation| present_operation.image_index == image_index)
        {
            assert!(present_operation.cleanup_fence.is_none());
            present_operation.cleanup_fence = Some(acquire_fence);
        } else {
            // There is no previous present operation of the same image index. We still have to do
            // something with the fence though, so we push a dummy present operation to the queue
            // just to clean up the fence.
            self.present_queue.push_back(SwapchainPresentOperation {
                generation,
                acquire_semaphore: None,
                image_index,
                pre_present_semaphore: None,
                present_semaphore: None,
                cleanup_fence: Some(acquire_fence),
            });
        }
    }

    fn allocate_semaphore(&mut self) -> Result<Semaphore, VulkanError> {
        if let Some(semaphore) = self.semaphore_pool.pop() {
            Ok(semaphore)
        } else {
            // SAFETY: The parameters are valid.
            let semaphore =
                unsafe { Semaphore::new_unchecked(&self.device, &SemaphoreCreateInfo::default()) }?;

            Ok(semaphore)
        }
    }

    fn deallocate_semaphore(&mut self, semaphore: Semaphore) {
        self.semaphore_pool.push(semaphore);
    }

    fn allocate_fence(&mut self) -> Result<Fence, VulkanError> {
        if let Some(fence) = self.fence_pool.pop() {
            Ok(fence)
        } else {
            // SAFETY: The parameters are valid.
            let fence = unsafe { Fence::new_unchecked(&self.device, &FenceCreateInfo::default()) }?;

            Ok(fence)
        }
    }

    fn deallocate_fence(&mut self, fence: Fence) {
        self.fence_pool.push(fence);
    }
}

const SWAPCHAIN_TAG_BITS: u32 = 2;
const SWAPCHAIN_TAG_MASK: u32 = (1 << SWAPCHAIN_TAG_BITS) - 1;

const SWAPCHAIN_UNUSED_TAG: u32 = 0;
const SWAPCHAIN_EXECUTION_LOCKED_TAG: u32 = 1;
const SWAPCHAIN_RECREATION_LOCKED_TAG: u32 = 2;
const SWAPCHAIN_REMOVED_TAG: u32 = 3;

unsafe impl Send for SwapchainLock {}
unsafe impl Sync for SwapchainLock {}

impl SwapchainLock {
    const fn new(data: SwapchainSyncState) -> Self {
        SwapchainLock {
            state: AtomicU32::new(SWAPCHAIN_UNUSED_TAG),
            data: UnsafeCell::new(data),
        }
    }

    /// Tries to lock the swapchain for use in a task graph execution.
    fn try_execute(&self, generation: u32) -> Result<(), SwapchainLockError> {
        let state = (generation << SWAPCHAIN_TAG_BITS) | SWAPCHAIN_UNUSED_TAG;
        let new_state = (generation << SWAPCHAIN_TAG_BITS) | SWAPCHAIN_EXECUTION_LOCKED_TAG;

        self.state
            .compare_exchange(state, new_state, Acquire, Relaxed)
            .map_err(SwapchainLockError::from_state)?;

        Ok(())
    }

    /// Tries to lock the swapchain for use in a recreation.
    fn try_recreate(&self, generation: u32) -> Result<(), SwapchainLockError> {
        let new_generation = generation.wrapping_add(1);
        let state = (generation << SWAPCHAIN_TAG_BITS) | SWAPCHAIN_UNUSED_TAG;
        let new_state = (new_generation << SWAPCHAIN_TAG_BITS) | SWAPCHAIN_RECREATION_LOCKED_TAG;

        self.state
            .compare_exchange(state, new_state, Acquire, Relaxed)
            .map_err(SwapchainLockError::from_state)?;

        Ok(())
    }

    /// Returns a mutable reference to the data.
    ///
    /// # Safety
    ///
    /// - The swapchain must have been locked.
    /// - There must not be other references to the data.
    #[allow(clippy::mut_from_ref)]
    unsafe fn get_mut_unchecked(&self) -> &mut SwapchainSyncState {
        // SAFETY: The caller must ensure that the lock has been locked -- which ensures correct
        // synchronization -- and that there aren't other references.
        unsafe { &mut *self.data.get() }
    }

    /// Unlocks the swapchain.
    ///
    /// # Safety
    ///
    /// - The swapchain must have been locked.
    unsafe fn unlock(&self) {
        const { assert!(SWAPCHAIN_UNUSED_TAG == 0) };

        // This sets the tag back to `SWAPCHAIN_UNUSED_TAG` by exploiting the fact that
        // `SWAPCHAIN_UNUSED_TAG` is `0`.
        self.state.fetch_and(!SWAPCHAIN_TAG_MASK, Release);
    }

    /// Tries to remove the swapchain.
    fn try_remove(&self, generation: u32) -> Result<(), SwapchainLockError> {
        let state = (generation << SWAPCHAIN_TAG_BITS) | SWAPCHAIN_UNUSED_TAG;
        let new_state = (generation << SWAPCHAIN_TAG_BITS) | SWAPCHAIN_REMOVED_TAG;

        self.state
            .compare_exchange(state, new_state, Acquire, Relaxed)
            .map_err(SwapchainLockError::from_state)?;

        Ok(())
    }
}

impl fmt::Debug for SwapchainLock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SwapchainLock").finish_non_exhaustive()
    }
}

pub(super) struct SwapchainLockGuard<'a> {
    state: &'a SwapchainState,
}

impl Drop for SwapchainLockGuard<'_> {
    fn drop(&mut self) {
        unsafe { self.state.unlock() };
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum SwapchainLockError {
    Unused,
    ExecutionLocked,
    RecreationLocked,
    Removed,
}

impl SwapchainLockError {
    fn from_state(state: u32) -> Self {
        match state & SWAPCHAIN_TAG_MASK {
            SWAPCHAIN_UNUSED_TAG => SwapchainLockError::Unused,
            SWAPCHAIN_EXECUTION_LOCKED_TAG => SwapchainLockError::ExecutionLocked,
            SWAPCHAIN_RECREATION_LOCKED_TAG => SwapchainLockError::RecreationLocked,
            SWAPCHAIN_REMOVED_TAG => SwapchainLockError::Removed,
            _ => unreachable!(),
        }
    }
}

impl fmt::Display for SwapchainLockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unused | Self::RecreationLocked => {
                f.write_str("the swapchain has already been recreated")
            }
            Self::ExecutionLocked => {
                f.write_str("a task graph using the swapchain is being executed")
            }
            Self::Removed => f.write_str("the swapchain has already been removed"),
        }
    }
}

impl Error for SwapchainLockError {}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum SwapchainSyncStage {
    SignalAcquire,
    WaitAcquire,
    SignalPrePresent,
    WaitPrePresent,
    SignalPresent,
}

// FIXME: imported/exported fences
#[derive(Debug)]
pub struct Flight {
    // HACK: We need this in order to collect garbage.
    resources: Weak<Resources>,
    frame_count: NonZero<u32>,
    biased_started_frame: AtomicU64,
    current_frame: AtomicU64,
    biased_complete_frame: AtomicU64,
    fences: SmallVec<[RwLock<Fence>; 3]>,
    garbage_queue: collector::LocalQueue,
    lock: Mutex<()>,
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
            lock: Mutex::new(()),
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
        self.biased_started_frame.load(Relaxed)
    }

    /// Returns the current frame counter value. This always starts out at `0` and increases by `1`
    /// after every successful [task graph execution].
    ///
    /// [task graph execution]: crate::graph::ExecutableTaskGraph::execute
    #[inline]
    #[must_use]
    pub fn current_frame(&self) -> u64 {
        self.current_frame.load(Relaxed)
    }

    /// Returns the latest complete frame stored at a bias of `frame_count + 1`. That means that if
    /// this value reaches `n + frame_count + 1` then frame `n` has been waited on. This starts out
    /// at `frame_count` because there is nothing to wait for in the first ever `frame_count`
    /// frames.
    fn biased_complete_frame(&self) -> u64 {
        self.biased_complete_frame.load(Relaxed)
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

    pub(crate) fn try_lock(&self) -> Option<MutexGuard<'_, ()>> {
        self.lock.try_lock()
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
        self.biased_complete_frame
            .fetch_update(Relaxed, Relaxed, |biased_complete_frame| {
                (biased_complete_frame < biased_frame).then_some(biased_frame)
            })
    }

    pub(crate) fn is_oldest_frame_complete(&self) -> bool {
        self.biased_complete_frame() > self.current_frame()
    }

    pub(crate) fn is_biased_frame_complete(&self, biased_frame: u64) -> bool {
        self.biased_complete_frame() >= biased_frame
    }

    pub(crate) unsafe fn start_next_frame(&self) {
        self.biased_started_frame.fetch_add(1, Relaxed);
    }

    pub(crate) unsafe fn undo_start_next_frame(&self) {
        self.biased_started_frame.fetch_sub(1, Relaxed);
    }

    pub(crate) unsafe fn next_frame(&self) {
        self.current_frame.fetch_add(1, Relaxed);
    }
}
