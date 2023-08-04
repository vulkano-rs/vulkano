// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{PresentMode, Swapchain};
use crate::{
    buffer::Buffer,
    device::{Device, DeviceOwned, Queue},
    image::{Image, ImageLayout},
    sync::{
        fence::Fence,
        future::{AccessCheckError, AccessError, GpuFuture, SubmitAnyBuilder},
        semaphore::Semaphore,
    },
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, VulkanError,
    VulkanObject,
};
use smallvec::smallvec;
use std::{
    fmt::Debug,
    mem::MaybeUninit,
    num::NonZeroU64,
    ops::Range,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::Duration,
};

/// Tries to take ownership of an image in order to draw on it.
///
/// The function returns the index of the image in the array of images that was returned
/// when creating the swapchain, plus a future that represents the moment when the image will
/// become available from the GPU (which may not be *immediately*).
///
/// If you try to draw on an image without acquiring it first, the execution will block. (TODO
/// behavior may change).
///
/// The second field in the tuple in the Ok result is a bool represent if the acquisition was
/// suboptimal. In this case the acquired image is still usable, but the swapchain should be
/// recreated as the Surface's properties no longer match the swapchain.
pub fn acquire_next_image(
    swapchain: Arc<Swapchain>,
    timeout: Option<Duration>,
) -> Result<(u32, bool, SwapchainAcquireFuture), Validated<VulkanError>> {
    let semaphore = Arc::new(Semaphore::from_pool(swapchain.device.clone())?);
    let fence = Fence::from_pool(swapchain.device.clone())?;

    let AcquiredImage {
        image_index,
        suboptimal,
    } = {
        // Check that this is not an old swapchain. From specs:
        // > swapchain must not have been replaced by being passed as the
        // > VkSwapchainCreateInfoKHR::oldSwapchain value to vkCreateSwapchainKHR
        let retired = swapchain.is_retired.lock();
        if *retired {
            return Err(VulkanError::OutOfDate.into());
        }

        let acquire_result =
            unsafe { acquire_next_image_raw(&swapchain, timeout, Some(&semaphore), Some(&fence)) };

        if matches!(
            acquire_result,
            Err(Validated::Error(VulkanError::FullScreenExclusiveModeLost))
        ) {
            swapchain
                .full_screen_exclusive_held
                .store(false, Ordering::SeqCst);
        }

        acquire_result?
    };

    Ok((
        image_index,
        suboptimal,
        SwapchainAcquireFuture {
            swapchain,
            semaphore: Some(semaphore),
            fence: Some(fence),
            image_index,
            finished: AtomicBool::new(false),
        },
    ))
}

/// Unsafe variant of `acquire_next_image`.
///
/// # Safety
///
/// - The semaphore and/or the fence must be kept alive until it is signaled.
/// - The swapchain must not have been replaced by being passed as the old swapchain when creating
///   a new one.
pub unsafe fn acquire_next_image_raw(
    swapchain: &Swapchain,
    timeout: Option<Duration>,
    semaphore: Option<&Semaphore>,
    fence: Option<&Fence>,
) -> Result<AcquiredImage, Validated<VulkanError>> {
    let fns = swapchain.device.fns();

    let timeout_ns = if let Some(timeout) = timeout {
        timeout
            .as_secs()
            .saturating_mul(1_000_000_000)
            .saturating_add(timeout.subsec_nanos() as u64)
    } else {
        u64::MAX
    };

    let mut out = MaybeUninit::uninit();
    let result = (fns.khr_swapchain.acquire_next_image_khr)(
        swapchain.device.handle(),
        swapchain.handle,
        timeout_ns,
        semaphore
            .map(|s| s.handle())
            .unwrap_or(ash::vk::Semaphore::null()),
        fence.map(|f| f.handle()).unwrap_or(ash::vk::Fence::null()),
        out.as_mut_ptr(),
    );

    let suboptimal = match result {
        ash::vk::Result::SUCCESS => false,
        ash::vk::Result::SUBOPTIMAL_KHR => true,
        ash::vk::Result::NOT_READY => return Err(VulkanError::NotReady.into()),
        ash::vk::Result::TIMEOUT => return Err(VulkanError::Timeout.into()),
        err => return Err(VulkanError::from(err).into()),
    };

    if let Some(semaphore) = semaphore {
        let mut state = semaphore.state();
        state.swapchain_acquire();
    }

    if let Some(fence) = fence {
        let mut state = fence.state();
        state.import_swapchain_acquire();
    }

    Ok(AcquiredImage {
        image_index: out.assume_init(),
        suboptimal,
    })
}

pub struct AcquiredImage {
    pub image_index: u32,
    pub suboptimal: bool,
}

/// Represents the moment when the GPU will have access to a swapchain image.
#[must_use]
pub struct SwapchainAcquireFuture {
    swapchain: Arc<Swapchain>,
    image_index: u32,
    // Semaphore that is signalled when the acquire is complete. Empty if the acquire has already
    // happened.
    semaphore: Option<Arc<Semaphore>>,
    // Fence that is signalled when the acquire is complete. Empty if the acquire has already
    // happened.
    fence: Option<Fence>,
    finished: AtomicBool,
}

impl SwapchainAcquireFuture {
    /// Returns the index of the image in the list of images returned when creating the swapchain.
    pub fn image_index(&self) -> u32 {
        self.image_index
    }

    /// Returns the corresponding swapchain.
    pub fn swapchain(&self) -> &Arc<Swapchain> {
        &self.swapchain
    }

    /// Blocks the current thread until the swapchain image has been acquired, or timeout
    ///
    /// If timeout is `None`, will potentially block forever
    ///
    /// You still need to join with this future for present to work
    pub fn wait(&self, timeout: Option<Duration>) -> Result<(), VulkanError> {
        match &self.fence {
            Some(fence) => fence.wait(timeout),
            None => Ok(()),
        }
    }
}

unsafe impl GpuFuture for SwapchainAcquireFuture {
    fn cleanup_finished(&mut self) {}

    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Validated<VulkanError>> {
        if let Some(ref semaphore) = self.semaphore {
            let sem = smallvec![semaphore.clone()];
            Ok(SubmitAnyBuilder::SemaphoresWait(sem))
        } else {
            Ok(SubmitAnyBuilder::Empty)
        }
    }

    fn flush(&self) -> Result<(), Validated<VulkanError>> {
        Ok(())
    }

    unsafe fn signal_finished(&self) {
        self.finished.store(true, Ordering::SeqCst);
    }

    fn queue_change_allowed(&self) -> bool {
        true
    }

    fn queue(&self) -> Option<Arc<Queue>> {
        None
    }

    fn check_buffer_access(
        &self,
        _buffer: &Buffer,
        _range: Range<DeviceSize>,
        _exclusive: bool,
        _queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        Err(AccessCheckError::Unknown)
    }

    fn check_image_access(
        &self,
        image: &Image,
        _range: Range<DeviceSize>,
        _exclusive: bool,
        expected_layout: ImageLayout,
        _queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        if self.swapchain.index_of_image(image) != Some(self.image_index) {
            return Err(AccessCheckError::Unknown);
        }

        if !self.swapchain.images[self.image_index as usize]
            .layout_initialized
            .load(Ordering::Relaxed)
            && expected_layout != ImageLayout::Undefined
        {
            return Err(AccessCheckError::Denied(AccessError::ImageNotInitialized {
                requested: expected_layout,
            }));
        }

        if expected_layout != ImageLayout::Undefined && expected_layout != ImageLayout::PresentSrc {
            return Err(AccessCheckError::Denied(
                AccessError::UnexpectedImageLayout {
                    allowed: ImageLayout::PresentSrc,
                    requested: expected_layout,
                },
            ));
        }

        Ok(())
    }

    #[inline]
    fn check_swapchain_image_acquired(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        before: bool,
    ) -> Result<(), AccessCheckError> {
        if before {
            Ok(())
        } else {
            if swapchain == self.swapchain.as_ref() && image_index == self.image_index {
                Ok(())
            } else {
                Err(AccessCheckError::Unknown)
            }
        }
    }
}

impl Drop for SwapchainAcquireFuture {
    fn drop(&mut self) {
        if thread::panicking() {
            return;
        }

        if let Some(fence) = &self.fence {
            fence.wait(None).unwrap(); // TODO: handle error?
            self.semaphore = None;
        }

        // TODO: if this future is destroyed without being presented, then eventually acquiring
        // a new image will block forever ; difficulty: hard
    }
}

unsafe impl DeviceOwned for SwapchainAcquireFuture {
    fn device(&self) -> &Arc<Device> {
        &self.swapchain.device
    }
}

/// Presents an image on the screen.
///
/// The actual behavior depends on the present mode that you passed when creating the swapchain.
pub fn present<F>(
    before: F,
    queue: Arc<Queue>,
    swapchain_info: SwapchainPresentInfo,
) -> PresentFuture<F>
where
    F: GpuFuture,
{
    assert!(swapchain_info.image_index < swapchain_info.swapchain.image_count());

    // TODO: restore this check with a dummy ImageAccess implementation
    /*let swapchain_image = me.images.lock().unwrap().get(index).unwrap().0.upgrade().unwrap();       // TODO: return error instead
    // Normally if `check_image_access` returns false we're supposed to call the `gpu_access`
    // function on the image instead. But since we know that this method on `SwapchainImage`
    // always returns false anyway (by design), we don't need to do it.
    assert!(before.check_image_access(&swapchain_image, ImageLayout::PresentSrc, true, &queue).is_ok());         // TODO: return error instead*/

    PresentFuture {
        previous: before,
        queue,
        swapchain_info,
        flushed: AtomicBool::new(false),
        finished: AtomicBool::new(false),
    }
}

/// Parameters to execute present operations on a queue.
#[derive(Clone, Debug)]
pub struct PresentInfo {
    /// The semaphores to wait for before beginning the execution of the present operations.
    ///
    /// The default value is empty.
    pub wait_semaphores: Vec<Arc<Semaphore>>,

    /// The present operations to perform.
    ///
    /// The default value is empty.
    pub swapchain_infos: Vec<SwapchainPresentInfo>,

    pub _ne: crate::NonExhaustive,
}

impl Default for PresentInfo {
    #[inline]
    fn default() -> Self {
        Self {
            wait_semaphores: Vec::new(),
            swapchain_infos: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Parameters for a single present operation on a swapchain.
#[derive(Clone, Debug)]
pub struct SwapchainPresentInfo {
    /// The swapchain to present to.
    ///
    /// There is no default value.
    pub swapchain: Arc<Swapchain>,

    /// The index of the swapchain image to present to.
    ///
    /// The image must have been acquired first; this is the index that `acquire_next_image`
    /// returns.
    ///
    /// There is no default value.
    pub image_index: u32,

    /* TODO: enable
    /// The fence to signal when the presentation has completed.
    ///
    /// If this is not `None`, then the
    /// [`ext_swapchain_maintenance1`](crate::device::DeviceExtensions::ext_swapchain_maintenance1)
    /// extension must be enabled on the device.
    ///
    /// The default value is `None`.
    pub fence: Option<Arc<Fence>>,
     */
    /// An id used to identify this present operation.
    ///
    /// If `present_id` is `Some`, the [`present_id`](crate::device::Features::present_id) feature
    /// must be enabled on the device. The id must be greater than any id previously used for
    /// `swapchain`. If a swapchain is recreated, this resets.
    ///
    /// The default value is `None`.
    pub present_id: Option<NonZeroU64>,

    /// The new present mode to use for presenting. This mode will be used for the current
    /// present, and any future presents where this value is `None`.
    ///
    /// If this is not `None`, then the provided present mode must be one of the present modes
    /// specified with [`present_modes`] when creating the
    /// swapchain.
    ///
    /// The default value is `None`.
    ///
    /// [`present_modes`]: crate::swapchain::SwapchainCreateInfo::present_modes
    pub present_mode: Option<PresentMode>,

    /// An optimization hint to the implementation, that only some parts of the swapchain image are
    /// going to be updated by the present operation.
    ///
    /// If `present_regions` is not empty, then the
    /// [`khr_incremental_present`](crate::device::DeviceExtensions::khr_incremental_present)
    /// extension must be enabled on the device. The implementation will update the provided
    /// regions of the swapchain image, and _may_ ignore the other areas. However, as this is just
    /// a hint, the Vulkan implementation is free to ignore the regions altogether and update
    /// everything.
    ///
    /// If `present_regions` is empty, that means that all of the swapchain image must be updated.
    ///
    /// The default value is empty.
    pub present_regions: Vec<RectangleLayer>,

    pub _ne: crate::NonExhaustive,
}

impl SwapchainPresentInfo {
    /// Returns a `SwapchainPresentInfo` with the specified `swapchain` and `image_index`.
    #[inline]
    pub fn swapchain_image_index(swapchain: Arc<Swapchain>, image_index: u32) -> Self {
        Self {
            swapchain,
            image_index,
            present_id: None,
            present_mode: None,
            present_regions: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Represents a rectangular region on an image layer.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RectangleLayer {
    /// Coordinates in pixels of the top-left hand corner of the rectangle.
    pub offset: [u32; 2],

    /// Dimensions in pixels of the rectangle.
    pub extent: [u32; 2],

    /// The layer of the image. For images with only one layer, the value of layer must be 0.
    pub layer: u32,
}

impl RectangleLayer {
    /// Returns true if this rectangle layer is compatible with swapchain.
    #[inline]
    pub fn is_compatible_with(&self, swapchain: &Swapchain) -> bool {
        self.offset[0] + self.extent[0] <= swapchain.image_extent()[0]
            && self.offset[1] + self.extent[1] <= swapchain.image_extent()[1]
            && self.layer < swapchain.image_array_layers()
    }
}

impl From<&RectangleLayer> for ash::vk::RectLayerKHR {
    #[inline]
    fn from(val: &RectangleLayer) -> Self {
        ash::vk::RectLayerKHR {
            offset: ash::vk::Offset2D {
                x: val.offset[0] as i32,
                y: val.offset[1] as i32,
            },
            extent: ash::vk::Extent2D {
                width: val.extent[0],
                height: val.extent[1],
            },
            layer: val.layer,
        }
    }
}

/// Represents a swapchain image being presented on the screen.
#[must_use = "Dropping this object will immediately block the thread until the GPU has finished processing the submission"]
pub struct PresentFuture<P>
where
    P: GpuFuture,
{
    previous: P,
    queue: Arc<Queue>,
    swapchain_info: SwapchainPresentInfo,
    // True if `flush()` has been called on the future, which means that the present command has
    // been submitted.
    flushed: AtomicBool,
    // True if `signal_finished()` has been called on the future, which means that the future has
    // been submitted and has already been processed by the GPU.
    finished: AtomicBool,
}

impl<P> PresentFuture<P>
where
    P: GpuFuture,
{
    /// Returns the index of the image in the list of images returned when creating the swapchain.
    pub fn image_id(&self) -> u32 {
        self.swapchain_info.image_index
    }

    /// Returns the corresponding swapchain.
    pub fn swapchain(&self) -> &Arc<Swapchain> {
        &self.swapchain_info.swapchain
    }
}

unsafe impl<P> GpuFuture for PresentFuture<P>
where
    P: GpuFuture,
{
    fn cleanup_finished(&mut self) {
        self.previous.cleanup_finished();
    }

    unsafe fn build_submission(&self) -> Result<SubmitAnyBuilder, Validated<VulkanError>> {
        if self.flushed.load(Ordering::SeqCst) {
            return Ok(SubmitAnyBuilder::Empty);
        }

        let mut swapchain_info = self.swapchain_info.clone();
        debug_assert!(swapchain_info.image_index < swapchain_info.swapchain.image_count());
        let device = swapchain_info.swapchain.device();

        if !device.enabled_features().present_id {
            swapchain_info.present_id = None;
        }

        if device.enabled_extensions().khr_incremental_present {
            for rectangle in &swapchain_info.present_regions {
                assert!(rectangle.is_compatible_with(swapchain_info.swapchain.as_ref()));
            }
        } else {
            swapchain_info.present_regions = Default::default();
        }

        let _queue = self.previous.queue();

        // TODO: if the swapchain image layout is not PRESENT, should add a transition command
        // buffer

        Ok(match self.previous.build_submission()? {
            SubmitAnyBuilder::Empty => SubmitAnyBuilder::QueuePresent(PresentInfo {
                swapchain_infos: vec![self.swapchain_info.clone()],
                ..Default::default()
            }),
            SubmitAnyBuilder::SemaphoresWait(semaphores) => {
                SubmitAnyBuilder::QueuePresent(PresentInfo {
                    wait_semaphores: semaphores.into_iter().collect(),
                    swapchain_infos: vec![self.swapchain_info.clone()],
                    ..Default::default()
                })
            }
            SubmitAnyBuilder::CommandBuffer(_, _) => {
                self.previous.flush()?;

                SubmitAnyBuilder::QueuePresent(PresentInfo {
                    swapchain_infos: vec![self.swapchain_info.clone()],
                    ..Default::default()
                })
            }
            SubmitAnyBuilder::BindSparse(_, _) => {
                self.previous.flush()?;

                SubmitAnyBuilder::QueuePresent(PresentInfo {
                    swapchain_infos: vec![self.swapchain_info.clone()],
                    ..Default::default()
                })
            }
            SubmitAnyBuilder::QueuePresent(mut present_info) => {
                if present_info.swapchain_infos.first().map_or(false, |prev| {
                    prev.present_mode.is_some() != self.swapchain_info.present_mode.is_some()
                }) {
                    // If the present mode Option variants don't match, create a new command.
                    self.previous.flush()?;

                    SubmitAnyBuilder::QueuePresent(PresentInfo {
                        swapchain_infos: vec![self.swapchain_info.clone()],
                        ..Default::default()
                    })
                } else {
                    // Otherwise, add our swapchain to the previous.
                    present_info
                        .swapchain_infos
                        .push(self.swapchain_info.clone());

                    SubmitAnyBuilder::QueuePresent(present_info)
                }
            }
        })
    }

    fn flush(&self) -> Result<(), Validated<VulkanError>> {
        unsafe {
            // If `flushed` already contains `true`, then `build_submission` will return `Empty`.

            let build_submission_result = self.build_submission();
            self.flushed.store(true, Ordering::SeqCst);

            match build_submission_result? {
                SubmitAnyBuilder::Empty => Ok(()),
                SubmitAnyBuilder::QueuePresent(present_info) => {
                    let PresentInfo {
                        wait_semaphores: _,
                        swapchain_infos,
                        _ne: _,
                    } = &present_info;

                    let has_present_mode = swapchain_infos
                        .first()
                        .map_or(false, |first| first.present_mode.is_some());

                    for swapchain_info in swapchain_infos {
                        let &SwapchainPresentInfo {
                            ref swapchain,
                            image_index: _,
                            present_id,
                            present_regions: _,
                            present_mode,
                            _ne: _,
                        } = swapchain_info;

                        if present_id.map_or(false, |present_id| {
                            !swapchain.try_claim_present_id(present_id)
                        }) {
                            return Err(Box::new(ValidationError {
                                problem: "the provided `present_id` was not greater than any \
                                    `present_id` passed previously for the same swapchain"
                                    .into(),
                                vuids: &["VUID-VkPresentIdKHR-presentIds-04999"],
                                ..Default::default()
                            })
                            .into());
                        }

                        if let Some(present_mode) = present_mode {
                            assert!(has_present_mode);

                            if !swapchain.present_modes().contains(&present_mode) {
                                return Err(Box::new(ValidationError {
                                    problem: "the requested present mode is not one of the modes \
                                        in `swapchain.present_modes()`"
                                        .into(),
                                    vuids: &[
                                        "VUID-VkSwapchainPresentModeInfoEXT-pPresentModes-07761",
                                    ],
                                    ..Default::default()
                                })
                                .into());
                            }
                        } else {
                            assert!(!has_present_mode);
                        }
                    }

                    match self.previous.check_swapchain_image_acquired(
                        &self.swapchain_info.swapchain,
                        self.swapchain_info.image_index,
                        true,
                    ) {
                        Ok(_) => (),
                        Err(AccessCheckError::Unknown) => {
                            return Err(Box::new(ValidationError {
                                problem: AccessError::SwapchainImageNotAcquired.to_string().into(),
                                ..Default::default()
                            })
                            .into());
                        }
                        Err(AccessCheckError::Denied(err)) => {
                            return Err(Box::new(ValidationError {
                                problem: err.to_string().into(),
                                ..Default::default()
                            })
                            .into());
                        }
                    }

                    Ok(self
                        .queue
                        .with(|mut q| q.present_unchecked(present_info))?
                        .map(|r| r.map(|_| ()))
                        .fold(Ok(()), Result::and)?)
                }
                _ => unreachable!(),
            }
        }
    }

    unsafe fn signal_finished(&self) {
        self.flushed.store(true, Ordering::SeqCst);
        self.finished.store(true, Ordering::SeqCst);
        self.previous.signal_finished();
    }

    fn queue_change_allowed(&self) -> bool {
        false
    }

    fn queue(&self) -> Option<Arc<Queue>> {
        debug_assert!(match self.previous.queue() {
            None => true,
            Some(q) => q == self.queue,
        });

        Some(self.queue.clone())
    }

    fn check_buffer_access(
        &self,
        buffer: &Buffer,
        range: Range<DeviceSize>,
        exclusive: bool,
        queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        self.previous
            .check_buffer_access(buffer, range, exclusive, queue)
    }

    fn check_image_access(
        &self,
        image: &Image,
        range: Range<DeviceSize>,
        exclusive: bool,
        expected_layout: ImageLayout,
        queue: &Queue,
    ) -> Result<(), AccessCheckError> {
        if self.swapchain_info.swapchain.index_of_image(image)
            == Some(self.swapchain_info.image_index)
        {
            // This future presents the swapchain image, which "unlocks" it. Therefore any attempt
            // to use this swapchain image afterwards shouldn't get granted automatic access.
            // Instead any attempt to access the image afterwards should get an authorization from
            // a later swapchain acquire future. Hence why we return `Unknown` here.
            Err(AccessCheckError::Unknown)
        } else {
            self.previous
                .check_image_access(image, range, exclusive, expected_layout, queue)
        }
    }

    #[inline]
    fn check_swapchain_image_acquired(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        before: bool,
    ) -> Result<(), AccessCheckError> {
        if before {
            self.previous
                .check_swapchain_image_acquired(swapchain, image_index, false)
        } else if swapchain == self.swapchain_info.swapchain.as_ref()
            && image_index == self.swapchain_info.image_index
        {
            Err(AccessError::SwapchainImageNotAcquired.into())
        } else {
            self.previous
                .check_swapchain_image_acquired(swapchain, image_index, false)
        }
    }
}

unsafe impl<P> DeviceOwned for PresentFuture<P>
where
    P: GpuFuture,
{
    fn device(&self) -> &Arc<Device> {
        self.queue.device()
    }
}

impl<P> Drop for PresentFuture<P>
where
    P: GpuFuture,
{
    fn drop(&mut self) {
        if thread::panicking() {
            return;
        }

        unsafe {
            if !*self.flushed.get_mut() {
                // Flushing may fail, that's okay. We will still wait for the queue later, so any
                // previous futures that were flushed correctly will still be waited upon.
                self.flush().ok();
            }

            if !*self.finished.get_mut() {
                // Block until the queue finished.
                self.queue().unwrap().with(|mut q| q.wait_idle()).unwrap();
                self.previous.signal_finished();
            }
        }
    }
}

/// Wait for an image to be presented to the user. Must be used with a `present_id` given to
/// `present_with_id`.
///
/// Returns a bool to represent if the presentation was suboptimal. In this case the swapchain is
/// still usable, but the swapchain should be recreated as the Surface's properties no longer match
/// the swapchain.
pub fn wait_for_present(
    swapchain: Arc<Swapchain>,
    present_id: u64,
    timeout: Option<Duration>,
) -> Result<bool, Validated<VulkanError>> {
    if !swapchain.device.enabled_features().present_wait {
        return Err(Box::new(ValidationError {
            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature("present_wait")])]),
            vuids: &["VUID-vkWaitForPresentKHR-presentWait-06234"],
            ..Default::default()
        })
        .into());
    }

    if present_id == 0 {
        return Err(Box::new(ValidationError {
            context: "present_id".into(),
            problem: "is 0".into(),
            ..Default::default()
        })
        .into());
    }

    let retired = swapchain.is_retired.lock();

    // VUID-vkWaitForPresentKHR-swapchain-04997
    if *retired {
        return Err(VulkanError::OutOfDate.into());
    }

    let timeout_ns = timeout.map(|dur| dur.as_nanos() as u64).unwrap_or(0);

    let result = unsafe {
        (swapchain.device.fns().khr_present_wait.wait_for_present_khr)(
            swapchain.device.handle(),
            swapchain.handle,
            present_id,
            timeout_ns,
        )
    };

    match result {
        ash::vk::Result::SUCCESS => Ok(false),
        ash::vk::Result::SUBOPTIMAL_KHR => Ok(true),
        ash::vk::Result::TIMEOUT => Err(VulkanError::Timeout.into()),
        err => {
            let err = VulkanError::from(err);

            if matches!(err, VulkanError::FullScreenExclusiveModeLost) {
                swapchain
                    .full_screen_exclusive_held
                    .store(false, Ordering::SeqCst);
            }

            Err(err.into())
        }
    }
}
