//! Synchronization state tracking of all resources.

use crate::{Id, InvalidSlotError, Object, Ref};
use ash::vk;
use concurrent_slotmap::{epoch, SlotMap};
use parking_lot::{Mutex, RwLock};
use smallvec::SmallVec;
use std::{
    any::Any,
    hash::Hash,
    num::{NonZeroU32, NonZeroU64},
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};
use thread_local::ThreadLocal;
use vulkano::{
    buffer::{AllocateBufferError, Buffer, BufferCreateInfo},
    command_buffer::allocator::StandardCommandBufferAllocator,
    device::{Device, DeviceOwned},
    image::{AllocateImageError, Image, ImageCreateInfo, ImageLayout, ImageMemory},
    memory::allocator::{AllocationCreateInfo, DeviceLayout, StandardMemoryAllocator},
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    sync::{
        fence::{Fence, FenceCreateFlags, FenceCreateInfo},
        semaphore::Semaphore,
        AccessFlags, PipelineStages,
    },
    Validated, VulkanError,
};

static REGISTERED_DEVICES: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// Tracks the synchronization state of all resources.
///
/// There can only exist one `Resources` collection per device, because there must only be one
/// source of truth in regards to the synchronization state of a resource. In a similar vein, each
/// resource in the collection must be unique.
// FIXME: Custom collector
#[derive(Debug)]
pub struct Resources {
    device: Arc<Device>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,

    global: epoch::GlobalHandle,
    locals: ThreadLocal<epoch::UniqueLocalHandle>,
    buffers: SlotMap<BufferState>,
    images: SlotMap<ImageState>,
    swapchains: SlotMap<SwapchainState>,
    flights: SlotMap<Flight>,
}

#[derive(Debug)]
pub struct BufferState {
    buffer: Arc<Buffer>,
    last_access: Mutex<BufferAccess>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferAccess {
    stage_mask: PipelineStages,
    access_mask: AccessFlags,
    queue_family_index: u32,
}

#[derive(Debug)]
pub struct ImageState {
    image: Arc<Image>,
    last_access: Mutex<ImageAccess>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageAccess {
    stage_mask: PipelineStages,
    access_mask: AccessFlags,
    image_layout: ImageLayout,
    queue_family_index: u32,
}

// FIXME: imported/exported semaphores
#[derive(Debug)]
pub struct SwapchainState {
    swapchain: Arc<Swapchain>,
    images: SmallVec<[Arc<Image>; 3]>,
    pub(crate) semaphores: SmallVec<[SwapchainSemaphoreState; 3]>,
    flight_id: Id<Flight>,
    pub(crate) current_image_index: AtomicU32,
    last_access: Mutex<ImageAccess>,
}

#[derive(Clone, Debug)]
pub(crate) struct SwapchainSemaphoreState {
    pub(crate) image_available_semaphore: Arc<Semaphore>,
    pub(crate) pre_present_complete_semaphore: Arc<Semaphore>,
    pub(crate) tasks_complete_semaphore: Arc<Semaphore>,
}

// FIXME: imported/exported fences
#[derive(Debug)]
pub struct Flight {
    frame_count: NonZeroU32,
    current_frame: AtomicU64,
    fences: SmallVec<[RwLock<Fence>; 3]>,
    pub(crate) state: Mutex<FlightState>,
}

#[derive(Debug)]
pub(crate) struct FlightState {
    pub(crate) death_rows: SmallVec<[DeathRow; 3]>,
}

pub(crate) type DeathRow = Vec<Arc<dyn Any + Send + Sync>>;

impl Resources {
    /// Creates a new `Resources` collection.
    ///
    /// # Panics
    ///
    /// - Panics if `device` already has a `Resources` collection associated with it.
    #[must_use]
    pub fn new(device: &Arc<Device>, create_info: &ResourcesCreateInfo<'_>) -> Arc<Self> {
        let mut registered_devices = REGISTERED_DEVICES.lock();
        let device_addr = Arc::as_ptr(device) as usize;

        assert!(
            !registered_devices.contains(&device_addr),
            "the device already has a `Resources` collection associated with it",
        );

        registered_devices.push(device_addr);

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let global = epoch::GlobalHandle::new();

        Arc::new(Resources {
            device: device.clone(),
            memory_allocator,
            command_buffer_allocator,
            locals: ThreadLocal::new(),
            buffers: SlotMap::with_global(create_info.max_buffers, global.clone()),
            images: SlotMap::with_global(create_info.max_images, global.clone()),
            swapchains: SlotMap::with_global(create_info.max_swapchains, global.clone()),
            flights: SlotMap::with_global(create_info.max_flights, global.clone()),
            global,
        })
    }

    /// Returns the standard memory allocator.
    #[inline]
    #[must_use]
    pub fn memory_allocator(&self) -> &Arc<StandardMemoryAllocator> {
        &self.memory_allocator
    }

    /// Creates a new buffer and adds it to the collection.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.size` is not zero.
    ///
    /// # Errors
    ///
    /// - Returns an error when [`Buffer::new`] returns an error.
    pub fn create_buffer(
        &self,
        create_info: BufferCreateInfo,
        allocation_info: AllocationCreateInfo,
        layout: DeviceLayout,
    ) -> Result<Id<Buffer>, Validated<AllocateBufferError>> {
        let buffer = Buffer::new(
            self.memory_allocator.clone(),
            create_info,
            allocation_info,
            layout,
        )?;

        // SAFETY: We just created the buffer.
        Ok(unsafe { self.add_buffer_unchecked(buffer) })
    }

    /// Creates a new image and adds it to the collection.
    ///
    /// # Errors
    ///
    /// - Returns an error when [`Image::new`] returns an error.
    pub fn create_image(
        &self,
        create_info: ImageCreateInfo,
        allocation_info: AllocationCreateInfo,
    ) -> Result<Id<Image>, Validated<AllocateImageError>> {
        let image = Image::new(self.memory_allocator.clone(), create_info, allocation_info)?;

        // SAFETY: We just created the image.
        Ok(unsafe { self.add_image_unchecked(image) })
    }

    /// Creates a swapchain and adds it to the collection. `flight_id` is the [flight] which will
    /// own the swapchain.
    ///
    /// # Panics
    ///
    /// - Panics if the instance of `surface` is not the same as that of `self.device()`.
    /// - Panics if `flight_id` is invalid.
    /// - Panics if `create_info.min_image_count` is not greater than or equal to the number of
    ///   [frames] of the flight corresponding to `flight_id`.
    ///
    /// # Errors
    ///
    /// - Returns an error when [`Swapchain::new`] returns an error.
    /// - Returns an error when [`add_swapchain`] returns an error.
    ///
    /// [`add_swapchain`]: Self::add_swapchain
    pub fn create_swapchain(
        &self,
        flight_id: Id<Flight>,
        surface: Arc<Surface>,
        create_info: SwapchainCreateInfo,
    ) -> Result<Id<Swapchain>, Validated<VulkanError>> {
        let frames_in_flight = self.flight(flight_id).unwrap().frame_count();

        assert!(create_info.min_image_count >= frames_in_flight);

        let (swapchain, images) = Swapchain::new(self.device().clone(), surface, create_info)?;

        // SAFETY: We just created the swapchain.
        Ok(unsafe { self.add_swapchain_unchecked(flight_id, swapchain, images) }?)
    }

    /// Creates a new [flight] with `frame_count` [frames] and adds it to the collection.
    ///
    /// # Panics
    ///
    /// - Panics if `frame_count` is zero.
    ///
    /// # Errors
    ///
    /// - Returns an error when [`Fence::new_unchecked`] returns an error.
    pub fn create_flight(&self, frame_count: u32) -> Result<Id<Flight>, VulkanError> {
        let frame_count =
            NonZeroU32::new(frame_count).expect("a flight with zero frames is not valid");

        let fences = (0..frame_count.get())
            .map(|_| {
                // SAFETY: The parameters are valid.
                unsafe {
                    Fence::new_unchecked(
                        self.device().clone(),
                        FenceCreateInfo {
                            flags: FenceCreateFlags::SIGNALED,
                            ..Default::default()
                        },
                    )
                }
                .map(RwLock::new)
            })
            .collect::<Result<_, VulkanError>>()?;

        let flight = Flight {
            frame_count,
            current_frame: AtomicU64::new(0),
            fences,
            state: Mutex::new(FlightState {
                death_rows: (0..frame_count.get()).map(|_| Vec::new()).collect(),
            }),
        };

        let slot = self
            .flights
            .insert_with_tag(flight, Flight::TAG, self.pin());

        Ok(unsafe { Id::new(slot) })
    }

    /// Adds a buffer to the collection.
    ///
    /// # Panics
    ///
    /// - Panics if any other references to the buffer exist.
    /// - Panics if the device of `buffer` is not the same as that of `self`.
    #[must_use]
    pub fn add_buffer(&self, mut buffer: Arc<Buffer>) -> Id<Buffer> {
        assert!(Arc::get_mut(&mut buffer).is_some());
        assert_eq!(buffer.device(), self.device());

        unsafe { self.add_buffer_unchecked(buffer) }
    }

    unsafe fn add_buffer_unchecked(&self, buffer: Arc<Buffer>) -> Id<Buffer> {
        let state = BufferState {
            buffer,
            last_access: Mutex::new(BufferAccess::NONE),
        };

        let slot = self.buffers.insert_with_tag(state, Buffer::TAG, self.pin());

        unsafe { Id::new(slot) }
    }

    /// Adds an image to the collection.
    ///
    /// # Panics
    ///
    /// - Panics if any other references to the image exist.
    /// - Panics if the device of `image` is not the same as that of `self`.
    /// - Panics if `image` is a swapchain image.
    #[must_use]
    pub fn add_image(&self, mut image: Arc<Image>) -> Id<Image> {
        assert!(Arc::get_mut(&mut image).is_some());
        assert_eq!(image.device(), self.device());

        assert!(
            !matches!(image.memory(), ImageMemory::Swapchain { .. }),
            "swapchain images cannot be added like regular images; please use \
            `Resources::add_swapchain` instead",
        );

        unsafe { self.add_image_unchecked(image) }
    }

    unsafe fn add_image_unchecked(&self, image: Arc<Image>) -> Id<Image> {
        let state = ImageState {
            image,
            last_access: Mutex::new(ImageAccess::NONE),
        };

        let slot = self.images.insert_with_tag(state, Image::TAG, self.pin());

        unsafe { Id::new(slot) }
    }

    /// Adds a swapchain to the collection. `(swapchain, images)` must correspond to the value
    /// returned by one of the [`Swapchain`] constructors or by [`Swapchain::recreate`].
    /// `flight_id` is the [flight] which will own the swapchain.
    ///
    /// # Panics
    ///
    /// - Panics if any other references to the swapchain or its images exist.
    /// - Panics if the device of `swapchain` is not the same as that of `self`.
    /// - Panics if the `images` don't comprise the images of `swapchain`.
    /// - Panics if `flight_id` is invalid.
    /// - Panics if `swapchain.image_count()` is not greater than or equal to the number of
    ///   [frames] of the flight corresponding to `flight_id`.
    ///
    /// # Errors
    ///
    /// - Returns an error when [`Semaphore::new_unchecked`] returns an error.
    pub fn add_swapchain(
        &self,
        flight_id: Id<Flight>,
        swapchain: Arc<Swapchain>,
        mut images: Vec<Arc<Image>>,
    ) -> Result<Id<Swapchain>, VulkanError> {
        assert_eq!(swapchain.device(), self.device());
        assert_eq!(images.len(), swapchain.image_count() as usize);

        let frames_in_flight = self.flight(flight_id).unwrap().frame_count();

        assert!(swapchain.image_count() >= frames_in_flight);

        for (index, image) in images.iter_mut().enumerate() {
            match image.memory() {
                ImageMemory::Swapchain {
                    swapchain: image_swapchain,
                    image_index,
                } => {
                    assert_eq!(image_swapchain, &swapchain);
                    assert_eq!(*image_index as usize, index);
                    assert!(Arc::get_mut(image).is_some());
                }
                _ => panic!("not a swapchain image"),
            }
        }

        // It is against the safety contract of `Arc::(de,in)crement_strong_count` to call them
        // with a pointer obtained through `Arc::as_ptr`, even though that's perfectly safe to do,
        // so we have to go through this hoop.
        let ptr = Arc::into_raw(swapchain);
        let mut swapchain = unsafe { Arc::from_raw(ptr) };

        // The following is extremely cursed, but as of right now the only way to assert that we
        // own the only references to the swapchain.
        {
            for _ in 0..images.len() {
                // SAFETY: The pointer was obtained through `Arc::into_raw` above, and we checked
                // that each of the images is a swapchain image belonging to the same swapchain
                // also above, which means that there are at least `images.len()` references to
                // this swapchain besides `swapchain` itself.
                unsafe { Arc::decrement_strong_count(ptr) };
            }

            let we_own_the_only_references = Arc::get_mut(&mut swapchain).is_some();

            for _ in 0..images.len() {
                // SAFETY: Same as the `Arc::decrement_strong_count` above.
                unsafe { Arc::increment_strong_count(ptr) };
            }

            assert!(we_own_the_only_references);
        }

        unsafe { self.add_swapchain_unchecked(flight_id, swapchain, images) }
    }

    unsafe fn add_swapchain_unchecked(
        &self,
        flight_id: Id<Flight>,
        swapchain: Arc<Swapchain>,
        images: Vec<Arc<Image>>,
    ) -> Result<Id<Swapchain>, VulkanError> {
        let guard = &self.pin();

        let frames_in_flight = unsafe { self.flight_unprotected(flight_id) }
            .unwrap()
            .frame_count();

        let semaphores = (0..frames_in_flight)
            .map(|_| {
                Ok(SwapchainSemaphoreState {
                    // SAFETY: The parameters are valid.
                    image_available_semaphore: Arc::new(unsafe {
                        Semaphore::new_unchecked(self.device().clone(), Default::default())
                    }?),
                    // SAFETY: The parameters are valid.
                    pre_present_complete_semaphore: Arc::new(unsafe {
                        Semaphore::new_unchecked(self.device().clone(), Default::default())
                    }?),
                    // SAFETY: The parameters are valid.
                    tasks_complete_semaphore: Arc::new(unsafe {
                        Semaphore::new_unchecked(self.device().clone(), Default::default())
                    }?),
                })
            })
            .collect::<Result<_, VulkanError>>()?;

        let state = SwapchainState {
            swapchain,
            images: images.into(),
            semaphores,
            flight_id,
            current_image_index: AtomicU32::new(u32::MAX),
            last_access: Mutex::new(ImageAccess::NONE),
        };

        let slot = self
            .swapchains
            .insert_with_tag(state, Swapchain::TAG, guard);

        Ok(unsafe { Id::new(slot) })
    }

    /// Calls [`Swapchain::recreate`] on the swapchain corresponding to `id` and adds the new
    /// swapchain to the collection. The old swapchain will be cleaned up as soon as possible.
    ///
    /// # Panics
    ///
    /// - Panics if called from multiple threads at the same time.
    /// - Panics if the flight is currently being executed.
    /// - Panics if `f` panics.
    /// - Panics if [`Swapchain::recreate`] panics.
    /// - Panics if `new_swapchain.image_count()` is not greater than or equal to the number of
    ///   [frames] of the flight that owns the swapchain.
    ///
    /// # Errors
    ///
    /// - Returns an error when [`Swapchain::recreate`] returns an error.
    pub fn recreate_swapchain(
        &self,
        id: Id<Swapchain>,
        f: impl FnOnce(SwapchainCreateInfo) -> SwapchainCreateInfo,
    ) -> Result<Id<Swapchain>, Validated<VulkanError>> {
        let guard = self.pin();

        let state = unsafe { self.swapchain_unprotected(id) }.unwrap();
        let swapchain = state.swapchain();
        let flight_id = state.flight_id;
        let flight = unsafe { self.flight_unprotected_unchecked(flight_id) };
        let mut flight_state = flight.state.try_lock().unwrap();

        let (new_swapchain, new_images) = swapchain.recreate(f(swapchain.create_info()))?;

        let frames_in_flight = flight.frame_count();

        assert!(new_swapchain.image_count() >= frames_in_flight);

        let death_row = &mut flight_state.death_rows[flight.previous_frame_index() as usize];
        death_row.push(swapchain.clone());

        let new_state = SwapchainState {
            swapchain: new_swapchain,
            images: new_images.into(),
            semaphores: state.semaphores.clone(),
            flight_id,
            current_image_index: AtomicU32::new(u32::MAX),
            last_access: Mutex::new(ImageAccess::NONE),
        };

        let slot = self
            .swapchains
            .insert_with_tag(new_state, Swapchain::TAG, guard);

        let _ = unsafe { self.remove_swapchain(id) };

        Ok(unsafe { Id::new(slot) })
    }

    /// Removes the buffer corresponding to `id`.
    ///
    /// # Safety
    ///
    /// - Unless the buffer is being kept alive by other means, it must not be in use in any
    ///   pending command buffer, and if it is used in any command buffer that's in the executable
    ///   or recording state, that command buffer must never be executed.
    pub unsafe fn remove_buffer(&self, id: Id<Buffer>) -> Result<Ref<'_, BufferState>> {
        self.buffers
            .remove(id.slot, self.pin())
            .map(Ref)
            .ok_or(InvalidSlotError::new(id))
    }

    /// Removes the image corresponding to `id`.
    ///
    /// # Safety
    ///
    /// - Unless the image is being kept alive by other means, it must not be in use in any pending
    ///   command buffer, and if it is used in any command buffer that's in the executable or
    ///   recording state, that command buffer must never be executed.
    pub unsafe fn remove_image(&self, id: Id<Image>) -> Result<Ref<'_, ImageState>> {
        self.images
            .remove(id.slot, self.pin())
            .map(Ref)
            .ok_or(InvalidSlotError::new(id))
    }

    /// Removes the swapchain corresponding to `id`.
    ///
    /// # Safety
    ///
    /// - Unless the swapchain is being kept alive by other means, it must not be in use in any
    ///   pending command buffer, and if it is used in any command buffer that's in the executable
    ///   or recording state, that command buffer must never be executed.
    pub unsafe fn remove_swapchain(&self, id: Id<Swapchain>) -> Result<Ref<'_, SwapchainState>> {
        self.swapchains
            .remove(id.slot, self.pin())
            .map(Ref)
            .ok_or(InvalidSlotError::new(id))
    }

    /// Returns the buffer corresponding to `id`.
    #[inline]
    pub fn buffer(&self, id: Id<Buffer>) -> Result<Ref<'_, BufferState>> {
        self.buffers
            .get(id.slot, self.pin())
            .map(Ref)
            .ok_or(InvalidSlotError::new(id))
    }

    #[inline]
    pub(crate) unsafe fn buffer_unprotected(&self, id: Id<Buffer>) -> Result<&BufferState> {
        // SAFETY: Enforced by the caller.
        unsafe { self.buffers.get_unprotected(id.slot) }.ok_or(InvalidSlotError::new(id))
    }

    #[inline]
    pub(crate) unsafe fn buffer_unchecked_unprotected(&self, id: Id<Buffer>) -> &BufferState {
        #[cfg(debug_assertions)]
        if unsafe { self.buffers.get_unprotected(id.slot) }.is_none() {
            std::process::abort();
        }

        // SAFETY: Enforced by the caller.
        unsafe { self.buffers.index_unchecked_unprotected(id.index()) }
    }

    /// Returns the image corresponding to `id`.
    #[inline]
    pub fn image(&self, id: Id<Image>) -> Result<Ref<'_, ImageState>> {
        self.images
            .get(id.slot, self.pin())
            .map(Ref)
            .ok_or(InvalidSlotError::new(id))
    }

    #[inline]
    pub(crate) unsafe fn image_unprotected(&self, id: Id<Image>) -> Result<&ImageState> {
        // SAFETY: Enforced by the caller.
        unsafe { self.images.get_unprotected(id.slot) }.ok_or(InvalidSlotError::new(id))
    }

    #[inline]
    pub(crate) unsafe fn image_unchecked_unprotected(&self, id: Id<Image>) -> &ImageState {
        #[cfg(debug_assertions)]
        if unsafe { self.images.get_unprotected(id.slot) }.is_none() {
            std::process::abort();
        }

        // SAFETY: Enforced by the caller.
        unsafe { self.images.index_unchecked_unprotected(id.index()) }
    }

    /// Returns the swapchain corresponding to `id`.
    #[inline]
    pub fn swapchain(&self, id: Id<Swapchain>) -> Result<Ref<'_, SwapchainState>> {
        self.swapchains
            .get(id.slot, self.pin())
            .map(Ref)
            .ok_or(InvalidSlotError::new(id))
    }

    #[inline]
    pub(crate) unsafe fn swapchain_unprotected(
        &self,
        id: Id<Swapchain>,
    ) -> Result<&SwapchainState> {
        // SAFETY: Enforced by the caller.
        unsafe { self.swapchains.get_unprotected(id.slot) }.ok_or(InvalidSlotError::new(id))
    }

    #[inline]
    pub(crate) unsafe fn swapchain_unchecked_unprotected(
        &self,
        id: Id<Swapchain>,
    ) -> &SwapchainState {
        #[cfg(debug_assertions)]
        if unsafe { self.swapchains.get_unprotected(id.slot) }.is_none() {
            std::process::abort();
        }

        // SAFETY: Enforced by the caller.
        unsafe { self.swapchains.index_unchecked_unprotected(id.index()) }
    }

    /// Returns the [flight] corresponding to `id`.
    #[inline]
    pub fn flight(&self, id: Id<Flight>) -> Result<Ref<'_, Flight>> {
        self.flights
            .get(id.slot, self.pin())
            .map(Ref)
            .ok_or(InvalidSlotError::new(id))
    }

    #[inline]
    pub(crate) unsafe fn flight_unprotected(&self, id: Id<Flight>) -> Result<&Flight> {
        // SAFETY: Enforced by the caller.
        unsafe { self.flights.get_unprotected(id.slot) }.ok_or(InvalidSlotError::new(id))
    }

    #[inline]
    pub(crate) unsafe fn flight_unprotected_unchecked(&self, id: Id<Flight>) -> &Flight {
        // SAFETY: Enforced by the caller.
        unsafe { self.flights.index_unchecked_unprotected(id.slot.index()) }
    }

    #[inline]
    pub(crate) fn pin(&self) -> epoch::Guard<'_> {
        self.locals.get_or(|| self.global.register_local()).pin()
    }

    pub(crate) fn command_buffer_allocator(&self) -> &Arc<StandardCommandBufferAllocator> {
        &self.command_buffer_allocator
    }

    pub(crate) fn try_advance_global_and_collect(&self, guard: &epoch::Guard<'_>) {
        if guard.try_advance_global() {
            self.buffers.try_collect(guard);
            self.images.try_collect(guard);
            self.swapchains.try_collect(guard);
            self.flights.try_collect(guard);
        }
    }
}

impl Drop for Resources {
    fn drop(&mut self) {
        for (flight_id, flight) in &mut self.flights {
            let prev_frame_index = flight.previous_frame_index();
            let fence = flight.fences[prev_frame_index as usize].get_mut();

            if let Err(err) = fence.wait(None) {
                if err == VulkanError::DeviceLost {
                    break;
                }

                eprintln!(
                    "failed to wait for flight {flight_id:?} to finish rendering graceful shutdown \
                    impossible: {err}; aborting",
                );
                std::process::abort();
            }
        }

        // FIXME:
        let _ = unsafe { self.device().wait_idle() };

        let mut registered_devices = REGISTERED_DEVICES.lock();

        // This can't panic because there's no way to construct this type without the device's
        // address being inserted into the list.
        let index = registered_devices
            .iter()
            .position(|&addr| addr == Arc::as_ptr(self.device()) as usize)
            .unwrap();

        registered_devices.remove(index);
    }
}

unsafe impl DeviceOwned for Resources {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl BufferState {
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

impl BufferAccess {
    /// A `BufferAccess` that signifies the lack thereof, for instance because the resource was
    /// never accessed.
    pub const NONE: Self = BufferAccess {
        stage_mask: PipelineStages::empty(),
        access_mask: AccessFlags::empty(),
        queue_family_index: vk::QUEUE_FAMILY_IGNORED,
    };

    /// Creates a new `BufferAccess`.
    ///
    /// # Panics
    ///
    /// - Panics if `access_types` contains any access type that's not valid for buffers.
    #[inline]
    #[must_use]
    pub const fn new(access_types: &[AccessType], queue_family_index: u32) -> Self {
        let mut access = BufferAccess {
            stage_mask: PipelineStages::empty(),
            access_mask: AccessFlags::empty(),
            queue_family_index,
        };
        let mut i = 0;

        while i < access_types.len() {
            let access_type = access_types[i];

            assert!(access_type.is_valid_buffer_access_type());

            access.stage_mask = access.stage_mask.union(access_type.stage_mask());
            access.access_mask = access.access_mask.union(access_type.access_mask());
            i += 1;
        }

        access
    }

    pub(crate) const fn from_masks(
        stage_mask: PipelineStages,
        access_mask: AccessFlags,
        queue_family_index: u32,
    ) -> Self {
        BufferAccess {
            stage_mask,
            access_mask,
            queue_family_index,
        }
    }

    /// Returns the stage mask of this access.
    #[inline]
    #[must_use]
    pub const fn stage_mask(&self) -> PipelineStages {
        self.stage_mask
    }

    /// Returns the access mask of this access.
    #[inline]
    #[must_use]
    pub const fn access_mask(&self) -> AccessFlags {
        self.access_mask
    }

    /// Returns the queue family index of this access.
    #[inline]
    #[must_use]
    pub const fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }
}

impl ImageState {
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

impl ImageAccess {
    /// An `ImageAccess` that signifies the lack thereof, for instance because the resource was
    /// never accessed.
    pub const NONE: Self = ImageAccess {
        stage_mask: PipelineStages::empty(),
        access_mask: AccessFlags::empty(),
        image_layout: ImageLayout::Undefined,
        queue_family_index: vk::QUEUE_FAMILY_IGNORED,
    };

    /// Creates a new `ImageAccess`.
    ///
    /// # Panics
    ///
    /// - Panics if `access_types` contains any access type that's not valid for images.
    #[inline]
    #[must_use]
    pub const fn new(
        access_types: &[AccessType],
        layout_type: ImageLayoutType,
        queue_family_index: u32,
    ) -> Self {
        let mut access = ImageAccess {
            stage_mask: PipelineStages::empty(),
            access_mask: AccessFlags::empty(),
            image_layout: ImageLayout::Undefined,
            queue_family_index,
        };
        let mut i = 0;

        while i < access_types.len() {
            let access_type = access_types[i];

            assert!(access_type.is_valid_image_access_type());

            let image_layout = access_type.image_layout(layout_type);

            access.stage_mask = access.stage_mask.union(access_type.stage_mask());
            access.access_mask = access.access_mask.union(access_type.access_mask());
            access.image_layout = if matches!(access.image_layout, ImageLayout::Undefined)
                || access.image_layout as i32 == image_layout as i32
            {
                image_layout
            } else {
                ImageLayout::General
            };
            i += 1;
        }

        access
    }

    pub(crate) const fn from_masks(
        stage_mask: PipelineStages,
        access_mask: AccessFlags,
        image_layout: ImageLayout,
        queue_family_index: u32,
    ) -> Self {
        ImageAccess {
            stage_mask,
            access_mask,
            image_layout,
            queue_family_index,
        }
    }

    /// Returns the stage mask of this access.
    #[inline]
    #[must_use]
    pub const fn stage_mask(&self) -> PipelineStages {
        self.stage_mask
    }

    /// Returns the access mask of this access.
    #[inline]
    #[must_use]
    pub const fn access_mask(&self) -> AccessFlags {
        self.access_mask
    }

    /// Returns the image layout of this access.
    #[inline]
    #[must_use]
    pub const fn image_layout(&self) -> ImageLayout {
        self.image_layout
    }

    /// Returns the queue family index of this access.
    #[inline]
    #[must_use]
    pub const fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }
}

impl SwapchainState {
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

impl Flight {
    /// Returns the number of [frames] in this [flight].
    #[inline]
    #[must_use]
    pub fn frame_count(&self) -> u32 {
        self.frame_count.get()
    }

    /// Returns the current frame counter value. This always starts out at 0 and is monotonically
    /// increasing with each passing [frame].
    #[inline]
    #[must_use]
    pub fn current_frame(&self) -> u64 {
        self.current_frame.load(Ordering::Relaxed)
    }

    /// Returns the index of the current [frame] in [flight].
    #[inline]
    #[must_use]
    pub fn current_frame_index(&self) -> u32 {
        (self.current_frame() % NonZeroU64::from(self.frame_count)) as u32
    }

    fn previous_frame_index(&self) -> u32 {
        (self.current_frame().wrapping_sub(1) % NonZeroU64::from(self.frame_count)) as u32
    }

    pub(crate) fn current_fence(&self) -> &RwLock<Fence> {
        &self.fences[self.current_frame_index() as usize]
    }

    /// Waits for the oldest [frame] in [flight] to finish.
    #[inline]
    pub fn wait(&self, timeout: Option<Duration>) -> Result<(), VulkanError> {
        self.fences[self.current_frame_index() as usize]
            .read()
            .wait(timeout)
    }

    /// Waits for the given [frame] to finish. `frame` must have been previously obtained using
    /// [`current_frame`] on `self`.
    ///
    /// # Panics
    ///
    /// - Panics if `frame` is greater than the current frame.
    ///
    /// [`current_frame`]: Self::current_frame
    #[inline]
    pub fn wait_for_frame(&self, frame: u64, timeout: Option<Duration>) -> Result<(), VulkanError> {
        let current_frame = self.current_frame();

        assert!(frame <= current_frame);

        if current_frame - frame > u64::from(self.frame_count()) {
            return Ok(());
        }

        self.fences[(frame % NonZeroU64::from(self.frame_count)) as usize]
            .read()
            .wait(timeout)
    }

    /// Queues the destruction of the given `object` after the destruction of the command buffer(s)
    /// for the previous [frame] in [flight].
    ///
    /// # Panics
    ///
    /// - Panics if called from multiple threads at the same time.
    /// - Panics if the flight is currently being executed.
    #[inline]
    pub fn destroy_object(&self, object: Arc<impl Any + Send + Sync>) {
        let mut state = self.state.try_lock().unwrap();
        state.death_rows[self.previous_frame_index() as usize].push(object);
    }

    /// Queues the destruction of the given `objects` after the destruction of the command
    /// buffer(s) for the previous [frame] in [flight].
    ///
    /// # Panics
    ///
    /// - Panics if called from multiple threads at the same time.
    /// - Panics if the flight is currently being executed.
    #[inline]
    pub fn destroy_objects(&self, objects: impl IntoIterator<Item = Arc<impl Any + Send + Sync>>) {
        let mut state = self.state.try_lock().unwrap();
        state.death_rows[self.previous_frame_index() as usize]
            .extend(objects.into_iter().map(|object| object as _));
    }

    pub(crate) unsafe fn next_frame(&self) {
        self.current_frame.fetch_add(1, Ordering::Relaxed);
    }
}

/// Parameters to create a new [`Resources`] collection.
#[derive(Debug)]
pub struct ResourcesCreateInfo<'a> {
    /// The maximum number of [`Buffer`]s that the collection can hold at once.
    pub max_buffers: u32,

    /// The maximum number of [`Image`]s that the collection can hold at once.
    pub max_images: u32,

    /// The maximum number of [`Swapchain`]s that the collection can hold at once.
    pub max_swapchains: u32,

    /// The maximum number of [`Flight`]s that the collection can hold at once.
    pub max_flights: u32,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for ResourcesCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        ResourcesCreateInfo {
            max_buffers: 1 << 24,
            max_images: 1 << 24,
            max_swapchains: 1 << 8,
            max_flights: 1 << 8,
            _ne: crate::NE,
        }
    }
}

macro_rules! access_types {
    (
        $(
            $(#[$meta:meta])*
            $name:ident {
                stage_mask: $($stage_flag:ident)|+,
                access_mask: $($access_flag:ident)|+,
                image_layout: $image_layout:ident,
                valid_for: $($valid_for:ident)|+,
            }
        )*
    ) => {
        /// Specifies which type of access is performed on a resource.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum AccessType {
            $(
                $(#[$meta])*
                $name,
            )*
        }

        impl AccessType {
            /// Returns the stage mask of this type of access.
            #[inline]
            #[must_use]
            pub const fn stage_mask(self) -> PipelineStages {
                match self {
                    $(
                        Self::$name => PipelineStages::empty()
                            $(.union(PipelineStages::$stage_flag))+,
                    )*
                }
            }

            /// Returns the access mask of this type of access.
            #[inline]
            #[must_use]
            pub const fn access_mask(self) -> AccessFlags {
                match self {
                    $(
                        Self::$name => AccessFlags::empty()
                            $(.union(AccessFlags::$access_flag))+,
                    )*
                }
            }

            /// Returns the image layout for this type of access.
            #[inline]
            #[must_use]
            pub const fn image_layout(self, layout_type: ImageLayoutType) -> ImageLayout {
                if layout_type.is_general() {
                    return ImageLayout::General;
                }

                match self {
                    $(
                        Self::$name => ImageLayout::$image_layout,
                    )*
                }
            }

            const fn valid_for(self) -> u8 {
                match self {
                    $(
                        Self::$name => $($valid_for)|+,
                    )*
                }
            }
        }
    };
}

const BUFFER: u8 = 1 << 0;
const IMAGE: u8 = 1 << 1;

access_types! {
    IndirectCommandRead {
        stage_mask: DRAW_INDIRECT,
        access_mask: INDIRECT_COMMAND_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    IndexRead {
        stage_mask: INDEX_INPUT,
        access_mask: INDEX_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    VertexAttributeRead {
        stage_mask: VERTEX_ATTRIBUTE_INPUT,
        access_mask: VERTEX_ATTRIBUTE_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    VertexShaderUniformRead {
        stage_mask: VERTEX_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    VertexShaderSampledRead {
        stage_mask: VERTEX_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
        valid_for: BUFFER | IMAGE,
    }

    VertexShaderStorageRead {
        stage_mask: VERTEX_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
        valid_for: BUFFER | IMAGE,
    }

    VertexShaderStorageWrite {
        stage_mask: VERTEX_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
        valid_for: BUFFER | IMAGE,
    }

    VertexShaderAccelerationStructureRead {
        stage_mask: VERTEX_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    TessellationControlShaderUniformRead {
        stage_mask: TESSELLATION_CONTROL_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    TessellationControlShaderSampledRead {
        stage_mask: TESSELLATION_CONTROL_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
        valid_for: BUFFER | IMAGE,
    }

    TessellationControlShaderStorageRead {
        stage_mask: TESSELLATION_CONTROL_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
        valid_for: BUFFER | IMAGE,
    }

    TessellationControlShaderStorageWrite {
        stage_mask: TESSELLATION_CONTROL_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
        valid_for: BUFFER | IMAGE,
    }

    TessellationControlShaderAccelerationStructureRead {
        stage_mask: TESSELLATION_CONTROL_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    TessellationEvaluationShaderUniformRead {
        stage_mask: TESSELLATION_EVALUATION_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    TessellationEvaluationShaderSampledRead {
        stage_mask: TESSELLATION_EVALUATION_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
        valid_for: BUFFER | IMAGE,
    }

    TessellationEvaluationShaderStorageRead {
        stage_mask: TESSELLATION_EVALUATION_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
        valid_for: BUFFER | IMAGE,
    }

    TessellationEvaluationShaderStorageWrite {
        stage_mask: TESSELLATION_EVALUATION_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
        valid_for: BUFFER | IMAGE,
    }

    TessellationEvaluationShaderAccelerationStructureRead {
        stage_mask: TESSELLATION_EVALUATION_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    GeometryShaderUniformRead {
        stage_mask: GEOMETRY_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    GeometryShaderSampledRead {
        stage_mask: GEOMETRY_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
        valid_for: BUFFER | IMAGE,
    }

    GeometryShaderStorageRead {
        stage_mask: GEOMETRY_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
        valid_for: BUFFER | IMAGE,
    }

    GeometryShaderStorageWrite {
        stage_mask: GEOMETRY_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
        valid_for: BUFFER | IMAGE,
    }

    GeometryShaderAccelerationStructureRead {
        stage_mask: GEOMETRY_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    FragmentShaderUniformRead {
        stage_mask: FRAGMENT_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    FragmentShaderColorInputAttachmentRead {
        stage_mask: FRAGMENT_SHADER,
        access_mask: INPUT_ATTACHMENT_READ,
        image_layout: ShaderReadOnlyOptimal,
        valid_for: IMAGE,
    }

    FragmentShaderDepthStencilInputAttachmentRead {
        stage_mask: FRAGMENT_SHADER,
        access_mask: INPUT_ATTACHMENT_READ,
        image_layout: DepthStencilReadOnlyOptimal,
        valid_for: IMAGE,
    }

    FragmentShaderSampledRead {
        stage_mask: FRAGMENT_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
        valid_for: BUFFER | IMAGE,
    }

    FragmentShaderStorageRead {
        stage_mask: FRAGMENT_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
        valid_for: BUFFER | IMAGE,
    }

    FragmentShaderStorageWrite {
        stage_mask: FRAGMENT_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
        valid_for: BUFFER | IMAGE,
    }

    FragmentShaderAccelerationStructureRead {
        stage_mask: FRAGMENT_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    DepthStencilAttachmentRead {
        stage_mask: EARLY_FRAGMENT_TESTS | LATE_FRAGMENT_TESTS,
        access_mask: DEPTH_STENCIL_ATTACHMENT_READ,
        image_layout: DepthStencilReadOnlyOptimal,
        valid_for: IMAGE,
    }

    DepthStencilAttachmentWrite {
        stage_mask: EARLY_FRAGMENT_TESTS | LATE_FRAGMENT_TESTS,
        access_mask: DEPTH_STENCIL_ATTACHMENT_WRITE,
        image_layout: DepthStencilAttachmentOptimal,
        valid_for: IMAGE,
    }

    DepthAttachmentWriteStencilReadOnly {
        stage_mask: EARLY_FRAGMENT_TESTS | LATE_FRAGMENT_TESTS,
        access_mask: DEPTH_STENCIL_ATTACHMENT_READ | DEPTH_STENCIL_ATTACHMENT_WRITE,
        image_layout: DepthAttachmentStencilReadOnlyOptimal,
        valid_for: IMAGE,
    }

    DepthReadOnlyStencilAttachmentWrite {
        stage_mask: EARLY_FRAGMENT_TESTS | LATE_FRAGMENT_TESTS,
        access_mask: DEPTH_STENCIL_ATTACHMENT_READ | DEPTH_STENCIL_ATTACHMENT_WRITE,
        image_layout: DepthReadOnlyStencilAttachmentOptimal,
        valid_for: IMAGE,
    }

    ColorAttachmentRead {
        stage_mask: COLOR_ATTACHMENT_OUTPUT,
        access_mask: COLOR_ATTACHMENT_READ,
        image_layout: ColorAttachmentOptimal,
        valid_for: IMAGE,
    }

    ColorAttachmentWrite {
        stage_mask: COLOR_ATTACHMENT_OUTPUT,
        access_mask: COLOR_ATTACHMENT_WRITE,
        image_layout: ColorAttachmentOptimal,
        valid_for: IMAGE,
    }

    ComputeShaderUniformRead {
        stage_mask: COMPUTE_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    ComputeShaderSampledRead {
        stage_mask: COMPUTE_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
        valid_for: BUFFER | IMAGE,
    }

    ComputeShaderStorageRead {
        stage_mask: COMPUTE_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
        valid_for: BUFFER | IMAGE,
    }

    ComputeShaderStorageWrite {
        stage_mask: COMPUTE_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
        valid_for: BUFFER | IMAGE,
    }

    ComputeShaderAccelerationStructureRead {
        stage_mask: COMPUTE_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    CopyTransferRead {
        stage_mask: COPY,
        access_mask: TRANSFER_READ,
        image_layout: TransferSrcOptimal,
        valid_for: BUFFER | IMAGE,
    }

    CopyTransferWrite {
        stage_mask: COPY,
        access_mask: TRANSFER_WRITE,
        image_layout: TransferDstOptimal,
        valid_for: BUFFER | IMAGE,
    }

    BlitTransferRead {
        stage_mask: BLIT,
        access_mask: TRANSFER_READ,
        image_layout: TransferSrcOptimal,
        valid_for: IMAGE,
    }

    BlitTransferWrite {
        stage_mask: BLIT,
        access_mask: TRANSFER_WRITE,
        image_layout: TransferDstOptimal,
        valid_for: IMAGE,
    }

    ResolveTransferRead {
        stage_mask: RESOLVE,
        access_mask: TRANSFER_READ,
        image_layout: TransferSrcOptimal,
        valid_for: IMAGE,
    }

    ResolveTransferWrite {
        stage_mask: RESOLVE,
        access_mask: TRANSFER_WRITE,
        image_layout: TransferDstOptimal,
        valid_for: IMAGE,
    }

    ClearTransferWrite {
        stage_mask: CLEAR,
        access_mask: TRANSFER_WRITE,
        image_layout: TransferDstOptimal,
        valid_for: IMAGE,
    }

    AccelerationStructureCopyTransferRead {
        stage_mask: ACCELERATION_STRUCTURE_COPY,
        access_mask: TRANSFER_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    AccelerationStructureCopyTransferWrite {
        stage_mask: ACCELERATION_STRUCTURE_COPY,
        access_mask: TRANSFER_WRITE,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    // TODO:
    // VideoDecodeRead {
    //     stage_mask: VIDEO_DECODE,
    //     access_mask: VIDEO_DECODE_READ,
    //     image_layout: Undefined,
    //     valid_for: BUFFER,
    // }

    // TODO:
    // VideoDecodeWrite {
    //     stage_mask: VIDEO_DECODE,
    //     access_mask: VIDEO_DECODE_WRITE,
    //     image_layout: VideoDecodeDst,
    //     valid_for: IMAGE,
    // }

    // TODO:
    // VideoDecodeDpbRead {
    //     stage_mask: VIDEO_DECODE,
    //     access_mask: VIDEO_DECODE_READ,
    //     image_layout: VideoDecodeDpb,
    //     valid_for: IMAGE,
    // }

    // TODO:
    // VideoDecodeDpbWrite {
    //     stage_mask: VIDEO_DECODE,
    //     access_mask: VIDEO_DECODE_WRITE,
    //     image_layout: VideoDecodeDpb,
    //     valid_for: IMAGE,
    // }

    // TODO:
    // VideoEncodeRead {
    //     stage_mask: VIDEO_ENCODE,
    //     access_mask: VIDEO_ENCODE_READ,
    //     image_layout: VideoEncodeSrc,
    //     valid_for: IMAGE,
    // }

    // TODO:
    // VideoEncodeWrite {
    //     stage_mask: VIDEO_ENCODE,
    //     access_mask: VIDEO_ENCODE_WRITE,
    //     image_layout: Undefined,
    //     valid_for: BUFFER,
    // }

    // TODO:
    // VideoEncodeDpbRead {
    //     stage_mask: VIDEO_ENCODE,
    //     access_mask: VIDEO_ENCODE_READ,
    //     image_layout: VideoEncodeDpb,
    //     valid_for: IMAGE,
    // }

    // TODO:
    // VideoEncodeDpbWrite {
    //     stage_mask: VIDEO_ENCODE,
    //     access_mask: VIDEO_ENCODE_WRITE,
    //     image_layout: VideoEncodeDpb,
    //     valid_for: IMAGE,
    // }

    // TODO:
    // RayTracingShaderUniformRead {
    //     stage_mask: RAY_TRACING_SHADER,
    //     access_mask: UNIFORM_READ,
    //     image_layout: Undefined,
    //     valid_for: BUFFER,
    // }

    // TODO:
    // RayTracingShaderColorInputAttachmentRead {
    //     stage_mask: RAY_TRACING_SHADER,
    //     access_mask: INPUT_ATTACHMENT_READ,
    //     image_layout: ShaderReadOnlyOptimal,
    //     valid_for: IMAGE,
    // }

    // TODO:
    // RayTracingShaderDepthStencilInputAttachmentRead {
    //     stage_mask: RAY_TRACING_SHADER,
    //     access_mask: INPUT_ATTACHMENT_READ,
    //     image_layout: DepthStencilReadOnlyOptimal,
    //     valid_for: IMAGE,
    // }

    // TODO:
    // RayTracingShaderSampledRead {
    //     stage_mask: RAY_TRACING_SHADER,
    //     access_mask: SHADER_SAMPLED_READ,
    //     image_layout: ShaderReadOnlyOptimal,
    //     valid_for: BUFFER | IMAGE,
    // }

    // TODO:
    // RayTracingShaderStorageRead {
    //     stage_mask: RAY_TRACING_SHADER,
    //     access_mask: SHADER_STORAGE_READ,
    //     image_layout: General,
    //     valid_for: BUFFER | IMAGE,
    // }

    // TODO:
    RayTracingShaderStorageWrite {
        stage_mask: RAY_TRACING_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
        valid_for: BUFFER | IMAGE,
    }

    // TODO:
    // RayTracingShaderBindingTableRead {
    //     stage_mask: RAY_TRACING_SHADER,
    //     access_mask: SHADER_BINDING_TABLE_READ,
    //     image_layout: Undefined,
    //     valid_for: BUFFER,
    // }

    // TODO:
    // RayTracingShaderAccelerationStructureRead {
    //     stage_mask: RAY_TRACING_SHADER,
    //     access_mask: ACCELERATION_STRUCTURE_READ,
    //     image_layout: Undefined,
    //     valid_for: BUFFER,
    // }

    TaskShaderUniformRead {
        stage_mask: TASK_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    TaskShaderSampledRead {
        stage_mask: TASK_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
        valid_for: BUFFER | IMAGE,
    }

    TaskShaderStorageRead {
        stage_mask: TASK_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
        valid_for: BUFFER | IMAGE,
    }

    TaskShaderStorageWrite {
        stage_mask: TASK_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
        valid_for: BUFFER | IMAGE,
    }

    TaskShaderAccelerationStructureRead {
        stage_mask: TASK_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    MeshShaderUniformRead {
        stage_mask: MESH_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    MeshShaderSampledRead {
        stage_mask: MESH_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
        valid_for: BUFFER | IMAGE,
    }

    MeshShaderStorageRead {
        stage_mask: MESH_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
        valid_for: BUFFER | IMAGE,
    }

    MeshShaderStorageWrite {
        stage_mask: MESH_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
        valid_for: BUFFER | IMAGE,
    }

    MeshShaderAccelerationStructureRead {
        stage_mask: MESH_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    AccelerationStructureBuildIndirectCommandRead {
        stage_mask: ACCELERATION_STRUCTURE_BUILD,
        access_mask: INDIRECT_COMMAND_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    AccelerationStructureBuildShaderRead {
        stage_mask: ACCELERATION_STRUCTURE_BUILD,
        access_mask: SHADER_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    AccelerationStructureBuildAccelerationStructureRead {
        stage_mask: ACCELERATION_STRUCTURE_BUILD,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    AccelerationStructureBuildAccelerationStructureWrite {
        stage_mask: ACCELERATION_STRUCTURE_BUILD,
        access_mask: ACCELERATION_STRUCTURE_WRITE,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    AccelerationStructureCopyAccelerationStructureRead {
        stage_mask: ACCELERATION_STRUCTURE_COPY,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    AccelerationStructureCopyAccelerationStructureWrite {
        stage_mask: ACCELERATION_STRUCTURE_COPY,
        access_mask: ACCELERATION_STRUCTURE_WRITE,
        image_layout: Undefined,
        valid_for: BUFFER,
    }

    /// Only use this for prototyping or debugging please. 🔫 Please. 🔫
    General {
        stage_mask: ALL_COMMANDS,
        access_mask: MEMORY_READ | MEMORY_WRITE,
        image_layout: General,
        valid_for: BUFFER | IMAGE,
    }
}

impl AccessType {
    pub(crate) const fn is_valid_buffer_access_type(self) -> bool {
        self.valid_for() & BUFFER != 0
    }

    pub(crate) const fn is_valid_image_access_type(self) -> bool {
        self.valid_for() & IMAGE != 0
    }
}

/// Specifies which type of layout an image resource is accessed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ImageLayoutType {
    /// The image is accessed in an optimal layout. This is what you should be using most of the
    /// time.
    ///
    /// The optimal layout depends on the access type. For instance, for color attachment output,
    /// the only valid optimal layout is [`ColorAttachmentOptimal`]. For transfer sources, it's
    /// [`TransferSrcOptimal`], and so on. Some access types don't have an optimal layout for them.
    /// In such cases, using this option makes no difference, as the [general layout] will
    /// always be used.
    ///
    /// [`ColorAttachmentOptimal`]: ImageLayout::ColorAttachmentOptimal
    /// [`TransferSrcOptimal`]: ImageLayout::TransferSrcOptimal
    /// [general layout]: ImageLayout::General
    Optimal,

    /// The image is accessed in the [general layout]. This layout may be less efficient to access
    /// on some hardware than an optimal layout. However, you may still want to use it in certain
    /// cases if you want to minimize the number of layout transitions.
    ///
    /// [general layout]: ImageLayout::General
    General,
}

impl ImageLayoutType {
    /// Returns `true` if the layout type is `Optimal`.
    #[inline]
    #[must_use]
    pub const fn is_optimal(self) -> bool {
        matches!(self, ImageLayoutType::Optimal)
    }

    /// Returns `true` if the layout type is `General`.
    #[inline]
    #[must_use]
    pub const fn is_general(self) -> bool {
        matches!(self, ImageLayoutType::General)
    }
}

/// Specifies which type of host access is performed on a resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HostAccessType {
    /// The resource is read on the host.
    Read,

    /// The resource is written on the host.
    Write,
}

type Result<T = (), E = InvalidSlotError> = ::std::result::Result<T, E>;
