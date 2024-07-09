//! Synchronization state tracking of all resources.

use crate::{Id, InvalidSlotError, ObjectType, Ref};
use ash::vk;
use concurrent_slotmap::{epoch, SlotMap};
use parking_lot::{Mutex, MutexGuard};
use rangemap::RangeMap;
use smallvec::SmallVec;
use std::{
    any::Any,
    hash::Hash,
    iter::FusedIterator,
    mem,
    num::NonZeroU32,
    ops::Range,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
};
use thread_local::ThreadLocal;
use vulkano::{
    buffer::{AllocateBufferError, Buffer, BufferCreateInfo},
    device::{Device, DeviceOwned},
    image::{
        AllocateImageError, Image, ImageAspects, ImageCreateFlags, ImageCreateInfo, ImageLayout,
        ImageMemory, ImageSubresourceRange,
    },
    memory::allocator::{AllocationCreateInfo, DeviceLayout, MemoryAllocator},
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    sync::{
        fence::{Fence, FenceCreateFlags, FenceCreateInfo},
        semaphore::Semaphore,
        AccessFlags, PipelineStages,
    },
    DeviceSize, Validated, VulkanError,
};

static REGISTERED_DEVICES: Mutex<Vec<usize>> = Mutex::new(Vec::new());

const BUFFER_TAG: u32 = ObjectType::Buffer as u32;
const IMAGE_TAG: u32 = ObjectType::Image as u32;
const SWAPCHAIN_TAG: u32 = ObjectType::Swapchain as u32;
const FLIGHT_TAG: u32 = ObjectType::Flight as u32;

/// Tracks the synchronization state of all resources.
///
/// There can only exist one `Resources` collection per device, because there must only be one
/// source of truth in regards to the synchronization state of a resource. In a similar vein, each
/// resource in the collection must be unique.
// FIXME: Custom collector
// FIXME: Swapchain recreation
#[derive(Debug)]
pub struct Resources {
    memory_allocator: Arc<dyn MemoryAllocator>,

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
    // FIXME: This is terribly inefficient.
    last_accesses: Mutex<RangeMap<DeviceSize, BufferAccess>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferAccess {
    access_type: AccessType,
    queue_family_index: u32,
}

#[derive(Debug)]
pub struct ImageState {
    image: Arc<Image>,
    // FIXME: This is terribly inefficient.
    last_accesses: Mutex<RangeMap<DeviceSize, ImageAccess>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageAccess {
    access_type: AccessType,
    layout_type: ImageLayoutType,
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
    last_accesses: Mutex<RangeMap<DeviceSize, ImageAccess>>,
}

#[derive(Debug)]
pub(crate) struct SwapchainSemaphoreState {
    pub(crate) image_available_semaphore: Semaphore,
    pub(crate) tasks_complete_semaphore: Semaphore,
}

// FIXME: imported/exported fences
#[derive(Debug)]
pub struct Flight {
    frame_count: NonZeroU32,
    current_frame: AtomicU32,
    fences: SmallVec<[Fence; 3]>,
    pub(crate) state: Mutex<FlightState>,
}

#[derive(Debug)]
pub(crate) struct FlightState {
    pub(crate) swapchains: SmallVec<[Id<Swapchain>; 1]>,
    pub(crate) death_rows: SmallVec<[DeathRow; 3]>,
}

pub(crate) type DeathRow = Vec<Arc<dyn Any + Send + Sync>>;

impl Resources {
    /// Creates a new `Resources` collection.
    ///
    /// # Panics
    ///
    /// - Panics if `memory_allocator.device()` already has a `Resources` collection associated
    ///   with it.
    #[must_use]
    pub fn new(
        memory_allocator: Arc<dyn MemoryAllocator>,
        create_info: ResourcesCreateInfo,
    ) -> Self {
        let device = memory_allocator.device();
        let mut registered_devices = REGISTERED_DEVICES.lock();
        let device_addr = Arc::as_ptr(device) as usize;

        assert!(
            !registered_devices.contains(&device_addr),
            "the device already has a `Resources` collection associated with it",
        );

        registered_devices.push(device_addr);

        let global = epoch::GlobalHandle::new();

        Resources {
            memory_allocator,
            locals: ThreadLocal::new(),
            buffers: SlotMap::with_global(create_info.max_buffers, global.clone()),
            images: SlotMap::with_global(create_info.max_images, global.clone()),
            swapchains: SlotMap::with_global(create_info.max_swapchains, global.clone()),
            flights: SlotMap::with_global(create_info.max_flights, global.clone()),
            global,
        }
    }

    /// Returns the memory allocator that the collection was created with.
    #[inline]
    #[must_use]
    pub fn memory_allocator(&self) -> &Arc<dyn MemoryAllocator> {
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
        let frames_in_flight = self
            .flights
            .get(flight_id.slot, self.pin())
            .unwrap()
            .frame_count();

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
            })
            .collect::<Result<_, VulkanError>>()?;

        let flight = Flight {
            frame_count,
            current_frame: AtomicU32::new(0),
            fences,
            state: Mutex::new(FlightState {
                swapchains: SmallVec::new(),
                death_rows: (0..frame_count.get()).map(|_| Vec::new()).collect(),
            }),
        };

        let slot = self.flights.insert_with_tag(flight, FLIGHT_TAG, self.pin());

        Ok(Id::new(slot))
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
            last_accesses: Mutex::new(RangeMap::new()),
        };

        unsafe { state.set_access(0..state.buffer.size(), BufferAccess::NONE) };

        let slot = self.buffers.insert_with_tag(state, BUFFER_TAG, self.pin());

        Id::new(slot)
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
            last_accesses: Mutex::new(RangeMap::new()),
        };

        unsafe { state.set_access(state.image.subresource_range(), ImageAccess::NONE) };

        let slot = self.images.insert_with_tag(state, IMAGE_TAG, self.pin());

        Id::new(slot)
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

        let frames_in_flight = self
            .flights
            .get(flight_id.slot, self.pin())
            .unwrap()
            .frame_count();

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

        let frames_in_flight = self
            .flights
            .get(flight_id.slot, guard)
            .unwrap()
            .frame_count();

        let semaphores = (0..frames_in_flight)
            .map(|_| {
                Ok(SwapchainSemaphoreState {
                    // SAFETY: The parameters are valid.
                    image_available_semaphore: unsafe {
                        Semaphore::new_unchecked(self.device().clone(), Default::default())
                    }?,
                    // SAFETY: The parameters are valid.
                    tasks_complete_semaphore: unsafe {
                        Semaphore::new_unchecked(self.device().clone(), Default::default())
                    }?,
                })
            })
            .collect::<Result<_, VulkanError>>()?;

        let state = SwapchainState {
            swapchain,
            images: images.into(),
            semaphores,
            flight_id,
            current_image_index: AtomicU32::new(u32::MAX),
            last_accesses: Mutex::new(RangeMap::new()),
        };

        unsafe {
            state.set_access(
                ImageSubresourceRange {
                    aspects: ImageAspects::COLOR,
                    mip_levels: 0..1,
                    array_layers: 0..state.swapchain.image_array_layers(),
                },
                ImageAccess::NONE,
            );
        }

        let slot = self.swapchains.insert_with_tag(state, SWAPCHAIN_TAG, guard);
        let id = Id::new(slot);

        self.flights
            .get(flight_id.slot, guard)
            .unwrap()
            .state
            // FIXME:
            .lock()
            .swapchains
            .push(id);

        Ok(id)
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
        let state = self
            .swapchains
            .remove(id.slot, self.pin())
            .map(Ref)
            .ok_or(InvalidSlotError::new(id))?;
        let flight_id = state.flight_id;

        let flight = self.flights.get(flight_id.slot, self.pin()).unwrap();
        // FIXME:
        let swapchains = &mut flight.state.lock().swapchains;
        let index = swapchains.iter().position(|&x| x == id).unwrap();
        swapchains.remove(index);

        Ok(state)
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
    pub(crate) fn pin(&self) -> epoch::Guard<'_> {
        self.locals.get_or(|| self.global.register_local()).pin()
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
        self.memory_allocator.device()
    }
}

impl BufferState {
    /// Returns the buffer.
    #[inline]
    #[must_use]
    pub fn buffer(&self) -> &Arc<Buffer> {
        &self.buffer
    }

    /// Returns all last accesses that overlap the given `range` of the buffer.
    ///
    /// # Panics
    ///
    /// - Panics if `range` doesn't denote a valid range of the buffer.
    #[inline]
    pub fn accesses(&self, range: BufferRange) -> BufferAccesses<'_> {
        assert!(range.end <= self.buffer.size());
        assert!(!range.is_empty());

        BufferAccesses {
            inner: MutexGuard::leak(self.last_accesses.lock()).overlapping(range),
            // SAFETY: We locked the mutex above.
            _guard: unsafe { AccessesGuard::new(&self.last_accesses) },
        }
    }

    /// Sets the last access of the given `range` of the buffer.
    ///
    /// # Safety
    ///
    /// - `access` must constitute the correct access that was last performed on the `range` of the
    ///   buffer.
    ///
    /// # Panics
    ///
    /// - Panics if `range` is empty.
    #[inline]
    pub unsafe fn set_access(&self, range: BufferRange, access: BufferAccess) {
        self.last_accesses.lock().insert(range, access);
    }
}

impl BufferAccess {
    /// A `BufferAccess` that signifies the lack thereof, for instance because the resource was
    /// never accessed.
    pub const NONE: Self = BufferAccess::new(AccessType::None, vk::QUEUE_FAMILY_IGNORED);

    /// Creates a new `BufferAccess`.
    #[inline]
    pub const fn new(access_type: AccessType, queue_family_index: u32) -> Self {
        BufferAccess {
            access_type,
            queue_family_index,
        }
    }

    /// Returns the stage mask of this access.
    #[inline]
    pub const fn stage_mask(&self) -> PipelineStages {
        self.access_type.stage_mask()
    }

    /// Returns the access mask of this access.
    #[inline]
    pub const fn access_mask(&self) -> AccessFlags {
        self.access_type.access_mask()
    }

    /// Returns the queue family index of this access.
    #[inline]
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

    /// Returns all last accesses that overlap the given `subresource_range` of the image.
    ///
    /// # Panics
    ///
    /// - Panics if `subresource_range` doesn't denote a valid subresource range of the image.
    #[inline]
    pub fn accesses(&self, subresource_range: ImageSubresourceRange) -> ImageAccesses<'_> {
        let subresource_ranges = SubresourceRanges::from_image(&self.image, subresource_range);
        let map = MutexGuard::leak(self.last_accesses.lock());

        ImageAccesses {
            mip_levels: self.image.mip_levels(),
            array_layers: self.image.array_layers(),
            subresource_ranges,
            overlapping: map.overlapping(0..0),
            map,
            // SAFETY: We locked the mutex above.
            _guard: unsafe { AccessesGuard::new(&self.last_accesses) },
        }
    }

    /// Sets the last access of the given `subresource_range` of the image.
    ///
    /// # Safety
    ///
    /// - `access` must constitute the correct access that was last performed on the
    ///   `subresource_range` of the image.
    ///
    /// # Panics
    ///
    /// - Panics if `range` is empty.
    #[inline]
    pub unsafe fn set_access(&self, subresource_range: ImageSubresourceRange, access: ImageAccess) {
        let mut last_accesses = self.last_accesses.lock();

        for range in SubresourceRanges::from_image(&self.image, subresource_range) {
            last_accesses.insert(range, access);
        }
    }
}

impl ImageAccess {
    /// An `ImageAccess` that signifies the lack thereof, for instance because the resource was
    /// never accessed.
    pub const NONE: Self = ImageAccess::new(
        AccessType::None,
        ImageLayoutType::Optimal,
        vk::QUEUE_FAMILY_IGNORED,
    );

    /// Creates a new `ImageAccess`.
    #[inline]
    pub const fn new(
        access_type: AccessType,
        mut layout_type: ImageLayoutType,
        queue_family_index: u32,
    ) -> Self {
        // Make sure that entries in the tree always compare equal if the effective access is the
        // same, so that they can be combined for easier pipeline barrier batching.
        if matches!(access_type.image_layout(), ImageLayout::General) {
            layout_type = ImageLayoutType::Optimal;
        }

        // Presentation must be done in the optimal image layout.
        if matches!(access_type, AccessType::Present) {
            layout_type = ImageLayoutType::Optimal;
        }

        ImageAccess {
            access_type,
            layout_type,
            queue_family_index,
        }
    }

    /// Returns the stage mask of this access.
    #[inline]
    #[must_use]
    pub const fn stage_mask(&self) -> PipelineStages {
        self.access_type.stage_mask()
    }

    /// Returns the access mask of this access.
    #[inline]
    #[must_use]
    pub const fn access_mask(&self) -> AccessFlags {
        self.access_type.access_mask()
    }

    /// Returns the image layout of this access.
    #[inline]
    #[must_use]
    pub const fn image_layout(&self) -> ImageLayout {
        if self.layout_type.is_general() {
            ImageLayout::General
        } else {
            self.access_type.image_layout()
        }
    }

    /// Returns the queue family index of this access.
    #[inline]
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

    pub(crate) fn accesses(&self, subresource_range: ImageSubresourceRange) -> ImageAccesses<'_> {
        assert_eq!(subresource_range.aspects, ImageAspects::COLOR);

        let subresource_ranges =
            SubresourceRanges::new(subresource_range, 1, self.swapchain.image_array_layers());
        let map = MutexGuard::leak(self.last_accesses.lock());

        ImageAccesses {
            mip_levels: 1,
            array_layers: self.swapchain.image_array_layers(),
            subresource_ranges,
            overlapping: map.overlapping(0..0),
            map,
            // SAFETY: We locked the mutex above.
            _guard: unsafe { AccessesGuard::new(&self.last_accesses) },
        }
    }

    pub(crate) unsafe fn set_access(
        &self,
        subresource_range: ImageSubresourceRange,
        access: ImageAccess,
    ) {
        assert_eq!(subresource_range.aspects, ImageAspects::COLOR);

        let mut last_accesses = self.last_accesses.lock();

        for range in
            SubresourceRanges::new(subresource_range, 1, self.swapchain.image_array_layers())
        {
            last_accesses.insert(range, access);
        }
    }
}

impl Flight {
    /// Returns the number of [frames] in this [flight].
    #[inline]
    #[must_use]
    pub fn frame_count(&self) -> u32 {
        self.frame_count.get()
    }

    /// Returns the index of the current [frame] in [flight].
    #[inline]
    #[must_use]
    pub fn current_frame(&self) -> u32 {
        self.current_frame.load(Ordering::Relaxed) % self.frame_count
    }

    /// Returns the fence for the current [frame] in [flight].
    #[inline]
    #[must_use]
    pub fn current_fence(&self) -> &Fence {
        &self.fences[self.current_frame() as usize]
    }

    pub(crate) unsafe fn next_frame(&self) {
        self.current_frame.fetch_add(1, Ordering::Relaxed);
    }
}

/// Parameters to create a new [`Resources`] collection.
#[derive(Debug)]
pub struct ResourcesCreateInfo {
    /// The maximum number of [`Buffer`]s that the collection can hold at once.
    pub max_buffers: u32,

    /// The maximum number of [`Image`]s that the collection can hold at once.
    pub max_images: u32,

    /// The maximum number of [`Swapchain`]s that the collection can hold at once.
    pub max_swapchains: u32,

    /// The maximum number of [`Flight`]s that the collection can hold at once.
    pub max_flights: u32,

    pub _ne: vulkano::NonExhaustive,
}

impl Default for ResourcesCreateInfo {
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

/// A subresource of a buffer that should be accessed.
pub type BufferRange = Range<DeviceSize>;

/// An iterator over the last accesses of a buffer subresource.
///
/// This type is created by the [`accesses`] method on [`BufferState`].
///
/// [`accesses`]: BufferState::accesses
pub struct BufferAccesses<'a> {
    inner: rangemap::map::Overlapping<'a, DeviceSize, BufferAccess, Range<DeviceSize>>,
    _guard: AccessesGuard<'a, BufferAccess>,
}

impl<'a> Iterator for BufferAccesses<'a> {
    type Item = (BufferRange, &'a BufferAccess);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|(range, access)| (range.clone(), access))
    }
}

impl FusedIterator for BufferAccesses<'_> {}

/// An iterator over the last accesses of an image subresource.
///
/// This type is created by the [`accesses`] method on [`ImageState`].
///
/// [`accesses`]: ImageState::accesses
pub struct ImageAccesses<'a> {
    mip_levels: u32,
    array_layers: u32,
    subresource_ranges: SubresourceRanges,
    overlapping: rangemap::map::Overlapping<'a, DeviceSize, ImageAccess, Range<DeviceSize>>,
    map: &'a RangeMap<DeviceSize, ImageAccess>,
    _guard: AccessesGuard<'a, ImageAccess>,
}

impl<'a> Iterator for ImageAccesses<'a> {
    type Item = (ImageSubresourceRange, &'a ImageAccess);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some((range, access)) = self.overlapping.next() {
            let subresource_range =
                range_to_subresources(range.clone(), self.mip_levels, self.array_layers);

            Some((subresource_range, access))
        } else if let Some(range) = self.subresource_ranges.next() {
            self.overlapping = self.map.overlapping(range);

            self.next()
        } else {
            None
        }
    }
}

impl FusedIterator for ImageAccesses<'_> {}

struct AccessesGuard<'a, V> {
    mutex: &'a Mutex<RangeMap<DeviceSize, V>>,
}

impl<'a, V> AccessesGuard<'a, V> {
    unsafe fn new(mutex: &'a Mutex<RangeMap<DeviceSize, V>>) -> Self {
        AccessesGuard { mutex }
    }
}

impl<V> Drop for AccessesGuard<'_, V> {
    fn drop(&mut self) {
        // SAFETY: Enforced by the caller of `AccessesGuard::new`.
        unsafe { self.mutex.force_unlock() }
    }
}

const _: () = assert!(mem::size_of::<ImageAspects>() == mem::size_of::<vk::ImageAspectFlags>());

#[derive(Clone)]
struct SubresourceRanges {
    aspects: u32,
    mip_levels: Range<DeviceSize>,
    array_layers: Range<u32>,
    aspect_size: DeviceSize,
    mip_level_size: DeviceSize,
    aspect_offset: DeviceSize,
    mip_level_offset: DeviceSize,
    granularity: SubresourceRangeGranularity,
}

#[derive(Clone, Copy)]
enum SubresourceRangeGranularity {
    Aspect,
    MipLevel,
    ArrayLayer,
}

impl SubresourceRanges {
    fn from_image(image: &Image, mut subresource_range: ImageSubresourceRange) -> Self {
        assert!(image.format().aspects().contains(subresource_range.aspects));

        if image.flags().intersects(ImageCreateFlags::DISJOINT)
            && subresource_range.aspects.intersects(ImageAspects::COLOR)
        {
            subresource_range.aspects -= ImageAspects::COLOR;
            subresource_range.aspects |= match image.format().planes().len() {
                2 => ImageAspects::PLANE_0 | ImageAspects::PLANE_1,
                3 => ImageAspects::PLANE_0 | ImageAspects::PLANE_1 | ImageAspects::PLANE_2,
                _ => unreachable!(),
            };
        }

        SubresourceRanges::new(subresource_range, image.mip_levels(), image.array_layers())
    }

    fn new(
        subresource_range: ImageSubresourceRange,
        image_mip_levels: u32,
        image_array_layers: u32,
    ) -> Self {
        assert!(subresource_range.mip_levels.end <= image_mip_levels);
        assert!(subresource_range.array_layers.end <= image_array_layers);
        assert!(!subresource_range.mip_levels.is_empty());
        assert!(!subresource_range.array_layers.is_empty());

        let mip_level_size = DeviceSize::from(image_array_layers);
        let mip_levels = DeviceSize::from(subresource_range.mip_levels.start) * mip_level_size
            ..DeviceSize::from(subresource_range.mip_levels.end) * mip_level_size;
        let aspect_size = mip_level_size * DeviceSize::from(image_mip_levels);
        let aspect_offset = 0;
        let mip_level_offset = mip_levels.end - mip_level_size;

        let granularity = if subresource_range.array_layers != (0..image_array_layers) {
            SubresourceRangeGranularity::ArrayLayer
        } else if subresource_range.mip_levels != (0..image_mip_levels) {
            SubresourceRangeGranularity::MipLevel
        } else {
            SubresourceRangeGranularity::Aspect
        };

        SubresourceRanges {
            aspects: vk::ImageAspectFlags::from(subresource_range.aspects).as_raw(),
            mip_levels,
            array_layers: subresource_range.array_layers,
            aspect_size,
            mip_level_size,
            aspect_offset,
            mip_level_offset,
            granularity,
        }
    }

    fn skip_unset_aspects(&mut self) {
        let aspect_count = self.aspects.trailing_zeros();
        self.aspects >>= aspect_count;
        self.aspect_offset += DeviceSize::from(aspect_count) * self.aspect_size;
    }

    fn next_aspect(&mut self) {
        self.aspects >>= 1;
        self.aspect_offset += self.aspect_size;
    }
}

impl Iterator for SubresourceRanges {
    type Item = Range<DeviceSize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.aspects != 0 {
            match self.granularity {
                SubresourceRangeGranularity::Aspect => {
                    self.skip_unset_aspects();

                    let aspect_count = DeviceSize::from(self.aspects.trailing_ones());
                    let start = self.aspect_offset;
                    let end = self.aspect_offset + aspect_count * self.aspect_size;

                    self.aspects >>= aspect_count;
                    self.aspect_offset += aspect_count * self.aspect_size;

                    Some(Range { start, end })
                }
                SubresourceRangeGranularity::MipLevel => {
                    self.skip_unset_aspects();

                    let start = self.aspect_offset + self.mip_levels.start;
                    let end = self.aspect_offset + self.mip_levels.end;

                    self.next_aspect();

                    Some(Range { start, end })
                }
                SubresourceRangeGranularity::ArrayLayer => {
                    self.mip_level_offset += self.mip_level_size;

                    if self.mip_level_offset == self.mip_levels.end {
                        self.mip_level_offset = self.mip_levels.start;
                        self.skip_unset_aspects();
                    }

                    let offset = self.aspect_offset + self.mip_level_offset;
                    let start = offset + DeviceSize::from(self.array_layers.start);
                    let end = offset + DeviceSize::from(self.array_layers.end);

                    if self.mip_level_offset == self.mip_levels.end - self.mip_level_size {
                        self.next_aspect();
                    }

                    Some(Range { start, end })
                }
            }
        } else {
            None
        }
    }
}

fn range_to_subresources(
    mut range: Range<DeviceSize>,
    image_mip_levels: u32,
    image_array_layers: u32,
) -> ImageSubresourceRange {
    debug_assert!(!range.is_empty());

    let aspect_size = DeviceSize::from(image_mip_levels) * DeviceSize::from(image_array_layers);
    let mip_level_size = DeviceSize::from(image_array_layers);

    if range.end - range.start > aspect_size {
        debug_assert!(range.start % aspect_size == 0);
        debug_assert!(range.end % aspect_size == 0);

        let aspect_start = (range.start / aspect_size) as u32;
        let aspect_end = (range.end / aspect_size) as u32;
        let aspects = u32::MAX >> (u32::BITS - (aspect_end - aspect_start)) << aspect_start;

        ImageSubresourceRange {
            aspects: vk::ImageAspectFlags::from_raw(aspects).into(),
            mip_levels: 0..image_mip_levels,
            array_layers: 0..image_array_layers,
        }
    } else {
        let aspect_index = (range.start / aspect_size) as u32;
        range.start %= aspect_size;
        range.end %= aspect_size;

        // Wraparound
        if range.end == 0 {
            range.end = aspect_size;
        }

        if range.end - range.start > mip_level_size {
            debug_assert!(range.start % mip_level_size == 0);
            debug_assert!(range.end % mip_level_size == 0);

            let mip_level_start = (range.start / mip_level_size) as u32;
            let mip_level_end = (range.end / mip_level_size) as u32;

            ImageSubresourceRange {
                aspects: vk::ImageAspectFlags::from_raw(1 << aspect_index).into(),
                mip_levels: mip_level_start..mip_level_end,
                array_layers: 0..image_array_layers,
            }
        } else {
            let mip_level = (range.start / mip_level_size) as u32;
            range.start %= mip_level_size;
            range.end %= mip_level_size;

            // Wraparound
            if range.end == 0 {
                range.end = mip_level_size;
            }

            let array_layer_start = range.start as u32;
            let array_layer_end = range.end as u32;

            ImageSubresourceRange {
                aspects: vk::ImageAspectFlags::from_raw(1 << aspect_index).into(),
                mip_levels: mip_level..mip_level + 1,
                array_layers: array_layer_start..array_layer_end,
            }
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
            }
        )*
    ) => {
        /// Specifies which type of access is performed on a subresource.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum AccessType {
            None,
            $(
                $(#[$meta])*
                $name,
            )*
            Present,
        }

        impl AccessType {
            /// Returns the stage mask of this type of access.
            #[inline]
            #[must_use]
            pub const fn stage_mask(self) -> PipelineStages {
                match self {
                    Self::None => PipelineStages::empty(),
                    $(
                        Self::$name => PipelineStages::empty()
                            $(.union(PipelineStages::$stage_flag))+,
                    )*
                    Self::Present => PipelineStages::empty(),
                }
            }

            /// Returns the access mask of this type of access.
            #[inline]
            #[must_use]
            pub const fn access_mask(self) -> AccessFlags {
                match self {
                    Self::None => AccessFlags::empty(),
                    $(
                        Self::$name => AccessFlags::empty()
                            $(.union(AccessFlags::$access_flag))+,
                    )*
                    Self::Present => AccessFlags::empty(),
                }
            }

            /// Returns the optimal image layout for this type of access, if any.
            #[inline]
            #[must_use]
            pub const fn image_layout(self) -> ImageLayout {
                match self {
                    Self::None => ImageLayout::Undefined,
                    $(
                        Self::$name => ImageLayout::$image_layout,
                    )*
                    Self::Present => ImageLayout::PresentSrc,
                }
            }
        }
    };
}

access_types! {
    IndirectCommandRead {
        stage_mask: DRAW_INDIRECT,
        access_mask: INDIRECT_COMMAND_READ,
        image_layout: Undefined,
    }

    IndexRead {
        stage_mask: INDEX_INPUT,
        access_mask: INDEX_READ,
        image_layout: Undefined,
    }

    VertexAttributeRead {
        stage_mask: VERTEX_ATTRIBUTE_INPUT,
        access_mask: VERTEX_ATTRIBUTE_READ,
        image_layout: Undefined,
    }

    VertexShaderUniformRead {
        stage_mask: VERTEX_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
    }

    VertexShaderSampledRead {
        stage_mask: VERTEX_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
    }

    VertexShaderStorageRead {
        stage_mask: VERTEX_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
    }

    VertexShaderStorageWrite {
        stage_mask: VERTEX_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
    }

    VertexShaderAccelerationStructureRead {
        stage_mask: VERTEX_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    TessellationControlShaderUniformRead {
        stage_mask: TESSELLATION_CONTROL_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
    }

    TessellationControlShaderSampledRead {
        stage_mask: TESSELLATION_CONTROL_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
    }

    TessellationControlShaderStorageRead {
        stage_mask: TESSELLATION_CONTROL_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
    }

    TessellationControlShaderStorageWrite {
        stage_mask: TESSELLATION_CONTROL_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
    }

    TessellationControlShaderAccelerationStructureRead {
        stage_mask: TESSELLATION_CONTROL_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    TessellationEvaluationShaderUniformRead {
        stage_mask: TESSELLATION_EVALUATION_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
    }

    TessellationEvaluationShaderSampledRead {
        stage_mask: TESSELLATION_EVALUATION_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
    }

    TessellationEvaluationShaderStorageRead {
        stage_mask: TESSELLATION_EVALUATION_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
    }

    TessellationEvaluationShaderStorageWrite {
        stage_mask: TESSELLATION_EVALUATION_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
    }

    TessellationEvaluationShaderAccelerationStructureRead {
        stage_mask: TESSELLATION_EVALUATION_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    GeometryShaderUniformRead {
        stage_mask: GEOMETRY_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
    }

    GeometryShaderSampledRead {
        stage_mask: GEOMETRY_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
    }

    GeometryShaderStorageRead {
        stage_mask: GEOMETRY_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
    }

    GeometryShaderStorageWrite {
        stage_mask: GEOMETRY_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
    }

    GeometryShaderAccelerationStructureRead {
        stage_mask: GEOMETRY_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    FragmentShaderUniformRead {
        stage_mask: FRAGMENT_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
    }

    FragmentShaderColorInputAttachmentRead {
        stage_mask: FRAGMENT_SHADER,
        access_mask: INPUT_ATTACHMENT_READ,
        image_layout: ShaderReadOnlyOptimal,
    }

    FragmentShaderDepthStencilInputAttachmentRead {
        stage_mask: FRAGMENT_SHADER,
        access_mask: INPUT_ATTACHMENT_READ,
        image_layout: DepthStencilReadOnlyOptimal,
    }

    FragmentShaderSampledRead {
        stage_mask: FRAGMENT_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
    }

    FragmentShaderStorageRead {
        stage_mask: FRAGMENT_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
    }

    FragmentShaderStorageWrite {
        stage_mask: FRAGMENT_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
    }

    FragmentShaderAccelerationStructureRead {
        stage_mask: FRAGMENT_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    DepthStencilAttachmentRead {
        stage_mask: EARLY_FRAGMENT_TESTS | LATE_FRAGMENT_TESTS,
        access_mask: DEPTH_STENCIL_ATTACHMENT_READ,
        image_layout: DepthStencilReadOnlyOptimal,
    }

    DepthStencilAttachmentWrite {
        stage_mask: EARLY_FRAGMENT_TESTS | LATE_FRAGMENT_TESTS,
        access_mask: DEPTH_STENCIL_ATTACHMENT_WRITE,
        image_layout: DepthStencilAttachmentOptimal,
    }

    DepthAttachmentWriteStencilReadOnly {
        stage_mask: EARLY_FRAGMENT_TESTS | LATE_FRAGMENT_TESTS,
        access_mask: DEPTH_STENCIL_ATTACHMENT_READ | DEPTH_STENCIL_ATTACHMENT_WRITE,
        image_layout: DepthAttachmentStencilReadOnlyOptimal,
    }

    DepthReadOnlyStencilAttachmentWrite {
        stage_mask: EARLY_FRAGMENT_TESTS | LATE_FRAGMENT_TESTS,
        access_mask: DEPTH_STENCIL_ATTACHMENT_READ | DEPTH_STENCIL_ATTACHMENT_WRITE,
        image_layout: DepthReadOnlyStencilAttachmentOptimal,
    }

    ColorAttachmentRead {
        stage_mask: COLOR_ATTACHMENT_OUTPUT,
        access_mask: COLOR_ATTACHMENT_READ,
        image_layout: ColorAttachmentOptimal,
    }

    ColorAttachmentWrite {
        stage_mask: COLOR_ATTACHMENT_OUTPUT,
        access_mask: COLOR_ATTACHMENT_WRITE,
        image_layout: ColorAttachmentOptimal,
    }

    ColorAttachmentReadWrite {
        stage_mask: COLOR_ATTACHMENT_OUTPUT,
        access_mask: COLOR_ATTACHMENT_READ | COLOR_ATTACHMENT_WRITE,
        image_layout: ColorAttachmentOptimal,
    }

    ComputeShaderUniformRead {
        stage_mask: COMPUTE_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
    }

    ComputeShaderSampledRead {
        stage_mask: COMPUTE_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
    }

    ComputeShaderStorageRead {
        stage_mask: COMPUTE_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
    }

    ComputeShaderStorageWrite {
        stage_mask: COMPUTE_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
    }

    ComputeShaderAccelerationStructureRead {
        stage_mask: COMPUTE_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    HostRead {
        stage_mask: HOST,
        access_mask: HOST_READ,
        image_layout: General,
    }

    HostWrite {
        stage_mask: HOST,
        access_mask: HOST_WRITE,
        image_layout: General,
    }

    CopyTransferRead {
        stage_mask: COPY,
        access_mask: TRANSFER_READ,
        image_layout: TransferSrcOptimal,
    }

    CopyTransferWrite {
        stage_mask: COPY,
        access_mask: TRANSFER_WRITE,
        image_layout: TransferDstOptimal,
    }

    BlitTransferRead {
        stage_mask: BLIT,
        access_mask: TRANSFER_READ,
        image_layout: TransferSrcOptimal,
    }

    BlitTransferWrite {
        stage_mask: BLIT,
        access_mask: TRANSFER_WRITE,
        image_layout: TransferDstOptimal,
    }

    ResolveTransferRead {
        stage_mask: RESOLVE,
        access_mask: TRANSFER_READ,
        image_layout: TransferSrcOptimal,
    }

    ResolveTransferWrite {
        stage_mask: RESOLVE,
        access_mask: TRANSFER_WRITE,
        image_layout: TransferDstOptimal,
    }

    ClearTransferWrite {
        stage_mask: CLEAR,
        access_mask: TRANSFER_WRITE,
        image_layout: TransferDstOptimal,
    }

    AccelerationStructureCopyTransferRead {
        stage_mask: ACCELERATION_STRUCTURE_COPY,
        access_mask: TRANSFER_READ,
        image_layout: Undefined,
    }

    AccelerationStructureCopyTransferWrite {
        stage_mask: ACCELERATION_STRUCTURE_COPY,
        access_mask: TRANSFER_WRITE,
        image_layout: Undefined,
    }

    // TODO:
    // VideoDecodeRead {
    //     stage_mask: VIDEO_DECODE,
    //     access_mask: VIDEO_DECODE_READ,
    //     image_layout: Undefined,
    // }

    // TODO:
    // VideoDecodeWrite {
    //     stage_mask: VIDEO_DECODE,
    //     access_mask: VIDEO_DECODE_WRITE,
    //     image_layout: VideoDecodeDst,
    // }

    // TODO:
    // VideoDecodeDpbRead {
    //     stage_mask: VIDEO_DECODE,
    //     access_mask: VIDEO_DECODE_READ,
    //     image_layout: VideoDecodeDpb,
    // }

    // TODO:
    // VideoDecodeDpbWrite {
    //     stage_mask: VIDEO_DECODE,
    //     access_mask: VIDEO_DECODE_WRITE,
    //     image_layout: VideoDecodeDpb,
    // }

    // TODO:
    // VideoEncodeRead {
    //     stage_mask: VIDEO_ENCODE,
    //     access_mask: VIDEO_ENCODE_READ,
    //     image_layout: VideoEncodeSrc,
    // }

    // TODO:
    // VideoEncodeWrite {
    //     stage_mask: VIDEO_ENCODE,
    //     access_mask: VIDEO_ENCODE_WRITE,
    //     image_layout: Undefined,
    // }

    // TODO:
    // VideoEncodeDpbRead {
    //     stage_mask: VIDEO_ENCODE,
    //     access_mask: VIDEO_ENCODE_READ,
    //     image_layout: VideoEncodeDpb,
    // }

    // TODO:
    // VideoEncodeDpbWrite {
    //     stage_mask: VIDEO_ENCODE,
    //     access_mask: VIDEO_ENCODE_WRITE,
    //     image_layout: VideoEncodeDpb,
    // }

    // TODO:
    // RayTracingShaderUniformRead {
    //     stage_mask: RAY_TRACING_SHADER,
    //     access_mask: UNIFORM_READ,
    //     image_layout: Undefined,
    // }

    // TODO:
    // RayTracingShaderColorInputAttachmentRead {
    //     stage_mask: RAY_TRACING_SHADER,
    //     access_mask: INPUT_ATTACHMENT_READ,
    //     image_layout: ShaderReadOnlyOptimal,
    // }

    // TODO:
    // RayTracingShaderDepthStencilInputAttachmentRead {
    //     stage_mask: RAY_TRACING_SHADER,
    //     access_mask: INPUT_ATTACHMENT_READ,
    //     image_layout: DepthStencilReadOnlyOptimal,
    // }

    // TODO:
    // RayTracingShaderSampledRead {
    //     stage_mask: RAY_TRACING_SHADER,
    //     access_mask: SHADER_SAMPLED_READ,
    //     image_layout: ShaderReadOnlyOptimal,
    // }

    // TODO:
    // RayTracingShaderStorageRead {
    //     stage_mask: RAY_TRACING_SHADER,
    //     access_mask: SHADER_STORAGE_READ,
    //     image_layout: General,
    // }

    // TODO:
    // RayTracingShaderStorageWrite {
    //     stage_mask: RAY_TRACING_SHADER,
    //     access_mask: SHADER_STORAGE_WRITE,
    //     image_layout: General,
    // }

    // TODO:
    // RayTracingShaderBindingTableRead {
    //     stage_mask: RAY_TRACING_SHADER,
    //     access_mask: SHADER_BINDING_TABLE_READ,
    //     image_layout: Undefined,
    // }

    // TODO:
    // RayTracingShaderAccelerationStructureRead {
    //     stage_mask: RAY_TRACING_SHADER,
    //     access_mask: ACCELERATION_STRUCTURE_READ,
    //     image_layout: Undefined,
    // }

    TaskShaderUniformRead {
        stage_mask: TASK_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
    }

    TaskShaderSampledRead {
        stage_mask: TASK_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
    }

    TaskShaderStorageRead {
        stage_mask: TASK_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
    }

    TaskShaderStorageWrite {
        stage_mask: TASK_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
    }

    TaskShaderAccelerationStructureRead {
        stage_mask: TASK_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    MeshShaderUniformRead {
        stage_mask: MESH_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
    }

    MeshShaderSampledRead {
        stage_mask: MESH_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
    }

    MeshShaderStorageRead {
        stage_mask: MESH_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
    }

    MeshShaderStorageWrite {
        stage_mask: MESH_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
    }

    MeshShaderAccelerationStructureRead {
        stage_mask: MESH_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    AccelerationStructureBuildShaderRead {
        stage_mask: ACCELERATION_STRUCTURE_BUILD,
        access_mask: SHADER_READ,
        image_layout: Undefined,
    }

    AccelerationStructureBuildAccelerationStructureRead {
        stage_mask: ACCELERATION_STRUCTURE_BUILD,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    AccelerationStructureBuildAccelerationStructureWrite {
        stage_mask: ACCELERATION_STRUCTURE_BUILD,
        access_mask: ACCELERATION_STRUCTURE_WRITE,
        image_layout: Undefined,
    }

    AccelerationStructureCopyAccelerationStructureRead {
        stage_mask: ACCELERATION_STRUCTURE_COPY,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    AccelerationStructureCopyAccelerationStructureWrite {
        stage_mask: ACCELERATION_STRUCTURE_COPY,
        access_mask: ACCELERATION_STRUCTURE_WRITE,
        image_layout: Undefined,
    }

    /// Only use this for prototyping or debugging please. 🔫 Please. 🔫
    General {
        stage_mask: ALL_COMMANDS,
        access_mask: MEMORY_READ | MEMORY_WRITE,
        image_layout: General,
    }
}

/// Specifies which type of layout an image subresource is accessed in.
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

type Result<T = (), E = InvalidSlotError> = ::std::result::Result<T, E>;

#[allow(clippy::erasing_op, clippy::identity_op)]
#[cfg(test)]
mod tests {
    use super::*;
    use vulkano::image::ImageAspects;

    #[test]
    fn subresource_ranges_aspect_granularity() {
        let mut iter = SubresourceRanges::new(
            ImageSubresourceRange {
                aspects: ImageAspects::COLOR,
                mip_levels: 0..4,
                array_layers: 0..8,
            },
            4,
            8,
        );

        assert_eq!(iter.next(), Some(0 * 32..1 * 32));
        assert_eq!(iter.next(), None);

        let mut iter = SubresourceRanges::new(
            ImageSubresourceRange {
                aspects: ImageAspects::DEPTH | ImageAspects::STENCIL,
                mip_levels: 0..2,
                array_layers: 0..12,
            },
            2,
            12,
        );

        assert_eq!(iter.next(), Some(1 * 24..3 * 24));
        assert_eq!(iter.next(), None);

        let mut iter = SubresourceRanges::new(
            ImageSubresourceRange {
                aspects: ImageAspects::COLOR | ImageAspects::METADATA | ImageAspects::PLANE_0,
                mip_levels: 0..5,
                array_layers: 0..10,
            },
            5,
            10,
        );

        assert_eq!(iter.next(), Some(0 * 50..1 * 50));
        assert_eq!(iter.next(), Some(3 * 50..5 * 50));
        assert_eq!(iter.next(), None);

        let mut iter = SubresourceRanges::new(
            ImageSubresourceRange {
                aspects: ImageAspects::COLOR
                    | ImageAspects::DEPTH
                    | ImageAspects::STENCIL
                    | ImageAspects::PLANE_0
                    | ImageAspects::PLANE_2
                    | ImageAspects::MEMORY_PLANE_2
                    | ImageAspects::MEMORY_PLANE_3,
                mip_levels: 0..3,
                array_layers: 0..20,
            },
            3,
            20,
        );

        assert_eq!(iter.next(), Some(0 * 60..3 * 60));
        assert_eq!(iter.next(), Some(4 * 60..5 * 60));
        assert_eq!(iter.next(), Some(6 * 60..7 * 60));
        assert_eq!(iter.next(), Some(9 * 60..11 * 60));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn subresource_ranges_mip_level_granularity() {
        let mut iter = SubresourceRanges::new(
            ImageSubresourceRange {
                aspects: ImageAspects::DEPTH,
                mip_levels: 1..3,
                array_layers: 0..8,
            },
            5,
            8,
        );

        assert_eq!(iter.next(), Some(1 * 40 + 1 * 8..1 * 40 + 3 * 8));
        assert_eq!(iter.next(), None);

        let mut iter = SubresourceRanges::new(
            ImageSubresourceRange {
                aspects: ImageAspects::PLANE_0 | ImageAspects::PLANE_1 | ImageAspects::PLANE_2,
                mip_levels: 1..3,
                array_layers: 0..12,
            },
            3,
            12,
        );

        assert_eq!(iter.next(), Some(4 * 36 + 1 * 12..4 * 36 + 3 * 12));
        assert_eq!(iter.next(), Some(5 * 36 + 1 * 12..5 * 36 + 3 * 12));
        assert_eq!(iter.next(), Some(6 * 36 + 1 * 12..6 * 36 + 3 * 12));
        assert_eq!(iter.next(), None);

        let mut iter = SubresourceRanges::new(
            ImageSubresourceRange {
                aspects: ImageAspects::DEPTH
                    | ImageAspects::STENCIL
                    | ImageAspects::PLANE_0
                    | ImageAspects::PLANE_1
                    | ImageAspects::PLANE_2,
                mip_levels: 1..3,
                array_layers: 0..10,
            },
            4,
            10,
        );

        dbg!(iter.clone().collect::<Vec<_>>());

        assert_eq!(iter.next(), Some(1 * 40 + 1 * 10..1 * 40 + 3 * 10));
        assert_eq!(iter.next(), Some(2 * 40 + 1 * 10..2 * 40 + 3 * 10));
        assert_eq!(iter.next(), Some(4 * 40 + 1 * 10..4 * 40 + 3 * 10));
        assert_eq!(iter.next(), Some(5 * 40 + 1 * 10..5 * 40 + 3 * 10));
        assert_eq!(iter.next(), Some(6 * 40 + 1 * 10..6 * 40 + 3 * 10));
        assert_eq!(iter.next(), None);

        let mut iter = SubresourceRanges::new(
            ImageSubresourceRange {
                aspects: ImageAspects::METADATA
                    | ImageAspects::PLANE_2
                    | ImageAspects::MEMORY_PLANE_1,
                mip_levels: 2..4,
                array_layers: 0..6,
            },
            4,
            6,
        );

        assert_eq!(iter.next(), Some(3 * 24 + 2 * 6..3 * 24 + 4 * 6));
        assert_eq!(iter.next(), Some(6 * 24 + 2 * 6..6 * 24 + 4 * 6));
        assert_eq!(iter.next(), Some(8 * 24 + 2 * 6..8 * 24 + 4 * 6));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn subresource_ranges_array_layer_granularity() {
        let mut iter = SubresourceRanges::new(
            ImageSubresourceRange {
                aspects: ImageAspects::STENCIL,
                mip_levels: 0..4,
                array_layers: 2..9,
            },
            4,
            10,
        );

        assert_eq!(iter.next(), Some(2 * 40 + 0 * 10 + 2..2 * 40 + 0 * 10 + 9));
        assert_eq!(iter.next(), Some(2 * 40 + 1 * 10 + 2..2 * 40 + 1 * 10 + 9));
        assert_eq!(iter.next(), Some(2 * 40 + 2 * 10 + 2..2 * 40 + 2 * 10 + 9));
        assert_eq!(iter.next(), Some(2 * 40 + 3 * 10 + 2..2 * 40 + 3 * 10 + 9));
        assert_eq!(iter.next(), None);

        let mut iter = SubresourceRanges::new(
            ImageSubresourceRange {
                aspects: ImageAspects::COLOR | ImageAspects::METADATA,
                mip_levels: 1..3,
                array_layers: 3..8,
            },
            3,
            8,
        );

        assert_eq!(iter.next(), Some(0 * 24 + 1 * 8 + 3..0 * 24 + 1 * 8 + 8));
        assert_eq!(iter.next(), Some(0 * 24 + 2 * 8 + 3..0 * 24 + 2 * 8 + 8));
        assert_eq!(iter.next(), Some(3 * 24 + 1 * 8 + 3..3 * 24 + 1 * 8 + 8));
        assert_eq!(iter.next(), Some(3 * 24 + 2 * 8 + 3..3 * 24 + 2 * 8 + 8));
        assert_eq!(iter.next(), None);

        let mut iter = SubresourceRanges::new(
            ImageSubresourceRange {
                aspects: ImageAspects::DEPTH | ImageAspects::PLANE_0 | ImageAspects::PLANE_1,
                mip_levels: 1..3,
                array_layers: 2..4,
            },
            5,
            6,
        );

        assert_eq!(iter.next(), Some(1 * 30 + 1 * 6 + 2..1 * 30 + 1 * 6 + 4));
        assert_eq!(iter.next(), Some(1 * 30 + 2 * 6 + 2..1 * 30 + 2 * 6 + 4));
        assert_eq!(iter.next(), Some(4 * 30 + 1 * 6 + 2..4 * 30 + 1 * 6 + 4));
        assert_eq!(iter.next(), Some(4 * 30 + 2 * 6 + 2..4 * 30 + 2 * 6 + 4));
        assert_eq!(iter.next(), Some(5 * 30 + 1 * 6 + 2..5 * 30 + 1 * 6 + 4));
        assert_eq!(iter.next(), Some(5 * 30 + 2 * 6 + 2..5 * 30 + 2 * 6 + 4));
        assert_eq!(iter.next(), None);

        let mut iter = SubresourceRanges::new(
            ImageSubresourceRange {
                aspects: ImageAspects::PLANE_2
                    | ImageAspects::MEMORY_PLANE_0
                    | ImageAspects::MEMORY_PLANE_1
                    | ImageAspects::MEMORY_PLANE_2,
                mip_levels: 5..6,
                array_layers: 0..3,
            },
            8,
            4,
        );

        assert_eq!(iter.next(), Some(6 * 32 + 5 * 4 + 0..6 * 32 + 5 * 4 + 3));
        assert_eq!(iter.next(), Some(7 * 32 + 5 * 4 + 0..7 * 32 + 5 * 4 + 3));
        assert_eq!(iter.next(), Some(8 * 32 + 5 * 4 + 0..8 * 32 + 5 * 4 + 3));
        assert_eq!(iter.next(), Some(9 * 32 + 5 * 4 + 0..9 * 32 + 5 * 4 + 3));
        assert_eq!(iter.next(), None);
    }
}
