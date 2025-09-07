//! Synchronization state tracking of all resources.

use crate::{
    assert_unsafe_precondition,
    collector::{self, DeferredBatch},
    descriptor_set::{BindlessContext, BindlessContextCreateInfo},
    Id, InvalidSlotError, Object, Ref,
};
use ash::vk;
use concurrent_slotmap::{hyaline, SlotMap};
use parking_lot::{Mutex, RwLock};
use smallvec::SmallVec;
use std::{
    hash::Hash,
    iter, mem,
    num::NonZero,
    ops::{BitOr, BitOrAssign},
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc, Weak,
    },
    time::Duration,
};
use vulkano::{
    buffer::{AllocateBufferError, Buffer, BufferCreateInfo},
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::allocator::StandardDescriptorSetAllocator,
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
    storage: Arc<ResourceStorage>,
    bindless_context: Option<BindlessContext>,
}

#[derive(Debug)]
pub(crate) struct ResourceStorage {
    device: Arc<Device>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,

    hyaline_collector: hyaline::CollectorHandle,
    buffers: SlotMap<Id, BufferState>,
    images: SlotMap<Id, ImageState>,
    swapchains: SlotMap<Id, SwapchainState>,
    flights: SlotMap<Id, Flight>,

    garbage_queue: collector::GlobalQueue,
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

// FIXME: imported/exported fences
#[derive(Debug)]
pub struct Flight {
    // HACK: `Flight::wait` needs this in order to collect garbage.
    resources: Weak<Resources>,
    frame_count: NonZero<u32>,
    biased_started_frame: AtomicU64,
    current_frame: AtomicU64,
    biased_complete_frame: AtomicU64,
    fences: SmallVec<[RwLock<Fence>; 3]>,
    garbage_queue: collector::LocalQueue,
    pub(crate) state: Mutex<()>,
}

impl Resources {
    /// Creates a new `Resources` collection.
    ///
    /// # Panics
    ///
    /// - Panics if `device` already has a `Resources` collection associated with it.
    pub fn new(
        device: &Arc<Device>,
        create_info: &ResourcesCreateInfo<'_>,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        let &ResourcesCreateInfo {
            max_buffers,
            max_images,
            max_swapchains,
            max_flights,
            bindless_context,
            _ne: _,
        } = create_info;

        let mut registered_devices = REGISTERED_DEVICES.lock();
        let device_addr = Arc::as_ptr(device) as usize;

        assert!(
            !registered_devices.contains(&device_addr),
            "the device already has a `Resources` collection associated with it",
        );

        registered_devices.push(device_addr);
        drop(registered_devices);

        let memory_allocator = Arc::new(StandardMemoryAllocator::new(device, &Default::default()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device,
            &Default::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device,
            &Default::default(),
        ));

        let hyaline_collector = hyaline::CollectorHandle::new();
        let storage = Arc::new(ResourceStorage {
            device: device.clone(),
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
            // SAFETY: We make sure that any guard we acquire through `ResourceStorage::pin` doesn't
            // outlive the `Resources` collection.
            buffers: unsafe {
                SlotMap::with_collector_and_key(max_buffers, hyaline_collector.clone())
            },
            // SAFETY: Same as the previous.
            images: unsafe {
                SlotMap::with_collector_and_key(max_images, hyaline_collector.clone())
            },
            // SAFETY: Same as the previous.
            swapchains: unsafe {
                SlotMap::with_collector_and_key(max_swapchains, hyaline_collector.clone())
            },
            // SAFETY: Same as the previous.
            flights: unsafe {
                SlotMap::with_collector_and_key(max_flights, hyaline_collector.clone())
            },
            // SAFETY: Same as the previous.
            garbage_queue: unsafe { collector::GlobalQueue::new(hyaline_collector.clone()) },
            hyaline_collector,
        });
        let bindless_context = bindless_context
            .map(|bindless_info| BindlessContext::new(&storage, bindless_info))
            .transpose()?;

        Ok(Arc::new(Resources {
            storage,
            bindless_context,
        }))
    }

    /// Returns the standard memory allocator.
    #[inline]
    #[must_use]
    pub fn memory_allocator(&self) -> &Arc<StandardMemoryAllocator> {
        &self.storage.memory_allocator
    }

    pub(crate) fn command_buffer_allocator(&self) -> &Arc<StandardCommandBufferAllocator> {
        &self.storage.command_buffer_allocator
    }

    pub(crate) fn descriptor_set_allocator(&self) -> &Arc<StandardDescriptorSetAllocator> {
        &self.storage.descriptor_set_allocator
    }

    pub(crate) fn garbage_queue(&self) -> &collector::GlobalQueue {
        &self.storage.garbage_queue
    }

    /// Returns the `BindlessContext`.
    ///
    /// Returns `None` if [`ResourcesCreateInfo::bindless_context`] was not specified when creating
    /// the collection.
    #[inline]
    pub fn bindless_context(&self) -> Option<&BindlessContext> {
        self.bindless_context.as_ref()
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
        create_info: &BufferCreateInfo<'_>,
        allocation_info: &AllocationCreateInfo<'_>,
        layout: DeviceLayout,
    ) -> Result<Id<Buffer>, Validated<AllocateBufferError>> {
        let buffer = Buffer::new(
            &self.storage.memory_allocator,
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
        create_info: &ImageCreateInfo<'_>,
        allocation_info: &AllocationCreateInfo<'_>,
    ) -> Result<Id<Image>, Validated<AllocateImageError>> {
        let image = Image::new(&self.storage.memory_allocator, create_info, allocation_info)?;

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
        surface: &Arc<Surface>,
        create_info: &SwapchainCreateInfo<'_>,
    ) -> Result<Id<Swapchain>, Validated<VulkanError>> {
        let frames_in_flight = self.flight(flight_id).unwrap().frame_count();

        assert!(create_info.min_image_count >= frames_in_flight);

        let (swapchain, images) = Swapchain::new(self.device(), surface, create_info)?;

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
    pub fn create_flight(self: &Arc<Self>, frame_count: u32) -> Result<Id<Flight>, VulkanError> {
        let frame_count =
            NonZero::new(frame_count).expect("a flight with zero frames is not valid");

        let fences = (0..frame_count.get())
            .map(|_| {
                // SAFETY: The parameters are valid.
                unsafe {
                    Fence::new_unchecked(
                        self.device(),
                        &FenceCreateInfo {
                            flags: FenceCreateFlags::SIGNALED,
                            ..Default::default()
                        },
                    )
                }
                .map(RwLock::new)
            })
            .collect::<Result<_, VulkanError>>()?;

        let flight = Flight {
            resources: Arc::downgrade(self),
            frame_count,
            biased_started_frame: AtomicU64::new(u64::from(frame_count.get())),
            current_frame: AtomicU64::new(0),
            biased_complete_frame: AtomicU64::new(u64::from(frame_count.get())),
            fences,
            garbage_queue: self.garbage_queue().register_local(),
            state: Mutex::new(()),
        };

        let id = self
            .storage
            .flights
            .insert_with_tag(flight, Flight::TAG, &self.storage.pin());

        Ok(unsafe { id.parametrize() })
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

        let id = self
            .storage
            .buffers
            .insert_with_tag(state, Buffer::TAG, &self.storage.pin());

        unsafe { id.parametrize() }
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

        let id = self
            .storage
            .images
            .insert_with_tag(state, Image::TAG, &self.storage.pin());

        unsafe { id.parametrize() }
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
        let guard = &self.storage.pin();

        let frames_in_flight = self
            .storage
            .flight_protected(flight_id, guard)
            .unwrap()
            .frame_count();

        let semaphores = (0..frames_in_flight)
            .map(|_| {
                Ok(SwapchainSemaphoreState {
                    // SAFETY: The parameters are valid.
                    image_available_semaphore: unsafe {
                        Semaphore::new_unchecked(self.device(), &Default::default())
                    }?,
                    // SAFETY: The parameters are valid.
                    pre_present_complete_semaphore: unsafe {
                        Semaphore::new_unchecked(self.device(), &Default::default())
                    }?,
                    // SAFETY: The parameters are valid.
                    tasks_complete_semaphore: unsafe {
                        Semaphore::new_unchecked(self.device(), &Default::default())
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
            last_access: Mutex::new(ImageAccess::NONE),
        };

        let id = self
            .storage
            .swapchains
            .insert_with_tag(state, Swapchain::TAG, guard);

        Ok(unsafe { id.parametrize() })
    }

    /// Calls [`Swapchain::recreate`] on the swapchain corresponding to `id` and adds the new
    /// swapchain to the collection. The old swapchain will be cleaned up as soon as possible.
    ///
    /// # Panics
    ///
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
        f: impl for<'a> FnOnce(&SwapchainCreateInfo<'a>) -> SwapchainCreateInfo<'a>,
    ) -> Result<Id<Swapchain>, Validated<VulkanError>> {
        let guard = &self.storage.pin();

        let state = self.storage.swapchain_protected(id, guard).unwrap();
        let swapchain = state.swapchain();
        let flight_id = state.flight_id;
        let flight = unsafe { self.storage.flight_unchecked_protected(flight_id, guard) };

        let (new_swapchain, new_images) = swapchain.recreate(&f(&swapchain.create_info()))?;

        let frames_in_flight = flight.frame_count();

        assert!(new_swapchain.image_count() >= frames_in_flight);

        let new_state = SwapchainState {
            swapchain: new_swapchain,
            images: new_images.into(),
            semaphores: state.semaphores.clone(),
            flight_id,
            current_image_index: AtomicU32::new(u32::MAX),
            last_access: Mutex::new(ImageAccess::NONE),
        };

        let new_id = self
            .storage
            .swapchains
            .insert_with_tag(new_state, Swapchain::TAG, guard);

        let mut batch = self.create_deferred_batch();
        batch.destroy_swapchain(id);

        // SAFETY: Swapchains are owned by a flight in order to facilitates this. No other flight
        // can be using the swapchain.
        unsafe { batch.enqueue_with_flights(iter::once(flight_id)) };

        Ok(unsafe { new_id.parametrize() })
    }

    pub(crate) fn invalidate_buffer<'a>(
        &'a self,
        id: Id<Buffer>,
        guard: &'a hyaline::Guard<'_>,
    ) -> Result<&'a BufferState> {
        let state = self
            .storage
            .buffers
            .invalidate(id.erase(), guard)
            .ok_or(InvalidSlotError::new(id))?;

        Ok(state)
    }

    pub(crate) fn invalidate_image<'a>(
        &'a self,
        id: Id<Image>,
        guard: &'a hyaline::Guard<'_>,
    ) -> Result<&'a ImageState> {
        let state = self
            .storage
            .images
            .invalidate(id.erase(), guard)
            .ok_or(InvalidSlotError::new(id))?;

        Ok(state)
    }

    pub(crate) fn invalidate_swapchain<'a>(
        &'a self,
        id: Id<Swapchain>,
        guard: &'a hyaline::Guard<'_>,
    ) -> Result<&'a SwapchainState> {
        let state = self
            .storage
            .swapchains
            .invalidate(id.erase(), guard)
            .ok_or(InvalidSlotError::new(id))?;

        Ok(state)
    }

    pub(crate) unsafe fn remove_invalidated_buffer_unchecked(&self, id: Id<Buffer>) {
        // SAFETY: Enforced by the caller.
        unsafe {
            self.storage
                .buffers
                .remove_invalidated_unchecked(id.erase())
        };
    }

    pub(crate) unsafe fn remove_invalidated_image_unchecked(&self, id: Id<Image>) {
        // SAFETY: Enforced by the caller.
        unsafe { self.storage.images.remove_invalidated_unchecked(id.erase()) };
    }

    pub(crate) unsafe fn remove_invalidated_swapchain_unchecked(&self, id: Id<Swapchain>) {
        // SAFETY: Enforced by the caller.
        unsafe {
            self.storage
                .swapchains
                .remove_invalidated_unchecked(id.erase())
        };
    }

    /// Returns the buffer corresponding to `id`.
    #[inline]
    pub fn buffer(&self, id: Id<Buffer>) -> Result<Ref<'_, BufferState>> {
        self.storage.buffer(id)
    }

    pub(crate) fn buffer_protected<'a>(
        &'a self,
        id: Id<Buffer>,
        guard: &'a hyaline::Guard<'a>,
    ) -> Result<&'a BufferState> {
        self.storage.buffer_protected(id, guard)
    }

    pub(crate) unsafe fn buffer_unchecked_protected<'a>(
        &'a self,
        id: Id<Buffer>,
        guard: &'a hyaline::Guard<'a>,
    ) -> &'a BufferState {
        // SAFETY: Enforced by the caller.
        unsafe { self.storage.buffer_unchecked_protected(id, guard) }
    }

    /// Returns the image corresponding to `id`.
    #[inline]
    pub fn image(&self, id: Id<Image>) -> Result<Ref<'_, ImageState>> {
        self.storage.image(id)
    }

    pub(crate) fn image_protected<'a>(
        &'a self,
        id: Id<Image>,
        guard: &'a hyaline::Guard<'a>,
    ) -> Result<&'a ImageState> {
        self.storage.image_protected(id, guard)
    }

    pub(crate) unsafe fn image_unchecked_protected<'a>(
        &'a self,
        id: Id<Image>,
        guard: &'a hyaline::Guard<'a>,
    ) -> &'a ImageState {
        // SAFETY: Enforced by the caller.
        unsafe { self.storage.image_unchecked_protected(id, guard) }
    }

    /// Returns the swapchain corresponding to `id`.
    #[inline]
    pub fn swapchain(&self, id: Id<Swapchain>) -> Result<Ref<'_, SwapchainState>> {
        self.storage.swapchain(id)
    }

    pub(crate) fn swapchain_protected<'a>(
        &'a self,
        id: Id<Swapchain>,
        guard: &'a hyaline::Guard<'a>,
    ) -> Result<&'a SwapchainState> {
        self.storage.swapchain_protected(id, guard)
    }

    pub(crate) unsafe fn swapchain_unchecked_protected<'a>(
        &'a self,
        id: Id<Swapchain>,
        guard: &'a hyaline::Guard<'a>,
    ) -> &'a SwapchainState {
        // SAFETY: Enforced by the caller.
        unsafe { self.storage.swapchain_unchecked_protected(id, guard) }
    }

    /// Returns the [flight] corresponding to `id`.
    #[inline]
    pub fn flight(&self, id: Id<Flight>) -> Result<Ref<'_, Flight>> {
        self.storage.flight(id)
    }

    pub(crate) fn flight_protected<'a>(
        &'a self,
        id: Id<Flight>,
        guard: &'a hyaline::Guard<'a>,
    ) -> Result<&'a Flight> {
        self.storage.flight_protected(id, guard)
    }

    pub(crate) unsafe fn flight_unchecked_protected<'a>(
        &'a self,
        id: Id<Flight>,
        guard: &'a hyaline::Guard<'a>,
    ) -> &'a Flight {
        // SAFETY: Enforced by the caller.
        unsafe { self.storage.flight_unchecked_protected(id, guard) }
    }

    pub(crate) fn flights_protected<'a>(
        &'a self,
        guard: &'a hyaline::Guard<'a>,
    ) -> impl Iterator<Item = (Id<Flight>, &'a Flight)> + use<'a> {
        self.storage
            .flights
            .iter(guard)
            .map(|(k, v)| (unsafe { k.parametrize() }, v))
    }

    /// Creates a new deferred function batch.
    ///
    /// You should batch as many deferred functions together as you can for optimal performance.
    pub fn create_deferred_batch(&self) -> DeferredBatch<'_> {
        DeferredBatch::new(self)
    }

    /// Waits for all frames of all flights to finish.
    ///
    /// This is equivalent to [`Device::wait_idle`], but unlike that method, this method
    /// additionally collects outstanding garbage.
    pub fn wait_idle(&self) -> Result<(), VulkanError> {
        let guard = &self.storage.pin();
        let mut frames = SmallVec::<[_; 8]>::new();

        for (_, flight) in self.storage.flights.iter(guard) {
            let biased_frame = flight.current_frame() + u64::from(flight.frame_count());
            frames.push((flight, biased_frame));
        }

        self.device().wait_idle()?;

        let mut collect = false;

        for &(flight, biased_frame) in &frames {
            // SAFETY: We waited for the frame.
            if unsafe { flight.update_biased_complete_frame(biased_frame) }.is_ok() {
                unsafe { flight.garbage_queue().collect(self, flight, guard) };
                collect = true;
            }
        }

        if collect {
            unsafe { self.garbage_queue().collect(self, guard) };
        }

        Ok(())
    }

    pub(crate) fn pin(&self) -> hyaline::Guard<'_> {
        self.storage.pin()
    }
}

unsafe impl DeviceOwned for Resources {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.storage.device
    }
}

impl ResourceStorage {
    pub(crate) fn hyaline_collector(&self) -> &hyaline::CollectorHandle {
        &self.hyaline_collector
    }

    pub(crate) fn pin(&self) -> hyaline::Guard<'_> {
        // SAFETY: The guard's lifetime is bound to the collection.
        unsafe { self.hyaline_collector.pin() }
    }

    pub(crate) fn buffer(&self, id: Id<Buffer>) -> Result<Ref<'_, BufferState>> {
        let guard = self.pin();

        let state = self
            .buffers
            // SAFETY: We unbind the lifetime because this would result in E0515 otherwise. This is
            // perfectly safe to do -- none of these methods actually borrow from the guard (there's
            // physically no way for them to; this is encoded in the type system). The lifetime is
            // bound to the returned reference to ensure that the reference doesn't outlive the
            // guard. We enforce that by `Ref` owning the `hyaline::Guard` instead.
            .get(id.erase(), unsafe {
                mem::transmute::<&hyaline::Guard<'_>, &hyaline::Guard<'_>>(&guard)
            })
            .ok_or(InvalidSlotError::new(id))?;

        Ok(Ref::new(state, guard))
    }

    pub(crate) fn buffer_protected<'a>(
        &'a self,
        id: Id<Buffer>,
        guard: &'a hyaline::Guard<'a>,
    ) -> Result<&'a BufferState> {
        self.buffers
            .get(id.erase(), guard)
            .ok_or(InvalidSlotError::new(id))
    }

    pub(crate) unsafe fn buffer_unchecked_protected<'a>(
        &'a self,
        id: Id<Buffer>,
        guard: &'a hyaline::Guard<'a>,
    ) -> &'a BufferState {
        assert_unsafe_precondition!(
            self.buffers.get(id.erase(), &self.pin()).is_some(),
            "`id` must be a valid buffer ID",
        );

        // SAFETY: Enforced by the caller.
        unsafe { self.buffers.get_unchecked(id.erase(), guard) }
    }

    pub(crate) fn image(&self, id: Id<Image>) -> Result<Ref<'_, ImageState>> {
        let guard = self.pin();

        let state = self
            .images
            // SAFETY: Same as in the `buffer` method above.
            .get(id.erase(), unsafe {
                mem::transmute::<&hyaline::Guard<'_>, &hyaline::Guard<'_>>(&guard)
            })
            .ok_or(InvalidSlotError::new(id))?;

        Ok(Ref::new(state, guard))
    }

    pub(crate) fn image_protected<'a>(
        &'a self,
        id: Id<Image>,
        guard: &'a hyaline::Guard<'a>,
    ) -> Result<&'a ImageState> {
        self.images
            .get(id.erase(), guard)
            .ok_or(InvalidSlotError::new(id))
    }

    pub(crate) unsafe fn image_unchecked_protected<'a>(
        &'a self,
        id: Id<Image>,
        guard: &'a hyaline::Guard<'a>,
    ) -> &'a ImageState {
        assert_unsafe_precondition!(
            self.images.get(id.erase(), &self.pin()).is_some(),
            "`id` must be a valid image ID",
        );

        // SAFETY: Enforced by the caller.
        unsafe { self.images.get_unchecked(id.erase(), guard) }
    }

    pub(crate) fn swapchain(&self, id: Id<Swapchain>) -> Result<Ref<'_, SwapchainState>> {
        let guard = self.pin();

        let state = self
            .swapchains
            // SAFETY: Same as in the `buffer` method above.
            .get(id.erase(), unsafe {
                mem::transmute::<&hyaline::Guard<'_>, &hyaline::Guard<'_>>(&guard)
            })
            .ok_or(InvalidSlotError::new(id))?;

        Ok(Ref::new(state, guard))
    }

    pub(crate) fn swapchain_protected<'a>(
        &'a self,
        id: Id<Swapchain>,
        guard: &'a hyaline::Guard<'a>,
    ) -> Result<&'a SwapchainState> {
        self.swapchains
            .get(id.erase(), guard)
            .ok_or(InvalidSlotError::new(id))
    }

    pub(crate) unsafe fn swapchain_unchecked_protected<'a>(
        &'a self,
        id: Id<Swapchain>,
        guard: &'a hyaline::Guard<'a>,
    ) -> &'a SwapchainState {
        assert_unsafe_precondition!(
            self.swapchains.get(id.erase(), &self.pin()).is_some(),
            "`id` must be a valid swapchain ID",
        );

        // SAFETY: Enforced by the caller.
        unsafe { self.swapchains.get_unchecked(id.erase(), guard) }
    }

    pub(crate) fn flight(&self, id: Id<Flight>) -> Result<Ref<'_, Flight>> {
        let guard = self.pin();

        let state = self
            .flights
            // SAFETY: We own the `hyaline::Guard`.
            .get(id.erase(), unsafe {
                mem::transmute::<&hyaline::Guard<'_>, &hyaline::Guard<'_>>(&guard)
            })
            .ok_or(InvalidSlotError::new(id))?;

        Ok(Ref::new(state, guard))
    }

    pub(crate) fn flight_protected<'a>(
        &'a self,
        id: Id<Flight>,
        guard: &'a hyaline::Guard<'a>,
    ) -> Result<&'a Flight> {
        self.flights
            .get(id.erase(), guard)
            .ok_or(InvalidSlotError::new(id))
    }

    pub(crate) unsafe fn flight_unchecked_protected<'a>(
        &'a self,
        id: Id<Flight>,
        guard: &'a hyaline::Guard<'a>,
    ) -> &'a Flight {
        // SAFETY: Enforced by the caller.
        unsafe { self.flights.get_unchecked(id.erase(), guard) }
    }
}

unsafe impl DeviceOwned for ResourceStorage {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl Drop for Resources {
    fn drop(&mut self) {
        let guard = &self.storage.pin();

        if let Err(err) = self.device().wait_idle() {
            if err == VulkanError::DeviceLost {
                return;
            }

            eprintln!(
                "failed to wait for idle rendering graceful shutdown impossible: {err}; aborting",
            );
            std::process::abort();
        }

        for (_, flight) in self.storage.flights.iter(guard) {
            // SAFETY:
            // * We are being dropped, which is proof that no outstanding references to any slots
            //   can exist and that we have exclusive access to this queue.
            // * We waited on the flight above.
            // * `self` is the correct `Resources` collection.
            unsafe { flight.garbage_queue.drop(self, guard) };
        }

        // SAFETY: Same as the `drop` above.
        unsafe { self.storage.garbage_queue.drop(self, guard) };

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
    pub const fn new(access_types: AccessTypes, queue_family_index: u32) -> Self {
        assert!(access_types.are_valid_buffer_access_types());

        BufferAccess {
            stage_mask: access_types.stage_mask(),
            access_mask: access_types.access_mask(),
            queue_family_index,
        }
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
        access_types: AccessTypes,
        layout_type: ImageLayoutType,
        queue_family_index: u32,
    ) -> Self {
        assert!(access_types.are_valid_image_access_types());

        ImageAccess {
            stage_mask: access_types.stage_mask(),
            access_mask: access_types.access_mask(),
            image_layout: access_types.image_layout(layout_type),
            queue_family_index,
        }
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

        // The `frame_count` bias cancels out under the modulus, however the `1` bias doesn't, so we
        // subtract it to get the same index as would be computed using the unbiased `frame`.
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

    unsafe fn update_biased_complete_frame(&self, biased_frame: u64) -> Result<u64, u64> {
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

/// Parameters to create a new [`Resources`] collection.
#[derive(Clone, Debug)]
pub struct ResourcesCreateInfo<'a> {
    /// The maximum number of [`Buffer`]s that the collection can hold at once.
    ///
    /// The default value is `16777216` (2<sup>24</sup>).
    pub max_buffers: u32,

    /// The maximum number of [`Image`]s that the collection can hold at once.
    ///
    /// The default value is `16777216` (2<sup>24</sup>).
    pub max_images: u32,

    /// The maximum number of [`Swapchain`]s that the collection can hold at once.
    ///
    /// The default value is `256` (2<sup>8</sup>).
    pub max_swapchains: u32,

    /// The maximum number of [`Flight`]s that the collection can hold at once.
    ///
    /// The default value is `256` (2<sup>8</sup>).
    pub max_flights: u32,

    /// Parameters to create a new [`BindlessContext`].
    ///
    /// If set, the device extensions given by [`BindlessContext::required_extensions`] and the
    /// device features given by [`BindlessContext::required_features`] must be enabled on the
    /// device.
    ///
    /// The default value is `None`.
    pub bindless_context: Option<&'a BindlessContextCreateInfo<'a>>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for ResourcesCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl ResourcesCreateInfo<'_> {
    /// Returns a default `ResourcesCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            max_buffers: 1 << 24,
            max_images: 1 << 24,
            max_swapchains: 1 << 8,
            max_flights: 1 << 8,
            bindless_context: None,
            _ne: crate::NE,
        }
    }
}

/// Specifies which types of accesses are performed on a resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AccessTypes {
    stage_mask: PipelineStages,
    access_mask: AccessFlags,
    image_layout: ImageLayout,
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
        impl AccessTypes {
            $(
                $(#[$meta])*
                pub const $name: Self = AccessTypes {
                    stage_mask: PipelineStages::empty()$(.union(PipelineStages::$stage_flag))+,
                    access_mask: AccessFlags::empty()$(.union(AccessFlags::$access_flag))+,
                    image_layout: ImageLayout::$image_layout,
                };
            )*
        }
    };
}

access_types! {
    INDIRECT_COMMAND_READ {
        stage_mask: DRAW_INDIRECT,
        access_mask: INDIRECT_COMMAND_READ,
        image_layout: Undefined,
    }

    INDEX_READ {
        stage_mask: INDEX_INPUT,
        access_mask: INDEX_READ,
        image_layout: Undefined,
    }

    VERTEX_ATTRIBUTE_READ {
        stage_mask: VERTEX_ATTRIBUTE_INPUT,
        access_mask: VERTEX_ATTRIBUTE_READ,
        image_layout: Undefined,
    }

    VERTEX_SHADER_UNIFORM_READ {
        stage_mask: VERTEX_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
    }

    VERTEX_SHADER_SAMPLED_READ {
        stage_mask: VERTEX_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
    }

    VERTEX_SHADER_STORAGE_READ {
        stage_mask: VERTEX_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
    }

    VERTEX_SHADER_STORAGE_WRITE {
        stage_mask: VERTEX_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
    }

    VERTEX_SHADER_ACCELERATION_STRUCTURE_READ {
        stage_mask: VERTEX_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    TESSELLATION_CONTROL_SHADER_UNIFORM_READ {
        stage_mask: TESSELLATION_CONTROL_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
    }

    TESSELLATION_CONTROL_SHADER_SAMPLED_READ {
        stage_mask: TESSELLATION_CONTROL_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
    }

    TESSELLATION_CONTROL_SHADER_STORAGE_READ {
        stage_mask: TESSELLATION_CONTROL_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
    }

    TESSELLATION_CONTROL_SHADER_STORAGE_WRITE {
        stage_mask: TESSELLATION_CONTROL_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
    }

    TESSELLATION_CONTROL_SHADER_ACCELERATION_STRUCTURE_READ {
        stage_mask: TESSELLATION_CONTROL_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    TESSELLATION_EVALUATION_SHADER_UNIFORM_READ {
        stage_mask: TESSELLATION_EVALUATION_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
    }

    TESSELLATION_EVALUATION_SHADER_SAMPLED_READ {
        stage_mask: TESSELLATION_EVALUATION_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
    }

    TESSELLATION_EVALUATION_SHADER_STORAGE_READ {
        stage_mask: TESSELLATION_EVALUATION_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
    }

    TESSELLATION_EVALUATION_SHADER_STORAGE_WRITE {
        stage_mask: TESSELLATION_EVALUATION_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
    }

    TESSELLATION_EVALUATION_SHADER_ACCELERATION_STRUCTURE_READ {
        stage_mask: TESSELLATION_EVALUATION_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    GEOMETRY_SHADER_UNIFORM_READ {
        stage_mask: GEOMETRY_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
    }

    GEOMETRY_SHADER_SAMPLED_READ {
        stage_mask: GEOMETRY_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
    }

    GEOMETRY_SHADER_STORAGE_READ {
        stage_mask: GEOMETRY_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
    }

    GEOMETRY_SHADER_STORAGE_WRITE {
        stage_mask: GEOMETRY_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
    }

    GEOMETRY_SHADER_ACCELERATION_STRUCTURE_READ {
        stage_mask: GEOMETRY_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    FRAGMENT_SHADER_UNIFORM_READ {
        stage_mask: FRAGMENT_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
    }

    FRAGMENT_SHADER_COLOR_INPUT_ATTACHMENT_READ {
        stage_mask: FRAGMENT_SHADER,
        access_mask: INPUT_ATTACHMENT_READ,
        image_layout: ShaderReadOnlyOptimal,
    }

    FRAGMENT_SHADER_DEPTH_STENCIL_INPUT_ATTACHMENT_READ {
        stage_mask: FRAGMENT_SHADER,
        access_mask: INPUT_ATTACHMENT_READ,
        image_layout: DepthStencilReadOnlyOptimal,
    }

    FRAGMENT_SHADER_SAMPLED_READ {
        stage_mask: FRAGMENT_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
    }

    FRAGMENT_SHADER_STORAGE_READ {
        stage_mask: FRAGMENT_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
    }

    FRAGMENT_SHADER_STORAGE_WRITE {
        stage_mask: FRAGMENT_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
    }

    FRAGMENT_SHADER_ACCELERATION_STRUCTURE_READ {
        stage_mask: FRAGMENT_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    DEPTH_STENCIL_ATTACHMENT_READ {
        stage_mask: EARLY_FRAGMENT_TESTS | LATE_FRAGMENT_TESTS,
        access_mask: DEPTH_STENCIL_ATTACHMENT_READ,
        image_layout: DepthStencilReadOnlyOptimal,
    }

    DEPTH_STENCIL_ATTACHMENT_WRITE {
        stage_mask: EARLY_FRAGMENT_TESTS | LATE_FRAGMENT_TESTS,
        access_mask: DEPTH_STENCIL_ATTACHMENT_WRITE,
        image_layout: DepthStencilAttachmentOptimal,
    }

    DEPTH_ATTACHMENT_WRITE_STENCIL_READ_ONLY {
        stage_mask: EARLY_FRAGMENT_TESTS | LATE_FRAGMENT_TESTS,
        access_mask: DEPTH_STENCIL_ATTACHMENT_READ | DEPTH_STENCIL_ATTACHMENT_WRITE,
        image_layout: DepthAttachmentStencilReadOnlyOptimal,
    }

    DEPTH_READ_ONLY_STENCIL_ATTACHMENT_WRITE {
        stage_mask: EARLY_FRAGMENT_TESTS | LATE_FRAGMENT_TESTS,
        access_mask: DEPTH_STENCIL_ATTACHMENT_READ | DEPTH_STENCIL_ATTACHMENT_WRITE,
        image_layout: DepthReadOnlyStencilAttachmentOptimal,
    }

    COLOR_ATTACHMENT_READ {
        stage_mask: COLOR_ATTACHMENT_OUTPUT,
        access_mask: COLOR_ATTACHMENT_READ,
        image_layout: ColorAttachmentOptimal,
    }

    COLOR_ATTACHMENT_WRITE {
        stage_mask: COLOR_ATTACHMENT_OUTPUT,
        access_mask: COLOR_ATTACHMENT_WRITE,
        image_layout: ColorAttachmentOptimal,
    }

    COMPUTE_SHADER_UNIFORM_READ {
        stage_mask: COMPUTE_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
    }

    COMPUTE_SHADER_SAMPLED_READ {
        stage_mask: COMPUTE_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
    }

    COMPUTE_SHADER_STORAGE_READ {
        stage_mask: COMPUTE_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
    }

    COMPUTE_SHADER_STORAGE_WRITE {
        stage_mask: COMPUTE_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
    }

    COMPUTE_SHADER_ACCELERATION_STRUCTURE_READ {
        stage_mask: COMPUTE_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    COPY_TRANSFER_READ {
        stage_mask: COPY,
        access_mask: TRANSFER_READ,
        image_layout: TransferSrcOptimal,
    }

    COPY_TRANSFER_WRITE {
        stage_mask: COPY,
        access_mask: TRANSFER_WRITE,
        image_layout: TransferDstOptimal,
    }

    BLIT_TRANSFER_READ {
        stage_mask: BLIT,
        access_mask: TRANSFER_READ,
        image_layout: TransferSrcOptimal,
    }

    BLIT_TRANSFER_WRITE {
        stage_mask: BLIT,
        access_mask: TRANSFER_WRITE,
        image_layout: TransferDstOptimal,
    }

    RESOLVE_TRANSFER_READ {
        stage_mask: RESOLVE,
        access_mask: TRANSFER_READ,
        image_layout: TransferSrcOptimal,
    }

    RESOLVE_TRANSFER_WRITE {
        stage_mask: RESOLVE,
        access_mask: TRANSFER_WRITE,
        image_layout: TransferDstOptimal,
    }

    CLEAR_TRANSFER_WRITE {
        stage_mask: CLEAR,
        access_mask: TRANSFER_WRITE,
        image_layout: TransferDstOptimal,
    }

    ACCELERATION_STRUCTURE_COPY_TRANSFER_READ {
        stage_mask: ACCELERATION_STRUCTURE_COPY,
        access_mask: TRANSFER_READ,
        image_layout: Undefined,
    }

    ACCELERATION_STRUCTURE_COPY_TRANSFER_WRITE {
        stage_mask: ACCELERATION_STRUCTURE_COPY,
        access_mask: TRANSFER_WRITE,
        image_layout: Undefined,
    }

    // TODO:
    // VIDEO_DECODE_READ {
    //     stage_mask: VIDEO_DECODE,
    //     access_mask: VIDEO_DECODE_READ,
    //     image_layout: Undefined,
    // }

    // TODO:
    // VIDEO_DECODE_WRITE {
    //     stage_mask: VIDEO_DECODE,
    //     access_mask: VIDEO_DECODE_WRITE,
    //     image_layout: VideoDecodeDst,
    // }

    // TODO:
    // VIDEO_DECODE_DPB_READ {
    //     stage_mask: VIDEO_DECODE,
    //     access_mask: VIDEO_DECODE_READ,
    //     image_layout: VideoDecodeDpb,
    // }

    // TODO:
    // VIDEO_DECODE_DPB_WRITE {
    //     stage_mask: VIDEO_DECODE,
    //     access_mask: VIDEO_DECODE_WRITE,
    //     image_layout: VideoDecodeDpb,
    // }

    // TODO:
    // VIDEO_ENCODE_READ {
    //     stage_mask: VIDEO_ENCODE,
    //     access_mask: VIDEO_ENCODE_READ,
    //     image_layout: VideoEncodeSrc,
    // }

    // TODO:
    // VIDEO_ENCODE_WRITE {
    //     stage_mask: VIDEO_ENCODE,
    //     access_mask: VIDEO_ENCODE_WRITE,
    //     image_layout: Undefined,
    // }

    // TODO:
    // VIDEO_ENCODE_DPB_READ {
    //     stage_mask: VIDEO_ENCODE,
    //     access_mask: VIDEO_ENCODE_READ,
    //     image_layout: VideoEncodeDpb,
    // }

    // TODO:
    // VIDEO_ENCODE_DPB_WRITE {
    //     stage_mask: VIDEO_ENCODE,
    //     access_mask: VIDEO_ENCODE_WRITE,
    //     image_layout: VideoEncodeDpb,
    // }

    RAY_TRACING_SHADER_UNIFORM_READ {
        stage_mask: RAY_TRACING_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
    }

    RAY_TRACING_SHADER_SAMPLED_READ {
        stage_mask: RAY_TRACING_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
    }

    RAY_TRACING_SHADER_STORAGE_READ {
        stage_mask: RAY_TRACING_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
    }

    RAY_TRACING_SHADER_STORAGE_WRITE {
        stage_mask: RAY_TRACING_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
    }

    RAY_TRACING_SHADER_BINDING_TABLE_READ {
        stage_mask: RAY_TRACING_SHADER,
        access_mask: SHADER_BINDING_TABLE_READ,
        image_layout: Undefined,
    }

    RAY_TRACING_SHADER_ACCELERATION_STRUCTURE_READ {
        stage_mask: RAY_TRACING_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    TASK_SHADER_UNIFORM_READ {
        stage_mask: TASK_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
    }

    TASK_SHADER_SAMPLED_READ {
        stage_mask: TASK_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
    }

    TASK_SHADER_STORAGE_READ {
        stage_mask: TASK_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
    }

    TASK_SHADER_STORAGE_WRITE {
        stage_mask: TASK_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
    }

    TASK_SHADER_ACCELERATION_STRUCTURE_READ {
        stage_mask: TASK_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    MESH_SHADER_UNIFORM_READ {
        stage_mask: MESH_SHADER,
        access_mask: UNIFORM_READ,
        image_layout: Undefined,
    }

    MESH_SHADER_SAMPLED_READ {
        stage_mask: MESH_SHADER,
        access_mask: SHADER_SAMPLED_READ,
        image_layout: ShaderReadOnlyOptimal,
    }

    MESH_SHADER_STORAGE_READ {
        stage_mask: MESH_SHADER,
        access_mask: SHADER_STORAGE_READ,
        image_layout: General,
    }

    MESH_SHADER_STORAGE_WRITE {
        stage_mask: MESH_SHADER,
        access_mask: SHADER_STORAGE_WRITE,
        image_layout: General,
    }

    MESH_SHADER_ACCELERATION_STRUCTURE_READ {
        stage_mask: MESH_SHADER,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    ACCELERATION_STRUCTURE_BUILD_INDIRECT_COMMAND_READ {
        stage_mask: ACCELERATION_STRUCTURE_BUILD,
        access_mask: INDIRECT_COMMAND_READ,
        image_layout: Undefined,
    }

    ACCELERATION_STRUCTURE_BUILD_SHADER_READ {
        stage_mask: ACCELERATION_STRUCTURE_BUILD,
        access_mask: SHADER_READ,
        image_layout: Undefined,
    }

    ACCELERATION_STRUCTURE_BUILD_ACCELERATION_STRUCTURE_READ {
        stage_mask: ACCELERATION_STRUCTURE_BUILD,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    ACCELERATION_STRUCTURE_BUILD_ACCELERATION_STRUCTURE_WRITE {
        stage_mask: ACCELERATION_STRUCTURE_BUILD,
        access_mask: ACCELERATION_STRUCTURE_WRITE,
        image_layout: Undefined,
    }

    ACCELERATION_STRUCTURE_COPY_ACCELERATION_STRUCTURE_READ {
        stage_mask: ACCELERATION_STRUCTURE_COPY,
        access_mask: ACCELERATION_STRUCTURE_READ,
        image_layout: Undefined,
    }

    ACCELERATION_STRUCTURE_COPY_ACCELERATION_STRUCTURE_WRITE {
        stage_mask: ACCELERATION_STRUCTURE_COPY,
        access_mask: ACCELERATION_STRUCTURE_WRITE,
        image_layout: Undefined,
    }

    /// Only use this for prototyping or debugging please. 🔫 Please. 🔫
    GENERAL {
        stage_mask: ALL_COMMANDS,
        access_mask: MEMORY_READ | MEMORY_WRITE,
        image_layout: General,
    }
}

impl AccessTypes {
    /// Returns the stage mask of these types of accesses.
    #[inline]
    #[must_use]
    pub const fn stage_mask(&self) -> PipelineStages {
        self.stage_mask
    }

    /// Returns the access mask of these types of accesses.
    #[inline]
    #[must_use]
    pub const fn access_mask(&self) -> AccessFlags {
        self.access_mask
    }

    /// Returns the image layout of these types of accesses.
    #[inline]
    #[must_use]
    pub const fn image_layout(&self, layout_type: ImageLayoutType) -> ImageLayout {
        if layout_type.is_optimal() {
            self.image_layout
        } else {
            ImageLayout::General
        }
    }

    /// Returns the union of `self` and `other`.
    ///
    /// If the image layouts don't match, [`ImageLayout::General`] is used.
    #[inline]
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        AccessTypes {
            stage_mask: self.stage_mask.union(other.stage_mask),
            access_mask: self.access_mask.union(other.access_mask),
            image_layout: if self.image_layout as i32 == other.image_layout as i32 {
                self.image_layout
            } else {
                ImageLayout::General
            },
        }
    }

    pub(crate) const fn are_valid_buffer_access_types(self) -> bool {
        const VALID_STAGE_FLAGS: PipelineStages = PipelineStages::DRAW_INDIRECT
            .union(PipelineStages::VERTEX_SHADER)
            .union(PipelineStages::TESSELLATION_CONTROL_SHADER)
            .union(PipelineStages::TESSELLATION_EVALUATION_SHADER)
            .union(PipelineStages::GEOMETRY_SHADER)
            .union(PipelineStages::FRAGMENT_SHADER)
            .union(PipelineStages::COMPUTE_SHADER)
            .union(PipelineStages::ALL_COMMANDS)
            .union(PipelineStages::COPY)
            .union(PipelineStages::INDEX_INPUT)
            .union(PipelineStages::VERTEX_ATTRIBUTE_INPUT)
            .union(PipelineStages::VIDEO_DECODE)
            .union(PipelineStages::VIDEO_ENCODE)
            .union(PipelineStages::ACCELERATION_STRUCTURE_BUILD)
            .union(PipelineStages::RAY_TRACING_SHADER)
            .union(PipelineStages::TASK_SHADER)
            .union(PipelineStages::MESH_SHADER)
            .union(PipelineStages::ACCELERATION_STRUCTURE_COPY);
        const VALID_ACCESS_FLAGS: AccessFlags = AccessFlags::INDIRECT_COMMAND_READ
            .union(AccessFlags::INDEX_READ)
            .union(AccessFlags::VERTEX_ATTRIBUTE_READ)
            .union(AccessFlags::UNIFORM_READ)
            .union(AccessFlags::TRANSFER_READ)
            .union(AccessFlags::TRANSFER_WRITE)
            .union(AccessFlags::MEMORY_READ)
            .union(AccessFlags::MEMORY_WRITE)
            .union(AccessFlags::SHADER_SAMPLED_READ)
            .union(AccessFlags::SHADER_STORAGE_READ)
            .union(AccessFlags::SHADER_STORAGE_WRITE)
            .union(AccessFlags::VIDEO_DECODE_READ)
            .union(AccessFlags::VIDEO_ENCODE_WRITE)
            .union(AccessFlags::ACCELERATION_STRUCTURE_READ)
            .union(AccessFlags::ACCELERATION_STRUCTURE_WRITE)
            .union(AccessFlags::SHADER_BINDING_TABLE_READ);

        VALID_STAGE_FLAGS.contains(self.stage_mask)
            && VALID_ACCESS_FLAGS.contains(self.access_mask)
            && matches!(
                self.image_layout,
                ImageLayout::Undefined
                    | ImageLayout::General
                    | ImageLayout::ShaderReadOnlyOptimal
                    | ImageLayout::TransferSrcOptimal
                    | ImageLayout::TransferDstOptimal,
            )
    }

    pub(crate) const fn are_valid_image_access_types(self) -> bool {
        !matches!(self.image_layout, ImageLayout::Undefined)
    }
}

impl BitOr for AccessTypes {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl BitOrAssign for AccessTypes {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.union(rhs);
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
