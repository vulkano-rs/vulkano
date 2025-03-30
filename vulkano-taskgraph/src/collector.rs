//! Garbage collection facilities.

use crate::{
    assert_unsafe_precondition,
    descriptor_set::{
        AccelerationStructureId, SampledImageId, SamplerId, StorageBufferId, StorageImageId,
    },
    resource::{Flight, Resources},
    Id,
};
use concurrent_slotmap::epoch;
use smallvec::SmallVec;
use std::{
    any::Any,
    cell::UnsafeCell,
    fmt,
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::ControlFlow,
    ptr,
    sync::{
        atomic::{
            self, AtomicU32, AtomicU64,
            Ordering::{Acquire, Relaxed, Release, SeqCst},
        },
        Arc,
    },
};
use virtual_buffer::{align_up, page_size, Allocation};
use vulkano::{buffer::Buffer, image::Image, swapchain::Swapchain};

/// A batch of deferred functions to be enqueued together.
///
/// You should batch as many deferred functions together as you can for optimal performance.
///
/// This type is created by the [`create_deferred_batch`] method on [`Resources`].
///
/// [`create_deferred_batch`]: Resources::create_deferred_batch
pub struct DeferredBatch<'a> {
    resources: &'a Resources,
    node_index: u32,
    node_ptr: *mut Node,
    deferreds: *mut Vec<Deferred>,
    guard: epoch::Guard<'a>,
}

impl<'a> DeferredBatch<'a> {
    pub(crate) fn new(resources: &'a Resources) -> Self {
        let guard = resources.pin();
        let garbage_queue = resources.garbage_queue();
        let node_index = garbage_queue.node_allocator().allocate(&guard);
        let node_ptr = unsafe { garbage_queue.nodes().get_unchecked_raw(node_index) }.cast_mut();
        let deferreds = UnsafeCell::raw_get(unsafe { ptr::addr_of_mut!((*node_ptr).deferreds) });

        DeferredBatch {
            resources,
            node_index,
            node_ptr,
            deferreds,
            guard,
        }
    }

    /// Defers the destruction of the buffer corresponding to `id`.
    ///
    /// # Panics
    ///
    /// - Panics if `id` is not a valid physical resource ID.
    #[inline]
    pub fn destroy_buffer(&mut self, id: Id<Buffer>) -> &mut Self {
        self.resources
            .invalidate_buffer(id, &self.guard)
            .expect("invalid buffer ID");

        self.defer(move |resources| {
            // SAFETY: We invalidated the resource above, and by our own invariant, the deferred
            // function must have been called after the global epoch was advanced at least 2 steps
            // and all flights that use the resource have passed.
            unsafe { resources.remove_buffer_unchecked(id) };
        })
    }

    /// Defers the destruction of the image corresponding to `id`.
    ///
    /// # Panics
    ///
    /// - Panics if `id` is not a valid physical resource ID.
    #[inline]
    pub fn destroy_image(&mut self, id: Id<Image>) -> &mut Self {
        self.resources
            .invalidate_image(id, &self.guard)
            .expect("invalid image ID");

        self.defer(move |resources| {
            // SAFETY: We invalidated the resource above, and by our own invariant, the deferred
            // function must have been called after the global epoch was advanced at least 2 steps
            // and all flights that use the resource have passed.
            unsafe { resources.remove_image_unchecked(id) };
        })
    }

    /// Defers the destruction of the swapchain corresponding to `id`.
    ///
    /// # Panics
    ///
    /// - Panics if `id` is not a valid physical resource ID.
    #[inline]
    pub fn destroy_swapchain(&mut self, id: Id<Swapchain>) -> &mut Self {
        self.resources
            .invalidate_swapchain(id, &self.guard)
            .expect("invalid swapchain ID");

        self.defer(move |resources| {
            // SAFETY: We invalidated the resource above, and by our own invariant, the deferred
            // function must have been called after the global epoch was advanced at least 2 steps
            // and all flights that use the resource have passed.
            unsafe { resources.remove_swapchain_unchecked(id) };
        })
    }

    /// Defers the destruction of the sampler descriptor corresponding to `id`.
    ///
    /// # Panics
    ///
    /// - Panics if the `Resources` collection wasn't created with a bindless context.
    /// - Panics if `id` is invalid.
    #[inline]
    pub fn destroy_sampler(&mut self, id: SamplerId) -> &mut Self {
        let bcx = self
            .resources
            .bindless_context()
            .expect("no bindless context");

        bcx.global_set()
            .invalidate_sampler(id, &self.guard)
            .expect("invalid sampler ID");

        self.defer(move |resources| {
            let bcx = resources.bindless_context().unwrap();

            // SAFETY: We invalidated the resource above, and by our own invariant, the deferred
            // function must have been called after the global epoch was advanced at least 2 steps
            // and all flights that use the descriptor have passed.
            unsafe { bcx.global_set().remove_sampler_unchecked(id) };
        })
    }

    /// Defers the destruction of the sampled image descriptor corresponding to `id`.
    ///
    /// # Panics
    ///
    /// - Panics if the `Resources` collection wasn't created with a bindless context.
    /// - Panics if `id` is invalid.
    #[inline]
    pub fn destroy_sampled_image(&mut self, id: SampledImageId) -> &mut Self {
        let bcx = self
            .resources
            .bindless_context()
            .expect("no bindless context");

        bcx.global_set()
            .invalidate_sampled_image(id, &self.guard)
            .expect("invalid sampled image ID");

        self.defer(move |resources| {
            let bcx = resources.bindless_context().unwrap();

            // SAFETY: We invalidated the resource above, and by our own invariant, the deferred
            // function must have been called after the global epoch was advanced at least 2 steps
            // and all flights that use the descriptor have passed.
            unsafe { bcx.global_set().remove_sampled_image_unchecked(id) };
        })
    }

    /// Defers the destruction of the storage image descriptor corresponding to `id`.
    ///
    /// # Panics
    ///
    /// - Panics if the `Resources` collection wasn't created with a bindless context.
    /// - Panics if `id` is invalid.
    #[inline]
    pub fn destroy_storage_image(&mut self, id: StorageImageId) -> &mut Self {
        let bcx = self
            .resources
            .bindless_context()
            .expect("no bindless context");

        bcx.global_set()
            .invalidate_storage_image(id, &self.guard)
            .expect("invalid storage image ID");

        self.defer(move |resources| {
            let bcx = resources.bindless_context().unwrap();

            // SAFETY: We invalidated the resource above, and by our own invariant, the deferred
            // function must have been called after the global epoch was advanced at least 2 steps
            // and all flights that use the descriptor have passed.
            unsafe { bcx.global_set().remove_storage_image_unchecked(id) };
        })
    }

    /// Defers the destruction of the storage buffer descriptor corresponding to `id`.
    ///
    /// # Panics
    ///
    /// - Panics if the `Resources` collection wasn't created with a bindless context.
    /// - Panics if `id` is invalid.
    #[inline]
    pub fn destroy_storage_buffer(&mut self, id: StorageBufferId) -> &mut Self {
        let bcx = self
            .resources
            .bindless_context()
            .expect("no bindless context");

        bcx.global_set()
            .invalidate_storage_buffer(id, &self.guard)
            .expect("invalid storage buffer ID");

        self.defer(move |resources| {
            let bcx = resources.bindless_context().unwrap();

            // SAFETY: We invalidated the resource above, and by our own invariant, the deferred
            // function must have been called after the global epoch was advanced at least 2 steps
            // and all flights that use the descriptor have passed.
            unsafe { bcx.global_set().remove_storage_buffer_unchecked(id) };
        })
    }

    /// Defers the destruction of the acceleration structure descriptor corresponding to `id`.
    ///
    /// # Panics
    ///
    /// - Panics if the `Resources` collection wasn't created with a bindless context.
    /// - Panics if `id` is invalid.
    #[inline]
    pub fn destroy_acceleration_structure(&mut self, id: AccelerationStructureId) -> &mut Self {
        let bcx = self
            .resources
            .bindless_context()
            .expect("no bindless context");

        bcx.global_set()
            .invalidate_acceleration_structure(id, &self.guard)
            .expect("invalid acceleration structure ID");

        self.defer(move |resources| {
            let bcx = resources.bindless_context().unwrap();

            // SAFETY: We invalidated the resource above, and by our own invariant, the deferred
            // function must have been called after the global epoch was advanced at least 2 steps
            // and all flights that use the descriptor have passed.
            unsafe { bcx.global_set().remove_acceleration_structure_unchecked(id) };
        })
    }

    /// Defers the destruction of the given `object`.
    #[inline]
    pub fn destroy_object(&mut self, object: impl Send + 'static) -> &mut Self {
        self.defer(|_| drop(object))
    }

    /// Defers the given function.
    #[inline]
    pub fn defer(&mut self, f: impl FnOnce(&Resources) + Send + 'static) -> &mut Self {
        if cfg!(debug_assertions) {
            // This whole block is conditional on debug assertions because this adds a word of
            // memory overhead and the inline storage of `Deferred` is limited to 3 words.

            let addr = ptr::from_ref(self.resources) as usize;

            self.deferreds_mut().push(Deferred::new(move |resources| {
                assert_unsafe_precondition!(
                    ptr::from_ref(resources) as usize == addr,
                    "`Deferred::call_in_place` must be called with the same `Resources` collection
                    as the one for which the deferred function was issued",
                );

                f(resources);
            }));
        } else {
            self.deferreds_mut().push(Deferred::new(f));
        }

        self
    }

    pub(crate) fn deferreds_mut(&mut self) -> &mut Vec<Deferred> {
        // SAFETY: By our own invariant, `self.deferreds` is valid so long as one of the `enqueue`
        // methods hasn't been called.
        unsafe { &mut *self.deferreds }
    }

    /// Enqueues the deferred functions for after all flights have been waited on.
    pub fn enqueue(self) {
        // See the `Drop` implementation.
    }

    /// Enqueues the deferred functions for after the flights corresponding to the given
    /// `flight_ids` have been waited on.
    ///
    /// # Safety
    ///
    /// - There must be no other flights that use any resource or descriptor that is being
    ///   destroyed by the deferred functions.
    ///
    /// # Panics
    ///
    /// - Panics if `flight_ids` contains an invalid ID.
    /// - Panics if `flight_ids` panics.
    ///
    /// # Leaks
    ///
    /// If the method panics for any reason (including if the provided iterator panics), then the
    /// deferred functions will never be called. For any destruction that was deferred, the
    /// corresponding resource or descriptor will be leaked.
    pub unsafe fn enqueue_with_flights(self, flight_ids: impl IntoIterator<Item = Id<Flight>>) {
        // It is imperative that this be an iterator as it's lazy, because `flight.current_frame()`
        // must only be called **after** the atomic fence in `Self::enqueue_inner`.
        let frames = flight_ids.into_iter().map(|flight_id| {
            let flight =
                // SAFETY: We own an `epoch::Guard`.
                unsafe { self.resources.flight_unprotected(flight_id) }.expect("invalid flight ID");

            (flight_id, flight.current_frame())
        });

        // SAFETY:
        // * We own `self`, which ensures that this method isn't called again.
        // * We have wrapped `self` in a `ManuallyDrop` to ensure that the `Drop` implementation
        //   doesn't call this method.
        // * The caller must ensure that `flight_ids` constitutes the correct set of flights for our
        //   deferred functions.
        unsafe { ManuallyDrop::new(self).enqueue_inner(frames) }
    }

    /// Enqueues the deferred functions for after the given `frames` have been waited on.
    ///
    /// `frames` provides mappings from a flight ID to the corresponding [frame] that should be
    /// waited on.
    ///
    /// # Safety
    ///
    /// - There must be no other flights that use any resource or descriptor that is being
    ///   destroyed by the deferred functions.
    /// - The given flights must not use any resource or descriptor that is being destroyed by the
    ///   deferred functions past the corresponding frame.
    ///
    /// # Panics
    ///
    /// - Panics if `frames` contains an invalid ID.
    /// - Panics if `frames` panics.
    ///
    /// # Leaks
    ///
    /// If the method panics for any reason (including if the provided iterator panics), then the
    /// deferred functions will never be called. For any destruction that was deferred, the
    /// corresponding resource or descriptor will be leaked.
    pub unsafe fn enqueue_with_frames(self, frames: impl IntoIterator<Item = (Id<Flight>, u64)>) {
        let frames = frames.into_iter().inspect(|&(flight_id, _)| {
            // SAFETY: We own an `epoch::Guard`.
            unsafe { self.resources.flight_unprotected(flight_id) }.expect("invalid flight ID");
        });

        // SAFETY:
        // * We own `self`, which ensures that this method isn't called again.
        // * We have wrapped `self` in a `ManuallyDrop` to ensure that the `Drop` implementation
        //   doesn't call this method.
        // * The caller must ensure that `frames` constitutes the correct set of frames for our
        //   deferred functions.
        unsafe { ManuallyDrop::new(self).enqueue_inner(frames) }
    }

    unsafe fn enqueue_inner(&mut self, frames: impl IntoIterator<Item = (Id<Flight>, u64)>) {
        struct ClearGuard<'a, 'b>(&'a mut DeferredBatch<'b>);

        impl Drop for ClearGuard<'_, '_> {
            #[cold]
            fn drop(&mut self) {
                self.0.deferreds_mut().clear();

                let node_allocator = self.0.resources.garbage_queue().node_allocator();

                // SAFETY: By our own invariant, the node denoted by `self.node_index` must have
                // been allocated by us, and since we haven't pushed it to any garbage queue, the
                // node is still unlinked and therefore safe to deallocate.
                unsafe { node_allocator.deallocate(self.0.node_index, &self.0.guard) };
            }
        }

        let node_ptr = self.node_ptr;
        let this = ClearGuard(self);

        // This fence ensures that the following loads of the global epoch and the frame counters
        // synchronize with the corresponding stores. `Acquire` is not strong enough because it
        // would allow reading older values, which would cause us to collect prematurely.
        atomic::fence(SeqCst);

        // SAFETY: By our own invariant, the node denoted by `self.node_ptr` must have been
        // allocated by us, which means that no other threads can be accessing this node's data
        // while we haven't pushed it to a garbage queue.
        unsafe { (*node_ptr).epoch = this.0.resources.global().epoch() };
        unsafe { (*node_ptr).frames.clear() };
        unsafe { (*node_ptr).frames.extend(frames) };

        mem::forget(this);

        if let &[(flight_id, _)] = unsafe { (*node_ptr).frames.as_slice() } {
            // SAFETY:
            // * We own an `epoch::Guard`.
            // * The caller must ensure that `frames` contained valid flight IDs when extending the
            //   node's `frames` above. Since we have owned an `epoch::Guard` from then until now,
            //   the flight cannot have been dropped yet even if it were to be removed between then
            //   and now.
            let flight = unsafe { self.resources.flight_unchecked_unprotected(flight_id) };

            let garbage_queue = flight.garbage_queue();

            // SAFETY:
            // * By our own invariant, the node denoted by `self.node_index` must have been
            //   allocated by us.
            // * The caller must ensure that the node's data is not accessed mutably again.
            // * The caller must ensure that this method is not called again.
            // * The caller must ensure that `frames` constitutes the correct set of frames for our
            //   deferred functions.
            unsafe { garbage_queue.push(self.node_index, &self.guard) };
        } else {
            let garbage_queue = self.resources.garbage_queue();

            // SAFETY: Same as the `push` above.
            unsafe { garbage_queue.push(self.node_index, &self.guard) };
        }
    }
}

impl Drop for DeferredBatch<'_> {
    /// Dropping a `DeferredBatch` does the same thing as calling [`enqueue`].
    ///
    /// [`enqueue`]: Self::enqueue
    fn drop(&mut self) {
        // SAFETY: We own an `epoch::Guard`.
        let flights = unsafe { self.resources.flights_unprotected() };

        // It is imperative that this be an iterator as it's lazy, because `flight.current_frame()`
        // must only be called **after** the atomic fence in `Self::enqueue_inner`.
        let frames = flights.map(|(flight_id, flight)| (flight_id, flight.current_frame()));

        // SAFETY:
        // * We're dropping `self`, which ensures that this method isn't called again.
        // * `frames` includes the current frame of every flight, so there can be no flight that
        //   uses any of the resources/descriptors being destroyed that isn't waited on.
        unsafe { self.enqueue_inner(frames) };
    }
}

/// Collects garbage.
///
/// You may want to implement this trait if you want to control where collection takes place, for
/// example on a background thread.
pub trait Collector: Any + Send + Sync {
    /// Collects garbage.
    ///
    /// The method is expected to empty the `deferreds` vector in one way or another. You can drain
    /// the vector, clear it, or `mem::swap` it; it doesn't matter. However, it would be very cash
    /// money of the method if it could preserve the capacity of the vector, because it is going to
    /// be recycled. If you're sending the vector to another thread, you could consider having a
    /// return channel for emptied vectors.
    ///
    /// You must call each deferred function, otherwise you will leak.
    ///
    /// # Safety
    ///
    /// - `ccx.resources()` must be the same collection as the one for which the deferred functions
    ///   were issued.
    /// - `ccx.frames()` must be the frames for which the deferred functions should wait on and
    ///   those frames must have been waited on.
    unsafe fn collect(&self, deferreds: &mut Vec<Deferred>, ccx: &mut CollectorContext<'_>);
}

impl fmt::Debug for dyn Collector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Collector").finish_non_exhaustive()
    }
}

/// The default garbage collector implementation.
///
/// This collector calls each deferred function in place on the current thread.
#[derive(Clone, Copy, Debug, Default)]
pub struct DefaultCollector;

impl Collector for DefaultCollector {
    unsafe fn collect(&self, deferreds: &mut Vec<Deferred>, ccx: &mut CollectorContext<'_>) {
        struct ClearGuard<'a>(&'a mut Vec<Deferred>);

        impl Drop for ClearGuard<'_> {
            fn drop(&mut self) {
                self.0.clear();
            }
        }

        for deferred in ClearGuard(deferreds).0.iter_mut() {
            // SAFETY:
            // * The deferred cannot be called again because we call it once and ensure that the
            //   `deferreds` vector gets cleared afterward in all cases.
            // * The caller must ensure that `ccx.resources()` is the collection for which the
            //   `deferreds` were issued.
            unsafe { deferred.call_in_place(ccx.resources()) };
        }
    }
}

/// The garbage collection context.
///
/// This gives [`Collector::collect`] access to the `Resources` collection for which the deferred
/// functions have been issued, as well as the flight IDs and associated [frames] which have been
/// waited on.
pub struct CollectorContext<'a> {
    resources: &'a Resources,
    frames: &'a [(Id<Flight>, u64)],
}

impl<'a> CollectorContext<'a> {
    /// Returns the `Resources` collection for which the deferred functions were issued.
    #[inline]
    pub fn resources(&self) -> &'a Resources {
        self.resources
    }

    /// Returns a map from the flight ID of each flight that was waited on to the corresponding
    /// [frame] that was waited on.
    #[inline]
    pub fn frames(&self) -> &'a [(Id<Flight>, u64)] {
        self.frames
    }
}

const DATA_WORDS: usize = 3;

/// A deferred function.
///
/// This is essentially a `dyn FnOnce(&Resources)` except sized and stored inline.
pub struct Deferred {
    call: unsafe fn(*mut (), &Resources),
    data: MaybeUninit<Data>,
}

type Data = [usize; DATA_WORDS];

// SAFETY: `Deferred::new` requires that the function is `Send`, so it's safe to send the
// `Deferred` to another thread as well.
unsafe impl Send for Deferred {}

// SAFETY: We only give owning or mutable access to the contained function, so it's safe to share a
// reference to the `Deferred` as it can't be used for anything.
unsafe impl Sync for Deferred {}

impl Deferred {
    pub(crate) fn destroy(object: impl Send + 'static) -> Self {
        Self::new(|_| drop(object))
    }

    pub(crate) fn new<F: FnOnce(&Resources) + Send + 'static>(f: F) -> Self {
        if size_of::<F>() <= size_of::<Data>() && align_of::<F>() <= align_of::<Data>() {
            let mut data = MaybeUninit::<Data>::uninit();

            // SAFETY: The pointer is valid for writes and we checked that `Data` has a layout that
            // can fit `F`.
            unsafe { data.as_mut_ptr().cast::<F>().write(f) };

            Deferred {
                // SAFETY: The caller of `call` must ensure that `data` is our same data and that
                // `resources` is the collection for which the deferred function was issued.
                call: |data, resources| unsafe { data.cast::<F>().read() }(resources),
                data,
            }
        } else {
            const { assert!(size_of::<Box<F>>() <= size_of::<Data>()) };
            const { assert!(align_of::<Box<F>>() <= align_of::<Data>()) };

            let mut data = MaybeUninit::<Data>::uninit();

            // SAFETY: The pointer is valid for writes and we checked that `Data` has a layout that
            // can fit `Box<F>`.
            unsafe { data.as_mut_ptr().cast::<Box<F>>().write(Box::new(f)) };

            Deferred {
                // SAFETY: The caller of `call` must ensure that `data` is our same data and that
                // `resources` is the collection for which the deferred function was issued.
                call: |data, resources| unsafe { data.cast::<Box<F>>().read() }(resources),
                data,
            }
        }
    }

    /// Calls the deferred function.
    ///
    /// # Safety
    ///
    /// - `resources` must be the collection for which this deferred function was issued.
    #[inline]
    pub unsafe fn call(mut self, resources: &Resources) {
        // SAFETY:
        // * We own the `Deferred`, which means that it cannot be called again.
        // * The caller must ensure that `resources` is the collection for which this deferred
        //   function was issued.
        unsafe { self.call_in_place(resources) };
    }

    /// Calls the deferred function in place.
    ///
    /// This is a bit easier on the optimizer, at the expense of being wildly unsound if called
    /// more than once. If you call this method more than once, you **will** get double-frees.
    ///
    /// See also [`call`] for the safer variant.
    ///
    /// # Safety
    ///
    /// - The deferred function must not have been called before.
    /// - `resources` must be the collection for which this deferred function was issued.
    ///
    /// [`call`]: Self::call
    #[inline]
    pub unsafe fn call_in_place(&mut self, resources: &Resources) {
        // SAFETY:
        // * The constructor of `Deferred` must ensure that `self.call` is safe to call with a
        //   pointer to `self.data`.
        // * The caller must ensure that the function is only called once.
        // * The caller must ensure that `resources` is the collection for which this deferred
        //   function was issued.
        unsafe { (self.call)(self.data.as_mut_ptr().cast(), resources) };
    }
}

impl fmt::Debug for Deferred {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Deferred").finish_non_exhaustive()
    }
}

pub(crate) struct GlobalQueue {
    inner: Queue,
}

impl GlobalQueue {
    pub(crate) fn new(global: epoch::GlobalHandle) -> Self {
        GlobalQueue {
            inner: Queue::new(NodeVec::new(), global),
        }
    }

    fn node_allocator(&self) -> &NodeAllocator {
        &self.inner.node_allocator
    }

    fn nodes(&self) -> &NodeVec {
        self.inner.nodes()
    }

    pub(crate) fn register_local(&self) -> LocalQueue {
        LocalQueue::new(
            self.node_allocator().nodes.clone(),
            self.node_allocator().global.clone(),
        )
    }

    unsafe fn push(&self, node_index: u32, guard: &epoch::Guard<'_>) {
        // SAFETY: Enforced by the caller.
        unsafe { self.inner.push(node_index, guard) };
    }

    pub(crate) unsafe fn collect(&self, resources: &Resources, guard: &epoch::Guard<'_>) {
        let predicate = |frames: &[(Id<Flight>, u64)]| {
            for &(flight_id, frame) in frames {
                // SAFETY: We own an `epoch::Guard`.
                let Ok(flight) = (unsafe { resources.flight_unprotected(flight_id) }) else {
                    continue;
                };

                if !flight.is_frame_complete(frame) {
                    return ControlFlow::Continue(false);
                }
            }

            ControlFlow::Continue(true)
        };

        // SAFETY:
        // * The `predicate` correctly collects only if all frames are completed.
        // * The caller must ensure that `resources` is the collection for which the deferred
        //   functions were issued.
        // * The caller must ensure that `guard.global()` is that of `resources.global()`.
        unsafe { self.inner.collect(predicate, resources, guard) };
    }

    pub(crate) unsafe fn drop(&self, resources: &Resources) {
        // SAFETY: Enforced by the caller.
        unsafe { self.inner.drop(resources) };
    }
}

impl fmt::Debug for GlobalQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlobalQueue").finish_non_exhaustive()
    }
}

pub(crate) struct LocalQueue {
    inner: Queue,
}

impl LocalQueue {
    fn new(nodes: Arc<NodeVec>, global: epoch::GlobalHandle) -> Self {
        LocalQueue {
            inner: Queue::new(nodes, global),
        }
    }

    unsafe fn push(&self, node_index: u32, guard: &epoch::Guard<'_>) {
        // SAFETY: Enforced by the caller.
        unsafe { self.inner.push(node_index, guard) };
    }

    pub(crate) unsafe fn collect(
        &self,
        resources: &Resources,
        flight: &Flight,
        guard: &epoch::Guard<'_>,
    ) {
        let predicate = |frames: &[(Id<Flight>, u64)]| {
            if let &[(_, frame)] = frames {
                if flight.is_frame_complete(frame) {
                    ControlFlow::Continue(true)
                } else {
                    ControlFlow::Break(())
                }
            } else {
                unreachable!();
            }
        };

        // SAFETY:
        // * The `predicate` correctly collects only if the frame is completed.
        // * The caller must ensure that `resources` is the collection for which the deferred
        //   functions were issued.
        // * The caller must ensure that `guard.global()` is that of `resources.global()`.
        unsafe { self.inner.collect(predicate, resources, guard) };
    }

    pub(crate) unsafe fn drop(&self, resources: &Resources) {
        // SAFETY: Enforced by the caller.
        unsafe { self.inner.drop(resources) };
    }
}

impl fmt::Debug for LocalQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalQueue").finish_non_exhaustive()
    }
}

const NIL: u32 = u32::MAX & !DELETED_BIT;
const DELETED_BIT: u32 = 1 << 31;

struct Queue {
    head: u32,
    node_allocator: NodeAllocator,
}

impl Queue {
    fn new(nodes: Arc<NodeVec>, global: epoch::GlobalHandle) -> Self {
        let sentinel_index = nodes.push();
        let sentinel_node = unsafe { nodes.get_unchecked(sentinel_index) };
        sentinel_node.next.store(NIL, Relaxed);

        Queue {
            head: sentinel_index,
            node_allocator: NodeAllocator::new(nodes, global),
        }
    }

    fn nodes(&self) -> &NodeVec {
        &self.node_allocator.nodes
    }

    unsafe fn push(&self, index: u32, guard: &epoch::Guard<'_>) {
        // SAFETY: The caller must ensure that `index` is valid and that the node is not mutated
        // again.
        let node = unsafe { self.nodes().get_unchecked(index) };

        node.next.store(NIL, Relaxed);

        // SAFETY: `self.head` is always valid.
        let head_node = unsafe { self.nodes().get_unchecked(self.head) };

        let mut prev_node = head_node;
        let mut curr_index = prev_node.next.load(Acquire);

        loop {
            debug_assert!(!is_deleted(curr_index));

            if curr_index == NIL {
                match prev_node
                    .next
                    .compare_exchange(NIL, index, Release, Acquire)
                {
                    Ok(_) => {
                        // We successfully linked `node` after the tail node.
                        break;
                    }
                    Err(new_curr_index) if is_deleted(new_curr_index) => {
                        // Another thread deleted `prev_node`. We have to restart from the
                        // beginning, as our snapshot of the queue is now inconsistent.
                        prev_node = head_node;
                        curr_index = prev_node.next.load(Acquire);
                        continue;
                    }
                    Err(new_curr_index) => {
                        // Another thread linked another node after the tail node. Try to move onto
                        // the node in the next iteration and try again.
                        curr_index = new_curr_index;
                        continue;
                    }
                }
            }

            // SAFETY: We always push indices of existing nodes into the queue and the nodes vector
            // never shrinks, so the index must have staid in bounds.
            let curr_node = unsafe { self.nodes().get_unchecked(curr_index) };

            let next_index = curr_node.next.load(Acquire);

            // Try to unlink any deleted nodes we encounter. We cannot link a node after a deleted
            // node, and we also cannot unlink a node following a deleted node.
            if is_deleted(next_index) {
                match prev_node.next.compare_exchange(
                    curr_index,
                    next_index & !DELETED_BIT,
                    Release,
                    Relaxed,
                ) {
                    Ok(_) => {
                        // Our thread unlinked `curr_node`. Deallocate `curr_node` and move onto the
                        // next node.

                        // SAFETY: We successfully unlinked the node from the queue such that it is
                        // not reachable going forward and no other thread can be deallocating this
                        // same node.
                        unsafe { self.node_allocator.deallocate(curr_index, guard) };

                        curr_index = next_index & !DELETED_BIT;
                        continue;
                    }
                    Err(new_curr_index) if is_deleted(new_curr_index) => {
                        // Another thread deleted `prev_node`. We have to restart from the
                        // beginning, as our snapshot of the queue is now inconsistent.
                        prev_node = head_node;
                        curr_index = prev_node.next.load(Acquire);
                        continue;
                    }
                    Err(_) => {
                        // Another thread unlinked `curr_node`. Move onto the next node without
                        // deallocating `curr_node`.
                        curr_index = next_index & !DELETED_BIT;
                        continue;
                    }
                }
            }

            prev_node = curr_node;
            curr_index = next_index;
        }
    }

    unsafe fn collect(
        &self,
        mut collect_predicate: impl FnMut(&[(Id<Flight>, u64)]) -> ControlFlow<(), bool>,
        resources: &Resources,
        guard: &epoch::Guard<'_>,
    ) {
        let epoch = self.node_allocator.global.epoch();

        // SAFETY: `self.head` is always valid.
        let head_node = unsafe { self.nodes().get_unchecked(self.head) };

        let mut prev_node = head_node;
        let mut curr_index = prev_node.next.load(Acquire);

        loop {
            debug_assert!(!is_deleted(curr_index));

            if curr_index == NIL {
                break;
            }

            // SAFETY: We always push indices of existing nodes into the queue and the nodes vector
            // never shrinks, so the index must have staid in bounds.
            let curr_node = unsafe { self.nodes().get_unchecked(curr_index) };

            let next_index = curr_node.next.load(Acquire);

            // Try to unlink any deleted nodes we encounter. We cannot collect and/or delete an
            // already deleted node, and we also cannot unlink a node following a deleted node.
            if is_deleted(next_index) {
                match prev_node.next.compare_exchange(
                    curr_index,
                    next_index & !DELETED_BIT,
                    Release,
                    Relaxed,
                ) {
                    Ok(_) => {
                        // Our thread unlinked `curr_node`. Deallocate `curr_node` and move onto the
                        // next node.

                        // SAFETY: We successfully unlinked the node from the queue such that it is
                        // not reachable going forward and no other thread can be deallocating this
                        // same node.
                        unsafe { self.node_allocator.deallocate(curr_index, guard) };

                        curr_index = next_index & !DELETED_BIT;
                        continue;
                    }
                    Err(new_curr_index) if is_deleted(new_curr_index) => {
                        // Another thread deleted `prev_node`. We have to restart from the
                        // beginning, as our snapshot of the queue is now
                        // inconsistent.
                        prev_node = head_node;
                        curr_index = prev_node.next.load(Acquire);
                        continue;
                    }
                    Err(_) => {
                        // Another thread unlinked `curr_node`. Move onto the next node without
                        // deallocating `curr_node`.
                        curr_index = next_index & !DELETED_BIT;
                        continue;
                    }
                }
            }

            // We must only collect if at least 2 epochs have passed, but since the epoch advances
            // in steps of 2, that amounts to 4. We also don't continue unless this holds, because a
            // pushed node should be tagged with an epoch at least that of the last pushed one, so
            // we know that we cannot collect any of the following nodes either.
            if !(epoch.wrapping_sub(curr_node.epoch) >= 4) {
                break;
            }

            match collect_predicate(&curr_node.frames) {
                ControlFlow::Continue(false) => {
                    // The node cannot be collected but there might be more collectable nodes. Move
                    // onto the next node.
                    prev_node = curr_node;
                    curr_index = next_index;
                    continue;
                }
                ControlFlow::Continue(true) => {
                    // The node can be collected and there might be more collectable nodes. Collect
                    // the node and move onto the next.
                }
                ControlFlow::Break(()) => {
                    // The node cannot be collected and there cannot be more collectable nodes.
                    break;
                }
            }

            match curr_node.next.compare_exchange(
                next_index,
                next_index | DELETED_BIT,
                Release,
                Relaxed,
            ) {
                Ok(_) => {
                    // Our thread deleted `curr_node`. Collect it and move onto the next node.

                    // SAFETY: We successfully marked the node as deleted such that no other thread
                    // can be collecting this same node.
                    let deferreds = unsafe { &mut *curr_node.deferreds.get() };

                    let mut ccx = CollectorContext {
                        resources,
                        frames: &curr_node.frames,
                    };

                    // SAFETY: The caller must ensure that `resources` is the correct collection
                    // for the `deferreds`. `ccx.frames()` are the frames for which the deferred
                    // functions should wait. The caller must ensure that those frames have been
                    // waited on.
                    unsafe { resources.collector().collect(deferreds, &mut ccx) };

                    prev_node = curr_node;
                    curr_index = next_index;
                }
                Err(_) => {
                    // Another thread deleted `curr_node` or unlinked `prev_node`. We have to
                    // restart from the beginning, as our snapshot of the queue is now inconsistent.
                    prev_node = head_node;
                    curr_index = prev_node.next.load(Acquire);
                }
            }
        }
    }

    unsafe fn drop(&self, resources: &Resources) {
        // SAFETY: `self.head` is always valid.
        let head_node = unsafe { self.nodes().get_unchecked(self.head) };

        let mut curr_index = head_node.next.load(Relaxed);

        while curr_index != NIL {
            // SAFETY: We always push indices of existing nodes into the queue and the nodes vector
            // never shrinks, so the index must have staid in bounds.
            let curr_node = unsafe { self.nodes().get_unchecked(curr_index) };

            let next_index = curr_node.next.load(Relaxed);

            // The caller must ensure that we have exclusive access to the queue, so we can simply
            // skip over deleted nodes without unlinking them.
            if is_deleted(next_index) {
                curr_index = next_index & !DELETED_BIT;
                continue;
            }

            // The caller must ensure that we have exclusive access to the queue, so we don't need
            // to use `compare_exchange` or stronger orderings here.
            curr_node.next.store(next_index | DELETED_BIT, Relaxed);

            // SAFETY: We marked the node as deleted such that the following `Collector::collect`
            // call cannot reentrantly collect this same node.
            let deferreds = unsafe { &mut *curr_node.deferreds.get() };

            let mut ccx = CollectorContext {
                resources,
                frames: &curr_node.frames,
            };

            // SAFETY: The caller must ensure that `resources` is the correct collection for the
            // `deferreds`. `ccx.frames()` are the frames for which the deferred functions should
            // wait. The caller must ensure that those frames have been waited on.
            unsafe { resources.collector().collect(deferreds, &mut ccx) };

            curr_index = next_index;
        }
    }
}

fn is_deleted(index: u32) -> bool {
    index & DELETED_BIT != 0
}

#[repr(C, align(128))]
struct Node {
    next: AtomicU32,
    next_free: AtomicU32,
    epoch: u32,
    frames: SmallVec<[(Id<Flight>, u64); 4]>,
    deferreds: UnsafeCell<Vec<Deferred>>,
}

#[repr(C)]
struct NodeAllocator {
    nodes: Arc<NodeVec>,
    global: epoch::GlobalHandle,
    free_list_head: AtomicU32,
    _alignment: CacheAligned,
    free_list_queue: [AtomicU64; 2],
}

impl NodeAllocator {
    fn new(nodes: Arc<NodeVec>, global: epoch::GlobalHandle) -> Self {
        Self {
            nodes,
            global,
            free_list_head: AtomicU32::new(NIL),
            _alignment: CacheAligned,
            free_list_queue: [
                AtomicU64::new(u64::from(NIL)),
                AtomicU64::new(u64::from(NIL)),
            ],
        }
    }

    fn allocate(&self, _guard: &epoch::Guard<'_>) -> u32 {
        let mut head_index = self.free_list_head.load(Acquire);

        while head_index != NIL {
            // SAFETY: We always push indices of existing nodes into the free-list and the nodes
            // vector never shrinks, so the index must have staid in bounds.
            let head_ptr = unsafe { self.nodes.get_unchecked_raw(head_index) };

            // SAFETY: The pointer is valid and we make sure to only access the node's linkage, as
            // we are **not** allowed any kind of access to the node's data, as another thread might
            // have already allocated this node and started mutating the node's data.
            let next_index = unsafe { &(*head_ptr).next_free }.load(Acquire);

            match self
                .free_list_head
                .compare_exchange_weak(head_index, next_index, Release, Acquire)
            {
                Ok(_) => return head_index,
                Err(new_head_index) => head_index = new_head_index,
            }
        }

        self.nodes.push()
    }

    unsafe fn deallocate(&self, index: u32, _guard: &epoch::Guard<'_>) {
        // SAFETY: The caller must ensure that `index` is valid.
        let node = unsafe { self.nodes.get_unchecked(index) };

        // This fence ensures that the following load of the global epoch synchronizes with the
        // corresponding store. `Acquire` is not strong enough because it would allow reading older
        // values, which would cause us to collect prematurely.
        atomic::fence(SeqCst);

        let epoch = self.global.epoch();
        let queued_list = &self.free_list_queue[((epoch >> 1) & 1) as usize];
        let mut queued_state = queued_list.load(Acquire);

        loop {
            let queued_head_index = (queued_state & 0xFFFF_FFFF) as u32;
            let queued_epoch = (queued_state >> 32) as u32;
            let epoch_interval = epoch.wrapping_sub(queued_epoch);

            if epoch_interval == 0 {
                node.next_free.store(queued_head_index, Relaxed);

                let new_state = u64::from(index) | (u64::from(queued_epoch) << 32);

                match queued_list.compare_exchange_weak(queued_state, new_state, Release, Acquire) {
                    Ok(_) => break,
                    Err(new_state) => queued_state = new_state,
                }
            } else {
                let global_epoch_is_behind_queue = epoch_interval & (1 << 31) != 0;

                debug_assert!(!global_epoch_is_behind_queue);
                debug_assert!(epoch_interval >= 4);

                node.next_free.store(NIL, Relaxed);

                let new_state = u64::from(index) | (u64::from(epoch) << 32);

                match queued_list.compare_exchange_weak(queued_state, new_state, Release, Acquire) {
                    Ok(_) => {
                        // SAFETY: Having ended up here, the global epoch must have been advanced
                        // at least 2 steps from the last push into the queued list and we removed
                        // the list from the queue, which means that no other threads can be
                        // accessing any of the nodes in the list.
                        unsafe { self.collect_unchecked(queued_head_index) };

                        break;
                    }
                    Err(new_state) => queued_state = new_state,
                }
            }
        }
    }

    unsafe fn collect_unchecked(&self, queued_head_index: u32) {
        if queued_head_index == NIL {
            return;
        }

        let mut queued_tail_index = queued_head_index;
        let mut queued_tail_node;

        loop {
            // SAFETY: We always push indices of existing nodes into the free-lists and the nodes
            // vector never shrinks, therefore the index must have staid in bounds.
            queued_tail_node = unsafe { self.nodes.get_unchecked(queued_tail_index) };

            let next_free_index = queued_tail_node.next_free.load(Acquire);

            if next_free_index == NIL {
                break;
            }

            queued_tail_index = next_free_index;
        }

        let mut head_index = self.free_list_head.load(Acquire);

        loop {
            queued_tail_node.next_free.store(head_index, Relaxed);

            match self.free_list_head.compare_exchange_weak(
                head_index,
                queued_head_index,
                Release,
                Acquire,
            ) {
                Ok(_) => break,
                Err(new_head_index) => head_index = new_head_index,
            }
        }
    }
}

const SLAB_CAPACITY: u32 = 256;
const MAX_SLABS: u32 = 4096;
const MAX_CAPACITY: u32 = MAX_SLABS * SLAB_CAPACITY;

struct NodeVec {
    allocation: Allocation,
    capacity: AtomicU32,
    reserved_len: AtomicU32,
}

impl NodeVec {
    fn new() -> Arc<Self> {
        assert!(align_of::<Node>() <= page_size());

        let size = usize::try_from(MAX_CAPACITY)
            .unwrap()
            .checked_mul(size_of::<Node>())
            .unwrap();

        Arc::new(NodeVec {
            allocation: Allocation::new(size).unwrap(),
            capacity: AtomicU32::new(0),
            reserved_len: AtomicU32::new(0),
        })
    }

    fn as_ptr(&self) -> *const Node {
        self.allocation.ptr().cast()
    }

    fn push(&self) -> u32 {
        // This cannot overflow because our capacity can never exceed `MAX_CAPACITY`.
        let index = self.reserved_len.fetch_add(1, Relaxed);

        // The `Acquire` ordering synchronizes with the `Release` ordering in
        // `Self::reserve_for_push`, making sure that the new capacity is visible here.
        let capacity = self.capacity.load(Acquire);

        if index >= capacity {
            self.reserve_for_push(capacity);
        }

        // SAFETY: We ensured that the index is in bounds above.
        let ptr = unsafe { self.as_ptr().add(index as usize) }.cast_mut();

        // SAFETY: The pointer is valid and we incremented `reserved_len` above such that no other
        // thread can be writing to this same node.
        unsafe { (*ptr).frames = SmallVec::new() };
        unsafe { (*ptr).deferreds = UnsafeCell::new(Vec::new()) };

        index
    }

    #[inline(never)]
    fn reserve_for_push(&self, old_capacity: u32) {
        if old_capacity >= MAX_CAPACITY {
            capacity_overflow();
        }

        let new_capacity = old_capacity + SLAB_CAPACITY;

        let page_size = page_size();

        // This cannot overflow because `new_capacity` can never exceed `MAX_CAPACITY` and our
        // allocation has a max capacity of `MAX_CAPACITY`.
        let old_size = align_up(old_capacity as usize * size_of::<Node>(), page_size);
        let new_size = align_up(new_capacity as usize * size_of::<Node>(), page_size);
        let ptr = self.allocation.ptr().wrapping_add(old_size);
        let size = new_size - old_size;

        self.allocation.commit(ptr, size).unwrap();

        // The `Release` ordering synchronizes with the `Acquire` ordering in `Self::push`, making
        // sure that the new capacity is visible there.
        if let Err(capacity) =
            self.capacity
                .compare_exchange(old_capacity, new_capacity, Release, Relaxed)
        {
            // We lost the race, but the winner must have updated the capacity same as we wanted to.
            assert!(capacity >= new_capacity);
        }
    }

    unsafe fn get_unchecked(&self, index: u32) -> &Node {
        // SAFETY: Enforced by the caller.
        let ptr = unsafe { self.get_unchecked_raw(index) };

        // SAFETY: Enforced by the caller.
        unsafe { &*ptr }
    }

    unsafe fn get_unchecked_raw(&self, index: u32) -> *const Node {
        // SAFETY: Enforced by the caller.
        unsafe { self.as_ptr().add(index as usize) }
    }
}

#[inline(never)]
fn capacity_overflow() -> ! {
    panic!("capacity overflow");
}

#[repr(align(128))]
struct CacheAligned;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::test_queues;
    use std::{sync::Barrier, thread};

    const _: () = assert!(size_of::<Node>() == 128);

    const ITERATIONS: u32 = 100_000;
    const THREADS: usize = 8;

    #[test]
    fn queue_basic_usage() {
        let (resources, _) = test_queues!();
        let queue = Queue::new(NodeVec::new(), resources.global().clone());
        let local = resources.global().register_local();

        {
            let guard = &local.pin();
            unsafe { queue.push(queue.node_allocator.allocate(guard), guard) };
            unsafe { queue.push(queue.node_allocator.allocate(guard), guard) };
        }

        assert_eq!(queue_len(&queue), (2, 0));

        advance_global_two_steps(&local);

        {
            let guard = &local.pin();
            let predicate = |_: &_| ControlFlow::Continue(true);
            unsafe { queue.collect(predicate, &resources, guard) };
        }

        assert_eq!(queue_len(&queue), (0, 2));

        advance_global_two_steps(&local);

        {
            let guard = &local.pin();
            let predicate = |_: &_| ControlFlow::Continue(true);
            unsafe { queue.collect(predicate, &resources, guard) };
        }

        assert_eq!(queue_len(&queue), (0, 0));
    }

    #[test]
    fn queue_collect_stress() {
        let (resources, _) = test_queues!();
        let queue = Queue::new(NodeVec::new(), resources.global().clone());
        let local = resources.global().register_local();

        // TODO: `Queue::push` is quadratic in isolation.
        {
            let guard = &local.pin();
            let mut next_index = queue.node_allocator.allocate(guard);
            let node = unsafe { queue.nodes().get_unchecked(next_index) };
            node.next.store(NIL, Relaxed);

            for _ in 0..ITERATIONS - 1 {
                let index = queue.node_allocator.allocate(guard);
                let node = unsafe { queue.nodes().get_unchecked(index) };
                node.next.store(next_index, Relaxed);
                next_index = index;
            }

            let head_node = unsafe { queue.nodes().get_unchecked(queue.head) };
            head_node.next.store(next_index, Relaxed);
        }

        assert_eq!(queue_len(&queue), (ITERATIONS, 0));

        advance_global_two_steps(&local);

        let barrier = Barrier::new(THREADS);

        thread::scope(|scope| {
            for _ in 0..THREADS {
                scope.spawn(|| {
                    let local = resources.global().register_local();
                    let guard = &local.pin();
                    let predicate = |_: &_| ControlFlow::Continue(true);

                    barrier.wait();

                    unsafe { queue.collect(predicate, &resources, guard) };
                });
            }
        });

        assert_eq!(queue_len(&queue), (0, 0));
    }

    #[test]
    fn queue_push_collect_stress() {
        let (resources, _) = test_queues!();
        let queue = Queue::new(NodeVec::new(), resources.global().clone());
        let barrier = Barrier::new(THREADS);

        thread::scope(|scope| {
            for _ in 0..THREADS / 2 {
                scope.spawn(|| {
                    let local = resources.global().register_local();

                    barrier.wait();

                    for _ in 0..ITERATIONS {
                        let guard = &local.pin();
                        let index = queue.node_allocator.allocate(guard);
                        unsafe { queue.push(index, guard) };
                    }
                });
            }

            for _ in 0..THREADS / 2 {
                scope.spawn(|| {
                    let local = resources.global().register_local();
                    let predicate = |_: &_| ControlFlow::Continue(true);

                    barrier.wait();

                    for _ in 0..ITERATIONS {
                        let guard = &local.pin();
                        unsafe { queue.collect(predicate, &resources, guard) };
                    }
                });
            }
        });
    }

    fn queue_len(queue: &Queue) -> (u32, u32) {
        let head_node = unsafe { queue.nodes().get_unchecked(queue.head) };
        let mut curr_index = head_node.next.load(Relaxed);
        let mut count = 0;
        let mut deleted_count = 0;

        while curr_index != NIL {
            let curr_node = unsafe { queue.nodes().get_unchecked(curr_index) };

            let next_index = curr_node.next.load(Relaxed);

            if is_deleted(next_index) {
                deleted_count += 1;
            } else {
                count += 1;
            }

            curr_index = next_index & !DELETED_BIT;
        }

        (count, deleted_count)
    }

    fn advance_global_two_steps(local: &epoch::UniqueLocalHandle) {
        for _ in 0..2 {
            local.pin().try_advance_global();
        }
    }
}
