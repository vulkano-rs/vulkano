use super::{AllocationType, Region, Suballocation, Suballocator, SuballocatorError};
use crate::{
    memory::{
        allocator::{
            align_up, suballocator::are_blocks_on_same_page, AllocationHandle, DeviceLayout,
        },
        DeviceAlignment,
    },
    DeviceSize,
};
use std::cell::Cell;

/// A [suballocator] which can allocate dynamically, but can only free all allocations at once.
///
/// With bump allocation, the used up space increases linearly as allocations are made and
/// allocations can never be freed individually, which is why this algorithm is also called *linear
/// allocation*. It is also known as *arena allocation*.
///
/// `BumpAllocator`s are best suited for very short-lived (say a few frames at best) resources that
/// need to be allocated often (say each frame), to really take advantage of the performance gains.
/// For creating long-lived allocations, [`FreeListAllocator`] is best suited. The way you would
/// typically use this allocator is to have one for each frame in flight. At the start of a frame,
/// you reset it and allocate your resources with it. You write to the resources, render with them,
/// and drop them at the end of the frame.
///
/// See also [the `Suballocator` implementation].
///
/// # Algorithm
///
/// What happens is that every time you make an allocation, you receive one with an offset
/// corresponding to the *free start* within the [region], and then the free start is *bumped*, so
/// that following allocations wouldn't alias it. As you can imagine, this is **extremely fast**,
/// because it doesn't need to keep a [free-list]. It only needs to do a few additions and
/// comparisons. But beware, **fast is about all this is**. It is horribly memory inefficient when
/// used wrong, and is very susceptible to [memory leaks].
///
/// Once you know that you are done with the allocations, meaning you know they have all been
/// dropped, you can safely reset the allocator using the [`reset`] method as long as the allocator
/// is not shared between threads. This is one of the reasons you are generally advised to use one
/// `BumpAllocator` per thread if you can.
///
/// # Efficiency
///
/// Allocation is *O*(1), and so is resetting the allocator (freeing all allocations).
///
/// [suballocator]: Suballocator
/// [the `Suballocator` implementation]: Suballocator#impl-Suballocator-for-Arc<BumpAllocator>
/// [region]: Suballocator#regions
/// [free-list]: Suballocator#free-lists
/// [memory leaks]: super#leakage
/// [`reset`]: Self::reset
/// [hierarchy]: Suballocator#memory-hierarchies
#[derive(Debug)]
pub struct BumpAllocator {
    region: Region,
    free_start: Cell<DeviceSize>,
    prev_allocation_type: Cell<AllocationType>,
}

impl BumpAllocator {
    /// Resets the free-start back to the beginning of the [region].
    ///
    /// [region]: Suballocator#regions
    #[inline]
    pub fn reset(&mut self) {
        *self.free_start.get_mut() = 0;
        *self.prev_allocation_type.get_mut() = AllocationType::Unknown;
    }
}

unsafe impl Suballocator for BumpAllocator {
    /// Creates a new `BumpAllocator` for the given [region].
    ///
    /// [region]: Suballocator#regions
    fn new(region: Region) -> Self {
        BumpAllocator {
            region,
            free_start: Cell::new(0),
            prev_allocation_type: Cell::new(AllocationType::Unknown),
        }
    }

    #[inline]
    fn allocate(
        &self,
        layout: DeviceLayout,
        allocation_type: AllocationType,
        buffer_image_granularity: DeviceAlignment,
    ) -> Result<Suballocation, SuballocatorError> {
        fn has_granularity_conflict(prev_ty: AllocationType, ty: AllocationType) -> bool {
            prev_ty == AllocationType::Unknown || prev_ty != ty
        }

        let size = layout.size();
        let alignment = layout.alignment();

        // These can't overflow because suballocation offsets are bounded by the region, whose end
        // can itself not exceed `DeviceLayout::MAX_SIZE`.
        let prev_end = self.region.offset() + self.free_start.get();
        let mut offset = align_up(prev_end, alignment);

        if buffer_image_granularity != DeviceAlignment::MIN
            && prev_end > 0
            && are_blocks_on_same_page(0, prev_end, offset, buffer_image_granularity)
            && has_granularity_conflict(self.prev_allocation_type.get(), allocation_type)
        {
            offset = align_up(offset, buffer_image_granularity);
        }

        let relative_offset = offset - self.region.offset();

        let free_start = relative_offset + size;

        if free_start > self.region.size() {
            return Err(SuballocatorError::OutOfRegionMemory);
        }

        self.free_start.set(free_start);
        self.prev_allocation_type.set(allocation_type);

        Ok(Suballocation {
            offset,
            size,
            allocation_type,
            handle: AllocationHandle::null(),
        })
    }

    #[inline]
    unsafe fn deallocate(&self, _suballocation: Suballocation) {
        // such complex, very wow
    }

    #[inline]
    fn free_size(&self) -> DeviceSize {
        self.region.size() - self.free_start.get()
    }

    #[inline]
    fn cleanup(&mut self) {
        self.reset();
    }
}
