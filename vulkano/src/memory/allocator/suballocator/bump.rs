use super::{
    AllocationType, Region, Suballocation, SuballocationNode, SuballocationType, Suballocator,
    SuballocatorError,
};
use crate::{
    memory::{
        allocator::{
            align_up, suballocator::are_blocks_on_same_page, AllocationHandle, DeviceLayout,
        },
        DeviceAlignment,
    },
    DeviceSize,
};
use std::iter::FusedIterator;

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
    free_start: DeviceSize,
    prev_allocation_type: AllocationType,
}

impl BumpAllocator {
    /// Resets the free-start back to the beginning of the [region].
    ///
    /// [region]: Suballocator#regions
    #[inline]
    pub fn reset(&mut self) {
        self.free_start = 0;
        self.prev_allocation_type = AllocationType::Unknown;
    }

    fn suballocation_node(&self, part: usize) -> SuballocationNode {
        if part == 0 {
            SuballocationNode {
                offset: self.region.offset(),
                size: self.free_start,
                allocation_type: self.prev_allocation_type.into(),
            }
        } else {
            debug_assert_eq!(part, 1);

            SuballocationNode {
                offset: self.region.offset() + self.free_start,
                size: self.free_size(),
                allocation_type: SuballocationType::Free,
            }
        }
    }
}

unsafe impl Suballocator for BumpAllocator {
    type Suballocations<'a> = Suballocations<'a>;

    /// Creates a new `BumpAllocator` for the given [region].
    ///
    /// [region]: Suballocator#regions
    fn new(region: Region) -> Self {
        BumpAllocator {
            region,
            free_start: 0,
            prev_allocation_type: AllocationType::Unknown,
        }
    }

    #[inline]
    fn allocate(
        &mut self,
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
        let prev_end = self.region.offset() + self.free_start;
        let mut offset = align_up(prev_end, alignment);

        if buffer_image_granularity != DeviceAlignment::MIN
            && prev_end > 0
            && are_blocks_on_same_page(0, prev_end, offset, buffer_image_granularity)
            && has_granularity_conflict(self.prev_allocation_type, allocation_type)
        {
            offset = align_up(offset, buffer_image_granularity);
        }

        let relative_offset = offset - self.region.offset();

        let free_start = relative_offset + size;

        if free_start > self.region.size() {
            return Err(SuballocatorError::OutOfRegionMemory);
        }

        self.free_start = free_start;
        self.prev_allocation_type = allocation_type;

        Ok(Suballocation {
            offset,
            size,
            allocation_type,
            handle: AllocationHandle::null(),
        })
    }

    #[inline]
    unsafe fn deallocate(&mut self, _suballocation: Suballocation) {
        // such complex, very wow
    }

    #[inline]
    fn free_size(&self) -> DeviceSize {
        self.region.size() - self.free_start
    }

    #[inline]
    fn cleanup(&mut self) {
        self.reset();
    }

    #[inline]
    fn suballocations(&self) -> Self::Suballocations<'_> {
        let start = if self.free_start == 0 { 1 } else { 0 };
        let end = if self.free_start == self.region.size() {
            1
        } else {
            2
        };

        Suballocations {
            allocator: self,
            start,
            end,
        }
    }
}

#[derive(Clone)]
pub struct Suballocations<'a> {
    allocator: &'a BumpAllocator,
    start: usize,
    end: usize,
}

impl Iterator for Suballocations<'_> {
    type Item = SuballocationNode;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len() != 0 {
            let node = self.allocator.suballocation_node(self.start);
            self.start += 1;

            Some(node)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();

        (len, Some(len))
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl DoubleEndedIterator for Suballocations<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len() != 0 {
            self.end -= 1;
            let node = self.allocator.suballocation_node(self.end);

            Some(node)
        } else {
            None
        }
    }
}

impl ExactSizeIterator for Suballocations<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.end - self.start
    }
}

impl FusedIterator for Suballocations<'_> {}
