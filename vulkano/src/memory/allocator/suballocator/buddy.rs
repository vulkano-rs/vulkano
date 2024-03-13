use super::{AllocationType, Region, Suballocation, Suballocator, SuballocatorError};
use crate::{
    memory::{
        allocator::{align_up, array_vec::ArrayVec, AllocationHandle, DeviceLayout},
        is_aligned, DeviceAlignment,
    },
    DeviceSize, NonZeroDeviceSize,
};
use std::{
    cell::{Cell, UnsafeCell},
    cmp,
};

/// A [suballocator] whose structure forms a binary tree of power-of-two-sized suballocations.
///
/// That is, all allocation sizes are rounded up to the next power of two. This helps reduce
/// [external fragmentation] by a lot, at the expense of possibly severe [internal fragmentation]
/// if you're not careful. For example, if you needed an allocation size of 64MiB, you would be
/// wasting no memory. But with an allocation size of 70MiB, you would use a whole 128MiB instead,
/// wasting 45% of the memory. Use this algorithm if you need to create and free a lot of
/// allocations, which would cause too much external fragmentation when using
/// [`FreeListAllocator`]. However, if the sizes of your allocations are more or less the same,
/// then using an allocation pool would be a better choice and would eliminate external
/// fragmentation completely.
///
/// See also [the `Suballocator` implementation].
///
/// # Algorithm
///
/// Say you have a [region] of size 256MiB, and you want to allocate 14MiB. Assuming there are no
/// existing allocations, the `BuddyAllocator` would split the 256MiB root *node* into two 128MiB
/// nodes. These two nodes are called *buddies*. The allocator would then proceed to split the left
/// node recursively until it wouldn't be able to fit the allocation anymore. In this example, that
/// would happen after 4 splits and end up with a node size of 16MiB. Since the allocation
/// requested was 14MiB, 2MiB would become internal fragmentation and be unusable for the lifetime
/// of the allocation. When an allocation is freed, this process is done backwards, checking if the
/// buddy of each node on the way up is free and if so they are coalesced.
///
/// Each possible node size has an *order*, with the smallest node size being of order 0 and the
/// largest of the highest order. With this notion, node sizes are proportional to 2<sup>*n*</sup>
/// where *n* is the order. The highest order is determined from the size of the region and a
/// constant minimum node size, which we chose to be 16B: log(*region&nbsp;size*&nbsp;/&nbsp;16) or
/// equiavalently log(*region&nbsp;size*)&nbsp;-&nbsp;4 (assuming
/// *region&nbsp;size*&nbsp;&ge;&nbsp;16).
///
/// It's safe to say that this algorithm works best if you have some level of control over your
/// allocation sizes, so that you don't end up allocating twice as much memory. An example of this
/// would be when you need to allocate regions for other allocators, such as for an allocation pool
/// or the [`BumpAllocator`].
///
/// # Efficiency
///
/// The time complexity of both allocation and freeing is *O*(*m*) in the worst case where *m* is
/// the highest order, which equates to *O*(log (*n*)) where *n* is the size of the region.
///
/// [suballocator]: Suballocator
/// [internal fragmentation]: super#internal-fragmentation
/// [external fragmentation]: super#external-fragmentation
/// [the `Suballocator` implementation]: Suballocator#impl-Suballocator-for-Arc<BuddyAllocator>
/// [region]: Suballocator#regions
#[derive(Debug)]
pub struct BuddyAllocator {
    region_offset: DeviceSize,
    // Total memory remaining in the region.
    free_size: Cell<DeviceSize>,
    state: UnsafeCell<BuddyAllocatorState>,
}

impl BuddyAllocator {
    pub(super) const MIN_NODE_SIZE: DeviceSize = 16;

    /// Arbitrary maximum number of orders, used to avoid a 2D `Vec`. Together with a minimum node
    /// size of 16, this is enough for a 32GiB region.
    const MAX_ORDERS: usize = 32;
}

unsafe impl Suballocator for BuddyAllocator {
    /// Creates a new `BuddyAllocator` for the given [region].
    ///
    /// # Panics
    ///
    /// - Panics if `region.size` is not a power of two.
    /// - Panics if `region.size` is not in the range \[16B,&nbsp;32GiB\].
    ///
    /// [region]: Suballocator#regions
    fn new(region: Region) -> Self {
        const EMPTY_FREE_LIST: Vec<DeviceSize> = Vec::new();

        assert!(region.size().is_power_of_two());
        assert!(region.size() >= BuddyAllocator::MIN_NODE_SIZE);

        let max_order = (region.size() / BuddyAllocator::MIN_NODE_SIZE).trailing_zeros() as usize;

        assert!(max_order < BuddyAllocator::MAX_ORDERS);

        let free_size = Cell::new(region.size());

        let mut free_list =
            ArrayVec::new(max_order + 1, [EMPTY_FREE_LIST; BuddyAllocator::MAX_ORDERS]);
        // The root node has the lowest offset and highest order, so it's the whole region.
        free_list[max_order].push(region.offset());
        let state = UnsafeCell::new(BuddyAllocatorState { free_list });

        BuddyAllocator {
            region_offset: region.offset(),
            free_size,
            state,
        }
    }

    #[inline]
    fn allocate(
        &self,
        layout: DeviceLayout,
        allocation_type: AllocationType,
        buffer_image_granularity: DeviceAlignment,
    ) -> Result<Suballocation, SuballocatorError> {
        /// Returns the largest power of two smaller or equal to the input, or zero if the input is
        /// zero.
        fn prev_power_of_two(val: DeviceSize) -> DeviceSize {
            const MAX_POWER_OF_TWO: DeviceSize = DeviceAlignment::MAX.as_devicesize();

            if let Some(val) = NonZeroDeviceSize::new(val) {
                // This can't overflow because `val` is non-zero, which means it has fewer leading
                // zeroes than the total number of bits.
                MAX_POWER_OF_TWO >> val.leading_zeros()
            } else {
                0
            }
        }

        let mut size = layout.size();
        let mut alignment = layout.alignment();

        if buffer_image_granularity != DeviceAlignment::MIN {
            debug_assert!(is_aligned(self.region_offset, buffer_image_granularity));

            if allocation_type == AllocationType::Unknown
                || allocation_type == AllocationType::NonLinear
            {
                // This can't overflow because `DeviceLayout` guarantees that `size` doesn't exceed
                // `DeviceLayout::MAX_SIZE`.
                size = align_up(size, buffer_image_granularity);
                alignment = cmp::max(alignment, buffer_image_granularity);
            }
        }

        // `DeviceLayout` guarantees that its size does not exceed `DeviceLayout::MAX_SIZE`,
        // which means it can't overflow when rounded up to the next power of two.
        let size = cmp::max(size, BuddyAllocator::MIN_NODE_SIZE).next_power_of_two();

        let min_order = (size / BuddyAllocator::MIN_NODE_SIZE).trailing_zeros() as usize;
        let state = unsafe { &mut *self.state.get() };

        // Start searching at the lowest possible order going up.
        for (order, free_list) in state.free_list.iter_mut().enumerate().skip(min_order) {
            for (index, &offset) in free_list.iter().enumerate() {
                if is_aligned(offset, alignment) {
                    free_list.remove(index);

                    // Go in the opposite direction, splitting nodes from higher orders. The lowest
                    // order doesn't need any splitting.
                    for (order, free_list) in state
                        .free_list
                        .iter_mut()
                        .enumerate()
                        .skip(min_order)
                        .take(order - min_order)
                        .rev()
                    {
                        // This can't discard any bits because `order` is confined to the range
                        // [0, log(region.size / BuddyAllocator::MIN_NODE_SIZE)].
                        let size = BuddyAllocator::MIN_NODE_SIZE << order;

                        // This can't overflow because suballocations are bounded by the region,
                        // whose end can itself not exceed `DeviceLayout::MAX_SIZE`.
                        let right_child = offset + size;

                        // Insert the right child in sorted order.
                        let (Ok(index) | Err(index)) = free_list.binary_search(&right_child);
                        free_list.insert(index, right_child);

                        // Repeat splitting for the left child if required in the next loop turn.
                    }

                    // This can't overflow because suballocation sizes in the free-list are
                    // constrained by the remaining size of the region.
                    self.free_size.set(self.free_size.get() - size);

                    return Ok(Suballocation {
                        offset,
                        size: layout.size(),
                        allocation_type,
                        handle: AllocationHandle::from_index(min_order),
                    });
                }
            }
        }

        if prev_power_of_two(self.free_size()) >= layout.size() {
            // A node large enough could be formed if the region wasn't so fragmented.
            Err(SuballocatorError::FragmentedRegion)
        } else {
            Err(SuballocatorError::OutOfRegionMemory)
        }
    }

    #[inline]
    unsafe fn deallocate(&self, suballocation: Suballocation) {
        let mut offset = suballocation.offset;
        let order = suballocation.handle.as_index();

        let min_order = order;
        let state = unsafe { &mut *self.state.get() };

        debug_assert!(!state.free_list[order].contains(&offset));

        // Try to coalesce nodes while incrementing the order.
        for (order, free_list) in state.free_list.iter_mut().enumerate().skip(min_order) {
            // This can't discard any bits because `order` is confined to the range
            // [0, log(region.size / BuddyAllocator::MIN_NODE_SIZE)].
            let size = BuddyAllocator::MIN_NODE_SIZE << order;

            // This can't overflow because the offsets in the free-list are confined to the range
            // [region.offset, region.offset + region.size).
            let buddy_offset = ((offset - self.region_offset) ^ size) + self.region_offset;

            match free_list.binary_search(&buddy_offset) {
                // If the buddy is in the free-list, we can coalesce.
                Ok(index) => {
                    free_list.remove(index);
                    offset = cmp::min(offset, buddy_offset);
                }
                // Otherwise free the node.
                Err(_) => {
                    let (Ok(index) | Err(index)) = free_list.binary_search(&offset);
                    free_list.insert(index, offset);

                    // This can't discard any bits for the same reason as above.
                    let size = BuddyAllocator::MIN_NODE_SIZE << min_order;

                    // The sizes of suballocations allocated by `self` are constrained by that of
                    // its region, so they can't possibly overflow when added up.
                    self.free_size.set(self.free_size.get() + size);

                    break;
                }
            }
        }
    }

    /// Returns the total amount of free space left in the [region] that is available to the
    /// allocator, which means that [internal fragmentation] is excluded.
    ///
    /// [region]: Suballocator#regions
    /// [internal fragmentation]: super#internal-fragmentation
    #[inline]
    fn free_size(&self) -> DeviceSize {
        self.free_size.get()
    }

    #[inline]
    fn cleanup(&mut self) {}
}

#[derive(Debug)]
struct BuddyAllocatorState {
    // Every order has its own free-list for convenience, so that we don't have to traverse a tree.
    // Each free-list is sorted by offset because we want to find the first-fit as this strategy
    // minimizes external fragmentation.
    free_list: ArrayVec<Vec<DeviceSize>, { BuddyAllocator::MAX_ORDERS }>,
}
