use super::{
    are_blocks_on_same_page, AllocationType, Region, Suballocation, SuballocationNode,
    SuballocationType, Suballocator, SuballocatorError,
};
use crate::{
    memory::{
        allocator::{align_up, AllocationHandle, DeviceLayout},
        is_aligned, DeviceAlignment,
    },
    DeviceSize,
};
use std::{cmp, iter::FusedIterator, marker::PhantomData, ptr::NonNull};

/// A [suballocator] that uses the most generic [free-list].
///
/// The strength of this allocator is that it can create and free allocations completely
/// dynamically, which means they can be any size and created/freed in any order. The downside is
/// that this always leads to horrific [external fragmentation] the more such dynamic allocations
/// are made. Therefore, this allocator is best suited for long-lived allocations. If you need
/// to create allocations of various sizes, but can't afford this fragmentation, then the
/// [`BuddyAllocator`] is your best buddy. If you need to create allocations which share a similar
/// size, consider an allocation pool. Lastly, if you need to allocate very often, then
/// [`BumpAllocator`] is best suited.
///
/// See also [the `Suballocator` implementation].
///
/// # Algorithm
///
/// The free-list stores suballocations which can have any offset and size. When an allocation
/// request is made, the list is searched using the best-fit strategy, meaning that the smallest
/// suballocation that fits the request is chosen. If required, the chosen suballocation is trimmed
/// at the ends and the ends are returned to the free-list. As such, no [internal fragmentation]
/// occurs. The front might need to be trimmed because of [alignment requirements] and the end
/// because of a larger than required size. When an allocation is freed, the allocator checks if
/// the adjacent suballocations are free, and if so it coalesces them into a bigger one before
/// putting it in the free-list.
///
/// # Efficiency
///
/// The free-list is sorted by size, which means that when allocating, finding a best-fit is always
/// possible in *O*(log(*n*)) time in the worst case. When freeing, the coalescing requires us to
/// remove the adjacent free suballocations from the free-list which is *O*(log(*n*)), and insert
/// the possibly coalesced suballocation into the free-list which has the same time complexity, so
/// in total freeing is *O*(log(*n*)).
///
/// There is one notable edge-case: after the allocator finds a best-fit, it is possible that it
/// needs to align the suballocation's offset to a higher value, after which the requested size
/// might no longer fit. In such a case, the next free suballocation in sorted order is tried until
/// a fit is successful. If this issue is encountered with all candidates, then the time complexity
/// would be *O*(*n*). However, this scenario is extremely unlikely which is why we are not
/// considering it in the above analysis. Additionally, if your free-list is filled with
/// allocations that all have the same size then that seems pretty sus. Sounds like you're in dire
/// need of an allocation pool.
///
/// [suballocator]: Suballocator
/// [free-list]: Suballocator#free-lists
/// [external fragmentation]: super#external-fragmentation
/// [the `Suballocator` implementation]: Suballocator#impl-Suballocator-for-Arc<FreeListAllocator>
/// [internal fragmentation]: super#internal-fragmentation
/// [alignment requirements]: super#alignment
#[derive(Debug)]
pub struct FreeListAllocator {
    region_offset: DeviceSize,
    // Total memory remaining in the region.
    free_size: DeviceSize,
    suballocations: SuballocationList,
}

unsafe impl Suballocator for FreeListAllocator {
    type Suballocations<'a> = Suballocations<'a>;

    /// Creates a new `FreeListAllocator` for the given [region].
    ///
    /// [region]: Suballocator#regions
    fn new(region: Region) -> Self {
        let node_allocator = slabbin::SlabAllocator::<SuballocationListNode>::new(32);
        let root_ptr = node_allocator.allocate();
        let root = SuballocationListNode {
            prev: None,
            next: None,
            offset: region.offset(),
            size: region.size(),
            allocation_type: SuballocationType::Free,
        };
        unsafe { root_ptr.as_ptr().write(root) };

        let mut free_list = Vec::with_capacity(32);
        free_list.push(root_ptr);

        let suballocations = SuballocationList {
            head: root_ptr,
            tail: root_ptr,
            len: 1,
            free_list,
            node_allocator,
        };

        FreeListAllocator {
            region_offset: region.offset(),
            free_size: region.size(),
            suballocations,
        }
    }

    #[inline]
    fn allocate(
        &mut self,
        layout: DeviceLayout,
        allocation_type: AllocationType,
        buffer_image_granularity: DeviceAlignment,
    ) -> Result<Suballocation, SuballocatorError> {
        fn has_granularity_conflict(prev_ty: SuballocationType, ty: AllocationType) -> bool {
            if prev_ty == SuballocationType::Free {
                false
            } else if prev_ty == SuballocationType::Unknown {
                true
            } else {
                prev_ty != ty.into()
            }
        }

        let size = layout.size();
        let alignment = layout.alignment();

        match self.suballocations.free_list.last() {
            Some(&last) if unsafe { (*last.as_ptr()).size } >= size => {
                // We create a dummy node to compare against in the below binary search. The only
                // fields of importance are `offset` and `size`. It is paramount that we set
                // `offset` to zero, so that in the case where there are multiple free
                // suballocations with the same size, we get the first one of them, that is, the
                // one with the lowest offset.
                let dummy_node = SuballocationListNode {
                    prev: None,
                    next: None,
                    offset: 0,
                    size,
                    allocation_type: SuballocationType::Unknown,
                };

                // This is almost exclusively going to return `Err`, but that's expected: we are
                // first comparing the size, looking for an allocation of the given `size`, however
                // the next-best will do as well (that is, a size somewhat larger). In that case we
                // get `Err`. If we do find a suballocation with the exact size however, we are
                // then comparing the offsets to make sure we get the suballocation with the lowest
                // offset, in case there are multiple with the same size. In that case we also
                // exclusively get `Err` except when the offset is zero.
                //
                // Note that `index == free_list.len()` can't be because we checked that the
                // free-list contains a suballocation that is big enough.
                let (Ok(index) | Err(index)) = self
                    .suballocations
                    .free_list
                    .binary_search_by_key(&dummy_node, |&ptr| unsafe { *ptr.as_ptr() });

                for (index, &node_ptr) in
                    self.suballocations.free_list.iter().enumerate().skip(index)
                {
                    let node = unsafe { *node_ptr.as_ptr() };

                    // This can't overflow because suballocation offsets are bounded by the region,
                    // whose end can itself not exceed `DeviceLayout::MAX_SIZE`.
                    let mut offset = align_up(node.offset, alignment);

                    if buffer_image_granularity != DeviceAlignment::MIN {
                        debug_assert!(is_aligned(self.region_offset, buffer_image_granularity));

                        if let Some(prev_ptr) = node.prev {
                            let prev = unsafe { *prev_ptr.as_ptr() };

                            if are_blocks_on_same_page(
                                prev.offset,
                                prev.size,
                                offset,
                                buffer_image_granularity,
                            ) && has_granularity_conflict(prev.allocation_type, allocation_type)
                            {
                                // This is overflow-safe for the same reason as above.
                                offset = align_up(offset, buffer_image_granularity);
                            }
                        }
                    }

                    // `offset`, no matter the alignment, can't end up as more than
                    // `DeviceAlignment::MAX` for the same reason as above. `DeviceLayout`
                    // guarantees that `size` doesn't exceed `DeviceLayout::MAX_SIZE`.
                    // `DeviceAlignment::MAX.as_devicesize() + DeviceLayout::MAX_SIZE` is equal to
                    // `DeviceSize::MAX`. Therefore, `offset + size` can't overflow.
                    //
                    // `node.offset + node.size` can't overflow for the same reason as above.
                    if offset + size <= node.offset + node.size {
                        self.suballocations.free_list.remove(index);

                        // SAFETY:
                        // - `node` is free.
                        // - `offset` is that of `node`, possibly rounded up.
                        // - We checked that `offset + size` falls within `node`.
                        unsafe { self.suballocations.split(node_ptr, offset, size) };

                        unsafe { (*node_ptr.as_ptr()).allocation_type = allocation_type.into() };

                        // This can't overflow because suballocation sizes in the free-list are
                        // constrained by the remaining size of the region.
                        self.free_size -= size;

                        return Ok(Suballocation {
                            offset,
                            size,
                            allocation_type,
                            handle: AllocationHandle::from_ptr(node_ptr.as_ptr().cast()),
                        });
                    }
                }

                // There is not enough space due to alignment requirements.
                Err(SuballocatorError::OutOfRegionMemory)
            }
            // There would be enough space if the region wasn't so fragmented. :(
            Some(_) if self.free_size() >= size => Err(SuballocatorError::FragmentedRegion),
            // There is not enough space.
            Some(_) => Err(SuballocatorError::OutOfRegionMemory),
            // There is no space at all.
            None => Err(SuballocatorError::OutOfRegionMemory),
        }
    }

    #[inline]
    unsafe fn deallocate(&mut self, suballocation: Suballocation) {
        let node_ptr = suballocation
            .handle
            .as_ptr()
            .cast::<SuballocationListNode>();

        // SAFETY: The caller must guarantee that `suballocation` refers to a currently allocated
        // allocation of `self`, which means that `node_ptr` is the same one we gave out on
        // allocation, making it a valid pointer.
        let node_ptr = unsafe { NonNull::new_unchecked(node_ptr) };
        let node = unsafe { *node_ptr.as_ptr() };

        debug_assert_ne!(node.allocation_type, SuballocationType::Free);

        // Suballocation sizes are constrained by the size of the region, so they can't possibly
        // overflow when added up.
        self.free_size += node.size;

        unsafe { (*node_ptr.as_ptr()).allocation_type = SuballocationType::Free };

        unsafe { self.suballocations.coalesce(node_ptr) };
        unsafe { self.suballocations.deallocate(node_ptr) };
    }

    #[inline]
    fn free_size(&self) -> DeviceSize {
        self.free_size
    }

    #[inline]
    fn cleanup(&mut self) {}

    #[inline]
    fn suballocations(&self) -> Self::Suballocations<'_> {
        self.suballocations.iter()
    }
}

#[derive(Debug)]
struct SuballocationList {
    head: NonNull<SuballocationListNode>,
    tail: NonNull<SuballocationListNode>,
    len: usize,
    // Free suballocations sorted by size in ascending order. This means we can always find a
    // best-fit in *O*(log(*n*)) time in the worst case, and iterating in order is very efficient.
    free_list: Vec<NonNull<SuballocationListNode>>,
    node_allocator: slabbin::SlabAllocator<SuballocationListNode>,
}

unsafe impl Send for SuballocationList {}
unsafe impl Sync for SuballocationList {}

#[derive(Clone, Copy, Debug)]
struct SuballocationListNode {
    prev: Option<NonNull<Self>>,
    next: Option<NonNull<Self>>,
    offset: DeviceSize,
    size: DeviceSize,
    allocation_type: SuballocationType,
}

impl PartialEq for SuballocationListNode {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size && self.offset == other.offset
    }
}

impl Eq for SuballocationListNode {}

impl PartialOrd for SuballocationListNode {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SuballocationListNode {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        // We want to sort the free-list by size.
        self.size
            .cmp(&other.size)
            // However there might be multiple free suballocations with the same size, so we need
            // to compare the offset as well to differentiate.
            .then(self.offset.cmp(&other.offset))
    }
}

impl SuballocationList {
    /// Fits a suballocation inside the target one, splitting the target at the ends if required.
    ///
    /// # Safety
    ///
    /// - `node_ptr` must refer to a currently free suballocation of `self`.
    /// - `offset` and `size` must refer to a subregion of the given suballocation.
    unsafe fn split(
        &mut self,
        node_ptr: NonNull<SuballocationListNode>,
        offset: DeviceSize,
        size: DeviceSize,
    ) {
        let node = unsafe { *node_ptr.as_ptr() };

        debug_assert_eq!(node.allocation_type, SuballocationType::Free);
        debug_assert!(offset >= node.offset);
        debug_assert!(offset + size <= node.offset + node.size);

        // These are guaranteed to not overflow because the caller must uphold that the given
        // region is contained within that of `node`.
        let padding_front = offset - node.offset;
        let padding_back = node.offset + node.size - offset - size;

        if padding_front > 0 {
            let padding_ptr = self.node_allocator.allocate();
            let padding = SuballocationListNode {
                prev: node.prev,
                next: Some(node_ptr),
                offset: node.offset,
                size: padding_front,
                allocation_type: SuballocationType::Free,
            };
            unsafe { padding_ptr.as_ptr().write(padding) };

            if let Some(prev_ptr) = padding.prev {
                unsafe { (*prev_ptr.as_ptr()).next = Some(padding_ptr) };
            }

            unsafe { (*node_ptr.as_ptr()).prev = Some(padding_ptr) };
            unsafe { (*node_ptr.as_ptr()).offset = offset };
            // The caller must uphold that the given region is contained within that of `node`, and
            // it follows that if there is padding, the size of the node must be larger than that
            // of the padding, so this can't overflow.
            unsafe { (*node_ptr.as_ptr()).size -= padding.size };

            if node_ptr == self.head {
                self.head = padding_ptr;
            }

            self.len += 1;

            // SAFETY: We just created this suballocation, so there's no way that it was
            // deallocated already.
            unsafe { self.deallocate(padding_ptr) };
        }

        if padding_back > 0 {
            let padding_ptr = self.node_allocator.allocate();
            let padding = SuballocationListNode {
                prev: Some(node_ptr),
                next: node.next,
                offset: offset + size,
                size: padding_back,
                allocation_type: SuballocationType::Free,
            };
            unsafe { padding_ptr.as_ptr().write(padding) };

            if let Some(next_ptr) = padding.next {
                unsafe { (*next_ptr.as_ptr()).prev = Some(padding_ptr) };
            }

            unsafe { (*node_ptr.as_ptr()).next = Some(padding_ptr) };
            // This is overflow-safe for the same reason as above.
            unsafe { (*node_ptr.as_ptr()).size -= padding.size };

            if node_ptr == self.tail {
                self.tail = padding_ptr;
            }

            self.len += 1;

            // SAFETY: Same as above.
            unsafe { self.deallocate(padding_ptr) };
        }
    }

    /// Inserts the target suballocation into the free-list.
    ///
    /// # Safety
    ///
    /// - `node_ptr` must refer to a currently allocated suballocation of `self`.
    unsafe fn deallocate(&mut self, node_ptr: NonNull<SuballocationListNode>) {
        debug_assert!(!self.free_list.contains(&node_ptr));

        let node = unsafe { *node_ptr.as_ptr() };
        let (Ok(index) | Err(index)) = self
            .free_list
            .binary_search_by_key(&node, |&ptr| unsafe { *ptr.as_ptr() });
        self.free_list.insert(index, node_ptr);
    }

    /// Coalesces the target (free) suballocation with adjacent ones that are also free.
    ///
    /// # Safety
    ///
    /// - `node_ptr` must refer to a currently free suballocation `self`.
    unsafe fn coalesce(&mut self, node_ptr: NonNull<SuballocationListNode>) {
        let node = unsafe { *node_ptr.as_ptr() };

        debug_assert_eq!(node.allocation_type, SuballocationType::Free);

        if let Some(prev_ptr) = node.prev {
            let prev = unsafe { *prev_ptr.as_ptr() };

            if prev.allocation_type == SuballocationType::Free {
                // SAFETY: We checked that the suballocation is free.
                self.allocate(prev_ptr);

                unsafe { (*node_ptr.as_ptr()).prev = prev.prev };
                unsafe { (*node_ptr.as_ptr()).offset = prev.offset };
                // The sizes of suballocations are constrained by that of the parent allocation, so
                // they can't possibly overflow when added up.
                unsafe { (*node_ptr.as_ptr()).size += prev.size };

                if let Some(prev_ptr) = prev.prev {
                    unsafe { (*prev_ptr.as_ptr()).next = Some(node_ptr) };
                }

                if prev_ptr == self.head {
                    self.head = node_ptr;
                }

                self.len -= 1;

                // SAFETY:
                // - The suballocation is free.
                // - The suballocation was removed from the free-list.
                // - The next suballocation and possibly a previous suballocation have been updated
                //   such that they no longer reference the suballocation.
                // - The head no longer points to the suballocation if it used to.
                // All of these conditions combined guarantee that `prev_ptr` cannot be used again.
                unsafe { self.node_allocator.deallocate(prev_ptr) };
            }
        }

        if let Some(next_ptr) = node.next {
            let next = unsafe { *next_ptr.as_ptr() };

            if next.allocation_type == SuballocationType::Free {
                // SAFETY: Same as above.
                self.allocate(next_ptr);

                unsafe { (*node_ptr.as_ptr()).next = next.next };
                // This is overflow-safe for the same reason as above.
                unsafe { (*node_ptr.as_ptr()).size += next.size };

                if let Some(next_ptr) = next.next {
                    unsafe { (*next_ptr.as_ptr()).prev = Some(node_ptr) };
                }

                if next_ptr == self.tail {
                    self.tail = node_ptr;
                }

                self.len -= 1;

                // SAFETY: Same as above.
                unsafe { self.node_allocator.deallocate(next_ptr) };
            }
        }
    }

    /// Removes the target suballocation from the free-list.
    ///
    /// # Safety
    ///
    /// - `node_ptr` must refer to a currently free suballocation of `self`.
    unsafe fn allocate(&mut self, node_ptr: NonNull<SuballocationListNode>) {
        debug_assert!(self.free_list.contains(&node_ptr));

        let node = unsafe { *node_ptr.as_ptr() };

        match self
            .free_list
            .binary_search_by_key(&node, |&ptr| unsafe { *ptr.as_ptr() })
        {
            Ok(index) => {
                self.free_list.remove(index);
            }
            Err(_) => unreachable!(),
        }
    }

    fn iter(&self) -> Suballocations<'_> {
        Suballocations {
            head: Some(self.head),
            tail: Some(self.tail),
            len: self.len,
            marker: PhantomData,
        }
    }
}

#[derive(Clone)]
pub struct Suballocations<'a> {
    head: Option<NonNull<SuballocationListNode>>,
    tail: Option<NonNull<SuballocationListNode>>,
    len: usize,
    marker: PhantomData<&'a SuballocationList>,
}

unsafe impl Send for Suballocations<'_> {}
unsafe impl Sync for Suballocations<'_> {}

impl Iterator for Suballocations<'_> {
    type Item = SuballocationNode;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.len != 0 {
            if let Some(head) = self.head {
                let head = unsafe { *head.as_ptr() };
                self.head = head.next;
                self.len -= 1;

                Some(SuballocationNode {
                    offset: head.offset,
                    size: head.size,
                    allocation_type: head.allocation_type,
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }

    #[inline]
    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl DoubleEndedIterator for Suballocations<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len != 0 {
            if let Some(tail) = self.tail {
                let tail = unsafe { *tail.as_ptr() };
                self.tail = tail.prev;
                self.len -= 1;

                Some(SuballocationNode {
                    offset: tail.offset,
                    size: tail.size,
                    allocation_type: tail.allocation_type,
                })
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl ExactSizeIterator for Suballocations<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

impl FusedIterator for Suballocations<'_> {}
