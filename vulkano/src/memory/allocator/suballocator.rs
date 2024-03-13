//! Suballocators are used to divide a *region* into smaller *suballocations*.
//!
//! See also [the parent module] for details about memory allocation in Vulkan.
//!
//! [the parent module]: super

pub use self::region::Region;
use super::{
    align_down, align_up, array_vec::ArrayVec, AllocationHandle, DeviceAlignment, DeviceLayout,
};
use crate::{image::ImageTiling, memory::is_aligned, DeviceSize, NonZeroDeviceSize};
use std::{
    cell::{Cell, UnsafeCell},
    cmp,
    error::Error,
    fmt::{self, Debug, Display},
    ptr::NonNull,
};

/// Suballocators are used to divide a *region* into smaller *suballocations*.
///
/// # Regions
///
/// As the name implies, a region is a contiguous portion of memory. It may be the whole dedicated
/// block of [`DeviceMemory`], or only a part of it. Or it may be a buffer, or only a part of a
/// buffer. Regions are just allocations like any other, but we use this term to refer specifically
/// to an allocation that is to be suballocated. Every suballocator is created with a region to
/// work with.
///
/// # Free-lists
///
/// A free-list, also kind of predictably, refers to a list of (sub)allocations within a region
/// that are currently free. Every (sub)allocator that can free allocations dynamically (in any
/// order) needs to keep a free-list of some sort. This list is then consulted when new allocations
/// are made, and can be used to coalesce neighboring allocations that are free into bigger ones.
///
/// # Memory hierarchies
///
/// Different applications have wildly different allocation needs, and there's no way to cover them
/// all with a single type of allocator. Furthermore, different allocators have different
/// trade-offs and are best suited to specific tasks. To account for all possible use-cases,
/// Vulkano offers the ability to create *memory hierarchies*. We refer to the `DeviceMemory` as
/// the root of any such hierarchy, even though technically the driver has levels that are further
/// up, because those `DeviceMemory` blocks need to be allocated from physical memory pages
/// themselves, but since those levels are not accessible to us we don't need to consider them. You
/// can create any number of levels/branches from there, bounded only by the amount of available
/// memory within a `DeviceMemory` block. You can suballocate the root into regions, which are then
/// suballocated into further regions and so on, creating hierarchies of arbitrary height.
///
/// # Examples
///
/// TODO
///
/// # Safety
///
/// First consider using the provided implementations as there should be no reason to implement
/// this trait, but if you **must**:
///
/// - `allocate` must return a memory block that is in bounds of the region.
/// - `allocate` must return a memory block that doesn't alias any other currently allocated memory
///   blocks:
///   - Two currently allocated memory blocks must not share any memory locations, meaning that the
///     intersection of the byte ranges of the two memory blocks must be empty.
///   - Two neighboring currently allocated memory blocks must not share any [page] whose size is
///     given by the [buffer-image granularity], unless either both were allocated with
///     [`AllocationType::Linear`] or both were allocated with [`AllocationType::NonLinear`].
///   - The size does **not** have to be padded to the alignment. That is, as long the offset is
///     aligned and the memory blocks don't share any memory locations, a memory block is not
///     considered to alias another even if the padded size shares memory locations with another
///     memory block.
/// - A memory block must stay allocated until either `deallocate` is called on it or the allocator
///   is dropped. If the allocator is cloned, it must produce the same allocator, and memory blocks
///   must stay allocated until either `deallocate` is called on the memory block using any of the
///   clones or all of the clones have been dropped.
///
/// [`DeviceMemory`]: crate::memory::DeviceMemory
/// [page]: super#pages
/// [buffer-image granularity]: super#buffer-image-granularity
pub unsafe trait Suballocator {
    /// Creates a new suballocator for the given [region].
    ///
    /// [region]: Self#regions
    fn new(region: Region) -> Self
    where
        Self: Sized;

    /// Creates a new suballocation within the [region].
    ///
    /// # Arguments
    ///
    /// - `layout` - The layout of the allocation.
    ///
    /// - `allocation_type` - The type of resources that can be bound to the allocation.
    ///
    /// - `buffer_image_granularity` - The [buffer-image granularity] device property.
    ///
    ///   This is provided as an argument here rather than on construction of the allocator to
    ///   allow for optimizations: if you are only ever going to be creating allocations with the
    ///   same `allocation_type` using this allocator, then you may hard-code this to
    ///   [`DeviceAlignment::MIN`], in which case, after inlining, the logic for aligning the
    ///   allocation to the buffer-image-granularity based on the allocation type of surrounding
    ///   allocations can be optimized out.
    ///
    ///   You don't need to consider the buffer-image granularity for instance when suballocating a
    ///   buffer, or when suballocating a [`DeviceMemory`] block that's only ever going to be used
    ///   for optimal images. However, if you do allocate both linear and non-linear resources and
    ///   don't specify the buffer-image granularity device property here, **you will get undefined
    ///   behavior down the line**. Note that [`AllocationType::Unknown`] counts as both linear and
    ///   non-linear at the same time: if you always use this as the `allocation_type` using this
    ///   allocator, then it is valid to set this to `DeviceAlignment::MIN`, but **you must ensure
    ///   all allocations are aligned to the buffer-image granularity at minimum**.
    ///
    /// [region]: Self#regions
    /// [buffer-image granularity]: super#buffer-image-granularity
    /// [`DeviceMemory`]: crate::memory::DeviceMemory
    fn allocate(
        &self,
        layout: DeviceLayout,
        allocation_type: AllocationType,
        buffer_image_granularity: DeviceAlignment,
    ) -> Result<Suballocation, SuballocatorError>;

    /// Deallocates the given `suballocation`.
    ///
    /// # Safety
    ///
    /// - `suballocation` must refer to a **currently allocated** suballocation of `self`.
    unsafe fn deallocate(&self, suballocation: Suballocation);

    /// Returns the total amount of free space that is left in the [region].
    ///
    /// [region]: Self#regions
    fn free_size(&self) -> DeviceSize;

    /// Tries to free some space, if applicable.
    ///
    /// There must be no current allocations as they might get freed.
    fn cleanup(&mut self);
}

impl Debug for dyn Suballocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Suballocator").finish_non_exhaustive()
    }
}

mod region {
    use super::{DeviceLayout, DeviceSize};

    /// A [region] for a [suballocator] to allocate within. All [suballocations] will be in bounds
    /// of this region.
    ///
    /// In order to prevent arithmetic overflow when allocating, the region's end must not exceed
    /// [`DeviceLayout::MAX_SIZE`].
    ///
    /// The suballocator knowing the offset of the region rather than only the size allows you to
    /// easily suballocate suballocations. Otherwise, if regions were always relative, you would
    /// have to pick some maximum alignment for a suballocation before suballocating it further, to
    /// satisfy alignment requirements. However, you might not even know the maximum alignment
    /// requirement. Instead you can feed a suballocator a region that is aligned any which way,
    /// and it makes sure that the *absolute offset* of the suballocation has the requested
    /// alignment, meaning the offset that's already offset by the region's offset.
    ///
    /// There's one important caveat: if suballocating a suballocation, and the suballocation and
    /// the suballocation's suballocations aren't both only linear or only nonlinear, then the
    /// region must be aligned to the [buffer-image granularity]. Otherwise, there might be a
    /// buffer-image granularity conflict between the parent suballocator's allocations and the
    /// child suballocator's allocations.
    ///
    /// [region]: super::Suballocator#regions
    /// [suballocator]: super::Suballocator
    /// [suballocations]: super::Suballocation
    /// [buffer-image granularity]: super::super#buffer-image-granularity
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct Region {
        offset: DeviceSize,
        size: DeviceSize,
    }

    impl Region {
        /// Creates a new `Region` from the given `offset` and `size`.
        ///
        /// Returns [`None`] if the end of the region would exceed [`DeviceLayout::MAX_SIZE`].
        #[inline]
        pub const fn new(offset: DeviceSize, size: DeviceSize) -> Option<Self> {
            if offset.saturating_add(size) <= DeviceLayout::MAX_SIZE {
                // SAFETY: We checked that the end of the region doesn't exceed
                // `DeviceLayout::MAX_SIZE`.
                Some(unsafe { Region::new_unchecked(offset, size) })
            } else {
                None
            }
        }

        /// Creates a new `Region` from the given `offset` and `size` without doing any checks.
        ///
        /// # Safety
        ///
        /// - The end of the region must not exceed [`DeviceLayout::MAX_SIZE`], that is the
        ///   infinite-precision sum of `offset` and `size` must not exceed the bound.
        #[inline]
        pub const unsafe fn new_unchecked(offset: DeviceSize, size: DeviceSize) -> Self {
            Region { offset, size }
        }

        /// Returns the offset where the region begins.
        #[inline]
        pub const fn offset(&self) -> DeviceSize {
            self.offset
        }

        /// Returns the size of the region.
        #[inline]
        pub const fn size(&self) -> DeviceSize {
            self.size
        }
    }
}

/// Tells the [suballocator] what type of resource will be bound to the allocation, so that it can
/// optimize memory usage while still respecting the [buffer-image granularity].
///
/// [suballocator]: Suballocator
/// [buffer-image granularity]: super#buffer-image-granularity
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AllocationType {
    /// The type of resource is unknown, it might be either linear or non-linear. What this means
    /// is that allocations created with this type must always be aligned to the buffer-image
    /// granularity.
    Unknown = 0,

    /// The resource is linear, e.g. buffers, linear images. A linear allocation following another
    /// linear allocation never needs to be aligned to the buffer-image granularity.
    Linear = 1,

    /// The resource is non-linear, e.g. optimal images. A non-linear allocation following another
    /// non-linear allocation never needs to be aligned to the buffer-image granularity.
    NonLinear = 2,
}

impl From<ImageTiling> for AllocationType {
    #[inline]
    fn from(tiling: ImageTiling) -> Self {
        match tiling {
            ImageTiling::Optimal => AllocationType::NonLinear,
            ImageTiling::Linear => AllocationType::Linear,
            ImageTiling::DrmFormatModifier => AllocationType::Unknown,
        }
    }
}

/// An allocation made using a [suballocator].
///
/// [suballocator]: Suballocator
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Suballocation {
    /// The **absolute** offset within the [region]. That means that this is already offset by the
    /// region's offset, **not relative to beginning of the region**. This offset will be aligned
    /// to the requested alignment.
    ///
    /// [region]: Suballocator#regions
    pub offset: DeviceSize,

    /// The size of the allocation. This will be exactly equal to the requested size.
    pub size: DeviceSize,

    /// The type of resources that can be bound to this memory block. This will be exactly equal to
    /// the requested allocation type.
    pub allocation_type: AllocationType,

    /// An opaque handle identifying the allocation within the allocator.
    pub handle: AllocationHandle,
}

/// Error that can be returned when creating an [allocation] using a [suballocator].
///
/// [allocation]: Suballocation
/// [suballocator]: Suballocator
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SuballocatorError {
    /// There is no more space available in the region.
    OutOfRegionMemory,

    /// The region has enough free space to satisfy the request but is too fragmented.
    FragmentedRegion,
}

impl Error for SuballocatorError {}

impl Display for SuballocatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::OutOfRegionMemory => "out of region memory",
            Self::FragmentedRegion => "the region is too fragmented",
        };

        f.write_str(msg)
    }
}

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
    free_size: Cell<DeviceSize>,
    state: UnsafeCell<FreeListAllocatorState>,
}

unsafe impl Send for FreeListAllocator {}

unsafe impl Suballocator for FreeListAllocator {
    /// Creates a new `FreeListAllocator` for the given [region].
    ///
    /// [region]: Suballocator#regions
    fn new(region: Region) -> Self {
        let free_size = Cell::new(region.size());

        let node_allocator = slabbin::SlabAllocator::<SuballocationListNode>::new(32);
        let mut free_list = Vec::with_capacity(32);
        let root_ptr = node_allocator.allocate();
        let root = SuballocationListNode {
            prev: None,
            next: None,
            offset: region.offset(),
            size: region.size(),
            ty: SuballocationType::Free,
        };
        unsafe { root_ptr.as_ptr().write(root) };
        free_list.push(root_ptr);

        let state = UnsafeCell::new(FreeListAllocatorState {
            node_allocator,
            free_list,
        });

        FreeListAllocator {
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
        let state = unsafe { &mut *self.state.get() };

        match state.free_list.last() {
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
                    ty: SuballocationType::Unknown,
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
                let (Ok(index) | Err(index)) = state
                    .free_list
                    .binary_search_by_key(&dummy_node, |&ptr| unsafe { *ptr.as_ptr() });

                for (index, &node_ptr) in state.free_list.iter().enumerate().skip(index) {
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
                            ) && has_granularity_conflict(prev.ty, allocation_type)
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
                        state.free_list.remove(index);

                        // SAFETY:
                        // - `node` is free.
                        // - `offset` is that of `node`, possibly rounded up.
                        // - We checked that `offset + size` falls within `node`.
                        unsafe { state.split(node_ptr, offset, size) };

                        unsafe { (*node_ptr.as_ptr()).ty = allocation_type.into() };

                        // This can't overflow because suballocation sizes in the free-list are
                        // constrained by the remaining size of the region.
                        self.free_size.set(self.free_size.get() - size);

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
    unsafe fn deallocate(&self, suballocation: Suballocation) {
        let node_ptr = suballocation
            .handle
            .as_ptr()
            .cast::<SuballocationListNode>();

        // SAFETY: The caller must guarantee that `suballocation` refers to a currently allocated
        // allocation of `self`, which means that `node_ptr` is the same one we gave out on
        // allocation, making it a valid pointer.
        let node_ptr = unsafe { NonNull::new_unchecked(node_ptr) };
        let node = unsafe { *node_ptr.as_ptr() };

        debug_assert!(node.ty != SuballocationType::Free);

        // Suballocation sizes are constrained by the size of the region, so they can't possibly
        // overflow when added up.
        self.free_size.set(self.free_size.get() + node.size);

        unsafe { (*node_ptr.as_ptr()).ty = SuballocationType::Free };

        let state = unsafe { &mut *self.state.get() };

        unsafe { state.coalesce(node_ptr) };
        unsafe { state.deallocate(node_ptr) };
    }

    #[inline]
    fn free_size(&self) -> DeviceSize {
        self.free_size.get()
    }

    #[inline]
    fn cleanup(&mut self) {}
}

#[derive(Debug)]
struct FreeListAllocatorState {
    node_allocator: slabbin::SlabAllocator<SuballocationListNode>,
    // Free suballocations sorted by size in ascending order. This means we can always find a
    // best-fit in *O*(log(*n*)) time in the worst case, and iterating in order is very efficient.
    free_list: Vec<NonNull<SuballocationListNode>>,
}

#[derive(Clone, Copy, Debug)]
struct SuballocationListNode {
    prev: Option<NonNull<Self>>,
    next: Option<NonNull<Self>>,
    offset: DeviceSize,
    size: DeviceSize,
    ty: SuballocationType,
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

/// Tells us if a suballocation is free, and if not, whether it is linear or not. This is needed in
/// order to be able to respect the buffer-image granularity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SuballocationType {
    Unknown,
    Linear,
    NonLinear,
    Free,
}

impl From<AllocationType> for SuballocationType {
    fn from(ty: AllocationType) -> Self {
        match ty {
            AllocationType::Unknown => SuballocationType::Unknown,
            AllocationType::Linear => SuballocationType::Linear,
            AllocationType::NonLinear => SuballocationType::NonLinear,
        }
    }
}

impl FreeListAllocatorState {
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

        debug_assert!(node.ty == SuballocationType::Free);
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
                ty: SuballocationType::Free,
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
                ty: SuballocationType::Free,
            };
            unsafe { padding_ptr.as_ptr().write(padding) };

            if let Some(next_ptr) = padding.next {
                unsafe { (*next_ptr.as_ptr()).prev = Some(padding_ptr) };
            }

            unsafe { (*node_ptr.as_ptr()).next = Some(padding_ptr) };
            // This is overflow-safe for the same reason as above.
            unsafe { (*node_ptr.as_ptr()).size -= padding.size };

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

        debug_assert!(node.ty == SuballocationType::Free);

        if let Some(prev_ptr) = node.prev {
            let prev = unsafe { *prev_ptr.as_ptr() };

            if prev.ty == SuballocationType::Free {
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

                // SAFETY:
                // - The suballocation is free.
                // - The suballocation was removed from the free-list.
                // - The next suballocation and possibly a previous suballocation have been updated
                //   such that they no longer reference the suballocation.
                // All of these conditions combined guarantee that `prev_ptr` cannot be used again.
                unsafe { self.node_allocator.deallocate(prev_ptr) };
            }
        }

        if let Some(next_ptr) = node.next {
            let next = unsafe { *next_ptr.as_ptr() };

            if next.ty == SuballocationType::Free {
                // SAFETY: Same as above.
                self.allocate(next_ptr);

                unsafe { (*node_ptr.as_ptr()).next = next.next };
                // This is overflow-safe for the same reason as above.
                unsafe { (*node_ptr.as_ptr()).size += next.size };

                if let Some(next_ptr) = next.next {
                    unsafe { (*next_ptr.as_ptr()).prev = Some(node_ptr) };
                }

                // SAFETY: Same as above.
                unsafe { self.node_allocator.deallocate(next_ptr) };
            }
        }
    }
}

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
    const MIN_NODE_SIZE: DeviceSize = 16;

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

/// Checks if resouces A and B share a page.
///
/// > **Note**: Assumes `a_offset + a_size > 0` and `a_offset + a_size <= b_offset`.
fn are_blocks_on_same_page(
    a_offset: DeviceSize,
    a_size: DeviceSize,
    b_offset: DeviceSize,
    page_size: DeviceAlignment,
) -> bool {
    debug_assert!(a_offset + a_size > 0);
    debug_assert!(a_offset + a_size <= b_offset);

    let a_end = a_offset + a_size - 1;
    let a_end_page = align_down(a_end, page_size);
    let b_start_page = align_down(b_offset, page_size);

    a_end_page == b_start_page
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam_queue::ArrayQueue;
    use parking_lot::Mutex;
    use std::thread;

    const fn unwrap<T: Copy>(opt: Option<T>) -> T {
        match opt {
            Some(x) => x,
            None => panic!(),
        }
    }

    const DUMMY_LAYOUT: DeviceLayout = unwrap(DeviceLayout::from_size_alignment(1, 1));

    #[test]
    fn free_list_allocator_capacity() {
        const THREADS: DeviceSize = 12;
        const ALLOCATIONS_PER_THREAD: DeviceSize = 100;
        const ALLOCATION_STEP: DeviceSize = 117;
        const REGION_SIZE: DeviceSize =
            (ALLOCATION_STEP * (THREADS + 1) * THREADS / 2) * ALLOCATIONS_PER_THREAD;

        let allocator = Mutex::new(FreeListAllocator::new(Region::new(0, REGION_SIZE).unwrap()));
        let allocs = ArrayQueue::new((ALLOCATIONS_PER_THREAD * THREADS) as usize);

        // Using threads to randomize allocation order.
        thread::scope(|scope| {
            for i in 1..=THREADS {
                let (allocator, allocs) = (&allocator, &allocs);

                scope.spawn(move || {
                    let layout = DeviceLayout::from_size_alignment(i * ALLOCATION_STEP, 1).unwrap();

                    for _ in 0..ALLOCATIONS_PER_THREAD {
                        allocs
                            .push(
                                allocator
                                    .lock()
                                    .allocate(layout, AllocationType::Unknown, DeviceAlignment::MIN)
                                    .unwrap(),
                            )
                            .unwrap();
                    }
                });
            }
        });

        let allocator = allocator.into_inner();

        assert!(allocator
            .allocate(DUMMY_LAYOUT, AllocationType::Unknown, DeviceAlignment::MIN)
            .is_err());
        assert_eq!(allocator.free_size(), 0);

        for alloc in allocs {
            unsafe { allocator.deallocate(alloc) };
        }

        assert_eq!(allocator.free_size(), REGION_SIZE);
        let alloc = allocator
            .allocate(
                DeviceLayout::from_size_alignment(REGION_SIZE, 1).unwrap(),
                AllocationType::Unknown,
                DeviceAlignment::MIN,
            )
            .unwrap();
        unsafe { allocator.deallocate(alloc) };
    }

    #[test]
    fn free_list_allocator_respects_alignment() {
        const REGION_SIZE: DeviceSize = 10 * 256;
        const LAYOUT: DeviceLayout = unwrap(DeviceLayout::from_size_alignment(1, 256));

        let allocator = FreeListAllocator::new(Region::new(0, REGION_SIZE).unwrap());
        let mut allocs = Vec::with_capacity(10);

        for _ in 0..10 {
            allocs.push(
                allocator
                    .allocate(LAYOUT, AllocationType::Unknown, DeviceAlignment::MIN)
                    .unwrap(),
            );
        }

        assert!(allocator
            .allocate(LAYOUT, AllocationType::Unknown, DeviceAlignment::MIN)
            .is_err());
        assert_eq!(allocator.free_size(), REGION_SIZE - 10);

        for alloc in allocs.drain(..) {
            unsafe { allocator.deallocate(alloc) };
        }
    }

    #[test]
    fn free_list_allocator_respects_granularity() {
        const GRANULARITY: DeviceAlignment = unwrap(DeviceAlignment::new(16));
        const REGION_SIZE: DeviceSize = 2 * GRANULARITY.as_devicesize();

        let allocator = FreeListAllocator::new(Region::new(0, REGION_SIZE).unwrap());
        let mut linear_allocs = Vec::with_capacity(REGION_SIZE as usize / 2);
        let mut nonlinear_allocs = Vec::with_capacity(REGION_SIZE as usize / 2);

        for i in 0..REGION_SIZE {
            if i % 2 == 0 {
                linear_allocs.push(
                    allocator
                        .allocate(DUMMY_LAYOUT, AllocationType::Linear, GRANULARITY)
                        .unwrap(),
                );
            } else {
                nonlinear_allocs.push(
                    allocator
                        .allocate(DUMMY_LAYOUT, AllocationType::NonLinear, GRANULARITY)
                        .unwrap(),
                );
            }
        }

        assert!(allocator
            .allocate(DUMMY_LAYOUT, AllocationType::Linear, GRANULARITY)
            .is_err());
        assert_eq!(allocator.free_size(), 0);

        for alloc in linear_allocs.drain(..) {
            unsafe { allocator.deallocate(alloc) };
        }

        let alloc = allocator
            .allocate(
                DeviceLayout::from_size_alignment(GRANULARITY.as_devicesize(), 1).unwrap(),
                AllocationType::Unknown,
                GRANULARITY,
            )
            .unwrap();
        unsafe { allocator.deallocate(alloc) };

        let alloc = allocator
            .allocate(DUMMY_LAYOUT, AllocationType::Unknown, GRANULARITY)
            .unwrap();
        assert!(allocator
            .allocate(DUMMY_LAYOUT, AllocationType::Unknown, GRANULARITY)
            .is_err());
        assert!(allocator
            .allocate(DUMMY_LAYOUT, AllocationType::Linear, GRANULARITY)
            .is_err());
        unsafe { allocator.deallocate(alloc) };

        for alloc in nonlinear_allocs.drain(..) {
            unsafe { allocator.deallocate(alloc) };
        }
    }

    #[test]
    fn buddy_allocator_capacity() {
        const MAX_ORDER: usize = 10;
        const REGION_SIZE: DeviceSize = BuddyAllocator::MIN_NODE_SIZE << MAX_ORDER;

        let allocator = BuddyAllocator::new(Region::new(0, REGION_SIZE).unwrap());
        let mut allocs = Vec::with_capacity(1 << MAX_ORDER);

        for order in 0..=MAX_ORDER {
            let layout =
                DeviceLayout::from_size_alignment(BuddyAllocator::MIN_NODE_SIZE << order, 1)
                    .unwrap();

            for _ in 0..1 << (MAX_ORDER - order) {
                allocs.push(
                    allocator
                        .allocate(layout, AllocationType::Unknown, DeviceAlignment::MIN)
                        .unwrap(),
                );
            }

            assert!(allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Unknown, DeviceAlignment::MIN)
                .is_err());
            assert_eq!(allocator.free_size(), 0);

            for alloc in allocs.drain(..) {
                unsafe { allocator.deallocate(alloc) };
            }
        }

        let mut orders = (0..MAX_ORDER).collect::<Vec<_>>();

        for mid in 0..MAX_ORDER {
            orders.rotate_left(mid);

            for &order in &orders {
                let layout =
                    DeviceLayout::from_size_alignment(BuddyAllocator::MIN_NODE_SIZE << order, 1)
                        .unwrap();

                allocs.push(
                    allocator
                        .allocate(layout, AllocationType::Unknown, DeviceAlignment::MIN)
                        .unwrap(),
                );
            }

            let alloc = allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Unknown, DeviceAlignment::MIN)
                .unwrap();
            assert!(allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Unknown, DeviceAlignment::MIN)
                .is_err());
            assert_eq!(allocator.free_size(), 0);
            unsafe { allocator.deallocate(alloc) };

            for alloc in allocs.drain(..) {
                unsafe { allocator.deallocate(alloc) };
            }
        }
    }

    #[test]
    fn buddy_allocator_respects_alignment() {
        const REGION_SIZE: DeviceSize = 4096;

        let allocator = BuddyAllocator::new(Region::new(0, REGION_SIZE).unwrap());

        {
            let layout = DeviceLayout::from_size_alignment(1, 4096).unwrap();

            let alloc = allocator
                .allocate(layout, AllocationType::Unknown, DeviceAlignment::MIN)
                .unwrap();
            assert!(allocator
                .allocate(layout, AllocationType::Unknown, DeviceAlignment::MIN)
                .is_err());
            assert_eq!(
                allocator.free_size(),
                REGION_SIZE - BuddyAllocator::MIN_NODE_SIZE,
            );
            unsafe { allocator.deallocate(alloc) };
        }

        {
            let layout_a = DeviceLayout::from_size_alignment(1, 256).unwrap();
            let allocations_a = REGION_SIZE / layout_a.alignment().as_devicesize();
            let layout_b = DeviceLayout::from_size_alignment(1, 16).unwrap();
            let allocations_b = REGION_SIZE / layout_b.alignment().as_devicesize() - allocations_a;

            let mut allocs =
                Vec::with_capacity((REGION_SIZE / BuddyAllocator::MIN_NODE_SIZE) as usize);

            for _ in 0..allocations_a {
                allocs.push(
                    allocator
                        .allocate(layout_a, AllocationType::Unknown, DeviceAlignment::MIN)
                        .unwrap(),
                );
            }

            assert!(allocator
                .allocate(layout_a, AllocationType::Unknown, DeviceAlignment::MIN)
                .is_err());
            assert_eq!(
                allocator.free_size(),
                REGION_SIZE - allocations_a * BuddyAllocator::MIN_NODE_SIZE,
            );

            for _ in 0..allocations_b {
                allocs.push(
                    allocator
                        .allocate(layout_b, AllocationType::Unknown, DeviceAlignment::MIN)
                        .unwrap(),
                );
            }

            assert!(allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Unknown, DeviceAlignment::MIN)
                .is_err());
            assert_eq!(allocator.free_size(), 0);

            for alloc in allocs {
                unsafe { allocator.deallocate(alloc) };
            }
        }
    }

    #[test]
    fn buddy_allocator_respects_granularity() {
        const GRANULARITY: DeviceAlignment = unwrap(DeviceAlignment::new(256));
        const REGION_SIZE: DeviceSize = 2 * GRANULARITY.as_devicesize();

        let allocator = BuddyAllocator::new(Region::new(0, REGION_SIZE).unwrap());

        {
            const ALLOCATIONS: DeviceSize = REGION_SIZE / BuddyAllocator::MIN_NODE_SIZE;

            let mut allocs = Vec::with_capacity(ALLOCATIONS as usize);

            for _ in 0..ALLOCATIONS {
                allocs.push(
                    allocator
                        .allocate(DUMMY_LAYOUT, AllocationType::Linear, GRANULARITY)
                        .unwrap(),
                );
            }

            assert!(allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Linear, GRANULARITY)
                .is_err());
            assert_eq!(allocator.free_size(), 0);

            for alloc in allocs {
                unsafe { allocator.deallocate(alloc) };
            }
        }

        {
            let alloc1 = allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Unknown, GRANULARITY)
                .unwrap();
            let alloc2 = allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Unknown, GRANULARITY)
                .unwrap();
            assert!(allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Linear, GRANULARITY)
                .is_err());
            assert_eq!(allocator.free_size(), 0);
            unsafe { allocator.deallocate(alloc1) };
            unsafe { allocator.deallocate(alloc2) };
        }
    }

    #[test]
    fn bump_allocator_respects_alignment() {
        const ALIGNMENT: DeviceSize = 16;
        const REGION_SIZE: DeviceSize = 10 * ALIGNMENT;

        let layout = DeviceLayout::from_size_alignment(1, ALIGNMENT).unwrap();
        let mut allocator = BumpAllocator::new(Region::new(0, REGION_SIZE).unwrap());

        for _ in 0..10 {
            allocator
                .allocate(layout, AllocationType::Unknown, DeviceAlignment::MIN)
                .unwrap();
        }

        assert!(allocator
            .allocate(layout, AllocationType::Unknown, DeviceAlignment::MIN)
            .is_err());

        for _ in 0..ALIGNMENT - 1 {
            allocator
                .allocate(DUMMY_LAYOUT, AllocationType::Unknown, DeviceAlignment::MIN)
                .unwrap();
        }

        assert!(allocator
            .allocate(layout, AllocationType::Unknown, DeviceAlignment::MIN)
            .is_err());
        assert_eq!(allocator.free_size(), 0);

        allocator.reset();
        assert_eq!(allocator.free_size(), REGION_SIZE);
    }

    #[test]
    fn bump_allocator_respects_granularity() {
        const ALLOCATIONS: DeviceSize = 10;
        const GRANULARITY: DeviceAlignment = unwrap(DeviceAlignment::new(1024));
        const REGION_SIZE: DeviceSize = ALLOCATIONS * GRANULARITY.as_devicesize();

        let mut allocator = BumpAllocator::new(Region::new(0, REGION_SIZE).unwrap());

        for i in 0..ALLOCATIONS {
            for _ in 0..GRANULARITY.as_devicesize() {
                allocator
                    .allocate(
                        DUMMY_LAYOUT,
                        if i % 2 == 0 {
                            AllocationType::NonLinear
                        } else {
                            AllocationType::Linear
                        },
                        GRANULARITY,
                    )
                    .unwrap();
            }
        }

        assert!(allocator
            .allocate(DUMMY_LAYOUT, AllocationType::Linear, GRANULARITY)
            .is_err());
        assert_eq!(allocator.free_size(), 0);

        allocator.reset();

        for i in 0..ALLOCATIONS {
            allocator
                .allocate(
                    DUMMY_LAYOUT,
                    if i % 2 == 0 {
                        AllocationType::Linear
                    } else {
                        AllocationType::NonLinear
                    },
                    GRANULARITY,
                )
                .unwrap();
        }

        assert!(allocator
            .allocate(DUMMY_LAYOUT, AllocationType::Linear, GRANULARITY)
            .is_err());
        assert_eq!(allocator.free_size(), GRANULARITY.as_devicesize() - 1);

        allocator.reset();
        assert_eq!(allocator.free_size(), REGION_SIZE);
    }
}
