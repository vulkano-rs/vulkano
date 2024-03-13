//! Suballocators are used to divide a *region* into smaller *suballocations*.
//!
//! See also [the parent module] for details about memory allocation in Vulkan.
//!
//! [the parent module]: super

pub use self::{
    buddy::BuddyAllocator, bump::BumpAllocator, free_list::FreeListAllocator, region::Region,
};
use super::{align_down, AllocationHandle, DeviceAlignment, DeviceLayout};
use crate::{image::ImageTiling, DeviceSize};
use std::{
    error::Error,
    fmt::{self, Debug, Display},
};

mod buddy;
mod bump;
mod free_list;

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
    type Suballocations<'a>: Iterator<Item = SuballocationNode>
        + DoubleEndedIterator
        + ExactSizeIterator
    where
        Self: Sized + 'a;

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
        &mut self,
        layout: DeviceLayout,
        allocation_type: AllocationType,
        buffer_image_granularity: DeviceAlignment,
    ) -> Result<Suballocation, SuballocatorError>;

    /// Deallocates the given `suballocation`.
    ///
    /// # Safety
    ///
    /// - `suballocation` must refer to a **currently allocated** suballocation of `self`.
    unsafe fn deallocate(&mut self, suballocation: Suballocation);

    /// Returns the total amount of free space that is left in the [region].
    ///
    /// [region]: Self#regions
    fn free_size(&self) -> DeviceSize;

    /// Tries to free some space, if applicable.
    ///
    /// There must be no current allocations as they might get freed.
    fn cleanup(&mut self);

    /// Returns an iterator over the current suballocations.
    fn suballocations(&self) -> Self::Suballocations<'_>
    where
        Self: Sized;
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

/// A node within a [suballocator]'s list/tree of suballocations.
///
/// [suballocator]: Suballocator
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SuballocationNode {
    /// The **absolute** offset within the [region]. That means that this is already offset by the
    /// region's offset, **not relative to beginning of the region**.
    ///
    /// [region]: Suballocator#regions
    pub offset: DeviceSize,

    /// The size of the allocation.
    pub size: DeviceSize,

    /// Tells us if the allocation is free, and if not, what type of resources can be bound to it.
    pub allocation_type: SuballocationType,
}

/// Tells us if an allocation within a [suballocator]'s list/tree of suballocations is free, and if
/// not, what type of resources can be bound to it. The suballocator needs to keep track of this in
/// order to be able to respect the buffer-image granularity.
///
/// [suballocator]: Suballocator
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SuballocationType {
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

    /// The allocation is free. It can take on any of the allocation types once allocated.
    Free = 3,
}

impl From<AllocationType> for SuballocationType {
    #[inline]
    fn from(ty: AllocationType) -> Self {
        match ty {
            AllocationType::Unknown => SuballocationType::Unknown,
            AllocationType::Linear => SuballocationType::Linear,
            AllocationType::NonLinear => SuballocationType::NonLinear,
        }
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

        let mut allocator = allocator.into_inner();

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

        let mut allocator = FreeListAllocator::new(Region::new(0, REGION_SIZE).unwrap());
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

        let mut allocator = FreeListAllocator::new(Region::new(0, REGION_SIZE).unwrap());
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

        let mut allocator = BuddyAllocator::new(Region::new(0, REGION_SIZE).unwrap());
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

        let mut allocator = BuddyAllocator::new(Region::new(0, REGION_SIZE).unwrap());

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

        let mut allocator = BuddyAllocator::new(Region::new(0, REGION_SIZE).unwrap());

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
