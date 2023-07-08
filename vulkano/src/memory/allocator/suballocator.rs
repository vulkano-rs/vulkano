// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Suballocators are used to divide a *region* into smaller *suballocations*.
//!
//! See also [the parent module] for details about memory allocation in Vulkan.
//!
//! [the parent module]: super

use self::host::SlotId;
use super::{
    align_down, align_up, array_vec::ArrayVec, DeviceAlignment, DeviceLayout, MemoryAllocatorError,
};
use crate::{
    device::{Device, DeviceOwned},
    image::ImageTiling,
    memory::{is_aligned, DeviceMemory, MemoryPropertyFlags},
    DeviceSize, NonZeroDeviceSize, VulkanError, VulkanObject,
};
use crossbeam_queue::ArrayQueue;
use parking_lot::Mutex;
use std::{
    cell::Cell,
    cmp,
    error::Error,
    ffi::c_void,
    fmt::{self, Display},
    mem::{self, ManuallyDrop, MaybeUninit},
    ops::Range,
    ptr::{self, NonNull},
    slice,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

/// Memory allocations are portions of memory that are reserved for a specific resource or purpose.
///
/// There's a few ways you can obtain a `MemoryAlloc` in Vulkano. Most commonly you will probably
/// want to use a [memory allocator]. If you already have a [`DeviceMemory`] block on hand that you
/// would like to turn into an allocation, you can use [the constructor]. Lastly, you can use a
/// [suballocator] if you want to create multiple smaller allocations out of a bigger one.
///
/// [memory allocator]: super::MemoryAllocator
/// [the constructor]: Self::new
/// [suballocator]: Suballocator
#[derive(Debug)]
pub struct MemoryAlloc {
    offset: DeviceSize,
    size: DeviceSize,
    // Needed when binding resources to the allocation in order to avoid aliasing memory.
    allocation_type: AllocationType,
    // Mapped pointer to the start of the allocation or `None` is the memory is not host-visible.
    mapped_ptr: Option<NonNull<c_void>>,
    // Used by the suballocators to align allocations to the non-coherent atom size when the memory
    // type is host-visible but not host-coherent. This will be `None` for any other memory type.
    atom_size: Option<DeviceAlignment>,
    // Used in the `Drop` impl to free the allocation if required.
    parent: AllocParent,
}

#[derive(Debug)]
enum AllocParent {
    FreeList {
        allocator: Arc<FreeListAllocator>,
        id: SlotId,
    },
    Buddy {
        allocator: Arc<BuddyAllocator>,
        order: usize,
        offset: DeviceSize,
    },
    Pool {
        allocator: Arc<PoolAllocatorInner>,
        index: DeviceSize,
    },
    Bump(Arc<BumpAllocator>),
    Root(Arc<DeviceMemory>),
    Dedicated(DeviceMemory),
}

// It is safe to share `mapped_ptr` between threads because the user would have to use unsafe code
// themself to get UB in the first place.
unsafe impl Send for MemoryAlloc {}
unsafe impl Sync for MemoryAlloc {}

impl MemoryAlloc {
    /// Creates a new `MemoryAlloc`.
    ///
    /// The memory is mapped automatically if it's host-visible.
    #[inline]
    pub fn new(device_memory: DeviceMemory) -> Result<Self, MemoryAllocatorError> {
        // Sanity check: this would lead to UB when suballocating.
        assert!(device_memory.allocation_size() <= DeviceLayout::MAX_SIZE);

        let device = device_memory.device();
        let physical_device = device.physical_device();
        let memory_type_index = device_memory.memory_type_index();
        let property_flags = &physical_device.memory_properties().memory_types
            [memory_type_index as usize]
            .property_flags;

        let mapped_ptr = if property_flags.intersects(MemoryPropertyFlags::HOST_VISIBLE) {
            // Sanity check: this would lead to UB when calculating pointer offsets.
            assert!(device_memory.allocation_size() <= isize::MAX.try_into().unwrap());

            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            // This is always valid because we are mapping the whole range.
            unsafe {
                (fns.v1_0.map_memory)(
                    device.handle(),
                    device_memory.handle(),
                    0,
                    ash::vk::WHOLE_SIZE,
                    ash::vk::MemoryMapFlags::empty(),
                    output.as_mut_ptr(),
                )
                .result()
                .map_err(VulkanError::from)?;

                Some(NonNull::new(output.assume_init()).unwrap())
            }
        } else {
            None
        };

        let atom_size = (property_flags.intersects(MemoryPropertyFlags::HOST_VISIBLE)
            && !property_flags.intersects(MemoryPropertyFlags::HOST_COHERENT))
        .then_some(physical_device.properties().non_coherent_atom_size);

        Ok(MemoryAlloc {
            offset: 0,
            size: device_memory.allocation_size(),
            allocation_type: AllocationType::Unknown,
            mapped_ptr,
            atom_size,
            parent: if device_memory.is_dedicated() {
                AllocParent::Dedicated(device_memory)
            } else {
                AllocParent::Root(Arc::new(device_memory))
            },
        })
    }

    /// Returns the offset of the allocation within the [`DeviceMemory`] block.
    #[inline]
    pub fn offset(&self) -> DeviceSize {
        self.offset
    }

    /// Returns the size of the allocation.
    #[inline]
    pub fn size(&self) -> DeviceSize {
        self.size
    }

    /// Returns the type of resources that can be bound to this allocation.
    #[inline]
    pub fn allocation_type(&self) -> AllocationType {
        self.allocation_type
    }

    /// Returns the mapped pointer to the start of the allocation if the memory is host-visible,
    /// otherwise returns [`None`].
    #[inline]
    pub fn mapped_ptr(&self) -> Option<NonNull<c_void>> {
        self.mapped_ptr
    }

    /// Returns a mapped slice to the data within the allocation if the memory is host-visible,
    /// otherwise returns [`None`].
    ///
    /// # Safety
    ///
    /// - While the returned slice exists, there must be no operations pending or executing in a
    ///   GPU queue that write to the same memory.
    #[inline]
    pub unsafe fn mapped_slice(&self) -> Option<&[u8]> {
        self.mapped_ptr
            .map(|ptr| slice::from_raw_parts(ptr.as_ptr().cast(), self.size as usize))
    }

    /// Returns a mapped mutable slice to the data within the allocation if the memory is
    /// host-visible, otherwise returns [`None`].
    ///
    /// # Safety
    ///
    /// - While the returned slice exists, there must be no operations pending or executing in a
    ///   GPU queue that access the same memory.
    #[inline]
    pub unsafe fn mapped_slice_mut(&mut self) -> Option<&mut [u8]> {
        self.mapped_ptr
            .map(|ptr| slice::from_raw_parts_mut(ptr.as_ptr().cast(), self.size as usize))
    }

    pub(crate) fn atom_size(&self) -> Option<DeviceAlignment> {
        self.atom_size
    }

    /// Invalidates the host (CPU) cache for a range of the allocation.
    ///
    /// You must call this method before the memory is read by the host, if the device previously
    /// wrote to the memory. It has no effect if the memory is not mapped or if the memory is
    /// [host-coherent].
    ///
    /// `range` is specified in bytes relative to the start of the allocation. The start and end of
    /// `range` must be a multiple of the [`non_coherent_atom_size`] device property, but
    /// `range.end` can also equal to `self.size()`.
    ///
    /// # Safety
    ///
    /// - If there are memory writes by the GPU that have not been propagated into the CPU cache,
    ///   then there must not be any references in Rust code to the specified `range` of the memory.
    ///
    /// # Panics
    ///
    /// - Panics if `range` is empty.
    /// - Panics if `range.end` exceeds `self.size`.
    /// - Panics if `range.start` or `range.end` are not a multiple of the `non_coherent_atom_size`.
    ///
    /// [host-coherent]: crate::memory::MemoryPropertyFlags::HOST_COHERENT
    /// [`non_coherent_atom_size`]: crate::device::Properties::non_coherent_atom_size
    #[inline]
    pub unsafe fn invalidate_range(&self, range: Range<DeviceSize>) -> Result<(), VulkanError> {
        // VUID-VkMappedMemoryRange-memory-00684
        if let Some(atom_size) = self.atom_size {
            let range = self.create_memory_range(range, atom_size);
            let device = self.device();
            let fns = device.fns();
            (fns.v1_0.invalidate_mapped_memory_ranges)(device.handle(), 1, &range)
                .result()
                .map_err(VulkanError::from)?;
        } else {
            self.debug_validate_memory_range(&range);
        }

        Ok(())
    }

    /// Flushes the host (CPU) cache for a range of the allocation.
    ///
    /// You must call this method after writing to the memory from the host, if the device is going
    /// to read the memory. It has no effect if the memory is not mapped or if the memory is
    /// [host-coherent].
    ///
    /// `range` is specified in bytes relative to the start of the allocation. The start and end of
    /// `range` must be a multiple of the [`non_coherent_atom_size`] device property, but
    /// `range.end` can also equal to `self.size()`.
    ///
    /// # Safety
    ///
    /// - There must be no operations pending or executing in a GPU queue that access the specified
    ///   `range` of the memory.
    ///
    /// # Panics
    ///
    /// - Panics if `range` is empty.
    /// - Panics if `range.end` exceeds `self.size`.
    /// - Panics if `range.start` or `range.end` are not a multiple of the `non_coherent_atom_size`.
    ///
    /// [host-coherent]: crate::memory::MemoryPropertyFlags::HOST_COHERENT
    /// [`non_coherent_atom_size`]: crate::device::Properties::non_coherent_atom_size
    #[inline]
    pub unsafe fn flush_range(&self, range: Range<DeviceSize>) -> Result<(), VulkanError> {
        // VUID-VkMappedMemoryRange-memory-00684
        if let Some(atom_size) = self.atom_size {
            let range = self.create_memory_range(range, atom_size);
            let device = self.device();
            let fns = device.fns();
            (fns.v1_0.flush_mapped_memory_ranges)(device.handle(), 1, &range)
                .result()
                .map_err(VulkanError::from)?;
        } else {
            self.debug_validate_memory_range(&range);
        }

        Ok(())
    }

    fn create_memory_range(
        &self,
        range: Range<DeviceSize>,
        atom_size: DeviceAlignment,
    ) -> ash::vk::MappedMemoryRange {
        assert!(!range.is_empty() && range.end <= self.size);

        // VUID-VkMappedMemoryRange-size-00685
        // Guaranteed because we always map the entire `DeviceMemory`.

        // VUID-VkMappedMemoryRange-offset-00687
        // VUID-VkMappedMemoryRange-size-01390
        assert!(
            is_aligned(range.start, atom_size)
                && (is_aligned(range.end, atom_size) || range.end == self.size)
        );

        // VUID-VkMappedMemoryRange-offset-00687
        // Guaranteed as long as `range.start` is aligned because the suballocators always align
        // `self.offset` to the non-coherent atom size for non-coherent host-visible memory.
        let offset = self.offset + range.start;

        let mut size = range.end - range.start;
        let device_memory = self.device_memory();

        // VUID-VkMappedMemoryRange-size-01390
        if offset + size < device_memory.allocation_size() {
            // We align the size in case `range.end == self.size`. We can do this without aliasing
            // other allocations because the suballocators ensure that all allocations are aligned
            // to the atom size for non-coherent host-visible memory.
            size = align_up(size, atom_size);
        }

        ash::vk::MappedMemoryRange {
            memory: device_memory.handle(),
            offset,
            size,
            ..Default::default()
        }
    }

    /// This exists because even if no cache control is required, the parameters should still be
    /// valid, otherwise you might have bugs in your code forever just because your memory happens
    /// to be host-coherent.
    fn debug_validate_memory_range(&self, range: &Range<DeviceSize>) {
        debug_assert!(!range.is_empty() && range.end <= self.size);

        let atom_size = self
            .device()
            .physical_device()
            .properties()
            .non_coherent_atom_size;
        debug_assert!(
            is_aligned(range.start, atom_size)
                && (is_aligned(range.end, atom_size) || range.end == self.size),
            "attempted to invalidate or flush a memory range that is not aligned to the \
            non-coherent atom size",
        );
    }

    /// Returns the underlying block of [`DeviceMemory`].
    #[inline]
    pub fn device_memory(&self) -> &DeviceMemory {
        match &self.parent {
            AllocParent::FreeList { allocator, .. } => &allocator.device_memory,
            AllocParent::Buddy { allocator, .. } => &allocator.device_memory,
            AllocParent::Pool { allocator, .. } => &allocator.device_memory,
            AllocParent::Bump(allocator) => &allocator.device_memory,
            AllocParent::Root(device_memory) => device_memory,
            AllocParent::Dedicated(device_memory) => device_memory,
        }
    }

    /// Returns the parent allocation if this allocation is a [suballocation], otherwise returns
    /// [`None`].
    ///
    /// [suballocation]: Suballocator
    #[inline]
    pub fn parent_allocation(&self) -> Option<&Self> {
        match &self.parent {
            AllocParent::FreeList { allocator, .. } => Some(&allocator.region),
            AllocParent::Buddy { allocator, .. } => Some(&allocator.region),
            AllocParent::Pool { allocator, .. } => Some(&allocator.region),
            AllocParent::Bump(allocator) => Some(&allocator.region),
            AllocParent::Root(_) => None,
            AllocParent::Dedicated(_) => None,
        }
    }

    /// Returns `true` if this allocation is the root of the [memory hierarchy].
    ///
    /// [memory hierarchy]: Suballocator#memory-hierarchies
    #[inline]
    pub fn is_root(&self) -> bool {
        matches!(&self.parent, AllocParent::Root(_))
    }

    /// Returns `true` if this allocation is a [dedicated allocation].
    ///
    /// [dedicated allocation]: crate::memory::MemoryAllocateInfo#structfield.dedicated_allocation
    #[inline]
    pub fn is_dedicated(&self) -> bool {
        matches!(&self.parent, AllocParent::Dedicated(_))
    }

    /// Returns the underlying block of [`DeviceMemory`] if this allocation [is the root
    /// allocation] and is not [aliased], otherwise returns the allocation back wrapped in [`Err`].
    ///
    /// [is the root allocation]: Self::is_root
    /// [aliased]: Self::alias
    #[inline]
    pub fn try_unwrap(self) -> Result<DeviceMemory, Self> {
        let this = ManuallyDrop::new(self);

        // SAFETY: This is safe because even if a panic happens, `self.parent` can not be
        // double-freed since `self` was wrapped in `ManuallyDrop`. If we fail to unwrap the
        // `DeviceMemory`, the copy of `self.parent` is forgotten and only then is the
        // `ManuallyDrop` wrapper removed from `self`.
        match unsafe { ptr::read(&this.parent) } {
            AllocParent::Root(device_memory) => {
                Arc::try_unwrap(device_memory).map_err(|device_memory| {
                    mem::forget(device_memory);
                    ManuallyDrop::into_inner(this)
                })
            }
            parent => {
                mem::forget(parent);
                Err(ManuallyDrop::into_inner(this))
            }
        }
    }

    /// Duplicates the allocation, creating aliased memory. Returns [`None`] if the allocation [is
    /// a dedicated allocation].
    ///
    /// You might consider using this method if you want to optimize memory usage by aliasing
    /// render targets for example, in which case you will have to double and triple check that the
    /// memory is not used concurrently unless it only involves reading. You are highly discouraged
    /// from doing this unless you have a reason to.
    ///
    /// # Safety
    ///
    /// - You must ensure memory accesses are synchronized yourself.
    ///
    /// [memory hierarchy]: Suballocator#memory-hierarchies
    /// [is a dedicated allocation]: Self::is_dedicated
    #[inline]
    pub unsafe fn alias(&self) -> Option<Self> {
        self.root().map(|device_memory| MemoryAlloc {
            parent: AllocParent::Root(device_memory.clone()),
            ..*self
        })
    }

    fn root(&self) -> Option<&Arc<DeviceMemory>> {
        match &self.parent {
            AllocParent::FreeList { allocator, .. } => Some(&allocator.device_memory),
            AllocParent::Buddy { allocator, .. } => Some(&allocator.device_memory),
            AllocParent::Pool { allocator, .. } => Some(&allocator.device_memory),
            AllocParent::Bump(allocator) => Some(&allocator.device_memory),
            AllocParent::Root(device_memory) => Some(device_memory),
            AllocParent::Dedicated(_) => None,
        }
    }

    /// Increases the offset of the allocation by the specified `amount` and shrinks its size by
    /// the same amount.
    ///
    /// # Panics
    ///
    /// - Panics if the `amount` exceeds the size of the allocation.
    #[inline]
    pub fn shift(&mut self, amount: DeviceSize) {
        assert!(amount <= self.size);

        unsafe { self.set_offset(self.offset + amount) };
        self.size -= amount;
    }

    /// Shrinks the size of the allocation to the specified `new_size`.
    ///
    /// # Panics
    ///
    /// - Panics if the `new_size` exceeds the current size of the allocation.
    #[inline]
    pub fn shrink(&mut self, new_size: DeviceSize) {
        assert!(new_size <= self.size);

        self.size = new_size;
    }

    /// Sets the offset of the allocation without checking for memory aliasing.
    ///
    /// See also [`shift`], which moves the offset safely.
    ///
    /// # Safety
    ///
    /// - You must ensure that the allocation doesn't alias any other allocations within the
    ///   [`DeviceMemory`] block, and if it does, then you must ensure memory accesses are
    ///   synchronized yourself.
    /// - You must ensure the allocation still fits inside the `DeviceMemory` block.
    ///
    /// [`shift`]: Self::shift
    #[inline]
    pub unsafe fn set_offset(&mut self, new_offset: DeviceSize) {
        if let Some(ptr) = self.mapped_ptr.as_mut() {
            *ptr = NonNull::new_unchecked(
                ptr.as_ptr()
                    .offset(new_offset as isize - self.offset as isize),
            );
        }
        self.offset = new_offset;
    }

    /// Sets the size of the allocation without checking for memory aliasing.
    ///
    /// See also [`shrink`], which sets the size safely.
    ///
    /// # Safety
    ///
    /// - You must ensure that the allocation doesn't alias any other allocations within the
    ///   [`DeviceMemory`] block, and if it does, then you must ensure memory accesses are
    ///   synchronized yourself.
    /// - You must ensure the allocation still fits inside the `DeviceMemory` block.
    ///
    /// [`shrink`]: Self::shrink
    #[inline]
    pub unsafe fn set_size(&mut self, new_size: DeviceSize) {
        self.size = new_size;
    }

    /// Sets the allocation type.
    ///
    /// This might cause memory aliasing due to [buffer-image granularity] conflicts if the
    /// allocation type is [`Linear`] or [`NonLinear`] and is changed to a different one.
    ///
    /// # Safety
    ///
    /// - You must ensure that the allocation doesn't alias any other allocations within the
    ///   [`DeviceMemory`] block, and if it does, then you must ensure memory accesses are
    ///   synchronized yourself.
    ///
    /// [buffer-image granularity]: super#buffer-image-granularity
    /// [`Linear`]: AllocationType::Linear
    /// [`NonLinear`]: AllocationType::NonLinear
    #[inline]
    pub unsafe fn set_allocation_type(&mut self, new_type: AllocationType) {
        self.allocation_type = new_type;
    }
}

impl Drop for MemoryAlloc {
    #[inline]
    fn drop(&mut self) {
        match &self.parent {
            AllocParent::FreeList { allocator, id } => {
                unsafe { allocator.free(*id) };
            }
            AllocParent::Buddy {
                allocator,
                order,
                offset,
            } => {
                unsafe { allocator.free(*order, *offset) };
            }
            AllocParent::Pool { allocator, index } => {
                unsafe { allocator.free(*index) };
            }
            // The bump allocator can't free individually, but we need to keep a reference to it so
            // it don't get reset or dropped while in use.
            AllocParent::Bump(_) => {}
            // A root allocation frees itself once all references to the `DeviceMemory` are dropped.
            AllocParent::Root(_) => {}
            // Dedicated allocations free themselves when the `DeviceMemory` is dropped.
            AllocParent::Dedicated(_) => {}
        }
    }
}

unsafe impl DeviceOwned for MemoryAlloc {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.device_memory().device()
    }
}

/// Suballocators are used to divide a *region* into smaller *suballocations*.
///
/// # Regions
///
/// As the name implies, a region is a contiguous portion of memory. It may be the whole dedicated
/// block of [`DeviceMemory`], or only a part of it. Regions are just [allocations] like any other,
/// but we use this term to refer specifically to an allocation that is to be suballocated. Every
/// suballocator is created with a region to work with.
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
/// Vulkano offers the ability to create *memory hierarchies*. We refer to the [`DeviceMemory`] as
/// the root of any such hierarchy, even though technically the driver has levels that are further
/// up, because those `DeviceMemory` blocks need to be allocated from physical memory [pages]
/// themselves, but since those levels are not accessible to us we don't need to consider them. You
/// can create any number of levels/branches from there, bounded only by the amount of available
/// memory within a `DeviceMemory` block. You can suballocate the root into regions, which are then
/// suballocated into further regions and so on, creating hierarchies of arbitrary height.
///
/// As an added bonus, memory hierarchies lend themselves perfectly to the concept of composability
/// we all love so much, making them a natural fit for Rust. For one, a region can be allocated any
/// way, and fed into any suballocator. Also, once you are done with a branch of a hierarchy,
/// meaning there are no more suballocations in use within the region of that branch, and you would
/// like to reuse the region, you can do so safely! All suballocators have a `try_into_region`
/// method for this purpose. This means that you can replace one suballocator with another without
/// consulting any of the higher levels in the hierarchy.
///
/// # Examples
///
/// Allocating a region to suballocatate:
///
/// ```
/// use vulkano::memory::{DeviceMemory, MemoryAllocateInfo, MemoryPropertyFlags, MemoryType};
/// use vulkano::memory::allocator::MemoryAlloc;
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
///
/// // First you need to find a suitable memory type.
/// let memory_type_index = device
///     .physical_device()
///     .memory_properties()
///     .memory_types
///     .iter()
///     .enumerate()
///     // In a real-world scenario, you would probably want to rank the memory types based on your
///     // requirements, instead of picking the first one that satisfies them. Also, you have to
///     // take the requirements of the resources you want to allocate memory for into consideration.
///     .find_map(|(index, MemoryType { property_flags, .. })| {
///         property_flags.intersects(MemoryPropertyFlags::DEVICE_LOCAL).then_some(index)
///     })
///     .unwrap() as u32;
///
/// let region = MemoryAlloc::new(
///     DeviceMemory::allocate(
///         device.clone(),
///         MemoryAllocateInfo {
///             allocation_size: 64 * 1024 * 1024,
///             memory_type_index,
///             ..Default::default()
///         },
///     )
///     .unwrap(),
/// )
/// .unwrap();
///
/// // You can now feed `region` into any suballocator.
/// ```
///
/// # Implementing the trait
///
/// Please don't.
///
/// [allocations]: MemoryAlloc
/// [pages]: super#pages
pub unsafe trait Suballocator: DeviceOwned {
    /// Whether this allocator needs to block or not.
    ///
    /// This is used by the [`GenericMemoryAllocator`] to specialize the allocation strategy to the
    /// suballocator at compile time.
    ///
    /// [`GenericMemoryAllocator`]: super::GenericMemoryAllocator
    const IS_BLOCKING: bool;

    /// Whether the allocator needs [`cleanup`] to be called before memory can be released.
    ///
    /// This is used by the [`GenericMemoryAllocator`] to specialize the allocation strategy to the
    /// suballocator at compile time.
    ///
    /// [`cleanup`]: Self::cleanup
    /// [`GenericMemoryAllocator`]: super::GenericMemoryAllocator
    const NEEDS_CLEANUP: bool;

    /// Creates a new suballocator for the given [region].
    ///
    /// [region]: Self#regions
    fn new(region: MemoryAlloc) -> Self
    where
        Self: Sized;

    /// Creates a new suballocation within the [region].
    ///
    /// [region]: Self#regions
    fn allocate(
        &self,
        create_info: SuballocationCreateInfo,
    ) -> Result<MemoryAlloc, SuballocatorError>;

    /// Returns a reference to the underlying [region].
    ///
    /// [region]: Self#regions
    fn region(&self) -> &MemoryAlloc;

    /// Returns the underlying [region] if there are no other strong references to the allocator,
    /// otherwise hands you back the allocator wrapped in [`Err`]. Allocations made with the
    /// allocator count as references for as long as they are alive.
    ///
    /// [region]: Self#regions
    fn try_into_region(self) -> Result<MemoryAlloc, Self>
    where
        Self: Sized;

    /// Returns the total amount of free space that is left in the [region].
    ///
    /// [region]: Self#regions
    fn free_size(&self) -> DeviceSize;

    /// Tries to free some space, if applicable.
    fn cleanup(&mut self);
}

/// Parameters to create a new [allocation] using a [suballocator].
///
/// [allocation]: MemoryAlloc
/// [suballocator]: Suballocator
#[derive(Clone, Debug)]
pub struct SuballocationCreateInfo {
    /// Memory layout required for the allocation.
    ///
    /// The default value is a layout with size [`DeviceLayout::MAX_SIZE`] and alignment
    /// [`DeviceAlignment::MIN`], which must be overridden.
    pub layout: DeviceLayout,

    /// Type of resources that can be bound to the allocation.
    ///
    /// The default value is [`AllocationType::Unknown`].
    pub allocation_type: AllocationType,

    pub _ne: crate::NonExhaustive,
}

impl Default for SuballocationCreateInfo {
    #[inline]
    fn default() -> Self {
        SuballocationCreateInfo {
            layout: DeviceLayout::new(
                NonZeroDeviceSize::new(DeviceLayout::MAX_SIZE).unwrap(),
                DeviceAlignment::MIN,
            )
            .unwrap(),
            allocation_type: AllocationType::Unknown,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Tells the [suballocator] what type of resource will be bound to the allocation, so that it can
/// optimize memory usage while still respecting the [buffer-image granularity].
///
/// [suballocator]: Suballocator
/// [buffer-image granularity]: super#buffer-image-granularity
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AllocationType {
    /// The type of resource is unknown, it might be either linear or non-linear. What this means is
    /// that allocations created with this type must always be aligned to the buffer-image
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

/// Error that can be returned when using a [suballocator].
///
/// [suballocator]: Suballocator
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SuballocatorError {
    /// There is no more space available in the region.
    OutOfRegionMemory,

    /// The region has enough free space to satisfy the request but is too fragmented.
    FragmentedRegion,

    /// The allocation was larger than the allocator's block size, meaning that this error would
    /// arise with the parameters no matter the state the allocator was in.
    ///
    /// This can be used to let the [`GenericMemoryAllocator`] know that allocating a new block of
    /// [`DeviceMemory`] and trying to suballocate it with the same parameters would not solve the
    /// issue.
    ///
    /// [`GenericMemoryAllocator`]: super::GenericMemoryAllocator
    BlockSizeExceeded,
}

impl Error for SuballocatorError {}

impl Display for SuballocatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::OutOfRegionMemory => "out of region memory",
                Self::FragmentedRegion => "the region is too fragmented",
                Self::BlockSizeExceeded =>
                    "the allocation size was greater than the suballocator's block size",
            }
        )
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
/// size, consider the [`PoolAllocator`]. Lastly, if you need to allocate very often, then
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
/// The allocator is synchronized internally with a lock, which is held only for a very short
/// period each time an allocation is created and freed. The free-list is sorted by size, which
/// means that when allocating, finding a best-fit is always possible in *O*(log(*n*)) time in the
/// worst case. When freeing, the coalescing requires us to remove the adjacent free suballocations
/// from the free-list which is *O*(log(*n*)), and insert the possibly coalesced suballocation into
/// the free-list which has the same time complexity, so in total freeing is *O*(log(*n*)).
///
/// There is one notable edge-case: after the allocator finds a best-fit, it is possible that it
/// needs to align the suballocation's offset to a higher value, after which the requested size
/// might no longer fit. In such a case, the next free suballocation in sorted order is tried until
/// a fit is successful. If this issue is encountered with all candidates, then the time complexity
/// would be *O*(*n*). However, this scenario is extremely unlikely which is why we are not
/// considering it in the above analysis. Additionally, if your free-list is filled with
/// allocations that all have the same size then that seems pretty sus. Sounds like you're in dire
/// need of a `PoolAllocator`.
///
/// # Examples
///
/// Most commonly you will not want to use this suballocator directly but rather use it within
/// [`GenericMemoryAllocator`], having one global [`StandardMemoryAllocator`] for most if not all
/// of your allocation needs.
///
/// Basic usage as a global allocator for long-lived resources:
///
/// ```
/// use vulkano::format::Format;
/// use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
/// use vulkano::memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator};
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
///
/// let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
///
/// # fn read_textures() -> Vec<Vec<u8>> { Vec::new() }
/// // Allocate some resources.
/// let textures_data: Vec<Vec<u8>> = read_textures();
/// let textures = textures_data.into_iter().map(|data| {
///     let image = Image::new(
///         &memory_allocator,
///         ImageCreateInfo {
///             image_type: ImageType::Dim2d,
///             format: Some(Format::R8G8B8A8_UNORM),
///             extent: [1024, 1024, 1],
///             usage: ImageUsage::SAMPLED,
///             ..Default::default()
///         },
///         AllocationCreateInfo::default(),
///     )
///     .unwrap();
///
///     // ...upload data...
///
///     image
/// });
/// ```
///
/// For use in allocating arenas for [`SubbufferAllocator`]:
///
/// ```
/// use std::sync::Arc;
/// use vulkano::buffer::allocator::SubbufferAllocator;
/// use vulkano::memory::allocator::StandardMemoryAllocator;
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
///
/// // We need to wrap the allocator in an `Arc` so that we can share ownership of it.
/// let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
/// let buffer_allocator = SubbufferAllocator::new(memory_allocator.clone(), Default::default());
///
/// // You can continue using `memory_allocator` for other things.
/// ```
///
/// Sometimes, it is neccessary to suballocate an allocation. If you don't want to allocate new
/// [`DeviceMemory`] blocks to suballocate, perhaps because of concerns of memory wastage or
/// allocation efficiency, you can use your existing global `StandardMemoryAllocator` to allocate
/// regions for your suballocation needs:
///
/// ```
/// use vulkano::memory::allocator::{
///     DeviceLayout, MemoryAllocator, StandardMemoryAllocator, SuballocationCreateInfo,
/// };
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// let memory_allocator = StandardMemoryAllocator::new_default(device.clone());
///
/// # let memory_type_index = 0;
/// let region = memory_allocator.allocate_from_type(
///     // When choosing the index, you have to make sure that the memory type is allowed for the
///     // type of resource that you want to bind the suballocations to.
///     memory_type_index,
///     SuballocationCreateInfo {
///         layout: DeviceLayout::from_size_alignment(
///             // This will be the size of your region.
///             16 * 1024 * 1024,
///             // It generally does not matter what the alignment is, because you're going to
///             // suballocate the allocation anyway, and not bind it directly.
///             1,
///         )
///         .unwrap(),
///         ..Default::default()
///     },
/// )
/// .unwrap();
///
/// // You can now feed the `region` into any suballocator.
/// ```
///
/// [suballocator]: Suballocator
/// [free-list]: Suballocator#free-lists
/// [external fragmentation]: super#external-fragmentation
/// [the `Suballocator` implementation]: Suballocator#impl-Suballocator-for-Arc<FreeListAllocator>
/// [internal fragmentation]: super#internal-fragmentation
/// [alignment requirements]: super#alignment
/// [`GenericMemoryAllocator`]: super::GenericMemoryAllocator
/// [`StandardMemoryAllocator`]: super::StandardMemoryAllocator
/// [`SubbufferAllocator`]: crate::buffer::allocator::SubbufferAllocator
#[derive(Debug)]
pub struct FreeListAllocator {
    region: MemoryAlloc,
    device_memory: Arc<DeviceMemory>,
    buffer_image_granularity: DeviceAlignment,
    atom_size: DeviceAlignment,
    // Total memory remaining in the region.
    free_size: AtomicU64,
    state: Mutex<FreeListAllocatorState>,
}

impl FreeListAllocator {
    /// Creates a new `FreeListAllocator` for the given [region].
    ///
    /// # Panics
    ///
    /// - Panics if `region.allocation_type` is not [`AllocationType::Unknown`]. This is done to
    ///   avoid checking for a special case of [buffer-image granularity] conflict.
    /// - Panics if `region` is a [dedicated allocation].
    ///
    /// [region]: Suballocator#regions
    /// [buffer-image granularity]: super#buffer-image-granularity
    /// [dedicated allocation]: MemoryAlloc::is_dedicated
    pub fn new(region: MemoryAlloc) -> Arc<Self> {
        // NOTE(Marc): This number was pulled straight out of my a-
        const AVERAGE_ALLOCATION_SIZE: DeviceSize = 64 * 1024;

        assert!(region.allocation_type == AllocationType::Unknown);

        let device_memory = region
            .root()
            .expect("dedicated allocations can't be suballocated")
            .clone();
        let buffer_image_granularity = device_memory
            .device()
            .physical_device()
            .properties()
            .buffer_image_granularity;

        let atom_size = region.atom_size.unwrap_or(DeviceAlignment::MIN);
        let free_size = AtomicU64::new(region.size);

        let capacity = (region.size / AVERAGE_ALLOCATION_SIZE) as usize;
        let mut nodes = host::PoolAllocator::new(capacity + 64);
        let mut free_list = Vec::with_capacity(capacity / 16 + 16);
        let root_id = nodes.allocate(SuballocationListNode {
            prev: None,
            next: None,
            offset: region.offset,
            size: region.size,
            ty: SuballocationType::Free,
        });
        free_list.push(root_id);
        let state = Mutex::new(FreeListAllocatorState { nodes, free_list });

        Arc::new(FreeListAllocator {
            region,
            device_memory,
            buffer_image_granularity,
            atom_size,
            free_size,
            state,
        })
    }

    /// # Safety
    ///
    /// - `node_id` must refer to an occupied suballocation allocated by `self`.
    unsafe fn free(&self, node_id: SlotId) {
        let mut state = self.state.lock();
        let node = state.nodes.get_mut(node_id);

        debug_assert!(node.ty != SuballocationType::Free);

        // Suballocation sizes are constrained by the size of the region, so they can't possibly
        // overflow when added up.
        self.free_size.fetch_add(node.size, Ordering::Release);

        node.ty = SuballocationType::Free;
        state.coalesce(node_id);
        state.free(node_id);
    }
}

unsafe impl Suballocator for Arc<FreeListAllocator> {
    const IS_BLOCKING: bool = true;

    const NEEDS_CLEANUP: bool = false;

    #[inline]
    fn new(region: MemoryAlloc) -> Self {
        FreeListAllocator::new(region)
    }

    /// Creates a new suballocation within the [region].
    ///
    /// # Errors
    ///
    /// - Returns [`OutOfRegionMemory`] if there are no free suballocations large enough so satisfy
    ///   the request.
    /// - Returns [`FragmentedRegion`] if a suballocation large enough to satisfy the request could
    ///   have been formed, but wasn't because of [external fragmentation].
    ///
    /// [region]: Suballocator#regions
    /// [`allocate`]: Suballocator::allocate
    /// [`OutOfRegionMemory`]: SuballocatorError::OutOfRegionMemory
    /// [`FragmentedRegion`]: SuballocatorError::FragmentedRegion
    /// [external fragmentation]: super#external-fragmentation
    #[inline]
    fn allocate(
        &self,
        create_info: SuballocationCreateInfo,
    ) -> Result<MemoryAlloc, SuballocatorError> {
        fn has_granularity_conflict(prev_ty: SuballocationType, ty: AllocationType) -> bool {
            if prev_ty == SuballocationType::Free {
                false
            } else if prev_ty == SuballocationType::Unknown {
                true
            } else {
                prev_ty != ty.into()
            }
        }

        let SuballocationCreateInfo {
            layout,
            allocation_type,
            _ne: _,
        } = create_info;

        let size = layout.size();
        let alignment = cmp::max(layout.alignment(), self.atom_size);
        let mut state = self.state.lock();

        unsafe {
            match state.free_list.last() {
                Some(&last) if state.nodes.get(last).size >= size => {
                    let index = match state
                        .free_list
                        .binary_search_by_key(&size, |&id| state.nodes.get(id).size)
                    {
                        // Exact fit.
                        Ok(index) => index,
                        // Next-best fit. Note that `index == free_list.len()` can't be because we
                        // checked that the free-list contains a suballocation that is big enough.
                        Err(index) => index,
                    };

                    for (index, &id) in state.free_list.iter().enumerate().skip(index) {
                        let suballoc = state.nodes.get(id);

                        // This can't overflow because suballocation offsets are constrained by
                        // the size of the root allocation, which can itself not exceed
                        // `DeviceLayout::MAX_SIZE`.
                        let mut offset = align_up(suballoc.offset, alignment);

                        if let Some(prev_id) = suballoc.prev {
                            let prev = state.nodes.get(prev_id);

                            if are_blocks_on_same_page(
                                prev.offset,
                                prev.size,
                                offset,
                                self.buffer_image_granularity,
                            ) && has_granularity_conflict(prev.ty, allocation_type)
                            {
                                // This is overflow-safe for the same reason as above.
                                offset = align_up(offset, self.buffer_image_granularity);
                            }
                        }

                        if offset + size <= suballoc.offset + suballoc.size {
                            state.free_list.remove(index);

                            // SAFETY:
                            // - `suballoc` is free.
                            // - `offset` is that of `suballoc`, possibly rounded up.
                            // - We checked that `offset + size` falls within `suballoc`.
                            state.split(id, offset, size);
                            state.nodes.get_mut(id).ty = allocation_type.into();

                            // This can't overflow because suballocation sizes in the free-list are
                            // constrained by the remaining size of the region.
                            self.free_size.fetch_sub(size, Ordering::Release);

                            let mapped_ptr = self.region.mapped_ptr.map(|ptr| {
                                // This can't overflow because offsets in the free-list are confined
                                // to the range [region.offset, region.offset + region.size).
                                let relative_offset = offset - self.region.offset;

                                // SAFETY: Allocation sizes are guaranteed to not exceed
                                // `isize::MAX` when they have a mapped pointer, and the original
                                // pointer was handed to us from the Vulkan implementation,
                                // so the offset better be in range.
                                let ptr = ptr.as_ptr().offset(relative_offset as isize);

                                // SAFETY: Same as the previous.
                                NonNull::new_unchecked(ptr)
                            });

                            return Ok(MemoryAlloc {
                                offset,
                                size,
                                allocation_type,
                                mapped_ptr,
                                atom_size: self.region.atom_size,
                                parent: AllocParent::FreeList {
                                    allocator: self.clone(),
                                    id,
                                },
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
    }

    #[inline]
    fn region(&self) -> &MemoryAlloc {
        &self.region
    }

    #[inline]
    fn try_into_region(self) -> Result<MemoryAlloc, Self> {
        Arc::try_unwrap(self).map(|allocator| allocator.region)
    }

    #[inline]
    fn free_size(&self) -> DeviceSize {
        self.free_size.load(Ordering::Acquire)
    }

    #[inline]
    fn cleanup(&mut self) {}
}

unsafe impl DeviceOwned for FreeListAllocator {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.device_memory.device()
    }
}

#[derive(Debug)]
struct FreeListAllocatorState {
    nodes: host::PoolAllocator<SuballocationListNode>,
    // Free suballocations sorted by size in ascending order. This means we can always find a
    // best-fit in *O*(log(*n*)) time in the worst case, and iterating in order is very efficient.
    free_list: Vec<SlotId>,
}

#[derive(Clone, Copy, Debug)]
struct SuballocationListNode {
    prev: Option<SlotId>,
    next: Option<SlotId>,
    offset: DeviceSize,
    size: DeviceSize,
    ty: SuballocationType,
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
    /// - `node_id` must have been allocated by `self`.
    /// - `node_id` must be in the free-list.
    unsafe fn allocate(&mut self, node_id: SlotId) {
        debug_assert!(self.free_list.contains(&node_id));

        let node = self.nodes.get(node_id);

        match self
            .free_list
            .binary_search_by_key(&node.size, |&id| self.nodes.get(id).size)
        {
            Ok(index) => {
                // If there are multiple free suballocations with the same size, the search might
                // have returned any one, so we need to find the one corresponding to the target ID.
                if self.free_list[index] == node_id {
                    self.free_list.remove(index);
                    return;
                }

                // Check all previous indices that point to suballocations with the same size.
                {
                    let mut index = index;
                    loop {
                        index = index.wrapping_sub(1);
                        if let Some(&id) = self.free_list.get(index) {
                            if id == node_id {
                                self.free_list.remove(index);
                                return;
                            }
                            if self.nodes.get(id).size != node.size {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                }

                // Check all next indices that point to suballocations with the same size.
                {
                    let mut index = index;
                    loop {
                        index += 1;
                        if let Some(&id) = self.free_list.get(index) {
                            if id == node_id {
                                self.free_list.remove(index);
                                return;
                            }
                            if self.nodes.get(id).size != node.size {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                }

                unreachable!();
            }
            Err(_) => unreachable!(),
        }
    }

    /// Fits a suballocation inside the target one, splitting the target at the ends if required.
    ///
    /// # Safety
    ///
    /// - `node_id` must have been allocated by `self`.
    /// - `node_id` must refer to a free suballocation.
    /// - `offset` and `size` must refer to a subregion of the given suballocation.
    unsafe fn split(&mut self, node_id: SlotId, offset: DeviceSize, size: DeviceSize) {
        let node = self.nodes.get(node_id);

        debug_assert!(node.ty == SuballocationType::Free);
        debug_assert!(offset >= node.offset);
        debug_assert!(offset + size <= node.offset + node.size);

        // These are guaranteed to not overflow because the caller must uphold that the given
        // region is contained within that of `node`.
        let padding_front = offset - node.offset;
        let padding_back = node.offset + node.size - offset - size;

        if padding_front > 0 {
            let padding = SuballocationListNode {
                prev: node.prev,
                next: Some(node_id),
                offset: node.offset,
                size: padding_front,
                ty: SuballocationType::Free,
            };
            let padding_id = self.nodes.allocate(padding);

            if let Some(prev_id) = padding.prev {
                self.nodes.get_mut(prev_id).next = Some(padding_id);
            }

            let node = self.nodes.get_mut(node_id);
            node.prev = Some(padding_id);
            node.offset = offset;
            // The caller must uphold that the given region is contained within that of `node`, and
            // it follows that if there is padding, the size of the node must be larger than that
            // of the padding, so this can't overflow.
            node.size -= padding.size;

            self.free(padding_id);
        }

        if padding_back > 0 {
            let padding = SuballocationListNode {
                prev: Some(node_id),
                next: node.next,
                offset: offset + size,
                size: padding_back,
                ty: SuballocationType::Free,
            };
            let padding_id = self.nodes.allocate(padding);

            if let Some(next_id) = padding.next {
                self.nodes.get_mut(next_id).prev = Some(padding_id);
            }

            let node = self.nodes.get_mut(node_id);
            node.next = Some(padding_id);
            // This is overflow-safe for the same reason as above.
            node.size -= padding.size;

            self.free(padding_id);
        }
    }

    /// Inserts the target suballocation into the free-list.
    ///
    /// # Safety
    ///
    /// - `node_id` must have been allocated by `self`.
    /// - The free-list must not contain the given suballocation already, as that would constitude
    ///   a double-free.
    unsafe fn free(&mut self, node_id: SlotId) {
        debug_assert!(!self.free_list.contains(&node_id));

        let node = self.nodes.get(node_id);
        let (Ok(index) | Err(index)) = self
            .free_list
            .binary_search_by_key(&node.size, |&id| self.nodes.get(id).size);
        self.free_list.insert(index, node_id);
    }

    /// Coalesces the target (free) suballocation with adjacent ones that are also free.
    ///
    /// # Safety
    ///
    /// - `node_id` must have been allocated by `self`.
    /// - `node_id` must refer to a free suballocation.
    unsafe fn coalesce(&mut self, node_id: SlotId) {
        let node = self.nodes.get(node_id);

        debug_assert!(node.ty == SuballocationType::Free);

        if let Some(prev_id) = node.prev {
            let prev = self.nodes.get(prev_id);

            if prev.ty == SuballocationType::Free {
                // SAFETY: We checked that the suballocation is free.
                self.allocate(prev_id);

                let node = self.nodes.get_mut(node_id);
                node.prev = prev.prev;
                node.offset = prev.offset;
                // The sizes of suballocations are constrained by that of the parent allocation, so
                // they can't possibly overflow when added up.
                node.size += prev.size;

                if let Some(prev_id) = node.prev {
                    self.nodes.get_mut(prev_id).next = Some(node_id);
                }

                // SAFETY:
                // - The suballocation is free.
                // - The suballocation was removed from the free-list.
                // - The next suballocation and possibly a previous suballocation have been updated
                //   such that they no longer reference the suballocation.
                // All of these conditions combined guarantee that `prev_id` can not be used again.
                self.nodes.free(prev_id);
            }
        }

        if let Some(next_id) = node.next {
            let next = self.nodes.get(next_id);

            if next.ty == SuballocationType::Free {
                // SAFETY: Same as above.
                self.allocate(next_id);

                let node = self.nodes.get_mut(node_id);
                node.next = next.next;
                // This is overflow-safe for the same reason as above.
                node.size += next.size;

                if let Some(next_id) = node.next {
                    self.nodes.get_mut(next_id).prev = Some(node_id);
                }

                // SAFETY: Same as above.
                self.nodes.free(next_id);
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
/// then the [`PoolAllocator`] would be a better choice and would eliminate external fragmentation
/// completely.
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
/// would be when you need to allocate regions for other allocators, such as the `PoolAllocator` or
/// the [`BumpAllocator`].
///
/// # Efficiency
///
/// The allocator is synchronized internally with a lock, which is held only for a very short
/// period each time an allocation is created and freed. The time complexity of both allocation and
/// freeing is *O*(*m*) in the worst case where *m* is the highest order, which equates to *O*(log
/// (*n*)) where *n* is the size of the region.
///
/// # Examples
///
/// Basic usage together with [`GenericMemoryAllocator`], to allocate resources that have a
/// moderately low life span (for example if you have a lot of images, each of which needs to be
/// resized every now and then):
///
/// ```
/// use std::sync::Arc;
/// use vulkano::memory::allocator::{
///     BuddyAllocator, GenericMemoryAllocator, GenericMemoryAllocatorCreateInfo,
/// };
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// let memory_allocator = GenericMemoryAllocator::<Arc<BuddyAllocator>>::new(
///     device.clone(),
///     GenericMemoryAllocatorCreateInfo {
///         // Your block sizes must be powers of two, because `BuddyAllocator` only accepts
///         // power-of-two-sized regions.
///         block_sizes: &[(0, 64 * 1024 * 1024)],
///         ..Default::default()
///     },
/// )
/// .unwrap();
///
/// // Now you can use `memory_allocator` to allocate whatever it is you need.
/// ```
///
/// [suballocator]: Suballocator
/// [internal fragmentation]: super#internal-fragmentation
/// [external fragmentation]: super#external-fragmentation
/// [the `Suballocator` implementation]: Suballocator#impl-Suballocator-for-Arc<BuddyAllocator>
/// [region]: Suballocator#regions
/// [`GenericMemoryAllocator`]: super::GenericMemoryAllocator
#[derive(Debug)]
pub struct BuddyAllocator {
    region: MemoryAlloc,
    device_memory: Arc<DeviceMemory>,
    buffer_image_granularity: DeviceAlignment,
    atom_size: DeviceAlignment,
    // Total memory remaining in the region.
    free_size: AtomicU64,
    state: Mutex<BuddyAllocatorState>,
}

impl BuddyAllocator {
    const MIN_NODE_SIZE: DeviceSize = 16;

    /// Arbitrary maximum number of orders, used to avoid a 2D `Vec`. Together with a minimum node
    /// size of 16, this is enough for a 64GiB region.
    const MAX_ORDERS: usize = 32;

    /// Creates a new `BuddyAllocator` for the given [region].
    ///
    /// # Panics
    ///
    /// - Panics if `region.allocation_type` is not [`AllocationType::Unknown`]. This is done to
    ///   avoid checking for a special case of [buffer-image granularity] conflict.
    /// - Panics if `region.size` is not a power of two.
    /// - Panics if `region.size` is not in the range \[16B,&nbsp;64GiB\].
    /// - Panics if `region` is a [dedicated allocation].
    ///
    /// [region]: Suballocator#regions
    /// [buffer-image granularity]: super#buffer-image-granularity
    /// [dedicated allocation]: MemoryAlloc::is_dedicated
    #[inline]
    pub fn new(region: MemoryAlloc) -> Arc<Self> {
        const EMPTY_FREE_LIST: Vec<DeviceSize> = Vec::new();

        assert!(region.allocation_type == AllocationType::Unknown);
        assert!(region.size.is_power_of_two());
        assert!(region.size >= BuddyAllocator::MIN_NODE_SIZE);

        let max_order = (region.size / BuddyAllocator::MIN_NODE_SIZE).trailing_zeros() as usize;

        assert!(max_order < BuddyAllocator::MAX_ORDERS);

        let device_memory = region
            .root()
            .expect("dedicated allocations can't be suballocated")
            .clone();
        let buffer_image_granularity = device_memory
            .device()
            .physical_device()
            .properties()
            .buffer_image_granularity;
        let atom_size = region.atom_size.unwrap_or(DeviceAlignment::MIN);
        let free_size = AtomicU64::new(region.size);

        let mut free_list =
            ArrayVec::new(max_order + 1, [EMPTY_FREE_LIST; BuddyAllocator::MAX_ORDERS]);
        // The root node has the lowest offset and highest order, so it's the whole region.
        free_list[max_order].push(region.offset);
        let state = Mutex::new(BuddyAllocatorState { free_list });

        Arc::new(BuddyAllocator {
            region,
            device_memory,
            buffer_image_granularity,
            atom_size,
            free_size,
            state,
        })
    }

    /// # Safety
    ///
    /// - `order` and `offset` must refer to an occupied suballocation allocated by `self`.
    unsafe fn free(&self, order: usize, mut offset: DeviceSize) {
        let min_order = order;
        let mut state = self.state.lock();

        debug_assert!(!state.free_list[order].contains(&offset));

        // Try to coalesce nodes while incrementing the order.
        for (order, free_list) in state.free_list.iter_mut().enumerate().skip(min_order) {
            // This can't discard any bits because `order` is confined to the range
            // [0, log(region.size / BuddyAllocator::MIN_NODE_SIZE)].
            let size = BuddyAllocator::MIN_NODE_SIZE << order;

            // This can't overflow because the offsets in the free-list are confined to the range
            // [region.offset, region.offset + region.size).
            let buddy_offset = ((offset - self.region.offset) ^ size) + self.region.offset;

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
                    self.free_size.fetch_add(size, Ordering::Release);

                    break;
                }
            }
        }
    }
}

unsafe impl Suballocator for Arc<BuddyAllocator> {
    const IS_BLOCKING: bool = true;

    const NEEDS_CLEANUP: bool = false;

    #[inline]
    fn new(region: MemoryAlloc) -> Self {
        BuddyAllocator::new(region)
    }

    /// Creates a new suballocation within the [region].
    ///
    /// # Errors
    ///
    /// - Returns [`OutOfRegionMemory`] if there are no free nodes large enough so satisfy the
    ///   request.
    /// - Returns [`FragmentedRegion`] if a node large enough to satisfy the request could have
    ///   been formed, but wasn't because of [external fragmentation].
    ///
    /// [region]: Suballocator#regions
    /// [`allocate`]: Suballocator::allocate
    /// [`OutOfRegionMemory`]: SuballocatorError::OutOfRegionMemory
    /// [`FragmentedRegion`]: SuballocatorError::FragmentedRegion
    /// [external fragmentation]: super#external-fragmentation
    #[inline]
    fn allocate(
        &self,
        create_info: SuballocationCreateInfo,
    ) -> Result<MemoryAlloc, SuballocatorError> {
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

        let SuballocationCreateInfo {
            layout,
            allocation_type,
            _ne: _,
        } = create_info;

        let mut size = layout.size();
        let mut alignment = cmp::max(layout.alignment(), self.atom_size);

        if allocation_type == AllocationType::Unknown
            || allocation_type == AllocationType::NonLinear
        {
            // This can't overflow because `DeviceLayout` guarantees that `size` doesn't exceed
            // `DeviceLayout::MAX_SIZE`.
            size = align_up(size, self.buffer_image_granularity);
            alignment = cmp::max(alignment, self.buffer_image_granularity);
        }

        // `DeviceLayout` guarantees that its size does not exceed `DeviceLayout::MAX_SIZE`,
        // which means it can't overflow when rounded up to the next power of two.
        let size = cmp::max(size, BuddyAllocator::MIN_NODE_SIZE).next_power_of_two();

        let min_order = (size / BuddyAllocator::MIN_NODE_SIZE).trailing_zeros() as usize;
        let mut state = self.state.lock();

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

                        // This can't overflow because offsets are confined to the size of the root
                        // allocation, which can itself not exceed `DeviceLayout::MAX_SIZE`.
                        let right_child = offset + size;

                        // Insert the right child in sorted order.
                        let (Ok(index) | Err(index)) = free_list.binary_search(&right_child);
                        free_list.insert(index, right_child);

                        // Repeat splitting for the left child if required in the next loop turn.
                    }

                    // This can't overflow because suballocation sizes in the free-list are
                    // constrained by the remaining size of the region.
                    self.free_size.fetch_sub(size, Ordering::Release);

                    let mapped_ptr = self.region.mapped_ptr.map(|ptr| {
                        // This can't overflow because offsets in the free-list are confined to the
                        // range [region.offset, region.offset + region.size).
                        let relative_offset = offset - self.region.offset;

                        // SAFETY: Allocation sizes are guaranteed to not exceed `isize::MAX` when
                        // they have a mapped pointer, and the original pointer was handed to us
                        // from the Vulkan implementation, so the offset better be in range.
                        let ptr = unsafe { ptr.as_ptr().offset(relative_offset as isize) };

                        // SAFETY: Same as the previous.
                        unsafe { NonNull::new_unchecked(ptr) }
                    });

                    return Ok(MemoryAlloc {
                        offset,
                        size: layout.size(),
                        allocation_type,
                        mapped_ptr,
                        atom_size: self.region.atom_size,
                        parent: AllocParent::Buddy {
                            allocator: self.clone(),
                            order: min_order,
                            offset, // The offset in the alloc itself can change.
                        },
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
    fn region(&self) -> &MemoryAlloc {
        &self.region
    }

    #[inline]
    fn try_into_region(self) -> Result<MemoryAlloc, Self> {
        Arc::try_unwrap(self).map(|allocator| allocator.region)
    }

    /// Returns the total amount of free space left in the [region] that is available to the
    /// allocator, which means that [internal fragmentation] is excluded.
    ///
    /// [region]: Suballocator#regions
    /// [internal fragmentation]: super#internal-fragmentation
    #[inline]
    fn free_size(&self) -> DeviceSize {
        self.free_size.load(Ordering::Acquire)
    }

    #[inline]
    fn cleanup(&mut self) {}
}

unsafe impl DeviceOwned for BuddyAllocator {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.device_memory.device()
    }
}

#[derive(Debug)]
struct BuddyAllocatorState {
    // Every order has its own free-list for convenience, so that we don't have to traverse a tree.
    // Each free-list is sorted by offset because we want to find the first-fit as this strategy
    // minimizes external fragmentation.
    free_list: ArrayVec<Vec<DeviceSize>, { BuddyAllocator::MAX_ORDERS }>,
}

/// A [suballocator] using a pool of fixed-size blocks as a [free-list].
///
/// Since the size of the blocks is fixed, you can not create allocations bigger than that. You can
/// create smaller ones, though, which leads to more and more [internal fragmentation] the smaller
/// the allocations get. This is generally a good trade-off, as internal fragmentation is nowhere
/// near as hard to deal with as [external fragmentation].
///
/// See also [the `Suballocator` implementation].
///
/// # Algorithm
///
/// The free-list contains indices of blocks in the region that are available, so allocation
/// consists merely of popping an index from the free-list. The same goes for freeing, all that is
/// required is to push the index of the block into the free-list. Note that this is only possible
/// because the blocks have a fixed size. Due to this one fact, the free-list doesn't need to be
/// sorted or traversed. As long as there is a free block, it will do, no matter which block it is.
///
/// Since the `PoolAllocator` doesn't keep a list of suballocations that are currently in use,
/// resolving [buffer-image granularity] conflicts on a case-by-case basis is not possible.
/// Therefore, it is an all or nothing situation:
///
/// - you use the allocator for only one type of allocation, [`Linear`] or [`NonLinear`], or
/// - you allow both but align the blocks to the granularity so that no conflics can happen.
///
/// The way this is done is that every suballocation inherits the allocation type of the region.
/// The latter is done by using a region whose allocation type is [`Unknown`]. You are discouraged
/// from using this type if you can avoid it.
///
/// The block size can end up bigger than specified if the allocator is created with a region whose
/// allocation type is `Unknown`. In that case all blocks are aligned to the buffer-image
/// granularity, which may or may not cause signifficant memory usage increase. Say for example
/// your driver reports a granularity of 4KiB. If you need a block size of 8KiB, you would waste no
/// memory. On the other hand, if you needed a block size of 6KiB, you would be wasting 25% of the
/// memory. In such a scenario you are highly encouraged to use a different allocation type.
///
/// The reverse is also true: with an allocation type other than `Unknown`, not all memory within a
/// block may be usable depending on the requested [suballocation]. For instance, with a block size
/// of 1152B (9 * 128B) and a suballocation with `alignment: 256`, a block at an odd index could
/// not utilize its first 128B, reducing its effective size to 1024B. This is usually only relevant
/// with small block sizes, as [alignment requirements] are usually rather small, but it completely
/// depends on the resource and driver.
///
/// In summary, the block size you choose has a signifficant impact on internal fragmentation due
/// to the two reasons described above. You need to choose your block size carefully, *especially*
/// if you require small allocations. Some rough guidelines:
///
/// - Always [align] your blocks to a sufficiently large power of 2. This does **not** mean your
///   block size must be a power of two. For example with a block size of 3KiB, your blocks would
///   be aligned to 1KiB.
/// - Prefer not using the allocation type `Unknown`. You can always create as many
///   `PoolAllocator`s as you like for different allocation types and sizes, and they can all work
///   within the same memory block. You should be safe from fragmentation if your blocks are
///   aligned to 1KiB.
/// - If you must use the allocation type `Unknown`, then you should be safe from fragmentation on
///   pretty much any driver if your blocks are aligned to 64KiB. Keep in mind that this might
///   change any time as new devices appear or new drivers come out. Always look at the properties
///   of the devices you want to support before relying on any such data.
///
/// # Efficiency
///
/// In theory, a pool allocator is the ideal one because it causes no external fragmentation, and
/// both allocation and freeing is *O*(1). It also never needs to lock and hence also lends itself
/// perfectly to concurrency. But of course, there is the trade-off that block sizes are not
/// dynamic.
///
/// As you can imagine, the `PoolAllocator` is the perfect fit if you know the sizes of the
/// allocations you will be making, and they are more or less in the same size class. But this
/// allocation algorithm really shines when combined with others, as most do. For one, nothing is
/// stopping you from having multiple `PoolAllocator`s for many different size classes. You could
/// consider a pool of pools, by layering `PoolAllocator` with itself, but this would have the
/// downside that the regions of the pools for all size classes would have to match. Usually this
/// is not desired. If you want pools for different size classes to all have about the same number
/// of blocks, or you even know that some size classes require more or less blocks (because of how
/// many resources you will be allocating for each), then you need an allocator that can allocate
/// regions of different sizes. You can use the [`FreeListAllocator`] for this, if external
/// fragmentation is not an issue, otherwise you might consider using the [`BuddyAllocator`]. On
/// the other hand, you might also want to consider having a `PoolAllocator` at the top of a
/// [hierarchy]. Again, this allocator never needs to lock making it *the* perfect fit for a global
/// concurrent allocator, which hands out large regions which can then be suballocated locally on a
/// thread, by the [`BumpAllocator`] for example.
///
/// # Examples
///
/// Basic usage together with [`GenericMemoryAllocator`]:
///
/// ```
/// use std::sync::Arc;
/// use vulkano::memory::allocator::{
///     GenericMemoryAllocator, GenericMemoryAllocatorCreateInfo, PoolAllocator,
/// };
///
/// # let device: std::sync::Arc<vulkano::device::Device> = return;
/// let memory_allocator = GenericMemoryAllocator::<Arc<PoolAllocator<{ 64 * 1024 }>>>::new(
///     device.clone(),
///     GenericMemoryAllocatorCreateInfo {
///         block_sizes: &[(0, 64 * 1024 * 1024)],
///         ..Default::default()
///     },
/// )
/// .unwrap();
///
/// // Now you can use `memory_allocator` to allocate whatever it is you need.
/// ```
///
/// [suballocator]: Suballocator
/// [free-list]: Suballocator#free-lists
/// [internal fragmentation]: super#internal-fragmentation
/// [external fragmentation]: super#external-fragmentation
/// [the `Suballocator` implementation]: Suballocator#impl-Suballocator-for-Arc<PoolAllocator<BLOCK_SIZE>>
/// [region]: Suballocator#regions
/// [buffer-image granularity]: super#buffer-image-granularity
/// [`Linear`]: AllocationType::Linear
/// [`NonLinear`]: AllocationType::NonLinear
/// [`Unknown`]: AllocationType::Unknown
/// [suballocation]: SuballocationCreateInfo
/// [alignment requirements]: super#memory-requirements
/// [align]: super#alignment
/// [hierarchy]: Suballocator#memory-hierarchies
/// [`GenericMemoryAllocator`]: super::GenericMemoryAllocator
#[derive(Debug)]
#[repr(transparent)]
pub struct PoolAllocator<const BLOCK_SIZE: DeviceSize> {
    inner: PoolAllocatorInner,
}

impl<const BLOCK_SIZE: DeviceSize> PoolAllocator<BLOCK_SIZE> {
    /// Creates a new `PoolAllocator` for the given [region].
    ///
    /// # Panics
    ///
    /// - Panics if `region.size < BLOCK_SIZE`.
    /// - Panics if `region` is a [dedicated allocation].
    ///
    /// [region]: Suballocator#regions
    /// [dedicated allocation]: MemoryAlloc::is_dedicated
    #[inline]
    pub fn new(
        region: MemoryAlloc,
        #[cfg(test)] buffer_image_granularity: DeviceAlignment,
    ) -> Arc<Self> {
        Arc::new(PoolAllocator {
            inner: PoolAllocatorInner::new(
                region,
                BLOCK_SIZE,
                #[cfg(test)]
                buffer_image_granularity,
            ),
        })
    }

    /// Size of a block. Can be bigger than `BLOCK_SIZE` due to alignment requirements.
    #[inline]
    pub fn block_size(&self) -> DeviceSize {
        self.inner.block_size
    }

    /// Total number of blocks available to the allocator. This is always equal to
    /// `self.region().size() / self.block_size()`.
    #[inline]
    pub fn block_count(&self) -> usize {
        self.inner.free_list.capacity()
    }

    /// Number of free blocks.
    #[inline]
    pub fn free_count(&self) -> usize {
        self.inner.free_list.len()
    }
}

unsafe impl<const BLOCK_SIZE: DeviceSize> Suballocator for Arc<PoolAllocator<BLOCK_SIZE>> {
    const IS_BLOCKING: bool = false;

    const NEEDS_CLEANUP: bool = false;

    #[inline]
    fn new(region: MemoryAlloc) -> Self {
        PoolAllocator::new(
            region,
            #[cfg(test)]
            DeviceAlignment::MIN,
        )
    }

    /// Creates a new suballocation within the [region].
    ///
    /// > **Note**: `create_info.allocation_type` is silently ignored because all suballocations
    /// > inherit the allocation type from the region.
    ///
    /// # Errors
    ///
    /// - Returns [`OutOfRegionMemory`] if the [free-list] is empty.
    /// - Returns [`OutOfRegionMemory`] if the allocation can't fit inside a block. Only the first
    ///   block in the free-list is tried, which means that if one block isn't usable due to
    ///   [internal fragmentation] but a different one would be, you still get this error. See the
    ///   [type-level documentation] for details on how to properly configure your allocator.
    /// - Returns [`BlockSizeExceeded`] if `create_info.size` exceeds `BLOCK_SIZE`.
    ///
    /// [region]: Suballocator#regions
    /// [`allocate`]: Suballocator::allocate
    /// [`OutOfRegionMemory`]: SuballocatorError::OutOfRegionMemory
    /// [free-list]: Suballocator#free-lists
    /// [internal fragmentation]: super#internal-fragmentation
    /// [type-level documentation]: PoolAllocator
    /// [`BlockSizeExceeded`]: SuballocatorError::BlockSizeExceeded
    #[inline]
    fn allocate(
        &self,
        create_info: SuballocationCreateInfo,
    ) -> Result<MemoryAlloc, SuballocatorError> {
        // SAFETY: `PoolAllocator<BLOCK_SIZE>` and `PoolAllocatorInner` have the same layout.
        //
        // This is not quite optimal, because we are always cloning the `Arc` even if allocation
        // fails, in which case the `Arc` gets cloned and dropped for no reason. Unfortunately,
        // there is currently no way to turn `&Arc<T>` into `&Arc<U>` that is sound.
        unsafe { Arc::from_raw(Arc::into_raw(self.clone()).cast::<PoolAllocatorInner>()) }
            .allocate(create_info)
    }

    #[inline]
    fn region(&self) -> &MemoryAlloc {
        &self.inner.region
    }

    #[inline]
    fn try_into_region(self) -> Result<MemoryAlloc, Self> {
        Arc::try_unwrap(self).map(|allocator| allocator.inner.region)
    }

    #[inline]
    fn free_size(&self) -> DeviceSize {
        self.free_count() as DeviceSize * self.block_size()
    }

    #[inline]
    fn cleanup(&mut self) {}
}

unsafe impl<const BLOCK_SIZE: DeviceSize> DeviceOwned for PoolAllocator<BLOCK_SIZE> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device_memory.device()
    }
}

#[derive(Debug)]
struct PoolAllocatorInner {
    region: MemoryAlloc,
    device_memory: Arc<DeviceMemory>,
    atom_size: DeviceAlignment,
    block_size: DeviceSize,
    // Unsorted list of free block indices.
    free_list: ArrayQueue<DeviceSize>,
}

impl PoolAllocatorInner {
    fn new(
        region: MemoryAlloc,
        mut block_size: DeviceSize,
        #[cfg(test)] buffer_image_granularity: DeviceAlignment,
    ) -> Self {
        let device_memory = region
            .root()
            .expect("dedicated allocations can't be suballocated")
            .clone();
        #[cfg(not(test))]
        let buffer_image_granularity = device_memory
            .device()
            .physical_device()
            .properties()
            .buffer_image_granularity;
        let atom_size = region.atom_size.unwrap_or(DeviceAlignment::MIN);
        if region.allocation_type == AllocationType::Unknown {
            block_size = align_up(block_size, buffer_image_granularity);
        }

        let block_count = region.size / block_size;
        let free_list = ArrayQueue::new(block_count as usize);
        for i in 0..block_count {
            free_list.push(i).unwrap();
        }

        PoolAllocatorInner {
            region,
            device_memory,
            atom_size,
            block_size,
            free_list,
        }
    }

    fn allocate(
        self: Arc<Self>,
        create_info: SuballocationCreateInfo,
    ) -> Result<MemoryAlloc, SuballocatorError> {
        let SuballocationCreateInfo {
            layout,
            allocation_type: _,
            _ne: _,
        } = create_info;

        let size = layout.size();
        let alignment = cmp::max(layout.alignment(), self.atom_size);
        let index = self
            .free_list
            .pop()
            .ok_or(SuballocatorError::OutOfRegionMemory)?;

        // Indices in the free-list are confined to the range [0, region.size / block_size], so
        // this can't overflow.
        let relative_offset = index * self.block_size;
        // This can't overflow because offsets are confined to the size of the root allocation,
        // which can itself not exceed `DeviceLayout::MAX_SIZE`.
        let offset = align_up(self.region.offset + relative_offset, alignment);

        if offset + size > self.region.offset + relative_offset + self.block_size {
            let _ = self.free_list.push(index);

            return if size > self.block_size {
                Err(SuballocatorError::BlockSizeExceeded)
            } else {
                // There is not enough space due to alignment requirements.
                Err(SuballocatorError::OutOfRegionMemory)
            };
        }

        let mapped_ptr = self.region.mapped_ptr.map(|ptr| {
            // SAFETY: Allocation sizes are guaranteed to not exceed `isize::MAX` when they have a
            // mapped pointer, and the original pointer was handed to us from the Vulkan
            // implementation, so the offset better be in range.
            let ptr = unsafe { ptr.as_ptr().offset(relative_offset as isize) };

            // SAFETY: Same as the previous.
            unsafe { NonNull::new_unchecked(ptr) }
        });

        Ok(MemoryAlloc {
            offset,
            size,
            allocation_type: self.region.allocation_type,
            mapped_ptr,
            atom_size: self.region.atom_size,
            parent: AllocParent::Pool {
                allocator: self,
                index,
            },
        })
    }

    /// # Safety
    ///
    /// - `index` must refer to an occupied suballocation allocated by `self`.
    unsafe fn free(&self, index: DeviceSize) {
        let _ = self.free_list.push(index);
    }
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
/// dropped, you can safely reset the allocator using the [`try_reset`] method as long as the
/// allocator is not shared between threads. It is hard to safely reset a bump allocator that is
/// used concurrently. In such a scenario it's best not to reset it at all and instead drop it once
/// it reaches the end of the [region], freeing the region to a higher level in the [hierarchy]
/// once all threads have dropped their reference to the allocator. This is one of the reasons you
/// are generally advised to use one `BumpAllocator` per thread if you can.
///
/// # Efficiency
///
/// Allocation is *O*(1), and so is resetting the allocator (freeing all allocations). Allocation
/// is always lock-free, and most of the time even wait-free. The only case in which it is not
/// wait-free is if a lot of allocations are made concurrently, which results in CPU-level
/// contention. Therefore, if you for example need to allocate a lot of buffers each frame from
/// multiple threads, you might get better performance by using one `BumpAllocator` per thread.
///
/// The reason synchronization can be avoided entirely is that the created allocations can be
/// dropped without needing to talk back to the allocator to free anything. The other allocation
/// algorithms all have a free-list which needs to be modified once an allocation is dropped. Since
/// Vulkano's buffers and images are `Sync`, that means that even if the allocator only allocates
/// from one thread, it can still be used to free from multiple threads.
///
/// [suballocator]: Suballocator
/// [the `Suballocator` implementation]: Suballocator#impl-Suballocator-for-Arc<BumpAllocator>
/// [region]: Suballocator#regions
/// [free-list]: Suballocator#free-lists
/// [memory leaks]: super#leakage
/// [`try_reset`]: Self::try_reset
/// [hierarchy]: Suballocator#memory-hierarchies
#[derive(Debug)]
pub struct BumpAllocator {
    region: MemoryAlloc,
    device_memory: Arc<DeviceMemory>,
    buffer_image_granularity: DeviceAlignment,
    atom_size: DeviceAlignment,
    // Encodes the previous allocation type in the 2 least signifficant bits and the free start in
    // the rest.
    state: AtomicU64,
}

impl BumpAllocator {
    /// Creates a new `BumpAllocator` for the given [region].
    ///
    /// # Panics
    ///
    /// - Panics if `region` is a [dedicated allocation].
    /// - Panics if `region.size` exceeds `DeviceLayout::MAX_SIZE >> 2`.
    ///
    /// [region]: Suballocator#regions
    /// [dedicated allocation]: MemoryAlloc::is_dedicated
    pub fn new(region: MemoryAlloc) -> Arc<Self> {
        // Sanity check: this would lead to UB because of the left-shifting by 2 needed to encode
        // the free-start into the state.
        assert!(region.size <= (DeviceLayout::MAX_SIZE >> 2));

        let device_memory = region
            .root()
            .expect("dedicated allocations can't be suballocated")
            .clone();
        let buffer_image_granularity = device_memory
            .device()
            .physical_device()
            .properties()
            .buffer_image_granularity;
        let atom_size = region.atom_size.unwrap_or(DeviceAlignment::MIN);
        let state = AtomicU64::new(region.allocation_type as DeviceSize);

        Arc::new(BumpAllocator {
            region,
            device_memory,
            buffer_image_granularity,
            atom_size,
            state,
        })
    }

    /// Resets the free-start back to the beginning of the [region] if there are no other strong
    /// references to the allocator.
    ///
    /// [region]: Suballocator#regions
    #[inline]
    pub fn try_reset(self: &mut Arc<Self>) -> Result<(), BumpAllocatorResetError> {
        Arc::get_mut(self)
            .map(|allocator| {
                *allocator.state.get_mut() = allocator.region.allocation_type as DeviceSize;
            })
            .ok_or(BumpAllocatorResetError)
    }

    /// Resets the free-start to the beginning of the [region] without checking if there are other
    /// strong references to the allocator.
    ///
    /// This could be useful if you cloned the [`Arc`] yourself, and can guarantee that no
    /// allocations currently hold a reference to it.
    ///
    /// As a safe alternative, you can let the `Arc` do all the work. Simply drop it once it
    /// reaches the end of the region. After all threads do that, the region will be freed to the
    /// next level up the [hierarchy]. If you only use the allocator on one thread and need shared
    /// ownership, you can use `Rc<RefCell<Arc<BumpAllocator>>>` together with [`try_reset`] for a
    /// safe alternative as well.
    ///
    /// # Safety
    ///
    /// - All allocations made with the allocator must have been dropped.
    ///
    /// [region]: Suballocator#regions
    /// [hierarchy]: Suballocator#memory-hierarchies
    /// [`try_reset`]: Self::try_reset
    #[inline]
    pub unsafe fn reset_unchecked(&self) {
        self.state
            .store(self.region.allocation_type as DeviceSize, Ordering::Release);
    }
}

unsafe impl Suballocator for Arc<BumpAllocator> {
    const IS_BLOCKING: bool = false;

    const NEEDS_CLEANUP: bool = true;

    #[inline]
    fn new(region: MemoryAlloc) -> Self {
        BumpAllocator::new(region)
    }

    /// Creates a new suballocation within the [region].
    ///
    /// # Errors
    ///
    /// - Returns [`OutOfRegionMemory`] if the requested allocation can't fit in the free space
    ///   remaining in the region.
    ///
    /// [region]: Suballocator#regions
    /// [`allocate`]: Suballocator::allocate
    /// [`OutOfRegionMemory`]: SuballocatorError::OutOfRegionMemory
    #[inline]
    fn allocate(
        &self,
        create_info: SuballocationCreateInfo,
    ) -> Result<MemoryAlloc, SuballocatorError> {
        const SPIN_LIMIT: u32 = 6;

        // NOTE(Marc): The following code is a minimal version `Backoff` taken from
        // crossbeam_utils v0.8.11, because we didn't want to add a dependency for a couple lines
        // that are used in one place only.
        /// Original documentation:
        /// https://docs.rs/crossbeam-utils/0.8.11/crossbeam_utils/struct.Backoff.html
        struct Backoff {
            step: Cell<u32>,
        }

        impl Backoff {
            fn new() -> Self {
                Backoff { step: Cell::new(0) }
            }

            fn spin(&self) {
                for _ in 0..1 << self.step.get().min(SPIN_LIMIT) {
                    core::hint::spin_loop();
                }

                if self.step.get() <= SPIN_LIMIT {
                    self.step.set(self.step.get() + 1);
                }
            }
        }

        fn has_granularity_conflict(prev_ty: AllocationType, ty: AllocationType) -> bool {
            prev_ty == AllocationType::Unknown || prev_ty != ty
        }

        let SuballocationCreateInfo {
            layout,
            allocation_type,
            _ne: _,
        } = create_info;

        let size = layout.size();
        let alignment = cmp::max(layout.alignment(), self.atom_size);
        let backoff = Backoff::new();
        let mut state = self.state.load(Ordering::Relaxed);

        loop {
            let free_start = state >> 2;
            let prev_alloc_type = match state & 0b11 {
                0 => AllocationType::Unknown,
                1 => AllocationType::Linear,
                2 => AllocationType::NonLinear,
                _ => unreachable!(),
            };

            // These can't overflow because offsets are constrained by the size of the root
            // allocation, which can itself not exceed `DeviceLayout::MAX_SIZE`.
            let prev_end = self.region.offset + free_start;
            let mut offset = align_up(prev_end, alignment);

            if prev_end > 0
                && are_blocks_on_same_page(0, prev_end, offset, self.buffer_image_granularity)
                && has_granularity_conflict(prev_alloc_type, allocation_type)
            {
                offset = align_up(offset, self.buffer_image_granularity);
            }

            let relative_offset = offset - self.region.offset;

            let free_start = relative_offset + size;

            if free_start > self.region.size {
                return Err(SuballocatorError::OutOfRegionMemory);
            }

            // This can't discard any bits because we checked that `region.size` does not exceed
            // `DeviceLayout::MAX_SIZE >> 2`.
            let new_state = free_start << 2 | allocation_type as DeviceSize;

            match self.state.compare_exchange_weak(
                state,
                new_state,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    let mapped_ptr = self.region.mapped_ptr.map(|ptr| {
                        // SAFETY: Allocation sizes are guaranteed to not exceed `isize::MAX` when
                        // they have a mapped pointer, and the original pointer was handed to us
                        // from the Vulkan implementation, so the offset better be in range.
                        let ptr = unsafe { ptr.as_ptr().offset(relative_offset as isize) };

                        // SAFETY: Same as the previous.
                        unsafe { NonNull::new_unchecked(ptr) }
                    });

                    return Ok(MemoryAlloc {
                        offset,
                        size,
                        allocation_type,
                        mapped_ptr,
                        atom_size: self.region.atom_size,
                        parent: AllocParent::Bump(self.clone()),
                    });
                }
                Err(new_state) => {
                    state = new_state;
                    backoff.spin();
                }
            }
        }
    }

    #[inline]
    fn region(&self) -> &MemoryAlloc {
        &self.region
    }

    #[inline]
    fn try_into_region(self) -> Result<MemoryAlloc, Self> {
        Arc::try_unwrap(self).map(|allocator| allocator.region)
    }

    #[inline]
    fn free_size(&self) -> DeviceSize {
        self.region.size - (self.state.load(Ordering::Acquire) >> 2)
    }

    #[inline]
    fn cleanup(&mut self) {
        let _ = self.try_reset();
    }
}

unsafe impl DeviceOwned for BumpAllocator {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.device_memory.device()
    }
}

/// Error that can be returned when resetting the [`BumpAllocator`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BumpAllocatorResetError;

impl Error for BumpAllocatorResetError {}

impl Display for BumpAllocatorResetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the allocator is still in use")
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

/// Allocators for memory on the host, used to speed up the allocators for the device.
mod host {
    use std::num::NonZeroUsize;

    /// Allocates objects from a pool on the host, which has the following benefits:
    ///
    /// - Allocation is much faster because there is no need to consult the global allocator or even
    ///   worse, the operating system, each time a small object needs to be created.
    /// - Freeing is extremely fast, because the whole pool can be dropped at once. This is
    ///   particularily useful for linked structures, whose nodes need to be freed one-by-one by
    ///   traversing the whole structure otherwise.
    /// - Cache locality is somewhat improved for linked structures with few nodes.
    ///
    /// The allocator doesn't hand out pointers but rather IDs that are relative to the pool. This
    /// simplifies the logic because the pool can easily be moved and hence also resized, but the
    /// downside is that the whole pool must be copied when it runs out of memory. It is therefore
    /// best to start out with a safely large capacity.
    #[derive(Debug)]
    pub(super) struct PoolAllocator<T> {
        pool: Vec<T>,
        // Unsorted list of free slots.
        free_list: Vec<SlotId>,
    }

    impl<T> PoolAllocator<T> {
        pub fn new(capacity: usize) -> Self {
            debug_assert!(capacity > 0);

            PoolAllocator {
                pool: Vec::with_capacity(capacity),
                free_list: Vec::new(),
            }
        }

        /// Allocates a slot and initializes it with the provided value. Returns the ID of the slot.
        pub fn allocate(&mut self, val: T) -> SlotId {
            if let Some(id) = self.free_list.pop() {
                *unsafe { self.get_mut(id) } = val;

                id
            } else {
                self.pool.push(val);

                // SAFETY: `self.pool` is guaranteed to be non-empty.
                SlotId(unsafe { NonZeroUsize::new_unchecked(self.pool.len()) })
            }
        }

        /// Returns the slot with the given ID to the allocator to be reused.
        ///
        /// # Safety
        ///
        /// - `id` must not be freed again, as that would constitute a double-free.
        /// - `id` must not be used to to access the slot again afterward, as that would constitute
        ///   a use-after-free.
        pub unsafe fn free(&mut self, id: SlotId) {
            debug_assert!(!self.free_list.contains(&id));
            self.free_list.push(id);
        }

        /// Returns a mutable reference to the slot with the given ID.
        ///
        /// # Safety
        ///
        /// - `SlotId` must have been allocated by `self`.
        pub unsafe fn get_mut(&mut self, id: SlotId) -> &mut T {
            debug_assert!(!self.free_list.contains(&id));
            debug_assert!(id.0.get() <= self.pool.len());

            // SAFETY:
            // - The caller must uphold that the `SlotId` was allocated with this allocator.
            // - The only way to obtain a `SlotId` is through `Self::allocate`.
            // - `Self::allocate` returns `SlotId`s in the range [1, self.pool.len()].
            // - `self.pool` only grows and never shrinks.
            self.pool.get_unchecked_mut(id.0.get() - 1)
        }
    }

    impl<T: Copy> PoolAllocator<T> {
        /// Returns a copy of the slot with the given ID.
        ///
        /// # Safety
        ///
        /// - `SlotId` must have been allocated by `self`.
        pub unsafe fn get(&self, id: SlotId) -> T {
            debug_assert!(!self.free_list.contains(&id));
            debug_assert!(id.0.get() <= self.pool.len());

            // SAFETY: Same as the `get_unchecked_mut` above.
            *self.pool.get_unchecked(id.0.get() - 1)
        }
    }

    /// ID of a slot in the pool of the `host::PoolAllocator`. This is used to limit the visibility
    /// of the actual ID to this `host` module, making it easier to reason about unsafe code.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) struct SlotId(NonZeroUsize);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryAllocateInfo;
    use std::thread;

    #[test]
    fn memory_alloc_set_offset() {
        let (device, _) = gfx_dev_and_queue!();
        let memory_type_index = device
            .physical_device()
            .memory_properties()
            .memory_types
            .iter()
            .position(|memory_type| {
                memory_type
                    .property_flags
                    .contains(MemoryPropertyFlags::HOST_VISIBLE)
            })
            .unwrap() as u32;
        let mut alloc = MemoryAlloc::new(
            DeviceMemory::allocate(
                device,
                MemoryAllocateInfo {
                    memory_type_index,
                    allocation_size: 1024,
                    ..Default::default()
                },
            )
            .unwrap(),
        )
        .unwrap();
        let ptr = alloc.mapped_ptr().unwrap().as_ptr();

        unsafe {
            alloc.set_offset(16);
            assert_eq!(alloc.mapped_ptr().unwrap().as_ptr(), ptr.offset(16));
            alloc.set_offset(0);
            assert_eq!(alloc.mapped_ptr().unwrap().as_ptr(), ptr.offset(0));
            alloc.set_offset(32);
            assert_eq!(alloc.mapped_ptr().unwrap().as_ptr(), ptr.offset(32));
        }
    }

    #[test]
    fn free_list_allocator_capacity() {
        const THREADS: DeviceSize = 12;
        const ALLOCATIONS_PER_THREAD: DeviceSize = 100;
        const ALLOCATION_STEP: DeviceSize = 117;
        const REGION_SIZE: DeviceSize =
            (ALLOCATION_STEP * (THREADS + 1) * THREADS / 2) * ALLOCATIONS_PER_THREAD;

        let allocator = dummy_allocator!(FreeListAllocator, REGION_SIZE);
        let allocs = ArrayQueue::new((ALLOCATIONS_PER_THREAD * THREADS) as usize);

        // Using threads to randomize allocation order.
        thread::scope(|scope| {
            for i in 1..=THREADS {
                let (allocator, allocs) = (&allocator, &allocs);

                scope.spawn(move || {
                    let info = dummy_info!(i * ALLOCATION_STEP);

                    for _ in 0..ALLOCATIONS_PER_THREAD {
                        allocs
                            .push(allocator.allocate(info.clone()).unwrap())
                            .unwrap();
                    }
                });
            }
        });

        assert!(allocator.allocate(dummy_info!()).is_err());
        assert!(allocator.free_size() == 0);

        drop(allocs);
        assert!(allocator.free_size() == REGION_SIZE);
        assert!(allocator.allocate(dummy_info!(REGION_SIZE)).is_ok());
    }

    #[test]
    fn free_list_allocator_respects_alignment() {
        const REGION_SIZE: DeviceSize = 10 * 256;

        let info = dummy_info!(1, 256);

        let allocator = dummy_allocator!(FreeListAllocator, REGION_SIZE);
        let mut allocs = Vec::with_capacity(10);

        for _ in 0..10 {
            allocs.push(allocator.allocate(info.clone()).unwrap());
        }

        assert!(allocator.allocate(info).is_err());
        assert!(allocator.free_size() == REGION_SIZE - 10);
    }

    #[test]
    fn free_list_allocator_respects_granularity() {
        const GRANULARITY: DeviceSize = 16;
        const REGION_SIZE: DeviceSize = 2 * GRANULARITY;

        let allocator = dummy_allocator!(FreeListAllocator, REGION_SIZE, GRANULARITY);
        let mut linear_allocs = Vec::with_capacity(GRANULARITY as usize);
        let mut nonlinear_allocs = Vec::with_capacity(GRANULARITY as usize);

        for i in 0..REGION_SIZE {
            if i % 2 == 0 {
                linear_allocs.push(allocator.allocate(dummy_info_linear!()).unwrap());
            } else {
                nonlinear_allocs.push(allocator.allocate(dummy_info_nonlinear!()).unwrap());
            }
        }

        assert!(allocator.allocate(dummy_info_linear!()).is_err());
        assert!(allocator.free_size() == 0);

        drop(linear_allocs);
        assert!(allocator.allocate(dummy_info!(GRANULARITY)).is_ok());

        let _alloc = allocator.allocate(dummy_info!()).unwrap();
        assert!(allocator.allocate(dummy_info!()).is_err());
        assert!(allocator.allocate(dummy_info_linear!()).is_err());
    }

    #[test]
    fn pool_allocator_capacity() {
        const BLOCK_SIZE: DeviceSize = 1024;

        fn dummy_allocator(
            device: Arc<Device>,
            allocation_size: DeviceSize,
        ) -> Arc<PoolAllocator<BLOCK_SIZE>> {
            let device_memory = DeviceMemory::allocate(
                device,
                MemoryAllocateInfo {
                    allocation_size,
                    memory_type_index: 0,
                    ..Default::default()
                },
            )
            .unwrap();

            PoolAllocator::new(
                MemoryAlloc::new(device_memory).unwrap(),
                DeviceAlignment::new(1).unwrap(),
            )
        }

        let (device, _) = gfx_dev_and_queue!();

        assert_should_panic!({ dummy_allocator(device.clone(), BLOCK_SIZE - 1) });

        let allocator = dummy_allocator(device.clone(), 2 * BLOCK_SIZE - 1);
        {
            let alloc = allocator.allocate(dummy_info!()).unwrap();
            assert!(allocator.allocate(dummy_info!()).is_err());

            drop(alloc);
            let _alloc = allocator.allocate(dummy_info!()).unwrap();
        }

        let allocator = dummy_allocator(device, 2 * BLOCK_SIZE);
        {
            let alloc1 = allocator.allocate(dummy_info!()).unwrap();
            let alloc2 = allocator.allocate(dummy_info!()).unwrap();
            assert!(allocator.allocate(dummy_info!()).is_err());

            drop(alloc1);
            let alloc1 = allocator.allocate(dummy_info!()).unwrap();
            assert!(allocator.allocate(dummy_info!()).is_err());

            drop(alloc1);
            drop(alloc2);
            let _alloc1 = allocator.allocate(dummy_info!()).unwrap();
            let _alloc2 = allocator.allocate(dummy_info!()).unwrap();
        }
    }

    #[test]
    fn pool_allocator_respects_alignment() {
        const BLOCK_SIZE: DeviceSize = 1024 + 128;

        let info_a = dummy_info!(BLOCK_SIZE, 256);
        let info_b = dummy_info!(1024, 256);

        let allocator = {
            let (device, _) = gfx_dev_and_queue!();
            let device_memory = DeviceMemory::allocate(
                device,
                MemoryAllocateInfo {
                    allocation_size: 10 * BLOCK_SIZE,
                    memory_type_index: 0,
                    ..Default::default()
                },
            )
            .unwrap();

            PoolAllocator::<BLOCK_SIZE>::new(
                MemoryAlloc::new(device_memory).unwrap(),
                DeviceAlignment::new(1).unwrap(),
            )
        };

        // This uses the fact that block indices are inserted into the free-list in order, so
        // the first allocation succeeds because the block has an even index, while the second
        // has an odd index.
        allocator.allocate(info_a.clone()).unwrap();
        assert!(allocator.allocate(info_a.clone()).is_err());
        allocator.allocate(info_a.clone()).unwrap();
        assert!(allocator.allocate(info_a).is_err());

        for _ in 0..10 {
            allocator.allocate(info_b.clone()).unwrap();
        }
    }

    #[test]
    fn pool_allocator_respects_granularity() {
        const BLOCK_SIZE: DeviceSize = 128;

        fn dummy_allocator(
            device: Arc<Device>,
            allocation_type: AllocationType,
        ) -> Arc<PoolAllocator<BLOCK_SIZE>> {
            let device_memory = DeviceMemory::allocate(
                device,
                MemoryAllocateInfo {
                    allocation_size: 1024,
                    memory_type_index: 0,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut region = MemoryAlloc::new(device_memory).unwrap();
            unsafe { region.set_allocation_type(allocation_type) };

            PoolAllocator::new(region, DeviceAlignment::new(256).unwrap())
        }

        let (device, _) = gfx_dev_and_queue!();

        let allocator = dummy_allocator(device.clone(), AllocationType::Unknown);
        assert!(allocator.block_count() == 4);

        let allocator = dummy_allocator(device.clone(), AllocationType::Linear);
        assert!(allocator.block_count() == 8);

        let allocator = dummy_allocator(device, AllocationType::NonLinear);
        assert!(allocator.block_count() == 8);
    }

    #[test]
    fn buddy_allocator_capacity() {
        const MAX_ORDER: usize = 10;
        const REGION_SIZE: DeviceSize = BuddyAllocator::MIN_NODE_SIZE << MAX_ORDER;

        let allocator = dummy_allocator!(BuddyAllocator, REGION_SIZE);
        let mut allocs = Vec::with_capacity(1 << MAX_ORDER);

        for order in 0..=MAX_ORDER {
            let size = BuddyAllocator::MIN_NODE_SIZE << order;

            for _ in 0..1 << (MAX_ORDER - order) {
                allocs.push(allocator.allocate(dummy_info!(size)).unwrap());
            }

            assert!(allocator.allocate(dummy_info!()).is_err());
            assert!(allocator.free_size() == 0);
            allocs.clear();
        }

        let mut orders = (0..MAX_ORDER).collect::<Vec<_>>();

        for mid in 0..MAX_ORDER {
            orders.rotate_left(mid);

            for &order in &orders {
                let size = BuddyAllocator::MIN_NODE_SIZE << order;
                allocs.push(allocator.allocate(dummy_info!(size)).unwrap());
            }

            let _alloc = allocator.allocate(dummy_info!()).unwrap();
            assert!(allocator.allocate(dummy_info!()).is_err());
            assert!(allocator.free_size() == 0);
            allocs.clear();
        }
    }

    #[test]
    fn buddy_allocator_respects_alignment() {
        const REGION_SIZE: DeviceSize = 4096;

        let allocator = dummy_allocator!(BuddyAllocator, REGION_SIZE);

        {
            let info = dummy_info!(1, 4096);

            let _alloc = allocator.allocate(info.clone()).unwrap();
            assert!(allocator.allocate(info).is_err());
            assert!(allocator.free_size() == REGION_SIZE - BuddyAllocator::MIN_NODE_SIZE);
        }

        {
            let info_a = dummy_info!(1, 256);
            let allocations_a = REGION_SIZE / info_a.layout.alignment().as_devicesize();
            let info_b = dummy_info!(1, 16);
            let allocations_b =
                REGION_SIZE / info_b.layout.alignment().as_devicesize() - allocations_a;

            let mut allocs =
                Vec::with_capacity((REGION_SIZE / BuddyAllocator::MIN_NODE_SIZE) as usize);

            for _ in 0..allocations_a {
                allocs.push(allocator.allocate(info_a.clone()).unwrap());
            }

            assert!(allocator.allocate(info_a).is_err());
            assert!(
                allocator.free_size()
                    == REGION_SIZE - allocations_a * BuddyAllocator::MIN_NODE_SIZE
            );

            for _ in 0..allocations_b {
                allocs.push(allocator.allocate(info_b.clone()).unwrap());
            }

            assert!(allocator.allocate(dummy_info!()).is_err());
            assert!(allocator.free_size() == 0);
        }
    }

    #[test]
    fn buddy_allocator_respects_granularity() {
        const GRANULARITY: DeviceSize = 256;
        const REGION_SIZE: DeviceSize = 2 * GRANULARITY;

        let allocator = dummy_allocator!(BuddyAllocator, REGION_SIZE, GRANULARITY);

        {
            const ALLOCATIONS: DeviceSize = REGION_SIZE / BuddyAllocator::MIN_NODE_SIZE;

            let mut allocs = Vec::with_capacity(ALLOCATIONS as usize);
            for _ in 0..ALLOCATIONS {
                allocs.push(allocator.allocate(dummy_info_linear!()).unwrap());
            }

            assert!(allocator.allocate(dummy_info_linear!()).is_err());
            assert!(allocator.free_size() == 0);
        }

        {
            let _alloc1 = allocator.allocate(dummy_info!()).unwrap();
            let _alloc2 = allocator.allocate(dummy_info!()).unwrap();
            assert!(allocator.allocate(dummy_info!()).is_err());
            assert!(allocator.free_size() == 0);
        }
    }

    #[test]
    fn bump_allocator_respects_alignment() {
        const ALIGNMENT: DeviceSize = 16;

        let info = dummy_info!(1, ALIGNMENT);
        let allocator = dummy_allocator!(BumpAllocator, ALIGNMENT * 10);

        for _ in 0..10 {
            allocator.allocate(info.clone()).unwrap();
        }

        assert!(allocator.allocate(info.clone()).is_err());

        for _ in 0..ALIGNMENT - 1 {
            allocator.allocate(dummy_info!()).unwrap();
        }

        assert!(allocator.allocate(info).is_err());
        assert!(allocator.free_size() == 0);
    }

    #[test]
    fn bump_allocator_respects_granularity() {
        const ALLOCATIONS: DeviceSize = 10;
        const GRANULARITY: DeviceSize = 1024;

        let mut allocator = dummy_allocator!(BumpAllocator, GRANULARITY * ALLOCATIONS, GRANULARITY);

        for i in 0..ALLOCATIONS {
            for _ in 0..GRANULARITY {
                allocator
                    .allocate(SuballocationCreateInfo {
                        allocation_type: if i % 2 == 0 {
                            AllocationType::NonLinear
                        } else {
                            AllocationType::Linear
                        },
                        ..dummy_info!()
                    })
                    .unwrap();
            }
        }

        assert!(allocator.allocate(dummy_info_linear!()).is_err());
        assert!(allocator.free_size() == 0);

        allocator.try_reset().unwrap();

        for i in 0..ALLOCATIONS {
            allocator
                .allocate(SuballocationCreateInfo {
                    allocation_type: if i % 2 == 0 {
                        AllocationType::Linear
                    } else {
                        AllocationType::NonLinear
                    },
                    ..dummy_info!()
                })
                .unwrap();
        }

        assert!(allocator.allocate(dummy_info_linear!()).is_err());
        assert!(allocator.free_size() == GRANULARITY - 1);
    }

    #[test]
    fn bump_allocator_syncness() {
        const THREADS: DeviceSize = 12;
        const ALLOCATIONS_PER_THREAD: DeviceSize = 100_000;
        const ALLOCATION_STEP: DeviceSize = 117;
        const REGION_SIZE: DeviceSize =
            (ALLOCATION_STEP * (THREADS + 1) * THREADS / 2) * ALLOCATIONS_PER_THREAD;

        let mut allocator = dummy_allocator!(BumpAllocator, REGION_SIZE);

        thread::scope(|scope| {
            for i in 1..=THREADS {
                let allocator = &allocator;

                scope.spawn(move || {
                    let info = dummy_info!(i * ALLOCATION_STEP);

                    for _ in 0..ALLOCATIONS_PER_THREAD {
                        allocator.allocate(info.clone()).unwrap();
                    }
                });
            }
        });

        assert!(allocator.allocate(dummy_info!()).is_err());
        assert!(allocator.free_size() == 0);

        allocator.try_reset().unwrap();
        assert!(allocator.free_size() == REGION_SIZE);
    }

    macro_rules! dummy_allocator {
        ($type:ty, $size:expr) => {
            dummy_allocator!($type, $size, 1)
        };
        ($type:ty, $size:expr, $granularity:expr) => {{
            let (device, _) = gfx_dev_and_queue!();
            let device_memory = DeviceMemory::allocate(
                device,
                MemoryAllocateInfo {
                    allocation_size: $size,
                    memory_type_index: 0,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut allocator = <$type>::new(MemoryAlloc::new(device_memory).unwrap());
            Arc::get_mut(&mut allocator)
                .unwrap()
                .buffer_image_granularity = DeviceAlignment::new($granularity).unwrap();

            allocator
        }};
    }

    macro_rules! dummy_info {
        () => {
            dummy_info!(1)
        };
        ($size:expr) => {
            dummy_info!($size, 1)
        };
        ($size:expr, $alignment:expr) => {
            SuballocationCreateInfo {
                layout: DeviceLayout::new(
                    NonZeroDeviceSize::new($size).unwrap(),
                    DeviceAlignment::new($alignment).unwrap(),
                )
                .unwrap(),
                allocation_type: AllocationType::Unknown,
                ..Default::default()
            }
        };
    }

    macro_rules! dummy_info_linear {
        ($($args:tt)*) => {
            SuballocationCreateInfo {
                allocation_type: AllocationType::Linear,
                ..dummy_info!($($args)*)
            }
        };
    }

    macro_rules! dummy_info_nonlinear {
        ($($args:tt)*) => {
            SuballocationCreateInfo {
                allocation_type: AllocationType::NonLinear,
                ..dummy_info!($($args)*)
            }
        };
    }

    pub(self) use {dummy_allocator, dummy_info, dummy_info_linear, dummy_info_nonlinear};
}
