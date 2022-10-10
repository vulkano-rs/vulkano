// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! In Vulkan, suballocation of [`DeviceMemory`] is left to the application, because every
//! application has slightly different needs and one can not incorporate an allocator into the
//! driver that would perform well in all cases. Vulkano stays true to this sentiment, but aims to
//! reduce the burden on the user as much as possible. You have a toolbox of configurable
//! [suballocators] to choose from that cover all allocation algorithms, which you can compose into
//! any kind of [hierarchy] you wish. This way you have maximum flexibility while still only using
//! a few `DeviceMemory` blocks and not writing any of the very error-prone code.
//!
//! If you just want to allocate memory and don't have any special needs, look no further than the
//! [`StandardMemoryAllocator`].
//!
//! # Why not just allocate `DeviceMemory`?
//!
//! But the driver has an allocator! Otherwise you wouldn't be able to allocate `DeviceMemory`,
//! right? Indeed, but that allocation is very expensive. Not only that, there is also a pretty low
//! limit on the number of allocations by the drivers. See, everything in Vulkan tries to keep you
//! away from allocating `DeviceMemory` too much. These limits are used by the implementation to
//! optimize on its end, while the application optimizes on the other end.
//!
//! # Alignment
//!
//! At the end of the day, memory needs to be backed by hardware somehow. A *memory cell* stores a
//! single *bit*, bits are grouped into *bytes* and bytes are grouped into *words*. Intuitively, it
//! should make sense that accessing single bits at a time would be very inefficient. That is why
//! computers always access a whole word of memory at once, at least. That means that if you tried
//! to do an unaligned access, you would need to access twice the number of memory locations.
//!
//! Example aligned access, performing bitwise NOT on the (64-bit) word at offset 0x08:
//!
//! ```plain
//!     | 08                      | 10                      | 18
//! ----+-------------------------+-------------------------+----
//! ••• | 35 35 35 35 35 35 35 35 | 01 23 45 67 89 ab cd ef | •••
//! ----+-------------------------+-------------------------+----
//!     ,            |            ,
//!     +------------|------------+
//!     '            v            '
//! ----+-------------------------+-------------------------+----
//! ••• | ca ca ca ca ca ca ca ca | 01 23 45 67 89 ab cd ef | •••
//! ----+-------------------------+-------------------------+----
//! ```
//!
//! Same example as above, but this time unaligned with a word at offset 0x0a:
//!
//! ```plain
//!     | 08    0a                | 10                      | 18
//! ----+-------------------------+-------------------------+----
//! ••• | cd ef 35 35 35 35 35 35 | 35 35 01 23 45 67 89 ab | •••
//! ----+-------------------------+-------------------------+----
//!            ,            |            ,
//!            +------------|------------+
//!            '            v            '
//! ----+-------------------------+-------------------------+----
//! ••• | cd ef ca ca ca ca ca ca | ca ca 01 23 45 67 89 ab | •••
//! ----+-------------------------+-------------------------+----
//! ```
//!
//! As you can see, in the unaligned case the hardware would need to read both the word at offset
//! 0x08 and the word at the offset 0x10 and then shift the bits from one register into the other.
//! Safe to say it should to be avoided, and this is why we need alignment. This example also goes
//! to show how inefficient unaligned writes are. Say you pieced together your word as described,
//! and now you want to perform the bitwise NOT and write the result back. Difficult, isn't it?
//! That's due to the fact that even though the chunks occupy different ranges in memory, they are
//! still said to *alias* each other, because if you try to write to one memory location, you would
//! be overwriting 2 or more different chunks of data.
//!
//! ## Pages
//!
//! It doesn't stop at the word, though. Words are further grouped into *pages*. These are
//! typically power-of-two multiples of the word size, much like words are typically powers of two
//! themselves. You can easily extend the concepts from the previous examples to pages if you think
//! of the examples as having a page size of 1 word. Two resources are said to alias if they share
//! a page, and therefore should be aligned to the page size. What the page size is depends on the
//! context, and a computer might have multiple different ones for different parts of hardware.
//!
//! ## Memory requirements
//!
//! A Vulkan device might have any number of reasons it would want certain alignments for certain
//! resources. For example, the device might have different caches for different types of
//! resources, which have different page sizes. Maybe the device wants to store images in some
//! other cache compared to buffers which needs different alignment. Or maybe images of different
//! layouts require different alignment, or buffers with different usage/mapping do. The specifics
//! don't matter in the end, this just goes to illustrate the point. This is why memory
//! requirements in Vulkan vary not only with the Vulkan implementation, but also with the type of
//! resource.
//!
//! ## Buffer-image granularity
//!
//! This unfortunately named granularity is the page size which a linear resource neighboring a
//! non-linear resource must be aligned to in order for them not to alias. The difference between
//! the memory requirements of the individual resources and the [buffer-image granularity] is that
//! the memory requirements only apply to the resource they are for, while the buffer-image
//! granularity applies to two neighboring resources. For example, you might create two buffers,
//! which might have two different memory requirements, but as long as those are satisfied, you can
//! put these buffers cheek to cheek. On the other hand, if one of them is an (optimal layout)
//! image, then they must not share any page, whose size is given by this granularity. The Vulkan
//! implementation can use this for additional optimizations if it needs to, or report a
//! granularity of 1.
//!
//! # Fragmentation
//!
//! Memory fragmentation refers to the wastage of memory that results from alignment requirements
//! and/or dynamic memory allocation. As such, some level of fragmentation is always going to be
//! inevitable. Different allocation algorithms each have their own characteristics and trade-offs
//! in relation to fragmentation.
//!
//! ## Internal Fragmentation
//!
//! This type of fragmentation arises from alignment requirements. These might be imposed by the
//! Vulkan implementation or the application itself.
//!
//! Say for example your allocations need to be aligned to 64B, then any allocation whose size is
//! not a multiple of the alignment will need padding at the end:
//!
//! ```plain
//!     | 0x040            | 0x080            | 0x0c0            | 0x100
//! ----+------------------+------------------+------------------+--------
//!     | ############     | ################ | ########         | #######
//! ••• | ### 48 B ###     | ##### 64 B ##### | # 32 B #         | ### •••
//!     | ############     | ################ | ########         | #######
//! ----+------------------+------------------+------------------+--------
//! ```
//!
//! If this alignment is imposed by the Vulkan implementation, then there's nothing one can do
//! about this. Simply put, that space is unusable. One also shouldn't want to do anything about
//! it, since these requirements have very good reasons, as described in further detail in previous
//! sections. They prevent resources from aliasing so that performance is optimal.
//!
//! It might seem strange that the application would want to cause internal fragmentation itself,
//! but this is often a good trade-off to reduce or even completely eliminate external
//! fragmentation. Internal fragmentation is very predictable, which makes it easier to deal with.
//!
//! ## External fragmentation
//!
//! With external fragmentation, what happens is that while the allocations might be using their
//! own memory totally efficiently, the way they are arranged in relation to each other would
//! prevent a new contiguous chunk of memory to be allocated even though there is enough free space
//! left. That is why this fragmentation is said to be external to the allocations. Also, the
//! allocations together with the fragments in-between add overhead both in terms of space and time
//! to the allocator, because it needs to keep track of more things overall.
//!
//! As an example, take these 4 allocations within some block, with the rest of the block assumed
//! to be full:
//!
//! ```plain
//! +-----+-------------------+-------+-----------+-- - - --+
//! |     |                   |       |           |         |
//! |  A  |         B         |   C   |     D     |   •••   |
//! |     |                   |       |           |         |
//! +-----+-------------------+-------+-----------+-- - - --+
//! ```
//!
//! The allocations were all done in order, and naturally there is no fragmentation at this point.
//! Now if we free B and D, since these are done out of order, we will be left with holes between
//! the other allocations, and we won't be able to fit allocation E anywhere:
//!
//!  ```plain
//! +-----+-------------------+-------+-----------+-- - - --+       +-------------------------+
//! |     |                   |       |           |         |   ?   |                         |
//! |  A  |                   |   C   |           |   •••   |  <==  |            E            |
//! |     |                   |       |           |         |       |                         |
//! +-----+-------------------+-------+-----------+-- - - --+       +-------------------------+
//! ```
//!
//! So fine, we use a different block for E, and just use this block for allocations that fit:
//!
//! ```plain
//! +-----+---+-----+---------+-------+-----+-----+-- - - --+
//! |     |   |     |         |       |     |     |         |
//! |  A  | H |  I  |    J    |   C   |  F  |  G  |   •••   |
//! |     |   |     |         |       |     |     |         |
//! +-----+---+-----+---------+-------+-----+-----+-- - - --+
//! ```
//!
//! Sure, now let's free some shall we? And voilà, the problem just became much worse:
//!
//! ```plain
//! +-----+---+-----+---------+-------+-----+-----+-- - - --+
//! |     |   |     |         |       |     |     |         |
//! |  A  |   |  I  |    J    |       |  F  |     |   •••   |
//! |     |   |     |         |       |     |     |         |
//! +-----+---+-----+---------+-------+-----+-----+-- - - --+
//! ```
//!
//! # Leakage
//!
//! Memory leaks happen when allocations are kept alive past their shelf life. This most often
//! occurs because of [cyclic references]. If you have structures that have cycles, then make sure
//! you read the documentation for [`Arc`]/[`Rc`] carefully to avoid memory leaks. You can also
//! introduce memory leaks willingly by using [`mem::forget`] or [`Box::leak`] to name a few. In
//! all of these examples the memory can never be reclaimed, but that doesn't have to be the case
//! for something to be considered a leak. Say for example you have a [region] which you
//! suballocate, and at some point you drop all the suballocations. When that happens, the region
//! can be returned (freed) to the next level up the hierarchy, or it can be reused by another
//! suballocator. But if you happen to keep alive just one suballocation for the duration of the
//! program for instance, then the whole region is also kept as it is for that time (and keep in
//! mind this bubbles up the hierarchy). Therefore, for the program, that memory might be a leak
//! depending on the allocator, because some allocators wouldn't be able to reuse the entire rest
//! of the region. You must always consider the lifetime of your resources when choosing the
//! appropriate allocator.
//!
//! [suballocators]: Suballocator
//! [hierarchy]: Suballocator#memory-hierarchies
//! [buffer-image granularity]: crate::device::Properties::buffer_image_granularity
//! [cyclic references]: Arc#breaking-cycles-with-weak
//! [`Rc`]: std::rc::Rc
//! [region]: Suballocator#regions

use self::host::SlotId;
use super::{DeviceMemory, MemoryAllocateInfo};
use crate::{device::DeviceOwned, DeviceSize, VulkanObject};
use crossbeam_queue::ArrayQueue;
use parking_lot::Mutex;
use std::{
    cell::Cell,
    error::Error,
    fmt::{self, Display},
    mem::{self, ManuallyDrop},
    ptr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

/// Memory allocations are portions of memory that are are reserved for a specific resource or
/// purpose.
///
/// There's a few ways you can obtain a `MemoryAlloc` in Vulkano. Most commonly you will probably
/// want to use one of the [generic memory allocators]. If you want a root allocation, and already
/// have a [`DeviceMemory`] block on hand, you can turn it into a `MemoryAlloc` by using the
/// [`From`] implementation. Lastly, you can use a [suballocator] if you want to create multiple
/// smaller allocations out of a bigger one.
///
/// [generic memory allocators]: MemoryAllocator
/// [`From`]: Self#impl-From<DeviceMemory>-for-MemoryAlloc
/// [suballocator]: Suballocator
#[derive(Debug)]
pub struct MemoryAlloc {
    offset: DeviceSize,
    size: DeviceSize,
    // Needed when binding resources to the allocation in order to avoid aliasing memory.
    allocation_type: AllocationType,
    // Used in the `Drop` impl to free the allocation if required.
    parent: AllocParent,
    // Underlying block of memory. This field is duplicated here to avoid walking up the hierarchy
    // when binding.
    memory: ash::vk::DeviceMemory,
    memory_type_index: u32,
    // Used by the suballocators to resolve buffer-image granularity conflicts. This field is
    // duplicated here to avoid walking up the hierarchy when creating a suballocator.
    buffer_image_granularity: DeviceSize,
}

#[derive(Debug)]
enum AllocParent {
    FreeList {
        allocator: Arc<FreeListAllocator>,
        id: SlotId,
    },
    Pool {
        allocator: Arc<PoolAllocatorInner>,
        index: DeviceSize,
    },
    Buddy {
        allocator: Arc<BuddyAllocator>,
        order: usize,
    },
    Bump(Arc<BumpAllocator>),
    Root(Arc<DeviceMemory>),
    Dedicated(DeviceMemory),
    #[cfg(test)]
    None,
}

impl MemoryAlloc {
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

    /// Returns the index of the memory type that this allocation resides in.
    #[inline]
    pub fn memory_type_index(&self) -> u32 {
        self.memory_type_index
    }

    /// Returns the parent allocation if this allocation is a [suballocation], otherwise returns the
    /// [`DeviceMemory`] wrapped in [`Err`].
    ///
    /// [suballocation]: Suballocator
    #[inline]
    pub fn parent_allocation(&self) -> Result<&Self, &DeviceMemory> {
        match &self.parent {
            AllocParent::FreeList { allocator, .. } => Ok(&allocator.region),
            AllocParent::Pool { allocator, .. } => Ok(&allocator.region),
            AllocParent::Buddy { allocator, .. } => Ok(&allocator.region),
            AllocParent::Bump(allocator) => Ok(&allocator.region),
            AllocParent::Root(device_memory) => Err(device_memory),
            AllocParent::Dedicated(device_memory) => Err(device_memory),
            #[cfg(test)]
            AllocParent::None => unreachable!(),
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
    /// [dedicated allocation]: MemoryAllocateInfo#structfield.dedicated_allocation
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
    /// This has the performance of traversing a linked list, *O*(*n*), where *n* is the height of
    /// the [memory hierarchy].
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
            AllocParent::FreeList { allocator, .. } => allocator.region.root(),
            AllocParent::Pool { allocator, .. } => allocator.region.root(),
            AllocParent::Buddy { allocator, .. } => allocator.region.root(),
            AllocParent::Bump(allocator) => allocator.region.root(),
            AllocParent::Root(device_memory) => Some(device_memory),
            AllocParent::Dedicated(_) => None,
            #[cfg(test)]
            AllocParent::None => unreachable!(),
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

        self.offset += amount;
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
    /// [buffer-image granularity]: self#buffer-image-granularity
    /// [`Linear`]: AllocationType::Linear
    /// [`NonLinear`]: AllocationType::NonLinear
    #[inline]
    pub unsafe fn set_allocation_type(&mut self, new_type: AllocationType) {
        self.allocation_type = new_type;
    }
}

impl From<DeviceMemory> for MemoryAlloc {
    /// Converts the `DeviceMemory` into a root allocation.
    #[inline]
    fn from(device_memory: DeviceMemory) -> Self {
        MemoryAlloc {
            offset: 0,
            size: device_memory.allocation_size(),
            allocation_type: AllocationType::Unknown,
            memory: device_memory.internal_object(),
            memory_type_index: device_memory.memory_type_index(),
            buffer_image_granularity: device_memory
                .device()
                .physical_device()
                .properties()
                .buffer_image_granularity,
            parent: AllocParent::Root(Arc::new(device_memory)),
        }
    }
}

impl Drop for MemoryAlloc {
    #[inline]
    fn drop(&mut self) {
        match &self.parent {
            AllocParent::FreeList { allocator, id } => {
                allocator.free(*id);
            }
            AllocParent::Pool { allocator, index } => {
                allocator.free(*index);
            }
            AllocParent::Buddy { allocator, order } => {
                allocator.free(*order, self.offset);
            }
            // The bump allocator can't free individually, but we need to keep a reference to it so
            // it don't get reset or dropped while in use.
            AllocParent::Bump(_) => {}
            // A root allocation frees itself once all references to the `DeviceMemory` are dropped.
            AllocParent::Root(_) => {}
            // Dedicated allocations free themselves when the `DeviceMemory` is dropped.
            AllocParent::Dedicated(_) => {}
            #[cfg(test)]
            AllocParent::None => {}
        }
    }
}

unsafe impl VulkanObject for MemoryAlloc {
    type Object = ash::vk::DeviceMemory;

    #[inline]
    fn internal_object(&self) -> Self::Object {
        self.memory
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
/// [allocations]: MemoryAlloc
/// [pages]: self#pages
pub trait Suballocator {
    /// Creates a new suballocator for the given [region].
    ///
    /// [region]: Self#regions
    fn new(region: MemoryAlloc) -> Self
    where
        Self: Sized;

    /// Creates a new suballocation within the [region].
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.size` is zero.
    /// - Panics if `create_info.alignment` is zero.
    /// - Panics if `create_info.alignment` is not a power of two.
    ///
    /// [region]: Self#regions
    #[inline]
    fn allocate(
        &self,
        create_info: SuballocationCreateInfo,
    ) -> Result<MemoryAlloc, SuballocationError> {
        assert!(create_info.size > 0);
        assert!(create_info.alignment > 0);
        assert!(create_info.alignment.is_power_of_two());

        unsafe { self.allocate_unchecked(create_info) }
    }

    /// Creates a new suballocation within the [region] without checking the parameters.
    ///
    /// See [`allocate`] for the safe version.
    ///
    /// # Safety
    ///
    /// - `create_info.size` must not be zero.
    /// - `create_info.alignment` must not be zero.
    /// - `create_info.alignment` must be a power of two.
    ///
    /// [region]: Self#regions
    /// [`allocate`]: Self::allocate
    unsafe fn allocate_unchecked(
        &self,
        create_info: SuballocationCreateInfo,
    ) -> Result<MemoryAlloc, SuballocationError>;

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

    /// Returns the amount of free space that is left in the [region].
    ///
    /// [region]: Self#regions
    fn free_size(&self) -> DeviceSize;
}

/// Parameters to create a new [allocation] using a [suballocator].
///
/// [allocation]: MemoryAlloc
/// [suballocator]: Suballocator
#[derive(Clone, Debug)]
pub struct SuballocationCreateInfo {
    /// Size of the allocation in bytes.
    ///
    /// The default value is `0`, which must be overridden.
    pub size: DeviceSize,

    /// [Alignment] of the allocation in bytes. Must be a power of 2.
    ///
    /// The default value is `0`, which must be overridden.
    ///
    /// [Alignment]: self#alignment
    pub alignment: DeviceSize,

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
            size: 0,
            alignment: 0,
            allocation_type: AllocationType::Unknown,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Tells the [suballocator] what type of resource will be bound to the allocation, so that it can
/// optimize memory usage while still respecting the [buffer-image granularity].
///
/// [suballocator]: Suballocator
/// [buffer-image granularity]: self#buffer-image-granularity
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

/// Error that can be returned when using a [suballocator].
///
/// [suballocator]: Suballocator
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SuballocationError {
    /// There is no more space available in the region.
    OutOfRegionMemory,

    /// The region has enough free space to satisfy the request but is too fragmented.
    FragmentedRegion,
}

impl Display for SuballocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                SuballocationError::OutOfRegionMemory => "out of region memory",
                SuballocationError::FragmentedRegion => "the region is too fragmented",
            }
        )
    }
}

impl Error for SuballocationError {}

/// A [suballocator] that uses the most generic [free-list].
///
/// The strength of this allocator is that it can create and free allocations completely
/// dynamically, which means they can be any size and created/freed in any order. The downside is
/// that this always leads to horrific [external fragmentation] the more such dynamic allocations
/// are made. Therefore, this allocator is best suited for long-lived allocations. If you need need
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
/// [suballocator]: Suballocator
/// [free-list]: Suballocator#free-lists
/// [external fragmentation]: self#external-fragmentation
/// [the `Suballocator` implementation]: Suballocator#impl-Suballocator-for-Arc<FreeListAllocator>
/// [internal fragmentation]: self#internal-fragmentation
/// [alignment requirements]: self#alignment
#[derive(Debug)]
pub struct FreeListAllocator {
    region: MemoryAlloc,
    inner: Mutex<FreeListAllocatorInner>,
}

impl FreeListAllocator {
    /// Creates a new `FreeListAllocator` for the given [region].
    ///
    /// # Panics
    ///
    /// - Panics if `region.allocation_type` is not [`AllocationType::Unknown`]. This is done to
    ///   avoid checking for a special case of [buffer-image granularity] conflict.
    ///
    /// [region]: Suballocator#regions
    /// [buffer-image granularity]: self#buffer-image-granularity
    #[inline]
    pub fn new(region: MemoryAlloc) -> Arc<Self> {
        // NOTE(Marc): This number was pulled straight out of my a-
        const AVERAGE_ALLOCATION_SIZE: DeviceSize = 64 * 1024;

        assert!(region.allocation_type == AllocationType::Unknown);

        let capacity = (region.size / AVERAGE_ALLOCATION_SIZE) as usize;
        let mut nodes = host::PoolAllocator::new(capacity + 64);
        let mut free_list = Vec::with_capacity(capacity / 16 + 16);
        let root_id = nodes.allocate(SuballocationListNode {
            prev: None,
            next: None,
            offset: 0,
            size: region.size,
            ty: SuballocationType::Free,
        });
        free_list.push(root_id);

        let inner = FreeListAllocatorInner {
            nodes,
            free_list,
            free_size: region.size,
        };

        Arc::new(FreeListAllocator {
            region,
            inner: Mutex::new(inner),
        })
    }

    fn free(&self, id: SlotId) {
        let mut inner = self.inner.lock();
        inner.nodes.get_mut(id).ty = SuballocationType::Free;
        inner.coalesce(id);
        inner.free(id);
    }
}

impl Suballocator for Arc<FreeListAllocator> {
    #[inline]
    fn new(region: MemoryAlloc) -> Self {
        FreeListAllocator::new(region)
    }

    /// Creates a new suballocation within the [region] without checking the parameters.
    ///
    /// See [`allocate`] for the safe version.
    ///
    /// # Errors
    ///
    /// - Returns [`OutOfRegionMemory`] if there are no free suballocations large enough so satisfy
    ///   the request.
    /// - Returns [`FragmentedRegion`] if a suballocation large enough to satisfy the request could
    ///   have been formed, but wasn't because of [external fragmentation].
    ///
    /// # Safety
    ///
    /// - `create_info.size` must not be zero.
    /// - `create_info.alignment` must not be zero.
    /// - `create_info.alignment` must be a power of two.
    ///
    /// [region]: Suballocator#regions
    /// [`allocate`]: Suballocator::allocate
    /// [`OutOfRegionMemory`]: SuballocationError::OutOfRegionMemory
    /// [`FragmentedRegion`]: SuballocationError::FragmentedRegion
    /// [external fragmentation]: self#external-fragmentation
    #[inline]
    unsafe fn allocate_unchecked(
        &self,
        create_info: SuballocationCreateInfo,
    ) -> Result<MemoryAlloc, SuballocationError> {
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
            size,
            alignment,
            allocation_type,
            _ne: _,
        } = create_info;

        let mut inner = self.inner.lock();

        match inner.free_list.last() {
            Some(&last) if inner.nodes.get(last).size >= size => {
                let index = match inner
                    .free_list
                    .binary_search_by_key(&size, |&x| inner.nodes.get(x).size)
                {
                    // Exact fit.
                    Ok(index) => index,
                    // Next-best fit. Note that `index == free_list.len()` can not be because we
                    // checked that the free-list contains a suballocation that is big enough.
                    Err(index) => index,
                };

                for &id in &inner.free_list[index..] {
                    let suballoc = inner.nodes.get(id);
                    let mut offset = align_up(self.region.offset + suballoc.offset, alignment);

                    if let Some(prev_id) = suballoc.prev {
                        let prev = inner.nodes.get(prev_id);

                        if are_blocks_on_same_page(
                            prev.offset,
                            prev.size,
                            offset,
                            self.region.buffer_image_granularity,
                        ) && has_granularity_conflict(prev.ty, allocation_type)
                        {
                            offset = align_up(offset, self.region.buffer_image_granularity);
                        }
                    }

                    if offset + size <= suballoc.offset + suballoc.size {
                        inner.allocate(id);
                        inner.split(id, offset, size);
                        inner.nodes.get_mut(id).ty = allocation_type.into();

                        return Ok(MemoryAlloc {
                            offset,
                            size,
                            allocation_type,
                            parent: AllocParent::FreeList {
                                allocator: self.clone(),
                                id,
                            },
                            memory: self.region.memory,
                            memory_type_index: self.region.memory_type_index,
                            buffer_image_granularity: self.region.buffer_image_granularity,
                        });
                    }
                }

                // There is not enough space due to alignment requirements.
                Err(SuballocationError::OutOfRegionMemory)
            }
            // There would be enough space if the region wasn't so fragmented. :(
            Some(_) if inner.free_size >= size => Err(SuballocationError::FragmentedRegion),
            // There is not enough space.
            Some(_) => Err(SuballocationError::OutOfRegionMemory),
            // There is no space at all.
            None => Err(SuballocationError::OutOfRegionMemory),
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
        self.inner.lock().free_size
    }
}

#[derive(Debug)]
struct FreeListAllocatorInner {
    nodes: host::PoolAllocator<SuballocationListNode>,
    // Free suballocations sorted by size in ascending order. This means we can always find a
    // best-fit in *O*(log(*n*)) time in the worst case, and iterating in order is very efficient.
    free_list: Vec<SlotId>,
    // Total memory remaining in the region.
    free_size: DeviceSize,
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

impl FreeListAllocatorInner {
    /// Removes the target suballocation from the free-list. The free-list must contain it.
    fn allocate(&mut self, node_id: SlotId) {
        debug_assert!(self.free_list.contains(&node_id));

        let node = self.nodes.get(node_id);
        self.free_size -= node.size;

        match self
            .free_list
            .binary_search_by_key(&node.size, |&x| self.nodes.get(x).size)
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
    fn split(&mut self, node_id: SlotId, offset: DeviceSize, size: DeviceSize) {
        let node = self.nodes.get(node_id);

        debug_assert!(node.ty == SuballocationType::Free);
        debug_assert!(offset >= node.offset);
        debug_assert!(offset + size <= node.offset + node.size);

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
            node.size -= padding.size;

            self.free(padding_id);
        }
    }

    /// Inserts the target suballocation into the free-list. The free-list must not contain it
    /// already.
    fn free(&mut self, node_id: SlotId) {
        debug_assert!(!self.free_list.contains(&node_id));

        let node = self.nodes.get(node_id);
        self.free_size += node.size;
        let index = match self
            .free_list
            .binary_search_by_key(&node.size, |&x| self.nodes.get(x).size)
        {
            Ok(index) => index,
            Err(index) => index,
        };
        self.free_list.insert(index, node_id);
    }

    /// Coalesces the target (free) suballocation with adjacent ones that are also free.
    fn coalesce(&mut self, node_id: SlotId) {
        let node = self.nodes.get(node_id);

        debug_assert!(node.ty == SuballocationType::Free);

        if let Some(prev_id) = node.prev {
            let prev = self.nodes.get(prev_id);

            if prev.ty == SuballocationType::Free {
                self.allocate(prev_id);
                self.nodes.free(prev_id);

                let node = self.nodes.get_mut(node_id);
                node.prev = prev.prev;
                node.offset = prev.offset;
                node.size += prev.size; // nom nom nom

                if let Some(prev_id) = node.prev {
                    self.nodes.get_mut(prev_id).next = Some(node_id);
                }
            }
        }

        if let Some(next_id) = node.next {
            let next = self.nodes.get(next_id);

            if next.ty == SuballocationType::Free {
                self.allocate(next_id);
                self.nodes.free(next_id);

                let node = self.nodes.get_mut(node_id);
                node.next = next.next;
                node.size += next.size;

                if let Some(next_id) = node.next {
                    self.nodes.get_mut(next_id).prev = Some(node_id);
                }
            }
        }
    }
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
/// regions of different sizes, which are still going to be predictable and tunable though. The
/// [`BuddyAllocator`] is perfectly suited for this task. You could also consider
/// [`FreeListAllocator`] if external fragmentation is not an issue, or you can't align your
/// regions nicely causing too much internal fragmentation with the buddy system. On the other
/// hand, you might also want to consider having a `PoolAllocator` at the top of a [hierarchy].
/// Again, this allocator never needs to lock making it *the* perfect fit for a global concurrent
/// allocator, which hands out large regions which can then be suballocated locally on a thread, by
/// the [`BumpAllocator`] for example, for optimal performance.
///
/// [suballocator]: Suballocator
/// [free-list]: Suballocator#free-lists
/// [internal fragmentation]: self#internal-fragmentation
/// [external fragmentation]: self#external-fragmentation
/// [the `Suballocator` implementation]: Suballocator#impl-Suballocator-for-Arc<PoolAllocator<BLOCK_SIZE>>
/// [region]: Suballocator#regions
/// [buffer-image granularity]: self#buffer-image-granularity
/// [`Linear`]: AllocationType::Linear
/// [`NonLinear`]: AllocationType::NonLinear
/// [`Unknown`]: AllocationType::Unknown
/// [suballocation]: SuballocationCreateInfo
/// [alignment requirements]: self#memory-requirements
/// [align]: self#alignment
/// [hierarchy]: Suballocator#memory-hierarchies
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
    ///
    /// [region]: Suballocator#regions
    #[inline]
    pub fn new(region: MemoryAlloc) -> Arc<Self> {
        // SAFETY: `PoolAllocator<BLOCK_SIZE>` and `PoolAllocatorInner` have the same layout.
        unsafe {
            Arc::from_raw(
                Arc::into_raw(PoolAllocatorInner::new(region, BLOCK_SIZE))
                    .cast::<PoolAllocator<BLOCK_SIZE>>(),
            )
        }
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

impl<const BLOCK_SIZE: DeviceSize> Suballocator for Arc<PoolAllocator<BLOCK_SIZE>> {
    #[inline]
    fn new(region: MemoryAlloc) -> Self {
        PoolAllocator::new(region)
    }

    /// Creates a new suballocation within the [region] without checking the parameters.
    ///
    /// See [`allocate`] for the safe version.
    ///
    /// > **Note**: `create_info.allocation_type` is silently ignored because all suballocations
    /// > inherit the same allocation type from the allocator.
    ///
    /// # Errors
    ///
    /// - Returns [`OutOfRegionMemory`] if the [free-list] is empty.
    /// - Returns [`OutOfRegionMemory`] if the allocation can't fit inside a block. Only the first
    ///   block in the free-list is tried, which means that if one block isn't usable due to
    ///   [internal fragmentation] but a different one would be, you still get this error. See the
    ///   [type-level documentation] for details on how to properly configure your allocator.
    ///
    /// # Safety
    ///
    /// - `create_info.size` must not be zero.
    /// - `create_info.alignment` must not be zero.
    /// - `create_info.alignment` must be a power of two.
    ///
    /// [region]: Suballocator#regions
    /// [`allocate`]: Suballocator::allocate
    /// [`OutOfRegionMemory`]: SuballocationError::OutOfRegionMemory
    /// [free-list]: Suballocator#free-lists
    /// [internal fragmentation]: self#internal-fragmentation
    /// [type-level documentation]: PoolAllocator
    #[inline]
    unsafe fn allocate_unchecked(
        &self,
        create_info: SuballocationCreateInfo,
    ) -> Result<MemoryAlloc, SuballocationError> {
        // SAFETY: `PoolAllocator<BLOCK_SIZE>` and `PoolAllocatorInner` have the same layout.
        //
        // This is not quite optimal, because we are always cloning the `Arc` even if allocation
        // fails, in which case the `Arc` gets cloned and dropped for no reason. Unfortunately,
        // there is currently no way to turn `&Arc<T>` into `&Arc<U>` that is sound.
        Arc::from_raw(Arc::into_raw(self.clone()).cast::<PoolAllocatorInner>())
            .allocate_unchecked(create_info)
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
}

#[derive(Debug)]
struct PoolAllocatorInner {
    region: MemoryAlloc,
    block_size: DeviceSize,
    // Unsorted list of free block indices.
    free_list: ArrayQueue<DeviceSize>,
}

impl PoolAllocatorInner {
    fn new(region: MemoryAlloc, mut block_size: DeviceSize) -> Arc<Self> {
        if region.allocation_type == AllocationType::Unknown {
            block_size = align_up(block_size, region.buffer_image_granularity);
        }

        let block_count = region.size / block_size;
        let free_list = ArrayQueue::new(block_count as usize);
        for i in 0..block_count {
            free_list.push(i).unwrap();
        }

        Arc::new(PoolAllocatorInner {
            region,
            block_size,
            free_list,
        })
    }

    unsafe fn allocate_unchecked(
        self: Arc<Self>,
        create_info: SuballocationCreateInfo,
    ) -> Result<MemoryAlloc, SuballocationError> {
        let SuballocationCreateInfo {
            size,
            alignment,
            allocation_type: _,
            _ne: _,
        } = create_info;

        let index = self
            .free_list
            .pop()
            .ok_or(SuballocationError::OutOfRegionMemory)?;
        let unaligned_offset = index * self.block_size;
        let offset = align_up(unaligned_offset, alignment);

        if offset + size > unaligned_offset + self.block_size {
            self.free_list.push(index).unwrap();

            return Err(SuballocationError::OutOfRegionMemory);
        }

        Ok(MemoryAlloc {
            offset,
            size,
            allocation_type: self.region.allocation_type,
            memory: self.region.memory,
            memory_type_index: self.region.memory_type_index,
            buffer_image_granularity: self.region.buffer_image_granularity,
            parent: AllocParent::Pool {
                allocator: self,
                index,
            },
        })
    }

    fn free(&self, index: DeviceSize) {
        self.free_list.push(index).unwrap();
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
/// Say you have a region of size 256MiB, and you want to allocate 14MiB. Assuming there are no
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
/// allocation sizes, so that you don't end allocating twice as much memory. An example of this
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
/// [suballocator]: Suballocator
/// [internal fragmentation]: self#internal-fragmentation
/// [external fragmentation]: self#external-fragmentation
/// [the `Suballocator` implementation]: Suballocator#impl-Suballocator-for-Arc<BuddyAllocator>
/// [region]: Suballocator#regions
/// [buffer-image granularity]: self#buffer-image-granularity
#[derive(Debug)]
pub struct BuddyAllocator {
    region: MemoryAlloc,
    order_count: usize,
    inner: Mutex<BuddyAllocatorInner>,
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
    ///
    /// [region]: Suballocator#regions
    /// [buffer-image granularity]: self#buffer-image-granularity
    #[inline]
    pub fn new(region: MemoryAlloc) -> Arc<Self> {
        const EMPTY_FREE_LIST: Vec<DeviceSize> = Vec::new();

        let max_order = (region.size / BuddyAllocator::MIN_NODE_SIZE).trailing_zeros() as usize;

        assert!(region.allocation_type == AllocationType::Unknown);
        assert!(region.size.is_power_of_two());
        assert!(
            region.size >= BuddyAllocator::MIN_NODE_SIZE && max_order < BuddyAllocator::MAX_ORDERS
        );

        let mut free_list = [EMPTY_FREE_LIST; BuddyAllocator::MAX_ORDERS];
        // The root node has the lowest offset and highest order, so it's the whole region.
        free_list[max_order].push(region.offset);
        let inner = BuddyAllocatorInner {
            free_list,
            free_size: region.size,
        };

        Arc::new(BuddyAllocator {
            region,
            order_count: max_order + 1,
            inner: Mutex::new(inner),
        })
    }

    /// Number of orders in the tree. This is always equal to log(*region&nbsp;size*)&nbsp;-&nbsp;3
    /// (that is, the highest order plus one).
    #[inline]
    pub fn order_count(&self) -> usize {
        self.order_count
    }

    fn free(&self, min_order: usize, mut offset: DeviceSize) {
        let mut inner = self.inner.lock();

        // Try to coalesce nodes while incrementing the order.
        for order in min_order..self.order_count {
            let size = Self::MIN_NODE_SIZE << order;
            let buddy_offset = ((offset - self.region.offset) ^ size) + self.region.offset;

            match inner.free_list[order].binary_search(&buddy_offset) {
                // If the buddy is in the free-list, we can coalesce.
                Ok(index) => {
                    inner.free_list[order].remove(index);
                    offset = DeviceSize::min(offset, buddy_offset);
                }
                // Otherwise free the node.
                Err(_) => {
                    let index = match inner.free_list[order].binary_search(&offset) {
                        Ok(index) => index,
                        Err(index) => index,
                    };
                    inner.free_list[order].insert(index, offset);
                    inner.free_size += Self::MIN_NODE_SIZE << min_order;

                    break;
                }
            }
        }
    }
}

impl Suballocator for Arc<BuddyAllocator> {
    #[inline]
    fn new(region: MemoryAlloc) -> Self {
        BuddyAllocator::new(region)
    }

    /// Creates a new suballocation within the [region] without checking the parameters.
    ///
    /// See [`allocate`] for the safe version.
    ///
    /// # Errors
    ///
    /// - Returns [`OutOfRegionMemory`] if there are no free nodes large enough so satisfy the
    ///   request.
    /// - Returns [`FragmentedRegion`] if a node large enough to satisfy the request could have
    ///   been formed, but wasn't because of [external fragmentation].
    ///
    /// # Safety
    ///
    /// - `create_info.size` must not be zero.
    /// - `create_info.alignment` must not be zero.
    /// - `create_info.alignment` must be a power of two.
    ///
    /// [region]: Suballocator#regions
    /// [`allocate`]: Suballocator::allocate
    /// [`OutOfRegionMemory`]: SuballocationError::OutOfRegionMemory
    /// [`FragmentedRegion`]: SuballocationError::FragmentedRegion
    /// [external fragmentation]: self#external-fragmentation
    #[inline]
    unsafe fn allocate_unchecked(
        &self,
        create_info: SuballocationCreateInfo,
    ) -> Result<MemoryAlloc, SuballocationError> {
        /// Returns the largest power of two smaller or equal to the input.
        fn prev_power_of_two(val: DeviceSize) -> DeviceSize {
            const MAX_POWER_OF_TWO: DeviceSize = 1 << (DeviceSize::BITS - 1);

            MAX_POWER_OF_TWO
                .checked_shr(val.leading_zeros())
                .unwrap_or(0)
        }

        let SuballocationCreateInfo {
            mut size,
            mut alignment,
            allocation_type,
            _ne: _,
        } = create_info;

        if allocation_type == AllocationType::Unknown
            || allocation_type == AllocationType::NonLinear
        {
            size = align_up(size, self.region.buffer_image_granularity);
            alignment = DeviceSize::max(alignment, self.region.buffer_image_granularity);
        }

        let size = DeviceSize::max(size, BuddyAllocator::MIN_NODE_SIZE).next_power_of_two();
        let min_order = (size / BuddyAllocator::MIN_NODE_SIZE).trailing_zeros() as usize;
        let mut inner = self.inner.lock();

        // Start searching at the lowest possible order going up.
        for order in min_order..self.order_count {
            for (index, offset) in inner.free_list[order].iter().copied().enumerate() {
                if offset % alignment == 0 {
                    inner.free_list[order].remove(index);

                    // Go in the opposite direction, splitting nodes from higher orders. The lowest
                    // order doesn't need any splitting.
                    for order in (min_order..order).rev() {
                        let size = BuddyAllocator::MIN_NODE_SIZE << order;
                        let right_child = offset + size;

                        // Insert the right child in sorted order.
                        let index = match inner.free_list[order].binary_search(&right_child) {
                            Ok(index) => index,
                            Err(index) => index,
                        };
                        inner.free_list[order].insert(index, right_child);

                        // Repeat splitting for the left child if required in the next loop turn.
                    }

                    inner.free_size -= size;

                    return Ok(MemoryAlloc {
                        offset,
                        size: create_info.size,
                        allocation_type,
                        parent: AllocParent::Buddy {
                            allocator: self.clone(),
                            order: min_order,
                        },
                        memory: self.region.memory,
                        memory_type_index: self.region.memory_type_index,
                        buffer_image_granularity: self.region.buffer_image_granularity,
                    });
                }
            }
        }

        if prev_power_of_two(inner.free_size) >= create_info.size {
            // A node large enough could be formed if the region wasn't so fragmented.
            Err(SuballocationError::FragmentedRegion)
        } else {
            Err(SuballocationError::OutOfRegionMemory)
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

    /// Returns the amount of free space left in the [region] that is available to the allocator,
    /// which means that [internal fragmentation] is excluded.
    ///
    /// [region]: Suballocator#regions
    /// [internal fragmentation]: self#internal-fragmentation
    #[inline]
    fn free_size(&self) -> DeviceSize {
        self.inner.lock().free_size
    }
}

#[derive(Debug)]
struct BuddyAllocatorInner {
    // Every order has its own free-list for convenience, so that we don't have to traverse a tree.
    // Each free-list is sorted by offset because we want to find the first-fit as this strategy
    // minimizes external fragmentation.
    free_list: [Vec<DeviceSize>; BuddyAllocator::MAX_ORDERS],
    // Total free space remaining in the region.
    free_size: DeviceSize,
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
/// [memory leaks]: self#leakage
/// [`try_reset`]: Self::try_reset
/// [hierarchy]: Suballocator#memory-hierarchies
#[derive(Debug)]
pub struct BumpAllocator {
    region: MemoryAlloc,
    // Encodes the previous allocation type in the 2 least signifficant bits and the free start in
    // the rest.
    state: AtomicU64,
}

impl BumpAllocator {
    /// Creates a new `BumpAllocator` for the given [region].
    ///
    /// [region]: Suballocator#regions
    #[inline]
    pub fn new(region: MemoryAlloc) -> Arc<Self> {
        Arc::new(BumpAllocator {
            state: AtomicU64::new(region.allocation_type as u64),
            region,
        })
    }

    /// Resets the free start back to the beginning of the [region] if there are no other strong
    /// references to the allocator.
    ///
    /// [region]: Suballocator#regions
    #[inline]
    pub fn try_reset(self: &mut Arc<Self>) -> Result<(), BumpAllocatorResetError> {
        Arc::get_mut(self)
            .map(|allocator| {
                *allocator.state.get_mut() = allocator.region.allocation_type as u64;
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
            .store(self.region.allocation_type as u64, Ordering::Relaxed);
    }
}

impl Suballocator for Arc<BumpAllocator> {
    #[inline]
    fn new(region: MemoryAlloc) -> Self {
        BumpAllocator::new(region)
    }

    /// Creates a new suballocation within the [region] without checking the parameters.
    ///
    /// See [`allocate`] for the safe version.
    ///
    /// # Errors
    ///
    /// - Returns [`OutOfRegionMemory`] if the requested allocation can't fit in the free space
    ///   remaining in the region.
    ///
    /// # Safety
    ///
    /// - `create_info.size` must not be zero.
    /// - `create_info.alignment` must not be zero.
    /// - `create_info.alignment` must be a power of two.
    ///
    /// [region]: Suballocator#regions
    /// [`allocate`]: Suballocator::allocate
    /// [`OutOfRegionMemory`]: SuballocationError::OutOfRegionMemory
    #[inline]
    unsafe fn allocate_unchecked(
        &self,
        create_info: SuballocationCreateInfo,
    ) -> Result<MemoryAlloc, SuballocationError> {
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
            size,
            alignment,
            allocation_type,
            _ne: _,
        } = create_info;

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
            let prev_end = self.region.offset + free_start;
            let mut offset = align_up(prev_end, alignment);

            if prev_end > 0
                && are_blocks_on_same_page(
                    prev_end,
                    0,
                    offset,
                    self.region.buffer_image_granularity,
                )
                && has_granularity_conflict(prev_alloc_type, allocation_type)
            {
                offset = align_up(offset, self.region.buffer_image_granularity);
            }

            let free_start = offset - self.region.offset + size;

            if free_start > self.region.size {
                return Err(SuballocationError::OutOfRegionMemory);
            }

            let new_state = free_start << 2 | allocation_type as u64;

            match self.state.compare_exchange_weak(
                state,
                new_state,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Ok(MemoryAlloc {
                        offset,
                        size,
                        allocation_type,
                        parent: AllocParent::Bump(self.clone()),
                        memory: self.region.memory,
                        memory_type_index: self.region.memory_type_index,
                        buffer_image_granularity: self.region.buffer_image_granularity,
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
        self.region.size - (self.state.load(Ordering::Relaxed) >> 2)
    }
}

/// Error that can be returned when resetting the [`BumpAllocator`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BumpAllocatorResetError;

impl Display for BumpAllocatorResetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the allocator is still in use")
    }
}

impl Error for BumpAllocatorResetError {}

fn align_up(val: DeviceSize, alignment: DeviceSize) -> DeviceSize {
    align_down(val + alignment - 1, alignment)
}

fn align_down(val: DeviceSize, alignment: DeviceSize) -> DeviceSize {
    debug_assert!(alignment.is_power_of_two());

    val & !(alignment - 1)
}

/// Checks if resouces A and B share a page.
///
/// > **Note**: Assumes `a_offset + a_size > 0` and `a_offset + a_size <= b_offset`.
fn are_blocks_on_same_page(
    a_offset: DeviceSize,
    a_size: DeviceSize,
    b_offset: DeviceSize,
    page_size: DeviceSize,
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
    /// downside is that the whole pool and possibly also the free-list must be copied when it runs
    /// out of memory. It is therefore best to start out with a safely large capacity.
    #[derive(Debug)]
    pub(super) struct PoolAllocator<T> {
        pool: Vec<T>,
        // LIFO list of free allocations, which means that newly freed allocations are always
        // reused first before bumping the free start.
        free_list: Vec<SlotId>,
    }

    impl<T> PoolAllocator<T> {
        pub fn new(capacity: usize) -> Self {
            debug_assert!(capacity > 0);

            let mut pool = Vec::new();
            let mut free_list = Vec::new();
            pool.reserve_exact(capacity);
            free_list.reserve_exact(capacity);
            // All IDs are free at the start.
            for index in (1..=capacity).rev() {
                free_list.push(SlotId(NonZeroUsize::new(index).unwrap()));
            }

            PoolAllocator { pool, free_list }
        }

        /// Allocates a slot and initializes it with the provided value. Returns the ID of the slot.
        pub fn allocate(&mut self, val: T) -> SlotId {
            let id = self.free_list.pop().unwrap_or_else(|| {
                // The free-list is empty, we need another pool.
                let new_len = self.pool.len() * 3 / 2;
                let additional = new_len - self.pool.len();
                self.pool.reserve_exact(additional);
                self.free_list.reserve_exact(additional);

                // Add the new IDs to the free-list.
                let len = self.pool.len();
                let cap = self.pool.capacity();
                for id in (len + 2..=cap).rev() {
                    // SAFETY: The `new_unchecked` is safe because:
                    // - `id` is bound to the range [len + 2, cap].
                    // - There is no way to add 2 to an unsigned integer (`len`) such that the
                    //   result is 0, except for an overflow, which is why rustc can't optimize this
                    //   out (unlike in the above loop where the range has a constant start).
                    // - `Vec::reserve_exact` panics if the new capacity exceeds `isize::MAX` bytes,
                    //   so the length of the pool can not be `usize::MAX - 1`.
                    let id = SlotId(unsafe { NonZeroUsize::new_unchecked(id) });
                    self.free_list.push(id);
                }

                // Smallest free ID.
                SlotId(NonZeroUsize::new(len + 1).unwrap())
            });

            if let Some(x) = self.pool.get_mut(id.0.get() - 1) {
                // We're reusing a slot, initialize it with the new value.
                *x = val;
            } else {
                // We're using a fresh slot. We always pick IDs in order into the free-list, so the
                //  next free ID must be for the slot right after the end of the occupied slots.
                debug_assert!(id.0.get() - 1 == self.pool.len());
                self.pool.push(val);
            }

            id
        }

        /// Returns the slot with the given ID to the allocator to be reused. The [`SlotId`] should
        /// not be used again afterward.
        pub fn free(&mut self, id: SlotId) {
            debug_assert!(!self.free_list.contains(&id));
            self.free_list.push(id);
        }

        /// Returns a mutable reference to the slot with the given ID.
        pub fn get_mut(&mut self, id: SlotId) -> &mut T {
            debug_assert!(!self.free_list.contains(&id));

            // SAFETY: This is safe because:
            // - The only way to obtain a `SlotId` is through `Self::allocate`.
            // - `Self::allocate` returns `SlotId`s in the range [1, self.pool.len()].
            // - `self.pool` only grows and never shrinks.
            unsafe { self.pool.get_unchecked_mut(id.0.get() - 1) }
        }
    }

    impl<T: Copy> PoolAllocator<T> {
        /// Returns a copy of the slot with the given ID.
        pub fn get(&self, id: SlotId) -> T {
            debug_assert!(!self.free_list.contains(&id));

            // SAFETY: Same as the `get_unchecked_mut` above.
            *unsafe { self.pool.get_unchecked(id.0.get() - 1) }
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
    use std::thread;

    const DUMMY_INFO: SuballocationCreateInfo = SuballocationCreateInfo {
        size: 1,
        alignment: 1,
        allocation_type: AllocationType::Unknown,
        _ne: crate::NonExhaustive(()),
    };

    const DUMMY_INFO_LINEAR: SuballocationCreateInfo = SuballocationCreateInfo {
        allocation_type: AllocationType::Linear,
        ..DUMMY_INFO
    };

    #[test]
    fn free_list_allocator_capacity() {
        const THREADS: DeviceSize = 12;
        const ALLOCATIONS_PER_THREAD: DeviceSize = 100;
        const ALLOCATION_STEP: DeviceSize = 117;
        const REGION_SIZE: DeviceSize =
            (ALLOCATION_STEP * (THREADS + 1) * THREADS / 2) * ALLOCATIONS_PER_THREAD;

        let allocator = FreeListAllocator::new(dummy_alloc!(REGION_SIZE));
        let allocs = ArrayQueue::new((ALLOCATIONS_PER_THREAD * THREADS) as usize);

        // Using threads to randomize allocation order.
        thread::scope(|scope| {
            for i in 1..=THREADS {
                let (allocator, allocs) = (&allocator, &allocs);

                scope.spawn(move || {
                    let size = i * ALLOCATION_STEP;

                    for _ in 0..ALLOCATIONS_PER_THREAD {
                        allocs
                            .push(
                                allocator
                                    .allocate(SuballocationCreateInfo { size, ..DUMMY_INFO })
                                    .unwrap(),
                            )
                            .unwrap();
                    }
                });
            }
        });

        assert!(allocator.allocate(DUMMY_INFO).is_err());
        assert!(allocator.free_size() == 0);

        drop(allocs);
        assert!(allocator.free_size() == REGION_SIZE);
        assert!(allocator
            .allocate(SuballocationCreateInfo {
                size: REGION_SIZE,
                ..DUMMY_INFO
            })
            .is_ok());
    }

    #[test]
    fn free_list_allocator_respects_alignment() {
        const INFO: SuballocationCreateInfo = SuballocationCreateInfo {
            alignment: 256,
            ..DUMMY_INFO
        };
        const REGION_SIZE: DeviceSize = 10 * INFO.alignment;

        let allocator = FreeListAllocator::new(dummy_alloc!(REGION_SIZE));
        let mut allocs = Vec::with_capacity(10);

        for _ in 0..10 {
            allocs.push(allocator.allocate(INFO).unwrap());
        }

        assert!(allocator.allocate(INFO).is_err());
        assert!(allocator.free_size() == REGION_SIZE - 10);
    }

    #[test]
    fn free_list_allocator_respects_granularity() {
        const GRANULARITY: DeviceSize = 16;
        const REGION_SIZE: DeviceSize = 2 * GRANULARITY;

        let allocator = FreeListAllocator::new(dummy_alloc!(REGION_SIZE, GRANULARITY));
        let mut linear_allocs = Vec::with_capacity(GRANULARITY as usize);
        let mut non_linear_allocs = Vec::with_capacity(GRANULARITY as usize);

        for i in 0..REGION_SIZE {
            if i % 2 == 0 {
                linear_allocs.push(
                    allocator
                        .allocate(SuballocationCreateInfo {
                            allocation_type: AllocationType::Linear,
                            ..DUMMY_INFO
                        })
                        .unwrap(),
                );
            } else {
                non_linear_allocs.push(
                    allocator
                        .allocate(SuballocationCreateInfo {
                            allocation_type: AllocationType::NonLinear,
                            ..DUMMY_INFO
                        })
                        .unwrap(),
                );
            }
        }

        assert!(allocator.allocate(DUMMY_INFO_LINEAR).is_err());
        assert!(allocator.free_size() == 0);

        drop(linear_allocs);
        assert!(allocator
            .allocate(SuballocationCreateInfo {
                size: GRANULARITY,
                ..DUMMY_INFO
            })
            .is_ok());

        let _alloc = allocator.allocate(DUMMY_INFO).unwrap();
        assert!(allocator.allocate(DUMMY_INFO).is_err());
        assert!(allocator.allocate(DUMMY_INFO_LINEAR).is_err());
    }

    #[test]
    fn pool_allocator_capacity() {
        const BLOCK_SIZE: DeviceSize = 1024;

        type Allocator = PoolAllocator<BLOCK_SIZE>;

        assert_should_panic!({ Allocator::new(dummy_alloc!(BLOCK_SIZE - 1)) });

        let allocator = Allocator::new(dummy_alloc!(2 * BLOCK_SIZE - 1));
        {
            let alloc = allocator.allocate(DUMMY_INFO).unwrap();
            assert!(allocator.allocate(DUMMY_INFO).is_err());

            drop(alloc);
            let _alloc = allocator.allocate(DUMMY_INFO).unwrap();
        }

        let allocator = Allocator::new(dummy_alloc!(2 * BLOCK_SIZE));
        {
            let alloc1 = allocator.allocate(DUMMY_INFO).unwrap();
            let alloc2 = allocator.allocate(DUMMY_INFO).unwrap();
            assert!(allocator.allocate(DUMMY_INFO).is_err());

            drop(alloc1);
            let alloc1 = allocator.allocate(DUMMY_INFO).unwrap();
            assert!(allocator.allocate(DUMMY_INFO).is_err());

            drop(alloc1);
            drop(alloc2);
            let _alloc1 = allocator.allocate(DUMMY_INFO).unwrap();
            let _alloc2 = allocator.allocate(DUMMY_INFO).unwrap();
        }
    }

    #[test]
    fn pool_allocator_respects_alignment() {
        const BLOCK_SIZE: DeviceSize = 1024 + 128;
        const INFO_A: SuballocationCreateInfo = SuballocationCreateInfo {
            size: BLOCK_SIZE,
            alignment: 256,
            ..DUMMY_INFO
        };
        const INFO_B: SuballocationCreateInfo = SuballocationCreateInfo {
            size: 1024,
            ..INFO_A
        };

        let allocator = PoolAllocator::<BLOCK_SIZE>::new(dummy_alloc!(10 * BLOCK_SIZE));

        // This uses the fact that block indices are inserted into the free-list in order, so
        // the first allocation succeeds because the block has an even index, while the second
        // has an odd index.
        allocator.allocate(INFO_A).unwrap();
        assert!(allocator.allocate(INFO_A).is_err());
        allocator.allocate(INFO_A).unwrap();
        assert!(allocator.allocate(INFO_A).is_err());

        for _ in 0..10 {
            allocator.allocate(INFO_B).unwrap();
        }
    }

    #[test]
    fn pool_allocator_respects_granularity() {
        type Allocator = PoolAllocator<128>;

        let allocator = Allocator::new(dummy_alloc!(1024, 256, AllocationType::Unknown));
        assert!(allocator.block_count() == 4);

        let allocator = Allocator::new(dummy_alloc!(1024, 256, AllocationType::Linear));
        assert!(allocator.block_count() == 8);

        let allocator = Allocator::new(dummy_alloc!(1024, 256, AllocationType::NonLinear));
        assert!(allocator.block_count() == 8);
    }

    #[test]
    fn buddy_allocator_capacity() {
        const MAX_ORDER: usize = 10;
        const REGION_SIZE: DeviceSize = BuddyAllocator::MIN_NODE_SIZE << MAX_ORDER;

        let allocator = BuddyAllocator::new(dummy_alloc!(REGION_SIZE));
        assert!(allocator.order_count() == MAX_ORDER + 1);
        let mut allocs = Vec::with_capacity(1 << MAX_ORDER);

        for order in 0..=MAX_ORDER {
            let size = BuddyAllocator::MIN_NODE_SIZE << order;

            for _ in 0..1 << (MAX_ORDER - order) {
                allocs.push(
                    allocator
                        .allocate(SuballocationCreateInfo { size, ..DUMMY_INFO })
                        .unwrap(),
                );
            }

            assert!(allocator.allocate(DUMMY_INFO).is_err());
            assert!(allocator.free_size() == 0);
            allocs.clear();
        }

        let mut orders = (0..MAX_ORDER).collect::<Vec<_>>();

        for mid in 0..MAX_ORDER {
            orders.rotate_left(mid);

            for &order in &orders {
                let size = BuddyAllocator::MIN_NODE_SIZE << order;
                allocs.push(
                    allocator
                        .allocate(SuballocationCreateInfo { size, ..DUMMY_INFO })
                        .unwrap(),
                );
            }

            let _alloc = allocator.allocate(DUMMY_INFO).unwrap();
            assert!(allocator.allocate(DUMMY_INFO).is_err());
            assert!(allocator.free_size() == 0);
            allocs.clear();
        }
    }

    #[test]
    fn buddy_allocator_respects_alignment() {
        const REGION_SIZE: DeviceSize = 4096;

        let allocator = BuddyAllocator::new(dummy_alloc!(REGION_SIZE));

        {
            const INFO: SuballocationCreateInfo = SuballocationCreateInfo {
                alignment: 4096,
                ..DUMMY_INFO
            };

            let _alloc = allocator.allocate(INFO).unwrap();
            assert!(allocator.allocate(INFO).is_err());
            assert!(allocator.free_size() == REGION_SIZE - BuddyAllocator::MIN_NODE_SIZE);
        }

        {
            const INFO_A: SuballocationCreateInfo = SuballocationCreateInfo {
                alignment: 256,
                ..DUMMY_INFO
            };
            const ALLOCATIONS_A: DeviceSize = REGION_SIZE / INFO_A.alignment;
            const INFO_B: SuballocationCreateInfo = SuballocationCreateInfo {
                alignment: 16,
                ..DUMMY_INFO
            };
            const ALLOCATIONS_B: DeviceSize = REGION_SIZE / INFO_B.alignment - ALLOCATIONS_A;

            let mut allocs =
                Vec::with_capacity((REGION_SIZE / BuddyAllocator::MIN_NODE_SIZE) as usize);

            for _ in 0..ALLOCATIONS_A {
                allocs.push(allocator.allocate(INFO_A).unwrap());
            }

            assert!(allocator.allocate(INFO_A).is_err());
            assert!(
                allocator.free_size()
                    == REGION_SIZE - ALLOCATIONS_A * BuddyAllocator::MIN_NODE_SIZE
            );

            for _ in 0..ALLOCATIONS_B {
                allocs.push(allocator.allocate(INFO_B).unwrap());
            }

            assert!(allocator.allocate(DUMMY_INFO).is_err());
            assert!(allocator.free_size() == 0);
        }
    }

    #[test]
    fn buddy_allocator_respects_granularity() {
        const GRANULARITY: DeviceSize = 256;
        const REGION_SIZE: DeviceSize = 2 * GRANULARITY;

        let allocator = BuddyAllocator::new(dummy_alloc!(REGION_SIZE, GRANULARITY));

        {
            const ALLOCATIONS: DeviceSize = REGION_SIZE / BuddyAllocator::MIN_NODE_SIZE;

            let mut allocs = Vec::with_capacity(ALLOCATIONS as usize);
            for _ in 0..ALLOCATIONS {
                allocs.push(allocator.allocate(DUMMY_INFO_LINEAR).unwrap());
            }

            assert!(allocator.allocate(DUMMY_INFO_LINEAR).is_err());
            assert!(allocator.free_size() == 0);
        }

        {
            let _alloc1 = allocator.allocate(DUMMY_INFO).unwrap();
            let _alloc2 = allocator.allocate(DUMMY_INFO).unwrap();
            assert!(allocator.allocate(DUMMY_INFO).is_err());
            assert!(allocator.free_size() == 0);
        }
    }

    #[test]
    fn bump_allocator_respects_alignment() {
        const INFO: SuballocationCreateInfo = SuballocationCreateInfo {
            alignment: 16,
            ..DUMMY_INFO
        };

        let allocator = BumpAllocator::new(dummy_alloc!(INFO.alignment * 10));

        for _ in 0..10 {
            allocator.allocate(INFO).unwrap();
        }

        assert!(allocator.allocate(INFO).is_err());

        for _ in 0..INFO.alignment - 1 {
            allocator.allocate(DUMMY_INFO).unwrap();
        }

        assert!(allocator.allocate(INFO).is_err());
        assert!(allocator.free_size() == 0);
    }

    #[test]
    fn bump_allocator_respects_granularity() {
        const ALLOCATIONS: DeviceSize = 10;
        const GRANULARITY: DeviceSize = 1024;

        let mut allocator =
            BumpAllocator::new(dummy_alloc!(GRANULARITY * ALLOCATIONS, GRANULARITY));

        for i in 0..ALLOCATIONS {
            for _ in 0..GRANULARITY {
                allocator
                    .allocate(SuballocationCreateInfo {
                        allocation_type: if i % 2 == 0 {
                            AllocationType::NonLinear
                        } else {
                            AllocationType::Linear
                        },
                        ..DUMMY_INFO
                    })
                    .unwrap();
            }
        }

        assert!(allocator.allocate(DUMMY_INFO_LINEAR).is_err());
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
                    ..DUMMY_INFO
                })
                .unwrap();
        }

        assert!(allocator.allocate(DUMMY_INFO_LINEAR).is_err());
        assert!(allocator.free_size() == GRANULARITY - 1);
    }

    #[test]
    fn bump_allocator_syncness() {
        const THREADS: DeviceSize = 12;
        const ALLOCATIONS_PER_THREAD: DeviceSize = 100_000;
        const ALLOCATION_STEP: DeviceSize = 117;
        const REGION_SIZE: DeviceSize =
            (ALLOCATION_STEP * (THREADS + 1) * THREADS / 2) * ALLOCATIONS_PER_THREAD;

        let mut allocator = BumpAllocator::new(dummy_alloc!(REGION_SIZE));

        thread::scope(|scope| {
            for i in 1..=THREADS {
                let allocator = &allocator;

                scope.spawn(move || {
                    let size = i * ALLOCATION_STEP;

                    for _ in 0..ALLOCATIONS_PER_THREAD {
                        allocator
                            .allocate(SuballocationCreateInfo { size, ..DUMMY_INFO })
                            .unwrap();
                    }
                });
            }
        });

        assert!(allocator.allocate(DUMMY_INFO).is_err());
        assert!(allocator.free_size() == 0);

        allocator.try_reset().unwrap();
        assert!(allocator.free_size() == REGION_SIZE);
    }

    macro_rules! dummy_alloc {
        ($size:expr) => {
            dummy_alloc!($size, 1)
        };
        ($size:expr, $granularity:expr) => {
            dummy_alloc!($size, $granularity, AllocationType::Unknown)
        };
        ($size:expr, $granularity:expr, $type:expr) => {
            MemoryAlloc {
                offset: 0,
                size: $size,
                allocation_type: $type,
                parent: AllocParent::None,
                memory: ash::vk::DeviceMemory::null(),
                memory_type_index: 0,
                buffer_image_granularity: $granularity,
            }
        };
    }

    pub(self) use dummy_alloc;
}
