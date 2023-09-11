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
//! reduce the burden on the user as much as possible. You have a toolbox of [suballocators] to
//! choose from that cover all allocation algorithms, which you can compose into any kind of
//! [hierarchy] you wish. This way you have maximum flexibility while still only using a few
//! `DeviceMemory` blocks and not writing any of the very error-prone code.
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
//! [`mem::forget`]: std::mem::forget
//! [region]: Suballocator#regions

mod layout;
pub mod suballocator;

use self::array_vec::ArrayVec;
pub use self::{
    layout::DeviceLayout,
    suballocator::{
        AllocationType, BuddyAllocator, BumpAllocator, FreeListAllocator, Suballocation,
        Suballocator, SuballocatorError,
    },
};
use super::{
    DedicatedAllocation, DeviceAlignment, DeviceMemory, ExternalMemoryHandleTypes,
    MemoryAllocateFlags, MemoryAllocateInfo, MemoryMapInfo, MemoryProperties, MemoryPropertyFlags,
    MemoryRequirements, MemoryType,
};
use crate::{
    device::{Device, DeviceOwned},
    instance::InstanceOwnedDebugWrapper,
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version,
    VulkanError,
};
use ash::vk::MAX_MEMORY_TYPES;
use parking_lot::Mutex;
use std::{
    error::Error,
    fmt::{Debug, Display, Error as FmtError, Formatter},
    ops::BitOr,
    ptr,
    sync::Arc,
};

/// General-purpose memory allocators which allocate from any memory type dynamically as needed.
///
/// # Safety
///
/// TODO
pub unsafe trait MemoryAllocator: DeviceOwned + Send + Sync + 'static {
    /// Finds the most suitable memory type index in `memory_type_bits` using the given `filter`.
    /// Returns [`None`] if the requirements are too strict and no memory type is able to satisfy
    /// them.
    fn find_memory_type_index(
        &self,
        memory_type_bits: u32,
        filter: MemoryTypeFilter,
    ) -> Option<u32>;

    /// Allocates memory from a specific memory type.
    ///
    /// # Arguments
    ///
    /// - `memory_type_index` - The index of the memory type to allocate from.
    ///
    /// - `layout` - The layout of the allocation.
    ///
    /// - `allocation_type` - The type of resources that can be bound to the allocation.
    ///
    /// - `never_allocate` - If `true` then the allocator should never allocate `DeviceMemory`,
    ///   instead only suballocate from existing blocks.
    fn allocate_from_type(
        &self,
        memory_type_index: u32,
        layout: DeviceLayout,
        allocation_type: AllocationType,
        never_allocate: bool,
    ) -> Result<MemoryAlloc, MemoryAllocatorError>;

    /// Allocates memory according to requirements.
    ///
    /// # Arguments
    ///
    /// - `requirements` - Requirements of the resource you want to allocate memory for.
    ///
    ///   If you plan to bind this memory directly to a non-sparse resource, then this must
    ///   correspond to the value returned by either [`RawBuffer::memory_requirements`] or
    ///   [`RawImage::memory_requirements`] for the respective buffer or image.
    ///
    /// - `allocation_type` - What type of resource this allocation will be used for.
    ///
    ///   This should be [`Linear`] for buffers and linear images, and [`NonLinear`] for optimal
    ///   images. You can not bind memory allocated with the [`Linear`] type to optimal images or
    ///   bind memory allocated with the [`NonLinear`] type to buffers and linear images. You
    ///   should never use the [`Unknown`] type unless you have to, as that can be less memory
    ///   efficient.
    ///
    /// - `dedicated_allocation` - Allows a dedicated allocation to be created.
    ///
    ///   You should always fill this field in if you are allocating memory for a non-sparse
    ///   resource, otherwise the allocator won't be able to create a dedicated allocation if one
    ///   is required or recommended.
    ///
    ///   This argument is silently ignored (treated as `None`) if the device API version is below
    ///   1.1 and the [`khr_dedicated_allocation`] extension is not enabled on the device.
    ///
    /// [`RawBuffer::memory_requirements`]: crate::buffer::sys::RawBuffer::memory_requirements
    /// [`RawImage::memory_requirements`]: crate::image::sys::RawImage::memory_requirements
    /// [`Linear`]: AllocationType::Linear
    /// [`NonLinear`]: AllocationType::NonLinear
    /// [`Unknown`]: AllocationType::Unknown
    /// [`khr_dedicated_allocation`]: crate::device::DeviceExtensions::khr_dedicated_allocation
    fn allocate(
        &self,
        requirements: MemoryRequirements,
        allocation_type: AllocationType,
        create_info: AllocationCreateInfo,
        dedicated_allocation: Option<DedicatedAllocation<'_>>,
    ) -> Result<MemoryAlloc, MemoryAllocatorError>;

    /// Creates an allocation with a whole device memory block dedicated to it.
    fn allocate_dedicated(
        &self,
        memory_type_index: u32,
        allocation_size: DeviceSize,
        dedicated_allocation: Option<DedicatedAllocation<'_>>,
        export_handle_types: ExternalMemoryHandleTypes,
    ) -> Result<MemoryAlloc, MemoryAllocatorError>;

    /// Deallocates the given `allocation`.
    ///
    /// # Safety
    ///
    /// - `allocation` must refer to a **currently allocated** allocation of `self`.
    unsafe fn deallocate(&self, allocation: MemoryAlloc);
}

impl Debug for dyn MemoryAllocator {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        f.debug_struct("MemoryAllocator").finish_non_exhaustive()
    }
}

/// Describes what memory property flags are required, preferred and not preferred when picking a
/// memory type index.
///
/// # Examples
///
/// For resources that the device frequently accesses, e.g. textures, render targets, or
/// intermediary buffers, you want device-local memory without any host access:
///
/// ```
/// # use vulkano::{
/// #     image::{Image, ImageCreateInfo, ImageUsage},
/// #     memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
/// # };
/// #
/// # let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = return;
/// # let format = return;
/// # let extent = return;
/// #
/// let texture = Image::new(
///     memory_allocator.clone(),
///     ImageCreateInfo {
///         format,
///         extent,
///         usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
///         ..Default::default()
///     },
///     AllocationCreateInfo {
///         memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
///         ..Default::default()
///     },
/// )
/// .unwrap();
/// ```
///
/// For staging, the resource is only ever written to sequentially. Also, since the device will
/// only read the staging resourse once, it would yield no benefit to place it in device-local
/// memory, in fact it would be wasteful. Therefore, it's best to put it in host-local memory:
///
/// ```
/// # use vulkano::{
/// #     buffer::{Buffer, BufferCreateInfo, BufferUsage},
/// #     memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
/// # };
/// #
/// # let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = return;
/// #
/// let staging_buffer = Buffer::new_sized(
///     memory_allocator.clone(),
///     BufferCreateInfo {
///         usage: BufferUsage::TRANSFER_SRC,
///         ..Default::default()
///     },
///     AllocationCreateInfo {
///         memory_type_filter: MemoryTypeFilter::PREFER_HOST |
///             MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
///         ..Default::default()
///     },
/// )
/// .unwrap();
/// #
/// # let staging_buffer: vulkano::buffer::Subbuffer<u32> = staging_buffer;
/// ```
///
/// For resources that the device accesses directly, aka a buffer/image used for anything other
/// than transfers, it's probably best to put it in device-local memory:
///
/// ```
/// # use vulkano::{
/// #     buffer::{Buffer, BufferCreateInfo, BufferUsage},
/// #     memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
/// # };
/// #
/// # let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = return;
/// #
/// let uniform_buffer = Buffer::new_sized(
///     memory_allocator.clone(),
///     BufferCreateInfo {
///         usage: BufferUsage::UNIFORM_BUFFER,
///         ..Default::default()
///     },
///     AllocationCreateInfo {
///         memory_type_filter: MemoryTypeFilter::PREFER_DEVICE |
///             MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
///         ..Default::default()
///     },
/// )
/// .unwrap();
/// #
/// # let uniform_buffer: vulkano::buffer::Subbuffer<u32> = uniform_buffer;
/// ```
///
/// For readback, e.g. getting the results of a compute shader back to the host:
///
/// ```
/// # use vulkano::{
/// #     buffer::{Buffer, BufferCreateInfo, BufferUsage},
/// #     memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
/// # };
/// #
/// # let memory_allocator: std::sync::Arc<vulkano::memory::allocator::StandardMemoryAllocator> = return;
/// #
/// let readback_buffer = Buffer::new_sized(
///     memory_allocator.clone(),
///     BufferCreateInfo {
///         usage: BufferUsage::TRANSFER_DST,
///         ..Default::default()
///     },
///     AllocationCreateInfo {
///         memory_type_filter: MemoryTypeFilter::PREFER_HOST |
///             MemoryTypeFilter::HOST_RANDOM_ACCESS,
///         ..Default::default()
///     },
/// )
/// .unwrap();
/// #
/// # let readback_buffer: vulkano::buffer::Subbuffer<u32> = readback_buffer;
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryTypeFilter {
    pub required_flags: MemoryPropertyFlags,
    pub preferred_flags: MemoryPropertyFlags,
    pub not_preferred_flags: MemoryPropertyFlags,
}

impl MemoryTypeFilter {
    /// Prefers picking a memory type with the [`DEVICE_LOCAL`] flag.
    ///
    /// Memory being device-local means that it is fastest to access for the device. However,
    /// for dedicated GPUs, getting memory in and out of VRAM requires to go through the PCIe bus,
    /// which is very slow and should therefore only be done when necessary.
    ///
    /// This filter is best suited for anything that the host doesn't access, but the device
    /// accesses frequently. For example textures, render targets, and intermediary buffers.
    ///
    /// For memory that the host does access but less frequently than the device, such as updating
    /// a uniform buffer each frame, device-local memory may also be preferred. In this case,
    /// because the memory is only written once before being consumed by the device and becoming
    /// outdated, it doesn't matter that the data is potentially transferred over the PCIe bus
    /// since it only happens once. Since this is only a preference, if you have some requirements
    /// such as the memory being [`HOST_VISIBLE`], those requirements will take precendence.
    ///
    /// For memory that the host doesn't access, and the device doesn't access directly, you may
    /// still prefer device-local memory if the memory is used regularly. For instance, an image
    /// that is swapped each frame.
    ///
    /// Keep in mind that for implementations with unified memory, there's only system RAM. That
    /// means that even if the implementation reports a memory type that is not `HOST_VISIBLE` and
    /// is `DEVICE_LOCAL`, its still the same unified memory as all other memory. However, it may
    /// still be faster to access. On the other hand, some such implementations don't report any
    /// memory types that are not `HOST_VISIBLE`, which makes sense since it's all system RAM. In
    /// that case you wouldn't need to do staging.
    ///
    /// Don't use this together with [`PREFER_HOST`], that makes no sense. If you need host access,
    /// make sure you combine this with either the [`HOST_SEQUENTIAL_WRITE`] or
    /// [`HOST_RANDOM_ACCESS`] filter.
    ///
    /// [`DEVICE_LOCAL`]: MemoryPropertyFlags::DEVICE_LOCAL
    /// [`HOST_VISIBLE`]: MemoryPropertyFlags::HOST_VISIBLE
    /// [`PREFER_HOST`]: Self::PREFER_HOST
    /// [`HOST_SEQUENTIAL_WRITE`]: Self::HOST_SEQUENTIAL_WRITE
    /// [`HOST_RANDOM_ACCESS`]: Self::HOST_RANDOM_ACCESS
    pub const PREFER_DEVICE: Self = Self {
        required_flags: MemoryPropertyFlags::empty(),
        preferred_flags: MemoryPropertyFlags::DEVICE_LOCAL,
        not_preferred_flags: MemoryPropertyFlags::empty(),
    };

    /// Prefers picking a memory type without the [`DEVICE_LOCAL`] flag.
    ///
    /// This option is best suited for resources that the host does access, but device doesn't
    /// access directly, such as staging buffers and readback buffers.
    ///
    /// For memory that the host does access but less frequently than the device, such as updating
    /// a uniform buffer each frame, you may still get away with using host-local memory if the
    /// updates are small and local enough. In such cases, the memory should be able to be quickly
    /// cached by the device, such that the data potentially being transferred over the PCIe bus
    /// wouldn't matter.
    ///
    /// For memory that the host doesn't access, and the device doesn't access directly, you may
    /// still prefer host-local memory if the memory is rarely used, such as for manually paging
    /// parts of device-local memory out in order to free up space on the device.
    ///
    /// Don't use this together with [`PREFER_DEVICE`], that makes no sense. If you need host
    /// access, make sure you combine this with either the [`HOST_SEQUENTIAL_WRITE`] or
    /// [`HOST_RANDOM_ACCESS`] filter.
    ///
    /// [`DEVICE_LOCAL`]: MemoryPropertyFlags::DEVICE_LOCAL
    /// [`PREFER_DEVICE`]: Self::PREFER_DEVICE
    /// [`HOST_SEQUENTIAL_WRITE`]: Self::HOST_SEQUENTIAL_WRITE
    /// [`HOST_RANDOM_ACCESS`]: Self::HOST_RANDOM_ACCESS
    pub const PREFER_HOST: Self = Self {
        required_flags: MemoryPropertyFlags::empty(),
        preferred_flags: MemoryPropertyFlags::empty(),
        not_preferred_flags: MemoryPropertyFlags::DEVICE_LOCAL,
    };

    /// This guarantees picking a memory type that has the [`HOST_VISIBLE`] flag. Using this filter
    /// allows the allocator to pick a memory type that is uncached and write-combined, which is
    /// ideal for sequential writes. However, this optimization might lead to poor performance for
    /// anything else. What counts as a sequential write is any kind of loop that writes memory
    /// locations in order, such as iterating over a slice while writing each element in order, or
    /// equivalently using [`slice::copy_from_slice`]. Copying sized data also counts, as rustc
    /// should write the memory locations in order. If you have a struct, make sure you write it
    /// member-by-member.
    ///
    /// Example use cases include staging buffers, as well as any other kind of buffer that you
    /// only write to from the host, like a uniform or vertex buffer.
    ///
    /// Don't use this together with [`HOST_RANDOM_ACCESS`], that makes no sense. If you do both a
    /// sequential write and read or random access, then you should use `HOST_RANDOM_ACCESS`
    /// instead. However, you could also consider using different allocations for the two purposes
    /// to get the most performance out, if that's possible.
    ///
    /// [`HOST_VISIBLE`]: MemoryPropertyFlags::HOST_VISIBLE
    /// [`HOST_COHERENT`]: MemoryPropertyFlags::HOST_COHERENT
    /// [`HOST_RANDOM_ACCESS`]: Self::HOST_RANDOM_ACCESS
    pub const HOST_SEQUENTIAL_WRITE: Self = Self {
        required_flags: MemoryPropertyFlags::HOST_VISIBLE,
        preferred_flags: MemoryPropertyFlags::empty(),
        not_preferred_flags: MemoryPropertyFlags::HOST_CACHED,
    };

    /// This guarantees picking a memory type that has the [`HOST_VISIBLE`] and [`HOST_CACHED`]
    /// flags, which is best suited for readback and/or random access.
    ///
    /// Example use cases include using the device for things other than rendering and getting the
    /// results back to the host. That might be compute shading, or image or video manipulation, or
    /// screenshotting.
    ///
    /// Don't use this together with [`HOST_SEQUENTIAL_WRITE`], that makes no sense. If you are
    /// sure you only ever need to sequentially write to the allocation, then using
    /// `HOST_SEQUENTIAL_WRITE` instead will yield better performance.
    ///
    /// [`HOST_VISIBLE`]: MemoryPropertyFlags::HOST_VISIBLE
    /// [`HOST_CACHED`]: MemoryPropertyFlags::HOST_CACHED
    /// [`HOST_SEQUENTIAL_WRITE`]: Self::HOST_SEQUENTIAL_WRITE
    pub const HOST_RANDOM_ACCESS: Self = Self {
        required_flags: MemoryPropertyFlags::HOST_VISIBLE.union(MemoryPropertyFlags::HOST_CACHED),
        preferred_flags: MemoryPropertyFlags::empty(),
        not_preferred_flags: MemoryPropertyFlags::empty(),
    };

    /// Returns a `MemoryTypeFilter` with none of the flags set.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            required_flags: MemoryPropertyFlags::empty(),
            preferred_flags: MemoryPropertyFlags::empty(),
            not_preferred_flags: MemoryPropertyFlags::empty(),
        }
    }

    /// Returns the union of `self` and `other`.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self {
            required_flags: self.required_flags.union(other.required_flags),
            preferred_flags: self.preferred_flags.union(other.preferred_flags),
            not_preferred_flags: self.not_preferred_flags.union(other.not_preferred_flags),
        }
    }
}

impl BitOr for MemoryTypeFilter {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

/// Parameters to create a new [allocation] using a [memory allocator].
///
/// [allocation]: MemoryAlloc
/// [memory allocator]: MemoryAllocator
#[derive(Clone, Debug)]
pub struct AllocationCreateInfo {
    /// Filter used to narrow down the memory type to be selected.
    ///
    /// The default value is [`MemoryTypeFilter::PREFER_DEVICE`].
    pub memory_type_filter: MemoryTypeFilter,

    /// Allows you to further constrain the possible choices of memory types, by only allowing the
    /// memory type indices that have a corresponding bit at the same index set to 1.
    ///
    /// The default value is [`u32::MAX`].
    pub memory_type_bits: u32,

    /// How eager the allocator should be to allocate [`DeviceMemory`].
    ///
    /// The default value is [`MemoryAllocatePreference::Unknown`].
    pub allocate_preference: MemoryAllocatePreference,

    pub _ne: crate::NonExhaustive,
}

impl Default for AllocationCreateInfo {
    #[inline]
    fn default() -> Self {
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            memory_type_bits: u32::MAX,
            allocate_preference: MemoryAllocatePreference::Unknown,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Describes whether allocating [`DeviceMemory`] is desired.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryAllocatePreference {
    /// There is no known preference, let the allocator decide.
    Unknown,

    /// The allocator should never allocate `DeviceMemory` and should instead only suballocate from
    /// existing blocks.
    ///
    /// This option is best suited if you can not afford the overhead of allocating `DeviceMemory`.
    NeverAllocate,

    /// The allocator should always allocate `DeviceMemory`.
    ///
    /// This option is best suited if you are allocating a long-lived resource that you know could
    /// benefit from having a dedicated allocation.
    AlwaysAllocate,
}

/// An allocation made using a [memory allocator].
///
/// [memory allocator]: MemoryAllocator
#[derive(Clone, Debug)]
pub struct MemoryAlloc {
    /// The underlying block of device memory.
    pub device_memory: Arc<DeviceMemory>,

    /// The suballocation within the device memory block, or [`None`] if this is a dedicated
    /// allocation.
    pub suballocation: Option<Suballocation>,

    /// The type of resources that can be bound to this memory block. This will be exactly equal to
    /// the requested allocation type.
    ///
    /// For dedicated allocations it doesn't matter what this is, as there aren't going to be any
    /// neighboring suballocations. Therefore the allocator implementation is advised to always set
    /// this to [`AllocationType::Unknown`] in that case for maximum flexibility.
    pub allocation_type: AllocationType,

    /// An opaque handle identifying the allocation inside the allocator.
    pub allocation_handle: AllocationHandle,
}

/// An opaque handle identifying an allocation inside an allocator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct AllocationHandle(pub *mut ());

unsafe impl Send for AllocationHandle {}
unsafe impl Sync for AllocationHandle {}

/// Error that can be returned when creating an [allocation] using a [memory allocator].
///
/// [allocation]: MemoryAlloc
/// [memory allocator]: MemoryAllocator
#[derive(Clone, Debug)]
pub enum MemoryAllocatorError {
    /// Allocating [`DeviceMemory`] failed.
    AllocateDeviceMemory(Validated<VulkanError>),

    /// Finding a suitable memory type failed.
    ///
    /// This is returned from [`MemoryAllocator::allocate`] when
    /// [`MemoryAllocator::find_memory_type_index`] returns [`None`].
    FindMemoryType,

    /// There is not enough memory in the pool.
    ///
    /// This is returned when using [`MemoryAllocatePreference::NeverAllocate`] and there is not
    /// enough memory in the pool.
    OutOfPoolMemory,

    /// A dedicated allocation is required but was explicitly forbidden.
    ///
    /// This is returned when using [`MemoryAllocatePreference::NeverAllocate`] and the
    /// implementation requires a dedicated allocation.
    DedicatedAllocationRequired,

    /// The block size for the allocator was exceeded.
    ///
    /// This is returned when using [`MemoryAllocatePreference::NeverAllocate`] and the allocation
    /// size exceeded the block size for all heaps of suitable memory types.
    BlockSizeExceeded,
}

impl Error for MemoryAllocatorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::AllocateDeviceMemory(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for MemoryAllocatorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        let msg = match self {
            Self::AllocateDeviceMemory(_) => "allocating device memory failed",
            Self::FindMemoryType => "finding a suitable memory type failed",
            Self::OutOfPoolMemory => "the pool doesn't have enough free space",
            Self::DedicatedAllocationRequired => {
                "a dedicated allocation is required but was explicitly forbidden"
            }
            Self::BlockSizeExceeded => {
                "the allocation size was greater than the block size for all heaps of suitable \
                memory types and dedicated allocations were explicitly forbidden"
            }
        };

        f.write_str(msg)
    }
}

/// Standard memory allocator intended as a global and general-purpose allocator.
///
/// This type of allocator is what you should always use, unless you know, for a fact, that it is
/// not suited to the task.
///
/// See also [`GenericMemoryAllocator`] for details about the allocation algorithm, and
/// [`FreeListAllocator`] for details about the suballocation algorithm.
pub type StandardMemoryAllocator = GenericMemoryAllocator<FreeListAllocator>;

impl StandardMemoryAllocator {
    /// Creates a new `StandardMemoryAllocator` with default configuration.
    pub fn new_default(device: Arc<Device>) -> Self {
        let MemoryProperties {
            memory_types,
            memory_heaps,
        } = device.physical_device().memory_properties();

        let mut block_sizes = [0; MAX_MEMORY_TYPES];
        let mut memory_type_bits = u32::MAX;

        for (index, memory_type) in memory_types.iter().enumerate() {
            const LARGE_HEAP_THRESHOLD: DeviceSize = 1024 * 1024 * 1024;

            let heap_size = memory_heaps[memory_type.heap_index as usize].size;

            block_sizes[index] = if heap_size >= LARGE_HEAP_THRESHOLD {
                256 * 1024 * 1024
            } else {
                64 * 1024 * 1024
            };

            if memory_type.property_flags.intersects(
                MemoryPropertyFlags::LAZILY_ALLOCATED
                    | MemoryPropertyFlags::PROTECTED
                    | MemoryPropertyFlags::DEVICE_COHERENT
                    | MemoryPropertyFlags::RDMA_CAPABLE,
            ) {
                // VUID-VkMemoryAllocateInfo-memoryTypeIndex-01872
                // VUID-vkAllocateMemory-deviceCoherentMemory-02790
                // Lazily allocated memory would just cause problems for suballocation in general.
                memory_type_bits &= !(1 << index);
            }
        }

        let create_info = GenericMemoryAllocatorCreateInfo {
            block_sizes: &block_sizes,
            memory_type_bits,
            ..Default::default()
        };

        unsafe { Self::new_unchecked(device, create_info) }
    }
}

/// A generic implementation of a [memory allocator].
///
/// The allocator keeps a pool of [`DeviceMemory`] blocks for each memory type and uses the type
/// parameter `S` to [suballocate] these blocks. You can also configure the sizes of these blocks.
/// This means that you can have as many `GenericMemoryAllocator`s as you you want for different
/// needs, or for performance reasons, as long as the block sizes are configured properly so that
/// too much memory isn't wasted.
///
/// See also [the `MemoryAllocator` implementation].
///
/// # Mapping behavior
///
/// Every time a new `DeviceMemory` block is allocated, it is mapped in full automatically as long
/// as it resides in host-visible memory. It remains mapped until it is dropped, which only happens
/// if the allocator is dropped. In other words, all eligible blocks are persistently mapped, so
/// you don't need to worry about whether or not your host-visible allocations are host-accessible.
///
/// # `DeviceMemory` allocation
///
/// If an allocation is created with the [`MemoryAllocatePreference::Unknown`] option, and the
/// allocator deems the allocation too big for suballocation (larger than half the block size), or
/// the implementation prefers or requires a dedicated allocation, then that allocation is made a
/// dedicated allocation. Using [`MemoryAllocatePreference::NeverAllocate`], a dedicated allocation
/// is never created, even if the allocation is larger than the block size or a dedicated
/// allocation is required. In such a case an error is returned instead. Using
/// [`MemoryAllocatePreference::AlwaysAllocate`], a dedicated allocation is always created.
///
/// In all other cases, `DeviceMemory` is only allocated if a pool runs out of memory and needs
/// another block. No `DeviceMemory` is allocated when the allocator is created, the blocks are
/// only allocated once they are needed.
///
/// [memory allocator]: MemoryAllocator
/// [suballocate]: Suballocator
/// [the `MemoryAllocator` implementation]: Self#impl-MemoryAllocator-for-GenericMemoryAllocator<S>
#[derive(Debug)]
pub struct GenericMemoryAllocator<S> {
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    buffer_image_granularity: DeviceAlignment,
    // Each memory type has a pool of `DeviceMemory` blocks.
    pools: ArrayVec<Pool<S>, MAX_MEMORY_TYPES>,
    // Global mask of memory types.
    memory_type_bits: u32,
    dedicated_allocation: bool,
    export_handle_types: ArrayVec<ExternalMemoryHandleTypes, MAX_MEMORY_TYPES>,
    flags: MemoryAllocateFlags,
    // How many `DeviceMemory` allocations should be allowed before restricting them.
    max_allocations: u32,
}

#[derive(Debug)]
struct Pool<S> {
    blocks: Mutex<Vec<Box<Block<S>>>>,
    // This is cached here for faster access, so we don't need to hop through 3 pointers.
    property_flags: MemoryPropertyFlags,
    atom_size: DeviceAlignment,
    block_size: DeviceSize,
}

impl<S> GenericMemoryAllocator<S> {
    // This is a false-positive, we only use this const for static initialization.
    #[allow(clippy::declare_interior_mutable_const)]
    const EMPTY_POOL: Pool<S> = Pool {
        blocks: Mutex::new(Vec::new()),
        property_flags: MemoryPropertyFlags::empty(),
        atom_size: DeviceAlignment::MIN,
        block_size: 0,
    };

    /// Creates a new `GenericMemoryAllocator<S>` using the provided suballocator `S` for
    /// suballocation of [`DeviceMemory`] blocks.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.block_sizes` doesn't contain as many elements as the number of
    ///   memory types.
    /// - Panics if `create_info.export_handle_types` is non-empty and doesn't contain as many
    ///   elements as the number of memory types.
    pub fn new(
        device: Arc<Device>,
        create_info: GenericMemoryAllocatorCreateInfo<'_>,
    ) -> Result<Self, Box<ValidationError>> {
        Self::validate_new(&device, &create_info)?;

        Ok(unsafe { Self::new_unchecked(device, create_info) })
    }

    fn validate_new(
        device: &Device,
        create_info: &GenericMemoryAllocatorCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        create_info: GenericMemoryAllocatorCreateInfo<'_>,
    ) -> Self {
        let GenericMemoryAllocatorCreateInfo {
            block_sizes,
            memory_type_bits,
            dedicated_allocation,
            export_handle_types,
            mut device_address,
            _ne: _,
        } = create_info;

        let buffer_image_granularity = device
            .physical_device()
            .properties()
            .buffer_image_granularity;

        let memory_types = &device.physical_device().memory_properties().memory_types;

        let mut pools = ArrayVec::new(memory_types.len(), [Self::EMPTY_POOL; MAX_MEMORY_TYPES]);

        for (index, &MemoryType { property_flags, .. }) in memory_types.iter().enumerate() {
            pools[index].property_flags = property_flags;

            if property_flags.intersects(MemoryPropertyFlags::HOST_VISIBLE)
                && !property_flags.intersects(MemoryPropertyFlags::HOST_COHERENT)
            {
                pools[index].atom_size =
                    device.physical_device().properties().non_coherent_atom_size;
            }

            pools[index].block_size = block_sizes[index];
        }

        let export_handle_types = {
            let mut types = ArrayVec::new(
                export_handle_types.len(),
                [ExternalMemoryHandleTypes::empty(); MAX_MEMORY_TYPES],
            );
            types.copy_from_slice(export_handle_types);

            types
        };

        // VUID-VkMemoryAllocateInfo-flags-03331
        device_address &= device.enabled_features().buffer_device_address
            && !device.enabled_extensions().ext_buffer_device_address;
        // Providers of `VkMemoryAllocateFlags`
        device_address &=
            device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_device_group;

        let flags = if device_address {
            MemoryAllocateFlags::DEVICE_ADDRESS
        } else {
            MemoryAllocateFlags::empty()
        };

        let max_memory_allocation_count = device
            .physical_device()
            .properties()
            .max_memory_allocation_count;
        let max_allocations = max_memory_allocation_count / 4 * 3;

        GenericMemoryAllocator {
            device: InstanceOwnedDebugWrapper(device),
            buffer_image_granularity,
            pools,
            dedicated_allocation,
            export_handle_types,
            flags,
            memory_type_bits,
            max_allocations,
        }
    }

    #[cold]
    fn allocate_device_memory(
        &self,
        memory_type_index: u32,
        allocation_size: DeviceSize,
        dedicated_allocation: Option<DedicatedAllocation<'_>>,
        export_handle_types: ExternalMemoryHandleTypes,
    ) -> Result<Arc<DeviceMemory>, Validated<VulkanError>> {
        let mut memory = DeviceMemory::allocate(
            self.device.clone(),
            MemoryAllocateInfo {
                allocation_size,
                memory_type_index,
                dedicated_allocation,
                export_handle_types,
                flags: self.flags,
                ..Default::default()
            },
        )?;

        if self.pools[memory_type_index as usize]
            .property_flags
            .intersects(MemoryPropertyFlags::HOST_VISIBLE)
        {
            // SAFETY:
            // - We checked that the memory is host-visible.
            // - The memory can't be mapped already, because we just allocated it.
            // - Mapping the whole range is always valid.
            unsafe {
                memory.map_unchecked(MemoryMapInfo {
                    offset: 0,
                    size: memory.allocation_size(),
                    _ne: crate::NonExhaustive(()),
                })?;
            }
        }

        Ok(Arc::new(memory))
    }
}

unsafe impl<S: Suballocator + Send + 'static> MemoryAllocator for GenericMemoryAllocator<S> {
    fn find_memory_type_index(
        &self,
        memory_type_bits: u32,
        filter: MemoryTypeFilter,
    ) -> Option<u32> {
        let required_flags = filter.required_flags.into();
        let preferred_flags = filter.preferred_flags.into();
        let not_preferred_flags = filter.not_preferred_flags.into();

        self.pools
            .iter()
            .map(|pool| ash::vk::MemoryPropertyFlags::from(pool.property_flags))
            .enumerate()
            // Filter out memory types which are supported by the memory type bits and have the
            // required flags set.
            .filter(|&(index, flags)| {
                memory_type_bits & (1 << index) != 0 && flags & required_flags == required_flags
            })
            // Rank memory types with more of the preferred flags higher, and ones with more of the
            // not preferred flags lower.
            .min_by_key(|&(_, flags)| {
                (!flags & preferred_flags).as_raw().count_ones()
                    + (flags & not_preferred_flags).as_raw().count_ones()
            })
            .map(|(index, _)| index as u32)
    }

    /// Allocates memory from a specific memory type.
    ///
    /// # Arguments
    ///
    /// - `memory_type_index` - The index of the memory type to allocate from.
    ///
    /// - `layout` - The layout of the allocation.
    ///
    /// - `allocation_type` - The type of resources that can be bound to the allocation.
    ///
    /// - `never_allocate` - If `true` then the allocator should never allocate `DeviceMemory`,
    ///   instead only suballocate from existing blocks.
    ///
    /// # Panics
    ///
    /// - Panics if `memory_type_index` is not less than the number of available memory types.
    ///
    /// # Errors
    ///
    /// - Returns [`AllocateDeviceMemory`] if allocating a new block failed.
    /// - Returns [`OutOfPoolMemory`] if `never_allocate` is `true` and the pool doesn't have
    ///   enough free space.
    /// - Returns [`BlockSizeExceeded`] if `create_info.layout.size()` is greater than the block
    ///   size corresponding to the heap that the memory type corresponding to `memory_type_index`
    ///   resides in.
    ///
    /// [`AllocateDeviceMemory`]: MemoryAllocatorError::AllocateDeviceMemory
    /// [`OutOfPoolMemory`]: MemoryAllocatorError::OutOfPoolMemory
    /// [`BlockSizeExceeded`]: MemoryAllocatorError::BlockSizeExceeded
    fn allocate_from_type(
        &self,
        memory_type_index: u32,
        mut layout: DeviceLayout,
        allocation_type: AllocationType,
        never_allocate: bool,
    ) -> Result<MemoryAlloc, MemoryAllocatorError> {
        let size = layout.size();
        let pool = &self.pools[memory_type_index as usize];

        if size > pool.block_size {
            return Err(MemoryAllocatorError::BlockSizeExceeded);
        }

        layout = layout.align_to(pool.atom_size).unwrap();

        let mut blocks = pool.blocks.lock();

        // TODO: Incremental sorting
        blocks.sort_by_key(|block| block.free_size());
        let (Ok(idx) | Err(idx)) = blocks.binary_search_by_key(&size, |block| block.free_size());

        for block in &blocks[idx..] {
            if let Ok(allocation) =
                block.allocate(layout, allocation_type, self.buffer_image_granularity)
            {
                return Ok(allocation);
            }
        }

        // For bump allocators, first do a garbage sweep and try to allocate again.
        if S::NEEDS_CLEANUP {
            blocks.iter_mut().for_each(|block| block.cleanup());
            blocks.sort_unstable_by_key(|block| block.free_size());

            if let Some(block) = blocks.last() {
                if let Ok(allocation) =
                    block.allocate(layout, allocation_type, self.buffer_image_granularity)
                {
                    return Ok(allocation);
                }
            }
        }

        if never_allocate {
            return Err(MemoryAllocatorError::OutOfPoolMemory);
        }

        // The pool doesn't have enough real estate, so we need a new block.
        let block = {
            let export_handle_types = if !self.export_handle_types.is_empty() {
                self.export_handle_types[memory_type_index as usize]
            } else {
                ExternalMemoryHandleTypes::empty()
            };
            let mut i = 0;

            loop {
                let allocation_size = pool.block_size >> i;

                match self.allocate_device_memory(
                    memory_type_index,
                    allocation_size,
                    None,
                    export_handle_types,
                ) {
                    Ok(device_memory) => {
                        break Block::new(device_memory);
                    }
                    // Retry up to 3 times, halving the allocation size each time so long as the
                    // resulting size is still large enough.
                    Err(Validated::Error(
                        VulkanError::OutOfHostMemory | VulkanError::OutOfDeviceMemory,
                    )) if i < 3 && pool.block_size >> (i + 1) >= size => {
                        i += 1;
                    }
                    Err(err) => return Err(MemoryAllocatorError::AllocateDeviceMemory(err)),
                }
            }
        };

        blocks.push(block);
        let block = blocks.last().unwrap();

        match block.allocate(layout, allocation_type, self.buffer_image_granularity) {
            Ok(allocation) => Ok(allocation),
            // This can't happen as we always allocate a block of sufficient size.
            Err(SuballocatorError::OutOfRegionMemory) => unreachable!(),
            // This can't happen as the block is fresher than Febreze and we're still holding an
            // exclusive lock.
            Err(SuballocatorError::FragmentedRegion) => unreachable!(),
        }
    }

    /// Allocates memory according to requirements.
    ///
    /// # Arguments
    ///
    /// - `requirements` - Requirements of the resource you want to allocate memory for.
    ///
    ///   If you plan to bind this memory directly to a non-sparse resource, then this must
    ///   correspond to the value returned by either [`RawBuffer::memory_requirements`] or
    ///   [`RawImage::memory_requirements`] for the respective buffer or image.
    ///
    /// - `allocation_type` - What type of resource this allocation will be used for.
    ///
    ///   This should be [`Linear`] for buffers and linear images, and [`NonLinear`] for optimal
    ///   images. You can not bind memory allocated with the [`Linear`] type to optimal images or
    ///   bind memory allocated with the [`NonLinear`] type to buffers and linear images. You
    ///   should never use the [`Unknown`] type unless you have to, as that can be less memory
    ///   efficient.
    ///
    /// - `dedicated_allocation` - Allows a dedicated allocation to be created.
    ///
    ///   You should always fill this field in if you are allocating memory for a non-sparse
    ///   resource, otherwise the allocator won't be able to create a dedicated allocation if one
    ///   is required or recommended.
    ///
    ///   This argument is silently ignored (treated as `None`) if the device API version is below
    ///   1.1 and the [`khr_dedicated_allocation`] extension is not enabled on the device.
    ///
    /// # Errors
    ///
    /// - Returns [`AllocateDeviceMemory`] if allocating a new block failed.
    /// - Returns [`FindMemoryType`] if finding a suitable memory type failed. This can happen if
    ///   the `create_info.requirements` correspond to those of an optimal image but
    ///   `create_info.memory_type_filter` requires host access.
    /// - Returns [`OutOfPoolMemory`] if `create_info.allocate_preference` is
    ///   [`MemoryAllocatePreference::NeverAllocate`] and none of the pools of suitable memory
    ///   types have enough free space.
    /// - Returns [`DedicatedAllocationRequired`] if `create_info.allocate_preference` is
    ///   [`MemoryAllocatePreference::NeverAllocate`] and
    ///   `create_info.requirements.requires_dedicated_allocation` is `true`.
    /// - Returns [`BlockSizeExceeded`] if `create_info.allocate_preference` is
    ///   [`MemoryAllocatePreference::NeverAllocate`] and `create_info.requirements.size` is
    ///   greater than the block size for all heaps of suitable memory types.
    ///
    /// [`RawBuffer::memory_requirements`]: crate::buffer::sys::RawBuffer::memory_requirements
    /// [`RawImage::memory_requirements`]: crate::image::sys::RawImage::memory_requirements
    /// [`Linear`]: AllocationType::Linear
    /// [`NonLinear`]: AllocationType::NonLinear
    /// [`Unknown`]: AllocationType::Unknown
    /// [`khr_dedicated_allocation`]: crate::device::DeviceExtensions::khr_dedicated_allocation
    /// [`AllocateDeviceMemory`]: MemoryAllocatorError::AllocateDeviceMemory
    /// [`FindMemoryType`]: MemoryAllocatorError::FindMemoryType
    /// [`OutOfPoolMemory`]: MemoryAllocatorError::OutOfPoolMemory
    /// [`DedicatedAllocationRequired`]: MemoryAllocatorError::DedicatedAllocationRequired
    /// [`BlockSizeExceeded`]: MemoryAllocatorError::BlockSizeExceeded
    fn allocate(
        &self,
        requirements: MemoryRequirements,
        allocation_type: AllocationType,
        create_info: AllocationCreateInfo,
        mut dedicated_allocation: Option<DedicatedAllocation<'_>>,
    ) -> Result<MemoryAlloc, MemoryAllocatorError> {
        let MemoryRequirements {
            layout,
            mut memory_type_bits,
            mut prefers_dedicated_allocation,
            requires_dedicated_allocation,
        } = requirements;

        memory_type_bits &= self.memory_type_bits;
        memory_type_bits &= create_info.memory_type_bits;

        let AllocationCreateInfo {
            memory_type_filter,
            memory_type_bits: _,
            allocate_preference,
            _ne: _,
        } = create_info;

        let size = layout.size();

        let mut memory_type_index = self
            .find_memory_type_index(memory_type_bits, memory_type_filter)
            .ok_or(MemoryAllocatorError::FindMemoryType)?;

        if !self.dedicated_allocation && !requires_dedicated_allocation {
            dedicated_allocation = None;
        }

        let export_handle_types = if self.export_handle_types.is_empty() {
            ExternalMemoryHandleTypes::empty()
        } else {
            self.export_handle_types[memory_type_index as usize]
        };

        loop {
            let pool = &self.pools[memory_type_index as usize];

            let res = match allocate_preference {
                MemoryAllocatePreference::Unknown => {
                    // VUID-vkBindBufferMemory-buffer-01444
                    // VUID-vkBindImageMemory-image-01445
                    if requires_dedicated_allocation {
                        self.allocate_dedicated(
                            memory_type_index,
                            size,
                            dedicated_allocation,
                            export_handle_types,
                        )
                    } else {
                        if size > pool.block_size / 2 {
                            prefers_dedicated_allocation = true;
                        }
                        if self.device.allocation_count() > self.max_allocations
                            && size <= pool.block_size
                        {
                            prefers_dedicated_allocation = false;
                        }

                        if prefers_dedicated_allocation {
                            self.allocate_dedicated(
                                memory_type_index,
                                size,
                                dedicated_allocation,
                                export_handle_types,
                            )
                            // Fall back to suballocation.
                            .or_else(|err| {
                                self.allocate_from_type(
                                    memory_type_index,
                                    layout,
                                    allocation_type,
                                    true, // A dedicated allocation already failed.
                                )
                                .map_err(|_| err)
                            })
                        } else {
                            self.allocate_from_type(
                                memory_type_index,
                                layout,
                                allocation_type,
                                false,
                            )
                            // Fall back to dedicated allocation. It is possible that the 1/8
                            // block size tried was greater than the allocation size, so
                            // there's hope.
                            .or_else(|_| {
                                self.allocate_dedicated(
                                    memory_type_index,
                                    size,
                                    dedicated_allocation,
                                    export_handle_types,
                                )
                            })
                        }
                    }
                }
                MemoryAllocatePreference::NeverAllocate => {
                    if requires_dedicated_allocation {
                        return Err(MemoryAllocatorError::DedicatedAllocationRequired);
                    }

                    self.allocate_from_type(memory_type_index, layout, allocation_type, true)
                }
                MemoryAllocatePreference::AlwaysAllocate => self.allocate_dedicated(
                    memory_type_index,
                    size,
                    dedicated_allocation,
                    export_handle_types,
                ),
            };

            match res {
                Ok(allocation) => return Ok(allocation),
                // Try a different memory type.
                Err(err) => {
                    memory_type_bits &= !(1 << memory_type_index);
                    memory_type_index = self
                        .find_memory_type_index(memory_type_bits, memory_type_filter)
                        .ok_or(err)?;
                }
            }
        }
    }

    #[cold]
    fn allocate_dedicated(
        &self,
        memory_type_index: u32,
        allocation_size: DeviceSize,
        dedicated_allocation: Option<DedicatedAllocation<'_>>,
        export_handle_types: ExternalMemoryHandleTypes,
    ) -> Result<MemoryAlloc, MemoryAllocatorError> {
        let device_memory = self
            .allocate_device_memory(
                memory_type_index,
                allocation_size,
                dedicated_allocation,
                export_handle_types,
            )
            .map_err(MemoryAllocatorError::AllocateDeviceMemory)?;

        Ok(MemoryAlloc {
            device_memory,
            suballocation: None,
            allocation_type: AllocationType::Unknown,
            allocation_handle: AllocationHandle(ptr::null_mut()),
        })
    }

    unsafe fn deallocate(&self, allocation: MemoryAlloc) {
        if let Some(suballocation) = allocation.suballocation {
            let memory_type_index = allocation.device_memory.memory_type_index();
            let pool = self.pools[memory_type_index as usize].blocks.lock();
            let block_ptr = allocation.allocation_handle.0 as *const Block<S>;

            // TODO: Maybe do a similar check for dedicated blocks.
            debug_assert!(
                pool.iter()
                    .any(|block| &**block as *const Block<S> == block_ptr),
                "attempted to deallocate a memory block that does not belong to this allocator",
            );

            // SAFETY: The caller must guarantee that `allocation` refers to one allocated by
            // `self`, therefore `block_ptr` must be the same one we gave out on allocation. We
            // know that this pointer must be valid, because all blocks are boxed and pinned in
            // memory and because a block isn't dropped until the allocator itself is dropped, at
            // which point it would be impossible to call this method. We also know that it must be
            // valid to create a reference to the block, because we locked the pool it belongs to.
            let block = &*block_ptr;

            // SAFETY: The caller must guarantee that `allocation` refers to a currently allocated
            // allocation of `self`.
            block.deallocate(suballocation);

            drop(pool);
        }
    }
}

unsafe impl<T: MemoryAllocator> MemoryAllocator for Arc<T> {
    fn find_memory_type_index(
        &self,
        memory_type_bits: u32,
        filter: MemoryTypeFilter,
    ) -> Option<u32> {
        (**self).find_memory_type_index(memory_type_bits, filter)
    }

    fn allocate_from_type(
        &self,
        memory_type_index: u32,
        layout: DeviceLayout,
        allocation_type: AllocationType,
        never_allocate: bool,
    ) -> Result<MemoryAlloc, MemoryAllocatorError> {
        (**self).allocate_from_type(memory_type_index, layout, allocation_type, never_allocate)
    }

    fn allocate(
        &self,
        requirements: MemoryRequirements,
        allocation_type: AllocationType,
        create_info: AllocationCreateInfo,
        dedicated_allocation: Option<DedicatedAllocation<'_>>,
    ) -> Result<MemoryAlloc, MemoryAllocatorError> {
        (**self).allocate(
            requirements,
            allocation_type,
            create_info,
            dedicated_allocation,
        )
    }

    fn allocate_dedicated(
        &self,
        memory_type_index: u32,
        allocation_size: DeviceSize,
        dedicated_allocation: Option<DedicatedAllocation<'_>>,
        export_handle_types: ExternalMemoryHandleTypes,
    ) -> Result<MemoryAlloc, MemoryAllocatorError> {
        (**self).allocate_dedicated(
            memory_type_index,
            allocation_size,
            dedicated_allocation,
            export_handle_types,
        )
    }

    unsafe fn deallocate(&self, allocation: MemoryAlloc) {
        (**self).deallocate(allocation)
    }
}

unsafe impl<S> DeviceOwned for GenericMemoryAllocator<S> {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

#[derive(Debug)]
struct Block<S> {
    device_memory: Arc<DeviceMemory>,
    suballocator: S,
}

impl<S: Suballocator> Block<S> {
    fn new(device_memory: Arc<DeviceMemory>) -> Box<Self> {
        let suballocator = S::new(0, device_memory.allocation_size());

        Box::new(Block {
            device_memory,
            suballocator,
        })
    }

    fn allocate(
        &self,
        layout: DeviceLayout,
        allocation_type: AllocationType,
        buffer_image_granularity: DeviceAlignment,
    ) -> Result<MemoryAlloc, SuballocatorError> {
        let suballocation =
            self.suballocator
                .allocate(layout, allocation_type, buffer_image_granularity)?;

        Ok(MemoryAlloc {
            device_memory: self.device_memory.clone(),
            suballocation: Some(suballocation),
            allocation_type,
            allocation_handle: AllocationHandle(self as *const Block<S> as _),
        })
    }

    unsafe fn deallocate(&self, suballocation: Suballocation) {
        self.suballocator.deallocate(suballocation)
    }

    fn free_size(&self) -> DeviceSize {
        self.suballocator.free_size()
    }

    fn cleanup(&mut self) {
        self.suballocator.cleanup();
    }
}

/// Parameters to create a new [`GenericMemoryAllocator`].
#[derive(Clone, Debug)]
pub struct GenericMemoryAllocatorCreateInfo<'a> {
    /// Lets you configure the block sizes for each memory type.
    ///
    /// Must contain one entry for each memory type. The allocator keeps a pool of [`DeviceMemory`]
    /// blocks for each memory type, and every time a new block is allocated, the block size
    /// corresponding to the memory type index is looked up here and used for the allocation.
    ///
    /// The block size is going to be the maximum size of a `DeviceMemory` block that is tried. If
    /// allocating a block with the size fails, the allocator tries 1/2, 1/4 and 1/8 of the block
    /// size in that order until one succeeds, else a dedicated allocation is attempted for the
    /// allocation. If an allocation is created with a size greater than half the block size it is
    /// always made a dedicated allocation. All of this doesn't apply when using
    /// [`MemoryAllocatePreference::NeverAllocate`] however.
    ///
    /// The default value is `&[]`, which must be overridden.
    pub block_sizes: &'a [DeviceSize],

    /// Lets you configure the allocator's global mask of memory type indices. Only the memory type
    /// indices that have a corresponding bit at the same index set will be allocated from when
    /// calling [`allocate`], otherwise [`MemoryAllocatorError::FindMemoryType`] is returned.
    ///
    /// You may use this to disallow problematic memory types, for instance ones with the
    /// [`PROTECTED`] flag, or any other flags you don't want.
    ///
    /// The default value is [`u32::MAX`].
    ///
    /// [`allocate`]: struct.GenericMemoryAllocator.html#method.allocate
    /// [`PROTECTED`]: MemoryPropertyFlags::DEVICE_COHERENT
    pub memory_type_bits: u32,

    /// Whether the allocator should use the dedicated allocation APIs.
    ///
    /// This means that when the allocator decides that an allocation should not be suballocated,
    /// but rather have its own block of [`DeviceMemory`], that that allocation will be made a
    /// dedicated allocation. Otherwise they are still given their own block of device memory, just
    /// that that block won't be [dedicated] to a particular resource.
    ///
    /// Dedicated allocations are an optimization which may result in better performance, so there
    /// really is no reason to disable this option, unless the restrictions that they bring with
    /// them are a problem. Namely, a dedicated allocation must only be used for the resource it
    /// was created for. Meaning that reusing the memory for something else is not possible,
    /// suballocating it is not possible, and aliasing it is also not possible.
    ///
    /// This option is silently ignored (treated as `false`) if the device API version is below 1.1
    /// and the [`khr_dedicated_allocation`] extension is not enabled on the device.
    ///
    /// The default value is `true`.
    ///
    /// [dedicated]: DeviceMemory::is_dedicated
    /// [`khr_dedicated_allocation`]: crate::device::DeviceExtensions::khr_dedicated_allocation
    pub dedicated_allocation: bool,

    /// Lets you configure the external memory handle types that the [`DeviceMemory`] blocks will
    /// be allocated with.
    ///
    /// Must be either empty or contain one element for each memory type. When `DeviceMemory` is
    /// allocated, the external handle types corresponding to the memory type index are looked up
    /// here and used for the allocation.
    ///
    /// The default value is `&[]`.
    pub export_handle_types: &'a [ExternalMemoryHandleTypes],

    /// Whether the allocator should allocate the [`DeviceMemory`] blocks with the
    /// [`DEVICE_ADDRESS`] flag set.
    ///
    /// This is required if you want to allocate memory for buffers that have the
    /// [`SHADER_DEVICE_ADDRESS`] usage set. For this option too, there is no reason to disable it.
    ///
    /// This option is silently ignored (treated as `false`) if the [`buffer_device_address`]
    /// feature is not enabled on the device or if the [`ext_buffer_device_address`] extension is
    /// enabled on the device. It is also ignored if the device API version is below 1.1 and the
    /// [`khr_device_group`] extension is not enabled on the device.
    ///
    /// The default value is `true`.
    ///
    /// [`DEVICE_ADDRESS`]: MemoryAllocateFlags::DEVICE_ADDRESS
    /// [`SHADER_DEVICE_ADDRESS`]: crate::buffer::BufferUsage::SHADER_DEVICE_ADDRESS
    /// [`buffer_device_address`]: crate::device::Features::buffer_device_address
    /// [`ext_buffer_device_address`]: crate::device::DeviceExtensions::ext_buffer_device_address
    /// [`khr_device_group`]: crate::device::DeviceExtensions::khr_device_group
    pub device_address: bool,

    pub _ne: crate::NonExhaustive,
}

impl GenericMemoryAllocatorCreateInfo<'_> {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            block_sizes,
            memory_type_bits: _,
            dedicated_allocation: _,
            export_handle_types,
            device_address: _,
            _ne: _,
        } = self;

        let memory_types = &device.physical_device().memory_properties().memory_types;

        assert!(
            block_sizes.len() == memory_types.len(),
            "`create_info.block_sizes` must contain as many elements as the number of memory types",
        );

        if !export_handle_types.is_empty() {
            if !(device.api_version() >= Version::V1_1
                && device.enabled_extensions().khr_external_memory)
            {
                return Err(Box::new(ValidationError {
                    context: "export_handle_types".into(),
                    problem: "is not empty".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::APIVersion(Version::V1_1)]),
                        RequiresAllOf(&[Requires::DeviceExtension("khr_external_memory")]),
                    ]),
                    ..Default::default()
                }));
            }

            assert!(
                export_handle_types.len() == memory_types.len(),
                "`create_info.export_handle_types` must contain as many elements as the number of \
                memory types if not empty",
            );

            for (index, export_handle_types) in export_handle_types.iter().enumerate() {
                export_handle_types
                    .validate_device(device)
                    .map_err(|err| err.add_context(format!("export_handle_types[{}]", index)))?;
            }
        }

        Ok(())
    }
}

impl Default for GenericMemoryAllocatorCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        GenericMemoryAllocatorCreateInfo {
            block_sizes: &[],
            memory_type_bits: u32::MAX,
            dedicated_allocation: true,
            export_handle_types: &[],
            device_address: true,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// > **Note**: Returns `0` on overflow.
#[inline(always)]
pub(crate) const fn align_up(val: DeviceSize, alignment: DeviceAlignment) -> DeviceSize {
    align_down(val.wrapping_add(alignment.as_devicesize() - 1), alignment)
}

#[inline(always)]
pub(crate) const fn align_down(val: DeviceSize, alignment: DeviceAlignment) -> DeviceSize {
    val & !(alignment.as_devicesize() - 1)
}

mod array_vec {
    use std::ops::{Deref, DerefMut};

    /// Minimal implementation of an `ArrayVec`. Useful when a `Vec` is needed but there is a known
    /// limit on the number of elements, so that it can occupy real estate on the stack.
    #[derive(Clone, Copy, Debug)]
    pub(super) struct ArrayVec<T, const N: usize> {
        len: usize,
        data: [T; N],
    }

    impl<T, const N: usize> ArrayVec<T, N> {
        pub fn new(len: usize, data: [T; N]) -> Self {
            assert!(len <= N);

            ArrayVec { len, data }
        }
    }

    impl<T, const N: usize> Deref for ArrayVec<T, N> {
        type Target = [T];

        fn deref(&self) -> &Self::Target {
            // SAFETY: `self.len <= N`.
            unsafe { self.data.get_unchecked(0..self.len) }
        }
    }

    impl<T, const N: usize> DerefMut for ArrayVec<T, N> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            // SAFETY: `self.len <= N`.
            unsafe { self.data.get_unchecked_mut(0..self.len) }
        }
    }
}
