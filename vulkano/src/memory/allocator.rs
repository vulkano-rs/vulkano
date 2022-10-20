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

use self::{array_vec::ArrayVec, host::SlotId};
use super::{
    DedicatedAllocation, MemoryAllocateFlags, MemoryAllocateInfo, MemoryRequirements, MemoryType,
};
use crate::{
    buffer::{
        sys::{UnsafeBuffer, UnsafeBufferCreateInfo},
        BufferCreationError,
    },
    device::{Device, DeviceOwned},
    image::{
        sys::{UnsafeImage, UnsafeImageCreateInfo},
        ImageCreationError, ImageTiling,
    },
    memory::{DeviceMemory, MemoryProperties},
    DeviceSize, Version, VulkanError, VulkanObject,
};
use ash::vk::{MAX_MEMORY_HEAPS, MAX_MEMORY_TYPES};
use crossbeam_queue::ArrayQueue;
use parking_lot::{Mutex, RwLock};
use std::{
    cell::Cell,
    error::Error,
    ffi::c_void,
    fmt::{self, Display},
    mem::{self, ManuallyDrop, MaybeUninit},
    num::NonZeroU64,
    ops::Range,
    ptr::{self, NonNull},
    slice,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

const B: DeviceSize = 1;
const K: DeviceSize = 1024 * B;
const M: DeviceSize = 1024 * K;
const G: DeviceSize = 1024 * M;

/// General-purpose memory allocators which allocate from any memory type dynamically as needed.
pub unsafe trait MemoryAllocator: DeviceOwned {
    /// Allocates memory from a specific memory type.
    fn allocate_from_type(
        &self,
        memory_type_index: u32,
        create_info: SuballocationCreateInfo,
    ) -> Result<MemoryAlloc, AllocationCreationError>;

    /// Allocates memory according to requirements.
    fn allocate(
        &self,
        create_info: AllocationCreateInfo<'_>,
    ) -> Result<MemoryAlloc, AllocationCreationError>;

    /// Conveniece method to create a (non-sparse) `UnsafeBuffer`, allocate memory for it, and bind
    /// the memory to it.
    ///
    /// The implementation of this method can be optimized as no checks need to be made, since the
    /// parameters for allocation come straight from the Vulkan implementation.
    fn create_buffer(
        &self,
        create_info: UnsafeBufferCreateInfo,
        usage: MemoryUsage,
        allocate_preference: MemoryAllocatePreference,
    ) -> Result<
        Result<(Arc<UnsafeBuffer>, MemoryAlloc), AllocationCreationError>,
        BufferCreationError,
    >;

    /// Conveniece method to create a (non-sparse) `UnsafeImage`, allocate memory for it, and bind
    /// the memory to it.
    ///
    /// The implementation of this method can be optimized as no checks need to be made, since the
    /// parameters for allocation come straight from the Vulkan implementation.
    fn create_image(
        &self,
        create_info: UnsafeImageCreateInfo,
        usage: MemoryUsage,
        allocate_preference: MemoryAllocatePreference,
    ) -> Result<Result<(Arc<UnsafeImage>, MemoryAlloc), AllocationCreationError>, ImageCreationError>;
}

/// Parameters to create a new [allocation] using a [memory allocator].
///
/// [allocation]: MemoryAlloc
/// [memory allocator]: MemoryAllocator
#[derive(Clone, Debug)]
pub struct AllocationCreateInfo<'d> {
    /// Requirements of the resource you want to allocate memory for.
    ///
    /// If you plan to bind this memory directly to a non-sparse resource, then this must
    /// correspond to the value returned by either [`UnsafeBuffer::memory_requirements`] or
    /// [`UnsafeImage::memory_requirements`] for the respective buffer or image.
    ///
    /// All of the fields must be non-zero, [`alignment`] must be a power of two, and
    /// [`memory_type_bits`] must be below 2<sup>*n*</sup> where *n* is the number of available
    /// memory types.
    ///
    /// The default is all zeros, which must be overridden.
    ///
    /// [`alignment`]: MemoryRequirements::alignment
    /// [`memory_type_bits`]: MemoryRequirements::memory_type_bits
    pub requirements: MemoryRequirements,

    /// What type of resource this allocation will be used for.
    ///
    /// This should be [`Linear`] for buffers and linear images, and [`NonLinear`] for optimal
    /// images. You can not bind memory allocated with the [`Linear`] type to optimal images or
    /// bind memory allocated with the [`NonLinear`] type to buffers and linear images. You should
    /// never use the [`Unknown`] type unless you have to, as that can be less memory efficient.
    ///
    /// The default value is [`AllocationType::Unknown`].
    ///
    /// [`Linear`]: AllocationType::Linear
    /// [`NonLinear`]: AllocationType::NonLinear
    /// [`Unknown`]: AllocationType::Unknown
    pub allocation_type: AllocationType,

    /// The intended usage for the allocation.
    ///
    /// The default value is [`MemoryUsage::GpuOnly`].
    pub usage: MemoryUsage,

    /// How eager the allocator should be to allocate [`DeviceMemory`].
    ///
    /// The default value is [`MemoryAllocatePreference::Unknown`].
    pub allocate_preference: MemoryAllocatePreference,

    /// Allows a dedicated allocation to be created.
    ///
    /// You should always fill this field in if you are allocating memory for a non-sparse
    /// resource, otherwise the allocator won't be able to create a dedicated allocation if one is
    /// recommended.
    ///
    /// This option is silently ignored (treated as `None`) if the device API version is below 1.1
    /// and the [`khr_dedicated_allocation`] extension is not enabled on the device.
    ///
    /// The default value is [`None`].
    ///
    /// [`requirements.prefer_dedicated`]: MemoryRequirements::prefer_dedicated
    /// [`khr_dedicated_allocation`]: crate::device::DeviceExtensions::khr_dedicated_allocation
    pub dedicated_allocation: Option<DedicatedAllocation<'d>>,

    pub _ne: crate::NonExhaustive,
}

impl Default for AllocationCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        AllocationCreateInfo {
            requirements: MemoryRequirements {
                size: 0,
                alignment: 0,
                memory_type_bits: 0,
                prefer_dedicated: false,
            },
            allocation_type: AllocationType::Unknown,
            usage: MemoryUsage::GpuOnly,
            allocate_preference: MemoryAllocatePreference::Unknown,
            dedicated_allocation: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Describes how a memory allocation is going to be used.
///
/// This is mostly an optimization, except for `MemoryUsage::GpuOnly` which will pick a memory type
/// that is not CPU-accessible if such a type exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryUsage {
    /// The memory is intended to only be used by the GPU.
    ///
    /// Prefers picking a memory type with the [`device_local`] flag and without the
    /// [`host_visible`] flag.
    ///
    /// This option is what you will always want to use unless the memory needs to be accessed by
    /// the CPU, because a memory type that can only be accessed by the GPU is going to give the
    /// best performance. Example use cases would be textures and other maps which are written to
    /// once and then never again, or resources that are only written and read by the GPU, like
    /// render targets and intermediary buffers.
    ///
    /// [`device_local`]: super::MemoryPropertyFlags::device_local
    /// [`host_visible`]: super::MemoryPropertyFlags::host_visible
    GpuOnly,

    /// The memory is intended for upload to the GPU.
    ///
    /// Guarantees picking a memory type with the [`host_visible`] flag. Prefers picking one
    /// without the [`host_cached`] flag and with the [`device_local`] flag.
    ///
    /// This option is best suited for resources that need to be constantly updated by the CPU,
    /// like vertex and index buffers for example. It is also neccessary for *staging buffers*,
    /// whose only purpose in life it is to get data into `device_local` memory or texels into an
    /// optimal image.
    ///
    /// [`host_visible`]: super::MemoryPropertyFlags::host_visible
    /// [`host_cached`]: super::MemoryPropertyFlags::host_cached
    /// [`device_local`]: super::MemoryPropertyFlags::device_local
    Upload,

    /// The memory is intended for download from the GPU.
    ///
    /// Guarantees picking a memory type with the [`host_visible`] flag. Prefers picking one with
    /// the [`host_cached`] flag and without the [`device_local`] flag.
    ///
    /// This option is best suited if you're using the GPU for things other than rendering and you
    /// need to get the results back to the CPU. That might be compute shading, or image or video
    /// manipulation, or screenshotting for example.
    ///
    /// [`host_visible`]: super::MemoryPropertyFlags::host_visible
    /// [`host_cached`]: super::MemoryPropertyFlags::host_cached
    /// [`device_local`]: super::MemoryPropertyFlags::device_local
    Download,
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

/// Memory allocations are portions of memory that are are reserved for a specific resource or
/// purpose.
///
/// There's a few ways you can obtain a `MemoryAlloc` in Vulkano. Most commonly you will probably
/// want to use a [memory allocator]. If you want a root allocation, and already have a
/// [`DeviceMemory`] block on hand, you can turn it into a `MemoryAlloc` by using the [`From`]
/// implementation. Lastly, you can use a [suballocator] if you want to create multiple smaller
/// allocations out of a bigger one.
///
/// [memory allocator]: MemoryAllocator
/// [`From`]: Self#impl-From<DeviceMemory>-for-MemoryAlloc
/// [suballocator]: Suballocator
#[derive(Debug)]
pub struct MemoryAlloc {
    offset: DeviceSize,
    size: DeviceSize,
    // Needed when binding resources to the allocation in order to avoid aliasing memory.
    allocation_type: AllocationType,
    // Underlying block of memory. These fields are duplicated here to avoid walking up the
    // hierarchy when binding.
    memory: ash::vk::DeviceMemory,
    memory_type_index: u32,
    // Used by the suballocators to align allocations to the non-coherent atom size when the memory
    // type is host-visible but not host-coherent. This will be `None` for any other memory type.
    non_coherent_atom_size: Option<NonZeroU64>,
    // Mapped pointer to the start of the allocation or `None` is the memory is not host-visible.
    mapped_ptr: Option<NonNull<c_void>>,
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
    /// [host-coherent]: super::MemoryPropertyFlags::host_coherent
    /// [`non_coherent_atom_size`]: crate::device::Properties::non_coherent_atom_size
    #[inline]
    pub unsafe fn invalidate_range(&self, range: Range<DeviceSize>) -> Result<(), VulkanError> {
        // VUID-VkMappedMemoryRange-memory-00684
        if let Some(atom_size) = self.non_coherent_atom_size {
            let range = self.create_memory_range(range, atom_size.get());
            let device = self.device();
            let fns = device.fns();
            (fns.v1_0.invalidate_mapped_memory_ranges)(device.internal_object(), 1, &range)
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
    /// [host-coherent]: super::MemoryPropertyFlags::host_coherent
    /// [`non_coherent_atom_size`]: crate::device::Properties::non_coherent_atom_size
    #[inline]
    pub unsafe fn flush_range(&self, range: Range<DeviceSize>) -> Result<(), VulkanError> {
        // VUID-VkMappedMemoryRange-memory-00684
        if let Some(atom_size) = self.non_coherent_atom_size {
            let range = self.create_memory_range(range, atom_size.get());
            let device = self.device();
            let fns = device.fns();
            (fns.v1_0.flush_mapped_memory_ranges)(device.internal_object(), 1, &range)
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
        atom_size: DeviceSize,
    ) -> ash::vk::MappedMemoryRange {
        assert!(!range.is_empty() && range.end <= self.size);

        // VUID-VkMappedMemoryRange-size-00685
        // Guaranteed because we always map the entire `DeviceMemory`.

        // VUID-VkMappedMemoryRange-offset-00687
        // VUID-VkMappedMemoryRange-size-01390
        assert!(
            range.start % atom_size == 0 && (range.end % atom_size == 0 || range.end == self.size)
        );

        // VUID-VkMappedMemoryRange-offset-00687
        // Guaranteed as long as `range.start` is aligned because the suballocators always align
        // `self.offset` to the non-coherent atom size for non-coherent host-visible memory.
        let offset = self.offset + range.start;

        let mut size = range.end - range.start;

        // VUID-VkMappedMemoryRange-size-01390
        if offset + size < self.device_memory().allocation_size() {
            // We align the size in case `range.end == self.size`. We can do this without aliasing
            // other allocations because the suballocators ensure that all allocations are aligned
            // to the atom size for non-coherent host-visible memory.
            size = align_up(size, atom_size);
        }

        ash::vk::MappedMemoryRange {
            memory: self.memory,
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
        debug_assert!({
            let atom_size = self
                .device()
                .physical_device()
                .properties()
                .non_coherent_atom_size;

            range.start % atom_size == 0 && (range.end % atom_size == 0 || range.end == self.size)
        });
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
            mapped_ptr: None,
            non_coherent_atom_size: None,
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
            AllocParent::Buddy { allocator, order } => {
                allocator.free(*order, self.offset);
            }
            AllocParent::Pool { allocator, index } => {
                allocator.free(*index);
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

unsafe impl VulkanObject for MemoryAlloc {
    type Object = ash::vk::DeviceMemory;

    #[inline]
    fn internal_object(&self) -> Self::Object {
        self.memory
    }
}

/// Error that can be returned when creating an [allocation] using a [memory allocator].
///
/// [allocation]: MemoryAlloc
/// [memory allocator]: MemoryAllocator
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AllocationCreationError {
    /// There is not enough memory on the host.
    OutOfHostMemory,

    /// There is not enough memory on the device.
    OutOfDeviceMemory,

    /// Too many [`DeviceMemory`] allocations exist already.
    TooManyObjects,

    /// There is not enough memory in the pool.
    ///
    /// This is returned when using [`MemoryAllocatePreference::NeverAllocate`] and there is not
    /// enough memory in the pool.
    OutOfPoolMemory,

    /// Failed to map memory.
    MemoryMapFailed,

    /// The block size for the suballocator was exceeded.
    ///
    /// This is returned when using [`GenericMemoryAllocator<Arc<PoolAllocator<BLOCK_SIZE>>>`] if
    /// the allocation size exceeded `BLOCK_SIZE`.
    BlockSizeExceeded,
}

impl Display for AllocationCreationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::OutOfHostMemory => "out of host memory",
                Self::OutOfDeviceMemory => "out of device memory",
                Self::TooManyObjects => "too many `DeviceMemory` allocations exist already",
                Self::OutOfPoolMemory => "the pool doesn't have enough free space",
                Self::MemoryMapFailed => "failed to map memory",
                Self::BlockSizeExceeded =>
                    "the allocation size was greater than the suballocator's block size",
            }
        )
    }
}

impl Error for AllocationCreationError {}

/// Standard memory allocator intended as a global and general-purpose allocator.
///
/// This type of allocator should work well in most cases, it is however **not** to be used when
/// allocations need to be made very frequently. For that purpose, use [`FastMemoryAllocator`].
///
/// See [`FreeListAllocator`] for details about the allocation algorithm.
pub type StandardMemoryAllocator = GenericMemoryAllocator<Arc<FreeListAllocator>>;

impl StandardMemoryAllocator {
    /// Creates a new `StandardMemoryAllocator` with default configuration.
    pub fn new_default(device: Arc<Device>) -> Self {
        #[allow(clippy::erasing_op, clippy::identity_op)]
        let create_info = GenericMemoryAllocatorCreateInfo {
            #[rustfmt::skip]
            block_sizes: &[
                (0 * B,  64 * M),
                (1 * G, 256 * M),
            ],
            ..Default::default()
        };

        unsafe { Self::new_unchecked(device, create_info) }
    }
}

/// Fast memory allocator intended as a local and special-purpose allocator.
///
/// This type of allocator is only useful when you need to allocate a lot, for example once or more
/// per frame. It is **not** to be used when allocations are long-lived. For that purpose use
/// [`StandardMemoryAllocator`].
///
/// See [`BumpAllocator`] for details about the allocation algorithm.
pub type FastMemoryAllocator = GenericMemoryAllocator<Arc<BumpAllocator>>;

impl FastMemoryAllocator {
    /// Creates a new `FastMemoryAllocator` with default configuration.
    pub fn new_default(device: Arc<Device>) -> Self {
        #[allow(clippy::erasing_op, clippy::identity_op)]
        let create_info = GenericMemoryAllocatorCreateInfo {
            #[rustfmt::skip]
            block_sizes: &[
                (  0 * B, 16 * M),
                (512 * M, 32 * M),
                (  1 * G, 64 * M),
            ],
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
/// # `DeviceMemory` allocation
///
/// If an allocation is created with the [`MemoryAllocatePreference::Unknown`] option, and the
/// allocator deems the allocation too big for suballocation (larger than half the block size), or
/// the implementation prefers a dedicated allocation, then that allocation is made a dedicated
/// allocation. Using [`MemoryAllocatePreference::NeverAllocate`], a dedicated allocation is never
/// created, even if the allocation is larger than the block size. In such a case an error is
/// returned instead. Using [`MemoryAllocatePreference::AlwaysAllocate`], a dedicated allocation is
/// always created.
///
/// In all other cases, `DeviceMemory` is only allocated if a pool runs out of memory and needs
/// another block. No `DeviceMemory` is allocated when the allocator is created, the blocks are
/// only allocated once they are needed.
///
/// # Locking behavior
///
/// The allocator never needs to lock while suballocating unless `S` needs to lock. The only time
/// when a pool must be locked is when a new `DeviceMemory` block is allocated for the pool. This
/// means that the allocator is suited to both locking and lock-free (sub)allocation algorithms.
///
/// [memory allocator]: MemoryAllocator
/// [suballocate]: Suballocator
/// [the `MemoryAllocator` implementation]: Self#impl-MemoryAllocator-for-GenericMemoryAllocator<S>
#[derive(Debug)]
pub struct GenericMemoryAllocator<S: Suballocator> {
    device: Arc<Device>,
    // Each memory type has a pool of `DeviceMemory` blocks.
    pools: ArrayVec<Pool<S>, MAX_MEMORY_TYPES>,
    // Each memory heap has its own block size.
    block_sizes: ArrayVec<DeviceSize, MAX_MEMORY_HEAPS>,
    dedicated_allocation: bool,
    flags: MemoryAllocateFlags,
    // Global mask of memory types.
    memory_type_bits: u32,
    // How many `DeviceMemory` allocations should be allowed before restricting them.
    max_memory_allocation_count: u32,
}

#[derive(Debug)]
struct Pool<S> {
    blocks: RwLock<Vec<S>>,
    // This is cached here for faster access, so we don't need to hop through 3 pointers.
    memory_type: ash::vk::MemoryType,
}

impl<S: Suballocator> GenericMemoryAllocator<S> {
    // This is a false-positive, we only use this const for static initialization.
    #[allow(clippy::declare_interior_mutable_const)]
    const EMPTY_POOL: Pool<S> = Pool {
        blocks: RwLock::new(Vec::new()),
        memory_type: ash::vk::MemoryType {
            property_flags: ash::vk::MemoryPropertyFlags::empty(),
            heap_index: 0,
        },
    };

    /// Creates a new `GenericMemoryAllocator<S>` using the provided suballocator `S` for
    /// suballocation of [`DeviceMemory`] blocks.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.block_sizes` is not sorted by threshold.
    /// - Panics if `create_info.block_sizes` contains duplicate thresholds.
    /// - Panics if `create_info.block_sizes` does not contain a baseline threshold of `0`.
    /// - Panics if the block size for a heap exceeds the size of the heap.
    pub fn new(device: Arc<Device>, create_info: GenericMemoryAllocatorCreateInfo<'_>) -> Self {
        Self::validate_new(&create_info);

        unsafe { Self::new_unchecked(device, create_info) }
    }

    fn validate_new(create_info: &GenericMemoryAllocatorCreateInfo<'_>) {
        let GenericMemoryAllocatorCreateInfo {
            block_sizes,
            dedicated_allocation: _,
            device_address: _,
            _ne: _,
        } = create_info;

        assert!(
            block_sizes.windows(2).all(|win| win[0].0 < win[1].0),
            "`create_info.block_sizes` must be sorted by threshold without duplicates",
        );
        assert!(
            matches!(block_sizes.first(), Some((0, _))),
            "`create_info.block_sizes` must contain a baseline threshold `0`",
        );
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        create_info: GenericMemoryAllocatorCreateInfo<'_>,
    ) -> Self {
        let GenericMemoryAllocatorCreateInfo {
            block_sizes,
            mut dedicated_allocation,
            mut device_address,
            _ne: _,
        } = create_info;

        let MemoryProperties {
            memory_types,
            memory_heaps,
        } = device.physical_device().memory_properties();

        let mut pools = ArrayVec::new(memory_types.len(), [Self::EMPTY_POOL; MAX_MEMORY_TYPES]);
        for (i, memory_type) in memory_types.iter().enumerate() {
            pools[i].memory_type = ash::vk::MemoryType {
                property_flags: memory_type.property_flags.into(),
                heap_index: memory_type.heap_index,
            };
        }

        let block_sizes = {
            let mut sizes = ArrayVec::new(memory_heaps.len(), [0; MAX_MEMORY_HEAPS]);
            for (i, memory_heap) in memory_heaps.iter().enumerate() {
                let idx = match block_sizes.binary_search_by_key(&memory_heap.size, |&(t, _)| t) {
                    Ok(idx) => idx,
                    Err(idx) => idx.saturating_sub(1),
                };
                sizes[i] = block_sizes[idx].1;

                // VUID-vkAllocateMemory-pAllocateInfo-01713
                assert!(sizes[i] <= memory_heap.size);
            }

            sizes
        };

        // Providers of `VkMemoryDedicatedAllocateInfo`
        dedicated_allocation &= device.api_version() >= Version::V1_1
            || device.enabled_extensions().khr_dedicated_allocation;

        // VUID-VkMemoryAllocateInfo-flags-03331
        device_address &= device.enabled_features().buffer_device_address
            && !device.enabled_extensions().ext_buffer_device_address;
        // Providers of `VkMemoryAllocateFlags`
        device_address &=
            device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_device_group;

        let mut memory_type_bits = u32::MAX;
        for (index, MemoryType { property_flags, .. }) in memory_types.iter().enumerate() {
            if property_flags.lazily_allocated
                || property_flags.protected
                || property_flags.device_coherent
                || property_flags.device_uncached
                || property_flags.rdma_capable
            {
                memory_type_bits &= !(1 << index);
            }
        }

        let max_memory_allocation_count = device
            .physical_device()
            .properties()
            .max_memory_allocation_count;
        let max_memory_allocation_count = max_memory_allocation_count * 3 / 4;

        GenericMemoryAllocator {
            device,
            pools,
            block_sizes,
            dedicated_allocation,
            flags: MemoryAllocateFlags {
                device_address,
                ..Default::default()
            },
            memory_type_bits,
            max_memory_allocation_count,
        }
    }

    fn validate_allocate_from_type(
        &self,
        memory_type_index: u32,
        create_info: &SuballocationCreateInfo,
    ) {
        let memory_type = &self.pools[memory_type_index as usize].memory_type;
        // VUID-VkMemoryAllocateInfo-memoryTypeIndex-01872
        assert!(
            !memory_type
                .property_flags
                .contains(ash::vk::MemoryPropertyFlags::PROTECTED)
                || self.device.enabled_features().protected_memory,
            "attempted to allocate from a protected memory type without the `protected_memory` \
            feature being enabled on the device",
        );

        let block_size = self.block_sizes[memory_type.heap_index as usize];
        // VUID-vkAllocateMemory-pAllocateInfo-01713
        assert!(
            create_info.size <= block_size,
            "attempted to create an allocation larger than the block size for the memory heap",
        );

        create_info.validate();
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn allocate_from_type_unchecked(
        &self,
        memory_type_index: u32,
        create_info: SuballocationCreateInfo,
        never_allocate: bool,
    ) -> Result<MemoryAlloc, AllocationCreationError> {
        let SuballocationCreateInfo {
            size,
            alignment: _,
            allocation_type: _,
            _ne: _,
        } = create_info;

        let pool = &self.pools[memory_type_index as usize];

        let mut blocks = if S::IS_BLOCKING {
            // If the allocation algorithm needs to block, then there's no point in trying to avoid
            // locks here either. In that case the best strategy is to take full advantage of it by
            // always taking an exclusive lock, which lets us sort the blocks by free size. If you
            // as a user want to avoid locks, simply don't share the allocator between threads. You
            // can create as many allocators as you wish, but keep in mind that that will waste a
            // huge amount of memory unless you configure your block sizes properly!

            let mut blocks = pool.blocks.write();
            blocks.sort_by_key(Suballocator::free_size);
            let (Ok(idx) | Err(idx)) = blocks.binary_search_by_key(&size, Suballocator::free_size);
            for block in &blocks[idx..] {
                match block.allocate_unchecked(create_info.clone()) {
                    Ok(alloc) => return Ok(alloc),
                    Err(SuballocationCreationError::BlockSizeExceeded) => {
                        return Err(AllocationCreationError::BlockSizeExceeded);
                    }
                    Err(_) => {}
                }
            }

            blocks
        } else {
            // If the allocation algorithm is lock-free, then we should avoid taking an exclusive
            // lock unless it is absolutely neccessary (meaning, only when allocating a new
            // `DeviceMemory` block and inserting it into a pool). This has the disadvantage that
            // traversing the pool is O(n), which is not a problem since the number of blocks is
            // expected to be small. If there are more than 10 blocks in a pool then that's a
            // configuration error. Also, sorting the blocks before each allocation would be less
            // efficient because to get the free size of the `PoolAllocator` and `BumpAllocator`
            // has the same performance as trying to allocate.

            let blocks = pool.blocks.read();
            // Search in reverse order because we always append new blocks at the end.
            for block in blocks.iter().rev() {
                match block.allocate_unchecked(create_info.clone()) {
                    Ok(alloc) => return Ok(alloc),
                    // This can happen when using the `PoolAllocator<BLOCK_SIZE>` if the allocation
                    // size is greater than `BLOCK_SIZE`.
                    Err(SuballocationCreationError::BlockSizeExceeded) => {
                        return Err(AllocationCreationError::BlockSizeExceeded);
                    }
                    Err(_) => {}
                }
            }

            let len = blocks.len();
            drop(blocks);
            let blocks = pool.blocks.write();
            if blocks.len() > len {
                // Another thread beat us to it and inserted a fresh block, try to allocate from it.
                match blocks[len].allocate_unchecked(create_info.clone()) {
                    Ok(alloc) => return Ok(alloc),
                    // This can happen if this is the first block that was inserted and when using
                    // the `PoolAllocator<BLOCK_SIZE>` if the allocation size is greater than
                    // `BLOCK_SIZE`.
                    Err(SuballocationCreationError::BlockSizeExceeded) => {
                        return Err(AllocationCreationError::BlockSizeExceeded);
                    }
                    Err(_) => {}
                }
            }

            blocks
        };

        // For bump allocators, first do a garbage sweep and try to allocate again.
        if S::NEEDS_CLEANUP {
            blocks.iter_mut().for_each(Suballocator::cleanup);
            blocks.sort_unstable_by_key(Suballocator::free_size);

            if let Some(block) = blocks.last() {
                if let Ok(alloc) = block.allocate_unchecked(create_info.clone()) {
                    return Ok(alloc);
                }
            }
        }

        if never_allocate {
            return Err(AllocationCreationError::OutOfPoolMemory);
        }

        // The pool doesn't have enough real estate, so we need a new block.
        let block = {
            let block_size = self.block_sizes[pool.memory_type.heap_index as usize];
            let mut i = 0;

            loop {
                let allocate_info = MemoryAllocateInfo {
                    allocation_size: block_size >> i,
                    memory_type_index,
                    dedicated_allocation: None,
                    flags: self.flags,
                    ..Default::default()
                };
                match DeviceMemory::allocate_unchecked(self.device.clone(), allocate_info, None) {
                    Ok(device_memory) => {
                        let property_flags = pool.memory_type.property_flags;
                        let non_coherent_atom_size = (property_flags
                            .contains(ash::vk::MemoryPropertyFlags::HOST_VISIBLE)
                            && !property_flags
                                .contains(ash::vk::MemoryPropertyFlags::HOST_COHERENT))
                        .then_some(
                            self.device
                                .physical_device()
                                .properties()
                                .non_coherent_atom_size,
                        )
                        .and_then(NonZeroU64::new);

                        break S::new(MemoryAlloc {
                            offset: 0,
                            size: device_memory.allocation_size(),
                            allocation_type: AllocationType::Unknown,
                            memory: device_memory.internal_object(),
                            memory_type_index,
                            mapped_ptr: self.mapped_ptr(&device_memory)?,
                            non_coherent_atom_size,
                            parent: AllocParent::Root(Arc::new(device_memory)),
                        });
                    }
                    // Retry up to 3 times, halving the allocation size each time.
                    Err(VulkanError::OutOfHostMemory | VulkanError::OutOfDeviceMemory) if i < 3 => {
                        i += 1;
                    }
                    Err(VulkanError::OutOfHostMemory) => {
                        return Err(AllocationCreationError::OutOfHostMemory);
                    }
                    Err(VulkanError::OutOfDeviceMemory) => {
                        return Err(AllocationCreationError::OutOfDeviceMemory);
                    }
                    Err(VulkanError::TooManyObjects) => {
                        return Err(AllocationCreationError::TooManyObjects);
                    }
                    Err(_) => unreachable!(),
                }
            }
        };

        blocks.push(block);
        let block = blocks.last().unwrap();

        match block.allocate_unchecked(create_info) {
            Ok(alloc) => Ok(alloc),
            // This can happen if the block ended up smaller than advertised because there wasn't
            // enough memory.
            Err(SuballocationCreationError::OutOfRegionMemory) => {
                Err(AllocationCreationError::OutOfDeviceMemory)
            }
            // This can not happen as the block is fresher than Febreze and we're still holding an
            // exclusive lock.
            Err(SuballocationCreationError::FragmentedRegion) => unreachable!(),
            // This can happen if this is the first block that was inserted and when using the
            // `PoolAllocator<BLOCK_SIZE>` if the allocation size is greater than `BLOCK_SIZE`.
            Err(SuballocationCreationError::BlockSizeExceeded) => {
                Err(AllocationCreationError::BlockSizeExceeded)
            }
        }
    }

    fn validate_allocate(&self, create_info: &AllocationCreateInfo<'_>) {
        let &AllocationCreateInfo {
            requirements,
            allocation_type: _,
            usage: _,
            allocate_preference: _,
            dedicated_allocation,
            _ne: _,
        } = create_info;

        SuballocationCreateInfo::from(create_info.clone()).validate();

        assert!(requirements.memory_type_bits != 0);
        assert!(requirements.memory_type_bits < 1 << self.pools.len());

        if let Some(dedicated_allocation) = dedicated_allocation {
            match dedicated_allocation {
                DedicatedAllocation::Buffer(buffer) => {
                    // VUID-VkMemoryDedicatedAllocateInfo-commonparent
                    assert_eq!(&self.device, buffer.device());

                    let required_size = buffer.memory_requirements().size;

                    // VUID-VkMemoryDedicatedAllocateInfo-buffer-02965
                    assert!(requirements.size != required_size);
                }
                DedicatedAllocation::Image(image) => {
                    // VUID-VkMemoryDedicatedAllocateInfo-commonparent
                    assert_eq!(&self.device, image.device());

                    let required_size = image.memory_requirements().size;

                    // VUID-VkMemoryDedicatedAllocateInfo-image-02964
                    assert!(requirements.size != required_size);
                }
            }
        }
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn allocate_unchecked(
        &self,
        create_info: AllocationCreateInfo<'_>,
    ) -> Result<MemoryAlloc, AllocationCreationError> {
        let AllocationCreateInfo {
            requirements:
                MemoryRequirements {
                    size,
                    alignment: _,
                    mut memory_type_bits,
                    mut prefer_dedicated,
                },
            allocation_type: _,
            usage,
            allocate_preference,
            dedicated_allocation,
            _ne: _,
        } = create_info;

        let create_info = SuballocationCreateInfo::from(create_info);

        memory_type_bits &= self.memory_type_bits;

        let mut required_flags = ash::vk::MemoryPropertyFlags::empty();
        let mut preferred_flags = ash::vk::MemoryPropertyFlags::empty();
        let mut not_preferred_flags = ash::vk::MemoryPropertyFlags::empty();

        match usage {
            MemoryUsage::GpuOnly => {
                preferred_flags |= ash::vk::MemoryPropertyFlags::DEVICE_LOCAL;
                not_preferred_flags |= ash::vk::MemoryPropertyFlags::HOST_VISIBLE;
            }
            MemoryUsage::Upload => {
                required_flags |= ash::vk::MemoryPropertyFlags::HOST_VISIBLE;
                preferred_flags |= ash::vk::MemoryPropertyFlags::DEVICE_LOCAL;
                not_preferred_flags |= ash::vk::MemoryPropertyFlags::HOST_CACHED;
            }
            MemoryUsage::Download => {
                required_flags |= ash::vk::MemoryPropertyFlags::HOST_VISIBLE;
                preferred_flags |= ash::vk::MemoryPropertyFlags::HOST_CACHED;
            }
        }

        let mut memory_type_index = self
            .find_memory_type_index(
                memory_type_bits,
                required_flags,
                preferred_flags,
                not_preferred_flags,
            )
            .expect("couldn't find a suitable memory type");

        loop {
            let memory_type = self.pools[memory_type_index as usize].memory_type;
            let block_size = self.block_sizes[memory_type.heap_index as usize];

            let res = match allocate_preference {
                MemoryAllocatePreference::Unknown => {
                    if size > block_size / 2 {
                        prefer_dedicated = true;
                    }
                    if self.device.allocation_count() > self.max_memory_allocation_count
                        && size < block_size
                    {
                        prefer_dedicated = false;
                    }

                    if prefer_dedicated {
                        self.allocate_dedicated(memory_type_index, size, dedicated_allocation)
                            // Fall back to suballocation.
                            .or_else(|e| {
                                if size < block_size {
                                    self.allocate_from_type_unchecked(
                                        memory_type_index,
                                        create_info.clone(),
                                        true, // A dedicated allocation already failed.
                                    )
                                    .map_err(|_| e)
                                } else {
                                    Err(e)
                                }
                            })
                    } else {
                        self.allocate_from_type_unchecked(
                            memory_type_index,
                            create_info.clone(),
                            false,
                        )
                        // Fall back to dedicated allocation. It is possible that the 1/8 block size
                        // that was tried was greater than the allocation size, so there's hope.
                        .or_else(|_| {
                            self.allocate_dedicated(memory_type_index, size, dedicated_allocation)
                        })
                    }
                }
                MemoryAllocatePreference::AlwaysAllocate => {
                    self.allocate_dedicated(memory_type_index, size, dedicated_allocation)
                }
                MemoryAllocatePreference::NeverAllocate => {
                    if size <= block_size {
                        self.allocate_from_type_unchecked(
                            memory_type_index,
                            create_info.clone(),
                            true,
                        )
                    } else {
                        Err(AllocationCreationError::OutOfPoolMemory)
                    }
                }
            };

            match res {
                Ok(alloc) => return Ok(alloc),
                // This is not recoverable.
                Err(AllocationCreationError::BlockSizeExceeded) => {
                    return Err(AllocationCreationError::BlockSizeExceeded);
                }
                // Try a different memory type.
                Err(e) => {
                    memory_type_bits &= !(1 << memory_type_index);
                    memory_type_index = self
                        .find_memory_type_index(
                            memory_type_bits,
                            required_flags,
                            preferred_flags,
                            not_preferred_flags,
                        )
                        .ok_or(e)?;
                }
            }
        }
    }

    fn find_memory_type_index(
        &self,
        memory_type_bits: u32,
        required_flags: ash::vk::MemoryPropertyFlags,
        preferred_flags: ash::vk::MemoryPropertyFlags,
        not_preferred_flags: ash::vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        self.pools
            .iter()
            .map(|pool| pool.memory_type.property_flags)
            .enumerate()
            // Filter out memory types which are supported by the memory type bits and have the
            // required flags set.
            .filter(|&(index, flags)| {
                memory_type_bits & (1 << index) != 0 && required_flags & flags == required_flags
            })
            // Rank memory types with more of the preferred flags higher, and ones with more of the
            // not preferred flags lower.
            .min_by_key(|&(_, flags)| {
                (preferred_flags & !flags).as_raw().count_ones()
                    + (not_preferred_flags & flags).as_raw().count_ones()
            })
            .map(|(index, _)| index as u32)
    }

    unsafe fn allocate_dedicated(
        &self,
        memory_type_index: u32,
        allocation_size: DeviceSize,
        mut dedicated_allocation: Option<DedicatedAllocation<'_>>,
    ) -> Result<MemoryAlloc, AllocationCreationError> {
        if !self.dedicated_allocation {
            dedicated_allocation = None;
        }

        let is_dedicated = dedicated_allocation.is_some();
        let allocate_info = MemoryAllocateInfo {
            allocation_size,
            memory_type_index,
            dedicated_allocation,
            flags: self.flags,
            ..Default::default()
        };
        let device_memory =
            DeviceMemory::allocate_unchecked(self.device.clone(), allocate_info, None).map_err(
                |e| match e {
                    VulkanError::OutOfHostMemory => AllocationCreationError::OutOfHostMemory,
                    VulkanError::OutOfDeviceMemory => AllocationCreationError::OutOfDeviceMemory,
                    VulkanError::TooManyObjects => AllocationCreationError::TooManyObjects,
                    _ => unreachable!(),
                },
            )?;
        let property_flags = self.pools[memory_type_index as usize]
            .memory_type
            .property_flags;
        let non_coherent_atom_size = (property_flags
            .contains(ash::vk::MemoryPropertyFlags::HOST_VISIBLE)
            && !property_flags.contains(ash::vk::MemoryPropertyFlags::HOST_COHERENT))
        .then_some(
            self.device
                .physical_device()
                .properties()
                .non_coherent_atom_size,
        )
        .and_then(NonZeroU64::new);

        Ok(MemoryAlloc {
            offset: 0,
            size: allocation_size,
            allocation_type: AllocationType::Unknown,
            memory: device_memory.internal_object(),
            memory_type_index,
            mapped_ptr: self.mapped_ptr(&device_memory)?,
            non_coherent_atom_size,
            parent: if is_dedicated {
                AllocParent::Dedicated(device_memory)
            } else {
                AllocParent::Root(Arc::new(device_memory))
            },
        })
    }

    unsafe fn mapped_ptr(
        &self,
        device_memory: &DeviceMemory,
    ) -> Result<Option<NonNull<c_void>>, AllocationCreationError> {
        let memory_type_index = device_memory.memory_type_index();

        if self.pools[memory_type_index as usize]
            .memory_type
            .property_flags
            .contains(ash::vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            let fns = self.device.fns();
            let mut output = MaybeUninit::uninit();
            // This is always valid because we are mapping the whole range.
            (fns.v1_0.map_memory)(
                self.device.internal_object(),
                device_memory.internal_object(),
                0,
                ash::vk::WHOLE_SIZE,
                ash::vk::MemoryMapFlags::empty(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(|e| match e.into() {
                VulkanError::OutOfHostMemory => AllocationCreationError::OutOfHostMemory,
                VulkanError::OutOfDeviceMemory => AllocationCreationError::OutOfDeviceMemory,
                VulkanError::MemoryMapFailed => AllocationCreationError::MemoryMapFailed,
                _ => unreachable!(),
            })?;

            Ok(NonNull::new(output.assume_init()))
        } else {
            Ok(None)
        }
    }
}

unsafe impl<S: Suballocator> MemoryAllocator for GenericMemoryAllocator<S> {
    /// Allocates memory from a specific memory type.
    ///
    /// # Panics
    ///
    /// - Panics if `memory_type_index` is not less than the number of available memory types.
    /// - Panics if `memory_type_index` refers to a memory type which has the [`protected`] flag set
    ///   and the [`protected_memory`] feature is not enabled on the device.
    /// - Panics if `create_info.size` is greater than the block size corresponding to the heap that
    ///   the memory type corresponding to `memory_type_index` resides in.
    /// - Panics if `create_info.size` is zero.
    /// - Panics if `create_info.alignment` is zero.
    /// - Panics if `create_info.alignment` is not a power of two.
    ///
    /// # Errors
    ///
    /// - Returns an error if allocating a new block is required and failed. This can be one of the
    ///   OOM errors or [`TooManyObjects`].
    /// - Returns [`BlockSizeExceeded`] if `S` is `PoolAllocator<BLOCK_SIZE>` and `create_info.size`
    ///   is greater than `BLOCK_SIZE`.
    ///
    /// [`protected`]: super::MemoryPropertyFlags::protected
    /// [`protected_memory`]: crate::device::Features::protected_memory
    /// [`TooManyObjects`]: AllocationCreationError::TooManyObjects
    /// [`BlockSizeExceeded`]: AllocationCreationError::BlockSizeExceeded
    fn allocate_from_type(
        &self,
        memory_type_index: u32,
        create_info: SuballocationCreateInfo,
    ) -> Result<MemoryAlloc, AllocationCreationError> {
        self.validate_allocate_from_type(memory_type_index, &create_info);

        if self.pools[memory_type_index as usize]
            .memory_type
            .property_flags
            .contains(ash::vk::MemoryPropertyFlags::LAZILY_ALLOCATED)
        {
            return unsafe { self.allocate_dedicated(memory_type_index, create_info.size, None) };
        }

        unsafe { self.allocate_from_type_unchecked(memory_type_index, create_info, false) }
    }

    /// Allocates memory according to requirements.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.requirements.size` is zero.
    /// - Panics if `create_info.requirements.alignment` is zero.
    /// - Panics if `create_info.requirements.alignment` is not a power of two.
    /// - Panics if `create_info.requirements.memory_type_bits` is zero.
    /// - Panics if `create_info.requirements.memory_type_bits` is not less than 2<sup>*n*</sup>
    ///   where *n* is the number of available memory types.
    /// - Panics if `create_info.dedicated_allocation` is `Some` and
    ///   `create_info.requirements.size` doesn't match the memory requirements of the resource.
    /// - Panics if finding a suitable memory type failed. This only happens if the
    ///   `create_info.requirements` correspond to those of an optimal image but
    ///   `create_info.usage` is not [`MemoryUsage::GpuOnly`].
    ///
    /// # Errors
    ///
    /// - Returns an error if allocating a new block is required and failed. This can be one of the
    ///   OOM errors or [`TooManyObjects`].
    /// - Returns [`BlockSizeExceeded`] if `S` is `PoolAllocator<BLOCK_SIZE>` and `create_info.size`
    ///   is greater than `BLOCK_SIZE` and a dedicated allocation was not created.
    /// - Returns [`OutOfPoolMemory`] if `create_info.allocate_preference` is
    ///   [`MemoryAllocatePreference::NeverAllocate`] and `create_info.requirements.size` is greater
    ///   than the block size for all heaps of suitable memory types.
    /// - Returns `OutOfPoolMemory` if `create_info.allocate_preference` is
    ///   [`MemoryAllocatePreference::NeverAllocate`] and none of the pools of suitable memory
    ///   types have enough free space.
    ///
    /// [`device_local`]: MemoryPropertyFlags::device_local
    /// [`host_visible`]: MemoryPropertyFlags::host_visible
    /// [`NoSuitableMemoryTypes`]: AllocationCreationError::NoSuitableMemoryTypes
    /// [`TooManyObjects`]: AllocationCreationError::TooManyObjects
    /// [`BlockSizeExceeded`]: AllocationCreationError::BlockSizeExceeded
    /// [`OutOfPoolMemory`]: AllocationCreationError::OutOfPoolMemory
    fn allocate(
        &self,
        create_info: AllocationCreateInfo<'_>,
    ) -> Result<MemoryAlloc, AllocationCreationError> {
        self.validate_allocate(&create_info);

        unsafe { self.allocate_unchecked(create_info) }
    }

    fn create_buffer(
        &self,
        create_info: UnsafeBufferCreateInfo,
        usage: MemoryUsage,
        allocate_preference: MemoryAllocatePreference,
    ) -> Result<
        Result<(Arc<UnsafeBuffer>, MemoryAlloc), AllocationCreationError>,
        BufferCreationError,
    > {
        let buffer = UnsafeBuffer::new(self.device().clone(), create_info)?;
        let create_info = AllocationCreateInfo {
            requirements: buffer.memory_requirements(),
            allocation_type: AllocationType::Linear,
            usage,
            allocate_preference,
            dedicated_allocation: Some(DedicatedAllocation::Buffer(&buffer)),
            ..Default::default()
        };

        Ok(match unsafe { self.allocate_unchecked(create_info) } {
            Ok(alloc) => {
                unsafe { buffer.bind_memory(alloc.device_memory(), alloc.offset()) }?;

                Ok((buffer, alloc))
            }
            Err(e) => Err(e),
        })
    }

    /// Conveniece method to create a (non-sparse) `UnsafeImage`, allocate memory for it, and bind
    /// the memory to it.
    ///
    /// # Panics
    ///
    /// - Panics if `create_info.tiling` is [`ImageTiling::Optimal`] and `usage` is not
    ///   [`MemoryUsage::GpuOnly`].
    fn create_image(
        &self,
        create_info: UnsafeImageCreateInfo,
        usage: MemoryUsage,
        allocate_preference: MemoryAllocatePreference,
    ) -> Result<Result<(Arc<UnsafeImage>, MemoryAlloc), AllocationCreationError>, ImageCreationError>
    {
        let allocation_type = create_info.tiling.into();
        let image = UnsafeImage::new(self.device().clone(), create_info)?;
        let create_info = AllocationCreateInfo {
            requirements: image.memory_requirements(),
            allocation_type,
            usage,
            allocate_preference,
            dedicated_allocation: Some(DedicatedAllocation::Image(&image)),
            ..Default::default()
        };

        Ok(match unsafe { self.allocate_unchecked(create_info) } {
            Ok(alloc) => {
                unsafe { image.bind_memory(alloc.device_memory(), alloc.offset()) }?;

                Ok((image, alloc))
            }
            Err(e) => Err(e),
        })
    }
}

unsafe impl<S: Suballocator> DeviceOwned for GenericMemoryAllocator<S> {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

/// Parameters to create a new [`GenericMemoryAllocator`].
#[derive(Clone, Debug)]
pub struct GenericMemoryAllocatorCreateInfo<'b> {
    /// Lets you configure the block sizes for various heap size classes.
    ///
    /// Each entry is a pair of the threshold for the heap size and the block size that should be
    /// used for that heap. Must be sorted by threshold and all thresholds must be unique. Must
    /// contain a baseline threshold of 0.
    ///
    /// The allocator keeps a pool of [`DeviceMemory`] blocks for each memory type, so each memory
    /// type that resides in a heap whose size crosses one of the thresholds will use the
    /// corresponding block size. If multiple thresholds apply to a given heap, the block size
    /// corresponding to the largest threshold is chosen.
    ///
    /// The block size is going to be the maximum size of a `DeviceMemory` block that is tried. If
    /// allocating a block with the size fails, the allocator tries 1/2, 1/4 and 1/8 of the block
    /// size in that order until one succeeds, else a dedicated allocation is attempted for the
    /// allocation. If an allocation is created with a size greater than half the block size it is
    /// always made a dedicated allocation. All of this doesn't apply when using
    /// [`MemoryAllocatePreference::NeverAllocate`] however.
    ///
    /// The default value is `&[]`, which must be overridden.
    pub block_sizes: &'b [(Threshold, BlockSize)],

    /// Whether the allocator should use the dedicated allocation APIs.
    ///
    /// This means that when the allocator dedices that an allocation should not be suballocated,
    /// but rather have its own block of [`DeviceMemory`], that that allocation will be made a
    /// dedicated allocation. Otherwise they are still made free-standing ([root]) allocations,
    /// just not [dedicated] ones.
    ///
    /// Dedicated allocations are an optimization which may result in better performance, so there
    /// really is no reason to disable this option, unless the restrictions that they bring with
    /// them are a problem. Namely, a dedicated allocation must only be used for the resource it
    /// was created for. Meaning that [reusing the memory] for something else is not possible,
    /// [suballocating it] is not possible, and [aliasing it] is also not possible.
    ///
    /// This option is silently ignored (treated as `false`) if the device API version is below 1.1
    /// and the [`khr_dedicated_allocation`] extension is not enabled on the device.
    ///
    /// The default value is `true`.
    ///
    /// [root]: MemoryAlloc::is_root
    /// [dedicated]: MemoryAlloc::is_dedicated
    /// [reusing the memory]: MemoryAlloc::try_unwrap
    /// [suballocating it]: Suballocator
    /// [aliasing it]: MemoryAlloc::alias
    /// [`khr_dedicated_allocation`]: crate::device::DeviceExtensions::khr_dedicated_allocation
    pub dedicated_allocation: bool,

    /// Whether the allocator should allocate the [`DeviceMemory`] blocks with the
    /// [`device_address`] flag set.
    ///
    /// This is required if you want to allocate memory for buffers that have the
    /// [`shader_device_address`] usage set. For this option too, there is no reason to disable it.
    ///
    /// This option is silently ignored (treated as `false`) if the [`buffer_device_address`]
    /// feature is not enabled on the device or if the [`ext_buffer_device_address`] extension is
    /// enabled on the device. It is also ignored if the device API version is below 1.1 and the
    /// [`khr_device_group`] extension is not enabled on the device.
    ///
    /// The default value is `true`.
    ///
    /// [`device_address`]: MemoryAllocateFlags::device_address
    /// [`shader_device_address`]: crate::buffer::BufferUsage::shader_device_address
    /// [`buffer_device_address`]: crate::device::Features::buffer_device_address
    /// [`ext_buffer_device_address`]: crate::device::DeviceExtensions::ext_buffer_device_address
    /// [`khr_device_group`]: crate::device::DeviceExtensions::khr_device_group
    pub device_address: bool,

    pub _ne: crate::NonExhaustive,
}

pub type Threshold = DeviceSize;

pub type BlockSize = DeviceSize;

impl Default for GenericMemoryAllocatorCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        GenericMemoryAllocatorCreateInfo {
            block_sizes: &[],
            dedicated_allocation: true,
            device_address: true,
            _ne: crate::NonExhaustive(()),
        }
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
/// # Implementing the trait
///
/// Please don't.
///
/// [allocations]: MemoryAlloc
/// [pages]: self#pages
pub unsafe trait Suballocator: DeviceOwned {
    /// Whether this allocator needs to block or not.
    ///
    /// This is used by the [`GenericMemoryAllocator`] to specialize the allocation strategy to the
    /// suballocator at compile time.
    const IS_BLOCKING: bool;

    /// Whether the allocator needs [`cleanup`] to be called before memory can be released.
    ///
    /// This is used by the [`GenericMemoryAllocator`] to specialize the allocation strategy to the
    /// suballocator at compile time.
    ///
    /// [`cleanup`]: Self::cleanup
    const NEEDS_CLEANUP: bool;

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
    ) -> Result<MemoryAlloc, SuballocationCreationError> {
        create_info.validate();

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
    ) -> Result<MemoryAlloc, SuballocationCreationError>;

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

impl From<AllocationCreateInfo<'_>> for SuballocationCreateInfo {
    #[inline]
    fn from(create_info: AllocationCreateInfo<'_>) -> Self {
        SuballocationCreateInfo {
            size: create_info.requirements.size,
            alignment: create_info.requirements.alignment,
            allocation_type: create_info.allocation_type,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl SuballocationCreateInfo {
    fn validate(&self) {
        assert!(self.size > 0);
        assert!(self.alignment > 0);
        assert!(self.alignment.is_power_of_two());
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

impl From<ImageTiling> for AllocationType {
    #[inline]
    fn from(tiling: ImageTiling) -> Self {
        match tiling {
            ImageTiling::Optimal => AllocationType::NonLinear,
            ImageTiling::Linear => AllocationType::Linear,
        }
    }
}

/// Error that can be returned when using a [suballocator].
///
/// [suballocator]: Suballocator
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SuballocationCreationError {
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
    BlockSizeExceeded,
}

impl Display for SuballocationCreationError {
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

impl Error for SuballocationCreationError {}

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
    device_memory: Arc<DeviceMemory>,
    buffer_image_granularity: DeviceSize,
    // Total memory remaining in the region.
    free_size: AtomicU64,
    inner: Mutex<FreeListAllocatorInner>,
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
    /// [buffer-image granularity]: self#buffer-image-granularity
    /// [dedicated allocation]: MemoryAlloc::is_dedicated
    #[inline]
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
        let free_size = AtomicU64::new(region.size);

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
        let inner = Mutex::new(FreeListAllocatorInner { nodes, free_list });

        Arc::new(FreeListAllocator {
            region,
            device_memory,
            buffer_image_granularity,
            free_size,
            inner,
        })
    }

    fn free(&self, id: SlotId) {
        let mut inner = self.inner.lock();
        self.free_size
            .fetch_add(inner.nodes.get(id).size, Ordering::Release);
        inner.nodes.get_mut(id).ty = SuballocationType::Free;
        inner.coalesce(id);
        inner.free(id);
    }
}

unsafe impl Suballocator for Arc<FreeListAllocator> {
    const IS_BLOCKING: bool = true;

    const NEEDS_CLEANUP: bool = false;

    #[inline]
    fn new(region: MemoryAlloc) -> Self {
        FreeListAllocator::new(region)
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
    /// # Errors
    ///
    /// - Returns [`OutOfRegionMemory`] if there are no free suballocations large enough so satisfy
    ///   the request.
    /// - Returns [`FragmentedRegion`] if a suballocation large enough to satisfy the request could
    ///   have been formed, but wasn't because of [external fragmentation].
    ///
    /// [region]: Suballocator#regions
    /// [`allocate`]: Suballocator::allocate
    /// [`OutOfRegionMemory`]: SuballocationCreationError::OutOfRegionMemory
    /// [`FragmentedRegion`]: SuballocationCreationError::FragmentedRegion
    /// [external fragmentation]: self#external-fragmentation
    #[inline]
    unsafe fn allocate_unchecked(
        &self,
        create_info: SuballocationCreateInfo,
    ) -> Result<MemoryAlloc, SuballocationCreationError> {
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

        let alignment = DeviceSize::max(
            alignment,
            self.region
                .non_coherent_atom_size
                .map(NonZeroU64::get)
                .unwrap_or(1),
        );
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
                            self.buffer_image_granularity,
                        ) && has_granularity_conflict(prev.ty, allocation_type)
                        {
                            offset = align_up(offset, self.buffer_image_granularity);
                        }
                    }

                    if offset + size <= suballoc.offset + suballoc.size {
                        inner.allocate(id);
                        inner.split(id, offset, size);
                        inner.nodes.get_mut(id).ty = allocation_type.into();
                        self.free_size.fetch_sub(size, Ordering::Release);

                        return Ok(MemoryAlloc {
                            offset,
                            size,
                            allocation_type,
                            memory: self.region.memory,
                            memory_type_index: self.region.memory_type_index,
                            mapped_ptr: self.region.mapped_ptr.and_then(|ptr| {
                                NonNull::new(
                                    ptr.as_ptr().add((offset - self.region.offset) as usize),
                                )
                            }),
                            non_coherent_atom_size: self.region.non_coherent_atom_size,
                            parent: AllocParent::FreeList {
                                allocator: self.clone(),
                                id,
                            },
                        });
                    }
                }

                // There is not enough space due to alignment requirements.
                Err(SuballocationCreationError::OutOfRegionMemory)
            }
            // There would be enough space if the region wasn't so fragmented. :(
            Some(_) if self.free_size() >= size => {
                Err(SuballocationCreationError::FragmentedRegion)
            }
            // There is not enough space.
            Some(_) => Err(SuballocationCreationError::OutOfRegionMemory),
            // There is no space at all.
            None => Err(SuballocationCreationError::OutOfRegionMemory),
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
struct FreeListAllocatorInner {
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

impl FreeListAllocatorInner {
    /// Removes the target suballocation from the free-list. The free-list must contain it.
    fn allocate(&mut self, node_id: SlotId) {
        debug_assert!(self.free_list.contains(&node_id));

        let node = self.nodes.get(node_id);

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
    device_memory: Arc<DeviceMemory>,
    buffer_image_granularity: DeviceSize,
    // Total memory remaining in the region.
    free_size: AtomicU64,
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
    /// - Panics if `region` is a [dedicated allocation].
    ///
    /// [region]: Suballocator#regions
    /// [buffer-image granularity]: self#buffer-image-granularity
    /// [dedicated allocation]: MemoryAlloc::is_dedicated
    #[inline]
    pub fn new(region: MemoryAlloc) -> Arc<Self> {
        const EMPTY_FREE_LIST: Vec<DeviceSize> = Vec::new();

        let max_order = (region.size / Self::MIN_NODE_SIZE).trailing_zeros() as usize;

        assert!(region.allocation_type == AllocationType::Unknown);
        assert!(region.size.is_power_of_two());
        assert!(region.size >= Self::MIN_NODE_SIZE && max_order < Self::MAX_ORDERS);

        let device_memory = region
            .root()
            .expect("dedicated allocations can't be suballocated")
            .clone();
        let buffer_image_granularity = device_memory
            .device()
            .physical_device()
            .properties()
            .buffer_image_granularity;
        let free_size = AtomicU64::new(region.size);

        let mut free_list = ArrayVec::new(max_order + 1, [EMPTY_FREE_LIST; Self::MAX_ORDERS]);
        // The root node has the lowest offset and highest order, so it's the whole region.
        free_list[max_order].push(region.offset);
        let inner = Mutex::new(BuddyAllocatorInner { free_list });

        Arc::new(BuddyAllocator {
            region,
            device_memory,
            buffer_image_granularity,
            free_size,
            inner,
        })
    }

    fn free(&self, min_order: usize, mut offset: DeviceSize) {
        let mut inner = self.inner.lock();

        // Try to coalesce nodes while incrementing the order.
        for (order, free_list) in inner.free_list.iter_mut().enumerate().skip(min_order) {
            let size = Self::MIN_NODE_SIZE << order;
            let buddy_offset = ((offset - self.region.offset) ^ size) + self.region.offset;

            match free_list.binary_search(&buddy_offset) {
                // If the buddy is in the free-list, we can coalesce.
                Ok(index) => {
                    free_list.remove(index);
                    offset = DeviceSize::min(offset, buddy_offset);
                }
                // Otherwise free the node.
                Err(_) => {
                    let index = match free_list.binary_search(&offset) {
                        Ok(index) => index,
                        Err(index) => index,
                    };
                    free_list.insert(index, offset);
                    self.free_size
                        .fetch_add(Self::MIN_NODE_SIZE << min_order, Ordering::Release);

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
    /// # Errors
    ///
    /// - Returns [`OutOfRegionMemory`] if there are no free nodes large enough so satisfy the
    ///   request.
    /// - Returns [`FragmentedRegion`] if a node large enough to satisfy the request could have
    ///   been formed, but wasn't because of [external fragmentation].
    ///
    /// [region]: Suballocator#regions
    /// [`allocate`]: Suballocator::allocate
    /// [`OutOfRegionMemory`]: SuballocationCreationError::OutOfRegionMemory
    /// [`FragmentedRegion`]: SuballocationCreationError::FragmentedRegion
    /// [external fragmentation]: self#external-fragmentation
    #[inline]
    unsafe fn allocate_unchecked(
        &self,
        create_info: SuballocationCreateInfo,
    ) -> Result<MemoryAlloc, SuballocationCreationError> {
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
            size = align_up(size, self.buffer_image_granularity);
            alignment = DeviceSize::max(alignment, self.buffer_image_granularity);
        }

        let size = DeviceSize::max(size, BuddyAllocator::MIN_NODE_SIZE).next_power_of_two();
        let alignment = DeviceSize::max(
            alignment,
            self.region
                .non_coherent_atom_size
                .map(NonZeroU64::get)
                .unwrap_or(1),
        );
        let min_order = (size / BuddyAllocator::MIN_NODE_SIZE).trailing_zeros() as usize;
        let mut inner = self.inner.lock();

        // Start searching at the lowest possible order going up.
        for (order, free_list) in inner.free_list.iter_mut().enumerate().skip(min_order) {
            for (index, &offset) in free_list.iter().enumerate() {
                if offset % alignment == 0 {
                    free_list.remove(index);

                    // Go in the opposite direction, splitting nodes from higher orders. The lowest
                    // order doesn't need any splitting.
                    for (order, free_list) in inner
                        .free_list
                        .iter_mut()
                        .enumerate()
                        .skip(min_order)
                        .take(order - min_order)
                        .rev()
                    {
                        let size = BuddyAllocator::MIN_NODE_SIZE << order;
                        let right_child = offset + size;

                        // Insert the right child in sorted order.
                        let index = match free_list.binary_search(&right_child) {
                            Ok(index) => index,
                            Err(index) => index,
                        };
                        free_list.insert(index, right_child);

                        // Repeat splitting for the left child if required in the next loop turn.
                    }

                    self.free_size.fetch_sub(size, Ordering::Release);

                    return Ok(MemoryAlloc {
                        offset,
                        size: create_info.size,
                        allocation_type,
                        memory: self.region.memory,
                        memory_type_index: self.region.memory_type_index,
                        mapped_ptr: self.region.mapped_ptr.and_then(|ptr| {
                            NonNull::new(ptr.as_ptr().add((offset - self.region.offset) as usize))
                        }),
                        non_coherent_atom_size: self.region.non_coherent_atom_size,
                        parent: AllocParent::Buddy {
                            allocator: self.clone(),
                            order: min_order,
                        },
                    });
                }
            }
        }

        if prev_power_of_two(self.free_size()) >= create_info.size {
            // A node large enough could be formed if the region wasn't so fragmented.
            Err(SuballocationCreationError::FragmentedRegion)
        } else {
            Err(SuballocationCreationError::OutOfRegionMemory)
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
    /// [internal fragmentation]: self#internal-fragmentation
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
struct BuddyAllocatorInner {
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
    /// - Panics if `region` is a [dedicated allocation].
    ///
    /// [region]: Suballocator#regions
    /// [dedicated allocation]: MemoryAlloc::is_dedicated
    #[inline]
    pub fn new(
        region: MemoryAlloc,
        #[cfg(test)] buffer_image_granularity: DeviceSize,
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
            1,
        )
    }

    /// Creates a new suballocation within the [region] without checking the parameters.
    ///
    /// See [`allocate`] for the safe version.
    ///
    /// > **Note**: `create_info.allocation_type` is silently ignored because all suballocations
    /// > inherit the same allocation type from the allocator.
    ///
    /// # Safety
    ///
    /// - `create_info.size` must not be zero.
    /// - `create_info.alignment` must not be zero.
    /// - `create_info.alignment` must be a power of two.
    ///
    /// # Errors
    ///
    /// - Returns [`OutOfRegionMemory`] if the [free-list] is empty.
    /// - Returns [`OutOfRegionMemory`] if the allocation can't fit inside a block. Only the first
    ///   block in the free-list is tried, which means that if one block isn't usable due to
    ///   [internal fragmentation] but a different one would be, you still get this error. See the
    ///   [type-level documentation] for details on how to properly configure your allocator.
    ///
    /// [region]: Suballocator#regions
    /// [`allocate`]: Suballocator::allocate
    /// [`OutOfRegionMemory`]: SuballocationCreationError::OutOfRegionMemory
    /// [free-list]: Suballocator#free-lists
    /// [internal fragmentation]: self#internal-fragmentation
    /// [type-level documentation]: PoolAllocator
    #[inline]
    unsafe fn allocate_unchecked(
        &self,
        create_info: SuballocationCreateInfo,
    ) -> Result<MemoryAlloc, SuballocationCreationError> {
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
    block_size: DeviceSize,
    // Unsorted list of free block indices.
    free_list: ArrayQueue<DeviceSize>,
}

impl PoolAllocatorInner {
    fn new(
        region: MemoryAlloc,
        mut block_size: DeviceSize,
        #[cfg(test)] buffer_image_granularity: DeviceSize,
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
            block_size,
            free_list,
        }
    }

    unsafe fn allocate_unchecked(
        self: Arc<Self>,
        create_info: SuballocationCreateInfo,
    ) -> Result<MemoryAlloc, SuballocationCreationError> {
        let SuballocationCreateInfo {
            size,
            alignment,
            allocation_type: _,
            _ne: _,
        } = create_info;

        let alignment = DeviceSize::max(
            alignment,
            self.region
                .non_coherent_atom_size
                .map(NonZeroU64::get)
                .unwrap_or(1),
        );
        let index = self
            .free_list
            .pop()
            .ok_or(SuballocationCreationError::OutOfRegionMemory)?;
        let unaligned_offset = index * self.block_size;
        let offset = align_up(unaligned_offset, alignment);

        if offset + size > unaligned_offset + self.block_size {
            self.free_list.push(index).unwrap();

            return Err(SuballocationCreationError::BlockSizeExceeded);
        }

        Ok(MemoryAlloc {
            offset,
            size,
            allocation_type: self.region.allocation_type,
            memory: self.region.memory,
            memory_type_index: self.region.memory_type_index,
            mapped_ptr: self.region.mapped_ptr.and_then(|ptr| {
                NonNull::new(ptr.as_ptr().add((offset - self.region.offset) as usize))
            }),
            non_coherent_atom_size: self.region.non_coherent_atom_size,
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
    device_memory: Arc<DeviceMemory>,
    buffer_image_granularity: DeviceSize,
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
    ///
    /// [region]: Suballocator#regions
    /// [dedicated allocation]: MemoryAlloc::is_dedicated
    #[inline]
    pub fn new(region: MemoryAlloc) -> Arc<Self> {
        let device_memory = region
            .root()
            .expect("dedicated allocations can't be suballocated")
            .clone();
        let buffer_image_granularity = device_memory
            .device()
            .physical_device()
            .properties()
            .buffer_image_granularity;
        let state = AtomicU64::new(region.allocation_type as u64);

        Arc::new(BumpAllocator {
            region,
            device_memory,
            buffer_image_granularity,
            state,
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

unsafe impl Suballocator for Arc<BumpAllocator> {
    const IS_BLOCKING: bool = false;

    const NEEDS_CLEANUP: bool = true;

    #[inline]
    fn new(region: MemoryAlloc) -> Self {
        BumpAllocator::new(region)
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
    /// # Errors
    ///
    /// - Returns [`OutOfRegionMemory`] if the requested allocation can't fit in the free space
    ///   remaining in the region.
    ///
    /// [region]: Suballocator#regions
    /// [`allocate`]: Suballocator::allocate
    /// [`OutOfRegionMemory`]: SuballocationCreationError::OutOfRegionMemory
    #[inline]
    unsafe fn allocate_unchecked(
        &self,
        create_info: SuballocationCreateInfo,
    ) -> Result<MemoryAlloc, SuballocationCreationError> {
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

        let alignment = DeviceSize::max(
            alignment,
            self.region
                .non_coherent_atom_size
                .map(NonZeroU64::get)
                .unwrap_or(1),
        );
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
                && are_blocks_on_same_page(prev_end, 0, offset, self.buffer_image_granularity)
                && has_granularity_conflict(prev_alloc_type, allocation_type)
            {
                offset = align_up(offset, self.buffer_image_granularity);
            }

            let free_start = offset - self.region.offset + size;

            if free_start > self.region.size {
                return Err(SuballocationCreationError::OutOfRegionMemory);
            }

            let new_state = free_start << 2 | allocation_type as u64;

            match self.state.compare_exchange_weak(
                state,
                new_state,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Ok(MemoryAlloc {
                        offset,
                        size,
                        allocation_type,
                        memory: self.region.memory,
                        memory_type_index: self.region.memory_type_index,
                        mapped_ptr: self.region.mapped_ptr.and_then(|ptr| {
                            NonNull::new(ptr.as_ptr().add((offset - self.region.offset) as usize))
                        }),
                        non_coherent_atom_size: self.region.non_coherent_atom_size,
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

        let allocator = dummy_allocator!(FreeListAllocator, REGION_SIZE);
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

        let allocator = dummy_allocator!(FreeListAllocator, REGION_SIZE);
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

        let allocator = dummy_allocator!(FreeListAllocator, REGION_SIZE, GRANULARITY);
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

            PoolAllocator::new(device_memory.into(), 1)
        }

        let (device, _) = gfx_dev_and_queue!();

        assert_should_panic!({ dummy_allocator(device.clone(), BLOCK_SIZE - 1) });

        let allocator = dummy_allocator(device.clone(), 2 * BLOCK_SIZE - 1);
        {
            let alloc = allocator.allocate(DUMMY_INFO).unwrap();
            assert!(allocator.allocate(DUMMY_INFO).is_err());

            drop(alloc);
            let _alloc = allocator.allocate(DUMMY_INFO).unwrap();
        }

        let allocator = dummy_allocator(device, 2 * BLOCK_SIZE);
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

            PoolAllocator::<BLOCK_SIZE>::new(device_memory.into(), 1)
        };

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
            let mut region = MemoryAlloc::from(device_memory);
            unsafe { region.set_allocation_type(allocation_type) };

            PoolAllocator::new(region, 256)
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

        let allocator = dummy_allocator!(BuddyAllocator, REGION_SIZE);

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

        let allocator = dummy_allocator!(BuddyAllocator, REGION_SIZE, GRANULARITY);

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

        let allocator = dummy_allocator!(BumpAllocator, INFO.alignment * 10);

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

        let mut allocator = dummy_allocator!(BumpAllocator, REGION_SIZE);

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

    macro_rules! dummy_allocator {
        ($type:ty, $size:expr) => {
            dummy_allocator!($type, $size, 1)
        };
        ($type:ty, $size:expr, $granularity:expr) => {
            dummy_allocator!($type, $size, $granularity, AllocationType::Unknown)
        };
        ($type:ty, $size:expr, $granularity:expr, $allocation_type:expr) => {{
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
            let mut allocator = <$type>::new(device_memory.into());
            Arc::get_mut(&mut allocator)
                .unwrap()
                .buffer_image_granularity = $granularity;

            allocator
        }};
    }

    pub(self) use dummy_allocator;
}
