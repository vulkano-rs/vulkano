//! Device memory allocation and memory pools.
//!
//! By default, memory allocation is automatically handled by the vulkano library when you create
//! a buffer or an image. But if you want more control, you have the possibility to customise the
//! memory allocation strategy.
//!
//! # Memory types and heaps
//!
//! A physical device is composed of one or more **memory heaps**. A memory heap is a pool of
//! memory that can be allocated.
//!
//! ```
//! // Enumerating memory heaps.
//! # let physical_device: vulkano::device::physical::PhysicalDevice = return;
//! for (index, heap) in physical_device
//!     .memory_properties()
//!     .memory_heaps
//!     .iter()
//!     .enumerate()
//! {
//!     println!("Heap #{:?} has a capacity of {:?} bytes", index, heap.size);
//! }
//! ```
//!
//! However you can't allocate directly from a memory heap. A memory heap is shared amongst one or
//! multiple **memory types**, which you can allocate memory from. Each memory type has different
//! characteristics.
//!
//! A memory type may or may not be visible to the host. In other words, it may or may not be
//! directly writable by the CPU. A memory type may or may not be device-local. A device-local
//! memory type has a much quicker access time from the GPU than a non-device-local type. Note
//! that non-device-local memory types are still accessible by the device, they are just slower.
//!
//! ```
//! // Enumerating memory types.
//! # let physical_device: vulkano::device::physical::PhysicalDevice = return;
//! for ty in physical_device.memory_properties().memory_types.iter() {
//!     println!("Memory type belongs to heap #{:?}", ty.heap_index);
//!     println!("Property flags: {:?}", ty.property_flags);
//! }
//! ```
//!
//! Memory types are order from "best" to "worse". In other words, the implementation prefers that
//! you use the memory types that are earlier in the list. This means that selecting a memory type
//! should always be done by enumerating them and taking the first one that matches our criteria.
//!
//! ## In practice
//!
//! In practice, desktop machines usually have two memory heaps: one that represents the RAM of
//! the CPU, and one that represents the RAM of the GPU. The CPU's RAM is host-accessible but not
//! device-local, while the GPU's RAM is not host-accessible but is device-local.
//!
//! Mobile machines usually have a single memory heap that is "equally local" to both the CPU and
//! the GPU. It is both host-accessible and device-local.
//!
//! # Allocating memory and memory pools
//!
//! Allocating memory can be done by calling `DeviceMemory::allocate()`.
//!
//! Here is an example:
//!
//! ```
//! use vulkano::memory::{DeviceMemory, MemoryAllocateInfo};
//!
//! # let device: std::sync::Arc<vulkano::device::Device> = return;
//! // Taking the first memory type for the sake of this example.
//! let memory_type_index = 0;
//!
//! let memory = DeviceMemory::allocate(
//!     device.clone(),
//!     MemoryAllocateInfo {
//!         allocation_size: 1024,
//!         memory_type_index,
//!         ..Default::default()
//!     },
//! )
//! .expect("Failed to allocate memory");
//!
//! // The memory is automatically freed when `memory` is destroyed.
//! ```
//!
//! However allocating and freeing memory is very slow (up to several hundred milliseconds
//! sometimes). Instead you are strongly encouraged to use a memory pool. A memory pool is not
//! a Vulkan concept but a vulkano concept.
//!
//! A memory pool is any object that implements the `MemoryPool` trait. You can implement that
//! trait on your own structure and then use it when you create buffers and images so that they
//! get memory from that pool. By default if you don't specify any pool when creating a buffer or
//! an image, an instance of `StandardMemoryPool` that is shared by the `Device` object is used.

use self::allocator::{
    align_up, AllocationHandle, AllocationType, DeviceLayout, MemoryAlloc, MemoryAllocator,
    Suballocation,
};
pub use self::{alignment::*, device_memory::*};
use crate::{
    buffer::{sys::RawBuffer, Subbuffer},
    device::{Device, DeviceOwned, DeviceOwnedDebugWrapper},
    image::{sys::RawImage, Image, ImageAspects},
    macros::vulkan_bitflags,
    sync::{semaphore::Semaphore, HostAccessError},
    DeviceSize, Validated, ValidationError, VulkanError,
};
use std::{
    cmp,
    mem::ManuallyDrop,
    num::NonZeroU64,
    ops::{Bound, Range, RangeBounds, RangeTo},
    ptr::NonNull,
    sync::Arc,
};

mod alignment;
pub mod allocator;
mod device_memory;

/// Memory that can be bound to resources.
///
/// Most commonly you will want to obtain this by first using a [memory allocator] and then
/// [constructing this object from its allocation]. Alternatively, if you want to bind a whole
/// block of `DeviceMemory` to a resource, or can't go through an allocator, you can use [the
/// dedicated constructor].
///
/// [memory allocator]: MemoryAllocator
/// [the dedicated constructor]: Self::new_dedicated
#[derive(Debug)]
pub struct ResourceMemory {
    device_memory: ManuallyDrop<DeviceOwnedDebugWrapper<Arc<DeviceMemory>>>,
    offset: DeviceSize,
    size: DeviceSize,
    allocation_type: AllocationType,
    allocation_handle: AllocationHandle,
    suballocation_handle: Option<AllocationHandle>,
    allocator: Option<Arc<dyn MemoryAllocator>>,
}

impl ResourceMemory {
    /// Creates a new `ResourceMemory` that has a whole device memory block dedicated to it. You
    /// may use this when you obtain the memory in a way other than through the use of a memory
    /// allocator, for instance by importing memory.
    ///
    /// This is safe because we take ownership of the device memory, so that there can be no
    /// aliasing resources. On the other hand, the device memory can never be reused: it will be
    /// freed once the returned object is dropped.
    pub fn new_dedicated(device_memory: DeviceMemory) -> Self {
        unsafe { Self::new_dedicated_unchecked(Arc::new(device_memory)) }
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_dedicated_unchecked(device_memory: Arc<DeviceMemory>) -> Self {
        ResourceMemory {
            offset: 0,
            size: device_memory.allocation_size(),
            allocation_type: AllocationType::Unknown,
            allocation_handle: AllocationHandle::null(),
            suballocation_handle: None,
            allocator: None,
            device_memory: ManuallyDrop::new(DeviceOwnedDebugWrapper(device_memory)),
        }
    }

    /// Creates a new `ResourceMemory` from an allocation of a memory allocator.
    ///
    /// Ownership of `allocation` is semantically transferred to this object, and it is deallocated
    /// when the returned object is dropped.
    ///
    /// # Safety
    ///
    /// - `allocation` must refer to a **currently allocated** allocation of `allocator`.
    /// - `allocation` must never be deallocated.
    #[inline]
    pub unsafe fn from_allocation(
        allocator: Arc<dyn MemoryAllocator>,
        allocation: MemoryAlloc,
    ) -> Self {
        if let Some(suballocation) = allocation.suballocation {
            ResourceMemory {
                offset: suballocation.offset,
                size: suballocation.size,
                allocation_type: suballocation.allocation_type,
                allocation_handle: allocation.allocation_handle,
                suballocation_handle: Some(suballocation.handle),
                allocator: Some(allocator),
                device_memory: ManuallyDrop::new(DeviceOwnedDebugWrapper(allocation.device_memory)),
            }
        } else {
            ResourceMemory {
                offset: 0,
                size: allocation.device_memory.allocation_size(),
                allocation_type: AllocationType::Unknown,
                allocation_handle: allocation.allocation_handle,
                suballocation_handle: None,
                allocator: Some(allocator),
                device_memory: ManuallyDrop::new(DeviceOwnedDebugWrapper(allocation.device_memory)),
            }
        }
    }

    /// Returns the underlying block of [`DeviceMemory`].
    #[inline]
    pub fn device_memory(&self) -> &Arc<DeviceMemory> {
        &self.device_memory
    }

    /// Returns the offset (in bytes) within the [`DeviceMemory`] block where this `ResourceMemory`
    /// beings.
    ///
    /// If this `ResourceMemory` is not a [suballocation], then this will be `0`.
    ///
    /// [suballocation]: Suballocation
    #[inline]
    pub fn offset(&self) -> DeviceSize {
        self.offset
    }

    /// Returns the size (in bytes) of the `ResourceMemory`.
    ///
    /// If this `ResourceMemory` is not a [suballocation], then this will be equal to the
    /// [allocation size] of the [`DeviceMemory`] block.
    ///
    /// [suballocation]: Suballocation
    #[inline]
    pub fn size(&self) -> DeviceSize {
        self.size
    }

    /// Returns the type of resources that can be bound to this `ResourceMemory`.
    ///
    /// If this `ResourceMemory` is not a [suballocation], then this will be
    /// [`AllocationType::Unknown`].
    ///
    /// [suballocation]: Suballocation
    #[inline]
    pub fn allocation_type(&self) -> AllocationType {
        self.allocation_type
    }

    fn suballocation(&self) -> Option<Suballocation> {
        self.suballocation_handle.map(|handle| Suballocation {
            offset: self.offset,
            size: self.size,
            allocation_type: self.allocation_type,
            handle,
        })
    }

    /// Returns the mapped pointer to a range of the `ResourceMemory`, or returns [`None`] if ouf
    /// of bounds.
    ///
    /// `range` is specified in bytes relative to the beginning of `self` and must fall within the
    /// range of the memory mapping given to [`DeviceMemory::map`].
    ///
    /// See [`MappingState::slice`] for the safety invariants of the returned pointer.
    #[inline]
    pub fn mapped_slice(
        &self,
        range: impl RangeBounds<DeviceSize>,
    ) -> Option<Result<NonNull<[u8]>, HostAccessError>> {
        let mut range = self::range(range, ..self.size())?;
        range.start += self.offset();
        range.end += self.offset();

        let res = if let Some(state) = self.device_memory().mapping_state() {
            state.slice(range).ok_or(HostAccessError::OutOfMappedRange)
        } else {
            Err(HostAccessError::NotHostMapped)
        };

        Some(res)
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn mapped_slice_unchecked(
        &self,
        range: impl RangeBounds<DeviceSize>,
    ) -> Result<NonNull<[u8]>, HostAccessError> {
        let mut range = range_unchecked(range, ..self.size());
        range.start += self.offset();
        range.end += self.offset();

        if let Some(state) = self.device_memory().mapping_state() {
            state.slice(range).ok_or(HostAccessError::OutOfMappedRange)
        } else {
            Err(HostAccessError::NotHostMapped)
        }
    }

    pub(crate) fn atom_size(&self) -> Option<DeviceAlignment> {
        let memory = self.device_memory();

        (!memory.is_coherent()).then_some(memory.atom_size())
    }

    /// Invalidates the host cache for a range of the `ResourceMemory`.
    ///
    /// If the device memory is not [host-coherent], you must call this function before the memory
    /// is read by the host, if the device previously wrote to the memory. It has no effect if the
    /// memory is host-coherent.
    ///
    /// # Safety
    ///
    /// - If there are memory writes by the device that have not been propagated into the host
    ///   cache, then there must not be any references in Rust code to any portion of the specified
    ///   `memory_range`.
    ///
    /// [host-coherent]: MemoryPropertyFlags::HOST_COHERENT
    /// [`non_coherent_atom_size`]: crate::device::DeviceProperties::non_coherent_atom_size
    #[inline]
    pub unsafe fn invalidate_range(
        &self,
        memory_range: MappedMemoryRange,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_memory_range(&memory_range)?;

        self.device_memory()
            .invalidate_range(self.create_memory_range(memory_range))
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn invalidate_range_unchecked(
        &self,
        memory_range: MappedMemoryRange,
    ) -> Result<(), VulkanError> {
        self.device_memory()
            .invalidate_range_unchecked(self.create_memory_range(memory_range))
    }

    /// Flushes the host cache for a range of the `ResourceMemory`.
    ///
    /// If the device memory is not [host-coherent], you must call this function after writing to
    /// the memory, if the device is going to read the memory. It has no effect if the memory is
    /// host-coherent.
    ///
    /// # Safety
    ///
    /// - There must be no operations pending or executing in a device queue, that access any
    ///   portion of the specified `memory_range`.
    ///
    /// [host-coherent]: MemoryPropertyFlags::HOST_COHERENT
    /// [`non_coherent_atom_size`]: crate::device::DeviceProperties::non_coherent_atom_size
    #[inline]
    pub unsafe fn flush_range(
        &self,
        memory_range: MappedMemoryRange,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_memory_range(&memory_range)?;

        self.device_memory()
            .flush_range(self.create_memory_range(memory_range))
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn flush_range_unchecked(
        &self,
        memory_range: MappedMemoryRange,
    ) -> Result<(), VulkanError> {
        self.device_memory()
            .flush_range_unchecked(self.create_memory_range(memory_range))
    }

    fn validate_memory_range(
        &self,
        memory_range: &MappedMemoryRange,
    ) -> Result<(), Box<ValidationError>> {
        let &MappedMemoryRange {
            offset,
            size,
            _ne: _,
        } = memory_range;

        if !(offset <= self.size() && size <= self.size() - offset) {
            return Err(Box::new(ValidationError {
                context: "memory_range".into(),
                problem: "is not contained within the allocation".into(),
                ..Default::default()
            }));
        }

        Ok(())
    }

    fn create_memory_range(&self, memory_range: MappedMemoryRange) -> MappedMemoryRange {
        let MappedMemoryRange {
            mut offset,
            mut size,
            _ne: _,
        } = memory_range;

        let memory = self.device_memory();

        offset += self.offset();

        // VUID-VkMappedMemoryRange-size-01390
        if memory_range.offset + size == self.size() {
            // We can align the end of the range like this without aliasing other allocations,
            // because the memory allocator must ensure that all allocations are aligned to the
            // atom size for non-host-coherent host-visible memory.
            let end = cmp::min(
                align_up(offset + size, memory.atom_size()),
                memory.allocation_size(),
            );
            size = end - offset;
        }

        MappedMemoryRange {
            offset,
            size,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl Drop for ResourceMemory {
    #[inline]
    fn drop(&mut self) {
        let device_memory = unsafe { ManuallyDrop::take(&mut self.device_memory) }.0;

        if let Some(allocator) = &self.allocator {
            let allocation = MemoryAlloc {
                device_memory,
                suballocation: self.suballocation(),
                allocation_handle: self.allocation_handle,
            };

            // SAFETY: Enforced by the safety contract of [`ResourceMemory::from_allocation`].
            unsafe { allocator.deallocate(allocation) };
        }
    }
}

unsafe impl DeviceOwned for ResourceMemory {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.device_memory().device()
    }
}

/// Properties of the memory in a physical device.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MemoryProperties {
    /// The available memory types.
    pub memory_types: Vec<MemoryType>,

    /// The available memory heaps.
    pub memory_heaps: Vec<MemoryHeap>,
}

impl From<ash::vk::PhysicalDeviceMemoryProperties> for MemoryProperties {
    #[inline]
    fn from(val: ash::vk::PhysicalDeviceMemoryProperties) -> Self {
        Self {
            memory_types: val.memory_types[0..val.memory_type_count as usize]
                .iter()
                .map(|vk_memory_type| MemoryType {
                    property_flags: vk_memory_type.property_flags.into(),
                    heap_index: vk_memory_type.heap_index,
                })
                .collect(),
            memory_heaps: val.memory_heaps[0..val.memory_heap_count as usize]
                .iter()
                .map(|vk_memory_heap| MemoryHeap {
                    size: vk_memory_heap.size,
                    flags: vk_memory_heap.flags.into(),
                })
                .collect(),
        }
    }
}

/// A memory type in a physical device.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MemoryType {
    /// The properties of this memory type.
    pub property_flags: MemoryPropertyFlags,

    /// The index of the memory heap that this memory type corresponds to.
    pub heap_index: u32,
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Properties of a memory type.
    MemoryPropertyFlags = MemoryPropertyFlags(u32);

    /// The memory is located on the device, and is allocated from a heap that also has the
    /// [`DEVICE_LOCAL`] flag set.
    ///
    /// For some devices, particularly integrated GPUs, the device shares memory with the host and
    /// all memory may be device-local, so the distinction is moot. However, if the device has
    /// non-device-local memory, it is usually faster for the device to access device-local memory.
    /// Therefore, device-local memory is preferred for data that will only be accessed by
    /// the device.
    ///
    /// If the device and host do not share memory, data transfer between host and device may
    /// involve sending the data over the data bus that connects the two. Accesses are faster if
    /// they do not have to cross this barrier: device-local memory is fast for the device to
    /// access, but slower to access by the host. However, there are devices that share memory with
    /// the host, yet have distinct device-local and non-device local memory types. In that case,
    /// the speed difference may not be large.
    ///
    /// For data transfer between host and device, it is most efficient if the memory is located
    /// at the destination of the transfer. Thus, if [`HOST_VISIBLE`] versions of both are
    /// available, device-local memory is preferred for host-to-device data transfer, while
    /// non-device-local memory is preferred for device-to-host data transfer. This is because data
    /// is usually written only once but potentially read several times, and because reads can take
    /// advantage of caching while writes cannot.
    ///
    /// Devices may have memory types that are neither `DEVICE_LOCAL` nor [`HOST_VISIBLE`]. This
    /// is regular host memory that is made available to the device exclusively. Although it will be
    /// slower to access from the device than `DEVICE_LOCAL` memory, it can be faster than
    /// [`HOST_VISIBLE`] memory. It can be used as overflow space if the device is out of memory.
    ///
    /// [`DEVICE_LOCAL`]: MemoryHeapFlags::DEVICE_LOCAL
    /// [`HOST_VISIBLE`]: MemoryPropertyFlags::HOST_VISIBLE
    DEVICE_LOCAL = DEVICE_LOCAL,

    /// The memory can be mapped into the memory space of the host and accessed as regular RAM.
    ///
    /// Memory of this type is required to transfer data between the host and the device. If
    /// the memory is going to be accessed by the device more than a few times, it is recommended
    /// to copy the data to non-`HOST_VISIBLE` memory first if it is available.
    ///
    /// `HOST_VISIBLE` memory is always at least either [`HOST_COHERENT`] or [`HOST_CACHED`],
    /// but it can be both.
    ///
    /// [`HOST_COHERENT`]: MemoryPropertyFlags::HOST_COHERENT
    /// [`HOST_CACHED`]: MemoryPropertyFlags::HOST_CACHED
    HOST_VISIBLE = HOST_VISIBLE,

    /// Host access to the memory does not require calling [`invalidate_range`] to make device
    /// writes visible to the host, nor [`flush_range`] to flush host writes back to the device.
    ///
    /// [`invalidate_range`]: DeviceMemory::invalidate_range
    /// [`flush_range`]: DeviceMemory::flush_range
    HOST_COHERENT = HOST_COHERENT,

    /// The memory is cached by the host.
    ///
    /// `HOST_CACHED` memory is fast for reads and random access from the host, so it is preferred
    /// for device-to-host data transfer. Memory that is [`HOST_VISIBLE`] but not `HOST_CACHED` is
    /// often slow for all accesses other than sequential writing, so it is more suited for
    /// host-to-device transfer, and it is often beneficial to write the data in sequence.
    ///
    /// [`HOST_VISIBLE`]: MemoryPropertyFlags::HOST_VISIBLE
    HOST_CACHED = HOST_CACHED,

    /// Allocations made from the memory are lazy.
    ///
    /// This means that no actual allocation is performed. Instead memory is automatically
    /// allocated by the Vulkan implementation based on need. You can call
    /// [`DeviceMemory::commitment`] to query how much memory is currently committed to an
    /// allocation.
    ///
    /// Memory of this type can only be used on images created with a certain flag, and is never
    /// [`HOST_VISIBLE`].
    ///
    /// [`HOST_VISIBLE`]: MemoryPropertyFlags::HOST_VISIBLE
    LAZILY_ALLOCATED = LAZILY_ALLOCATED,

    /// The memory can only be accessed by the device, and allows protected queue access.
    ///
    /// Memory of this type is never [`HOST_VISIBLE`], [`HOST_COHERENT`] or [`HOST_CACHED`].
    ///
    /// [`HOST_VISIBLE`]: MemoryPropertyFlags::HOST_VISIBLE
    /// [`HOST_COHERENT`]: MemoryPropertyFlags::HOST_COHERENT
    /// [`HOST_CACHED`]: MemoryPropertyFlags::HOST_CACHED
    PROTECTED = PROTECTED
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
    ]),

    /// Device accesses to the memory are automatically made available and visible to other device
    /// accesses.
    ///
    /// Memory of this type is slower to access by the device, so it is best avoided for general
    /// purpose use. Because of its coherence properties, however, it may be useful for debugging.
    DEVICE_COHERENT = DEVICE_COHERENT_AMD
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(amd_device_coherent_memory)]),
    ]),

    /// The memory is not cached on the device.
    ///
    /// `DEVICE_UNCACHED` memory is always also [`DEVICE_COHERENT`].
    ///
    /// [`DEVICE_COHERENT`]: MemoryPropertyFlags::DEVICE_COHERENT
    DEVICE_UNCACHED = DEVICE_UNCACHED_AMD
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(amd_device_coherent_memory)]),
    ]),

    /// Other devices can access the memory via remote direct memory access (RDMA).
    RDMA_CAPABLE = RDMA_CAPABLE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_external_memory_rdma)]),
    ]),
}

/// A memory heap in a physical device.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MemoryHeap {
    /// The size of the heap in bytes.
    pub size: DeviceSize,

    /// Attributes of the heap.
    pub flags: MemoryHeapFlags,
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Attributes of a memory heap.
    MemoryHeapFlags = MemoryHeapFlags(u32);

    /// The heap corresponds to device-local memory.
    DEVICE_LOCAL = DEVICE_LOCAL,

    /// If used on a logical device that represents more than one physical device, allocations are
    /// replicated across each physical device's instance of this heap.
    MULTI_INSTANCE = MULTI_INSTANCE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([InstanceExtension(khr_device_group_creation)]),
    ]),
}

/// Represents requirements expressed by the Vulkan implementation when it comes to binding memory
/// to a resource.
#[derive(Clone, Copy, Debug)]
pub struct MemoryRequirements {
    /// Memory layout required for the resource.
    pub layout: DeviceLayout,

    /// Indicates which memory types can be used. Each bit that is set to 1 means that the memory
    /// type whose index is the same as the position of the bit can be used.
    pub memory_type_bits: u32,

    /// Whether implementation prefers to use dedicated allocations (in other words, allocate
    /// a whole block of memory dedicated to this resource alone).
    /// This will be `false` if the device API version is less than 1.1 and the
    /// [`khr_get_memory_requirements2`](crate::device::DeviceExtensions::khr_get_memory_requirements2)
    /// extension is not enabled on the device.
    pub prefers_dedicated_allocation: bool,

    /// Whether implementation requires the use of a dedicated allocation (in other words, allocate
    /// a whole block of memory dedicated to this resource alone).
    /// This will be `false` if the device API version is less than 1.1 and the
    /// [`khr_get_memory_requirements2`](crate::device::DeviceExtensions::khr_get_memory_requirements2)
    /// extension is not enabled on the device.
    pub requires_dedicated_allocation: bool,
}

/// Indicates a specific resource to allocate memory for.
///
/// Using dedicated allocations can yield better performance, but requires the
/// [`khr_dedicated_allocation`](crate::device::DeviceExtensions::khr_dedicated_allocation)
/// extension to be enabled on the device.
///
/// If a dedicated allocation is performed, it must not be bound to any resource other than the
/// one that was passed with the enumeration.
#[derive(Clone, Copy, Debug)]
pub enum DedicatedAllocation<'a> {
    /// Allocation dedicated to a buffer.
    Buffer(&'a RawBuffer),
    /// Allocation dedicated to an image.
    Image(&'a RawImage),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DedicatedTo {
    Buffer(NonZeroU64),
    Image(NonZeroU64),
}

impl From<DedicatedAllocation<'_>> for DedicatedTo {
    fn from(dedicated_allocation: DedicatedAllocation<'_>) -> Self {
        match dedicated_allocation {
            DedicatedAllocation::Buffer(buffer) => Self::Buffer(buffer.id()),
            DedicatedAllocation::Image(image) => Self::Image(image.id()),
        }
    }
}

/// The properties for exporting or importing external memory, when a buffer or image is created
/// with a specific configuration.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct ExternalMemoryProperties {
    /// Whether a dedicated memory allocation is required for the queried external handle type.
    pub dedicated_only: bool,

    /// Whether memory can be exported to an external source with the queried
    /// external handle type.
    pub exportable: bool,

    /// Whether memory can be imported from an external source with the queried
    /// external handle type.
    pub importable: bool,

    /// Which external handle types can be re-exported after the queried external handle type has
    /// been imported.
    pub export_from_imported_handle_types: ExternalMemoryHandleTypes,

    /// Which external handle types can be enabled along with the queried external handle type
    /// when creating the buffer or image.
    pub compatible_handle_types: ExternalMemoryHandleTypes,
}

impl From<ash::vk::ExternalMemoryProperties> for ExternalMemoryProperties {
    #[inline]
    fn from(val: ash::vk::ExternalMemoryProperties) -> Self {
        Self {
            dedicated_only: val
                .external_memory_features
                .intersects(ash::vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY),
            exportable: val
                .external_memory_features
                .intersects(ash::vk::ExternalMemoryFeatureFlags::EXPORTABLE),
            importable: val
                .external_memory_features
                .intersects(ash::vk::ExternalMemoryFeatureFlags::IMPORTABLE),
            export_from_imported_handle_types: val.export_from_imported_handle_types.into(),
            compatible_handle_types: val.compatible_handle_types.into(),
        }
    }
}

/// Parameters to execute sparse bind operations on a queue.
#[derive(Clone, Debug)]
pub struct BindSparseInfo {
    /// The semaphores to wait for before beginning the execution of this batch of
    /// sparse bind operations.
    ///
    /// The default value is empty.
    pub wait_semaphores: Vec<Arc<Semaphore>>,

    /// The bind operations to perform for buffers.
    ///
    /// The default value is empty.
    pub buffer_binds: Vec<(Subbuffer<[u8]>, Vec<SparseBufferMemoryBind>)>,

    /// The bind operations to perform for images with an opaque memory layout.
    ///
    /// This should be used for mip tail regions, the metadata aspect, and for the normal regions
    /// of images that do not have the `sparse_residency` flag set.
    ///
    /// The default value is empty.
    pub image_opaque_binds: Vec<(Arc<Image>, Vec<SparseImageOpaqueMemoryBind>)>,

    /// The bind operations to perform for images with a known memory layout.
    ///
    /// This type of sparse bind can only be used for images that have the `sparse_residency`
    /// flag set.
    /// Only the normal texel regions can be bound this way, not the mip tail regions or metadata
    /// aspect.
    ///
    /// The default value is empty.
    pub image_binds: Vec<(Arc<Image>, Vec<SparseImageMemoryBind>)>,

    /// The semaphores to signal after the execution of this batch of sparse bind operations
    /// has completed.
    ///
    /// The default value is empty.
    pub signal_semaphores: Vec<Arc<Semaphore>>,

    pub _ne: crate::NonExhaustive,
}

impl Default for BindSparseInfo {
    #[inline]
    fn default() -> Self {
        Self {
            wait_semaphores: Vec::new(),
            buffer_binds: Vec::new(),
            image_opaque_binds: Vec::new(),
            image_binds: Vec::new(),
            signal_semaphores: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Parameters for a single sparse bind operation on a buffer.
#[derive(Clone, Debug, Default)]
pub struct SparseBufferMemoryBind {
    /// The offset in bytes from the start of the buffer's memory, where memory is to be (un)bound.
    ///
    /// The default value is `0`.
    pub offset: DeviceSize,

    /// The size in bytes of the memory to be (un)bound.
    ///
    /// The default value is `0`, which must be overridden.
    pub size: DeviceSize,

    /// If `Some`, specifies the memory and an offset into that memory that is to be bound.
    /// The provided memory must match the buffer's memory requirements.
    ///
    /// If `None`, specifies that existing memory at the specified location is to be unbound.
    ///
    /// The default value is `None`.
    pub memory: Option<(Arc<DeviceMemory>, DeviceSize)>,
}

/// Parameters for a single sparse bind operation on parts of an image with an opaque memory
/// layout.
///
/// This type of sparse bind should be used for mip tail regions, the metadata aspect, and for the
/// normal regions of images that do not have the `sparse_residency` flag set.
#[derive(Clone, Debug, Default)]
pub struct SparseImageOpaqueMemoryBind {
    /// The offset in bytes from the start of the image's memory, where memory is to be (un)bound.
    ///
    /// The default value is `0`.
    pub offset: DeviceSize,

    /// The size in bytes of the memory to be (un)bound.
    ///
    /// The default value is `0`, which must be overridden.
    pub size: DeviceSize,

    /// If `Some`, specifies the memory and an offset into that memory that is to be bound.
    /// The provided memory must match the image's memory requirements.
    ///
    /// If `None`, specifies that existing memory at the specified location is to be unbound.
    ///
    /// The default value is `None`.
    pub memory: Option<(Arc<DeviceMemory>, DeviceSize)>,

    /// Sets whether the binding should apply to the metadata aspect of the image, or to the
    /// normal texel data.
    ///
    /// The default value is `false`.
    pub metadata: bool,
}

/// Parameters for a single sparse bind operation on parts of an image with a known memory layout.
///
/// This type of sparse bind can only be used for images that have the `sparse_residency` flag set.
/// Only the normal texel regions can be bound this way, not the mip tail regions or metadata
/// aspect.
#[derive(Clone, Debug, Default)]
pub struct SparseImageMemoryBind {
    /// The aspects of the image where memory is to be (un)bound.
    ///
    /// The default value is `ImageAspects::empty()`, which must be overridden.
    pub aspects: ImageAspects,

    /// The mip level of the image where memory is to be (un)bound.
    ///
    /// The default value is `0`.
    pub mip_level: u32,

    /// The array layer of the image where memory is to be (un)bound.
    ///
    /// The default value is `0`.
    pub array_layer: u32,

    /// The offset in texels (or for compressed images, texel blocks) from the origin of the image,
    /// where memory is to be (un)bound.
    ///
    /// This must be a multiple of the
    /// [`SparseImageFormatProperties::image_granularity`](crate::image::SparseImageFormatProperties::image_granularity)
    /// value of the image.
    ///
    /// The default value is `[0; 3]`.
    pub offset: [u32; 3],

    /// The extent in texels (or for compressed images, texel blocks) of the image where
    /// memory is to be (un)bound.
    ///
    /// This must be a multiple of the
    /// [`SparseImageFormatProperties::image_granularity`](crate::image::SparseImageFormatProperties::image_granularity)
    /// value of the image, or `offset + extent` for that dimension must equal the image's total
    /// extent.
    ///
    /// The default value is `[0; 3]`, which must be overridden.
    pub extent: [u32; 3],

    /// If `Some`, specifies the memory and an offset into that memory that is to be bound.
    /// The provided memory must match the image's memory requirements.
    ///
    /// If `None`, specifies that existing memory at the specified location is to be unbound.
    ///
    /// The default value is `None`.
    pub memory: Option<(Arc<DeviceMemory>, DeviceSize)>,
}

#[inline(always)]
pub(crate) fn is_aligned(offset: DeviceSize, alignment: DeviceAlignment) -> bool {
    offset & (alignment.as_devicesize() - 1) == 0
}

/// Performs bounds-checking of a Vulkan memory range. Analog of `std::slice::range`.
pub(crate) fn range(
    range: impl RangeBounds<DeviceSize>,
    bounds: RangeTo<DeviceSize>,
) -> Option<Range<DeviceSize>> {
    let len = bounds.end;

    let start = match range.start_bound() {
        Bound::Included(&start) => start,
        Bound::Excluded(start) => start.checked_add(1)?,
        Bound::Unbounded => 0,
    };

    let end = match range.end_bound() {
        Bound::Included(end) => end.checked_add(1)?,
        Bound::Excluded(&end) => end,
        Bound::Unbounded => len,
    };

    (start <= end && end <= len).then_some(Range { start, end })
}

/// Converts a `RangeBounds` into a `Range` without doing any bounds checking.
pub(crate) fn range_unchecked(
    range: impl RangeBounds<DeviceSize>,
    bounds: RangeTo<DeviceSize>,
) -> Range<DeviceSize> {
    let len = bounds.end;

    let start = match range.start_bound() {
        Bound::Included(&start) => start,
        Bound::Excluded(start) => start + 1,
        Bound::Unbounded => 0,
    };

    let end = match range.end_bound() {
        Bound::Included(end) => end + 1,
        Bound::Excluded(&end) => end,
        Bound::Unbounded => len,
    };

    debug_assert!(start <= end && end <= len);

    Range { start, end }
}
