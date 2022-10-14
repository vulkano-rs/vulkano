// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

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
//! for (index, heap) in physical_device.memory_properties().memory_heaps.iter().enumerate() {
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
//!     println!("Host-accessible: {:?}", ty.property_flags.host_visible);
//!     println!("Device-local: {:?}", ty.property_flags.device_local);
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
//! ).expect("Failed to allocate memory");
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

pub use self::{
    device_memory::{
        DeviceMemory, DeviceMemoryError, ExternalMemoryHandleType, ExternalMemoryHandleTypes,
        MappedDeviceMemory, MemoryAllocateFlags, MemoryAllocateInfo, MemoryImportInfo,
        MemoryMapError,
    },
    pool::MemoryPool,
};
use crate::{
    buffer::{sys::UnsafeBuffer, BufferAccess},
    image::{sys::UnsafeImage, ImageAccess, ImageAspects},
    macros::vulkan_bitflags,
    sync::Semaphore,
    DeviceSize,
};
use std::sync::Arc;

pub mod allocator;
mod device_memory;
pub mod pool;

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
    /// Properties of a memory type.
    #[non_exhaustive]
    MemoryPropertyFlags = MemoryPropertyFlags(u32);

    /// The memory is located on the device, and is allocated from a heap that also has the
    /// [`device_local`](MemoryHeapFlags::device_local) flag set.
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
    /// at the destination of the transfer. Thus, if `host_visible` versions of both are available,
    /// device-local memory is preferred for host-to-device data transfer, while non-device-local
    /// memory is preferred for device-to-host data transfer. This is because data is usually
    /// written only once but potentially read several times, and because reads can take advantage
    /// of caching while writes cannot.
    ///
    /// Devices may have memory types that are neither `device_local` nor `host_visible`. This is
    /// regular host memory that is made available to the device exclusively. Although it will be
    /// slower to access from the device than `device_local` memory, it can be faster than
    /// `host_visible` memory. It can be used as overflow space if the device is out of memory.
    device_local = DEVICE_LOCAL,

    /// The memory can be mapped into the memory space of the host and accessed as regular RAM.
    ///
    /// Memory of this type is required to transfer data between the host and the device. If
    /// the memory is going to be accessed by the device more than a few times, it is recommended
    /// to copy the data to non-`host_visible` memory first if it is available.
    ///
    /// `host_visible` memory is always at least either `host_coherent` or `host_cached`, but it
    /// can be both.
    host_visible = HOST_VISIBLE,

    /// Host access to the memory does not require calling
    /// [`invalidate_range`](MappedDeviceMemory::invalidate_range) to make device writes visible to
    /// the host, nor [`flush_range`](MappedDeviceMemory::flush_range) to flush host writes back
    /// to the device.
    host_coherent = HOST_COHERENT,

    /// The memory is cached by the host.
    ///
    /// `host_cached` memory is fast for reads and random access from the host, so it is preferred
    /// for device-to-host data transfer. Memory that is `host_visible` but not `host_cached` is
    /// often slow for all accesses other than sequential writing, so it is more suited for
    /// host-to-device transfer, and it is often beneficial to write the data in sequence.
    host_cached = HOST_CACHED,

    /// Allocations made from the memory are lazy.
    ///
    /// This means that no actual allocation is performed. Instead memory is automatically
    /// allocated by the Vulkan implementation based on need. You can call
    /// [`DeviceMemory::commitment`] to query how much memory is currently committed to an
    /// allocation.
    ///
    /// Memory of this type can only be used on images created with a certain flag, and is never
    /// `host_visible`.
    lazily_allocated = LAZILY_ALLOCATED,

    /// The memory can only be accessed by the device, and allows protected queue access.
    ///
    /// Memory of this type is never `host_visible`, `host_coherent` or `host_cached`.
    protected = PROTECTED {
        api_version: V1_1,
    },

    /// Device accesses to the memory are automatically made available and visible to other device
    /// accesses.
    ///
    /// Memory of this type is slower to access by the device, so it is best avoided for general
    /// purpose use. Because of its coherence properties, however, it may be useful for debugging.
    device_coherent = DEVICE_COHERENT_AMD {
        device_extensions: [amd_device_coherent_memory],
    },

    /// The memory is not cached on the device.
    ///
    /// `device_uncached` memory is always also `device_coherent`.
    device_uncached = DEVICE_UNCACHED_AMD {
        device_extensions: [amd_device_coherent_memory],
    },

    /// Other devices can access the memory via remote direct memory access (RDMA).
    rdma_capable = RDMA_CAPABLE_NV {
        device_extensions: [nv_external_memory_rdma],
    },
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
    /// Attributes of a memory heap.
    #[non_exhaustive]
    MemoryHeapFlags = MemoryHeapFlags(u32);

    /// The heap corresponds to device-local memory.
    device_local = DEVICE_LOCAL,

    /// If used on a logical device that represents more than one physical device, allocations are
    /// replicated across each physical device's instance of this heap.
    multi_instance = MULTI_INSTANCE {
        api_version: V1_1,
        instance_extensions: [khr_device_group_creation],
    },
}

/// Represents requirements expressed by the Vulkan implementation when it comes to binding memory
/// to a resource.
#[derive(Clone, Copy, Debug)]
pub struct MemoryRequirements {
    /// Number of bytes of memory required.
    pub size: DeviceSize,

    /// Alignment of the requirement buffer. The base memory address must be a multiple
    /// of this value.
    pub alignment: DeviceSize,

    /// Indicates which memory types can be used. Each bit that is set to 1 means that the memory
    /// type whose index is the same as the position of the bit can be used.
    pub memory_type_bits: u32,

    /// Whether implementation prefers to use dedicated allocations (in other words, allocate
    /// a whole block of memory dedicated to this resource alone). This will be `false` if the
    /// [`khr_get_memory_requirements2`](crate::device::DeviceExtensions::khr_get_memory_requirements2)
    /// extension is not enabled on the device.
    ///
    /// > **Note**: As its name says, using a dedicated allocation is an optimization and not a
    /// > requirement.
    pub prefer_dedicated: bool,
}

impl From<ash::vk::MemoryRequirements> for MemoryRequirements {
    #[inline]
    fn from(val: ash::vk::MemoryRequirements) -> Self {
        MemoryRequirements {
            size: val.size,
            alignment: val.alignment,
            memory_type_bits: val.memory_type_bits,
            prefer_dedicated: false,
        }
    }
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
    Buffer(&'a UnsafeBuffer),
    /// Allocation dedicated to an image.
    Image(&'a UnsafeImage),
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
    pub buffer_binds: Vec<(Arc<dyn BufferAccess>, Vec<SparseBufferMemoryBind>)>,

    /// The bind operations to perform for images with an opaque memory layout.
    ///
    /// This should be used for mip tail regions, the metadata aspect, and for the normal regions
    /// of images that do not have the `sparse_residency` flag set.
    ///
    /// The default value is empty.
    pub image_opaque_binds: Vec<(Arc<dyn ImageAccess>, Vec<SparseImageOpaqueMemoryBind>)>,

    /// The bind operations to perform for images with a known memory layout.
    ///
    /// This type of sparse bind can only be used for images that have the `sparse_residency`
    /// flag set.
    /// Only the normal texel regions can be bound this way, not the mip tail regions or metadata
    /// aspect.
    ///
    /// The default value is empty.
    pub image_binds: Vec<(Arc<dyn ImageAccess>, Vec<SparseImageMemoryBind>)>,

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
    pub resource_offset: DeviceSize,

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

/// Parameters for a single sparse bind operation on parts of an image with an opaque memory layout.
///
/// This type of sparse bind should be used for mip tail regions, the metadata aspect, and for the
/// normal regions of images that do not have the `sparse_residency` flag set.
#[derive(Clone, Debug, Default)]
pub struct SparseImageOpaqueMemoryBind {
    /// The offset in bytes from the start of the image's memory, where memory is to be (un)bound.
    ///
    /// The default value is `0`.
    pub resource_offset: DeviceSize,

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
