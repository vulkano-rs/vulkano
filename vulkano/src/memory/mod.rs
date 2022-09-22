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
        MappedDeviceMemory, MemoryAllocateInfo, MemoryImportInfo, MemoryMapError,
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

    /// The memory is located on the device. This usually means that it's efficient for the
    /// device to access this memory.
    device_local = DEVICE_LOCAL,

    /// The memory can be accessed by the host.
    host_visible = HOST_VISIBLE,

    /// Modifications made by the host or the device on this memory type are
    /// instantaneously visible to the other party. If memory does not have this flag, changes to
    /// the memory are not visible until they are flushed or invalidated.
    host_coherent = HOST_COHERENT,

    /// The memory is cached by the host. Host memory accesses to cached memory are faster than for
    /// uncached memory, but the cache may not be coherent.
    host_cached = HOST_CACHED,

    /// Allocations made from this memory type are lazy.
    ///
    /// This means that no actual allocation is performed. Instead memory is automatically
    /// allocated by the Vulkan implementation based on need.
    ///
    /// Memory of this type can only be used on images created with a certain flag. Memory of this
    /// type is never host-visible.
    lazily_allocated = LAZILY_ALLOCATED,

    /// The memory can only be accessed by the device, and allows protected queue access.
    ///
    /// Memory of this type is never host visible, host coherent or host cached.
    protected = PROTECTED {
        api_version: V1_1,
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
#[derive(Debug, Copy, Clone)]
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
#[derive(Debug, Copy, Clone)]
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
