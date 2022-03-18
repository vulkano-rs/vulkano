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
//! for heap in physical_device.memory_heaps() {
//!     println!("Heap #{:?} has a capacity of {:?} bytes", heap.id(), heap.size());
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
//! for ty in physical_device.memory_types() {
//!     println!("Memory type belongs to heap #{:?}", ty.heap().id());
//!     println!("Host-accessible: {:?}", ty.is_host_visible());
//!     println!("Device-local: {:?}", ty.is_device_local());
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
//! let memory_type = device.physical_device().memory_types().next().unwrap();
//!
//! let memory = DeviceMemory::allocate(
//!     device.clone(),
//!     MemoryAllocateInfo {
//!         allocation_size: 1024,
//!         memory_type_index: memory_type.id(),
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
//! an image, an instance of `StdMemoryPool` that is shared by the `Device` object is used.

pub use self::{
    device_memory::{
        DeviceMemory, DeviceMemoryAllocationError, DeviceMemoryExportError,
        ExternalMemoryHandleType, ExternalMemoryHandleTypes, MappedDeviceMemory,
        MemoryAllocateInfo, MemoryImportInfo, MemoryMapError,
    },
    pool::MemoryPool,
};
use crate::{buffer::sys::UnsafeBuffer, image::sys::UnsafeImage, DeviceSize};

mod device_memory;
pub mod pool;

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
