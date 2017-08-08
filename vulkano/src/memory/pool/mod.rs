// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use device::DeviceOwned;
use instance::MemoryType;
use memory::DedicatedAlloc;
use memory::DeviceMemory;
use memory::MappedDeviceMemory;
use memory::DeviceMemoryAllocError;

pub use self::host_visible::StdHostVisibleMemoryTypePool;
pub use self::host_visible::StdHostVisibleMemoryTypePoolAlloc;
pub use self::non_host_visible::StdNonHostVisibleMemoryTypePool;
pub use self::non_host_visible::StdNonHostVisibleMemoryTypePoolAlloc;
pub use self::pool::StdMemoryPool;
pub use self::pool::StdMemoryPoolAlloc;

mod host_visible;
mod non_host_visible;
mod pool;

/// Pool of GPU-visible memory that can be allocated from.
pub unsafe trait MemoryPool: DeviceOwned {
    /// Object that represents a single allocation. Its destructor should free the chunk.
    type Alloc: MemoryPoolAlloc;

    /// Allocates memory from the pool.
    ///
    /// # Safety
    ///
    /// Implementation safety:
    ///
    /// - The returned object must match the requirements.
    /// - When a linear object is allocated next to an optimal object, it is mandatory that
    ///   the boundary is aligned to the value of the `buffer_image_granularity` limit.
    ///
    /// Note that it is not unsafe to *call* this function, but it is unsafe to bind the memory
    /// returned by this function to a resource.
    ///
    /// # Panic
    ///
    /// - Panics if `memory_type` doesn't belong to the same physical device as the device which
    ///   was used to create this pool.
    /// - Panics if the memory type is not host-visible and `map` is `MappingRequirement::Map`.
    /// - Panics if `size` is 0.
    /// - Panics if `alignment` is 0.
    ///
    fn alloc_generic(&self, ty: MemoryType, size: usize, alignment: usize, layout: AllocLayout,
                     map: MappingRequirement) -> Result<Self::Alloc, DeviceMemoryAllocError>;

    /// Allocates memory from the pool.
    ///
    /// Contrary to `alloc_generic`, this function may allocate a whole new block of memory
    /// dedicated to a resource. The default provided implementation will do so for allocations
    /// of more than 20MB and for images that have the color_attachment or depth_stencil_attachment
    /// usages enabled.
    ///
    /// # Safety
    ///
    /// Implementation safety:
    ///
    /// - The returned object must match the requirements.
    /// - When a linear object is allocated next to an optimal object, it is mandatory that
    ///   the boundary is aligned to the value of the `buffer_image_granularity` limit.
    /// - If `dedicated` is not `None`, the returned memory must either not be dedicated or be
    ///   dedicated to the resource that was passed.
    ///
    /// Note that it is not unsafe to *call* this function, but it is unsafe to bind the memory
    /// returned by this function to a resource.
    ///
    /// # Panic
    ///
    /// - Panics if `memory_type` doesn't belong to the same physical device as the device which
    ///   was used to create this pool.
    /// - Panics if the memory type is not host-visible and `map` is `MappingRequirement::Map`.
    /// - Panics if `size` is 0.
    /// - Panics if `alignment` is 0.
    ///
    fn alloc(&self, ty: MemoryType, size: usize, alignment: usize, layout: AllocLayout,
             map: MappingRequirement, dedicated: DedicatedAlloc)
             -> Result<PotentialDedicatedAllocation<Self::Alloc>, DeviceMemoryAllocError>
    {
        if !self.device().loaded_extensions().khr_dedicated_allocation {
            let alloc = self.alloc_generic(ty, size, alignment, layout, map)?;
            return Ok(alloc.into());
        }

        let use_dedicated = match dedicated {
            DedicatedAlloc::None => false,
            DedicatedAlloc::Buffer(ref buf) => size >= 20 * 1024 * 1024,
            DedicatedAlloc::Image(ref img) => {
                size >= 20 * 1024 * 1024 || img.usage_color_attachment() ||
                    img.usage_depth_stencil_attachment()
            },
        };

        if !use_dedicated {
            let alloc = self.alloc_generic(ty, size, alignment, layout, map)?;
            return Ok(alloc.into());
        }

        // If we reach here, then we perform a dedicated alloc.

        match map {
            MappingRequirement::Map => {
                let mem = DeviceMemory::dedicated_alloc_and_map(self.device().clone(), ty, size,
                                                                dedicated)?;
                Ok(PotentialDedicatedAllocation::DedicatedMapped(mem))
            },
            MappingRequirement::DoNotMap => {
                let mem = DeviceMemory::dedicated_alloc(self.device().clone(), ty, size,
                                                        dedicated)?;
                Ok(PotentialDedicatedAllocation::Dedicated(mem))
            },
        }
    }
}

/// Object that represents a single allocation. Its destructor should free the chunk.
pub unsafe trait MemoryPoolAlloc {
    /// Returns the memory object from which this is allocated. Returns `None` if the memory is
    /// not mapped.
    fn mapped_memory(&self) -> Option<&MappedDeviceMemory>;

    /// Returns the memory object from which this is allocated.
    fn memory(&self) -> &DeviceMemory;

    /// Returns the offset at the start of the memory where the first byte of this allocation
    /// resides.
    fn offset(&self) -> usize;
}

/// Whether an allocation should map the memory or not.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum MappingRequirement {
    /// Should map.
    Map,
    /// Shouldn't map.
    DoNotMap,
}

/// Layout of the object being allocated.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum AllocLayout {
    /// The object has a linear layout.
    Linear,
    /// The object has an optimal layout.
    Optimal,
}

/// Enumeration that can contain either a generic allocation coming from a pool, or a dedicated
/// allocation for one specific resource.
#[derive(Debug)]
pub enum PotentialDedicatedAllocation<A> {
    Generic(A),
    Dedicated(DeviceMemory),
    DedicatedMapped(MappedDeviceMemory),
}

unsafe impl<A> MemoryPoolAlloc for PotentialDedicatedAllocation<A>
    where A: MemoryPoolAlloc
{
    #[inline]
    fn mapped_memory(&self) -> Option<&MappedDeviceMemory> {
        match *self {
            PotentialDedicatedAllocation::Generic(ref alloc) => alloc.mapped_memory(),
            PotentialDedicatedAllocation::Dedicated(ref mem) => None,
            PotentialDedicatedAllocation::DedicatedMapped(ref mem) => Some(mem),
        }
    }

    #[inline]
    fn memory(&self) -> &DeviceMemory {
        match *self {
            PotentialDedicatedAllocation::Generic(ref alloc) => alloc.memory(),
            PotentialDedicatedAllocation::Dedicated(ref mem) => mem,
            PotentialDedicatedAllocation::DedicatedMapped(ref mem) => mem.as_ref(),
        }
    }

    #[inline]
    fn offset(&self) -> usize {
        match *self {
            PotentialDedicatedAllocation::Generic(ref alloc) => alloc.offset(),
            PotentialDedicatedAllocation::Dedicated(_) => 0,
            PotentialDedicatedAllocation::DedicatedMapped(_) => 0,
        }
    }
}

impl<A> From<A> for PotentialDedicatedAllocation<A> {
    #[inline]
    fn from(alloc: A) -> PotentialDedicatedAllocation<A> {
        PotentialDedicatedAllocation::Generic(alloc)
    }
}
