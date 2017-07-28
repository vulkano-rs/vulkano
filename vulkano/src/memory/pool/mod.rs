// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use instance::MemoryType;
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
pub unsafe trait MemoryPool {
    /// Object that represents a single allocation. Its destructor should free the chunk.
    type Alloc: MemoryPoolAlloc;

    /// Allocates memory from the pool.
    ///
    /// # Safety
    ///
    /// - The returned object must match the requirements.
    /// - When a linear object is allocated next to an optimal object, it is mandatory that
    ///   the boundary is aligned to the value of the `buffer_image_granularity` limit.
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
             map: MappingRequirement) -> Result<Self::Alloc, DeviceMemoryAllocError>;
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
