// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use instance::MemoryType;
use memory::DeviceMemory;
use memory::MappedDeviceMemory;
use OomError;

pub use self::pool::StdMemoryPool;
pub use self::pool::StdMemoryPoolAlloc;
pub use self::host_visible::StdHostVisibleMemoryTypePool;
pub use self::host_visible::StdHostVisibleMemoryTypePoolAlloc;
pub use self::non_host_visible::StdNonHostVisibleMemoryTypePool;
pub use self::non_host_visible::StdNonHostVisibleMemoryTypePoolAlloc;

mod host_visible;
mod non_host_visible;
mod pool;

/// Pool of GPU-visible memory that can be allocated from.
pub unsafe trait MemoryPool: 'static + Send + Sync {
    /// Object that represents a single allocation. Its destructor should free the chunk.
    type Alloc: MemoryPoolAlloc;

    /// Allocates memory from the pool.
    ///
    /// # Panic
    ///
    /// - Panicks if `device` and `memory_type` don't belong to the same physical device.
    /// - Panicks if `size` is 0.
    /// - Panicks if `alignment` is 0.
    ///
    fn alloc(&Arc<Self>, ty: MemoryType, size: usize, alignment: usize)
             -> Result<Self::Alloc, OomError>;
}

/// Object that represents a single allocation. Its destructor should free the chunk.
pub unsafe trait MemoryPoolAlloc: 'static + Send + Sync {
    /// Returns the memory object from which this is allocated. Returns `None` if the memory is
    /// not mapped.
    fn mapped_memory(&self) -> Option<&MappedDeviceMemory>;

    /// Returns the memory object from which this is allocated.
    fn memory(&self) -> &DeviceMemory;

    /// Returns the offset at the start of the memory where the first byte of this allocation
    /// resides.
    fn offset(&self) -> usize;
}
