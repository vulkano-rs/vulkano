// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! In the Vulkan API, descriptor sets must be allocated from *descriptor pools*.
//!
//! A descriptor pool holds and manages the memory of one or more descriptor sets. If you destroy a
//! descriptor pool, all of its descriptor sets are automatically destroyed.
//!
//! In vulkano, creating a descriptor set requires passing an implementation of the
//! [`DescriptorSetAllocator`] trait, which you can implement yourself or use the vulkano-provided
//! [`StandardDescriptorSetAllocator`].

pub use self::standard::StandardDescriptorSetAllocator;
use super::{layout::DescriptorSetLayout, sys::UnsafeDescriptorSet};
use crate::{device::DeviceOwned, OomError};
use std::sync::Arc;

pub mod standard;

/// Types that manage the memory of descriptor sets.
///
/// # Safety
///
/// A Vulkan descriptor pool must be externally synchronized as if it owned the descriptor sets that
/// were allocated from it. This includes allocating from the pool, freeing from the pool and
/// resetting the pool or individual descriptor sets. The implementation of `DescriptorSetAllocator`
/// is expected to manage this.
///
/// The destructor of the [`DescriptorSetAlloc`] is expected to free the descriptor set, reset the
/// descriptor set, or add it to a pool so that it gets reused. If the implementation frees or
/// resets the descriptor set, it must not forget that this operation must be externally
/// synchronized.
pub unsafe trait DescriptorSetAllocator: DeviceOwned {
    /// Object that represented an allocated descriptor set.
    ///
    /// The destructor of this object should free the descriptor set.
    type Alloc: DescriptorSetAlloc;

    /// Allocates a descriptor set.
    fn allocate(
        &self,
        layout: &Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
    ) -> Result<Self::Alloc, OomError>;
}

/// An allocated descriptor set.
pub trait DescriptorSetAlloc: Send + Sync {
    /// Returns the inner unsafe descriptor set object.
    fn inner(&self) -> &UnsafeDescriptorSet;

    /// Returns the inner unsafe descriptor set object.
    fn inner_mut(&mut self) -> &mut UnsafeDescriptorSet;
}
