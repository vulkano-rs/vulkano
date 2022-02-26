// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! A pool from which descriptor sets can be allocated.

pub use self::{
    standard::StdDescriptorPool,
    sys::{
        DescriptorPoolAllocError, DescriptorSetAllocateInfo, UnsafeDescriptorPool,
        UnsafeDescriptorPoolCreateInfo,
    },
};
use super::{layout::DescriptorSetLayout, sys::UnsafeDescriptorSet};
use crate::{device::DeviceOwned, OomError};

pub mod standard;
mod sys;

/// A pool from which descriptor sets can be allocated.
///
/// Since the destructor of `Alloc` must free the descriptor set, this trait is usually implemented
/// on `Arc<T>` or `&'a T` and not `T` directly, so that the `Alloc` object can hold the pool.
pub unsafe trait DescriptorPool: DeviceOwned {
    /// Object that represented an allocated descriptor set.
    ///
    /// The destructor of this object should free the descriptor set.
    type Alloc: DescriptorPoolAlloc;

    /// Allocates a descriptor set.
    fn allocate(
        &mut self,
        layout: &DescriptorSetLayout,
        variable_descriptor_count: u32,
    ) -> Result<Self::Alloc, OomError>;
}

/// An allocated descriptor set.
pub trait DescriptorPoolAlloc: Send + Sync {
    /// Returns the inner unsafe descriptor set object.
    fn inner(&self) -> &UnsafeDescriptorSet;

    /// Returns the inner unsafe descriptor set object.
    fn inner_mut(&mut self) -> &mut UnsafeDescriptorSet;
}
