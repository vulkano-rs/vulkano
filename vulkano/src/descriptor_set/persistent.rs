// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! A simple, immutable descriptor set that is expected to be long-lived.
//!
//! Creating a persistent descriptor set allocates from a pool, and can't be modified once created.
//! You are therefore encouraged to create them at initialization and not the during
//! performance-critical paths.
//!
//! > **Note**: You can control of the pool that is used to create the descriptor set, if you wish
//! > so. By creating a implementation of the `DescriptorPool` trait that doesn't perform any
//! > actual allocation, you can skip this allocation and make it acceptable to use a persistent
//! > descriptor set in performance-critical paths..
//!
//! # Examples
//! TODO:

use super::{pool::DescriptorPoolAlloc, sys::UnsafeDescriptorSet, CopyDescriptorSet};
use crate::{
    descriptor_set::{
        allocator::{DescriptorSetAlloc, DescriptorSetAllocator, StandardDescriptorSetAlloc},
        update::WriteDescriptorSet,
        DescriptorSet, DescriptorSetLayout, DescriptorSetResources,
    },
    device::{Device, DeviceOwned},
    Validated, ValidationError, VulkanError, VulkanObject,
};
use smallvec::SmallVec;
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

/// A simple, immutable descriptor set that is expected to be long-lived.
pub struct PersistentDescriptorSet<P = StandardDescriptorSetAlloc> {
    inner: UnsafeDescriptorSet<P>,
    resources: DescriptorSetResources,
}

impl PersistentDescriptorSet {
    /// Creates and returns a new descriptor set with a variable descriptor count of 0.
    ///
    /// See `new_with_pool` for more.
    #[inline]
    pub fn new<A>(
        allocator: &A,
        layout: Arc<DescriptorSetLayout>,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
        descriptor_copies: impl IntoIterator<Item = CopyDescriptorSet>,
    ) -> Result<Arc<PersistentDescriptorSet<A::Alloc>>, Validated<VulkanError>>
    where
        A: DescriptorSetAllocator + ?Sized,
    {
        Self::new_variable(allocator, layout, 0, descriptor_writes, descriptor_copies)
    }

    /// Creates and returns a new descriptor set with the requested variable descriptor count,
    /// allocating it from the provided pool.
    ///
    /// # Panics
    ///
    /// - Panics if `layout` was created for push descriptors rather than descriptor sets.
    /// - Panics if `variable_descriptor_count` is too large for the given `layout`.
    pub fn new_variable<A>(
        allocator: &A,
        layout: Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
        descriptor_copies: impl IntoIterator<Item = CopyDescriptorSet>,
    ) -> Result<Arc<PersistentDescriptorSet<A::Alloc>>, Validated<VulkanError>>
    where
        A: DescriptorSetAllocator + ?Sized,
    {
        let mut set = PersistentDescriptorSet {
            inner: UnsafeDescriptorSet::new(allocator, &layout, variable_descriptor_count)?,
            resources: DescriptorSetResources::new(&layout, variable_descriptor_count),
        };

        unsafe {
            set.update(descriptor_writes, descriptor_copies)?;
        }

        Ok(Arc::new(set))
    }
}

impl<P> PersistentDescriptorSet<P>
where
    P: DescriptorSetAlloc,
{
    unsafe fn update(
        &mut self,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
        descriptor_copies: impl IntoIterator<Item = CopyDescriptorSet>,
    ) -> Result<(), Box<ValidationError>> {
        let descriptor_writes: SmallVec<[_; 8]> = descriptor_writes.into_iter().collect();
        let descriptor_copies: SmallVec<[_; 8]> = descriptor_copies.into_iter().collect();

        unsafe {
            self.inner.update(&descriptor_writes, &descriptor_copies)?;
        }

        for write in descriptor_writes {
            self.resources.write(&write, self.inner.layout());
        }

        for copy in descriptor_copies {
            self.resources.copy(&copy);
        }

        Ok(())
    }
}

unsafe impl<P> DescriptorSet for PersistentDescriptorSet<P>
where
    P: DescriptorSetAlloc,
{
    fn alloc(&self) -> &DescriptorPoolAlloc {
        self.inner.alloc().inner()
    }

    fn resources(&self) -> &DescriptorSetResources {
        &self.resources
    }
}

unsafe impl<P> VulkanObject for PersistentDescriptorSet<P>
where
    P: DescriptorSetAlloc,
{
    type Handle = ash::vk::DescriptorSet;

    fn handle(&self) -> Self::Handle {
        self.inner.handle()
    }
}

unsafe impl<P> DeviceOwned for PersistentDescriptorSet<P>
where
    P: DescriptorSetAlloc,
{
    fn device(&self) -> &Arc<Device> {
        self.layout().device()
    }
}

impl<P> PartialEq for PersistentDescriptorSet<P>
where
    P: DescriptorSetAlloc,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<P> Eq for PersistentDescriptorSet<P> where P: DescriptorSetAlloc {}

impl<P> Hash for PersistentDescriptorSet<P>
where
    P: DescriptorSetAlloc,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}
