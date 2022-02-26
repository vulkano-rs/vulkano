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
//! # Example
//! TODO:

use crate::descriptor_set::pool::standard::StdDescriptorPoolAlloc;
use crate::descriptor_set::pool::{DescriptorPool, DescriptorPoolAlloc};
use crate::descriptor_set::update::WriteDescriptorSet;
use crate::descriptor_set::{
    DescriptorSet, DescriptorSetCreationError, DescriptorSetInner, DescriptorSetLayout,
    DescriptorSetResources, UnsafeDescriptorSet,
};
use crate::device::{Device, DeviceOwned};
use crate::VulkanObject;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// A simple, immutable descriptor set that is expected to be long-lived.
pub struct PersistentDescriptorSet<P = StdDescriptorPoolAlloc> {
    alloc: P,
    inner: DescriptorSetInner,
}

impl PersistentDescriptorSet {
    /// Creates and returns a new descriptor set with a variable descriptor count of 0.
    ///
    /// See `new_with_pool` for more.
    #[inline]
    pub fn new(
        layout: Arc<DescriptorSetLayout>,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
    ) -> Result<Arc<PersistentDescriptorSet>, DescriptorSetCreationError> {
        let mut pool = Device::standard_descriptor_pool(layout.device());
        Self::new_with_pool(layout, 0, &mut pool, descriptor_writes)
    }

    /// Creates and returns a new descriptor set with the requested variable descriptor count.
    ///
    /// See `new_with_pool` for more.
    #[inline]
    pub fn new_variable(
        layout: Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
    ) -> Result<Arc<PersistentDescriptorSet>, DescriptorSetCreationError> {
        let mut pool = Device::standard_descriptor_pool(layout.device());
        Self::new_with_pool(
            layout,
            variable_descriptor_count,
            &mut pool,
            descriptor_writes,
        )
    }

    /// Creates and returns a new descriptor set with the requested variable descriptor count,
    /// allocating it from the provided pool.
    ///
    /// # Panics
    ///
    /// - Panics if `layout` was created for push descriptors rather than descriptor sets.
    /// - Panics if `variable_descriptor_count` is too large for the given `layout`.
    pub fn new_with_pool<P>(
        layout: Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
        pool: &mut P,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
    ) -> Result<Arc<PersistentDescriptorSet<P::Alloc>>, DescriptorSetCreationError>
    where
        P: ?Sized + DescriptorPool,
    {
        assert!(
            !layout.push_descriptor(),
            "the provided descriptor set layout is for push descriptors, and cannot be used to build a descriptor set object"
        );

        let max_count = layout.variable_descriptor_count();

        assert!(
            variable_descriptor_count <= max_count,
            "the provided variable_descriptor_count ({}) is greater than the maximum number of variable count descriptors in the set ({})",
            variable_descriptor_count,
            max_count,
        );

        let alloc = pool.allocate(&layout, variable_descriptor_count)?;
        let inner = DescriptorSetInner::new(
            alloc.inner().internal_object(),
            layout,
            variable_descriptor_count,
            descriptor_writes,
        )?;

        Ok(Arc::new(PersistentDescriptorSet { alloc, inner }))
    }
}

unsafe impl<P> DescriptorSet for PersistentDescriptorSet<P>
where
    P: DescriptorPoolAlloc,
{
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        self.alloc.inner()
    }

    #[inline]
    fn layout(&self) -> &Arc<DescriptorSetLayout> {
        self.inner.layout()
    }

    #[inline]
    fn resources(&self) -> &DescriptorSetResources {
        self.inner.resources()
    }
}

unsafe impl<P> DeviceOwned for PersistentDescriptorSet<P>
where
    P: DescriptorPoolAlloc,
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.layout().device()
    }
}

impl<P> PartialEq for PersistentDescriptorSet<P>
where
    P: DescriptorPoolAlloc,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner().internal_object() == other.inner().internal_object()
            && self.device() == other.device()
    }
}

impl<P> Eq for PersistentDescriptorSet<P> where P: DescriptorPoolAlloc {}

impl<P> Hash for PersistentDescriptorSet<P>
where
    P: DescriptorPoolAlloc,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().internal_object().hash(state);
        self.device().hash(state);
    }
}
