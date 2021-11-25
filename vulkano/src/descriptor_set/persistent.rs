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

use crate::buffer::BufferView;
use crate::descriptor_set::builder::DescriptorSetBuilder;
use crate::descriptor_set::pool::standard::StdDescriptorPoolAlloc;
use crate::descriptor_set::pool::{DescriptorPool, DescriptorPoolAlloc};
use crate::descriptor_set::resources::DescriptorSetResources;
use crate::descriptor_set::{
    BufferAccess, DescriptorSet, DescriptorSetError, DescriptorSetLayout, UnsafeDescriptorSet,
};
use crate::device::{Device, DeviceOwned};
use crate::image::ImageViewAbstract;
use crate::sampler::Sampler;
use crate::VulkanObject;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// A simple, immutable descriptor set that is expected to be long-lived.
pub struct PersistentDescriptorSet<P = StdDescriptorPoolAlloc> {
    alloc: P,
    resources: DescriptorSetResources,
    layout: Arc<DescriptorSetLayout>,
}

impl PersistentDescriptorSet {
    /// Starts the process of building a `PersistentDescriptorSet`. Returns a builder.
    pub fn start(layout: Arc<DescriptorSetLayout>) -> PersistentDescriptorSetBuilder {
        assert!(
            !layout.desc().is_push_descriptor(),
            "the provided descriptor set layout is for push descriptors, and cannot be used to build a descriptor set object"
        );

        PersistentDescriptorSetBuilder {
            inner: DescriptorSetBuilder::start(layout),
        }
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
        &self.layout
    }

    #[inline]
    fn resources(&self) -> &DescriptorSetResources {
        &self.resources
    }
}

unsafe impl<P> DeviceOwned for PersistentDescriptorSet<P>
where
    P: DescriptorPoolAlloc,
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.layout.device()
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

/// Prototype of a `PersistentDescriptorSet`.
pub struct PersistentDescriptorSetBuilder {
    inner: DescriptorSetBuilder,
}

impl PersistentDescriptorSetBuilder {
    /// Call this function if the next element of the set is an array in order to set the value of
    /// each element.
    ///
    /// Returns an error if the descriptor is empty, there are no remaining descriptors, or if the
    /// builder is already in an error.
    ///
    /// This function can be called even if the descriptor isn't an array, and it is valid to enter
    /// the "array", add one element, then leave.
    #[inline]
    pub fn enter_array(&mut self) -> Result<&mut Self, DescriptorSetError> {
        self.inner.enter_array()?;
        Ok(self)
    }

    /// Leaves the array. Call this once you added all the elements of the array.
    ///
    /// Returns an error if the array is missing elements, or if the builder is not in an array.
    #[inline]
    pub fn leave_array(&mut self) -> Result<&mut Self, DescriptorSetError> {
        self.inner.leave_array()?;
        Ok(self)
    }

    /// Skips the current descriptor if it is empty.
    #[inline]
    pub fn add_empty(&mut self) -> Result<&mut Self, DescriptorSetError> {
        self.inner.add_empty()?;
        Ok(self)
    }

    /// Binds a buffer as the next descriptor.
    ///
    /// An error is returned if the buffer isn't compatible with the descriptor.
    #[inline]
    pub fn add_buffer(
        &mut self,
        buffer: Arc<dyn BufferAccess>,
    ) -> Result<&mut Self, DescriptorSetError> {
        self.inner.add_buffer(buffer)?;
        Ok(self)
    }

    /// Binds a buffer view as the next descriptor.
    ///
    /// An error is returned if the buffer isn't compatible with the descriptor.
    #[inline]
    pub fn add_buffer_view<B>(
        &mut self,
        view: Arc<BufferView<B>>,
    ) -> Result<&mut Self, DescriptorSetError>
    where
        B: BufferAccess + 'static,
    {
        self.inner.add_buffer_view(view)?;
        Ok(self)
    }

    /// Binds an image view as the next descriptor.
    ///
    /// An error is returned if the image view isn't compatible with the descriptor.
    #[inline]
    pub fn add_image(
        &mut self,
        image_view: Arc<dyn ImageViewAbstract>,
    ) -> Result<&mut Self, DescriptorSetError> {
        self.inner.add_image(image_view)?;
        Ok(self)
    }

    /// Binds an image view with a sampler as the next descriptor.
    ///
    /// If the descriptor set layout contains immutable samplers for this descriptor, use
    /// `add_image` instead.
    ///
    /// An error is returned if the image view isn't compatible with the descriptor.
    #[inline]
    pub fn add_sampled_image(
        &mut self,
        image_view: Arc<dyn ImageViewAbstract>,
        sampler: Arc<Sampler>,
    ) -> Result<&mut Self, DescriptorSetError> {
        self.inner.add_sampled_image(image_view, sampler)?;
        Ok(self)
    }

    /// Binds a sampler as the next descriptor.
    ///
    /// An error is returned if the sampler isn't compatible with the descriptor.
    #[inline]
    pub fn add_sampler(&mut self, sampler: Arc<Sampler>) -> Result<&mut Self, DescriptorSetError> {
        self.inner.add_sampler(sampler)?;
        Ok(self)
    }

    /// Builds a `PersistentDescriptorSet` from the builder.
    #[inline]
    pub fn build(
        self,
    ) -> Result<Arc<PersistentDescriptorSet<StdDescriptorPoolAlloc>>, DescriptorSetError> {
        let mut pool = Device::standard_descriptor_pool(self.inner.device());
        self.build_with_pool(&mut pool)
    }

    /// Builds a `PersistentDescriptorSet` from the builder.
    pub fn build_with_pool<P>(
        self,
        pool: &mut P,
    ) -> Result<Arc<PersistentDescriptorSet<P::Alloc>>, DescriptorSetError>
    where
        P: ?Sized + DescriptorPool,
    {
        let writes = self.inner.build()?;
        let mut alloc = pool.alloc(writes.layout(), writes.variable_descriptor_count())?;
        let mut resources =
            DescriptorSetResources::new(writes.layout(), writes.variable_descriptor_count());

        unsafe {
            alloc.inner_mut().write(writes.layout(), writes.writes());
            resources.update(writes.writes());
        }

        Ok(Arc::new(PersistentDescriptorSet {
            alloc,
            resources,
            layout: writes.layout().clone(),
        }))
    }
}
