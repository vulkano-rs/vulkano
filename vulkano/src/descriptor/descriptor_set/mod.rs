// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Descriptor sets creation and management
//!
//! This module is dedicated to managing descriptor sets. There are three concepts in Vulkan
//! related to descriptor sets:
//!
//! - A `DescriptorSetLayout` is a Vulkan object that describes to the Vulkan implementation the
//!   layout of a future descriptor set. When you allocate a descriptor set, you have to pass an
//!   instance of this object. This is represented with the `UnsafeDescriptorSetLayout` type in
//!   vulkano.
//! - A `DescriptorPool` is a Vulkan object that holds the memory of descriptor sets and that can
//!   be used to allocate and free individual descriptor sets. This is represented with the
//!   `UnsafeDescriptorPool` type in vulkano.
//! - A `DescriptorSet` contains the bindings to resources and is allocated from a pool. This is
//!   represented with the `UnsafeDescriptorSet` type in vulkano.
//!
//! In addition to this, vulkano defines the following:
//!
//! - The `DescriptorPool` trait can be implemented on types from which you can allocate and free
//!   descriptor sets. However it is different from Vulkan descriptor pools in the sense that an
//!   implementation of the `DescriptorPool` trait can manage multiple Vulkan descriptor pools.
//! - The `StdDescriptorPool` type is a default implementation of the `DescriptorPool` trait.
//! - The `DescriptorSet` trait is implemented on types that wrap around Vulkan descriptor sets in
//!   a safe way. A Vulkan descriptor set is inherently unsafe, so we need safe wrappers around
//!   them.
//! - The `SimpleDescriptorSet` type is a default implementation of the `DescriptorSet` trait.
//! - The `DescriptorSetsCollection` trait is implemented on collections of types that implement
//!   `DescriptorSet`. It is what you pass to the draw functions.

use std::hash::Hash;
use std::hash::Hasher;

use buffer::BufferAccess;
use descriptor::descriptor::DescriptorDesc;
use device::DeviceOwned;
use image::ImageViewAccess;
use SafeDeref;
use VulkanObject;

pub use self::collection::DescriptorSetsCollection;
pub use self::fixed_size_pool::FixedSizeDescriptorSet;
pub use self::fixed_size_pool::FixedSizeDescriptorSetBuilder;
pub use self::fixed_size_pool::FixedSizeDescriptorSetBuilderArray;
pub use self::fixed_size_pool::FixedSizeDescriptorSetsPool;
pub use self::persistent::PersistentDescriptorSet;
pub use self::persistent::PersistentDescriptorSetBuf;
pub use self::persistent::PersistentDescriptorSetBufView;
pub use self::persistent::PersistentDescriptorSetBuildError;
pub use self::persistent::PersistentDescriptorSetBuilder;
pub use self::persistent::PersistentDescriptorSetBuilderArray;
pub use self::persistent::PersistentDescriptorSetError;
pub use self::persistent::PersistentDescriptorSetImg;
pub use self::persistent::PersistentDescriptorSetSampler;
pub use self::std_pool::StdDescriptorPool;
pub use self::std_pool::StdDescriptorPoolAlloc;
pub use self::sys::DescriptorPool;
pub use self::sys::DescriptorPoolAlloc;
pub use self::sys::DescriptorPoolAllocError;
pub use self::sys::DescriptorWrite;
pub use self::sys::DescriptorsCount;
pub use self::sys::UnsafeDescriptorPool;
pub use self::sys::UnsafeDescriptorPoolAllocIter;
pub use self::sys::UnsafeDescriptorSet;
pub use self::unsafe_layout::UnsafeDescriptorSetLayout;

pub mod collection;

mod fixed_size_pool;
mod persistent;
mod std_pool;
mod sys;
mod unsafe_layout;

/// Trait for objects that contain a collection of resources that will be accessible by shaders.
///
/// Objects of this type can be passed when submitting a draw command.
pub unsafe trait DescriptorSet: DescriptorSetDesc + DeviceOwned {
    /// Returns the inner `UnsafeDescriptorSet`.
    fn inner(&self) -> &UnsafeDescriptorSet;

    /// Returns the number of buffers within this descriptor set.
    fn num_buffers(&self) -> usize;

    /// Returns the `index`th buffer of this descriptor set, or `None` if out of range. Also
    /// returns the index of the descriptor that uses this buffer.
    ///
    /// The valid range is between 0 and `num_buffers()`.
    fn buffer(&self, index: usize) -> Option<(&dyn BufferAccess, u32)>;

    /// Returns the number of images within this descriptor set.
    fn num_images(&self) -> usize;

    /// Returns the `index`th image of this descriptor set, or `None` if out of range. Also returns
    /// the index of the descriptor that uses this image.
    ///
    /// The valid range is between 0 and `num_images()`.
    fn image(&self, index: usize) -> Option<(&dyn ImageViewAccess, u32)>;
}

unsafe impl<T> DescriptorSet for T
where
    T: SafeDeref,
    T::Target: DescriptorSet,
{
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        (**self).inner()
    }

    #[inline]
    fn num_buffers(&self) -> usize {
        (**self).num_buffers()
    }

    #[inline]
    fn buffer(&self, index: usize) -> Option<(&dyn BufferAccess, u32)> {
        (**self).buffer(index)
    }

    #[inline]
    fn num_images(&self) -> usize {
        (**self).num_images()
    }

    #[inline]
    fn image(&self, index: usize) -> Option<(&dyn ImageViewAccess, u32)> {
        (**self).image(index)
    }
}

impl PartialEq for dyn DescriptorSet + Send + Sync {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner().internal_object() == other.inner().internal_object()
            && self.device() == other.device()
    }
}

impl Eq for dyn DescriptorSet + Send + Sync {}

impl Hash for dyn DescriptorSet + Send + Sync {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().internal_object().hash(state);
        self.device().hash(state);
    }
}

/// Trait for objects that describe the layout of the descriptors of a set.
pub unsafe trait DescriptorSetDesc {
    /// Returns the number of binding slots in the set.
    fn num_bindings(&self) -> usize;

    /// Returns a description of a descriptor, or `None` if out of range.
    fn descriptor(&self, binding: usize) -> Option<DescriptorDesc>;
}

unsafe impl<T> DescriptorSetDesc for T
where
    T: SafeDeref,
    T::Target: DescriptorSetDesc,
{
    #[inline]
    fn num_bindings(&self) -> usize {
        (**self).num_bindings()
    }

    #[inline]
    fn descriptor(&self, binding: usize) -> Option<DescriptorDesc> {
        (**self).descriptor(binding)
    }
}
