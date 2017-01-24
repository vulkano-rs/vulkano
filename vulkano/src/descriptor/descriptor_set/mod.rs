// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
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

use std::sync::Arc;

use descriptor::descriptor::DescriptorDesc;

pub use self::collection::DescriptorSetsCollection;
pub use self::pool::DescriptorPool;
pub use self::pool::DescriptorPoolAlloc;
pub use self::pool::DescriptorPoolAllocError;
pub use self::pool::DescriptorWrite;
pub use self::pool::DescriptorsCount;
pub use self::pool::UnsafeDescriptorPool;
pub use self::pool::UnsafeDescriptorPoolAllocIter;
pub use self::pool::UnsafeDescriptorSet;
pub use self::std_pool::StdDescriptorPool;
pub use self::std_pool::StdDescriptorPoolAlloc;
pub use self::simple::*;
pub use self::unsafe_layout::UnsafeDescriptorSetLayout;

pub mod collection;

mod pool;
mod simple;
mod std_pool;
mod unsafe_layout;

/// Trait for objects that contain a collection of resources that will be accessible by shaders.
///
/// Objects of this type can be passed when submitting a draw command.
pub unsafe trait DescriptorSet {
    /// Returns the inner `UnsafeDescriptorSet`.
    fn inner(&self) -> &UnsafeDescriptorSet;
}

unsafe impl<T: ?Sized> DescriptorSet for Arc<T> where T: DescriptorSet {
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        (**self).inner()
    }
}

unsafe impl<'a, T: ?Sized> DescriptorSet for &'a T where T: 'a + DescriptorSet {
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        (**self).inner()
    }
}

/// Trait for objects that describe the layout of the descriptors of a set.
pub unsafe trait DescriptorSetDesc {
    /// Iterator that describes individual descriptors.
    type Iter: ExactSizeIterator<Item = DescriptorDesc>;

    /// Describes the layout of the descriptors of the pipeline.
    fn desc(&self) -> Self::Iter;
}

unsafe impl<T> DescriptorSetDesc for Arc<T> where T: DescriptorSetDesc {
    type Iter = <T as DescriptorSetDesc>::Iter;

    #[inline]
    fn desc(&self) -> Self::Iter {
        (**self).desc()
    }
}

unsafe impl<'a, T> DescriptorSetDesc for &'a T where T: 'a + DescriptorSetDesc {
    type Iter = <T as DescriptorSetDesc>::Iter;

    #[inline]
    fn desc(&self) -> Self::Iter {
        (**self).desc()
    }
}
