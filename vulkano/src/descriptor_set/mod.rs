// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Bindings between shaders and the resources they access.
//!
//! # Overview
//!
//! In order to access a buffer or an image from a shader, that buffer or image must be put in a
//! *descriptor*. Each descriptor contains one buffer or one image alongside with the way that it
//! can be accessed. A descriptor can also be an array, in which case it contains multiple buffers
//! or images that all have the same layout.
//!
//! Descriptors are grouped in what is called *descriptor sets*. In Vulkan you don't bind
//! individual descriptors one by one, but you create then bind descriptor sets one by one. As
//! binding a descriptor set has (small but non-null) a cost, you are encouraged to put descriptors
//! that are often used together in the same set so that you can keep the same set binding through
//! multiple draws.
//!
//! # Example
//!
//! > **Note**: This section describes the simple way to bind resources. There are more optimized
//! > ways.
//!
//! There are two steps to give access to a resource in a shader: creating the descriptor set, and
//! passing the descriptor sets when drawing.
//!
//! ## Creating a descriptor set
//!
//! TODO: write example for: PersistentDescriptorSet::start(layout.clone()).add_buffer(data_buffer.clone())
//!
//! ## Passing the descriptor set when drawing
//!
//! TODO: write
//!
//! # When drawing
//!
//! When you call a function that adds a draw command to a command buffer, one of the parameters
//! corresponds to the list of descriptor sets to use. Vulkano will check that what you passed is
//! compatible with the layout of the pipeline.
//!
//! TODO: talk about perfs of changing sets
//!
//! # Descriptor sets creation and management
//!
//! There are three concepts in Vulkan related to descriptor sets:
//!
//! - A `DescriptorSetLayout` is a Vulkan object that describes to the Vulkan implementation the
//!   layout of a future descriptor set. When you allocate a descriptor set, you have to pass an
//!   instance of this object. This is represented with the `DescriptorSetLayout` type in
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

pub use self::collection::DescriptorSetsCollection;
use self::descriptor::DescriptorDesc;
pub use self::fixed_size_pool::FixedSizeDescriptorSetsPool;
pub use self::layout::DescriptorSetLayout;
pub use self::persistent::PersistentDescriptorSet;
pub use self::persistent::PersistentDescriptorSetBuildError;
pub use self::persistent::PersistentDescriptorSetError;
pub use self::std_pool::StdDescriptorPool;
pub use self::std_pool::StdDescriptorPoolAlloc;
pub use self::sys::DescriptorWrite;
pub use self::sys::UnsafeDescriptorSet;
use crate::buffer::BufferAccess;
use crate::device::DeviceOwned;
use crate::image::view::ImageViewAbstract;
use crate::SafeDeref;
use crate::VulkanObject;
use std::hash::Hash;
use std::hash::Hasher;

pub mod collection;
pub mod descriptor;
pub mod fixed_size_pool;
mod layout;
pub mod persistent;
pub mod pool;
mod std_pool;
mod sys;

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
    fn image(&self, index: usize) -> Option<(&dyn ImageViewAbstract, u32)>;
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
    fn image(&self, index: usize) -> Option<(&dyn ImageViewAbstract, u32)> {
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
