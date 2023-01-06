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
//! # Examples
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
//!   instance of this object. This is represented with the [`DescriptorSetLayout`] type in
//!   vulkano.
//! - A `DescriptorPool` is a Vulkan object that holds the memory of descriptor sets and that can
//!   be used to allocate and free individual descriptor sets. This is represented with the
//!   [`DescriptorPool`] type in vulkano.
//! - A `DescriptorSet` contains the bindings to resources and is allocated from a pool. This is
//!   represented with the [`UnsafeDescriptorSet`] type in vulkano.
//!
//! In addition to this, vulkano defines the following:
//!
//! - The [`DescriptorSetAllocator`] trait can be implemented on types from which you can allocate
//!   and free descriptor sets. However it is different from Vulkan descriptor pools in the sense
//!   that an implementation of the [`DescriptorSetAllocator`] trait can manage multiple Vulkan
//!   descriptor pools.
//! - The [`StandardDescriptorSetAllocator`] type is a default implementation of the
//!   [`DescriptorSetAllocator`] trait.
//! - The [`DescriptorSet`] trait is implemented on types that wrap around Vulkan descriptor sets in
//!   a safe way. A Vulkan descriptor set is inherently unsafe, so we need safe wrappers around
//!   them.
//! - The [`DescriptorSetsCollection`] trait is implemented on collections of types that implement
//!   [`DescriptorSet`]. It is what you pass to the draw functions.
//!
//! [`DescriptorPool`]: pool::DescriptorPool
//! [`DescriptorSetAllocator`]: allocator::DescriptorSetAllocator
//! [`StandardDescriptorSetAllocator`]: allocator::StandardDescriptorSetAllocator

pub(crate) use self::update::{check_descriptor_write, DescriptorWriteInfo};
pub use self::{
    collection::DescriptorSetsCollection,
    persistent::PersistentDescriptorSet,
    update::{DescriptorSetUpdateError, WriteDescriptorSet, WriteDescriptorSetElements},
};
use self::{layout::DescriptorSetLayout, sys::UnsafeDescriptorSet};
use crate::{
    buffer::{view::BufferView, Subbuffer},
    descriptor_set::layout::DescriptorType,
    device::DeviceOwned,
    image::view::ImageViewAbstract,
    sampler::Sampler,
    DeviceSize, OomError, VulkanObject,
};
use ahash::HashMap;
use smallvec::{smallvec, SmallVec};
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    hash::{Hash, Hasher},
    ops::Range,
    ptr,
    sync::Arc,
};

pub mod allocator;
mod collection;
pub mod layout;
pub mod persistent;
pub mod pool;
pub mod sys;
mod update;

/// Trait for objects that contain a collection of resources that will be accessible by shaders.
///
/// Objects of this type can be passed when submitting a draw command.
pub unsafe trait DescriptorSet: DeviceOwned + Send + Sync {
    /// Returns the inner `UnsafeDescriptorSet`.
    fn inner(&self) -> &UnsafeDescriptorSet;

    /// Returns the layout of this descriptor set.
    fn layout(&self) -> &Arc<DescriptorSetLayout>;

    /// Returns the variable descriptor count that this descriptor set was allocated with.
    fn variable_descriptor_count(&self) -> u32;

    /// Creates a [`DescriptorSetWithOffsets`] with the given dynamic offsets.
    fn offsets(
        self: Arc<Self>,
        dynamic_offsets: impl IntoIterator<Item = u32>,
    ) -> DescriptorSetWithOffsets
    where
        Self: Sized + 'static,
    {
        DescriptorSetWithOffsets::new(self, dynamic_offsets)
    }

    /// Returns the resources bound to this descriptor set.
    fn resources(&self) -> &DescriptorSetResources;
}

impl PartialEq for dyn DescriptorSet {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner()
    }
}

impl Eq for dyn DescriptorSet {}

impl Hash for dyn DescriptorSet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
    }
}

pub(crate) struct DescriptorSetInner {
    layout: Arc<DescriptorSetLayout>,
    variable_descriptor_count: u32,
    resources: DescriptorSetResources,
}

impl DescriptorSetInner {
    pub(crate) fn new(
        handle: ash::vk::DescriptorSet,
        layout: Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
    ) -> Result<Self, DescriptorSetUpdateError> {
        assert!(
            !layout.push_descriptor(),
            "the provided descriptor set layout is for push descriptors, and cannot be used to \
            build a descriptor set object",
        );

        let max_count = layout.variable_descriptor_count();

        assert!(
            variable_descriptor_count <= max_count,
            "the provided variable_descriptor_count ({}) is greater than the maximum number of \
            variable count descriptors in the layout ({})",
            variable_descriptor_count,
            max_count,
        );

        let mut resources = DescriptorSetResources::new(&layout, variable_descriptor_count);

        let descriptor_writes = descriptor_writes.into_iter();
        let (lower_size_bound, _) = descriptor_writes.size_hint();
        let mut descriptor_write_info: SmallVec<[_; 8]> = SmallVec::with_capacity(lower_size_bound);
        let mut write_descriptor_set: SmallVec<[_; 8]> = SmallVec::with_capacity(lower_size_bound);

        for write in descriptor_writes {
            let layout_binding =
                check_descriptor_write(&write, &layout, variable_descriptor_count)?;

            resources.update(&write);
            descriptor_write_info.push(write.to_vulkan_info(layout_binding.descriptor_type));
            write_descriptor_set.push(write.to_vulkan(handle, layout_binding.descriptor_type));
        }

        if !write_descriptor_set.is_empty() {
            for (info, write) in descriptor_write_info
                .iter()
                .zip(write_descriptor_set.iter_mut())
            {
                match info {
                    DescriptorWriteInfo::Image(info) => {
                        write.descriptor_count = info.len() as u32;
                        write.p_image_info = info.as_ptr();
                    }
                    DescriptorWriteInfo::Buffer(info) => {
                        write.descriptor_count = info.len() as u32;
                        write.p_buffer_info = info.as_ptr();
                    }
                    DescriptorWriteInfo::BufferView(info) => {
                        write.descriptor_count = info.len() as u32;
                        write.p_texel_buffer_view = info.as_ptr();
                    }
                }
            }
        }

        unsafe {
            let fns = layout.device().fns();

            (fns.v1_0.update_descriptor_sets)(
                layout.device().handle(),
                write_descriptor_set.len() as u32,
                write_descriptor_set.as_ptr(),
                0,
                ptr::null(),
            );
        }

        Ok(DescriptorSetInner {
            layout,
            variable_descriptor_count,
            resources,
        })
    }

    pub(crate) fn layout(&self) -> &Arc<DescriptorSetLayout> {
        &self.layout
    }

    pub(crate) fn resources(&self) -> &DescriptorSetResources {
        &self.resources
    }
}

/// The resources that are bound to a descriptor set.
#[derive(Clone)]
pub struct DescriptorSetResources {
    binding_resources: HashMap<u32, DescriptorBindingResources>,
}

impl DescriptorSetResources {
    /// Creates a new `DescriptorSetResources` matching the provided descriptor set layout, and
    /// all descriptors set to `None`.
    #[inline]
    pub fn new(layout: &DescriptorSetLayout, variable_descriptor_count: u32) -> Self {
        assert!(variable_descriptor_count <= layout.variable_descriptor_count());

        let binding_resources = layout
            .bindings()
            .iter()
            .map(|(&binding_num, binding)| {
                let count = if binding.variable_descriptor_count {
                    variable_descriptor_count
                } else {
                    binding.descriptor_count
                } as usize;

                let binding_resources = match binding.descriptor_type {
                    DescriptorType::UniformBuffer
                    | DescriptorType::StorageBuffer
                    | DescriptorType::UniformBufferDynamic
                    | DescriptorType::StorageBufferDynamic => {
                        DescriptorBindingResources::Buffer(smallvec![None; count])
                    }
                    DescriptorType::UniformTexelBuffer | DescriptorType::StorageTexelBuffer => {
                        DescriptorBindingResources::BufferView(smallvec![None; count])
                    }
                    DescriptorType::SampledImage
                    | DescriptorType::StorageImage
                    | DescriptorType::InputAttachment => {
                        DescriptorBindingResources::ImageView(smallvec![None; count])
                    }
                    DescriptorType::CombinedImageSampler => {
                        if binding.immutable_samplers.is_empty() {
                            DescriptorBindingResources::ImageViewSampler(smallvec![None; count])
                        } else {
                            DescriptorBindingResources::ImageView(smallvec![None; count])
                        }
                    }
                    DescriptorType::Sampler => {
                        if binding.immutable_samplers.is_empty() {
                            DescriptorBindingResources::Sampler(smallvec![None; count])
                        } else if layout.push_descriptor() {
                            // For push descriptors, no resource is written by default, this needs
                            // to be done explicitly via a dummy write.
                            DescriptorBindingResources::None(smallvec![None; count])
                        } else {
                            // For regular descriptor sets, all descriptors are considered valid
                            // from the start.
                            DescriptorBindingResources::None(smallvec![Some(()); count])
                        }
                    }
                };
                (binding_num, binding_resources)
            })
            .collect();

        Self { binding_resources }
    }

    /// Applies a descriptor write to the resources.
    ///
    /// # Panics
    ///
    /// - Panics if the binding number of a write does not exist in the resources.
    /// - See also [`DescriptorBindingResources::update`].
    #[inline]
    pub fn update(&mut self, write: &WriteDescriptorSet) {
        self.binding_resources
            .get_mut(&write.binding())
            .expect("descriptor write has invalid binding number")
            .update(write)
    }

    /// Returns a reference to the bound resources for `binding`. Returns `None` if the binding
    /// doesn't exist.
    #[inline]
    pub fn binding(&self, binding: u32) -> Option<&DescriptorBindingResources> {
        self.binding_resources.get(&binding)
    }
}

/// The resources that are bound to a single descriptor set binding.
#[derive(Clone)]
pub enum DescriptorBindingResources {
    None(Elements<()>),
    Buffer(Elements<(Subbuffer<[u8]>, Range<DeviceSize>)>),
    BufferView(Elements<Arc<BufferView>>),
    ImageView(Elements<Arc<dyn ImageViewAbstract>>),
    ImageViewSampler(Elements<(Arc<dyn ImageViewAbstract>, Arc<Sampler>)>),
    Sampler(Elements<Arc<Sampler>>),
}

type Elements<T> = SmallVec<[Option<T>; 1]>;

impl DescriptorBindingResources {
    /// Applies a descriptor write to the resources.
    ///
    /// # Panics
    ///
    /// - Panics if the resource types do not match.
    /// - Panics if the write goes out of bounds.
    #[inline]
    pub fn update(&mut self, write: &WriteDescriptorSet) {
        fn write_resources<T: Clone>(first: usize, resources: &mut [Option<T>], elements: &[T]) {
            resources
                .get_mut(first..first + elements.len())
                .expect("descriptor write for binding out of bounds")
                .iter_mut()
                .zip(elements)
                .for_each(|(resource, element)| {
                    *resource = Some(element.clone());
                });
        }

        let first = write.first_array_element() as usize;

        match (self, write.elements()) {
            (
                DescriptorBindingResources::None(resources),
                WriteDescriptorSetElements::None(num_elements),
            ) => {
                resources
                    .get_mut(first..first + *num_elements as usize)
                    .expect("descriptor write for binding out of bounds")
                    .iter_mut()
                    .for_each(|resource| {
                        *resource = Some(());
                    });
            }
            (
                DescriptorBindingResources::Buffer(resources),
                WriteDescriptorSetElements::Buffer(elements),
            ) => write_resources(first, resources, elements),
            (
                DescriptorBindingResources::BufferView(resources),
                WriteDescriptorSetElements::BufferView(elements),
            ) => write_resources(first, resources, elements),
            (
                DescriptorBindingResources::ImageView(resources),
                WriteDescriptorSetElements::ImageView(elements),
            ) => write_resources(first, resources, elements),
            (
                DescriptorBindingResources::ImageViewSampler(resources),
                WriteDescriptorSetElements::ImageViewSampler(elements),
            ) => write_resources(first, resources, elements),
            (
                DescriptorBindingResources::Sampler(resources),
                WriteDescriptorSetElements::Sampler(elements),
            ) => write_resources(first, resources, elements),
            _ => panic!(
                "descriptor write for binding {} has wrong resource type",
                write.binding(),
            ),
        }
    }
}

#[derive(Clone)]
pub struct DescriptorSetWithOffsets {
    descriptor_set: Arc<dyn DescriptorSet>,
    dynamic_offsets: SmallVec<[u32; 4]>,
}

impl DescriptorSetWithOffsets {
    pub fn new(
        descriptor_set: Arc<dyn DescriptorSet>,
        dynamic_offsets: impl IntoIterator<Item = u32>,
    ) -> Self {
        Self {
            descriptor_set,
            dynamic_offsets: dynamic_offsets.into_iter().collect(),
        }
    }

    #[inline]
    pub fn as_ref(&self) -> (&Arc<dyn DescriptorSet>, &[u32]) {
        (&self.descriptor_set, &self.dynamic_offsets)
    }

    #[inline]
    pub fn into_tuple(self) -> (Arc<dyn DescriptorSet>, impl ExactSizeIterator<Item = u32>) {
        (self.descriptor_set, self.dynamic_offsets.into_iter())
    }
}

impl<S> From<Arc<S>> for DescriptorSetWithOffsets
where
    S: DescriptorSet + 'static,
{
    fn from(descriptor_set: Arc<S>) -> Self {
        DescriptorSetWithOffsets::new(descriptor_set, std::iter::empty())
    }
}

#[derive(Clone, Copy, Debug)]
pub enum DescriptorSetCreationError {
    DescriptorSetUpdateError(DescriptorSetUpdateError),
    OomError(OomError),
}

impl Error for DescriptorSetCreationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::DescriptorSetUpdateError(err) => Some(err),
            Self::OomError(err) => Some(err),
        }
    }
}

impl Display for DescriptorSetCreationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::DescriptorSetUpdateError(_) => {
                write!(f, "an error occurred while updating the descriptor set")
            }
            Self::OomError(_) => write!(f, "out of memory"),
        }
    }
}

impl From<DescriptorSetUpdateError> for DescriptorSetCreationError {
    fn from(err: DescriptorSetUpdateError) -> Self {
        Self::DescriptorSetUpdateError(err)
    }
}

impl From<OomError> for DescriptorSetCreationError {
    fn from(err: OomError) -> Self {
        Self::OomError(err)
    }
}
