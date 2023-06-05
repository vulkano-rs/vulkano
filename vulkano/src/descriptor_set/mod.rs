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

pub(crate) use self::update::DescriptorWriteInfo;
pub use self::{
    collection::DescriptorSetsCollection,
    persistent::PersistentDescriptorSet,
    update::{
        CopyDescriptorSet, DescriptorBufferInfo, DescriptorImageViewInfo, WriteDescriptorSet,
        WriteDescriptorSetElements,
    },
};
use self::{layout::DescriptorSetLayout, sys::UnsafeDescriptorSet};
use crate::{
    acceleration_structure::AccelerationStructure,
    buffer::view::BufferView,
    descriptor_set::layout::{
        DescriptorBindingFlags, DescriptorSetLayoutCreateFlags, DescriptorType,
    },
    device::DeviceOwned,
    image::ImageLayout,
    sampler::Sampler,
    ValidationError, VulkanObject,
};
use ahash::HashMap;
use smallvec::{smallvec, SmallVec};
use std::{
    hash::{Hash, Hasher},
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
        descriptor_copies: impl IntoIterator<Item = CopyDescriptorSet>,
    ) -> Result<Self, ValidationError> {
        assert!(
            !layout
                .flags()
                .intersects(DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR),
            "the provided descriptor set layout is for push descriptors, and cannot be used to \
            build a descriptor set object",
        );

        let max_variable_descriptor_count = layout.variable_descriptor_count();

        assert!(
            variable_descriptor_count <= max_variable_descriptor_count,
            "the provided variable_descriptor_count ({}) is greater than the maximum number of \
            variable count descriptors in the layout ({})",
            variable_descriptor_count,
            max_variable_descriptor_count,
        );

        let mut resources = DescriptorSetResources::new(&layout, variable_descriptor_count);

        struct PerDescriptorWrite {
            write_info: DescriptorWriteInfo,
            acceleration_structures: ash::vk::WriteDescriptorSetAccelerationStructureKHR,
            inline_uniform_block: ash::vk::WriteDescriptorSetInlineUniformBlock,
        }

        let writes_iter = descriptor_writes.into_iter();
        let (lower_size_bound, _) = writes_iter.size_hint();
        let mut writes_vk: SmallVec<[_; 8]> = SmallVec::with_capacity(lower_size_bound);
        let mut per_writes_vk: SmallVec<[_; 8]> = SmallVec::with_capacity(lower_size_bound);

        for (index, write) in writes_iter.enumerate() {
            write
                .validate(&layout, variable_descriptor_count)
                .map_err(|err| err.add_context(format!("descriptor_writes[{}]", index)))?;
            resources.write(&write, &layout);

            let layout_binding = &layout.bindings()[&write.binding()];
            writes_vk.push(write.to_vulkan(handle, layout_binding.descriptor_type));
            per_writes_vk.push(PerDescriptorWrite {
                write_info: write.to_vulkan_info(layout_binding.descriptor_type),
                acceleration_structures: Default::default(),
                inline_uniform_block: Default::default(),
            });
        }

        if !writes_vk.is_empty() {
            for (write_vk, per_write_vk) in writes_vk.iter_mut().zip(per_writes_vk.iter_mut()) {
                match &mut per_write_vk.write_info {
                    DescriptorWriteInfo::Image(info) => {
                        write_vk.descriptor_count = info.len() as u32;
                        write_vk.p_image_info = info.as_ptr();
                    }
                    DescriptorWriteInfo::Buffer(info) => {
                        write_vk.descriptor_count = info.len() as u32;
                        write_vk.p_buffer_info = info.as_ptr();
                    }
                    DescriptorWriteInfo::BufferView(info) => {
                        write_vk.descriptor_count = info.len() as u32;
                        write_vk.p_texel_buffer_view = info.as_ptr();
                    }
                    DescriptorWriteInfo::InlineUniformBlock(data) => {
                        write_vk.descriptor_count = data.len() as u32;
                        write_vk.p_next = &per_write_vk.inline_uniform_block as *const _ as _;
                        per_write_vk.inline_uniform_block.data_size = write_vk.descriptor_count;
                        per_write_vk.inline_uniform_block.p_data = data.as_ptr() as *const _;
                    }
                    DescriptorWriteInfo::AccelerationStructure(info) => {
                        write_vk.descriptor_count = info.len() as u32;
                        write_vk.p_next = &per_write_vk.acceleration_structures as *const _ as _;
                        per_write_vk
                            .acceleration_structures
                            .acceleration_structure_count = write_vk.descriptor_count;
                        per_write_vk
                            .acceleration_structures
                            .p_acceleration_structures = info.as_ptr();
                    }
                }
            }
        }

        let copies_iter = descriptor_copies.into_iter();
        let (lower_size_bound, _) = copies_iter.size_hint();
        let mut copies_vk: SmallVec<[_; 8]> = SmallVec::with_capacity(lower_size_bound);

        for (index, copy) in copies_iter.enumerate() {
            copy.validate(&layout, variable_descriptor_count)
                .map_err(|err| err.add_context(format!("descriptor_copies[{}]", index)))?;
            resources.copy(&copy);

            let &CopyDescriptorSet {
                ref src_set,
                src_binding,
                src_first_array_element,
                dst_binding,
                dst_first_array_element,
                descriptor_count,
                _ne: _,
            } = &copy;

            copies_vk.push(ash::vk::CopyDescriptorSet {
                src_set: src_set.inner().handle(),
                src_binding,
                src_array_element: src_first_array_element,
                dst_set: handle,
                dst_binding,
                dst_array_element: dst_first_array_element,
                descriptor_count,
                ..Default::default()
            });
        }

        unsafe {
            let fns = layout.device().fns();
            (fns.v1_0.update_descriptor_sets)(
                layout.device().handle(),
                writes_vk.len() as u32,
                writes_vk.as_ptr(),
                copies_vk.len() as u32,
                copies_vk.as_ptr(),
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
                let count = if binding
                    .binding_flags
                    .intersects(DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT)
                {
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
                        } else if layout
                            .flags()
                            .intersects(DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR)
                        {
                            // For push descriptors, no resource is written by default, this needs
                            // to be done explicitly via a dummy write.
                            DescriptorBindingResources::None(smallvec![None; count])
                        } else {
                            // For regular descriptor sets, all descriptors are considered valid
                            // from the start.
                            DescriptorBindingResources::None(smallvec![Some(()); count])
                        }
                    }
                    DescriptorType::InlineUniformBlock => {
                        DescriptorBindingResources::InlineUniformBlock
                    }
                    DescriptorType::AccelerationStructure => {
                        DescriptorBindingResources::AccelerationStructure(smallvec![None; count])
                    }
                };
                (binding_num, binding_resources)
            })
            .collect();

        Self { binding_resources }
    }

    /// Returns a reference to the bound resources for `binding`. Returns `None` if the binding
    /// doesn't exist.
    #[inline]
    pub fn binding(&self, binding: u32) -> Option<&DescriptorBindingResources> {
        self.binding_resources.get(&binding)
    }

    #[inline]
    pub(crate) fn write(&mut self, write: &WriteDescriptorSet, layout: &DescriptorSetLayout) {
        let descriptor_type = layout
            .bindings()
            .get(&write.binding())
            .expect("descriptor write has invalid binding number")
            .descriptor_type;
        self.binding_resources
            .get_mut(&write.binding())
            .expect("descriptor write has invalid binding number")
            .write(write, descriptor_type)
    }

    #[inline]
    pub(crate) fn copy(&mut self, copy: &CopyDescriptorSet) {
        let src = copy
            .src_set
            .resources()
            .binding_resources
            .get(&copy.src_binding)
            .expect("descriptor copy has invalid src_binding number");
        self.binding_resources
            .get_mut(&copy.dst_binding)
            .expect("descriptor copy has invalid dst_binding number")
            .copy(
                src,
                copy.src_first_array_element,
                copy.dst_first_array_element,
                copy.descriptor_count,
            );
    }
}

/// The resources that are bound to a single descriptor set binding.
#[derive(Clone)]
pub enum DescriptorBindingResources {
    None(Elements<()>),
    Buffer(Elements<DescriptorBufferInfo>),
    BufferView(Elements<Arc<BufferView>>),
    ImageView(Elements<DescriptorImageViewInfo>),
    ImageViewSampler(Elements<(DescriptorImageViewInfo, Arc<Sampler>)>),
    Sampler(Elements<Arc<Sampler>>),
    InlineUniformBlock,
    AccelerationStructure(Elements<Arc<AccelerationStructure>>),
}

type Elements<T> = SmallVec<[Option<T>; 1]>;

impl DescriptorBindingResources {
    pub(crate) fn write(&mut self, write: &WriteDescriptorSet, descriptor_type: DescriptorType) {
        fn write_resources<T: Clone>(
            first: usize,
            resources: &mut [Option<T>],
            elements: &[T],
            element_func: impl Fn(&T) -> T,
        ) {
            resources
                .get_mut(first..first + elements.len())
                .expect("descriptor write for binding out of bounds")
                .iter_mut()
                .zip(elements)
                .for_each(|(resource, element)| {
                    *resource = Some(element_func(element));
                });
        }

        let default_image_layout = descriptor_type.default_image_layout();
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
            ) => write_resources(first, resources, elements, Clone::clone),
            (
                DescriptorBindingResources::BufferView(resources),
                WriteDescriptorSetElements::BufferView(elements),
            ) => write_resources(first, resources, elements, Clone::clone),
            (
                DescriptorBindingResources::ImageView(resources),
                WriteDescriptorSetElements::ImageView(elements),
            ) => write_resources(first, resources, elements, |element| {
                let mut element = element.clone();

                if element.image_layout == ImageLayout::Undefined {
                    element.image_layout = default_image_layout;
                }

                element
            }),
            (
                DescriptorBindingResources::ImageViewSampler(resources),
                WriteDescriptorSetElements::ImageViewSampler(elements),
            ) => write_resources(first, resources, elements, |element| {
                let mut element = element.clone();

                if element.0.image_layout == ImageLayout::Undefined {
                    element.0.image_layout = default_image_layout;
                }

                element
            }),
            (
                DescriptorBindingResources::Sampler(resources),
                WriteDescriptorSetElements::Sampler(elements),
            ) => write_resources(first, resources, elements, Clone::clone),
            (
                DescriptorBindingResources::InlineUniformBlock,
                WriteDescriptorSetElements::InlineUniformBlock(_),
            ) => (),
            (
                DescriptorBindingResources::AccelerationStructure(resources),
                WriteDescriptorSetElements::AccelerationStructure(elements),
            ) => write_resources(first, resources, elements, Clone::clone),
            _ => panic!(
                "descriptor write for binding {} has wrong resource type",
                write.binding(),
            ),
        }
    }

    pub(crate) fn copy(
        &mut self,
        src: &DescriptorBindingResources,
        src_start: u32,
        dst_start: u32,
        count: u32,
    ) {
        let src_start = src_start as usize;
        let dst_start = dst_start as usize;
        let count = count as usize;

        match (src, self) {
            (DescriptorBindingResources::None(src), DescriptorBindingResources::None(dst)) => {
                dst[dst_start..dst_start + count]
                    .clone_from_slice(&src[src_start..src_start + count]);
            }
            (DescriptorBindingResources::Buffer(src), DescriptorBindingResources::Buffer(dst)) => {
                dst[dst_start..dst_start + count]
                    .clone_from_slice(&src[src_start..src_start + count]);
            }
            (
                DescriptorBindingResources::BufferView(src),
                DescriptorBindingResources::BufferView(dst),
            ) => {
                dst[dst_start..dst_start + count]
                    .clone_from_slice(&src[src_start..src_start + count]);
            }
            (
                DescriptorBindingResources::ImageView(src),
                DescriptorBindingResources::ImageView(dst),
            ) => {
                dst[dst_start..dst_start + count]
                    .clone_from_slice(&src[src_start..src_start + count]);
            }
            (
                DescriptorBindingResources::ImageViewSampler(src),
                DescriptorBindingResources::ImageViewSampler(dst),
            ) => {
                dst[dst_start..dst_start + count]
                    .clone_from_slice(&src[src_start..src_start + count]);
            }
            (
                DescriptorBindingResources::Sampler(src),
                DescriptorBindingResources::Sampler(dst),
            ) => {
                dst[dst_start..dst_start + count]
                    .clone_from_slice(&src[src_start..src_start + count]);
            }
            (
                DescriptorBindingResources::InlineUniformBlock,
                DescriptorBindingResources::InlineUniformBlock,
            ) => (),
            (
                DescriptorBindingResources::AccelerationStructure(src),
                DescriptorBindingResources::AccelerationStructure(dst),
            ) => {
                dst[dst_start..dst_start + count]
                    .clone_from_slice(&src[src_start..src_start + count]);
            }
            _ => panic!("descriptor copy has wrong resource type"),
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
