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
//! TODO: write
//!
//! ## Passing the descriptor set when drawing
//!
//! TODO: write
//!
//! # When drawing
//!
//! When you call a function that adds a draw command to a command buffer, vulkano will check that
//! the descriptor sets you bound are compatible with the layout of the pipeline.
//!
//! TODO: talk about perfs of changing sets
//!
//! # Descriptor sets creation and management
//!
//! There are three concepts in Vulkan related to descriptor sets:
//!
//! - A `VkDescriptorSetLayout` is a Vulkan object that describes to the Vulkan implementation the
//!   layout of a future descriptor set. When you allocate a descriptor set, you have to pass an
//!   instance of this object. This is represented with the [`DescriptorSetLayout`] type in
//!   vulkano.
//! - A `VkDescriptorPool` is a Vulkan object that holds the memory of descriptor sets and that can
//!   be used to allocate and free individual descriptor sets. This is represented with the
//!   [`DescriptorPool`] type in vulkano.
//! - A `VkDescriptorSet` contains the bindings to resources and is allocated from a pool. This is
//!   represented with the [`UnsafeDescriptorSet`] type in vulkano.
//!
//! In addition to this, vulkano defines the following:
//!
//! - The [`DescriptorSetAllocator`] trait can be implemented on types from which you can allocate
//!   and free descriptor sets. However it is different from Vulkan descriptor pools in the sense
//!   that an implementation of the `DescriptorSetAllocator` trait can manage multiple Vulkan
//!   descriptor pools.
//! - The [`StandardDescriptorSetAllocator`] type is a default implementation of the
//!   [`DescriptorSetAllocator`] trait.
//! - The [`DescriptorSet`] type wraps around an `UnsafeDescriptorSet` a safe way. A Vulkan
//!   descriptor set is inherently unsafe, so we need safe wrappers around them.
//! - The [`DescriptorSetsCollection`] trait is implemented on collections of descriptor sets. It
//!   is what you pass to the bind function.
//!
//! [`DescriptorPool`]: pool::DescriptorPool
//! [`UnsafeDescriptorSet`]: sys::UnsafeDescriptorSet
//! [`DescriptorSetAllocator`]: allocator::DescriptorSetAllocator
//! [`StandardDescriptorSetAllocator`]: allocator::StandardDescriptorSetAllocator

pub(crate) use self::update::DescriptorWriteInfo;
use self::{
    allocator::DescriptorSetAllocator,
    layout::DescriptorSetLayout,
    pool::{DescriptorPool, DescriptorPoolAlloc},
    sys::UnsafeDescriptorSet,
};
pub use self::{
    collection::DescriptorSetsCollection,
    update::{
        CopyDescriptorSet, DescriptorBufferInfo, DescriptorImageViewInfo, WriteDescriptorSet,
        WriteDescriptorSetElements,
    },
};
use crate::{
    acceleration_structure::AccelerationStructure,
    buffer::view::BufferView,
    descriptor_set::layout::{
        DescriptorBindingFlags, DescriptorSetLayoutCreateFlags, DescriptorType,
    },
    device::{Device, DeviceOwned},
    image::{sampler::Sampler, ImageLayout},
    Validated, ValidationError, VulkanError, VulkanObject,
};
use ahash::HashMap;
use parking_lot::{RwLock, RwLockReadGuard};
use smallvec::{smallvec, SmallVec};
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

pub mod allocator;
mod collection;
pub mod layout;
pub mod pool;
pub mod sys;
mod update;

/// An object that contains a collection of resources that will be accessible by shaders.
///
/// Descriptor sets can be bound when recording a command buffer.
#[derive(Debug)]
pub struct DescriptorSet {
    inner: UnsafeDescriptorSet,
    resources: RwLock<DescriptorSetResources>,
}

impl DescriptorSet {
    /// Creates and returns a new descriptor set with a variable descriptor count of 0.
    pub fn new(
        allocator: Arc<dyn DescriptorSetAllocator>,
        layout: Arc<DescriptorSetLayout>,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
        descriptor_copies: impl IntoIterator<Item = CopyDescriptorSet>,
    ) -> Result<Arc<DescriptorSet>, Validated<VulkanError>> {
        Self::new_variable(allocator, layout, 0, descriptor_writes, descriptor_copies)
    }

    /// Creates and returns a new descriptor set with the requested variable descriptor count.
    pub fn new_variable(
        allocator: Arc<dyn DescriptorSetAllocator>,
        layout: Arc<DescriptorSetLayout>,
        variable_descriptor_count: u32,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
        descriptor_copies: impl IntoIterator<Item = CopyDescriptorSet>,
    ) -> Result<Arc<DescriptorSet>, Validated<VulkanError>> {
        let mut set = DescriptorSet {
            inner: UnsafeDescriptorSet::new(allocator, &layout, variable_descriptor_count)?,
            resources: RwLock::new(DescriptorSetResources::new(
                &layout,
                variable_descriptor_count,
            )),
        };

        set.update(descriptor_writes, descriptor_copies)?;

        Ok(Arc::new(set))
    }

    /// Returns the allocation of the descriptor set.
    #[inline]
    pub fn alloc(&self) -> &DescriptorPoolAlloc {
        &self.inner.alloc().inner
    }

    /// Returns the descriptor pool that the descriptor set was allocated from.
    #[inline]
    pub fn pool(&self) -> &DescriptorPool {
        self.inner.pool()
    }

    /// Returns the layout of this descriptor set.
    #[inline]
    pub fn layout(&self) -> &Arc<DescriptorSetLayout> {
        self.alloc().layout()
    }

    /// Returns the variable descriptor count that this descriptor set was allocated with.
    #[inline]
    pub fn variable_descriptor_count(&self) -> u32 {
        self.alloc().variable_descriptor_count()
    }

    /// Creates a [`DescriptorSetWithOffsets`] with the given dynamic offsets.
    pub fn offsets(
        self: Arc<Self>,
        dynamic_offsets: impl IntoIterator<Item = u32>,
    ) -> DescriptorSetWithOffsets {
        DescriptorSetWithOffsets::new(self, dynamic_offsets)
    }

    /// Returns the resources bound to this descriptor set.
    #[inline]
    pub fn resources(&self) -> RwLockReadGuard<'_, DescriptorSetResources> {
        self.resources.read()
    }

    /// Updates the descriptor set with new values.
    pub fn update(
        &mut self,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
        descriptor_copies: impl IntoIterator<Item = CopyDescriptorSet>,
    ) -> Result<(), Box<ValidationError>> {
        let descriptor_writes: SmallVec<[_; 8]> = descriptor_writes.into_iter().collect();
        let descriptor_copies: SmallVec<[_; 8]> = descriptor_copies.into_iter().collect();

        self.inner
            .validate_update(&descriptor_writes, &descriptor_copies)?;

        unsafe {
            Self::update_inner(
                &self.inner,
                self.resources.get_mut(),
                &descriptor_writes,
                &descriptor_copies,
            );
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn update_unchecked(
        &mut self,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
        descriptor_copies: impl IntoIterator<Item = CopyDescriptorSet>,
    ) {
        let descriptor_writes: SmallVec<[_; 8]> = descriptor_writes.into_iter().collect();
        let descriptor_copies: SmallVec<[_; 8]> = descriptor_copies.into_iter().collect();

        Self::update_inner(
            &self.inner,
            self.resources.get_mut(),
            &descriptor_writes,
            &descriptor_copies,
        );
    }

    /// Updates the descriptor set with new values.
    ///
    /// # Safety
    ///
    /// - Host access to the descriptor set must be externally synchronized.
    pub unsafe fn update_by_ref(
        &self,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
        descriptor_copies: impl IntoIterator<Item = CopyDescriptorSet>,
    ) -> Result<(), Box<ValidationError>> {
        let descriptor_writes: SmallVec<[_; 8]> = descriptor_writes.into_iter().collect();
        let descriptor_copies: SmallVec<[_; 8]> = descriptor_copies.into_iter().collect();

        self.inner
            .validate_update(&descriptor_writes, &descriptor_copies)?;

        Self::update_inner(
            &self.inner,
            &mut self.resources.write(),
            &descriptor_writes,
            &descriptor_copies,
        );

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn update_by_ref_unchecked(
        &self,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
        descriptor_copies: impl IntoIterator<Item = CopyDescriptorSet>,
    ) {
        let descriptor_writes: SmallVec<[_; 8]> = descriptor_writes.into_iter().collect();
        let descriptor_copies: SmallVec<[_; 8]> = descriptor_copies.into_iter().collect();

        Self::update_inner(
            &self.inner,
            &mut self.resources.write(),
            &descriptor_writes,
            &descriptor_copies,
        );
    }

    unsafe fn update_inner(
        inner: &UnsafeDescriptorSet,
        resources: &mut DescriptorSetResources,
        descriptor_writes: &[WriteDescriptorSet],
        descriptor_copies: &[CopyDescriptorSet],
    ) {
        inner.update_unchecked(descriptor_writes, descriptor_copies);

        for write in descriptor_writes {
            resources.write(write, inner.layout());
        }

        for copy in descriptor_copies {
            resources.copy(copy);
        }
    }
}

unsafe impl VulkanObject for DescriptorSet {
    type Handle = ash::vk::DescriptorSet;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.inner.handle()
    }
}

unsafe impl DeviceOwned for DescriptorSet {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

impl PartialEq for DescriptorSet {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Eq for DescriptorSet {}

impl Hash for DescriptorSet {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

/// The resources that are bound to a descriptor set.
#[derive(Clone, Debug)]
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
        let resources = copy.src_set.resources();
        let src = resources
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
#[derive(Clone, Debug)]
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

        match write.elements() {
            WriteDescriptorSetElements::None(num_elements) => match self {
                DescriptorBindingResources::None(resources) => {
                    resources
                        .get_mut(first..first + *num_elements as usize)
                        .expect("descriptor write for binding out of bounds")
                        .iter_mut()
                        .for_each(|resource| {
                            *resource = Some(());
                        });
                }
                _ => panic!(
                    "descriptor write for binding {} has wrong resource type",
                    write.binding(),
                ),
            },
            WriteDescriptorSetElements::Buffer(elements) => match self {
                DescriptorBindingResources::Buffer(resources) => {
                    write_resources(first, resources, elements, Clone::clone)
                }
                _ => panic!(
                    "descriptor write for binding {} has wrong resource type",
                    write.binding(),
                ),
            },
            WriteDescriptorSetElements::BufferView(elements) => match self {
                DescriptorBindingResources::BufferView(resources) => {
                    write_resources(first, resources, elements, Clone::clone)
                }
                _ => panic!(
                    "descriptor write for binding {} has wrong resource type",
                    write.binding(),
                ),
            },
            WriteDescriptorSetElements::ImageView(elements) => match self {
                DescriptorBindingResources::ImageView(resources) => {
                    write_resources(first, resources, elements, |element| {
                        let mut element = element.clone();

                        if element.image_layout == ImageLayout::Undefined {
                            element.image_layout = default_image_layout;
                        }

                        element
                    })
                }
                _ => panic!(
                    "descriptor write for binding {} has wrong resource type",
                    write.binding(),
                ),
            },
            WriteDescriptorSetElements::ImageViewSampler(elements) => match self {
                DescriptorBindingResources::ImageViewSampler(resources) => {
                    write_resources(first, resources, elements, |element| {
                        let mut element = element.clone();

                        if element.0.image_layout == ImageLayout::Undefined {
                            element.0.image_layout = default_image_layout;
                        }

                        element
                    })
                }
                _ => panic!(
                    "descriptor write for binding {} has wrong resource type",
                    write.binding(),
                ),
            },
            WriteDescriptorSetElements::Sampler(elements) => match self {
                DescriptorBindingResources::Sampler(resources) => {
                    write_resources(first, resources, elements, Clone::clone)
                }
                _ => panic!(
                    "descriptor write for binding {} has wrong resource type",
                    write.binding(),
                ),
            },
            WriteDescriptorSetElements::InlineUniformBlock(_) => match self {
                DescriptorBindingResources::InlineUniformBlock => (),
                _ => panic!(
                    "descriptor write for binding {} has wrong resource type",
                    write.binding(),
                ),
            },
            WriteDescriptorSetElements::AccelerationStructure(elements) => match self {
                DescriptorBindingResources::AccelerationStructure(resources) => {
                    write_resources(first, resources, elements, Clone::clone)
                }
                _ => panic!(
                    "descriptor write for binding {} has wrong resource type",
                    write.binding(),
                ),
            },
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

        match src {
            DescriptorBindingResources::None(src) => match self {
                DescriptorBindingResources::None(dst) => dst[dst_start..dst_start + count]
                    .clone_from_slice(&src[src_start..src_start + count]),
                _ => panic!("descriptor copy has wrong resource type"),
            },
            DescriptorBindingResources::Buffer(src) => match self {
                DescriptorBindingResources::Buffer(dst) => dst[dst_start..dst_start + count]
                    .clone_from_slice(&src[src_start..src_start + count]),
                _ => panic!("descriptor copy has wrong resource type"),
            },
            DescriptorBindingResources::BufferView(src) => match self {
                DescriptorBindingResources::BufferView(dst) => dst[dst_start..dst_start + count]
                    .clone_from_slice(&src[src_start..src_start + count]),
                _ => panic!("descriptor copy has wrong resource type"),
            },
            DescriptorBindingResources::ImageView(src) => match self {
                DescriptorBindingResources::ImageView(dst) => dst[dst_start..dst_start + count]
                    .clone_from_slice(&src[src_start..src_start + count]),
                _ => panic!("descriptor copy has wrong resource type"),
            },
            DescriptorBindingResources::ImageViewSampler(src) => match self {
                DescriptorBindingResources::ImageViewSampler(dst) => dst
                    [dst_start..dst_start + count]
                    .clone_from_slice(&src[src_start..src_start + count]),
                _ => panic!("descriptor copy has wrong resource type"),
            },
            DescriptorBindingResources::Sampler(src) => match self {
                DescriptorBindingResources::Sampler(dst) => dst[dst_start..dst_start + count]
                    .clone_from_slice(&src[src_start..src_start + count]),
                _ => panic!("descriptor copy has wrong resource type"),
            },
            DescriptorBindingResources::InlineUniformBlock => match self {
                DescriptorBindingResources::InlineUniformBlock => (),
                _ => panic!("descriptor copy has wrong resource type"),
            },
            DescriptorBindingResources::AccelerationStructure(src) => match self {
                DescriptorBindingResources::AccelerationStructure(dst) => dst
                    [dst_start..dst_start + count]
                    .clone_from_slice(&src[src_start..src_start + count]),
                _ => panic!("descriptor copy has wrong resource type"),
            },
        }
    }
}

#[derive(Clone)]
pub struct DescriptorSetWithOffsets {
    descriptor_set: Arc<DescriptorSet>,
    dynamic_offsets: SmallVec<[u32; 4]>,
}

impl DescriptorSetWithOffsets {
    pub fn new(
        descriptor_set: Arc<DescriptorSet>,
        dynamic_offsets: impl IntoIterator<Item = u32>,
    ) -> Self {
        Self {
            descriptor_set,
            dynamic_offsets: dynamic_offsets.into_iter().collect(),
        }
    }

    #[inline]
    pub fn as_ref(&self) -> (&Arc<DescriptorSet>, &[u32]) {
        (&self.descriptor_set, &self.dynamic_offsets)
    }

    #[inline]
    pub fn into_tuple(self) -> (Arc<DescriptorSet>, impl ExactSizeIterator<Item = u32>) {
        (self.descriptor_set, self.dynamic_offsets.into_iter())
    }
}

impl From<Arc<DescriptorSet>> for DescriptorSetWithOffsets {
    #[inline]
    fn from(descriptor_set: Arc<DescriptorSet>) -> Self {
        DescriptorSetWithOffsets::new(descriptor_set, std::iter::empty())
    }
}
