// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::builder::DescriptorSetBuilder;
use super::builder::DescriptorSetBuilderOutput;
use super::resources::DescriptorSetResources;
use super::DescriptorSetError;
use crate::buffer::BufferView;
use crate::descriptor_set::layout::DescriptorSetDesc;
use crate::descriptor_set::layout::DescriptorSetLayout;
use crate::descriptor_set::pool::DescriptorPoolAlloc;
use crate::descriptor_set::pool::DescriptorPoolAllocError;
use crate::descriptor_set::pool::UnsafeDescriptorPool;
use crate::descriptor_set::BufferAccess;
use crate::descriptor_set::DescriptorSet;
use crate::descriptor_set::UnsafeDescriptorSet;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::image::ImageViewAbstract;
use crate::sampler::Sampler;
use crate::OomError;
use crate::VulkanObject;
use crossbeam_queue::SegQueue;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

/// `SingleLayoutDescSetPool` is a convenience wrapper provided by Vulkano not to be confused with
/// `VkDescriptorPool`. Its function is to provide access to pool(s) to allocate `DescriptorSet`'s
/// from and optimizes for a specific layout. For a more general purpose pool see `descriptor_set::pool::StdDescriptorPool`.
pub struct SingleLayoutDescSetPool {
    // The `SingleLayoutPool` struct contains an actual Vulkan pool. Every time it is full or additional
    // runtime array capacity is needed, we create a new pool and replace the current one with the new one.
    inner: Option<Arc<SingleLayoutPool>>,
    // The Vulkan device.
    device: Arc<Device>,
    // The capacity available to runtime arrays when we create a new Vulkan pool.
    runtime_array_capacity: usize,
    // The amount of sets available to use when we create a new Vulkan pool.
    set_count: usize,
    // The descriptor layout that we target and allocate with
    target_layout: Arc<DescriptorSetLayout>,
}

impl SingleLayoutDescSetPool {
    /// Initializes a new pool. The pool is configured to allocate sets that corresponds to the
    /// parameters passed to this function.
    ///
    /// `runtime_array_capacity` is the capacity that is reserved in a set for runtime arrays.
    /// This capacity will automatically be adjusted if a set is created that exceeed this value.
    /// If a set doesn't contain a runtime array this does nothing.
    pub fn new(
        layout: Arc<DescriptorSetLayout>,
        runtime_array_capacity: usize,
    ) -> Result<Self, DescriptorSetError> {
        let device = layout.device().clone();
        let target_layout = Arc::new(DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetDesc::new((0..layout.num_bindings()).into_iter().map(|binding_i| {
                let mut desc = layout.descriptor(binding_i);

                if let Some(desc) = &mut desc {
                    if desc.variable_count {
                        desc.descriptor_count = runtime_array_capacity as u32;
                    }
                }

                desc
            })),
        )?);

        Ok(Self {
            inner: None,
            device,
            runtime_array_capacity,
            set_count: 4,
            target_layout,
        })
    }

    /// Starts the process of building a new descriptor set.
    ///
    /// The set will corresponds to the set layout that was passed to `new`.
    pub fn next<'a>(&'a mut self) -> Result<SingleLayoutDescSetBuilder<'a>, DescriptorSetError> {
        let runtime_array_capacity = self.runtime_array_capacity;
        let layout = self.target_layout.clone();

        Ok(SingleLayoutDescSetBuilder {
            pool: self,
            inner: DescriptorSetBuilder::start(layout, runtime_array_capacity)?,
            poisoned: false,
        })
    }

    fn next_alloc(
        &mut self,
        runtime_array_length: usize,
    ) -> Result<SingleLayoutPoolAlloc, OomError> {
        loop {
            let mut not_enough_sets = false;

            if runtime_array_length <= self.runtime_array_capacity {
                if let Some(ref mut p_inner) = self.inner {
                    if let Some(existing) = p_inner.reserve.pop() {
                        return Ok(SingleLayoutPoolAlloc {
                            pool: p_inner.clone(),
                            inner: Some(existing),
                        });
                    } else {
                        not_enough_sets = true;
                    }
                }
            }

            let mut not_enough_capacity = false;

            while runtime_array_length > self.runtime_array_capacity {
                self.runtime_array_capacity *= 2;
                not_enough_capacity = true;
            }

            if not_enough_sets {
                self.set_count *= 2;
            }

            if not_enough_capacity {
                self.target_layout = Arc::new(DescriptorSetLayout::new(
                    self.device.clone(),
                    DescriptorSetDesc::new((0..self.target_layout.num_bindings()).into_iter().map(
                        |binding_i| {
                            let mut desc = self.target_layout.descriptor(binding_i);

                            if let Some(desc) = &mut desc {
                                if desc.variable_count {
                                    desc.descriptor_count = self.runtime_array_capacity as u32;
                                }
                            }

                            desc
                        },
                    )),
                )?);
            }

            let count = *self.target_layout.descriptors_count() * self.set_count as u32;
            let mut unsafe_pool = UnsafeDescriptorPool::new(
                self.device.clone(),
                &count,
                self.set_count as u32,
                false,
            )?;

            let reserve = unsafe {
                match unsafe_pool.alloc((0..self.set_count).map(|_| &*self.target_layout)) {
                    Ok(alloc_iter) => {
                        let reserve = SegQueue::new();

                        for alloc in alloc_iter {
                            reserve.push(alloc);
                        }

                        reserve
                    }
                    Err(DescriptorPoolAllocError::OutOfHostMemory) => {
                        return Err(OomError::OutOfHostMemory);
                    }
                    Err(DescriptorPoolAllocError::OutOfDeviceMemory) => {
                        return Err(OomError::OutOfDeviceMemory);
                    }
                    Err(DescriptorPoolAllocError::FragmentedPool) => {
                        // This can't happen as we don't free individual sets.
                        unreachable!()
                    }
                    Err(DescriptorPoolAllocError::OutOfPoolMemory) => unreachable!(),
                }
            };

            self.inner = Some(Arc::new(SingleLayoutPool {
                inner: unsafe_pool,
                reserve,
            }));
        }
    }
}

struct SingleLayoutPool {
    // The actual Vulkan descriptor pool. This field isn't actually used anywhere, but we need to
    // keep the pool alive in order to keep the descriptor sets valid.
    inner: UnsafeDescriptorPool,

    // List of descriptor sets. When `alloc` is called, a descriptor will be extracted from this
    // list. When a `SingleLayoutPoolAlloc` is dropped, its descriptor set is put back in this list.
    reserve: SegQueue<UnsafeDescriptorSet>,
}

struct SingleLayoutPoolAlloc {
    // The `SingleLayoutPool` were we allocated from. We need to keep a copy of it in each allocation
    // so that we can put back the allocation in the list in our `Drop` impl.
    pool: Arc<SingleLayoutPool>,

    // The actual descriptor set, wrapped inside an `Option` so that we can extract it in our
    // `Drop` impl.
    inner: Option<UnsafeDescriptorSet>,
}

impl DescriptorPoolAlloc for SingleLayoutPoolAlloc {
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        self.inner.as_ref().unwrap()
    }

    #[inline]
    fn inner_mut(&mut self) -> &mut UnsafeDescriptorSet {
        self.inner.as_mut().unwrap()
    }
}

impl Drop for SingleLayoutPoolAlloc {
    fn drop(&mut self) {
        let inner = self.inner.take().unwrap();
        self.pool.reserve.push(inner);
    }
}

/// A descriptor set created from a `SingleLayoutDescSetPool`.
pub struct SingleLayoutDescSet {
    inner: SingleLayoutPoolAlloc,
    resources: DescriptorSetResources,
    layout: Arc<DescriptorSetLayout>,
}

unsafe impl DescriptorSet for SingleLayoutDescSet {
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        self.inner.inner()
    }

    #[inline]
    fn layout(&self) -> &Arc<DescriptorSetLayout> {
        &self.layout
    }

    #[inline]
    fn num_buffers(&self) -> usize {
        self.resources.num_buffers()
    }

    #[inline]
    fn buffer(&self, index: usize) -> Option<(&dyn BufferAccess, u32)> {
        self.resources.buffer(index)
    }

    #[inline]
    fn num_images(&self) -> usize {
        self.resources.num_images()
    }

    #[inline]
    fn image(&self, index: usize) -> Option<(&dyn ImageViewAbstract, u32)> {
        self.resources.image(index)
    }
}

unsafe impl DeviceOwned for SingleLayoutDescSet {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.layout.device()
    }
}

impl PartialEq for SingleLayoutDescSet {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner().internal_object() == other.inner().internal_object()
            && self.device() == other.device()
    }
}

impl Eq for SingleLayoutDescSet {}

impl Hash for SingleLayoutDescSet {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().internal_object().hash(state);
        self.device().hash(state);
    }
}

/// Prototype of a `SingleLayoutDescSet`.
pub struct SingleLayoutDescSetBuilder<'a> {
    pool: &'a mut SingleLayoutDescSetPool,
    inner: DescriptorSetBuilder,
    poisoned: bool,
}

impl<'a> SingleLayoutDescSetBuilder<'a> {
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
        if self.poisoned {
            Err(DescriptorSetError::BuilderPoisoned)
        } else {
            match self.inner.enter_array() {
                Ok(_) => Ok(self),
                Err(e) => {
                    self.poisoned = true;
                    Err(e)
                }
            }
        }
    }

    /// Leaves the array. Call this once you added all the elements of the array.
    ///
    /// Returns an error if the array is missing elements, or if the builder is not in an array.
    #[inline]
    pub fn leave_array(&mut self) -> Result<&mut Self, DescriptorSetError> {
        if self.poisoned {
            Err(DescriptorSetError::BuilderPoisoned)
        } else {
            match self.inner.leave_array() {
                Ok(_) => Ok(self),
                Err(e) => {
                    self.poisoned = true;
                    Err(e)
                }
            }
        }
    }

    /// Skips the current descriptor if it is empty.
    #[inline]
    pub fn add_empty(&mut self) -> Result<&mut Self, DescriptorSetError> {
        if self.poisoned {
            Err(DescriptorSetError::BuilderPoisoned)
        } else {
            match self.inner.add_empty() {
                Ok(_) => Ok(self),
                Err(e) => {
                    self.poisoned = true;
                    Err(e)
                }
            }
        }
    }

    /// Binds a buffer as the next descriptor.
    ///
    /// An error is returned if the buffer isn't compatible with the descriptor.
    #[inline]
    pub fn add_buffer(
        &mut self,
        buffer: Arc<dyn BufferAccess + 'static>,
    ) -> Result<&mut Self, DescriptorSetError> {
        if self.poisoned {
            Err(DescriptorSetError::BuilderPoisoned)
        } else {
            match self.inner.add_buffer(buffer) {
                Ok(_) => Ok(self),
                Err(e) => {
                    self.poisoned = true;
                    Err(e)
                }
            }
        }
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
        if self.poisoned {
            Err(DescriptorSetError::BuilderPoisoned)
        } else {
            match self.inner.add_buffer_view(view) {
                Ok(_) => Ok(self),
                Err(e) => {
                    self.poisoned = true;
                    Err(e)
                }
            }
        }
    }

    /// Binds an image view as the next descriptor.
    ///
    /// An error is returned if the image view isn't compatible with the descriptor.
    #[inline]
    pub fn add_image(
        &mut self,
        image_view: Arc<dyn ImageViewAbstract + Send + Sync + 'static>,
    ) -> Result<&mut Self, DescriptorSetError> {
        if self.poisoned {
            Err(DescriptorSetError::BuilderPoisoned)
        } else {
            match self.inner.add_image(image_view) {
                Ok(_) => Ok(self),
                Err(e) => {
                    self.poisoned = true;
                    Err(e)
                }
            }
        }
    }

    /// Binds an image view with a sampler as the next descriptor.
    ///
    /// An error is returned if the image view isn't compatible with the descriptor.
    #[inline]
    pub fn add_sampled_image(
        &mut self,
        image_view: Arc<dyn ImageViewAbstract + Send + Sync + 'static>,
        sampler: Arc<Sampler>,
    ) -> Result<&mut Self, DescriptorSetError> {
        if self.poisoned {
            Err(DescriptorSetError::BuilderPoisoned)
        } else {
            match self.inner.add_sampled_image(image_view, sampler) {
                Ok(_) => Ok(self),
                Err(e) => {
                    self.poisoned = true;
                    Err(e)
                }
            }
        }
    }

    /// Binds a sampler as the next descriptor.
    ///
    /// An error is returned if the sampler isn't compatible with the descriptor.
    #[inline]
    pub fn add_sampler(&mut self, sampler: Arc<Sampler>) -> Result<&mut Self, DescriptorSetError> {
        if self.poisoned {
            Err(DescriptorSetError::BuilderPoisoned)
        } else {
            match self.inner.add_sampler(sampler) {
                Ok(_) => Ok(self),
                Err(e) => {
                    self.poisoned = true;
                    Err(e)
                }
            }
        }
    }

    /// Builds a `SingleLayoutDescSet` from the builder.
    pub fn build(self) -> Result<SingleLayoutDescSet, DescriptorSetError> {
        if self.poisoned {
            return Err(DescriptorSetError::BuilderPoisoned);
        }

        let DescriptorSetBuilderOutput {
            writes,
            resources,
            runtime_array_length,
            ..
        } = self.inner.output(false)?;

        let mut alloc = self.pool.next_alloc(runtime_array_length)?;
        unsafe {
            alloc
                .inner_mut()
                .write(&self.pool.device, writes.into_iter());
        }

        Ok(SingleLayoutDescSet {
            inner: alloc,
            resources,
            layout: self.pool.target_layout.clone(),
        })
    }
}
