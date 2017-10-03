// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crossbeam::sync::SegQueue;
use std::sync::Arc;

use OomError;
use buffer::BufferAccess;
use buffer::BufferViewRef;
use descriptor::descriptor::DescriptorDesc;
use descriptor::descriptor_set::DescriptorPool;
use descriptor::descriptor_set::DescriptorPoolAlloc;
use descriptor::descriptor_set::DescriptorPoolAllocError;
use descriptor::descriptor_set::DescriptorSet;
use descriptor::descriptor_set::DescriptorSetDesc;
use descriptor::descriptor_set::UnsafeDescriptorPool;
use descriptor::descriptor_set::UnsafeDescriptorSet;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use descriptor::descriptor_set::persistent::*;
use descriptor::pipeline_layout::PipelineLayoutAbstract;
use device::Device;
use device::DeviceOwned;
use image::ImageViewAccess;
use sampler::Sampler;

/// Pool of descriptor sets of a specific capacity and that are automatically reclaimed.
///
/// You are encouraged to use this type when you need a different descriptor set at each frame, or
/// regularly during the execution.
///
/// # Example
///
/// At initialization, create a `FixedSizeDescriptorSetsPool`. The first parameter of the `new`
/// function can be a graphics pipeline, a compute pipeline, or anything that represents a pipeline
/// layout.
///
/// ```rust
/// use vulkano::descriptor::descriptor_set::FixedSizeDescriptorSetsPool;
/// # use vulkano::pipeline::GraphicsPipelineAbstract;
/// # use std::sync::Arc;
/// # let graphics_pipeline: Arc<GraphicsPipelineAbstract> = return;
/// // use vulkano::pipeline::GraphicsPipelineAbstract;
/// // let graphics_pipeline: Arc<GraphicsPipelineAbstract> = ...;
///
/// let pool = FixedSizeDescriptorSetsPool::new(graphics_pipeline.clone(), 0);
/// ```
///
/// You would then typically store the pool in a struct for later. Its type is
/// `FixedSizeDescriptorSetsPool<T>` where `T` is the type of what was passed to `new()`. In the
/// example above, it would be `FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract>>`.
///
/// Then whenever you need a descriptor set, call `pool.next()` to start the process of building
/// it.
///
/// ```rust
/// # use std::sync::Arc;
/// # use vulkano::descriptor::descriptor_set::FixedSizeDescriptorSetsPool;
/// # use vulkano::pipeline::GraphicsPipelineAbstract;
/// # let mut pool: FixedSizeDescriptorSetsPool<Arc<GraphicsPipelineAbstract>> = return;
/// let descriptor_set = pool.next()
///     //.add_buffer(...)
///     //.add_sampled_image(...)
///     .build().unwrap();
/// ```
///
/// Note that `next()` requires exclusive (`mut`) access to the pool. You can use a `Mutex` around
/// the pool if you can't provide this.
///
#[derive(Clone)]
pub struct FixedSizeDescriptorSetsPool<L> {
    pipeline_layout: L,
    set_id: usize,
    set_layout: Arc<UnsafeDescriptorSetLayout>,
    // We hold a local implementation of the `DescriptorPool` trait for our own purpose. Since we
    // don't want to expose this trait impl in our API, we use a separate struct.
    pool: LocalPool,
}

impl<L> FixedSizeDescriptorSetsPool<L> {
    /// Initializes a new pool. The pool is configured to allocate sets that corresponds to the
    /// parameters passed to this function.
    pub fn new(layout: L, set_id: usize) -> FixedSizeDescriptorSetsPool<L>
        where L: PipelineLayoutAbstract
    {
        assert!(layout.num_sets() > set_id);

        let device = layout.device().clone();

        let set_layout = layout
            .descriptor_set_layout(set_id)
            .expect("Unable to get the descriptor set layout")
            .clone();

        FixedSizeDescriptorSetsPool {
            pipeline_layout: layout,
            set_id,
            set_layout,
            pool: LocalPool {
                device: device,
                next_capacity: 3,
                current_pool: None,
            },
        }
    }

    /// Starts the process of building a new descriptor set.
    ///
    /// The set will corresponds to the set layout that was passed to `new`.
    #[inline]
    pub fn next(&mut self) -> FixedSizeDescriptorSetBuilder<L, ()>
        where L: PipelineLayoutAbstract + Clone
    {
        let inner = PersistentDescriptorSet::start(self.pipeline_layout.clone(), self.set_id);

        FixedSizeDescriptorSetBuilder {
            pool: self,
            inner: inner,
        }
    }
}

/// A descriptor set created from a `FixedSizeDescriptorSetsPool`.
pub struct FixedSizeDescriptorSet<L, R> {
    inner: PersistentDescriptorSet<L, R, LocalPoolAlloc>,
}

unsafe impl<L, R> DescriptorSet for FixedSizeDescriptorSet<L, R>
    where L: PipelineLayoutAbstract,
          R: PersistentDescriptorSetResources
{
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        self.inner.inner()
    }

    #[inline]
    fn num_buffers(&self) -> usize {
        self.inner.num_buffers()
    }

    #[inline]
    fn buffer(&self, index: usize) -> Option<(&BufferAccess, u32)> {
        self.inner.buffer(index)
    }

    #[inline]
    fn num_images(&self) -> usize {
        self.inner.num_images()
    }

    #[inline]
    fn image(&self, index: usize) -> Option<(&ImageViewAccess, u32)> {
        self.inner.image(index)
    }
}

unsafe impl<L, R> DescriptorSetDesc for FixedSizeDescriptorSet<L, R>
    where L: PipelineLayoutAbstract
{
    #[inline]
    fn num_bindings(&self) -> usize {
        self.inner.num_bindings()
    }

    #[inline]
    fn descriptor(&self, binding: usize) -> Option<DescriptorDesc> {
        self.inner.descriptor(binding)
    }
}

unsafe impl<L, R> DeviceOwned for FixedSizeDescriptorSet<L, R>
    where L: DeviceOwned
{
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

// The fields of this struct can be considered as fields of the `FixedSizeDescriptorSet`. They are
// in a separate struct because we don't want to expose the fact that we implement the
// `DescriptorPool` trait.
#[derive(Clone)]
struct LocalPool {
    // The `LocalPoolInner` struct contains an actual Vulkan pool. Every time it is full, we create
    // a new pool and replace the current one with the new one.
    current_pool: Option<Arc<LocalPoolInner>>,
    // Capacity to use when we create a new Vulkan pool.
    next_capacity: u32,
    // The Vulkan device.
    device: Arc<Device>,
}

struct LocalPoolInner {
    // The actual Vulkan descriptor pool. This field isn't actually used anywhere, but we need to
    // keep the pool alive in order to keep the descriptor sets valid.
    actual_pool: UnsafeDescriptorPool,

    // List of descriptor sets. When `alloc` is called, a descriptor will be extracted from this
    // list. When a `LocalPoolAlloc` is dropped, its descriptor set is put back in this list.
    reserve: SegQueue<UnsafeDescriptorSet>,
}

struct LocalPoolAlloc {
    // The `LocalPoolInner` we were allocated from. We need to keep a copy of it in each allocation
    // so that we can put back the allocation in the list in our `Drop` impl.
    pool: Arc<LocalPoolInner>,

    // The actual descriptor set, wrapped inside an `Option` so that we can extract it in our
    // `Drop` impl.
    actual_alloc: Option<UnsafeDescriptorSet>,
}

unsafe impl DescriptorPool for LocalPool {
    type Alloc = LocalPoolAlloc;

    fn alloc(&mut self, layout: &UnsafeDescriptorSetLayout) -> Result<Self::Alloc, OomError> {
        loop {
            // Try to extract a descriptor from the current pool if any exist.
            // This is the most common case.
            if let Some(ref mut current_pool) = self.current_pool {
                if let Some(already_existing_set) = current_pool.reserve.try_pop() {
                    return Ok(LocalPoolAlloc {
                                  actual_alloc: Some(already_existing_set),
                                  pool: current_pool.clone(),
                              });
                }
            }

            // If we failed to grab an existing set, that means the current pool is full. Create a
            // new one of larger capacity.
            let count = *layout.descriptors_count() * self.next_capacity;
            let mut new_pool =
                UnsafeDescriptorPool::new(self.device.clone(), &count, self.next_capacity, false)?;
            let alloc = unsafe {
                match new_pool.alloc((0 .. self.next_capacity).map(|_| layout)) {
                    Ok(iter) => {
                        let stack = SegQueue::new();
                        for elem in iter {
                            stack.push(elem);
                        }
                        stack
                    },
                    Err(DescriptorPoolAllocError::OutOfHostMemory) => {
                        return Err(OomError::OutOfHostMemory);
                    },
                    Err(DescriptorPoolAllocError::OutOfDeviceMemory) => {
                        return Err(OomError::OutOfDeviceMemory);
                    },
                    Err(DescriptorPoolAllocError::FragmentedPool) => {
                        // This can't happen as we don't free individual sets.
                        unreachable!()
                    },
                    Err(DescriptorPoolAllocError::OutOfPoolMemory) => {
                        unreachable!()
                    },
                }
            };

            self.next_capacity = self.next_capacity.saturating_mul(2);
            self.current_pool = Some(Arc::new(LocalPoolInner {
                                                  actual_pool: new_pool,
                                                  reserve: alloc,
                                              }));
        }
    }
}

unsafe impl DeviceOwned for LocalPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl DescriptorPoolAlloc for LocalPoolAlloc {
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        self.actual_alloc.as_ref().unwrap()
    }

    #[inline]
    fn inner_mut(&mut self) -> &mut UnsafeDescriptorSet {
        self.actual_alloc.as_mut().unwrap()
    }
}

impl Drop for LocalPoolAlloc {
    fn drop(&mut self) {
        let inner = self.actual_alloc.take().unwrap();
        self.pool.reserve.push(inner);
    }
}

/// Prototype of a `FixedSizeDescriptorSet`.
///
/// The template parameter `L` is the pipeline layout to use, and the template parameter `R` is
/// an unspecified type that represents the list of resources.
///
/// See the docs of `FixedSizeDescriptorSetsPool` for an example.
pub struct FixedSizeDescriptorSetBuilder<'a, L: 'a, R> {
    pool: &'a mut FixedSizeDescriptorSetsPool<L>,
    inner: PersistentDescriptorSetBuilder<L, R>,
}

impl<'a, L, R> FixedSizeDescriptorSetBuilder<'a, L, R>
    where L: PipelineLayoutAbstract
{
    /// Builds a `FixedSizeDescriptorSet` from the builder.
    #[inline]
    pub fn build(self) -> Result<FixedSizeDescriptorSet<L, R>, PersistentDescriptorSetBuildError> {
        let inner = self.inner.build_with_pool(&mut self.pool.pool)?;
        Ok(FixedSizeDescriptorSet { inner: inner })
    }

    /// Call this function if the next element of the set is an array in order to set the value of
    /// each element.
    ///
    /// Returns an error if the descriptor is empty.
    ///
    /// This function can be called even if the descriptor isn't an array, and it is valid to enter
    /// the "array", add one element, then leave.
    #[inline]
    pub fn enter_array(
        self)
        -> Result<FixedSizeDescriptorSetBuilderArray<'a, L, R>, PersistentDescriptorSetError> {
        Ok(FixedSizeDescriptorSetBuilderArray {
               pool: self.pool,
               inner: self.inner.enter_array()?,
           })
    }

    /// Skips the current descriptor if it is empty.
    #[inline]
    pub fn add_empty(
        self)
        -> Result<FixedSizeDescriptorSetBuilder<'a, L, R>, PersistentDescriptorSetError> {
        Ok(FixedSizeDescriptorSetBuilder {
               pool: self.pool,
               inner: self.inner.add_empty()?,
           })
    }

    /// Binds a buffer as the next descriptor.
    ///
    /// An error is returned if the buffer isn't compatible with the descriptor.
    ///
    /// # Panic
    ///
    /// Panics if the buffer doesn't have the same device as the pipeline layout.
    ///
    #[inline]
    pub fn add_buffer<T>(self, buffer: T)
                         -> Result<FixedSizeDescriptorSetBuilder<'a,
                                                                 L,
                                                                 (R,
                                                                  PersistentDescriptorSetBuf<T>)>,
                                   PersistentDescriptorSetError>
        where T: BufferAccess
    {
        Ok(FixedSizeDescriptorSetBuilder {
               pool: self.pool,
               inner: self.inner.add_buffer(buffer)?,
           })
    }

    /// Binds a buffer view as the next descriptor.
    ///
    /// An error is returned if the buffer isn't compatible with the descriptor.
    ///
    /// # Panic
    ///
    /// Panics if the buffer view doesn't have the same device as the pipeline layout.
    ///
    pub fn add_buffer_view<T>(self, view: T)
        -> Result<FixedSizeDescriptorSetBuilder<'a, L, (R, PersistentDescriptorSetBufView<T>)>, PersistentDescriptorSetError>
        where T: BufferViewRef
    {
        Ok(FixedSizeDescriptorSetBuilder {
               pool: self.pool,
               inner: self.inner.add_buffer_view(view)?,
           })
    }

    /// Binds an image view as the next descriptor.
    ///
    /// An error is returned if the image view isn't compatible with the descriptor.
    ///
    /// # Panic
    ///
    /// Panics if the image view doesn't have the same device as the pipeline layout.
    ///
    #[inline]
    pub fn add_image<T>(self, image_view: T)
                        -> Result<FixedSizeDescriptorSetBuilder<'a,
                                                                L,
                                                                (R, PersistentDescriptorSetImg<T>)>,
                                  PersistentDescriptorSetError>
        where T: ImageViewAccess
    {
        Ok(FixedSizeDescriptorSetBuilder {
               pool: self.pool,
               inner: self.inner.add_image(image_view)?,
           })
    }

    /// Binds an image view with a sampler as the next descriptor.
    ///
    /// An error is returned if the image view isn't compatible with the descriptor.
    ///
    /// # Panic
    ///
    /// Panics if the image view or the sampler doesn't have the same device as the pipeline layout.
    ///
    #[inline]
    pub fn add_sampled_image<T>(self, image_view: T, sampler: Arc<Sampler>)
        -> Result<FixedSizeDescriptorSetBuilder<'a, L, ((R, PersistentDescriptorSetImg<T>), PersistentDescriptorSetSampler)>, PersistentDescriptorSetError>
        where T: ImageViewAccess
    {
        Ok(FixedSizeDescriptorSetBuilder {
               pool: self.pool,
               inner: self.inner.add_sampled_image(image_view, sampler)?,
           })
    }

    /// Binds a sampler as the next descriptor.
    ///
    /// An error is returned if the sampler isn't compatible with the descriptor.
    ///
    /// # Panic
    ///
    /// Panics if the sampler doesn't have the same device as the pipeline layout.
    ///
    #[inline]
    pub fn add_sampler(self, sampler: Arc<Sampler>)
                       -> Result<FixedSizeDescriptorSetBuilder<'a,
                                                               L,
                                                               (R, PersistentDescriptorSetSampler)>,
                                 PersistentDescriptorSetError> {
        Ok(FixedSizeDescriptorSetBuilder {
               pool: self.pool,
               inner: self.inner.add_sampler(sampler)?,
           })
    }
}

/// Same as `FixedSizeDescriptorSetBuilder`, but we're in an array.
pub struct FixedSizeDescriptorSetBuilderArray<'a, L: 'a, R> {
    pool: &'a mut FixedSizeDescriptorSetsPool<L>,
    inner: PersistentDescriptorSetBuilderArray<L, R>,
}

impl<'a, L, R> FixedSizeDescriptorSetBuilderArray<'a, L, R>
    where L: PipelineLayoutAbstract
{
    /// Leaves the array. Call this once you added all the elements of the array.
    pub fn leave_array(
        self)
        -> Result<FixedSizeDescriptorSetBuilder<'a, L, R>, PersistentDescriptorSetError> {
        Ok(FixedSizeDescriptorSetBuilder {
               pool: self.pool,
               inner: self.inner.leave_array()?,
           })
    }

    /// Binds a buffer as the next element in the array.
    ///
    /// An error is returned if the buffer isn't compatible with the descriptor.
    ///
    /// # Panic
    ///
    /// Panics if the buffer doesn't have the same device as the pipeline layout.
    ///
    pub fn add_buffer<T>(self, buffer: T)
        -> Result<FixedSizeDescriptorSetBuilderArray<'a, L, (R, PersistentDescriptorSetBuf<T>)>, PersistentDescriptorSetError>
        where T: BufferAccess
    {
        Ok(FixedSizeDescriptorSetBuilderArray {
               pool: self.pool,
               inner: self.inner.add_buffer(buffer)?,
           })
    }

    /// Binds a buffer view as the next element in the array.
    ///
    /// An error is returned if the buffer isn't compatible with the descriptor.
    ///
    /// # Panic
    ///
    /// Panics if the buffer view doesn't have the same device as the pipeline layout.
    ///
    pub fn add_buffer_view<T>(self, view: T)
        -> Result<FixedSizeDescriptorSetBuilderArray<'a, L, (R, PersistentDescriptorSetBufView<T>)>, PersistentDescriptorSetError>
        where T: BufferViewRef
    {
        Ok(FixedSizeDescriptorSetBuilderArray {
               pool: self.pool,
               inner: self.inner.add_buffer_view(view)?,
           })
    }

    /// Binds an image view as the next element in the array.
    ///
    /// An error is returned if the image view isn't compatible with the descriptor.
    ///
    /// # Panic
    ///
    /// Panics if the image view doesn't have the same device as the pipeline layout.
    ///
    pub fn add_image<T>(self, image_view: T)
        -> Result<FixedSizeDescriptorSetBuilderArray<'a, L, (R, PersistentDescriptorSetImg<T>)>, PersistentDescriptorSetError>
        where T: ImageViewAccess
    {
        Ok(FixedSizeDescriptorSetBuilderArray {
               pool: self.pool,
               inner: self.inner.add_image(image_view)?,
           })
    }

    /// Binds an image view with a sampler as the next element in the array.
    ///
    /// An error is returned if the image view isn't compatible with the descriptor.
    ///
    /// # Panic
    ///
    /// Panics if the image or the sampler doesn't have the same device as the pipeline layout.
    ///
    pub fn add_sampled_image<T>(self, image_view: T, sampler: Arc<Sampler>)
        -> Result<FixedSizeDescriptorSetBuilderArray<'a, L, ((R, PersistentDescriptorSetImg<T>), PersistentDescriptorSetSampler)>, PersistentDescriptorSetError>
        where T: ImageViewAccess
    {
        Ok(FixedSizeDescriptorSetBuilderArray {
               pool: self.pool,
               inner: self.inner.add_sampled_image(image_view, sampler)?,
           })
    }

    /// Binds a sampler as the next element in the array.
    ///
    /// An error is returned if the sampler isn't compatible with the descriptor.
    ///
    /// # Panic
    ///
    /// Panics if the sampler doesn't have the same device as the pipeline layout.
    ///
    pub fn add_sampler(self, sampler: Arc<Sampler>)
        -> Result<FixedSizeDescriptorSetBuilderArray<'a, L, (R, PersistentDescriptorSetSampler)>, PersistentDescriptorSetError>
    {
        Ok(FixedSizeDescriptorSetBuilderArray {
               pool: self.pool,
               inner: self.inner.add_sampler(sampler)?,
           })
    }
}
