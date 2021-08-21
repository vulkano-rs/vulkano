use super::bound::BoundResources;
use super::builder::RuntimeDescriptorSetBuilder;
use super::builder::RuntimeDescriptorSetBuilderOutput;
use super::RuntimeDescriptorSetError;
use crate::descriptor_set::layout::DescriptorSetDesc;
use crate::descriptor_set::layout::DescriptorSetLayout;
use crate::descriptor_set::pool::DescriptorPool;
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

pub struct RuntimeDescriptorSetPool {
    layout: Arc<DescriptorSetLayout>,
    pool: RtPool,
}

impl RuntimeDescriptorSetPool {
    pub fn new(layout: Arc<DescriptorSetLayout>, runtime_array_capacity: usize) -> Self {
        Self {
            pool: RtPool {
                inner: None,
                device: layout.device().clone(),
                rt_arr_cap: runtime_array_capacity,
                set_count: 4,
            },
            layout,
        }
    }

    pub fn next<'a>(
        &'a mut self,
    ) -> Result<RuntimeDescriptorSetPoolSetBuilder<'a>, RuntimeDescriptorSetError> {
        let runtime_array_capacity = self.pool.rt_arr_cap;
        let layout = self.layout.clone();

        Ok(RuntimeDescriptorSetPoolSetBuilder {
            pool: &mut self.pool,
            inner: RuntimeDescriptorSetBuilder::start(layout, runtime_array_capacity)?,
            poisoned: false,
        })
    }
}

struct RtPool {
    inner: Option<Arc<RtPoolInner>>,
    device: Arc<Device>,
    rt_arr_cap: usize,
    set_count: usize,
}

struct RtPoolInner {
    inner: UnsafeDescriptorPool,
    reserve: SegQueue<UnsafeDescriptorSet>,
}

unsafe impl DescriptorPool for RtPool {
    type Alloc = RtPoolAlloc;

    fn alloc(&mut self, layout: &DescriptorSetLayout) -> Result<Self::Alloc, OomError> {
        assert!(layout.num_bindings() > 0);

        let layout_rt_arr_cap = match layout.descriptor(layout.num_bindings() - 1) {
            Some(desc) => {
                if desc.variable_count {
                    desc.array_count as usize
                } else {
                    0
                }
            }
            None => 0,
        };

        loop {
            let mut not_enough_sets = false;

            if layout_rt_arr_cap <= self.rt_arr_cap {
                if let Some(ref mut p_inner) = self.inner {
                    if let Some(existing) = p_inner.reserve.pop() {
                        return Ok(RtPoolAlloc {
                            pool: p_inner.clone(),
                            inner: Some(existing),
                        });
                    } else {
                        not_enough_sets = true;
                    }
                }
            }

            while layout_rt_arr_cap > self.rt_arr_cap {
                self.rt_arr_cap *= 2;
            }

            if not_enough_sets {
                self.set_count *= 2;
            }

            let target_layout = DescriptorSetLayout::new(
                self.device.clone(),
                DescriptorSetDesc::new((0..layout.num_bindings()).into_iter().map(|binding_i| {
                    let mut desc = layout.descriptor(binding_i);

                    if let Some(desc) = &mut desc {
                        if desc.variable_count {
                            desc.array_count = self.rt_arr_cap as u32;
                        }
                    }

                    desc
                })),
            )?;

            let count = *layout.descriptors_count() * self.set_count as u32;
            let mut unsafe_pool = UnsafeDescriptorPool::new(
                self.device.clone(),
                &count,
                self.set_count as u32,
                false,
            )?;
            let reserve = unsafe {
                match unsafe_pool.alloc((0..self.set_count).map(|_| &target_layout)) {
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

            self.inner = Some(Arc::new(RtPoolInner {
                inner: unsafe_pool,
                reserve,
            }));
        }
    }
}

unsafe impl DeviceOwned for RtPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

struct RtPoolAlloc {
    pool: Arc<RtPoolInner>,
    inner: Option<UnsafeDescriptorSet>,
}

impl DescriptorPoolAlloc for RtPoolAlloc {
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        self.inner.as_ref().unwrap()
    }

    #[inline]
    fn inner_mut(&mut self) -> &mut UnsafeDescriptorSet {
        self.inner.as_mut().unwrap()
    }
}

impl Drop for RtPoolAlloc {
    fn drop(&mut self) {
        let inner = self.inner.take().unwrap();
        self.pool.reserve.push(inner);
    }
}

pub struct RuntimeDescriptorSetPoolSet {
    inner: RtPoolAlloc,
    bound_resources: BoundResources,
    layout: Arc<DescriptorSetLayout>,
}

unsafe impl DescriptorSet for RuntimeDescriptorSetPoolSet {
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
        self.bound_resources.num_buffers()
    }

    #[inline]
    fn buffer(&self, index: usize) -> Option<(&dyn BufferAccess, u32)> {
        self.bound_resources.buffer(index)
    }

    #[inline]
    fn num_images(&self) -> usize {
        self.bound_resources.num_images()
    }

    #[inline]
    fn image(&self, index: usize) -> Option<(&dyn ImageViewAbstract, u32)> {
        self.bound_resources.image(index)
    }
}

unsafe impl DeviceOwned for RuntimeDescriptorSetPoolSet {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.layout.device()
    }
}

impl PartialEq for RuntimeDescriptorSetPoolSet {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner().internal_object() == other.inner().internal_object()
            && self.device() == other.device()
    }
}

impl Eq for RuntimeDescriptorSetPoolSet {}

impl Hash for RuntimeDescriptorSetPoolSet {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().internal_object().hash(state);
        self.device().hash(state);
    }
}

pub struct RuntimeDescriptorSetPoolSetBuilder<'a> {
    pool: &'a mut RtPool,
    inner: RuntimeDescriptorSetBuilder,
    poisoned: bool,
}

impl<'a> RuntimeDescriptorSetPoolSetBuilder<'a> {
    pub fn enter_array(&mut self) -> Result<&mut Self, RuntimeDescriptorSetError> {
        if self.poisoned {
            Err(RuntimeDescriptorSetError::BuilderPoisoned)
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

    pub fn leave_array(&mut self) -> Result<&mut Self, RuntimeDescriptorSetError> {
        if self.poisoned {
            Err(RuntimeDescriptorSetError::BuilderPoisoned)
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

    pub fn add_empty(&mut self) -> Result<&mut Self, RuntimeDescriptorSetError> {
        if self.poisoned {
            Err(RuntimeDescriptorSetError::BuilderPoisoned)
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

    pub fn add_buffer(
        &mut self,
        buffer: Arc<dyn BufferAccess + Send + Sync + 'static>,
    ) -> Result<&mut Self, RuntimeDescriptorSetError> {
        if self.poisoned {
            Err(RuntimeDescriptorSetError::BuilderPoisoned)
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

    pub fn add_image(
        &mut self,
        image_view: Arc<dyn ImageViewAbstract + Send + Sync + 'static>,
    ) -> Result<&mut Self, RuntimeDescriptorSetError> {
        if self.poisoned {
            Err(RuntimeDescriptorSetError::BuilderPoisoned)
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

    pub fn add_sampled_image(
        &mut self,
        image_view: Arc<dyn ImageViewAbstract + Send + Sync + 'static>,
        sampler: Arc<Sampler>,
    ) -> Result<&mut Self, RuntimeDescriptorSetError> {
        if self.poisoned {
            Err(RuntimeDescriptorSetError::BuilderPoisoned)
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

    pub fn add_sampler(
        &mut self,
        sampler: Arc<Sampler>,
    ) -> Result<&mut Self, RuntimeDescriptorSetError> {
        if self.poisoned {
            Err(RuntimeDescriptorSetError::BuilderPoisoned)
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

    pub fn build(self) -> Result<RuntimeDescriptorSetPoolSet, RuntimeDescriptorSetError> {
        if self.poisoned {
            return Err(RuntimeDescriptorSetError::BuilderPoisoned);
        }

        let RuntimeDescriptorSetBuilderOutput {
            layout,
            writes,
            bound_resources,
        } = self.inner.output()?;

        let set = unsafe {
            let mut set = self.pool.alloc(&layout)?;
            set.inner_mut()
                .write(self.pool.device(), writes.into_iter());
            set
        };

        Ok(RuntimeDescriptorSetPoolSet {
            inner: set,
            bound_resources,
            layout,
        })
    }
}
