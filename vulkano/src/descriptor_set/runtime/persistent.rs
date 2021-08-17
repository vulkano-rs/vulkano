use super::bound::BoundResources;
use super::builder::RuntimeDescriptorSetBuilder;
use super::builder::RuntimeDescriptorSetBuilderOutput;
use super::RuntimeDescriptorSetError;
use crate::descriptor_set::pool::standard::StdDescriptorPoolAlloc;
use crate::descriptor_set::pool::DescriptorPool;
use crate::descriptor_set::pool::DescriptorPoolAlloc;
use crate::descriptor_set::BufferAccess;
use crate::descriptor_set::DescriptorSet;
use crate::descriptor_set::DescriptorSetLayout;
use crate::descriptor_set::UnsafeDescriptorSet;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::image::ImageViewAbstract;
use crate::sampler::Sampler;
use crate::VulkanObject;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

pub struct RuntimePersistentDescriptorSet<P = StdDescriptorPoolAlloc> {
    inner: P,
    bound_resources: BoundResources,
    layout: Arc<DescriptorSetLayout>,
}

impl RuntimePersistentDescriptorSet {
    pub fn start(
        layout: Arc<DescriptorSetLayout>,
        runtime_array_capacity: Option<usize>,
    ) -> Result<RuntimePersistentDescriptorSetBuilder, RuntimeDescriptorSetError> {
        Ok(RuntimePersistentDescriptorSetBuilder {
            inner: RuntimeDescriptorSetBuilder::start(layout, runtime_array_capacity.unwrap_or(0))?,
            poisoned: false,
        })
    }
}

unsafe impl<P> DescriptorSet for RuntimePersistentDescriptorSet<P>
where
    P: DescriptorPoolAlloc,
{
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

unsafe impl<P> DeviceOwned for RuntimePersistentDescriptorSet<P> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.layout.device()
    }
}

impl<P> PartialEq for RuntimePersistentDescriptorSet<P>
where
    P: DescriptorPoolAlloc,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner().internal_object() == other.inner().internal_object()
            && self.device() == other.device()
    }
}

impl<P> Eq for RuntimePersistentDescriptorSet<P> where P: DescriptorPoolAlloc {}

impl<P> Hash for RuntimePersistentDescriptorSet<P>
where
    P: DescriptorPoolAlloc,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().internal_object().hash(state);
        self.device().hash(state);
    }
}

pub struct RuntimePersistentDescriptorSetBuilder {
    inner: RuntimeDescriptorSetBuilder,
    poisoned: bool,
}

impl RuntimePersistentDescriptorSetBuilder {
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

    pub fn add_buffer<T>(
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

    pub fn add_image<T>(
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

    pub fn add_sampled_image<T>(
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

    pub fn build(
        self,
    ) -> Result<RuntimePersistentDescriptorSet<StdDescriptorPoolAlloc>, RuntimeDescriptorSetError>
    {
        let mut pool = Device::standard_descriptor_pool(self.inner.device());
        self.build_with_pool(&mut pool)
    }

    pub fn build_with_pool<P>(
        self,
        pool: &mut P,
    ) -> Result<RuntimePersistentDescriptorSet<P::Alloc>, RuntimeDescriptorSetError>
    where
        P: ?Sized + DescriptorPool,
    {
        if self.poisoned {
            return Err(RuntimeDescriptorSetError::BuilderPoisoned);
        }

        let RuntimeDescriptorSetBuilderOutput {
            layout,
            writes,
            bound_resources,
        } = self.inner.output()?;

        let set = unsafe {
            let mut set = pool.alloc(&layout)?;
            set.inner_mut().write(pool.device(), writes.into_iter());
            set
        };

        Ok(RuntimePersistentDescriptorSet {
            inner: set,
            bound_resources,
            layout,
        })
    }
}
