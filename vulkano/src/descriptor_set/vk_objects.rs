use std::mem;
use std::ptr;
use std::sync::Arc;

use buffer::BufferResource;
use descriptor_set::layout_def::PipelineLayoutDesc;
use descriptor_set::layout_def::DescriptorSetDesc;
use descriptor_set::layout_def::DescriptorWrite;
use descriptor_set::layout_def::DescriptorBind;
use descriptor_set::pool::DescriptorPool;
use device::Device;

use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// An actual descriptor set with the resources that are binded to it.
pub struct DescriptorSet<S> {
    set: vk::DescriptorSet,
    pool: Arc<DescriptorPool>,
    layout: Arc<DescriptorSetLayout<S>>,
}

impl<S> DescriptorSet<S> where S: DescriptorSetDesc {
    ///
    /// # Panic
    ///
    /// - Panicks if the pool and the layout were not created from the same `Device`.
    ///
    pub fn new(pool: &Arc<DescriptorPool>, layout: &Arc<DescriptorSetLayout<S>>, init: S::Init)
               -> Result<Arc<DescriptorSet<S>>, OomError>
    {
        unsafe {
            let mut set = try!(DescriptorSet::uninitialized(pool, layout));
            Arc::get_mut(&mut set).unwrap().unchecked_write(layout.description().decode_init(init));
            Ok(set)
        }
    }

    ///
    /// # Panic
    ///
    /// - Panicks if the pool and the layout were not created from the same `Device`.
    ///
    // FIXME: this has to check whether there's still enough room in the pool
    pub unsafe fn uninitialized(pool: &Arc<DescriptorPool>, layout: &Arc<DescriptorSetLayout<S>>)
                                -> Result<Arc<DescriptorSet<S>>, OomError>
    {
        assert_eq!(&**pool.device() as *const Device, &*layout.device as *const Device);

        let vk = pool.device().pointers();

        let set = {
            let infos = vk::DescriptorSetAllocateInfo {
                sType: vk::STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                pNext: ptr::null(),
                descriptorPool: pool.internal_object(),
                descriptorSetCount: 1,
                pSetLayouts: &layout.layout,
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.AllocateDescriptorSets(pool.device().internal_object(), &infos,
                                                        &mut output)));
            output
        };

        Ok(Arc::new(DescriptorSet {
            set: set,
            pool: pool.clone(),
            layout: layout.clone(),
        }))
    }

    /// Modifies a descriptor set.
    ///
    /// The parameter depends on your implementation of `DescriptorSetDesc`.
    ///
    /// This function trusts the implementation of `DescriptorSetDesc` when it comes to making sure
    /// that the correct resource type is written to the correct descriptor.
    pub fn write(&mut self, write: S::Write) {
        let write = self.layout.description().decode_write(write);
        unsafe { self.unchecked_write(write); }
    }

    /// Modifies a descriptor set without checking that the writes are correct.
    pub unsafe fn unchecked_write(&mut self, write: Vec<DescriptorWrite>) {
        let vk = self.pool.device().pointers();

        // FIXME: store resources in the descriptor set so that they aren't destroyed

        // TODO: the architecture of this function is going to be tricky

        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let buffer_descriptors = write.iter().enumerate().map(|(num, write)| {
            match write.content {
                DescriptorBind::UniformBuffer(ref buffer) => {
                    Some(vk::DescriptorBufferInfo {
                        buffer: buffer.internal_object(),
                        offset: 0,      // FIXME: allow buffer slices
                        range: buffer.size() as u64,       // FIXME: allow buffer slices
                    })
                },
            }
        }).collect::<Vec<_>>();

        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let vk_writes = write.iter().enumerate().map(|(num, write)| {
            vk::WriteDescriptorSet {
                sType: vk::STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                pNext: ptr::null(),
                dstSet: self.set,
                dstBinding: write.binding,
                dstArrayElement: write.array_element,
                descriptorCount: 1,
                descriptorType: vk::DESCRIPTOR_TYPE_UNIFORM_BUFFER,     // FIXME:
                pImageInfo: ptr::null(),        // FIXME:
                pBufferInfo: if let Some(ref b) = buffer_descriptors[num] { b } else { ptr::null() },
                pTexelBufferView: ptr::null(),      // FIXME:
            }
        }).collect::<Vec<_>>();

        if !vk_writes.is_empty() {
            vk.UpdateDescriptorSets(self.pool.device().internal_object(),
                                    vk_writes.len() as u32, vk_writes.as_ptr(), 0, ptr::null());
        }
    }
}

unsafe impl<S> VulkanObject for DescriptorSet<S> {
    type Object = vk::DescriptorSet;

    #[inline]
    fn internal_object(&self) -> vk::DescriptorSet {
        self.set
    }
}

impl<S> Drop for DescriptorSet<S> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.pool.device().pointers();
            vk.FreeDescriptorSets(self.pool.device().internal_object(),
                                  self.pool.internal_object(), 1, &self.set);
        }
    }
}


/// Implemented on all `DescriptorSet` objects. Hides the template parameters.
pub unsafe trait AbstractDescriptorSet: ::VulkanObjectU64 {}
unsafe impl<S> AbstractDescriptorSet for DescriptorSet<S> {}

/// Describes the layout of all descriptors within a descriptor set.
pub struct DescriptorSetLayout<S> {
    layout: vk::DescriptorSetLayout,
    device: Arc<Device>,
    description: S,
}

impl<S> DescriptorSetLayout<S> where S: DescriptorSetDesc {
    pub fn new(device: &Arc<Device>, description: S)
               -> Result<Arc<DescriptorSetLayout<S>>, OomError>
    {
        let vk = device.pointers();

        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let bindings = description.descriptors().into_iter().map(|desc| {
            vk::DescriptorSetLayoutBinding {
                binding: desc.binding,
                descriptorType: desc.ty.vk_enum(),
                descriptorCount: desc.array_count,
                stageFlags: desc.stages.into(),
                pImmutableSamplers: ptr::null(),        // FIXME: not yet implemented
            }
        }).collect::<Vec<_>>();

        let layout = unsafe {
            let infos = vk::DescriptorSetLayoutCreateInfo {
                sType: vk::STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                bindingCount: bindings.len() as u32,
                pBindings: bindings.as_ptr(),
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateDescriptorSetLayout(device.internal_object(), &infos,
                                                           ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(DescriptorSetLayout {
            layout: layout,
            device: device.clone(),
            description: description,
        }))
    }

    #[inline]
    pub fn description(&self) -> &S {
        &self.description
    }
}

unsafe impl<S> VulkanObject for DescriptorSetLayout<S> {
    type Object = vk::DescriptorSetLayout;

    #[inline]
    fn internal_object(&self) -> vk::DescriptorSetLayout {
        self.layout
    }
}

impl<S> Drop for DescriptorSetLayout<S> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyDescriptorSetLayout(self.device.internal_object(), self.layout, ptr::null());
        }
    }
}

/// Implemented on all `DescriptorSetLayout` objects. Hides the template parameters.
pub unsafe trait AbstractDescriptorSetLayout: ::VulkanObjectU64 {}
unsafe impl<S> AbstractDescriptorSetLayout for DescriptorSetLayout<S> {}

/// A collection of `DescriptorSetLayout` structs.
// TODO: push constants.
pub struct PipelineLayout<P> {
    device: Arc<Device>,
    layout: vk::PipelineLayout,
    description: P,
    layouts: Vec<Arc<AbstractDescriptorSetLayout>>,     // TODO: is it necessary to keep the layouts alive? check the specs
}

impl<P> PipelineLayout<P> where P: PipelineLayoutDesc {
    /// Creates a new `PipelineLayout`.
    pub fn new(device: &Arc<Device>, description: P, layouts: P::DescriptorSetLayouts)
               -> Result<Arc<PipelineLayout<P>>, OomError>
    {
        let vk = device.pointers();

        let layouts = description.decode_descriptor_set_layouts(layouts);
        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let layouts_ids = layouts.iter().map(|l| {
            // FIXME: check that they belong to the same device
            ::VulkanObjectU64::internal_object(&**l)
        }).collect::<Vec<_>>();

        let layout = unsafe {
            let infos = vk::PipelineLayoutCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                setLayoutCount: layouts_ids.len() as u32,
                pSetLayouts: layouts_ids.as_ptr(),
                pushConstantRangeCount: 0,      // TODO: unimplemented
                pPushConstantRanges: ptr::null(),    // TODO: unimplemented
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreatePipelineLayout(device.internal_object(), &infos,
                                                      ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(PipelineLayout {
            device: device.clone(),
            layout: layout,
            description: description,
            layouts: layouts,
        }))
    }

    #[inline]
    pub fn description(&self) -> &P {
        &self.description
    }
}

unsafe impl<P> VulkanObject for PipelineLayout<P> {
    type Object = vk::PipelineLayout;

    #[inline]
    fn internal_object(&self) -> vk::PipelineLayout {
        self.layout
    }
}

impl<P> Drop for PipelineLayout<P> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyDescriptorSetLayout(self.device.internal_object(), self.layout, ptr::null());
        }
    }
}
