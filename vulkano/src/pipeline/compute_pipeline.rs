use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::sync::Arc;

use pipeline::GenericPipeline;
use shader::EntryPoint;

use device::Device;
use OomError;
use Success;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

///
///
/// The template parameter contains the descriptor set to use with this pipeline.
pub struct ComputePipeline<D, C> {
    device: Arc<Device>,
    pipeline: vk::Pipeline,
    marker: PhantomData<(D, C)>,
}

impl<D, C> ComputePipeline<D, C> {
    ///
    ///
    /// # Panic
    ///
    /// Panicks if the pipeline layout and/or shader don't belong to the device.
    pub fn new<D, S, P>(device: &Arc<Device>, pipeline_layout: &Arc<PipelineLayout<D, C>>,
                        shader: &ComputeShaderEntryPoint<D, S, P>, specialization: &S)
                        -> Result<ComputePipeline<D, C>, OomError>
        where S: SpecializationConstants
    {
        let vk = device.pointers();

        let pipeline = unsafe {
            let spec_descriptors = specialization.descriptors();
            let specialization = vk::SpecializationInfo {
                mapEntryCount: spec_descriptors.len(),
                pMapEntries: spec_descriptors.as_ptr() as *const _,
                dataSize: mem::size_of_val(specialization),
                pData: specialization as *const S as *const _,
            };

            let stage = vk::PipelineShaderStageCreateInfo {
                sType: vk::STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                shader: shader,
                pSpecializationInfo: if mem::size_of_val(specialization) == 0 {
                    ptr::null()
                } else {
                    &specialization
                },
            };

            let infos = VkComputePipelineCreateInfo {
                sType: vk::STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                stage: stage,
                layout: pipeline_layout.internal_object(),
                basePipelineHandle: vk::NULL_HANDLE,
                basePipelineIndex: 0,
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateComputePipelines(device.internal_object(), vk::NULL_HANDLE,
                                                        1, &infos, ptr::null(), &mut output)));
            output
        };
    }
}

impl GenericPipeline for ComputePipeline {
}

impl<D, C> Drop for ComputePipeline<D, C> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyPipeline(self.device.internal_object(), self.pipeline, ptr::null());
        }
    }
}
