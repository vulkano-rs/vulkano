// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::mem;
use std::ptr;
use std::sync::Arc;

use descriptor_set::PipelineLayout;
//use descriptor_set::pipeline_layout::PipelineLayoutDesc;
//use descriptor_set::pipeline_layout::PipelineLayoutSuperset;
use pipeline::shader::ComputeShaderEntryPoint;

use device::Device;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

///
///
/// The template parameter contains the descriptor set to use with this pipeline.
pub struct ComputePipeline<Pl> {
    pipeline: vk::Pipeline,
    device: Arc<Device>,
    pipeline_layout: Arc<Pl>,
}

impl<Pl> ComputePipeline<Pl> {
    ///
    ///
    /// # Panic
    ///
    /// Panicks if the pipeline layout and/or shader don't belong to the device.
    pub fn new<Csl>(device: &Arc<Device>, pipeline_layout: &Arc<Pl>,
                    shader: &ComputeShaderEntryPoint<Csl>) 
                    -> Result<Arc<ComputePipeline<Pl>>, OomError>
        where Pl: PipelineLayout// + PipelineLayoutSuperset<Csl>, Csl: PipelineLayoutDesc
    {
        let vk = device.pointers();

        // TODO: should be an error instead
        //assert!(PipelineLayoutSuperset::is_superset_of(pipeline_layout, shader));

        let pipeline = unsafe {
            /*let spec_descriptors = specialization.descriptors();
            let specialization = vk::SpecializationInfo {
                mapEntryCount: spec_descriptors.len(),
                pMapEntries: spec_descriptors.as_ptr() as *const _,
                dataSize: mem::size_of_val(specialization),
                pData: specialization as *const S as *const _,
            };*/

            let stage = vk::PipelineShaderStageCreateInfo {
                sType: vk::STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                stage: vk::SHADER_STAGE_COMPUTE_BIT,
                module: shader.module().internal_object(),
                pName: shader.name().as_ptr(),
                pSpecializationInfo: //if mem::size_of_val(specialization) == 0 {
                    ptr::null()
                /*} else {
                    &specialization
                }*/,
            };

            let infos = vk::ComputePipelineCreateInfo {
                sType: vk::STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                stage: stage,
                layout: pipeline_layout.inner_pipeline_layout().internal_object(),
                basePipelineHandle: 0,
                basePipelineIndex: 0,
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateComputePipelines(device.internal_object(), 0,
                                                        1, &infos, ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(ComputePipeline {
            device: device.clone(),
            pipeline: pipeline,
            pipeline_layout: pipeline_layout.clone(),
        }))
    }

    /// Returns the `Device` this compute pipeline was created with.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the pipeline layout used in this compute pipeline.
    #[inline]
    pub fn layout(&self) -> &Arc<Pl> {
        &self.pipeline_layout
    }
}

unsafe impl<Pl> VulkanObject for ComputePipeline<Pl> {
    type Object = vk::Pipeline;

    #[inline]
    fn internal_object(&self) -> vk::Pipeline {
        self.pipeline
    }
}

impl<Pl> Drop for ComputePipeline<Pl> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyPipeline(self.device.internal_object(), self.pipeline, ptr::null());
        }
    }
}
