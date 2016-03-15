use std::mem;
use std::ptr;
use std::sync::Arc;

use descriptor_set::Layout;
use descriptor_set::LayoutPossibleSuperset;
use descriptor_set::PipelineLayout;
use pipeline::GenericPipeline;
use shader::ComputeShaderEntryPoint;

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
    pipeline_layout: Arc<PipelineLayout<Pl>>,
}

impl<Pl> ComputePipeline<Pl> {
    ///
    ///
    /// # Panic
    ///
    /// Panicks if the pipeline layout and/or shader don't belong to the device.
    pub fn new<Csl>(device: &Arc<Device>, pipeline_layout: &Arc<PipelineLayout<Pl>>,
                    shader: &ComputeShaderEntryPoint<Csl>) 
                    -> Result<Arc<ComputePipeline<Pl>>, OomError>
        where Pl: LayoutPossibleSuperset<Csl>, Csl: Layout
    {
        let vk = device.pointers();

        // TODO: should be an error instead
        assert!(LayoutPossibleSuperset::is_superset_of(pipeline_layout.layout(), shader.layout()));

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
                layout: pipeline_layout.internal_object(),
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

    /// Returns the `PipelineLayout` used in this compute pipeline.
    #[inline]
    pub fn layout(&self) -> &Arc<PipelineLayout<Pl>> {
        &self.pipeline_layout
    }
}

impl<Pl> GenericPipeline for ComputePipeline<Pl> {
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
