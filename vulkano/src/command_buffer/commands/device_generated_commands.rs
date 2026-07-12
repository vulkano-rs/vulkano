use crate::{
    command_buffer::RecordingCommandBuffer,
    device::DeviceOwned,
    pipeline::{ComputePipeline, Pipeline, PipelineCreateFlags},
    Requires, RequiresAllOf, RequiresOneOf, ValidationError, VulkanObject,
};
use std::sync::Arc;

impl RecordingCommandBuffer {
    #[inline]
    pub fn update_pipeline_indirect_buffer(
        &mut self,
        pipeline: &Arc<ComputePipeline>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_update_pipeline_indirect_buffer(pipeline)?;

        Ok(unsafe { self.update_pipeline_indirect_buffer_unchecked(pipeline) })
    }

    fn validate_update_pipeline_indirect_buffer(
        &self,
        pipeline: &Arc<ComputePipeline>,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .device()
            .enabled_features()
            .device_generated_compute_pipelines
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                    Requires::DeviceFeature("device_generated_compute_pipelines"),
                ])]),
                vuids: &[
                    "VUID-vkCmdUpdatePipelineIndirectBufferNV-deviceGeneratedComputePipelines-09021",
                ],
                ..Default::default()
            }));
        }

        // VUID-vkCmdUpdatePipelineIndirectBufferNV-commonparent
        assert_eq!(self.device(), pipeline.device());

        if !pipeline
            .flags()
            .intersects(PipelineCreateFlags::INDIRECT_BINDABLE)
        {
            return Err(Box::new(ValidationError {
                context: "pipeline.flags()".into(),
                problem: "does not contain `PipelineCreateFlags::INDIRECT_BINDABLE`".into(),
                vuids: &["VUID-vkCmdUpdatePipelineIndirectBufferNV-pipeline-09019"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn update_pipeline_indirect_buffer_unchecked(
        &mut self,
        pipeline: &Arc<ComputePipeline>,
    ) -> &mut Self {
        let fns = self.device().fns();
        unsafe {
            (fns.nv_device_generated_commands_compute
                .cmd_update_pipeline_indirect_buffer_nv)(
                self.handle(),
                pipeline.bind_point().into(),
                pipeline.handle(),
            )
        };

        self
    }
}