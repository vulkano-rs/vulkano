use crate::command_buffer::auto::Resource;
use crate::command_buffer::{AutoCommandBufferBuilder, RecordingCommandBuffer, ResourceInCommand};
use crate::device::DeviceOwned;
use crate::pipeline::{ComputePipeline, Pipeline};
use crate::sync::{AccessFlags, PipelineStageAccess, PipelineStageAccessFlags};
use crate::{ValidationError, VulkanObject};
use std::sync::Arc;

impl<L> AutoCommandBufferBuilder<L> {
    pub unsafe fn update_pipeline_indirect_buffer(
        &mut self,
        pipeline: Arc<ComputePipeline>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_update_pipeline_indirect_buffer(&pipeline)?;

        Ok(unsafe { self.update_pipeline_indirect_buffer_unchecked(pipeline) })
    }

    fn validate_update_pipeline_indirect_buffer(
        &self,
        pipeline: &Arc<ComputePipeline>,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_update_pipeline_indirect_buffer(pipeline)?;
        // TODO
        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn update_pipeline_indirect_buffer_unchecked(
        &mut self,
        pipeline: Arc<ComputePipeline>,
    ) -> &mut Self {
        let buffer = pipeline.indirect_pipeline_buffer().unwrap();

        let mut used_resources = Vec::new();
        used_resources.push((
            ResourceInCommand::Destination.into(),
            Resource::Buffer {
                buffer: buffer.clone(),
                range: 0..buffer.size(),
                memory_access: PipelineStageAccessFlags::CommandPreprocess_CommandPreprocessRead
                    | PipelineStageAccessFlags::UpdatePipelineIndirectBuffer_UpdatePipelineIndirectBufferWrite,
            }
            ));

        self.add_command(
            "update_pipeline_indirect_buffer",
            used_resources,
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.update_pipeline_indirect_buffer_unchecked(&pipeline) };
            },
        );

        self
    }
}

impl RecordingCommandBuffer {
    #[inline]
    pub unsafe fn update_pipeline_indirect_buffer(
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
        // TODO
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
