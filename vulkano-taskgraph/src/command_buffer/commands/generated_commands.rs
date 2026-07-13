use crate::{
    command_buffer::{RecordingCommandBuffer, Result},
    Id,
};
use ash::vk;
use smallvec::SmallVec;
use std::sync::Arc;
use vulkano::{
    buffer::Buffer,
    device::DeviceOwned,
    device_generated_commands::{GeneratedCommandsPipeline, IndirectCommandsLayout},
    pipeline::{compute::ComputePipeline, graphics::GraphicsPipeline},
    VulkanObject,
};

/// # Commands to generate and execute device-generated commands
impl RecordingCommandBuffer<'_> {
    /// Preprocesses input data for device-generated commands, panicking on a validation error.
    ///
    /// The preprocessing step executes in a separate logical pipeline from either graphics or
    /// compute, and must be explicitly synchronized against the command execution.
    ///
    /// This is a shortcut for `try_preprocess_generated_commands().unwrap()`.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_preprocess_generated_commands`] returns a [`ValidationError`].
    ///
    /// [shader safety requirements]: vulkano::shader#safety
    /// [`try_preprocess_generated_commands`]: Self::try_preprocess_generated_commands
    #[track_caller]
    pub unsafe fn preprocess_generated_commands(
        &mut self,
        generated_commands_info: &GeneratedCommandsInfo<'_>,
    ) -> &mut Self {
        unsafe { self.try_preprocess_generated_commands(generated_commands_info) }.unwrap()
    }

    /// Preprocesses input data for device-generated commands.
    ///
    /// The preprocessing step executes in a separate logical pipeline from either graphics or
    /// compute, and must be explicitly synchronized against the command execution.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    ///
    /// [shader safety requirements]: vulkano::shader#safety
    pub unsafe fn try_preprocess_generated_commands(
        &mut self,
        generated_commands_info: &GeneratedCommandsInfo<'_>,
    ) -> Result<&mut Self> {
        Ok(unsafe { self.preprocess_generated_commands_unchecked(generated_commands_info) })
    }

    pub unsafe fn preprocess_generated_commands_unchecked(
        &mut self,
        generated_commands_info: &GeneratedCommandsInfo<'_>,
    ) -> &mut Self {
        let &GeneratedCommandsInfo {
            ref pipeline,
            ref indirect_commands_layout,
            streams,
            sequence_count,
            preprocess_buffer,
            preprocess_offset,
            preprocess_size,
            sequence_count_buffer,
            sequence_count_buffer_offset,
            sequence_index_buffer,
            sequence_index_buffer_offset,
            _ne: _,
        } = generated_commands_info;

        let preprocess_buffer = unsafe { self.accesses.buffer_unchecked(preprocess_buffer) };

        let streams_vk: SmallVec<[_; 4]> = streams
            .iter()
            .map(|stream| {
                let buffer = unsafe { self.accesses.buffer_unchecked(stream.buffer) };

                vk::IndirectCommandsStreamNV::default()
                    .buffer(buffer.handle())
                    .offset(stream.offset)
            })
            .collect();

        let mut commands_info_vk = vk::GeneratedCommandsInfoNV::default()
            .pipeline_bind_point(pipeline.bind_point().into())
            .pipeline(pipeline.handle())
            .indirect_commands_layout(indirect_commands_layout.handle())
            .streams(&streams_vk)
            .sequences_count(sequence_count)
            .preprocess_buffer(preprocess_buffer.handle())
            .preprocess_offset(preprocess_offset)
            .preprocess_size(preprocess_size);

        if let Some(sequence_count_buffer) = sequence_count_buffer {
            let sequence_count_buffer =
                unsafe { self.accesses.buffer_unchecked(sequence_count_buffer) };

            commands_info_vk = commands_info_vk
                .sequences_count_buffer(sequence_count_buffer.handle())
                .sequences_count_offset(sequence_count_buffer_offset);
        }

        if let Some(sequence_index_buffer) = sequence_index_buffer {
            let sequence_index_buffer =
                unsafe { self.accesses.buffer_unchecked(sequence_index_buffer) };

            commands_info_vk = commands_info_vk
                .sequences_index_buffer(sequence_index_buffer.handle())
                .sequences_index_offset(sequence_index_buffer_offset);
        }

        let fns = self.device().fns();
        unsafe {
            (fns.nv_device_generated_commands
                .cmd_preprocess_generated_commands_nv)(self.handle(), &commands_info_vk)
        };

        self
    }

    /// Generates and executes device-generated commands, panicking on a validation error.
    ///
    /// If `is_preprocessed` is `true`, the preprocessing step is skipped and the previously
    /// preprocessed data is used. Otherwise, the preprocessing and execution are performed
    /// together.
    ///
    /// This is a shortcut for `try_execute_generated_commands().unwrap()`.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    ///
    /// # Panics
    ///
    /// - Panics if [`try_execute_generated_commands`] returns a [`ValidationError`].
    ///
    /// [shader safety requirements]: vulkano::shader#safety
    /// [`try_execute_generated_commands`]: Self::try_execute_generated_commands
    #[track_caller]
    pub unsafe fn execute_generated_commands(
        &mut self,
        is_preprocessed: bool,
        generated_commands_info: &GeneratedCommandsInfo<'_>,
    ) -> &mut Self {
        unsafe { self.try_execute_generated_commands(is_preprocessed, generated_commands_info) }
            .unwrap()
    }

    /// Generates and executes device-generated commands.
    ///
    /// If `is_preprocessed` is `true`, the preprocessing step is skipped and the previously
    /// preprocessed data is used. Otherwise, the preprocessing and execution are performed
    /// together.
    ///
    /// # Safety
    ///
    /// - The general [shader safety requirements] apply.
    ///
    /// [shader safety requirements]: vulkano::shader#safety
    pub unsafe fn try_execute_generated_commands(
        &mut self,
        is_preprocessed: bool,
        generated_commands_info: &GeneratedCommandsInfo<'_>,
    ) -> Result<&mut Self> {
        Ok(unsafe {
            self.execute_generated_commands_unchecked(is_preprocessed, generated_commands_info)
        })
    }

    pub unsafe fn execute_generated_commands_unchecked(
        &mut self,
        is_preprocessed: bool,
        generated_commands_info: &GeneratedCommandsInfo<'_>,
    ) -> &mut Self {
        let &GeneratedCommandsInfo {
            ref pipeline,
            ref indirect_commands_layout,
            streams,
            sequence_count,
            preprocess_buffer,
            preprocess_offset,
            preprocess_size,
            sequence_count_buffer,
            sequence_count_buffer_offset,
            sequence_index_buffer,
            sequence_index_buffer_offset,
            _ne: _,
        } = generated_commands_info;

        let preprocess_buffer = unsafe { self.accesses.buffer_unchecked(preprocess_buffer) };

        let streams_vk: SmallVec<[_; 4]> = streams
            .iter()
            .map(|stream| {
                let buffer = unsafe { self.accesses.buffer_unchecked(stream.buffer) };

                vk::IndirectCommandsStreamNV::default()
                    .buffer(buffer.handle())
                    .offset(stream.offset)
            })
            .collect();

        let mut commands_info_vk = vk::GeneratedCommandsInfoNV::default()
            .pipeline_bind_point(pipeline.bind_point().into())
            .pipeline(pipeline.handle())
            .indirect_commands_layout(indirect_commands_layout.handle())
            .streams(&streams_vk)
            .sequences_count(sequence_count)
            .preprocess_buffer(preprocess_buffer.handle())
            .preprocess_offset(preprocess_offset)
            .preprocess_size(preprocess_size);

        if let Some(sequence_count_buffer) = sequence_count_buffer {
            let sequence_count_buffer =
                unsafe { self.accesses.buffer_unchecked(sequence_count_buffer) };

            commands_info_vk = commands_info_vk
                .sequences_count_buffer(sequence_count_buffer.handle())
                .sequences_count_offset(sequence_count_buffer_offset);
        }

        if let Some(sequence_index_buffer) = sequence_index_buffer {
            let sequence_index_buffer =
                unsafe { self.accesses.buffer_unchecked(sequence_index_buffer) };

            commands_info_vk = commands_info_vk
                .sequences_index_buffer(sequence_index_buffer.handle())
                .sequences_index_offset(sequence_index_buffer_offset);
        }

        let fns = self.device().fns();
        unsafe {
            (fns.nv_device_generated_commands
                .cmd_execute_generated_commands_nv)(
                self.handle(),
                is_preprocessed.into(),
                &commands_info_vk,
            )
        };

        self
    }
}

/// Parameters for generating and executing device-generated commands.
#[derive(Clone, Debug)]
pub struct GeneratedCommandsInfo<'a> {
    /// The pipeline used in the generation and execution process.
    ///
    /// There is no default value.
    pub pipeline: GeneratedCommandsPipeline,

    /// The indirect commands layout that provides the command sequence to generate.
    ///
    /// There is no default value.
    pub indirect_commands_layout: Arc<IndirectCommandsLayout>,

    /// The input streams providing the data for the tokens used in `indirect_commands_layout`.
    ///
    /// The length must match `indirect_commands_layout`'s stream count.
    ///
    /// The default value is empty.
    pub streams: &'a [IndirectCommandsStream<'a>],

    /// The maximum number of sequences to reserve. If `sequence_count_buffer` is `None`, this is
    /// also the actual number of sequences generated.
    ///
    /// The default value is `0`.
    pub sequence_count: u32,

    /// The buffer that is used for preprocessing the input data for execution. The contents and
    /// layout of this buffer are opaque to applications and must not be modified outside functions
    /// related to device-generated commands or copied to another buffer for reuse.
    ///
    /// There is no default value.
    pub preprocess_buffer: Id<Buffer>,

    /// The byte offset into `preprocess_buffer` where the preprocessed data is stored.
    ///
    /// The default value is `0`.
    pub preprocess_offset: u64,

    /// The maximum byte size within `preprocess_buffer` after `preprocess_offset` that is
    /// available for preprocessing.
    ///
    /// The default value is `0`.
    pub preprocess_size: u64,

    /// A buffer in which the actual number of sequences is provided as a single `u32` value.
    ///
    /// If this is `Some`, then `sequence_count` serves as an upper bound.
    ///
    /// The default value is `None`.
    pub sequence_count_buffer: Option<Id<Buffer>>,

    /// The byte offset into `sequence_count_buffer` where the count value is stored.
    ///
    /// The default value is `0`.
    pub sequence_count_buffer_offset: u64,

    /// A buffer that encodes the used sequence indices as a `u32` array.
    ///
    /// The default value is `None`.
    pub sequence_index_buffer: Option<Id<Buffer>>,

    /// The byte offset into `sequence_index_buffer` where the index values start.
    ///
    /// The default value is `0`.
    pub sequence_index_buffer_offset: u64,

    pub _ne: crate::NonExhaustive<'a>,
}

impl GeneratedCommandsInfo<'_> {
    /// Returns a `GeneratedCommandsInfo` for a graphics pipeline.
    pub fn graphics_pipeline(
        pipeline: Arc<GraphicsPipeline>,
        indirect_commands_layout: Arc<IndirectCommandsLayout>,
        preprocess_buffer: Id<Buffer>,
        preprocess_offset: u64,
        preprocess_size: u64,
    ) -> Self {
        Self {
            pipeline: GeneratedCommandsPipeline::Graphics(pipeline),
            indirect_commands_layout,
            streams: &[],
            sequence_count: 0,
            preprocess_buffer,
            preprocess_offset,
            preprocess_size,
            sequence_count_buffer: None,
            sequence_count_buffer_offset: 0,
            sequence_index_buffer: None,
            sequence_index_buffer_offset: 0,
            _ne: crate::NE,
        }
    }

    /// Returns a `GeneratedCommandsInfo` for a compute pipeline.
    pub fn compute_pipeline(
        pipeline: Arc<ComputePipeline>,
        indirect_commands_layout: Arc<IndirectCommandsLayout>,
        preprocess_buffer: Id<Buffer>,
        preprocess_offset: u64,
        preprocess_size: u64,
    ) -> Self {
        Self {
            pipeline: GeneratedCommandsPipeline::Compute(pipeline),
            indirect_commands_layout,
            streams: &[],
            sequence_count: 0,
            preprocess_buffer,
            preprocess_offset,
            preprocess_size,
            sequence_count_buffer: None,
            sequence_count_buffer_offset: 0,
            sequence_index_buffer: None,
            sequence_index_buffer_offset: 0,
            _ne: crate::NE,
        }
    }

    /// Returns a `GeneratedCommandsInfo` for a dynamically selected pipeline.
    pub fn dynamic_pipeline(
        indirect_commands_layout: Arc<IndirectCommandsLayout>,
        preprocess_buffer: Id<Buffer>,
        preprocess_offset: u64,
        preprocess_size: u64,
    ) -> Self {
        Self {
            pipeline: GeneratedCommandsPipeline::Dynamic,
            indirect_commands_layout,
            streams: &[],
            sequence_count: 0,
            preprocess_buffer,
            preprocess_offset,
            preprocess_size,
            sequence_count_buffer: None,
            sequence_count_buffer_offset: 0,
            sequence_index_buffer: None,
            sequence_index_buffer_offset: 0,
            _ne: crate::NE,
        }
    }
}

/// An input stream providing data for the tokens used in an [`IndirectCommandsLayout`].
#[derive(Clone, Debug)]
pub struct IndirectCommandsStream<'a> {
    /// The buffer storing the functional arguments for each sequence. These arguments can be
    /// written by the device.
    ///
    /// The default value is [`Id::INVALID`], which must be overridden.
    pub buffer: Id<Buffer>,

    /// The byte offset into `buffer` where the arguments start.
    ///
    /// The default value is `0`.
    pub offset: u64,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for IndirectCommandsStream<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl IndirectCommandsStream<'_> {
    /// Returns a default `IndirectCommandsStream`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            buffer: Id::INVALID,
            offset: 0,
            _ne: crate::NE,
        }
    }
}
