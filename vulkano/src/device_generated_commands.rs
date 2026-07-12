//! Generating command buffer content on the device.
//!
//! Device-generated commands allow the device to create and execute command buffer content without
//! a round-trip to the host. A layout is defined that describes the sequence of commands to
//! generate, one or more buffers are filled with the data to be interpreted by that layout, and
//! the device generates and executes the commands.

use crate::{
    NE, NonNullDeviceAddress, Validated, ValidationError, VulkanError, VulkanObject, buffer::{Buffer, BufferUsage, IndexType}, device::{Device, DeviceOwned}, macros::{vulkan_bitflags, vulkan_enum}, memory::MemoryRequirements, pipeline::{
        ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineCreateFlags, PipelineLayout, compute::ComputePipelineCreateInfo,
    }, shader::ShaderStages,
};
use ash::vk::{self, DeviceAddress};
use std::{collections::BTreeMap, mem::MaybeUninit, ptr, sync::Arc};
use vulkano::{Requires, RequiresAllOf, RequiresOneOf};

/// An opaque handle to an indirect commands layout object, which describes the sequence of
/// commands that should be generated.
#[derive(Debug)]
pub struct IndirectCommandsLayout {
    handle: vk::IndirectCommandsLayoutNV,
    device: Arc<Device>,

    flags: IndirectCommandsLayoutUsageFlags,
    pipeline_bind_point: PipelineBindPoint,
    stream_count: u32,
    token_types: Vec<IndirectCommandsTokenType>,
}

impl IndirectCommandsLayout {
    #[inline]
    pub fn new(
        device: Arc<Device>,
        create_info: IndirectCommandsLayoutCreateInfo<'_>,
    ) -> Result<Arc<IndirectCommandsLayout>, Validated<VulkanError>> {
        Self::validate_new(&device, &create_info)?;
        Ok(unsafe { Self::new_unchecked(device, create_info) }?)
    }

    fn validate_new(
        device: &Device,
        create_info: &IndirectCommandsLayoutCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !device.enabled_features().device_generated_commands {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "nv_device_generated_commands",
                )])]),
                vuids: &["VUID-vkCreateIndirectCommandsLayoutNV-deviceGeneratedCommands-02929"],
                ..Default::default()
            }));
        }

        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        create_info: IndirectCommandsLayoutCreateInfo<'_>,
    ) -> Result<Arc<IndirectCommandsLayout>, VulkanError> {
        let flags = create_info.flags;
        let pipeline_bind_point = create_info.pipeline_bind_point;
        let stream_count = create_info.stream_strides.len() as u32;
        let token_types = create_info
            .tokens
            .iter()
            .map(|token| token.token_type)
            .collect();

        let create_info_fields2_vk = create_info.to_vk_fields2();
        let create_info_fields1_vk = create_info.to_vk_fields1(&create_info_fields2_vk);
        let create_info_vk = create_info.to_vk(&create_info_fields1_vk);

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.nv_device_generated_commands
                    .create_indirect_commands_layout_nv)(
                    device.handle(),
                    &create_info_vk,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
            unsafe { output.assume_init() }
        };

        Ok(Arc::new(IndirectCommandsLayout {
            handle,
            device,
            flags,
            pipeline_bind_point,
            stream_count,
            token_types,
        }))
    }

    pub fn memory_requirements(
        &self,
        pipeline: &GeneratedCommandsPipeline,
        max_sequence_count: u32,
    ) -> MemoryRequirements {
        let memory_requirements_info_vk = vk::GeneratedCommandsMemoryRequirementsInfoNV::default()
            .pipeline_bind_point(pipeline.bind_point().into())
            .pipeline(pipeline.handle())
            .indirect_commands_layout(self.handle)
            .max_sequences_count(max_sequence_count);

        let memory_requirements_vk2 = {
            let fns = self.device.fns();
            let mut output = vk::MemoryRequirements2::default();
            unsafe {
                (fns.nv_device_generated_commands
                    .get_generated_commands_memory_requirements_nv)(
                    self.device.handle(),
                    &memory_requirements_info_vk,
                    &mut output,
                )
            };
            output
        };

        let memory_requirements_extension_vk2 =
            MemoryRequirements::to_mut_vk2_extensions(self.device());

        MemoryRequirements::from_vk2(&memory_requirements_vk2, &memory_requirements_extension_vk2)
    }

    /// Returns the flags the layout was created with.
    #[inline]
    pub fn flags(&self) -> IndirectCommandsLayoutUsageFlags {
        self.flags
    }

    /// Returns the pipeline bind point the layout was created with.
    #[inline]
    pub fn pipeline_bind_point(&self) -> PipelineBindPoint {
        self.pipeline_bind_point
    }

    /// Returns the number of streams the layout was created with.
    #[inline]
    pub fn stream_count(&self) -> u32 {
        self.stream_count
    }

    /// Returns the token types of the layout, in order.
    #[inline]
    pub fn token_types(&self) -> &[IndirectCommandsTokenType] {
        &self.token_types
    }
}

unsafe impl VulkanObject for IndirectCommandsLayout {
    type Handle = vk::IndirectCommandsLayoutNV;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for IndirectCommandsLayout {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl Drop for IndirectCommandsLayout {
    #[inline]
    fn drop(&mut self) {
        let fns = self.device.fns();
        unsafe {
            (fns.nv_device_generated_commands
                .destroy_indirect_commands_layout_nv)(
                self.device.handle(),
                self.handle,
                ptr::null(),
            )
        }
    }
}

/// Parameters to create a new [`IndirectCommandsLayout`].
#[derive(Clone, Debug)]
pub struct IndirectCommandsLayoutCreateInfo<'a> {
    /// The usage flags for the indirect commands layout.
    ///
    /// The default value is empty.
    pub flags: IndirectCommandsLayoutUsageFlags,

    /// The pipeline bind point used for the pipeline.
    ///
    /// Must be either [`PipelineBindPoint::Compute`] or [`PipelineBindPoint::Graphics`].
    ///
    /// The default value is [`PipelineBindPoint::Graphics`].
    pub pipeline_bind_point: PipelineBindPoint,

    /// The tokens that define the command sequence to generate.
    ///
    /// The default value is empty, which must be overridden.
    pub tokens: &'a [IndirectCommandsLayoutToken],

    /// The strides in bytes for each input stream.
    ///
    /// The default value is empty, which must be overridden.
    pub stream_strides: &'a [u32],

    pub _ne: crate::NonExhaustive<'static>,
}

impl<'a> IndirectCommandsLayoutCreateInfo<'a> {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        self.flags
            .validate_device(device)
            .map_err(|err| err.add_context("flags"))?;

        self.pipeline_bind_point
            .validate_device(device)
            .map_err(|err| err.add_context("pipeline_bind_point"))?;

        if self.pipeline_bind_point != PipelineBindPoint::Compute
            && self.pipeline_bind_point != PipelineBindPoint::Graphics
        {
            return Err(Box::new(ValidationError {
                context: "pipeline_bind_point".into(),
                problem: "is not `PipelineBindPoint::Compute` or \
                    `PipelineBindPoint::Graphics`"
                    .into(),
                vuids: &["VUID-VkIndirectCommandsLayoutCreateInfoNV-pipelineBindPoint-02930"],
                ..Default::default()
            }));
        }

        {
            let token_count = self.tokens.len() as u32;

            if token_count == 0
                || token_count
                    > device
                        .physical_device()
                        .properties()
                        .max_indirect_commands_token_count
                        .unwrap_or(0)
            {
                return Err(Box::new(ValidationError {
                    context: "tokens".into(),
                    problem: "the length is zero, or is greater than the \
                        `max_indirect_commands_token_count` limit"
                        .into(),
                    vuids: &["VUID-VkIndirectCommandsLayoutCreateInfoNV-tokenCount-02931"],
                    ..Default::default()
                }));
            }
        }

        self.tokens
            .iter()
            .map(|token| token.validate(device, self.stream_strides.len() as u32))
            .fold(Ok(()), Result::or)?;

        if self
            .tokens
            .iter()
            .skip(1)
            .any(|token| token.token_type == IndirectCommandsTokenType::ShaderGroup)
        {
            return Err(Box::new(ValidationError {
                context: "tokens".into(),
                problem: "a token of type \
                    `IndirectCommandsTokenType::ShaderGroup` is present at a position \
                    other than the first"
                    .into(),
                vuids: &["VUID-VkIndirectCommandsLayoutCreateInfoNV-pTokens-02932"],
                ..Default::default()
            }));
        }

        if self
            .tokens
            .iter()
            .skip(1)
            .any(|token| token.token_type == IndirectCommandsTokenType::Pipeline)
        {
            return Err(Box::new(ValidationError {
                context: "tokens".into(),
                problem: "a token of type \
                    `IndirectCommandsTokenType::Pipeline` is present at a position \
                    other than the first"
                    .into(),
                vuids: &["VUID-VkIndirectCommandsLayoutCreateInfoNV-pTokens-09585"],
                ..Default::default()
            }));
        }

        if self
            .tokens
            .iter()
            .filter(|token| token.token_type == IndirectCommandsTokenType::StateFlags)
            .count()
            > 1
        {
            return Err(Box::new(ValidationError {
                context: "tokens".into(),
                problem: "more than one token of type \
                    `IndirectCommandsTokenType::StateFlags` is present"
                    .into(),
                vuids: &["VUID-VkIndirectCommandsLayoutCreateInfoNV-pTokens-02933"],
                ..Default::default()
            }));
        }

        if self
            .tokens
            .iter()
            .take(self.tokens.len() - 1)
            .filter(|token| {
                token.token_type == IndirectCommandsTokenType::Draw
                    || token.token_type == IndirectCommandsTokenType::DrawIndexed
                    || token.token_type == IndirectCommandsTokenType::DrawTasks
                    || token.token_type == IndirectCommandsTokenType::DrawMeshTasks
                    || token.token_type == IndirectCommandsTokenType::Dispatch
            })
            .count()
            > 0
        {
            return Err(Box::new(ValidationError {
                context: "tokens".into(),
                problem: "a token with an action type is present at a position \
                    other than the last"
                    .into(),
                vuids: &["VUID-VkIndirectCommandsLayoutCreateInfoNV-pTokens-02934"],
                ..Default::default()
            }));
        }

        self.tokens
            .last()
            .map(|token| match token.token_type {
                IndirectCommandsTokenType::DrawIndexed
                | IndirectCommandsTokenType::Draw
                | IndirectCommandsTokenType::DrawTasks
                | IndirectCommandsTokenType::DrawMeshTasks => {
                    if self.pipeline_bind_point == PipelineBindPoint::Graphics {
                        Ok(())
                    } else {
                        Err(Box::new(ValidationError {
                            problem: "the last token has a draw type, but \
                                `pipeline_bind_point` is not \
                                `PipelineBindPoint::Graphics`"
                                .into(),
                            vuids: &["VUID-VkIndirectCommandsLayoutCreateInfoNV-pTokens-02935"],
                            ..Default::default()
                        }))
                    }
                }
                IndirectCommandsTokenType::Dispatch => {
                    if self.pipeline_bind_point == PipelineBindPoint::Compute {
                        Ok(())
                    } else {
                        Err(Box::new(ValidationError {
                            problem: "the last token has the `Dispatch` type, but \
                                `pipeline_bind_point` is not \
                                `PipelineBindPoint::Compute`"
                                .into(),
                            vuids: &["VUID-VkIndirectCommandsLayoutCreateInfoNV-pTokens-02935"],
                            ..Default::default()
                        }))
                    }
                }
                _ => Err(Box::new(ValidationError {
                    context: "tokens".into(),
                    problem: "the last token does not have an action type".into(),
                    vuids: &["VUID-VkIndirectCommandsLayoutCreateInfoNV-pTokens-02935"],
                    ..Default::default()
                })),
            })
            .unwrap()?;

        {
            let stream_count = self.stream_strides.len() as u32;

            if stream_count == 0
                || stream_count
                    > device
                        .physical_device()
                        .properties()
                        .max_indirect_commands_stream_count
                        .unwrap_or(0)
            {
                return Err(Box::new(ValidationError {
                    context: "stream_strides".into(),
                    problem: "the length is zero, or is greater than the \
                        `max_indirect_commands_stream_count` limit"
                        .into(),
                    vuids: &["VUID-VkIndirectCommandsLayoutCreateInfoNV-streamCount-02936"],
                    ..Default::default()
                }));
            }

            if self.stream_strides.iter().copied().any(|stream_stride| {
                stream_stride == 0
                    || stream_stride
                        > device
                            .physical_device()
                            .properties()
                            .max_indirect_commands_stream_stride
                            .unwrap_or(0)
            }) {
                return Err(Box::new(ValidationError {
                    context: "stream_strides".into(),
                    problem: "contains a value that is zero, or is greater than the \
                        `max_indirect_commands_stream_stride` limit"
                        .into(),
                    vuids: &["VUID-VkIndirectCommandsLayoutCreateInfoNV-pStreamStrides-02937"],
                    ..Default::default()
                }));
            }

        }

        if self.pipeline_bind_point == PipelineBindPoint::Compute {
            if !device.enabled_features().device_generated_compute {
                return Err(Box::new(ValidationError {
                    context: "pipeline_bind_point".into(),
                    problem: "is `PipelineBindPoint::Compute`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "device_generated_compute",
                    )])]),
                    vuids: &["VUID-VkIndirectCommandsLayoutCreateInfoNV-pipelineBindPoint-09088"],
                    ..Default::default()
                }));
            }

            if self
                .tokens
                .iter()
                .filter(|token| match token.token_type {
                    IndirectCommandsTokenType::PushConstant
                    | IndirectCommandsTokenType::Pipeline
                    | IndirectCommandsTokenType::Dispatch => false,
                    _ => true,
                })
                .count()
                > 0
            {
                return Err(Box::new(ValidationError {
                    problem: "`pipeline_bind_point` is `PipelineBindPoint::Compute`, but \
                        `tokens` contains a token with a type other than \
                        `IndirectCommandsTokenType::PushConstant`, \
                        `IndirectCommandsTokenType::Pipeline`, or \
                        `IndirectCommandsTokenType::Dispatch`"
                        .into(),
                    vuids: &["VUID-VkIndirectCommandsLayoutCreateInfoNV-pipelineBindPoint-09089"],
                    ..Default::default()
                }));
            }

            // Pipeline token can only be the first one and we have at least one token
            if self.tokens[0].token_type == IndirectCommandsTokenType::Pipeline
                && !device.enabled_features().device_generated_compute_pipelines
            {
                return Err(Box::new(ValidationError {
                    context: "tokens[0].token_type".into(),
                    problem: "is `IndirectCommandsTokenType::Pipeline`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "device_generated_compute_pipelines",
                    )])]),
                    vuids: &["VUID-VkIndirectCommandsLayoutCreateInfoNV-pipelineBindPoint-09090"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &'a self,
        fields1_vk: &'a IndirectCommandsLayoutCreateInfoFields1Vk<'_>,
    ) -> vk::IndirectCommandsLayoutCreateInfoNV<'a> {
        vk::IndirectCommandsLayoutCreateInfoNV::default()
            .flags(self.flags.into())
            .pipeline_bind_point(self.pipeline_bind_point.into())
            .tokens(fields1_vk.tokens.as_slice())
            .stream_strides(self.stream_strides)
    }

    pub(crate) fn to_vk_fields1(
        &self,
        fields2_vk: &'a IndirectCommandsLayoutCreateInfoFields2Vk,
    ) -> IndirectCommandsLayoutCreateInfoFields1Vk<'a> {
        let tokens = self
            .tokens
            .iter()
            .zip(fields2_vk.token_index_types.iter())
            .map(|(token, token_fields1_vk)| token.to_vk(token_fields1_vk))
            .collect::<Vec<_>>();
        IndirectCommandsLayoutCreateInfoFields1Vk {
            tokens,
        }
    }

    pub(crate) fn to_vk_fields2(&self) -> IndirectCommandsLayoutCreateInfoFields2Vk {
        let token_index_types = self
            .tokens
            .iter()
            .map(|token| token.to_vk_field1())
            .collect::<Vec<_>>();
        IndirectCommandsLayoutCreateInfoFields2Vk {
            token_index_types,
        }
    }
}

impl<'a> Default for IndirectCommandsLayoutCreateInfo<'a> {
    fn default() -> IndirectCommandsLayoutCreateInfo<'a> {
        IndirectCommandsLayoutCreateInfo {
            flags: Default::default(),
            pipeline_bind_point: PipelineBindPoint::Graphics,
            tokens: &[],
            stream_strides: &[],
            _ne: NE,
        }
    }
}

pub(crate) struct IndirectCommandsLayoutCreateInfoFields1Vk<'a> {
    pub(crate) tokens: Vec<vk::IndirectCommandsLayoutTokenNV<'a>>,
}

pub(crate) struct IndirectCommandsLayoutCreateInfoFields2Vk {
    pub(crate) token_index_types: Vec<IndirectCommandsLayoutTokenFieldVk1>,
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Usage hints of an [`IndirectCommandsLayout`].
    IndirectCommandsLayoutUsageFlags = IndirectCommandsLayoutUsageFlagsNV(u32);

    /// Specifies that the layout is always used with the manual preprocessing step and executed
    /// with `is_preprocessed` set to `true`.
    EXPLICIT_PREPROCESS = EXPLICIT_PREPROCESS,

    /// Specifies that the input data for the sequences is not implicitly indexed from
    /// 0..sequences_used, but an application-provided buffer encoding the index is provided.
    INDEXED_SEQUENCES = INDEXED_SEQUENCES,

    /// Specifies that the processing of sequences can happen at an implementation-dependent
    /// order, which is not guaranteed to be coherent using the same input data. This flag is
    /// ignored when the pipeline bind point is [`PipelineBindPoint::Compute`], as the dispatch
    /// sequence is always unordered.
    ///
    /// [`PipelineBindPoint::Compute`]: crate::pipeline::PipelineBindPoint::Compute
    UNORDERED_SEQUENCES = UNORDERED_SEQUENCES,
}

/// A token in an [`IndirectCommandsLayout`] that specifies details of the command arguments
/// that need to be known at layout creation time.
#[derive(Clone, Debug)]
pub struct IndirectCommandsLayoutToken {
    /// The token command type.
    ///
    /// The default value is [`IndirectCommandsTokenType::ShaderGroup`].
    pub token_type: IndirectCommandsTokenType,

    /// The index of the input stream containing the token argument data.
    ///
    /// The default value is `0`.
    pub stream: u32,

    /// A relative starting offset within the input stream memory for the token argument data.
    ///
    /// The default value is `0`.
    pub offset: u32,

    /// Used for the vertex buffer binding command.
    ///
    /// The default value is `0`.
    pub vertex_binding_unit: u32,

    /// Sets if the vertex buffer stride is provided by the binding command rather than the
    /// current bound graphics pipeline state.
    ///
    /// The default value is `false`.
    pub vertex_dynamic_stride: bool,

    /// The push constant data for the push constant command. Must be `Some` if and only if
    /// `token_type` is [`IndirectCommandsTokenType::PushConstant`].
    ///
    /// The default value is `None`.
    pub pushconstant_data: Option<IndirectCommandsLayoutTokenPushConstant>,

    /// The active states for the state flag command.
    ///
    /// The default value is empty.
    pub indirect_state_flags: IndirectStateFlags,

    /// Maps custom `u32` values to be treated as specific [`IndexType`] values.
    ///
    /// The default value is empty.
    pub index_types: BTreeMap<u32, IndexType>,

    pub _ne: crate::NonExhaustive<'static>,
}

impl IndirectCommandsLayoutToken {
    pub(crate) fn validate(
        &self,
        device: &Device,
        stream_count: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.token_type
            .validate_device(device)
            .map_err(|err| err.add_context("token_type"))?;

        if self.stream >= stream_count {
            return Err(Box::new(ValidationError {
                context: "stream".into(),
                problem: "is not less than `stream_count`".into(),
                vuids: &["VUID-VkIndirectCommandsLayoutTokenNV-stream-02951"],
                ..Default::default()
            }));
        }

        if self.offset
            > device
                .physical_device()
                .properties()
                .max_indirect_commands_token_offset
                .unwrap()
        {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "is greater than the `max_indirect_commands_token_offset` limit".into(),
                vuids: &["VUID-VkIndirectCommandsLayoutTokenNV-offset-02952"],
                ..Default::default()
            }));
        }

        // TODO: Alignment of offset VUID-VkIndirectCommandsLayoutTokenNV-offset-06888

        if self.token_type == IndirectCommandsTokenType::VertexBuffer {
            // TODO: vertex binding unit VUID-VkIndirectCommandsLayoutTokenNV-tokenType-02976
        }

        if self.token_type == IndirectCommandsTokenType::PushConstant
            && self.pushconstant_data.is_none()
        {
            return Err(Box::new(ValidationError {
                problem: "`token_type` is `IndirectCommandsTokenType::PushConstant`, but \
                    `pushconstant_data` is `None`"
                    .into(),
                ..Default::default()
            }));
        }

        if self.token_type != IndirectCommandsTokenType::PushConstant
            && self.pushconstant_data.is_some()
        {
            return Err(Box::new(ValidationError {
                problem: "`token_type` is not `IndirectCommandsTokenType::PushConstant`, \
                    but `pushconstant_data` is `Some`"
                    .into(),
                ..Default::default()
            }));
        }

        if let Some(pushconstant_data) = &self.pushconstant_data {
            pushconstant_data.validate(device)?;
        }

        if self.token_type == IndirectCommandsTokenType::StateFlags
            && self.indirect_state_flags == IndirectStateFlags::empty()
        {
            return Err(Box::new(ValidationError {
                problem: "`token_type` is `IndirectCommandsTokenType::StateFlags`, but \
                    `indirect_state_flags` is empty"
                    .into(),
                vuids: &["VUID-VkIndirectCommandsLayoutTokenNV-tokenType-02984"],
                ..Default::default()
            }));
        }

        // TODO: push data VUID-VkIndirectCommandsLayoutTokenNV-tokenType-11334

        Ok(())
    }
    pub(crate) fn to_vk<'a>(
        &self,
        fields1_vk: &'a IndirectCommandsLayoutTokenFieldVk1,
    ) -> vk::IndirectCommandsLayoutTokenNV<'a> {
        let as_vk = vk::IndirectCommandsLayoutTokenNV::default()
            .token_type(self.token_type.into())
            .stream(self.stream)
            .offset(self.offset)
            .vertex_binding_unit(self.vertex_binding_unit)
            .vertex_dynamic_stride(self.vertex_dynamic_stride)
            .indirect_state_flags(self.indirect_state_flags.into())
            .index_types(fields1_vk.index_types.as_slice())
            .index_type_values(fields1_vk.index_type_values.as_slice());
        match self.pushconstant_data.as_ref() {
            None => as_vk,
            Some(pushconstant_data) => as_vk
                .pushconstant_pipeline_layout(pushconstant_data.pipeline_layout.handle())
                .pushconstant_shader_stage_flags(pushconstant_data.shader_stage_flags.into())
                .pushconstant_offset(pushconstant_data.offset)
                .pushconstant_size(pushconstant_data.size),
        }
    }

    pub(crate) fn to_vk_field1(&self) -> IndirectCommandsLayoutTokenFieldVk1 {
        let (index_type_values, index_types) = self
            .index_types
            .iter()
            .map(|(index_type_value, index_type)| {
                (index_type_value, vk::IndexType::from(*index_type))
            })
            .unzip();

        IndirectCommandsLayoutTokenFieldVk1 {
            index_types,
            index_type_values,
        }
    }
}

impl Default for IndirectCommandsLayoutToken {
    fn default() -> IndirectCommandsLayoutToken {
        IndirectCommandsLayoutToken {
            token_type: IndirectCommandsTokenType::ShaderGroup,
            stream: 0,
            offset: 0,
            vertex_binding_unit: 0,
            vertex_dynamic_stride: false,
            pushconstant_data: None,
            indirect_state_flags: Default::default(),
            index_types: Default::default(),
            _ne: NE,
        }
    }
}

pub(crate) struct IndirectCommandsLayoutTokenFieldVk1 {
    pub(crate) index_types: Vec<vk::IndexType>,
    pub(crate) index_type_values: Vec<u32>,
}

vulkan_enum! {
    #[non_exhaustive]

    /// The type of indirect command token, specifying what type of command arguments are provided
    /// in an indirect commands stream.
    IndirectCommandsTokenType = IndirectCommandsTokenTypeNV(i32);

    /// Binds a pipeline shader group.
    ShaderGroup = SHADER_GROUP,

    /// Sets indirect state flags.
    StateFlags = STATE_FLAGS,

    /// Equivalent to [`bind_index_buffer`].
    ///
    /// [`bind_index_buffer`]: crate::command_buffer::AutoCommandBufferBuilder::bind_index_buffer
    IndexBuffer = INDEX_BUFFER,

    /// Equivalent to [`bind_vertex_buffers`].
    ///
    /// [`bind_vertex_buffers`]: crate::command_buffer::AutoCommandBufferBuilder::bind_vertex_buffers
    VertexBuffer = VERTEX_BUFFER,

    /// Equivalent to [`push_constants`].
    ///
    /// [`push_constants`]: crate::command_buffer::AutoCommandBufferBuilder::push_constants
    PushConstant = PUSH_CONSTANT,

    // TODO: enable
    //PushData = PUSH_DATA,

    /// Equivalent to [`draw_indexed_indirect`].
    ///
    /// [`draw_indexed_indirect`]: crate::command_buffer::AutoCommandBufferBuilder::draw_indexed_indirect
    DrawIndexed = DRAW_INDEXED,

    /// Equivalent to [`draw_indirect`].
    ///
    /// [`draw_indirect`]: crate::command_buffer::AutoCommandBufferBuilder::draw_indirect
    Draw = DRAW,

    /// Equivalent to [`draw_mesh_tasks_indirect`] for the NV mesh shader extension.
    ///
    /// [`draw_mesh_tasks_indirect`]: crate::command_buffer::AutoCommandBufferBuilder::draw_mesh_tasks_indirect
    DrawTasks = DRAW_TASKS,

    /// Equivalent to [`bind_pipeline_graphics`] or [`bind_pipeline_compute`].
    ///
    /// [`bind_pipeline_graphics`]: crate::command_buffer::AutoCommandBufferBuilder::bind_pipeline_graphics
    /// [`bind_pipeline_compute`]: crate::command_buffer::AutoCommandBufferBuilder::bind_pipeline_compute
    Pipeline = PIPELINE
    RequiresOneOf([
        RequiresAllOf([
            DeviceFeature(device_generated_compute_pipelines),
            DeviceExtension(nv_device_generated_commands_compute),
        ]),
    ]),

    /// Equivalent to [`dispatch_indirect`].
    ///
    /// [`dispatch_indirect`]: crate::command_buffer::AutoCommandBufferBuilder::dispatch_indirect
    Dispatch = DISPATCH
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_device_generated_commands_compute)]),
    ]),

    /// Equivalent to [`draw_mesh_tasks_indirect`] for the EXT mesh shader extension.
    ///
    /// [`draw_mesh_tasks_indirect`]: crate::command_buffer::AutoCommandBufferBuilder::draw_mesh_tasks_indirect
    DrawMeshTasks = DRAW_MESH_TASKS,
}

/// The push constant data for an [`IndirectCommandsLayoutToken`] with token type
/// [`IndirectCommandsTokenType::PushConstant`].
#[derive(Clone, Debug)]
pub struct IndirectCommandsLayoutTokenPushConstant {
    /// The pipeline layout used for the push constant command.
    ///
    /// There is no default value.
    pub pipeline_layout: Arc<PipelineLayout>,

    /// The shader stage flags used for the push constant command.
    ///
    /// The default value is [`ShaderStages::empty()`].
    pub shader_stage_flags: ShaderStages,

    /// The offset in bytes used for the push constant command.
    ///
    /// Must be a multiple of 4.
    ///
    /// The default value is `0`.
    pub offset: u32,

    /// The size in bytes used for the push constant command.
    ///
    /// Must be a multiple of 4.
    ///
    /// The default value is `0`.
    pub size: u32,
}

impl IndirectCommandsLayoutTokenPushConstant {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        self.shader_stage_flags
            .validate_device(device)
            .map_err(|err| err.add_context("shader_stage_flags"))?;

        if !self.offset.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-VkIndirectCommandsLayoutTokenNV-tokenType-02978"],
                ..Default::default()
            }));
        }

        if !self.size.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                context: "size".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-VkIndirectCommandsLayoutTokenNV-tokenType-02979"],
                ..Default::default()
            }));
        }

        if self.offset
            >= device
                .physical_device()
                .properties()
                .max_push_constants_size
        {
            return Err(Box::new(ValidationError {
                context: "offset".into(),
                problem: "is not less than the `max_push_constants_size` limit".into(),
                vuids: &["VUID-VkIndirectCommandsLayoutTokenNV-tokenType-02980"],
                ..Default::default()
            }));
        }

        if self.size
            > device
                .physical_device()
                .properties()
                .max_push_constants_size
                - self.offset
        {
            return Err(Box::new(ValidationError {
                problem: "`size` is greater than the `max_push_constants_size` limit \
                    minus `offset`"
                    .into(),
                vuids: &["VUID-VkIndirectCommandsLayoutTokenNV-tokenType-02981"],
                ..Default::default()
            }));
        }

        let start = self.offset;
        let end = self.offset + self.size;

        for range in self.pipeline_layout.push_constant_ranges() {
            let range_start = range.offset;
            let range_end = range.offset + range.size;

            if range_start < end && range_end > start {
                if !self.shader_stage_flags.contains(range.stages) {
                    return Err(Box::new(ValidationError {
                        problem: "for a push constant range in `pipeline_layout` that \
                            overlaps the byte range specified by `offset` and `size`, \
                            `shader_stage_flags` does not contain all stages in that \
                            range"
                            .into(),
                        vuids: &[
                            "VUID-VkIndirectCommandsLayoutTokenNV-tokenType-02983",
                        ],
                        ..Default::default()
                    }));
                }
            }
        }

        {
            let mut remaining_stages = self.shader_stage_flags;

            for range in self.pipeline_layout.push_constant_ranges() {
                let range_start = range.offset;
                let range_end = range.offset + range.size;

                if range_start <= start && range_end >= end {
                    remaining_stages -= remaining_stages & range.stages;
                }
            }

            if !remaining_stages.is_empty() {
                return Err(Box::new(ValidationError {
                    problem: "for a shader stage in `shader_stage_flags`, there is no \
                        push constant range in `pipeline_layout` that includes both \
                        that stage and the full byte range specified by `offset` and \
                        `size`"
                        .into(),
                    vuids: &[
                        "VUID-VkIndirectCommandsLayoutTokenNV-tokenType-02982",
                    ],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// A subset of the graphics pipeline state that can be altered using indirect state flags.
    IndirectStateFlags = IndirectStateFlagsNV(u32);

    /// Allows toggling the [`FrontFace`] rasterization state for subsequent drawing commands.
    ///
    /// [`FrontFace`]: crate::pipeline::graphics::rasterization::FrontFace
    FLAG_FRONTFACE = FLAG_FRONTFACE,
}

impl ComputePipeline {
    pub fn indirect_device_address(
        &self,
    ) -> Result<NonNullDeviceAddress, Box<ValidationError>> {
        self.validate_indirect_device_address()?;

        Ok(unsafe { self.indirect_device_address_unchecked() })
    }

    fn validate_indirect_device_address(&self) -> Result<(), Box<ValidationError>> {
        if !self
            .device()
            .enabled_features()
            .device_generated_compute_pipelines
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "device_generated_compute_pipelines",
                )])]),
                vuids: &[
                    "VUID-vkGetPipelineIndirectDeviceAddressNV-deviceGeneratedComputePipelines-09078",
                ],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn indirect_device_address_unchecked(&self) -> NonNullDeviceAddress {
        let address_info_vk = vk::PipelineIndirectDeviceAddressInfoNV::default()
            .pipeline_bind_point(self.bind_point().into())
            .pipeline(self.handle());

        let device_address_vk = {
            let fns = self.device().fns();
            unsafe {
                (fns.nv_device_generated_commands_compute
                    .get_pipeline_indirect_device_address_nv)(
                    self.device().handle(),
                    &address_info_vk,
                )
            }
        };

        NonNullDeviceAddress::new(device_address_vk).unwrap()
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
    pub preprocess_buffer: &'a Arc<Buffer>,

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
    pub sequence_count_buffer: Option<&'a Arc<Buffer>>,

    /// The byte offset into `sequence_count_buffer` where the count value is stored.
    ///
    /// The default value is `0`.
    pub sequence_count_buffer_offset: u64,

    /// A buffer that encodes the used sequence indices as a `u32` array.
    ///
    /// The default value is `None`.
    pub sequence_index_buffer: Option<&'a Arc<Buffer>>,

    /// The byte offset into `sequence_index_buffer` where the index values start.
    ///
    /// The default value is `0`.
    pub sequence_index_buffer_offset: u64,

    pub _ne: crate::NonExhaustive<'static>,
}

impl<'a> GeneratedCommandsInfo<'a> {
    pub fn graphics_pipeline(
        pipeline: Arc<GraphicsPipeline>,
        indirect_commands_layout: Arc<IndirectCommandsLayout>,
        preprocess_buffer: &'a Arc<Buffer>,
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
            _ne: NE,
        }
    }

    pub fn compute_pipeline(
        pipeline: Arc<ComputePipeline>,
        indirect_commands_layout: Arc<IndirectCommandsLayout>,
        preprocess_buffer: &'a Arc<Buffer>,
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
            _ne: NE,
        }
    }

    pub fn dynamic_pipeline(
        indirect_commands_layout: Arc<IndirectCommandsLayout>,
        preprocess_buffer: &'a Arc<Buffer>,
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
            _ne: NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let properties = device.physical_device().properties();

        if self.sequence_count
            > properties
                .max_indirect_sequence_count
                .unwrap_or(0)
        {
            return Err(Box::new(ValidationError {
                context: "sequence_count".into(),
                problem: "is greater than the `max_indirect_sequence_count` limit".into(),
                vuids: &["VUID-VkGeneratedCommandsInfoNV-sequencesCount-02917"],
                ..Default::default()
            }));
        }

        if !self
            .preprocess_buffer
            .usage()
            .intersects(BufferUsage::INDIRECT_BUFFER)
        {
            return Err(Box::new(ValidationError {
                context: "preprocess_buffer.usage()".into(),
                problem: "does not contain `BufferUsage::INDIRECT_BUFFER`".into(),
                vuids: &["VUID-VkGeneratedCommandsInfoNV-preprocessBuffer-02918"],
                ..Default::default()
            }));
        }

        if let Some(min_alignment) =
            properties.min_indirect_commands_buffer_offset_alignment
        {
            if !self.preprocess_offset.is_multiple_of(min_alignment as u64) {
                return Err(Box::new(ValidationError {
                    context: "preprocess_offset".into(),
                    problem: "is not a multiple of the \
                        `min_indirect_commands_buffer_offset_alignment` device property"
                        .into(),
                    vuids: &["VUID-VkGeneratedCommandsInfoNV-preprocessOffset-02919"],
                    ..Default::default()
                }));
            }
        }

        if let Some(sequence_count_buffer) = self.sequence_count_buffer {
            if !sequence_count_buffer
                .usage()
                .intersects(BufferUsage::INDIRECT_BUFFER)
            {
                return Err(Box::new(ValidationError {
                    context: "sequence_count_buffer.usage()".into(),
                    problem: "does not contain `BufferUsage::INDIRECT_BUFFER`".into(),
                    vuids: &["VUID-VkGeneratedCommandsInfoNV-sequencesCountBuffer-02922"],
                    ..Default::default()
                }));
            }

            if let Some(min_alignment) =
                properties.min_sequences_count_buffer_offset_alignment
            {
                if !self
                    .sequence_count_buffer_offset
                    .is_multiple_of(min_alignment as u64)
                {
                    return Err(Box::new(ValidationError {
                        context: "sequence_count_buffer_offset".into(),
                        problem: "is not a multiple of the \
                            `min_sequences_count_buffer_offset_alignment` device property"
                            .into(),
                        vuids: &[
                            "VUID-VkGeneratedCommandsInfoNV-sequencesCountBuffer-02923",
                        ],
                        ..Default::default()
                    }));
                }
            }
        }

        if let Some(sequence_index_buffer) = self.sequence_index_buffer {
            if !sequence_index_buffer
                .usage()
                .intersects(BufferUsage::INDIRECT_BUFFER)
            {
                return Err(Box::new(ValidationError {
                    context: "sequence_index_buffer.usage()".into(),
                    problem: "does not contain `BufferUsage::INDIRECT_BUFFER`".into(),
                    vuids: &["VUID-VkGeneratedCommandsInfoNV-sequencesIndexBuffer-02925"],
                    ..Default::default()
                }));
            }

            if let Some(min_alignment) =
                properties.min_sequences_index_buffer_offset_alignment
            {
                if !self
                    .sequence_index_buffer_offset
                    .is_multiple_of(min_alignment as u64)
                {
                    return Err(Box::new(ValidationError {
                        context: "sequence_index_buffer_offset".into(),
                        problem: "is not a multiple of the \
                            `min_sequences_index_buffer_offset_alignment` device property"
                            .into(),
                        vuids: &[
                            "VUID-VkGeneratedCommandsInfoNV-sequencesIndexBuffer-02926",
                        ],
                        ..Default::default()
                    }));
                }
            }
        }

        if self.streams.len() as u32
            != self.indirect_commands_layout.stream_count()
        {
            return Err(Box::new(ValidationError {
                problem: "the length of `streams` does not match \
                    `indirect_commands_layout.stream_count()`"
                    .into(),
                vuids: &["VUID-VkGeneratedCommandsInfoNV-streamCount-02916"],
                ..Default::default()
            }));
        }

        if let Some(min_alignment) =
            properties.min_indirect_commands_buffer_offset_alignment
        {
            for (index, stream) in self.streams.iter().enumerate() {
                if !stream.offset.is_multiple_of(min_alignment as u64) {
                    return Err(Box::new(ValidationError {
                        context: format!("streams[{}].offset", index).into(),
                        problem: "is not a multiple of the \
                            `min_indirect_commands_buffer_offset_alignment` device property"
                            .into(),
                        vuids: &["VUID-VkIndirectCommandsStreamNV-offset-02943"],
                        ..Default::default()
                    }));
                }
            }
        }

        if self.indirect_commands_layout.pipeline_bind_point() == PipelineBindPoint::Compute {
            match &self.pipeline {
                GeneratedCommandsPipeline::Compute(pipeline) => {
                    if !pipeline
                        .flags()
                        .intersects(PipelineCreateFlags::INDIRECT_BINDABLE)
                    {
                        return Err(Box::new(ValidationError {
                            context: "pipeline".into(),
                            problem: "`pipeline_bind_point` is \
                                `PipelineBindPoint::Compute`, but the pipeline was not \
                                created with the `PipelineCreateFlags::INDIRECT_BINDABLE` \
                                flag"
                                .into(),
                            vuids: &[
                                "VUID-VkGeneratedCommandsInfoNV-pipelineBindPoint-09084",
                            ],
                            ..Default::default()
                        }));
                    }
                }
                GeneratedCommandsPipeline::Dynamic => {
                    if !self
                        .indirect_commands_layout
                        .token_types()
                        .iter()
                        .any(|&t| t == IndirectCommandsTokenType::Pipeline)
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`pipeline` is `Dynamic`, but \
                                `indirect_commands_layout` does not contain a \
                                `IndirectCommandsTokenType::Pipeline` token"
                                .into(),
                            vuids: &[
                                "VUID-VkGeneratedCommandsInfoNV-pipelineBindPoint-09087",
                            ],
                            ..Default::default()
                        }));
                    }
                }
                _ => {}
            }

            if self
                .indirect_commands_layout
                .token_types()
                .iter()
                .any(|&t| t == IndirectCommandsTokenType::Pipeline)
            {
                if !matches!(self.pipeline, GeneratedCommandsPipeline::Dynamic) {
                    return Err(Box::new(ValidationError {
                        problem: "`indirect_commands_layout` contains a \
                            `IndirectCommandsTokenType::Pipeline` token, but `pipeline` \
                            is not `Dynamic`"
                            .into(),
                        vuids: &[
                            "VUID-VkGeneratedCommandsInfoNV-pipelineBindPoint-09087",
                        ],
                        ..Default::default()
                    }));
                }
            }
        }

        if self
            .indirect_commands_layout
            .flags()
            .intersects(IndirectCommandsLayoutUsageFlags::INDEXED_SEQUENCES)
        {
            if self.sequence_index_buffer.is_none() {
                return Err(Box::new(ValidationError {
                    problem: "`indirect_commands_layout.flags()` contains \
                        `IndirectCommandsLayoutUsageFlags::INDEXED_SEQUENCES`, but \
                        `sequence_index_buffer` is `None`"
                        .into(),
                    vuids: &[
                        "VUID-VkGeneratedCommandsInfoNV-sequencesIndexBuffer-02924",
                    ],
                    ..Default::default()
                }));
            }
        } else if self.sequence_index_buffer.is_some() {
            return Err(Box::new(ValidationError {
                problem: "`indirect_commands_layout.flags()` does not contain \
                    `IndirectCommandsLayoutUsageFlags::INDEXED_SEQUENCES`, but \
                    `sequence_index_buffer` is `Some`"
                    .into(),
                vuids: &[
                    "VUID-VkGeneratedCommandsInfoNV-sequencesIndexBuffer-02924",
                ],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk(
        &self,
        fields1_vk: &'a GeneratedCommandsInfoFieldsVk1,
    ) -> vk::GeneratedCommandsInfoNV<'a> {
        let result = vk::GeneratedCommandsInfoNV::default()
            .pipeline_bind_point(self.pipeline.bind_point().into())
            .pipeline(self.pipeline.handle())
            .indirect_commands_layout(self.indirect_commands_layout.handle())
            .streams(fields1_vk.streams.as_slice())
            .sequences_count(self.sequence_count)
            .preprocess_buffer(self.preprocess_buffer.handle())
            .preprocess_offset(self.preprocess_offset)
            .preprocess_size(self.preprocess_size);
        let result = match self.sequence_count_buffer {
            None => result,
            Some(buffer) => result
                .sequences_count_buffer(buffer.handle())
                .sequences_count_offset(self.sequence_count_buffer_offset),
        };
        let result = match self.sequence_index_buffer {
            None => result,
            Some(buffer) => result
                .sequences_index_buffer(buffer.handle())
                .sequences_index_offset(self.sequence_index_buffer_offset),
        };
        result
    }

    pub(crate) fn to_vk_fields1(&self) -> GeneratedCommandsInfoFieldsVk1 {
        let streams = self.streams.iter().map(|stream| stream.to_vk()).collect();
        GeneratedCommandsInfoFieldsVk1 { streams }
    }
}

pub(crate) struct GeneratedCommandsInfoFieldsVk1 {
    streams: Vec<vk::IndirectCommandsStreamNV>,
}

/// The pipeline used in the generation and execution of device-generated commands.
#[derive(Clone, Debug)]
pub enum GeneratedCommandsPipeline {
    /// The pipeline is selected dynamically via an
    /// [`IndirectCommandsTokenType::Pipeline`] token.
    Dynamic,

    /// A graphics pipeline.
    Graphics(Arc<GraphicsPipeline>),

    /// A compute pipeline.
    Compute(Arc<ComputePipeline>),
}

impl GeneratedCommandsPipeline {
    pub fn bind_point(&self) -> PipelineBindPoint {
        match self {
            GeneratedCommandsPipeline::Dynamic => PipelineBindPoint::Compute,
            GeneratedCommandsPipeline::Graphics(pipeline) => pipeline.bind_point(),
            GeneratedCommandsPipeline::Compute(pipeline) => pipeline.bind_point(),
        }
    }

    pub fn handle(&self) -> vk::Pipeline {
        match self {
            GeneratedCommandsPipeline::Dynamic => vk::Pipeline::null(),
            GeneratedCommandsPipeline::Graphics(pipeline) => pipeline.handle(),
            GeneratedCommandsPipeline::Compute(pipeline) => pipeline.handle(),
        }
    }
}

/// An input stream providing data for the tokens used in an [`IndirectCommandsLayout`].
#[derive(Clone, Debug)]
pub struct IndirectCommandsStream<'a> {
    /// The buffer storing the functional arguments for each sequence. These arguments can be
    /// written by the device.
    ///
    /// There is no default value.
    pub buffer: &'a Arc<Buffer>,

    /// The byte offset into `buffer` where the arguments start.
    ///
    /// The default value is `0`.
    pub offset: u64,
}

impl<'a> IndirectCommandsStream<'a> {
    pub(crate) fn to_vk(&self) -> vk::IndirectCommandsStreamNV {
        vk::IndirectCommandsStreamNV::default()
            .buffer(self.buffer.handle())
            .offset(self.offset)
    }
}

/// Parameters specifying where a compute pipeline's metadata will be stored for use with
/// device-generated commands.
#[derive(Clone, Debug)]
pub struct ComputePipelineIndirectBufferInfo {
    /// The device address where the pipeline's metadata will be stored.
    ///
    /// This must be aligned to the alignment returned by
    /// [`Device::pipeline_indirect_memory_requirements`].
    ///
    /// There is no default value.
    pub buffer: DeviceAddress,

    /// The size of the pipeline's metadata region.
    ///
    /// This must be at least the size returned by
    /// [`Device::pipeline_indirect_memory_requirements`].
    ///
    /// There is no default value.
    pub size: u64,

    /// If nonzero, the device address where the pipeline's metadata was originally saved, used
    /// to re-populate `buffer` for replay.
    ///
    /// If this is nonzero, then it must be an address retrieved from an identically created
    /// pipeline on the same implementation, and the pipeline metadata must also be placed on an
    /// identically created buffer and at the same offset.
    ///
    /// The default value is `0`.
    pub pipeline_device_address_capture_replay: DeviceAddress,

    pub _ne: crate::NonExhaustive<'static>,
}

impl ComputePipelineIndirectBufferInfo {
    pub fn buffer(
        buffer: DeviceAddress,
        size: u64,
    ) -> Self {
        Self {
            buffer,
            size,
            pipeline_device_address_capture_replay: DeviceAddress::default(),
            _ne: NE,
        }
    }

    pub(crate) fn validate(
        &self,
        device: &Device,
        pipeline_create_info: &ComputePipelineCreateInfo<'_>,
    ) -> Result<(), Box<ValidationError>> {
        if !device.enabled_features().device_generated_compute_pipelines {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "device_generated_compute_pipelines",
                )])]),
                vuids: &["VUID-VkComputePipelineIndirectBufferInfoNV-\
                    deviceGeneratedComputePipelines-09009"],
                ..Default::default()
            }));
        }

        let memory_requirements = device.pipeline_indirect_memory_requirements(
            pipeline_create_info,
        );

        if !self.buffer
            .is_multiple_of(memory_requirements.layout.alignment().as_devicesize())
        {
            return Err(Box::new(ValidationError {
                context: "buffer".into(),
                problem: "is not aligned to the required memory alignment".into(),
                vuids: &["VUID-VkComputePipelineIndirectBufferInfoNV-deviceAddress-09011"],
                ..Default::default()
            }));
        }

        if self.size < memory_requirements.layout.size() {
            return Err(Box::new(ValidationError {
                context: "size".into(),
                problem: "is less than the required memory size".into(),
                vuids: &["VUID-VkComputePipelineIndirectBufferInfoNV-size-09013"],
                ..Default::default()
            }));
        }

        if self.pipeline_device_address_capture_replay != 0 {
            if !device
                .enabled_features()
                .device_generated_compute_capture_replay
            {
                return Err(Box::new(ValidationError {
                    context: "pipeline_device_address_capture_replay".into(),
                    problem: "is not zero".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[
                        Requires::DeviceFeature(
                            "device_generated_compute_capture_replay",
                        ),
                    ])]),
                    vuids: &["VUID-VkComputePipelineIndirectBufferInfoNV-\
                        pipelineDeviceAddressCaptureReplay-09014"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::ComputePipelineIndirectBufferInfoNV<'_> {
        vk::ComputePipelineIndirectBufferInfoNV::default()
            .device_address(self.buffer)
            .size(self.size)
            .pipeline_device_address_capture_replay(
                self.pipeline_device_address_capture_replay
            )
    }
}