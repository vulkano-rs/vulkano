use crate::buffer::{BufferUsage, IndexType, Subbuffer};
use crate::device::{Device, DeviceOwned};
use crate::macros::{vulkan_bitflags, vulkan_enum};
use crate::memory::MemoryRequirements;
use crate::pipeline::compute::ComputePipelineCreateInfo;
use crate::pipeline::{
    ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
};
use crate::shader::ShaderStages;
use crate::{NonNullDeviceAddress, VulkanError};
use crate::{Validated, ValidationError, VulkanObject};
use ash::vk;
use std::collections::BTreeMap;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;
use vulkano::{Requires, RequiresAllOf, RequiresOneOf};

#[derive(Debug)]
pub struct IndirectCommandsLayout {
    handle: vk::IndirectCommandsLayoutNV,
    device: Arc<Device>,

    push_constant_pipeline_layouts: Vec<Arc<PipelineLayout>>,
}

impl IndirectCommandsLayout {
    #[inline]
    pub fn new(
        device: Arc<Device>,
        create_info: IndirectCommandsLayoutCreateInfo,
    ) -> Result<Arc<IndirectCommandsLayout>, Validated<VulkanError>> {
        Self::validate_new(&device, &create_info)?;
        Ok(unsafe { Self::new_unchecked(device, create_info) }?)
    }

    fn validate_new(
        device: &Device,
        create_info: &IndirectCommandsLayoutCreateInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !device.enabled_features().device_generated_commands {
            return Err(Box::new(ValidationError {
                problem: "using device generated commands".into(),
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
        create_info: IndirectCommandsLayoutCreateInfo,
    ) -> Result<Arc<IndirectCommandsLayout>, VulkanError> {
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
        Ok(unsafe { Self::from_handle(device, handle, create_info) })
    }

    #[inline]
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: vk::IndirectCommandsLayoutNV,
        create_info: IndirectCommandsLayoutCreateInfo,
    ) -> Arc<IndirectCommandsLayout> {
        let push_constant_pipeline_layouts = create_info
            .tokens
            .into_iter()
            .filter_map(move |token| token.pushconstant_data)
            .map(|push_data| push_data.pipeline_layout)
            .collect();
        Arc::new(IndirectCommandsLayout {
            handle,
            device,
            push_constant_pipeline_layouts,
        })
    }

    pub fn memory_requirements(
        &self,
        pipeline: &GeneratedCommandsPipeline,
        max_sequence_count: u32,
    ) -> MemoryRequirements {
        let memory_requirements_info_vk = vk::GeneratedCommandsMemoryRequirementsInfoNV::default()
            .pipeline_bind_point(pipeline.bind_point().unwrap_or(PipelineBindPoint::Compute).into())
            .pipeline(pipeline.handle())
            .indirect_commands_layout(self.handle)
            .max_sequences_count(max_sequence_count);

        let memory_requirements_vk2 = {
            let fns = self.device.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.nv_device_generated_commands
                    .get_generated_commands_memory_requirements_nv)(
                    self.device.handle(),
                    &memory_requirements_info_vk,
                    output.as_mut_ptr(),
                )
            };
            unsafe { output.assume_init() }
        };

        let memory_requirements_extension_vk2 =
            MemoryRequirements::to_mut_vk2_extensions(self.device());

        MemoryRequirements::from_vk2(&memory_requirements_vk2, &memory_requirements_extension_vk2)
    }

    // TODO: Move this function somewhere else
    pub fn pipeline_indirect_memory_requirements(
        device: &Device,
        pipeline_create_info: &ComputePipelineCreateInfo,
    ) -> MemoryRequirements {
        // TODO: Validate extensions

        let create_info_fields2_vk = pipeline_create_info.to_vk_fields2();
        let create_info_fields1_vk = pipeline_create_info.to_vk_fields1(&create_info_fields2_vk);
        let mut create_info_extensions_vk = pipeline_create_info.to_vk_extensions();
        let create_info_vk =
            pipeline_create_info.to_vk(&create_info_fields1_vk, &mut create_info_extensions_vk);

        let memory_requirements_vk2 = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            unsafe {
                (fns.nv_device_generated_commands_compute
                    .get_pipeline_indirect_memory_requirements_nv)(
                    device.handle(),
                    &create_info_vk,
                    output.as_mut_ptr(),
                )
            };
            unsafe { output.assume_init() }
        };

        let memory_requirements_extension_vk2 =
            MemoryRequirements::to_mut_vk2_extensions(device);

        MemoryRequirements::from_vk2(&memory_requirements_vk2, &memory_requirements_extension_vk2)
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

#[derive(Clone, Debug)]
pub struct IndirectCommandsLayoutCreateInfo {
    pub flags: IndirectCommandsLayoutUsageFlags,
    pub pipeline_bind_point: PipelineBindPoint,
    pub tokens: Vec<IndirectCommandsLayoutToken>,
    pub stream_strides: Vec<u32>,
    pub _ne: crate::NonExhaustive,
}

impl IndirectCommandsLayoutCreateInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        self.flags
            .validate_device(device)
            .map_err(|err| err.add_context("flags"))?;

        self.pipeline_bind_point
            .validate_device(device)
            .map_err(|err| err.add_context("pipeline_bind_point"))?;

        if self.pipeline_bind_point != PipelineBindPoint::Compute
            || self.pipeline_bind_point != PipelineBindPoint::Graphics
        {
            return Err(Box::new(ValidationError {
                problem: "pipeline_bind_point must be either Compute or Graphics".into(),
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
                    problem: "token count is outside of bounds".into(),
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
                problem: "ShaderGroup token must be the first token".into(),
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
                problem: "Pipeline token must be the first token".into(),
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
                problem: "tokens must include at most one token of type StateFlags".into(),
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
                problem: "action tokens may only be the last token".into(),
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
                            problem: "draw command tokens require the pipeline bind point Graphics"
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
                            problem:
                                "dispatch command tokens require the pipeline bind point Compute"
                                    .into(),
                            vuids: &["VUID-VkIndirectCommandsLayoutCreateInfoNV-pTokens-02935"],
                            ..Default::default()
                        }))
                    }
                }
                _ => Err(Box::new(ValidationError {
                    problem: "the last token must be an action command".into(),
                    vuids: &["VUID-VkIndirectCommandsLayoutCreateInfoNV-pTokens-02935"],
                    ..Default::default()
                })),
            })
            .unwrap()?; // Guaranteed to be Some

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
                    problem: "stream count is outside of bounds".into(),
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
                    problem: "a stream stride value is outside of bounds".into(),
                    vuids: &["VUID-VkIndirectCommandsLayoutCreateInfoNV-pStreamStrides-02937"],
                    ..Default::default()
                }));
            }

            // TODO: Validate alignment
        }

        if self.pipeline_bind_point == PipelineBindPoint::Compute {
            if !device.enabled_features().device_generated_compute {
                return Err(Box::new(ValidationError {
                    problem: "pipeline bind point compute".into(),
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
                    problem: "pipeline bind point compute can only have pipeline and push constant state tokens".into(),
                    vuids: &["VUID-VkIndirectCommandsLayoutCreateInfoNV-pipelineBindPoint-09089"],
                    ..Default::default()

                }));
            }

            // Pipeline token can only be the first one and we have at least one token
            if self.tokens[0].token_type == IndirectCommandsTokenType::Pipeline
                && !device.enabled_features().device_generated_compute_pipelines
            {
                return Err(Box::new(ValidationError {
                    problem: "token type Pipeline".into(),
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

    pub(crate) fn to_vk<'a>(
        &'a self,
        fields1_vk: &'a IndirectCommandsLayoutCreateInfoFields1Vk<'_>,
    ) -> vk::IndirectCommandsLayoutCreateInfoNV<'a> {
        vk::IndirectCommandsLayoutCreateInfoNV::default()
            .flags(self.flags.into())
            .pipeline_bind_point(self.pipeline_bind_point.into())
            .tokens(fields1_vk.tokens.as_slice())
            .stream_strides(self.stream_strides.as_slice())
    }

    pub(crate) fn to_vk_fields1<'a>(
        &self,
        fields2_vk: &'a IndirectCommandsLayoutCreateInfoFields2Vk,
    ) -> IndirectCommandsLayoutCreateInfoFields1Vk<'a> {
        let tokens = self
            .tokens
            .iter()
            .zip(fields2_vk.token_index_types.iter())
            .map(|(token, token_fields1_vk)| token.to_vk(token_fields1_vk))
            .collect();
        IndirectCommandsLayoutCreateInfoFields1Vk { tokens }
    }

    pub(crate) fn to_vk_fields2(&self) -> IndirectCommandsLayoutCreateInfoFields2Vk {
        let token_index_types = self
            .tokens
            .iter()
            .map(|token| token.to_vk_field1())
            .collect();
        IndirectCommandsLayoutCreateInfoFields2Vk { token_index_types }
    }
}

impl Default for IndirectCommandsLayoutCreateInfo {
    fn default() -> IndirectCommandsLayoutCreateInfo {
        IndirectCommandsLayoutCreateInfo {
            flags: Default::default(),
            pipeline_bind_point: PipelineBindPoint::Graphics,
            tokens: vec![],
            stream_strides: vec![],
            _ne: crate::NonExhaustive(()),
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

    IndirectCommandsLayoutUsageFlags = IndirectCommandsLayoutUsageFlagsNV(u32);

    EXPLICIT_PREPROCESS = EXPLICIT_PREPROCESS,

    INDEXED_SEQUENCES = INDEXED_SEQUENCES,

    UNORDERED_SEQUENCES = UNORDERED_SEQUENCES,
}

#[derive(Clone, Debug)]
pub struct IndirectCommandsLayoutToken {
    pub token_type: IndirectCommandsTokenType,
    pub stream: u32,
    pub offset: u32,
    pub vertex_binding_unit: u32,
    pub vertex_dynamic_stride: bool,
    pub pushconstant_data: Option<IndirectCommandsLayoutTokenPushConstant>,
    pub indirect_state_flags: IndirectStateFlags,
    pub index_types: BTreeMap<u32, IndexType>,
    pub _ne: crate::NonExhaustive,
}

impl IndirectCommandsLayoutToken {
    pub(crate) fn validate(&self, device: &Device, stream_count: u32) -> Result<(), Box<ValidationError>> {
        self.token_type
            .validate_device(device)
            .map_err(|err| err.add_context("token_type"))?;

        if self.stream >= stream_count {
            return Err(Box::new(ValidationError {
                problem: "stream index is outside of bounds".into(),
                vuids: &["VUID-VkIndirectCommandsLayoutTokenNV-stream-02951"],
                ..Default::default()
            }));
        }

        if self.offset > device.physical_device().properties().max_indirect_commands_token_offset.unwrap() {
            return Err(Box::new(ValidationError {
                problem: "token offset is too big".into(),
                vuids: &["VUID-VkIndirectCommandsLayoutTokenNV-offset-02952"],
                ..Default::default()
            }));
        }

        // TODO: Alignment of offset VUID-VkIndirectCommandsLayoutTokenNV-offset-06888

        if self.token_type == IndirectCommandsTokenType::VertexBuffer {
            // TODO: vertex binding unit VUID-VkIndirectCommandsLayoutTokenNV-tokenType-02976
        }

        if (self.token_type == IndirectCommandsTokenType::PushConstant) != self.pushconstant_data.is_some() {
            return Err(Box::new(ValidationError {
                problem: "push constant data should be set if and only if token type is PushConstant".into(),
                ..Default::default()
            }));
        }

        if let Some(pushconstant_data) = &self.pushconstant_data {
            pushconstant_data.validate(device)?;
        }

        if self.token_type == IndirectCommandsTokenType::StateFlags && self.indirect_state_flags == IndirectStateFlags::empty() {
            return Err(Box::new(ValidationError {
                problem: "token type is StateFlags but indirect state flags is empty".into(),
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
            _ne: crate::NonExhaustive(()),
        }
    }
}

pub(crate) struct IndirectCommandsLayoutTokenFieldVk1 {
    pub(crate) index_types: Vec<vk::IndexType>,
    pub(crate) index_type_values: Vec<u32>,
}

vulkan_enum! {
    #[non_exhaustive]

    IndirectCommandsTokenType = IndirectCommandsTokenTypeNV(i32);

    ///
    ShaderGroup = SHADER_GROUP,

    ///
    StateFlags = STATE_FLAGS,

    ///
    IndexBuffer = INDEX_BUFFER,

    ///
    VertexBuffer = VERTEX_BUFFER,

    ///
    PushConstant = PUSH_CONSTANT,

    // TODO: enable
    //PushData = PUSH_DATA,

    ///
    DrawIndexed = DRAW_INDEXED,

    ///
    Draw = DRAW,

    ///
    DrawTasks = DRAW_TASKS,

    Pipeline = PIPELINE
    RequiresOneOf([
        RequiresAllOf([
            DeviceFeature(device_generated_compute_pipelines),
            DeviceExtension(nv_device_generated_commands_compute),
        ]),
    ]),

    Dispatch = DISPATCH
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_device_generated_commands_compute)]),
    ]),

    DrawMeshTasks = DRAW_MESH_TASKS,
}

#[derive(Clone, Debug)]
pub struct IndirectCommandsLayoutTokenPushConstant {
    pub pipeline_layout: Arc<PipelineLayout>,
    pub shader_stage_flags: ShaderStages,
    pub offset: u32,
    pub size: u32,
}

impl IndirectCommandsLayoutTokenPushConstant {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        self.shader_stage_flags.validate_device(device)
            .map_err(|err| err.add_context("shader_stage_flags"))?;

        if !self.offset.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                problem: "offset is not a multiple of 4".into(),
                vuids: &["VUID-VkIndirectCommandsLayoutTokenNV-tokenType-02978"],
                ..Default::default()
            }));
        }

        if !self.size.is_multiple_of(4) {
            return Err(Box::new(ValidationError {
                problem: "size is not a multiple of 4".into(),
                vuids: &["VUID-VkIndirectCommandsLayoutTokenNV-tokenType-02979"],
                ..Default::default()
            }));
        }

        if self.offset >= device.physical_device().properties().max_push_constants_size {
            return Err(Box::new(ValidationError {
                problem: "offset is too large for physical device limit".into(),
                vuids: &["VUID-VkIndirectCommandsLayoutTokenNV-tokenType-02980"],
                ..Default::default()
            }));
        }

        if self.size > device.physical_device().properties().max_push_constants_size - self.offset {
            return Err(Box::new(ValidationError {
                problem: "size is too large for physical device limit".into(),
                vuids: &["VUID-VkIndirectCommandsLayoutTokenNV-tokenType-02981"],
                ..Default::default()
            }));
        }

        // TODO: Validate push constant ranges in pipeline_layout

        Ok(())
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    IndirectStateFlags = IndirectStateFlagsNV(u32);

    FLAG_FRONTFACE = FLAG_FRONTFACE,
}

impl ComputePipeline {
    pub fn indirect_device_address(&self) -> NonNullDeviceAddress {
        // TODO: validate

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

        // TODO: use checks
        NonNullDeviceAddress::new(device_address_vk).unwrap()
    }
}

#[derive(Clone, Debug)]
pub struct GeneratedCommandsInfo {
    pub pipeline: GeneratedCommandsPipeline,
    pub indirect_commands_layout: Arc<IndirectCommandsLayout>,
    pub streams: Vec<IndirectCommandsStream>,
    pub sequence_count: u32,
    pub preprocess_buffer: Subbuffer<[u8]>,
    pub sequence_count_buffer: Option<Subbuffer<u32>>,
    pub sequence_index_buffer: Option<Subbuffer<u32>>,
    pub(crate) _ne: crate::NonExhaustive,
}

impl GeneratedCommandsInfo {

    pub fn graphics_pipeline(pipeline: Arc<GraphicsPipeline>, indirect_commands_layout: Arc<IndirectCommandsLayout>, preprocess_buffer: Subbuffer<[u8]>) -> Self {
        Self {
            pipeline: GeneratedCommandsPipeline::Graphics(pipeline),
            indirect_commands_layout,
            streams: vec![],
            sequence_count: 0,
            preprocess_buffer,
            sequence_count_buffer: None,
            sequence_index_buffer: None,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub fn compute_pipeline(pipeline: Arc<ComputePipeline>, indirect_commands_layout: Arc<IndirectCommandsLayout>, preprocess_buffer: Subbuffer<[u8]>) -> Self {
        Self {
            pipeline: GeneratedCommandsPipeline::Compute(pipeline),
            indirect_commands_layout,
            streams: vec![],
            sequence_count: 0,
            preprocess_buffer,
            sequence_count_buffer: None,
            sequence_index_buffer: None,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub fn dynamic_pipeline(indirect_commands_layout: Arc<IndirectCommandsLayout>, preprocess_buffer: Subbuffer<[u8]>) -> Self {
        Self {
            pipeline: GeneratedCommandsPipeline::Dynamic(),
            indirect_commands_layout,
            streams: vec![],
            sequence_count: 0,
            preprocess_buffer,
            sequence_count_buffer: None,
            sequence_index_buffer: None,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate(&self) -> Result<(), Box<ValidationError>> {
        todo!()
    }

    pub(crate) fn to_vk<'a>(
        &self,
        fields1_vk: &'a GeneratedCommandsInfoFieldsVk1,
    ) -> vk::GeneratedCommandsInfoNV<'a> {
        let result = vk::GeneratedCommandsInfoNV::default()
            .pipeline_bind_point(
                self.pipeline
                    .bind_point()
                    .map(|bind_point| bind_point.into())
                    .unwrap_or(vk::PipelineBindPoint::default()),
            )
            .pipeline(self.pipeline.handle())
            .indirect_commands_layout(self.indirect_commands_layout.handle())
            .streams(fields1_vk.streams.as_slice())
            .sequences_count(self.sequence_count)
            .preprocess_buffer(self.preprocess_buffer.buffer().handle())
            .preprocess_offset(self.preprocess_buffer.offset())
            .preprocess_size(self.preprocess_buffer.size());
        let result = match self.sequence_count_buffer.as_ref() {
            None => result,
            Some(buffer) => result
                .sequences_count_buffer(buffer.buffer().handle())
                .sequences_count_offset(buffer.offset()),
        };
        let result = match self.sequence_index_buffer.as_ref() {
            None => result,
            Some(buffer) => result
                .sequences_index_buffer(buffer.buffer().handle())
                .sequences_index_offset(buffer.offset()),
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

#[derive(Clone, Debug)]
pub enum GeneratedCommandsPipeline {
    Dynamic(),
    Graphics(Arc<GraphicsPipeline>),
    Compute(Arc<ComputePipeline>),
}

impl GeneratedCommandsPipeline {
    pub fn bind_point(&self) -> Option<PipelineBindPoint> {
        match self {
            GeneratedCommandsPipeline::Dynamic() => None,
            GeneratedCommandsPipeline::Graphics(pipeline) => Some(pipeline.bind_point()),
            GeneratedCommandsPipeline::Compute(pipeline) => Some(pipeline.bind_point()),
        }
    }

    pub fn handle(&self) -> vk::Pipeline {
        match self {
            GeneratedCommandsPipeline::Dynamic() => vk::Pipeline::null(),
            GeneratedCommandsPipeline::Graphics(pipeline) => pipeline.handle(),
            GeneratedCommandsPipeline::Compute(pipeline) => pipeline.handle(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct IndirectCommandsStream {
    pub buffer: Subbuffer<[u8]>,
}

impl IndirectCommandsStream {
    pub(crate) fn to_vk(&self) -> vk::IndirectCommandsStreamNV {
        vk::IndirectCommandsStreamNV::default()
            .buffer(self.buffer.buffer().handle())
            .offset(self.buffer.offset())
    }
}

#[derive(Clone, Debug)]
pub struct ComputePipelineIndirectBufferInfo {
    pub subbuffer: Subbuffer<[u8]>,
    // TODO: capture replay
    pub _ne: crate::NonExhaustive,
}

impl ComputePipelineIndirectBufferInfo {
    pub(crate) fn validate(&self, device: &Device, pipeline_create_info: &ComputePipelineCreateInfo) -> Result<(), Box<ValidationError>> {
        if !device.enabled_features().device_generated_compute_pipelines {
            return Err(Box::new(ValidationError {
                problem: "compute pipeline indirect buffer info".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature("device_generated_compute_pipelines")])]),
                vuids: &["VUID-VkComputePipelineIndirectBufferInfoNV-deviceGeneratedComputePipelines-09009"],
                ..Default::default()
            }));
        }

        let memory_requirements = IndirectCommandsLayout::pipeline_indirect_memory_requirements(device, pipeline_create_info);

        if !self.subbuffer.offset().is_multiple_of(memory_requirements.layout.alignment().as_devicesize()) {
            return Err(Box::new(ValidationError {
                problem: "offset of the subbuffer is not aligned correctly".into(),
                vuids: &["VUID-VkComputePipelineIndirectBufferInfoNV-deviceAddress-09011"],
                ..Default::default()
            }));
        }

        if self.subbuffer.size() < memory_requirements.layout.size() {
            return Err(Box::new(ValidationError {
                problem: "size of the subbuffer is smaller than the required minimum size".into(),
                vuids: &["VUID-VkComputePipelineIndirectBufferInfoNV-size-09013"],
                ..Default::default()
            }));
        }

        if !self.subbuffer.buffer().usage().contains(BufferUsage::TRANSFER_DST | BufferUsage::INDIRECT_BUFFER) {
            return Err(Box::new(ValidationError {
                problem: "pipeline indirect buffer must have usage set for TRANSFER_DST and INDIRECT_BUFFER".into(),
                vuids: &["VUID-VkComputePipelineIndirectBufferInfoNV-deviceAddress-09012"],
                ..Default::default()
            }));
        }

        // TODO: capture replay

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> vk::ComputePipelineIndirectBufferInfoNV<'_> {
        vk::ComputePipelineIndirectBufferInfoNV::default()
            .device_address(self.subbuffer.device_address().unwrap().get())
            .size(self.subbuffer.size())
    }
}
