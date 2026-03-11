use crate::buffer::IndexType;
use crate::device::{Device, DeviceOwned};
use crate::macros::{vulkan_bitflags, vulkan_enum};
use crate::pipeline::{PipelineBindPoint, PipelineLayout};
use crate::shader::ShaderStages;
use crate::VulkanError;
use crate::{Validated, ValidationError, VulkanObject};
use ash::vk;
use std::collections::BTreeMap;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

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
        todo!()
        //Ok(())
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
}

unsafe impl VulkanObject for IndirectCommandsLayout {
    type Handle = ash::vk::IndirectCommandsLayoutNV;

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
}

impl IndirectCommandsLayoutToken {
    pub(crate) fn validate(&self) -> Result<(), Box<ValidationError>> {
        todo!()
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

    ///
    DrawIndexed = DRAW_INDEXED,

    ///
    Draw = DRAW,

    ///
    DrawTasks = DRAW_TASKS,
}

#[derive(Clone, Debug)]
pub struct IndirectCommandsLayoutTokenPushConstant {
    pub pipeline_layout: Arc<PipelineLayout>,
    pub shader_stage_flags: ShaderStages,
    pub offset: u32,
    pub size: u32,
}

vulkan_bitflags! {
    #[non_exhaustive]

    IndirectStateFlags = IndirectStateFlagsNV(u32);

    FLAG_FRONTFACE = FLAG_FRONTFACE,
}
