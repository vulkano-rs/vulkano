use crate::ResourceAccess;
use std::sync::Arc;
use vulkano::{
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    shader::{EntryPoint, ShaderModule, SpecializationConstant},
    Validated, VulkanError,
};

/// Load shader module from its module load function with specialization constants
pub fn load_module<const N: usize>(
    module_fn: fn(
        Arc<vulkano::device::Device>,
    ) -> Result<Arc<ShaderModule>, Validated<vulkano::VulkanError>>,
    access: &ResourceAccess,
    specialization_info: [(u32, SpecializationConstant); N],
) -> EntryPoint {
    let shader_module = module_fn(access.device()).unwrap();
    if N > 0 {
        shader_module
            .specialize(specialization_info.into_iter().collect())
            .unwrap()
            .entry_point("main")
            .unwrap()
    } else {
        shader_module.entry_point("main").unwrap()
    }
}

/// Generate compute pipelines from its module load function.
pub fn create_compute_pipeline<const N: usize>(
    module_fn: fn(
        Arc<vulkano::device::Device>,
    ) -> Result<Arc<ShaderModule>, Validated<vulkano::VulkanError>>,
    access: &ResourceAccess,
    bindless_layout: bool,
    specialization_info: [(u32, SpecializationConstant); N],
) -> Result<Arc<ComputePipeline>, Validated<VulkanError>> {
    let compute_shader = load_module(module_fn, access, specialization_info);

    let stage = PipelineShaderStageCreateInfo::new(compute_shader);

    let layout = if bindless_layout {
        access.pipeline_layout_from_stages(&[stage.clone()])?
    } else {
        PipelineLayout::new(
            access.device(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(access.device())
                .unwrap(),
        )?
    };

    ComputePipeline::new(
        access.device(),
        None,
        ComputePipelineCreateInfo::new(stage, layout),
    )
}
