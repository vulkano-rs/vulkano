// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! A pipeline that performs general-purpose operations.
//!
//! A compute pipeline takes buffers and/or images as both inputs and outputs. It operates
//! "standalone", with no additional infrastructure such as render passes or vertex input. Compute
//! pipelines can be used by themselves for performing work on the Vulkan device, but they can also
//! assist graphics operations by precalculating or postprocessing the operations from another kind
//! of pipeline. While it theoretically possible to perform graphics operations entirely in a
//! compute pipeline, a graphics pipeline is better suited to that task.
//!
//! A compute pipeline is relatively simple to create, requiring only a pipeline layout and a single
//! shader, the *compute shader*. The compute shader is the actual program that performs the work.
//! Once created, you can execute a compute pipeline by *binding* it in a command buffer, binding
//! any descriptor sets and/or push constants that the pipeline needs, and then issuing a `dispatch`
//! command on the command buffer.

use super::{PipelineCreateFlags, PipelineShaderStageCreateInfo};
use crate::{
    device::{Device, DeviceOwned, DeviceOwnedDebugWrapper},
    instance::InstanceOwnedDebugWrapper,
    macros::impl_id_counter,
    pipeline::{cache::PipelineCache, layout::PipelineLayout, Pipeline, PipelineBindPoint},
    shader::{spirv::ExecutionModel, DescriptorBindingRequirements, ShaderStage},
    Validated, ValidationError, VulkanError, VulkanObject,
};
use ahash::HashMap;
use std::{ffi::CString, fmt::Debug, mem::MaybeUninit, num::NonZeroU64, ptr, sync::Arc};

/// A pipeline object that describes to the Vulkan implementation how it should perform compute
/// operations.
///
/// The template parameter contains the descriptor set to use with this pipeline.
///
/// Pass an optional `Arc` to a `PipelineCache` to enable pipeline caching. The vulkan
/// implementation will handle the `PipelineCache` and check if it is available.
/// Check the documentation of the `PipelineCache` for more information.
#[derive(Debug)]
pub struct ComputePipeline {
    handle: ash::vk::Pipeline,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZeroU64,

    flags: PipelineCreateFlags,
    layout: DeviceOwnedDebugWrapper<Arc<PipelineLayout>>,

    descriptor_binding_requirements: HashMap<(u32, u32), DescriptorBindingRequirements>,
    num_used_descriptor_sets: u32,
}

impl ComputePipeline {
    /// Creates a new `ComputePipeline`.
    #[inline]
    pub fn new(
        device: Arc<Device>,
        cache: Option<Arc<PipelineCache>>,
        create_info: ComputePipelineCreateInfo,
    ) -> Result<Arc<ComputePipeline>, Validated<VulkanError>> {
        Self::validate_new(&device, cache.as_ref().map(AsRef::as_ref), &create_info)?;

        unsafe { Ok(Self::new_unchecked(device, cache, create_info)?) }
    }

    fn validate_new(
        device: &Device,
        cache: Option<&PipelineCache>,
        create_info: &ComputePipelineCreateInfo,
    ) -> Result<(), Box<ValidationError>> {
        // VUID-vkCreateComputePipelines-pipelineCache-parent
        if let Some(cache) = &cache {
            assert_eq!(device, cache.device().as_ref());
        }
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;
        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        cache: Option<Arc<PipelineCache>>,
        create_info: ComputePipelineCreateInfo,
    ) -> Result<Arc<ComputePipeline>, VulkanError> {
        let &ComputePipelineCreateInfo {
            flags,
            ref stage,
            ref layout,
            ref base_pipeline,
            _ne: _,
        } = &create_info;

        let stage_vk;
        let name_vk;
        let specialization_info_vk;
        let specialization_map_entries_vk: Vec<_>;
        let mut specialization_data_vk: Vec<u8>;
        let required_subgroup_size_create_info;

        {
            let &PipelineShaderStageCreateInfo {
                flags,
                ref entry_point,
                ref required_subgroup_size,
                _ne: _,
            } = stage;

            let entry_point_info = entry_point.info();
            name_vk = CString::new(entry_point_info.name.as_str()).unwrap();

            specialization_data_vk = Vec::new();
            specialization_map_entries_vk = entry_point
                .module()
                .specialization_info()
                .iter()
                .map(|(&constant_id, value)| {
                    let data = value.as_bytes();
                    let offset = specialization_data_vk.len() as u32;
                    let size = data.len();
                    specialization_data_vk.extend(data);

                    ash::vk::SpecializationMapEntry {
                        constant_id,
                        offset,
                        size,
                    }
                })
                .collect();

            specialization_info_vk = ash::vk::SpecializationInfo {
                map_entry_count: specialization_map_entries_vk.len() as u32,
                p_map_entries: specialization_map_entries_vk.as_ptr(),
                data_size: specialization_data_vk.len(),
                p_data: specialization_data_vk.as_ptr() as *const _,
            };
            required_subgroup_size_create_info =
                required_subgroup_size.map(|required_subgroup_size| {
                    ash::vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo {
                        required_subgroup_size,
                        ..Default::default()
                    }
                });
            stage_vk = ash::vk::PipelineShaderStageCreateInfo {
                p_next: required_subgroup_size_create_info.as_ref().map_or(
                    ptr::null(),
                    |required_subgroup_size_create_info| {
                        required_subgroup_size_create_info as *const _ as _
                    },
                ),
                flags: flags.into(),
                stage: ShaderStage::from(entry_point_info.execution_model).into(),
                module: entry_point.module().handle(),
                p_name: name_vk.as_ptr(),
                p_specialization_info: if specialization_info_vk.data_size == 0 {
                    ptr::null()
                } else {
                    &specialization_info_vk
                },
                ..Default::default()
            };
        }

        let create_infos_vk = ash::vk::ComputePipelineCreateInfo {
            flags: flags.into(),
            stage: stage_vk,
            layout: layout.handle(),
            base_pipeline_handle: base_pipeline
                .as_ref()
                .map_or(ash::vk::Pipeline::null(), VulkanObject::handle),
            base_pipeline_index: -1,
            ..Default::default()
        };

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_compute_pipelines)(
                device.handle(),
                cache.as_ref().map_or_else(Default::default, |c| c.handle()),
                1,
                &create_infos_vk,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Self::from_handle(device, handle, create_info))
    }

    /// Creates a new `ComputePipeline` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::Pipeline,
        create_info: ComputePipelineCreateInfo,
    ) -> Arc<ComputePipeline> {
        let ComputePipelineCreateInfo {
            flags,
            stage,
            layout,
            base_pipeline: _,
            _ne: _,
        } = create_info;

        let descriptor_binding_requirements: HashMap<_, _> = stage
            .entry_point
            .info()
            .descriptor_binding_requirements
            .iter()
            .map(|(&loc, reqs)| (loc, reqs.clone()))
            .collect();
        let num_used_descriptor_sets = descriptor_binding_requirements
            .keys()
            .map(|loc| loc.0)
            .max()
            .map(|x| x + 1)
            .unwrap_or(0);

        Arc::new(ComputePipeline {
            handle,
            device: InstanceOwnedDebugWrapper(device),
            id: Self::next_id(),

            flags,
            layout: DeviceOwnedDebugWrapper(layout),

            descriptor_binding_requirements,
            num_used_descriptor_sets,
        })
    }

    /// Returns the `Device` that the pipeline was created with.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the flags that the pipeline was created with.
    #[inline]
    pub fn flags(&self) -> PipelineCreateFlags {
        self.flags
    }
}

impl Pipeline for ComputePipeline {
    #[inline]
    fn bind_point(&self) -> PipelineBindPoint {
        PipelineBindPoint::Compute
    }

    #[inline]
    fn layout(&self) -> &Arc<PipelineLayout> {
        &self.layout
    }

    #[inline]
    fn num_used_descriptor_sets(&self) -> u32 {
        self.num_used_descriptor_sets
    }

    #[inline]
    fn descriptor_binding_requirements(
        &self,
    ) -> &HashMap<(u32, u32), DescriptorBindingRequirements> {
        &self.descriptor_binding_requirements
    }
}

impl_id_counter!(ComputePipeline);

unsafe impl VulkanObject for ComputePipeline {
    type Handle = ash::vk::Pipeline;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for ComputePipeline {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.device()
    }
}

impl Drop for ComputePipeline {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.destroy_pipeline)(self.device.handle(), self.handle, ptr::null());
        }
    }
}

/// Parameters to create a new `ComputePipeline`.
#[derive(Clone, Debug)]
pub struct ComputePipelineCreateInfo {
    /// Additional properties of the pipeline.
    ///
    /// The default value is empty.
    pub flags: PipelineCreateFlags,

    /// The compute shader stage to use.
    ///
    /// There is no default value.
    pub stage: PipelineShaderStageCreateInfo,

    /// The pipeline layout to use.
    ///
    /// There is no default value.
    pub layout: Arc<PipelineLayout>,

    /// The pipeline to use as a base when creating this pipeline.
    ///
    /// If this is `Some`, then `flags` must contain [`PipelineCreateFlags::DERIVATIVE`],
    /// and the `flags` of the provided pipeline must contain
    /// [`PipelineCreateFlags::ALLOW_DERIVATIVES`].
    ///
    /// The default value is `None`.
    pub base_pipeline: Option<Arc<ComputePipeline>>,

    pub _ne: crate::NonExhaustive,
}

impl ComputePipelineCreateInfo {
    /// Returns a `ComputePipelineCreateInfo` with the specified `stage` and `layout`.
    #[inline]
    pub fn stage_layout(stage: PipelineShaderStageCreateInfo, layout: Arc<PipelineLayout>) -> Self {
        Self {
            flags: PipelineCreateFlags::empty(),
            stage,
            layout,
            base_pipeline: None,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            ref stage,
            ref layout,
            ref base_pipeline,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkComputePipelineCreateInfo-flags-parameter"])
        })?;

        stage
            .validate(device)
            .map_err(|err| err.add_context("stage"))?;

        if flags.intersects(PipelineCreateFlags::DERIVATIVE) {
            let base_pipeline = base_pipeline.as_ref().ok_or_else(|| {
                Box::new(ValidationError {
                    problem: "`flags` contains `PipelineCreateFlags::DERIVATIVE`, but \
                        `base_pipeline` is `None`"
                        .into(),
                    vuids: &["VUID-VkComputePipelineCreateInfo-flags-07984"],
                    ..Default::default()
                })
            })?;

            if !base_pipeline
                .flags()
                .intersects(PipelineCreateFlags::ALLOW_DERIVATIVES)
            {
                return Err(Box::new(ValidationError {
                    context: "base_pipeline.flags()".into(),
                    problem: "does not contain `PipelineCreateFlags::ALLOW_DERIVATIVES`".into(),
                    vuids: &["VUID-vkCreateComputePipelines-flags-00696"],
                    ..Default::default()
                }));
            }
        } else if base_pipeline.is_some() {
            return Err(Box::new(ValidationError {
                problem: "`flags` does not contain `PipelineCreateFlags::DERIVATIVE`, but \
                    `base_pipeline` is `Some`"
                    .into(),
                ..Default::default()
            }));
        }

        let &PipelineShaderStageCreateInfo {
            flags: _,
            ref entry_point,
            required_subgroup_size: _vk,
            _ne: _,
        } = &stage;

        let entry_point_info = entry_point.info();

        if !matches!(entry_point_info.execution_model, ExecutionModel::GLCompute) {
            return Err(Box::new(ValidationError {
                context: "stage.entry_point".into(),
                problem: "is not a `ShaderStage::Compute` entry point".into(),
                vuids: &["VUID-VkComputePipelineCreateInfo-stage-00701"],
                ..Default::default()
            }));
        }

        // TODO: Make sure that all VUIDs are indeed checked.
        layout
            .ensure_compatible_with_shader(
                entry_point_info
                    .descriptor_binding_requirements
                    .iter()
                    .map(|(k, v)| (*k, v)),
                entry_point_info.push_constant_requirements.as_ref(),
            )
            .map_err(|err| {
                Box::new(ValidationError {
                    context: "stage.entry_point".into(),
                    vuids: &[
                        "VUID-VkComputePipelineCreateInfo-layout-07987",
                        "VUID-VkComputePipelineCreateInfo-layout-07988",
                        "VUID-VkComputePipelineCreateInfo-layout-07990",
                        "VUID-VkComputePipelineCreateInfo-layout-07991",
                    ],
                    ..ValidationError::from_error(err)
                })
            })?;

        // TODO:
        // VUID-VkComputePipelineCreateInfo-stage-00702
        // VUID-VkComputePipelineCreateInfo-layout-01687

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        buffer::{Buffer, BufferCreateInfo, BufferUsage},
        command_buffer::{
            allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        },
        descriptor_set::{
            allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
        },
        memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
        pipeline::{
            compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
            ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
            PipelineShaderStageCreateInfo,
        },
        shader::{ShaderModule, ShaderModuleCreateInfo, ShaderStages},
        sync::{now, GpuFuture},
    };
    use std::sync::Arc;

    // TODO: test for basic creation
    // TODO: test for pipeline layout error

    #[test]
    fn specialization_constants() {
        // This test checks whether specialization constants work.
        // It executes a single compute shader (one invocation) that writes the value of a spec.
        // constant to a buffer. The buffer content is then checked for the right value.

        let (device, queue) = gfx_dev_and_queue!();

        let cs = unsafe {
            /*
            #version 450

            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

            layout(constant_id = 83) const int VALUE = 0xdeadbeef;

            layout(set = 0, binding = 0) buffer Output {
                int write;
            } write;

            void main() {
                write.write = VALUE;
            }
            */
            const MODULE: [u32; 120] = [
                119734787, 65536, 524289, 14, 0, 131089, 1, 393227, 1, 1280527431, 1685353262,
                808793134, 0, 196622, 0, 1, 327695, 5, 4, 1852399981, 0, 393232, 4, 17, 1, 1, 1,
                196611, 2, 450, 262149, 4, 1852399981, 0, 262149, 7, 1886680399, 29813, 327686, 7,
                0, 1953067639, 101, 262149, 9, 1953067639, 101, 262149, 11, 1431060822, 69, 327752,
                7, 0, 35, 0, 196679, 7, 3, 262215, 9, 34, 0, 262215, 9, 33, 0, 262215, 11, 1, 83,
                131091, 2, 196641, 3, 2, 262165, 6, 32, 1, 196638, 7, 6, 262176, 8, 2, 7, 262203,
                8, 9, 2, 262187, 6, 10, 0, 262194, 6, 11, 3735928559, 262176, 12, 2, 6, 327734, 2,
                4, 0, 3, 131320, 5, 327745, 12, 13, 9, 10, 196670, 13, 11, 65789, 65592,
            ];
            let module =
                ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(&MODULE)).unwrap();
            module
                .specialize([(83, 0x12345678i32.into())].into_iter().collect())
                .unwrap()
                .entry_point("main")
                .unwrap()
        };

        let pipeline = {
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let data_buffer = Buffer::from_data(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            0,
        )
        .unwrap();

        let ds_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let set = PersistentDescriptorSet::new(
            ds_allocator,
            pipeline.layout().set_layouts().get(0).unwrap().clone(),
            [WriteDescriptorSet::buffer(0, data_buffer.clone())],
            [],
        )
        .unwrap();

        let cb_allocator = StandardCommandBufferAllocator::new(device.clone(), Default::default());
        let mut cbb = AutoCommandBufferBuilder::primary(
            &cb_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        cbb.bind_pipeline_compute(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set,
            )
            .unwrap()
            .dispatch([1, 1, 1])
            .unwrap();
        let cb = cbb.build().unwrap();

        let future = now(device)
            .then_execute(queue, cb)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let data_buffer_content = data_buffer.read().unwrap();
        assert_eq!(*data_buffer_content, 0x12345678);
    }

    #[test]
    fn required_subgroup_size() {
        // This test checks whether required_subgroup_size works.
        // It executes a single compute shader (one invocation) that writes the subgroup size
        // to a buffer. The buffer content is then checked for the right value.

        let (device, queue) = gfx_dev_and_queue!(subgroup_size_control);

        if !device
            .physical_device()
            .properties()
            .required_subgroup_size_stages
            .unwrap_or_default()
            .intersects(ShaderStages::COMPUTE)
        {
            return;
        }

        let cs = unsafe {
            /*
            #version 450

            #extension GL_KHR_shader_subgroup_basic: enable

            layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Output {
                uint write;
            } write;

            void main() {
                if (gl_GlobalInvocationID.x == 0) {
                    write.write = gl_SubgroupSize;
                }
            }
            */
            const MODULE: [u32; 246] = [
                119734787, 65536, 851978, 30, 0, 131089, 1, 131089, 61, 393227, 1, 1280527431,
                1685353262, 808793134, 0, 196622, 0, 1, 458767, 5, 4, 1852399981, 0, 9, 23, 393232,
                4, 17, 128, 1, 1, 196611, 2, 450, 655364, 1197427783, 1279741775, 1885560645,
                1953718128, 1600482425, 1701734764, 1919509599, 1769235301, 25974, 524292,
                1197427783, 1279741775, 1852399429, 1685417059, 1768185701, 1952671090, 6649449,
                589828, 1264536647, 1935626824, 1701077352, 1970495346, 1869768546, 1650421877,
                1667855201, 0, 262149, 4, 1852399981, 0, 524293, 9, 1197436007, 1633841004,
                1986939244, 1952539503, 1231974249, 68, 262149, 18, 1886680399, 29813, 327686, 18,
                0, 1953067639, 101, 262149, 20, 1953067639, 101, 393221, 23, 1398762599,
                1919378037, 1399879023, 6650473, 262215, 9, 11, 28, 327752, 18, 0, 35, 0, 196679,
                18, 3, 262215, 20, 34, 0, 262215, 20, 33, 0, 196679, 23, 0, 262215, 23, 11, 36,
                196679, 24, 0, 262215, 29, 11, 25, 131091, 2, 196641, 3, 2, 262165, 6, 32, 0,
                262167, 7, 6, 3, 262176, 8, 1, 7, 262203, 8, 9, 1, 262187, 6, 10, 0, 262176, 11, 1,
                6, 131092, 14, 196638, 18, 6, 262176, 19, 2, 18, 262203, 19, 20, 2, 262165, 21, 32,
                1, 262187, 21, 22, 0, 262203, 11, 23, 1, 262176, 25, 2, 6, 262187, 6, 27, 128,
                262187, 6, 28, 1, 393260, 7, 29, 27, 28, 28, 327734, 2, 4, 0, 3, 131320, 5, 327745,
                11, 12, 9, 10, 262205, 6, 13, 12, 327850, 14, 15, 13, 10, 196855, 17, 0, 262394,
                15, 16, 17, 131320, 16, 262205, 6, 24, 23, 327745, 25, 26, 20, 22, 196670, 26, 24,
                131321, 17, 131320, 17, 65789, 65592,
            ];
            let module =
                ShaderModule::new(device.clone(), ShaderModuleCreateInfo::new(&MODULE)).unwrap();
            module.entry_point("main").unwrap()
        };

        let properties = device.physical_device().properties();
        let subgroup_size = properties.min_subgroup_size.unwrap_or(1);

        let pipeline = {
            let stage = PipelineShaderStageCreateInfo {
                required_subgroup_size: Some(subgroup_size),
                ..PipelineShaderStageCreateInfo::new(cs)
            };
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let data_buffer = Buffer::from_data(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            0,
        )
        .unwrap();

        let ds_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let set = PersistentDescriptorSet::new(
            ds_allocator,
            pipeline.layout().set_layouts().get(0).unwrap().clone(),
            [WriteDescriptorSet::buffer(0, data_buffer.clone())],
            [],
        )
        .unwrap();

        let cb_allocator = StandardCommandBufferAllocator::new(device.clone(), Default::default());
        let mut cbb = AutoCommandBufferBuilder::primary(
            &cb_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        cbb.bind_pipeline_compute(pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set,
            )
            .unwrap()
            .dispatch([128, 1, 1])
            .unwrap();
        let cb = cbb.build().unwrap();

        let future = now(device)
            .then_execute(queue, cb)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let data_buffer_content = data_buffer.read().unwrap();
        assert_eq!(*data_buffer_content, subgroup_size);
    }
}
