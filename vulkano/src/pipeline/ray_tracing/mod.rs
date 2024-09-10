use std::{collections::hash_map::Entry, mem::MaybeUninit, num::NonZeroU64, ptr, sync::Arc};

use ahash::{HashMap, HashSet};
use ash::vk::StridedDeviceAddressRegionKHR;
use smallvec::SmallVec;

use crate::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    device::{Device, DeviceOwned, DeviceOwnedDebugWrapper, DeviceOwnedVulkanObject},
    instance::InstanceOwnedDebugWrapper,
    macros::impl_id_counter,
    memory::{
        allocator::{align_up, AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
        DeviceAlignment,
    },
    shader::DescriptorBindingRequirements,
    Validated, ValidationError, VulkanError, VulkanObject,
};

use super::{
    cache::PipelineCache, DynamicState, Pipeline, PipelineBindPoint, PipelineCreateFlags,
    PipelineLayout, PipelineShaderStageCreateInfo,
};

#[derive(Debug)]
pub struct RayTracingPipeline {
    handle: ash::vk::Pipeline,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZeroU64,

    flags: PipelineCreateFlags,
    layout: DeviceOwnedDebugWrapper<Arc<PipelineLayout>>,

    descriptor_binding_requirements: HashMap<(u32, u32), DescriptorBindingRequirements>,
    num_used_descriptor_sets: u32,

    groups: SmallVec<[RayTracingShaderGroupCreateInfo; 5]>,
    stages: SmallVec<[PipelineShaderStageCreateInfo; 5]>,
}

impl RayTracingPipeline {
    /// Creates a new `RayTracingPipeline`.
    #[inline]
    pub fn new(
        device: Arc<Device>,
        cache: Option<Arc<PipelineCache>>,
        create_info: RayTracingPipelineCreateInfo,
    ) -> Result<Arc<Self>, Validated<VulkanError>> {
        // Self::validate_new(&device, cache.as_deref(), &create_info)?;

        unsafe { Ok(Self::new_unchecked(device, cache, create_info)?) }
    }

    fn validate_new(
        device: &Device,
        cache: Option<&PipelineCache>,
        create_info: &RayTracingPipelineCreateInfo,
    ) -> Result<(), Box<ValidationError>> {
        todo!()
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        cache: Option<Arc<PipelineCache>>,
        create_info: RayTracingPipelineCreateInfo,
    ) -> Result<Arc<Self>, VulkanError> {
        let RayTracingPipelineCreateInfo {
            flags,
            stages,
            groups,
            max_pipeline_ray_recursion_depth,
            dynamic_state,
            layout,
            base_pipeline,
            ..
        } = &create_info;

        let owned_stages_vk: SmallVec<[_; 5]> =
            stages.iter().map(|s| s.to_owned_vulkan()).collect();
        let stages_vk: SmallVec<[_; 5]> = owned_stages_vk.iter().map(|s| s.to_vulkan()).collect();

        let groups_vk: SmallVec<[_; 5]> = groups
            .iter()
            .map(|g| ash::vk::RayTracingShaderGroupCreateInfoKHR {
                ty: g.group_type,
                general_shader: g.general_shader.unwrap_or(ash::vk::SHADER_UNUSED_KHR),
                closest_hit_shader: g.closest_hit_shader.unwrap_or(ash::vk::SHADER_UNUSED_KHR),
                any_hit_shader: g.any_hit_shader.unwrap_or(ash::vk::SHADER_UNUSED_KHR),
                intersection_shader: g.intersection_shader.unwrap_or(ash::vk::SHADER_UNUSED_KHR),
                // TODO: RayTracing: p_shader_group_capture_replay_handle
                ..Default::default()
            })
            .collect();

        let dynamic_state_list_vk: SmallVec<[_; 4]> =
            dynamic_state.iter().copied().map(Into::into).collect();
        let dynamic_state_vk =
            (!dynamic_state_list_vk.is_empty()).then(|| ash::vk::PipelineDynamicStateCreateInfo {
                flags: ash::vk::PipelineDynamicStateCreateFlags::empty(),
                dynamic_state_count: dynamic_state_list_vk.len() as u32,
                p_dynamic_states: dynamic_state_list_vk.as_ptr(),
                ..Default::default()
            });

        let create_infos_vk = ash::vk::RayTracingPipelineCreateInfoKHR {
            flags: (*flags).into(),
            stage_count: stages_vk.len() as u32,
            p_stages: stages_vk.as_ptr(),
            group_count: groups_vk.len() as u32,
            p_groups: groups_vk.as_ptr(),
            max_pipeline_ray_recursion_depth: *max_pipeline_ray_recursion_depth,
            layout: layout.handle(),
            base_pipeline_handle: base_pipeline
                .as_deref()
                .map_or(ash::vk::Pipeline::null(), |p| p.handle()),
            base_pipeline_index: 0,
            p_dynamic_state: dynamic_state_vk.as_ref().map_or(ptr::null(), |d| d),
            // TODO: RayTracing: library
            ..Default::default()
        };

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();

            (fns.khr_ray_tracing_pipeline
                .create_ray_tracing_pipelines_khr)(
                device.handle(),
                ash::vk::DeferredOperationKHR::null(), // TODO: RayTracing: deferred_operation
                cache.map_or(ash::vk::PipelineCache::null(), |c| c.handle()),
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

    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::Pipeline,
        create_info: RayTracingPipelineCreateInfo,
    ) -> Arc<Self> {
        let RayTracingPipelineCreateInfo {
            flags,
            stages,
            groups,
            layout,
            ..
        } = create_info;

        let mut descriptor_binding_requirements: HashMap<
            (u32, u32),
            DescriptorBindingRequirements,
        > = HashMap::default();
        for stage in &stages {
            for (&loc, reqs) in stage
                .entry_point
                .info()
                .descriptor_binding_requirements
                .iter()
            {
                match descriptor_binding_requirements.entry(loc) {
                    Entry::Occupied(entry) => {
                        entry.into_mut().merge(reqs).expect("Could not produce an intersection of the shader descriptor requirements");
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(reqs.clone());
                    }
                }
            }
        }
        let num_used_descriptor_sets = descriptor_binding_requirements
            .keys()
            .map(|loc| loc.0)
            .max()
            .map(|x| x + 1)
            .unwrap_or(0);
        Arc::new(Self {
            handle,
            device: InstanceOwnedDebugWrapper(device),
            id: Self::next_id(),

            flags,
            layout: DeviceOwnedDebugWrapper(layout),

            descriptor_binding_requirements,
            num_used_descriptor_sets,

            groups,
            stages,
        })
    }

    pub fn groups(&self) -> &[RayTracingShaderGroupCreateInfo] {
        &self.groups
    }

    pub fn stages(&self) -> &[PipelineShaderStageCreateInfo] {
        &self.stages
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl Pipeline for RayTracingPipeline {
    #[inline]
    fn bind_point(&self) -> PipelineBindPoint {
        PipelineBindPoint::RayTracing
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

impl_id_counter!(RayTracingPipeline);

unsafe impl VulkanObject for RayTracingPipeline {
    type Handle = ash::vk::Pipeline;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for RayTracingPipeline {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.device()
    }
}

impl Drop for RayTracingPipeline {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.destroy_pipeline)(self.device.handle(), self.handle, ptr::null());
        }
    }
}

/// Parameters to create a new `RayTracingPipeline`.
#[derive(Clone, Debug)]
pub struct RayTracingPipelineCreateInfo {
    /// Additional properties of the pipeline.
    ///
    /// The default value is empty.
    pub flags: PipelineCreateFlags,

    /// The compute shader stage to use.
    ///
    /// There is no default value.
    pub stages: SmallVec<[PipelineShaderStageCreateInfo; 5]>,

    pub groups: SmallVec<[RayTracingShaderGroupCreateInfo; 5]>,

    pub max_pipeline_ray_recursion_depth: u32,

    pub dynamic_state: HashSet<DynamicState>,

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
    pub base_pipeline: Option<Arc<RayTracingPipeline>>,

    pub _ne: crate::NonExhaustive,
}

impl RayTracingPipelineCreateInfo {
    pub fn layout(layout: Arc<PipelineLayout>) -> Self {
        Self {
            flags: PipelineCreateFlags::empty(),
            stages: SmallVec::new(),
            groups: SmallVec::new(),
            max_pipeline_ray_recursion_depth: 0,
            dynamic_state: Default::default(),

            layout,

            base_pipeline: None,
            _ne: crate::NonExhaustive(()),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RayTracingShaderGroupCreateInfo {
    pub group_type: ash::vk::RayTracingShaderGroupTypeKHR, // TODO: Custom type
    pub general_shader: Option<u32>,
    pub closest_hit_shader: Option<u32>,
    pub any_hit_shader: Option<u32>,
    pub intersection_shader: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct ShaderBindingTable {
    raygen: StridedDeviceAddressRegionKHR,
    miss: StridedDeviceAddressRegionKHR,
    hit: StridedDeviceAddressRegionKHR,
    callable: StridedDeviceAddressRegionKHR,
    buffer: Subbuffer<[u8]>,
}

impl ShaderBindingTable {
    pub fn raygen(&self) -> &StridedDeviceAddressRegionKHR {
        &self.raygen
    }

    pub fn miss(&self) -> &StridedDeviceAddressRegionKHR {
        &self.miss
    }

    pub fn hit(&self) -> &StridedDeviceAddressRegionKHR {
        &self.hit
    }

    pub fn callable(&self) -> &StridedDeviceAddressRegionKHR {
        &self.callable
    }

    pub(crate) fn buffer(&self) -> &Subbuffer<[u8]> {
        &self.buffer
    }

    pub fn new(
        allocator: Arc<dyn MemoryAllocator>,
        ray_tracing_pipeline: &RayTracingPipeline,
        miss_shader_count: u64,
        hit_shader_count: u64,
        callable_shader_count: u64,
    ) -> Result<Self, Validated<VulkanError>> {
        let handle_data = ray_tracing_pipeline
            .device()
            .get_ray_tracing_shader_group_handles(
                &ray_tracing_pipeline,
                0,
                ray_tracing_pipeline.groups().len() as u32,
            )?;

        let properties = ray_tracing_pipeline.device().physical_device().properties();
        let handle_size_aligned = align_up(
            handle_data.handle_size() as u64,
            DeviceAlignment::new(properties.shader_group_handle_alignment.unwrap() as u64)
                .expect("unexpected shader_group_handle_alignment"),
        );

        let shader_group_base_alignment =
            DeviceAlignment::new(properties.shader_group_base_alignment.unwrap() as u64)
                .expect("unexpected shader_group_base_alignment");

        let raygen_stride = align_up(handle_size_aligned, shader_group_base_alignment);

        let mut raygen = StridedDeviceAddressRegionKHR {
            stride: raygen_stride,
            size: raygen_stride,
            device_address: 0,
        };
        let mut miss = StridedDeviceAddressRegionKHR {
            stride: handle_size_aligned,
            size: align_up(
                handle_size_aligned * miss_shader_count,
                shader_group_base_alignment,
            ),
            device_address: 0,
        };
        let mut hit = StridedDeviceAddressRegionKHR {
            stride: handle_size_aligned,
            size: align_up(
                handle_size_aligned * hit_shader_count,
                shader_group_base_alignment,
            ),
            device_address: 0,
        };
        let mut callable = StridedDeviceAddressRegionKHR {
            stride: handle_size_aligned,
            size: align_up(
                handle_size_aligned * callable_shader_count,
                shader_group_base_alignment,
            ),
            device_address: 0,
        };

        let sbt_buffer = Buffer::new_slice::<u8>(
            allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC
                    | BufferUsage::SHADER_DEVICE_ADDRESS
                    | BufferUsage::SHADER_BINDING_TABLE,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            raygen.size + miss.size + hit.size + callable.size,
        )
        .expect("todo: raytracing: better error type");
        sbt_buffer
            .buffer()
            .set_debug_utils_object_name("Shader Binding Table Buffer".into())
            .unwrap();

        raygen.device_address = sbt_buffer.buffer().device_address().unwrap().get();
        miss.device_address = raygen.device_address + raygen.size;
        hit.device_address = miss.device_address + miss.size;
        callable.device_address = hit.device_address + hit.size;

        {
            let mut sbt_buffer_write = sbt_buffer.write().unwrap();

            let mut handle_iter = handle_data.iter();

            let handle_size = handle_data.handle_size() as usize;
            sbt_buffer_write[..handle_size].copy_from_slice(handle_iter.next().unwrap());
            let mut offset = raygen.size as usize;
            for _ in 0..miss_shader_count {
                sbt_buffer_write[offset..offset + handle_size]
                    .copy_from_slice(handle_iter.next().unwrap());
                offset += miss.stride as usize;
            }
            offset = (raygen.size + miss.size) as usize;
            for _ in 0..hit_shader_count {
                sbt_buffer_write[offset..offset + handle_size]
                    .copy_from_slice(handle_iter.next().unwrap());
                offset += hit.stride as usize;
            }
            offset = (raygen.size + miss.size + hit.size) as usize;
            for _ in 0..callable_shader_count {
                sbt_buffer_write[offset..offset + handle_size]
                    .copy_from_slice(handle_iter.next().unwrap());
                offset += callable.stride as usize;
            }
        }

        Ok(Self {
            raygen,
            miss,
            hit,
            callable,
            buffer: sbt_buffer,
        })
    }
}
