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
    Validated, VulkanError, VulkanObject,
};

use super::{
    cache::PipelineCache, DynamicState, Pipeline, PipelineBindPoint, PipelineCreateFlags,
    PipelineLayout, PipelineShaderStageCreateInfo, PipelineShaderStageCreateInfoExtensionsVk,
    PipelineShaderStageCreateInfoFields1Vk, PipelineShaderStageCreateInfoFields2Vk,
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

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        cache: Option<Arc<PipelineCache>>,
        create_info: RayTracingPipelineCreateInfo,
    ) -> Result<Arc<Self>, VulkanError> {
        let handle = {
            let fields3_vk = create_info.to_vk_fields3();
            let fields2_vk = create_info.to_vk_fields2(&fields3_vk);
            let mut fields1_extensions_vk = create_info.to_vk_fields1_extensions();
            let fields1_vk = create_info.to_vk_fields1(&fields2_vk, &mut fields1_extensions_vk);
            let create_infos_vk = create_info.to_vk(&fields1_vk);

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

    pub(crate) fn to_vk<'a>(
        &self,
        fields1_vk: &'a RayTracingPipelineCreateInfoFields1Vk<'_>,
    ) -> ash::vk::RayTracingPipelineCreateInfoKHR<'a> {
        let &Self {
            flags,
            max_pipeline_ray_recursion_depth,

            ref layout,
            ref base_pipeline,
            ..
        } = self;

        let RayTracingPipelineCreateInfoFields1Vk {
            stages_vk,
            groups_vk,
            dynamic_state_vk,
        } = fields1_vk;

        let mut val_vk = ash::vk::RayTracingPipelineCreateInfoKHR::default()
            .flags(flags.into())
            .stages(stages_vk)
            .groups(groups_vk)
            .layout(layout.handle())
            .max_pipeline_ray_recursion_depth(max_pipeline_ray_recursion_depth)
            .base_pipeline_handle(
                base_pipeline
                    .as_ref()
                    .map_or(ash::vk::Pipeline::null(), |p| p.handle()),
            )
            .base_pipeline_index(-1);

        if let Some(dynamic_state_vk) = dynamic_state_vk {
            val_vk = val_vk.dynamic_state(dynamic_state_vk);
        }

        return val_vk;
    }

    pub(crate) fn to_vk_fields1<'a>(
        &self,
        fields2_vk: &'a RayTracingPipelineCreateInfoFields2Vk<'_>,
        extensions_vk: &'a mut RayTracingPipelineCreateInfoFields1ExtensionsVk,
    ) -> RayTracingPipelineCreateInfoFields1Vk<'a> {
        let Self { stages, groups, .. } = self;
        let RayTracingPipelineCreateInfoFields2Vk {
            stages_fields1_vk,
            dynamic_states_vk,
        } = fields2_vk;
        let RayTracingPipelineCreateInfoFields1ExtensionsVk {
            stages_extensions_vk,
        } = extensions_vk;

        let stages_vk: SmallVec<[_; 5]> = stages
            .iter()
            .zip(stages_fields1_vk)
            .zip(stages_extensions_vk)
            .map(|((stage, fields1), fields1_extensions_vk)| {
                stage.to_vk(fields1, fields1_extensions_vk)
            })
            .collect();

        let groups_vk = groups
            .iter()
            .map(RayTracingShaderGroupCreateInfo::to_vk)
            .collect();

        let dynamic_state_vk = (!dynamic_states_vk.is_empty()).then(|| {
            ash::vk::PipelineDynamicStateCreateInfo::default()
                .flags(ash::vk::PipelineDynamicStateCreateFlags::empty())
                .dynamic_states(dynamic_states_vk)
        });

        RayTracingPipelineCreateInfoFields1Vk {
            stages_vk,
            groups_vk,
            dynamic_state_vk,
        }
    }

    pub(crate) fn to_vk_fields1_extensions(
        &self,
    ) -> RayTracingPipelineCreateInfoFields1ExtensionsVk {
        let Self { stages, .. } = self;

        let stages_extensions_vk = stages
            .iter()
            .map(|stage| stage.to_vk_extensions())
            .collect();

        RayTracingPipelineCreateInfoFields1ExtensionsVk {
            stages_extensions_vk,
        }
    }

    pub(crate) fn to_vk_fields2<'a>(
        &self,
        fields3_vk: &'a RayTracingPipelineCreateInfoFields3Vk,
    ) -> RayTracingPipelineCreateInfoFields2Vk<'a> {
        let Self {
            stages,
            dynamic_state,
            ..
        } = self;

        let stages_fields1_vk = stages
            .iter()
            .zip(fields3_vk.stages_fields2_vk.iter())
            .map(|(stage, fields3)| stage.to_vk_fields1(fields3))
            .collect();

        let dynamic_states_vk = dynamic_state.iter().copied().map(Into::into).collect();

        RayTracingPipelineCreateInfoFields2Vk {
            stages_fields1_vk,
            dynamic_states_vk,
        }
    }

    pub(crate) fn to_vk_fields3<'a>(&self) -> RayTracingPipelineCreateInfoFields3Vk {
        let Self { stages, .. } = self;

        let stages_fields2_vk = stages.iter().map(|stage| stage.to_vk_fields2()).collect();

        RayTracingPipelineCreateInfoFields3Vk { stages_fields2_vk }
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

impl RayTracingShaderGroupCreateInfo {
    pub(crate) fn to_vk(&self) -> ash::vk::RayTracingShaderGroupCreateInfoKHR<'static> {
        // We are not using pointers in the struct, so 'static is used.
        ash::vk::RayTracingShaderGroupCreateInfoKHR::default()
            .ty(self.group_type)
            .general_shader(self.general_shader.unwrap_or(ash::vk::SHADER_UNUSED_KHR))
            .closest_hit_shader(
                self.closest_hit_shader
                    .unwrap_or(ash::vk::SHADER_UNUSED_KHR),
            )
            .any_hit_shader(self.any_hit_shader.unwrap_or(ash::vk::SHADER_UNUSED_KHR))
            .intersection_shader(
                self.intersection_shader
                    .unwrap_or(ash::vk::SHADER_UNUSED_KHR),
            )
    }
}

pub struct RayTracingPipelineCreateInfoFields1Vk<'a> {
    pub(crate) stages_vk: SmallVec<[ash::vk::PipelineShaderStageCreateInfo<'a>; 5]>,
    pub(crate) groups_vk: SmallVec<[ash::vk::RayTracingShaderGroupCreateInfoKHR<'static>; 5]>,
    pub(crate) dynamic_state_vk: Option<ash::vk::PipelineDynamicStateCreateInfo<'a>>,
}

pub struct RayTracingPipelineCreateInfoFields1ExtensionsVk {
    pub(crate) stages_extensions_vk: SmallVec<[PipelineShaderStageCreateInfoExtensionsVk; 5]>,
}

pub struct RayTracingPipelineCreateInfoFields2Vk<'a> {
    pub(crate) stages_fields1_vk: SmallVec<[PipelineShaderStageCreateInfoFields1Vk<'a>; 5]>,
    pub(crate) dynamic_states_vk: SmallVec<[ash::vk::DynamicState; 4]>,
}

pub struct RayTracingPipelineCreateInfoFields3Vk {
    pub(crate) stages_fields2_vk: SmallVec<[PipelineShaderStageCreateInfoFields2Vk; 5]>,
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
        // TODO: RayTracing: Add unit test for copy algorithm

        Ok(Self {
            raygen,
            miss,
            hit,
            callable,
            buffer: sbt_buffer,
        })
    }
}
