//! Ray tracing pipeline functionality for GPU-accelerated ray tracing.
//!
//! # Overview
//! Ray tracing pipelines enable high-performance ray tracing by defining a set of shader stages
//! that handle ray generation, intersection testing, and shading calculations. The pipeline
//! consists of different shader stages organized into shader groups.
//!
//! # Shader Types
//!
//! ## Ray Generation Shader
//! - Entry point for ray tracing
//! - Generates and traces primary rays
//! - Controls the overall ray tracing process
//!
//! ## Intersection Shaders
//! - **Built-in Triangle Intersection**: Handles standard triangle geometry intersection
//! - **Custom Intersection**: Implements custom geometry intersection testing
//!
//! ## Hit Shaders
//! - **Closest Hit**: Executes when a ray finds its closest intersection
//! - **Any Hit**: Optional shader that runs on every potential intersection
//!
//! ## Miss Shader
//! - Executes when a ray doesn't intersect any geometry
//! - Typically handles environment mapping or background colors
//!
//! ## Callable Shader
//! - Utility shader that can be called from other shader stages
//! - Enables code reuse across different shader stages
//!
//! # Pipeline Organization
//! Shaders are organized into groups:
//! - General groups: Contains ray generation, miss, or callable shaders
//! - Triangle hit groups: Contains closest-hit and optional any-hit shaders
//! - Procedural hit groups: Contains intersection, closest-hit, and optional any-hit shaders
//!
//! The ray tracing pipeline uses a Shader Binding Table (SBT) to organize and access
//! these shader groups during execution.

use super::{
    cache::PipelineCache, DynamicState, Pipeline, PipelineBindPoint, PipelineCreateFlags,
    PipelineLayout, PipelineShaderStageCreateInfo, PipelineShaderStageCreateInfoExtensionsVk,
    PipelineShaderStageCreateInfoFields1Vk, PipelineShaderStageCreateInfoFields2Vk,
};
use crate::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    device::{Device, DeviceOwned, DeviceOwnedDebugWrapper},
    instance::InstanceOwnedDebugWrapper,
    macros::impl_id_counter,
    memory::{
        allocator::{align_up, AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
        DeviceAlignment,
    },
    shader::{spirv::ExecutionModel, DescriptorBindingRequirements},
    Validated, ValidationError, VulkanError, VulkanObject,
};
use ash::vk::StridedDeviceAddressRegionKHR;
use foldhash::{HashMap, HashSet};
use smallvec::SmallVec;
use std::{collections::hash_map::Entry, mem::MaybeUninit, num::NonZeroU64, ptr, sync::Arc};

/// Defines how the implementation should perform ray tracing operations.
///
/// This object uses the `VK_KHR_ray_tracing_pipeline` extension.
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
        Self::validate_new(&device, cache.as_deref(), &create_info)?;

        unsafe { Ok(Self::new_unchecked(device, cache, create_info)?) }
    }

    fn validate_new(
        device: &Arc<Device>,
        cache: Option<&PipelineCache>,
        create_info: &RayTracingPipelineCreateInfo,
    ) -> Result<(), Validated<VulkanError>> {
        if let Some(cache) = &cache {
            assert_eq!(device, cache.device());
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

    pub fn flags(&self) -> PipelineCreateFlags {
        self.flags
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

    /// The ray tracing shader stages to use.
    ///
    /// There is no default value.
    pub stages: SmallVec<[PipelineShaderStageCreateInfo; 5]>,

    /// The shader groups to use. They reference the shader stages in `stages`.
    ///
    /// The default value is empty.
    pub groups: SmallVec<[RayTracingShaderGroupCreateInfo; 5]>,

    /// The maximum recursion depth of the pipeline.
    ///
    /// The default value is 1.
    pub max_pipeline_ray_recursion_depth: u32,

    /// The dynamic state to use.
    ///
    /// May only contain `DynamicState::RayTracingPipelineStackSize`.
    ///
    /// The default value is empty.
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
            max_pipeline_ray_recursion_depth: 1,
            dynamic_state: Default::default(),

            layout,

            base_pipeline: None,
            _ne: crate::NonExhaustive(()),
        }
    }

    fn validate(&self, device: &Arc<Device>) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            ref stages,
            ref groups,
            ref layout,
            ref base_pipeline,
            ref dynamic_state,
            max_pipeline_ray_recursion_depth,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkRayTracingPipelineCreateInfoKHR-flags-parameter"])
        })?;

        if flags.intersects(PipelineCreateFlags::DERIVATIVE) {
            let base_pipeline = base_pipeline.as_ref().ok_or_else(|| {
                Box::new(ValidationError {
                    problem: "`flags` contains `PipelineCreateFlags::DERIVATIVE`, but \
                        `base_pipeline` is `None`"
                        .into(),
                    vuids: &["VUID-VkRayTracingPipelineCreateInfoKHR-flags-07984
"],
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
                    vuids: &["VUID-vkCreateRayTracingPipelinesKHR-flags-03416"],
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

        for stage in stages {
            stage.validate(device).map_err(|err| {
                err.add_context("stages")
                    .set_vuids(&["VUID-VkRayTracingPipelineCreateInfoKHR-pStages-parameter"])
            })?;

            let entry_point_info = stage.entry_point.info();

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
                            "VUID-VkRayTracingPipelineCreateInfoKHR-layout-07987",
                            "VUID-VkRayTracingPipelineCreateInfoKHR-layout-07988",
                            "VUID-VkRayTracingPipelineCreateInfoKHR-layout-07990",
                            "VUID-VkRayTracingPipelineCreateInfoKHR-layout-07991",
                        ],
                        ..ValidationError::from_error(err)
                    })
                })?;
        }

        for group in groups {
            group.validate(stages).map_err(|err| {
                err.add_context("groups")
                    .set_vuids(&["VUID-VkRayTracingPipelineCreateInfoKHR-pGroups-parameter"])
            })?;
        }

        // TODO: Enable
        // if dynamic_state
        //     .iter()
        //     .any(|&state| state != DynamicState::RayTracingPipelineStackSize)
        // {
        //     return Err(Box::new(ValidationError {
        //         problem:
        //             format!("`dynamic_state` contains a dynamic state other than
        // RayTracingPipelineStackSize: {:?}", dynamic_state).into(),         vuids:
        // &["VUID-VkRayTracingPipelineCreateInfoKHR-pDynamicStates-03602"],
        //         ..Default::default()
        //     }));
        // }
        if !dynamic_state.is_empty() {
            todo!("Dynamic state for ray tracing pipelines is not yet supported");
        }

        let max_ray_recursion_depth = device
            .physical_device()
            .properties()
            .max_ray_recursion_depth
            .unwrap();
        if max_pipeline_ray_recursion_depth > max_ray_recursion_depth {
            return Err(Box::new(ValidationError {
                problem: format!(
                    "`max_pipeline_ray_recursion_depth` is greater than the device's max value of {}",
                    max_ray_recursion_depth
                ).into(),
                vuids: &["VUID-VkRayTracingPipelineCreateInfoKHR-maxPipelineRayRecursionDepth-03589"],
                ..Default::default()
            }));
        }

        Ok(())
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

        val_vk
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

    pub(crate) fn to_vk_fields3(&self) -> RayTracingPipelineCreateInfoFields3Vk {
        let Self { stages, .. } = self;

        let stages_fields2_vk = stages.iter().map(|stage| stage.to_vk_fields2()).collect();

        RayTracingPipelineCreateInfoFields3Vk { stages_fields2_vk }
    }
}

/// Enum representing different types of Ray Tracing Shader Groups.
///
/// Contains the index of the shader to use for each type of shader group.
/// The index corresponds to the position of the shader in the `stages` field of the
/// `RayTracingPipelineCreateInfo`.
#[derive(Debug, Clone)]
pub enum RayTracingShaderGroupCreateInfo {
    /// General shader group type, typically used for ray generation and miss shaders.
    ///
    /// Contains a single shader stage that can be:
    /// - Ray generation shader
    /// - Miss shader
    /// - Callable shader
    General {
        /// Index of the general shader stage
        general_shader: u32,
    },

    /// Procedural hit shader group type, used for custom intersection testing.
    ///
    /// Used when implementing custom intersection shapes or volumes.
    /// Requires an intersection shader and can optionally include closest hit
    /// and any hit shaders.
    ProceduralHit {
        /// Optional index of the closest hit shader stage
        closest_hit_shader: Option<u32>,
        /// Optional index of the any hit shader stage
        any_hit_shader: Option<u32>,
        /// Index of the intersection shader stage
        intersection_shader: u32,
    },

    /// Triangle hit shader group type, used for built-in triangle intersection.
    ///
    /// Used for standard triangle geometry intersection testing.
    /// Can optionally include closest hit and any hit shaders.
    TrianglesHit {
        /// Optional index of the closest hit shader stage
        closest_hit_shader: Option<u32>,
        /// Optional index of the any hit shader stage
        any_hit_shader: Option<u32>,
    },
}

impl RayTracingShaderGroupCreateInfo {
    fn validate(
        &self,
        stages: &[PipelineShaderStageCreateInfo],
    ) -> Result<(), Box<ValidationError>> {
        let get_shader_type =
            |shader: u32| stages[shader as usize].entry_point.info().execution_model;

        match self {
            RayTracingShaderGroupCreateInfo::General { general_shader } => {
                match get_shader_type(*general_shader) {
                    ExecutionModel::RayGenerationKHR
                    | ExecutionModel::MissKHR
                    | ExecutionModel::CallableKHR => Ok(()),
                    _ => Err(Box::new(ValidationError {
                        problem: "general shader in GENERAL group must be a RayGeneration, Miss, or Callable shader".into(),
                        vuids: &["VUID-VkRayTracingShaderGroupCreateInfoKHR-type-03474"],
                        ..Default::default()
                    })),
                }?;
            }
            RayTracingShaderGroupCreateInfo::ProceduralHit {
                intersection_shader,
                any_hit_shader,
                closest_hit_shader,
            } => {
                if get_shader_type(*intersection_shader) != ExecutionModel::IntersectionKHR {
                    return Err(Box::new(ValidationError {
                        problem: "intersection shader in PROCEDURAL_HIT_GROUP must be an Intersection shader".into(),
                        vuids: &["VUID-VkRayTracingShaderGroupCreateInfoKHR-type-03476"],
                        ..Default::default()
                    }));
                }

                if let Some(any_hit_shader) = any_hit_shader {
                    if get_shader_type(*any_hit_shader) != ExecutionModel::AnyHitKHR {
                        return Err(Box::new(ValidationError {
                            problem: "any hit shader must be an AnyHit shader".into(),
                            vuids: &[
                                "VUID-VkRayTracingShaderGroupCreateInfoKHR-anyHitShader-03479",
                            ],
                            ..Default::default()
                        }));
                    }
                }

                if let Some(closest_hit_shader) = closest_hit_shader {
                    if get_shader_type(*closest_hit_shader) != ExecutionModel::ClosestHitKHR {
                        return Err(Box::new(ValidationError {
                            problem: "closest hit shader must be a ClosestHit shader".into(),
                            vuids: &[
                                "VUID-VkRayTracingShaderGroupCreateInfoKHR-closestHitShader-03478",
                            ],
                            ..Default::default()
                        }));
                    }
                }
            }
            RayTracingShaderGroupCreateInfo::TrianglesHit {
                any_hit_shader,
                closest_hit_shader,
            } => {
                if let Some(any_hit_shader) = any_hit_shader {
                    if get_shader_type(*any_hit_shader) != ExecutionModel::AnyHitKHR {
                        return Err(Box::new(ValidationError {
                            problem: "any hit shader must be an AnyHit shader".into(),
                            vuids: &[
                                "VUID-VkRayTracingShaderGroupCreateInfoKHR-anyHitShader-03479",
                            ],
                            ..Default::default()
                        }));
                    }
                }

                if let Some(closest_hit_shader) = closest_hit_shader {
                    if get_shader_type(*closest_hit_shader) != ExecutionModel::ClosestHitKHR {
                        return Err(Box::new(ValidationError {
                            problem: "closest hit shader must be a ClosestHit shader".into(),
                            vuids: &[
                                "VUID-VkRayTracingShaderGroupCreateInfoKHR-closestHitShader-03478",
                            ],
                            ..Default::default()
                        }));
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk(&self) -> ash::vk::RayTracingShaderGroupCreateInfoKHR<'static> {
        match self {
            RayTracingShaderGroupCreateInfo::General { general_shader } => {
                ash::vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(ash::vk::RayTracingShaderGroupTypeKHR::GENERAL)
                    .general_shader(*general_shader)
                    .closest_hit_shader(ash::vk::SHADER_UNUSED_KHR)
                    .any_hit_shader(ash::vk::SHADER_UNUSED_KHR)
                    .intersection_shader(ash::vk::SHADER_UNUSED_KHR)
            }
            RayTracingShaderGroupCreateInfo::ProceduralHit {
                closest_hit_shader,
                any_hit_shader,
                intersection_shader,
            } => ash::vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(ash::vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
                .general_shader(ash::vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(closest_hit_shader.unwrap_or(ash::vk::SHADER_UNUSED_KHR))
                .any_hit_shader(any_hit_shader.unwrap_or(ash::vk::SHADER_UNUSED_KHR))
                .intersection_shader(*intersection_shader),
            RayTracingShaderGroupCreateInfo::TrianglesHit {
                closest_hit_shader,
                any_hit_shader,
            } => ash::vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(ash::vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(ash::vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(closest_hit_shader.unwrap_or(ash::vk::SHADER_UNUSED_KHR))
                .any_hit_shader(any_hit_shader.unwrap_or(ash::vk::SHADER_UNUSED_KHR))
                .intersection_shader(ash::vk::SHADER_UNUSED_KHR),
        }
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

/// An object that holds the addresses of the shader groups in a shader binding table.
#[derive(Debug, Clone)]
pub struct ShaderBindingTableAddresses {
    pub raygen: StridedDeviceAddressRegionKHR,
    pub miss: StridedDeviceAddressRegionKHR,
    pub hit: StridedDeviceAddressRegionKHR,
    pub callable: StridedDeviceAddressRegionKHR,
}

/// An object that holds the shader binding table buffer and its addresses.
#[derive(Debug, Clone)]
pub struct ShaderBindingTable {
    addresses: ShaderBindingTableAddresses,
    _buffer: Subbuffer<[u8]>,
}

impl ShaderBindingTable {
    pub fn addresses(&self) -> &ShaderBindingTableAddresses {
        &self.addresses
    }

    /// Automatically creates a shader binding table from a ray tracing pipeline.
    pub fn new(
        allocator: Arc<dyn MemoryAllocator>,
        ray_tracing_pipeline: &RayTracingPipeline,
    ) -> Result<Self, Validated<VulkanError>> {
        let mut miss_shader_count: u64 = 0;
        let mut hit_shader_count: u64 = 0;
        let mut callable_shader_count: u64 = 0;

        for group in ray_tracing_pipeline.groups() {
            match group {
                RayTracingShaderGroupCreateInfo::General { general_shader } => {
                    match ray_tracing_pipeline.stages()[*general_shader as usize]
                        .entry_point
                        .info()
                        .execution_model
                    {
                        ExecutionModel::RayGenerationKHR => {}
                        ExecutionModel::MissKHR => miss_shader_count += 1,
                        ExecutionModel::CallableKHR => callable_shader_count += 1,
                        _ => {
                            panic!("Unexpected shader type in general shader group");
                        }
                    }
                }
                RayTracingShaderGroupCreateInfo::ProceduralHit { .. }
                | RayTracingShaderGroupCreateInfo::TrianglesHit { .. } => {
                    hit_shader_count += 1;
                }
            }
        }

        let handle_data = ray_tracing_pipeline
            .device()
            .ray_tracing_shader_group_handles(
                ray_tracing_pipeline,
                0,
                ray_tracing_pipeline.groups().len() as u32,
            )?;

        let properties = ray_tracing_pipeline.device().physical_device().properties();
        let handle_size_aligned = align_up(
            handle_data.handle_size() as u64,
            DeviceAlignment::new(properties.shader_group_handle_alignment.unwrap() as u64).unwrap(),
        );

        let shader_group_base_alignment =
            DeviceAlignment::new(properties.shader_group_base_alignment.unwrap() as u64).unwrap();

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
        .expect("todo: raytracing: better error type for buffer errors");

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
            addresses: ShaderBindingTableAddresses {
                raygen,
                miss,
                hit,
                callable,
            },
            _buffer: sbt_buffer,
        })
    }
}
