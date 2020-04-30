// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use smallvec::SmallVec;
use std::mem;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;
use std::u32;

use descriptor::pipeline_layout::PipelineLayoutAbstract;
use descriptor::pipeline_layout::PipelineLayoutDesc;
use descriptor::pipeline_layout::PipelineLayoutDescTweaks;
use device::Device;
use pipeline::ray_tracing_pipeline::Inner as RayTracingPipelineInner;
use pipeline::ray_tracing_pipeline::RayTracingPipeline;
use pipeline::ray_tracing_pipeline::RayTracingPipelineCreationError;
use pipeline::shader::EmptyEntryPointDummy;
use pipeline::shader::RayTracingEntryPointAbstract;
use pipeline::shader::SpecializationConstants;

use check_errors;
use vk;
use VulkanObject;

pub struct RayTracingPipelineGroupBuilder<Cs, Css, As, Ass, Is, Iss> {
    pub closest_hit_shader: Option<(Cs, Css)>,
    pub any_hit_shader: Option<(As, Ass)>,
    pub intersection_shader: Option<(Is, Iss)>,
}

impl
    RayTracingPipelineGroupBuilder<
        EmptyEntryPointDummy,
        (),
        EmptyEntryPointDummy,
        (),
        EmptyEntryPointDummy,
        (),
    >
{
    pub(super) fn new() -> Self {
        RayTracingPipelineGroupBuilder {
            closest_hit_shader: None,
            any_hit_shader: None,
            intersection_shader: None,
        }
    }
}

impl<Cs1, Css1, As1, Ass1, Is1, Iss1>
    RayTracingPipelineGroupBuilder<Cs1, Css1, As1, Ass1, Is1, Iss1>
where
    Cs1: RayTracingEntryPointAbstract,
    Css1: SpecializationConstants,
    Cs1::PipelineLayout: Clone + 'static + Send + Sync, // TODO: shouldn't be required
    As1: RayTracingEntryPointAbstract,
    Ass1: SpecializationConstants,
    As1::PipelineLayout: Clone + 'static + Send + Sync, // TODO: shouldn't be required
    Is1: RayTracingEntryPointAbstract,
    Iss1: SpecializationConstants,
    Is1::PipelineLayout: Clone + 'static + Send + Sync, // TODO: shouldn't be required
{
    /// Adds a closest hit shader to group.
    // TODO: correct specialization constants
    #[inline]
    pub fn closest_hit_shader<Cs2, Css2>(
        self,
        shader: Cs2,
        specialization_constants: Css2,
    ) -> RayTracingPipelineGroupBuilder<Cs2, Css2, As1, Ass1, Is1, Iss1>
    where
        Cs2: RayTracingEntryPointAbstract<SpecializationConstants = Css2>,
        Css2: SpecializationConstants,
    {
        RayTracingPipelineGroupBuilder {
            closest_hit_shader: Some((shader, specialization_constants)),
            any_hit_shader: self.any_hit_shader,
            intersection_shader: self.intersection_shader,
        }
    }

    /// Adds a any hit shader to group.
    // TODO: correct specialization constants
    #[inline]
    pub fn any_hit_shader<As2, Ass2>(
        self,
        shader: As2,
        specialization_constants: Ass2,
    ) -> RayTracingPipelineGroupBuilder<Cs1, Css1, As2, Ass2, Is1, Iss1>
    where
        As2: RayTracingEntryPointAbstract<SpecializationConstants = Ass2>,
        Ass2: SpecializationConstants,
    {
        RayTracingPipelineGroupBuilder {
            closest_hit_shader: self.closest_hit_shader,
            any_hit_shader: Some((shader, specialization_constants)),
            intersection_shader: self.intersection_shader,
        }
    }

    /// Adds an intersection shader to group.
    // TODO: correct specialization constants
    #[inline]
    pub fn intersection_shader<Is2, Iss2>(
        self,
        shader: Is2,
        specialization_constants: Iss2,
    ) -> RayTracingPipelineGroupBuilder<Cs1, Css1, As1, Ass1, Is2, Iss2>
    where
        Is2: RayTracingEntryPointAbstract<SpecializationConstants = Iss2>,
        Iss2: SpecializationConstants,
    {
        RayTracingPipelineGroupBuilder {
            closest_hit_shader: self.closest_hit_shader,
            any_hit_shader: self.any_hit_shader,
            intersection_shader: Some((shader, specialization_constants)),
        }
    }
}

/// Prototype for a `RayTracingPipeline`.
// TODO: we can optimize this by filling directly the raw vk structs
pub struct RayTracingPipelineBuilder<Rs, Rss, Ms, Mss, Cs, Css, As, Ass, Is, Iss> {
    nv_extension: bool,
    // TODO: Should be a list
    raygen_shader: Option<(Rs, Rss)>,
    // TODO: Should be a list
    miss_shader: Option<(Ms, Mss)>,
    // TODO: Should be a list
    group: RayTracingPipelineGroupBuilder<Cs, Css, As, Ass, Is, Iss>,
    max_recursion_depth: u32,
}

impl
    RayTracingPipelineBuilder<
        EmptyEntryPointDummy,
        (),
        EmptyEntryPointDummy,
        (),
        EmptyEntryPointDummy,
        (),
        EmptyEntryPointDummy,
        (),
        EmptyEntryPointDummy,
        (),
    >
{
    /// Builds a new empty builder using the `nv_ray_tracing` extension.
    pub(super) fn nv(max_recursion_depth: u32) -> Self {
        RayTracingPipelineBuilder {
            nv_extension: true,
            raygen_shader: None,
            miss_shader: None,
            group: RayTracingPipelineGroupBuilder {
                closest_hit_shader: None,
                any_hit_shader: None,
                intersection_shader: None,
            },
            max_recursion_depth,
        }
    }

    /// Builds a new empty builder using the `khr_ray_tracing` extension.
    pub(super) fn khr(max_recursion_depth: u32) -> Self {
        RayTracingPipelineBuilder {
            nv_extension: false,
            raygen_shader: None,
            miss_shader: None,
            group: RayTracingPipelineGroupBuilder {
                closest_hit_shader: None,
                any_hit_shader: None,
                intersection_shader: None,
            },
            max_recursion_depth,
        }
    }
}

impl<Rs, Rss, Ms, Mss, Cs, Css, As, Ass, Is, Iss>
    RayTracingPipelineBuilder<Rs, Rss, Ms, Mss, Cs, Css, As, Ass, Is, Iss>
where
    Rs: RayTracingEntryPointAbstract,
    Rss: SpecializationConstants,
    Rs::PipelineLayout: Clone + 'static + Send + Sync, // TODO: shouldn't be required
    Ms: RayTracingEntryPointAbstract,
    Mss: SpecializationConstants,
    Ms::PipelineLayout: Clone + 'static + Send + Sync, // TODO: shouldn't be required
    Cs: RayTracingEntryPointAbstract,
    Css: SpecializationConstants,
    Cs::PipelineLayout: Clone + 'static + Send + Sync, // TODO: shouldn't be required
    As: RayTracingEntryPointAbstract,
    Ass: SpecializationConstants,
    As::PipelineLayout: Clone + 'static + Send + Sync, // TODO: shouldn't be required
    Is: RayTracingEntryPointAbstract,
    Iss: SpecializationConstants,
    Is::PipelineLayout: Clone + 'static + Send + Sync, // TODO: shouldn't be required
{
    /// Builds the ray tracing pipeline, using an inferred a pipeline layout.
    // TODO: replace Box<PipelineLayoutAbstract> with a PipelineUnion struct without template params
    pub fn build(
        self,
        device: Arc<Device>,
    ) -> Result<
        RayTracingPipeline<Box<dyn PipelineLayoutAbstract + Send + Sync>>,
        RayTracingPipelineCreationError,
    > {
        self.with_auto_layout(device, &[])
    }

    /// Builds the ray tracing pipeline using the `nv_ray_tracing` extension.
    ///
    /// Does the same as `build`, except that `build` automatically builds the pipeline layout
    /// object corresponding to the union of your shaders while this function allows you to specify
    /// the pipeline layout.
    pub fn with_pipeline_layout_nv<L>(
        self,
        device: Arc<Device>,
        pipeline_layout: L,
    ) -> Result<RayTracingPipeline<L>, RayTracingPipelineCreationError>
    where
        L: PipelineLayoutAbstract,
    {
        self.with_pipeline_layout(device, pipeline_layout)
    }

    fn with_auto_layout(
        self,
        device: Arc<Device>,
        dynamic_buffers: &[(usize, usize)],
    ) -> Result<
        RayTracingPipeline<Box<dyn PipelineLayoutAbstract + Send + Sync>>,
        RayTracingPipelineCreationError,
    > {
        let pipeline_layout;

        // Must be at least one stage with raygen
        if let Some((ref shader, _)) = self.raygen_shader {
            let layout = shader.layout().clone();
            if let Some((ref shader, _)) = self.miss_shader {
                let layout = layout.union(shader.layout().clone());
                if let Some((ref shader, _)) = self.group.closest_hit_shader {
                    let layout = layout.union(shader.layout().clone());
                    if let Some((ref shader, _)) = self.group.intersection_shader {
                        let layout = layout.union(shader.layout().clone());
                        if let Some((ref shader, _)) = self.group.any_hit_shader {
                            let layout = layout.union(shader.layout().clone());
                            pipeline_layout = Box::new(
                                PipelineLayoutDescTweaks::new(
                                    layout,
                                    dynamic_buffers.into_iter().cloned(),
                                )
                                .build(device.clone())
                                .unwrap(),
                            ) as Box<_>; // TODO: error
                        } else {
                            pipeline_layout = Box::new(
                                PipelineLayoutDescTweaks::new(
                                    layout,
                                    dynamic_buffers.into_iter().cloned(),
                                )
                                .build(device.clone())
                                .unwrap(),
                            ) as Box<_>; // TODO: error
                        }
                    } else {
                        if let Some((ref shader, _)) = self.group.any_hit_shader {
                            let layout = layout.union(shader.layout().clone());
                            pipeline_layout = Box::new(
                                PipelineLayoutDescTweaks::new(
                                    layout,
                                    dynamic_buffers.into_iter().cloned(),
                                )
                                .build(device.clone())
                                .unwrap(),
                            ) as Box<_>; // TODO: error
                        } else {
                            pipeline_layout = Box::new(
                                PipelineLayoutDescTweaks::new(
                                    layout,
                                    dynamic_buffers.into_iter().cloned(),
                                )
                                .build(device.clone())
                                .unwrap(),
                            ) as Box<_>; // TODO: error
                        }
                    }
                } else {
                    if let Some((ref shader, _)) = self.group.intersection_shader {
                        let layout = layout.union(shader.layout().clone());
                        if let Some((ref shader, _)) = self.group.any_hit_shader {
                            let layout = layout.union(shader.layout().clone());
                            pipeline_layout = Box::new(
                                PipelineLayoutDescTweaks::new(
                                    layout,
                                    dynamic_buffers.into_iter().cloned(),
                                )
                                .build(device.clone())
                                .unwrap(),
                            ) as Box<_>; // TODO: error
                        } else {
                            pipeline_layout = Box::new(
                                PipelineLayoutDescTweaks::new(
                                    layout,
                                    dynamic_buffers.into_iter().cloned(),
                                )
                                .build(device.clone())
                                .unwrap(),
                            ) as Box<_>; // TODO: error
                        }
                    } else {
                        if let Some((ref shader, _)) = self.group.any_hit_shader {
                            let layout = layout.union(shader.layout().clone());
                            pipeline_layout = Box::new(
                                PipelineLayoutDescTweaks::new(
                                    layout,
                                    dynamic_buffers.into_iter().cloned(),
                                )
                                .build(device.clone())
                                .unwrap(),
                            ) as Box<_>; // TODO: error
                        } else {
                            pipeline_layout = Box::new(
                                PipelineLayoutDescTweaks::new(
                                    layout,
                                    dynamic_buffers.into_iter().cloned(),
                                )
                                .build(device.clone())
                                .unwrap(),
                            ) as Box<_>; // TODO: error
                        }
                    }
                }
            } else {
                if let Some((ref shader, _)) = self.group.closest_hit_shader {
                    let layout = layout.union(shader.layout().clone());
                    if let Some((ref shader, _)) = self.group.intersection_shader {
                        let layout = layout.union(shader.layout().clone());
                        if let Some((ref shader, _)) = self.group.any_hit_shader {
                            let layout = layout.union(shader.layout().clone());
                            pipeline_layout = Box::new(
                                PipelineLayoutDescTweaks::new(
                                    layout,
                                    dynamic_buffers.into_iter().cloned(),
                                )
                                .build(device.clone())
                                .unwrap(),
                            ) as Box<_>; // TODO: error
                        } else {
                            pipeline_layout = Box::new(
                                PipelineLayoutDescTweaks::new(
                                    layout,
                                    dynamic_buffers.into_iter().cloned(),
                                )
                                .build(device.clone())
                                .unwrap(),
                            ) as Box<_>; // TODO: error
                        }
                    } else {
                        if let Some((ref shader, _)) = self.group.any_hit_shader {
                            let layout = layout.union(shader.layout().clone());
                            pipeline_layout = Box::new(
                                PipelineLayoutDescTweaks::new(
                                    layout,
                                    dynamic_buffers.into_iter().cloned(),
                                )
                                .build(device.clone())
                                .unwrap(),
                            ) as Box<_>; // TODO: error
                        } else {
                            pipeline_layout = Box::new(
                                PipelineLayoutDescTweaks::new(
                                    layout,
                                    dynamic_buffers.into_iter().cloned(),
                                )
                                .build(device.clone())
                                .unwrap(),
                            ) as Box<_>; // TODO: error
                        }
                    }
                } else {
                    if let Some((ref shader, _)) = self.group.intersection_shader {
                        let layout = layout.union(shader.layout().clone());
                        if let Some((ref shader, _)) = self.group.any_hit_shader {
                            let layout = layout.union(shader.layout().clone());
                            pipeline_layout = Box::new(
                                PipelineLayoutDescTweaks::new(
                                    layout,
                                    dynamic_buffers.into_iter().cloned(),
                                )
                                .build(device.clone())
                                .unwrap(),
                            ) as Box<_>; // TODO: error
                        } else {
                            pipeline_layout = Box::new(
                                PipelineLayoutDescTweaks::new(
                                    layout,
                                    dynamic_buffers.into_iter().cloned(),
                                )
                                .build(device.clone())
                                .unwrap(),
                            ) as Box<_>; // TODO: error
                        }
                    } else {
                        if let Some((ref shader, _)) = self.group.any_hit_shader {
                            let layout = layout.union(shader.layout().clone());
                            pipeline_layout = Box::new(
                                PipelineLayoutDescTweaks::new(
                                    layout,
                                    dynamic_buffers.into_iter().cloned(),
                                )
                                .build(device.clone())
                                .unwrap(),
                            ) as Box<_>; // TODO: error
                        } else {
                            pipeline_layout = Box::new(
                                PipelineLayoutDescTweaks::new(
                                    layout,
                                    dynamic_buffers.into_iter().cloned(),
                                )
                                .build(device.clone())
                                .unwrap(),
                            ) as Box<_>; // TODO: error
                        }
                    }
                }
            }
        } else {
            return Err(RayTracingPipelineCreationError::NoRaygenShader);
        }

        if self.max_recursion_depth > device.physical_device().max_recursion_depth() {
            return Err(RayTracingPipelineCreationError::MaxRecursionDepthExceeded {
                max: device.physical_device().max_recursion_depth() as usize,
                obtained: self.max_recursion_depth as usize,
            });
        }

        // TODO: Callable, Miss and Raygen are alone in a General Shader Group
        // TODO: Intersection shader is required in procedural group and disallowed in triangle group
        // TODO: Anyhit and closest hit are optional in procedural group and disallowed in triangle group
        // TODO: groupCount must be greater than 0
        // TODO: layout must be consistent with all shaders specified in pStages
        // TODO: The number of resources in layout accessible to each shader stage that is used by
        //       the pipeline must be less than or equal to VkPhysicalDeviceLimits::maxPerStageResources
        // TODO: If VkPhysicalDeviceRayTracingFeaturesKHR::rayTracingShaderGroupHandleCaptureReplayMixed
        //       is VK_FALSE then pShaderGroupCaptureReplayHandle must not be provided if it has not been
        //       provided on a previous call to ray tracing pipeline creation
        // TODO: If VkPhysicalDeviceRayTracingFeaturesKHR::rayTracingShaderGroupHandleCaptureReplayMixed
        //       is VK_FALSE then the caller must guarantee that no ray tracing pipeline creation commands
        //       with pShaderGroupCaptureReplayHandle provided execute simultaneously with ray tracing
        //       pipeline creation commands without pShaderGroupCaptureReplayHandle provided
        self.with_pipeline_layout(device, pipeline_layout)
    }

    fn with_pipeline_layout<L>(
        self,
        device: Arc<Device>,
        pipeline_layout: L,
    ) -> Result<RayTracingPipeline<L>, RayTracingPipelineCreationError>
    where
        L: PipelineLayoutAbstract,
    {
        let vk = device.pointers();

        // Creating the specialization constants of the various stages.
        let raygen_stage = {
            let raygen_shader_specialization = {
                let spec_descriptors = Rss::descriptors();
                let constants = &self.raygen_shader.as_ref().unwrap().1;
                vk::SpecializationInfo {
                    mapEntryCount: spec_descriptors.len() as u32,
                    pMapEntries: spec_descriptors.as_ptr() as *const _,
                    dataSize: mem::size_of_val(constants),
                    pData: constants as *const Rss as *const _,
                }
            };
            vk::PipelineShaderStageCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0, // reserved
                stage: vk::SHADER_STAGE_RAYGEN_BIT_KHR,
                module: self
                    .raygen_shader
                    .as_ref()
                    .unwrap()
                    .0
                    .module()
                    .internal_object(),
                pName: self.raygen_shader.as_ref().unwrap().0.name().as_ptr(),
                pSpecializationInfo: &raygen_shader_specialization as *const _,
            }
        };
        let miss_stage = match self.miss_shader {
            Some((shader, constants)) => {
                let specialization = {
                    let spec_descriptors = Mss::descriptors();
                    vk::SpecializationInfo {
                        mapEntryCount: spec_descriptors.len() as u32,
                        pMapEntries: spec_descriptors.as_ptr() as *const _,
                        dataSize: mem::size_of_val(&constants),
                        pData: &constants as *const Mss as *const _,
                    }
                };
                Some(vk::PipelineShaderStageCreateInfo {
                    sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    pNext: ptr::null(),
                    flags: 0, // reserved
                    stage: vk::SHADER_STAGE_MISS_BIT_KHR,
                    module: shader.module().internal_object(),
                    pName: shader.name().as_ptr(),
                    pSpecializationInfo: &specialization as *const _,
                })
            }
            None => None,
        };
        let closest_hit_stage = match self.group.closest_hit_shader {
            Some((shader, constants)) => {
                let specialization = {
                    let spec_descriptors = Css::descriptors();
                    vk::SpecializationInfo {
                        mapEntryCount: spec_descriptors.len() as u32,
                        pMapEntries: spec_descriptors.as_ptr() as *const _,
                        dataSize: mem::size_of_val(&constants),
                        pData: &constants as *const Css as *const _,
                    }
                };
                Some(vk::PipelineShaderStageCreateInfo {
                    sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    pNext: ptr::null(),
                    flags: 0, // reserved
                    stage: vk::SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                    module: shader.module().internal_object(),
                    pName: shader.name().as_ptr(),
                    pSpecializationInfo: &specialization as *const _,
                })
            }
            None => None,
        };
        let any_hit_stage = match self.group.any_hit_shader {
            Some((shader, constants)) => {
                let specialization = {
                    let spec_descriptors = Ass::descriptors();
                    vk::SpecializationInfo {
                        mapEntryCount: spec_descriptors.len() as u32,
                        pMapEntries: spec_descriptors.as_ptr() as *const _,
                        dataSize: mem::size_of_val(&constants),
                        pData: &constants as *const Ass as *const _,
                    }
                };
                Some(vk::PipelineShaderStageCreateInfo {
                    sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    pNext: ptr::null(),
                    flags: 0, // reserved
                    stage: vk::SHADER_STAGE_ANY_HIT_BIT_KHR,
                    module: shader.module().internal_object(),
                    pName: shader.name().as_ptr(),
                    pSpecializationInfo: &specialization as *const _,
                })
            }
            None => None,
        };
        let intersection_stage = match self.group.intersection_shader {
            Some((shader, constants)) => {
                let specialization = {
                    let spec_descriptors = Iss::descriptors();
                    vk::SpecializationInfo {
                        mapEntryCount: spec_descriptors.len() as u32,
                        pMapEntries: spec_descriptors.as_ptr() as *const _,
                        dataSize: mem::size_of_val(&constants),
                        pData: &constants as *const Iss as *const _,
                    }
                };
                Some(vk::PipelineShaderStageCreateInfo {
                    sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    pNext: ptr::null(),
                    flags: 0, // reserved
                    stage: vk::SHADER_STAGE_INTERSECTION_BIT_KHR,
                    module: shader.module().internal_object(),
                    pName: shader.name().as_ptr(),
                    pSpecializationInfo: &specialization as *const _,
                })
            }
            None => None,
        };

        let (pipeline, group_count) = if self.nv_extension {
            let mut stages = SmallVec::<[_; 5]>::new();
            let mut groups = SmallVec::<[_; 5]>::new();

            // Raygen
            groups.push(vk::RayTracingShaderGroupCreateInfoNV {
                sType: vk::STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV,
                pNext: ptr::null(),
                type_: vk::RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV,
                generalShader: stages.len() as u32,
                closestHitShader: vk::SHADER_UNUSED,
                anyHitShader: vk::SHADER_UNUSED,
                intersectionShader: vk::SHADER_UNUSED,
            });
            stages.push(raygen_stage);

            // Miss
            match miss_stage {
                Some(miss_stage) => {
                    groups.push(vk::RayTracingShaderGroupCreateInfoNV {
                        sType: vk::STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV,
                        pNext: ptr::null(),
                        type_: vk::RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV,
                        generalShader: stages.len() as u32,
                        closestHitShader: vk::SHADER_UNUSED,
                        anyHitShader: vk::SHADER_UNUSED,
                        intersectionShader: vk::SHADER_UNUSED,
                    });
                    stages.push(miss_stage);
                }
                None => {}
            }

            // Groups
            if closest_hit_stage.is_some()
                || any_hit_stage.is_some()
                || intersection_stage.is_some()
            {
                let mut info = vk::RayTracingShaderGroupCreateInfoNV {
                    sType: vk::STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV,
                    pNext: ptr::null(),
                    type_: vk::RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_NV,
                    generalShader: vk::SHADER_UNUSED,
                    closestHitShader: vk::SHADER_UNUSED,
                    anyHitShader: vk::SHADER_UNUSED,
                    intersectionShader: vk::SHADER_UNUSED,
                };

                match closest_hit_stage {
                    Some(stage) => {
                        info.closestHitShader = stages.len() as u32;
                        stages.push(stage);
                    }
                    None => {}
                }

                match any_hit_stage {
                    Some(stage) => {
                        info.anyHitShader = stages.len() as u32;
                        stages.push(stage);
                    }
                    None => {}
                }

                match intersection_stage {
                    Some(stage) => {
                        // If there is an intersection stage, then the type is procedural, not triangles
                        info.type_ = vk::RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_NV;
                        info.intersectionShader = stages.len() as u32;
                        stages.push(stage);
                    }
                    None => {}
                }

                groups.push(info);
            }

            unsafe {
                let infos = vk::RayTracingPipelineCreateInfoNV {
                    sType: vk::STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_NV,
                    pNext: ptr::null(),
                    flags: 0,
                    stageCount: stages.len() as u32,
                    pStages: stages.as_ptr(),
                    groupCount: groups.len() as u32,
                    pGroups: groups.as_ptr(),
                    layout: PipelineLayoutAbstract::sys(&pipeline_layout).internal_object(),
                    maxRecursionDepth: self.max_recursion_depth,
                    basePipelineHandle: 0, // TODO:
                    basePipelineIndex: -1, // TODO:
                };

                let mut output = MaybeUninit::uninit();
                check_errors(vk.CreateRayTracingPipelinesNV(
                    device.internal_object(),
                    0,
                    1,
                    &infos,
                    ptr::null(),
                    output.as_mut_ptr(),
                ))?;
                (output.assume_init(), groups.len() as u32)
            }
        } else {
            let mut stages = SmallVec::<[_; 5]>::new();
            let mut groups = SmallVec::<[_; 5]>::new();

            // Raygen
            groups.push(vk::RayTracingShaderGroupCreateInfoKHR {
                sType: vk::STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                pNext: ptr::null(),
                type_: vk::RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                generalShader: stages.len() as u32,
                closestHitShader: vk::SHADER_UNUSED,
                anyHitShader: vk::SHADER_UNUSED,
                intersectionShader: vk::SHADER_UNUSED,
                pShaderGroupCaptureReplayHandle: ptr::null(), // TODO:
            });
            stages.push(raygen_stage);

            // Miss
            match miss_stage {
                Some(miss_stage) => {
                    groups.push(vk::RayTracingShaderGroupCreateInfoKHR {
                        sType: vk::STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                        pNext: ptr::null(),
                        type_: vk::RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                        generalShader: stages.len() as u32,
                        closestHitShader: vk::SHADER_UNUSED,
                        anyHitShader: vk::SHADER_UNUSED,
                        intersectionShader: vk::SHADER_UNUSED,
                        pShaderGroupCaptureReplayHandle: ptr::null(), // TODO:
                    });
                    stages.push(miss_stage);
                }
                None => {}
            }

            // Groups
            if closest_hit_stage.is_some()
                || any_hit_stage.is_some()
                || intersection_stage.is_some()
            {
                let mut info = vk::RayTracingShaderGroupCreateInfoKHR {
                    sType: vk::STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                    pNext: ptr::null(),
                    type_: vk::RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                    generalShader: vk::SHADER_UNUSED,
                    closestHitShader: vk::SHADER_UNUSED,
                    anyHitShader: vk::SHADER_UNUSED,
                    intersectionShader: vk::SHADER_UNUSED,
                    pShaderGroupCaptureReplayHandle: ptr::null(), // TODO:
                };

                match closest_hit_stage {
                    Some(stage) => {
                        info.closestHitShader = stages.len() as u32;
                        stages.push(stage);
                    }
                    None => {}
                }

                match any_hit_stage {
                    Some(stage) => {
                        info.anyHitShader = stages.len() as u32;
                        stages.push(stage);
                    }
                    None => {}
                }

                match intersection_stage {
                    Some(stage) => {
                        // If there is an intersection stage, then the type is procedural, not triangles
                        info.type_ = vk::RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
                        info.intersectionShader = stages.len() as u32;
                        stages.push(stage);
                    }
                    None => {}
                }

                groups.push(info);
            }

            let library_info = vk::PipelineLibraryCreateInfoKHR {
                sType: vk::STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
                pNext: ptr::null(),
                libraryCount: 0,         // TODO:
                pLibraries: ptr::null(), // TODO:
            };

            unsafe {
                let infos = vk::RayTracingPipelineCreateInfoKHR {
                    sType: vk::STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
                    pNext: ptr::null(),
                    flags: 0,
                    stageCount: stages.len() as u32,
                    pStages: stages.as_ptr(),
                    groupCount: groups.len() as u32,
                    pGroups: groups.as_ptr(),
                    maxRecursionDepth: self.max_recursion_depth,
                    libraries: library_info,
                    pLibraryInterface: ptr::null(), // TODO:
                    layout: PipelineLayoutAbstract::sys(&pipeline_layout).internal_object(),
                    basePipelineHandle: 0, // TODO:
                    basePipelineIndex: -1, // TODO:
                };

                let mut output = MaybeUninit::uninit();
                check_errors(vk.CreateRayTracingPipelinesKHR(
                    device.internal_object(),
                    0,
                    1,
                    &infos,
                    ptr::null(),
                    output.as_mut_ptr(),
                ))?;
                (output.assume_init(), groups.len() as u32)
            }
        };

        // Some drivers return `VK_SUCCESS` but provide a null handle if they
        // fail to create the pipeline (due to invalid shaders, etc)
        // This check ensures that we don't create an invalid `RayTracingPipeline` instance
        if pipeline == vk::NULL_HANDLE {
            panic!("vkCreateRayTracingPipelines provided a NULL handle");
        }

        Ok(RayTracingPipeline {
            inner: RayTracingPipelineInner {
                device: device.clone(),
                pipeline,
            },
            layout: pipeline_layout,
            group_count,
            max_recursion_depth: self.max_recursion_depth,
            nv_extension: self.nv_extension,
        })
    }

    // TODO: add build_with_cache method
}

impl<Rs1, Rss1, Ms1, Mss1, Cs1, Css1, As1, Ass1, Is1, Iss1>
    RayTracingPipelineBuilder<Rs1, Rss1, Ms1, Mss1, Cs1, Css1, As1, Ass1, Is1, Iss1>
{
    // TODO: add pipeline derivate system

    /// Adds a raygen shader group to use.
    // TODO: raygen_shader should be a list
    // TODO: correct specialization constants
    #[inline]
    pub fn raygen_shader<Rs2, Rss2>(
        self,
        shader: Rs2,
        specialization_constants: Rss2,
    ) -> RayTracingPipelineBuilder<Rs2, Rss2, Ms1, Mss1, Cs1, Css1, As1, Ass1, Is1, Iss1>
    where
        Rs2: RayTracingEntryPointAbstract<SpecializationConstants = Rss2>,
        Rss2: SpecializationConstants,
    {
        RayTracingPipelineBuilder {
            nv_extension: self.nv_extension,
            max_recursion_depth: self.max_recursion_depth,
            raygen_shader: Some((shader, specialization_constants)),
            miss_shader: self.miss_shader,
            group: self.group,
        }
    }

    /// Adds a miss shader group to use.
    // TODO: miss_shader should be a list
    // TODO: correct specialization constants
    #[inline]
    pub fn miss_shader<Ms2, Mss2>(
        self, shader: Ms2, specialization_constants: Mss2,
    ) -> RayTracingPipelineBuilder<Rs1, Rss1, Ms2, Mss2, Cs1, Css1, As1, Ass1, Is1, Iss1>
    where
        Ms2: RayTracingEntryPointAbstract<SpecializationConstants = Mss2>,
        Mss2: SpecializationConstants,
    {
        RayTracingPipelineBuilder {
            nv_extension: self.nv_extension,
            max_recursion_depth: self.max_recursion_depth,
            raygen_shader: self.raygen_shader,
            miss_shader: Some((shader, specialization_constants)),
            group: self.group,
        }
    }

    /// Add a ray tracing group
    #[inline]
    pub fn group<Cs2, Css2, As2, Ass2, Is2, Iss2>(
        self,
        group: RayTracingPipelineGroupBuilder<Cs2, Css2, As2, Ass2, Is2, Iss2>,
    ) -> RayTracingPipelineBuilder<Rs1, Rss1, Ms1, Mss1, Cs2, Css2, As2, Ass2, Is2, Iss2>
    where
        Cs2: RayTracingEntryPointAbstract<SpecializationConstants = Css2>,
        Css2: SpecializationConstants,
        As2: RayTracingEntryPointAbstract<SpecializationConstants = Ass2>,
        Ass2: SpecializationConstants,
        Is2: RayTracingEntryPointAbstract<SpecializationConstants = Iss2>,
        Iss2: SpecializationConstants,
    {
        RayTracingPipelineBuilder {
            nv_extension: self.nv_extension,
            max_recursion_depth: self.max_recursion_depth,
            raygen_shader: self.raygen_shader,
            miss_shader: self.miss_shader,
            group,
        }
    }
}

impl<Rs, Rss, Ms, Mss, Cs, Css, As, Ass, Is, Iss> Clone
    for RayTracingPipelineBuilder<Rs, Rss, Ms, Mss, Cs, Css, As, Ass, Is, Iss>
where
    Rs: Clone,
    Rss: Clone,
    Ms: Clone,
    Mss: Clone,
    Cs: Clone,
    Css: Clone,
    As: Clone,
    Ass: Clone,
    Is: Clone,
    Iss: Clone,
{
    fn clone(&self) -> Self {
        RayTracingPipelineBuilder {
            nv_extension: self.nv_extension,
            raygen_shader: self.raygen_shader.clone(),
            miss_shader: self.miss_shader.clone(),
            group: RayTracingPipelineGroupBuilder {
                closest_hit_shader: self.group.closest_hit_shader.clone(),
                any_hit_shader: self.group.any_hit_shader.clone(),
                intersection_shader: self.group.intersection_shader.clone(),
            },
            max_recursion_depth: self.max_recursion_depth,
        }
    }
}
