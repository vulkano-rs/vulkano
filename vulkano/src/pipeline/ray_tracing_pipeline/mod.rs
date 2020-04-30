// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;
use std::u32;

use check_errors;
use descriptor::descriptor::DescriptorDesc;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use descriptor::pipeline_layout::PipelineLayoutDesc;
use descriptor::pipeline_layout::PipelineLayoutDescPcRange;
use descriptor::pipeline_layout::PipelineLayoutSys;
use descriptor::PipelineLayoutAbstract;
use device::{Device, DeviceOwned, Queue};
use pipeline::shader::EmptyEntryPointDummy;
use vk;
use SafeDeref;
use VulkanObject;

pub use self::builder::RayTracingPipelineBuilder;
pub use self::creation_error::RayTracingPipelineCreationError;
use pipeline::ray_tracing_pipeline::builder::RayTracingPipelineGroupBuilder;

mod builder;
mod creation_error;
// FIXME: restore
//mod tests;

/// Defines how the implementation should perform a draw operation.
///
/// This object contains the shaders and the various fixed states that describe how the
/// implementation should perform the various operations needed by a draw command.
pub struct RayTracingPipeline<Layout> {
    inner: Inner,
    layout: Layout,

    group_count: u32,
    max_recursion_depth: u32,
    nv_extension: bool,
}

struct Inner {
    pipeline: vk::Pipeline,
    device: Arc<Device>,
}

impl RayTracingPipeline<()> {
    /// Starts the building process of a ray tracing pipeline using the
    /// `nv_ray_tracing` extension.
    /// Returns a builder object that you can fill with the various parameters.
    pub fn nv<'a>(
        max_recursion_depth: u32,
    ) -> RayTracingPipelineBuilder<
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
    > {
        RayTracingPipelineBuilder::nv(max_recursion_depth)
    }

    /// Starts the building process of a ray tracing pipeline using the
    /// `khr_ray_tracing` extension.
    /// Returns a builder object that you can fill with the various parameters.
    pub fn khr<'a>(
        max_recursion_depth: u32,
    ) -> RayTracingPipelineBuilder<
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
    > {
        RayTracingPipelineBuilder::khr(max_recursion_depth)
    }

    pub fn group<'a>() -> RayTracingPipelineGroupBuilder<
        EmptyEntryPointDummy,
        (),
        EmptyEntryPointDummy,
        (),
        EmptyEntryPointDummy,
        (),
    > {
        RayTracingPipelineGroupBuilder::new()
    }
}

impl<L> RayTracingPipeline<L> {
    #[inline]
    pub fn group_handles(&self, queue: Arc<Queue>) -> Vec<u8> {
        if self.nv_extension {
            self.group_handles_nv(queue)
        } else {
            self.group_handles_khr(queue)
        }
    }

    #[inline]
    fn group_handles_nv(&self, queue: Arc<Queue>) -> Vec<u8> {
        let binding_table_size = (self.group_count
            * self.device().physical_device().shader_group_handle_size())
            as usize;
        let mut shader_handle_storage = Vec::<u8>::with_capacity(binding_table_size.clone());
        shader_handle_storage.resize(binding_table_size.clone(), 0);

        debug_assert!(self.device().loaded_extensions().nv_ray_tracing);

        unsafe {
            let vk = self.device().pointers();
            check_errors(vk.GetRayTracingShaderGroupHandlesNV(
                self.device().internal_object(),
                self.inner.pipeline.clone(),
                0,
                self.group_count,
                binding_table_size,
                shader_handle_storage.as_mut_ptr() as *mut _,
            ))
            .unwrap();
        };
        shader_handle_storage
    }

    #[inline]
    fn group_handles_khr(&self, queue: Arc<Queue>) -> Vec<u8> {
        let binding_table_size = (self.group_count
            * self.device().physical_device().shader_group_handle_size())
            as usize;
        let mut shader_handle_storage = Vec::<u8>::with_capacity(binding_table_size.clone());
        shader_handle_storage.resize(binding_table_size.clone(), 0);

        debug_assert!(self.device().loaded_extensions().khr_ray_tracing);

        unsafe {
            let vk = self.device().pointers();
            check_errors(vk.GetRayTracingShaderGroupHandlesKHR(
                self.device().internal_object(),
                self.inner.pipeline.clone(),
                0,
                self.group_count,
                binding_table_size,
                shader_handle_storage.as_mut_ptr() as *mut _,
            ))
            .unwrap();
        };
        shader_handle_storage
    }
}

impl<L> fmt::Debug for RayTracingPipeline<L> {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "<Vulkan ray tracing pipeline {:?}>",
            self.inner.pipeline
        )
    }
}

impl<L> RayTracingPipeline<L> {
    /// Returns the `Device` this ray tracing pipeline was created with.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.inner.device
    }

    /// Returns the pipeline layout used in this ray tracing pipeline.
    #[inline]
    pub fn layout(&self) -> &L {
        &self.layout
    }

    /// Returns the if the pipeline is using `nv_ray_tracing`
    #[inline]
    pub fn use_nv_extension(&self) -> bool {
        self.nv_extension.clone()
    }
}

/// Trait implemented on all ray tracing pipelines.
pub unsafe trait RayTracingPipelineAbstract: PipelineLayoutAbstract {
    /// Returns an opaque object that represents the inside of the ray tracing pipeline.
    fn inner(&self) -> RayTracingPipelineSys;
    fn use_nv_extension(&self) -> bool;
}

unsafe impl<L> RayTracingPipelineAbstract for RayTracingPipeline<L>
where
    L: PipelineLayoutAbstract,
{
    #[inline]
    fn inner(&self) -> RayTracingPipelineSys {
        RayTracingPipelineSys(self.inner.pipeline, PhantomData)
    }

    #[inline]
    fn use_nv_extension(&self) -> bool {
        self.nv_extension
    }
}

unsafe impl<T> RayTracingPipelineAbstract for T
where
    T: SafeDeref,
    T::Target: RayTracingPipelineAbstract,
{
    #[inline]
    fn inner(&self) -> RayTracingPipelineSys {
        (**self).inner()
    }

    #[inline]
    fn use_nv_extension(&self) -> bool {
        (**self).use_nv_extension()
    }
}

/// Opaque object that represents the inside of the ray tracing pipeline.
#[derive(Debug, Copy, Clone)]
pub struct RayTracingPipelineSys<'a>(vk::Pipeline, PhantomData<&'a ()>);

unsafe impl<'a> VulkanObject for RayTracingPipelineSys<'a> {
    type Object = vk::Pipeline;

    const TYPE: vk::ObjectType = vk::OBJECT_TYPE_PIPELINE;

    #[inline]
    fn internal_object(&self) -> vk::Pipeline {
        self.0
    }
}

unsafe impl<L> PipelineLayoutAbstract for RayTracingPipeline<L>
where
    L: PipelineLayoutAbstract,
{
    #[inline]
    fn sys(&self) -> PipelineLayoutSys {
        self.layout.sys()
    }

    #[inline]
    fn descriptor_set_layout(&self, index: usize) -> Option<&Arc<UnsafeDescriptorSetLayout>> {
        self.layout.descriptor_set_layout(index)
    }
}

unsafe impl<L> PipelineLayoutDesc for RayTracingPipeline<L>
where
    L: PipelineLayoutDesc,
{
    #[inline]
    fn num_sets(&self) -> usize {
        self.layout.num_sets()
    }

    #[inline]
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        self.layout.num_bindings_in_set(set)
    }

    #[inline]
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        self.layout.descriptor(set, binding)
    }

    #[inline]
    fn num_push_constants_ranges(&self) -> usize {
        self.layout.num_push_constants_ranges()
    }

    #[inline]
    fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
        self.layout.push_constants_range(num)
    }
}

unsafe impl<L> DeviceOwned for RayTracingPipeline<L> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.inner.device
    }
}
