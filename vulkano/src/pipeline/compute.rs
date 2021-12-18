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

use crate::check_errors;
use crate::descriptor_set::layout::{DescriptorSetDesc, DescriptorSetLayout};
use crate::device::{Device, DeviceOwned};
use crate::pipeline::cache::PipelineCache;
use crate::pipeline::layout::{
    PipelineLayout, PipelineLayoutCreationError, PipelineLayoutSupersetError,
};
use crate::pipeline::{Pipeline, PipelineBindPoint};
use crate::shader::{DescriptorRequirements, EntryPoint, SpecializationConstants};
use crate::DeviceSize;
use crate::Error;
use crate::OomError;
use crate::VulkanObject;
use fnv::FnvHashMap;
use std::error;
use std::fmt;
use std::mem;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

/// A pipeline object that describes to the Vulkan implementation how it should perform compute
/// operations.
///
/// The template parameter contains the descriptor set to use with this pipeline.
///
/// Pass an optional `Arc` to a `PipelineCache` to enable pipeline caching. The vulkan
/// implementation will handle the `PipelineCache` and check if it is available.
/// Check the documentation of the `PipelineCache` for more information.
pub struct ComputePipeline {
    handle: ash::vk::Pipeline,
    device: Arc<Device>,
    layout: Arc<PipelineLayout>,
    descriptor_requirements: FnvHashMap<(u32, u32), DescriptorRequirements>,
    num_used_descriptor_sets: u32,
}

impl ComputePipeline {
    /// Builds a new `ComputePipeline`.
    ///
    /// `func` is a closure that is given a mutable reference to the inferred descriptor set
    /// definitions. This can be used to make changes to the layout before it's created, for example
    /// to add dynamic buffers or immutable samplers.
    pub fn new<Css, F>(
        device: Arc<Device>,
        shader: EntryPoint,
        specialization_constants: &Css,
        cache: Option<Arc<PipelineCache>>,
        func: F,
    ) -> Result<Arc<ComputePipeline>, ComputePipelineCreationError>
    where
        Css: SpecializationConstants,
        F: FnOnce(&mut [DescriptorSetDesc]),
    {
        let mut descriptor_set_layout_descs =
            DescriptorSetDesc::from_requirements(shader.descriptor_requirements());
        func(&mut descriptor_set_layout_descs);
        let descriptor_set_layouts = descriptor_set_layout_descs
            .iter()
            .map(|desc| Ok(DescriptorSetLayout::new(device.clone(), desc.clone())?))
            .collect::<Result<Vec<_>, PipelineLayoutCreationError>>()?;

        let layout = PipelineLayout::new(
            device.clone(),
            descriptor_set_layouts,
            shader.push_constant_requirements().cloned(),
        )?;

        unsafe {
            ComputePipeline::with_unchecked_pipeline_layout(
                device,
                shader,
                specialization_constants,
                layout,
                cache,
            )
        }
    }

    /// Builds a new `ComputePipeline` with a specific pipeline layout.
    ///
    /// An error will be returned if the pipeline layout isn't a superset of what the shader
    /// uses.
    pub fn with_pipeline_layout<Css>(
        device: Arc<Device>,
        shader: EntryPoint,
        specialization_constants: &Css,
        layout: Arc<PipelineLayout>,
        cache: Option<Arc<PipelineCache>>,
    ) -> Result<Arc<ComputePipeline>, ComputePipelineCreationError>
    where
        Css: SpecializationConstants,
    {
        let spec_descriptors = Css::descriptors();

        for (constant_id, reqs) in shader.specialization_constant_requirements() {
            let map_entry = spec_descriptors
                .iter()
                .find(|desc| desc.constant_id == constant_id)
                .ok_or(ComputePipelineCreationError::IncompatibleSpecializationConstants)?;

            if map_entry.size as DeviceSize != reqs.size {
                return Err(ComputePipelineCreationError::IncompatibleSpecializationConstants);
            }
        }

        layout.ensure_compatible_with_shader(
            shader.descriptor_requirements(),
            shader.push_constant_requirements(),
        )?;

        unsafe {
            ComputePipeline::with_unchecked_pipeline_layout(
                device,
                shader,
                specialization_constants,
                layout,
                cache,
            )
        }
    }

    /// Same as `with_pipeline_layout`, but doesn't check whether the pipeline layout is a
    /// superset of what the shader expects.
    pub unsafe fn with_unchecked_pipeline_layout<Css>(
        device: Arc<Device>,
        shader: EntryPoint,
        specialization_constants: &Css,
        layout: Arc<PipelineLayout>,
        cache: Option<Arc<PipelineCache>>,
    ) -> Result<Arc<ComputePipeline>, ComputePipelineCreationError>
    where
        Css: SpecializationConstants,
    {
        let fns = device.fns();

        let handle = {
            let spec_descriptors = Css::descriptors();
            let specialization = ash::vk::SpecializationInfo {
                map_entry_count: spec_descriptors.len() as u32,
                p_map_entries: spec_descriptors.as_ptr() as *const _,
                data_size: mem::size_of_val(specialization_constants),
                p_data: specialization_constants as *const Css as *const _,
            };

            let stage = ash::vk::PipelineShaderStageCreateInfo {
                flags: ash::vk::PipelineShaderStageCreateFlags::empty(),
                stage: ash::vk::ShaderStageFlags::COMPUTE,
                module: shader.module().internal_object(),
                p_name: shader.name().as_ptr(),
                p_specialization_info: if specialization.data_size == 0 {
                    ptr::null()
                } else {
                    &specialization
                },
                ..Default::default()
            };

            let infos = ash::vk::ComputePipelineCreateInfo {
                flags: ash::vk::PipelineCreateFlags::empty(),
                stage,
                layout: layout.internal_object(),
                base_pipeline_handle: ash::vk::Pipeline::null(),
                base_pipeline_index: 0,
                ..Default::default()
            };

            let cache_handle = match cache {
                Some(ref cache) => cache.internal_object(),
                None => ash::vk::PipelineCache::null(),
            };

            let mut output = MaybeUninit::uninit();
            check_errors(fns.v1_0.create_compute_pipelines(
                device.internal_object(),
                cache_handle,
                1,
                &infos,
                ptr::null(),
                output.as_mut_ptr(),
            ))?;
            output.assume_init()
        };

        let descriptor_requirements: FnvHashMap<_, _> = shader
            .descriptor_requirements()
            .map(|(loc, reqs)| (loc, reqs.clone()))
            .collect();
        let num_used_descriptor_sets = descriptor_requirements
            .keys()
            .map(|loc| loc.0)
            .max()
            .map(|x| x + 1)
            .unwrap_or(0);

        Ok(Arc::new(ComputePipeline {
            handle,
            device: device.clone(),
            layout,
            descriptor_requirements,
            num_used_descriptor_sets,
        }))
    }

    /// Returns the `Device` this compute pipeline was created with.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns an iterator over the descriptor requirements for this pipeline.
    #[inline]
    pub fn descriptor_requirements(
        &self,
    ) -> impl ExactSizeIterator<Item = ((u32, u32), &DescriptorRequirements)> {
        self.descriptor_requirements
            .iter()
            .map(|(loc, reqs)| (*loc, reqs))
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
}

impl fmt::Debug for ComputePipeline {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan compute pipeline {:?}>", self.handle)
    }
}

impl PartialEq for ComputePipeline {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.internal_object() == other.internal_object()
    }
}

impl Eq for ComputePipeline {}

unsafe impl VulkanObject for ComputePipeline {
    type Object = ash::vk::Pipeline;

    #[inline]
    fn internal_object(&self) -> ash::vk::Pipeline {
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
            fns.v1_0
                .destroy_pipeline(self.device.internal_object(), self.handle, ptr::null());
        }
    }
}

/// Error that can happen when creating a compute pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputePipelineCreationError {
    /// Not enough memory.
    OomError(OomError),
    /// Error while creating the pipeline layout object.
    PipelineLayoutCreationError(PipelineLayoutCreationError),
    /// The pipeline layout is not compatible with what the shader expects.
    IncompatiblePipelineLayout(PipelineLayoutSupersetError),
    /// The provided specialization constants are not compatible with what the shader expects.
    IncompatibleSpecializationConstants,
}

impl error::Error for ComputePipelineCreationError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            ComputePipelineCreationError::OomError(ref err) => Some(err),
            ComputePipelineCreationError::PipelineLayoutCreationError(ref err) => Some(err),
            ComputePipelineCreationError::IncompatiblePipelineLayout(ref err) => Some(err),
            ComputePipelineCreationError::IncompatibleSpecializationConstants => None,
        }
    }
}

impl fmt::Display for ComputePipelineCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                ComputePipelineCreationError::OomError(_) => "not enough memory available",
                ComputePipelineCreationError::PipelineLayoutCreationError(_) => {
                    "error while creating the pipeline layout object"
                }
                ComputePipelineCreationError::IncompatiblePipelineLayout(_) => {
                    "the pipeline layout is not compatible with what the shader expects"
                }
                ComputePipelineCreationError::IncompatibleSpecializationConstants => {
                    "the provided specialization constants are not compatible with what the shader expects"
                }
            }
        )
    }
}

impl From<OomError> for ComputePipelineCreationError {
    #[inline]
    fn from(err: OomError) -> ComputePipelineCreationError {
        ComputePipelineCreationError::OomError(err)
    }
}

impl From<PipelineLayoutCreationError> for ComputePipelineCreationError {
    #[inline]
    fn from(err: PipelineLayoutCreationError) -> ComputePipelineCreationError {
        ComputePipelineCreationError::PipelineLayoutCreationError(err)
    }
}

impl From<PipelineLayoutSupersetError> for ComputePipelineCreationError {
    #[inline]
    fn from(err: PipelineLayoutSupersetError) -> ComputePipelineCreationError {
        ComputePipelineCreationError::IncompatiblePipelineLayout(err)
    }
}

impl From<Error> for ComputePipelineCreationError {
    #[inline]
    fn from(err: Error) -> ComputePipelineCreationError {
        match err {
            err @ Error::OutOfHostMemory => {
                ComputePipelineCreationError::OomError(OomError::from(err))
            }
            err @ Error::OutOfDeviceMemory => {
                ComputePipelineCreationError::OomError(OomError::from(err))
            }
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::buffer::BufferUsage;
    use crate::buffer::CpuAccessibleBuffer;
    use crate::command_buffer::AutoCommandBufferBuilder;
    use crate::command_buffer::CommandBufferUsage;
    use crate::descriptor_set::PersistentDescriptorSet;
    use crate::descriptor_set::WriteDescriptorSet;
    use crate::pipeline::ComputePipeline;
    use crate::pipeline::Pipeline;
    use crate::pipeline::PipelineBindPoint;
    use crate::shader::ShaderModule;
    use crate::shader::SpecializationConstants;
    use crate::shader::SpecializationMapEntry;
    use crate::sync::now;
    use crate::sync::GpuFuture;

    // TODO: test for basic creation
    // TODO: test for pipeline layout error

    #[test]
    fn specialization_constants() {
        // This test checks whether specialization constants work.
        // It executes a single compute shader (one invocation) that writes the value of a spec.
        // constant to a buffer. The buffer content is then checked for the right value.

        let (device, queue) = gfx_dev_and_queue!();

        let module = unsafe {
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
            const MODULE: [u8; 480] = [
                3, 2, 35, 7, 0, 0, 1, 0, 1, 0, 8, 0, 14, 0, 0, 0, 0, 0, 0, 0, 17, 0, 2, 0, 1, 0, 0,
                0, 11, 0, 6, 0, 1, 0, 0, 0, 71, 76, 83, 76, 46, 115, 116, 100, 46, 52, 53, 48, 0,
                0, 0, 0, 14, 0, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 15, 0, 5, 0, 5, 0, 0, 0, 4, 0, 0, 0,
                109, 97, 105, 110, 0, 0, 0, 0, 16, 0, 6, 0, 4, 0, 0, 0, 17, 0, 0, 0, 1, 0, 0, 0, 1,
                0, 0, 0, 1, 0, 0, 0, 3, 0, 3, 0, 2, 0, 0, 0, 194, 1, 0, 0, 5, 0, 4, 0, 4, 0, 0, 0,
                109, 97, 105, 110, 0, 0, 0, 0, 5, 0, 4, 0, 7, 0, 0, 0, 79, 117, 116, 112, 117, 116,
                0, 0, 6, 0, 5, 0, 7, 0, 0, 0, 0, 0, 0, 0, 119, 114, 105, 116, 101, 0, 0, 0, 5, 0,
                4, 0, 9, 0, 0, 0, 119, 114, 105, 116, 101, 0, 0, 0, 5, 0, 4, 0, 11, 0, 0, 0, 86,
                65, 76, 85, 69, 0, 0, 0, 72, 0, 5, 0, 7, 0, 0, 0, 0, 0, 0, 0, 35, 0, 0, 0, 0, 0, 0,
                0, 71, 0, 3, 0, 7, 0, 0, 0, 3, 0, 0, 0, 71, 0, 4, 0, 9, 0, 0, 0, 34, 0, 0, 0, 0, 0,
                0, 0, 71, 0, 4, 0, 9, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 71, 0, 4, 0, 11, 0, 0, 0,
                1, 0, 0, 0, 83, 0, 0, 0, 19, 0, 2, 0, 2, 0, 0, 0, 33, 0, 3, 0, 3, 0, 0, 0, 2, 0, 0,
                0, 21, 0, 4, 0, 6, 0, 0, 0, 32, 0, 0, 0, 1, 0, 0, 0, 30, 0, 3, 0, 7, 0, 0, 0, 6, 0,
                0, 0, 32, 0, 4, 0, 8, 0, 0, 0, 2, 0, 0, 0, 7, 0, 0, 0, 59, 0, 4, 0, 8, 0, 0, 0, 9,
                0, 0, 0, 2, 0, 0, 0, 43, 0, 4, 0, 6, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 50, 0, 4, 0,
                6, 0, 0, 0, 11, 0, 0, 0, 239, 190, 173, 222, 32, 0, 4, 0, 12, 0, 0, 0, 2, 0, 0, 0,
                6, 0, 0, 0, 54, 0, 5, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 248, 0, 2,
                0, 5, 0, 0, 0, 65, 0, 5, 0, 12, 0, 0, 0, 13, 0, 0, 0, 9, 0, 0, 0, 10, 0, 0, 0, 62,
                0, 3, 0, 13, 0, 0, 0, 11, 0, 0, 0, 253, 0, 1, 0, 56, 0, 1, 0,
            ];
            ShaderModule::from_bytes(device.clone(), &MODULE).unwrap()
        };

        #[derive(Debug, Copy, Clone)]
        #[allow(non_snake_case)]
        #[repr(C)]
        struct SpecConsts {
            VALUE: i32,
        }
        unsafe impl SpecializationConstants for SpecConsts {
            fn descriptors() -> &'static [SpecializationMapEntry] {
                static DESCRIPTORS: [SpecializationMapEntry; 1] = [SpecializationMapEntry {
                    constant_id: 83,
                    offset: 0,
                    size: 4,
                }];
                &DESCRIPTORS
            }
        }

        let pipeline = ComputePipeline::new(
            device.clone(),
            module.entry_point("main").unwrap(),
            &SpecConsts { VALUE: 0x12345678 },
            None,
            |_| {},
        )
        .unwrap();

        let data_buffer =
            CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), false, 0).unwrap();

        let set = PersistentDescriptorSet::new(
            pipeline
                .layout()
                .descriptor_set_layouts()
                .get(0)
                .unwrap()
                .clone(),
            [WriteDescriptorSet::buffer(0, data_buffer.clone())],
        )
        .unwrap();

        let mut cbb = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        cbb.bind_pipeline_compute(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set,
            )
            .dispatch([1, 1, 1])
            .unwrap();
        let cb = cbb.build().unwrap();

        let future = now(device.clone())
            .then_execute(queue.clone(), cb)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let data_buffer_content = data_buffer.read().unwrap();
        assert_eq!(*data_buffer_content, 0x12345678);
    }
}
