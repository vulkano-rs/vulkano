// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::check_errors;
use crate::descriptor_set::layout::DescriptorSetDesc;
use crate::descriptor_set::layout::DescriptorSetLayout;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::pipeline::cache::PipelineCache;
use crate::pipeline::layout::PipelineLayout;
use crate::pipeline::layout::PipelineLayoutCreationError;
use crate::pipeline::layout::PipelineLayoutSupersetError;
use crate::pipeline::shader::EntryPointAbstract;
use crate::pipeline::shader::SpecializationConstants;
use crate::Error;
use crate::OomError;
use crate::VulkanObject;
use std::error;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

/// A pipeline object that describes to the Vulkan implementation how it should perform compute
/// operations.
///
/// The template parameter contains the descriptor set to use with this pipeline.
///
/// All compute pipeline objects implement the `ComputePipelineAbstract` trait. You can turn any
/// `Arc<ComputePipeline>` into an `Arc<ComputePipelineAbstract>` if necessary.
///
/// Pass an optional `Arc` to a `PipelineCache` to enable pipeline caching. The vulkan
/// implementation will handle the `PipelineCache` and check if it is available.
/// Check the documentation of the `PipelineCache` for more information.
pub struct ComputePipeline {
    inner: Inner,
    pipeline_layout: Arc<PipelineLayout>,
}

struct Inner {
    pipeline: ash::vk::Pipeline,
    device: Arc<Device>,
}

impl ComputePipeline {
    /// Builds a new `ComputePipeline`.
    ///
    /// `func` is a closure that is given a mutable reference to the inferred descriptor set
    /// definitions. This can be used to make changes to the layout before it's created, for example
    /// to add dynamic buffers or immutable samplers.
    pub fn new<Cs, Css, F>(
        device: Arc<Device>,
        shader: &Cs,
        spec_constants: &Css,
        cache: Option<Arc<PipelineCache>>,
        func: F,
    ) -> Result<ComputePipeline, ComputePipelineCreationError>
    where
        Cs: EntryPointAbstract,
        Css: SpecializationConstants,
        F: FnOnce(&mut [DescriptorSetDesc]),
    {
        let mut descriptor_set_layout_descs = shader.descriptor_set_layout_descs().to_owned();
        func(&mut descriptor_set_layout_descs);

        let descriptor_set_layouts = descriptor_set_layout_descs
            .iter()
            .map(|desc| {
                Ok(Arc::new(DescriptorSetLayout::new(
                    device.clone(),
                    desc.clone(),
                )?))
            })
            .collect::<Result<Vec<_>, PipelineLayoutCreationError>>()?;
        let pipeline_layout = Arc::new(PipelineLayout::new(
            device.clone(),
            descriptor_set_layouts,
            shader.push_constant_range().iter().cloned(),
        )?);

        unsafe {
            ComputePipeline::with_unchecked_pipeline_layout(
                device,
                shader,
                spec_constants,
                pipeline_layout,
                cache,
            )
        }
    }

    /// Builds a new `ComputePipeline` with a specific pipeline layout.
    ///
    /// An error will be returned if the pipeline layout isn't a superset of what the shader
    /// uses.
    pub fn with_pipeline_layout<Cs, Css>(
        device: Arc<Device>,
        shader: &Cs,
        spec_constants: &Css,
        pipeline_layout: Arc<PipelineLayout>,
        cache: Option<Arc<PipelineCache>>,
    ) -> Result<ComputePipeline, ComputePipelineCreationError>
    where
        Cs: EntryPointAbstract,
        Css: SpecializationConstants,
    {
        if Css::descriptors() != shader.spec_constants() {
            return Err(ComputePipelineCreationError::IncompatibleSpecializationConstants);
        }

        unsafe {
            pipeline_layout.ensure_compatible_with_shader(
                shader.descriptor_set_layout_descs(),
                shader.push_constant_range(),
            )?;
            ComputePipeline::with_unchecked_pipeline_layout(
                device,
                shader,
                spec_constants,
                pipeline_layout,
                cache,
            )
        }
    }

    /// Same as `with_pipeline_layout`, but doesn't check whether the pipeline layout is a
    /// superset of what the shader expects.
    pub unsafe fn with_unchecked_pipeline_layout<Cs, Css>(
        device: Arc<Device>,
        shader: &Cs,
        spec_constants: &Css,
        pipeline_layout: Arc<PipelineLayout>,
        cache: Option<Arc<PipelineCache>>,
    ) -> Result<ComputePipeline, ComputePipelineCreationError>
    where
        Cs: EntryPointAbstract,
        Css: SpecializationConstants,
    {
        let fns = device.fns();

        let pipeline = {
            let spec_descriptors = Css::descriptors();
            let specialization = ash::vk::SpecializationInfo {
                map_entry_count: spec_descriptors.len() as u32,
                p_map_entries: spec_descriptors.as_ptr() as *const _,
                data_size: mem::size_of_val(spec_constants),
                p_data: spec_constants as *const Css as *const _,
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
                layout: pipeline_layout.internal_object(),
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

        Ok(ComputePipeline {
            inner: Inner {
                device: device.clone(),
                pipeline: pipeline,
            },
            pipeline_layout: pipeline_layout,
        })
    }

    /// Returns the `Device` this compute pipeline was created with.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.inner.device
    }

    /// Returns the pipeline layout used in this compute pipeline.
    #[inline]
    pub fn layout(&self) -> &Arc<PipelineLayout> {
        &self.pipeline_layout
    }
}

impl fmt::Debug for ComputePipeline {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan compute pipeline {:?}>", self.inner.pipeline)
    }
}

impl PartialEq for ComputePipeline {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.internal_object() == other.internal_object()
    }
}

impl Eq for ComputePipeline {}

/// Opaque object that represents the inside of the compute pipeline. Can be made into a trait
/// object.
#[derive(Debug, Copy, Clone)]
pub struct ComputePipelineSys<'a>(ash::vk::Pipeline, PhantomData<&'a ()>);

unsafe impl<'a> VulkanObject for ComputePipelineSys<'a> {
    type Object = ash::vk::Pipeline;

    #[inline]
    fn internal_object(&self) -> ash::vk::Pipeline {
        self.0
    }
}

unsafe impl DeviceOwned for ComputePipeline {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.device()
    }
}

unsafe impl VulkanObject for ComputePipeline {
    type Object = ash::vk::Pipeline;

    #[inline]
    fn internal_object(&self) -> ash::vk::Pipeline {
        self.inner.pipeline
    }
}

impl Drop for Inner {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            fns.v1_0
                .destroy_pipeline(self.device.internal_object(), self.pipeline, ptr::null());
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
    use crate::descriptor_set::layout::DescriptorDesc;
    use crate::descriptor_set::layout::DescriptorDescTy;
    use crate::descriptor_set::layout::DescriptorSetDesc;
    use crate::descriptor_set::PersistentDescriptorSet;
    use crate::pipeline::shader::ShaderModule;
    use crate::pipeline::shader::ShaderStages;
    use crate::pipeline::shader::SpecializationConstants;
    use crate::pipeline::shader::SpecializationMapEntry;
    use crate::pipeline::ComputePipeline;
    use crate::pipeline::PipelineBindPoint;
    use crate::sync::now;
    use crate::sync::GpuFuture;
    use std::ffi::CStr;
    use std::sync::Arc;

    // TODO: test for basic creation
    // TODO: test for pipeline layout error

    #[test]
    fn spec_constants() {
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
            ShaderModule::new(device.clone(), &MODULE).unwrap()
        };

        let shader = unsafe {
            static NAME: [u8; 5] = [109, 97, 105, 110, 0]; // "main"
            module.compute_entry_point(
                CStr::from_ptr(NAME.as_ptr() as *const _),
                [DescriptorSetDesc::new([Some(DescriptorDesc {
                    ty: DescriptorDescTy::StorageBuffer,
                    descriptor_count: 1,
                    stages: ShaderStages {
                        compute: true,
                        ..ShaderStages::none()
                    },
                    mutable: false,
                    variable_count: false,
                })])],
                None,
                SpecConsts::descriptors(),
            )
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

        let pipeline = Arc::new(
            ComputePipeline::new(
                device.clone(),
                &shader,
                &SpecConsts { VALUE: 0x12345678 },
                None,
                |_| {},
            )
            .unwrap(),
        );

        let data_buffer =
            CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), false, 0).unwrap();
        let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
        let mut builder = PersistentDescriptorSet::start(layout.clone());

        builder.add_buffer(data_buffer.clone()).unwrap();

        let set = builder.build().unwrap();

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
