// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::check_errors;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::pipeline::cache::PipelineCache;
use crate::pipeline::layout::PipelineLayout;
use crate::pipeline::layout::PipelineLayoutCreationError;
use crate::pipeline::layout::PipelineLayoutNotSupersetError;
use crate::pipeline::shader::EntryPointAbstract;
use crate::pipeline::shader::SpecializationConstants;
use crate::vk;
use crate::Error;
use crate::OomError;
use crate::SafeDeref;
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
    pipeline: vk::Pipeline,
    device: Arc<Device>,
}

impl ComputePipeline {
    /// Builds a new `ComputePipeline`.
    pub fn new<Cs, Css>(
        device: Arc<Device>,
        shader: &Cs,
        spec_constants: &Css,
        cache: Option<Arc<PipelineCache>>,
    ) -> Result<ComputePipeline, ComputePipelineCreationError>
    where
        Cs: EntryPointAbstract,
        Css: SpecializationConstants,
    {
        unsafe {
            let pipeline_layout = Arc::new(PipelineLayout::new(
                device.clone(),
                shader.layout_desc().clone(),
            )?);
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
            pipeline_layout
                .desc()
                .ensure_superset_of(shader.layout_desc())?;
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
        let vk = device.pointers();

        let pipeline = {
            let spec_descriptors = Css::descriptors();
            let specialization = vk::SpecializationInfo {
                mapEntryCount: spec_descriptors.len() as u32,
                pMapEntries: spec_descriptors.as_ptr() as *const _,
                dataSize: mem::size_of_val(spec_constants),
                pData: spec_constants as *const Css as *const _,
            };

            let stage = vk::PipelineShaderStageCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                stage: vk::SHADER_STAGE_COMPUTE_BIT,
                module: shader.module().internal_object(),
                pName: shader.name().as_ptr(),
                pSpecializationInfo: if specialization.dataSize == 0 {
                    ptr::null()
                } else {
                    &specialization
                },
            };

            let infos = vk::ComputePipelineCreateInfo {
                sType: vk::STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                stage,
                layout: pipeline_layout.internal_object(),
                basePipelineHandle: 0,
                basePipelineIndex: 0,
            };

            let cache_handle = match cache {
                Some(ref cache) => cache.internal_object(),
                None => vk::NULL_HANDLE,
            };

            let mut output = MaybeUninit::uninit();
            check_errors(vk.CreateComputePipelines(
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
}

impl fmt::Debug for ComputePipeline {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan compute pipeline {:?}>", self.inner.pipeline)
    }
}

impl ComputePipeline {
    /// Returns the `Device` this compute pipeline was created with.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.inner.device
    }
}

/// Trait implemented on all compute pipelines.
pub unsafe trait ComputePipelineAbstract: DeviceOwned {
    /// Returns an opaque object that represents the inside of the compute pipeline.
    fn inner(&self) -> ComputePipelineSys;

    /// Returns the pipeline layout used in this compute pipeline.
    fn layout(&self) -> &Arc<PipelineLayout>;
}

unsafe impl ComputePipelineAbstract for ComputePipeline {
    #[inline]
    fn inner(&self) -> ComputePipelineSys {
        ComputePipelineSys(self.inner.pipeline, PhantomData)
    }

    #[inline]
    fn layout(&self) -> &Arc<PipelineLayout> {
        &self.pipeline_layout
    }
}

unsafe impl<T> ComputePipelineAbstract for T
where
    T: SafeDeref,
    T::Target: ComputePipelineAbstract,
{
    #[inline]
    fn inner(&self) -> ComputePipelineSys {
        (**self).inner()
    }

    #[inline]
    fn layout(&self) -> &Arc<PipelineLayout> {
        (**self).layout()
    }
}

/// Opaque object that represents the inside of the compute pipeline. Can be made into a trait
/// object.
#[derive(Debug, Copy, Clone)]
pub struct ComputePipelineSys<'a>(vk::Pipeline, PhantomData<&'a ()>);

unsafe impl<'a> VulkanObject for ComputePipelineSys<'a> {
    type Object = vk::Pipeline;

    const TYPE: vk::ObjectType = vk::OBJECT_TYPE_PIPELINE;

    #[inline]
    fn internal_object(&self) -> vk::Pipeline {
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
    type Object = vk::Pipeline;

    const TYPE: vk::ObjectType = vk::OBJECT_TYPE_PIPELINE;

    #[inline]
    fn internal_object(&self) -> vk::Pipeline {
        self.inner.pipeline
    }
}

impl Drop for Inner {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyPipeline(self.device.internal_object(), self.pipeline, ptr::null());
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
    IncompatiblePipelineLayout(PipelineLayoutNotSupersetError),
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

impl From<PipelineLayoutNotSupersetError> for ComputePipelineCreationError {
    #[inline]
    fn from(err: PipelineLayoutNotSupersetError) -> ComputePipelineCreationError {
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
    use crate::descriptor::descriptor::DescriptorBufferDesc;
    use crate::descriptor::descriptor::DescriptorDesc;
    use crate::descriptor::descriptor::DescriptorDescTy;
    use crate::descriptor::descriptor::ShaderStages;
    use crate::descriptor::descriptor_set::PersistentDescriptorSet;
    use crate::pipeline::layout::PipelineLayoutDesc;
    use crate::pipeline::shader::ShaderModule;
    use crate::pipeline::shader::SpecializationConstants;
    use crate::pipeline::shader::SpecializationMapEntry;
    use crate::pipeline::ComputePipeline;
    use crate::pipeline::ComputePipelineAbstract;
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
                PipelineLayoutDesc::new_unchecked(
                    vec![vec![Some(DescriptorDesc {
                        ty: DescriptorDescTy::Buffer(DescriptorBufferDesc {
                            dynamic: Some(false),
                            storage: true,
                        }),
                        array_count: 1,
                        stages: ShaderStages {
                            compute: true,
                            ..ShaderStages::none()
                        },
                        readonly: true,
                    })]],
                    vec![],
                ),
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
            )
            .unwrap(),
        );

        let data_buffer =
            CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), false, 0).unwrap();
        let layout = pipeline.layout().descriptor_set_layout(0).unwrap();
        let set = PersistentDescriptorSet::start(layout.clone())
            .add_buffer(data_buffer.clone())
            .unwrap()
            .build()
            .unwrap();

        let mut cbb = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        cbb.dispatch([1, 1, 1], pipeline.clone(), set, (), vec![])
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
