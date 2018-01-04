// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::sync::Arc;

use descriptor::descriptor::DescriptorDesc;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use descriptor::pipeline_layout::PipelineLayout;
use descriptor::pipeline_layout::PipelineLayoutAbstract;
use descriptor::pipeline_layout::PipelineLayoutCreationError;
use descriptor::pipeline_layout::PipelineLayoutDesc;
use descriptor::pipeline_layout::PipelineLayoutDescPcRange;
use descriptor::pipeline_layout::PipelineLayoutNotSupersetError;
use descriptor::pipeline_layout::PipelineLayoutSuperset;
use descriptor::pipeline_layout::PipelineLayoutSys;
use pipeline::shader::EntryPointAbstract;
use pipeline::shader::SpecializationConstants;

use Error;
use OomError;
use SafeDeref;
use VulkanObject;
use check_errors;
use device::Device;
use device::DeviceOwned;
use vk;

/// A pipeline object that describes to the Vulkan implementation how it should perform compute
/// operations.
///
/// The template parameter contains the descriptor set to use with this pipeline.
///
/// All compute pipeline objects implement the `ComputePipelineAbstract` trait. You can turn any
/// `Arc<ComputePipeline<Pl>>` into an `Arc<ComputePipelineAbstract>` if necessary.
pub struct ComputePipeline<Pl> {
    inner: Inner,
    pipeline_layout: Pl,
}

struct Inner {
    pipeline: vk::Pipeline,
    device: Arc<Device>,
}

impl ComputePipeline<()> {
    /// Builds a new `ComputePipeline`.
    pub fn new<Cs>(
        device: Arc<Device>, shader: &Cs, specialization: &Cs::SpecializationConstants)
        -> Result<ComputePipeline<PipelineLayout<Cs::PipelineLayout>>, ComputePipelineCreationError>
        where Cs::PipelineLayout: Clone,
              Cs: EntryPointAbstract
    {
        unsafe {
            let pipeline_layout = shader.layout().clone().build(device.clone())?;
            ComputePipeline::with_unchecked_pipeline_layout(device,
                                                            shader,
                                                            specialization,
                                                            pipeline_layout)
        }
    }
}

impl<Pl> ComputePipeline<Pl> {
    /// Builds a new `ComputePipeline` with a specific pipeline layout.
    ///
    /// An error will be returned if the pipeline layout isn't a superset of what the shader
    /// uses.
    pub fn with_pipeline_layout<Cs>(device: Arc<Device>, shader: &Cs,
                                    specialization: &Cs::SpecializationConstants,
                                    pipeline_layout: Pl)
                                    -> Result<ComputePipeline<Pl>, ComputePipelineCreationError>
        where Cs::PipelineLayout: Clone,
              Cs: EntryPointAbstract,
              Pl: PipelineLayoutAbstract
    {
        unsafe {
            PipelineLayoutSuperset::ensure_superset_of(&pipeline_layout, shader.layout())?;
            ComputePipeline::with_unchecked_pipeline_layout(device,
                                                            shader,
                                                            specialization,
                                                            pipeline_layout)
        }
    }

    /// Same as `with_pipeline_layout`, but doesn't check whether the pipeline layout is a
    /// superset of what the shader expects.
    pub unsafe fn with_unchecked_pipeline_layout<Cs>(
        device: Arc<Device>, shader: &Cs, specialization: &Cs::SpecializationConstants,
        pipeline_layout: Pl)
        -> Result<ComputePipeline<Pl>, ComputePipelineCreationError>
        where Cs::PipelineLayout: Clone,
              Cs: EntryPointAbstract,
              Pl: PipelineLayoutAbstract
    {
        let vk = device.pointers();

        let pipeline = {
            let spec_descriptors = Cs::SpecializationConstants::descriptors();
            let specialization = vk::SpecializationInfo {
                mapEntryCount: spec_descriptors.len() as u32,
                pMapEntries: spec_descriptors.as_ptr() as *const _,
                dataSize: mem::size_of_val(specialization),
                pData: specialization as *const Cs::SpecializationConstants as *const _,
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
                stage: stage,
                layout: PipelineLayoutAbstract::sys(&pipeline_layout).internal_object(),
                basePipelineHandle: 0,
                basePipelineIndex: 0,
            };

            let mut output = mem::uninitialized();
            check_errors(vk.CreateComputePipelines(device.internal_object(),
                                                   0,
                                                   1,
                                                   &infos,
                                                   ptr::null(),
                                                   &mut output))?;
            output
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

impl<Pl> fmt::Debug for ComputePipeline<Pl> {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Vulkan compute pipeline {:?}>", self.inner.pipeline)
    }
}

impl<Pl> ComputePipeline<Pl> {
    /// Returns the `Device` this compute pipeline was created with.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.inner.device
    }

    /// Returns the pipeline layout used in this compute pipeline.
    #[inline]
    pub fn layout(&self) -> &Pl {
        &self.pipeline_layout
    }
}

/// Trait implemented on all compute pipelines.
pub unsafe trait ComputePipelineAbstract: PipelineLayoutAbstract {
    /// Returns an opaque object that represents the inside of the compute pipeline.
    fn inner(&self) -> ComputePipelineSys;
}

unsafe impl<Pl> ComputePipelineAbstract for ComputePipeline<Pl>
    where Pl: PipelineLayoutAbstract
{
    #[inline]
    fn inner(&self) -> ComputePipelineSys {
        ComputePipelineSys(self.inner.pipeline, PhantomData)
    }
}

unsafe impl<T> ComputePipelineAbstract for T
    where T: SafeDeref,
          T::Target: ComputePipelineAbstract
{
    #[inline]
    fn inner(&self) -> ComputePipelineSys {
        (**self).inner()
    }
}

/// Opaque object that represents the inside of the compute pipeline. Can be made into a trait
/// object.
#[derive(Debug, Copy, Clone)]
pub struct ComputePipelineSys<'a>(vk::Pipeline, PhantomData<&'a ()>);

unsafe impl<'a> VulkanObject for ComputePipelineSys<'a> {
    type Object = vk::Pipeline;

    const TYPE: vk::DebugReportObjectTypeEXT = vk::DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT;

    #[inline]
    fn internal_object(&self) -> vk::Pipeline {
        self.0
    }
}

unsafe impl<Pl> PipelineLayoutAbstract for ComputePipeline<Pl>
    where Pl: PipelineLayoutAbstract
{
    #[inline]
    fn sys(&self) -> PipelineLayoutSys {
        self.layout().sys()
    }

    #[inline]
    fn descriptor_set_layout(&self, index: usize) -> Option<&Arc<UnsafeDescriptorSetLayout>> {
        self.layout().descriptor_set_layout(index)
    }
}

unsafe impl<Pl> PipelineLayoutDesc for ComputePipeline<Pl>
    where Pl: PipelineLayoutDesc
{
    #[inline]
    fn num_sets(&self) -> usize {
        self.pipeline_layout.num_sets()
    }

    #[inline]
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        self.pipeline_layout.num_bindings_in_set(set)
    }

    #[inline]
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        self.pipeline_layout.descriptor(set, binding)
    }

    #[inline]
    fn num_push_constants_ranges(&self) -> usize {
        self.pipeline_layout.num_push_constants_ranges()
    }

    #[inline]
    fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
        self.pipeline_layout.push_constants_range(num)
    }
}

unsafe impl<Pl> DeviceOwned for ComputePipeline<Pl> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.device()
    }
}

// TODO: remove in favor of ComputePipelineAbstract?
unsafe impl<Pl> VulkanObject for ComputePipeline<Pl> {
    type Object = vk::Pipeline;

    const TYPE: vk::DebugReportObjectTypeEXT = vk::DEBUG_REPORT_OBJECT_TYPE_PIPELINE_EXT;

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
}

impl error::Error for ComputePipelineCreationError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            ComputePipelineCreationError::OomError(_) => "not enough memory available",
            ComputePipelineCreationError::PipelineLayoutCreationError(_) =>
                "error while creating the pipeline layout object",
            ComputePipelineCreationError::IncompatiblePipelineLayout(_) =>
                "the pipeline layout is not compatible with what the shader expects",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            ComputePipelineCreationError::OomError(ref err) => Some(err),
            ComputePipelineCreationError::PipelineLayoutCreationError(ref err) => Some(err),
            ComputePipelineCreationError::IncompatiblePipelineLayout(ref err) => Some(err),
        }
    }
}

impl fmt::Display for ComputePipelineCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
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
            },
            err @ Error::OutOfDeviceMemory => {
                ComputePipelineCreationError::OomError(OomError::from(err))
            },
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

#[cfg(test)]
mod tests {
    use buffer::BufferUsage;
    use buffer::CpuAccessibleBuffer;
    use command_buffer::AutoCommandBufferBuilder;
    use descriptor::descriptor::DescriptorBufferDesc;
    use descriptor::descriptor::DescriptorDesc;
    use descriptor::descriptor::DescriptorDescTy;
    use descriptor::descriptor::ShaderStages;
    use descriptor::descriptor_set::PersistentDescriptorSet;
    use descriptor::pipeline_layout::PipelineLayoutDesc;
    use descriptor::pipeline_layout::PipelineLayoutDescPcRange;
    use pipeline::ComputePipeline;
    use pipeline::shader::ShaderModule;
    use pipeline::shader::SpecializationConstants;
    use pipeline::shader::SpecializationMapEntry;
    use std::ffi::CStr;
    use std::sync::Arc;
    use sync::GpuFuture;
    use sync::now;

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
                3,
                2,
                35,
                7,
                0,
                0,
                1,
                0,
                1,
                0,
                8,
                0,
                14,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                17,
                0,
                2,
                0,
                1,
                0,
                0,
                0,
                11,
                0,
                6,
                0,
                1,
                0,
                0,
                0,
                71,
                76,
                83,
                76,
                46,
                115,
                116,
                100,
                46,
                52,
                53,
                48,
                0,
                0,
                0,
                0,
                14,
                0,
                3,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                15,
                0,
                5,
                0,
                5,
                0,
                0,
                0,
                4,
                0,
                0,
                0,
                109,
                97,
                105,
                110,
                0,
                0,
                0,
                0,
                16,
                0,
                6,
                0,
                4,
                0,
                0,
                0,
                17,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                3,
                0,
                3,
                0,
                2,
                0,
                0,
                0,
                194,
                1,
                0,
                0,
                5,
                0,
                4,
                0,
                4,
                0,
                0,
                0,
                109,
                97,
                105,
                110,
                0,
                0,
                0,
                0,
                5,
                0,
                4,
                0,
                7,
                0,
                0,
                0,
                79,
                117,
                116,
                112,
                117,
                116,
                0,
                0,
                6,
                0,
                5,
                0,
                7,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                119,
                114,
                105,
                116,
                101,
                0,
                0,
                0,
                5,
                0,
                4,
                0,
                9,
                0,
                0,
                0,
                119,
                114,
                105,
                116,
                101,
                0,
                0,
                0,
                5,
                0,
                4,
                0,
                11,
                0,
                0,
                0,
                86,
                65,
                76,
                85,
                69,
                0,
                0,
                0,
                72,
                0,
                5,
                0,
                7,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                35,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                71,
                0,
                3,
                0,
                7,
                0,
                0,
                0,
                3,
                0,
                0,
                0,
                71,
                0,
                4,
                0,
                9,
                0,
                0,
                0,
                34,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                71,
                0,
                4,
                0,
                9,
                0,
                0,
                0,
                33,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                71,
                0,
                4,
                0,
                11,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                83,
                0,
                0,
                0,
                19,
                0,
                2,
                0,
                2,
                0,
                0,
                0,
                33,
                0,
                3,
                0,
                3,
                0,
                0,
                0,
                2,
                0,
                0,
                0,
                21,
                0,
                4,
                0,
                6,
                0,
                0,
                0,
                32,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                30,
                0,
                3,
                0,
                7,
                0,
                0,
                0,
                6,
                0,
                0,
                0,
                32,
                0,
                4,
                0,
                8,
                0,
                0,
                0,
                2,
                0,
                0,
                0,
                7,
                0,
                0,
                0,
                59,
                0,
                4,
                0,
                8,
                0,
                0,
                0,
                9,
                0,
                0,
                0,
                2,
                0,
                0,
                0,
                43,
                0,
                4,
                0,
                6,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                50,
                0,
                4,
                0,
                6,
                0,
                0,
                0,
                11,
                0,
                0,
                0,
                239,
                190,
                173,
                222,
                32,
                0,
                4,
                0,
                12,
                0,
                0,
                0,
                2,
                0,
                0,
                0,
                6,
                0,
                0,
                0,
                54,
                0,
                5,
                0,
                2,
                0,
                0,
                0,
                4,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                3,
                0,
                0,
                0,
                248,
                0,
                2,
                0,
                5,
                0,
                0,
                0,
                65,
                0,
                5,
                0,
                12,
                0,
                0,
                0,
                13,
                0,
                0,
                0,
                9,
                0,
                0,
                0,
                10,
                0,
                0,
                0,
                62,
                0,
                3,
                0,
                13,
                0,
                0,
                0,
                11,
                0,
                0,
                0,
                253,
                0,
                1,
                0,
                56,
                0,
                1,
                0,
            ];
            ShaderModule::new(device.clone(), &MODULE).unwrap()
        };

        let shader = unsafe {
            #[derive(Debug, Copy, Clone)]
            struct Layout;
            unsafe impl PipelineLayoutDesc for Layout {
                fn num_sets(&self) -> usize {
                    1
                }
                fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
                    match set {
                        0 => Some(1),
                        _ => None,
                    }
                }
                fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
                    match (set, binding) {
                        (0, 0) => Some(DescriptorDesc {
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
                                       }),
                        _ => None,
                    }
                }
                fn num_push_constants_ranges(&self) -> usize {
                    0
                }
                fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
                    None
                }
            }

            static NAME: [u8; 5] = [109, 97, 105, 110, 0]; // "main"
            module.compute_entry_point(CStr::from_ptr(NAME.as_ptr() as *const _), Layout)
        };

        #[derive(Debug, Copy, Clone)]
        #[allow(non_snake_case)]
        #[repr(C)]
        struct SpecConsts {
            VALUE: i32,
        }
        unsafe impl SpecializationConstants for SpecConsts {
            fn descriptors() -> &'static [SpecializationMapEntry] {
                static DESCRIPTORS: [SpecializationMapEntry; 1] = [
                    SpecializationMapEntry {
                        constant_id: 83,
                        offset: 0,
                        size: 4,
                    },
                ];
                &DESCRIPTORS
            }
        }

        let pipeline = Arc::new(ComputePipeline::new(device.clone(),
                                                     &shader,
                                                     &SpecConsts { VALUE: 0x12345678 })
                                    .unwrap());

        let data_buffer = CpuAccessibleBuffer::from_data(device.clone(), BufferUsage::all(), 0)
            .unwrap();
        let set = PersistentDescriptorSet::start(pipeline.clone(), 0)
            .add_buffer(data_buffer.clone())
            .unwrap()
            .build()
            .unwrap();

        let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(),
                                                                               queue.family())
            .unwrap()
            .dispatch([1, 1, 1], pipeline, set, ())
            .unwrap()
            .build()
            .unwrap();

        let future = now(device.clone())
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        let data_buffer_content = data_buffer.read().unwrap();
        assert_eq!(*data_buffer_content, 0x12345678);
    }
}
