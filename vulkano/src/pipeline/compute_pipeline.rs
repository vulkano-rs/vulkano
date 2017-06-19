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
use descriptor::pipeline_layout::PipelineLayoutSys;
use descriptor::pipeline_layout::PipelineLayoutCreationError;
use descriptor::pipeline_layout::PipelineLayoutDesc;
use descriptor::pipeline_layout::PipelineLayoutDescNames;
use descriptor::pipeline_layout::PipelineLayoutDescPcRange;
use descriptor::pipeline_layout::PipelineLayoutSuperset;
use descriptor::pipeline_layout::PipelineLayoutAbstract;
use descriptor::pipeline_layout::PipelineLayoutNotSupersetError;
use pipeline::shader::ComputeShaderEntryPoint;
use pipeline::shader::SpecializationConstants;

use device::Device;
use device::DeviceOwned;
use Error;
use OomError;
use SafeDeref;
use VulkanObject;
use check_errors;
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
    pub fn new<Css, Csl>(device: Arc<Device>, shader: &ComputeShaderEntryPoint<Css, Csl>,
                         specialization: &Css) 
                         -> Result<ComputePipeline<PipelineLayout<Csl>>, ComputePipelineCreationError>
        where Csl: PipelineLayoutDescNames + Clone,
              Css: SpecializationConstants
    {
        let vk = device.pointers();

        let pipeline_layout = shader.layout().clone().build(device.clone())?;

        debug_assert!(PipelineLayoutSuperset::ensure_superset_of(pipeline_layout.desc(),
                                                                 shader.layout()).is_ok());

        let pipeline = unsafe {
            let spec_descriptors = <Css as SpecializationConstants>::descriptors();
            let specialization = vk::SpecializationInfo {
                mapEntryCount: spec_descriptors.len() as u32,
                pMapEntries: spec_descriptors.as_ptr() as *const _,
                dataSize: mem::size_of_val(specialization),
                pData: specialization as *const Css as *const _,
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
            try!(check_errors(vk.CreateComputePipelines(device.internal_object(), 0,
                                                        1, &infos, ptr::null(), &mut output)));
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
    where T: SafeDeref, T::Target: ComputePipelineAbstract
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

    #[inline]
    fn internal_object(&self) -> vk::Pipeline {
        self.0
    }
}

unsafe impl<Pl> PipelineLayoutAbstract for ComputePipeline<Pl> where Pl: PipelineLayoutAbstract {
    #[inline]
    fn sys(&self) -> PipelineLayoutSys {
        self.layout().sys()
    }

    #[inline]
    fn descriptor_set_layout(&self, index: usize) -> Option<&Arc<UnsafeDescriptorSetLayout>> {
        self.layout().descriptor_set_layout(index)
    }
}

unsafe impl<Pl> PipelineLayoutDesc for ComputePipeline<Pl> where Pl: PipelineLayoutDesc {
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

unsafe impl<Pl> PipelineLayoutDescNames for ComputePipeline<Pl> where Pl: PipelineLayoutDescNames {
    #[inline]
    fn descriptor_by_name(&self, name: &str) -> Option<(usize, usize)> {
        self.pipeline_layout.descriptor_by_name(name)
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
            ComputePipelineCreationError::PipelineLayoutCreationError(_) => "error while creating \
                                                                             the pipeline layout \
                                                                             object",
            ComputePipelineCreationError::IncompatiblePipelineLayout(_) => "the pipeline layout is \
                                                                            not compatible with what \
                                                                            the shader expects",
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
            _ => panic!("unexpected error: {:?}", err)
        }
    }
}

#[cfg(test)]
mod tests {
    // TODO: test for basic creation
    // TODO: test for pipeline layout error
}
