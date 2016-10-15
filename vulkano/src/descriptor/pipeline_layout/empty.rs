// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use device::Device;
use descriptor::descriptor::DescriptorDesc;
use descriptor::descriptor::ShaderStages;
use descriptor::pipeline_layout::PipelineLayoutRef;
use descriptor::pipeline_layout::PipelineLayoutDesc;
use descriptor::pipeline_layout::PipelineLayout;
use descriptor::pipeline_layout::PipelineLayoutSys;
use descriptor::pipeline_layout::PipelineLayoutCreationError;

/// Implementation of `PipelineLayoutRef` for an empty pipeline.
pub struct EmptyPipeline {
    inner: PipelineLayout
}

impl EmptyPipeline {
    /// Builds a new empty pipeline.
    pub fn new(device: &Arc<Device>) -> Result<Arc<EmptyPipeline>, PipelineLayoutCreationError> {
        let inner = {
            try!(PipelineLayout::new(device, Box::new(EmptyPipelineDesc) as Box<_>))
        };

        Ok(Arc::new(EmptyPipeline {
            inner: inner
        }))
    }
}

unsafe impl PipelineLayoutRef for EmptyPipeline {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }

    #[inline]
    fn sys(&self) -> PipelineLayoutSys {
        self.inner.sys()
    }

    #[inline]
    fn desc(&self) -> &PipelineLayoutDesc {
        self.inner.desc()
    }
}

unsafe impl PipelineLayoutDesc for EmptyPipeline {
    #[inline]
    fn num_sets(&self) -> usize {
        0
    }

    #[inline]
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        None
    }

    #[inline]
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        None
    }

    #[inline]
    fn num_push_constants_ranges(&self) -> usize {
        0
    }

    #[inline]
    fn push_constants_range(&self, num: usize) -> Option<(usize, usize, ShaderStages)> {
        None
    }
}

/// Description of an empty pipeline layout.
#[derive(Debug, Copy, Clone)]
pub struct EmptyPipelineDesc;

unsafe impl PipelineLayoutDesc for EmptyPipelineDesc {
    #[inline]
    fn num_sets(&self) -> usize {
        0
    }

    #[inline]
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        None
    }

    #[inline]
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        None
    }

    #[inline]
    fn num_push_constants_ranges(&self) -> usize {
        0
    }

    #[inline]
    fn push_constants_range(&self, num: usize) -> Option<(usize, usize, ShaderStages)> {
        None
    }
}


#[cfg(test)]
mod tests {
    use descriptor::pipeline_layout::empty::EmptyPipeline;

    #[test]
    fn create() {
        let (device, _) = gfx_dev_and_queue!();
        let _layout = EmptyPipeline::new(&device).unwrap();
    }
}
