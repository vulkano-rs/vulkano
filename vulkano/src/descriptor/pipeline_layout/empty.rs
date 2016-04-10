// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::iter;
use std::iter::Empty;
use std::sync::Arc;

use device::Device;
use descriptor::descriptor::DescriptorDesc;
use descriptor::pipeline_layout::PipelineLayout;
use descriptor::pipeline_layout::PipelineLayoutDesc;
use descriptor::pipeline_layout::UnsafePipelineLayout;
use OomError;

/// Implementation of `PipelineLayout` for an empty pipeline.
pub struct EmptyPipeline {
    inner: UnsafePipelineLayout
}

impl EmptyPipeline {
    /// Builds a new empty pipeline.
    pub fn new(device: &Arc<Device>) -> Result<Arc<EmptyPipeline>, OomError> {
        let inner = unsafe {
            try!(UnsafePipelineLayout::new(device, iter::empty(), iter::empty()))
        };

        Ok(Arc::new(EmptyPipeline {
            inner: inner
        }))
    }
}

unsafe impl PipelineLayout for EmptyPipeline {
    #[inline]
    fn inner_pipeline_layout(&self) -> &UnsafePipelineLayout {
        &self.inner
    }
}

unsafe impl PipelineLayoutDesc for EmptyPipeline {
    type SetsIter = Empty<Self::DescIter>;
    type DescIter = Empty<DescriptorDesc>;

    fn descriptors_desc(&self) -> Self::SetsIter {
        iter::empty()
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
