// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use descriptor::descriptor::DescriptorDesc;
use descriptor::pipeline_layout::PipelineLayoutDesc;
use descriptor::pipeline_layout::PipelineLayoutDescPcRange;

/// Description of an empty pipeline layout.
///
/// # Example
///
/// ```
/// # use std::sync::Arc;
/// # use vulkano::device::Device;
/// use vulkano::descriptor::pipeline_layout::EmptyPipelineDesc;
/// use vulkano::descriptor::pipeline_layout::PipelineLayoutDesc;
///
/// # let device: Arc<Device> = return;
/// let pipeline_layout = EmptyPipelineDesc.build(device.clone()).unwrap();
/// ```
#[derive(Debug, Copy, Clone)]
pub struct EmptyPipelineDesc;

unsafe impl PipelineLayoutDesc for EmptyPipelineDesc {
    #[inline]
    fn num_sets(&self) -> usize {
        0
    }

    #[inline]
    fn num_bindings_in_set(&self, _: usize) -> Option<usize> {
        None
    }

    #[inline]
    fn descriptor(&self, _: usize, _: usize) -> Option<DescriptorDesc> {
        None
    }

    #[inline]
    fn num_push_constants_ranges(&self) -> usize {
        0
    }

    #[inline]
    fn push_constants_range(&self, _: usize) -> Option<PipelineLayoutDescPcRange> {
        None
    }
}
