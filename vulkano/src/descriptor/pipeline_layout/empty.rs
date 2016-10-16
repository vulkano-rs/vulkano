// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use descriptor::descriptor::DescriptorDesc;
use descriptor::descriptor::ShaderStages;
use descriptor::pipeline_layout::PipelineLayoutDesc;
use descriptor::pipeline_layout::PipelineLayoutDescNames;

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

unsafe impl PipelineLayoutDescNames for EmptyPipelineDesc {
    #[inline]
    fn descriptor_by_name(&self, name: &str) -> Option<(usize, usize)> {
        None
    }
}
