// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use descriptor_set::AbstractDescriptorSet;
use descriptor_set::AbstractDescriptorSetLayout;
use descriptor_set::DescriptorBind;
use descriptor_set::DescriptorDesc;
use descriptor_set::SetLayout;
use descriptor_set::SetLayoutInit;
use descriptor_set::SetLayoutWrite;
use descriptor_set::DescriptorWrite;
use descriptor_set::Layout as PipelineLayoutDesc;

/// FIXME: it should be unsafe to create this struct
pub struct RuntimeDesc;

unsafe impl PipelineLayoutDesc for RuntimeDesc {
    type DescriptorSets = Vec<Arc<AbstractDescriptorSet>>;
    type DescriptorSetLayouts = Vec<Arc<AbstractDescriptorSetLayout>>;
    type PushConstants = ();

    #[inline]
    fn decode_descriptor_sets(&self, sets: Self::DescriptorSets) -> Vec<Arc<AbstractDescriptorSet>> {
        sets
    }

    #[inline]
    fn decode_descriptor_set_layouts(&self, layouts: Self::DescriptorSetLayouts)
                                     -> Vec<Arc<AbstractDescriptorSetLayout>>
    {
        layouts
    }
}

/// Implementation of `PipelineLayoutDesc` that describes an empty pipeline.
#[derive(Debug, Copy, Clone, Default)]
pub struct EmptyPipelineDesc;
unsafe impl PipelineLayoutDesc for EmptyPipelineDesc {
    type DescriptorSets = ();
    type DescriptorSetLayouts = ();
    type PushConstants = ();

    #[inline]
    fn decode_descriptor_set_layouts(&self, _: Self::DescriptorSetLayouts)
                                     -> Vec<Arc<AbstractDescriptorSetLayout>> { vec![] }
    #[inline]
    fn decode_descriptor_sets(&self, _: Self::DescriptorSets)
                              -> Vec<Arc<AbstractDescriptorSet>> { vec![] }
}

/// FIXME: should be unsafe to create this struct
pub struct RuntimeDescriptorSetDesc {
    pub descriptors: Vec<DescriptorDesc>,
}

unsafe impl SetLayout for RuntimeDescriptorSetDesc {
    fn descriptors(&self) -> Vec<DescriptorDesc> {
        self.descriptors.clone()
    }
}

unsafe impl<T> SetLayoutWrite<T> for RuntimeDescriptorSetDesc
    where T: IntoIterator<Item = (u32, DescriptorBind)>
{
    fn decode(&self, data: T) -> Vec<DescriptorWrite> {
        data.into_iter().map(|(binding, bind)| {
            // TODO: check correctness?

            DescriptorWrite {
                binding: binding,
                array_element: 0,       // FIXME:
                content: bind,
            }
        }).collect()
    }
}

unsafe impl<T> SetLayoutInit<T> for RuntimeDescriptorSetDesc
    where T: IntoIterator<Item = (u32, DescriptorBind)>
{
    fn decode(&self, data: T) -> Vec<DescriptorWrite> {
        SetLayoutWrite::decode(self, data)
    }
}
