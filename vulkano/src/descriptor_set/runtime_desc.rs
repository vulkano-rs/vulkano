use std::mem;
use std::option::IntoIter as OptionIntoIter;
use std::ptr;
use std::sync::Arc;

use buffer::BufferResource;
use descriptor_set::AbstractDescriptorSet;
use descriptor_set::AbstractDescriptorSetLayout;
use descriptor_set::DescriptorBind;
use descriptor_set::DescriptorDesc;
use descriptor_set::DescriptorSetDesc;
use descriptor_set::DescriptorWrite;
use descriptor_set::PipelineLayoutDesc;
use device::Device;

use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

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

/// Dummy implementation of `PipelineLayoutDesc` that describes an empty pipeline.
///
/// The descriptors, descriptor sets and push constants are all `()`. You have to pass `()` when
/// drawing when you use a `EmptyPipelineDesc`.
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

unsafe impl DescriptorSetDesc for RuntimeDescriptorSetDesc {
    type Write = Vec<(u32, DescriptorBind)>;

    type Init = Vec<(u32, DescriptorBind)>;

    fn descriptors(&self) -> Vec<DescriptorDesc> {
        self.descriptors.clone()
    }

    fn decode_write(&self, data: Self::Write) -> Vec<DescriptorWrite> {
        data.into_iter().map(|(binding, bind)| {
            // TODO: check correctness?

            DescriptorWrite {
                binding: binding,
                array_element: 0,       // FIXME:
                content: bind,
            }
        }).collect()
    }

    fn decode_init(&self, data: Self::Init) -> Vec<DescriptorWrite> {
        self.decode_write(data)
    }
}
