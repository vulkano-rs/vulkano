// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use descriptor::descriptor::DescriptorBufferDesc;
use descriptor::descriptor::DescriptorDesc;
use descriptor::descriptor::DescriptorDescTy;
use descriptor::pipeline_layout::PipelineLayoutDesc;
use descriptor::pipeline_layout::PipelineLayoutDescPcRange;
use fnv::FnvHashSet;

/// Transforms a `PipelineLayoutDesc`.
///
/// Used to adjust automatically inferred `PipelineLayoutDesc`s with information that cannot be inferred.
pub struct PipelineLayoutDescTweaks<T> {
    inner: T,
    dynamic_buffers: FnvHashSet<(usize, usize)>,
}

impl<T> PipelineLayoutDescTweaks<T>
where
    T: PipelineLayoutDesc,
{
    /// Describe a layout, ensuring that each `(set, binding)` in `dynamic_buffers` is a dynamic buffers.
    pub fn new<I>(inner: T, dynamic_buffers: I) -> Self
    where
        I: IntoIterator<Item = (usize, usize)>,
    {
        let dynamic_buffers = dynamic_buffers.into_iter().collect();
        for &(set, binding) in &dynamic_buffers {
            debug_assert!(
                inner
                    .descriptor(set, binding)
                    .map_or(false, |desc| match desc.ty {
                        DescriptorDescTy::Buffer(_) => true,
                        _ => false,
                    }),
                "tried to make the non-buffer descriptor at set {} binding {} a dynamic buffer",
                set,
                binding
            );
        }
        Self {
            inner,
            dynamic_buffers,
        }
    }
}

unsafe impl<T> PipelineLayoutDesc for PipelineLayoutDescTweaks<T>
where
    T: PipelineLayoutDesc,
{
    #[inline]
    fn num_sets(&self) -> usize {
        self.inner.num_sets()
    }

    #[inline]
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        self.inner.num_bindings_in_set(set)
    }

    #[inline]
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        self.inner
            .descriptor(set, binding)
            .map(|desc| match desc.ty {
                DescriptorDescTy::Buffer(ref buffer_desc)
                    if self.dynamic_buffers.contains(&(set, binding)) =>
                {
                    DescriptorDesc {
                        ty: DescriptorDescTy::Buffer(DescriptorBufferDesc {
                            dynamic: Some(true),
                            ..*buffer_desc
                        }),
                        ..desc
                    }
                }
                _ => desc,
            })
    }

    #[inline]
    fn num_push_constants_ranges(&self) -> usize {
        self.inner.num_push_constants_ranges()
    }

    // TODO: needs tests
    #[inline]
    fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
        self.inner.push_constants_range(num)
    }
}
