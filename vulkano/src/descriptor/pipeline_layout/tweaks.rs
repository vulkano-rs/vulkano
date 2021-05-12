// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::RuntimePipelineDesc;
use crate::descriptor::descriptor::DescriptorBufferDesc;
use crate::descriptor::descriptor::DescriptorDesc;
use crate::descriptor::descriptor::DescriptorDescTy;
use crate::descriptor::pipeline_layout::PipelineLayoutDesc;
use fnv::FnvHashSet;

/// Transforms a `PipelineLayoutDesc`.
///
/// Used to adjust automatically inferred `PipelineLayoutDesc`s with information that cannot be inferred.
pub fn tweak<T, I>(inner: T, dynamic_buffers: I) -> RuntimePipelineDesc
where
    T: PipelineLayoutDesc,
    I: IntoIterator<Item = (usize, usize)>,
{
    let dynamic_buffers: FnvHashSet<(usize, usize)> = dynamic_buffers.into_iter().collect();
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

    unsafe {
        let descriptor_sets = (0..inner.num_sets())
            .map(|set| {
                (0..inner.num_bindings_in_set(set).unwrap_or(0))
                    .map(|binding| {
                        inner.descriptor(set, binding).map(|desc| match desc.ty {
                            DescriptorDescTy::Buffer(ref buffer_desc)
                                if dynamic_buffers.contains(&(set, binding)) =>
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
                    })
                    .collect()
            })
            .collect();

        // TODO: needs tests
        let push_constants = (0..inner.num_push_constants_ranges())
            .map(|num| inner.push_constants_range(num).unwrap())
            .collect();

        RuntimePipelineDesc::new_unchecked(descriptor_sets, push_constants)
    }
}
