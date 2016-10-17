// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::cmp;
use std::sync::Arc;
use descriptor::descriptor::DescriptorDesc;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use descriptor::pipeline_layout::PipelineLayoutDesc;
use descriptor::pipeline_layout::PipelineLayoutDescNames;
use descriptor::pipeline_layout::PipelineLayoutDescPcRange;

pub struct PipelineLayoutDescUnion<A, B> {
    a: A,
    b: B,
}

impl<A, B> PipelineLayoutDescUnion<A, B> {
    // FIXME: check collisions
    pub fn new(a: A, b: B) -> PipelineLayoutDescUnion<A, B> {
        PipelineLayoutDescUnion { a: a, b: b }
    }
}

unsafe impl<A, B> PipelineLayoutDesc for PipelineLayoutDescUnion<A, B>
    where A: PipelineLayoutDesc, B: PipelineLayoutDesc
{
    #[inline]
    fn num_sets(&self) -> usize {
        cmp::max(self.a.num_sets(), self.b.num_sets())
    }

    #[inline]
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        let a = self.a.num_bindings_in_set(set);
        let b = self.b.num_bindings_in_set(set);

        match (a, b) {
            (Some(a), Some(b)) => Some(cmp::max(a, b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }

    #[inline]
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        let a = self.a.descriptor(set, binding);
        let b = self.b.descriptor(set, binding);

        match (a, b) {
            (Some(a), Some(b)) => Some(a.union(&b).expect("Can't be union-ed")),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }

    #[inline]
    fn provided_set_layout(&self, set: usize) -> Option<Arc<UnsafeDescriptorSetLayout>> {
        self.a.provided_set_layout(set).or(self.b.provided_set_layout(set))
    }

    #[inline]
    fn num_push_constants_ranges(&self) -> usize {
        // FIXME: wrong
        0
    }

    #[inline]
    fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
        // FIXME:
        None
    }
}

unsafe impl<A, B> PipelineLayoutDescNames for PipelineLayoutDescUnion<A, B>
    where A: PipelineLayoutDescNames, B: PipelineLayoutDescNames
{
    #[inline]
    fn descriptor_by_name(&self, name: &str) -> Option<(usize, usize)> {
        let a = self.a.descriptor_by_name(name);
        let b = self.b.descriptor_by_name(name);

        match (a, b) {
            (None, None) => None,
            (Some(r), None) => Some(r),
            (None, Some(r)) => Some(r),
            (Some(a), Some(b)) => { assert_eq!(a, b); Some(a) }
        }
    }
}
