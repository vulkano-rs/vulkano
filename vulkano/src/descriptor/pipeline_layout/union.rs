// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use descriptor::descriptor::DescriptorDesc;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use descriptor::pipeline_layout::PipelineLayoutDesc;
use descriptor::pipeline_layout::PipelineLayoutDescPcRange;
use std::cmp;
use std::sync::Arc;

/// Contains the union of two pipeline layout description.
///
/// If `A` and `B` both implement `PipelineLayoutDesc`, then this struct also implements
/// `PipelineLayoutDesc` and will correspond to the union of the `A` object and the `B` object.
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
    where A: PipelineLayoutDesc,
          B: PipelineLayoutDesc
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
        self.a
            .provided_set_layout(set)
            .or(self.b.provided_set_layout(set))
    }

    #[inline]
    fn num_push_constants_ranges(&self) -> usize {
        // We simply call `push_constants_range` repeatidely to determine when it is over.
        // TODO: consider caching this
        (self.a.num_push_constants_ranges() ..)
            .filter(|&n| self.push_constants_range(n).is_none())
            .next()
            .unwrap()
    }

    // TODO: needs tests
    #[inline]
    fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
        // The strategy here is that we return the same ranges as `self.a`, except that if there
        // happens to be a range with a similar stage in `self.b` then we adjust the offset and
        // size of the range coming from `self.a` to include the range of `self.b`.
        //
        // After all the ranges of `self.a` have been returned, we return the ones from `self.b`
        // that don't intersect with any range of `self.a`.

        if let Some(mut pc) = self.a.push_constants_range(num) {
            // We try to find the ranges in `self.b` that share the same stages as us.
            for n in 0 .. self.b.num_push_constants_ranges() {
                let other_pc = self.b.push_constants_range(n).unwrap();

                if other_pc.stages.intersects(&pc.stages) {
                    if other_pc.offset < pc.offset {
                        pc.size += pc.offset - other_pc.offset;
                        pc.size = cmp::max(pc.size, other_pc.size);
                        pc.offset = other_pc.offset;

                    } else if other_pc.offset > pc.offset {
                        pc.size = cmp::max(pc.size, other_pc.size + (other_pc.offset - pc.offset));
                    }
                }
            }

            return Some(pc);
        }

        let mut num = num - self.a.num_push_constants_ranges();
        'outer_loop: for b_r in 0 .. self.b.num_push_constants_ranges() {
            let pc = self.b.push_constants_range(b_r).unwrap();

            for n in 0 .. self.a.num_push_constants_ranges() {
                let other_pc = self.a.push_constants_range(n).unwrap();
                if other_pc.stages.intersects(&pc.stages) {
                    continue 'outer_loop;
                }
            }

            if num == 0 {
                return Some(pc);
            } else {
                num -= 1;
            }
        }

        None
    }
}
