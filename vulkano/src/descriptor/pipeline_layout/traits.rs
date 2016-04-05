// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use descriptor::descriptor::DescriptorDesc;
use descriptor::descriptor_set::DescriptorSetsCollection;
use descriptor::pipeline_layout::UnsafePipelineLayout;

/// Trait for objects that describe the layout of the descriptors and push constants of a pipeline.
pub unsafe trait PipelineLayout: PipelineLayoutDesc + 'static + Send + Sync {
    /// Returns the inner `UnsafePipelineLayout`.
    // TODO: should be named "inner()" after https://github.com/rust-lang/rust/issues/12808 is fixed
    fn inner_pipeline_layout(&self) -> &UnsafePipelineLayout;
}

/// Trait for objects that describe the layout of the descriptors and push constants of a pipeline.
pub unsafe trait PipelineLayoutDesc {
    /// Iterator that describes descriptor sets.
    type SetsIter: Iterator<Item = Self::DescIter>;
    /// Iterator that describes individual descriptors.
    type DescIter: Iterator<Item = DescriptorDesc>;

    /// Describes the layout of the descriptors of the pipeline.
    fn descriptors_desc(&self) -> Self::SetsIter;

    // TODO: describe push constants
}

/// Traits that allow determining whether a pipeline layout is a superset of another one.
///
/// This trait is automatically implemented on all types that implement `PipelineLayout`.
/// TODO: once specialization lands, we can add implementations that don't perform deep comparisons
pub unsafe trait PipelineLayoutSuperset<Other>: PipelineLayoutDesc
    where Other: PipelineLayoutDesc
{
    /// Returns true if `self` is a superset of `Other`.
    fn is_superset_of(&self, &Other) -> bool;
}

unsafe impl<T, U> PipelineLayoutSuperset<U> for T
    where T: PipelineLayoutDesc, U: PipelineLayoutDesc
{
    fn is_superset_of(&self, other: &U) -> bool {
        let mut other_descriptor_sets = other.descriptors_desc();

        for my_set in self.descriptors_desc() {
            let mut other_set = match other_descriptor_sets.next() {
                None => return false,
                Some(s) => s,
            };

            for my_desc in my_set {
                let other_desc = match other_set.next() {
                    None => return false,
                    Some(d) => d,
                };

                if !my_desc.is_superset_of(&other_desc) {
                    return false;
                }
            }
        }

        true
    }
}

/// Traits that allow determining whether 
pub unsafe trait PipelineLayoutSetsCompatible<Other>: PipelineLayoutDesc
    where Other: DescriptorSetsCollection
{
    /// Returns true if `Other` can be used with a pipeline that uses `self` as layout.
    fn is_compatible(&self, &Other) -> bool;
}

unsafe impl<T, U> PipelineLayoutSetsCompatible<U> for T
    where T: PipelineLayoutDesc, U: DescriptorSetsCollection
{
    fn is_compatible(&self, sets: &U) -> bool {
        let mut other_descriptor_sets = DescriptorSetsCollection::description(sets);

        for my_set in self.descriptors_desc() {
            let mut other_set = match other_descriptor_sets.next() {
                None => return false,
                Some(s) => s,
            };

            for my_desc in my_set {
                let other_desc = match other_set.next() {
                    None => return false,
                    Some(d) => d,
                };

                if !my_desc.is_superset_of(&other_desc) {
                    return false;
                }
            }
        }

        true
    }
}

/// Traits that allow determining whether 
// TODO: require a trait on Pc
pub unsafe trait PipelineLayoutPushConstantsCompatible<Pc>: PipelineLayout {
    /// Returns true if `Pc` can be used with a pipeline that uses `self` as layout.
    fn is_compatible(&self, &Pc) -> bool;
}

unsafe impl<T, U> PipelineLayoutPushConstantsCompatible<U> for T where T: PipelineLayout {
    fn is_compatible(&self, _: &U) -> bool {
        // FIXME:
        true
    }
}
