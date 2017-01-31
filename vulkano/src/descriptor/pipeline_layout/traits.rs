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
use descriptor::descriptor::ShaderStages;
use descriptor::descriptor_set::DescriptorSetsCollection;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use descriptor::pipeline_layout::PipelineLayout;
use descriptor::pipeline_layout::PipelineLayoutDescUnion;
use descriptor::pipeline_layout::PipelineLayoutSys;
use descriptor::pipeline_layout::PipelineLayoutCreationError;
use device::Device;
use SafeDeref;

/// Trait for objects that describe the layout of the descriptors and push constants of a pipeline.
pub unsafe trait PipelineLayoutRef {
    /// Returns an opaque object that allows internal access to the pipeline layout.
    ///
    /// Can be obtained by calling `PipelineLayoutRef::sys()` on the pipeline layout.
    ///
    /// > **Note**: This is an internal function that you normally don't need to call.
    fn sys(&self) -> PipelineLayoutSys;

    /// Returns the description of the pipeline layout.
    ///
    /// Can be obtained by calling `PipelineLayoutRef::desc()` on the pipeline layout.
    // TODO: meh for `PipelineLayoutDescNames instead of `PipelineLayoutDesc`
    fn desc(&self) -> &PipelineLayoutDescNames;

    /// Returns the device that this pipeline layout belongs to.
    ///
    /// Can be obtained by calling `PipelineLayoutRef::device()` on the pipeline layout.
    fn device(&self) -> &Arc<Device>;

    /// Returns the `UnsafeDescriptorSetLayout` object of the specified set index.
    ///
    /// Returns `None` if out of range or if the set is empty for this index.
    fn descriptor_set_layout(&self, index: usize) -> Option<&Arc<UnsafeDescriptorSetLayout>>;
}

unsafe impl<T> PipelineLayoutRef for T where T: SafeDeref, T::Target: PipelineLayoutRef {
    #[inline]
    fn sys(&self) -> PipelineLayoutSys {
        (**self).sys()
    }

    #[inline]
    fn desc(&self) -> &PipelineLayoutDescNames {
        (**self).desc()
    }

    #[inline]
    fn device(&self) -> &Arc<Device> {
        (**self).device()
    }

    #[inline]
    fn descriptor_set_layout(&self, index: usize) -> Option<&Arc<UnsafeDescriptorSetLayout>> {
        (**self).descriptor_set_layout(index)
    }
}

/// Trait for objects that describe the layout of the descriptors and push constants of a pipeline.
pub unsafe trait PipelineLayoutDesc {
    /// Returns the number of sets in the layout. Includes possibly empty sets.
    ///
    /// In other words, this should be equal to the highest set number plus one.
    fn num_sets(&self) -> usize;

    /// Returns the number of descriptors in the set. Includes possibly empty descriptors.
    ///
    /// Returns `None` if the set is out of range.
    fn num_bindings_in_set(&self, set: usize) -> Option<usize>;

    /// Returns the descriptor for the given binding of the given set.
    ///
    /// Returns `None` if out of range.
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc>;

    /// If the `PipelineLayoutDesc` implementation is able to provide an existing
    /// `UnsafeDescriptorSetLayout` for a given set, it can do so by returning it here.
    #[inline]
    fn provided_set_layout(&self, set: usize) -> Option<Arc<UnsafeDescriptorSetLayout>> {
        None
    }

    /// Returns the number of push constant ranges of the layout.
    fn num_push_constants_ranges(&self) -> usize;

    /// Returns a description of the given push constants range.
    ///
    /// Contrary to the descriptors, a push constants range can't be empty.
    ///
    /// Returns `None` if out of range.
    ///
    /// Each bit of `stages` must only be present in a single push constants range of the
    /// description.
    fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange>;

    /// Builds the union of this layout and another.
    #[inline]
    fn union<T>(self, other: T) -> PipelineLayoutDescUnion<Self, T> where Self: Sized {
        PipelineLayoutDescUnion::new(self, other)
    }

    /// Turns the layout description into a `PipelineLayout` object that can be used by Vulkan.
    ///
    /// > **Note**: This is just a shortcut for `PipelineLayout::new`.
    #[inline]
    fn build(self, device: &Arc<Device>)
             -> Result<PipelineLayout<Self>, PipelineLayoutCreationError>
        where Self: Sized
    {
        PipelineLayout::new(device, self)
    }
}

/// Description of a range of the push constants of a pipeline layout.
// TODO: should contain the layout as well
#[derive(Debug, Copy, Clone)]
pub struct PipelineLayoutDescPcRange {
    /// Offset in bytes from the start of the push constants to this range.
    pub offset: usize,
    /// Size in bytes of the range.
    pub size: usize,
    /// The stages which can access this range. Note that the same shader stage can't access two
    /// different ranges.
    pub stages: ShaderStages,
}

unsafe impl<T> PipelineLayoutDesc for T where T: SafeDeref, T::Target: PipelineLayoutDesc {
    #[inline]
    fn num_sets(&self) -> usize {
        (**self).num_sets()
    }

    #[inline]
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        (**self).num_bindings_in_set(set)
    }

    #[inline]
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        (**self).descriptor(set, binding)
    }

    #[inline]
    fn num_push_constants_ranges(&self) -> usize {
        (**self).num_push_constants_ranges()
    }

    #[inline]
    fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
        (**self).push_constants_range(num)
    }
}

/// Extension trait for `PipelineLayoutDesc`. Allows retreiving a descriptor by its name.
pub unsafe trait PipelineLayoutDescNames: PipelineLayoutDesc {
    /// Returns the set ID and descriptor ID within set of the descriptor with the given name.
    ///
    /// Returns `None` if the name was not found.
    fn descriptor_by_name(&self, name: &str) -> Option<(usize, usize)>;
}

unsafe impl<T> PipelineLayoutDescNames for T where T: SafeDeref, T::Target: PipelineLayoutDescNames {
    #[inline]
    fn descriptor_by_name(&self, name: &str) -> Option<(usize, usize)> {
        (**self).descriptor_by_name(name)
    }
}

/// Traits that allow determining whether a pipeline layout is a superset of another one.
///
/// This trait is automatically implemented on all types that implement `PipelineLayoutRef`.
/// TODO: once specialization lands, we can add implementations that don't perform deep comparisons
pub unsafe trait PipelineLayoutSuperset<Other: ?Sized>: PipelineLayoutDesc
    where Other: PipelineLayoutDesc
{
    /// Returns true if `self` is a superset of `Other`.
    // TODO: return a Result instead of a bool
    fn is_superset_of(&self, &Other) -> bool;
}

unsafe impl<T: ?Sized, U: ?Sized> PipelineLayoutSuperset<U> for T
    where T: PipelineLayoutDesc, U: PipelineLayoutDesc
{
    fn is_superset_of(&self, other: &U) -> bool {
        for set_num in 0 .. cmp::max(self.num_sets(), other.num_sets()) {
            let other_num_bindings = other.num_bindings_in_set(set_num).unwrap_or(0);

            if self.num_bindings_in_set(set_num).unwrap_or(0) < other_num_bindings {
                return false;
            }

            for desc_num in 0 .. other_num_bindings {
                match (self.descriptor(set_num, desc_num), other.descriptor(set_num, desc_num)) {
                    (Some(mine), Some(other)) => {
                        if !mine.is_superset_of(&other) {
                            return false;
                        }
                    },
                    (None, Some(_)) => return false,
                    _ => ()
                }
            }
        }

        // FIXME: check push constants

        true
    }
}

/// Traits that allow determining whether 
pub unsafe trait PipelineLayoutSetsCompatible<Other: ?Sized>: PipelineLayoutDesc
    where Other: DescriptorSetsCollection
{
    /// Returns true if `Other` can be used with a pipeline that uses `self` as layout.
    fn is_compatible(&self, &Other) -> bool;
}

unsafe impl<T: ?Sized, U: ?Sized> PipelineLayoutSetsCompatible<U> for T
    where T: PipelineLayoutDesc, U: DescriptorSetsCollection
{
    fn is_compatible(&self, sets: &U) -> bool {
        /*let mut other_descriptor_sets = DescriptorSetsCollection::description(sets);

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
        }*/

        // FIXME: 
        true
    }
}

/// Traits that allow determining whether 
// TODO: require a trait on Pc
pub unsafe trait PipelineLayoutPushConstantsCompatible<Pc: ?Sized>: PipelineLayoutDesc {
    /// Returns true if `Pc` can be used with a pipeline that uses `self` as layout.
    fn is_compatible(&self, &Pc) -> bool;
}

unsafe impl<T: ?Sized, U: ?Sized> PipelineLayoutPushConstantsCompatible<U> for T
    where T: PipelineLayoutDesc
{
    fn is_compatible(&self, _: &U) -> bool {
        // FIXME:
        true
    }
}
