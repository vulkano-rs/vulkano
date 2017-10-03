// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::cmp;
use std::error;
use std::fmt;
use std::sync::Arc;

use SafeDeref;
use descriptor::descriptor::DescriptorDesc;
use descriptor::descriptor::DescriptorDescSupersetError;
use descriptor::descriptor::ShaderStages;
use descriptor::descriptor_set::DescriptorSetsCollection;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use descriptor::pipeline_layout::PipelineLayout;
use descriptor::pipeline_layout::PipelineLayoutCreationError;
use descriptor::pipeline_layout::PipelineLayoutDescUnion;
use descriptor::pipeline_layout::PipelineLayoutSys;
use descriptor::pipeline_layout::limits_check;
use device::Device;
use device::DeviceOwned;

/// Trait for objects that describe the layout of the descriptors and push constants of a pipeline.
pub unsafe trait PipelineLayoutAbstract: PipelineLayoutDesc + DeviceOwned {
    /// Returns an opaque object that allows internal access to the pipeline layout.
    ///
    /// Can be obtained by calling `PipelineLayoutAbstract::sys()` on the pipeline layout.
    ///
    /// > **Note**: This is an internal function that you normally don't need to call.
    fn sys(&self) -> PipelineLayoutSys;

    /// Returns the `UnsafeDescriptorSetLayout` object of the specified set index.
    ///
    /// Returns `None` if out of range or if the set is empty for this index.
    fn descriptor_set_layout(&self, index: usize) -> Option<&Arc<UnsafeDescriptorSetLayout>>;
}

unsafe impl<T> PipelineLayoutAbstract for T
    where T: SafeDeref,
          T::Target: PipelineLayoutAbstract
{
    #[inline]
    fn sys(&self) -> PipelineLayoutSys {
        (**self).sys()
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
    /// Returns `None` if out of range or if the descriptor is empty.
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc>;

    /// If the `PipelineLayoutDesc` implementation is able to provide an existing
    /// `UnsafeDescriptorSetLayout` for a given set, it can do so by returning it here.
    #[inline]
    fn provided_set_layout(&self, _set: usize) -> Option<Arc<UnsafeDescriptorSetLayout>> {
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
    fn union<T>(self, other: T) -> PipelineLayoutDescUnion<Self, T>
        where Self: Sized
    {
        PipelineLayoutDescUnion::new(self, other)
    }

    /// Checks whether this description fulfills the device limits requirements.
    #[inline]
    fn check_against_limits(&self, device: &Device)
                            -> Result<(), limits_check::PipelineLayoutLimitsError> {
        limits_check::check_desc_against_limits(self, device.physical_device().limits())
    }

    /// Turns the layout description into a `PipelineLayout` object that can be used by Vulkan.
    ///
    /// > **Note**: This is just a shortcut for `PipelineLayout::new`.
    #[inline]
    fn build(self, device: Arc<Device>) -> Result<PipelineLayout<Self>, PipelineLayoutCreationError>
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

unsafe impl<T> PipelineLayoutDesc for T
    where T: SafeDeref,
          T::Target: PipelineLayoutDesc
{
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

/// Traits that allow determining whether a pipeline layout is a superset of another one.
///
/// This trait is automatically implemented on all types that implement `PipelineLayoutAbstract`.
/// TODO: once specialization lands, we can add implementations that don't perform deep comparisons
pub unsafe trait PipelineLayoutSuperset<Other: ?Sized>: PipelineLayoutDesc
    where Other: PipelineLayoutDesc
{
    /// Makes sure that `self` is a superset of `Other`. Returns an `Err` if this is not the case.
    fn ensure_superset_of(&self, &Other) -> Result<(), PipelineLayoutNotSupersetError>;
}

unsafe impl<T: ?Sized, U: ?Sized> PipelineLayoutSuperset<U> for T
    where T: PipelineLayoutDesc,
          U: PipelineLayoutDesc
{
    fn ensure_superset_of(&self, other: &U) -> Result<(), PipelineLayoutNotSupersetError> {
        for set_num in 0 .. cmp::max(self.num_sets(), other.num_sets()) {
            let other_num_bindings = other.num_bindings_in_set(set_num).unwrap_or(0);
            let self_num_bindings = self.num_bindings_in_set(set_num).unwrap_or(0);

            if self_num_bindings < other_num_bindings {
                return Err(PipelineLayoutNotSupersetError::DescriptorsCountMismatch {
                               set_num: set_num as u32,
                               self_num_descriptors: self_num_bindings as u32,
                               other_num_descriptors: other_num_bindings as u32,
                           });
            }

            for desc_num in 0 .. other_num_bindings {
                match (self.descriptor(set_num, desc_num), other.descriptor(set_num, desc_num)) {
                    (Some(mine), Some(other)) => {
                        if let Err(err) = mine.is_superset_of(&other) {
                            return Err(PipelineLayoutNotSupersetError::IncompatibleDescriptors {
                                           error: err,
                                           set_num: set_num as u32,
                                           descriptor: desc_num as u32,
                                       });
                        }
                    },
                    (None, Some(_)) =>
                        return Err(PipelineLayoutNotSupersetError::ExpectedEmptyDescriptor {
                                       set_num: set_num as u32,
                                       descriptor: desc_num as u32,
                                   }),
                    _ => (),
                }
            }
        }

        // FIXME: check push constants

        Ok(())
    }
}

/// Error that can happen when creating a graphics pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PipelineLayoutNotSupersetError {
    /// There are more descriptors in the child than in the parent layout.
    DescriptorsCountMismatch {
        set_num: u32,
        self_num_descriptors: u32,
        other_num_descriptors: u32,
    },

    /// Expected an empty descriptor, but got something instead.
    ExpectedEmptyDescriptor { set_num: u32, descriptor: u32 },

    /// Two descriptors are incompatible.
    IncompatibleDescriptors {
        error: DescriptorDescSupersetError,
        set_num: u32,
        descriptor: u32,
    },
}

impl error::Error for PipelineLayoutNotSupersetError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            PipelineLayoutNotSupersetError::DescriptorsCountMismatch { .. } => {
                "there are more descriptors in the child than in the parent layout"
            },
            PipelineLayoutNotSupersetError::ExpectedEmptyDescriptor { .. } => {
                "expected an empty descriptor, but got something instead"
            },
            PipelineLayoutNotSupersetError::IncompatibleDescriptors { .. } => {
                "two descriptors are incompatible"
            },
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            PipelineLayoutNotSupersetError::IncompatibleDescriptors { ref error, .. } => {
                Some(error)
            },
            _ => None,
        }
    }
}

impl fmt::Display for PipelineLayoutNotSupersetError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
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
    where T: PipelineLayoutDesc,
          U: DescriptorSetsCollection
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
pub unsafe trait PipelineLayoutPushConstantsCompatible<Pc: ?Sized>
    : PipelineLayoutDesc {
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
