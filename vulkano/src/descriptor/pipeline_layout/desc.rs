// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::descriptor::descriptor::DescriptorBufferDesc;
use crate::descriptor::descriptor::DescriptorDesc;
use crate::descriptor::descriptor::DescriptorDescSupersetError;
use crate::descriptor::descriptor::DescriptorDescTy;
use crate::descriptor::descriptor::ShaderStages;
use crate::descriptor::descriptor_set::DescriptorSetsCollection;
use crate::descriptor::pipeline_layout::limits_check;
use crate::device::Device;
use fnv::FnvHashSet;
use std::cmp;
use std::error;
use std::fmt;

/// Object that describes the layout of the descriptors and push constants of a pipeline.
#[derive(Debug, Clone)]
pub struct PipelineLayoutDesc {
    descriptor_sets: Vec<Vec<Option<DescriptorDesc>>>,
    push_constants: Vec<PipelineLayoutDescPcRange>,
}

impl PipelineLayoutDesc {
    /// Builds a new `PipelineLayoutDesc` from the descriptors and push constants descriptions.
    pub fn new(
        descriptor_sets: Vec<Vec<Option<DescriptorDesc>>>,
        push_constants: Vec<PipelineLayoutDescPcRange>,
    ) -> Result<PipelineLayoutDesc, RuntimePipelineDescError> {
        unsafe {
            for (a_id, a) in push_constants.iter().enumerate() {
                for b in push_constants.iter().skip(a_id + 1) {
                    if a.offset <= b.offset && a.offset + a.size > b.offset {
                        return Err(RuntimePipelineDescError::PushConstantsConflict {
                            first_offset: a.offset,
                            first_size: a.size,
                            second_offset: b.offset,
                        });
                    }

                    if b.offset <= a.offset && b.offset + b.size > a.offset {
                        return Err(RuntimePipelineDescError::PushConstantsConflict {
                            first_offset: b.offset,
                            first_size: b.size,
                            second_offset: a.offset,
                        });
                    }
                }
            }

            Ok(Self::new_unchecked(descriptor_sets, push_constants))
        }
    }

    /// Builds a new `PipelineLayoutDesc` from the descriptors and push constants descriptions.
    ///
    /// # Safety
    ///
    /// The provided push constants must not conflict with each other.
    #[inline]
    pub unsafe fn new_unchecked(
        descriptor_sets: Vec<Vec<Option<DescriptorDesc>>>,
        push_constants: Vec<PipelineLayoutDescPcRange>,
    ) -> PipelineLayoutDesc {
        PipelineLayoutDesc {
            descriptor_sets,
            push_constants,
        }
    }

    /// Creates a description of an empty pipeline layout description, with no descriptor sets or
    /// push constants.
    #[inline]
    pub const fn empty() -> PipelineLayoutDesc {
        PipelineLayoutDesc {
            descriptor_sets: Vec::new(),
            push_constants: Vec::new(),
        }
    }

    /// Returns a description of the descriptor sets.
    #[inline]
    pub fn descriptor_sets(&self) -> &[Vec<Option<DescriptorDesc>>] {
        &self.descriptor_sets
    }

    /// Returns a description of the push constants.
    #[inline]
    pub fn push_constants(&self) -> &[PipelineLayoutDescPcRange] {
        &self.push_constants
    }

    #[inline]
    fn descriptor(&self, set_num: usize, binding_num: usize) -> Option<&DescriptorDesc> {
        self.descriptor_sets
            .get(set_num)
            .and_then(|b| b.get(binding_num).and_then(|b| b.as_ref()))
    }

    /// Transforms a `PipelineLayoutDesc`.
    ///
    /// Used to adjust automatically inferred `PipelineLayoutDesc`s with information that cannot be inferred.
    pub fn tweak<I>(&mut self, dynamic_buffers: I)
    where
        I: IntoIterator<Item = (usize, usize)>,
    {
        let dynamic_buffers: FnvHashSet<(usize, usize)> = dynamic_buffers.into_iter().collect();
        for &(set_num, binding_num) in &dynamic_buffers {
            debug_assert!(
                self.descriptor(set_num, binding_num)
                    .map_or(false, |desc| match desc.ty {
                        DescriptorDescTy::Buffer(_) => true,
                        _ => false,
                    }),
                "tried to make the non-buffer descriptor at set {} binding {} a dynamic buffer",
                set_num,
                binding_num
            );
        }

        for (set_num, set) in self.descriptor_sets.iter_mut().enumerate() {
            for (binding_num, binding) in set.iter_mut().enumerate() {
                if dynamic_buffers.contains(&(set_num, binding_num)) {
                    if let Some(desc) = binding {
                        if let DescriptorDescTy::Buffer(ref buffer_desc) = desc.ty {
                            desc.ty = DescriptorDescTy::Buffer(DescriptorBufferDesc {
                                dynamic: Some(true),
                                ..*buffer_desc
                            });
                        }
                    }
                }
            }
        }
    }

    /// Builds the union of this layout description and another.
    #[inline]
    pub fn union(&self, other: &PipelineLayoutDesc) -> PipelineLayoutDesc {
        unsafe {
            let descriptor_sets = {
                let self_sets = self.descriptor_sets();
                let other_sets = other.descriptor_sets();
                let num_sets = cmp::max(self_sets.len(), other_sets.len());

                (0..num_sets)
                    .map(|set_num| {
                        let self_bindings = self_sets.get(set_num);
                        let other_bindings = other_sets.get(set_num);
                        let num_bindings = cmp::max(
                            self_bindings.map(|s| s.len()).unwrap_or(0),
                            other_bindings.map(|s| s.len()).unwrap_or(0),
                        );

                        (0..num_bindings)
                            .map(|binding_num| {
                                let self_desc = self.descriptor(set_num, binding_num);
                                let other_desc = other.descriptor(set_num, binding_num);

                                match (self_desc, other_desc) {
                                    (Some(a), Some(b)) => {
                                        Some(a.union(&b).expect("Can't be union-ed"))
                                    }
                                    (Some(a), None) => Some(a.clone()),
                                    (None, Some(b)) => Some(b.clone()),
                                    (None, None) => None,
                                }
                            })
                            .collect()
                    })
                    .collect()
            };

            let push_constants = self
                .push_constants
                .iter()
                .map(|&(mut new_range)| {
                    // Find the ranges in `other` that share the same stages as `self_range`.
                    // If there is a range with a similar stage in `other`, then adjust the offset
                    // and size to include it.
                    for other_range in &other.push_constants {
                        if !other_range.stages.intersects(&new_range.stages) {
                            continue;
                        }

                        if other_range.offset < new_range.offset {
                            new_range.size += new_range.offset - other_range.offset;
                            new_range.size = cmp::max(new_range.size, other_range.size);
                            new_range.offset = other_range.offset;
                        } else if other_range.offset > new_range.offset {
                            new_range.size = cmp::max(
                                new_range.size,
                                other_range.size + (other_range.offset - new_range.offset),
                            );
                        }
                    }

                    new_range
                })
                .chain(
                    // Add the ones from `other` that were filtered out previously.
                    other
                        .push_constants
                        .iter()
                        .filter(|other_range| {
                            self.push_constants.iter().all(|self_range| {
                                !other_range.stages.intersects(&self_range.stages)
                            })
                        })
                        .map(|range| *range),
                )
                .collect();

            PipelineLayoutDesc::new_unchecked(descriptor_sets, push_constants)
        }
    }

    /// Checks whether this description fulfills the device limits requirements.
    #[inline]
    pub fn check_against_limits(
        &self,
        device: &Device,
    ) -> Result<(), limits_check::PipelineLayoutLimitsError> {
        limits_check::check_desc_against_limits(self, device.physical_device().limits())
    }

    /// Makes sure that `self` is a superset of `other`. Returns an `Err` if this is not the case.
    pub fn ensure_superset_of(
        &self,
        other: &PipelineLayoutDesc,
    ) -> Result<(), PipelineLayoutNotSupersetError> {
        let self_sets = self.descriptor_sets();
        let other_sets = other.descriptor_sets();

        for set_num in 0..cmp::max(self_sets.len(), other_sets.len()) {
            let self_bindings = self_sets.get(set_num);
            let other_bindings = other_sets.get(set_num);
            let self_num_bindings = self_bindings.map(|s| s.len()).unwrap_or(0);
            let other_num_bindings = other_bindings.map(|s| s.len()).unwrap_or(0);

            if self_num_bindings < other_num_bindings {
                return Err(PipelineLayoutNotSupersetError::DescriptorsCountMismatch {
                    set_num: set_num as u32,
                    self_num_descriptors: self_num_bindings as u32,
                    other_num_descriptors: other_num_bindings as u32,
                });
            }

            for binding_num in 0..other_num_bindings {
                let self_desc = self.descriptor(set_num, binding_num);
                let other_desc = self.descriptor(set_num, binding_num);

                match (self_desc, other_desc) {
                    (Some(mine), Some(other)) => {
                        if let Err(err) = mine.is_superset_of(&other) {
                            return Err(PipelineLayoutNotSupersetError::IncompatibleDescriptors {
                                error: err,
                                set_num: set_num as u32,
                                binding_num: binding_num as u32,
                            });
                        }
                    }
                    (None, Some(_)) => {
                        return Err(PipelineLayoutNotSupersetError::ExpectedEmptyDescriptor {
                            set_num: set_num as u32,
                            binding_num: binding_num as u32,
                        })
                    }
                    _ => (),
                }
            }
        }

        // FIXME: check push constants

        Ok(())
    }

    /// Returns true if `sets` can be used with a pipeline that uses `self` as layout.
    pub fn is_descriptor_sets_compatible<S>(&self, sets: &S) -> bool
    where
        S: DescriptorSetsCollection,
    {
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

    /// Returns true if `constants` can be used with a pipeline that uses `self` as layout.
    pub fn is_push_constants_compatible<Pc>(&self, constants: &Pc) -> bool
    where
        Pc: ?Sized,
    {
        // FIXME:
        true
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

/// Error when building a persistent descriptor set.
#[derive(Debug, Clone)]
pub enum RuntimePipelineDescError {
    /// Conflict between different push constants ranges.
    PushConstantsConflict {
        first_offset: usize,
        first_size: usize,
        second_offset: usize,
    },
}

impl error::Error for RuntimePipelineDescError {}

impl fmt::Display for RuntimePipelineDescError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                RuntimePipelineDescError::PushConstantsConflict { .. } => {
                    "conflict between different push constants ranges"
                }
            }
        )
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
    ExpectedEmptyDescriptor { set_num: u32, binding_num: u32 },

    /// Two descriptors are incompatible.
    IncompatibleDescriptors {
        error: DescriptorDescSupersetError,
        set_num: u32,
        binding_num: u32,
    },
}

impl error::Error for PipelineLayoutNotSupersetError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            PipelineLayoutNotSupersetError::IncompatibleDescriptors { ref error, .. } => {
                Some(error)
            }
            _ => None,
        }
    }
}

impl fmt::Display for PipelineLayoutNotSupersetError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                PipelineLayoutNotSupersetError::DescriptorsCountMismatch { .. } => {
                    "there are more descriptors in the child than in the parent layout"
                }
                PipelineLayoutNotSupersetError::ExpectedEmptyDescriptor { .. } => {
                    "expected an empty descriptor, but got something instead"
                }
                PipelineLayoutNotSupersetError::IncompatibleDescriptors { .. } => {
                    "two descriptors are incompatible"
                }
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::descriptor::descriptor::ShaderStages;
    use crate::descriptor::pipeline_layout::PipelineLayoutDesc;
    use crate::descriptor::pipeline_layout::PipelineLayoutDescPcRange;
    use crate::descriptor::pipeline_layout::RuntimePipelineDescError;

    #[test]
    fn pc_conflict() {
        let r = PipelineLayoutDesc::new(
            vec![],
            vec![
                PipelineLayoutDescPcRange {
                    offset: 0,
                    size: 8,
                    stages: ShaderStages::all(),
                },
                PipelineLayoutDescPcRange {
                    offset: 4,
                    size: 8,
                    stages: ShaderStages::all(),
                },
            ],
        );

        assert!(matches!(
            r,
            Err(RuntimePipelineDescError::PushConstantsConflict {
                first_offset: 0,
                first_size: 8,
                second_offset: 4,
            })
        ));
    }
}
