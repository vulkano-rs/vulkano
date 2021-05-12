// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::descriptor::descriptor::DescriptorDesc;
use crate::descriptor::pipeline_layout::PipelineLayoutDesc;
use crate::descriptor::pipeline_layout::PipelineLayoutDescPcRange;
use std::error;
use std::fmt;

/// Runtime description of a pipeline layout.
#[derive(Debug, Clone)]
pub struct RuntimePipelineDesc {
    descriptor_sets: Vec<Vec<Option<DescriptorDesc>>>,
    push_constants: Vec<PipelineLayoutDescPcRange>,
}

impl RuntimePipelineDesc {
    /// Builds a new `RuntimePipelineDesc` from the descriptors and push constants descriptions.
    pub fn new(
        descriptor_sets: Vec<Vec<Option<DescriptorDesc>>>,
        push_constants: Vec<PipelineLayoutDescPcRange>,
    ) -> Result<RuntimePipelineDesc, RuntimePipelineDescError> {
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

    /// Builds a new `RuntimePipelineDesc` from the descriptors and push constants descriptions.
    ///
    /// # Safety
    ///
    /// The provided push constants must not conflict with each other.
    pub unsafe fn new_unchecked(
        descriptor_sets: Vec<Vec<Option<DescriptorDesc>>>,
        push_constants: Vec<PipelineLayoutDescPcRange>,
    ) -> RuntimePipelineDesc {
        RuntimePipelineDesc {
            descriptor_sets,
            push_constants,
        }
    }

    /// Creates a description of an empty pipeline layout description, with no descriptor sets or
    /// push constants.
    pub const fn empty() -> RuntimePipelineDesc {
        RuntimePipelineDesc {
            descriptor_sets: Vec::new(),
            push_constants: Vec::new(),
        }
    }
}

unsafe impl PipelineLayoutDesc for RuntimePipelineDesc {
    #[inline]
    fn num_sets(&self) -> usize {
        self.descriptor_sets.len()
    }

    #[inline]
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        self.descriptor_sets.get(set).map(|s| s.len())
    }

    #[inline]
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        self.descriptor_sets
            .get(set)
            .and_then(|s| s.get(binding).cloned().unwrap_or(None))
    }

    #[inline]
    fn num_push_constants_ranges(&self) -> usize {
        self.push_constants.len()
    }

    #[inline]
    fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
        self.push_constants.get(num).cloned()
    }
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

#[cfg(test)]
mod tests {
    use crate::descriptor::descriptor::ShaderStages;
    use crate::descriptor::pipeline_layout::PipelineLayoutDescPcRange;
    use crate::descriptor::pipeline_layout::RuntimePipelineDesc;
    use crate::descriptor::pipeline_layout::RuntimePipelineDescError;

    #[test]
    fn pc_conflict() {
        let r = RuntimePipelineDesc::new(
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
