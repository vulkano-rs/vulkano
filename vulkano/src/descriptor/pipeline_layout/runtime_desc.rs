// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;

use smallvec::SmallVec;

use descriptor::descriptor::DescriptorDesc;
use descriptor::pipeline_layout::PipelineLayoutDesc;
use descriptor::pipeline_layout::PipelineLayoutDescPcRange;

/// Runtime description of a pipeline layout.
#[derive(Debug, Clone)]
pub struct RuntimePipelineDesc {
    descriptors: SmallVec<[SmallVec<[Option<DescriptorDesc>; 5]>; 3]>,
    push_constants: SmallVec<[PipelineLayoutDescPcRange; 6]>,
}

impl RuntimePipelineDesc {
    /// Builds a new `RuntimePipelineDesc` from the descriptors and push constants descriptions.
    pub fn new<TSetsIter, TPushConstsIter, TDescriptorsIter>(
        desc: TSetsIter, push_constants: TPushConstsIter)
        -> Result<RuntimePipelineDesc, RuntimePipelineDescError>
        where TSetsIter: IntoIterator<Item = TDescriptorsIter>,
              TDescriptorsIter: IntoIterator<Item = Option<DescriptorDesc>>,
              TPushConstsIter: IntoIterator<Item = PipelineLayoutDescPcRange>
    {
        let descriptors = desc.into_iter().map(|s| s.into_iter().collect()).collect();
        let push_constants: SmallVec<[PipelineLayoutDescPcRange; 6]> =
            push_constants.into_iter().collect();

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

        Ok(RuntimePipelineDesc {
               descriptors,
               push_constants,
           })
    }
}

unsafe impl PipelineLayoutDesc for RuntimePipelineDesc {
    #[inline]
    fn num_sets(&self) -> usize {
        self.descriptors.len()
    }

    #[inline]
    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        self.descriptors.get(set).map(|s| s.len())
    }

    #[inline]
    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        self.descriptors
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

impl error::Error for RuntimePipelineDescError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            RuntimePipelineDescError::PushConstantsConflict { .. } => {
                "conflict between different push constants ranges"
            },
        }
    }
}

impl fmt::Display for RuntimePipelineDescError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}

#[cfg(test)]
mod tests {
    use descriptor::descriptor::DescriptorDesc;
    use descriptor::descriptor::ShaderStages;
    use descriptor::pipeline_layout::PipelineLayoutDescPcRange;
    use descriptor::pipeline_layout::RuntimePipelineDesc;
    use descriptor::pipeline_layout::RuntimePipelineDescError;
    use std::iter;

    #[test]
    fn pc_conflict() {
        let range1 = PipelineLayoutDescPcRange {
            offset: 0,
            size: 8,
            stages: ShaderStages::all(),
        };

        let range2 = PipelineLayoutDescPcRange {
            offset: 4,
            size: 8,
            stages: ShaderStages::all(),
        };

        let r = RuntimePipelineDesc::new::<_, _, iter::Empty<Option<DescriptorDesc>>>
                                    (iter::empty(), iter::once(range1).chain(iter::once(range2)));

        match r {
            Err(RuntimePipelineDescError::PushConstantsConflict {
                    first_offset: 0,
                    first_size: 8,
                    second_offset: 4,
                }) => (),
            _ => panic!(),   // test failed
        }
    }
}
