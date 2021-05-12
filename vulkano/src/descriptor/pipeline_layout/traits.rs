// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::PipelineLayoutDesc;
use crate::descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use crate::descriptor::pipeline_layout::PipelineLayoutSys;
use crate::device::DeviceOwned;
use crate::SafeDeref;
use std::sync::Arc;

/// Trait for objects that describe the layout of the descriptors and push constants of a pipeline.
pub unsafe trait PipelineLayoutAbstract: DeviceOwned {
    /// Returns an opaque object that allows internal access to the pipeline layout.
    ///
    /// Can be obtained by calling `PipelineLayoutAbstract::sys()` on the pipeline layout.
    ///
    /// > **Note**: This is an internal function that you normally don't need to call.
    fn sys(&self) -> PipelineLayoutSys;

    /// Returns the description of the pipeline layout.
    fn desc(&self) -> &PipelineLayoutDesc;

    /// Returns the `UnsafeDescriptorSetLayout` object of the specified set index.
    ///
    /// Returns `None` if out of range or if the set is empty for this index.
    fn descriptor_set_layout(&self, index: usize) -> Option<&Arc<UnsafeDescriptorSetLayout>>;
}

unsafe impl<T> PipelineLayoutAbstract for T
where
    T: SafeDeref,
    T::Target: PipelineLayoutAbstract,
{
    #[inline]
    fn sys(&self) -> PipelineLayoutSys {
        (**self).sys()
    }

    #[inline]
    fn desc(&self) -> &PipelineLayoutDesc {
        (**self).desc()
    }

    #[inline]
    fn descriptor_set_layout(&self, index: usize) -> Option<&Arc<UnsafeDescriptorSetLayout>> {
        (**self).descriptor_set_layout(index)
    }
}
