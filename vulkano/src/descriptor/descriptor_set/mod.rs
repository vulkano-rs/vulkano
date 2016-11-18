// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use command_buffer::cmd::CommandsListSink;
use descriptor::descriptor::DescriptorDesc;

pub use self::collection::DescriptorSetsCollection;
pub use self::collection::TrackedDescriptorSetsCollection;
pub use self::pool::DescriptorPool;
pub use self::simple::*;
pub use self::sys::UnsafeDescriptorSet;
pub use self::sys::DescriptorWrite;
pub use self::unsafe_layout::UnsafeDescriptorSetLayout;

pub mod collection;

mod pool;
mod simple;
mod sys;
mod unsafe_layout;

/// Trait for objects that contain a collection of resources that will be accessible by shaders.
///
/// Objects of this type can be passed when submitting a draw command.
pub unsafe trait DescriptorSet {
    /// Returns the inner `UnsafeDescriptorSet`.
    fn inner(&self) -> &UnsafeDescriptorSet;
}

unsafe impl<T> DescriptorSet for Arc<T> where T: DescriptorSet {
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        (**self).inner()
    }
}

unsafe impl<'a, T> DescriptorSet for &'a T where T: 'a + DescriptorSet {
    #[inline]
    fn inner(&self) -> &UnsafeDescriptorSet {
        (**self).inner()
    }
}

/// Trait for objects that describe the layout of the descriptors of a set.
pub unsafe trait DescriptorSetDesc {
    /// Iterator that describes individual descriptors.
    type Iter: ExactSizeIterator<Item = DescriptorDesc>;

    /// Describes the layout of the descriptors of the pipeline.
    fn desc(&self) -> Self::Iter;
}

// TODO: re-read docs
/// Extension trait for descriptor sets so that it can be used with the standard commands list
/// interface.
pub unsafe trait TrackedDescriptorSet: DescriptorSet {
    fn add_transition<'a>(&'a self, &mut CommandsListSink<'a>);
}
