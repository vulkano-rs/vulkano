// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

pub use self::collection::DescriptorSetsCollection;
pub use self::pool::DescriptorPool;
pub use self::sys::UnsafeDescriptorSet;
pub use self::unsafe_layout::UnsafeDescriptorSetLayout;

mod collection;
mod pool;
mod sys;
mod unsafe_layout;

pub unsafe trait DescriptorSet: 'static + Send + Sync {
    /// Returns the inner `UnsafeDescriptorSet`.
    // TODO: should be named "inner()" after https://github.com/rust-lang/rust/issues/12808 is fixed
    fn inner_descriptor_set(&self) -> &UnsafeDescriptorSet;
}
