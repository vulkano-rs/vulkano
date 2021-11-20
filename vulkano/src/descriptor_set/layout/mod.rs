// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Describes the layout of all descriptors within a descriptor set.
//!
//! When creating a new descriptor set, you must provide a *layout* object to create it from. You
//! can create a descriptor set layout manually, but it is normally created automatically by each
//! pipeline layout.

pub use self::desc::DescriptorDesc;
pub use self::desc::DescriptorRequirementsNotMet;
pub use self::desc::DescriptorSetDesc;
pub use self::desc::DescriptorType;
pub use self::sys::DescriptorSetLayout;
pub use self::sys::DescriptorSetLayoutError;

mod desc;
mod sys;
