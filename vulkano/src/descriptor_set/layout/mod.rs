// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

pub use self::desc::DescriptorBufferDesc;
pub use self::desc::DescriptorDesc;
pub use self::desc::DescriptorDescSupersetError;
pub use self::desc::DescriptorDescTy;
pub use self::desc::DescriptorImageDesc;
pub use self::desc::DescriptorImageDescArray;
pub use self::desc::DescriptorImageDescDimensions;
pub use self::desc::DescriptorSetDesc;
pub use self::desc::DescriptorType;
pub use self::desc::ShaderStages;
pub use self::desc::ShaderStagesSupersetError;
pub use self::sys::DescriptorSetLayout;

mod desc;
mod sys;
