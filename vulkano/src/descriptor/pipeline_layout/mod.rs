// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

pub use self::empty::EmptyPipeline;
pub use self::sys::UnsafePipelineLayout;
pub use self::traits::PipelineLayout;
pub use self::traits::PipelineLayoutDesc;
pub use self::traits::PipelineLayoutSuperset;
pub use self::traits::PipelineLayoutSetsCompatible;
pub use self::traits::PipelineLayoutPushConstantsCompatible;

pub mod custom_pipeline_macro;
pub mod empty;

mod sys;
mod traits;
