// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Describes the layout of descriptors and push constants used by a pipeline.
//! 
//! This module contains all the structs and traits related to describing the layout of
//! descriptors and push constants used by the shaders of a graphics or compute pipeline.
//! 
//! The layout itself only **describes** the descriptors and push constants, and does not contain
//! the content of the push constants or the actual list of resources that are going to be available
//! through the descriptors. Push constants are set when you submit a draw command, and the list
//! of resources is set by creating *descriptor set* objects and passing these sets when you
//! submit a draw command.

pub use self::empty::EmptyPipelineDesc;
pub use self::sys::PipelineLayout;
pub use self::sys::PipelineLayoutCreationError;
pub use self::sys::PipelineLayoutSys;
pub use self::traits::PipelineLayoutRef;
pub use self::traits::PipelineLayoutDesc;
pub use self::traits::PipelineLayoutSuperset;
pub use self::traits::PipelineLayoutSetsCompatible;
pub use self::traits::PipelineLayoutPushConstantsCompatible;

pub mod custom_pipeline_macro;

mod empty;
mod sys;
mod traits;
