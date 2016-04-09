// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Tells the Vulkan implementation how it should perform a graphical or compute operation.
//! 
//! There are two kinds of pipelines:
//! 
//! - Compute pipelines, for compute operations that reads/write raw pixels from images and
//!   buffers.
//! - Graphics pipelines, for graphical operations.
//! 
//! All the sub-modules of this module (with the exception of `cache`) correspond to the various
//! steps of pipelines.

pub use self::compute_pipeline::ComputePipeline;
pub use self::graphics_pipeline::GraphicsPipeline;
pub use self::graphics_pipeline::GraphicsPipelineParams;

mod compute_pipeline;
mod graphics_pipeline;

pub mod blend;
pub mod cache;
pub mod depth_stencil;
pub mod input_assembly;
pub mod multisample;
pub mod raster;
pub mod shader;
pub mod vertex;
pub mod viewport;
