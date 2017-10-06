// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! A pipeline layout describes the layout of descriptors and push constants used by a graphics
//! pipeline or a compute pipeline.
//!
//! The layout itself only *describes* the descriptors and push constants, and does not contain
//! the content of the push constants or the actual list of resources that are going to be
//! available through the descriptors. Push constants are set when you submit a draw command, and
//! the list of resources is set by creating *descriptor set* objects and passing these sets when
//! you submit a draw command.
//!
//! # Pipeline layout objects
//!
//! A pipeline layout is something that you must describe to the Vulkan implementation by creating
//! a **pipeline layout object**, represented by the `PipelineLayout` struct in vulkano.
//!
//! Each graphics pipeline or compute pipeline that you create therefore holds a
//! **pipeline layout object** By default, creating a pipeline automatically builds a new pipeline
//! layout object describing the union of all the descriptors and push constants of all the shaders
//! used by the pipeline.
//!
//! The `PipelineLayout` struct describes the pipeline layout to both the Vulkan implementation and
//! to vulkano. It holds a template parameter whose type must implement the `PipelineLayoutDesc`
//! trait.
//!
//! # The PipelineLayoutAbstract trait
//!
//! All the functions in vulkano that operate on pipeline layout objects (for example, creating a
//! descriptor set) do not take directly a `PipelineLayout` struct as parameter. Instead they can
//! take any object that implements the `PipelineLayoutAbstract` trait.
//!
//! This trait represents any object that holds a `PipelineLayout`. It is implemented on the
//! `PipelineLayout` struct itself (obviously), but also notably on `GraphicsPipeline` and
//! `ComputePipeline`. In other words, you can for example create a descriptor set by passing a
//! graphics pipeline as parameter.
//!
//! # Custom pipeline layouts
//!
//! In some situations, it is better (as in, faster) to share the same descriptor set or sets
//! between multiple pipelines that each use different descriptors. To do so, you have to create a
//! pipeline layout object in advance and pass it when you create the pipelines.
//!
//! TODO: write this section

pub use self::empty::EmptyPipelineDesc;
pub use self::limits_check::PipelineLayoutLimitsError;
pub use self::runtime_desc::RuntimePipelineDesc;
pub use self::runtime_desc::RuntimePipelineDescError;
pub use self::sys::PipelineLayout;
pub use self::sys::PipelineLayoutCreationError;
pub use self::sys::PipelineLayoutSys;
pub use self::traits::PipelineLayoutAbstract;
pub use self::traits::PipelineLayoutDesc;
pub use self::traits::PipelineLayoutDescPcRange;
pub use self::traits::PipelineLayoutNotSupersetError;
pub use self::traits::PipelineLayoutPushConstantsCompatible;
pub use self::traits::PipelineLayoutSetsCompatible;
pub use self::traits::PipelineLayoutSuperset;
pub use self::union::PipelineLayoutDescUnion;

mod empty;
mod limits_check;
mod runtime_desc;
mod sys;
mod traits;
mod union;
