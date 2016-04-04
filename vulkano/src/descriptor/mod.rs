// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Collection of data and resources accessed by the shaders.
//!
//! In order to read the content of a buffer or an image from a shader, that buffer or image
//! must be put in a *descriptor*. Each descriptor contains one or several buffers or images
//! alongside with the way that it can be accessed.
//!
//! Descriptors are grouped in what is called *descriptor sets*. In Vulkan you don't bind
//! individual descriptors one by one, but you bind descriptor sets one by one. Therefore you are
//! encouraged to put descriptors that are often used together in the same set.
//!
//! In addition to descriptors, you also have the possibility to feed raw data to your shaders.
//! Each variable that you pass to a shader is called a *push constant*.
//!
//! The layout of all the descriptors and the push constants used by a pipeline is grouped in
//! a *pipeline layout*.
//!
//! # Pipeline initialization
//!
//! When you build a pipeline object (a `GraphicsPipeline` or a `ComputePipeline`), you have to
//! pass a pointer to a struct that implements the `PipelineLayout` trait. This object will
//! describe to the Vulkan implementation the types and layouts of the descriptors and push
//! constants that are going to be accessed by the shaders of the pipeline.
//!
//! The `PipelineLayout` trait is unsafe. You are encouraged not to implemented it yourself, but
//! instead use one of the already-existing implementations that are available to you:
//! 
//! - The shader analyser (from the `vulkano-shaders` crate) will generate an object named
//!   `Layout` for each shader module.
//! - You can merge multiple implementations of `PipelineLayout` into one with the
//!   `merge_pipelines!` macro.
//! - You can create a pipeline layout whose content is only known at runtime with the Foo object.
//!   This is the most costly solution, as vulkano will need to perform runtime checks.
//!
//! # Descriptor sets
//! 
//! When you draw, you have to pass two parameters (two parameters that are relevant here): the
//! list of descriptor sets to use, and the push constants. Vulkano will check that what you
//! passed is compatible with the pipeline layout that you used when creating the pipeline.
//! 
//! Descriptor sets have to be created in advance from a descriptor pool. You can use the same
//! descriptor set multiple time with multiple draw commands, and keep alive descriptor sets
//! between frames. Creating a descriptor set is quite cheap, so it won't kill your performances
//! to create new sets at each frame.
//! 
//! TODO: talk about perfs of changing sets

pub use self::descriptor_set::DescriptorSet;
pub use self::pipeline_layout::PipelineLayout;

pub mod descriptor;
pub mod descriptor_set;
pub mod pipeline_layout;
