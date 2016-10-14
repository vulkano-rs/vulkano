// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Provides a way for shaders to access the content of buffers and images, or read arbitrary data.
//!
//! If you except vertex attributes, there are two ways in Vulkan to pass data to a shader:
//!
//! - You can pass a very small amount of data (only a guaranteed 128 bytes) through the *push
//!   constants* mechanism. Push constants are the fastest and easiest way to pass data.
//! - You can make your shader read from a buffer or an image by binding it to a *descriptor*.
//!
//! Here is an example fragment shader in GLSL that uses both:
//!
//! ```ignore
//! #version 450
//!
//! // This is a descriptor that contains a texture.
//! layout(set = 0, binding = 0) uniform sampler2D u_texture;
//!
//! layout(push_constant) uniform PushConstants {
//!     // This is a push constant.
//!     float opacity;
//! } push_constants;
//!
//! layout(location = 0) in vec2 v_tex_coords;
//! layout(location = 0) out vec4 f_output;
//!
//! void main() {
//!     f_output.rgb = texture(u_texture, v_tex_coords).rgb;
//!     f_output.a = push_constants.opacity;
//! }
//! ```
//!
//! # Descriptors
//!
//! In order to read the content of a buffer or an image from a shader, that buffer or image
//! must be put in a *descriptor*. Each descriptor contains one buffer or one image alongside with
//! the way that it can be accessed. A descriptor can also be an array, in which case it contains
//! multiple buffers or images with the same layout.
//!
//! Descriptors are grouped in what is called *descriptor sets*. In Vulkan you don't bind
//! individual descriptors one by one, but you create then bind descriptor sets one by one. You are
//! therefore encouraged to put descriptors that are often used together in the same set.
//!
//! The layout of all the descriptors and the push constants used by all the stages of a graphics
//! or compute pipeline is grouped in a *pipeline layout* object.
//!
//! # Pipeline initialization
//!
//! When you build a pipeline object (a `GraphicsPipeline` or a `ComputePipeline`), you have to
//! pass a reference to a struct that implements the `PipelineLayoutRef` trait. This object will
//! describe to the Vulkan implementation the types and layouts of the descriptors and push
//! constants that are going to be accessed by the shaders of the pipeline.
//!
//! The `PipelineLayoutRef` trait is unsafe. You are encouraged not to implemented it yourself, but
//! instead use the `pipeline_layout!` macro, which will generate a struct that implements this
//! trait for you.
//!
//! Here is an example usage:
//!
//! ```ignore       // TODO: make it pass doctests
//! mod pipeline_layout {
//!     pipeline_layout!{
//!         push_constants: {
//!             opacity: f32
//!         },
//!         set0: {
//!             u_texture: CombinedImageSampler
//!         }
//!     }
//! }
//!
//! let _pipeline_layout = pipeline_layout::CustomPipeline::new(&device).unwrap();
//! ```
//!
//! # When drawing
//!
//! When you call a function that adds a draw command to a command buffer, one of the parameters
//! corresponds to the list of descriptor sets to use, and another parameter contains the push
//! constants. Vulkano will check that what you passed is compatible with the pipeline layout that
//! you used when creating the pipeline.
//!
//! It is encouraged, but not mandatory, that the descriptor sets you pass when drawing were
//! created from the same pipeline layout as the one you create the graphics or compute pipeline
//! with.
//!
//! Descriptor sets have to be created in advance from a descriptor pool. You can use the same
//! descriptor set multiple time with multiple draw commands, and keep alive descriptor sets
//! between frames. Creating a descriptor set is quite cheap, so it won't kill your performances
//! to create new sets at each frame.
//!
//! TODO: talk about perfs of changing sets

pub use self::descriptor_set::DescriptorSet;
pub use self::pipeline_layout::PipelineLayoutRef;

pub mod descriptor;
pub mod descriptor_set;
pub mod pipeline_layout;
