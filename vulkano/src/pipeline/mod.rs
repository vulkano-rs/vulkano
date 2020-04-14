// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Describes a graphical or compute operation.
//!
//! In Vulkan, before you can add a draw or a compute command to a command buffer you have to
//! create a *pipeline object* that describes this command.
//!
//! When you create a pipeline object, the implementation will usually generate some GPU machine
//! code that will execute the operation (similar to a compiler that generates an executable for
//! the CPU). Consequently it is a CPU-intensive operation that should be performed at
//! initialization or during a loading screen.
//!
//! There are three kinds of pipelines:
//!
//! - `ComputePipeline`s, for compute operations (general-purpose operations that read/write data
//!   in buffers or raw pixels in images).
//! - `GraphicsPipeline`s, for depth-buffer based graphical operations (operations that take
//!   vertices as input and write pixels to a framebuffer).
//! - `RayTracingPipeline`s, for ray tracing based graphical operations (operations that take
//!   acceleration structures and geometry as input and write pixels to a storage image).
//!
//! # Creating a compute pipeline.
//!
//! In order to create a compute pipeline, you first need a *shader entry point*.
//!
//! TODO: write the rest
//! For now vulkano has no "clean" way to create shaders ; everything's a bit hacky
//!
//! # Creating a graphics pipeline
//!
//! A graphics operation takes vertices or vertices and indices as input, and writes pixels to a
//! framebuffer. It consists of multiple steps:
//!
//! - A *shader* named the *vertex shader* is run once for each vertex of the input.
//! - Vertices are assembled into primitives.
//! - Optionally, a shader named the *tessellation control shader* is run once for each primitive
//!   and indicates the tessellation level to apply for this primitive.
//! - Optionally, a shader named the *tessellation evaluation shader* is run once for each vertex,
//!   including the ones newly created by the tessellation.
//! - Optionally, a shader named the *geometry shader* is run once for each line or triangle.
//! - The vertex coordinates (as outputted by the geometry shader, or by the tessellation
//!   evaluation shader if there's no geometry shader, or by the vertex shader if there's no
//!   geometry shader nor tessellation evaluation shader) are turned into screen-space coordinates.
//! - The list of pixels that cover each triangle are determined.
//! - A shader named the fragment shader is run once for each pixel that covers one of the
//!   triangles.
//! - The depth test and/or the stencil test are performed.
//! - The output of the fragment shader is written to the framebuffer attachments, possibly by
//!   mixing it with the existing values.
//!
//! All the sub-modules of this module (with the exception of `cache`) correspond to the various
//! stages of graphical pipelines.
//!
//! > **Note**: With the exception of the addition of the tessellation shaders and the geometry
//! > shader, these steps haven't changed in the past decade. If you are familiar with shaders in
//! > OpenGL 2 for example, don't worry as it works in the same in Vulkan.
//!
//! > **Note**: All the stages that consist in executing a shader are performed by a microprocessor
//! > (unless you happen to use a software implementation of Vulkan). As for the other stages,
//! > some hardware (usually desktop graphics cards) have dedicated chips that will execute them
//! > while some other hardware (usually mobile) perform them with the microprocessor as well. In
//! > the latter situation, the implementation will usually glue these steps to your shaders.
//!
//! Creating a graphics pipeline follows the same principle as a compute pipeline, except that
//! you must pass multiple shaders alongside with configuration for the other steps.
//!
//! TODO: add an example
//!
//! # Creating a ray tracing pipeline
//!
//! A ray tracing operation takes a scene as input. The scene is represented as an acceleration
//! structure in order for rays to quickly access geometry. The acceleration structures is
//! typically a bounding volume hiearchy of axis-aligned bounding boxes. The acceleration structure
//! is filled with triangles (vertices and indices) or with axis-aligned bounding boxes for
//! procedural geometry. Tracing consists of multiple steps:
//!
//! - A *shaders* called the *ray-generation shader* is run once for every ray created and can call
//!   a trace command on the scene and from its result, it can store color in the storage image from
//!   the resulting ray payload.
//! - A number of optional shaders called the *miss shaders* are run for every ray which does not
//!   encounter geometry. It can create a new trace command or write data to the ray payload.
//! - A number of optional *shader groups* containing the following shaders:
//!   - Optionally, a shader called the *closest-hit shader* is run once for each ray which has
//!     finished traversing the scene and has encountered geometry. It may read the intersection
//!     data and read or write to the ray payload.
//!   - Optionally, a shader called the *any-hit shader* is run for every interesction each ray has
//!     in the scene. It can evaluate if a geometry intersection should be accepted or not. It may
//!     read the intersection data and read or write to the ray payload
//!   - Optionally an *intersection shader* is run for procedural geometry is run for every
//!     bounding box each ray intersects. It then reports if the procedural geometry in the
//!     bounding box is intersected by the ray by writing to the intersection data.
//! - A number of optional shaders called the *callable shaders* are run from ray-generation,
//!   closest-hit, miss or other callable shaders to run on arbitrary data similarly to an
//!   indirect compute shader call.

// TODO: graphics pipeline params are deprecated, but are still the primary implementation in order
// to avoid duplicating code, so we hide the warnings for now
#![allow(deprecated)]

pub use self::compute_pipeline::ComputePipeline;
pub use self::compute_pipeline::ComputePipelineAbstract;
pub use self::compute_pipeline::ComputePipelineCreationError;
pub use self::compute_pipeline::ComputePipelineSys;
pub use self::graphics_pipeline::GraphicsPipeline;
pub use self::graphics_pipeline::GraphicsPipelineAbstract;
pub use self::graphics_pipeline::GraphicsPipelineBuilder;
pub use self::graphics_pipeline::GraphicsPipelineCreationError;
pub use self::graphics_pipeline::GraphicsPipelineSys;
pub use self::ray_tracing_pipeline::RayTracingPipeline;
pub use self::ray_tracing_pipeline::RayTracingPipelineAbstract;
pub use self::ray_tracing_pipeline::RayTracingPipelineBuilder;
pub use self::ray_tracing_pipeline::RayTracingPipelineCreationError;
pub use self::ray_tracing_pipeline::RayTracingPipelineSys;

mod compute_pipeline;
mod graphics_pipeline;
mod ray_tracing_pipeline;

pub mod blend;
pub mod cache;
pub mod depth_stencil;
pub mod input_assembly;
pub mod multisample;
pub mod raster;
pub mod shader;
pub mod vertex;
pub mod viewport;

#[derive(Copy, Clone)]
pub enum PipelineType {
    Graphics,
    Compute,
    RayTracing,
}
