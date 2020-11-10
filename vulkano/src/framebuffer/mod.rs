// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Targets on which your draw commands are executed.
//!
//! # Render passes and framebuffers
//!
//! There are two concepts in Vulkan:
//!
//! - A *render pass* describes the target which you are going to render to. It is a collection
//!   of descriptions of one or more attachments (ie. image that are rendered to), and of one or
//!   multiples subpasses. The render pass contains the format and number of samples of each
//!   attachment, and the attachments that are attached to each subpass. They are represented
//!   in vulkano with the `RenderPass` object.
//! - A *framebuffer* contains the list of actual images that are attached. It is created from a
//!   render pass and has to match its characteristics. They are represented in vulkano with the
//!   `Framebuffer` object.
//!
//! Render passes are typically created at initialization only (for example during a loading
//! screen) because they can be costly, while framebuffers can be created and destroyed either at
//! initialization or during the frame.
//!
//! Consequently you can create graphics pipelines from a render pass object alone.
//! A `Framebuffer` object is only needed when you actually add draw commands to a command buffer.
//!
//! # Render passes
//!
//! In vulkano a render pass is represented by the `RenderPass` struct. This struct has a template
//! parameter that contains the description of the render pass. The `RenderPassAbstract` trait is
//! implemented on all instances of `RenderPass<_>` and makes it easier to store render passes
//! without having to explicitly write its type.
//!
//! The template parameter of the `RenderPass` struct must implement the `RenderPassDesc` trait.
//! In order to create a render pass, you can create an object that implements this trait, then
//! call the `build_render_pass` method on it.
//!
//! ```
//! use vulkano::framebuffer::EmptySinglePassRenderPassDesc;
//! use vulkano::framebuffer::RenderPassDesc;
//!
//! # let device: std::sync::Arc<vulkano::device::Device> = return;
//! let desc = EmptySinglePassRenderPassDesc;
//! let render_pass = desc.build_render_pass(device.clone()).unwrap();
//! // The type of `render_pass` is `RenderPass<EmptySinglePassRenderPassDesc>`.
//! ```
//!
//! This example creates a render pass with no attachment and one single subpass that doesn't draw
//! on anything. While it's sometimes useful, most of the time it's not what you want.
//!
//! The easiest way to create a "real" render pass is to use the `single_pass_renderpass!` macro.
//!
//! ```
//! # #[macro_use] extern crate vulkano;
//! # fn main() {
//! # let device: std::sync::Arc<vulkano::device::Device> = return;
//! use vulkano::format::Format;
//!
//! let render_pass = single_pass_renderpass!(device.clone(),
//!     attachments: {
//!         // `foo` is a custom name we give to the first and only attachment.
//!         foo: {
//!             load: Clear,
//!             store: Store,
//!             format: Format::R8G8B8A8Unorm,
//!             samples: 1,
//!         }
//!     },
//!     pass: {
//!         color: [foo],       // Repeat the attachment name here.
//!         depth_stencil: {}
//!     }
//! ).unwrap();
//! # }
//! ```
//!
//! See the documentation of the macro for more details. TODO: put link here
//!
//! Once a `RenderPass<_>` struct is created, it implements the same render-pass-related traits as
//! its template parameter.
//!
//! # Framebuffers
//!
//! See [the documentation of the `Framebuffer` struct](struct.Framebuffer.html) for information
//! about how to create a framebuffer.
//!

pub use self::attachments_list::AttachmentsList;
pub use self::compat_atch::ensure_image_view_compatible;
pub use self::compat_atch::IncompatibleRenderPassAttachmentError;
pub use self::desc::AttachmentDescription;
pub use self::desc::LoadOp;
pub use self::desc::PassDependencyDescription;
pub use self::desc::PassDescription;
pub use self::desc::RenderPassDesc;
pub use self::desc::RenderPassDescAttachments;
pub use self::desc::RenderPassDescDependencies;
pub use self::desc::RenderPassDescSubpasses;
pub use self::desc::StoreOp;
pub use self::empty::EmptySinglePassRenderPassDesc;
pub use self::framebuffer::Framebuffer;
pub use self::framebuffer::FramebufferBuilder;
pub use self::framebuffer::FramebufferCreationError;
pub use self::framebuffer::FramebufferSys;
pub use self::sys::RenderPass;
pub use self::sys::RenderPassCreationError;
pub use self::sys::RenderPassSys;
pub use self::traits::FramebufferAbstract;
pub use self::traits::RenderPassAbstract;
pub use self::traits::RenderPassCompatible;
pub use self::traits::RenderPassDescClearValues;
pub use self::traits::RenderPassSubpassInterface;
pub use self::traits::Subpass;

#[macro_use]
mod macros;
mod attachments_list;
mod compat_atch;
mod desc;
mod empty;
mod framebuffer;
mod sys;
mod traits;
