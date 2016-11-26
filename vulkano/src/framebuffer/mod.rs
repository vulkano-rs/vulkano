// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Targets on which your draw commands are executed.
//! 
//! # Render passes and framebuffers
//!
//! There are two concepts in Vulkan:
//! 
//! - A `RenderPassRef` is a collection of one or multiples passes called subpasses. Each subpass
//!   contains the format and dimensions of the attachments that are part of the subpass. The
//!   render pass only defines the layout of the rendering process.
//! - A `FramebufferRef` contains the list of actual images that are attached. It is created from a
//!   render pass and has to match its characteristics.
//!
//! You can create graphics pipelines from a render pass object alone.
//! A `FramebufferRef` is only needed when you add draw commands to a command buffer.
//!
//! # Render passes
//!
//! In vulkano, a render pass is any object that implements the `RenderPassRef` trait.
//!
//! You can create a render pass by creating a `RenderPass` object. But as its name tells,
//! it is unsafe because a lot of safety checks aren't performed.
//!
//! Instead you are encouraged to use a safe wrapper around an `RenderPass`.
//! There are two ways to do this:   TODO add more ways
//!
//! - Creating an instance of an `EmptySinglePassRenderPass`, which describes a render pass with no
//!   attachment and with one subpass.
//! - Using the `single_pass_renderpass!` macro. See the documentation of this macro.
//!
//! Render passes have three characteristics:
//!
//! - A list of attachments with their format.
//! - A list of subpasses, that defines for each subpass which attachment is used for which
//!   purpose.
//! - A list of dependencies between subpasses. Vulkan implementations are free to reorder the
//!   subpasses, which means that you need to declare dependencies if the output of a subpass
//!   needs to be read in a following subpass.
//!
//! ## Example
//!
//! With `EmptySinglePassRenderPass`:
//!
//! ```no_run
//! use vulkano::framebuffer::EmptySinglePassRenderPass;
//!
//! # let device: std::sync::Arc<vulkano::device::Device> = unsafe { ::std::mem::uninitialized() };
//! let renderpass = EmptySinglePassRenderPass::new(&device);
//! ```
//!
//! # Framebuffers
//!
//! Creating a framebuffer is done by passing the render pass object, the dimensions of the
//! framebuffer, and the list of attachments to `FramebufferRef::new()`.
//!
//! The slightly tricky part is that the type that contains the list of attachments depends on
//! the trait implementation of `RenderPassRef`. For example if you use an
//! `EmptySinglePassRenderPass`, you have to pass `()` for the list of attachments.
//!

pub use self::basic_render_pass::BasicRenderPassDesc;
pub use self::basic_render_pass::BasicRenderPassDescAttachment;
pub use self::empty::EmptySinglePassRenderPassDesc;
pub use self::framebuffer::Framebuffer;
pub use self::framebuffer::FramebufferCreationError;
pub use self::framebuffer::FramebufferSys;
pub use self::sys::RenderPass;
pub use self::sys::RenderPassSys;
pub use self::sys::RenderPassCreationError;
pub use self::traits::FramebufferRef;
pub use self::traits::RenderPassRef;
pub use self::traits::RenderPassDesc;
pub use self::traits::RenderPassAttachmentsList;
pub use self::traits::RenderPassClearValues;
pub use self::traits::RenderPassCompatible;
pub use self::traits::RenderPassSubpassInterface;
pub use self::traits::LayoutAttachmentDescription;
pub use self::traits::LayoutPassDescription;
pub use self::traits::LayoutPassDependencyDescription;
pub use self::traits::StoreOp;
pub use self::traits::LoadOp;
pub use self::traits::Subpass;

mod basic_render_pass;
#[macro_use]
mod macros;
mod empty;
#[doc(hidden)] pub mod framebuffer;
mod sys;
#[doc(hidden)] pub mod traits;      // TODO: pub-hidden because of that trait visibility bug
