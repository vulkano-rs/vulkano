// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Description of the steps of the rendering process, and the images used as input or output.
//!
//! # Render passes and framebuffers
//!
//! There are two concepts in Vulkan:
//!
//! - A *render pass* describes the overall process of drawing a frame. It is subdivided into one
//!   or more subpasses.
//! - A *framebuffer* contains the list of image views that are attached during the drawing of
//!   each subpass.
//!
//! Render passes are typically created at initialization only (for example during a loading
//! screen) because they can be costly, while framebuffers can be created and destroyed either at
//! initialization or during the frame.
//!
//! Consequently you can create graphics pipelines from a render pass object alone.
//! A `Framebuffer` object is only needed when you actually add draw commands to a command buffer.

pub use self::attachments_list::AttachmentsList;
pub use self::compat_atch::ensure_image_view_compatible;
pub use self::compat_atch::IncompatibleRenderPassAttachmentError;
pub use self::desc::AttachmentDesc;
pub use self::desc::LoadOp;
pub use self::desc::RenderPassDesc;
pub use self::desc::StoreOp;
pub use self::desc::SubpassDependencyDesc;
pub use self::desc::SubpassDesc;
pub use self::framebuffer::Framebuffer;
pub use self::framebuffer::FramebufferAbstract;
pub use self::framebuffer::FramebufferBuilder;
pub use self::framebuffer::FramebufferCreationError;
pub use self::framebuffer::FramebufferSys;
pub use self::render_pass::RenderPass;
pub use self::render_pass::RenderPassCreationError;
pub use self::render_pass::RenderPassSys;
pub use self::render_pass::Subpass;

#[macro_use]
mod macros;
mod attachments_list;
mod compat_atch;
mod desc;
mod framebuffer;
mod render_pass;
