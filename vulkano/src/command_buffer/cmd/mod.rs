// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! All the commands used in the internals of vulkano.

pub use self::begin_render_pass::CmdBeginRenderPass;
pub use self::bind_index_buffer::CmdBindIndexBuffer;
pub use self::bind_descriptor_sets::{CmdBindDescriptorSets, CmdBindDescriptorSetsError};
pub use self::bind_pipeline::{CmdBindPipeline, CmdBindPipelineSys};
pub use self::bind_vertex_buffers::CmdBindVertexBuffers;
//pub use self::blit_image_unsynced::{BlitRegion, BlitRegionAspect};
//pub use self::blit_image_unsynced::{CmdBlitImageUnsynced, CmdBlitImageUnsyncedError};
pub use self::clear_attachments::CmdClearAttachments;
pub use self::copy_buffer::{CmdCopyBuffer, CmdCopyBufferError};
pub use self::dispatch::{CmdDispatch, CmdDispatchError};
//pub use self::dispatch_indirect::{CmdDispatchIndirect, CmdDispatchIndirectError};
pub use self::dispatch_raw::{CmdDispatchRaw, CmdDispatchRawError};
pub use self::draw::CmdDraw;
//pub use self::draw_indexed::CmdDrawIndexed;
pub use self::draw_indexed_raw::CmdDrawIndexedRaw;
pub use self::draw_indirect_raw::CmdDrawIndirectRaw;
pub use self::draw_raw::CmdDrawRaw;
pub use self::end_render_pass::CmdEndRenderPass;
pub use self::execute::CmdExecuteCommands;
pub use self::fill_buffer::{CmdFillBuffer, CmdFillBufferError};
pub use self::next_subpass::CmdNextSubpass;
pub use self::pipeline_barrier::CmdPipelineBarrier;
pub use self::push_constants::{CmdPushConstants, CmdPushConstantsError};
pub use self::set_event::CmdSetEvent;
pub use self::set_state::{CmdSetState};
pub use self::update_buffer::{CmdUpdateBuffer, CmdUpdateBufferError};

mod begin_render_pass;
mod bind_descriptor_sets;
mod bind_index_buffer;
mod bind_pipeline;
mod bind_vertex_buffers;
//mod blit_image_unsynced;
mod clear_attachments;
mod copy_buffer;
mod dispatch;
//mod dispatch_indirect;
mod dispatch_raw;
mod draw;
//mod draw_indexed;
mod draw_indexed_raw;
mod draw_indirect_raw;
mod draw_raw;
mod end_render_pass;
mod execute;
mod fill_buffer;
mod next_subpass;
mod pipeline_barrier;
mod push_constants;
mod set_event;
mod set_state;
mod update_buffer;
