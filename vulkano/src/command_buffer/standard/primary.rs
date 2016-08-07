// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ops::Range;
use std::sync::Arc;

use buffer::Buffer;
use command_buffer::CommandBuffer;
use command_buffer::pool::CommandPool;
use command_buffer::pool::StandardCommandPool;
use command_buffer::standard::StdCommandBufferBuilder;
use command_buffer::standard::sync_helper::BuilderSyncHelper;
use command_buffer::sys::Flags;
use command_buffer::sys::Kind;
use command_buffer::sys::UnsafeCommandBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use framebuffer::EmptySinglePassRenderPass;
use image::Image;
use image::sys::Layout;
use sync::PipelineStages;
use sync::AccessFlagBits;

/// An empty primary command buffer builder.
pub struct StdPrimaryCommandBufferBuilder<P = Arc<StandardCommandPool>> where P: CommandPool {
    inner: UnsafeCommandBufferBuilder<P>,
    sync_helper: BuilderSyncHelper,
}

impl<P> StdPrimaryCommandBufferBuilder<P> where P: CommandPool {
    /// Builds a new empty primary command buffer.
    #[inline]
    pub fn new(pool: P) -> StdPrimaryCommandBufferBuilder<P> {
        let kind = Kind::Primary::<EmptySinglePassRenderPass, EmptySinglePassRenderPass>;
        let cb = UnsafeCommandBufferBuilder::new(pool, kind, Flags::SimultaneousUse).unwrap();  // TODO: allow handling this error

        StdPrimaryCommandBufferBuilder {
            inner: cb,
        }
    }
}

unsafe impl<P> StdCommandBufferBuilder for StdPrimaryCommandBufferBuilder<P>
    where P: CommandPool
{
    type BuildOutput = StdPrimaryCommandBuffer<P>;
    type Pool = P;

    #[inline]
    unsafe fn add_command<F>(&mut self, cmd: F)
        where F: FnOnce(&mut UnsafeCommandBufferBuilder<P>)
    {
        cmd(&mut self.inner)
    }

    #[inline]
    unsafe fn add_buffer_usage<B>(&mut self, buffer: &Arc<B>, slice: Range<usize>, write: bool,
                                  stages: PipelineStages, accesses: AccessFlagBits)
        where B: Buffer
    {
        unimplemented!()
    }

    #[inline]
    unsafe fn add_image_usage<I>(&mut self, image: &Arc<I>, mipmaps: Range<u32>,
                                 array_layers: Range<u32>, write: bool, layout: Layout,
                                 stages: PipelineStages, accesses: AccessFlagBits)
        where I: Image
    {
        unimplemented!()
    }

    #[inline]
    fn build(self) -> StdPrimaryCommandBuffer<P> {
        StdPrimaryCommandBuffer {
            inner: self.inner.build().unwrap(),     // TODO: allow handling this error
        }
    }
}

/// An empty primary command buffer.
pub struct StdPrimaryCommandBuffer<P = Arc<StandardCommandPool>> where P: CommandPool {
    inner: UnsafeCommandBuffer<P>
}

unsafe impl<P> CommandBuffer for StdPrimaryCommandBuffer<P> where P: CommandPool {
    type Pool = P;

    #[inline]
    fn inner(&self) -> &UnsafeCommandBuffer<Self::Pool> {
        &self.inner
    }

    #[inline]
    fn set_one_time_submit_flag(&self) -> Result<(), ()> {
        // FIXME:
        Ok(())
    }
}
