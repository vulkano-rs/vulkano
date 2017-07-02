// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use smallvec::SmallVec;
use std::fmt;
use std::mem;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use OomError;
use VulkanObject;
use buffer::BufferAccess;
use buffer::BufferInner;
use check_errors;
use command_buffer::CommandBuffer;
use command_buffer::pool::CommandPool;
use command_buffer::pool::CommandPoolAlloc;
use command_buffer::pool::CommandPoolBuilderAlloc;
use descriptor::descriptor::ShaderStages;
use descriptor::descriptor_set::UnsafeDescriptorSet;
use descriptor::pipeline_layout::PipelineLayoutAbstract;
use device::Device;
use device::DeviceOwned;
use format::ClearValue;
use framebuffer::EmptySinglePassRenderPassDesc;
use framebuffer::Framebuffer;
use framebuffer::FramebufferAbstract;
use framebuffer::RenderPass;
use framebuffer::RenderPassAbstract;
use framebuffer::Subpass;
use framebuffer::SubpassContents;
use image::ImageAccess;
use image::ImageLayout;
use instance::QueueFamily;
use pipeline::ComputePipelineAbstract;
use pipeline::GraphicsPipelineAbstract;
use pipeline::input_assembly::IndexType;
use pipeline::viewport::Scissor;
use pipeline::viewport::Viewport;
use sync::AccessFlagBits;
use sync::Event;
use sync::PipelineStages;
use vk;

/// Determines the kind of command buffer that we want to create.
#[derive(Debug, Clone)]
pub enum Kind<R, F> {
    /// A primary command buffer can execute all commands and can call secondary command buffers.
    Primary,

    /// A secondary command buffer can execute all dispatch and transfer operations, but not
    /// drawing operations.
    Secondary,

    /// A secondary command buffer within a render pass can only call draw operations that can
    /// be executed from within a specific subpass.
    SecondaryRenderPass {
        /// Which subpass this secondary command buffer can be called from.
        subpass: Subpass<R>,

        /// The framebuffer object that will be used when calling the command buffer.
        /// This parameter is optional and is an optimization hint for the implementation.
        framebuffer: Option<F>,
    },
}

impl
    Kind<RenderPass<EmptySinglePassRenderPassDesc>,
         Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>, ()>> {
    /// Equivalent to `Kind::Primary`.
    ///
    /// > **Note**: If you use `let kind = Kind::Primary;` in your code, you will probably get a
    /// > compilation error because the Rust compiler couldn't determine the template parameters
    /// > of `Kind`. To solve that problem in an easy way you can use this function instead.
    #[inline]
    pub fn primary()
        -> Kind<RenderPass<EmptySinglePassRenderPassDesc>,
                Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>, ()>>
    {
        Kind::Primary
    }

    /// Equivalent to `Kind::Secondary`.
    ///
    /// > **Note**: If you use `let kind = Kind::Secondary;` in your code, you will probably get a
    /// > compilation error because the Rust compiler couldn't determine the template parameters
    /// > of `Kind`. To solve that problem in an easy way you can use this function instead.
    #[inline]
    pub fn secondary()
        -> Kind<RenderPass<EmptySinglePassRenderPassDesc>,
                Framebuffer<RenderPass<EmptySinglePassRenderPassDesc>, ()>>
    {
        Kind::Secondary
    }
}

/// Flags to pass when creating a command buffer.
///
/// The safest option is `SimultaneousUse`, but it may be slower than the other two.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Flags {
    /// The command buffer can be used multiple times, but must not execute more than once
    /// simultaneously.
    None,

    /// The command buffer can be executed multiple times in parallel.
    SimultaneousUse,

    /// The command buffer can only be submitted once. Any further submit is forbidden.
    OneTimeSubmit,
}

/// Command buffer being built.
///
/// You can add commands to an `UnsafeCommandBufferBuilder` by using the `AddCommand` trait.
/// The `AddCommand<&Cmd>` trait is implemented on the `UnsafeCommandBufferBuilder` for any `Cmd`
/// that is a raw Vulkan command.
///
/// When you are finished adding commands, you can use the `CommandBufferBuild` trait to turn this
/// builder into an `UnsafeCommandBuffer`.
pub struct UnsafeCommandBufferBuilder<P> {
    // The command buffer obtained from the pool. Contains `None` if `build()` has been called.
    cmd: Option<P>,

    // The raw `cmd`. Avoids having to specify a trait bound on `P`.
    cmd_raw: vk::CommandBuffer,

    // Device that owns the command buffer.
    // TODO: necessary?
    device: Arc<Device>,

    // Flags that were used at creation.
    // TODO: necessary?
    flags: Flags,
}

impl<P> fmt::Debug for UnsafeCommandBufferBuilder<P> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<Vulkan command buffer builder #{}>", self.cmd_raw)
    }
}

impl<P> UnsafeCommandBufferBuilder<P> {
    /// Creates a new builder.
    ///
    /// # Safety
    ///
    /// Creating and destroying an unsafe command buffer is not unsafe per se, but the commands
    /// that you add to it are unchecked, do not have any synchronization, and are not kept alive.
    ///
    /// In other words, it is your job to make sure that the commands you add are valid, that they
    /// don't use resources that have been destroyed, and that they do not introduce any race
    /// condition.
    ///
    /// > **Note**: Some checks are still made with `debug_assert!`. Do not expect to be able to
    /// > submit invalid commands.
    pub unsafe fn new<Pool, R, F, A>(pool: &Pool, kind: Kind<R, F>, flags: Flags)
                                     -> Result<UnsafeCommandBufferBuilder<P>, OomError>
        where Pool: CommandPool<Builder = P, Alloc = A>,
              P: CommandPoolBuilderAlloc<Alloc = A>,
              A: CommandPoolAlloc,
              R: RenderPassAbstract,
              F: FramebufferAbstract
    {
        let secondary = match kind {
            Kind::Primary => false,
            Kind::Secondary |
            Kind::SecondaryRenderPass { .. } => true,
        };

        let cmd = pool.alloc(secondary, 1)?
            .next()
            .expect("Requested one command buffer from the command pool, but got zero.");
        UnsafeCommandBufferBuilder::already_allocated(cmd, kind, flags)
    }

    /// Creates a new command buffer builder from an already-allocated command buffer.
    ///
    /// # Safety
    ///
    /// See the `new` method.
    ///
    /// The kind must match how the command buffer was allocated.
    ///
    pub unsafe fn already_allocated<R, F>(alloc: P, kind: Kind<R, F>, flags: Flags)
                                          -> Result<UnsafeCommandBufferBuilder<P>, OomError>
        where R: RenderPassAbstract,
              F: FramebufferAbstract,
              P: CommandPoolBuilderAlloc
    {
        let device = alloc.device().clone();
        let vk = device.pointers();
        let cmd = alloc.inner().internal_object();

        let vk_flags = {
            let a = match flags {
                Flags::None => 0,
                Flags::SimultaneousUse => vk::COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
                Flags::OneTimeSubmit => vk::COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };

            let b = match kind {
                Kind::Primary | Kind::Secondary => 0,
                Kind::SecondaryRenderPass { .. } => {
                    vk::COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT
                },
            };

            a | b
        };

        let (rp, sp) = if let Kind::SecondaryRenderPass { ref subpass, .. } = kind {
            (subpass.render_pass().inner().internal_object(), subpass.index())
        } else {
            (0, 0)
        };

        let framebuffer = if let Kind::SecondaryRenderPass {
            ref subpass,
            framebuffer: Some(ref framebuffer),
        } = kind
        {
            // TODO: restore check
            //assert!(framebuffer.is_compatible_with(subpass.render_pass()));     // TODO: proper error
            FramebufferAbstract::inner(&framebuffer).internal_object()
        } else {
            0
        };

        let inheritance = vk::CommandBufferInheritanceInfo {
            sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
            pNext: ptr::null(),
            renderPass: rp,
            subpass: sp,
            framebuffer: framebuffer,
            occlusionQueryEnable: 0, // TODO:
            queryFlags: 0, // TODO:
            pipelineStatistics: 0, // TODO:
        };

        let infos = vk::CommandBufferBeginInfo {
            sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: ptr::null(),
            flags: vk_flags,
            pInheritanceInfo: &inheritance,
        };

        check_errors(vk.BeginCommandBuffer(cmd, &infos))?;

        Ok(UnsafeCommandBufferBuilder {
               cmd: Some(alloc),
               cmd_raw: cmd,
               device: device.clone(),
               flags: flags,
           })
    }

    /// Returns the queue family of the builder.
    #[inline]
    pub fn queue_family(&self) -> QueueFamily
        where P: CommandPoolBuilderAlloc
    {
        self.cmd.as_ref().unwrap().queue_family()
    }

    /// Turns the builder into an actual command buffer.
    #[inline]
    pub fn build(mut self) -> Result<UnsafeCommandBuffer<P::Alloc>, OomError>
        where P: CommandPoolBuilderAlloc
    {
        unsafe {
            let cmd = self.cmd.take().unwrap();
            let vk = self.device.pointers();
            check_errors(vk.EndCommandBuffer(cmd.inner().internal_object()))?;
            let cmd_raw = cmd.inner().internal_object();

            Ok(UnsafeCommandBuffer {
                   cmd: cmd.into_alloc(),
                   cmd_raw: cmd_raw,
                   device: self.device.clone(),
                   flags: self.flags,
                   already_submitted: AtomicBool::new(false),
               })
        }
    }

    /// Calls `vkCmdBeginRenderPass` on the builder.
    #[inline]
    pub unsafe fn begin_render_pass<F, I>(&mut self, framebuffer: &F,
                                          subpass_contents: SubpassContents, clear_values: I)
        where F: ?Sized + FramebufferAbstract,
              I: Iterator<Item = ClearValue>
    {
        let vk = self.device().pointers();
        let cmd = self.internal_object();

        // TODO: allow passing a different render pass
        let raw_render_pass = RenderPassAbstract::inner(&framebuffer).internal_object();
        let raw_framebuffer = FramebufferAbstract::inner(&framebuffer).internal_object();

        let raw_clear_values: SmallVec<[_; 12]> = clear_values
            .map(|clear_value| match clear_value {
                     ClearValue::None => {
                         vk::ClearValue::color(vk::ClearColorValue::float32([0.0; 4]))
                     },
                     ClearValue::Float(val) => {
                         vk::ClearValue::color(vk::ClearColorValue::float32(val))
                     },
                     ClearValue::Int(val) => {
                         vk::ClearValue::color(vk::ClearColorValue::int32(val))
                     },
                     ClearValue::Uint(val) => {
                         vk::ClearValue::color(vk::ClearColorValue::uint32(val))
                     },
                     ClearValue::Depth(val) => {
                         vk::ClearValue::depth_stencil(vk::ClearDepthStencilValue {
                                                           depth: val,
                                                           stencil: 0,
                                                       })
                     },
                     ClearValue::Stencil(val) => {
                         vk::ClearValue::depth_stencil(vk::ClearDepthStencilValue {
                                                           depth: 0.0,
                                                           stencil: val,
                                                       })
                     },
                     ClearValue::DepthStencil((depth, stencil)) => {
                         vk::ClearValue::depth_stencil(vk::ClearDepthStencilValue {
                                                           depth: depth,
                                                           stencil: stencil,
                                                       })
                     },
                 })
            .collect();

        // TODO: allow customizing
        let rect = [
            0 .. framebuffer.dimensions()[0],
            0 .. framebuffer.dimensions()[1],
        ];

        let begin = vk::RenderPassBeginInfo {
            sType: vk::STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            pNext: ptr::null(),
            renderPass: raw_render_pass,
            framebuffer: raw_framebuffer,
            renderArea: vk::Rect2D {
                offset: vk::Offset2D {
                    x: rect[0].start as i32,
                    y: rect[1].start as i32,
                },
                extent: vk::Extent2D {
                    width: rect[0].end - rect[0].start,
                    height: rect[1].end - rect[1].start,
                },
            },
            clearValueCount: raw_clear_values.len() as u32,
            pClearValues: raw_clear_values.as_ptr(),
        };

        vk.CmdBeginRenderPass(cmd, &begin, subpass_contents as u32);
    }

    /// Calls `vkCmdBindDescriptorSets` on the builder.
    ///
    /// Does nothing if the list of descriptor sets is empty, as it would be a no-op and isn't a
    /// valid usage of the command anyway.
    #[inline]
    pub unsafe fn bind_descriptor_sets<'s, Pl, S, I>(&mut self, graphics: bool,
                                                     pipeline_layout: &Pl, first_binding: u32,
                                                     sets: S, dynamic_offsets: I)
        where Pl: ?Sized + PipelineLayoutAbstract,
              S: Iterator<Item = &'s UnsafeDescriptorSet>,
              I: Iterator<Item = u32>
    {
        let vk = self.device().pointers();
        let cmd = self.internal_object();

        let sets: SmallVec<[_; 12]> = sets.map(|s| s.internal_object()).collect();
        if sets.is_empty() {
            return;
        }
        let dynamic_offsets: SmallVec<[u32; 32]> = dynamic_offsets.collect();

        let num_bindings = sets.len() as u32;
        debug_assert!(first_binding + num_bindings <= pipeline_layout.num_sets() as u32);

        let bind_point = if graphics {
            vk::PIPELINE_BIND_POINT_GRAPHICS
        } else {
            vk::PIPELINE_BIND_POINT_COMPUTE
        };

        vk.CmdBindDescriptorSets(cmd,
                                 bind_point,
                                 pipeline_layout.sys().internal_object(),
                                 first_binding,
                                 num_bindings,
                                 sets.as_ptr(),
                                 dynamic_offsets.len() as u32,
                                 dynamic_offsets.as_ptr());
    }

    /// Calls `vkCmdBindIndexBuffer` on the builder.
    #[inline]
    pub unsafe fn bind_index_buffer<B>(&mut self, buffer: &B, index_ty: IndexType)
        where B: ?Sized + BufferAccess
    {
        let vk = self.device().pointers();
        let cmd = self.internal_object();

        let inner = buffer.inner();
        debug_assert!(inner.offset < inner.buffer.size());
        debug_assert!(inner.buffer.usage_index_buffer());

        vk.CmdBindIndexBuffer(cmd,
                              inner.buffer.internal_object(),
                              inner.offset as vk::DeviceSize,
                              index_ty as vk::IndexType);
    }

    /// Calls `vkCmdBindPipeline` on the builder with a compute pipeline.
    #[inline]
    pub unsafe fn bind_pipeline_compute<Cp>(&mut self, pipeline: &Cp)
        where Cp: ?Sized + ComputePipelineAbstract
    {
        let vk = self.device().pointers();
        let cmd = self.internal_object();
        vk.CmdBindPipeline(cmd,
                           vk::PIPELINE_BIND_POINT_COMPUTE,
                           pipeline.inner().internal_object());
    }

    /// Calls `vkCmdBindPipeline` on the builder with a graphics pipeline.
    #[inline]
    pub unsafe fn bind_pipeline_graphics<Gp>(&mut self, pipeline: &Gp)
        where Gp: ?Sized + GraphicsPipelineAbstract
    {
        let vk = self.device().pointers();
        let cmd = self.internal_object();
        let inner = GraphicsPipelineAbstract::inner(pipeline).internal_object();
        vk.CmdBindPipeline(cmd, vk::PIPELINE_BIND_POINT_GRAPHICS, inner);
    }

    /// Calls `vkCmdBindVertexBuffers` on the builder.
    ///
    /// Does nothing if the list of buffers is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn bind_vertex_buffers(&mut self, first_binding: u32,
                                      params: UnsafeCommandBufferBuilderBindVertexBuffer) {
        debug_assert_eq!(params.raw_buffers.len(), params.offsets.len());

        if params.raw_buffers.is_empty() {
            return;
        }

        let vk = self.device().pointers();
        let cmd = self.internal_object();

        let num_bindings = params.raw_buffers.len() as u32;

        debug_assert!({
                          let max_bindings = self.device()
                              .physical_device()
                              .limits()
                              .max_vertex_input_bindings();
                          first_binding + num_bindings <= max_bindings
                      });

        vk.CmdBindVertexBuffers(cmd,
                                first_binding,
                                num_bindings,
                                params.raw_buffers.as_ptr(),
                                params.offsets.as_ptr());
    }

    // TODO: missing structs
    /*/// Calls `vkCmdClearAttachments` on the builder.
    ///
    /// Does nothing if the list of attachments or the list of rects is empty, as it would be a
    /// no-op and isn't a valid usage of the command anyway.
    #[inline]
    pub unsafe fn clear_attachments<A, R>(&mut self, attachments: A, rects: R)
        where A: Iterator<Item = >,
              R: Iterator<Item = >
    {
        let attachments: SmallVec<[_; 16]> = attachments.map().collect();
        let rects: SmallVec<[_; 4]> = rects.map().collect();

        if attachments.is_empty() || rects.is_empty() {
            return;
        }

        let vk = self.device().pointers();
        let cmd = self.internal_object();
        vk.CmdClearAttachments(cmd, attachments.len() as u32, attachments.as_ptr(),
                               rects.len() as u32, rects.as_ptr());
    }*/

    /// Calls `vkCmdCopyBuffer` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_buffer<S, D, R>(&mut self, source: &S, destination: &D, regions: R)
        where S: ?Sized + BufferAccess,
              D: ?Sized + BufferAccess,
              R: Iterator<Item = (usize, usize, usize)>
    {
        // TODO: debug assert that there's no overlap in the destinations?

        let source = source.inner();
        debug_assert!(source.offset < source.buffer.size());
        debug_assert!(source.buffer.usage_transfer_source());

        let destination = destination.inner();
        debug_assert!(destination.offset < destination.buffer.size());
        debug_assert!(destination.buffer.usage_transfer_destination());

        let regions: SmallVec<[_; 8]> = regions
            .map(|(sr, de, sz)| {
                     vk::BufferCopy {
                         srcOffset: (sr + source.offset) as vk::DeviceSize,
                         dstOffset: (de + destination.offset) as vk::DeviceSize,
                         size: sz as vk::DeviceSize,
                     }
                 })
            .collect();

        if regions.is_empty() {
            return;
        }

        let vk = self.device().pointers();
        let cmd = self.internal_object();
        vk.CmdCopyBuffer(cmd,
                         source.buffer.internal_object(),
                         destination.buffer.internal_object(),
                         regions.len() as u32,
                         regions.as_ptr());
    }

    /// Calls `vkCmdCopyBufferToImage` on the builder.
    ///
    /// Does nothing if the list of regions is empty, as it would be a no-op and isn't a valid
    /// usage of the command anyway.
    #[inline]
    pub unsafe fn copy_buffer_to_image<S, D, R>(&mut self, source: &S, destination: &D,
                                                destination_layout: ImageLayout, regions: R)
        where S: ?Sized + BufferAccess,
              D: ?Sized + ImageAccess,
              R: Iterator<Item = UnsafeCommandBufferBuilderBufferImageCopy>
    {
        let source = source.inner();
        debug_assert!(source.offset < source.buffer.size());
        debug_assert!(source.buffer.usage_transfer_source());

        debug_assert_eq!(destination.samples(), 1);
        let destination = destination.inner();
        debug_assert!(destination.image.usage_transfer_destination());
        debug_assert!(destination_layout == ImageLayout::General ||
                      destination_layout == ImageLayout::TransferDstOptimal);

        let regions: SmallVec<[_; 8]> = regions
            .map(|copy| {
                debug_assert!(copy.image_layer_count <= destination.num_layers as u32);
                debug_assert!(copy.image_mip_level < destination.num_mipmap_levels as u32);

                vk::BufferImageCopy {
                    bufferOffset: (source.offset + copy.buffer_offset) as vk::DeviceSize,
                    bufferRowLength: copy.buffer_row_length,
                    bufferImageHeight: copy.buffer_image_height,
                    imageSubresource: vk::ImageSubresourceLayers {
                        aspectMask: copy.image_aspect.to_vk_bits(),
                        mipLevel: copy.image_mip_level + destination.first_mipmap_level as u32,
                        baseArrayLayer: copy.image_base_array_layer + destination.first_layer as u32,
                        layerCount: copy.image_layer_count,
                    },
                    imageOffset: vk::Offset3D {
                        x: copy.image_offset[0],
                        y: copy.image_offset[1],
                        z: copy.image_offset[2],
                    },
                    imageExtent: vk::Extent3D {
                        width: copy.image_extent[0],
                        height: copy.image_extent[1],
                        depth: copy.image_extent[2],
                    },
                }
            })
            .collect();

        if regions.is_empty() {
            return;
        }

        let vk = self.device().pointers();
        let cmd = self.internal_object();
        vk.CmdCopyBufferToImage(cmd,
                                source.buffer.internal_object(),
                                destination.image.internal_object(),
                                destination_layout as u32,
                                regions.len() as u32,
                                regions.as_ptr());
    }

    /// Calls `vkCmdDispatch` on the builder.
    #[inline]
    pub unsafe fn dispatch(&mut self, dimensions: [u32; 3]) {
        debug_assert!({
                          let max_dims = self.device()
                              .physical_device()
                              .limits()
                              .max_compute_work_group_count();
                          dimensions[0] <= max_dims[0] && dimensions[1] <= max_dims[1] &&
                              dimensions[2] <= max_dims[2]
                      });

        let vk = self.device().pointers();
        let cmd = self.internal_object();
        vk.CmdDispatch(cmd, dimensions[0], dimensions[1], dimensions[2]);
    }

    /// Calls `vkCmdDispatchIndirect` on the builder.
    #[inline]
    pub unsafe fn dispatch_indirect<B>(&mut self, buffer: &B)
        where B: ?Sized + BufferAccess
    {
        let vk = self.device().pointers();
        let cmd = self.internal_object();

        let inner = buffer.inner();
        debug_assert!(inner.offset < buffer.size());
        debug_assert!(inner.buffer.usage_indirect_buffer());
        debug_assert_eq!(inner.offset % 4, 0);

        vk.CmdDispatchIndirect(cmd,
                               inner.buffer.internal_object(),
                               inner.offset as vk::DeviceSize);
    }

    /// Calls `vkCmdDraw` on the builder.
    #[inline]
    pub unsafe fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32,
                       first_instance: u32) {
        let vk = self.device().pointers();
        let cmd = self.internal_object();
        vk.CmdDraw(cmd,
                   vertex_count,
                   instance_count,
                   first_vertex,
                   first_instance);
    }

    /// Calls `vkCmdDrawIndexed` on the builder.
    #[inline]
    pub unsafe fn draw_indexed(&mut self, index_count: u32, instance_count: u32,
                               first_index: u32, vertex_offset: i32, first_instance: u32) {
        let vk = self.device().pointers();
        let cmd = self.internal_object();
        vk.CmdDrawIndexed(cmd,
                          index_count,
                          instance_count,
                          first_index,
                          vertex_offset,
                          first_instance);
    }

    /// Calls `vkCmdDrawIndirect` on the builder.
    #[inline]
    pub unsafe fn draw_indirect<B>(&mut self, buffer: &B, draw_count: u32, stride: u32)
        where B: ?Sized + BufferAccess
    {
        let vk = self.device().pointers();
        let cmd = self.internal_object();

        debug_assert!(draw_count == 0 ||
                          ((stride % 4) == 0) &&
                              stride as usize >= mem::size_of::<vk::DrawIndirectCommand>());

        let inner = buffer.inner();
        debug_assert!(inner.offset < buffer.size());
        debug_assert!(inner.buffer.usage_indirect_buffer());

        vk.CmdDrawIndirect(cmd,
                           inner.buffer.internal_object(),
                           inner.offset as vk::DeviceSize,
                           draw_count,
                           stride);
    }

    /// Calls `vkCmdDrawIndexedIndirect` on the builder.
    #[inline]
    pub unsafe fn draw_indexed_indirect<B>(&mut self, buffer: &B, draw_count: u32, stride: u32)
        where B: ?Sized + BufferAccess
    {
        let vk = self.device().pointers();
        let cmd = self.internal_object();

        let inner = buffer.inner();
        debug_assert!(inner.offset < buffer.size());
        debug_assert!(inner.buffer.usage_indirect_buffer());

        vk.CmdDrawIndexedIndirect(cmd,
                                  inner.buffer.internal_object(),
                                  inner.offset as vk::DeviceSize,
                                  draw_count,
                                  stride);
    }

    /// Calls `vkCmdEndRenderPass` on the builder.
    #[inline]
    pub unsafe fn end_render_pass(&mut self) {
        let vk = self.device().pointers();
        let cmd = self.internal_object();
        vk.CmdEndRenderPass(cmd);
    }

    /// Calls `vkCmdExecuteCommands` on the builder.
    ///
    /// Does nothing if the list of command buffers is empty, as it would be a no-op and isn't a
    /// valid usage of the command anyway.
    #[inline]
    pub unsafe fn execute_commands(&mut self, cbs: UnsafeCommandBufferBuilderExecuteCommands) {
        if cbs.raw_cbs.is_empty() {
            return;
        }

        let vk = self.device().pointers();
        let cmd = self.internal_object();
        vk.CmdExecuteCommands(cmd, cbs.raw_cbs.len() as u32, cbs.raw_cbs.as_ptr());
    }

    /// Calls `vkCmdFillBuffer` on the builder.
    #[inline]
    pub unsafe fn fill_buffer<B>(&mut self, buffer: &B, data: u32)
        where B: ?Sized + BufferAccess
    {
        let vk = self.device().pointers();
        let cmd = self.internal_object();

        let size = buffer.size();

        let (buffer_handle, offset) = {
            let BufferInner {
                buffer: buffer_inner,
                offset,
            } = buffer.inner();
            debug_assert!(buffer_inner.usage_transfer_destination());
            debug_assert_eq!(offset % 4, 0);
            (buffer_inner.internal_object(), offset)
        };

        vk.CmdFillBuffer(cmd,
                         buffer_handle,
                         offset as vk::DeviceSize,
                         size as vk::DeviceSize,
                         data);
    }

    /// Calls `vkCmdNextSubpass` on the builder.
    #[inline]
    pub unsafe fn next_subpass(&mut self, subpass_contents: SubpassContents) {
        let vk = self.device().pointers();
        let cmd = self.internal_object();
        vk.CmdNextSubpass(cmd, subpass_contents as u32);
    }

    #[inline]
    pub unsafe fn pipeline_barrier(&mut self, command: &UnsafeCommandBufferBuilderPipelineBarrier) {
        // If barrier is empty, don't do anything.
        if command.src_stage_mask == 0 || command.dst_stage_mask == 0 {
            debug_assert!(command.src_stage_mask == 0 && command.dst_stage_mask == 0);
            debug_assert!(command.memory_barriers.is_empty());
            debug_assert!(command.buffer_barriers.is_empty());
            debug_assert!(command.image_barriers.is_empty());
            return;
        }

        let vk = self.device().pointers();
        let cmd = self.internal_object();

        debug_assert_ne!(command.src_stage_mask, 0);
        debug_assert_ne!(command.dst_stage_mask, 0);

        vk.CmdPipelineBarrier(cmd,
                              command.src_stage_mask,
                              command.dst_stage_mask,
                              command.dependency_flags,
                              command.memory_barriers.len() as u32,
                              command.memory_barriers.as_ptr(),
                              command.buffer_barriers.len() as u32,
                              command.buffer_barriers.as_ptr(),
                              command.image_barriers.len() as u32,
                              command.image_barriers.as_ptr());
    }

    /// Calls `vkCmdPushConstants` on the builder.
    #[inline]
    pub unsafe fn push_constants<Pl, D>(&mut self, pipeline_layout: &Pl, stages: ShaderStages,
                                        offset: u32, size: u32, data: &D)
        where Pl: ?Sized + PipelineLayoutAbstract,
              D: ?Sized
    {
        let vk = self.device().pointers();
        let cmd = self.internal_object();

        debug_assert!(stages != ShaderStages::none());
        debug_assert!(size > 0);
        debug_assert_eq!(size % 4, 0);
        debug_assert_eq!(offset % 4, 0);
        debug_assert!(mem::size_of_val(data) >= size as usize);

        vk.CmdPushConstants(cmd,
                            pipeline_layout.sys().internal_object(),
                            stages.into(),
                            offset as u32,
                            size as u32,
                            data as *const D as *const _);
    }

    /// Calls `vkCmdResetEvent` on the builder.
    #[inline]
    pub unsafe fn reset_event(&mut self, event: &Event, stages: PipelineStages) {
        let vk = self.device().pointers();
        let cmd = self.internal_object();

        debug_assert!(!stages.host);
        debug_assert_ne!(stages, PipelineStages::none());

        vk.CmdResetEvent(cmd, event.internal_object(), stages.into());
    }

    /// Calls `vkCmdSetBlendConstants` on the builder.
    #[inline]
    pub unsafe fn set_blend_constants(&mut self, constants: [f32; 4]) {
        let vk = self.device().pointers();
        let cmd = self.internal_object();
        vk.CmdSetBlendConstants(cmd, constants); // TODO: correct to pass array?
    }

    /// Calls `vkCmdSetDepthBias` on the builder.
    #[inline]
    pub unsafe fn set_depth_bias(&mut self, constant_factor: f32, clamp: f32, slope_factor: f32) {
        let vk = self.device().pointers();
        let cmd = self.internal_object();
        debug_assert!(clamp == 0.0 || self.device().enabled_features().depth_bias_clamp);
        vk.CmdSetDepthBias(cmd, constant_factor, clamp, slope_factor);
    }

    /// Calls `vkCmdSetDepthBounds` on the builder.
    #[inline]
    pub unsafe fn set_depth_bounds(&mut self, min: f32, max: f32) {
        let vk = self.device().pointers();
        let cmd = self.internal_object();
        debug_assert!(min >= 0.0 && min <= 1.0);
        debug_assert!(max >= 0.0 && max <= 1.0);
        vk.CmdSetDepthBounds(cmd, min, max);
    }

    /// Calls `vkCmdSetEvent` on the builder.
    #[inline]
    pub unsafe fn set_event(&mut self, event: &Event, stages: PipelineStages) {
        let vk = self.device().pointers();
        let cmd = self.internal_object();

        debug_assert!(!stages.host);
        debug_assert_ne!(stages, PipelineStages::none());

        vk.CmdSetEvent(cmd, event.internal_object(), stages.into());
    }

    /// Calls `vkCmdSetLineWidth` on the builder.
    #[inline]
    pub unsafe fn set_line_width(&mut self, line_width: f32) {
        let vk = self.device().pointers();
        let cmd = self.internal_object();
        debug_assert!(line_width == 1.0 || self.device().enabled_features().wide_lines);
        vk.CmdSetLineWidth(cmd, line_width);
    }

    // TODO: missing enum for face_mask
    /*/// Calls `vkCmdSetStencilCompareMask` on the builder.
    #[inline]
    pub unsafe fn set_stencil_compare_mask(&mut self, face_mask: , compare_mask: u32) {
        let vk = self.device().pointers();
        let cmd = self.internal_object();
        vk.CmdSetStencilCompareMask(cmd, face_mask, compare_mask);
    }*/

    // TODO: missing enum for face_mask
    /*/// Calls `vkCmdSetStencilReference` on the builder.
    #[inline]
    pub unsafe fn set_stencil_reference(&mut self, face_mask: , reference: u32) {
        let vk = self.device().pointers();
        let cmd = self.internal_object();
        vk.CmdSetStencilReference(cmd, face_mask, reference);
    }*/

    // TODO: missing enum for face_mask
    /*/// Calls `vkCmdSetStencilWriteMask` on the builder.
    #[inline]
    pub unsafe fn set_stencil_write_mask(&mut self, face_mask: , write_mask: u32) {
        let vk = self.device().pointers();
        let cmd = self.internal_object();
        vk.CmdSetStencilWriteMask(cmd, face_mask, write_mask);
    }*/

    /// Calls `vkCmdSetScissor` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_scissor<I>(&mut self, first_scissor: u32, scissors: I)
        where I: Iterator<Item = Scissor>
    {
        let scissors = scissors
            .map(|v| v.clone().into())
            .collect::<SmallVec<[_; 16]>>();
        if scissors.is_empty() {
            return;
        }

        // TODO: missing a debug assert for limits on the actual scissor values
        debug_assert!((first_scissor == 0 && scissors.len() == 1) ||
                          self.device().enabled_features().multi_viewport);
        debug_assert!({
                          let max = self.device().physical_device().limits().max_viewports();
                          first_scissor + scissors.len() as u32 <= max
                      });

        let vk = self.device().pointers();
        let cmd = self.internal_object();
        vk.CmdSetScissor(cmd, first_scissor, scissors.len() as u32, scissors.as_ptr());
    }

    /// Calls `vkCmdSetViewport` on the builder.
    ///
    /// If the list is empty then the command is automatically ignored.
    #[inline]
    pub unsafe fn set_viewport<I>(&mut self, first_viewport: u32, viewports: I)
        where I: Iterator<Item = Viewport>
    {
        let viewports = viewports
            .map(|v| v.clone().into())
            .collect::<SmallVec<[_; 16]>>();
        if viewports.is_empty() {
            return;
        }

        debug_assert!((first_viewport == 0 && viewports.len() == 1) ||
                          self.device().enabled_features().multi_viewport);
        debug_assert!({
                          let max = self.device().physical_device().limits().max_viewports();
                          first_viewport + viewports.len() as u32 <= max
                      });

        let vk = self.device().pointers();
        let cmd = self.internal_object();
        vk.CmdSetViewport(cmd,
                          first_viewport,
                          viewports.len() as u32,
                          viewports.as_ptr());
    }

    /// Calls `vkCmdUpdateBuffer` on the builder.
    #[inline]
    pub unsafe fn update_buffer<B, D>(&mut self, buffer: &B, data: &D)
        where B: ?Sized + BufferAccess,
              D: ?Sized
    {
        let vk = self.device().pointers();
        let cmd = self.internal_object();

        let size = buffer.size();
        debug_assert_eq!(size % 4, 0);
        debug_assert!(size <= 65536);
        debug_assert!(size <= mem::size_of_val(data));

        let (buffer_handle, offset) = {
            let BufferInner {
                buffer: buffer_inner,
                offset,
            } = buffer.inner();
            debug_assert!(buffer_inner.usage_transfer_destination());
            debug_assert_eq!(offset % 4, 0);
            (buffer_inner.internal_object(), offset)
        };

        vk.CmdUpdateBuffer(cmd,
                           buffer_handle,
                           offset as vk::DeviceSize,
                           size as vk::DeviceSize,
                           data as *const D as *const _);
    }
}

unsafe impl<P> DeviceOwned for UnsafeCommandBufferBuilder<P> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl<P> VulkanObject for UnsafeCommandBufferBuilder<P> {
    type Object = vk::CommandBuffer;

    #[inline]
    fn internal_object(&self) -> vk::CommandBuffer {
        debug_assert!(self.cmd.is_some());
        self.cmd_raw
    }
}

/// Prototype for a `vkCmdBindVertexBuffers`.
pub struct UnsafeCommandBufferBuilderBindVertexBuffer {
    // Raw handles of the buffers to bind.
    raw_buffers: SmallVec<[vk::Buffer; 4]>,
    // Raw offsets of the buffers to bind.
    offsets: SmallVec<[vk::DeviceSize; 4]>,
}

impl UnsafeCommandBufferBuilderBindVertexBuffer {
    /// Builds a new empty list.
    #[inline]
    pub fn new() -> UnsafeCommandBufferBuilderBindVertexBuffer {
        UnsafeCommandBufferBuilderBindVertexBuffer {
            raw_buffers: SmallVec::new(),
            offsets: SmallVec::new(),
        }
    }

    /// Adds a buffer to the list.
    #[inline]
    pub fn add<B>(&mut self, buffer: &B)
        where B: ?Sized + BufferAccess
    {
        let inner = buffer.inner();
        debug_assert!(inner.buffer.usage_vertex_buffer());
        self.raw_buffers.push(inner.buffer.internal_object());
        self.offsets.push(inner.offset as vk::DeviceSize);
    }
}

/// Prototype for a `vkCmdExecuteCommands`.
pub struct UnsafeCommandBufferBuilderExecuteCommands {
    // Raw handles of the command buffers to execute.
    raw_cbs: SmallVec<[vk::CommandBuffer; 4]>,
}

impl UnsafeCommandBufferBuilderExecuteCommands {
    /// Builds a new empty list.
    #[inline]
    pub fn new() -> UnsafeCommandBufferBuilderExecuteCommands {
        UnsafeCommandBufferBuilderExecuteCommands { raw_cbs: SmallVec::new() }
    }

    /// Adds a command buffer to the list.
    #[inline]
    pub fn add<C>(&mut self, cb: &C)
        where C: ?Sized + CommandBuffer
    {
        // TODO: debug assert that it is a secondary command buffer?
        self.raw_cbs.push(cb.inner().internal_object());
    }
}

// TODO: move somewhere else?
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct UnsafeCommandBufferBuilderImageAspect {
    pub color: bool,
    pub depth: bool,
    pub stencil: bool,
}

impl UnsafeCommandBufferBuilderImageAspect {
    pub(crate) fn to_vk_bits(&self) -> vk::ImageAspectFlagBits {
        let mut out = 0;
        if self.color {
            out |= vk::IMAGE_ASPECT_COLOR_BIT
        };
        if self.depth {
            out |= vk::IMAGE_ASPECT_DEPTH_BIT
        };
        if self.stencil {
            out |= vk::IMAGE_ASPECT_STENCIL_BIT
        };
        out
    }
}

// TODO: move somewhere else?
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct UnsafeCommandBufferBuilderBufferImageCopy {
    pub buffer_offset: usize,
    pub buffer_row_length: u32,
    pub buffer_image_height: u32,
    pub image_aspect: UnsafeCommandBufferBuilderImageAspect,
    pub image_mip_level: u32,
    pub image_base_array_layer: u32,
    pub image_layer_count: u32,
    pub image_offset: [i32; 3],
    pub image_extent: [u32; 3],
}

/// Command that adds a pipeline barrier to a command buffer builder.
///
/// A pipeline barrier is a low-level system-ish command that is often necessary for safety. By
/// default all commands that you add to a command buffer can potentially run simultaneously.
/// Adding a pipeline barrier separates commands before the barrier from commands after the barrier
/// and prevents them from running simultaneously.
///
/// Please take a look at the Vulkan specifications for more information. Pipeline barriers are a
/// complex topic and explaining them in this documentation would be redundant.
///
/// > **Note**: We use a builder-like API here so that users can pass multiple buffers or images of
/// > multiple different types. Doing so with a single function would be very tedious in terms of
/// > API.
pub struct UnsafeCommandBufferBuilderPipelineBarrier {
    src_stage_mask: vk::PipelineStageFlags,
    dst_stage_mask: vk::PipelineStageFlags,
    dependency_flags: vk::DependencyFlags,
    memory_barriers: SmallVec<[vk::MemoryBarrier; 2]>,
    buffer_barriers: SmallVec<[vk::BufferMemoryBarrier; 8]>,
    image_barriers: SmallVec<[vk::ImageMemoryBarrier; 8]>,
}

impl UnsafeCommandBufferBuilderPipelineBarrier {
    /// Creates a new empty pipeline barrier command.
    #[inline]
    pub fn new() -> UnsafeCommandBufferBuilderPipelineBarrier {
        UnsafeCommandBufferBuilderPipelineBarrier {
            src_stage_mask: 0,
            dst_stage_mask: 0,
            dependency_flags: vk::DEPENDENCY_BY_REGION_BIT,
            memory_barriers: SmallVec::new(),
            buffer_barriers: SmallVec::new(),
            image_barriers: SmallVec::new(),
        }
    }

    /// Returns true if no barrier or execution dependency has been added yet.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.src_stage_mask == 0 || self.dst_stage_mask == 0
    }

    /// Merges another pipeline builder into this one.
    #[inline]
    pub fn merge(&mut self, other: UnsafeCommandBufferBuilderPipelineBarrier) {
        self.src_stage_mask |= other.src_stage_mask;
        self.dst_stage_mask |= other.dst_stage_mask;
        self.dependency_flags &= other.dependency_flags;

        self.memory_barriers
            .extend(other.memory_barriers.into_iter());
        self.buffer_barriers
            .extend(other.buffer_barriers.into_iter());
        self.image_barriers.extend(other.image_barriers.into_iter());
    }

    /// Adds an execution dependency. This means that all the stages in `source` of the previous
    /// commands must finish before any of the stages in `destination` of the following commands can start.
    ///
    /// # Safety
    ///
    /// - If the pipeline stages include geometry or tessellation stages, then the corresponding
    ///   features must have been enabled in the device.
    /// - There are certain rules regarding the pipeline barriers inside render passes.
    ///
    #[inline]
    pub unsafe fn add_execution_dependency(&mut self, source: PipelineStages,
                                           destination: PipelineStages, by_region: bool) {
        if !by_region {
            self.dependency_flags = 0;
        }

        debug_assert_ne!(source, PipelineStages::none());
        debug_assert_ne!(destination, PipelineStages::none());

        self.src_stage_mask |= Into::<vk::PipelineStageFlags>::into(source);
        self.dst_stage_mask |= Into::<vk::PipelineStageFlags>::into(destination);
    }

    /// Adds a memory barrier. This means that all the memory writes by the given source stages
    /// for the given source accesses must be visible by the given destination stages for the given
    /// destination accesses.
    ///
    /// Also adds an execution dependency similar to `add_execution_dependency`.
    ///
    /// # Safety
    ///
    /// - Same as `add_execution_dependency`.
    ///
    pub unsafe fn add_memory_barrier(&mut self, source_stage: PipelineStages,
                                     source_access: AccessFlagBits,
                                     destination_stage: PipelineStages,
                                     destination_access: AccessFlagBits, by_region: bool) {
        debug_assert!(source_access.is_compatible_with(&source_stage));
        debug_assert!(destination_access.is_compatible_with(&destination_stage));

        self.add_execution_dependency(source_stage, destination_stage, by_region);

        self.memory_barriers.push(vk::MemoryBarrier {
                                      sType: vk::STRUCTURE_TYPE_MEMORY_BARRIER,
                                      pNext: ptr::null(),
                                      srcAccessMask: source_access.into(),
                                      dstAccessMask: destination_access.into(),
                                  });
    }

    /// Adds a buffer memory barrier. This means that all the memory writes to the given buffer by
    /// the given source stages for the given source accesses must be visible by the given dest
    /// stages for the given destination accesses.
    ///
    /// Also adds an execution dependency similar to `add_execution_dependency`.
    ///
    /// Also allows transfering buffer ownership between queues.
    ///
    /// # Safety
    ///
    /// - Same as `add_execution_dependency`.
    /// - The buffer must be alive for at least as long as the command buffer to which this barrier
    ///   is added.
    /// - Queue ownership transfers must be correct.
    ///
    pub unsafe fn add_buffer_memory_barrier<B>(&mut self, buffer: &B, source_stage: PipelineStages,
                                               source_access: AccessFlagBits,
                                               destination_stage: PipelineStages,
                                               destination_access: AccessFlagBits, by_region: bool,
                                               queue_transfer: Option<(u32, u32)>, offset: usize,
                                               size: usize)
        where B: ?Sized + BufferAccess
    {
        debug_assert!(source_access.is_compatible_with(&source_stage));
        debug_assert!(destination_access.is_compatible_with(&destination_stage));

        self.add_execution_dependency(source_stage, destination_stage, by_region);

        debug_assert!(size <= buffer.size());
        let BufferInner {
            buffer,
            offset: org_offset,
        } = buffer.inner();
        let offset = offset + org_offset;

        let (src_queue, dest_queue) = if let Some((src_queue, dest_queue)) = queue_transfer {
            (src_queue, dest_queue)
        } else {
            (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
        };

        self.buffer_barriers.push(vk::BufferMemoryBarrier {
                                      sType: vk::STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                      pNext: ptr::null(),
                                      srcAccessMask: source_access.into(),
                                      dstAccessMask: destination_access.into(),
                                      srcQueueFamilyIndex: src_queue,
                                      dstQueueFamilyIndex: dest_queue,
                                      buffer: buffer.internal_object(),
                                      offset: offset as vk::DeviceSize,
                                      size: size as vk::DeviceSize,
                                  });
    }

    /// Adds an image memory barrier. This is the equivalent of `add_buffer_memory_barrier` but
    /// for images.
    ///
    /// In addition to transfering image ownership between queues, it also allows changing the
    /// layout of images.
    ///
    /// Also adds an execution dependency similar to `add_execution_dependency`.
    ///
    /// # Safety
    ///
    /// - Same as `add_execution_dependency`.
    /// - The buffer must be alive for at least as long as the command buffer to which this barrier
    ///   is added.
    /// - Queue ownership transfers must be correct.
    /// - Image layouts transfers must be correct.
    /// - Access flags must be compatible with the image usage flags passed at image creation.
    ///
    pub unsafe fn add_image_memory_barrier<I>(&mut self, image: &I, mipmaps: Range<u32>,
                                              layers: Range<u32>, source_stage: PipelineStages,
                                              source_access: AccessFlagBits,
                                              destination_stage: PipelineStages,
                                              destination_access: AccessFlagBits, by_region: bool,
                                              queue_transfer: Option<(u32, u32)>,
                                              current_layout: ImageLayout, new_layout: ImageLayout)
        where I: ?Sized + ImageAccess
    {
        debug_assert!(source_access.is_compatible_with(&source_stage));
        debug_assert!(destination_access.is_compatible_with(&destination_stage));

        self.add_execution_dependency(source_stage, destination_stage, by_region);

        debug_assert_ne!(new_layout, ImageLayout::Undefined);
        debug_assert_ne!(new_layout, ImageLayout::Preinitialized);

        debug_assert!(mipmaps.start < mipmaps.end);
        debug_assert!(mipmaps.end <= image.mipmap_levels());
        debug_assert!(layers.start < layers.end);
        debug_assert!(layers.end <= image.dimensions().array_layers());

        let (src_queue, dest_queue) = if let Some((src_queue, dest_queue)) = queue_transfer {
            (src_queue, dest_queue)
        } else {
            (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
        };

        let aspect_mask = if image.has_color() {
            vk::IMAGE_ASPECT_COLOR_BIT
        } else if image.has_depth() && image.has_stencil() {
            vk::IMAGE_ASPECT_DEPTH_BIT | vk::IMAGE_ASPECT_STENCIL_BIT
        } else if image.has_depth() {
            vk::IMAGE_ASPECT_DEPTH_BIT
        } else if image.has_stencil() {
            vk::IMAGE_ASPECT_STENCIL_BIT
        } else {
            unreachable!()
        };

        let image = image.inner();

        self.image_barriers.push(vk::ImageMemoryBarrier {
                                     sType: vk::STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                     pNext: ptr::null(),
                                     srcAccessMask: source_access.into(),
                                     dstAccessMask: destination_access.into(),
                                     oldLayout: current_layout as u32,
                                     newLayout: new_layout as u32,
                                     srcQueueFamilyIndex: src_queue,
                                     dstQueueFamilyIndex: dest_queue,
                                     image: image.image.internal_object(),
                                     subresourceRange: vk::ImageSubresourceRange {
                                         aspectMask: aspect_mask,
                                         baseMipLevel: mipmaps.start + image.first_mipmap_level as u32,
                                         levelCount: mipmaps.end - mipmaps.start,
                                         baseArrayLayer: layers.start + image.first_layer as u32,
                                         layerCount: layers.end - layers.start,
                                     },
                                 });
    }
}

/// Command buffer that has been built.
///
/// Doesn't perform any synchronization and doesn't keep the object it uses alive.
pub struct UnsafeCommandBuffer<P> {
    // The Vulkan command buffer.
    cmd: P,

    // The raw version of `cmd`. Avoids having to require a trait on `P`.
    cmd_raw: vk::CommandBuffer,

    // Device that owns the command buffer.
    // TODO: necessary?
    device: Arc<Device>,

    // Flags that were used at creation.
    // TODO: necessary?
    flags: Flags,

    // True if the command buffer has always been submitted once. Only relevant if `flags` is
    // `OneTimeSubmit`.
    // TODO: remove?
    already_submitted: AtomicBool,
}

unsafe impl<P> DeviceOwned for UnsafeCommandBuffer<P> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl<P> VulkanObject for UnsafeCommandBuffer<P> {
    type Object = vk::CommandBuffer;

    #[inline]
    fn internal_object(&self) -> vk::CommandBuffer {
        self.cmd_raw
    }
}
