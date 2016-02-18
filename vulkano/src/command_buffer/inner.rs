use std::mem;
use std::ptr;
use std::sync::Arc;

use buffer::Buffer;
use buffer::BufferSlice;
use command_buffer::CommandBufferPool;
use command_buffer::DynamicState;
use device::Queue;
use framebuffer::ClearValue;
use framebuffer::Framebuffer;
use framebuffer::RenderPass;
use framebuffer::RenderPassLayout;
use memory::MemorySourceChunk;
use pipeline::GraphicsPipeline;
use pipeline::vertex::MultiVertex;
use sync::Fence;
use sync::Resource;
use sync::Semaphore;

use device::Device;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// Actual implementation of all command buffer builders.
///
/// Doesn't check whether the command type is appropriate for the command buffer type.
pub struct InnerCommandBufferBuilder {
    device: Arc<Device>,
    pool: Arc<CommandBufferPool>,
    cmd: Option<vk::CommandBuffer>,

    // List of all resources that are used by this command buffer.
    resources: Vec<Arc<Resource>>,

    // Current pipeline object binded to the graphics bind point.
    graphics_pipeline: Option<vk::Pipeline>,

    // Current pipeline object binded to the compute bind point.
    compute_pipeline: Option<vk::Pipeline>,

    // Current state of the dynamic state within the command buffer.
    dynamic_state: DynamicState,

    // When we use a buffer whose sharing mode is exclusive in a different queue family, we have
    // to transfer back ownership to the original queue family. To do so, we store the list of
    // barriers that must be queued before calling `vkEndCommandBuffer`.
    buffer_restore_queue_family: Vec<vk::BufferMemoryBarrier>,
}

impl InnerCommandBufferBuilder {
    /// Creates a new builder.
    pub fn new(pool: &Arc<CommandBufferPool>, secondary: bool)
               -> Result<InnerCommandBufferBuilder, OomError>
    {
        let device = pool.device();
        let vk = device.pointers();

        let cmd = unsafe {
            let infos = vk::CommandBufferAllocateInfo {
                sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                pNext: ptr::null(),
                commandPool: pool.internal_object(),
                level: if secondary {
                    vk::COMMAND_BUFFER_LEVEL_SECONDARY
                } else {
                    vk::COMMAND_BUFFER_LEVEL_PRIMARY
                },
                // vulkan can allocate multiple command buffers at once, hence the 1
                commandBufferCount: 1,
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.AllocateCommandBuffers(device.internal_object(), &infos,
                                                        &mut output)));
            output
        };

        unsafe {
            let infos = vk::CommandBufferBeginInfo {
                sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                pNext: ptr::null(),
                flags: vk::COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,       // TODO:
                pInheritanceInfo: ptr::null(),     // TODO: 
            };

            try!(check_errors(vk.BeginCommandBuffer(cmd, &infos)));
        }

        Ok(InnerCommandBufferBuilder {
            device: device.clone(),
            pool: pool.clone(),
            cmd: Some(cmd),
            resources: Vec::new(),
            graphics_pipeline: None,
            compute_pipeline: None,
            dynamic_state: DynamicState::none(),
            buffer_restore_queue_family: Vec::new(),
        })
    }

    /// Executes the content of another command buffer.
    ///
    /// # Safety
    ///
    /// Care must be taken to respect the rules about secondary command buffers.
    pub unsafe fn execute_commands<'a, I>(mut self, iter: I)
                                          -> InnerCommandBufferBuilder
        where I: Iterator<Item = &'a InnerCommandBuffer>
    {
        {
            let mut command_buffers = Vec::with_capacity(iter.size_hint().0);

            for cb in iter {
                command_buffers.push(cb.cmd);
                for r in cb.resources.iter() { self.resources.push(r.clone()); }
            }

            let vk = self.device.pointers();
            vk.CmdExecuteCommands(self.cmd.unwrap(), command_buffers.len() as u32,
                                  command_buffers.as_ptr());
        }

        self
    }

    /// Writes data to a buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if the size of `data` is not the same as the size of the buffer slice.
    /// - Panicks if the size of `data` is superior to 65536 bytes.
    /// - Panicks if the offset or size is not a multiple of 4.
    /// - Panicks if the buffer wasn't created with the right usage.
    /// - Panicks if the queue family doesn't support transfer operations.
    ///
    /// # Safety
    ///
    /// - Care must be taken to respect the rules about secondary command buffers.
    ///
    pub unsafe fn update_buffer<'a, B, T: 'a, M: 'a>(self, buffer: B, data: &T)
                                                 -> InnerCommandBufferBuilder
        where B: Into<BufferSlice<'a, T, M>>
    {
        {
            let vk = self.device.pointers();
            let buffer = buffer.into();

            assert!(self.pool.queue_family().supports_transfers());
            assert_eq!(buffer.size(), mem::size_of_val(data));
            assert!(buffer.size() <= 65536);
            assert!(buffer.offset() % 4 == 0);
            assert!(buffer.size() % 4 == 0);
            assert!(buffer.usage_transfer_dest());

            // FIXME: check that the queue family supports transfers
            // FIXME: add the buffer to the list of resources
            // FIXME: check queue family of the buffer

            vk.CmdUpdateBuffer(self.cmd.unwrap(), buffer.internal_object(),
                               buffer.offset() as vk::DeviceSize,
                               buffer.size() as vk::DeviceSize, data as *const T as *const _);
        }

        self
    }

    /// Fills a buffer with data.
    ///
    /// # Panic
    ///
    /// - Panicks if `offset + data` is superior to the size of the buffer.
    /// - Panicks if the offset or size is not a multiple of 4.
    /// - Panicks if the buffer wasn't created with the right usage.
    /// - Panicks if the queue family doesn't support transfer operations.
    ///
    /// # Safety
    ///
    /// - Type safety is not enforced by the API.
    /// - Care must be taken to respect the rules about secondary command buffers.
    ///
    pub unsafe fn fill_buffer<T: 'static, M>(mut self, buffer: &Arc<Buffer<T, M>>, offset: usize,
                                             size: usize, data: u32) -> InnerCommandBufferBuilder
        where M: MemorySourceChunk + 'static
    {
        {
            let vk = self.device.pointers();

            assert!(self.pool.queue_family().supports_transfers());
            assert!(offset + size <= buffer.size());
            assert!(offset % 4 == 0);
            assert!(size % 4 == 0);
            assert!(buffer.usage_transfer_dest());

            self.resources.push(buffer.clone());

            // FIXME: check that the queue family supports transfers
            // FIXME: check queue family of the buffer

            vk.CmdFillBuffer(self.cmd.unwrap(), buffer.internal_object(),
                             offset as vk::DeviceSize, size as vk::DeviceSize, data);
        }

        self
    }

    /*fn copy_buffer<I>(source: &Arc<Buffer>, destination: &Arc<Buffer>, copies: I)
                      -> InnerCommandBufferBuilder
        where I: IntoIter<Item = CopyCommand>
    {
            assert!(self.pool.queue_family().supports_transfers());
        // TODO: check values
        let copies = copies.into_iter().map(|command| {
            vk::BufferCopy {
                srcOffset: command.source_offset,
                dstOffset: command.destination_offset,
                size: command.size,
            }
        }).collect::<Vec<_>>();

        vk.CmdCopyBuffer(self.cmd.unwrap(), source.internal_object(), destination.internal_object(),
                         copies.len(), copies.as_ptr());
    }*/

    /// Calls `vkCmdDraw`.
    // FIXME: push constants
    pub unsafe fn draw<V>(mut self, pipeline: &Arc<GraphicsPipeline<V>>,
                      vertices: V, dynamic: &DynamicState)
                                -> InnerCommandBufferBuilder
        where V: MultiVertex
    {

        // FIXME: add buffers to the resources

        {
            self.bind_gfx_pipeline_state(pipeline, dynamic);

            let vk = self.device.pointers();

            let ids = vertices.ids();
            let offsets = (0 .. ids.len()).map(|_| 0).collect::<Vec<_>>();
            vk.CmdBindVertexBuffers(self.cmd.unwrap(), 0, ids.len() as u32, ids.as_ptr(),
                                    offsets.as_ptr());
            vk.CmdDraw(self.cmd.unwrap(), 3, 1, 0, 0);  // FIXME: params
        }

        self
    }

    fn bind_gfx_pipeline_state<V>(&mut self, pipeline: &Arc<GraphicsPipeline<V>>,
                                  dynamic: &DynamicState)
    {
        let vk = self.device.pointers();

        if self.graphics_pipeline != Some(pipeline.internal_object()) {
            // FIXME: add pipeline to resources list
            unsafe {
                vk.CmdBindPipeline(self.cmd.unwrap(), vk::PIPELINE_BIND_POINT_GRAPHICS,
                                   pipeline.internal_object());
            }
            self.graphics_pipeline = Some(pipeline.internal_object());
        }

        if let Some(line_width) = dynamic.line_width {
            assert!(pipeline.has_dynamic_line_width());
            // TODO: check limits
            if self.dynamic_state.line_width != Some(line_width) {
                unsafe { vk.CmdSetLineWidth(self.cmd.unwrap(), line_width) };
                self.dynamic_state.line_width = Some(line_width);
            }
        } else {
            assert!(!pipeline.has_dynamic_line_width());
        }
    }

    /// Calls `vkCmdBeginRenderPass`.
    ///
    /// # Panic
    ///
    /// - Panicks if the framebuffer is not compatible with the renderpass.
    ///
    /// # Safety
    ///
    /// - Care must be taken to respect the rules about secondary command buffers.
    ///
    #[inline]
    pub unsafe fn begin_renderpass<R, F>(self, renderpass: &Arc<RenderPass<R>>,
                                     framebuffer: &Arc<Framebuffer<F>>,
                                     secondary_cmd_buffers: bool,
                                     clear_values: &[ClearValue]) -> InnerCommandBufferBuilder
        where R: RenderPassLayout
    {
        // FIXME: framebuffer synchronization

        assert!(framebuffer.is_compatible_with(renderpass));

        let clear_values = clear_values.iter().map(|value| {
            match *value {
                ClearValue::None => vk::ClearValue::color({
                    vk::ClearColorValue::float32([0.0, 0.0, 0.0, 0.0])
                }),
                ClearValue::Float(data) => vk::ClearValue::color(vk::ClearColorValue::float32(data)),
                ClearValue::Int(data) => vk::ClearValue::color(vk::ClearColorValue::int32(data)),
                ClearValue::Uint(data) => vk::ClearValue::color(vk::ClearColorValue::uint32(data)),
                ClearValue::Depth(d) => vk::ClearValue::depth_stencil({
                    vk::ClearDepthStencilValue { depth: d, stencil: 0 }
                }),
                ClearValue::Stencil(s) => vk::ClearValue::depth_stencil({
                    vk::ClearDepthStencilValue { depth: 0.0, stencil: s }
                }),
                ClearValue::DepthStencil((d, s)) => vk::ClearValue::depth_stencil({
                    vk::ClearDepthStencilValue { depth: d, stencil: s }
                }),
            }
        }).collect::<Vec<_>>();

        // TODO: change attachment image layouts if necessary, for both initial and final
        /*for attachment in R::attachments() {

        }*/

        {
            let vk = self.device.pointers();

            let infos = vk::RenderPassBeginInfo {
                sType: vk::STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                pNext: ptr::null(),
                renderPass: renderpass.internal_object(),
                framebuffer: framebuffer.internal_object(),
                renderArea: vk::Rect2D {                // TODO: let user customize
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: framebuffer.width(),
                        height: framebuffer.height(),
                    },
                },
                clearValueCount: clear_values.len() as u32,
                pClearValues: clear_values.as_ptr(),
            };

            let content = if secondary_cmd_buffers {
                vk::SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS
            } else {
                vk::SUBPASS_CONTENTS_INLINE
            };

            vk.CmdBeginRenderPass(self.cmd.unwrap(), &infos, content);
        }

        self
    }

    #[inline]
    pub unsafe fn next_subpass(self, secondary_cmd_buffers: bool) -> InnerCommandBufferBuilder {
        {
            let vk = self.device.pointers();

            let content = if secondary_cmd_buffers {
                vk::SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS
            } else {
                vk::SUBPASS_CONTENTS_INLINE
            };

            vk.CmdNextSubpass(self.cmd.unwrap(), content);
        }

        self
    }

    #[inline]
    pub unsafe fn end_renderpass(self) -> InnerCommandBufferBuilder {
        {
            let vk = self.device.pointers();
            vk.CmdEndRenderPass(self.cmd.unwrap());
        }

        self
    }

    /// Finishes building the command buffer.
    pub fn build(mut self) -> Result<InnerCommandBuffer, OomError> {
        unsafe {
            let vk = self.device.pointers();
            let cmd = self.cmd.take().unwrap();

            // committing the necessary barriers
            if !self.buffer_restore_queue_family.is_empty() {
                vk.CmdPipelineBarrier(cmd, vk::PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                      vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                                      0, ptr::null(),
                                      self.buffer_restore_queue_family.len() as u32,
                                      self.buffer_restore_queue_family.as_ptr(),
                                      0, ptr::null());
            }

            // ending the commands recording
            try!(check_errors(vk.EndCommandBuffer(cmd)));

            Ok(InnerCommandBuffer {
                device: self.device.clone(),
                pool: self.pool.clone(),
                cmd: cmd,
                resources: mem::replace(&mut self.resources, Vec::new()),
            })
        }
    }
}

impl Drop for InnerCommandBufferBuilder {
    #[inline]
    fn drop(&mut self) {
        if let Some(cmd) = self.cmd {
            unsafe {
                let vk = self.device.pointers();
                vk.EndCommandBuffer(cmd);
                vk.FreeCommandBuffers(self.device.internal_object(), self.pool.internal_object(),
                                      1, &cmd);
            }
        }
    }
}

/// Actual implementation of all command buffers.
pub struct InnerCommandBuffer {
    device: Arc<Device>,
    pool: Arc<CommandBufferPool>,
    cmd: vk::CommandBuffer,
    resources: Vec<Arc<Resource>>,
}

impl InnerCommandBuffer {
    /// Submits the command buffer to a queue.
    ///
    /// Queues are not thread-safe, therefore we need to get a `&mut`.
    ///
    /// # Panic
    ///
    /// - Panicks if the queue doesn't belong to the device this command buffer was created with.
    /// - Panicks if the queue doesn't belong to the family the pool was created with.
    ///
    pub fn submit(&self, queue: &mut Queue) -> Result<(), OomError> {       // TODO: wrong error type
        // FIXME: the whole function should be checked
        let vk = self.device.pointers();

        assert_eq!(queue.device().internal_object(), self.pool.device().internal_object());
        assert_eq!(queue.family().id(), self.pool.queue_family().id());

        // FIXME: fence shouldn't be discarded, as it could be ignored by resources and
        //        destroyed while in use
        let fence = if self.resources.iter().any(|r| r.requires_fence()) {
            Some(try!(Fence::new(queue.device())))
        } else {
            None
        };

        let mut post_semaphores = Vec::new();
        let mut post_semaphores_ids = Vec::new();
        let mut pre_semaphores = Vec::new();
        let mut pre_semaphores_ids = Vec::new();
        let mut pre_semaphores_stages = Vec::new();

        // FIXME: pre-semaphores shouldn't be discarded as they could be deleted while in use
        //        they should be included in a return value instead
        // FIXME: same for post-semaphores

        for resource in self.resources.iter() {
            let post_semaphore = if resource.requires_semaphore() {
                let semaphore = try!(Semaphore::new(queue.device()));
                post_semaphores.push(semaphore.clone());
                post_semaphores_ids.push(semaphore.internal_object());
                Some(semaphore)

            } else {
                None
            };

            // FIXME: for the moment `write` is always true ; that shouldn't be the case
            let (s1, s2) = resource.gpu_access(true, queue, fence.clone(), post_semaphore);

            if let Some(s) = s1 {
                pre_semaphores_ids.push(s.internal_object());
                pre_semaphores_stages.push(vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT);     // TODO:
                pre_semaphores.push(s);
            }

            if let Some(s) = s2 {
                pre_semaphores_ids.push(s.internal_object());
                pre_semaphores_stages.push(vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT);     // TODO:
                pre_semaphores.push(s);
            }
        }

        let infos = vk::SubmitInfo {
            sType: vk::STRUCTURE_TYPE_SUBMIT_INFO,
            pNext: ptr::null(),
            waitSemaphoreCount: pre_semaphores_ids.len() as u32,
            pWaitSemaphores: pre_semaphores_ids.as_ptr(),
            pWaitDstStageMask: pre_semaphores_stages.as_ptr(),
            commandBufferCount: 1,
            pCommandBuffers: &self.cmd,
            signalSemaphoreCount: post_semaphores_ids.len() as u32,
            pSignalSemaphores: post_semaphores_ids.as_ptr(),
        };

        unsafe {
            let fence = if let Some(ref fence) = fence { fence.internal_object() } else { 0 };
            try!(check_errors(vk.QueueSubmit(queue.internal_object(), 1, &infos, fence)));
        }

        // FIXME: the return value shouldn't be () because the command buffer
        //        could be deleted while in use

        Ok(())
    }

/*  TODO:
    fn reset() -> InnerCommandBufferBuilder {

    }*/
}

impl Drop for InnerCommandBuffer {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.FreeCommandBuffers(self.device.internal_object(), self.pool.internal_object(),
                                  1, &self.cmd);
        }
    }
}
