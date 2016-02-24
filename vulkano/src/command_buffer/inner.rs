use std::mem;
use std::ptr;
use std::sync::Arc;

use buffer::Buffer;
use buffer::BufferSlice;
use buffer::AbstractBuffer;
use command_buffer::CommandBufferPool;
use command_buffer::DynamicState;
use descriptor_set::PipelineLayoutDesc;
use descriptor_set::DescriptorSetsCollection;
use device::Queue;
use formats::ClearValue;
use formats::FloatOrCompressedFormatMarker;
use formats::FloatFormatMarker;
use formats::StrongStorage;
use framebuffer::Framebuffer;
use framebuffer::RenderPass;
use framebuffer::RenderPassLayout;
use image::AbstractImageView;
use image::Image;
use image::ImageTypeMarker;
use memory::MemorySourceChunk;
use pipeline::GenericPipeline;
use pipeline::GraphicsPipeline;
use pipeline::input_assembly::Index;
use pipeline::vertex::MultiVertex;
use sync::Fence;
use sync::Resource;
use sync::Semaphore;

use device::Device;
use OomError;
use SynchronizedVulkanObject;
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
    buffer_resources: Vec<Arc<AbstractBuffer>>,

    // Same as `resources`. Should be merged with `resources` once Rust allows turning a
    // `Arc<AbstractImageView>` into an `Arc<AbstractBuffer>`.
    image_resources: Vec<Arc<AbstractImageView>>,

    // List of pipelines that are used by this command buffer.
    //
    // These are stored just so that they don't get destroyed.
    pipelines: Vec<Arc<GenericPipeline>>,

    // Current pipeline object binded to the graphics bind point.
    graphics_pipeline: Option<vk::Pipeline>,

    // Current pipeline object binded to the compute bind point.
    compute_pipeline: Option<vk::Pipeline>,

    // Current state of the dynamic state within the command buffer.
    dynamic_state: DynamicState,
}

impl InnerCommandBufferBuilder {
    /// Creates a new builder.
    pub fn new(pool: &Arc<CommandBufferPool>, secondary: bool)
               -> Result<InnerCommandBufferBuilder, OomError>
    {
        let device = pool.device();
        let vk = device.pointers();

        let pool_obj = pool.internal_object_guard();

        let cmd = unsafe {
            let infos = vk::CommandBufferAllocateInfo {
                sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                pNext: ptr::null(),
                commandPool: *pool_obj,
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
            buffer_resources: Vec::new(),
            image_resources: Vec::new(),
            pipelines: Vec::new(),
            graphics_pipeline: None,
            compute_pipeline: None,
            dynamic_state: DynamicState::none(),
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
        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let mut command_buffers = Vec::with_capacity(iter.size_hint().0);

        for cb in iter {
            command_buffers.push(cb.cmd);
            for p in cb.pipelines.iter() { self.pipelines.push(p.clone()); }
            for r in cb.buffer_resources.iter() { self.buffer_resources.push(r.clone()); }
            for r in cb.image_resources.iter() { self.image_resources.push(r.clone()); }
        }

        {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
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
    pub unsafe fn update_buffer<B, T>(self, buffer: B, data: &T)
                                      -> InnerCommandBufferBuilder
        where B: Into<BufferSlice<T>>
    {
        let buffer = buffer.into();

        assert!(self.pool.queue_family().supports_transfers());
        assert_eq!(buffer.size(), mem::size_of_val(data));
        assert!(buffer.size() <= 65536);
        assert!(buffer.offset() % 4 == 0);
        assert!(buffer.size() % 4 == 0);
        assert!(buffer.buffer().usage_transfer_dest());

        // FIXME: check that the queue family supports transfers
        // FIXME: add the buffer to the list of resources
        // FIXME: check queue family of the buffer

        {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            vk.CmdUpdateBuffer(self.cmd.unwrap(), buffer.buffer().internal_object(),
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

        assert!(self.pool.queue_family().supports_transfers());
        assert!(offset + size <= buffer.size());
        assert!(offset % 4 == 0);
        assert!(size % 4 == 0);
        assert!(buffer.usage_transfer_dest());

        self.add_buffer_resource(buffer.clone(), true, offset, size);

        // FIXME: check that the queue family supports transfers
        // FIXME: check queue family of the buffer

        {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            vk.CmdFillBuffer(self.cmd.unwrap(), buffer.internal_object(),
                             offset as vk::DeviceSize, size as vk::DeviceSize, data);
        }

        self
    }

    /// Copies data between buffers.
    ///
    /// # Panic
    ///
    /// - Panicks if the buffers don't belong to the same device.
    /// - Panicks if one of the buffers wasn't created with the right usage.
    /// - Panicks if the queue family doesn't support transfer operations.
    ///
    /// # Safety
    ///
    /// - Type safety is not enforced by the API.
    /// - Care must be taken to respect the rules about secondary command buffers.
    ///
    // TODO: doesn't support slices
    pub unsafe fn copy_buffer<T: ?Sized + 'static, Ms, Md>(mut self, source: &Arc<Buffer<T, Ms>>,
                                                           destination: &Arc<Buffer<T, Md>>)
                                                           -> InnerCommandBufferBuilder
        where Ms: MemorySourceChunk + 'static, Md: MemorySourceChunk + 'static
    {
        assert_eq!(&**source.device() as *const _, &**destination.device() as *const _);
        assert!(self.pool.queue_family().supports_transfers());
        assert!(source.usage_transfer_src());
        assert!(destination.usage_transfer_dest());

        let copy = vk::BufferCopy {
            srcOffset: 0,
            dstOffset: 0,
            size: source.size() as u64,     // FIXME: what is destination is too small?
        };

        self.add_buffer_resource(source.clone(), false, 0, source.size());
        self.add_buffer_resource(destination.clone(), true, 0, source.size());

        {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            vk.CmdCopyBuffer(self.cmd.unwrap(), source.internal_object(),
                             destination.internal_object(), 1, &copy);
        }

        self
    }

    ///
    /// Note that compressed formats are not supported.
    ///
    /// # Safety
    ///
    /// - Care must be taken to respect the rules about secondary command buffers.
    ///
    pub unsafe fn clear_color_image<'a, Ty, F, M>(self, image: &Arc<Image<Ty, F, M>>,
                                                  color: F::ClearValue) -> InnerCommandBufferBuilder
        where Ty: ImageTypeMarker, F: FloatFormatMarker
    {
        let color = match color.into() {
            ClearValue::Float(data) => vk::ClearColorValue::float32(data),
            ClearValue::Int(data) => vk::ClearColorValue::int32(data),
            ClearValue::Uint(data) => vk::ClearColorValue::uint32(data),
            _ => unreachable!()   // FloatOrCompressedFormatMarker has been improperly implemented
        };

        let range = vk::ImageSubresourceRange {
            aspectMask: vk::IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel: 0,        // FIXME:
            levelCount: 1,      // FIXME:
            baseArrayLayer: 0,      // FIXME:
            layerCount: 1,      // FIXME:
        };

        {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            vk.CmdClearColorImage(self.cmd.unwrap(), image.internal_object(), vk::IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL /* FIXME: */,
                                  &color, 1, &range);
        }

        self
    }

    /// Copies data from a buffer to a color image.
    ///
    /// This operation can be performed by any kind of queue.
    ///
    /// # Safety
    ///
    /// - Care must be taken to respect the rules about secondary command buffers.
    ///
    pub unsafe fn copy_buffer_to_color_image<S, Ty, F, Im>(self, source: S, image: &Arc<Image<Ty, F, Im>>)
                                                           -> InnerCommandBufferBuilder
        where S: Into<BufferSlice<[F::Pixel]>>, F: StrongStorage + FloatOrCompressedFormatMarker,
              Ty: ImageTypeMarker
    {
        let source = source.into();
        //self.add_buffer_resource(source)      // FIXME:

        let region = vk::BufferImageCopy {
            bufferOffset: source.offset() as vk::DeviceSize,
            bufferRowLength: 0,
            bufferImageHeight: 0,
            imageSubresource: vk::ImageSubresourceLayers {
                aspectMask: vk::IMAGE_ASPECT_COLOR_BIT,
                mipLevel: 0,            // FIXME:
                baseArrayLayer: 0,          // FIXME:
                layerCount: 1,          // FIXME:
            },
            imageOffset: vk::Offset3D {
                x: 0,           // FIXME:
                y: 0,           // FIXME:
                z: 0,           // FIXME:
            },
            imageExtent: vk::Extent3D {
                width: 93,         // FIXME:
                height: 93,            // FIXME:
                depth: 1,         // FIXME:
            },
        };

        {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            vk.CmdCopyBufferToImage(self.cmd.unwrap(), source.buffer().internal_object(), image.internal_object(),
                                    vk::IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL /* FIXME */,
                                    1, &region);
        }

        self
    }

    /// Calls `vkCmdDraw`.
    // FIXME: push constants
    pub unsafe fn draw<V, Pl, L>(mut self, pipeline: &Arc<GraphicsPipeline<V, Pl>>,
                             vertices: V, dynamic: &DynamicState,
                             sets: L) -> InnerCommandBufferBuilder
        where V: 'static + MultiVertex, L: 'static + DescriptorSetsCollection,
              Pl: 'static + PipelineLayoutDesc
    {
        // FIXME: add buffers to the resources

        self.bind_gfx_pipeline_state(pipeline, dynamic, sets);

        assert!(vertices.buffers().all(|b| b.usage_vertex_buffer()));
        let buffers = vertices.buffers();
        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let offsets = (0 .. buffers.len()).map(|_| 0).collect::<Vec<_>>();
        let ids = buffers.map(|b| b.internal_object()).collect::<Vec<_>>();

        {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            vk.CmdBindVertexBuffers(self.cmd.unwrap(), 0, ids.len() as u32, ids.as_ptr(),
                                    offsets.as_ptr());
            vk.CmdDraw(self.cmd.unwrap(), 4, 1, 0, 0);  // FIXME: params
        }

        self
    }

    /// Calls `vkCmdDrawIndexed`.
    // FIXME: push constants
    pub unsafe fn draw_indexed<V, Pl, L, I, Ib>(mut self, pipeline: &Arc<GraphicsPipeline<V, Pl>>,
                                                 vertices: V, indices: Ib, dynamic: &DynamicState,
                                                 sets: L) -> InnerCommandBufferBuilder
        where V: 'static + MultiVertex, L: 'static + DescriptorSetsCollection,
              Pl: 'static + PipelineLayoutDesc,
              Ib: Into<BufferSlice<[I]>>, I: 'static + Index
    {

        // FIXME: add buffers to the resources

        self.bind_gfx_pipeline_state(pipeline, dynamic, sets);


        let indices = indices.into();

        assert!(vertices.buffers().all(|b| b.usage_vertex_buffer()));
        let buffers = vertices.buffers();
        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let offsets = (0 .. buffers.len()).map(|_| 0).collect::<Vec<_>>();
        let ids = buffers.map(|b| b.internal_object()).collect::<Vec<_>>();

        assert!(indices.buffer().usage_index_buffer());

        self.add_buffer_resource(indices.buffer().clone(), false, indices.offset(), indices.size());

        {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            vk.CmdBindIndexBuffer(self.cmd.unwrap(), indices.buffer().internal_object(),
                                  indices.offset() as u64, I::ty() as u32);
            vk.CmdBindVertexBuffers(self.cmd.unwrap(), 0, ids.len() as u32, ids.as_ptr(),
                                    offsets.as_ptr());
            vk.CmdDrawIndexed(self.cmd.unwrap(), indices.len() as u32, 1, 0, 0, 0);  // FIXME: params
        }

        self
    }

    fn bind_gfx_pipeline_state<V, Pl, L>(&mut self, pipeline: &Arc<GraphicsPipeline<V, Pl>>,
                                         dynamic: &DynamicState, sets: L)
        where V: 'static + MultiVertex, L: 'static + DescriptorSetsCollection,
              Pl: 'static + PipelineLayoutDesc
    {
        unsafe {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            assert!(sets.is_compatible_with(pipeline.layout()));

            if self.graphics_pipeline != Some(pipeline.internal_object()) {
                vk.CmdBindPipeline(self.cmd.unwrap(), vk::PIPELINE_BIND_POINT_GRAPHICS,
                                   pipeline.internal_object());
                self.pipelines.push(pipeline.clone());
                self.graphics_pipeline = Some(pipeline.internal_object());
            }

            if let Some(line_width) = dynamic.line_width {
                assert!(pipeline.has_dynamic_line_width());
                // TODO: check limits
                if self.dynamic_state.line_width != Some(line_width) {
                    vk.CmdSetLineWidth(self.cmd.unwrap(), line_width);
                    self.dynamic_state.line_width = Some(line_width);
                }
            } else {
                assert!(!pipeline.has_dynamic_line_width());
            }

            // FIXME: keep these alive
            // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
            let descriptor_sets = sets.list().collect::<Vec<_>>();
            // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
            let descriptor_sets = descriptor_sets.into_iter().map(|set| set.internal_object()).collect::<Vec<_>>();

            // TODO: shouldn't rebind everything every time
            if !descriptor_sets.is_empty() {
                vk.CmdBindDescriptorSets(self.cmd.unwrap(), vk::PIPELINE_BIND_POINT_GRAPHICS,
                                         pipeline.layout().internal_object(), 0,
                                         descriptor_sets.len() as u32, descriptor_sets.as_ptr(),
                                         0, ptr::null());   // FIXME: dynamic offsets
            }
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
    pub unsafe fn begin_renderpass<R, F>(mut self, renderpass: &Arc<RenderPass<R>>,
                                         framebuffer: &Arc<Framebuffer<F>>,
                                         secondary_cmd_buffers: bool,
                                         clear_values: &[ClearValue]) -> InnerCommandBufferBuilder
        where R: RenderPassLayout, F: RenderPassLayout
    {
        assert!(framebuffer.is_compatible_with(renderpass));

        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
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

        // FIXME: change attachment image layouts if necessary, for both initial and final
        /*for attachment in R::attachments() {

        }*/

        for attachment in framebuffer.attachments() {
            self.image_resources.push(attachment.clone());
        }

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

        {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            vk.CmdBeginRenderPass(self.cmd.unwrap(), &infos, content);
        }

        self
    }

    #[inline]
    pub unsafe fn next_subpass(self, secondary_cmd_buffers: bool) -> InnerCommandBufferBuilder {
        let content = if secondary_cmd_buffers {
            vk::SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS
        } else {
            vk::SUBPASS_CONTENTS_INLINE
        };

        {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            vk.CmdNextSubpass(self.cmd.unwrap(), content);
        }

        self
    }

    #[inline]
    pub unsafe fn end_renderpass(self) -> InnerCommandBufferBuilder {
        {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            vk.CmdEndRenderPass(self.cmd.unwrap());
        }

        self
    }

    /// Finishes building the command buffer.
    pub fn build(mut self) -> Result<InnerCommandBuffer, OomError> {
        unsafe {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            let cmd = self.cmd.take().unwrap();

            // ending the commands recording
            try!(check_errors(vk.EndCommandBuffer(cmd)));

            Ok(InnerCommandBuffer {
                device: self.device.clone(),
                pool: self.pool.clone(),
                cmd: cmd,
                buffer_resources: mem::replace(&mut self.buffer_resources, Vec::new()),
                image_resources: mem::replace(&mut self.image_resources, Vec::new()),
                pipelines: mem::replace(&mut self.pipelines, Vec::new()),
            })
        }
    }

    /// Adds a buffer resource to the list of resources used by this command buffer.
    // FIXME: add access flags
    fn add_buffer_resource(&mut self, buffer: Arc<AbstractBuffer>, write: bool, offset: usize,
                           size: usize)
    {
        // TODO: handle memory barriers
        self.buffer_resources.push(buffer);
    }

    /// Adds an image resource to the list of resources used by this command buffer.
    fn add_image_resource(&mut self) {   // TODO:
        unimplemented!()
    }
}

impl Drop for InnerCommandBufferBuilder {
    #[inline]
    fn drop(&mut self) {
        if let Some(cmd) = self.cmd {
            unsafe {
                let vk = self.device.pointers();
                vk.EndCommandBuffer(cmd);

                let pool = self.pool.internal_object_guard();
                vk.FreeCommandBuffers(self.device.internal_object(), *pool, 1, &cmd);
            }
        }
    }
}

/// Actual implementation of all command buffers.
pub struct InnerCommandBuffer {
    device: Arc<Device>,
    pool: Arc<CommandBufferPool>,
    cmd: vk::CommandBuffer,
    buffer_resources: Vec<Arc<AbstractBuffer>>,
    image_resources: Vec<Arc<AbstractImageView>>,
    pipelines: Vec<Arc<GenericPipeline>>,
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
    pub fn submit(&self, queue: &Arc<Queue>) -> Result<(), OomError> {       // TODO: wrong error type
        // FIXME: the whole function should be checked
        let vk = self.device.pointers();

        assert_eq!(queue.device().internal_object(), self.pool.device().internal_object());
        assert_eq!(queue.family().id(), self.pool.queue_family().id());

        // FIXME: fence shouldn't be discarded, as it could be ignored by resources and
        //        destroyed while in use
        let fence = if self.buffer_resources.iter().any(|r| r.requires_fence()) ||
                       self.image_resources.iter().any(|r| r.requires_fence())
        {
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

        for resource in self.buffer_resources.iter() {
            let post_semaphore = if resource.requires_semaphore() {
                let semaphore = try!(Semaphore::new(queue.device()));
                post_semaphores.push(semaphore.clone());
                post_semaphores_ids.push(semaphore.internal_object());
                Some(semaphore)

            } else {
                None
            };

            // FIXME: for the moment `write` is always true ; that shouldn't be the case
            // FIXME: wrong offset and size
            let sem = unsafe {
                resource.gpu_access(true, 0, 18, queue, fence.clone(), post_semaphore)
            };

            if let Some(s) = sem {
                pre_semaphores_ids.push(s.internal_object());
                pre_semaphores_stages.push(vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT);     // TODO:
                pre_semaphores.push(s);
            }
        }

        for resource in self.image_resources.iter() {
            let post_semaphore = if resource.requires_semaphore() {
                let semaphore = try!(Semaphore::new(queue.device()));
                post_semaphores.push(semaphore.clone());
                post_semaphores_ids.push(semaphore.internal_object());
                Some(semaphore)

            } else {
                None
            };

            // FIXME: for the moment `write` is always true ; that shouldn't be the case
            let sem = unsafe {
                resource.gpu_access(true, queue, fence.clone(), post_semaphore)
            };

            if let Some(s) = sem {
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
            try!(check_errors(vk.QueueSubmit(*queue.internal_object_guard(), 1, &infos, fence)));
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
            let pool = self.pool.internal_object_guard();
            vk.FreeCommandBuffers(self.device.internal_object(), *pool, 1, &self.cmd);
        }
    }
}
