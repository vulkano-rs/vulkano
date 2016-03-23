use std::mem;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;
use smallvec::SmallVec;

use buffer::Buffer;
use buffer::BufferSlice;
use buffer::TypedBuffer;
use command_buffer::AbstractCommandBuffer;
use command_buffer::CommandBufferPool;
use command_buffer::DynamicState;
use descriptor_set::Layout as PipelineLayoutDesc;
use descriptor_set::DescriptorSetsCollection;
use descriptor_set::AbstractDescriptorSet;
use device::Queue;
use format::ClearValue;
use format::FormatDesc;
use format::PossibleFloatOrCompressedFormatDesc;
use format::PossibleFloatFormatDesc;
use format::StrongStorage;
use framebuffer::AbstractFramebuffer;
use framebuffer::RenderPass;
use framebuffer::Framebuffer;
use framebuffer::Subpass;
use image::AbstractImage;
use image::AbstractImageView;
use image::Image;
use image::ImageTypeMarker;
use memory::MemorySource;
use pipeline::GenericPipeline;
use pipeline::ComputePipeline;
use pipeline::GraphicsPipeline;
use pipeline::input_assembly::Index;
use pipeline::vertex::Definition as VertexDefinition;
use pipeline::vertex::Source as VertexSource;
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

    // List of secondary command buffers.
    secondary_command_buffers: Vec<Arc<AbstractCommandBuffer>>,

    // List of descriptor sets used in this CB.
    descriptor_sets: Vec<Arc<AbstractDescriptorSet>>,

    // List of framebuffers used in this CB.
    framebuffers: Vec<Arc<AbstractFramebuffer>>,

    // List of renderpasses used in this CB.
    renderpasses: Vec<Arc<RenderPass>>,

    // List of all resources that are used by this command buffer.
    buffer_resources: Vec<Arc<Buffer>>,

    image_resources: Vec<Arc<AbstractImage>>,

    // Same as `resources`. Should be merged with `resources` once Rust allows turning a
    // `Arc<AbstractImageView>` into an `Arc<Buffer>`.
    image_views_resources: Vec<Arc<AbstractImageView>>,

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
    pub fn new<R>(pool: &Arc<CommandBufferPool>, secondary: bool, secondary_cont: Option<Subpass<R>>,
                  secondary_cont_fb: Option<&Arc<Framebuffer<R>>>)
                  -> Result<InnerCommandBufferBuilder, OomError>
        where R: RenderPass + 'static
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
                    assert!(secondary_cont.is_some());
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

        let mut renderpasses = Vec::new();
        let mut framebuffers = Vec::new();

        unsafe {
            // TODO: one time submit
            let flags = vk::COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT |     // TODO:
                        if secondary_cont.is_some() { vk::COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT } else { 0 };

            let (rp, sp) = if let Some(ref sp) = secondary_cont {
                renderpasses.push(sp.render_pass().clone() as Arc<_>);
                (sp.render_pass().render_pass().internal_object(), sp.index())
            } else {
                (0, 0)
            };

            let framebuffer = if let Some(fb) = secondary_cont_fb {
                framebuffers.push(fb.clone() as Arc<_>);
                fb.internal_object()
            } else {
                0
            };

            let inheritance = vk::CommandBufferInheritanceInfo {
                sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
                pNext: ptr::null(),
                renderPass: rp,
                subpass: sp,
                framebuffer: framebuffer,
                occlusionQueryEnable: 0,            // TODO:
                queryFlags: 0,          // TODO:
                pipelineStatistics: 0,          // TODO:
            };

            let infos = vk::CommandBufferBeginInfo {
                sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                pNext: ptr::null(),
                flags: flags,
                pInheritanceInfo: &inheritance,
            };

            try!(check_errors(vk.BeginCommandBuffer(cmd, &infos)));
        }

        Ok(InnerCommandBufferBuilder {
            device: device.clone(),
            pool: pool.clone(),
            cmd: Some(cmd),
            secondary_command_buffers: Vec::new(),
            descriptor_sets: Vec::new(),
            framebuffers: framebuffers,
            renderpasses: renderpasses,
            buffer_resources: Vec::new(),
            image_resources: Vec::new(),
            image_views_resources: Vec::new(),
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
    pub unsafe fn execute_commands<'a>(mut self, cb_arc: Arc<AbstractCommandBuffer>,
                                       cb: &InnerCommandBuffer) -> InnerCommandBufferBuilder
    {
        self.secondary_command_buffers.push(cb_arc);

        for p in cb.pipelines.iter() { self.pipelines.push(p.clone()); }
        for r in cb.buffer_resources.iter() { self.buffer_resources.push(r.clone()); }
        for r in cb.image_views_resources.iter() { self.image_views_resources.push(r.clone()); }
        for r in cb.image_resources.iter() { self.image_resources.push(r.clone()); }

        {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            vk.CmdExecuteCommands(self.cmd.unwrap(), 1, &cb.cmd);
        }

        // Resetting the state of the command buffer.
        // The specs actually don't say anything about this, but one of the speakers at the
        // GDC 2016 conference said this was the case. Since keeping the state is purely an
        // optimization, we disable it just in case. This might be removed when things get
        // clarified.
        self.graphics_pipeline = None;
        self.compute_pipeline = None;
        self.dynamic_state = DynamicState::none();

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
    ///
    /// # Safety
    ///
    /// - Care must be taken to respect the rules about secondary command buffers.
    ///
    pub unsafe fn update_buffer<'a, B, T, Bt>(mut self, buffer: B, data: &T)
                                              -> InnerCommandBufferBuilder
        where B: Into<BufferSlice<'a, T, Bt>>, Bt: Buffer + 'static
    {
        let buffer = buffer.into();

        assert_eq!(buffer.size(), mem::size_of_val(data));
        assert!(buffer.size() <= 65536);
        assert!(buffer.offset() % 4 == 0);
        assert!(buffer.size() % 4 == 0);
        assert!(buffer.buffer().inner_buffer().usage_transfer_dest());

        // FIXME: check queue family of the buffer

        self.add_buffer_resource(buffer.buffer().clone(), true, buffer.offset(), buffer.size());

        {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            vk.CmdUpdateBuffer(self.cmd.unwrap(), buffer.buffer().inner_buffer().internal_object(),
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
    pub unsafe fn fill_buffer<B>(mut self, buffer: &Arc<B>, offset: usize,
                                 size: usize, data: u32) -> InnerCommandBufferBuilder
        where B: Buffer + 'static
    {
        assert!(self.pool.queue_family().supports_transfers());
        assert!(offset + size <= buffer.size());
        assert!(offset % 4 == 0);
        assert!(size % 4 == 0);
        assert!(buffer.inner_buffer().usage_transfer_dest());

        self.add_buffer_resource(buffer.clone(), true, offset, size);

        // FIXME: check that the queue family supports transfers
        // FIXME: check queue family of the buffer

        {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            vk.CmdFillBuffer(self.cmd.unwrap(), buffer.inner_buffer().internal_object(),
                             offset as vk::DeviceSize, size as vk::DeviceSize, data);
        }

        self
    }

    /// Copies data between buffers.
    ///
    /// There is no restriction for the type of queue that can perform this.
    ///
    /// # Panic
    ///
    /// - Panicks if the buffers don't belong to the same device.
    /// - Panicks if one of the buffers wasn't created with the right usage.
    ///
    /// # Safety
    ///
    /// - Type safety is not enforced by the API.
    /// - Care must be taken to respect the rules about secondary command buffers.
    ///
    // TODO: doesn't support slices
    pub unsafe fn copy_buffer<T: ?Sized + 'static, Bs, Bd>(mut self, source: &Arc<Bs>,
                                                           destination: &Arc<Bd>)
                                                           -> InnerCommandBufferBuilder
        where Bs: TypedBuffer<Content = T> + 'static, Bd: TypedBuffer<Content = T> + 'static
    {
        assert_eq!(&**source.inner_buffer().device() as *const _,
                   &**destination.inner_buffer().device() as *const _);
        assert!(source.inner_buffer().usage_transfer_src());
        assert!(destination.inner_buffer().usage_transfer_dest());

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
            vk.CmdCopyBuffer(self.cmd.unwrap(), source.inner_buffer().internal_object(),
                             destination.inner_buffer().internal_object(), 1, &copy);
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
        where Ty: ImageTypeMarker, F: PossibleFloatFormatDesc, M: MemorySource   // FIXME: should accept uint and int images too
    {
        assert!(image.format().is_float()); // FIXME: should accept uint and int images too

        let color = match image.format().decode_clear_value(color) {
            ClearValue::Float(data) => vk::ClearColorValue::float32(data),
            ClearValue::Int(data) => vk::ClearColorValue::int32(data),
            ClearValue::Uint(data) => vk::ClearColorValue::uint32(data),
            _ => unreachable!()   // PossibleFloatFormatDesc has been improperly implemented
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
    pub unsafe fn copy_buffer_to_color_image<'a, S, Sb, Ty, F, Im>(mut self, source: S, image: &Arc<Image<Ty, F, Im>>)
                                                                   -> InnerCommandBufferBuilder
        where S: Into<BufferSlice<'a, [F::Pixel], Sb>>, F: StrongStorage + 'static + PossibleFloatOrCompressedFormatDesc,     // FIXME: wrong trait
              Ty: ImageTypeMarker + 'static, Im: MemorySource + 'static, Sb: Buffer + 'static
    {
        assert!(image.format().is_float_or_compressed());

        let source = source.into();
        self.add_buffer_resource(source.buffer().clone(), false, source.offset(), source.size());
        self.add_image_resource(image.clone() as Arc<_>, true);

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
            vk.CmdCopyBufferToImage(self.cmd.unwrap(), source.buffer().inner_buffer().internal_object(), image.internal_object(),
                                    vk::IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL /* FIXME */,
                                    1, &region);
        }

        self
    }

    pub unsafe fn dispatch<Pl, L>(mut self, pipeline: &Arc<ComputePipeline<Pl>>, sets: L,
                                  x: u32, y: u32, z: u32) -> InnerCommandBufferBuilder
        where L: 'static + DescriptorSetsCollection,
              Pl: 'static + PipelineLayoutDesc
    {
        self.bind_compute_pipeline_state(pipeline, sets);

        {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            vk.CmdDispatch(self.cmd.unwrap(), x, y, z);
        }

        self
    }

    /// Calls `vkCmdDraw`.
    // FIXME: push constants
    pub unsafe fn draw<V, Pv, Pl, L, Rp>(mut self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Rp>>,
                             vertices: V, dynamic: &DynamicState,
                             sets: L) -> InnerCommandBufferBuilder
        where Pv: 'static + VertexDefinition + VertexSource<V>, L: 'static + DescriptorSetsCollection,
              Pl: 'static + PipelineLayoutDesc, Rp: 'static
    {
        // FIXME: add buffers to the resources

        self.bind_gfx_pipeline_state(pipeline, dynamic, sets);

        let vertices = pipeline.vertex_definition().decode(vertices);

        let offsets = (0 .. vertices.0.len()).map(|_| 0).collect::<SmallVec<[_; 8]>>();
        let ids = vertices.0.map(|b| {
            assert!(b.inner_buffer().usage_vertex_buffer());
            self.add_buffer_resource(b.clone(), false, 0, b.size());
            b.inner_buffer().internal_object()
        }).collect::<SmallVec<[_; 8]>>();

        {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            vk.CmdBindVertexBuffers(self.cmd.unwrap(), 0, ids.len() as u32, ids.as_ptr(),
                                    offsets.as_ptr());
            vk.CmdDraw(self.cmd.unwrap(), vertices.1 as u32, vertices.2 as u32, 0, 0);  // FIXME: params
        }

        self
    }

    /// Calls `vkCmdDrawIndexed`.
    // FIXME: push constants
    pub unsafe fn draw_indexed<'a, V, Pv, Pl, Rp, L, I, Ib, Ibb>(mut self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Rp>>,
                                                          vertices: V, indices: Ib, dynamic: &DynamicState,
                                                          sets: L) -> InnerCommandBufferBuilder
        where L: 'static + DescriptorSetsCollection,
              Pv: 'static + VertexDefinition + VertexSource<V>,
              Pl: 'static + PipelineLayoutDesc, Rp: 'static,
              Ib: Into<BufferSlice<'a, [I], Ibb>>, I: 'static + Index, Ibb: Buffer + 'static
    {

        // FIXME: add buffers to the resources

        self.bind_gfx_pipeline_state(pipeline, dynamic, sets);


        let indices = indices.into();

        let vertices = pipeline.vertex_definition().decode(vertices);

        let offsets = (0 .. vertices.0.len()).map(|_| 0).collect::<SmallVec<[_; 8]>>();
        let ids = vertices.0.map(|b| {
            assert!(b.inner_buffer().usage_vertex_buffer());
            self.add_buffer_resource(b.clone(), false, 0, b.size());
            b.inner_buffer().internal_object()
        }).collect::<SmallVec<[_; 8]>>();

        assert!(indices.buffer().inner_buffer().usage_index_buffer());

        self.add_buffer_resource(indices.buffer().clone(), false, indices.offset(), indices.size());

        {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            vk.CmdBindIndexBuffer(self.cmd.unwrap(), indices.buffer().inner_buffer().internal_object(),
                                  indices.offset() as u64, I::ty() as u32);
            vk.CmdBindVertexBuffers(self.cmd.unwrap(), 0, ids.len() as u32, ids.as_ptr(),
                                    offsets.as_ptr());
            vk.CmdDrawIndexed(self.cmd.unwrap(), indices.len() as u32, vertices.2 as u32,
                              0, 0, 0);  // FIXME: params
        }

        self
    }

    fn bind_compute_pipeline_state<Pl, L>(&mut self, pipeline: &Arc<ComputePipeline<Pl>>, sets: L)
        where L: 'static + DescriptorSetsCollection,
              Pl: 'static + PipelineLayoutDesc
    {
        unsafe {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            assert!(sets.is_compatible_with(pipeline.layout()));

            if self.compute_pipeline != Some(pipeline.internal_object()) {
                vk.CmdBindPipeline(self.cmd.unwrap(), vk::PIPELINE_BIND_POINT_COMPUTE,
                                   pipeline.internal_object());
                self.pipelines.push(pipeline.clone());
                self.compute_pipeline = Some(pipeline.internal_object());
            }

            let mut descriptor_sets = sets.list().collect::<SmallVec<[_; 32]>>();
            for d in descriptor_sets.iter() { self.descriptor_sets.push(d.clone()); }
            let descriptor_sets = descriptor_sets.into_iter().map(|set| set.internal_object()).collect::<SmallVec<[_; 32]>>();

            // TODO: shouldn't rebind everything every time
            if !descriptor_sets.is_empty() {
                vk.CmdBindDescriptorSets(self.cmd.unwrap(), vk::PIPELINE_BIND_POINT_COMPUTE,
                                         pipeline.layout().internal_object(), 0,
                                         descriptor_sets.len() as u32, descriptor_sets.as_ptr(),
                                         0, ptr::null());   // FIXME: dynamic offsets
            }
        }
    }

    fn bind_gfx_pipeline_state<V, Pl, L, Rp>(&mut self, pipeline: &Arc<GraphicsPipeline<V, Pl, Rp>>,
                                             dynamic: &DynamicState, sets: L)
        where V: 'static + VertexDefinition, L: 'static + DescriptorSetsCollection,
              Pl: 'static + PipelineLayoutDesc, Rp: 'static
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

            if let Some(ref viewports) = dynamic.viewports {
                assert!(pipeline.has_dynamic_viewports());
                assert_eq!(viewports.len(), pipeline.num_viewports() as usize);
                // TODO: check limits
                // TODO: cache state?
                let viewports = viewports.iter().map(|v| v.clone().into()).collect::<SmallVec<[_; 16]>>();
                vk.CmdSetViewport(self.cmd.unwrap(), 0, viewports.len() as u32, viewports.as_ptr());
            } else {
                assert!(!pipeline.has_dynamic_viewports());
            }

            if let Some(ref scissors) = dynamic.scissors {
                assert!(pipeline.has_dynamic_scissors());
                assert_eq!(scissors.len(), pipeline.num_viewports() as usize);
                // TODO: check limits
                // TODO: cache state?
                // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
                let scissors = scissors.iter().map(|v| v.clone().into()).collect::<SmallVec<[_; 16]>>();
                vk.CmdSetScissor(self.cmd.unwrap(), 0, scissors.len() as u32, scissors.as_ptr());
            } else {
                assert!(!pipeline.has_dynamic_scissors());
            }

            let mut descriptor_sets = sets.list().collect::<SmallVec<[_; 32]>>();
            for d in descriptor_sets.iter() { self.descriptor_sets.push(d.clone()); }
            let descriptor_sets = descriptor_sets.into_iter().map(|set| set.internal_object()).collect::<SmallVec<[_; 32]>>();

            // FIXME: input attachments of descriptor sets have to be checked against input
            //        attachments of the render pass

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
    pub unsafe fn begin_renderpass<R, F>(mut self, render_pass: &Arc<R>,
                                         framebuffer: &Arc<Framebuffer<F>>,
                                         secondary_cmd_buffers: bool,
                                         clear_values: &[ClearValue]) -> InnerCommandBufferBuilder
        where R: RenderPass + 'static, F: RenderPass + 'static
    {
        assert!(framebuffer.is_compatible_with(render_pass));

        self.framebuffers.push(framebuffer.clone() as Arc<_>);
        self.renderpasses.push(render_pass.clone() as Arc<_>);

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
        }).collect::<SmallVec<[_; 16]>>();

        // FIXME: change attachment image layouts if necessary, for both initial and final
        /*for attachment in R::attachments() {

        }*/

        for attachment in framebuffer.attachments() {
            self.image_views_resources.push(attachment.clone());
        }

        let infos = vk::RenderPassBeginInfo {
            sType: vk::STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            pNext: ptr::null(),
            renderPass: render_pass.render_pass().internal_object(),
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

            // Computing the list of buffer resources by removing duplicates.
            let buffer_resources = self.buffer_resources.iter().enumerate().filter_map(|(num, elem)| {
                if self.buffer_resources.iter().take(num)
                                       .find(|e| &***e as *const Buffer == &**elem as *const Buffer).is_some()
                {
                    None
                } else {
                    Some(elem.clone())
                }
            }).collect::<Vec<_>>();

            // Computing the list of image resources by removing duplicates.
            // TODO: image views as well
            let image_resources = self.image_resources.iter().enumerate().filter_map(|(num, elem)| {
                if self.image_resources.iter().take(num)
                                       .find(|e| &***e as *const AbstractImage == &**elem as *const AbstractImage).is_some()
                {
                    None
                } else {
                    Some(elem.clone())
                }
            }).collect::<Vec<_>>();

            Ok(InnerCommandBuffer {
                device: self.device.clone(),
                pool: self.pool.clone(),
                cmd: cmd,
                secondary_command_buffers: mem::replace(&mut self.secondary_command_buffers, Vec::new()),
                descriptor_sets: mem::replace(&mut self.descriptor_sets, Vec::new()),
                framebuffers: mem::replace(&mut self.framebuffers, Vec::new()),
                renderpasses: mem::replace(&mut self.renderpasses, Vec::new()),
                buffer_resources: buffer_resources,
                image_resources: image_resources,
                image_views_resources: mem::replace(&mut self.image_views_resources, Vec::new()),
                pipelines: mem::replace(&mut self.pipelines, Vec::new()),
            })
        }
    }

    /// Adds a buffer resource to the list of resources used by this command buffer.
    // FIXME: add access flags
    fn add_buffer_resource(&mut self, buffer: Arc<Buffer>, write: bool, offset: usize,
                           size: usize)
    {
        // TODO: handle memory barriers
        self.buffer_resources.push(buffer);
    }

    /// Adds an image resource to the list of resources used by this command buffer.
    fn add_image_resource(&mut self, image: Arc<AbstractImage>, _write: bool) {   // TODO:
        self.image_resources.push(image);
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
    secondary_command_buffers: Vec<Arc<AbstractCommandBuffer>>,
    descriptor_sets: Vec<Arc<AbstractDescriptorSet>>,
    framebuffers: Vec<Arc<AbstractFramebuffer>>,
    renderpasses: Vec<Arc<RenderPass>>,
    buffer_resources: Vec<Arc<Buffer>>,
    image_resources: Vec<Arc<AbstractImage>>,
    image_views_resources: Vec<Arc<AbstractImageView>>,
    pipelines: Vec<Arc<GenericPipeline>>,
}

/// Submits the command buffer to a queue.
///
/// Queues are not thread-safe, therefore we need to get a `&mut`.
///
/// # Panic
///
/// - Panicks if the queue doesn't belong to the device this command buffer was created with.
/// - Panicks if the queue doesn't belong to the family the pool was created with.
///
pub fn submit(me: &InnerCommandBuffer, me_arc: Arc<AbstractCommandBuffer>,
              queue: &Arc<Queue>) -> Result<Arc<Submission>, OomError>   // TODO: wrong error type
{
    // FIXME: the whole function should be checked
    let vk = me.device.pointers();

    assert_eq!(queue.device().internal_object(), me.pool.device().internal_object());
    assert_eq!(queue.family().id(), me.pool.queue_family().id());

    let fence = try!(Fence::new(queue.device()));

    let mut post_semaphores = Vec::new();
    let mut post_semaphores_ids = Vec::new();
    let mut pre_semaphores = Vec::new();
    let mut pre_semaphores_ids = Vec::new();
    let mut pre_semaphores_stages = Vec::new();

    // Each queue has a dedicated semaphore which must be signalled and waited upon by each
    // command buffer submission.
    {
        let (semaphore, wait) = unsafe { try!(queue.dedicated_semaphore()) };
        if wait {
            pre_semaphores_ids.push(semaphore.internal_object());
            pre_semaphores_stages.push(vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT);     // TODO:
            pre_semaphores.push(semaphore.clone());
        }
        post_semaphores_ids.push(semaphore.internal_object());
        post_semaphores.push(semaphore);
    }

    // Creating additional semaphores, one for each queue transition.
    let queue_transitions_hint: u32 = 2;        // TODO: get as function parameter
    // TODO: use a pool
    let semaphores_to_signal = {
        let mut list = SmallVec::new();
        for _ in 0 .. queue_transitions_hint {
            let sem = try!(Semaphore::new(queue.device()));
            post_semaphores_ids.push(sem.internal_object());
            post_semaphores.push(sem.clone());
            list.push(sem);
        }
        list
    };

    // We can now create the `Submission` object.
    let submission = Arc::new(Submission {
        cmd: Some(me_arc),
        fence: fence.clone(),
        queue: queue.clone(),
        guarded: Mutex::new(SubmissionGuarded {
            signalled_semaphores: semaphores_to_signal,
            signalled_queues: SmallVec::new(),
        }),
        keep_alive_semaphores: Mutex::new(SmallVec::new()),
    });

    // Now we determine which earlier submissions we must depend upon.
    let mut dependencies = SmallVec::<[Arc<Submission>; 6]>::new();

    // Buffers first.
    for resource in me.buffer_resources.iter() {
        let deps = unsafe {
            // FIXME: for the moment `write` is always true ; that shouldn't be the case
            // FIXME: wrong offset and size
            resource.gpu_access(true, 0 .. 18, &submission)
        };

        dependencies.extend(deps.into_iter());
    }

    // Then images.
    for resource in me.image_resources.iter() {
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
            resource.gpu_access(true, queue, Some(fence.clone()), post_semaphore)
        };

        if let Some(s) = sem {
            pre_semaphores_ids.push(s.internal_object());
            pre_semaphores_stages.push(vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT);     // TODO:
            pre_semaphores.push(s);
        }
    }

    // For each dependency, we either wait on one of its semaphores, or create a new one.
    for dependency in dependencies.iter() {
        let mut guard = dependency.guarded.lock().unwrap();
        let current_queue_id = (queue.family().id(), queue.id_within_family());

        // If the current queue is in the list of already-signalled queue, we ignore the
        // dependency.
        if guard.signalled_queues.iter().find(|&&elem| elem == current_queue_id).is_some() {
            continue;
        }

        // Otherwise, try to extract a semaphore from the semaphores that were signalled by the
        // dependency.
        let semaphore = guard.signalled_semaphores.pop();
        guard.signalled_queues.push(current_queue_id);

        let semaphore = if let Some(semaphore) = semaphore {
            semaphore

        } else {
            // This path is the slow path in the case where the user gave the wrong hint about
            // the number of queue transitions.
            // The only thing left to do is submit a dummy command buffer and a dummy semaphore
            // in the source queue.
            unimplemented!()    // FIXME:
        };

        pre_semaphores_ids.push(semaphore.internal_object());
        pre_semaphores_stages.push(vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT);     // TODO:
        pre_semaphores.push(semaphore);
    }

    // Don't forget to merge the semaphores into the submission.
    {
        let mut keep_alive_semaphores = submission.keep_alive_semaphores.lock().unwrap();
        *keep_alive_semaphores = post_semaphores.into_iter()
                                                .chain(pre_semaphores.into_iter()).collect();
    }

    debug_assert_eq!(pre_semaphores_ids.len(), pre_semaphores_stages.len());

    let infos = vk::SubmitInfo {
        sType: vk::STRUCTURE_TYPE_SUBMIT_INFO,
        pNext: ptr::null(),
        waitSemaphoreCount: pre_semaphores_ids.len() as u32,
        pWaitSemaphores: pre_semaphores_ids.as_ptr(),
        pWaitDstStageMask: pre_semaphores_stages.as_ptr(),
        commandBufferCount: 1,
        pCommandBuffers: &me.cmd,
        signalSemaphoreCount: post_semaphores_ids.len() as u32,
        pSignalSemaphores: post_semaphores_ids.as_ptr(),
    };

    unsafe {
        let fence = fence.internal_object();
        try!(check_errors(vk.QueueSubmit(*queue.internal_object_guard(), 1, &infos, fence)));
    }

    Ok(submission)
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

#[must_use]
pub struct Submission {
    cmd: Option<Arc<AbstractCommandBuffer>>,

    fence: Arc<Fence>,

    // The queue on which this was submitted.
    queue: Arc<Queue>,

    // Additional variables that are behind a mutex.
    guarded: Mutex<SubmissionGuarded>,

    // List of semaphores to keep alive while the submission hasn't finished execution.
    keep_alive_semaphores: Mutex<SmallVec<[Arc<Semaphore>; 8]>>,
}

struct SubmissionGuarded {
    // Reserve of semaphores that have been signalled by this submission and that must be
    // waited upon.
    signalled_semaphores: SmallVec<[Arc<Semaphore>; 4]>,

    // Queue familiy index and queue index of each queue that got submitted a command buffer
    // that was waiting on this submission to be complete.
    signalled_queues: SmallVec<[(u32, u32); 4]>,
}

impl Submission {
    /// Returns `true` is destroying this `Submission` object would block the CPU for some time.
    #[inline]
    pub fn destroying_would_block(&self) -> bool {
        self.fence.ready().unwrap_or(false)     // TODO: what to do in case of error?
    }

    /// Returns the `queue` the command buffers were submitted to.
    #[inline]
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }
}

impl Drop for Submission {
    #[inline]
    fn drop(&mut self) {
        self.fence.wait(5 * 1000 * 1000 * 1000 /* 5 seconds */).unwrap();

        // TODO: return `signalled_semaphores` to the semaphore pools
    }
}
