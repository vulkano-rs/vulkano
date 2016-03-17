use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash;
use std::mem;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;
use smallvec::SmallVec;

use buffer::Buffer;
use buffer::BufferSlice;
use buffer::GpuAccessRange as BufferGpuAccessRange;
use buffer::AbstractBuffer;
use buffer::BufferMemorySource;
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
use image::GpuAccessRange as ImageGpuAccessRange;
use image::Image;
use image::ImageTypeMarker;
use image::ImageMemorySource;
use image::Layout as ImageLayout;
use pipeline::GenericPipeline;
use pipeline::ComputePipeline;
use pipeline::GraphicsPipeline;
use pipeline::input_assembly::Index;
use pipeline::vertex::Definition as VertexDefinition;
use pipeline::vertex::Source as VertexSource;
use sync::Fence;
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
/// Mostly safe, except that it doesn't check whether the command types are appropriate for the
/// command buffer type.
pub struct InnerCommandBufferBuilder {
    device: Arc<Device>,
    pool: Arc<CommandBufferPool>,

    // Should always be `Some`, except after we call `build`. If this value is still `Some`
    // in the builder's destructor, we assume that the command buffer is to be destroyed.
    cmd: Option<vk::CommandBuffer>,

    // For each buffer and image used by this command buffer, stores the way the buffer or
    // image's `gpu_access` method is going to be called when the CB is submitted.
    externsync_buffer_resources: HashMap<AbstractBufferKey, SmallVec<[BufferGpuAccessRange; 2]>>,
    externsync_image_resources: HashMap<AbstractImageKey, SmallVec<[ImageGpuAccessRange; 2]>>,

    // When we are inside a render pass (between `CmdBeginRenderPass` and `CmdEndRenderPass`),
    // all commands go within `renderpass_staged` instead of being appended directly to the
    // command buffer and all used resources go inside the `renderpass_*_resources` variables.
    //
    // When `CmdEndRenderPass` is called, all the commands here are flushed.
    //
    // This allows us to know which image layout transitions and which pipeline barriers are needed
    // before the real command buffer enters the render pass, so that we can call
    // `CmdPipelineBarrier` before `CmdBeginRenderPass`.
    renderpass_staged: Vec<Box<FnMut(&vk::DevicePointers, vk::CommandBuffer)>>,
    renderpass_buffer_resources: Vec<(Arc<AbstractBuffer>, BufferInnerSync)>,
    renderpass_image_resources: Vec<(Arc<AbstractImage>, ImageInnerSync)>,

    // These variables exist to keep the objects that are used by this command buffer alive.
    keep_alive_secondary_command_buffers: Vec<Arc<AbstractCommandBuffer>>,
    keep_alive_descriptor_sets: Vec<Arc<AbstractDescriptorSet>>,
    keep_alive_framebuffers: Vec<Arc<AbstractFramebuffer>>,
    keep_alive_renderpasses: Vec<Arc<RenderPass>>,
    keep_alive_image_views_resources: Vec<Arc<AbstractImageView>>,
    keep_alive_pipelines: Vec<Arc<GenericPipeline>>,

    // Current pipeline object binded to the graphics bind point.
    current_graphics_pipeline: Option<vk::Pipeline>,

    // Current pipeline object binded to the compute bind point.
    current_compute_pipeline: Option<vk::Pipeline>,

    // Current state of the dynamic state within the command buffer.
    current_dynamic_state: DynamicState,
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

        let mut keepalive_renderpasses = Vec::new();
        let mut keepalive_framebuffers = Vec::new();

        unsafe {
            // TODO: one time submit
            let flags = vk::COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT |     // TODO:
                        if secondary_cont.is_some() { vk::COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT } else { 0 };

            let (rp, sp) = if let Some(ref sp) = secondary_cont {
                keepalive_renderpasses.push(sp.render_pass().clone() as Arc<_>);
                (sp.render_pass().render_pass().internal_object(), sp.index())
            } else {
                (0, 0)
            };

            let framebuffer = if let Some(fb) = secondary_cont_fb {
                keepalive_framebuffers.push(fb.clone() as Arc<_>);
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
            externsync_buffer_resources: HashMap::new(),
            externsync_image_resources: HashMap::new(),
            renderpass_staged: Vec::new(),
            renderpass_buffer_resources: Vec::new(),
            renderpass_image_resources: Vec::new(),
            keep_alive_secondary_command_buffers: Vec::new(),
            keep_alive_descriptor_sets: Vec::new(),
            keep_alive_framebuffers: keepalive_framebuffers,
            keep_alive_renderpasses: keepalive_renderpasses,
            keep_alive_image_views_resources: Vec::new(),
            keep_alive_pipelines: Vec::new(),
            current_graphics_pipeline: None,
            current_compute_pipeline: None,
            current_dynamic_state: DynamicState::none(),
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
        self.keep_alive_secondary_command_buffers.push(cb_arc);

        for (k, v) in cb.externsync_buffer_resources.iter() { self.externsync_buffer_resources.insert(k.clone(), v.clone()); }        // FIXME: merge properly
        for (k, v) in cb.externsync_image_resources.iter() { self.externsync_image_resources.insert(k.clone(), v.clone()); }        // FIXME: merge properly

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
        self.current_graphics_pipeline = None;
        self.current_compute_pipeline = None;
        self.current_dynamic_state = DynamicState::none();

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
    pub unsafe fn update_buffer<'a, B, T, Bo: ?Sized + 'static, Bm: 'static>(mut self, buffer: B, data: &T)
                                                          -> InnerCommandBufferBuilder
        where B: Into<BufferSlice<'a, T, Bo, Bm>>, Bm: BufferMemorySource
    {
        let buffer = buffer.into();

        assert_eq!(buffer.size(), mem::size_of_val(data));
        assert!(buffer.size() <= 65536);
        assert!(buffer.offset() % 4 == 0);
        assert!(buffer.size() % 4 == 0);
        assert!(buffer.buffer().usage_transfer_dest());

        self.register_resources(Some((buffer.buffer().clone() as Arc<_>, BufferInnerSync {
            range: buffer.offset() .. buffer.offset() + buffer.size(),
            write: true,
        })), None);

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
        where M: BufferMemorySource + 'static
    {

        assert!(self.pool.queue_family().supports_transfers());
        assert!(offset + size <= buffer.size());
        assert!(offset % 4 == 0);
        assert!(size % 4 == 0);
        assert!(buffer.usage_transfer_dest());

        self.register_resources(Some((buffer.clone() as Arc<_>, BufferInnerSync {
            range: offset .. offset + size,
            write: true,
        })), None);

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
    pub unsafe fn copy_buffer<T: ?Sized + 'static, Ms, Md>(mut self, source: &Arc<Buffer<T, Ms>>,
                                                           destination: &Arc<Buffer<T, Md>>)
                                                           -> InnerCommandBufferBuilder
        where Ms: BufferMemorySource + 'static, Md: BufferMemorySource + 'static
    {
        assert_eq!(&**source.device() as *const _, &**destination.device() as *const _);
        assert!(source.usage_transfer_src());
        assert!(destination.usage_transfer_dest());

        self.register_resources(
            vec![
                (source.clone() as Arc<_>, BufferInnerSync {
                    range: 0 .. source.size(),
                    write: false,
                }),
                (destination.clone() as Arc<_>, BufferInnerSync {
                    range: 0 .. destination.size(),
                    write: true,
                })
            ],
            None
        );

        let copy = vk::BufferCopy {
            srcOffset: 0,
            dstOffset: 0,
            size: source.size() as u64,     // FIXME: what is destination is too small?
        };

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
        where Ty: ImageTypeMarker, F: PossibleFloatFormatDesc, M: ImageMemorySource   // FIXME: should accept uint and int images too
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
            vk.CmdClearColorImage(self.cmd.unwrap(), image.internal_object(),
                                  vk::IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL /* FIXME: */,
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
    pub unsafe fn copy_buffer_to_color_image<'a, S, So: ?Sized, Sm, Ty, F, Im>(mut self, source: S, image: &Arc<Image<Ty, F, Im>>)
                                                                   -> InnerCommandBufferBuilder
        where S: Into<BufferSlice<'a, [F::Pixel], So, Sm>>, F: StrongStorage + 'static + PossibleFloatOrCompressedFormatDesc,     // FIXME: wrong trait
              Ty: ImageTypeMarker + 'static, Sm: BufferMemorySource + 'static, So: 'static,
              Im: ImageMemorySource + 'static
    {
        assert!(image.format().is_float_or_compressed());

        let source = source.into();

        self.register_resources(
            Some((source.buffer().clone() as Arc<_>, BufferInnerSync {
                range: source.offset() .. source.offset() + source.size(),
                write: false,
            })).into_iter(),
            Some((image.clone() as Arc<_>, ImageInnerSync {
                mipmap_levels_range: 0 .. 1,     // FIXME:
                array_layers_range: 0 .. 1,      // FIXME:
                write: true,
                layout: ImageLayout::TransferDstOptimal,       // TODO: can be General as well
            })).into_iter()
        );

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
            assert!(b.usage_vertex_buffer());
            self.renderpass_buffer_resources.push((b.clone(), BufferInnerSync {
                range: 0 .. b.size(),       // FIXME:
                write: false,
            }));
            b.internal_object()
        }).collect::<SmallVec<[_; 8]>>();

        let num_vertices = vertices.1 as u32;
        let num_instances = vertices.2 as u32;

        self.renderpass_staged.push(Box::new(move |vk, cmd| {
            vk.CmdBindVertexBuffers(cmd, 0, ids.len() as u32, ids.as_ptr(),
                                    offsets.as_ptr());
            vk.CmdDraw(cmd, num_vertices, num_instances, 0, 0);  // FIXME: params
        }));

        self
    }

    /// Calls `vkCmdDrawIndexed`.
    // FIXME: push constants
    pub unsafe fn draw_indexed<'a, V, Pv, Pl, Rp, L, I, Ib, Ibo: ?Sized + 'static, Ibm: 'static>(mut self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Rp>>,
                                                          vertices: V, indices: Ib, dynamic: &DynamicState,
                                                          sets: L) -> InnerCommandBufferBuilder
        where L: 'static + DescriptorSetsCollection,
              Pv: 'static + VertexDefinition + VertexSource<V>,
              Pl: 'static + PipelineLayoutDesc, Rp: 'static,
              Ib: Into<BufferSlice<'a, [I], Ibo, Ibm>>, I: 'static + Index,
              Ibm: BufferMemorySource
    {

        // FIXME: add buffers to the resources

        self.bind_gfx_pipeline_state(pipeline, dynamic, sets);


        let indices = indices.into();

        let vertices = pipeline.vertex_definition().decode(vertices);

        let offsets = (0 .. vertices.0.len()).map(|_| 0).collect::<SmallVec<[_; 8]>>();
        let ids = vertices.0.map(|b| {
            assert!(b.usage_vertex_buffer());
            self.renderpass_buffer_resources.push((b.clone(), BufferInnerSync {
                range: 0 .. b.size(),       // FIXME:
                write: false,
            }));
            b.internal_object()
        }).collect::<SmallVec<[_; 8]>>();

        assert!(indices.buffer().usage_index_buffer());

        self.renderpass_buffer_resources.push((indices.buffer().clone(), BufferInnerSync {
            range: indices.offset() .. indices.offset() + indices.size(),
            write: false,
        }));

        let indices_buffer_internal = indices.buffer().internal_object();
        let indices_offset = indices.offset();
        let indices_len = indices.len();
        let vertices_num = vertices.2 as u32;

        self.renderpass_staged.push(Box::new(move |vk, cmd| {
            vk.CmdBindIndexBuffer(cmd, indices_buffer_internal, indices_offset as u64,
                                  I::ty() as u32);
            vk.CmdBindVertexBuffers(cmd, 0, ids.len() as u32, ids.as_ptr(), offsets.as_ptr());
            vk.CmdDrawIndexed(cmd, indices_len as u32, vertices_num as u32,
                              0, 0, 0);  // FIXME: params
        }));

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

            if self.current_compute_pipeline != Some(pipeline.internal_object()) {
                let pipeline_internal = pipeline.internal_object();
                self.renderpass_staged.push(Box::new(move |vk, cmd| {
                    vk.CmdBindPipeline(cmd, vk::PIPELINE_BIND_POINT_COMPUTE, pipeline_internal);
                }));
                self.keep_alive_pipelines.push(pipeline.clone());
                self.current_compute_pipeline = Some(pipeline.internal_object());
            }

            let mut descriptor_sets = sets.list().collect::<SmallVec<[_; 32]>>();
            for d in descriptor_sets.iter() { self.keep_alive_descriptor_sets.push(d.clone()); }
            let descriptor_sets = descriptor_sets.into_iter().map(|set| set.internal_object()).collect::<SmallVec<[_; 32]>>();

            // TODO: shouldn't rebind everything every time
            if !descriptor_sets.is_empty() {
                let pipeline_layout_internal = pipeline.layout().internal_object();
                self.renderpass_staged.push(Box::new(move |vk, cmd| {
                    vk.CmdBindDescriptorSets(cmd, vk::PIPELINE_BIND_POINT_COMPUTE,
                                             pipeline_layout_internal, 0,
                                             descriptor_sets.len() as u32, descriptor_sets.as_ptr(),
                                             0, ptr::null());   // FIXME: dynamic offsets
                }));
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

            if self.current_graphics_pipeline != Some(pipeline.internal_object()) {
                let pipeline_internal = pipeline.internal_object();
                self.renderpass_staged.push(Box::new(move |vk, cmd| {
                    vk.CmdBindPipeline(cmd, vk::PIPELINE_BIND_POINT_GRAPHICS, pipeline_internal);
                }));
                self.keep_alive_pipelines.push(pipeline.clone());
                self.current_graphics_pipeline = Some(pipeline.internal_object());
            }

            if let Some(line_width) = dynamic.line_width {
                assert!(pipeline.has_dynamic_line_width());
                // TODO: check limits
                if self.current_dynamic_state.line_width != Some(line_width) {
                    self.renderpass_staged.push(Box::new(move |vk, cmd| {
                        vk.CmdSetLineWidth(cmd, line_width);
                    }));
                    self.current_dynamic_state.line_width = Some(line_width);
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
                self.renderpass_staged.push(Box::new(move |vk, cmd| {
                    vk.CmdSetViewport(cmd, 0, viewports.len() as u32, viewports.as_ptr());
                }));
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
                self.renderpass_staged.push(Box::new(move |vk, cmd| {
                    vk.CmdSetScissor(cmd, 0, scissors.len() as u32, scissors.as_ptr());
                }));
            } else {
                assert!(!pipeline.has_dynamic_scissors());
            }

            let mut descriptor_sets = sets.list().collect::<SmallVec<[_; 32]>>();
            for d in descriptor_sets.iter() { self.keep_alive_descriptor_sets.push(d.clone()); }
            let descriptor_sets = descriptor_sets.into_iter().map(|set| set.internal_object()).collect::<SmallVec<[_; 32]>>();

            // FIXME: input attachments of descriptor sets have to be checked against input
            //        attachments of the render pass

            // TODO: shouldn't rebind everything every time
            if !descriptor_sets.is_empty() {
                let pipeline_layout_internal = pipeline.layout().internal_object();

                self.renderpass_staged.push(Box::new(move |vk, cmd| {
                    vk.CmdBindDescriptorSets(cmd, vk::PIPELINE_BIND_POINT_GRAPHICS,
                                             pipeline_layout_internal, 0,
                                             descriptor_sets.len() as u32, descriptor_sets.as_ptr(),
                                             0, ptr::null());   // FIXME: dynamic offsets
                }));
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

        self.keep_alive_framebuffers.push(framebuffer.clone() as Arc<_>);
        self.keep_alive_renderpasses.push(render_pass.clone() as Arc<_>);

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

        for attachment in framebuffer.attachments() {
            self.keep_alive_image_views_resources.push(attachment.clone());
            self.renderpass_image_resources.push((attachment.image(), ImageInnerSync {
                mipmap_levels_range: 0 .. 1,        // FIXME:
                array_layers_range: 0 .. 1,     // FIXME:
                write: true,
                layout: ImageLayout::PresentSrc,
            }));
        }

        let content = if secondary_cmd_buffers {
            vk::SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS
        } else {
            vk::SUBPASS_CONTENTS_INLINE
        };

        let renderpass_internal = render_pass.render_pass().internal_object();
        let framebuffer_internal = framebuffer.internal_object();
        let framebuffer_width = framebuffer.width();
        let framebuffer_height = framebuffer.height();

        self.renderpass_staged.push(Box::new(move |vk, cmd| {
            let infos = vk::RenderPassBeginInfo {
                sType: vk::STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                pNext: ptr::null(),
                renderPass: renderpass_internal,
                framebuffer: framebuffer_internal,
                renderArea: vk::Rect2D {                // TODO: let user customize
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: framebuffer_width,
                        height: framebuffer_height,
                    },
                },
                clearValueCount: clear_values.len() as u32,
                pClearValues: clear_values.as_ptr(),
            };

            vk.CmdBeginRenderPass(cmd, &infos, content);
        }));

        self
    }

    #[inline]
    pub unsafe fn next_subpass(mut self, secondary_cmd_buffers: bool) -> InnerCommandBufferBuilder {
        let content = if secondary_cmd_buffers {
            vk::SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS
        } else {
            vk::SUBPASS_CONTENTS_INLINE
        };

        self.renderpass_staged.push(Box::new(move |vk, cmd| {
            vk.CmdNextSubpass(cmd, content);
        }));

        self
    }

    #[inline]
    pub unsafe fn end_renderpass(mut self) -> InnerCommandBufferBuilder {
        {
            let bufs = mem::replace(&mut self.renderpass_buffer_resources, Vec::new()).into_iter();
            let img = mem::replace(&mut self.renderpass_image_resources, Vec::new()).into_iter();
            self.register_resources(bufs, img);
        }

        {
            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            let cmd = self.cmd.unwrap();

            for mut command in mem::replace(&mut self.renderpass_staged, Vec::new()).into_iter() {
                command(vk, cmd);
            }

            vk.CmdEndRenderPass(cmd);
        }

        // FIXME: for each attachment, update externsync_image_resources with the layout transitions

        self
    }

    /// Finishes building the command buffer.
    pub fn build(mut self) -> Result<InnerCommandBuffer, OomError> {
        unsafe {
            debug_assert!(self.renderpass_staged.is_empty());
            debug_assert!(self.renderpass_buffer_resources.is_empty());
            debug_assert!(self.renderpass_image_resources.is_empty());

            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            let cmd = self.cmd.take().unwrap();

            // We need to transition back queue ownership and image layouts.
            {
                let buffer_barriers: SmallVec<[vk::BufferMemoryBarrier; 8]> = SmallVec::new();     // TODO: infer VkBufferMemoryBarrier

                let image_barriers: SmallVec<[_; 8]> = self.externsync_image_resources.iter_mut().flat_map(|(image, ranges)| {
                    let image = image.clone();
                    ranges.iter_mut().filter_map(move |range| {
                        let mandatory_layout = image.0.memory().mandatory_layout(range.mipmap_level_start .. range.mipmap_level_start + range.mipmap_levels_count,
                                                                                 range.array_layer_start .. range.array_layer_start + range.array_layers_count);

                        if let Some(mandatory_layout) = mandatory_layout {
                            if mandatory_layout != range.layout_transition {
                                let barrier = vk::ImageMemoryBarrier {
                                    sType: vk::STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                    pNext: ptr::null(),
                                    srcAccessMask: 0x0001ffff,  // FIXME:
                                    dstAccessMask: 0x0001ffff,  // FIXME:
                                    oldLayout: range.layout_transition as u32,
                                    newLayout: mandatory_layout as u32,
                                    srcQueueFamilyIndex: vk::QUEUE_FAMILY_IGNORED,
                                    dstQueueFamilyIndex: vk::QUEUE_FAMILY_IGNORED,
                                    image: image.0.internal_object(),
                                    subresourceRange: vk::ImageSubresourceRange {
                                        aspectMask: 1,        // FIXME:
                                        baseMipLevel: range.mipmap_level_start,
                                        levelCount: range.mipmap_levels_count,
                                        baseArrayLayer: range.array_layer_start,
                                        layerCount: range.array_layers_count,
                                    }
                                };

                                range.layout_transition = mandatory_layout;

                                return Some(barrier);
                            }
                        }

                        None
                    })
                }).collect();

                if !buffer_barriers.is_empty() && !image_barriers.is_empty() {
                    vk.CmdPipelineBarrier(self.cmd.unwrap(), 0x00010000 /* TODO */, 0x00010000 /* TODO */,
                                          0 /* TODO */, 0, ptr::null(),
                                          buffer_barriers.len() as u32, buffer_barriers.as_ptr(),
                                          image_barriers.len() as u32, image_barriers.as_ptr());
                }
            }

            // Ending the commands recording.
            try!(check_errors(vk.EndCommandBuffer(cmd)));

            // Building the command buffer wrapper.
            Ok(InnerCommandBuffer {
                device: self.device.clone(),
                pool: self.pool.clone(),
                cmd: cmd,
                externsync_buffer_resources: mem::replace(&mut self.externsync_buffer_resources, HashMap::new()),
                externsync_image_resources: mem::replace(&mut self.externsync_image_resources, HashMap::new()),
                keep_alive_secondary_command_buffers: mem::replace(&mut self.keep_alive_secondary_command_buffers, Vec::new()),
                keep_alive_descriptor_sets: mem::replace(&mut self.keep_alive_descriptor_sets, Vec::new()),
                keep_alive_framebuffers: mem::replace(&mut self.keep_alive_framebuffers, Vec::new()),
                keep_alive_renderpasses: mem::replace(&mut self.keep_alive_renderpasses, Vec::new()),
                keep_alive_image_views_resources: mem::replace(&mut self.keep_alive_image_views_resources, Vec::new()),
                keep_alive_pipelines: mem::replace(&mut self.keep_alive_pipelines, Vec::new()),
            })
        }
    }

    /// Must be called before each command that uses resources. The parameter must indicate how
    /// each resource is going to be used by the following command.
    ///
    /// The function ensures that the resources will be in the appropriate state.
    fn register_resources<Ib, Im>(&mut self, buffers: Ib, images: Im)
        where Ib: IntoIterator<Item = (Arc<AbstractBuffer>, BufferInnerSync)>,
              Im: IntoIterator<Item = (Arc<AbstractImage>, ImageInnerSync)>,
    {
        // TODO: check that there's no overlap in the resources
        // TODO: handle memory aliasing?

        let mut buffer_barriers = SmallVec::<[vk::BufferMemoryBarrier; 16]>::new();     // TODO: infer VkBufferMemoryBarrier
        let mut image_barriers = SmallVec::<[_; 16]>::new();

        for (buffer, inner_sync) in buffers {
            let buffer_memory = buffer.memory();

            // Memory chunk implementations can just tell us to ignore any synchronization as an
            // optimization.
            if !buffer_memory.requires_synchronization() {
                continue;
            }

            // Align the range and check for correctness with debug_assert!.
            let range = {
                let range = BufferGpuAccessRange {
                    range_start: inner_sync.range.start,
                    range_size: inner_sync.range.end - inner_sync.range.start,
                    write: inner_sync.write,
                    expected_queue_family_owner: None,
                    queue_family_owner_transition: None,
                };

                let aligned = buffer_memory.align(range);

                debug_assert!(aligned.range_start <= range.range_start);
                debug_assert!(aligned.range_size >= range.range_size +
                                                    (range.range_start - aligned.range_start));
                debug_assert_eq!(aligned.expected_queue_family_owner,
                                 range.expected_queue_family_owner);
                debug_assert_eq!(aligned.queue_family_owner_transition,
                                 range.queue_family_owner_transition);
                aligned
            };

            match self.externsync_buffer_resources.entry(AbstractBufferKey(buffer.clone())) {
                Entry::Occupied(mut e) => {
                    // FIXME: merge ranges correctly
                    e.get_mut().push(range);
                },
                Entry::Vacant(e) => {
                    let mut v = SmallVec::new();
                    v.push(range);
                    e.insert(v);
                },
            };
        }

        for (image, inner_sync) in images {
            let image_memory = image.memory();

            // Memory chunk implementations can just tell us to ignore any synchronization as an
            // optimization.
            if !image_memory.requires_synchronization() {
                continue;
            }

            // Align the range and check for correctness with debug_assert!.
            let range = {
                let range = ImageGpuAccessRange {
                    mipmap_level_start: inner_sync.mipmap_levels_range.start,
                    mipmap_levels_count: inner_sync.mipmap_levels_range.end -
                                         inner_sync.mipmap_levels_range.start,
                    array_layer_start: inner_sync.array_layers_range.start,
                    array_layers_count: inner_sync.array_layers_range.end -
                                        inner_sync.array_layers_range.start,
                    write: inner_sync.write,
                    expected_queue_family_owner: None,
                    queue_family_owner_transition: None,
                    expected_layout: inner_sync.layout,
                    layout_transition: inner_sync.layout,
                };

                let aligned = image_memory.align(range);

                debug_assert!(aligned.mipmap_level_start <= range.mipmap_level_start);
                debug_assert!(aligned.mipmap_levels_count >= range.mipmap_levels_count +
                                          (range.mipmap_level_start - aligned.mipmap_level_start));
                debug_assert!(aligned.array_layer_start <= range.array_layer_start);
                debug_assert!(aligned.array_layers_count >= range.array_layers_count +
                                            (range.array_layer_start - aligned.array_layer_start));
                debug_assert_eq!(aligned.write, range.write);
                debug_assert_eq!(aligned.expected_queue_family_owner,
                                 range.expected_queue_family_owner);
                debug_assert_eq!(aligned.queue_family_owner_transition,
                                 range.queue_family_owner_transition);
                debug_assert_eq!(aligned.expected_layout, range.expected_layout);
                debug_assert_eq!(aligned.layout_transition, range.layout_transition);
                aligned
            };

            let mandatory_layout = image_memory.mandatory_layout(range.mipmap_level_start .. range.mipmap_level_start + range.mipmap_levels_count,
                                                                 range.array_layer_start .. range.array_layer_start + range.array_layers_count);

            // TODO: handle memory barriers

            match self.externsync_image_resources.entry(AbstractImageKey(image.clone())) {
                Entry::Occupied(mut e) => {
                    // FIXME: merge ranges correctly
                    e.get_mut().push(range);
                },
                Entry::Vacant(e) => {
                    let mut v = SmallVec::new();
                    v.push(range);
                    e.insert(v);

                    if let Some(mandatory_layout) = mandatory_layout {
                        if mandatory_layout != inner_sync.layout {
                            image_barriers.push(vk::ImageMemoryBarrier {
                                sType: vk::STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                pNext: ptr::null(),
                                srcAccessMask: 0x0001ffff,  // FIXME:
                                dstAccessMask: 0x0001ffff,  // FIXME:
                                oldLayout: mandatory_layout as u32,
                                newLayout: inner_sync.layout as u32,
                                srcQueueFamilyIndex: vk::QUEUE_FAMILY_IGNORED,
                                dstQueueFamilyIndex: vk::QUEUE_FAMILY_IGNORED,
                                image: image.internal_object(),
                                subresourceRange: vk::ImageSubresourceRange {
                                    aspectMask: 1,        // FIXME:
                                    baseMipLevel: range.mipmap_level_start,
                                    levelCount: range.mipmap_levels_count,
                                    baseArrayLayer: range.array_layer_start,
                                    layerCount: range.array_layers_count,
                                }
                            });
                        }
                    }
                },
            };
        }

        if !buffer_barriers.is_empty() && !image_barriers.is_empty() {
            unsafe {
                let vk = self.device.pointers();
                vk.CmdPipelineBarrier(self.cmd.unwrap(), 0x00010000 /* TODO */, 0x00010000 /* TODO */,
                                      0 /* TODO */, 0, ptr::null(),
                                      buffer_barriers.len() as u32, buffer_barriers.as_ptr(),
                                      image_barriers.len() as u32, image_barriers.as_ptr());
            }
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
    externsync_buffer_resources: HashMap<AbstractBufferKey, SmallVec<[BufferGpuAccessRange; 2]>>,
    externsync_image_resources: HashMap<AbstractImageKey, SmallVec<[ImageGpuAccessRange; 2]>>,
    keep_alive_secondary_command_buffers: Vec<Arc<AbstractCommandBuffer>>,
    keep_alive_descriptor_sets: Vec<Arc<AbstractDescriptorSet>>,
    keep_alive_framebuffers: Vec<Arc<AbstractFramebuffer>>,
    keep_alive_renderpasses: Vec<Arc<RenderPass>>,
    keep_alive_image_views_resources: Vec<Arc<AbstractImageView>>,
    keep_alive_pipelines: Vec<Arc<GenericPipeline>>,
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
              queue: &Arc<Queue>) -> Result<Submission, OomError>   // TODO: wrong error type
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

    let submission_id = queue.device().fetch_submission_id();

    for (resource, ranges) in me.externsync_buffer_resources.iter() {
        // FIXME: for the moment `write` is always true ; that shouldn't be the case
        // FIXME: wrong offset and size
        unsafe {
            let result = resource.0.memory().gpu_access(queue, submission_id, ranges, Some(&fence));

            if let Some(s) = result.pre_semaphore {
                pre_semaphores_ids.push(s.internal_object());
                pre_semaphores_stages.push(vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT);     // TODO:
                pre_semaphores.push(s);
            }

            if let Some(s) = result.post_semaphore {
                post_semaphores.push(s.clone());
                post_semaphores_ids.push(s.internal_object());
            }
        }
    }

    for (resource, ranges) in me.externsync_image_resources.iter() {
        unsafe {
            let result = resource.0.memory().gpu_access(queue, submission_id, ranges, Some(&fence));

            if let Some(s) = result.pre_semaphore {
                pre_semaphores_ids.push(s.internal_object());
                pre_semaphores_stages.push(vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT);     // TODO:
                pre_semaphores.push(s);
            }

            if let Some(s) = result.post_semaphore {
                post_semaphores.push(s.clone());
                post_semaphores_ids.push(s.internal_object());
            }
        }
    }

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

    Ok(Submission {
        cmd: Some(me_arc),
        semaphores: post_semaphores.iter().cloned().chain(pre_semaphores.iter().cloned()).collect(),
        fence: fence,
    })
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
    semaphores: Vec<Arc<Semaphore>>,
    fence: Arc<Fence>,
}

impl Submission {
    #[doc(hidden)]
    #[inline]
    pub fn from_raw(semaphores: Vec<Arc<Semaphore>>, fence: Arc<Fence>) -> Submission {
        Submission {
            cmd: None,
            semaphores: semaphores,
            fence: fence,
        }
    }

    /// Returns `true` is destroying this `Submission` object would block the CPU for some time.
    #[inline]
    pub fn destroying_would_block(&self) -> bool {
        self.fence.ready().unwrap_or(false)     // TODO: what to do in case of error?
    }
}

impl Drop for Submission {
    #[inline]
    fn drop(&mut self) {
        self.fence.wait(5 * 1000 * 1000 * 1000 /* 5 seconds */).unwrap();
    }
}

#[derive(Clone)]
struct AbstractBufferKey(Arc<AbstractBuffer>);

impl PartialEq for AbstractBufferKey {
    #[inline]
    fn eq(&self, other: &AbstractBufferKey) -> bool {
        &*self.0 as *const AbstractBuffer == &*other.0 as *const AbstractBuffer
    }
}

impl Eq for AbstractBufferKey {}

impl hash::Hash for AbstractBufferKey {
    #[inline]
    fn hash<H>(&self, state: &mut H) where H: hash::Hasher {
        let ptr = &*self.0 as *const AbstractBuffer as *const () as usize;
        hash::Hash::hash(&ptr, state)
    }
}

#[derive(Clone)]
struct AbstractImageKey(Arc<AbstractImage>);

impl PartialEq for AbstractImageKey {
    #[inline]
    fn eq(&self, other: &AbstractImageKey) -> bool {
        &*self.0 as *const AbstractImage == &*other.0 as *const AbstractImage
    }
}

impl Eq for AbstractImageKey {}

impl hash::Hash for AbstractImageKey {
    #[inline]
    fn hash<H>(&self, state: &mut H) where H: hash::Hasher {
        let ptr = &*self.0 as *const AbstractImage as *const () as usize;
        hash::Hash::hash(&ptr, state)
    }
}

struct BufferInnerSync {
    range: Range<usize>,
    write: bool,
    // TODO: access mask
}

struct ImageInnerSync {
    mipmap_levels_range: Range<u32>,
    array_layers_range: Range<u32>,
    write: bool,
    layout: ImageLayout,
    // TODO: access mask
}
