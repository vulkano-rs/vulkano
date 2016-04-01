// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt;
use std::hash;
use std::mem;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;
use std::u64;
use smallvec::SmallVec;

use buffer::Buffer;
use buffer::BufferSlice;
use buffer::TypedBuffer;
use buffer::traits::AccessRange as BufferAccessRange;
use command_buffer::CommandBufferPool;
use command_buffer::DynamicState;
use descriptor_set::AbstractDescriptorSet;
use descriptor_set::Layout as PipelineLayoutDesc;
use descriptor_set::DescriptorSetsCollection;
use device::Queue;
use format::ClearValue;
use format::FormatDesc;
use format::FormatTy;
use format::PossibleFloatFormatDesc;
use framebuffer::RenderPass;
use framebuffer::Framebuffer;
use framebuffer::Subpass;
use image::Image;
use image::ImageView;
use image::sys::Layout as ImageLayout;
use image::traits::ImageClearValue;
use image::traits::ImageContent;
use image::traits::AccessRange as ImageAccessRange;
use pipeline::ComputePipeline;
use pipeline::GraphicsPipeline;
use pipeline::input_assembly::Index;
use pipeline::vertex::Definition as VertexDefinition;
use pipeline::vertex::Source as VertexSource;
use sync::Fence;
use sync::FenceWaitError;
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
//
// Implementation notes.
//
// Adding a command to an `InnerCommandBufferBuilder` does not immediately add it to the
// `vk::CommandBuffer`. Instead the command is added to a list of staging commands. The reason
// for this design is that we want to minimize the number of pipeline barriers. In order to know
// what must be in a pipeline barrier, we have to be ahead of the actual commands.
//
pub struct InnerCommandBufferBuilder {
    device: Arc<Device>,
    pool: Arc<CommandBufferPool>,
    cmd: Option<vk::CommandBuffer>,

    // If true, we're inside a secondary command buffer (compute or graphics).
    is_secondary: bool,

    // If true, we're inside a secondary graphics command buffer.
    is_secondary_graphics: bool,

    // List of accesses made by this command buffer to buffers and images, exclusing the staging
    // commands and the staging render pass.
    //
    // If a buffer/image is missing in this list, that means it hasn't been used by this command
    // buffer yet and is still in its default state.
    //
    // This list is only updated by the `flush()` function.
    buffers_state: HashMap<(BufferKey, usize), InternalBufferBlockAccess>,
    images_state: HashMap<(ImageKey, (u32, u32)), InternalImageBlockAccess>,

    // List of commands that are waiting to be submitted to the Vulkan command buffer. Doesn't
    // include commands that were submitted within a render pass.
    staging_commands: Vec<Box<FnMut(&vk::DevicePointers, vk::CommandBuffer)>>,

    // List of resources accesses made by the comands in `staging_commands`. Doesn't include
    // commands added to the current render pass.
    staging_required_buffer_accesses: HashMap<(BufferKey, usize), InternalBufferBlockAccess>,
    staging_required_image_accesses: HashMap<(ImageKey, (u32, u32)), InternalImageBlockAccess>,

    // List of commands that are waiting to be submitted to the Vulkan command buffer when we're
    // inside a render pass. Flushed when `end_renderpass` is called.
    render_pass_staging_commands: Vec<Box<FnMut(&vk::DevicePointers, vk::CommandBuffer)>>,

    // List of resources accesses made by the current render pass. Merged with
    // `staging_required_buffer_accesses` and `staging_required_image_accesses` when
    // `end_renderpass` is called.
    render_pass_staging_required_buffer_accesses: HashMap<(BufferKey, usize), InternalBufferBlockAccess>,
    render_pass_staging_required_image_accesses: HashMap<(ImageKey, (u32, u32)), InternalImageBlockAccess>,

    // List of resources that must be kept alive because they are used by this command buffer.
    keep_alive: Vec<Arc<KeepAlive>>,

    // Current pipeline object binded to the graphics bind point. Includes all staging commands.
    current_graphics_pipeline: Option<vk::Pipeline>,

    // Current pipeline object binded to the compute bind point. Includes all staging commands.
    current_compute_pipeline: Option<vk::Pipeline>,

    // Current state of the dynamic state within the command buffer. Includes all staging commands.
    current_dynamic_state: DynamicState,
}

impl InnerCommandBufferBuilder {
    /// Creates a new builder.
    pub fn new<R>(pool: &Arc<CommandBufferPool>, secondary: bool, secondary_cont: Option<Subpass<R>>,
                  secondary_cont_fb: Option<&Arc<Framebuffer<R>>>)
                  -> Result<InnerCommandBufferBuilder, OomError>
        where R: RenderPass + 'static + Send + Sync
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

        let mut keep_alive = Vec::new();

        unsafe {
            // TODO: one time submit
            let flags = vk::COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT |     // TODO:
                        if secondary_cont.is_some() { vk::COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT } else { 0 };

            let (rp, sp) = if let Some(ref sp) = secondary_cont {
                keep_alive.push(sp.render_pass().clone() as Arc<_>);
                (sp.render_pass().render_pass().internal_object(), sp.index())
            } else {
                (0, 0)
            };

            let framebuffer = if let Some(fb) = secondary_cont_fb {
                keep_alive.push(fb.clone() as Arc<_>);
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
            is_secondary: secondary,
            is_secondary_graphics: secondary_cont.is_some(),
            buffers_state: HashMap::new(),
            images_state: HashMap::new(),
            staging_commands: Vec::new(),
            staging_required_buffer_accesses: HashMap::new(),
            staging_required_image_accesses: HashMap::new(),
            render_pass_staging_commands: Vec::new(),
            render_pass_staging_required_buffer_accesses: HashMap::new(),
            render_pass_staging_required_image_accesses: HashMap::new(),
            keep_alive: keep_alive,
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
    pub unsafe fn execute_commands<'a>(mut self, cb_arc: Arc<KeepAlive>,
                                       cb: &InnerCommandBuffer) -> InnerCommandBufferBuilder
    {
        debug_assert!(!self.is_secondary);
        debug_assert!(!self.is_secondary_graphics);

        // By keeping alive the secondary command buffer itself, we also keep alive all
        // the resources stored by it.
        self.keep_alive.push(cb_arc);

        // Merging the resources of the command buffer.
        if self.render_pass_staging_commands.is_empty() {
            // We're outside of a render pass.

            // Flushing if required.
            let mut conflict = false;
            for (buffer, access) in cb.buffers_state.iter() {
                if let Some(&entry) = self.staging_required_buffer_accesses.get(&buffer) {
                    if entry.write || access.write {
                        conflict = true;
                        break;
                    }
                }
            }
            for (image, access) in cb.images_state.iter() {
                if let Some(entry) = self.staging_required_image_accesses.get(&image) {
                    // TODO: should be reviewed
                    if entry.write || access.write || entry.new_layout != access.old_layout {
                        conflict = true;
                        break;
                    }
                }
            }
            if conflict {
                self.flush(false);
            }

            // Inserting in `staging_required_buffer_accesses`.
            for (buffer, access) in cb.buffers_state.iter() {
                match self.staging_required_buffer_accesses.entry(buffer.clone()) {
                    Entry::Vacant(e) => { e.insert(access.clone()); },
                    Entry::Occupied(mut entry) => {
                        let mut entry = entry.get_mut();
                        entry.stages &= access.stages;
                        entry.accesses &= access.stages;
                        entry.write = entry.write || access.write;
                    }
                }
            }

            // Inserting in `staging_required_image_accesses`.
            for (image, access) in cb.images_state.iter() {
                match self.staging_required_image_accesses.entry(image.clone()) {
                    Entry::Vacant(e) => { e.insert(access.clone()); },
                    Entry::Occupied(mut entry) => {
                        let mut entry = entry.get_mut();
                        entry.stages &= access.stages;
                        entry.accesses &= access.stages;
                        entry.write = entry.write || access.write;
                        debug_assert_eq!(entry.new_layout, access.old_layout);
                        entry.aspects |= access.aspects;
                        entry.new_layout = access.new_layout;
                    }
                }
            }

            // Adding the command to the staging commands.
            {
                let cb_cmd = cb.cmd;
                self.staging_commands.push(Box::new(move |vk, cmd| {
                    vk.CmdExecuteCommands(cmd, 1, &cb_cmd);
                }));
            }

        } else {
            // We're inside a render pass.
            for (buffer, access) in cb.buffers_state.iter() {
                // TODO: check for collisions
                self.render_pass_staging_required_buffer_accesses.insert(buffer.clone(), access.clone());
            }

            for (image, access) in cb.images_state.iter() {
                // TODO: check for collisions
                self.render_pass_staging_required_image_accesses.insert(image.clone(), access.clone());
            }

            // Adding the command to the staging commands.
            {
                let cb_cmd = cb.cmd;
                self.render_pass_staging_commands.push(Box::new(move |vk, cmd| {
                    vk.CmdExecuteCommands(cmd, 1, &cb_cmd);
                }));
            }
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
    pub unsafe fn update_buffer<'a, B, T, Bt>(mut self, buffer: B, data: &T)
                                              -> InnerCommandBufferBuilder
        where B: Into<BufferSlice<'a, T, Bt>>, Bt: Buffer + 'static, T: Clone + 'static
    {
        debug_assert!(self.render_pass_staging_commands.is_empty());

        let buffer = buffer.into();

        assert_eq!(buffer.size(), mem::size_of_val(data));
        assert!(buffer.size() <= 65536);
        assert!(buffer.offset() % 4 == 0);
        assert!(buffer.size() % 4 == 0);
        assert!(buffer.buffer().inner_buffer().usage_transfer_dest());

        // FIXME: check queue family of the buffer

        self.add_buffer_resource_outside(buffer.buffer().clone() as Arc<_>, true,
                                         buffer.offset() .. buffer.offset() + buffer.size(),
                                         vk::PIPELINE_STAGE_TRANSFER_BIT,
                                         vk::ACCESS_TRANSFER_WRITE_BIT);

        {
            let buffer_offset = buffer.offset() as vk::DeviceSize;
            let buffer_size = buffer.size() as vk::DeviceSize;
            let buffer = buffer.buffer().inner_buffer().internal_object();
            let mut data = Some(data.clone());        // TODO: meh for Cloning, but I guess there's no other choice

            self.staging_commands.push(Box::new(move |vk, cmd| {
                let data = data.take().unwrap();
                vk.CmdUpdateBuffer(cmd, buffer, buffer_offset, buffer_size,
                                   &data as *const T as *const _);
            }));
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
        debug_assert!(self.render_pass_staging_commands.is_empty());

        assert!(self.pool.queue_family().supports_transfers());
        assert!(offset + size <= buffer.size());
        assert!(offset % 4 == 0);
        assert!(size % 4 == 0);
        assert!(buffer.inner_buffer().usage_transfer_dest());

        self.add_buffer_resource_outside(buffer.clone() as Arc<_>, true, offset .. offset + size,
                                         vk::PIPELINE_STAGE_TRANSFER_BIT,
                                         vk::ACCESS_TRANSFER_WRITE_BIT);

        // FIXME: check that the queue family supports transfers
        // FIXME: check queue family of the buffer

        {
            let buffer = buffer.clone();
            self.staging_commands.push(Box::new(move |vk, cmd| {
                vk.CmdFillBuffer(cmd, buffer.inner_buffer().internal_object(),
                                 offset as vk::DeviceSize, size as vk::DeviceSize, data);
            }));
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
        debug_assert!(self.render_pass_staging_commands.is_empty());

        assert_eq!(&**source.inner_buffer().device() as *const _,
                   &**destination.inner_buffer().device() as *const _);
        assert!(source.inner_buffer().usage_transfer_src());
        assert!(destination.inner_buffer().usage_transfer_dest());

        self.add_buffer_resource_outside(source.clone() as Arc<_>, false, 0 .. source.size(),
                                         vk::PIPELINE_STAGE_TRANSFER_BIT,
                                         vk::ACCESS_TRANSFER_READ_BIT);
        self.add_buffer_resource_outside(destination.clone() as Arc<_>, true, 0 .. source.size(),
                                         vk::PIPELINE_STAGE_TRANSFER_BIT,
                                         vk::ACCESS_TRANSFER_WRITE_BIT);

        {
            let source_size = source.size() as u64;     // FIXME: what is destination is too small?
            let source = source.inner_buffer().internal_object();
            let destination = destination.inner_buffer().internal_object();

            self.staging_commands.push(Box::new(move |vk, cmd| {
                let copy = vk::BufferCopy {
                    srcOffset: 0,
                    dstOffset: 0,
                    size: source_size,
                };

                vk.CmdCopyBuffer(cmd, source, destination, 1, &copy);
            }));
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
    pub unsafe fn clear_color_image<'a, I, V>(mut self, image: &Arc<I>, color: V)
                                              -> InnerCommandBufferBuilder
        where I: ImageClearValue<V> + 'static   // FIXME: should accept uint and int images too
    {
        debug_assert!(self.render_pass_staging_commands.is_empty());

        assert!(image.format().is_float()); // FIXME: should accept uint and int images too

        let color = image.decode(color).unwrap(); /* FIXME: error */

        {
            let image = image.inner_image().internal_object();

            self.staging_commands.push(Box::new(move |vk, cmd| {
                let color = match color {
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

                vk.CmdClearColorImage(cmd, image, vk::IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL /* FIXME: */,
                                      &color, 1, &range);
            }));
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
    pub unsafe fn copy_buffer_to_color_image<'a, P, S, Sb, Img>(mut self, source: S, image: &Arc<Img>,
                                                                mip_level: u32, array_layers_range: Range<u32>,
                                                                offset: [u32; 3], extent: [u32; 3])
                                                             -> InnerCommandBufferBuilder
        where S: Into<BufferSlice<'a, [P], Sb>>, Img: ImageContent<P> + Image + 'static,
              Sb: Buffer + 'static
    {
        // FIXME: check the parameters

        debug_assert!(self.render_pass_staging_commands.is_empty());

        //assert!(image.format().is_float_or_compressed());

        let source = source.into();
        self.add_buffer_resource_outside(source.buffer().clone() as Arc<_>, false,
                                         source.offset() .. source.offset() + source.size(),
                                         vk::PIPELINE_STAGE_TRANSFER_BIT,
                                         vk::ACCESS_TRANSFER_READ_BIT);
        self.add_image_resource_outside(image.clone() as Arc<_>, mip_level .. mip_level + 1,
                                        array_layers_range.clone(), true,
                                        ImageLayout::TransferDstOptimal,
                                        vk::PIPELINE_STAGE_TRANSFER_BIT,
                                        vk::ACCESS_TRANSFER_WRITE_BIT);

        {
            let source_offset = source.offset() as vk::DeviceSize;
            let source = source.buffer().inner_buffer().internal_object();
            let image = image.inner_image().internal_object();

            self.staging_commands.push(Box::new(move |vk, cmd| {
                let region = vk::BufferImageCopy {
                    bufferOffset: source_offset,
                    bufferRowLength: 0,
                    bufferImageHeight: 0,
                    imageSubresource: vk::ImageSubresourceLayers {
                        aspectMask: vk::IMAGE_ASPECT_COLOR_BIT,
                        mipLevel: mip_level,
                        baseArrayLayer: array_layers_range.start,
                        layerCount: array_layers_range.end - array_layers_range.start,
                    },
                    imageOffset: vk::Offset3D {
                        x: offset[0] as i32,
                        y: offset[1] as i32,
                        z: offset[2] as i32,
                    },
                    imageExtent: vk::Extent3D {
                        width: extent[0],
                        height: extent[1],
                        depth: extent[2],
                    },
                };

                vk.CmdCopyBufferToImage(cmd, source, image,
                                        vk::IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL /* FIXME */,
                                        1, &region);
            }));
        }

        self
    }

    pub unsafe fn blit<Si, Di>(mut self, source: &Arc<Si>, source_mip_level: u32,
                               source_array_layers: Range<u32>, src_coords: [Range<i32>; 3],
                               destination: &Arc<Di>, dest_mip_level: u32,
                               dest_array_layers: Range<u32>, dest_coords: [Range<i32>; 3])
                               -> InnerCommandBufferBuilder
        where Si: Image + 'static, Di: Image + 'static
    {
        // FIXME: check the parameters

        debug_assert!(self.render_pass_staging_commands.is_empty());

        assert!(source.supports_blit_source());
        assert!(destination.supports_blit_destination());

        self.add_image_resource_outside(source.clone() as Arc<_>,
                                        source_mip_level .. source_mip_level + 1,
                                        source_array_layers.clone(), false,
                                        ImageLayout::TransferSrcOptimal,
                                        vk::PIPELINE_STAGE_TRANSFER_BIT,
                                        vk::ACCESS_TRANSFER_READ_BIT);
        self.add_image_resource_outside(destination.clone() as Arc<_>,
                                        dest_mip_level .. dest_mip_level + 1,
                                        dest_array_layers.clone(), true,
                                        ImageLayout::TransferDstOptimal,
                                        vk::PIPELINE_STAGE_TRANSFER_BIT,
                                        vk::ACCESS_TRANSFER_WRITE_BIT);

        {
            let source = source.inner_image().internal_object();
            let destination = destination.inner_image().internal_object();

            self.staging_commands.push(Box::new(move |vk, cmd| {
                let region = vk::ImageBlit {
                    srcSubresource: vk::ImageSubresourceLayers {
                        aspectMask: vk::IMAGE_ASPECT_COLOR_BIT,
                        mipLevel: source_mip_level,
                        baseArrayLayer: source_array_layers.start,
                        layerCount: source_array_layers.end - source_array_layers.start,
                    },
                    srcOffsets: [
                        vk::Offset3D {
                            x: src_coords[0].start,
                            y: src_coords[1].start,
                            z: src_coords[2].start,
                        }, vk::Offset3D {
                            x: src_coords[0].end,
                            y: src_coords[1].end,
                            z: src_coords[2].end,
                        }
                    ],
                    dstSubresource: vk::ImageSubresourceLayers {
                        aspectMask: vk::IMAGE_ASPECT_COLOR_BIT,
                        mipLevel: dest_mip_level,
                        baseArrayLayer: dest_array_layers.start,
                        layerCount: dest_array_layers.end - dest_array_layers.start,
                    },
                    dstOffsets: [
                        vk::Offset3D {
                            x: dest_coords[0].start,
                            y: dest_coords[1].start,
                            z: dest_coords[2].start,
                        }, vk::Offset3D {
                            x: dest_coords[0].end,
                            y: dest_coords[1].end,
                            z: dest_coords[2].end,
                        }
                    ],
                };

                vk.CmdBlitImage(cmd, source, ImageLayout::TransferSrcOptimal as u32,
                                destination, ImageLayout::TransferDstOptimal as u32,
                                1, &region, vk::FILTER_LINEAR);
            }));
        }

        self
    }

    pub unsafe fn dispatch<Pl, L>(mut self, pipeline: &Arc<ComputePipeline<Pl>>, sets: L,
                                  dimensions: [u32; 3]) -> InnerCommandBufferBuilder
        where L: 'static + DescriptorSetsCollection + Send + Sync,
              Pl: 'static + PipelineLayoutDesc + Send + Sync
    {
        debug_assert!(self.render_pass_staging_commands.is_empty());

        self.bind_compute_pipeline_state(pipeline, sets);

        self.staging_commands.push(Box::new(move |vk, cmd| {
            vk.CmdDispatch(cmd, dimensions[0], dimensions[1], dimensions[2]);
        }));

        self
    }

    /// Calls `vkCmdDraw`.
    // FIXME: push constants
    pub unsafe fn draw<V, Pv, Pl, L, Rp>(mut self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Rp>>,
                             vertices: V, dynamic: &DynamicState,
                             sets: L) -> InnerCommandBufferBuilder
        where Pv: 'static + VertexDefinition + VertexSource<V>, L: 'static + DescriptorSetsCollection + Send + Sync,
              Pl: 'static + PipelineLayoutDesc + Send + Sync, Rp: 'static + Send + Sync
    {
        // FIXME: add buffers to the resources

        self.bind_gfx_pipeline_state(pipeline, dynamic, sets);

        let vertices = pipeline.vertex_definition().decode(vertices);

        let offsets = (0 .. vertices.0.len()).map(|_| 0).collect::<SmallVec<[_; 8]>>();
        let ids = vertices.0.map(|b| {
            assert!(b.inner_buffer().usage_vertex_buffer());
            self.add_buffer_resource_inside(b.clone(), false, 0 .. b.size(),
                                            vk::PIPELINE_STAGE_VERTEX_INPUT_BIT,
                                            vk::ACCESS_VERTEX_ATTRIBUTE_READ_BIT);
            b.inner_buffer().internal_object()
        }).collect::<SmallVec<[_; 8]>>();

        {
            let mut ids = Some(ids);
            let mut offsets = Some(offsets);
            let num_vertices = vertices.1 as u32;
            let num_instances = vertices.2 as u32;

            self.render_pass_staging_commands.push(Box::new(move |vk, cmd| {
                let ids = ids.take().unwrap();
                let offsets = offsets.take().unwrap();

                vk.CmdBindVertexBuffers(cmd, 0, ids.len() as u32, ids.as_ptr(), offsets.as_ptr());
                vk.CmdDraw(cmd, num_vertices, num_instances, 0, 0);  // FIXME: params
            }));
        }

        self
    }

    /// Calls `vkCmdDrawIndexed`.
    // FIXME: push constants
    pub unsafe fn draw_indexed<'a, V, Pv, Pl, Rp, L, I, Ib, Ibb>(mut self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Rp>>,
                                                          vertices: V, indices: Ib, dynamic: &DynamicState,
                                                          sets: L) -> InnerCommandBufferBuilder
        where L: 'static + DescriptorSetsCollection + Send + Sync,
              Pv: 'static + VertexDefinition + VertexSource<V>,
              Pl: 'static + PipelineLayoutDesc + Send + Sync, Rp: 'static + Send + Sync,
              Ib: Into<BufferSlice<'a, [I], Ibb>>, I: 'static + Index, Ibb: Buffer + 'static
    {
        // FIXME: add buffers to the resources

        self.bind_gfx_pipeline_state(pipeline, dynamic, sets);


        let indices = indices.into();

        let vertices = pipeline.vertex_definition().decode(vertices);

        let offsets = (0 .. vertices.0.len()).map(|_| 0).collect::<SmallVec<[_; 8]>>();
        let ids = vertices.0.map(|b| {
            assert!(b.inner_buffer().usage_vertex_buffer());
            self.add_buffer_resource_inside(b.clone(), false, 0 .. b.size(),
                                            vk::PIPELINE_STAGE_VERTEX_INPUT_BIT,
                                            vk::ACCESS_VERTEX_ATTRIBUTE_READ_BIT);
            b.inner_buffer().internal_object()
        }).collect::<SmallVec<[_; 8]>>();

        assert!(indices.buffer().inner_buffer().usage_index_buffer());

        self.add_buffer_resource_inside(indices.buffer().clone() as Arc<_>, false,
                                        indices.offset() .. indices.offset() + indices.size(),
                                        vk::PIPELINE_STAGE_VERTEX_INPUT_BIT,
                                        vk::ACCESS_INDEX_READ_BIT);

        {
            let mut ids = Some(ids);
            let mut offsets = Some(offsets);
            let indices_offset = indices.offset() as u64;
            let indices_len = indices.len() as u32;
            let indices_ty = I::ty() as u32;
            let indices = indices.buffer().inner_buffer().internal_object();
            let num_instances = vertices.2 as u32;

            self.render_pass_staging_commands.push(Box::new(move |vk, cmd| {
                let ids = ids.take().unwrap();
                let offsets = offsets.take().unwrap();

                vk.CmdBindIndexBuffer(cmd, indices, indices_offset, indices_ty);
                vk.CmdBindVertexBuffers(cmd, 0, ids.len() as u32, ids.as_ptr(), offsets.as_ptr());
                vk.CmdDrawIndexed(cmd, indices_len, num_instances, 0, 0, 0);  // FIXME: params
            }));
        }

        self
    }

    fn bind_compute_pipeline_state<Pl, L>(&mut self, pipeline: &Arc<ComputePipeline<Pl>>, sets: L)
        where L: 'static + DescriptorSetsCollection,
              Pl: 'static + PipelineLayoutDesc + Send + Sync
    {
        unsafe {
            assert!(sets.is_compatible_with(pipeline.layout()));

            if self.current_compute_pipeline != Some(pipeline.internal_object()) {
                self.keep_alive.push(pipeline.clone());
                let pipeline = pipeline.internal_object();
                self.staging_commands.push(Box::new(move |vk, cmd| {
                    vk.CmdBindPipeline(cmd, vk::PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline);
                }));
                self.current_compute_pipeline = Some(pipeline);
            }

            let mut descriptor_sets = sets.list().collect::<SmallVec<[_; 32]>>();

            for set in descriptor_sets.iter() {
                for &(ref img, block, layout) in AbstractDescriptorSet::images_list(&**set).iter() {
                    self.add_image_resource_outside(img.clone(), 0 .. 1 /* FIXME */, 0 .. 1 /* FIXME */,
                                                   false, layout, vk::PIPELINE_STAGE_ALL_COMMANDS_BIT /* FIXME */,
                                                   vk::ACCESS_SHADER_READ_BIT | vk::ACCESS_UNIFORM_READ_BIT /* TODO */);
                }
                for buffer in AbstractDescriptorSet::buffers_list(&**set).iter() {
                    self.add_buffer_resource_outside(buffer.clone(), false, 0 .. buffer.size() /* TODO */,
                                                    vk::PIPELINE_STAGE_ALL_COMMANDS_BIT /* FIXME */,
                                                    vk::ACCESS_SHADER_READ_BIT | vk::ACCESS_UNIFORM_READ_BIT /* TODO */);
                }
            }

            for d in descriptor_sets.iter() { self.keep_alive.push(mem::transmute(d.clone()) /* FIXME: */); }
            let mut descriptor_sets = Some(descriptor_sets.into_iter().map(|set| set.internal_object()).collect::<SmallVec<[_; 32]>>());

            // TODO: shouldn't rebind everything every time
            if !descriptor_sets.as_ref().unwrap().is_empty() {
                let playout = pipeline.layout().internal_object();
                self.staging_commands.push(Box::new(move |vk, cmd| {
                    let descriptor_sets = descriptor_sets.take().unwrap();
                    vk.CmdBindDescriptorSets(cmd, vk::PIPELINE_BIND_POINT_COMPUTE,
                                             playout, 0, descriptor_sets.len() as u32,
                                             descriptor_sets.as_ptr(), 0, ptr::null());   // FIXME: dynamic offsets
                }));
            }
        }
    }

    fn bind_gfx_pipeline_state<V, Pl, L, Rp>(&mut self, pipeline: &Arc<GraphicsPipeline<V, Pl, Rp>>,
                                             dynamic: &DynamicState, sets: L)
        where V: 'static + VertexDefinition + Send + Sync, L: 'static + DescriptorSetsCollection + Send + Sync,
              Pl: 'static + PipelineLayoutDesc + Send + Sync, Rp: 'static + Send + Sync
    {
        unsafe {
            assert!(sets.is_compatible_with(pipeline.layout()));

            if self.current_graphics_pipeline != Some(pipeline.internal_object()) {
                self.keep_alive.push(pipeline.clone());
                let pipeline = pipeline.internal_object();
                self.render_pass_staging_commands.push(Box::new(move |vk, cmd| {
                    vk.CmdBindPipeline(cmd, vk::PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                }));
                self.current_graphics_pipeline = Some(pipeline);
            }

            if let Some(line_width) = dynamic.line_width {
                assert!(pipeline.has_dynamic_line_width());
                // TODO: check limits
                if self.current_dynamic_state.line_width != Some(line_width) {
                    self.render_pass_staging_commands.push(Box::new(move |vk, cmd| {
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
                let mut viewports = Some(viewports.iter().map(|v| v.clone().into()).collect::<SmallVec<[_; 16]>>());
                self.render_pass_staging_commands.push(Box::new(move |vk, cmd| {
                    let viewports = viewports.take().unwrap();
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
                let mut scissors = Some(scissors.iter().map(|v| v.clone().into()).collect::<SmallVec<[_; 16]>>());
                self.render_pass_staging_commands.push(Box::new(move |vk, cmd| {
                    let scissors = scissors.take().unwrap();
                    vk.CmdSetScissor(cmd, 0, scissors.len() as u32, scissors.as_ptr());
                }));
            } else {
                assert!(!pipeline.has_dynamic_scissors());
            }

            let mut descriptor_sets = sets.list().collect::<SmallVec<[_; 32]>>();
            for set in descriptor_sets.iter() {
                for &(ref img, block, layout) in AbstractDescriptorSet::images_list(&**set).iter() {
                    self.add_image_resource_inside(img.clone(), 0 .. 1 /* FIXME */, 0 .. 1 /* FIXME */,
                                                   false, layout, layout, vk::PIPELINE_STAGE_ALL_COMMANDS_BIT /* FIXME */,
                                                   vk::ACCESS_SHADER_READ_BIT | vk::ACCESS_UNIFORM_READ_BIT /* TODO */);
                }
                for buffer in AbstractDescriptorSet::buffers_list(&**set).iter() {
                    self.add_buffer_resource_inside(buffer.clone(), false, 0 .. buffer.size() /* TODO */,
                                                    vk::PIPELINE_STAGE_ALL_COMMANDS_BIT /* FIXME */,
                                                    vk::ACCESS_SHADER_READ_BIT | vk::ACCESS_UNIFORM_READ_BIT /* TODO */);
                }
            }
            for d in descriptor_sets.iter() { self.keep_alive.push(mem::transmute(d.clone()) /* FIXME: */); }
            let mut descriptor_sets = Some(descriptor_sets.into_iter().map(|set| set.internal_object()).collect::<SmallVec<[_; 32]>>());

            // FIXME: input attachments of descriptor sets have to be checked against input
            //        attachments of the render pass

            // TODO: shouldn't rebind everything every time
            if !descriptor_sets.as_ref().unwrap().is_empty() {
                let playout = pipeline.layout().internal_object();
                self.render_pass_staging_commands.push(Box::new(move |vk, cmd| {
                    let descriptor_sets = descriptor_sets.take().unwrap();
                    vk.CmdBindDescriptorSets(cmd, vk::PIPELINE_BIND_POINT_GRAPHICS, playout,
                                             0, descriptor_sets.len() as u32,
                                             descriptor_sets.as_ptr(), 0, ptr::null());   // FIXME: dynamic offsets
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
        debug_assert!(self.render_pass_staging_commands.is_empty());
        debug_assert!(self.render_pass_staging_required_buffer_accesses.is_empty());
        debug_assert!(self.render_pass_staging_required_image_accesses.is_empty());

        assert!(framebuffer.is_compatible_with(render_pass));

        self.keep_alive.push(framebuffer.clone() as Arc<_>);
        self.keep_alive.push(render_pass.clone() as Arc<_>);

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

        for &(ref attachment, ref image, initial_layout, final_layout) in framebuffer.attachments() {
            self.keep_alive.push(mem::transmute(attachment.clone()) /* FIXME: */);

            let stages = vk::PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                         vk::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                         vk::PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                         vk::PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT; // FIXME:

            let accesses = vk::ACCESS_COLOR_ATTACHMENT_READ_BIT |
                           vk::ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                           vk::ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                           vk::ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                           vk::ACCESS_INPUT_ATTACHMENT_READ_BIT;       // FIXME:

            // FIXME: parameters
            self.add_image_resource_inside(image.clone(), 0 .. 1, 0 .. 1, true,
                                           initial_layout, final_layout, stages, accesses);
        }

        {
            let mut clear_values = Some(clear_values);
            let render_pass = render_pass.render_pass().internal_object();
            let (fw, fh) = (framebuffer.width(), framebuffer.height());
            let framebuffer = framebuffer.internal_object();

            self.render_pass_staging_commands.push(Box::new(move |vk, cmd| {
                let clear_values = clear_values.take().unwrap();

                let infos = vk::RenderPassBeginInfo {
                    sType: vk::STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                    pNext: ptr::null(),
                    renderPass: render_pass,
                    framebuffer: framebuffer,
                    renderArea: vk::Rect2D {                // TODO: let user customize
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: fw,
                            height: fh,
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

                vk.CmdBeginRenderPass(cmd, &infos, content);
            }));
        }

        self
    }

    #[inline]
    pub unsafe fn next_subpass(mut self, secondary_cmd_buffers: bool) -> InnerCommandBufferBuilder {
        debug_assert!(!self.render_pass_staging_commands.is_empty());

        let content = if secondary_cmd_buffers {
            vk::SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS
        } else {
            vk::SUBPASS_CONTENTS_INLINE
        };

        self.render_pass_staging_commands.push(Box::new(move |vk, cmd| {
            vk.CmdNextSubpass(cmd, content);
        }));

        self
    }

    /// Ends the current render pass by calling `vkCmdEndRenderPass`.
    ///
    /// # Safety
    ///
    /// Assumes that you're inside a render pass and that all subpasses have been processed.
    ///
    #[inline]
    pub unsafe fn end_renderpass(mut self) -> InnerCommandBufferBuilder {
        debug_assert!(!self.render_pass_staging_commands.is_empty());
        self.flush_render_pass();
        self.staging_commands.push(Box::new(move |vk, cmd| {
            vk.CmdEndRenderPass(cmd);
        }));
        self
    }

    /// Adds a buffer resource to the list of resources used by this command buffer.
    fn add_buffer_resource_outside(&mut self, buffer: Arc<Buffer>, write: bool,
                                   range: Range<usize>, stages: vk::PipelineStageFlagBits,
                                   accesses: vk::AccessFlagBits)
    {
        // Flushing if required.
        let mut conflict = false;
        for block in buffer.blocks(range.clone()) {
            let key = (BufferKey(buffer.clone()), block);
            if let Some(&entry) = self.staging_required_buffer_accesses.get(&key) {
                if entry.write || write {
                    conflict = true;
                    break;
                }
            }
        }
        if conflict {
            self.flush(false);
        }

        // Inserting in `staging_required_buffer_accesses`.
        for block in buffer.blocks(range.clone()) {
            let key = (BufferKey(buffer.clone()), block);
            match self.staging_required_buffer_accesses.entry(key) {
                Entry::Vacant(e) => {
                    e.insert(InternalBufferBlockAccess {
                        stages: stages,
                        accesses: accesses,
                        write: write,
                    });
                },
                Entry::Occupied(mut entry) => {
                    let mut entry = entry.get_mut();
                    entry.stages &= stages;
                    entry.accesses &= stages;
                    entry.write = entry.write || write;
                }
            }
        }
    }

    /// Adds an image resource to the list of resources used by this command buffer.
    fn add_image_resource_outside(&mut self, image: Arc<Image>, mipmap_levels_range: Range<u32>,
                                  array_layers_range: Range<u32>, write: bool, layout: ImageLayout,
                                  stages: vk::PipelineStageFlagBits, accesses: vk::AccessFlagBits)
    {
        // Flushing if required.
        let mut conflict = false;
        for block in image.blocks(mipmap_levels_range.clone(), array_layers_range.clone()) {
            let key = (ImageKey(image.clone()), block);
            if let Some(entry) = self.staging_required_image_accesses.get(&key) {
                // TODO: should be reviewed
                if entry.write || write || entry.new_layout != layout {
                    conflict = true;
                    break;
                }
            }
        }
        if conflict {
            self.flush(false);
        }

        // Inserting in `staging_required_image_accesses`.
        for block in image.blocks(mipmap_levels_range.clone(), array_layers_range.clone()) {
            let key = (ImageKey(image.clone()), block);
            let aspect_mask = match image.format().ty() {
                FormatTy::Float | FormatTy::Uint | FormatTy::Sint | FormatTy::Compressed => {
                    vk::IMAGE_ASPECT_COLOR_BIT
                },
                FormatTy::Depth => vk::IMAGE_ASPECT_DEPTH_BIT,
                FormatTy::Stencil => vk::IMAGE_ASPECT_STENCIL_BIT,
                FormatTy::DepthStencil => vk::IMAGE_ASPECT_DEPTH_BIT | vk::IMAGE_ASPECT_STENCIL_BIT,
            };

            match self.staging_required_image_accesses.entry(key) {
                Entry::Vacant(e) => {
                    e.insert(InternalImageBlockAccess {
                        stages: stages,
                        accesses: accesses,
                        write: write,
                        aspects: aspect_mask,
                        old_layout: layout,
                        new_layout: layout,
                    });
                },
                Entry::Occupied(mut entry) => {
                    let mut entry = entry.get_mut();
                    entry.stages &= stages;
                    entry.accesses &= stages;
                    entry.write = entry.write || write;
                    debug_assert_eq!(entry.new_layout, layout);
                    entry.aspects |= aspect_mask;
                }
            }
        }
    }

    /// Adds a buffer resource to the list of resources used by this command buffer.
    fn add_buffer_resource_inside(&mut self, buffer: Arc<Buffer>, write: bool,
                                  range: Range<usize>, stages: vk::PipelineStageFlagBits,
                                  accesses: vk::AccessFlagBits)
    {
        // TODO: check for collisions
        for block in buffer.blocks(range.clone()) {
            let key = (BufferKey(buffer.clone()), block);
            self.render_pass_staging_required_buffer_accesses.insert(key, InternalBufferBlockAccess {
                stages: stages,
                accesses: accesses,
                write: write,
            });
        }
    }

    /// Adds an image resource to the list of resources used by this command buffer.
    fn add_image_resource_inside(&mut self, image: Arc<Image>, mipmap_levels_range: Range<u32>,
                                 array_layers_range: Range<u32>, write: bool,
                                 initial_layout: ImageLayout, final_layout: ImageLayout,
                                 stages: vk::PipelineStageFlagBits, accesses: vk::AccessFlagBits)
    {
        // TODO: check for collisions
        for block in image.blocks(mipmap_levels_range.clone(), array_layers_range.clone()) {
            let key = (ImageKey(image.clone()), block);
            let aspect_mask = match image.format().ty() {
                FormatTy::Float | FormatTy::Uint | FormatTy::Sint | FormatTy::Compressed => {
                    vk::IMAGE_ASPECT_COLOR_BIT
                },
                FormatTy::Depth => vk::IMAGE_ASPECT_DEPTH_BIT,
                FormatTy::Stencil => vk::IMAGE_ASPECT_STENCIL_BIT,
                FormatTy::DepthStencil => vk::IMAGE_ASPECT_DEPTH_BIT | vk::IMAGE_ASPECT_STENCIL_BIT,
            };

            self.render_pass_staging_required_image_accesses.insert(key, InternalImageBlockAccess {
                stages: stages,
                accesses: accesses,
                write: write,
                aspects: aspect_mask,
                old_layout: initial_layout,
                new_layout: final_layout,
            });
        }
    }

    /// Flushes the staging render pass commands. Only call this before `vkCmdEndRenderPass` and
    /// before `vkEndCommandBuffer`.
    unsafe fn flush_render_pass(&mut self) {
        // Determine whether there's a conflict between the accesses done within the
        // render pass and the accesses from before the render pass.
        let mut conflict = false;
        for (key, access) in self.render_pass_staging_required_buffer_accesses.iter() {
            if let Some(ex_acc) = self.staging_required_buffer_accesses.get(&key) {
                if access.write || ex_acc.write {
                    conflict = true;
                    break;
                }
            }
        }
        if !conflict {
            for (key, access) in self.render_pass_staging_required_image_accesses.iter() {
                if let Some(ex_acc) = self.staging_required_image_accesses.get(&key) {
                    if access.write || ex_acc.write ||
                       (ex_acc.aspects & access.aspects) != ex_acc.aspects ||
                       access.old_layout != ex_acc.new_layout
                    {
                        conflict = true;
                        break;
                    }
                }
            }
        }
        if conflict {
            // Calling `flush` here means that a `vkCmdPipelineBarrier` will be inserted right
            // before the `vkCmdBeginRenderPass`.
            debug_assert!(!self.is_secondary_graphics);
            self.flush(false);
        }

        // Now merging the render pass accesses with the outter accesses.
        // Conflicts shouldn't happen since we checked above, but they are partly checked again
        // with `debug_assert`s.
        for ((buffer, block), access) in self.render_pass_staging_required_buffer_accesses.drain() {
            match self.staging_required_buffer_accesses.entry((buffer.clone(), block)) {
                Entry::Vacant(e) => { e.insert(access); },
                Entry::Occupied(mut entry) => {
                    let mut entry = entry.get_mut();
                    debug_assert!(!entry.write && !access.write);
                    entry.stages |= access.stages;
                    entry.accesses |= access.accesses;
                }
            }
        }
        for ((image, block), access) in self.render_pass_staging_required_image_accesses.drain() {
            match self.staging_required_image_accesses.entry((image.clone(), block)) {
                Entry::Vacant(e) => { e.insert(access); },
                Entry::Occupied(mut entry) => {
                    let mut entry = entry.get_mut();
                    debug_assert!(!entry.write && !access.write);
                    debug_assert_eq!(entry.new_layout, access.old_layout);
                    entry.stages |= access.stages;
                    entry.accesses |= access.accesses;
                    entry.new_layout = access.new_layout;
                }
            }
        }

        // Merging the commands as well.
        for command in self.render_pass_staging_commands.drain(..) {
            self.staging_commands.push(command);
        }
    }

    /// Flush the staging commands.
    fn flush(&mut self, ignore_empty_staging_commands: bool) {
        let cmd = self.cmd.unwrap();
        let vk = self.device.pointers();

        // If `staging_commands` is empty, that means we are doing two flushes in a row. This
        // means that a command conflicts with itself, for example a buffer reading and writing
        // simultaneously the same block.
        //
        // The `ignore_empty_staging_commands` parameter is here to ignore that check, because
        // in some situations it is legitimate to have two flushes in a row.
        //
        // TODO: handle error better
        if !ignore_empty_staging_commands {
            assert!(!self.staging_commands.is_empty(), "Invalid command detected");
        }

        // Merging the `staging_access` variables to the `state` variables,
        // and determining the list of barriers that are required and updating the resources states.
        // TODO: inefficient because multiple entries for contiguous blocks should be merged
        //       into one
        let mut buffer_barriers: SmallVec<[_; 8]> = SmallVec::new();
        let mut image_barriers: SmallVec<[_; 8]> = SmallVec::new();

        let mut src_stages = 0;
        let mut dst_stages = 0;

        for (buffer, access) in self.staging_required_buffer_accesses.drain() {
            match self.buffers_state.entry(buffer.clone()) {
                Entry::Vacant(entry) => {
                    if (buffer.0).0.host_accesses(buffer.1) && !self.is_secondary {
                        src_stages |= vk::PIPELINE_STAGE_HOST_BIT;
                        dst_stages |= access.stages;

                        let range = (buffer.0).0.block_memory_range(buffer.1);

                        debug_assert!(!self.is_secondary_graphics);
                        buffer_barriers.push(vk::BufferMemoryBarrier {
                            sType: vk::STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                            pNext: ptr::null(),
                            srcAccessMask: vk::ACCESS_HOST_READ_BIT | vk::ACCESS_HOST_WRITE_BIT,
                            dstAccessMask: access.accesses,
                            srcQueueFamilyIndex: vk::QUEUE_FAMILY_IGNORED,
                            dstQueueFamilyIndex: vk::QUEUE_FAMILY_IGNORED,
                            buffer: (buffer.0).0.inner_buffer().internal_object(),
                            offset: range.start as u64,
                            size: (range.end - range.start) as u64,
                        });
                    }

                    entry.insert(access);
                },

                Entry::Occupied(mut entry) => {
                    let entry = entry.get_mut();

                    if entry.write || access.write {
                        src_stages |= entry.stages;
                        dst_stages |= access.stages;

                        let range = (buffer.0).0.block_memory_range(buffer.1);

                        debug_assert!(!self.is_secondary_graphics);
                        buffer_barriers.push(vk::BufferMemoryBarrier {
                            sType: vk::STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                            pNext: ptr::null(),
                            srcAccessMask: entry.accesses,
                            dstAccessMask: access.accesses,
                            srcQueueFamilyIndex: vk::QUEUE_FAMILY_IGNORED,
                            dstQueueFamilyIndex: vk::QUEUE_FAMILY_IGNORED,
                            buffer: (buffer.0).0.inner_buffer().internal_object(),
                            offset: range.start as u64,
                            size: (range.end - range.start) as u64,
                        });

                        entry.stages = access.stages;
                        entry.accesses = access.accesses;
                    }

                    entry.write = entry.write || access.write;
                },
            }
        }

        for (image, access) in self.staging_required_image_accesses.drain() {
            match self.images_state.entry(image.clone()) {
                Entry::Vacant(entry) => {
                    // This is the first ever use of this image block in this command buffer.
                    // Therefore we need to query the image for the layout that it is going to
                    // have at the entry of this command buffer.
                    let (extern_layout, host, mem) = if !self.is_secondary {
                        (image.0).0.initial_layout(image.1, access.old_layout)
                    } else {
                        (access.old_layout, false, false)
                    };

                    let src_access = {
                        let mut v = 0;
                        if host { v |= vk::ACCESS_HOST_READ_BIT | vk::ACCESS_HOST_WRITE_BIT; }
                        if mem { v |= vk::ACCESS_MEMORY_READ_BIT | vk::ACCESS_MEMORY_WRITE_BIT; }
                        v
                    };

                    if extern_layout != access.old_layout || host || mem {
                        dst_stages |= access.stages;
                        
                        let range_mipmaps = (image.0).0.block_mipmap_levels_range(image.1);
                        let range_layers = (image.0).0.block_array_layers_range(image.1);

                        debug_assert!(!self.is_secondary_graphics);
                        image_barriers.push(vk::ImageMemoryBarrier {
                            sType: vk::STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                            pNext: ptr::null(),
                            srcAccessMask: src_access,
                            dstAccessMask: access.accesses,
                            oldLayout: extern_layout as u32,
                            newLayout: access.old_layout as u32,
                            srcQueueFamilyIndex: vk::QUEUE_FAMILY_IGNORED,
                            dstQueueFamilyIndex: vk::QUEUE_FAMILY_IGNORED,
                            image: (image.0).0.inner_image().internal_object(),
                            subresourceRange: vk::ImageSubresourceRange {
                                aspectMask: access.aspects,
                                baseMipLevel: range_mipmaps.start,
                                levelCount: range_mipmaps.end - range_mipmaps.start,
                                baseArrayLayer: range_layers.start,
                                layerCount: range_layers.end - range_layers.start,
                            },
                        });
                    }

                    entry.insert(InternalImageBlockAccess {
                        stages: access.stages,
                        accesses: access.accesses,
                        write: access.write,
                        aspects: access.aspects,
                        old_layout: extern_layout,
                        new_layout: access.new_layout,
                    });
                },

                Entry::Occupied(mut entry) => {
                    let mut entry = entry.get_mut();

                    // TODO: not always necessary
                    src_stages |= entry.stages;
                    dst_stages |= access.stages;

                    let range_mipmaps = (image.0).0.block_mipmap_levels_range(image.1);
                    let range_layers = (image.0).0.block_array_layers_range(image.1);

                    debug_assert!(!self.is_secondary_graphics);
                    image_barriers.push(vk::ImageMemoryBarrier {
                        sType: vk::STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        pNext: ptr::null(),
                        srcAccessMask: entry.accesses,
                        dstAccessMask: access.accesses,
                        oldLayout: entry.new_layout as u32,
                        newLayout: access.old_layout as u32,
                        srcQueueFamilyIndex: vk::QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex: vk::QUEUE_FAMILY_IGNORED,
                        image: (image.0).0.inner_image().internal_object(),
                        subresourceRange: vk::ImageSubresourceRange {
                            aspectMask: access.aspects,
                            baseMipLevel: range_mipmaps.start,
                            levelCount: range_mipmaps.end - range_mipmaps.start,
                            baseArrayLayer: range_layers.start,
                            layerCount: range_layers.end - range_layers.start,
                        },
                    });

                    // TODO: incomplete
                    entry.stages = access.stages;
                    entry.accesses = access.accesses;
                    entry.new_layout = access.new_layout;
                },
            };
        }

        // Adding the pipeline barrier.
        if !buffer_barriers.is_empty() || !image_barriers.is_empty() {
            let (src_stages, dst_stages) = match (src_stages, dst_stages) {
                (0, 0) => (vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT, vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT),
                (src, 0) => (src, vk::PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT),
                (0, dest) => (dest, dest),
                (src, dest) => (src, dest),
            };

            debug_assert!(src_stages != 0 && dst_stages != 0);

            unsafe {
                vk.CmdPipelineBarrier(cmd, src_stages, dst_stages,
                                      vk::DEPENDENCY_BY_REGION_BIT, 0, ptr::null(),
                                      buffer_barriers.len() as u32, buffer_barriers.as_ptr(),
                                      image_barriers.len() as u32, image_barriers.as_ptr());
            }
        }

        // Now flushing all commands.
        for mut command in self.staging_commands.drain(..) {
            command(&vk, cmd);
        }
    }

    /// Finishes building the command buffer.
    pub fn build(mut self) -> Result<InnerCommandBuffer, OomError> {
        unsafe {
            self.flush_render_pass();
            self.flush(true);

            // Ensuring that each image is in its final layout. We do so by inserting elements
            // in `staging_required_image_accesses`, which lets the system think that we are
            // accessing the images, and then flushing again.
            // TODO: ideally things that don't collide should be added before the first flush,
            //       and things that collide done here
            if !self.is_secondary {
                for (image, access) in self.images_state.iter() {
                    let (final_layout, host, mem) = (image.0).0.final_layout(image.1, access.new_layout);
                    if final_layout != access.new_layout || host || mem {
                        let mut accesses = 0;
                        if host { accesses |= vk::ACCESS_HOST_READ_BIT | vk::ACCESS_HOST_WRITE_BIT; }
                        if mem { accesses |= vk::ACCESS_MEMORY_READ_BIT | vk::ACCESS_MEMORY_WRITE_BIT; }

                        self.staging_required_image_accesses.insert(image.clone(), InternalImageBlockAccess {
                            stages: vk::PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            accesses: accesses,
                            write: false,
                            aspects: access.aspects,
                            old_layout: final_layout,
                            new_layout: final_layout,
                        });
                    }
                }

                // Checking each buffer to see if it must be flushed to the host.
                for (buffer, access) in self.buffers_state.iter() {
                    if !(buffer.0).0.host_accesses(buffer.1) {
                        continue;
                    }

                    self.staging_required_buffer_accesses.insert(buffer.clone(), InternalBufferBlockAccess {
                        stages: vk::PIPELINE_STAGE_HOST_BIT,
                        accesses: vk::ACCESS_HOST_READ_BIT | vk::ACCESS_HOST_WRITE_BIT,
                        write: false,
                    });
                }

                self.flush(true);
            }

            debug_assert!(self.staging_required_buffer_accesses.is_empty());
            debug_assert!(self.staging_required_image_accesses.is_empty());

            let vk = self.device.pointers();
            let _ = self.pool.internal_object_guard();      // the pool needs to be synchronized
            let cmd = self.cmd.take().unwrap();

            // Ending the commands recording.
            try!(check_errors(vk.EndCommandBuffer(cmd)));

            Ok(InnerCommandBuffer {
                device: self.device.clone(),
                pool: self.pool.clone(),
                cmd: cmd,
                buffers_state: self.buffers_state.clone(),      // TODO: meh
                images_state: self.images_state.clone(),        // TODO: meh
                extern_buffers_sync: {
                    let mut map = HashMap::new();
                    for ((buf, bl), access) in self.buffers_state.drain() {
                        let value = BufferAccessRange {
                            block: bl,
                            write: access.write,
                        };

                        match map.entry(buf) {
                            Entry::Vacant(e) => {
                                let mut v = SmallVec::new();
                                v.push(value);
                                e.insert(v);
                            },
                            Entry::Occupied(mut e) => {
                                e.get_mut().push(value);
                            },
                        }
                    }

                    map.into_iter().map(|(buf, val)| (buf.0, val)).collect()
                },
                extern_images_sync: {
                    let mut map = HashMap::new();
                    for ((img, bl), access) in self.images_state.drain() {
                        let value = ImageAccessRange {
                            block: bl,
                            write: access.write,
                            initial_layout: access.old_layout,
                            final_layout: access.new_layout,
                        };

                        match map.entry(img) {
                            Entry::Vacant(e) => {
                                let mut v = SmallVec::new();
                                v.push(value);
                                e.insert(v);
                            },
                            Entry::Occupied(mut e) => {
                                e.get_mut().push(value);
                            },
                        }
                    }

                    map.into_iter().map(|(img, val)| (img.0, val)).collect()
                },
                keep_alive: mem::replace(&mut self.keep_alive, Vec::new()),
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
                vk.EndCommandBuffer(cmd);       // TODO: really needed?

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
    buffers_state: HashMap<(BufferKey, usize), InternalBufferBlockAccess>,
    images_state: HashMap<(ImageKey, (u32, u32)), InternalImageBlockAccess>,
    extern_buffers_sync: SmallVec<[(Arc<Buffer>, SmallVec<[BufferAccessRange; 4]>); 32]>,
    extern_images_sync: SmallVec<[(Arc<Image>, SmallVec<[ImageAccessRange; 8]>); 32]>,
    keep_alive: Vec<Arc<KeepAlive>>,
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
pub fn submit(me: &InnerCommandBuffer, me_arc: Arc<KeepAlive>,
              queue: &Arc<Queue>) -> Result<Arc<Submission>, OomError>   // TODO: wrong error type
{
    let vk = me.device.pointers();

    assert_eq!(queue.device().internal_object(), me.pool.device().internal_object());
    assert_eq!(queue.family().id(), me.pool.queue_family().id());

    let fence = try!(Fence::new(queue.device()));

    let mut keep_alive_semaphores = SmallVec::<[_; 8]>::new();
    let mut post_semaphores_ids = SmallVec::<[_; 8]>::new();
    let mut pre_semaphores_ids = SmallVec::<[_; 8]>::new();
    let mut pre_semaphores_stages = SmallVec::<[_; 8]>::new();

    // Each queue has a dedicated semaphore which must be signalled and waited upon by each
    // command buffer submission.
    // TODO: for now that's not true ^  as semaphores are only used once then destroyed ;
    //       waiting on https://github.com/KhronosGroup/Vulkan-Docs/issues/155
    {
        let signalled = try!(Semaphore::new(queue.device()));
        let wait = unsafe { queue.dedicated_semaphore(signalled.clone()) };
        if let Some(wait) = wait {
            pre_semaphores_ids.push(wait.internal_object());
            pre_semaphores_stages.push(vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT);     // TODO:
            keep_alive_semaphores.push(wait);
        }
        post_semaphores_ids.push(signalled.internal_object());
        keep_alive_semaphores.push(signalled);
    }

    // Creating additional semaphores, one for each queue transition.
    let queue_transitions_hint: u32 = 2;        // TODO: get as function parameter
    // TODO: use a pool
    let semaphores_to_signal = {
        let mut list = SmallVec::new();
        for _ in 0 .. queue_transitions_hint {
            let sem = try!(Semaphore::new(queue.device()));
            post_semaphores_ids.push(sem.internal_object());
            keep_alive_semaphores.push(sem.clone());
            list.push(sem);
        }
        list
    };

    // We can now create the `Submission` object.
    // We need to create it early because we pass it when calling `gpu_access`.
    let submission = Arc::new(Submission {
        fence: fence.clone(),
        queue: queue.clone(),
        guarded: Mutex::new(SubmissionGuarded {
            signalled_semaphores: semaphores_to_signal,
            signalled_queues: SmallVec::new(),
        }),
        keep_alive_cb: Mutex::new({ let mut v = SmallVec::new(); v.push(me_arc); v }),
        keep_alive_semaphores: Mutex::new(SmallVec::new()),
    });

    // List of command buffers to submit before and after the main one.
    let mut before_command_buffers: SmallVec<[_; 4]> = SmallVec::new();
    let mut after_command_buffers: SmallVec<[_; 4]> = SmallVec::new();

    {
        // There is a possibility that a parallel thread is currently submitting a command buffer to
        // another queue. Once we start invoking the `gpu_access` access functions, there is a
        // possibility the other thread detects that it depends on this one and submits its command
        // buffer before us.
        //
        // Since we need to access the content of `guarded` in our dependencies, we lock our own
        // `guarded` here to avoid being used as a dependency before being submitted.
        // TODO: this needs a new block (https://github.com/rust-lang/rfcs/issues/811)
        let _submission_lock = submission.guarded.lock().unwrap();

        // Now we determine which earlier submissions we must depend upon.
        let mut dependencies = SmallVec::<[Arc<Submission>; 6]>::new();

        // Buffers first.
        for &(ref resource, ref ranges) in me.extern_buffers_sync.iter() {
            let result = unsafe { resource.gpu_access(&mut ranges.iter().cloned(), &submission) };
            if let Some(semaphore) = result.additional_wait_semaphore {
                pre_semaphores_ids.push(semaphore.internal_object());
                pre_semaphores_stages.push(vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT);     // TODO:
                keep_alive_semaphores.push(semaphore);
            }

            if let Some(semaphore) = result.additional_signal_semaphore {
                post_semaphores_ids.push(semaphore.internal_object());
                keep_alive_semaphores.push(semaphore);
            }

            dependencies.extend(result.dependencies.into_iter());
        }

        // Then images.
        for &(ref resource, ref ranges) in me.extern_images_sync.iter() {
            let result = unsafe { resource.gpu_access(&mut ranges.iter().cloned(), &submission) };

            if let Some(semaphore) = result.additional_wait_semaphore {
                pre_semaphores_ids.push(semaphore.internal_object());
                pre_semaphores_stages.push(vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT);     // TODO:
                keep_alive_semaphores.push(semaphore);
            }

            if let Some(semaphore) = result.additional_signal_semaphore {
                post_semaphores_ids.push(semaphore.internal_object());
                keep_alive_semaphores.push(semaphore);
            }

            for transition in result.before_transitions {
                let cb = transition_cb(&me.pool, resource.clone(), transition.block, transition.from, transition.to).unwrap();
                before_command_buffers.push(cb.cmd);
                submission.keep_alive_cb.lock().unwrap().push(Arc::new(cb));
            }

            for transition in result.after_transitions {
                let cb = transition_cb(&me.pool, resource.clone(), transition.block, transition.from, transition.to).unwrap();
                after_command_buffers.push(cb.cmd);
                submission.keep_alive_cb.lock().unwrap().push(Arc::new(cb));
            }

            dependencies.extend(result.dependencies.into_iter());
        }

        // For each dependency, we either wait on one of its semaphores, or create a new one.
        for dependency in dependencies.iter() {
            let current_queue_id = (queue.family().id(), queue.id_within_family());

            // If we submit to the same queue as your dependency, no need to worry about this.
            if current_queue_id == (dependency.queue.family().id(),
                                    dependency.queue.id_within_family())
            {
                continue;
            }

            let mut guard = dependency.guarded.lock().unwrap();

            // If the current queue is in the list of already-signalled queue of the dependency, we
            // ignore it.
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
            keep_alive_semaphores.push(semaphore);

            // Note that it may look dangerous to unlock the dependency's mutex here, because the
            // queue has already been added to the list of signalled queues but the command that
            // signals the semaphore hasn't been sent yet.
            //
            // However submitting to a queue must lock the queue, which guarantees that no other
            // parallel queue submission should happen on this same queue. This means that the problem
            // is non-existing.
        }

        unsafe {
            let mut infos = SmallVec::<[_; 3]>::new();

            if !before_command_buffers.is_empty() {
                let semaphore = Semaphore::new(queue.device()).unwrap();
                let semaphore_id = semaphore.internal_object();
                pre_semaphores_stages.push(vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT);     // TODO:
                pre_semaphores_ids.push(semaphore.internal_object());
                keep_alive_semaphores.push(semaphore);

                infos.push(vk::SubmitInfo {
                    sType: vk::STRUCTURE_TYPE_SUBMIT_INFO,
                    pNext: ptr::null(),
                    waitSemaphoreCount: 0,
                    pWaitSemaphores: ptr::null(),
                    pWaitDstStageMask: ptr::null(),
                    commandBufferCount: before_command_buffers.len() as u32,
                    pCommandBuffers: before_command_buffers.as_ptr(),
                    signalSemaphoreCount: 1,
                    pSignalSemaphores: &semaphore_id,
                });
            }

            let after_semaphore = if !after_command_buffers.is_empty() {
                let semaphore = Semaphore::new(queue.device()).unwrap();
                let semaphore_id = semaphore.internal_object();
                post_semaphores_ids.push(semaphore.internal_object());
                keep_alive_semaphores.push(semaphore);
                semaphore_id
            } else {
                0
            };

            debug_assert_eq!(pre_semaphores_ids.len(), pre_semaphores_stages.len());
            infos.push(vk::SubmitInfo {
                sType: vk::STRUCTURE_TYPE_SUBMIT_INFO,
                pNext: ptr::null(),
                waitSemaphoreCount: pre_semaphores_ids.len() as u32,
                pWaitSemaphores: pre_semaphores_ids.as_ptr(),
                pWaitDstStageMask: pre_semaphores_stages.as_ptr(),
                commandBufferCount: 1,
                pCommandBuffers: &me.cmd,
                signalSemaphoreCount: if after_command_buffers.is_empty() { post_semaphores_ids.len() as u32 } else { 1 },
                pSignalSemaphores: if after_command_buffers.is_empty() { post_semaphores_ids.as_ptr() } else { &after_semaphore },
            });

            if !after_command_buffers.is_empty() {
                let stage = vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT;     // TODO:
                infos.push(vk::SubmitInfo {
                    sType: vk::STRUCTURE_TYPE_SUBMIT_INFO,
                    pNext: ptr::null(),
                    waitSemaphoreCount: 1,
                    pWaitSemaphores: &after_semaphore,
                    pWaitDstStageMask: &stage,
                    commandBufferCount: after_command_buffers.len() as u32,
                    pCommandBuffers: after_command_buffers.as_ptr(),
                    signalSemaphoreCount: post_semaphores_ids.len() as u32,
                    pSignalSemaphores: post_semaphores_ids.as_ptr(),
                });
            }

            let fence = fence.internal_object();
            try!(check_errors(vk.QueueSubmit(*queue.internal_object_guard(), infos.len() as u32,
                                             infos.as_ptr(), fence)));
        }

        // Don't forget to add all the semaphores in the list of semaphores that must be kept alive.
        {
            let mut ka_sem = submission.keep_alive_semaphores.lock().unwrap();
            *ka_sem = keep_alive_semaphores;
        }

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
    fence: Arc<Fence>,

    // The queue on which this was submitted.
    queue: Arc<Queue>,

    // Additional variables that are behind a mutex.
    guarded: Mutex<SubmissionGuarded>,

    // The command buffers that were submitted and that needs to be kept alive until the submission
    // is complete by the GPU.
    //
    // The fact that it is behind a `Mutex` is a hack. The list of CBs can only be known
    // after the `Submission` has been created and put in an `Arc`. Therefore we need a `Mutex`
    // in order to write this list. The list isn't accessed anymore afterwards, so it shouldn't
    // slow things down too much.
    // TODO: consider an UnsafeCell
    keep_alive_cb: Mutex<SmallVec<[Arc<KeepAlive>; 8]>>,

    // List of semaphores to keep alive while the submission hasn't finished execution.
    //
    // The fact that it is behind a `Mutex` is a hack. See `keep_alive_cb`.
    // TODO: consider an UnsafeCell
    keep_alive_semaphores: Mutex<SmallVec<[Arc<Semaphore>; 8]>>,
}

impl fmt::Debug for Submission {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Submission object>")      // TODO: better description
    }
}

#[derive(Debug)]
struct SubmissionGuarded {
    // Reserve of semaphores that have been signalled by this submission and that can be
    // waited upon. The semaphore must be removed from the list if it is going to be waiting upon.
    signalled_semaphores: SmallVec<[Arc<Semaphore>; 4]>,

    // Queue familiy index and queue index of each queue that got submitted a command buffer
    // that was waiting on this submission to be complete.
    //
    // If a queue is in this list, that means that all side-effects of this submission are going
    // to be visible to any further command buffer submitted to this queue. Note that this is only
    // true due to the fact that we have a per-queue semaphore (see `Queue::dedicated_semaphore`).
    signalled_queues: SmallVec<[(u32, u32); 4]>,
}

impl Submission {
    /// Returns `true` if destroying this `Submission` object would block the CPU for some time.
    #[inline]
    pub fn destroying_would_block(&self) -> bool {
        !self.finished()
    }

    /// Returns `true` if the GPU has finished executing this submission.
    #[inline]
    pub fn finished(&self) -> bool {
        self.fence.ready().unwrap_or(false)     // TODO: what to do in case of error?   
    }

    /// Waits until the submission has finished being executed by the device.
    #[inline]
    pub fn wait(&self, timeout_ns: u64) -> Result<(), FenceWaitError> {
        self.fence.wait(timeout_ns)
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
        match self.fence.wait(u64::MAX) {
            Ok(_) => (),
            Err(FenceWaitError::DeviceLostError) => (),
            Err(FenceWaitError::Timeout) => panic!(),       // The driver has some sort of problem.
            Err(FenceWaitError::OomError(_)) => panic!(),   // What else to do here?
        }

        // TODO: return `signalled_semaphores` to the semaphore pools
    }
}

pub trait KeepAlive: 'static + Send + Sync {}
impl<T> KeepAlive for T where T: 'static + Send + Sync {}

#[derive(Clone)]
struct BufferKey(Arc<Buffer>);

impl PartialEq for BufferKey {
    #[inline]
    fn eq(&self, other: &BufferKey) -> bool {
        &*self.0 as *const Buffer == &*other.0 as *const Buffer
    }
}

impl Eq for BufferKey {}

impl hash::Hash for BufferKey {
    #[inline]
    fn hash<H>(&self, state: &mut H) where H: hash::Hasher {
        let ptr = &*self.0 as *const Buffer as *const () as usize;
        hash::Hash::hash(&ptr, state)
    }
}

#[derive(Copy, Clone, Debug)]
struct InternalBufferBlockAccess {
    // Stages in which the resource is used.
    // Note that this field can have different semantics depending on where this struct is used.
    // For example it can be the stages since the latest barrier instead of just the stages.
    stages: vk::PipelineStageFlagBits,

    // The way this resource is accessed.
    // Just like `stages`, this has different semantics depending on the usage of this struct.
    accesses: vk::AccessFlagBits,

    write: bool,
}

#[derive(Clone)]
struct ImageKey(Arc<Image>);

impl PartialEq for ImageKey {
    #[inline]
    fn eq(&self, other: &ImageKey) -> bool {
        &*self.0 as *const Image == &*other.0 as *const Image
    }
}

impl Eq for ImageKey {}

impl hash::Hash for ImageKey {
    #[inline]
    fn hash<H>(&self, state: &mut H) where H: hash::Hasher {
        let ptr = &*self.0 as *const Image as *const () as usize;
        hash::Hash::hash(&ptr, state)
    }
}

#[derive(Copy, Clone, Debug)]
struct InternalImageBlockAccess {
    // Stages in which the resource is used.
    // Note that this field can have different semantics depending on where this struct is used.
    // For example it can be the stages since the latest barrier instead of just the stages.
    stages: vk::PipelineStageFlagBits,

    // The way this resource is accessed.
    // Just like `stages`, this has different semantics depending on the usage of this struct.
    accesses: vk::AccessFlagBits,

    write: bool,
    aspects: vk::ImageAspectFlags,
    old_layout: ImageLayout,
    new_layout: ImageLayout,
}

/// Builds an `InnerCommandBuffer` whose only purpose is to transition an image between two
/// layouts.
fn transition_cb(pool: &Arc<CommandBufferPool>, image: Arc<Image>, block: (u32, u32),
                 old_layout: ImageLayout, new_layout: ImageLayout)
                 -> Result<InnerCommandBuffer, OomError>
{
    let device = pool.device();
    let vk = device.pointers();
    let pool_obj = pool.internal_object_guard();

    let cmd = unsafe {
        let infos = vk::CommandBufferAllocateInfo {
            sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: ptr::null(),
            commandPool: *pool_obj,
            level: vk::COMMAND_BUFFER_LEVEL_PRIMARY,
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
            flags: vk::COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo: ptr::null(),
        };

        // TODO: leak if this returns an err
        try!(check_errors(vk.BeginCommandBuffer(cmd, &infos)));

        let range_mipmaps = image.block_mipmap_levels_range(block);
        let range_layers = image.block_array_layers_range(block);
        let aspect_mask = match image.format().ty() {
            FormatTy::Float | FormatTy::Uint | FormatTy::Sint | FormatTy::Compressed => {
                vk::IMAGE_ASPECT_COLOR_BIT
            },
            FormatTy::Depth => vk::IMAGE_ASPECT_DEPTH_BIT,
            FormatTy::Stencil => vk::IMAGE_ASPECT_STENCIL_BIT,
            FormatTy::DepthStencil => vk::IMAGE_ASPECT_DEPTH_BIT | vk::IMAGE_ASPECT_STENCIL_BIT,
        };

        let barrier = vk::ImageMemoryBarrier {
            sType: vk::STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            pNext: ptr::null(),
            srcAccessMask: 0,      // TODO: ?
            dstAccessMask: 0x0001ffff,      // TODO: ?
            oldLayout: old_layout as u32,
            newLayout: new_layout as u32,
            srcQueueFamilyIndex: vk::QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex: vk::QUEUE_FAMILY_IGNORED,
            image: image.inner_image().internal_object(),
            subresourceRange: vk::ImageSubresourceRange {
                aspectMask: aspect_mask,
                baseMipLevel: range_mipmaps.start,
                levelCount: range_mipmaps.end - range_mipmaps.start,
                baseArrayLayer: range_layers.start,
                layerCount: range_layers.end - range_layers.start,
            },
        };

        vk.CmdPipelineBarrier(cmd, vk::PIPELINE_STAGE_ALL_COMMANDS_BIT,
                              vk::PIPELINE_STAGE_ALL_COMMANDS_BIT, vk::DEPENDENCY_BY_REGION_BIT,
                              0, ptr::null(), 0, ptr::null(), 1, &barrier);

        // TODO: leak if this returns an err
        try!(check_errors(vk.EndCommandBuffer(cmd)));
    }

    Ok(InnerCommandBuffer {
        device: device.clone(),
        pool: pool.clone(),
        cmd: cmd,
        buffers_state: HashMap::new(),
        images_state: HashMap::new(),
        extern_buffers_sync: SmallVec::new(),
        extern_images_sync: SmallVec::new(),
        keep_alive: Vec::new(),
    })
}
