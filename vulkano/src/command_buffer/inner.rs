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
use std::hash::BuildHasherDefault;
use std::mem;
use std::ops::Range;
use std::ptr;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use std::u64;
use fnv::FnvHasher;
use smallvec::SmallVec;

use buffer::Buffer;
use buffer::BufferSlice;
use buffer::TypedBuffer;
use buffer::traits::AccessRange as BufferAccessRange;
use command_buffer::CommandBuffer;
use command_buffer::DrawIndirectCommand;
use command_buffer::DynamicState;
use command_buffer::pool::CommandPool;
use command_buffer::pool::CommandPoolFinished;
use command_buffer::pool::StandardCommandPool;
use command_buffer::sys::BufferCopyCommand;
use command_buffer::sys::BufferCopyError;
use command_buffer::sys::BufferCopyRegion;
use command_buffer::sys::DispatchCommand;
use command_buffer::sys::DrawCommand;
use command_buffer::sys::DrawTy;
use command_buffer::sys::ExecuteCommand;
use command_buffer::sys::BufferFillCommand;
use command_buffer::sys::BufferFillError;
use command_buffer::sys::PipelineBarrierCommand;
use command_buffer::sys::BeginRenderPassCommand;
use command_buffer::sys::NextSubpassCommand;
use command_buffer::sys::EndRenderPassCommand;
use command_buffer::sys::BufferUpdateCommand;
use command_buffer::sys::SubmissionDesc;
use command_buffer::sys::Kind;
use command_buffer::sys::Flags;
use command_buffer::sys::UnsafeCommandBuffer;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use command_buffer::sys::UnsafeSubmission;
use descriptor::descriptor_set::DescriptorSetsCollection;
use descriptor::PipelineLayout;
use device::Queue;
use format::ClearValue;
use format::FormatTy;
use format::PossibleFloatFormatDesc;
use framebuffer::RenderPass;
use framebuffer::Framebuffer;
use framebuffer::Subpass;
use image::Image;
use image::sys::Layout as ImageLayout;
use image::traits::ImageClearValue;
use image::traits::ImageContent;
use image::traits::AccessRange as ImageAccessRange;
use pipeline::ComputePipeline;
use pipeline::GraphicsPipeline;
use pipeline::input_assembly::Index;
use pipeline::vertex::Source as VertexSource;
use sync::AccessFlagBits;
use sync::Fence;
use sync::FenceWaitError;
use sync::PipelineStages;
use sync::Semaphore;

use device::Device;
use OomError;
use SynchronizedVulkanObject;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

// TODO: that sucks but we have to lock everything when submitting a command buffer to a queue
//       this is because of synchronization issues when querying resources for their dependencies
lazy_static! {
    static ref GLOBAL_MUTEX: Mutex<()> = Mutex::new(());
}

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
pub struct InnerCommandBufferBuilder<P> where P: CommandPool {
    // The command buffer. It is an option because it is temporarily moved out from time to time.
    // If it contains `None`, that indicates an earlier panic.
    cmd: Option<UnsafeCommandBufferBuilder<P>>,

    // List of accesses made by this command buffer to buffers and images, exclusing the staging
    // commands and the staging render pass.
    //
    // If a buffer/image is missing in this list, that means it hasn't been used by this command
    // buffer yet and is still in its default state.
    //
    // This list is only updated by the `flush()` function.
    buffers_state: HashMap<(BufferKey, usize), InternalBufferBlockAccess, BuildHasherDefault<FnvHasher>>,
    images_state: HashMap<(ImageKey, (u32, u32)), InternalImageBlockAccess, BuildHasherDefault<FnvHasher>>,

    // List of commands that are waiting to be submitted to the Vulkan command buffer. Doesn't
    // include commands that were submitted within a render pass.
    staging_commands: Vec<Box<Fn(UnsafeCommandBufferBuilder<P>) -> UnsafeCommandBufferBuilder<P> + Send + Sync>>,

    // List of resources accesses made by the comands in `staging_commands`. Doesn't include
    // commands added to the current render pass.
    staging_required_buffer_accesses: HashMap<(BufferKey, usize), InternalBufferBlockAccess, BuildHasherDefault<FnvHasher>>,
    staging_required_image_accesses: HashMap<(ImageKey, (u32, u32)), InternalImageBlockAccess, BuildHasherDefault<FnvHasher>>,

    // List of commands that are waiting to be submitted to the Vulkan command buffer when we're
    // inside a render pass. Flushed when `end_renderpass` is called.
    render_pass_staging_commands: Vec<Box<Fn(UnsafeCommandBufferBuilder<P>) -> UnsafeCommandBufferBuilder<P> + Send + Sync>>,

    // List of resources accesses made by the current render pass. Merged with
    // `staging_required_buffer_accesses` and `staging_required_image_accesses` when
    // `end_renderpass` is called.
    render_pass_staging_required_buffer_accesses: HashMap<(BufferKey, usize), InternalBufferBlockAccess, BuildHasherDefault<FnvHasher>>,
    render_pass_staging_required_image_accesses: HashMap<(ImageKey, (u32, u32)), InternalImageBlockAccess, BuildHasherDefault<FnvHasher>>,
}

impl<P> InnerCommandBufferBuilder<P> where P: CommandPool {
    /// Creates a new builder.
    pub fn new<R>(pool: P, secondary: bool, secondary_cont: Option<Subpass<R>>,
                  secondary_cont_fb: Option<&Arc<Framebuffer<R>>>)
                  -> Result<InnerCommandBufferBuilder<P>, OomError>
        where R: RenderPass + 'static + Send + Sync
    {
        let device = pool.device().clone();
        let vk = device.pointers();

        let cmd = try!(UnsafeCommandBufferBuilder::new(pool, ));

        Ok(InnerCommandBufferBuilder {
            cmd: Some(cmd),
            buffers_state: HashMap::with_hasher(BuildHasherDefault::<FnvHasher>::default()),
            images_state: HashMap::with_hasher(BuildHasherDefault::<FnvHasher>::default()),
            staging_commands: Vec::new(),
            staging_required_buffer_accesses: HashMap::with_hasher(BuildHasherDefault::<FnvHasher>::default()),
            staging_required_image_accesses: HashMap::with_hasher(BuildHasherDefault::<FnvHasher>::default()),
            render_pass_staging_commands: Vec::new(),
            render_pass_staging_required_buffer_accesses: HashMap::with_hasher(BuildHasherDefault::<FnvHasher>::default()),
            render_pass_staging_required_image_accesses: HashMap::with_hasher(BuildHasherDefault::<FnvHasher>::default()),
        })
    }

    /// Executes the content of another command buffer.
    ///
    /// # Safety
    ///
    /// Care must be taken to respect the rules about secondary command buffers.
    pub unsafe fn execute_commands<'a, S, Tmp>(mut self, cb_arc: Arc<Tmp>,
                                          cb: &InnerCommandBuffer<S>)
                                          -> InnerCommandBufferBuilder<P>
        where S: CommandPool
    {
        /*debug_assert!(cb.is_secondary);
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

        self*/
        unimplemented!()
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
                                              -> InnerCommandBufferBuilder<P>
        where B: Into<BufferSlice<'a, T, Bt>>, Bt: Buffer + 'static, T: Clone + Send + Sync + 'static
    {
        debug_assert!(self.render_pass_staging_commands.is_empty());

        let buffer = buffer.into();
        let prototype = BufferUpdateCommand::new(buffer, data).unwrap();

        self.add_buffer_resource_outside(buffer.buffer().clone() as Arc<_>, true,
                                         buffer.offset() .. buffer.offset() + buffer.size(),
                                         PipelineStages::transfer(),
                                         AccessFlagBits { transfer_write: true,
                                                          .. AccessFlagBits::none() });

        self.staging_commands.push(Box::new(move |cb| prototype.submit(cb)));
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
    pub unsafe fn fill_buffer<'a, S, T: ?Sized, B>(mut self, buffer: S, data: u32)
                                                  -> InnerCommandBufferBuilder<P>
        where S: Into<BufferSlice<'a, T, B>>,
              B: Buffer + Send + Sync + 'static
    {
        debug_assert!(self.render_pass_staging_commands.is_empty());

        let buffer = buffer.into();

        let prototype = BufferFillCommand::untyped(buffer, data).unwrap();

        self.add_buffer_resource_outside(buffer.buffer().clone() as Arc<_>, true,
                                         buffer.offset() .. buffer.offset() + buffer.size(),
                                         PipelineStages::transfer(),
                                         AccessFlagBits { transfer_write: true,
                                                          .. AccessFlagBits::none() });

        self.staging_commands.push(Box::new(move |cb| prototype.submit(cb)));
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
                                                           -> InnerCommandBufferBuilder<P>
        where Bs: TypedBuffer<Content = T> + 'static, Bd: TypedBuffer<Content = T> + 'static
    {
        debug_assert!(self.render_pass_staging_commands.is_empty());

        let prototype = BufferCopyCommand::new(source, destination, Some(BufferCopyRegion {
            source_offset: 0,
            destination_offset: 0,
            size: source.size(),
        })).unwrap();

        self.add_buffer_resource_outside(source.clone() as Arc<_>, false, 0 .. source.size(),
                                         PipelineStages::transfer(),
                                         AccessFlagBits { transfer_write: true,
                                                          .. AccessFlagBits::none() });
        self.add_buffer_resource_outside(destination.clone() as Arc<_>, true, 0 .. source.size(),
                                         PipelineStages::transfer(),
                                         AccessFlagBits { transfer_write: true,
                                                          .. AccessFlagBits::none() });

        self.staging_commands.push(Box::new(move |cb| prototype.submit(cb)));
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
                                              -> InnerCommandBufferBuilder<P>
        where I: ImageClearValue<V> + 'static   // FIXME: should accept uint and int images too
    {
        /*debug_assert!(self.render_pass_staging_commands.is_empty());

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

        self*/
        unimplemented!()
    }

    /// Copies data from a buffer to a color image.
    ///
    /// This operation can be performed by any kind of queue.
    ///
    /// # Safety
    ///
    /// - Care must be taken to respect the rules about secondary command buffers.
    ///
    pub unsafe fn copy_buffer_to_color_image<'a, Pi, S, Sb, Img>(mut self, source: S, image: &Arc<Img>,
                                                                mip_level: u32, array_layers_range: Range<u32>,
                                                                offset: [u32; 3], extent: [u32; 3])
                                                             -> InnerCommandBufferBuilder<P>
        where S: Into<BufferSlice<'a, [Pi], Sb>>, Img: ImageContent<Pi> + Image + 'static,
              Sb: Buffer + 'static
    {
        // FIXME: check the parameters

        /*debug_assert!(self.render_pass_staging_commands.is_empty());

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

        self*/
        unimplemented!()
    }

    /// Copies data from a color image to a buffer.
    ///
    /// This operation can be performed by any kind of queue.
    ///
    /// # Safety
    ///
    /// - Care must be taken to respect the rules about secondary command buffers.
    ///
    pub unsafe fn copy_color_image_to_buffer<'a, Pi, S, Sb, Img>(mut self, dest: S, image: &Arc<Img>,
                                                                mip_level: u32, array_layers_range: Range<u32>,
                                                                offset: [u32; 3], extent: [u32; 3])
                                                             -> InnerCommandBufferBuilder<P>
        where S: Into<BufferSlice<'a, [Pi], Sb>>, Img: ImageContent<Pi> + Image + 'static,
              Sb: Buffer + 'static
    {
        // FIXME: check the parameters

        /*debug_assert!(self.render_pass_staging_commands.is_empty());

        //assert!(image.format().is_float_or_compressed());

        let dest = dest.into();
        self.add_buffer_resource_outside(dest.buffer().clone() as Arc<_>, true,
                                         dest.offset() .. dest.offset() + dest.size(),
                                         vk::PIPELINE_STAGE_TRANSFER_BIT,
                                         vk::ACCESS_TRANSFER_WRITE_BIT);
        self.add_image_resource_outside(image.clone() as Arc<_>, mip_level .. mip_level + 1,
                                        array_layers_range.clone(), false,
                                        ImageLayout::TransferSrcOptimal,
                                        vk::PIPELINE_STAGE_TRANSFER_BIT,
                                        vk::ACCESS_TRANSFER_READ_BIT);

        {
            let dest_offset = dest.offset() as vk::DeviceSize;
            let dest = dest.buffer().inner_buffer().internal_object();
            let image = image.inner_image().internal_object();

            self.staging_commands.push(Box::new(move |vk, cmd| {
                let region = vk::BufferImageCopy {
                    bufferOffset: dest_offset,
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

                vk.CmdCopyImageToBuffer(cmd, image,
                                        vk::IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL /* FIXME */,
                                        dest, 1, &region);
            }));
        }

        self*/
        unimplemented!()
    }

    pub unsafe fn blit<Si, Di>(mut self, source: &Arc<Si>, source_mip_level: u32,
                               source_array_layers: Range<u32>, src_coords: [Range<i32>; 3],
                               destination: &Arc<Di>, dest_mip_level: u32,
                               dest_array_layers: Range<u32>, dest_coords: [Range<i32>; 3])
                               -> InnerCommandBufferBuilder<P>
        where Si: Image + 'static, Di: Image + 'static
    {
        /*// FIXME: check the parameters

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

        self*/
        unimplemented!()
    }

    pub unsafe fn dispatch<Pl, L, Pc>(mut self, pipeline: &Arc<ComputePipeline<Pl>>, sets: L,
                                  dimensions: [u32; 3], push_constants: &Pc) -> InnerCommandBufferBuilder<P>
        where L: DescriptorSetsCollection + Send + Sync,
              Pl: 'static + PipelineLayout + Send + Sync,
              Pc: 'static + Clone + Send + Sync
    {
        /*debug_assert!(self.render_pass_staging_commands.is_empty());

        self.bind_compute_pipeline_state(pipeline, sets, push_constants);

        self.staging_commands.push(Box::new(move |vk, cmd| {
            vk.CmdDispatch(cmd, dimensions[0], dimensions[1], dimensions[2]);
        }));

        self*/
        unimplemented!()
    }

    /// Calls `vkCmdDraw`.
    // FIXME: push constants
    pub unsafe fn draw<V, Pv, Pl, L, Rp, Pc>(mut self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Rp>>,
                             vertices: V, dynamic: &DynamicState,
                             sets: L, push_constants: &Pc) -> InnerCommandBufferBuilder<P>
        where Pv: 'static + VertexSource<V>, L: DescriptorSetsCollection + Send + Sync,
              Pl: 'static + PipelineLayout + Send + Sync, Rp: 'static + Send + Sync,
              Pc: 'static + Clone + Send + Sync
    {
        /*// FIXME: add buffers to the resources

        self.bind_gfx_pipeline_state(pipeline, dynamic, sets, push_constants);

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

        self*/
        unimplemented!()
    }

    /// Calls `vkCmdDrawIndexed`.
    // FIXME: push constants
    pub unsafe fn draw_indexed<'a, V, Pv, Pl, Rp, L, I, Ib, Ibb, Pc>(mut self, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Rp>>,
                                                          vertices: V, indices: Ib, dynamic: &DynamicState,
                                                          sets: L, push_constants: &Pc) -> InnerCommandBufferBuilder<P>
        where L: DescriptorSetsCollection + Send + Sync,
              Pv: 'static + VertexSource<V>,
              Pl: 'static + PipelineLayout + Send + Sync, Rp: 'static + Send + Sync,
              Ib: Into<BufferSlice<'a, [I], Ibb>>, I: 'static + Index, Ibb: Buffer + 'static,
              Pc: 'static + Clone + Send + Sync
    {
        /*// FIXME: add buffers to the resources

        self.bind_gfx_pipeline_state(pipeline, dynamic, sets, push_constants);


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

        self*/
        unimplemented!()
    }

    /// Calls `vkCmdDrawIndirect`.
    // FIXME: push constants
    pub unsafe fn draw_indirect<I, V, Pv, Pl, L, Rp, Pc>(mut self, buffer: &Arc<I>, pipeline: &Arc<GraphicsPipeline<Pv, Pl, Rp>>,
                             vertices: V, dynamic: &DynamicState,
                             sets: L, push_constants: &Pc) -> InnerCommandBufferBuilder<P>
        where Pv: 'static + VertexSource<V>, L: DescriptorSetsCollection + Send + Sync,
              Pl: 'static + PipelineLayout + Send + Sync, Rp: 'static + Send + Sync, Pc: 'static + Clone + Send + Sync,
              I: 'static + TypedBuffer<Content = [DrawIndirectCommand]>
    {
        // FIXME: add buffers to the resources

       /* self.bind_gfx_pipeline_state(pipeline, dynamic, sets, push_constants);

        let vertices = pipeline.vertex_definition().decode(vertices);

        let offsets = (0 .. vertices.0.len()).map(|_| 0).collect::<SmallVec<[_; 8]>>();
        let ids = vertices.0.map(|b| {
            assert!(b.inner_buffer().usage_vertex_buffer());
            self.add_buffer_resource_inside(b.clone(), false, 0 .. b.size(),
                                            vk::PIPELINE_STAGE_VERTEX_INPUT_BIT,
                                            vk::ACCESS_VERTEX_ATTRIBUTE_READ_BIT);
            b.inner_buffer().internal_object()
        }).collect::<SmallVec<[_; 8]>>();

        self.add_buffer_resource_inside(buffer.clone(), false, 0 .. buffer.size(),
                                        vk::PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                        vk::ACCESS_INDIRECT_COMMAND_READ_BIT);

        {
            let mut ids = Some(ids);
            let mut offsets = Some(offsets);
            let buffer_internal = buffer.inner_buffer().internal_object();
            let buffer_draw_count = buffer.len() as u32;
            let buffer_size = buffer.size() as u32;

            self.render_pass_staging_commands.push(Box::new(move |vk, cmd| {
                let ids = ids.take().unwrap();
                let offsets = offsets.take().unwrap();

                vk.CmdBindVertexBuffers(cmd, 0, ids.len() as u32, ids.as_ptr(), offsets.as_ptr());
                vk.CmdDrawIndirect(cmd, buffer_internal, 0, buffer_draw_count,
                                   mem::size_of::<DrawIndirectCommand>() as u32);
            }));
        }

        self*/
        unimplemented!()
    }

    /*fn bind_compute_pipeline_state<Pl, L, Pc>(&mut self, pipeline: &Arc<ComputePipeline<Pl>>, sets: L,
                                          push_constants: &Pc)
        where L: DescriptorSetsCollection,
              Pl: 'static + PipelineLayout + Send + Sync,
              Pc: 'static + Clone + Send + Sync
    {
        unsafe {
            //assert!(sets.is_compatible_with(pipeline.layout()));

            if self.current_compute_pipeline != Some(pipeline.internal_object()) {
                self.keep_alive.push(pipeline.clone());
                let pipeline = pipeline.internal_object();
                self.staging_commands.push(Box::new(move |vk, cmd| {
                    vk.CmdBindPipeline(cmd, vk::PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline);
                }));
                self.current_compute_pipeline = Some(pipeline);
            }

            let mut descriptor_sets = DescriptorSetsCollection::list(&sets).collect::<SmallVec<[_; 32]>>();

            for set in descriptor_sets.iter() {
                for &(ref img, block, layout) in set.inner_descriptor_set().images_list().iter() {
                    self.add_image_resource_outside(img.clone(), 0 .. 1 /* FIXME */, 0 .. 1 /* FIXME */,
                                                   false, layout, vk::PIPELINE_STAGE_ALL_COMMANDS_BIT /* FIXME */,
                                                   vk::ACCESS_SHADER_READ_BIT | vk::ACCESS_UNIFORM_READ_BIT /* TODO */);
                }
                for buffer in set.inner_descriptor_set().buffers_list().iter() {
                    self.add_buffer_resource_outside(buffer.clone(), false, 0 .. buffer.size() /* TODO */,
                                                    vk::PIPELINE_STAGE_ALL_COMMANDS_BIT /* FIXME */,
                                                    vk::ACCESS_SHADER_READ_BIT | vk::ACCESS_UNIFORM_READ_BIT /* TODO */);
                }
            }

            for d in descriptor_sets.iter() { self.keep_alive.push(mem::transmute(d.clone()) /* FIXME: */); }
            let mut descriptor_sets = Some(descriptor_sets.into_iter().map(|set| set.inner_descriptor_set().internal_object()).collect::<SmallVec<[_; 32]>>());

            // TODO: shouldn't rebind everything every time
            if !descriptor_sets.as_ref().unwrap().is_empty() {
                let pipeline = PipelineLayout::inner_pipeline_layout(&**pipeline.layout()).internal_object();
                self.staging_commands.push(Box::new(move |vk, cmd| {
                    let descriptor_sets = descriptor_sets.take().unwrap();
                    vk.CmdBindDescriptorSets(cmd, vk::PIPELINE_BIND_POINT_COMPUTE,
                                             pipeline, 0, descriptor_sets.len() as u32,
                                             descriptor_sets.as_ptr(), 0, ptr::null());   // FIXME: dynamic offsets
                }));
            }

            if mem::size_of_val(push_constants) >= 1 {
                let pipeline = PipelineLayout::inner_pipeline_layout(&**pipeline.layout()).internal_object();
                let size = mem::size_of_val(push_constants);
                let push_constants = push_constants.clone();
                assert!((size % 4) == 0);

                self.staging_commands.push(Box::new(move |vk, cmd| {
                    vk.CmdPushConstants(cmd, pipeline, 0x7fffffff, 0, size as u32,
                                        &push_constants as *const Pc as *const _);
                }));
            }
        }
    }

    fn bind_gfx_pipeline_state<V, Pl, L, Rp, Pc>(&mut self, pipeline: &Arc<GraphicsPipeline<V, Pl, Rp>>,
                                                 dynamic: &DynamicState, sets: L, push_constants: &Pc)
        where V: 'static + Send + Sync, L: DescriptorSetsCollection + Send + Sync,
              Pl: 'static + PipelineLayout + Send + Sync, Rp: 'static + Send + Sync, Pc: 'static + Clone + Send + Sync
    {
        unsafe {
            //assert!(sets.is_compatible_with(pipeline.layout()));

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

            let mut descriptor_sets = DescriptorSetsCollection::list(&sets).collect::<SmallVec<[_; 32]>>();
            for set in descriptor_sets.iter() {
                for &(ref img, block, layout) in set.inner_descriptor_set().images_list().iter() {
                    self.add_image_resource_inside(img.clone(), 0 .. 1 /* FIXME */, 0 .. 1 /* FIXME */,
                                                   false, layout, layout, vk::PIPELINE_STAGE_ALL_COMMANDS_BIT /* FIXME */,
                                                   vk::ACCESS_SHADER_READ_BIT | vk::ACCESS_UNIFORM_READ_BIT /* TODO */);
                }
                for buffer in set.inner_descriptor_set().buffers_list().iter() {
                    self.add_buffer_resource_inside(buffer.clone(), false, 0 .. buffer.size() /* TODO */,
                                                    vk::PIPELINE_STAGE_ALL_COMMANDS_BIT /* FIXME */,
                                                    vk::ACCESS_SHADER_READ_BIT | vk::ACCESS_UNIFORM_READ_BIT /* TODO */);
                }
            }
            for d in descriptor_sets.iter() { self.keep_alive.push(mem::transmute(d.clone()) /* FIXME: */); }
            let mut descriptor_sets = Some(descriptor_sets.into_iter().map(|set| set.inner_descriptor_set().internal_object()).collect::<SmallVec<[_; 32]>>());

            if mem::size_of_val(push_constants) >= 1 {
                let pipeline = PipelineLayout::inner_pipeline_layout(&**pipeline.layout()).internal_object();
                let size = mem::size_of_val(push_constants);
                let push_constants = push_constants.clone();
                assert!((size % 4) == 0);

                self.render_pass_staging_commands.push(Box::new(move |vk, cmd| {
                    vk.CmdPushConstants(cmd, pipeline, 0x7fffffff, 0, size as u32,
                                        &push_constants as *const Pc as *const _);
                }));
            }

            // FIXME: input attachments of descriptor sets have to be checked against input
            //        attachments of the render pass

            // TODO: shouldn't rebind everything every time
            if !descriptor_sets.as_ref().unwrap().is_empty() {
                let pipeline = PipelineLayout::inner_pipeline_layout(&**pipeline.layout()).internal_object();
                self.render_pass_staging_commands.push(Box::new(move |vk, cmd| {
                    let descriptor_sets = descriptor_sets.take().unwrap();
                    vk.CmdBindDescriptorSets(cmd, vk::PIPELINE_BIND_POINT_GRAPHICS, pipeline,
                                             0, descriptor_sets.len() as u32,
                                             descriptor_sets.as_ptr(), 0, ptr::null());   // FIXME: dynamic offsets
                }));
            }
        }
    }*/

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
                                         clear_values: &[ClearValue]) -> InnerCommandBufferBuilder<P>
        where R: RenderPass + 'static, F: RenderPass + 'static
    {
        debug_assert!(self.render_pass_staging_commands.is_empty());
        debug_assert!(self.render_pass_staging_required_buffer_accesses.is_empty());
        debug_assert!(self.render_pass_staging_required_image_accesses.is_empty());

        let prototype = BeginRenderPassCommand::new(render_pass, framebuffer,
                                                    clear_values.iter().cloned(),
                                                    secondary_cmd_buffers).unwrap();

        for &(ref attachment, ref image, initial_layout, final_layout) in framebuffer.attachments() {
            let stages = PipelineStages {
                fragment_shader: true,
                color_attachment_output: true,
                early_fragment_tests: true,
                late_fragment_tests: true,
                .. PipelineStages::none()
            }; // FIXME:

            let accesses = AccessFlagBits {
                input_attachment_read: true,
                color_attachment_read: true,
                color_attachment_write: true,
                depth_stencil_attachment_read: true,
                depth_stencil_attachment_write: true,
                .. AccessFlagBits::none()
            };   // FIXME:

            // FIXME: parameters
            self.add_image_resource_inside(image.clone(), 0 .. 1, 0 .. 1, true,
                                           initial_layout, final_layout, stages, accesses);
        }

        self.render_pass_staging_commands.push(Box::new(move |cb| prototype.submit(cb)));
        self
    }

    #[inline]
    pub unsafe fn next_subpass(mut self, secondary_cmd_buffers: bool) -> InnerCommandBufferBuilder<P> {
        debug_assert!(!self.render_pass_staging_commands.is_empty());

        let prototype = NextSubpassCommand::new(secondary_cmd_buffers);
        self.render_pass_staging_commands.push(Box::new(move |cb| prototype.submit(cb)));
        self
    }

    /// Ends the current render pass by calling `vkCmdEndRenderPass`.
    ///
    /// # Safety
    ///
    /// Assumes that you're inside a render pass and that all subpasses have been processed.
    ///
    #[inline]
    pub unsafe fn end_renderpass(mut self) -> InnerCommandBufferBuilder<P> {
        debug_assert!(!self.render_pass_staging_commands.is_empty());

        self.flush_render_pass();

        let prototype = EndRenderPassCommand::new();
        self.render_pass_staging_commands.push(Box::new(move |cb| prototype.submit(cb)));
        self
    }

    /// Adds a buffer resource to the list of resources used by this command buffer.
    fn add_buffer_resource_outside(&mut self, buffer: Arc<Buffer>, write: bool,
                                   range: Range<usize>, stages: PipelineStages,
                                   accesses: AccessFlagBits)
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
                    entry.stages |= stages;
                    entry.accesses |= accesses;
                    entry.write = entry.write || write;
                }
            }
        }
    }

    /// Adds an image resource to the list of resources used by this command buffer.
    fn add_image_resource_outside(&mut self, image: Arc<Image>, mipmap_levels_range: Range<u32>,
                                  array_layers_range: Range<u32>, write: bool, layout: ImageLayout,
                                  stages: PipelineStages, accesses: AccessFlagBits)
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
                    entry.stages |= stages;
                    entry.accesses |= accesses;
                    entry.write = entry.write || write;
                    debug_assert_eq!(entry.new_layout, layout);
                    entry.aspects |= aspect_mask;
                }
            }
        }
    }

    /// Adds a buffer resource to the list of resources used by this command buffer.
    fn add_buffer_resource_inside(&mut self, buffer: Arc<Buffer>, write: bool,
                                  range: Range<usize>, stages: PipelineStages,
                                  accesses: AccessFlagBits)
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
                                 stages: PipelineStages, accesses: AccessFlagBits)
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

    /// Flushes the staging render pass commands. Only call this right before `vkCmdEndRenderPass`
    /// or `vkEndCommandBuffer`.
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
            //debug_assert!(!self.is_secondary_graphics);       // TODO: restore
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

    /// Flush the staging commands, inserting a pipeline barrier if necessary.
    fn flush(&mut self, ignore_empty_staging_commands: bool) {
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
        // TODO: inefficient because multiple entries for contiguous blocks could be merged
        //       into one
        let mut prototype = PipelineBarrierCommand::new();

        for (buffer, access) in self.staging_required_buffer_accesses.drain() {
            match self.buffers_state.entry(buffer.clone()) {
                Entry::Vacant(entry) => {
                    // This is the first time we use this buffer, so we may have to add a barrier
                    // with the host if necessary.
                    if (buffer.0).0.host_accesses(buffer.1) && !self.cmd.as_ref().unwrap().is_secondary() {
                        let range = (buffer.0).0.block_memory_range(buffer.1);
                        let slice = BufferSlice::unchecked(buffer.0, range);
                        let src_stages = PipelineStages { host: true, .. PipelineStages::none() };
                        let src_access = AccessFlagBits { host_read: true, host_write: true,
                                                          .. AccessFlagBits::none() };

                        prototype.add_buffer_memory_barrier(slice, src_stages, src_access,
                                                            access.stages, access.accesses,
                                                            true, None);
                    }

                    entry.insert(access);
                },

                Entry::Occupied(mut entry) => {
                    // Buffer was already in use. Checking for conflicts and adding a barrier if
                    // necessary.
                    let entry = entry.get_mut();

                    if !entry.write && access.write {
                        // For writes-after-read, we only need an execution dependency.
                        prototype.add_execution_dependency(entry.stages, access.stages, true);

                        entry.stages = access.stages;
                        entry.accesses = access.accesses;

                    } else if entry.write || access.write {
                        // Read-after-write, write-after-read or write-after-write all requires
                        // a memory barrier.
                        let range = (buffer.0).0.block_memory_range(buffer.1);
                        let slice = BufferSlice::unchecked((buffer.0).0, range);

                        prototype.add_buffer_memory_barrier(slice, entry.stages, entry.accesses,
                                                            access.stages, access.accesses,
                                                            true, None);

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
                    let (extern_layout, host, mem) = if !self.cmd.as_ref().unwrap().is_secondary() {
                        (image.0).0.initial_layout(image.1, access.old_layout)
                    } else {
                        (access.old_layout, false, false)
                    };

                    let src_access = AccessFlagBits {
                        host_read: host,
                        host_write: host,
                        memory_read: mem,       // TODO: looks cheesy, see https://github.com/KhronosGroup/Vulkan-Docs/issues/131
                        .. AccessFlagBits::none()
                    };

                    if extern_layout != access.old_layout || host || mem {
                        let range_mipmaps = (image.0).0.block_mipmap_levels_range(image.1);
                        let range_layers = (image.0).0.block_array_layers_range(image.1);

                        prototype.add_image_memory_barrier((image.0).0, range_mipmaps, range_layers,
                                                           access.stages /* TODO: unclear */, src_access,
                                                           access.stages, access.accesses, true,
                                                           None, extern_layout, access.old_layout);
                        // TODO: pass access.aspects
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
                    // Image was already in use. Checking for conflicts and adding a barrier if
                    // necessary.
                    let mut entry = entry.get_mut();

                    // TODO: check for conflicts

                    let range_mipmaps = (image.0).0.block_mipmap_levels_range(image.1);
                    let range_layers = (image.0).0.block_array_layers_range(image.1);

                    prototype.add_image_memory_barrier((image.0).0, range_mipmaps, range_layers,
                                                       entry.stages, entry.accesses,
                                                       access.stages, access.accesses, true, None,
                                                       entry.new_layout, access.old_layout);
                    // TODO: pass access.aspects

                    // TODO: incomplete
                    entry.stages = access.stages;
                    entry.accesses = access.accesses;
                    entry.new_layout = access.new_layout;
                },
            };
        }

        // Adding the pipeline barrier.
        self.cmd = Some(prototype.submit(self.cmd.take().unwrap()));

        // Now flushing all commands.
        for mut command in self.staging_commands.drain(..) {
            self.cmd = Some(command(self.cmd.take().unwrap()));
        }
    }

    /// Finishes building the command buffer.
    pub fn build(mut self) -> Result<InnerCommandBuffer<P>, OomError> {
        unsafe {
            self.flush_render_pass();
            self.flush(true);

            // Additional transitions at the end of the cb. We do so by inserting elements
            // in `staging_required_image_accesses`, which lets the system think that we are
            // accessing the images, and then flushing again.
            // TODO: ideally things that don't collide should be added before the first flush,
            //       and things that collide done here
            if !self.cmd.as_ref().unwrap().is_secondary() {
                // Ensuring that each image is in its final layout.
                for (image, access) in self.images_state.iter() {
                    let (final_layout, host, mem) = (image.0).0.final_layout(image.1, access.new_layout);
                    if final_layout != access.new_layout || host || mem {
                        self.staging_required_image_accesses.insert(image.clone(), InternalImageBlockAccess {
                            stages: PipelineStages {
                                top_of_pipe: true,
                                .. PipelineStages::none()
                            },
                            accesses: AccessFlagBits {
                                host_read: host,
                                host_write: host,
                                memory_write: mem,
                                .. AccessFlagBits::none()
                            },
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
                        stages: PipelineStages {
                            host: true,
                            .. PipelineStages::none()
                        },
                        accesses: AccessFlagBits {
                            host_read: true,
                            host_write: true,
                            .. AccessFlagBits::none()
                        },
                        write: false,
                    });
                }

                self.flush(true);
            }

            debug_assert!(self.staging_required_buffer_accesses.is_empty());
            debug_assert!(self.staging_required_image_accesses.is_empty());

            Ok(InnerCommandBuffer {
                cmd: try!(self.cmd.take().unwrap().build()),
                buffers_state: self.buffers_state.clone(),      // TODO: meh
                images_state: self.images_state.clone(),        // TODO: meh
                extern_buffers_sync: {
                    let mut map = HashMap::with_hasher(BuildHasherDefault::<FnvHasher>::default());
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
                    let mut map = HashMap::with_hasher(BuildHasherDefault::<FnvHasher>::default());
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
            })
        }
    }
}

/// Actual implementation of all command buffers.
pub struct InnerCommandBuffer<P = Arc<StandardCommandPool>> where P: CommandPool {
    cmd: UnsafeCommandBuffer<P>,
    buffers_state: HashMap<(BufferKey, usize), InternalBufferBlockAccess, BuildHasherDefault<FnvHasher>>,
    images_state: HashMap<(ImageKey, (u32, u32)), InternalImageBlockAccess, BuildHasherDefault<FnvHasher>>,
    extern_buffers_sync: SmallVec<[(Arc<Buffer>, SmallVec<[BufferAccessRange; 4]>); 32]>,
    extern_images_sync: SmallVec<[(Arc<Image>, SmallVec<[ImageAccessRange; 8]>); 32]>,
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
pub fn submit<C, P>(command_buffer: &Arc<C>, queue: &Arc<Queue>)
                    -> Result<Arc<Submission>, OomError>   // TODO: wrong error type
    where C: CommandBuffer + Send + Sync + 'static,
          P: CommandPool,
{
    debug_assert!(!command_buffer.inner_cb().is_secondary());

    // TODO: see comment of GLOBAL_MUTEX
    let _global_lock = GLOBAL_MUTEX.lock().unwrap();

    // TODO: check if this change is okay (maybe the Arc can be omitted?) - Mixthos
    //let fence = try!(Fence::new(queue.device()));
    let fence = Arc::new(try!(Fence::raw(queue.device())));

    let mut signal_semaphores = SmallVec::<[_; 8]>::new();
    let mut wait_semaphores = SmallVec::<[_; 8]>::new();

    // Each queue has a dedicated semaphore which must be signalled and waited upon by each
    // command buffer submission.
    // TODO: for now that's not true ^  as semaphores are only used once then destroyed ;
    //       waiting on https://github.com/KhronosGroup/Vulkan-Docs/issues/155
    {
        // TODO: check if this change is okay (maybe the Arc can be omitted?) - Mixthos
        //let signalled = try!(Semaphore::new(queue.device()));
        let signalled = Arc::new(try!(Semaphore::raw(queue.device())));
        let wait = unsafe { queue.dedicated_semaphore(signalled.clone()) };
        if let Some(wait) = wait {
            wait_semaphores.push((wait, PipelineStages { top_of_pipe: true, .. PipelineStages::none() }));      // TODO:
        }
        signal_semaphores.push(signalled);
    }

    // Creating additional semaphores, one for each queue transition.
    let queue_transitions_hint: u32 = 2;        // TODO: get as function parameter
    // TODO: use a pool
    let semaphores_to_signal = {
        let mut list = SmallVec::new();
        for _ in 0 .. queue_transitions_hint {
            // TODO: check if this change is okay (maybe the Arc can be omitted?) - Mixthos
            //let sem = try!(Semaphore::new(queue.device()));
            let sem = Arc::new(try!(Semaphore::raw(queue.device())));
            signal_semaphores.push(sem.clone());
            list.push(sem);
        }
        list
    };

    // We can now create the `Submission` object.
    // We need to create it early because we pass it when calling `gpu_access`.
    let submission = Arc::new(Submission {
        guarded: Mutex::new(SubmissionGuarded {
            inner: None,
            signalled_semaphores: semaphores_to_signal,
            signalled_queues: SmallVec::new(),
        }),
    });

    // Pipeline barriers for image layout and queue ownership transitions before and after the
    // command buffer to submit.
    let mut before_pipeline = PipelineBarrierCommand::new();
    let mut after_pipeline = PipelineBarrierCommand::new();

    {
        // There is a possibility that a parallel thread is currently submitting a command buffer
        // to another queue. Once we start invoking the `gpu_access` access functions, there is a
        // possibility the other thread detects that it depends on this one and submits its command
        // buffer before us.
        //
        // Since we need to access the content of `guarded` in our dependencies, we lock our own
        // `guarded` here to avoid being used as a dependency before being submitted.
        // TODO: this needs a new block (https://github.com/rust-lang/rfcs/issues/811)
        let submission_lock = submission.guarded.lock().unwrap();

        // Now we determine which earlier submissions we must depend upon.
        let mut dependencies = SmallVec::<[Arc<Submission>; 6]>::new();

        // Buffers first.
        for &(ref resource, ref ranges) in me.extern_buffers_sync.iter() {
            let result = unsafe { resource.gpu_access(&mut ranges.iter().cloned(), &submission) };
            if let Some(semaphore) = result.additional_wait_semaphore {
                wait_semaphores.push((semaphore, PipelineStages { top_of_pipe: true, .. PipelineStages::none() }));     // TODO:
            }

            if let Some(semaphore) = result.additional_signal_semaphore {
                signal_semaphores.push(semaphore);
            }

            dependencies.extend(result.dependencies.into_iter());
        }

        // Then images.
        for &(ref resource, ref ranges) in me.extern_images_sync.iter() {
            let result = unsafe { resource.gpu_access(&mut ranges.iter().cloned(), &submission) };

            if let Some(semaphore) = result.additional_wait_semaphore {
                wait_semaphores.push((semaphore, PipelineStages { top_of_pipe: true, .. PipelineStages::none() }));     // TODO:
            }

            if let Some(semaphore) = result.additional_signal_semaphore {
                signal_semaphores.push(semaphore);
            }

            for transition in result.before_pipeline {
                let mipmaps_range = resource.block_mipmap_levels_range(transition.block);
                let layers_range = resource.block_array_layers_range(transition.block);

                before_pipeline.add_image_memory_barrier(resource, mipmaps_range, layers_range,
                                                         PipelineStages { top_of_pipe: true, .. PipelineStages::none() },
                                                         AccessFlagBits::none(),
                                                         PipelineStages { top_of_pipe: true, .. PipelineStages::none() },
                                                         AccessFlagBits::all(),
                                                         true, None, transition.from,
                                                         transition.to);
            }

            for transition in result.after_pipeline {
                let mipmaps_range = resource.block_mipmap_levels_range(transition.block);
                let layers_range = resource.block_array_layers_range(transition.block);

                after_pipeline.add_image_memory_barrier(resource, mipmaps_range, layers_range,
                                                        PipelineStages { top_of_pipe: true, .. PipelineStages::none() },
                                                        AccessFlagBits::all(),
                                                        PipelineStages { top_of_pipe: true, .. PipelineStages::none() },
                                                        AccessFlagBits::none(),
                                                        true, None, transition.from,
                                                        transition.to);
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

            wait_semaphores.push((semaphore, PipelineStages { top_of_pipe: true, .. PipelineStages::none() }));     // TODO:

            // Note that it may look dangerous to unlock the dependency's mutex here, because the
            // queue has already been added to the list of signalled queues but the command that
            // signals the semaphore hasn't been sent yet.
            //
            // However submitting to a queue must lock the queue, which guarantees that no other
            // parallel queue submission should happen on this same queue. This means that the
            // problem is non-existing.
        }

        let before_command_buffer = if !before_pipeline.is_empty() {
            let mut cb = try!(UnsafeCommandBufferBuilder::new(Kind::Primary, Flags::OneTimeSubmit));
            let cb = try!(before_pipeline.submit(cb).build());
            Some(cb)
        } else {
            None
        };

        let after_command_buffer = if !after_pipeline.is_empty() {
            let mut cb = try!(UnsafeCommandBufferBuilder::new(Kind::Primary, Flags::OneTimeSubmit));
            let cb = try!(after_pipeline.submit(cb).build());
            Some(cb)
        } else {
            None
        };

        let batch = SubmissionDesc {
            command_buffers: before_command_buffer.into_iter()
                                                  .chain(Some(command_buffer).into_iter())
                                                  .chain(after_command_buffer.into_iter()),
            wait_semaphores: wait_semaphores.into_iter(),
            signal_semaphores: signal_semaphores.into_iter(),
        };

        submission_lock.inner = Some(try!(UnsafeSubmission::new(queue, Some(&fence), Some(batch))));
    }

    Ok(submission)
}

#[must_use]
pub struct Submission {
    guarded: Mutex<SubmissionGuarded>,
}

impl fmt::Debug for Submission {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "<Submission object>")      // TODO: better description
    }
}

struct SubmissionGuarded {
    // The inner submission.
    inner: Option<UnsafeSubmission>,

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
        self.inner.fence().unwrap().ready().unwrap_or(false)     // TODO: what to do in case of error?   
    }

    /// Waits until the submission has finished being executed by the device.
    #[inline]
    pub fn wait(&self, timeout: Duration) -> Result<(), FenceWaitError> {
        self.inner.fence().unwrap().wait(timeout)
    }

    /// Returns the `queue` the command buffers were submitted to.
    #[inline]
    pub fn queue(&self) -> &Arc<Queue> {
        self.inner.queue()
    }
}

impl Drop for Submission {
    #[inline]
    fn drop(&mut self) {
        let timeout = Duration::new(u64::MAX / 1_000_000_000, (u64::MAX % 1_000_000_000) as u32);
        match self.inner.fence().wait(timeout) {
            Ok(_) => (),
            Err(FenceWaitError::DeviceLostError) => (),
            Err(FenceWaitError::Timeout) => panic!(),       // The driver has some sort of problem.
            Err(FenceWaitError::OomError(_)) => panic!(),   // What else to do here?
        }

        // TODO: return `signalled_semaphores` to the semaphore pools
    }
}

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
    stages: PipelineStages,

    // The way this resource is accessed.
    // Just like `stages`, this has different semantics depending on the usage of this struct.
    accesses: AccessFlagBits,

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
    stages: PipelineStages,

    // The way this resource is accessed.
    // Just like `stages`, this has different semantics depending on the usage of this struct.
    accesses: AccessFlagBits,

    write: bool,
    aspects: vk::ImageAspectFlags,
    old_layout: ImageLayout,
    new_layout: ImageLayout,
}
