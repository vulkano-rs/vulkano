// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Lowest-level interface for command buffers creation.
//! 
//! This module provides the structs necessary to create command buffers. The main purpose of this
//! module is to provide a nice API, and almost no safety checks are performed except through
//! `debug_assert!`.
//! 
//! # Safety
//! 
//! Each individual function is documented and indicates what exactly must be done in order to be
//! safe.
//! 
//! Things to generally look for when you use an `UnsafeCommandBuffer` are:
//! 
//! - The objects must be kept alive for at least as long as the command buffer.
//! - The objects must be properly synchronized and transitionned with pipeline barriers.
//! - The capabilities of the queue must be checked (for example running a draw operation on a
//!   transfer-only queue won't work).
//! - Some commands can only be called from inside or outside of a render pass.
//!

use std::cmp;
use std::mem;
use std::ops::Range;
use std::ptr;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::u32;
use smallvec::SmallVec;

use buffer::Buffer;
use buffer::BufferInner;
use buffer::TrackedBufferPipelineBarrierRequest;
use buffer::sys::UnsafeBuffer;
use command_buffer::pool::AllocatedCommandBuffer;
use command_buffer::pool::CommandPool;
use command_buffer::pool::CommandPoolFinished;
use descriptor::pipeline_layout::PipelineLayoutRef;
use descriptor::descriptor_set::UnsafeDescriptorSet;
use descriptor::descriptor::ShaderStages;
use device::Device;
use format::ClearValue;
use format::FormatTy;
use framebuffer::RenderPassRef;
use framebuffer::RenderPassSys;
use framebuffer::Subpass;
use framebuffer::traits::Framebuffer;
use image::Image;
use image::sys::Layout;
use image::sys::UnsafeImage;
use image::traits::TrackedImagePipelineBarrierRequest;
use pipeline::ComputePipeline;
use pipeline::GraphicsPipeline;
use pipeline::input_assembly::IndexType;
use sync::AccessFlagBits;
use sync::PipelineStages;

use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

pub struct UnsafeCommandBufferBuilder<P> where P: CommandPool {
    cmd: Option<vk::CommandBuffer>,
    pool: Option<P>,
    device: Arc<Device>,

    // Flags that were used at creation.
    flags: Flags,

    // True if we are a secondary command buffer.
    secondary_cb: bool,

    // True if we are within a render pass.
    within_render_pass: bool,
}

impl<P> UnsafeCommandBufferBuilder<P> where P: CommandPool {
    /// Creates a new builder.
    pub fn new<R, F>(pool: P, kind: Kind<R, F>, flags: Flags)
                     -> Result<UnsafeCommandBufferBuilder<P>, OomError>
        where R: RenderPassRef, F: Framebuffer
    {
        let secondary = match kind {
            Kind::Primary => false,
            Kind::Secondary | Kind::SecondaryRenderPass { .. } => true,
        };

        let cmd = try!(pool.alloc(secondary, 1)).next().unwrap();
        
        match unsafe { UnsafeCommandBufferBuilder::already_allocated(pool, cmd, kind, flags) } {
            Ok(cmd) => Ok(cmd),
            Err(err) => {
                // FIXME: uncomment this and solve the fact that `pool` has been moved
                //unsafe { pool.free(secondary, Some(cmd.into()).into_iter()) };
                Err(err)
            },
        }
    }

    /// Creates a new command buffer builder from an already-allocated command buffer.
    ///
    /// # Safety
    ///
    /// - The allocated command buffer must belong to the pool and must not be used anywhere else
    ///   in the code for the duration of this command buffer.
    ///
    pub unsafe fn already_allocated<R, F>(pool: P, cmd: AllocatedCommandBuffer,
                                          kind: Kind<R, F>, flags: Flags)
                                          -> Result<UnsafeCommandBufferBuilder<P>, OomError>
        where R: RenderPassRef, F: Framebuffer
    {
        let device = pool.device().clone();
        let vk = device.pointers();
        let cmd = cmd.internal_object();

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
            (subpass.render_pass().sys().internal_object(), subpass.index())
        } else {
            (0, 0)
        };

        let framebuffer = if let Kind::SecondaryRenderPass { ref subpass, framebuffer: Some(ref framebuffer) } = kind {
            // TODO: restore check
            //assert!(framebuffer.is_compatible_with(subpass.render_pass()));     // TODO: proper error
            framebuffer.internal_object()
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
            flags: vk_flags,
            pInheritanceInfo: &inheritance,
        };

        try!(check_errors(vk.BeginCommandBuffer(cmd, &infos)));

        Ok(UnsafeCommandBufferBuilder {
            device: device.clone(),
            pool: Some(pool),
            cmd: Some(cmd),
            flags: flags,
            secondary_cb: match kind {
                Kind::Primary => false,
                Kind::Secondary | Kind::SecondaryRenderPass { .. } => true,
            },
            within_render_pass: match kind {
                Kind::Primary | Kind::Secondary => false,
                Kind::SecondaryRenderPass { .. } => true,
            },
        })
    }

    /// Finishes building the command buffer.
    pub fn build(mut self) -> Result<UnsafeCommandBuffer<P>, OomError> {
        unsafe {
            let vk = self.device.pointers();
            let cmd = self.cmd.take().unwrap();
            try!(check_errors(vk.EndCommandBuffer(cmd)));

            Ok(UnsafeCommandBuffer {
                cmd: cmd,
                device: self.device.clone(),
                pool: self.pool.take().unwrap().finish(),
                flags: self.flags,
                already_submitted: AtomicBool::new(false),
                secondary_cb: self.secondary_cb,
            })
        }
    }

    /// Returns the pool used to create this command buffer builder.
    #[inline]
    pub fn pool(&self) -> &P {
        self.pool.as_ref().unwrap()
    }

    /// Returns the device this command buffer belongs to.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns true if this is a secondary command buffer.
    #[inline]
    pub fn is_secondary(&self) -> bool {
        self.secondary_cb
    }

    /// Clears an image with a color format, from outside of a render pass.
    ///
    /// If `general_layout` is true, then the `General` image layout is used. Otherwise the
    /// `TransferDstOptimal` layout is used.
    ///
    /// # Panic
    ///
    /// - Panics if the image was not created with the same device as this command buffer.
    /// - Panics if the clear values is not a color value.
    ///
    /// # Safety
    ///
    /// - The image must be kept alive and must be properly synchronized while this command
    ///   buffer runs.
    /// - The ranges must be in range of the image.
    /// - The image must have a non-compressed color format.
    /// - The clear value must match the format of the image.
    /// - The queue family must support graphics or compute operations.
    /// - The image must have been created with the "transfer_dest" usage.
    /// - Must be called outside of a render pass.
    ///
    pub unsafe fn clear_color_image<I>(&mut self, image: &UnsafeImage, general_layout: bool,
                                       color: ClearValue, ranges: I)
        where I: Iterator<Item = ImageSubresourcesRange>
    {
        assert_eq!(image.device().internal_object(), self.device.internal_object());

        let clear_value = match color {
            ClearValue::None => panic!(),
            ClearValue::Float(val) => {
                debug_assert_eq!(image.format().ty(), FormatTy::Float);
                vk::ClearColorValue::float32(val)
            },
            ClearValue::Int(val) => {
                debug_assert_eq!(image.format().ty(), FormatTy::Sint);
                vk::ClearColorValue::int32(val)
            },
            ClearValue::Uint(val) => {
                debug_assert_eq!(image.format().ty(), FormatTy::Uint);
                vk::ClearColorValue::uint32(val)
            },
            ClearValue::Depth(_) => panic!(),
            ClearValue::Stencil(_) => panic!(),
            ClearValue::DepthStencil(_) => panic!(),
        };

        let ranges: SmallVec<[_; 4]> = ranges.filter_map(|range| {
            debug_assert!(range.first_mipmap_level + range.num_mipmap_levels <=
                          image.mipmap_levels());
            debug_assert!(range.first_array_layer + range.num_array_layers <=
                          image.dimensions().array_layers());

            if range.num_mipmap_levels == 0 {
                return None;
            }

            if range.num_array_layers == 0 {
                return None;
            }

            Some(vk::ImageSubresourceRange {
                aspectMask: vk::IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel: range.first_mipmap_level,
                levelCount: range.num_mipmap_levels,
                baseArrayLayer: range.first_array_layer,
                layerCount: range.num_array_layers,
            })
        }).collect();

        // Do nothing if no range to clear.
        if ranges.is_empty() {
            return;
        }

        let layout = if general_layout { vk::IMAGE_LAYOUT_GENERAL }
                     else { vk::IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL };

        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdClearColorImage(cmd, image.internal_object(), layout, &clear_value,
                              ranges.len() as u32, ranges.as_ptr());
    }

    /// Clears an image with a depth, stencil or depth-stencil format, from outside of a
    /// render pass.
    ///
    /// If the `ClearValue` is a depth value, then only the depth component will be cleared. Same
    /// for stencil. If it contains a depth-stencil value, then they will both be cleared.
    ///
    /// If `general_layout` is true, then the `General` image layout is used. Otherwise the
    /// `TransferDstOptimal` layout is used.
    ///
    /// # Panic
    ///
    /// - Panics if the image was not created with the same device as this command buffer.
    /// - Panics if the mipmap levels range or the array layers range is invalid, ie. if the end
    ///   is inferior to the start.
    /// - Panics if the clear values is not a depth, stencil or depth-stencil value.
    ///
    /// # Safety
    ///
    /// - The image must be kept alive and must be properly synchronized while this command
    ///   buffer runs.
    /// - The ranges must be in range of the image.
    /// - The image must have a depth, stencil or depth-stencil format.
    /// - The clear value must match the format of the image.
    /// - The queue family must support graphics operations.
    /// - The image must have been created with the "transfer_dest" usage.
    /// - Must be called outside of a render pass.
    ///
    pub unsafe fn clear_depth_stencil_image<I>(&mut self, image: &UnsafeImage, general_layout: bool,
                                               color: ClearValue, ranges: I)
        where I: Iterator<Item = ImageSubresourcesRange>
    {
        assert_eq!(image.device().internal_object(), self.device.internal_object());

        let (clear_value, aspect_mask) = match color {
            ClearValue::None => panic!(),
            ClearValue::Float(_) => panic!(),
            ClearValue::Int(_) => panic!(),
            ClearValue::Uint(_) => panic!(),
            ClearValue::Depth(val) => {
                debug_assert!(image.format().ty() == FormatTy::Depth ||
                              image.format().ty() == FormatTy::DepthStencil);
                let clear = vk::ClearDepthStencilValue { depth: val, stencil: 0 };
                let aspect = vk::IMAGE_ASPECT_DEPTH_BIT;
                (clear, aspect)
            },
            ClearValue::Stencil(val) => {
                debug_assert!(image.format().ty() == FormatTy::Stencil ||
                              image.format().ty() == FormatTy::DepthStencil);
                let clear = vk::ClearDepthStencilValue { depth: 0.0, stencil: val };
                let aspect = vk::IMAGE_ASPECT_STENCIL_BIT;
                (clear, aspect)
            },
            ClearValue::DepthStencil((depth, stencil)) => {
                debug_assert_eq!(image.format().ty(), FormatTy::DepthStencil);
                let clear = vk::ClearDepthStencilValue { depth: depth, stencil: stencil };
                let aspect = vk::IMAGE_ASPECT_DEPTH_BIT | vk::IMAGE_ASPECT_STENCIL_BIT;
                (clear, aspect)
            },
        };

        let ranges: SmallVec<[_; 4]> = ranges.filter_map(|range| {
            debug_assert!(range.first_mipmap_level + range.num_mipmap_levels <=
                          image.mipmap_levels());
            debug_assert!(range.first_array_layer + range.num_array_layers <=
                          image.dimensions().array_layers());

            if range.num_mipmap_levels == 0 {
                return None;
            }

            if range.num_array_layers == 0 {
                return None;
            }

            Some(vk::ImageSubresourceRange {
                aspectMask: aspect_mask,
                baseMipLevel: range.first_mipmap_level,
                levelCount: range.num_mipmap_levels,
                baseArrayLayer: range.first_array_layer,
                layerCount: range.num_array_layers,
            })
        }).collect();

        // Do nothing if no range to clear.
        if ranges.is_empty() {
            return;
        }

        let layout = if general_layout { vk::IMAGE_LAYOUT_GENERAL }
                     else { vk::IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL };

        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdClearDepthStencilImage(cmd, image.internal_object(), layout, &clear_value,
                                     ranges.len() as u32, ranges.as_ptr());
    }

    /// Clears attachments of the current render pass.
    ///
    /// You must pass a list of attachment ids and clear values, and a list of rectangles. Each
    /// rectangle of each attachment will be cleared. The rectangle's format is
    /// `[(x, width), (y, height), (array_layer, num_array_layers)]`.
    ///
    /// No memory barriers are needed between this function and preceding or subsequent draw or
    /// attachment clear commands in the same subpass.
    ///
    /// # Panic
    ///
    /// - Panics if one of the clear values is `None`.
    ///
    /// # Safety
    ///
    /// - The attachments ids must be valid, and the clear value must match the format of the
    ///   attachments.
    /// - Must be called from within a render pass.
    /// - The rects must be in range of the framebuffer.
    ///
    pub unsafe fn clear_attachments<Ia, Ir>(&mut self, attachments: Ia, rects: Ir)
        where Ia: Iterator<Item = (u32, ClearValue)>,
              Ir: Iterator<Item = [(u32, u32); 3]>,
    {
        let rects: SmallVec<[_; 3]> = rects.filter_map(|rect| {
            if rect[0].1 == 0 || rect[1].1 == 0 || rect[2].1 == 0 {
                return None;
            }

            Some(vk::ClearRect {
                rect: vk::Rect2D {
                    offset: vk::Offset2D {
                        x: rect[0].0 as i32,
                        y: rect[1].0 as i32,
                    },
                    extent: vk::Extent2D {
                        width: rect[0].1,
                        height: rect[1].1,
                    },
                },
                baseArrayLayer: rect[2].0,
                layerCount: rect[2].1,
            })
        }).collect();

        let attachments: SmallVec<[_; 8]> = attachments.map(|(attachment, clear_value)| {
            let (clear_value, aspect_mask) = match clear_value {
                ClearValue::None => panic!(),
                ClearValue::Float(val) => {
                    let clear = vk::ClearValue::color(vk::ClearColorValue::float32(val));
                    let aspect = vk::IMAGE_ASPECT_COLOR_BIT;
                    (clear, aspect)
                },
                ClearValue::Int(val) => {
                    let clear = vk::ClearValue::color(vk::ClearColorValue::int32(val));
                    let aspect = vk::IMAGE_ASPECT_COLOR_BIT;
                    (clear, aspect)
                },
                ClearValue::Uint(val) => {
                    let clear = vk::ClearValue::color(vk::ClearColorValue::uint32(val));
                    let aspect = vk::IMAGE_ASPECT_COLOR_BIT;
                    (clear, aspect)
                },
                ClearValue::Depth(val) => {
                    let clear = vk::ClearValue::depth_stencil(vk::ClearDepthStencilValue {
                        depth: val, stencil: 0
                    });
                    let aspect = vk::IMAGE_ASPECT_DEPTH_BIT;
                    (clear, aspect)
                },
                ClearValue::Stencil(val) => {
                    let clear = vk::ClearValue::depth_stencil(vk::ClearDepthStencilValue {
                        depth: 0.0, stencil: val
                    });
                    let aspect = vk::IMAGE_ASPECT_STENCIL_BIT;
                    (clear, aspect)
                },
                ClearValue::DepthStencil((depth, stencil)) => {
                    let clear = vk::ClearValue::depth_stencil(vk::ClearDepthStencilValue {
                        depth: depth, stencil: stencil,
                    });
                    let aspect = vk::IMAGE_ASPECT_DEPTH_BIT | vk::IMAGE_ASPECT_STENCIL_BIT;
                    (clear, aspect)
                },
            };

            vk::ClearAttachment {
                aspectMask: aspect_mask,
                colorAttachment: attachment,
                clearValue: clear_value,
            }
        }).collect();

        // Do nothing if nothing to do.
        if rects.is_empty() || attachments.is_empty() {
            return;
        }

        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdClearAttachments(cmd, attachments.len() as u32, attachments.as_ptr(),
                               rects.len() as u32, rects.as_ptr());
    }

    /// Fills a buffer by repeating a 32 bits data.
    ///
    /// This is similar to the `memset` function in C/C++.
    ///
    /// # Panic
    ///
    /// - Panics if the buffer was not created with the same device as this command buffer.
    ///
    /// # Safety
    ///
    /// - The buffer must be kept alive and must be properly synchronized while this command
    ///   buffer runs.
    /// - The queue family must support graphics or compute operations.
    /// - Type safety is not checked.
    /// - The offset must be a multiple of four.
    /// - The size must be a multiple of four, or must point to the end of the buffer.
    /// - The buffer must have been created with the "transfer_dest" usage.
    /// - Must be called outside of a render pass.
    ///
    pub unsafe fn fill_buffer<B>(&mut self, buffer: &B, data: u32)
        where B: Buffer
    {
        let size = buffer.size();
        let BufferInner { buffer, offset } = buffer.inner();

        assert_eq!(buffer.device().internal_object(), self.device.internal_object());

        debug_assert_eq!(offset % 4, 0);

        let size = if offset + size == buffer.size() {
            vk::WHOLE_SIZE
        } else {
            debug_assert_eq!(size % 4, 0);
            size as vk::DeviceSize
        };

        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdFillBuffer(cmd, buffer.internal_object(), offset as vk::DeviceSize,
                         size as vk::DeviceSize, data);
    }

    /// Fills a buffer with some data.
    ///
    /// The actual size that is copied is the minimum between the size of the slice and the size
    /// of the data.
    ///
    /// # Panic
    ///
    /// - Panics if the buffer was not created with the same device as this command buffer.
    ///
    /// # Safety
    ///
    /// - The buffer must be kept alive and must be properly synchronized while this command
    ///   buffer runs.
    /// - Type safety is not checked.
    /// - The offset and size must be multiples of four.
    /// - The size must be less than or equal to 65536 bytes (ie. 64kB).
    /// - The buffer must have been created with the "transfer_dest" usage.
    /// - Must be called outside of a render pass.
    ///
    pub unsafe fn update_buffer<B, D: ?Sized>(&mut self, buffer: &B, data: &D)
        where B: Buffer, D: Copy + 'static
    {
        let size = buffer.size();
        let BufferInner { buffer, offset } = buffer.inner();

        assert_eq!(buffer.device().internal_object(), self.device.internal_object());

        let size = cmp::min(size, mem::size_of_val(data));

        debug_assert_eq!(offset % 4, 0);
        debug_assert_eq!(size % 4, 0);
        debug_assert!(size <= 65536);

        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdUpdateBuffer(cmd, buffer.internal_object(), offset as vk::DeviceSize,
                           size as vk::DeviceSize, data as *const D as *const _);
    }

    /// Copies data from a source buffer to a destination buffer.
    ///
    /// This is similar to the `memcpy` function in C/C++.
    ///
    /// Automatically filters out empty regions.
    ///
    /// # Panic
    ///
    /// - Panics if one of the buffers was not created with the same device as this
    ///   command buffer.
    ///
    /// # Safety
    ///
    /// - The buffers must be kept alive and must be properly synchronized while this command
    ///   buffer runs.
    /// - Type safety is not checked.
    /// - The source buffer must have been created with the "transfer_src" usage.
    /// - The destination buffer must have been created with the "transfer_dest" usage.
    /// - Must be called outside of a render pass.
    /// - The offsets and size of the regions must be in range.
    ///
    pub unsafe fn copy_buffer<S, D, I>(&mut self, src: &S, dest: &D, regions: I)
        where S: Buffer,
              D: Buffer,
              I: IntoIterator<Item = BufferCopyRegion>
    {
        let src_size = src.size();
        let BufferInner { buffer: src_buffer, offset: src_offset } = src.inner();
        let dest_size = dest.size();
        let BufferInner { buffer: dest_buffer, offset: dest_offset } = dest.inner();

        assert_eq!(src_buffer.device().internal_object(), self.device.internal_object());
        assert_eq!(src_buffer.device().internal_object(), dest_buffer.device().internal_object());

        let regions: SmallVec<[_; 4]> = {
            let mut res = SmallVec::new();
            for region in regions.into_iter() {
                if region.size == 0 { continue; }
                debug_assert!(region.source_offset < src_size);
                debug_assert!(region.source_offset + region.size <= src_size);
                debug_assert!(region.destination_offset < dest_size);
                debug_assert!(region.destination_offset + region.size <= dest_size);

                res.push(vk::BufferCopy {
                    srcOffset: (region.source_offset + src_offset) as vk::DeviceSize,
                    dstOffset: (region.destination_offset + dest_offset) as vk::DeviceSize,
                    size: region.size as vk::DeviceSize,
                });
            }
            res
        };

        // Calling vkCmdCopyBuffer with 0 regions is forbidden. We just don't call the function
        // then.
        if regions.is_empty() {
            return;
        }

        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdCopyBuffer(cmd, src_buffer.internal_object(), dest_buffer.internal_object(),
                         regions.len() as u32, regions.as_ptr());
    }

    /// Executes secondary command buffers..
    // TODO: check for same device
    // TODO: crappy API
    pub unsafe fn execute_commands<I>(&mut self, command_buffers: I)
        where I: IntoIterator<Item = vk::CommandBuffer>
    {
        let raw_cbs: SmallVec<[_; 16]> = command_buffers.into_iter().collect();

        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdExecuteCommands(cmd, raw_cbs.len() as u32, raw_cbs.as_ptr());
    }

    /// Adds a pipeline barrier to the command buffer.
    ///
    /// This function itself is not unsafe, but creating a pipeline barrier builder is.
    #[inline]
    pub fn pipeline_barrier(&mut self, barrier: PipelineBarrierBuilder) {
        // If barrier is empty, don't do anything.
        if barrier.src_stage_mask == 0 || barrier.dst_stage_mask == 0 {
            debug_assert!(barrier.src_stage_mask == 0 && barrier.dst_stage_mask == 0);
            debug_assert!(barrier.memory_barriers.is_empty());
            debug_assert!(barrier.buffer_barriers.is_empty());
            debug_assert!(barrier.image_barriers.is_empty());
            return;
        }

        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();

        unsafe {
            vk.CmdPipelineBarrier(cmd, barrier.src_stage_mask, barrier.dst_stage_mask,
                                  barrier.dependency_flags, barrier.memory_barriers.len() as u32,
                                  barrier.memory_barriers.as_ptr(),
                                  barrier.buffer_barriers.len() as u32,
                                  barrier.buffer_barriers.as_ptr(),
                                  barrier.image_barriers.len() as u32,
                                  barrier.image_barriers.as_ptr());
        }
    }

    /// Enters a render pass.
    ///
    /// Any clear value that is equal to `None` is replaced with a dummy value. It is expected that
    /// `None` is passed only for attachments that are not cleared.
    ///
    /// # Panic
    ///
    /// - Panics if the render pass or framebuffer was not created with the same device as this
    ///   command buffer.
    /// - Panics if one of the ranges is invalid.
    ///
    /// # Safety
    ///
    /// - Must be called outside of a render pass.
    /// - The queue family must support graphics operations.
    /// - The render pass and the framebuffer must be kept alive.
    /// - The render pass and the framebuffer must be compatible.
    /// - The clear values must be valid for the attachments.
    ///
    pub unsafe fn begin_render_pass<I, F>(&mut self, render_pass: RenderPassSys,
                                          framebuffer: &F, clear_values: I,
                                          rect: [Range<u32>; 2], secondary: bool)
        where I: Iterator<Item = ClearValue>,
              F: Framebuffer
    {
        // TODO: restore these checks
        //assert_eq!(render_pass.device().internal_object(), framebuffer.device().internal_object());
        //assert_eq!(self.device.internal_object(), framebuffer.device().internal_object());

        let clear_values: SmallVec<[_; 12]> = clear_values.map(|clear_value| {
            match clear_value {
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
                        depth: val, stencil: 0
                    })
                },
                ClearValue::Stencil(val) => {
                    vk::ClearValue::depth_stencil(vk::ClearDepthStencilValue {
                        depth: 0.0, stencil: val
                    })
                },
                ClearValue::DepthStencil((depth, stencil)) => {
                    vk::ClearValue::depth_stencil(vk::ClearDepthStencilValue {
                        depth: depth, stencil: stencil,
                    })
                },
            }
        }).collect();

        assert!(rect[0].start <= rect[0].end);
        assert!(rect[1].start <= rect[1].end);

        let infos = vk::RenderPassBeginInfo {
            sType: vk::STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            pNext: ptr::null(),
            renderPass: render_pass.internal_object(),
            framebuffer: framebuffer.internal_object(),
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
            clearValueCount: clear_values.len() as u32,
            pClearValues: clear_values.as_ptr(),
        };

        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdBeginRenderPass(cmd, &infos,
                              if secondary { vk::SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS }
                              else { vk::SUBPASS_CONTENTS_INLINE });
    }

    /// Goes to the next subpass of the render pass.
    ///
    /// # Safety
    ///
    /// - Must be called inside of a render pass.
    /// - Must not be at the last subpass of the render pass.
    ///
    #[inline]
    pub unsafe fn next_subpass(&mut self, secondary: bool) {
        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdNextSubpass(cmd, if secondary { vk::SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS }
                               else { vk::SUBPASS_CONTENTS_INLINE });
    }

    /// Ends the current render pass.
    ///
    /// # Safety
    ///
    /// - Must be called inside of a render pass.
    /// - Must be at the last subpass of the render pass.
    ///
    #[inline]
    pub unsafe fn end_render_pass(&mut self) {
        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdEndRenderPass(cmd);
    }

    /// Binds a graphics pipeline to the graphics pipeline bind point.
    ///
    /// # Safety
    ///
    /// - The queue family must support graphics operations.
    /// - If the variable multisample rate feature is not supported, the current subpass has no
    ///   attachments, and this is not the first call to this function with a graphics pipeline
    ///   after transitioning to the current subpass, then the sample count specified by this
    ///   pipeline must match that set in the previous pipeline.
    ///
    #[inline]
    pub unsafe fn bind_pipeline_graphics<V, L, R>(&mut self, pipeline: &GraphicsPipeline<V, L, R>) {
        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdBindPipeline(cmd, vk::PIPELINE_BIND_POINT_GRAPHICS, pipeline.internal_object());
    }

    /// Binds a compute pipeline to the compute pipeline bind point.
    ///
    /// # Safety
    ///
    /// - The queue family must support compute operations.
    ///
    #[inline]
    pub unsafe fn bind_pipeline_compute<L>(&mut self, pipeline: &ComputePipeline<L>) {
        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdBindPipeline(cmd, vk::PIPELINE_BIND_POINT_COMPUTE, pipeline.internal_object());
    }

    /// Calls `vkCmdDraw`.
    #[inline]
    pub unsafe fn draw(&mut self, vertex_count: u32, instance_count: u32, first_vertex: u32,
                       first_instance: u32)
    {
        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdDraw(cmd, vertex_count, instance_count, first_vertex, first_instance);
    }

    /// Calls `vkCmdDrawIndexed`.
    #[inline]
    pub unsafe fn draw_indexed(&mut self, vertex_count: u32, instance_count: u32,
                               first_index: u32, vertex_offset: i32, first_instance: u32)
    {
        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdDrawIndexed(cmd, vertex_count, instance_count, first_index, vertex_offset,
                          first_instance);
    }

    /// Calls `vkCmdDrawIndirect`.
    ///
    /// # Panic
    ///
    /// - Panics if the buffer was not created with the same device as this command buffer.
    ///
    // TODO: don't request UnsafeBuffer
    #[inline]
    pub unsafe fn draw_indirect(&mut self, buffer: &UnsafeBuffer, offset: usize, draw_count: u32,
                                stride: u32)
    {
        assert_eq!(buffer.device().internal_object(), self.device.internal_object());

        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdDrawIndirect(cmd, buffer.internal_object(), offset as vk::DeviceSize, draw_count,
                           stride);
    }

    /// Calls `vkCmdDrawIndexedIndirect`.
    ///
    /// # Panic
    ///
    /// - Panics if the buffer was not created with the same device as this command buffer.
    ///
    // TODO: don't request UnsafeBuffer
    #[inline]
    pub unsafe fn draw_indexed_indirect(&mut self, buffer: &UnsafeBuffer, offset: usize,
                                        draw_count: u32, stride: u32)
    {
        assert_eq!(buffer.device().internal_object(), self.device.internal_object());

        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdDrawIndexedIndirect(cmd, buffer.internal_object(), offset as vk::DeviceSize,
                                  draw_count, stride);
    }

    /// Calls `vkCmdDispatch`.
    #[inline]
    pub unsafe fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdDispatch(cmd, x, y, z);
    }

    /// Calls `vkCmdDispatchIndirect`.
    ///
    /// # Panic
    ///
    /// - Panics if the buffer was not created with the same device as this command buffer.
    ///
    // TODO: don't request UnsafeBuffer
    #[inline]
    pub unsafe fn dispatch_indirect(&mut self, buffer: &UnsafeBuffer, offset: usize) {
        assert_eq!(buffer.device().internal_object(), self.device.internal_object());

        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdDispatchIndirect(cmd, buffer.internal_object(), offset as vk::DeviceSize);
    }

    /// Calls `vkCmdBindVertexBuffers`.
    ///
    /// The iterator yields a list of buffers and offset of the first byte.
    ///
    /// # Panic
    ///
    /// - Panics if one of the buffers was not created with the same device as this command buffer.
    ///
    // TODO: don't request vk::Buffer
    #[inline]
    pub unsafe fn bind_vertex_buffers<'a, I>(&mut self, first_binding: u32, buffers: I)
        where I: IntoIterator<Item = (vk::Buffer, usize)>
    {
        let mut raw_buffers: SmallVec<[_; 8]> = SmallVec::new();
        let mut raw_offsets: SmallVec<[_; 8]> = SmallVec::new();

        for (buf, off) in buffers {
            // TODO: restore
            //assert_eq!(buf.device().internal_object(), self.device.internal_object());
            raw_buffers.push(buf);
            raw_offsets.push(off as vk::DeviceSize);
        }

        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdBindVertexBuffers(cmd, first_binding, raw_buffers.len() as u32, raw_buffers.as_ptr(),
                                raw_offsets.as_ptr());
    }

    /// Calls `vkCmdBindIndexBuffer`.
    ///
    /// # Panic
    ///
    /// - Panics if the buffer was not created with the same device as this command buffer.
    ///
    #[inline]
    pub unsafe fn bind_index_buffer<B>(&mut self, buffer: &B, index_ty: IndexType)
        where B: Buffer
    {
        let BufferInner { buffer, offset } = buffer.inner();

        assert_eq!(buffer.device().internal_object(), self.device.internal_object());

        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdBindIndexBuffer(cmd, buffer.internal_object(), offset as vk::DeviceSize,
                              index_ty as u32);
    }

    /// Calls `vkCmdBindDescriptorSets`.
    ///
    /// # Panic
    ///
    /// - Panics if the layout or one of the sets were not created with the same device as this
    ///   command buffer.
    ///
    #[inline]
    // TODO: change the API to take implementations of DescriptorSet instead of UnsafeDescriptorSet
    pub unsafe fn bind_descriptor_sets<'a, L, Ides, Idyn>(&mut self, graphics_bind_point: bool,
                                                          layout: &L, first_set: u32,
                                                          descriptor_sets: Ides,
                                                          dynamic_offsets: Idyn)
        where L: PipelineLayoutRef,
              Ides: IntoIterator<Item = &'a UnsafeDescriptorSet>,
              Idyn: IntoIterator<Item = u32>
    {
        let bind_point = if graphics_bind_point { vk::PIPELINE_BIND_POINT_GRAPHICS }
                         else { vk::PIPELINE_BIND_POINT_COMPUTE };

        assert_eq!(layout.device().internal_object(), self.device.internal_object());

        let descriptor_sets: SmallVec<[_; 16]> = descriptor_sets.into_iter().map(|set| {
            assert_eq!(set.layout().device().internal_object(), self.device.internal_object());
            set.internal_object()
        }).collect();

        let dynamic_offsets: SmallVec<[_; 64]> = dynamic_offsets.into_iter().collect();

        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdBindDescriptorSets(cmd, bind_point, layout.sys().internal_object(), first_set,
                                 descriptor_sets.len() as u32, descriptor_sets.as_ptr(),
                                 dynamic_offsets.len() as u32, dynamic_offsets.as_ptr());
    }

    /// Calls `vkCmdPushConstants`.
    ///
    /// # Panic
    ///
    /// - Panics if the layout was not created with the same device as this command buffer.
    ///
    #[inline]
    pub unsafe fn push_constants<L, D: ?Sized>(&mut self, layout: &L,
                                               stages: ShaderStages, offset: usize, data: &D)
        where L: PipelineLayoutRef
    {
        assert_eq!(layout.device().internal_object(), self.device.internal_object());

        debug_assert!(offset <= u32::MAX as usize);
        debug_assert!(mem::size_of_val(data) <= u32::MAX as usize);

        let vk = self.device.pointers();
        let cmd = self.cmd.clone().take().unwrap();
        vk.CmdPushConstants(cmd, layout.sys().internal_object(), stages.into(), offset as u32,
                            mem::size_of_val(data) as u32, data as *const D as *const _);
    }
}

unsafe impl<P> VulkanObject for UnsafeCommandBufferBuilder<P> where P: CommandPool {
    type Object = vk::CommandBuffer;

    #[inline]
    fn internal_object(&self) -> vk::CommandBuffer {
        self.cmd.unwrap()
    }
}

impl<P> Drop for UnsafeCommandBufferBuilder<P> where P: CommandPool {
    #[inline]
    fn drop(&mut self) {
        if let Some(cmd) = self.cmd {
            unsafe {
                let vk = self.device.pointers();
                vk.EndCommandBuffer(cmd);       // TODO: really needed?

                self.pool.as_ref().unwrap().free(self.secondary_cb, Some(cmd.into()).into_iter());
            }
        }
    }
}

/// Determines the kind of command buffer that we want to create.
#[derive(Debug, Clone)]
pub enum Kind<'a, R, F: 'a> {
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
        framebuffer: Option<&'a F>,
    },
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

/// Range of an image subresource.
pub struct ImageSubresourcesRange {
    pub first_mipmap_level: u32,
    pub num_mipmap_levels: u32,
    pub first_array_layer: u32,
    pub num_array_layers: u32,
}

/// A copy between two buffers.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct BufferCopyRegion {
    /// Offset of the first byte to read from the source buffer.
    pub source_offset: usize,
    /// Offset of the first byte to write to the destination buffer.
    pub destination_offset: usize,
    /// Size in bytes of the copy.
    pub size: usize,
}

/// Prototype for a pipeline barrier that's going to be added to a command buffer builder.
///
/// Note: we use a builder-like API here so that users can pass multiple buffers or images of
/// multiple different types. Doing so with a single function would be very tedious in terms of
/// API.
pub struct PipelineBarrierBuilder {
    src_stage_mask: vk::PipelineStageFlags,
    dst_stage_mask: vk::PipelineStageFlags,
    dependency_flags: vk::DependencyFlags,
    memory_barriers: SmallVec<[vk::MemoryBarrier; 2]>,
    buffer_barriers: SmallVec<[vk::BufferMemoryBarrier; 8]>,
    image_barriers: SmallVec<[vk::ImageMemoryBarrier; 8]>,
}

impl PipelineBarrierBuilder {
    /// Adds a command that adds a pipeline barrier to a command buffer.
    #[inline]
    pub fn new() -> PipelineBarrierBuilder {
        PipelineBarrierBuilder {
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
    pub fn merge(&mut self, other: PipelineBarrierBuilder) {
        self.src_stage_mask |= other.src_stage_mask;
        self.dst_stage_mask |= other.dst_stage_mask;
        self.dependency_flags &= other.dependency_flags;

        self.memory_barriers.extend(other.memory_barriers.into_iter());
        self.buffer_barriers.extend(other.buffer_barriers.into_iter());
        self.image_barriers.extend(other.image_barriers.into_iter());
    }

    /// Adds an execution dependency. This means that all the stages in `source` of the previous
    /// commands must finish before any of the stages in `dest` of the following commands can start.
    ///
    /// # Safety
    ///
    /// - If the pipeline stages include geometry or tessellation stages, then the corresponding
    ///   features must have been enabled.
    /// - There are certain rules regarding the pipeline barriers inside render passes.
    ///
    #[inline]
    pub unsafe fn add_execution_dependency(&mut self, source: PipelineStages, dest: PipelineStages,
                                           by_region: bool)
    {
        if !by_region {
            self.dependency_flags = 0;
        }

        self.src_stage_mask |= source.into();
        self.dst_stage_mask |= dest.into();
    }

    /// Adds a memory barrier. This means that all the memory writes by the given source stages
    /// for the given source accesses must be visible by the given dest stages for the given dest
    /// accesses.
    ///
    /// Also adds an execution dependency.
    ///
    /// # Safety
    ///
    /// - If the pipeline stages include geometry or tessellation stages, then the corresponding
    ///   features must have been enabled.
    /// - There are certain rules regarding the pipeline barriers inside render passes.
    ///
    pub unsafe fn add_memory_barrier(&mut self, source_stage: PipelineStages,
                                     source_access: AccessFlagBits, dest_stage: PipelineStages,
                                     dest_access: AccessFlagBits, by_region: bool)
    {
        self.add_execution_dependency(source_stage, dest_stage, by_region);

        self.memory_barriers.push(vk::MemoryBarrier {
            sType: vk::STRUCTURE_TYPE_MEMORY_BARRIER,
            pNext: ptr::null(),
            srcAccessMask: source_access.into(),
            dstAccessMask: dest_access.into(),
        });
    }

    pub unsafe fn add_buffer_barrier_request<B>(&mut self, buffer: &B,
                                                request: TrackedBufferPipelineBarrierRequest)
        where B: Buffer
    {
        if !request.by_region {
            self.dependency_flags = 0;
        }

        self.src_stage_mask |= request.source_stage.into();
        self.dst_stage_mask |= request.destination_stages.into();

        if let Some(memory_barrier) = request.memory_barrier {
            let (src_queue, dest_queue) = /*if let Some((src_queue, dest_queue)) = queue_transfer {
                (src_queue, dest_queue)
            } else {*/
                (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
            /*}*/;
            
            // TODO: add more debug asserts?

            let size = buffer.size();
            let BufferInner { buffer, offset } = buffer.inner();
            debug_assert!(memory_barrier.offset + offset as isize >= 0);

            self.buffer_barriers.push(vk::BufferMemoryBarrier {
                sType: vk::STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                pNext: ptr::null(),
                srcAccessMask: memory_barrier.source_access.into(),
                dstAccessMask: memory_barrier.destination_access.into(),
                srcQueueFamilyIndex: src_queue,
                dstQueueFamilyIndex: dest_queue,
                buffer: buffer.internal_object(),
                offset: (memory_barrier.offset + offset as isize) as vk::DeviceSize,
                size: (memory_barrier.size + size) as vk::DeviceSize,
            });
        }
    }

    pub unsafe fn add_image_barrier_request<I>(&mut self, image: &I,
                                               request: TrackedImagePipelineBarrierRequest)
        where I: Image
    {
        if !request.by_region {
            self.dependency_flags = 0;
        }

        self.src_stage_mask |= request.source_stage.into();
        self.dst_stage_mask |= request.destination_stages.into();

        if let Some(memory_barrier) = request.memory_barrier {
            let (src_queue, dest_queue) = /*if let Some((src_queue, dest_queue)) = queue_transfer {
                (src_queue, dest_queue)
            } else {*/
                (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
            /*}*/;

            // TODO: add more debug asserts

            debug_assert!(memory_barrier.first_mipmap +
                          memory_barrier.num_mipmaps <= image.inner().mipmap_levels());     // TODO: don't use inner()
            debug_assert!(memory_barrier.first_layer +
                          memory_barrier.num_layers <= image.dimensions().array_layers());

            self.image_barriers.push(vk::ImageMemoryBarrier {
                sType: vk::STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                pNext: ptr::null(),
                srcAccessMask: memory_barrier.source_access.into(),
                dstAccessMask: memory_barrier.destination_access.into(),
                oldLayout: memory_barrier.old_layout as u32,
                newLayout: memory_barrier.new_layout as u32,
                srcQueueFamilyIndex: src_queue,
                dstQueueFamilyIndex: dest_queue,
                image: image.inner().internal_object(),
                subresourceRange: vk::ImageSubresourceRange {
                    aspectMask: if image.has_color() {
                        vk::IMAGE_ASPECT_COLOR_BIT
                    } else {
                        vk::IMAGE_ASPECT_DEPTH_BIT | vk::IMAGE_ASPECT_STENCIL_BIT
                    },
                    baseMipLevel: memory_barrier.first_mipmap,
                    levelCount: memory_barrier.num_mipmaps,
                    baseArrayLayer: memory_barrier.first_layer,
                    layerCount: memory_barrier.num_layers,
                },
            });
        }
    }

    /// Adds a buffer memory barrier. This means that all the memory writes to the given buffer by
    /// the given source stages for the given source accesses must be visible by the given dest
    /// stages for the given dest accesses.
    ///
    /// Also adds an execution dependency.
    ///
    /// Also allows transfering buffer ownership between queues.
    ///
    /// # Safety
    ///
    /// - If the pipeline stages include geometry or tessellation stages, then the corresponding
    ///   features must have been enabled.
    /// - There are certain rules regarding the pipeline barriers inside render passes.
    /// - The buffer must be alive for at least as long as the command buffer to which this barrier
    ///   is added.
    /// - Queue ownership transfers must be correct.
    ///
    pub unsafe fn add_buffer_memory_barrier<B>
                  (&mut self, buffer: B, source_stage: PipelineStages,
                   source_access: AccessFlagBits, dest_stage: PipelineStages,
                   dest_access: AccessFlagBits, by_region: bool,
                   queue_transfer: Option<(u32, u32)>)
        where B: Buffer
    {
        self.add_execution_dependency(source_stage, dest_stage, by_region);

        let size = buffer.size();
        let BufferInner { buffer, offset } = buffer.inner();

        let (src_queue, dest_queue) = if let Some((src_queue, dest_queue)) = queue_transfer {
            (src_queue, dest_queue)
        } else {
            (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
        };

        self.buffer_barriers.push(vk::BufferMemoryBarrier {
            sType: vk::STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            pNext: ptr::null(),
            srcAccessMask: source_access.into(),
            dstAccessMask: dest_access.into(),
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
    /// # Safety
    ///
    /// - If the pipeline stages include geometry or tessellation stages, then the corresponding
    ///   features must have been enabled.
    /// - There are certain rules regarding the pipeline barriers inside render passes.
    /// - The buffer must be alive for at least as long as the command buffer to which this barrier
    ///   is added.
    /// - Queue ownership transfers must be correct.
    /// - Image layouts transfers must be correct.
    /// - Access flags must be compatible with the image usage flags passed at image creation.
    ///
    pub unsafe fn add_image_memory_barrier<I>(&mut self, image: &Arc<I>, mipmaps: Range<u32>,
                  layers: Range<u32>, source_stage: PipelineStages, source_access: AccessFlagBits,
                  dest_stage: PipelineStages, dest_access: AccessFlagBits, by_region: bool,
                  queue_transfer: Option<(u32, u32)>, current_layout: Layout, new_layout: Layout)
        where I: Image
    {
        self.add_execution_dependency(source_stage, dest_stage, by_region);

        debug_assert!(mipmaps.start < mipmaps.end);
        // TODO: debug_assert!(mipmaps.end <= image.mipmap_levels());
        debug_assert!(layers.start < layers.end);
        debug_assert!(layers.end <= image.dimensions().array_layers());

        let (src_queue, dest_queue) = if let Some((src_queue, dest_queue)) = queue_transfer {
            (src_queue, dest_queue)
        } else {
            (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
        };

        self.image_barriers.push(vk::ImageMemoryBarrier {
            sType: vk::STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            pNext: ptr::null(),
            srcAccessMask: source_access.into(),
            dstAccessMask: dest_access.into(),
            oldLayout: current_layout as u32,
            newLayout: new_layout as u32,
            srcQueueFamilyIndex: src_queue,
            dstQueueFamilyIndex: dest_queue,
            image: image.inner().internal_object(),
            subresourceRange: vk::ImageSubresourceRange {
                aspectMask: 1 | 2 | 4 | 8,      // FIXME: wrong
                baseMipLevel: mipmaps.start,
                levelCount: mipmaps.end - mipmaps.start,
                baseArrayLayer: layers.start,
                layerCount: layers.end - layers.start,
            },
        });
    }
}

pub struct UnsafeCommandBuffer<P> where P: CommandPool {
    // The Vulkan command buffer.
    cmd: vk::CommandBuffer,

    // Device that owns the command buffer.
    device: Arc<Device>,

    // Pool that owns the command buffer.
    pool: P::Finished,

    // Flags that were used at creation.
    flags: Flags,

    // True if the command buffer has always been submitted once. Only relevant if `flags` is
    // `OneTimeSubmit`.
    already_submitted: AtomicBool,

    // True if we are a secondary command buffer.
    secondary_cb: bool,
}

impl<P> UnsafeCommandBuffer<P> where P: CommandPool {
    /// Returns the device used to create this command buffer.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

// Since the only moment where we access the pool is in the `Drop` trait, we can safely implement
// `Sync` on the command buffer.
// TODO: this could be generalized with a general-purpose wrapper that only allows &mut access
unsafe impl<P> Sync for UnsafeCommandBuffer<P> where P: CommandPool {}

unsafe impl<P> VulkanObject for UnsafeCommandBuffer<P> where P: CommandPool {
    type Object = vk::CommandBuffer;

    #[inline]
    fn internal_object(&self) -> vk::CommandBuffer {
        self.cmd
    }
}

impl<P> Drop for UnsafeCommandBuffer<P> where P: CommandPool {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            self.pool.free(self.secondary_cb, Some(self.cmd.into()).into_iter());
        }
    }
}
