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
use smallvec::SmallVec;

use buffer::Buffer;
use buffer::BufferSlice;
use command_buffer::pool::AllocatedCommandBuffer;
use command_buffer::pool::CommandPool;
use device::Device;
use format::ClearValue;
use format::FormatTy;
use framebuffer::RenderPass;
use framebuffer::Framebuffer;
use framebuffer::Subpass;
use image::Image;
use image::sys::Layout;
use image::sys::UnsafeImage;
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
    pub fn new<R, Rf>(pool: P, kind: Kind<R, Rf>, flags: Flags)
                      -> Result<UnsafeCommandBufferBuilder<P>, OomError>
        where R: RenderPass + 'static + Send + Sync,
              Rf: RenderPass + 'static + Send + Sync
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
    pub unsafe fn already_allocated<R, Rf>(pool: P, cmd: AllocatedCommandBuffer,
                                           kind: Kind<R, Rf>, flags: Flags)
                                           -> Result<UnsafeCommandBufferBuilder<P>, OomError>
        where R: RenderPass + 'static + Send + Sync,
              Rf: RenderPass + 'static + Send + Sync
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

        let (rp, sp) = if let Kind::SecondaryRenderPass { subpass, .. } = kind {
            (subpass.render_pass().inner().internal_object(), subpass.index())
        } else {
            (0, 0)
        };

        let framebuffer = if let Kind::SecondaryRenderPass { subpass, framebuffer: Some(ref framebuffer) } = kind {
            assert!(framebuffer.is_compatible_with(subpass.render_pass()));     // TODO: proper error
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
    /// - Panics if the mipmap levels range or the array layers range is invalid, ie. if the end
    ///   is inferior to the start.
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
            assert!(range.mipmap_levels.start <= range.mipmap_levels.end);
            assert!(range.array_layers.start <= range.array_layers.end);
            debug_assert!(range.mipmap_levels.end <= image.mipmap_levels());
            debug_assert!(range.array_layers.end <= image.dimensions().array_layers());

            if range.mipmap_levels.start == range.mipmap_levels.end {
                return None;
            }

            if range.array_layers.start == range.array_layers.end {
                return None;
            }

            Some(vk::ImageSubresourceRange {
                aspectMask: vk::IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel: range.mipmap_levels.start,
                levelCount: range.mipmap_levels.end - range.mipmap_levels.start,
                baseArrayLayer: range.array_layers.start,
                layerCount: range.array_layers.end - range.array_layers.start,
            })
        }).collect();

        // Do nothing if no range to clear.
        if ranges.is_empty() {
            return;
        }

        let layout = if general_layout { vk::IMAGE_LAYOUT_GENERAL }
                     else { vk::IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL };

        let vk = self.device.pointers();
        let cmd = self.cmd.take().unwrap();
        vk.CmdClearColorImage(cmd, image.internal_object(), layout, &clear_value,
                              ranges.len() as u32, ranges.as_ptr());
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
    pub unsafe fn fill_buffer<T: ?Sized, B>(&mut self, buffer: BufferSlice<T, B>, data: u32)
        where B: Buffer
    {
        assert_eq!(buffer.buffer().inner().device().internal_object(),
                   self.device.internal_object());

        debug_assert_eq!(buffer.offset() % 4, 0);

        let size = if buffer.offset() + buffer.size() == buffer.buffer().size() {
            vk::WHOLE_SIZE
        } else {
            debug_assert_eq!(buffer.size() % 4, 0);
            buffer.size() as vk::DeviceSize
        };

        let vk = self.device.pointers();
        let cmd = self.cmd.take().unwrap();
        vk.CmdFillBuffer(cmd, buffer.buffer().inner().internal_object(),
                         buffer.offset() as vk::DeviceSize,
                         buffer.size() as vk::DeviceSize, data);
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
    // TODO: does the data have to be 4-bytes aligned? https://github.com/KhronosGroup/Vulkan-Docs/issues/263
    pub unsafe fn update_buffer<T: ?Sized, B, D: ?Sized>(&mut self, buffer: BufferSlice<T, B>,
                                                         data: &D)
        where B: Buffer, D: Copy + 'static
    {
        assert_eq!(buffer.buffer().inner().device().internal_object(),
                   self.device.internal_object());

        let size = cmp::min(buffer.size(), mem::size_of_val(data));

        debug_assert_eq!(buffer.offset() % 4, 0);
        debug_assert_eq!(size % 4, 0);
        debug_assert!(size <= 65536);

        let vk = self.device.pointers();
        let cmd = self.cmd.take().unwrap();
        vk.CmdUpdateBuffer(cmd, buffer.buffer().inner().internal_object(),
                           buffer.offset() as vk::DeviceSize, size as vk::DeviceSize,
                           data as *const D as *const _);
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
    pub unsafe fn copy_buffer<Bs, Bd, I>(&mut self, src: &Arc<Bs>, dest: &Arc<Bd>, regions: I)
        where Bs: Buffer,
              Bd: Buffer,
              I: IntoIterator<Item = BufferCopyRegion>
    {
        assert_eq!(src.inner().device().internal_object(),
                   self.device.internal_object());
        assert_eq!(src.inner().device().internal_object(),
                   dest.inner().device().internal_object());

        let regions: SmallVec<[_; 4]> = {
            let mut res = SmallVec::new();
            for region in regions.into_iter() {
                if region.size == 0 { continue; }

                res.push(vk::BufferCopy {
                    srcOffset: region.source_offset as vk::DeviceSize,
                    dstOffset: region.destination_offset as vk::DeviceSize,
                    size: region.size as vk::DeviceSize,
                });
            }
            res
        };

        if regions.is_empty() {
            return;
        }

        let vk = self.device.pointers();
        let cmd = self.cmd.take().unwrap();
        vk.CmdCopyBuffer(cmd, src.inner().internal_object(),
                         dest.inner().internal_object(), regions.len() as u32,
                         regions.as_ptr());
    }

    /// Adds a pipeline barrier to the command buffer.
    ///
    /// This function itself is not unsafe, but creating a pipeline barrier builder is.
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
        let cmd = self.cmd.take().unwrap();

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
#[derive(Clone)]        // TODO: Debug
pub enum Kind<'a, R: 'a, Rf> {
    /// A primary command buffer can execute all commands and can call secondary command buffers.
    Primary,

    /// A secondary command buffer can execute all dispatch and transfer operations, but not
    /// drawing operations.
    Secondary,

    /// A secondary command buffer within a render pass can only call draw operations that can
    /// be executed from within a specific subpass.
    SecondaryRenderPass {
        /// Which subpass this secondary command buffer can be called from.
        subpass: Subpass<'a, R>,
        /// The framebuffer object that will be used when calling the command buffer.
        /// This parameter is optional and is an optimization hint for the implementation.
        framebuffer: Option<Arc<Framebuffer<Rf>>>,
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
    pub mipmap_levels: Range<u32>,
    pub array_layers: Range<u32>,
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
    pub unsafe fn add_buffer_memory_barrier<'a, T: ?Sized, B>
                  (&mut self, buffer: BufferSlice<'a, T, B>, source_stage: PipelineStages,
                   source_access: AccessFlagBits, dest_stage: PipelineStages,
                   dest_access: AccessFlagBits, by_region: bool,
                   queue_transfer: Option<(u32, u32)>)
        where B: Buffer
    {
        self.add_execution_dependency(source_stage, dest_stage, by_region);

        debug_assert!(buffer.size() + buffer.offset() <= buffer.buffer().size());

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
            buffer: buffer.buffer().inner().internal_object(),
            offset: buffer.offset() as vk::DeviceSize,
            size: buffer.size() as vk::DeviceSize,
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

// Since the only moment where we access the pool is in the `Drop` trait, we can safely implement
// `Sync` on the command buffer.
// TODO: this could be generalized with a general-purpose wrapper that only allows &mut access
unsafe impl<P> Sync for UnsafeCommandBuffer<P> where P: CommandPool {}
