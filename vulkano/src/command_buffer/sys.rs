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
use std::ptr;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use smallvec::SmallVec;

use buffer::Buffer;
use buffer::BufferSlice;
use command_buffer::pool::AllocatedCommandBuffer;
use command_buffer::pool::CommandPool;
use device::Device;
use framebuffer::RenderPass;
use framebuffer::Framebuffer;
use framebuffer::Subpass;

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
            (subpass.render_pass().render_pass().internal_object(), subpass.index())
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

    /// Fills a buffer by repeating a 32 bits data.
    ///
    /// This is similar to the `memset` function in C/C++.
    ///
    /// # Panic
    ///
    /// - Panicks if the buffer was not created with the same device as this command buffer.
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
        assert_eq!(buffer.buffer().inner_buffer().device().internal_object(),
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
        vk.CmdFillBuffer(cmd, buffer.buffer().inner_buffer().internal_object(),
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
    /// - Panicks if the buffer was not created with the same device as this command buffer.
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
        assert_eq!(buffer.buffer().inner_buffer().device().internal_object(),
                   self.device.internal_object());

        let size = cmp::min(buffer.size(), mem::size_of_val(data));

        debug_assert_eq!(buffer.offset() % 4, 0);
        debug_assert_eq!(size % 4, 0);
        debug_assert!(size <= 65536);

        let vk = self.device.pointers();
        let cmd = self.cmd.take().unwrap();
        vk.CmdUpdateBuffer(cmd, buffer.buffer().inner_buffer().internal_object(),
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
    /// - Panicks if one of the buffers was not created with the same device as this
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
        assert_eq!(src.inner_buffer().device().internal_object(),
                   self.device.internal_object());
        assert_eq!(src.inner_buffer().device().internal_object(),
                   dest.inner_buffer().device().internal_object());

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
        vk.CmdCopyBuffer(cmd, src.inner_buffer().internal_object(),
                         dest.inner_buffer().internal_object(), regions.len() as u32,
                         regions.as_ptr());
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
