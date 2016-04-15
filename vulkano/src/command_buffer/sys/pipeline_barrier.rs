// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ops::Range;
use std::ptr;
use std::sync::Arc;
use smallvec::SmallVec;

use buffer::Buffer;
use buffer::BufferSlice;
use command_buffer::pool::CommandPool;
use command_buffer::sys::CommandPrototype;
use command_buffer::sys::KeepAlive;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use image::Image;
use image::sys::Layout;
use sync::AccessFlagBits;
use sync::PipelineStages;

use VulkanObject;
use VulkanPointers;
use vk;

/// Prototype for a command that fills a buffer with data.
pub struct PipelineBarrierCommand {
    keep_alive: SmallVec<[Arc<KeepAlive + 'static>; 16]>,

    device: Option<vk::Device>,

    src_stage_mask: vk::PipelineStageFlags,
    dst_stage_mask: vk::PipelineStageFlags,
    dependency_flags: vk::DependencyFlags,
    memory_barriers: SmallVec<[vk::MemoryBarrier; 2]>,
    buffer_barriers: SmallVec<[vk::BufferMemoryBarrier; 8]>,
    image_barriers: SmallVec<[vk::ImageMemoryBarrier; 8]>,

    requires_geometry_shader_feature: bool,
    requires_tessellation_shader_feature: bool,
}

impl PipelineBarrierCommand {
    /// Adds a command that adds a pipeline barrier to a command buffer.
    pub fn new() -> PipelineBarrierCommand {
        PipelineBarrierCommand {
            keep_alive: SmallVec::new(),
            device: None,
            src_stage_mask: 0,
            dst_stage_mask: 0,
            dependency_flags: vk::DEPENDENCY_BY_REGION_BIT,
            memory_barriers: SmallVec::new(),
            buffer_barriers: SmallVec::new(),
            image_barriers: SmallVec::new(),
            requires_geometry_shader_feature: false,
            requires_tessellation_shader_feature: false,
        }
    }

    /// Adds an execution dependency. This means that all the stages in `source` of the previous
    /// commands must finish before any of the stages in `dest` of the following commands can start.
    #[inline]
    pub fn add_execution_dependency(&mut self, source: PipelineStages, dest: PipelineStages,
                                    by_region: bool)
    {
        if source.geometry_shader || dest.geometry_shader {
            self.requires_geometry_shader_feature = true;
        }

        if source.tessellation_control_shader || dest.tessellation_control_shader ||
           source.tessellation_evaluation_shader || dest.tessellation_evaluation_shader
        {
            self.requires_tessellation_shader_feature = true;
        }

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
    pub fn add_memory_barrier(&mut self, source_stage: PipelineStages,
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
    pub fn add_buffer_memory_barrier<'a, T: ?Sized, B>
           (&mut self, buffer: BufferSlice<'a, T, B>, source_stage: PipelineStages,
            source_access: AccessFlagBits, dest_stage: PipelineStages, dest_access: AccessFlagBits,
            by_region: bool, queue_transfer: Option<(u32, u32)>)
        where B: Buffer
    {
        // FIXME: document panics or replace with errors

        self.add_execution_dependency(source_stage, dest_stage, by_region);

        debug_assert!(buffer.size() + buffer.offset() <= buffer.buffer().size());

        let (src_queue, dest_queue) = if let Some((src_queue, dest_queue)) = queue_transfer {
            assert!(buffer.buffer().exclusive_sharing_mode());
            // FIXME: check that we are being executed on one of these queues
            // FIXME: check validity of the queue families
            (src_queue, dest_queue)
        } else {
            (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
        };

        self.keep_alive.push(buffer.buffer().clone() as Arc<_>);

        self.buffer_barriers.push(vk::BufferMemoryBarrier {
            sType: vk::STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            pNext: ptr::null(),
            srcAccessMask: source_access.into(),
            dstAccessMask: dest_access.into(),
            srcQueueFamilyIndex: src_queue,
            dstQueueFamilyIndex: dest_queue,
            buffer: buffer.buffer().inner_buffer().internal_object(),
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
    /// - Image layouts validity is not checked.
    ///
    pub unsafe fn add_image_memory_barrier<I>(&mut self, image: &Arc<I>, mipmaps: Range<u32>,
                  layers: Range<u32>, source_stage: PipelineStages,
                  source_access: AccessFlagBits, dest_stage: PipelineStages, dest_access: AccessFlagBits,
                  by_region: bool, queue_transfer: Option<(u32, u32)>, current_layout: Layout,
                  new_layout: Layout)
        where I: Image + 'static + Send + Sync
    {
        // FIXME: document panics or replace with errors

        self.add_execution_dependency(source_stage, dest_stage, by_region);

        assert!(mipmaps.start < mipmaps.end);
        assert!(mipmaps.end <= image.mipmap_levels());
        assert!(layers.start < layers.end);
        assert!(layers.end <= image.dimensions().array_layers());

        let (src_queue, dest_queue) = if let Some((src_queue, dest_queue)) = queue_transfer {
            assert!(image.exclusive_sharing_mode());
            // FIXME: check that we are being executed on one of these queues
            // FIXME: check validity of the queue families
            (src_queue, dest_queue)
        } else {
            (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
        };

        self.keep_alive.push(image.clone() as Arc<_>);

        self.image_barriers.push(vk::ImageMemoryBarrier {
            sType: vk::STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            pNext: ptr::null(),
            srcAccessMask: source_access.into(),
            dstAccessMask: dest_access.into(),
            oldLayout: current_layout as u32,
            newLayout: new_layout as u32,
            srcQueueFamilyIndex: src_queue,
            dstQueueFamilyIndex: dest_queue,
            image: image.inner_image().internal_object(),
            subresourceRange: vk::ImageSubresourceRange {
                aspectMask: 1 | 2 | 4 | 8,      // FIXME: wrong
                baseMipLevel: mipmaps.start,
                levelCount: mipmaps.end - mipmaps.start,
                baseArrayLayer: layers.start,
                layerCount: layers.end - layers.start,
            },
        });
    }

    /// Submits the command to the command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks one of the objects in the barrier was not allocated with the same device as
    ///   the command buffer.
    ///
    pub fn submit<P>(&mut self, mut cb: UnsafeCommandBufferBuilder<P>)
                     -> UnsafeCommandBufferBuilder<P>
        where P: CommandPool
    {
        unsafe {
            // If barrier is empty, don't do anything.
            if self.src_stage_mask == 0 || self.dst_stage_mask == 0 {
                debug_assert!(self.src_stage_mask == 0 && self.dst_stage_mask == 0);
                debug_assert!(self.memory_barriers.is_empty());
                debug_assert!(self.buffer_barriers.is_empty());
                debug_assert!(self.image_barriers.is_empty());
                debug_assert!(self.keep_alive.is_empty());
                return cb;
            }

            // FIXME: check requires_geometry_shader_feature and requires_tessellation_shader_feature

            // If we are inside a render pass, there are additional constraints that need to be
            // checked.
            if cb.within_render_pass {
                // FIXME: see section 6.5.1 of specs
            }

            if let Some(device) = self.device {
                assert_eq!(device, cb.device().internal_object());
            }

            for ka in self.keep_alive.into_iter() {
                cb.keep_alive.push(ka);
            }

            let _pool_lock = cb.pool().lock();

            {
                let vk = cb.device.pointers();
                let cmd = cb.cmd.clone().unwrap();
                vk.CmdPipelineBarrier(cmd, self.src_stage_mask, self.dst_stage_mask,
                                      self.dependency_flags, self.memory_barriers.len() as u32,
                                      self.memory_barriers.as_ptr(),
                                      self.buffer_barriers.len() as u32,
                                      self.buffer_barriers.as_ptr(),
                                      self.image_barriers.len() as u32,
                                      self.image_barriers.as_ptr());
            }

            cb
        }
    }
}

unsafe impl CommandPrototype for PipelineBarrierCommand {
    unsafe fn submit<P>(&mut self, cb: UnsafeCommandBufferBuilder<P>)
                        -> UnsafeCommandBufferBuilder<P>
        where P: CommandPool
    {
        self.submit(cb)
    }
}
