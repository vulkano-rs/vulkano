// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::marker::PhantomData;
use std::ops::Range;
use std::ptr;
use std::u32;
use smallvec::SmallVec;

use buffer::Buffer;
use buffer::BufferInner;
use command_buffer::CommandsListSink;
use command_buffer::RawCommandBufferPrototype;
use image::Image;
use image::Layout;
use sync::AccessFlagBits;
use sync::PipelineStages;

use VulkanObject;
use VulkanPointers;
use vk;

/// Prototype for a pipeline barrier that's going to be added to a command buffer builder.
///
/// Note: we use a builder-like API here so that users can pass multiple buffers or images of
/// multiple different types. Doing so with a single function would be very tedious in terms of
/// API.
pub struct PipelineBarrierBuilder<'a> {
    src_stage_mask: vk::PipelineStageFlags,
    dst_stage_mask: vk::PipelineStageFlags,
    dependency_flags: vk::DependencyFlags,
    memory_barriers: SmallVec<[vk::MemoryBarrier; 2]>,
    buffer_barriers: SmallVec<[vk::BufferMemoryBarrier; 8]>,
    image_barriers: SmallVec<[vk::ImageMemoryBarrier; 8]>,
    marker: PhantomData<&'a ()>,
}

impl<'a> PipelineBarrierBuilder<'a> {
    #[inline]
    pub fn new() -> PipelineBarrierBuilder<'a> {
        PipelineBarrierBuilder {
            src_stage_mask: 0,
            dst_stage_mask: 0,
            dependency_flags: vk::DEPENDENCY_BY_REGION_BIT,
            memory_barriers: SmallVec::new(),
            buffer_barriers: SmallVec::new(),
            image_barriers: SmallVec::new(),
            marker: PhantomData,
        }
    }

    /// Returns true if no barrier or execution dependency has been added yet.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.src_stage_mask == 0 || self.dst_stage_mask == 0
    }

    /// Adds a pipeline barrier to the command buffer.
    ///
    /// This function itself is not unsafe, but creating a pipeline barrier builder is.
    #[inline]
    pub fn append_to(self, builder: &mut CommandsListSink<'a>) {
        // If barrier is empty, don't do anything.
        if self.src_stage_mask == 0 || self.dst_stage_mask == 0 {
            debug_assert!(self.src_stage_mask == 0 && self.dst_stage_mask == 0);
            debug_assert!(self.memory_barriers.is_empty());
            debug_assert!(self.buffer_barriers.is_empty());
            debug_assert!(self.image_barriers.is_empty());
            return;
        }

        builder.add_command(Box::new(move |raw: &mut RawCommandBufferPrototype| {
            unsafe {
                let vk = raw.device.pointers();
                let cmd = raw.command_buffer.clone().take().unwrap();

                vk.CmdPipelineBarrier(cmd, self.src_stage_mask, self.dst_stage_mask,
                                      self.dependency_flags, self.memory_barriers.len() as u32,
                                      self.memory_barriers.as_ptr(),
                                      self.buffer_barriers.len() as u32,
                                      self.buffer_barriers.as_ptr(),
                                      self.image_barriers.len() as u32,
                                      self.image_barriers.as_ptr());
            }
        }));
    }

    /// Merges another pipeline builder into this one.
    #[inline]
    pub fn merge(&mut self, other: PipelineBarrierBuilder<'a>) {
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
    pub unsafe fn add_buffer_memory_barrier<B: ?Sized>
                  (&mut self, buffer: &'a B, source_stage: PipelineStages,
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
    pub unsafe fn add_image_memory_barrier<I: ?Sized>(&mut self, image: &'a I, mipmaps: Range<u32>,
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
