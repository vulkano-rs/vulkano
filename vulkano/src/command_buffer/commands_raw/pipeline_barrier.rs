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

use buffer::BufferAccess;
use buffer::BufferInner;
use command_buffer::CommandAddError;
use command_buffer::cb::AddCommand;
use command_buffer::cb::UnsafeCommandBufferBuilder;
use command_buffer::pool::CommandPool;
use image::ImageAccess;
use image::ImageLayout;
use sync::AccessFlagBits;
use sync::PipelineStages;

use device::DeviceOwned;
use VulkanObject;
use VulkanPointers;
use vk;

/// Command that adds a pipeline barrier to a command buffer builder.
///
/// A pipeline barrier is a low-level system-ish command that is often necessary for safety. By
/// default all commands that you add to a command buffer can potentially run simultaneously.
/// Adding a pipeline barrier separates commands before the barrier from commands after the barrier
/// and prevents them from running simultaneously.
///
/// Please take a look at the Vulkan specifications for more information. Pipeline barriers are a
/// complex topic and explaining them in this documentation would be redundant.
///
/// > **Note**: We use a builder-like API here so that users can pass multiple buffers or images of
/// > multiple different types. Doing so with a single function would be very tedious in terms of
/// > API.
pub struct CmdPipelineBarrier<'a> {
    src_stage_mask: vk::PipelineStageFlags,
    dst_stage_mask: vk::PipelineStageFlags,
    dependency_flags: vk::DependencyFlags,
    memory_barriers: SmallVec<[vk::MemoryBarrier; 2]>,
    buffer_barriers: SmallVec<[vk::BufferMemoryBarrier; 8]>,
    image_barriers: SmallVec<[vk::ImageMemoryBarrier; 8]>,
    marker: PhantomData<&'a ()>,
}

impl<'a> CmdPipelineBarrier<'a> {
    /// Creates a new empty pipeline barrier command.
    #[inline]
    pub fn new() -> CmdPipelineBarrier<'a> {
        CmdPipelineBarrier {
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

    /// Merges another pipeline builder into this one.
    #[inline]
    pub fn merge(&mut self, other: CmdPipelineBarrier<'a>) {
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
    ///   features must have been enabled in the device.
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
    /// Also adds an execution dependency similar to `add_execution_dependency`.
    ///
    /// # Safety
    ///
    /// - Same as `add_execution_dependency`.
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
    /// Also adds an execution dependency similar to `add_execution_dependency`.
    ///
    /// Also allows transfering buffer ownership between queues.
    ///
    /// # Safety
    ///
    /// - Same as `add_execution_dependency`.
    /// - The buffer must be alive for at least as long as the command buffer to which this barrier
    ///   is added.
    /// - Queue ownership transfers must be correct.
    ///
    pub unsafe fn add_buffer_memory_barrier<B: ?Sized>
                  (&mut self, buffer: &'a B, source_stage: PipelineStages,
                   source_access: AccessFlagBits, dest_stage: PipelineStages,
                   dest_access: AccessFlagBits, by_region: bool,
                   queue_transfer: Option<(u32, u32)>, offset: usize, size: usize)
        where B: BufferAccess
    {
        self.add_execution_dependency(source_stage, dest_stage, by_region);

        debug_assert!(size <= buffer.size());
        let BufferInner { buffer, offset: org_offset } = buffer.inner();
        let offset = offset + org_offset;

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
    /// Also adds an execution dependency similar to `add_execution_dependency`.
    ///
    /// # Safety
    ///
    /// - Same as `add_execution_dependency`.
    /// - The buffer must be alive for at least as long as the command buffer to which this barrier
    ///   is added.
    /// - Queue ownership transfers must be correct.
    /// - Image layouts transfers must be correct.
    /// - Access flags must be compatible with the image usage flags passed at image creation.
    ///
    pub unsafe fn add_image_memory_barrier<I: ?Sized>(&mut self, image: &'a I, mipmaps: Range<u32>,
                  layers: Range<u32>, source_stage: PipelineStages, source_access: AccessFlagBits,
                  dest_stage: PipelineStages, dest_access: AccessFlagBits, by_region: bool,
                  queue_transfer: Option<(u32, u32)>, current_layout: ImageLayout, new_layout: ImageLayout)
        where I: ImageAccess
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

unsafe impl<'a, P> AddCommand<&'a CmdPipelineBarrier<'a>> for UnsafeCommandBufferBuilder<P>
    where P: CommandPool
{
    type Out = UnsafeCommandBufferBuilder<P>;

    #[inline]
    fn add(self, command: &'a CmdPipelineBarrier<'a>) -> Result<Self::Out, CommandAddError> {
        // If barrier is empty, don't do anything.
        if command.src_stage_mask == 0 || command.dst_stage_mask == 0 {
            debug_assert!(command.src_stage_mask == 0 && command.dst_stage_mask == 0);
            debug_assert!(command.memory_barriers.is_empty());
            debug_assert!(command.buffer_barriers.is_empty());
            debug_assert!(command.image_barriers.is_empty());
            return Ok(self);
        }

        unsafe {
            let vk = self.device().pointers();
            let cmd = self.internal_object();

            vk.CmdPipelineBarrier(cmd, command.src_stage_mask, command.dst_stage_mask,
                                  command.dependency_flags, command.memory_barriers.len() as u32,
                                  command.memory_barriers.as_ptr(),
                                  command.buffer_barriers.len() as u32,
                                  command.buffer_barriers.as_ptr(),
                                  command.image_barriers.len() as u32,
                                  command.image_barriers.as_ptr());
        }

        Ok(self)
    }
}
