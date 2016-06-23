// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::mem;
use std::ptr;
use std::sync::Arc;
use smallvec::SmallVec;

use command_buffer::pool::CommandPool;
use command_buffer::sys::KeepAlive;
use command_buffer::sys::UnsafeCommandBufferBuilder;
use descriptor::descriptor_set::DescriptorSetsCollection;
use descriptor::pipeline_layout::PipelineLayout;
use descriptor::pipeline_layout::PipelineLayoutSetsCompatible;
use descriptor::pipeline_layout::PipelineLayoutPushConstantsCompatible;

use VulkanObject;
use VulkanPointers;
use vk;

/// Prototype for a command that binds an index buffer.
pub struct DescriptorSetsBindCommand {
    // Buffer to keep alive.
    keep_alive: SmallVec<[Arc<KeepAlive + 'static>; 8]>,

    // The device of the buffer, or 0 if the list of buffers is empty.
    device: vk::Device,

    bind_point: vk::PipelineBindPoint,

    layout: vk::PipelineLayout,

    descriptor_sets: SmallVec<[vk::DescriptorSet; 8]>,
    dynamic_offsets: SmallVec<[u32; 4]>,

    pc_stage_flags: vk::ShaderStageFlags,
    push_constants: Vec<u8>,
}

impl DescriptorSetsBindCommand {
    /// Builds a new command that will bind the given sets and push constants.
    ///
    /// If `graphics` is true, then the graphics pipeline bind point is used. Otherwise it is the
    /// compute pipeline bind point.
    ///
    /// This function panicks if the push constants do not respect the limitations of the device,
    /// but only after checking that the push constants are compatible with the pipeline layout.
    /// If such a panic occurs, it means that the pipeline layout did not do a good job of
    /// enforcing these limitations.
    ///
    /// # Panic
    ///
    /// - Panicks if the sets and the pipeline layout were not created with the same device.
    /// - Panicks if the size of `push_constants` is not a multiple of 4.
    /// - Panicks if the size of `push_constants` exceeds `maxPushConstantsSize`.
    ///
    pub fn new<L, S, P>(graphics: bool, pipeline_layout: &L, sets: S, push_constants: &P)
                        -> Result<DescriptorSetsBindCommand, DescriptorSetsBindError>
        where L: PipelineLayout + PipelineLayoutSetsCompatible<S> +
                 PipelineLayoutPushConstantsCompatible<P>,
              S: DescriptorSetsCollection,
              P: Copy
    {
        if !PipelineLayoutSetsCompatible::is_compatible(pipeline_layout, &sets) {
            return Err(DescriptorSetsBindError::IncompatibleSet);
        }

        if !PipelineLayoutPushConstantsCompatible::is_compatible(pipeline_layout, push_constants) {
            return Err(DescriptorSetsBindError::IncompatiblePushConstants);
        }

        let bind_point = if graphics { vk::PIPELINE_BIND_POINT_GRAPHICS }
                         else { vk::PIPELINE_BIND_POINT_COMPUTE };

        let device = pipeline_layout.inner_pipeline_layout().device().internal_object();

        let mut keep_alive = SmallVec::new();
        let mut descriptor_sets = SmallVec::new();
        let /*mut*/ dynamic_offsets = SmallVec::new();      // TODO: not implemented

        for set in sets.list() {
            assert_eq!(set.inner_descriptor_set().layout().device().internal_object(), device);
            descriptor_sets.push(set.inner_descriptor_set().internal_object());
            keep_alive.push(unsafe { mem::transmute(set) });       // FIXME: meh
        }

        let push_constants = unsafe {
            let size = mem::size_of_val(push_constants);
            assert!(size % 4 == 0);
            assert!(size >= pipeline_layout.inner_pipeline_layout().device().physical_device()
                                           .limits().max_push_constants_size() as usize);

            let mut pc_buf = Vec::with_capacity(size);
            ptr::copy_nonoverlapping(push_constants as *const P, pc_buf.as_mut_ptr() as *mut P, 1);
            pc_buf.set_len(size);
            pc_buf
        };

        Ok(DescriptorSetsBindCommand {
            keep_alive: keep_alive,
            device: device,
            bind_point: bind_point,
            layout: pipeline_layout.inner_pipeline_layout().internal_object(),
            descriptor_sets: descriptor_sets,
            dynamic_offsets: dynamic_offsets,
            pc_stage_flags: vk::SHADER_STAGE_ALL,       // TODO:
            push_constants: push_constants,
        })
    }

    /// Submits the command to the command buffer.
    ///
    /// # Panic
    ///
    /// - Panicks if the sets were not allocated with the same device as the command buffer.
    /// - Panicks if the queue doesn't not support graphics/compute operations depending on the
    ///   bind point.
    ///
    pub fn submit<P>(&mut self, mut cb: UnsafeCommandBufferBuilder<P>)
                     -> UnsafeCommandBufferBuilder<P>
        where P: CommandPool
    {
        unsafe {
            let _pool_lock = cb.pool().lock();

            // Various checks.
            // Note that surprisingly the specs allow this function to be called from outside a
            // render pass.
            assert_eq!(self.device, cb.device().internal_object());
            if self.bind_point == vk::PIPELINE_BIND_POINT_GRAPHICS {
                assert!(cb.pool().queue_family().supports_graphics());
            } else if self.bind_point == vk::PIPELINE_BIND_POINT_COMPUTE {
                assert!(cb.pool().queue_family().supports_compute());
            }

            for ka in self.keep_alive.into_iter() {
                cb.keep_alive.push(ka);
            }

            // Now binding.
            {
                let vk = cb.device.pointers();
                let cmd = cb.cmd.clone().unwrap();

                // TODO: For the moment we rebind the descriptor sets every time. In practice we
                //       should cache the descriptor sets in the `UnsafeCommandBufferBuilder` and
                //       bind only what is necessary. Dynamic offsets make this difficult.

                if !self.descriptor_sets.is_empty() {
                    vk.CmdBindDescriptorSets(cmd, self.bind_point, self.layout, 0,
                                             self.descriptor_sets.len() as u32,
                                             self.descriptor_sets.as_ptr(),
                                             self.dynamic_offsets.len() as u32,
                                             self.dynamic_offsets.as_ptr());
                }

                if !self.push_constants.is_empty() {
                    vk.CmdPushConstants(cmd, self.layout, self.pc_stage_flags, 0,
                                        self.push_constants.len() as u32,
                                        self.push_constants.as_ptr() as *const _);
                }
            }

            cb
        }
    }
}

error_ty!{DescriptorSetsBindError => "Error that can happen when binding descriptor sets.",
    IncompatibleSet => "one of the sets is not compatible with the pipeline layout",
    IncompatiblePushConstants => "the push constants are not compatible with the pipeline layout",
}
