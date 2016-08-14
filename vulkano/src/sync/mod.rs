// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Synchronization primitives for Vulkan objects.
//! 
//! In Vulkan, you have to manually ensure two things:
//! 
//! - That a buffer or an image are not read and written simultaneously (similarly to the CPU).
//! - That writes to a buffer or an image are propagated to other queues by inserting memory
//!   barriers.
//!
//! But don't worry ; this is automatically enforced by this library (as long as you don't use
//! any unsafe function). See the `memory` module for more info.
//!

use std::ops;
use std::sync::Arc;
use device::Queue;
use vk;

pub use self::event::Event;
pub use self::fence::Fence;
pub use self::fence::FenceWaitError;
pub use self::semaphore::Semaphore;

mod event;
mod fence;
mod semaphore;

/// Base trait for objects that can be used as resources and must be synchronized.
// TODO: remove
pub unsafe trait Resource {
    /// Returns in which queue family or families this resource can be used.
    fn sharing_mode(&self) -> &SharingMode;

    /// Returns true if the `gpu_access` function should be passed a fence.
    #[inline]
    fn requires_fence(&self) -> bool {
        true
    }

    /// Returns true if the `gpu_access` function should be passed a semaphore.
    #[inline]
    fn requires_semaphore(&self) -> bool {
        true
    }
}

/// Declares in which queue(s) a resource can be used.
///
/// When you create a buffer or an image, you have to tell the Vulkan library in which queue
/// families it will be used. The vulkano library requires you to tell in which queue famiily
/// the resource will be used, even for exclusive mode.
#[derive(Debug, Clone, PartialEq, Eq)]
// TODO: remove
pub enum SharingMode {
    /// The resource is used is only one queue family.
    Exclusive(u32),
    /// The resource is used in multiple queue families. Can be slower than `Exclusive`.
    Concurrent(Vec<u32>),       // TODO: Vec is too expensive here
}

impl<'a> From<&'a Arc<Queue>> for SharingMode {
    #[inline]
    fn from(queue: &'a Arc<Queue>) -> SharingMode {
        SharingMode::Exclusive(queue.family().id())
    }
}

impl<'a> From<&'a [&'a Arc<Queue>]> for SharingMode {
    #[inline]
    fn from(queues: &'a [&'a Arc<Queue>]) -> SharingMode {
        SharingMode::Concurrent(queues.iter().map(|queue| {
            queue.family().id()
        }).collect())
    }
}

/// Declares in which queue(s) a resource can be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sharing<I> where I: Iterator<Item = u32> {
    /// The resource is used is only one queue family.
    Exclusive,
    /// The resource is used in multiple queue families. Can be slower than `Exclusive`.
    Concurrent(I),
}

macro_rules! pipeline_stages {
    ($($elem:ident => $val:expr,)+) => (
        #[derive(Debug, Copy, Clone)]
        #[allow(missing_docs)]
        pub struct PipelineStages {
            $(
                pub $elem: bool,
            )+
        }

        impl PipelineStages {
            /// Builds an `PipelineStages` struct with none of the stages set.
            pub fn none() -> PipelineStages {
                PipelineStages {
                    $(
                        $elem: false,
                    )+
                }
            }
        }

        impl ops::BitOr for PipelineStages {
            type Output = PipelineStages;

            #[inline]
            fn bitor(self, rhs: PipelineStages) -> PipelineStages {
                PipelineStages {
                    $(
                        $elem: self.$elem || rhs.$elem,
                    )+
                }
            }
        }

        impl ops::BitOrAssign for PipelineStages {
            #[inline]
            fn bitor_assign(&mut self, rhs: PipelineStages) {
                $(
                    self.$elem = self.$elem || rhs.$elem;
                )+
            }
        }

        #[doc(hidden)]
        impl Into<vk::PipelineStageFlagBits> for PipelineStages {
            #[inline]
            fn into(self) -> vk::PipelineStageFlagBits {
                let mut result = 0;
                $(
                    if self.$elem { result |= $val }
                )+
                result
            }
        }
    );
}

pipeline_stages!{
    top_of_pipe => vk::PIPELINE_STAGE_TOP_OF_PIPE_BIT,
    draw_indirect => vk::PIPELINE_STAGE_DRAW_INDIRECT_BIT,
    vertex_input => vk::PIPELINE_STAGE_VERTEX_INPUT_BIT,
    vertex_shader => vk::PIPELINE_STAGE_VERTEX_SHADER_BIT,
    tessellation_control_shader => vk::PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT,
    tessellation_evaluation_shader => vk::PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT,
    geometry_shader => vk::PIPELINE_STAGE_GEOMETRY_SHADER_BIT,
    fragment_shader => vk::PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
    early_fragment_tests => vk::PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
    late_fragment_tests => vk::PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
    color_attachment_output => vk::PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
    compute_shader => vk::PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    transfer => vk::PIPELINE_STAGE_TRANSFER_BIT,
    bottom_of_pipe => vk::PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
    host => vk::PIPELINE_STAGE_HOST_BIT,
    all_graphics => vk::PIPELINE_STAGE_ALL_GRAPHICS_BIT,
    all_commands => vk::PIPELINE_STAGE_ALL_COMMANDS_BIT,
}

macro_rules! access_flags {
    ($($elem:ident => $val:expr,)+) => (
        #[derive(Debug, Copy, Clone)]
        #[allow(missing_docs)]
        pub struct AccessFlagBits {
            $(
                pub $elem: bool,
            )+
        }

        impl AccessFlagBits {
            /// Builds an `AccessFlagBits` struct with all bits set.
            pub fn all() -> AccessFlagBits {
                AccessFlagBits {
                    $(
                        $elem: true,
                    )+
                }
            }

            /// Builds an `AccessFlagBits` struct with none of the bits set.
            pub fn none() -> AccessFlagBits {
                AccessFlagBits {
                    $(
                        $elem: false,
                    )+
                }
            }
        }

        impl ops::BitOr for AccessFlagBits {
            type Output = AccessFlagBits;

            #[inline]
            fn bitor(self, rhs: AccessFlagBits) -> AccessFlagBits {
                AccessFlagBits {
                    $(
                        $elem: self.$elem || rhs.$elem,
                    )+
                }
            }
        }

        impl ops::BitOrAssign for AccessFlagBits {
            #[inline]
            fn bitor_assign(&mut self, rhs: AccessFlagBits) {
                $(
                    self.$elem = self.$elem || rhs.$elem;
                )+
            }
        }

        #[doc(hidden)]
        impl Into<vk::AccessFlagBits> for AccessFlagBits {
            #[inline]
            fn into(self) -> vk::AccessFlagBits {
                let mut result = 0;
                $(
                    if self.$elem { result |= $val }
                )+
                result
            }
        }
    );
}

access_flags!{
    indirect_command_read => vk::ACCESS_INDIRECT_COMMAND_READ_BIT,
    index_read => vk::ACCESS_INDEX_READ_BIT,
    vertex_attribute_read => vk::ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
    uniform_read => vk::ACCESS_UNIFORM_READ_BIT,
    input_attachment_read => vk::ACCESS_INPUT_ATTACHMENT_READ_BIT,
    shader_read => vk::ACCESS_SHADER_READ_BIT,
    shader_write => vk::ACCESS_SHADER_WRITE_BIT,
    color_attachment_read => vk::ACCESS_COLOR_ATTACHMENT_READ_BIT,
    color_attachment_write => vk::ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    depth_stencil_attachment_read => vk::ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
    depth_stencil_attachment_write => vk::ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
    transfer_read => vk::ACCESS_TRANSFER_READ_BIT,
    transfer_write => vk::ACCESS_TRANSFER_WRITE_BIT,
    host_read => vk::ACCESS_HOST_READ_BIT,
    host_write => vk::ACCESS_HOST_WRITE_BIT,
    memory_read => vk::ACCESS_MEMORY_READ_BIT,
    memory_write => vk::ACCESS_MEMORY_WRITE_BIT,
}
