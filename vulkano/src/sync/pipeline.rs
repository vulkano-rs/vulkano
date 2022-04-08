// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    buffer::sys::UnsafeBuffer,
    image::{sys::UnsafeImage, ImageAspects, ImageLayout, ImageSubresourceRange},
    DeviceSize,
};
use smallvec::SmallVec;
use std::{
    ops::{self, Range},
    sync::Arc,
};

macro_rules! pipeline_stages {
    ($($elem:ident, $var:ident => $val:ident, $queue:expr;)+) => (
        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
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

        impl From<PipelineStages> for ash::vk::PipelineStageFlags {
            #[inline]
            fn from(val: PipelineStages) -> Self {
                let mut result = ash::vk::PipelineStageFlags::empty();
                $(
                    if val.$elem { result |= ash::vk::PipelineStageFlags::$val }
                )+
                result
            }
        }

        impl From<PipelineStages> for ash::vk::PipelineStageFlags2 {
            #[inline]
            fn from(val: PipelineStages) -> Self {
                let mut result = ash::vk::PipelineStageFlags2::empty();
                $(
                    if val.$elem { result |= ash::vk::PipelineStageFlags2::$val }
                )+
                result
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

        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        #[repr(u64)]
        pub enum PipelineStage {
            $(
                $var = ash::vk::PipelineStageFlags2::$val.as_raw(),
            )+
        }

        impl PipelineStage {
            #[inline]
            pub fn required_queue_flags(&self) -> ash::vk::QueueFlags {
                match self {
                    $(
                        Self::$var => $queue,
                    )+
                }
            }
        }
    );
}

impl PipelineStages {
    /// Returns the access types that are supported with the given pipeline stages.
    ///
    /// Corresponds to the table
    /// "[Supported access types](https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-access-types-supported)"
    /// in the Vulkan specification.
    #[inline]
    pub fn supported_access(&self) -> AccessFlags {
        if self.all_commands {
            return AccessFlags::all();
        }

        let PipelineStages {
            top_of_pipe,
            mut draw_indirect,
            mut vertex_input,
            mut vertex_shader,
            mut tessellation_control_shader,
            mut tessellation_evaluation_shader,
            mut geometry_shader,
            mut fragment_shader,
            mut early_fragment_tests,
            mut late_fragment_tests,
            mut color_attachment_output,
            compute_shader,
            transfer,
            bottom_of_pipe,
            host,
            all_graphics,
            all_commands,
            ray_tracing_shader,
        } = *self;

        if all_graphics {
            draw_indirect = true;
            //task_shader = true;
            //mesh_shader = true;
            vertex_input = true;
            vertex_shader = true;
            tessellation_control_shader = true;
            tessellation_evaluation_shader = true;
            geometry_shader = true;
            fragment_shader = true;
            early_fragment_tests = true;
            late_fragment_tests = true;
            color_attachment_output = true;
            //conditional_rendering = true;
            //transform_feedback = true;
            //fragment_shading_rate_attachment = true;
            //fragment_density_process = true;
        }

        AccessFlags {
            indirect_command_read: draw_indirect, /*|| acceleration_structure_build*/
            index_read: vertex_input,
            vertex_attribute_read: vertex_input,
            uniform_read:
                // task_shader
                // mesh_shader
                ray_tracing_shader
                || vertex_shader
                || tessellation_control_shader
                || tessellation_evaluation_shader
                || geometry_shader
                || fragment_shader
                || compute_shader,
            shader_read:
                // acceleration_structure_build
                // task_shader
                // mesh_shader
                ray_tracing_shader
                || vertex_shader
                || tessellation_control_shader
                || tessellation_evaluation_shader
                || geometry_shader
                || fragment_shader
                || compute_shader,
            shader_write:
                // task_shader
                //  mesh_shader
                ray_tracing_shader
                || vertex_shader
                || tessellation_control_shader
                || tessellation_evaluation_shader
                || geometry_shader
                || fragment_shader
                || compute_shader,
            input_attachment_read:
                // subpass_shading
                fragment_shader,
            color_attachment_read: color_attachment_output,
            color_attachment_write: color_attachment_output,
            depth_stencil_attachment_read: early_fragment_tests || late_fragment_tests,
            depth_stencil_attachment_write: early_fragment_tests || late_fragment_tests,
            transfer_read: transfer,
                // acceleration_structure_build
            transfer_write: transfer,
                // acceleration_structure_build
            host_read: host,
            host_write: host,
            memory_read: true,
            memory_write: true,

            /*
            color_attachment_read_noncoherent: color_attachment_output,
            preprocess_read: command_preprocess,
            preprocess_write: command_preprocess,
            conditional_rendering_read: conditional_rendering,
            fragment_shading_rate_attachment_read: fragment_shading_rate_attachment,
            invocation_mask_read: invocation_mask,
            transform_feedback_write: transform_feedback,
            transform_feedback_counter_write: transform_feedback,
            transform_feedback_counter_read: transform_feedback || draw_indirect,
            acceleration_structure_read: task_shader || mesh_shader || vertex_shader || tessellation_control_shader || tessellation_evaluation_shader || geometry_shader || fragment_shader || compute_shader || ray_tracing_shader || acceleration_structure_build,
            acceleration_structure_write: acceleration_structure_build,
            fragment_density_map_read: fragment_density_process,
            */
        }
    }
}

impl From<PipelineStage> for ash::vk::PipelineStageFlags {
    #[inline]
    fn from(val: PipelineStage) -> Self {
        Self::from_raw(val as u32)
    }
}

impl From<PipelineStage> for ash::vk::PipelineStageFlags2 {
    #[inline]
    fn from(val: PipelineStage) -> Self {
        Self::from_raw(val as u64)
    }
}

pipeline_stages! {
    top_of_pipe, TopOfPipe => TOP_OF_PIPE, ash::vk::QueueFlags::empty();
    draw_indirect, DrawIndirect => DRAW_INDIRECT, ash::vk::QueueFlags::GRAPHICS | ash::vk::QueueFlags::COMPUTE;
    vertex_input, VertexInput => VERTEX_INPUT, ash::vk::QueueFlags::GRAPHICS;
    vertex_shader, VertexShader => VERTEX_SHADER, ash::vk::QueueFlags::GRAPHICS;
    tessellation_control_shader, TessellationControlShader => TESSELLATION_CONTROL_SHADER, ash::vk::QueueFlags::GRAPHICS;
    tessellation_evaluation_shader, TessellationEvaluationShader => TESSELLATION_EVALUATION_SHADER, ash::vk::QueueFlags::GRAPHICS;
    geometry_shader, GeometryShader => GEOMETRY_SHADER, ash::vk::QueueFlags::GRAPHICS;
    fragment_shader, FragmentShader => FRAGMENT_SHADER, ash::vk::QueueFlags::GRAPHICS;
    early_fragment_tests, EarlyFragmentTests => EARLY_FRAGMENT_TESTS, ash::vk::QueueFlags::GRAPHICS;
    late_fragment_tests, LateFragmentTests => LATE_FRAGMENT_TESTS, ash::vk::QueueFlags::GRAPHICS;
    color_attachment_output, ColorAttachmentOutput => COLOR_ATTACHMENT_OUTPUT, ash::vk::QueueFlags::GRAPHICS;
    compute_shader, ComputeShader => COMPUTE_SHADER, ash::vk::QueueFlags::COMPUTE;
    transfer, Transfer => TRANSFER, ash::vk::QueueFlags::GRAPHICS | ash::vk::QueueFlags::COMPUTE | ash::vk::QueueFlags::TRANSFER;
    bottom_of_pipe, BottomOfPipe => BOTTOM_OF_PIPE, ash::vk::QueueFlags::empty();
    host, Host => HOST, ash::vk::QueueFlags::empty();
    all_graphics, AllGraphics => ALL_GRAPHICS, ash::vk::QueueFlags::GRAPHICS;
    all_commands, AllCommands => ALL_COMMANDS, ash::vk::QueueFlags::empty();
    ray_tracing_shader, RayTracingShader => RAY_TRACING_SHADER_KHR, ash::vk::QueueFlags::GRAPHICS | ash::vk::QueueFlags::COMPUTE | ash::vk::QueueFlags::TRANSFER;
}

macro_rules! access_flags {
    ($($elem:ident => $val:ident,)+) => (
        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        #[allow(missing_docs)]
        pub struct AccessFlags {
            $(
                pub $elem: bool,
            )+
        }

        impl AccessFlags {
            /// Builds an `AccessFlags` struct with all bits set.
            pub fn all() -> AccessFlags {
                AccessFlags {
                    $(
                        $elem: true,
                    )+
                }
            }

            /// Builds an `AccessFlags` struct with none of the bits set.
            pub fn none() -> AccessFlags {
                AccessFlags {
                    $(
                        $elem: false,
                    )+
                }
            }

            /// Returns whether all flags in `other` are also set in `self`.
            pub const fn contains(&self, other: &Self) -> bool {
                $(
                    (self.$elem || !other.$elem)
                )&&+
            }
        }

        impl From<AccessFlags> for ash::vk::AccessFlags {
            #[inline]
            fn from(val: AccessFlags) -> Self {
                let mut result = ash::vk::AccessFlags::empty();
                $(
                    if val.$elem { result |= ash::vk::AccessFlags::$val }
                )+
                result
            }
        }

        impl From<AccessFlags> for ash::vk::AccessFlags2 {
            #[inline]
            fn from(val: AccessFlags) -> Self {
                let mut result = ash::vk::AccessFlags2::empty();
                $(
                    if val.$elem { result |= ash::vk::AccessFlags2::$val }
                )+
                result
            }
        }

        impl ops::BitOr for AccessFlags {
            type Output = AccessFlags;

            #[inline]
            fn bitor(self, rhs: AccessFlags) -> AccessFlags {
                AccessFlags {
                    $(
                        $elem: self.$elem || rhs.$elem,
                    )+
                }
            }
        }

        impl ops::BitOrAssign for AccessFlags {
            #[inline]
            fn bitor_assign(&mut self, rhs: AccessFlags) {
                $(
                    self.$elem = self.$elem || rhs.$elem;
                )+
            }
        }
    );
}

access_flags! {
    indirect_command_read => INDIRECT_COMMAND_READ,
    index_read => INDEX_READ,
    vertex_attribute_read => VERTEX_ATTRIBUTE_READ,
    uniform_read => UNIFORM_READ,
    input_attachment_read => INPUT_ATTACHMENT_READ,
    shader_read => SHADER_READ,
    shader_write => SHADER_WRITE,
    color_attachment_read => COLOR_ATTACHMENT_READ,
    color_attachment_write => COLOR_ATTACHMENT_WRITE,
    depth_stencil_attachment_read => DEPTH_STENCIL_ATTACHMENT_READ,
    depth_stencil_attachment_write => DEPTH_STENCIL_ATTACHMENT_WRITE,
    transfer_read => TRANSFER_READ,
    transfer_write => TRANSFER_WRITE,
    host_read => HOST_READ,
    host_write => HOST_WRITE,
    memory_read => MEMORY_READ,
    memory_write => MEMORY_WRITE,
}

/// The full specification of memory access by the pipeline for a particular resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PipelineMemoryAccess {
    /// The pipeline stages the resource will be accessed in.
    pub stages: PipelineStages,
    /// The type of memory access that will be performed.
    pub access: AccessFlags,
    /// Whether the resource needs exclusive (mutable) access or can be shared.
    pub exclusive: bool,
}

/// Dependency info for a pipeline barrier.
///
/// A pipeline barrier creates a dependency between commands submitted before the barrier (the
/// source scope) and commands submitted after it (the destination scope). A pipeline barrier
/// consists of multiple individual barriers that concern a either single resource or
/// operate globally.
///
/// Each barrier has a set of source/destination pipeline stages and source/destination memory
/// access types. The pipeline stages create an *execution dependency*: the `source_stages` of
/// commands submitted before the barrier must be completely finished before before any of the
/// `destination_stages` of commands after the barrier are allowed to start. The memory access types
/// create a *memory dependency*: in addition to the execution dependency, any `source_access`
/// performed before the barrier must be made available and visible before any `destination_access`
/// are made after the barrier.
#[derive(Clone, Debug)]
pub struct DependencyInfo {
    /// Memory barriers for global operations and accesses, not limited to a single resource.
    pub memory_barriers: SmallVec<[MemoryBarrier; 2]>,

    /// Memory barriers for individual buffers.
    pub buffer_memory_barriers: SmallVec<[BufferMemoryBarrier; 8]>,

    /// Memory barriers for individual images.
    pub image_memory_barriers: SmallVec<[ImageMemoryBarrier; 8]>,

    pub _ne: crate::NonExhaustive,
}

impl DependencyInfo {
    /// Returns whether `self` contains any barriers.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.memory_barriers.is_empty()
            && self.buffer_memory_barriers.is_empty()
            && self.image_memory_barriers.is_empty()
    }

    /// Clears all barriers.
    #[inline]
    pub fn clear(&mut self) {
        self.memory_barriers.clear();
        self.buffer_memory_barriers.clear();
        self.image_memory_barriers.clear();
    }
}

impl Default for DependencyInfo {
    #[inline]
    fn default() -> Self {
        Self {
            memory_barriers: SmallVec::new(),
            buffer_memory_barriers: SmallVec::new(),
            image_memory_barriers: SmallVec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// A memory barrier that is applied globally.
#[derive(Clone, Debug)]
pub struct MemoryBarrier {
    /// The pipeline stages in the source scope to wait for.
    ///
    /// The default value is [`PipelineStages::none()`].
    pub source_stages: PipelineStages,

    /// The memory accesses in the source scope to make available and visible.
    ///
    /// The default value is [`AccessFlags::none()`].
    pub source_access: AccessFlags,

    /// The pipeline stages in the destination scope that must wait for `source_stages`.
    ///
    /// The default value is [`PipelineStages::none()`].
    pub destination_stages: PipelineStages,

    /// The memory accesses in the destination scope that must wait for `source_access` to be made
    /// available and visible.
    pub destination_access: AccessFlags,

    pub _ne: crate::NonExhaustive,
}

impl Default for MemoryBarrier {
    #[inline]
    fn default() -> Self {
        Self {
            source_stages: PipelineStages::none(),
            source_access: AccessFlags::none(),
            destination_stages: PipelineStages::none(),
            destination_access: AccessFlags::none(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// A memory barrier that is applied to a single buffer.
#[derive(Clone, Debug)]
pub struct BufferMemoryBarrier {
    /// The pipeline stages in the source scope to wait for.
    ///
    /// The default value is [`PipelineStages::none()`].
    pub source_stages: PipelineStages,

    /// The memory accesses in the source scope to make available and visible.
    ///
    /// The default value is [`AccessFlags::none()`].
    pub source_access: AccessFlags,

    /// The pipeline stages in the destination scope that must wait for `source_stages`.
    ///
    /// The default value is [`PipelineStages::none()`].
    pub destination_stages: PipelineStages,

    /// The memory accesses in the destination scope that must wait for `source_access` to be made
    /// available and visible.
    pub destination_access: AccessFlags,

    /// For resources created with [`Sharing::Exclusive`](crate::sync::Sharing), transfers
    /// ownership of a resource from one queue family to another.
    pub queue_family_transfer: Option<QueueFamilyTransfer>,

    /// The buffer to apply the barrier to.
    pub buffer: Arc<UnsafeBuffer>,

    /// The byte range of `buffer` to apply the barrier to.
    pub range: Range<DeviceSize>,

    pub _ne: crate::NonExhaustive,
}

impl BufferMemoryBarrier {
    #[inline]
    pub fn buffer(buffer: Arc<UnsafeBuffer>) -> Self {
        Self {
            source_stages: PipelineStages::none(),
            source_access: AccessFlags::none(),
            destination_stages: PipelineStages::none(),
            destination_access: AccessFlags::none(),
            queue_family_transfer: None,
            buffer,
            range: 0..0,
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// A memory barrier that is applied to a single image.
#[derive(Clone, Debug)]
pub struct ImageMemoryBarrier {
    /// The pipeline stages in the source scope to wait for.
    ///
    /// The default value is [`PipelineStages::none()`].
    pub source_stages: PipelineStages,

    /// The memory accesses in the source scope to make available and visible.
    ///
    /// The default value is [`AccessFlags::none()`].
    pub source_access: AccessFlags,

    /// The pipeline stages in the destination scope that must wait for `source_stages`.
    ///
    /// The default value is [`PipelineStages::none()`].
    pub destination_stages: PipelineStages,

    /// The memory accesses in the destination scope that must wait for `source_access` to be made
    /// available and visible.
    pub destination_access: AccessFlags,

    /// The layout that the specified `subresource_range` of `image` is expected to be in when the
    /// source scope completes.
    pub old_layout: ImageLayout,

    /// The layout that the specified `subresource_range` of `image` will be transitioned to before
    /// the destination scope begins.
    pub new_layout: ImageLayout,

    /// For resources created with [`Sharing::Exclusive`](crate::sync::Sharing), transfers
    /// ownership of a resource from one queue family to another.
    pub queue_family_transfer: Option<QueueFamilyTransfer>,

    /// The image to apply the barrier to.
    pub image: Arc<UnsafeImage>,

    /// The subresource range of `image` to apply the barrier to.
    pub subresource_range: ImageSubresourceRange,

    pub _ne: crate::NonExhaustive,
}

impl ImageMemoryBarrier {
    #[inline]
    pub fn image(image: Arc<UnsafeImage>) -> Self {
        Self {
            source_stages: PipelineStages::none(),
            source_access: AccessFlags::none(),
            destination_stages: PipelineStages::none(),
            destination_access: AccessFlags::none(),
            old_layout: ImageLayout::Undefined,
            new_layout: ImageLayout::Undefined,
            queue_family_transfer: None,
            image,
            subresource_range: ImageSubresourceRange {
                aspects: ImageAspects::none(), // Can't use image format aspects because `color` can't be specified with `planeN`.
                mip_levels: 0..0,
                array_layers: 0..0,
            },
            _ne: Default::default(),
        }
    }
}

/// Specifies a queue family ownership transfer for a resource.
#[derive(Clone, Copy, Debug)]
pub struct QueueFamilyTransfer {
    /// The queue family that currently owns the resource.
    pub source_index: u32,

    /// The queue family to transfer ownership to.
    pub destination_index: u32,
}
