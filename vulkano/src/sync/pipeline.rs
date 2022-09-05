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
    macros::{vulkan_bitflags, vulkan_enum},
    DeviceSize,
};
use smallvec::SmallVec;
use std::{ops::Range, sync::Arc};

vulkan_enum! {
    // TODO: document
    #[non_exhaustive]
    PipelineStage = PipelineStageFlags2(u64);

    // TODO: document
    TopOfPipe = TOP_OF_PIPE,

    // TODO: document
    DrawIndirect = DRAW_INDIRECT,

    // TODO: document
    VertexInput = VERTEX_INPUT,

    // TODO: document
    VertexShader = VERTEX_SHADER,

    // TODO: document
    TessellationControlShader = TESSELLATION_CONTROL_SHADER,

    // TODO: document
    TessellationEvaluationShader = TESSELLATION_EVALUATION_SHADER,

    // TODO: document
    GeometryShader = GEOMETRY_SHADER,

    // TODO: document
    FragmentShader = FRAGMENT_SHADER,

    // TODO: document
    EarlyFragmentTests = EARLY_FRAGMENT_TESTS,

    // TODO: document
    LateFragmentTests = LATE_FRAGMENT_TESTS,

    // TODO: document
    ColorAttachmentOutput = COLOR_ATTACHMENT_OUTPUT,

    // TODO: document
    ComputeShader = COMPUTE_SHADER,

    // TODO: document
    Transfer = TRANSFER,

    // TODO: document
    BottomOfPipe = BOTTOM_OF_PIPE,

    // TODO: document
    Host = HOST,

    // TODO: document
    AllGraphics = ALL_GRAPHICS,

    // TODO: document
    AllCommands = ALL_COMMANDS,

    /*
    // TODO: document
    TransformFeedback = TRANSFORM_FEEDBACK_EXT {
        extensions: [ext_transform_feedback],
    },

    // TODO: document
    ConditionalRendering = CONDITIONAL_RENDERING_EXT {
        extensions: [ext_conditional_rendering],
    },

    // TODO: document
    AccelerationStructureBuild = ACCELERATION_STRUCTURE_BUILD_KHR {
        extensions: [khr_acceleration_structure, nv_ray_tracing],
    },
    */

    // TODO: document
    RayTracingShader = RAY_TRACING_SHADER_KHR {
        extensions: [khr_ray_tracing_pipeline, nv_ray_tracing],
    },

    /*
    // TODO: document
    FragmentDensityProcess = FRAGMENT_DENSITY_PROCESS_EXT {
        extensions: [ext_fragment_density_map],
    },

    // TODO: document
    FragmentShadingRateAttachment = FRAGMENT_SHADING_RATE_ATTACHMENT_KHR {
        extensions: [khr_fragment_shading_rate],
    },

    // TODO: document
    CommandPreprocess = COMMAND_PREPROCESS_NV {
        extensions: [nv_device_generated_commands],
    },

    // TODO: document
    TaskShader = TASK_SHADER_NV {
        extensions: [nv_mesh_shader],
    },

    // TODO: document
    MeshShader = MESH_SHADER_NV {
        extensions: [nv_mesh_shader],
    },
     */
}

impl PipelineStage {
    #[inline]
    pub fn required_queue_flags(&self) -> ash::vk::QueueFlags {
        match self {
            Self::TopOfPipe => ash::vk::QueueFlags::empty(),
            Self::DrawIndirect => ash::vk::QueueFlags::GRAPHICS | ash::vk::QueueFlags::COMPUTE,
            Self::VertexInput => ash::vk::QueueFlags::GRAPHICS,
            Self::VertexShader => ash::vk::QueueFlags::GRAPHICS,
            Self::TessellationControlShader => ash::vk::QueueFlags::GRAPHICS,
            Self::TessellationEvaluationShader => ash::vk::QueueFlags::GRAPHICS,
            Self::GeometryShader => ash::vk::QueueFlags::GRAPHICS,
            Self::FragmentShader => ash::vk::QueueFlags::GRAPHICS,
            Self::EarlyFragmentTests => ash::vk::QueueFlags::GRAPHICS,
            Self::LateFragmentTests => ash::vk::QueueFlags::GRAPHICS,
            Self::ColorAttachmentOutput => ash::vk::QueueFlags::GRAPHICS,
            Self::ComputeShader => ash::vk::QueueFlags::COMPUTE,
            Self::Transfer => {
                ash::vk::QueueFlags::GRAPHICS
                    | ash::vk::QueueFlags::COMPUTE
                    | ash::vk::QueueFlags::TRANSFER
            }
            Self::BottomOfPipe => ash::vk::QueueFlags::empty(),
            Self::Host => ash::vk::QueueFlags::empty(),
            Self::AllGraphics => ash::vk::QueueFlags::GRAPHICS,
            Self::AllCommands => ash::vk::QueueFlags::empty(),
            Self::RayTracingShader => {
                ash::vk::QueueFlags::GRAPHICS
                    | ash::vk::QueueFlags::COMPUTE
                    | ash::vk::QueueFlags::TRANSFER
            }
        }
    }
}

impl From<PipelineStage> for ash::vk::PipelineStageFlags {
    #[inline]
    fn from(val: PipelineStage) -> Self {
        Self::from_raw(val as u32)
    }
}

vulkan_bitflags! {
    // TODO: document
    #[non_exhaustive]
    PipelineStages = PipelineStageFlags2(u64);

    // TODO: document
    top_of_pipe = TOP_OF_PIPE,

    // TODO: document
    draw_indirect = DRAW_INDIRECT,

    // TODO: document
    vertex_input = VERTEX_INPUT,

    // TODO: document
    vertex_shader = VERTEX_SHADER,

    // TODO: document
    tessellation_control_shader = TESSELLATION_CONTROL_SHADER,

    // TODO: document
    tessellation_evaluation_shader = TESSELLATION_EVALUATION_SHADER,

    // TODO: document
    geometry_shader = GEOMETRY_SHADER,

    // TODO: document
    fragment_shader = FRAGMENT_SHADER,

    // TODO: document
    early_fragment_tests = EARLY_FRAGMENT_TESTS,

    // TODO: document
    late_fragment_tests = LATE_FRAGMENT_TESTS,

    // TODO: document
    color_attachment_output = COLOR_ATTACHMENT_OUTPUT,

    // TODO: document
    compute_shader = COMPUTE_SHADER,

    // TODO: document
    transfer = TRANSFER,

    // TODO: document
    bottom_of_pipe = BOTTOM_OF_PIPE,

    // TODO: document
    host = HOST,

    // TODO: document
    all_graphics = ALL_GRAPHICS,

    // TODO: document
    all_commands = ALL_COMMANDS,

    /*
    // TODO: document
    transform_feedback = TRANSFORM_FEEDBACK_EXT {
        extensions: [ext_transform_feedback],
    },

    // TODO: document
    conditional_rendering = CONDITIONAL_RENDERING_EXT {
        extensions: [ext_conditional_rendering],
    },

    // TODO: document
    acceleration_structure_build = ACCELERATION_STRUCTURE_BUILD_KHR {
        extensions: [khr_acceleration_structure, nv_ray_tracing],
    },
     */

    // TODO: document
    ray_tracing_shader = RAY_TRACING_SHADER_KHR {
        extensions: [khr_ray_tracing_pipeline, nv_ray_tracing],
    },

    /*
    // TODO: document
    fragment_density_process = FRAGMENT_DENSITY_PROCESS_EXT {
        extensions: [ext_fragment_density_map],
    },

    // TODO: document
    fragment_shading_rate_attachment = FRAGMENT_SHADING_RATE_ATTACHMENT_KHR {
        extensions: [khr_fragment_shading_rate],
    },

    // TODO: document
    command_preprocess = COMMAND_PREPROCESS_NV {
        extensions: [nv_device_generated_commands],
    },

    // TODO: document
    task_shader = TASK_SHADER_NV {
        extensions: [nv_mesh_shader],
    },

    // TODO: document
    mesh_shader = MESH_SHADER_NV {
        extensions: [nv_mesh_shader],
    },
     */
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
            top_of_pipe: _,
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
            bottom_of_pipe: _,
            host,
            all_graphics,
            all_commands: _,
            ray_tracing_shader,
            _ne: _,
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

            ..AccessFlags::empty()
        }
    }
}

impl From<PipelineStages> for ash::vk::PipelineStageFlags {
    #[inline]
    fn from(val: PipelineStages) -> Self {
        Self::from_raw(ash::vk::PipelineStageFlags2::from(val).as_raw() as u32)
    }
}

vulkan_bitflags! {
    // TODO: document
    #[non_exhaustive]
    AccessFlags = AccessFlags2(u64);

    // TODO: document
    indirect_command_read = INDIRECT_COMMAND_READ,

    // TODO: document
    index_read = INDEX_READ,

    // TODO: document
    vertex_attribute_read = VERTEX_ATTRIBUTE_READ,

    // TODO: document
    uniform_read = UNIFORM_READ,

    // TODO: document
    input_attachment_read = INPUT_ATTACHMENT_READ,

    // TODO: document
    shader_read = SHADER_READ,

    // TODO: document
    shader_write = SHADER_WRITE,

    // TODO: document
    color_attachment_read = COLOR_ATTACHMENT_READ,

    // TODO: document
    color_attachment_write = COLOR_ATTACHMENT_WRITE,

    // TODO: document
    depth_stencil_attachment_read = DEPTH_STENCIL_ATTACHMENT_READ,

    // TODO: document
    depth_stencil_attachment_write = DEPTH_STENCIL_ATTACHMENT_WRITE,

    // TODO: document
    transfer_read = TRANSFER_READ,

    // TODO: document
    transfer_write = TRANSFER_WRITE,

    // TODO: document
    host_read = HOST_READ,

    // TODO: document
    host_write = HOST_WRITE,

    // TODO: document
    memory_read = MEMORY_READ,

    // TODO: document
    memory_write = MEMORY_WRITE,

    /*
    // Provided by VK_EXT_transform_feedback
    transform_feedback_write = TRANSFORM_FEEDBACK_WRITE_EXT {
        extensions: [ext_transform_feedback],
    },

    // Provided by VK_EXT_transform_feedback
    transform_feedback_counter_read = TRANSFORM_FEEDBACK_COUNTER_READ_EXT {
        extensions: [ext_transform_feedback],
    },

    // Provided by VK_EXT_transform_feedback
    transform_feedback_counter_write = TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT {
        extensions: [ext_transform_feedback],
    },

    // Provided by VK_EXT_conditional_rendering
    conditional_rendering_read = CONDITIONAL_RENDERING_READ_EXT {
        extensions: [ext_conditional_rendering],
    },

    // Provided by VK_EXT_blend_operation_advanced
    color_attachment_read_noncoherent = COLOR_ATTACHMENT_READ_NONCOHERENT_EXT {
        extensions: [ext_blend_operation_advanced],
    },

    // Provided by VK_KHR_acceleration_structure
    acceleration_structure_read = ACCELERATION_STRUCTURE_READ_KHR {
        extensions: [khr_acceleration_structure, nv_ray_tracing],
    },

    // Provided by VK_KHR_acceleration_structure
    acceleration_structure_write = ACCELERATION_STRUCTURE_WRITE_KHR {
        extensions: [khr_acceleration_structure, nv_ray_tracing],
    },

    // Provided by VK_EXT_fragment_density_map
    fragment_density_map_read = FRAGMENT_DENSITY_MAP_READ_EXT {
        extensions: [ext_fragment_density_map],
    },

    // Provided by VK_KHR_fragment_shading_rate
    fragment_shading_rate_attachment_read = FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR {
        extensions: [khr_fragment_shading_rate],
    },

    // Provided by VK_NV_device_generated_commands
    command_preprocess_read = COMMAND_PREPROCESS_READ_NV {
        extensions: [nv_device_generated_commands],
    },

    // Provided by VK_NV_device_generated_commands
    command_preprocess_write = COMMAND_PREPROCESS_WRITE_NV {
        extensions: [nv_device_generated_commands],
    },
     */
}

impl AccessFlags {
    pub(crate) fn all() -> AccessFlags {
        AccessFlags {
            indirect_command_read: true,
            index_read: true,
            vertex_attribute_read: true,
            uniform_read: true,
            input_attachment_read: true,
            shader_read: true,
            shader_write: true,
            color_attachment_read: true,
            color_attachment_write: true,
            depth_stencil_attachment_read: true,
            depth_stencil_attachment_write: true,
            transfer_read: true,
            transfer_write: true,
            host_read: true,
            host_write: true,
            memory_read: true,
            memory_write: true,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl From<AccessFlags> for ash::vk::AccessFlags {
    #[inline]
    fn from(val: AccessFlags) -> Self {
        Self::from_raw(ash::vk::AccessFlags2::from(val).as_raw() as u32)
    }
}

/// The full specification of memory access by the pipeline for a particular resource.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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
    /// The default value is [`PipelineStages::empty()`].
    pub source_stages: PipelineStages,

    /// The memory accesses in the source scope to make available and visible.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub source_access: AccessFlags,

    /// The pipeline stages in the destination scope that must wait for `source_stages`.
    ///
    /// The default value is [`PipelineStages::empty()`].
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
            source_stages: PipelineStages::empty(),
            source_access: AccessFlags::empty(),
            destination_stages: PipelineStages::empty(),
            destination_access: AccessFlags::empty(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// A memory barrier that is applied to a single buffer.
#[derive(Clone, Debug)]
pub struct BufferMemoryBarrier {
    /// The pipeline stages in the source scope to wait for.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub source_stages: PipelineStages,

    /// The memory accesses in the source scope to make available and visible.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub source_access: AccessFlags,

    /// The pipeline stages in the destination scope that must wait for `source_stages`.
    ///
    /// The default value is [`PipelineStages::empty()`].
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
            source_stages: PipelineStages::empty(),
            source_access: AccessFlags::empty(),
            destination_stages: PipelineStages::empty(),
            destination_access: AccessFlags::empty(),
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
    /// The default value is [`PipelineStages::empty()`].
    pub source_stages: PipelineStages,

    /// The memory accesses in the source scope to make available and visible.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub source_access: AccessFlags,

    /// The pipeline stages in the destination scope that must wait for `source_stages`.
    ///
    /// The default value is [`PipelineStages::empty()`].
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
            source_stages: PipelineStages::empty(),
            source_access: AccessFlags::empty(),
            destination_stages: PipelineStages::empty(),
            destination_access: AccessFlags::empty(),
            old_layout: ImageLayout::Undefined,
            new_layout: ImageLayout::Undefined,
            queue_family_transfer: None,
            image,
            subresource_range: ImageSubresourceRange {
                aspects: ImageAspects::empty(), // Can't use image format aspects because `color` can't be specified with `planeN`.
                mip_levels: 0..0,
                array_layers: 0..0,
            },
            _ne: crate::NonExhaustive(()),
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
