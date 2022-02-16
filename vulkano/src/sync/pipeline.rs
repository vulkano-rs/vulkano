// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ops;

macro_rules! pipeline_stages {
    ($($elem:ident, $var:ident => $val:expr, $queue:expr;)+) => (
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
                    if val.$elem { result |= $val }
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
        #[repr(u32)]
        pub enum PipelineStage {
            $(
                $var = $val.as_raw(),
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
    /// Returns the access types that are allowed with the given pipeline stages.
    ///
    /// Corresponds to the table
    /// "[Supported access types](https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-access-types-supported)"
    /// in the Vulkan specification.
    #[inline]
    pub fn allowed_access(&self) -> AccessFlags {
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

pipeline_stages! {
    top_of_pipe, TopOfPipe => ash::vk::PipelineStageFlags::TOP_OF_PIPE, ash::vk::QueueFlags::empty();
    draw_indirect, DrawIndirect => ash::vk::PipelineStageFlags::DRAW_INDIRECT, ash::vk::QueueFlags::GRAPHICS | ash::vk::QueueFlags::COMPUTE;
    vertex_input, VertexInput => ash::vk::PipelineStageFlags::VERTEX_INPUT, ash::vk::QueueFlags::GRAPHICS;
    vertex_shader, VertexShader => ash::vk::PipelineStageFlags::VERTEX_SHADER, ash::vk::QueueFlags::GRAPHICS;
    tessellation_control_shader, TessellationControlShader => ash::vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER, ash::vk::QueueFlags::GRAPHICS;
    tessellation_evaluation_shader, TessellationEvaluationShader => ash::vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER, ash::vk::QueueFlags::GRAPHICS;
    geometry_shader, GeometryShader => ash::vk::PipelineStageFlags::GEOMETRY_SHADER, ash::vk::QueueFlags::GRAPHICS;
    fragment_shader, FragmentShader => ash::vk::PipelineStageFlags::FRAGMENT_SHADER, ash::vk::QueueFlags::GRAPHICS;
    early_fragment_tests, EarlyFragmentTests => ash::vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS, ash::vk::QueueFlags::GRAPHICS;
    late_fragment_tests, LateFragmentTests => ash::vk::PipelineStageFlags::LATE_FRAGMENT_TESTS, ash::vk::QueueFlags::GRAPHICS;
    color_attachment_output, ColorAttachmentOutput => ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, ash::vk::QueueFlags::GRAPHICS;
    compute_shader, ComputeShader => ash::vk::PipelineStageFlags::COMPUTE_SHADER, ash::vk::QueueFlags::COMPUTE;
    transfer, Transfer => ash::vk::PipelineStageFlags::TRANSFER, ash::vk::QueueFlags::GRAPHICS | ash::vk::QueueFlags::COMPUTE | ash::vk::QueueFlags::TRANSFER;
    bottom_of_pipe, BottomOfPipe => ash::vk::PipelineStageFlags::BOTTOM_OF_PIPE, ash::vk::QueueFlags::empty();
    host, Host => ash::vk::PipelineStageFlags::HOST, ash::vk::QueueFlags::empty();
    all_graphics, AllGraphics => ash::vk::PipelineStageFlags::ALL_GRAPHICS, ash::vk::QueueFlags::GRAPHICS;
    all_commands, AllCommands => ash::vk::PipelineStageFlags::ALL_COMMANDS, ash::vk::QueueFlags::empty();
    ray_tracing_shader, RayTracingShader => ash::vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR, ash::vk::QueueFlags::GRAPHICS | ash::vk::QueueFlags::COMPUTE | ash::vk::QueueFlags::TRANSFER;
}

macro_rules! access_flags {
    ($($elem:ident => $val:expr,)+) => (
        #[derive(Debug, Copy, Clone)]
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
                    if val.$elem { result |= $val }
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
    indirect_command_read => ash::vk::AccessFlags::INDIRECT_COMMAND_READ,
    index_read => ash::vk::AccessFlags::INDEX_READ,
    vertex_attribute_read => ash::vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
    uniform_read => ash::vk::AccessFlags::UNIFORM_READ,
    input_attachment_read => ash::vk::AccessFlags::INPUT_ATTACHMENT_READ,
    shader_read => ash::vk::AccessFlags::SHADER_READ,
    shader_write => ash::vk::AccessFlags::SHADER_WRITE,
    color_attachment_read => ash::vk::AccessFlags::COLOR_ATTACHMENT_READ,
    color_attachment_write => ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
    depth_stencil_attachment_read => ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
    depth_stencil_attachment_write => ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
    transfer_read => ash::vk::AccessFlags::TRANSFER_READ,
    transfer_write => ash::vk::AccessFlags::TRANSFER_WRITE,
    host_read => ash::vk::AccessFlags::HOST_READ,
    host_write => ash::vk::AccessFlags::HOST_WRITE,
    memory_read => ash::vk::AccessFlags::MEMORY_READ,
    memory_write => ash::vk::AccessFlags::MEMORY_WRITE,
}

/// The full specification of memory access by the pipeline for a particular resource.
#[derive(Clone, Copy, Debug)]
pub struct PipelineMemoryAccess {
    /// The pipeline stages the resource will be accessed in.
    pub stages: PipelineStages,
    /// The type of memory access that will be performed.
    pub access: AccessFlags,
    /// Whether the resource needs exclusive (mutable) access or can be shared.
    pub exclusive: bool,
}
