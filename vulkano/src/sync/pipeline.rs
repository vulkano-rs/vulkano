// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{
    buffer::sys::Buffer,
    device::QueueFlags,
    image::{sys::Image, ImageAspects, ImageLayout, ImageSubresourceRange},
    macros::{vulkan_bitflags, vulkan_bitflags_enum},
    DeviceSize,
};
use smallvec::SmallVec;
use std::{ops::Range, sync::Arc};

vulkan_bitflags_enum! {
    #[non_exhaustive]
    /// A set of [`PipelineStage`] values.
    PipelineStages impl {
        /// Returns whether `self` contains stages that are only available in
        /// `VkPipelineStageFlagBits2`.
        pub(crate) fn is_2(self) -> bool {
            !(self
                - (PipelineStages::TOP_OF_PIPE
                    | PipelineStages::DRAW_INDIRECT
                    | PipelineStages::VERTEX_INPUT
                    | PipelineStages::VERTEX_SHADER
                    | PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER
                    | PipelineStages::GEOMETRY_SHADER
                    | PipelineStages::FRAGMENT_SHADER
                    | PipelineStages::EARLY_FRAGMENT_TESTS
                    | PipelineStages::LATE_FRAGMENT_TESTS
                    | PipelineStages::COLOR_ATTACHMENT_OUTPUT
                    | PipelineStages::COMPUTE_SHADER
                    | PipelineStages::ALL_TRANSFER
                    | PipelineStages::BOTTOM_OF_PIPE
                    | PipelineStages::HOST
                    | PipelineStages::ALL_GRAPHICS
                    | PipelineStages::ALL_COMMANDS
                    | PipelineStages::TRANSFORM_FEEDBACK
                    | PipelineStages::CONDITIONAL_RENDERING
                    | PipelineStages::ACCELERATION_STRUCTURE_BUILD
                    | PipelineStages::RAY_TRACING_SHADER
                    | PipelineStages::FRAGMENT_DENSITY_PROCESS
                    | PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT
                    | PipelineStages::COMMAND_PREPROCESS
                    | PipelineStages::TASK_SHADER
                    | PipelineStages::MESH_SHADER))
                .is_empty()
        }

        /// Replaces and unsets flags that are equivalent to multiple other flags.
        ///
        /// This may set flags that are not supported by the device, so this is for internal use only
        /// and should not be passed on to Vulkan.
        pub(crate) fn normalize(mut self) -> Self {
            if self.intersects(PipelineStages::ALL_COMMANDS) {
                self -= PipelineStages::ALL_COMMANDS;
                self |= PipelineStages::TOP_OF_PIPE
                    | PipelineStages::DRAW_INDIRECT
                    | PipelineStages::VERTEX_INPUT
                    | PipelineStages::VERTEX_SHADER
                    | PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER
                    | PipelineStages::GEOMETRY_SHADER
                    | PipelineStages::FRAGMENT_SHADER
                    | PipelineStages::EARLY_FRAGMENT_TESTS
                    | PipelineStages::LATE_FRAGMENT_TESTS
                    | PipelineStages::COLOR_ATTACHMENT_OUTPUT
                    | PipelineStages::COMPUTE_SHADER
                    | PipelineStages::ALL_TRANSFER
                    | PipelineStages::BOTTOM_OF_PIPE
                    | PipelineStages::HOST
                    | PipelineStages::ALL_GRAPHICS
                    | PipelineStages::COPY
                    | PipelineStages::RESOLVE
                    | PipelineStages::BLIT
                    | PipelineStages::CLEAR
                    | PipelineStages::INDEX_INPUT
                    | PipelineStages::VERTEX_ATTRIBUTE_INPUT
                    | PipelineStages::PRE_RASTERIZATION_SHADERS
                    | PipelineStages::VIDEO_DECODE
                    | PipelineStages::VIDEO_ENCODE
                    | PipelineStages::TRANSFORM_FEEDBACK
                    | PipelineStages::CONDITIONAL_RENDERING
                    | PipelineStages::ACCELERATION_STRUCTURE_BUILD
                    | PipelineStages::RAY_TRACING_SHADER
                    | PipelineStages::FRAGMENT_DENSITY_PROCESS
                    | PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT
                    | PipelineStages::COMMAND_PREPROCESS
                    | PipelineStages::TASK_SHADER
                    | PipelineStages::MESH_SHADER
                    | PipelineStages::SUBPASS_SHADING
                    | PipelineStages::INVOCATION_MASK;
            }

            if self.intersects(PipelineStages::ALL_GRAPHICS) {
                self -= PipelineStages::ALL_GRAPHICS;
                self |= PipelineStages::DRAW_INDIRECT
                    | PipelineStages::TASK_SHADER
                    | PipelineStages::MESH_SHADER
                    | PipelineStages::VERTEX_INPUT
                    | PipelineStages::VERTEX_SHADER
                    | PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER
                    | PipelineStages::GEOMETRY_SHADER
                    | PipelineStages::FRAGMENT_SHADER
                    | PipelineStages::EARLY_FRAGMENT_TESTS
                    | PipelineStages::LATE_FRAGMENT_TESTS
                    | PipelineStages::COLOR_ATTACHMENT_OUTPUT
                    | PipelineStages::TRANSFORM_FEEDBACK
                    | PipelineStages::CONDITIONAL_RENDERING
                    | PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT
                    | PipelineStages::FRAGMENT_DENSITY_PROCESS
                    | PipelineStages::INVOCATION_MASK;
            }

            if self.intersects(PipelineStages::VERTEX_INPUT) {
                self -= PipelineStages::VERTEX_INPUT;
                self |= PipelineStages::INDEX_INPUT | PipelineStages::VERTEX_ATTRIBUTE_INPUT;
            }

            if self.intersects(PipelineStages::PRE_RASTERIZATION_SHADERS) {
                self -= PipelineStages::PRE_RASTERIZATION_SHADERS;
                self |= PipelineStages::VERTEX_SHADER
                    | PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER
                    | PipelineStages::GEOMETRY_SHADER
                    | PipelineStages::TASK_SHADER
                    | PipelineStages::MESH_SHADER;
            }

            if self.intersects(PipelineStages::ALL_TRANSFER) {
                self -= PipelineStages::ALL_TRANSFER;
                self |= PipelineStages::COPY
                    | PipelineStages::RESOLVE
                    | PipelineStages::BLIT
                    | PipelineStages::CLEAR;
                //PipelineStages::ACCELERATION_STRUCTURE_COPY;
            }

            self
        }

        /// Returns the access types that are supported with the given pipeline stages.
        ///
        /// Corresponds to the table
        /// "[Supported access types](https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-access-types-supported)"
        /// in the Vulkan specification.
        #[inline]
        pub fn supported_access(mut self) -> AccessFlags {
            if self.is_empty() {
                return AccessFlags::empty();
            }

            self = self.normalize();
            let mut result = AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE;

            if self.intersects(PipelineStages::DRAW_INDIRECT) {
                result |=
                    AccessFlags::INDIRECT_COMMAND_READ | AccessFlags::TRANSFORM_FEEDBACK_COUNTER_READ;
            }

            if self.intersects(PipelineStages::VERTEX_INPUT) {}

            if self.intersects(PipelineStages::VERTEX_SHADER) {
                result |= AccessFlags::SHADER_READ
                    | AccessFlags::UNIFORM_READ
                    | AccessFlags::SHADER_SAMPLED_READ
                    | AccessFlags::SHADER_STORAGE_READ
                    | AccessFlags::SHADER_WRITE
                    | AccessFlags::SHADER_STORAGE_WRITE
                    | AccessFlags::ACCELERATION_STRUCTURE_READ;
            }

            if self.intersects(PipelineStages::TESSELLATION_CONTROL_SHADER) {
                result |= AccessFlags::SHADER_READ
                    | AccessFlags::UNIFORM_READ
                    | AccessFlags::SHADER_SAMPLED_READ
                    | AccessFlags::SHADER_STORAGE_READ
                    | AccessFlags::SHADER_WRITE
                    | AccessFlags::SHADER_STORAGE_WRITE
                    | AccessFlags::ACCELERATION_STRUCTURE_READ;
            }

            if self.intersects(PipelineStages::TESSELLATION_EVALUATION_SHADER) {
                result |= AccessFlags::SHADER_READ
                    | AccessFlags::UNIFORM_READ
                    | AccessFlags::SHADER_SAMPLED_READ
                    | AccessFlags::SHADER_STORAGE_READ
                    | AccessFlags::SHADER_WRITE
                    | AccessFlags::SHADER_STORAGE_WRITE
                    | AccessFlags::ACCELERATION_STRUCTURE_READ;
            }

            if self.intersects(PipelineStages::GEOMETRY_SHADER) {
                result |= AccessFlags::SHADER_READ
                    | AccessFlags::UNIFORM_READ
                    | AccessFlags::SHADER_SAMPLED_READ
                    | AccessFlags::SHADER_STORAGE_READ
                    | AccessFlags::SHADER_WRITE
                    | AccessFlags::SHADER_STORAGE_WRITE
                    | AccessFlags::ACCELERATION_STRUCTURE_READ;
            }

            if self.intersects(PipelineStages::FRAGMENT_SHADER) {
                result |= AccessFlags::SHADER_READ
                    | AccessFlags::UNIFORM_READ
                    | AccessFlags::SHADER_SAMPLED_READ
                    | AccessFlags::SHADER_STORAGE_READ
                    | AccessFlags::SHADER_WRITE
                    | AccessFlags::SHADER_STORAGE_WRITE
                    | AccessFlags::ACCELERATION_STRUCTURE_READ
                    | AccessFlags::INPUT_ATTACHMENT_READ;
            }

            if self.intersects(PipelineStages::EARLY_FRAGMENT_TESTS) {
                result |= AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
            }

            if self.intersects(PipelineStages::LATE_FRAGMENT_TESTS) {
                result |= AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
            }

            if self.intersects(PipelineStages::COLOR_ATTACHMENT_OUTPUT) {
                result |= AccessFlags::COLOR_ATTACHMENT_READ
                    | AccessFlags::COLOR_ATTACHMENT_WRITE
                    | AccessFlags::COLOR_ATTACHMENT_READ_NONCOHERENT;
            }

            if self.intersects(PipelineStages::COMPUTE_SHADER) {
                result |= AccessFlags::SHADER_READ
                    | AccessFlags::UNIFORM_READ
                    | AccessFlags::SHADER_SAMPLED_READ
                    | AccessFlags::SHADER_STORAGE_READ
                    | AccessFlags::SHADER_WRITE
                    | AccessFlags::SHADER_STORAGE_WRITE
                    | AccessFlags::ACCELERATION_STRUCTURE_READ;
            }

            if self.intersects(PipelineStages::ALL_TRANSFER) {}

            if self.intersects(PipelineStages::BOTTOM_OF_PIPE) {}

            if self.intersects(PipelineStages::HOST) {
                result |= AccessFlags::HOST_READ | AccessFlags::HOST_WRITE;
            }

            if self.intersects(PipelineStages::ALL_GRAPHICS) {}

            if self.intersects(PipelineStages::ALL_COMMANDS) {}

            if self.intersects(PipelineStages::COPY) {
                result |= AccessFlags::TRANSFER_READ | AccessFlags::TRANSFER_WRITE;
            }

            if self.intersects(PipelineStages::RESOLVE) {
                result |= AccessFlags::TRANSFER_READ | AccessFlags::TRANSFER_WRITE;
            }

            if self.intersects(PipelineStages::BLIT) {
                result |= AccessFlags::TRANSFER_READ | AccessFlags::TRANSFER_WRITE;
            }

            if self.intersects(PipelineStages::CLEAR) {
                result |= AccessFlags::TRANSFER_WRITE;
            }

            if self.intersects(PipelineStages::INDEX_INPUT) {
                result |= AccessFlags::INDEX_READ;
            }

            if self.intersects(PipelineStages::VERTEX_ATTRIBUTE_INPUT) {
                result |= AccessFlags::VERTEX_ATTRIBUTE_READ;
            }

            if self.intersects(PipelineStages::PRE_RASTERIZATION_SHADERS) {}

            if self.intersects(PipelineStages::VIDEO_DECODE) {
                result |= AccessFlags::VIDEO_DECODE_READ | AccessFlags::VIDEO_DECODE_WRITE;
            }

            if self.intersects(PipelineStages::VIDEO_ENCODE) {
                result |= AccessFlags::VIDEO_ENCODE_READ | AccessFlags::VIDEO_ENCODE_WRITE;
            }

            if self.intersects(PipelineStages::TRANSFORM_FEEDBACK) {
                result |= AccessFlags::TRANSFORM_FEEDBACK_WRITE
                    | AccessFlags::TRANSFORM_FEEDBACK_COUNTER_WRITE
                    | AccessFlags::TRANSFORM_FEEDBACK_COUNTER_READ;
            }

            if self.intersects(PipelineStages::CONDITIONAL_RENDERING) {
                result |= AccessFlags::CONDITIONAL_RENDERING_READ;
            }

            if self.intersects(PipelineStages::ACCELERATION_STRUCTURE_BUILD) {
                result |= AccessFlags::INDIRECT_COMMAND_READ
                    | AccessFlags::SHADER_READ
                    | AccessFlags::SHADER_SAMPLED_READ
                    | AccessFlags::SHADER_STORAGE_READ
                    | AccessFlags::SHADER_STORAGE_WRITE
                    | AccessFlags::TRANSFER_READ
                    | AccessFlags::TRANSFER_WRITE
                    | AccessFlags::ACCELERATION_STRUCTURE_READ
                    | AccessFlags::ACCELERATION_STRUCTURE_WRITE;
            }

            if self.intersects(PipelineStages::RAY_TRACING_SHADER) {
                result |= AccessFlags::SHADER_READ
                    | AccessFlags::UNIFORM_READ
                    | AccessFlags::SHADER_SAMPLED_READ
                    | AccessFlags::SHADER_STORAGE_READ
                    | AccessFlags::SHADER_WRITE
                    | AccessFlags::SHADER_STORAGE_WRITE
                    | AccessFlags::ACCELERATION_STRUCTURE_READ;
            }

            if self.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS) {
                result |= AccessFlags::FRAGMENT_DENSITY_MAP_READ;
            }

            if self.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT) {
                result |= AccessFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_READ;
            }

            if self.intersects(PipelineStages::COMMAND_PREPROCESS) {
                result |= AccessFlags::COMMAND_PREPROCESS_READ | AccessFlags::COMMAND_PREPROCESS_WRITE;
            }

            if self.intersects(PipelineStages::TASK_SHADER) {
                result |= AccessFlags::SHADER_READ
                    | AccessFlags::UNIFORM_READ
                    | AccessFlags::SHADER_SAMPLED_READ
                    | AccessFlags::SHADER_STORAGE_READ
                    | AccessFlags::SHADER_WRITE
                    | AccessFlags::SHADER_STORAGE_WRITE
                    | AccessFlags::ACCELERATION_STRUCTURE_READ;
            }

            if self.intersects(PipelineStages::MESH_SHADER) {
                result |= AccessFlags::SHADER_READ
                    | AccessFlags::UNIFORM_READ
                    | AccessFlags::SHADER_SAMPLED_READ
                    | AccessFlags::SHADER_STORAGE_READ
                    | AccessFlags::SHADER_WRITE
                    | AccessFlags::SHADER_STORAGE_WRITE
                    | AccessFlags::ACCELERATION_STRUCTURE_READ;
            }

            if self.intersects(PipelineStages::SUBPASS_SHADING) {
                result |= AccessFlags::INPUT_ATTACHMENT_READ;
            }

            if self.intersects(PipelineStages::INVOCATION_MASK) {
                result |= AccessFlags::INVOCATION_MASK_READ;
            }

            result
        }
    },

    /// A single stage in the device's processing pipeline.
    PipelineStage impl {
        #[inline]
        pub fn required_queue_flags(self) -> QueueFlags {
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-pipeline-stages-supported
            match self {
                Self::TopOfPipe => QueueFlags::empty(),
                Self::DrawIndirect => QueueFlags::GRAPHICS | QueueFlags::COMPUTE,
                Self::VertexInput => QueueFlags::GRAPHICS,
                Self::VertexShader => QueueFlags::GRAPHICS,
                Self::TessellationControlShader => QueueFlags::GRAPHICS,
                Self::TessellationEvaluationShader => QueueFlags::GRAPHICS,
                Self::GeometryShader => QueueFlags::GRAPHICS,
                Self::FragmentShader => QueueFlags::GRAPHICS,
                Self::EarlyFragmentTests => QueueFlags::GRAPHICS,
                Self::LateFragmentTests => QueueFlags::GRAPHICS,
                Self::ColorAttachmentOutput => QueueFlags::GRAPHICS,
                Self::ComputeShader => QueueFlags::COMPUTE,
                Self::AllTransfer => QueueFlags::GRAPHICS | QueueFlags::COMPUTE | QueueFlags::TRANSFER,
                Self::BottomOfPipe => QueueFlags::empty(),
                Self::Host => QueueFlags::empty(),
                Self::AllGraphics => QueueFlags::GRAPHICS,
                Self::AllCommands => QueueFlags::empty(),
                Self::Copy => todo!(
                    "The spec doesn't currently say which queue flags support this pipeline stage"
                ),
                Self::Resolve => todo!(
                    "The spec doesn't currently say which queue flags support this pipeline stage"
                ),
                Self::Blit => todo!(
                    "The spec doesn't currently say which queue flags support this pipeline stage"
                ),
                Self::Clear => todo!(
                    "The spec doesn't currently say which queue flags support this pipeline stage"
                ),
                Self::IndexInput => QueueFlags::GRAPHICS,
                Self::VertexAttributeInput => QueueFlags::GRAPHICS,
                Self::PreRasterizationShaders => QueueFlags::GRAPHICS,
                Self::VideoDecode => QueueFlags::VIDEO_DECODE,
                Self::VideoEncode => QueueFlags::VIDEO_ENCODE,
                Self::ConditionalRendering => QueueFlags::GRAPHICS | QueueFlags::COMPUTE,
                Self::TransformFeedback => QueueFlags::GRAPHICS,
                Self::CommandPreprocess => QueueFlags::GRAPHICS | QueueFlags::COMPUTE,
                Self::FragmentShadingRateAttachment => QueueFlags::GRAPHICS,
                Self::TaskShader => QueueFlags::GRAPHICS,
                Self::MeshShader => QueueFlags::GRAPHICS,
                Self::AccelerationStructureBuild => QueueFlags::COMPUTE,
                Self::RayTracingShader => QueueFlags::COMPUTE,
                Self::FragmentDensityProcess => QueueFlags::GRAPHICS,
                Self::SubpassShading => QueueFlags::GRAPHICS,
                Self::InvocationMask => todo!(
                    "The spec doesn't currently say which queue flags support this pipeline stage"
                ),
            }
        }
    },

    = PipelineStageFlags2(u64);

    /// A pseudo-stage representing the start of the pipeline.
    TOP_OF_PIPE, TopOfPipe = TOP_OF_PIPE,

    /// Indirect buffers are read.
    DRAW_INDIRECT, DrawIndirect = DRAW_INDIRECT,

    /// Vertex and index buffers are read.
    ///
    /// It is currently equivalent to setting all of the following flags, but automatically
    /// omitting any that are not supported in a given context. It also implicitly includes future
    /// flags that are added to Vulkan, if they are not yet supported by Vulkano.
    /// - `index_input`
    /// - `vertex_attribute_input`
    VERTEX_INPUT, VertexInput = VERTEX_INPUT,

    /// Vertex shaders are executed.
    VERTEX_SHADER, VertexShader = VERTEX_SHADER,

    /// Tessellation control shaders are executed.
    TESSELLATION_CONTROL_SHADER, TessellationControlShader = TESSELLATION_CONTROL_SHADER,

    /// Tessellation evaluation shaders are executed.
    TESSELLATION_EVALUATION_SHADER, TessellationEvaluationShader = TESSELLATION_EVALUATION_SHADER,

    /// Geometry shaders are executed.
    GEOMETRY_SHADER, GeometryShader = GEOMETRY_SHADER,

    /// Fragment shaders are executed.
    FRAGMENT_SHADER, FragmentShader = FRAGMENT_SHADER,

    /// Early fragment tests (depth and stencil tests before fragment shading) are performed.
    /// Subpass load operations for framebuffer attachments with a depth/stencil format are
    /// performed.
    EARLY_FRAGMENT_TESTS, EarlyFragmentTests = EARLY_FRAGMENT_TESTS,

    /// Late fragment tests (depth and stencil tests after fragment shading) are performed.
    /// Subpass store operations for framebuffer attachments with a depth/stencil format are
    /// performed.
    LATE_FRAGMENT_TESTS, LateFragmentTests = LATE_FRAGMENT_TESTS,

    /// The final color values are output from the pipeline after blending.
    /// Subpass load and store operations, multisample resolve operations for framebuffer
    /// attachments with a color or depth/stencil format, and `clear_attachments` are performed.
    COLOR_ATTACHMENT_OUTPUT, ColorAttachmentOutput = COLOR_ATTACHMENT_OUTPUT,

    /// Compute shaders are executed.
    COMPUTE_SHADER, ComputeShader = COMPUTE_SHADER,

    /// The set of all current and future transfer pipeline stages.
    ///
    /// It is currently equivalent to setting all of the following flags, but automatically
    /// omitting any that are not supported in a given context. It also implicitly includes future
    /// flags that are added to Vulkan, if they are not yet supported by Vulkano.
    /// - `copy`
    /// - `blit`
    /// - `resolve`
    /// - `clear`
    /// - `acceleration_structure_copy`
    ALL_TRANSFER, AllTransfer = ALL_TRANSFER,

    /// A pseudo-stage representing the end of the pipeline.
    BOTTOM_OF_PIPE, BottomOfPipe = BOTTOM_OF_PIPE,

    /// A pseudo-stage representing reads and writes to device memory on the host.
    HOST, Host = HOST,

    /// The set of all current and future graphics pipeline stages.
    ///
    /// It is currently equivalent to setting all of the following flags, but automatically
    /// omitting any that are not supported in a given context. It also implicitly includes future
    /// flags that are added to Vulkan, if they are not yet supported by Vulkano.
    /// - `draw_indirect`
    /// - `task_shader`
    /// - `mesh_shader`
    /// - `vertex_input`
    /// - `vertex_shader`
    /// - `tessellation_control_shader`
    /// - `tessellation_evaluation_shader`
    /// - `geometry_shader`
    /// - `fragment_shader`
    /// - `early_fragment_tests`
    /// - `late_fragment_tests`
    /// - `color_attachment_output`
    /// - `conditional_rendering`
    /// - `transform_feedback`
    /// - `fragment_shading_rate_attachment`
    /// - `fragment_density_process`
    /// - `invocation_mask`
    ALL_GRAPHICS, AllGraphics = ALL_GRAPHICS,

    /// The set of all current and future pipeline stages of all types.
    ///
    /// It is currently equivalent to setting all flags in `PipelineStages`, but automatically
    /// omitting any that are not supported in a given context. It also implicitly includes future
    /// flags that are added to Vulkan, if they are not yet supported by Vulkano.
    ALL_COMMANDS, AllCommands = ALL_COMMANDS,

    /// The `copy_buffer`, `copy_image`, `copy_buffer_to_image`, `copy_image_to_buffer` and
    /// `copy_query_pool_results` commands are executed.
    COPY, Copy = COPY {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// The `resolve_image` command is executed.
    RESOLVE, Resolve = RESOLVE {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// The `blit_image` command is executed.
    BLIT, Blit = BLIT {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// The `clear_color_image`, `clear_depth_stencil_image`, `fill_buffer` and `update_buffer`
    /// commands are executed.
    CLEAR, Clear = CLEAR {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// Index buffers are read.
    INDEX_INPUT, IndexInput = INDEX_INPUT {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// Vertex buffers are read.
    VERTEX_ATTRIBUTE_INPUT, VertexAttributeInput = VERTEX_ATTRIBUTE_INPUT {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// The various pre-rasterization shader types are executed.
    ///
    /// It is currently equivalent to setting all of the following flags, but automatically
    /// omitting any that are not supported in a given context. It also implicitly includes future
    /// flags that are added to Vulkan, if they are not yet supported by Vulkano.
    /// - `vertex_shader`
    /// - `tessellation_control_shader`
    /// - `tessellation_evaluation_shader`
    /// - `geometry_shader`
    /// - `task_shader`
    /// - `mesh_shader`
    PRE_RASTERIZATION_SHADERS, PreRasterizationShaders = PRE_RASTERIZATION_SHADERS {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// Video decode operations are performed.
    VIDEO_DECODE, VideoDecode = VIDEO_DECODE_KHR {
        device_extensions: [khr_video_decode_queue],
    },

    /// Video encode operations are performed.
    VIDEO_ENCODE, VideoEncode = VIDEO_ENCODE_KHR {
        device_extensions: [khr_video_encode_queue],
    },

    /// Vertex attribute output values are written to the transform feedback buffers.
    TRANSFORM_FEEDBACK, TransformFeedback = TRANSFORM_FEEDBACK_EXT {
        device_extensions: [ext_transform_feedback],
    },

    /// The predicate of conditional rendering is read.
    CONDITIONAL_RENDERING, ConditionalRendering = CONDITIONAL_RENDERING_EXT {
        device_extensions: [ext_conditional_rendering],
    },

    /// Acceleration_structure commands are executed.
    ACCELERATION_STRUCTURE_BUILD, AccelerationStructureBuild = ACCELERATION_STRUCTURE_BUILD_KHR {
        device_extensions: [khr_acceleration_structure, nv_ray_tracing],
    },

    /// The various ray tracing shader types are executed.
    RAY_TRACING_SHADER, RayTracingShader = RAY_TRACING_SHADER_KHR {
        device_extensions: [khr_ray_tracing_pipeline, nv_ray_tracing],
    },

    /// The fragment density map is read to generate the fragment areas.
    FRAGMENT_DENSITY_PROCESS, FragmentDensityProcess = FRAGMENT_DENSITY_PROCESS_EXT {
        device_extensions: [ext_fragment_density_map],
    },

    /// The fragment shading rate attachment or shading rate image is read to determine the
    /// fragment shading rate for portions of a rasterized primitive.
    FRAGMENT_SHADING_RATE_ATTACHMENT, FragmentShadingRateAttachment = FRAGMENT_SHADING_RATE_ATTACHMENT_KHR {
        device_extensions: [khr_fragment_shading_rate],
    },

    /// Device-side preprocessing for generated commands via the `preprocess_generated_commands`
    /// command is handled.
    COMMAND_PREPROCESS, CommandPreprocess = COMMAND_PREPROCESS_NV {
        device_extensions: [nv_device_generated_commands],
    },

    /// Task shaders are executed.
    TASK_SHADER, TaskShader = TASK_SHADER_NV {
        device_extensions: [nv_mesh_shader],
    },

    /// Mesh shaders are executed.
    MESH_SHADER, MeshShader = MESH_SHADER_NV {
        device_extensions: [nv_mesh_shader],
    },

    /// Subpass shading shaders are executed.
    SUBPASS_SHADING, SubpassShading = SUBPASS_SHADING_HUAWEI {
        device_extensions: [huawei_subpass_shading],
    },

    /// The invocation mask image is read to optimize ray dispatch.
    INVOCATION_MASK, InvocationMask = INVOCATION_MASK_HUAWEI {
        device_extensions: [huawei_invocation_mask],
    },

    /*
    ACCELERATION_STRUCTURE_COPY, AccelerationStructureCopy = ACCELERATION_STRUCTURE_COPY_KHR {
        device_extensions: [khr_ray_tracing_maintenance1],
    },

    MICROMAP_BUILD, MicromapBuild = MICROMAP_BUILD_EXT {
        device_extensions: [ext_opacity_micromap],
    },

    OPTICAL_FLOW, OpticalFlow = OPTICAL_FLOW_NV {
        device_extensions: [nv_optical_flow],
    },
     */
}

impl From<PipelineStage> for ash::vk::PipelineStageFlags {
    #[inline]
    fn from(val: PipelineStage) -> Self {
        Self::from_raw(val as u32)
    }
}

impl From<PipelineStages> for ash::vk::PipelineStageFlags {
    #[inline]
    fn from(val: PipelineStages) -> Self {
        Self::from_raw(ash::vk::PipelineStageFlags2::from(val).as_raw() as u32)
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// A set of memory access types that are included in a memory dependency.
    AccessFlags impl {
        /// Returns whether `self` contains stages that are only available in
        /// `VkAccessFlagBits2`.
        pub(crate) fn is_2(self) -> bool {
            !(self
                - (AccessFlags::INDIRECT_COMMAND_READ
                    | AccessFlags::INDEX_READ
                    | AccessFlags::VERTEX_ATTRIBUTE_READ
                    | AccessFlags::UNIFORM_READ
                    | AccessFlags::INPUT_ATTACHMENT_READ
                    | AccessFlags::SHADER_READ
                    | AccessFlags::SHADER_WRITE
                    | AccessFlags::COLOR_ATTACHMENT_READ
                    | AccessFlags::COLOR_ATTACHMENT_WRITE
                    | AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                    | AccessFlags::TRANSFER_READ
                    | AccessFlags::TRANSFER_WRITE
                    | AccessFlags::HOST_READ
                    | AccessFlags::HOST_WRITE
                    | AccessFlags::MEMORY_READ
                    | AccessFlags::MEMORY_WRITE
                    | AccessFlags::SHADER_SAMPLED_READ
                    | AccessFlags::SHADER_STORAGE_READ
                    | AccessFlags::SHADER_STORAGE_WRITE
                    | AccessFlags::VIDEO_DECODE_READ
                    | AccessFlags::VIDEO_DECODE_WRITE
                    | AccessFlags::VIDEO_ENCODE_READ
                    | AccessFlags::VIDEO_ENCODE_WRITE
                    | AccessFlags::TRANSFORM_FEEDBACK_WRITE
                    | AccessFlags::TRANSFORM_FEEDBACK_COUNTER_READ
                    | AccessFlags::TRANSFORM_FEEDBACK_COUNTER_WRITE
                    | AccessFlags::CONDITIONAL_RENDERING_READ
                    | AccessFlags::COMMAND_PREPROCESS_READ
                    | AccessFlags::COMMAND_PREPROCESS_WRITE
                    | AccessFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_READ
                    | AccessFlags::ACCELERATION_STRUCTURE_READ
                    | AccessFlags::ACCELERATION_STRUCTURE_WRITE
                    | AccessFlags::FRAGMENT_DENSITY_MAP_READ
                    | AccessFlags::COLOR_ATTACHMENT_READ_NONCOHERENT
                    | AccessFlags::INVOCATION_MASK_READ))
                .is_empty()
        }

        /// Replaces and unsets flags that are equivalent to multiple other flags.
        ///
        /// This may set flags that are not supported by the device, so this is for internal use only
        /// and should not be passed on to Vulkan.
        #[allow(dead_code)] // TODO: use this function
        pub(crate) fn normalize(mut self) -> Self {
            if self.intersects(AccessFlags::SHADER_READ) {
                self -= AccessFlags::SHADER_READ;
                self |= AccessFlags::UNIFORM_READ
                    | AccessFlags::SHADER_SAMPLED_READ
                    | AccessFlags::SHADER_STORAGE_READ;
            }

            if self.intersects(AccessFlags::SHADER_WRITE) {
                self -= AccessFlags::SHADER_WRITE;
                self |= AccessFlags::SHADER_STORAGE_WRITE;
            }

            self
        }
    }
    = AccessFlags2(u64);

    /// Read access to an indirect buffer.
    INDIRECT_COMMAND_READ = INDIRECT_COMMAND_READ,

    /// Read access to an index buffer.
    INDEX_READ = INDEX_READ,

    /// Read access to a vertex buffer.
    VERTEX_ATTRIBUTE_READ = VERTEX_ATTRIBUTE_READ,

    /// Read access to a uniform buffer in a shader.
    UNIFORM_READ = UNIFORM_READ,

    /// Read access to an input attachment in a fragment shader, within a render pass.
    INPUT_ATTACHMENT_READ = INPUT_ATTACHMENT_READ,

    /// Read access to a buffer or image in a shader.
    ///
    /// It is currently equivalent to setting all of the following flags, but automatically
    /// omitting any that are not supported in a given context. It also implicitly includes future
    /// flags that are added to Vulkan, if they are not yet supported by Vulkano.
    /// - `uniform_read`
    /// - `shader_sampled_read`
    /// - `shader_storage_read`
    SHADER_READ = SHADER_READ,

    /// Write access to a buffer or image in a shader.
    ///
    /// It is currently equivalent to `shader_storage_write`. It also implicitly includes future
    /// flags that are added to Vulkan, if they are not yet supported by Vulkano.
    SHADER_WRITE = SHADER_WRITE,

    /// Read access to a color attachment during blending, logic operations or
    /// subpass load operations.
    COLOR_ATTACHMENT_READ = COLOR_ATTACHMENT_READ,

    /// Write access to a color, resolve or depth/stencil resolve attachment during a render pass
    /// or subpass store operations.
    COLOR_ATTACHMENT_WRITE = COLOR_ATTACHMENT_WRITE,

    /// Read access to a depth/stencil attachment during depth/stencil operations or
    /// subpass load operations.
    DEPTH_STENCIL_ATTACHMENT_READ = DEPTH_STENCIL_ATTACHMENT_READ,

    /// Write access to a depth/stencil attachment during depth/stencil operations or
    /// subpass store operations.
    DEPTH_STENCIL_ATTACHMENT_WRITE = DEPTH_STENCIL_ATTACHMENT_WRITE,

    /// Read access to a buffer or image during a copy, blit or resolve command.
    TRANSFER_READ = TRANSFER_READ,

    /// Write access to a buffer or image during a copy, blit, resolve or clear command.
    TRANSFER_WRITE = TRANSFER_WRITE,

    /// Read access performed by the host.
    HOST_READ = HOST_READ,

    /// Write access performed by the host.
    HOST_WRITE = HOST_WRITE,

    /// Any type of read access.
    ///
    /// This is equivalent to setting all `_read` flags that are allowed in the given context.
    MEMORY_READ = MEMORY_READ,

    /// Any type of write access.
    ///
    /// This is equivalent to setting all `_write` flags that are allowed in the given context.
    MEMORY_WRITE = MEMORY_WRITE,

    /// Read access to a uniform texel buffer or sampled image in a shader.
    SHADER_SAMPLED_READ = SHADER_SAMPLED_READ {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// Read access to a storage buffer, storage texel buffer or storage image in a shader.
    SHADER_STORAGE_READ = SHADER_STORAGE_READ {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// Write access to a storage buffer, storage texel buffer or storage image in a shader.
    SHADER_STORAGE_WRITE = SHADER_STORAGE_WRITE {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// Read access to an image or buffer as part of a video decode operation.
    VIDEO_DECODE_READ = VIDEO_DECODE_READ_KHR {
        device_extensions: [khr_video_decode_queue],
    },

    /// Write access to an image or buffer as part of a video decode operation.
    VIDEO_DECODE_WRITE = VIDEO_DECODE_WRITE_KHR {
        device_extensions: [khr_video_decode_queue],
    },

    /// Read access to an image or buffer as part of a video encode operation.
    VIDEO_ENCODE_READ = VIDEO_ENCODE_READ_KHR {
        device_extensions: [khr_video_encode_queue],
    },

    /// Write access to an image or buffer as part of a video encode operation.
    VIDEO_ENCODE_WRITE = VIDEO_ENCODE_WRITE_KHR {
        device_extensions: [khr_video_encode_queue],
    },

    /// Write access to a transform feedback buffer during transform feedback operations.
    TRANSFORM_FEEDBACK_WRITE = TRANSFORM_FEEDBACK_WRITE_EXT {
        device_extensions: [ext_transform_feedback],
    },

    /// Read access to a transform feedback counter buffer during transform feedback operations.
    TRANSFORM_FEEDBACK_COUNTER_READ = TRANSFORM_FEEDBACK_COUNTER_READ_EXT {
        device_extensions: [ext_transform_feedback],
    },

    /// Write access to a transform feedback counter buffer during transform feedback operations.
    TRANSFORM_FEEDBACK_COUNTER_WRITE = TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT {
        device_extensions: [ext_transform_feedback],
    },

    /// Read access to a predicate during conditional rendering.
    CONDITIONAL_RENDERING_READ = CONDITIONAL_RENDERING_READ_EXT {
        device_extensions: [ext_conditional_rendering],
    },

    /// Read access to preprocess buffers input to `preprocess_generated_commands`.
    COMMAND_PREPROCESS_READ = COMMAND_PREPROCESS_READ_NV {
        device_extensions: [nv_device_generated_commands],
    },

    /// Read access to sequences buffers output by `preprocess_generated_commands`.
    COMMAND_PREPROCESS_WRITE = COMMAND_PREPROCESS_WRITE_NV {
        device_extensions: [nv_device_generated_commands],
    },

    /// Read access to a fragment shading rate attachment during rasterization.
    FRAGMENT_SHADING_RATE_ATTACHMENT_READ = FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR {
        device_extensions: [khr_fragment_shading_rate],
    },

    /// Read access to an acceleration structure or acceleration structure scratch buffer during
    /// trace, build or copy commands.
    ACCELERATION_STRUCTURE_READ = ACCELERATION_STRUCTURE_READ_KHR {
        device_extensions: [khr_acceleration_structure, nv_ray_tracing],
    },

    /// Write access to an acceleration structure or acceleration structure scratch buffer during
    /// trace, build or copy commands.
    ACCELERATION_STRUCTURE_WRITE = ACCELERATION_STRUCTURE_WRITE_KHR {
        device_extensions: [khr_acceleration_structure, nv_ray_tracing],
    },

    /// Read access to a fragment density map attachment during dynamic fragment density map
    /// operations.
    FRAGMENT_DENSITY_MAP_READ = FRAGMENT_DENSITY_MAP_READ_EXT {
        device_extensions: [ext_fragment_density_map],
    },

    /// Read access to color attachments when performing advanced blend operations.
    COLOR_ATTACHMENT_READ_NONCOHERENT = COLOR_ATTACHMENT_READ_NONCOHERENT_EXT {
        device_extensions: [ext_blend_operation_advanced],
    },

    /// Read access to an invocation mask image.
    INVOCATION_MASK_READ = INVOCATION_MASK_READ_HUAWEI {
        device_extensions: [huawei_invocation_mask],
    },

    /*
    SHADER_BINDING_TABLE_READ = SHADER_BINDING_TABLE_READ_KHR {
        device_extensions: [khr_ray_tracing_maintenance1],
    },

    MICROMAP_READ = MICROMAP_READ_EXT {
        device_extensions: [ext_opacity_micromap],
    },

    MICROMAP_WRITE = MICROMAP_WRITE_EXT {
        device_extensions: [ext_opacity_micromap],
    },

    OPTICAL_FLOW_READ = OPTICAL_FLOW_READ_NV {
        device_extensions: [nv_optical_flow],
    },

    OPTICAL_FLOW_WRITE = OPTICAL_FLOW_WRITE_NV {
        device_extensions: [nv_optical_flow],
    },
    */
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
/// access types. The pipeline stages create an *execution dependency*: the `src_stages` of
/// commands submitted before the barrier must be completely finished before before any of the
/// `dst_stages` of commands after the barrier are allowed to start. The memory access types
/// create a *memory dependency*: in addition to the execution dependency, any `src_access`
/// performed before the barrier must be made available and visible before any `dst_access`
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
    pub src_stages: PipelineStages,

    /// The memory accesses in the source scope to make available and visible.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub src_access: AccessFlags,

    /// The pipeline stages in the destination scope that must wait for `src_stages`.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub dst_stages: PipelineStages,

    /// The memory accesses in the destination scope that must wait for `src_access` to be made
    /// available and visible.
    pub dst_access: AccessFlags,

    pub _ne: crate::NonExhaustive,
}

impl Default for MemoryBarrier {
    #[inline]
    fn default() -> Self {
        Self {
            src_stages: PipelineStages::empty(),
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStages::empty(),
            dst_access: AccessFlags::empty(),
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
    pub src_stages: PipelineStages,

    /// The memory accesses in the source scope to make available and visible.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub src_access: AccessFlags,

    /// The pipeline stages in the destination scope that must wait for `src_stages`.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub dst_stages: PipelineStages,

    /// The memory accesses in the destination scope that must wait for `src_access` to be made
    /// available and visible.
    pub dst_access: AccessFlags,

    /// For resources created with [`Sharing::Exclusive`](crate::sync::Sharing), transfers
    /// ownership of a resource from one queue family to another.
    pub queue_family_transfer: Option<QueueFamilyTransfer>,

    /// The buffer to apply the barrier to.
    pub buffer: Arc<Buffer>,

    /// The byte range of `buffer` to apply the barrier to.
    pub range: Range<DeviceSize>,

    pub _ne: crate::NonExhaustive,
}

impl BufferMemoryBarrier {
    #[inline]
    pub fn buffer(buffer: Arc<Buffer>) -> Self {
        Self {
            src_stages: PipelineStages::empty(),
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStages::empty(),
            dst_access: AccessFlags::empty(),
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
    pub src_stages: PipelineStages,

    /// The memory accesses in the source scope to make available and visible.
    ///
    /// The default value is [`AccessFlags::empty()`].
    pub src_access: AccessFlags,

    /// The pipeline stages in the destination scope that must wait for `src_stages`.
    ///
    /// The default value is [`PipelineStages::empty()`].
    pub dst_stages: PipelineStages,

    /// The memory accesses in the destination scope that must wait for `src_access` to be made
    /// available and visible.
    pub dst_access: AccessFlags,

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
    pub image: Arc<Image>,

    /// The subresource range of `image` to apply the barrier to.
    pub subresource_range: ImageSubresourceRange,

    pub _ne: crate::NonExhaustive,
}

impl ImageMemoryBarrier {
    #[inline]
    pub fn image(image: Arc<Image>) -> Self {
        Self {
            src_stages: PipelineStages::empty(),
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStages::empty(),
            dst_access: AccessFlags::empty(),
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
