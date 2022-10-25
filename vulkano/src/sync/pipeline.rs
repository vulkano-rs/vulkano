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
    /// A single stage in the device's processing pipeline.
    #[non_exhaustive]
    PipelineStage = PipelineStageFlags2(u64);

    /// A pseudo-stage representing the start of the pipeline.
    TopOfPipe = TOP_OF_PIPE,

    /// Indirect buffers are read.
    DrawIndirect = DRAW_INDIRECT,

    /// Vertex and index buffers are read.
    ///
    /// It is currently equivalent to setting all of the following flags, but automatically
    /// omitting any that are not supported in a given context. It also implicitly includes future
    /// flags that are added to Vulkan, if they are not yet supported by Vulkano.
    /// - `index_input`
    /// - `vertex_attribute_input`
    VertexInput = VERTEX_INPUT,

    /// Vertex shaders are executed.
    VertexShader = VERTEX_SHADER,

    /// Tessellation control shaders are executed.
    TessellationControlShader = TESSELLATION_CONTROL_SHADER,

    /// Tessellation evaluation shaders are executed.
    TessellationEvaluationShader = TESSELLATION_EVALUATION_SHADER,

    /// Geometry shaders are executed.
    GeometryShader = GEOMETRY_SHADER,

    /// Fragment shaders are executed.
    FragmentShader = FRAGMENT_SHADER,

    /// Early fragment tests (depth and stencil tests before fragment shading) are performed.
    /// Subpass load operations for framebuffer attachments with a depth/stencil format are
    /// performed.
    EarlyFragmentTests = EARLY_FRAGMENT_TESTS,

    /// Late fragment tests (depth and stencil tests after fragment shading) are performed.
    /// Subpass store operations for framebuffer attachments with a depth/stencil format are
    /// performed.
    LateFragmentTests = LATE_FRAGMENT_TESTS,

    /// The final color values are output from the pipeline after blending.
    /// Subpass load and store operations, multisample resolve operations for framebuffer
    /// attachments with a color or depth/stencil format, and `clear_attachments` are performed.
    ColorAttachmentOutput = COLOR_ATTACHMENT_OUTPUT,

    /// Compute shaders are executed.
    ComputeShader = COMPUTE_SHADER,

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
    AllTransfer = ALL_TRANSFER,

    /// A pseudo-stage representing the end of the pipeline.
    BottomOfPipe = BOTTOM_OF_PIPE,

    /// A pseudo-stage representing reads and writes to device memory on the host.
    Host = HOST,

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
    AllGraphics = ALL_GRAPHICS,

    /// The set of all current and future pipeline stages of all types.
    ///
    /// It is currently equivalent to setting all flags in `PipelineStages`, but automatically
    /// omitting any that are not supported in a given context. It also implicitly includes future
    /// flags that are added to Vulkan, if they are not yet supported by Vulkano.
    AllCommands = ALL_COMMANDS,

    /// The `copy_buffer`, `copy_image`, `copy_buffer_to_image`, `copy_image_to_buffer` and
    /// `copy_query_pool_results` commands are executed.
    Copy = COPY {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// The `resolve_image` command is executed.
    Resolve = RESOLVE {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// The `blit_image` command is executed.
    Blit = BLIT {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// The `clear_color_image`, `clear_depth_stencil_image`, `fill_buffer` and `update_buffer`
    /// commands are executed.
    Clear = CLEAR {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// Index buffers are read.
    IndexInput = INDEX_INPUT {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// Vertex buffers are read.
    VertexAttributeInput = VERTEX_ATTRIBUTE_INPUT {
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
    PreRasterizationShaders = PRE_RASTERIZATION_SHADERS {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// Video decode operations are performed.
    VideoDecode = VIDEO_DECODE_KHR {
        device_extensions: [khr_video_decode_queue],
    },

    /// Video encode operations are performed.
    VideoEncode = VIDEO_ENCODE_KHR {
        device_extensions: [khr_video_encode_queue],
    },

    /// Vertex attribute output values are written to the transform feedback buffers.
    TransformFeedback = TRANSFORM_FEEDBACK_EXT {
        device_extensions: [ext_transform_feedback],
    },

    /// The predicate of conditional rendering is read.
    ConditionalRendering = CONDITIONAL_RENDERING_EXT {
        device_extensions: [ext_conditional_rendering],
    },

    /// Acceleration_structure commands are executed.
    AccelerationStructureBuild = ACCELERATION_STRUCTURE_BUILD_KHR {
        device_extensions: [khr_acceleration_structure, nv_ray_tracing],
    },

    /// The various ray tracing shader types are executed.
    RayTracingShader = RAY_TRACING_SHADER_KHR {
        device_extensions: [khr_ray_tracing_pipeline, nv_ray_tracing],
    },

    /// The fragment density map is read to generate the fragment areas.
    FragmentDensityProcess = FRAGMENT_DENSITY_PROCESS_EXT {
        device_extensions: [ext_fragment_density_map],
    },

    /// The fragment shading rate attachment or shading rate image is read to determine the
    /// fragment shading rate for portions of a rasterized primitive.
    FragmentShadingRateAttachment = FRAGMENT_SHADING_RATE_ATTACHMENT_KHR {
        device_extensions: [khr_fragment_shading_rate],
    },

    /// Device-side preprocessing for generated commands via the `preprocess_generated_commands`
    /// command is handled.
    CommandPreprocess = COMMAND_PREPROCESS_NV {
        device_extensions: [nv_device_generated_commands],
    },

    /// Task shaders are executed.
    TaskShader = TASK_SHADER_NV {
        device_extensions: [nv_mesh_shader],
    },

    /// Mesh shaders are executed.
    MeshShader = MESH_SHADER_NV {
        device_extensions: [nv_mesh_shader],
    },

    /// Subpass shading shaders are executed.
    SubpassShading = SUBPASS_SHADING_HUAWEI {
        device_extensions: [huawei_subpass_shading],
    },

    /// The invocation mask image is read to optimize ray dispatch.
    InvocationMask = INVOCATION_MASK_HUAWEI {
        device_extensions: [huawei_invocation_mask],
    },

    /*
    AccelerationStructureCopy = ACCELERATION_STRUCTURE_COPY_KHR {
        device_extensions: [khr_ray_tracing_maintenance1],
    },

    MicromapBuild = MICROMAP_BUILD_EXT {
        device_extensions: [ext_opacity_micromap],
    },

    OpticalFlow = OPTICAL_FLOW_NV {
        device_extensions: [nv_optical_flow],
    },
     */
}

impl PipelineStage {
    #[inline]
    pub fn required_queue_flags(&self) -> ash::vk::QueueFlags {
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-pipeline-stages-supported
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
            Self::AllTransfer => {
                ash::vk::QueueFlags::GRAPHICS
                    | ash::vk::QueueFlags::COMPUTE
                    | ash::vk::QueueFlags::TRANSFER
            }
            Self::BottomOfPipe => ash::vk::QueueFlags::empty(),
            Self::Host => ash::vk::QueueFlags::empty(),
            Self::AllGraphics => ash::vk::QueueFlags::GRAPHICS,
            Self::AllCommands => ash::vk::QueueFlags::empty(),
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
            Self::IndexInput => ash::vk::QueueFlags::GRAPHICS,
            Self::VertexAttributeInput => ash::vk::QueueFlags::GRAPHICS,
            Self::PreRasterizationShaders => ash::vk::QueueFlags::GRAPHICS,
            Self::VideoDecode => ash::vk::QueueFlags::VIDEO_DECODE_KHR,
            Self::VideoEncode => ash::vk::QueueFlags::VIDEO_ENCODE_KHR,
            Self::ConditionalRendering => {
                ash::vk::QueueFlags::GRAPHICS | ash::vk::QueueFlags::COMPUTE
            }
            Self::TransformFeedback => ash::vk::QueueFlags::GRAPHICS,
            Self::CommandPreprocess => ash::vk::QueueFlags::GRAPHICS | ash::vk::QueueFlags::COMPUTE,
            Self::FragmentShadingRateAttachment => ash::vk::QueueFlags::GRAPHICS,
            Self::TaskShader => ash::vk::QueueFlags::GRAPHICS,
            Self::MeshShader => ash::vk::QueueFlags::GRAPHICS,
            Self::AccelerationStructureBuild => ash::vk::QueueFlags::COMPUTE,
            Self::RayTracingShader => ash::vk::QueueFlags::COMPUTE,
            Self::FragmentDensityProcess => ash::vk::QueueFlags::GRAPHICS,
            Self::SubpassShading => ash::vk::QueueFlags::GRAPHICS,
            Self::InvocationMask => todo!(
                "The spec doesn't currently say which queue flags support this pipeline stage"
            ),
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
    /// A set of stages in the device's processing pipeline.
    #[non_exhaustive]
    PipelineStages = PipelineStageFlags2(u64);

    /// A pseudo-stage representing the start of the pipeline.
    top_of_pipe = TOP_OF_PIPE,

    /// Indirect buffers are read.
    draw_indirect = DRAW_INDIRECT,

    /// Vertex and index buffers are read.
    ///
    /// It is currently equivalent to setting all of the following flags, but automatically
    /// omitting any that are not supported in a given context. It also implicitly includes future
    /// flags that are added to Vulkan, if they are not yet supported by Vulkano.
    /// - `index_input`
    /// - `vertex_attribute_input`
    vertex_input = VERTEX_INPUT,

    /// Vertex shaders are executed.
    vertex_shader = VERTEX_SHADER,

    /// Tessellation control shaders are executed.
    tessellation_control_shader = TESSELLATION_CONTROL_SHADER,

    /// Tessellation evaluation shaders are executed.
    tessellation_evaluation_shader = TESSELLATION_EVALUATION_SHADER,

    /// Geometry shaders are executed.
    geometry_shader = GEOMETRY_SHADER,

    /// Fragment shaders are executed.
    fragment_shader = FRAGMENT_SHADER,

    /// Early fragment tests (depth and stencil tests before fragment shading) are performed.
    /// Subpass load operations for framebuffer attachments with a depth/stencil format are
    /// performed.
    early_fragment_tests = EARLY_FRAGMENT_TESTS,

    /// Late fragment tests (depth and stencil tests after fragment shading) are performed.
    /// Subpass store operations for framebuffer attachments with a depth/stencil format are
    /// performed.
    late_fragment_tests = LATE_FRAGMENT_TESTS,

    /// The final color values are output from the pipeline after blending.
    /// Subpass load and store operations, multisample resolve operations for framebuffer
    /// attachments with a color or depth/stencil format, and `clear_attachments` are performed.
    color_attachment_output = COLOR_ATTACHMENT_OUTPUT,

    /// Compute shaders are executed.
    compute_shader = COMPUTE_SHADER,

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
    all_transfer = ALL_TRANSFER,

    /// A pseudo-stage representing the end of the pipeline.
    bottom_of_pipe = BOTTOM_OF_PIPE,

    /// A pseudo-stage representing reads and writes to device memory on the host.
    host = HOST,

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
    all_graphics = ALL_GRAPHICS,

    /// The set of all current and future pipeline stages of all types.
    ///
    /// It is currently equivalent to setting all flags in `PipelineStages`, but automatically
    /// omitting any that are not supported in a given context. It also implicitly includes future
    /// flags that are added to Vulkan, if they are not yet supported by Vulkano.
    all_commands = ALL_COMMANDS,

    /// The `copy_buffer`, `copy_image`, `copy_buffer_to_image`, `copy_image_to_buffer` and
    /// `copy_query_pool_results` commands are executed.
    copy = COPY {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// The `resolve_image` command is executed.
    resolve = RESOLVE {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// The `blit_image` command is executed.
    blit = BLIT {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// The `clear_color_image`, `clear_depth_stencil_image`, `fill_buffer` and `update_buffer`
    /// commands are executed.
    clear = CLEAR {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// Index buffers are read.
    index_input = INDEX_INPUT {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// Vertex buffers are read.
    vertex_attribute_input = VERTEX_ATTRIBUTE_INPUT {
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
    pre_rasterization_shaders = PRE_RASTERIZATION_SHADERS {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// Video decode operations are performed.
    video_decode = VIDEO_DECODE_KHR {
        device_extensions: [khr_video_decode_queue],
    },

    /// Video encode operations are performed.
    video_encode = VIDEO_ENCODE_KHR {
        device_extensions: [khr_video_encode_queue],
    },

    /// Vertex attribute output values are written to the transform feedback buffers.
    transform_feedback = TRANSFORM_FEEDBACK_EXT {
        device_extensions: [ext_transform_feedback],
    },

    /// The predicate of conditional rendering is read.
    conditional_rendering = CONDITIONAL_RENDERING_EXT {
        device_extensions: [ext_conditional_rendering],
    },

    /// Acceleration_structure commands are executed.
    acceleration_structure_build = ACCELERATION_STRUCTURE_BUILD_KHR {
        device_extensions: [khr_acceleration_structure, nv_ray_tracing],
    },

    /// The various ray tracing shader types are executed.
    ray_tracing_shader = RAY_TRACING_SHADER_KHR {
        device_extensions: [khr_ray_tracing_pipeline, nv_ray_tracing],
    },

    /// The fragment density map is read to generate the fragment areas.
    fragment_density_process = FRAGMENT_DENSITY_PROCESS_EXT {
        device_extensions: [ext_fragment_density_map],
    },

    /// The fragment shading rate attachment or shading rate image is read to determine the
    /// fragment shading rate for portions of a rasterized primitive.
    fragment_shading_rate_attachment = FRAGMENT_SHADING_RATE_ATTACHMENT_KHR {
        device_extensions: [khr_fragment_shading_rate, nv_shading_rate_image],
    },

    /// Device-side preprocessing for generated commands via the `preprocess_generated_commands`
    /// command is handled.
    command_preprocess = COMMAND_PREPROCESS_NV {
        device_extensions: [nv_device_generated_commands],
    },

    /// Task shaders are executed.
    task_shader = TASK_SHADER_NV {
        device_extensions: [nv_mesh_shader],
    },

    /// Mesh shaders are executed.
    mesh_shader = MESH_SHADER_NV {
        device_extensions: [nv_mesh_shader],
    },

    /// Subpass shading shaders are executed.
    subpass_shading = SUBPASS_SHADING_HUAWEI {
        device_extensions: [huawei_subpass_shading],
    },

    /// The invocation mask image is read to optimize ray dispatch.
    invocation_mask = INVOCATION_MASK_HUAWEI {
        device_extensions: [huawei_invocation_mask],
    },

    /*
    acceleration_structure_copy = ACCELERATION_STRUCTURE_COPY_KHR {
        device_extensions: [khr_ray_tracing_maintenance1],
    },

    micromap_build = MICROMAP_BUILD_EXT {
        device_extensions: [ext_opacity_micromap],
    },

    optical_flow = OPTICAL_FLOW_NV {
        device_extensions: [nv_optical_flow],
    },
     */
}

impl PipelineStages {
    /// Returns whether `self` contains stages that are only available in
    /// `VkPipelineStageFlagBits2`.
    pub(crate) fn is_2(&self) -> bool {
        !Self {
            top_of_pipe: false,
            draw_indirect: false,
            vertex_input: false,
            vertex_shader: false,
            tessellation_control_shader: false,
            tessellation_evaluation_shader: false,
            geometry_shader: false,
            fragment_shader: false,
            early_fragment_tests: false,
            late_fragment_tests: false,
            color_attachment_output: false,
            compute_shader: false,
            all_transfer: false,
            bottom_of_pipe: false,
            host: false,
            all_graphics: false,
            all_commands: false,
            transform_feedback: false,
            conditional_rendering: false,
            acceleration_structure_build: false,
            ray_tracing_shader: false,
            fragment_density_process: false,
            fragment_shading_rate_attachment: false,
            command_preprocess: false,
            task_shader: false,
            mesh_shader: false,
            ..*self
        }
        .is_empty()
    }

    /// Replaces and unsets flags that are equivalent to multiple other flags.
    ///
    /// This may set flags that are not supported by the device, so this is for internal use only
    /// and should not be passed on to Vulkan.
    pub(crate) fn normalize(mut self) -> Self {
        if self.all_commands {
            self = Self {
                all_commands: false,

                top_of_pipe: true,
                draw_indirect: true,
                vertex_input: true,
                vertex_shader: true,
                tessellation_control_shader: true,
                tessellation_evaluation_shader: true,
                geometry_shader: true,
                fragment_shader: true,
                early_fragment_tests: true,
                late_fragment_tests: true,
                color_attachment_output: true,
                compute_shader: true,
                all_transfer: true,
                bottom_of_pipe: true,
                host: true,
                all_graphics: true,
                copy: true,
                resolve: true,
                blit: true,
                clear: true,
                index_input: true,
                vertex_attribute_input: true,
                pre_rasterization_shaders: true,
                video_decode: true,
                video_encode: true,
                transform_feedback: true,
                conditional_rendering: true,
                acceleration_structure_build: true,
                ray_tracing_shader: true,
                fragment_density_process: true,
                fragment_shading_rate_attachment: true,
                command_preprocess: true,
                task_shader: true,
                mesh_shader: true,
                subpass_shading: true,
                invocation_mask: true,
                _ne: crate::NonExhaustive(()),
            }
        }

        if self.all_graphics {
            self = Self {
                all_graphics: false,

                draw_indirect: true,
                task_shader: true,
                mesh_shader: true,
                vertex_input: true,
                vertex_shader: true,
                tessellation_control_shader: true,
                tessellation_evaluation_shader: true,
                geometry_shader: true,
                fragment_shader: true,
                early_fragment_tests: true,
                late_fragment_tests: true,
                color_attachment_output: true,
                transform_feedback: true,
                conditional_rendering: true,
                fragment_shading_rate_attachment: true,
                fragment_density_process: true,
                invocation_mask: true,
                ..self
            }
        }

        if self.vertex_input {
            self = Self {
                vertex_input: false,

                index_input: true,
                vertex_attribute_input: true,
                ..self
            }
        }

        if self.pre_rasterization_shaders {
            self = Self {
                pre_rasterization_shaders: false,

                vertex_shader: true,
                tessellation_control_shader: true,
                tessellation_evaluation_shader: true,
                geometry_shader: true,
                task_shader: true,
                mesh_shader: true,
                ..self
            }
        }

        if self.all_transfer {
            self = Self {
                all_transfer: false,

                copy: true,
                resolve: true,
                blit: true,
                clear: true,
                //acceleration_structure_copy: true,
                ..self
            }
        }

        self
    }

    /// Returns the access types that are supported with the given pipeline stages.
    ///
    /// Corresponds to the table
    /// "[Supported access types](https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-access-types-supported)"
    /// in the Vulkan specification.
    #[inline]
    pub fn supported_access(&self) -> AccessFlags {
        let PipelineStages {
            top_of_pipe: _,
            draw_indirect,
            vertex_input: _,
            vertex_shader,
            tessellation_control_shader,
            tessellation_evaluation_shader,
            geometry_shader,
            fragment_shader,
            early_fragment_tests,
            late_fragment_tests,
            color_attachment_output,
            compute_shader,
            all_transfer: _,
            bottom_of_pipe: _,
            host,
            all_graphics: _,
            all_commands: _,
            copy,
            resolve,
            blit,
            clear,
            index_input,
            vertex_attribute_input,
            pre_rasterization_shaders: _,
            video_decode,
            video_encode,
            transform_feedback,
            conditional_rendering,
            acceleration_structure_build,
            ray_tracing_shader,
            fragment_density_process,
            fragment_shading_rate_attachment,
            command_preprocess,
            task_shader,
            mesh_shader,
            subpass_shading,
            invocation_mask,
            //acceleration_structure_copy,
            _ne: _,
        } = self.normalize();

        AccessFlags {
            indirect_command_read: draw_indirect || acceleration_structure_build,
            index_read: index_input,
            vertex_attribute_read: vertex_attribute_input,
            uniform_read: task_shader
                || mesh_shader
                || ray_tracing_shader
                || vertex_shader
                || tessellation_control_shader
                || tessellation_evaluation_shader
                || geometry_shader
                || fragment_shader
                || compute_shader,
            shader_read: acceleration_structure_build
                || task_shader
                || mesh_shader
                || ray_tracing_shader
                // || micromap_build
                || vertex_shader
                || tessellation_control_shader
                || tessellation_evaluation_shader
                || geometry_shader
                || fragment_shader
                || compute_shader,
            shader_write: task_shader
                || mesh_shader
                || ray_tracing_shader
                || vertex_shader
                || tessellation_control_shader
                || tessellation_evaluation_shader
                || geometry_shader
                || fragment_shader
                || compute_shader,
            input_attachment_read: subpass_shading || fragment_shader,
            color_attachment_read: color_attachment_output,
            color_attachment_write: color_attachment_output,
            depth_stencil_attachment_read: early_fragment_tests || late_fragment_tests,
            depth_stencil_attachment_write: early_fragment_tests || late_fragment_tests,
            transfer_read: copy || blit || resolve || acceleration_structure_build,
            transfer_write: copy || blit || resolve || clear || acceleration_structure_build,
            host_read: host,
            host_write: host,
            memory_read: true,
            memory_write: true,
            shader_sampled_read: acceleration_structure_build
                || task_shader
                || mesh_shader
                || ray_tracing_shader
                // || micromap_build
                || vertex_shader
                || tessellation_control_shader
                || tessellation_evaluation_shader
                || geometry_shader
                || fragment_shader
                || compute_shader,
            shader_storage_read: acceleration_structure_build
                || task_shader
                || mesh_shader
                || ray_tracing_shader
                // || micromap_build
                || vertex_shader
                || tessellation_control_shader
                || tessellation_evaluation_shader
                || geometry_shader
                || fragment_shader
                || compute_shader,
            shader_storage_write: acceleration_structure_build
                || task_shader
                || mesh_shader
                || ray_tracing_shader
                || vertex_shader
                || tessellation_control_shader
                || tessellation_evaluation_shader
                || geometry_shader
                || fragment_shader
                || compute_shader,
            video_decode_read: video_decode,
            video_decode_write: video_decode,
            video_encode_read: video_encode,
            video_encode_write: video_encode,
            transform_feedback_write: transform_feedback,
            transform_feedback_counter_write: transform_feedback,
            transform_feedback_counter_read: transform_feedback || draw_indirect,
            conditional_rendering_read: conditional_rendering,
            command_preprocess_read: command_preprocess,
            command_preprocess_write: command_preprocess,
            fragment_shading_rate_attachment_read: fragment_shading_rate_attachment,
            acceleration_structure_read: task_shader
                || mesh_shader
                || vertex_shader
                || tessellation_control_shader
                || tessellation_evaluation_shader
                || geometry_shader
                || fragment_shader
                || compute_shader
                || ray_tracing_shader
                || acceleration_structure_build,
            acceleration_structure_write: acceleration_structure_build,
            fragment_density_map_read: fragment_density_process,
            color_attachment_read_noncoherent: color_attachment_output,
            invocation_mask_read: invocation_mask,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl From<PipelineStages> for ash::vk::PipelineStageFlags {
    #[inline]
    fn from(val: PipelineStages) -> Self {
        Self::from_raw(ash::vk::PipelineStageFlags2::from(val).as_raw() as u32)
    }
}

impl From<PipelineStage> for PipelineStages {
    #[inline]
    fn from(val: PipelineStage) -> Self {
        let mut result = Self::empty();

        match val {
            PipelineStage::TopOfPipe => result.top_of_pipe = true,
            PipelineStage::DrawIndirect => result.draw_indirect = true,
            PipelineStage::VertexInput => result.vertex_input = true,
            PipelineStage::VertexShader => result.vertex_shader = true,
            PipelineStage::TessellationControlShader => result.tessellation_control_shader = true,
            PipelineStage::TessellationEvaluationShader => {
                result.tessellation_evaluation_shader = true
            }
            PipelineStage::GeometryShader => result.geometry_shader = true,
            PipelineStage::FragmentShader => result.fragment_shader = true,
            PipelineStage::EarlyFragmentTests => result.early_fragment_tests = true,
            PipelineStage::LateFragmentTests => result.late_fragment_tests = true,
            PipelineStage::ColorAttachmentOutput => result.color_attachment_output = true,
            PipelineStage::ComputeShader => result.compute_shader = true,
            PipelineStage::AllTransfer => result.all_transfer = true,
            PipelineStage::BottomOfPipe => result.bottom_of_pipe = true,
            PipelineStage::Host => result.host = true,
            PipelineStage::AllGraphics => result.all_graphics = true,
            PipelineStage::AllCommands => result.all_commands = true,
            PipelineStage::Copy => result.copy = true,
            PipelineStage::Resolve => result.resolve = true,
            PipelineStage::Blit => result.blit = true,
            PipelineStage::Clear => result.clear = true,
            PipelineStage::IndexInput => result.index_input = true,
            PipelineStage::VertexAttributeInput => result.vertex_attribute_input = true,
            PipelineStage::PreRasterizationShaders => result.pre_rasterization_shaders = true,
            PipelineStage::VideoDecode => result.video_decode = true,
            PipelineStage::VideoEncode => result.video_encode = true,
            PipelineStage::TransformFeedback => result.transform_feedback = true,
            PipelineStage::ConditionalRendering => result.conditional_rendering = true,
            PipelineStage::AccelerationStructureBuild => result.acceleration_structure_build = true,
            PipelineStage::RayTracingShader => result.ray_tracing_shader = true,
            PipelineStage::FragmentDensityProcess => result.fragment_density_process = true,
            PipelineStage::FragmentShadingRateAttachment => {
                result.fragment_shading_rate_attachment = true
            }
            PipelineStage::CommandPreprocess => result.command_preprocess = true,
            PipelineStage::TaskShader => result.task_shader = true,
            PipelineStage::MeshShader => result.mesh_shader = true,
            PipelineStage::SubpassShading => result.subpass_shading = true,
            PipelineStage::InvocationMask => result.invocation_mask = true,
        }

        result
    }
}

vulkan_bitflags! {
    /// A set of memory access types that are included in a memory dependency.
    #[non_exhaustive]
    AccessFlags = AccessFlags2(u64);

    /// Read access to an indirect buffer.
    indirect_command_read = INDIRECT_COMMAND_READ,

    /// Read access to an index buffer.
    index_read = INDEX_READ,

    /// Read access to a vertex buffer.
    vertex_attribute_read = VERTEX_ATTRIBUTE_READ,

    /// Read access to a uniform buffer in a shader.
    uniform_read = UNIFORM_READ,

    /// Read access to an input attachment in a fragment shader, within a render pass.
    input_attachment_read = INPUT_ATTACHMENT_READ,

    /// Read access to a buffer or image in a shader.
    ///
    /// It is currently equivalent to setting all of the following flags, but automatically
    /// omitting any that are not supported in a given context. It also implicitly includes future
    /// flags that are added to Vulkan, if they are not yet supported by Vulkano.
    /// - `uniform_read`
    /// - `shader_sampled_read`
    /// - `shader_storage_read`
    shader_read = SHADER_READ,

    /// Write access to a buffer or image in a shader.
    ///
    /// It is currently equivalent to `shader_storage_write`. It also implicitly includes future
    /// flags that are added to Vulkan, if they are not yet supported by Vulkano.
    shader_write = SHADER_WRITE,

    /// Read access to a color attachment during blending, logic operations or
    /// subpass load operations.
    color_attachment_read = COLOR_ATTACHMENT_READ,

    /// Write access to a color, resolve or depth/stencil resolve attachment during a render pass
    /// or subpass store operations.
    color_attachment_write = COLOR_ATTACHMENT_WRITE,

    /// Read access to a depth/stencil attachment during depth/stencil operations or
    /// subpass load operations.
    depth_stencil_attachment_read = DEPTH_STENCIL_ATTACHMENT_READ,

    /// Write access to a depth/stencil attachment during depth/stencil operations or
    /// subpass store operations.
    depth_stencil_attachment_write = DEPTH_STENCIL_ATTACHMENT_WRITE,

    /// Read access to a buffer or image during a copy, blit or resolve command.
    transfer_read = TRANSFER_READ,

    /// Write access to a buffer or image during a copy, blit, resolve or clear command.
    transfer_write = TRANSFER_WRITE,

    /// Read access performed by the host.
    host_read = HOST_READ,

    /// Write access performed by the host.
    host_write = HOST_WRITE,

    /// Any type of read access.
    ///
    /// This is equivalent to setting all `_read` flags that are allowed in the given context.
    memory_read = MEMORY_READ,

    /// Any type of write access.
    ///
    /// This is equivalent to setting all `_write` flags that are allowed in the given context.
    memory_write = MEMORY_WRITE,

    /// Read access to a uniform texel buffer or sampled image in a shader.
    shader_sampled_read = SHADER_SAMPLED_READ {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// Read access to a storage buffer, storage texel buffer or storage image in a shader.
    shader_storage_read = SHADER_STORAGE_READ {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// Write access to a storage buffer, storage texel buffer or storage image in a shader.
    shader_storage_write = SHADER_STORAGE_WRITE {
        api_version: V1_3,
        device_extensions: [khr_synchronization2],
    },

    /// Read access to an image or buffer as part of a video decode operation.
    video_decode_read = VIDEO_DECODE_READ_KHR {
        device_extensions: [khr_video_decode_queue],
    },

    /// Write access to an image or buffer as part of a video decode operation.
    video_decode_write = VIDEO_DECODE_WRITE_KHR {
        device_extensions: [khr_video_decode_queue],
    },

    /// Read access to an image or buffer as part of a video encode operation.
    video_encode_read = VIDEO_ENCODE_READ_KHR {
        device_extensions: [khr_video_encode_queue],
    },

    /// Write access to an image or buffer as part of a video encode operation.
    video_encode_write = VIDEO_ENCODE_WRITE_KHR {
        device_extensions: [khr_video_encode_queue],
    },

    /// Write access to a transform feedback buffer during transform feedback operations.
    transform_feedback_write = TRANSFORM_FEEDBACK_WRITE_EXT {
        device_extensions: [ext_transform_feedback],
    },

    /// Read access to a transform feedback counter buffer during transform feedback operations.
    transform_feedback_counter_read = TRANSFORM_FEEDBACK_COUNTER_READ_EXT {
        device_extensions: [ext_transform_feedback],
    },

    /// Write access to a transform feedback counter buffer during transform feedback operations.
    transform_feedback_counter_write = TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT {
        device_extensions: [ext_transform_feedback],
    },

    /// Read access to a predicate during conditional rendering.
    conditional_rendering_read = CONDITIONAL_RENDERING_READ_EXT {
        device_extensions: [ext_conditional_rendering],
    },

    /// Read access to preprocess buffers input to `preprocess_generated_commands`.
    command_preprocess_read = COMMAND_PREPROCESS_READ_NV {
        device_extensions: [nv_device_generated_commands],
    },

    /// Read access to sequences buffers output by `preprocess_generated_commands`.
    command_preprocess_write = COMMAND_PREPROCESS_WRITE_NV {
        device_extensions: [nv_device_generated_commands],
    },

    /// Read access to a fragment shading rate attachment during rasterization.
    fragment_shading_rate_attachment_read = FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR {
        device_extensions: [khr_fragment_shading_rate],
    },

    /// Read access to an acceleration structure or acceleration structure scratch buffer during
    /// trace, build or copy commands.
    acceleration_structure_read = ACCELERATION_STRUCTURE_READ_KHR {
        device_extensions: [khr_acceleration_structure, nv_ray_tracing],
    },

    /// Write access to an acceleration structure or acceleration structure scratch buffer during
    /// trace, build or copy commands.
    acceleration_structure_write = ACCELERATION_STRUCTURE_WRITE_KHR {
        device_extensions: [khr_acceleration_structure, nv_ray_tracing],
    },

    /// Read access to a fragment density map attachment during dynamic fragment density map
    /// operations.
    fragment_density_map_read = FRAGMENT_DENSITY_MAP_READ_EXT {
        device_extensions: [ext_fragment_density_map],
    },

    /// Read access to color attachments when performing advanced blend operations.
    color_attachment_read_noncoherent = COLOR_ATTACHMENT_READ_NONCOHERENT_EXT {
        device_extensions: [ext_blend_operation_advanced],
    },

    /// Read access to an invocation mask image.
    invocation_mask_read = INVOCATION_MASK_READ_HUAWEI {
        device_extensions: [huawei_invocation_mask],
    },

    /*
    shader_binding_table_read = SHADER_BINDING_TABLE_READ_KHR {
        device_extensions: [khr_ray_tracing_maintenance1],
    },

    micromap_read = MICROMAP_READ_EXT {
        device_extensions: [ext_opacity_micromap],
    },

    micromap_write = MICROMAP_WRITE_EXT {
        device_extensions: [ext_opacity_micromap],
    },

    optical_flow_read = OPTICAL_FLOW_READ_NV {
        device_extensions: [nv_optical_flow],
    },

    optical_flow_write = OPTICAL_FLOW_WRITE_NV {
        device_extensions: [nv_optical_flow],
    },
    */
}

impl AccessFlags {
    /// Returns whether `self` contains stages that are only available in
    /// `VkAccessFlagBits2`.
    pub(crate) fn is_2(&self) -> bool {
        !Self {
            indirect_command_read: false,
            index_read: false,
            vertex_attribute_read: false,
            uniform_read: false,
            input_attachment_read: false,
            shader_read: false,
            shader_write: false,
            color_attachment_read: false,
            color_attachment_write: false,
            depth_stencil_attachment_read: false,
            depth_stencil_attachment_write: false,
            transfer_read: false,
            transfer_write: false,
            host_read: false,
            host_write: false,
            memory_read: false,
            memory_write: false,
            transform_feedback_write: false,
            transform_feedback_counter_read: false,
            transform_feedback_counter_write: false,
            conditional_rendering_read: false,
            color_attachment_read_noncoherent: false,
            acceleration_structure_read: false,
            acceleration_structure_write: false,
            fragment_density_map_read: false,
            fragment_shading_rate_attachment_read: false,
            command_preprocess_read: false,
            command_preprocess_write: false,
            ..*self
        }
        .is_empty()
    }

    /// Replaces and unsets flags that are equivalent to multiple other flags.
    ///
    /// This may set flags that are not supported by the device, so this is for internal use only
    /// and should not be passed on to Vulkan.
    #[allow(dead_code)] // TODO: use this function
    pub(crate) fn normalize(mut self) -> Self {
        if self.shader_read {
            self = Self {
                shader_read: false,

                uniform_read: true,
                shader_sampled_read: true,
                shader_storage_read: true,
                ..self
            }
        }

        if self.shader_write {
            self = Self {
                shader_write: false,

                shader_storage_write: true,
                ..self
            }
        }

        self
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
    pub buffer: Arc<UnsafeBuffer>,

    /// The byte range of `buffer` to apply the barrier to.
    pub range: Range<DeviceSize>,

    pub _ne: crate::NonExhaustive,
}

impl BufferMemoryBarrier {
    #[inline]
    pub fn buffer(buffer: Arc<UnsafeBuffer>) -> Self {
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
    pub image: Arc<UnsafeImage>,

    /// The subresource range of `image` to apply the barrier to.
    pub subresource_range: ImageSubresourceRange,

    pub _ne: crate::NonExhaustive,
}

impl ImageMemoryBarrier {
    #[inline]
    pub fn image(image: Arc<UnsafeImage>) -> Self {
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
