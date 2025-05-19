use crate::{
    buffer::Buffer,
    descriptor_set::layout::DescriptorType,
    device::{Device, QueueFlags},
    image::{
        Image, ImageAspects, ImageCreateFlags, ImageLayout, ImageSubresourceRange, ImageUsage,
    },
    macros::{vulkan_bitflags, vulkan_bitflags_enum},
    shader::ShaderStages,
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError, VulkanObject,
};
use ash::vk;
use foldhash::HashMap;
use smallvec::SmallVec;
use std::{
    ops::Range,
    sync::{Arc, LazyLock},
};

vulkan_bitflags_enum! {
    #[non_exhaustive]
    /// A set of [`PipelineStage`] values.
    PipelineStages impl {
        /// Returns whether `self` contains stages that are only available in
        /// `VkPipelineStageFlagBits2`.
        pub(crate) fn contains_flags2(self) -> bool {
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
        pub(crate) fn expand(mut self, queue_flags: QueueFlags) -> Self {
            if self.intersects(PipelineStages::ALL_COMMANDS) {
                self -= PipelineStages::ALL_COMMANDS;
                self |= queue_flags.into();
            }

            if self.intersects(PipelineStages::ALL_GRAPHICS) {
                self -= PipelineStages::ALL_GRAPHICS;
                self |= QueueFlags::GRAPHICS.into();
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
                    | PipelineStages::CLEAR
                    | PipelineStages::ACCELERATION_STRUCTURE_COPY;
            }

            self
        }

        /// Replaces unsupported flags with more general supported ones.
        pub(crate) fn into_supported(mut self, device: &Device) -> Self {
            if !device.enabled_features().ray_tracing_maintenance1 {
                if self.intersects(PipelineStages::ACCELERATION_STRUCTURE_COPY) {
                    self -= PipelineStages::ACCELERATION_STRUCTURE_COPY;
                    self |= PipelineStages::ACCELERATION_STRUCTURE_BUILD;
                }
            }

            if !device.enabled_features().synchronization2 {
                if self.intersects(
                    PipelineStages::INDEX_INPUT
                    | PipelineStages::VERTEX_ATTRIBUTE_INPUT
                ) {
                    self -= PipelineStages::INDEX_INPUT | PipelineStages::VERTEX_ATTRIBUTE_INPUT;
                    self |= PipelineStages::VERTEX_INPUT;
                }

                if self.intersects(
                    PipelineStages::COPY
                    | PipelineStages::RESOLVE
                    | PipelineStages::BLIT
                    | PipelineStages::CLEAR
                ) {
                    self -= PipelineStages::COPY
                        | PipelineStages::RESOLVE
                        | PipelineStages::BLIT
                        | PipelineStages::CLEAR;
                    self |= PipelineStages::ALL_TRANSFER;
                }

                if self.contains_flags2() {
                    return if (self - PipelineStages::from(QueueFlags::GRAPHICS)).is_empty() {
                        PipelineStages::ALL_GRAPHICS
                    } else {
                        PipelineStages::ALL_COMMANDS
                    };
                }
            }

            match self.validate_device(device) {
                Ok(_) => self,
                Err(_) if (self - PipelineStages::from(QueueFlags::GRAPHICS)).is_empty() => PipelineStages::ALL_GRAPHICS,
                Err(_) => PipelineStages::ALL_COMMANDS,
            }
        }
    },

    /// A single stage in the device's processing pipeline.
    PipelineStage,

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
    COPY, Copy = COPY
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),

    /// The `resolve_image` command is executed.
    RESOLVE, Resolve = RESOLVE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),

    /// The `blit_image` command is executed.
    BLIT, Blit = BLIT
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),

    /// The `clear_color_image`, `clear_depth_stencil_image`, `fill_buffer` and `update_buffer`
    /// commands are executed.
    CLEAR, Clear = CLEAR
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),

    /// Index buffers are read.
    INDEX_INPUT, IndexInput = INDEX_INPUT
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),

    /// Vertex buffers are read.
    VERTEX_ATTRIBUTE_INPUT, VertexAttributeInput = VERTEX_ATTRIBUTE_INPUT
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),

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
    PRE_RASTERIZATION_SHADERS, PreRasterizationShaders = PRE_RASTERIZATION_SHADERS
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),

    /// Video decode operations are performed.
    VIDEO_DECODE, VideoDecode = VIDEO_DECODE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_decode_queue)]),
    ]),

    /// Video encode operations are performed.
    VIDEO_ENCODE, VideoEncode = VIDEO_ENCODE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_encode_queue)]),
    ]),

    /// Vertex attribute output values are written to the transform feedback buffers.
    TRANSFORM_FEEDBACK, TransformFeedback = TRANSFORM_FEEDBACK_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_transform_feedback)]),
    ]),

    /// The predicate of conditional rendering is read.
    CONDITIONAL_RENDERING, ConditionalRendering = CONDITIONAL_RENDERING_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_conditional_rendering)]),
    ]),

    /// Acceleration_structure commands are executed.
    ACCELERATION_STRUCTURE_BUILD, AccelerationStructureBuild = ACCELERATION_STRUCTURE_BUILD_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_acceleration_structure)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    /// The various ray tracing shader types are executed.
    RAY_TRACING_SHADER, RayTracingShader = RAY_TRACING_SHADER_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_pipeline)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    /// The fragment density map is read to generate the fragment areas.
    FRAGMENT_DENSITY_PROCESS, FragmentDensityProcess = FRAGMENT_DENSITY_PROCESS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_fragment_density_map)]),
    ]),

    /// The fragment shading rate attachment or shading rate image is read to determine the
    /// fragment shading rate for portions of a rasterized primitive.
    FRAGMENT_SHADING_RATE_ATTACHMENT, FragmentShadingRateAttachment = FRAGMENT_SHADING_RATE_ATTACHMENT_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_fragment_shading_rate)]),
    ]),

    /// Device-side preprocessing for generated commands via the `preprocess_generated_commands`
    /// command is handled.
    COMMAND_PREPROCESS, CommandPreprocess = COMMAND_PREPROCESS_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_device_generated_commands)]),
    ]),

    /// Task shaders are executed.
    TASK_SHADER, TaskShader = TASK_SHADER_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_mesh_shader)]),
        RequiresAllOf([DeviceExtension(nv_mesh_shader)]),
    ]),

    /// Mesh shaders are executed.
    MESH_SHADER, MeshShader = MESH_SHADER_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_mesh_shader)]),
        RequiresAllOf([DeviceExtension(nv_mesh_shader)]),
    ]),

    /// Subpass shaders are executed.
    SUBPASS_SHADER, SubpassShader = SUBPASS_SHADER_HUAWEI
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(huawei_subpass_shading),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(huawei_subpass_shading),
        ]),
    ]),

    /// The invocation mask image is read to optimize ray dispatch.
    INVOCATION_MASK, InvocationMask = INVOCATION_MASK_HUAWEI
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(huawei_invocation_mask),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(huawei_invocation_mask),
        ]),
    ]),

    /// The `copy_acceleration_structure` command is executed.
    ACCELERATION_STRUCTURE_COPY, AccelerationStructureCopy = ACCELERATION_STRUCTURE_COPY_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_ray_tracing_maintenance1)]),
    ]),

    /// Micromap commands are executed.
    MICROMAP_BUILD, MicromapBuild = MICROMAP_BUILD_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_opacity_micromap)]),
    ]),

    /// Optical flow operations are performed.
    OPTICAL_FLOW, OpticalFlow = OPTICAL_FLOW_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_optical_flow)]),
    ]),
}

impl From<PipelineStage> for vk::PipelineStageFlags {
    #[inline]
    fn from(val: PipelineStage) -> Self {
        Self::from_raw(val as u32)
    }
}

impl From<PipelineStages> for vk::PipelineStageFlags {
    #[inline]
    fn from(val: PipelineStages) -> Self {
        Self::from_raw(vk::PipelineStageFlags2::from(val).as_raw() as u32)
    }
}

impl From<QueueFlags> for PipelineStages {
    /// Corresponds to the table "[Supported pipeline stage flags]" in the Vulkan specification.
    ///
    /// [Supported pipeline stage flags]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-pipeline-stages-supported
    #[inline]
    fn from(val: QueueFlags) -> Self {
        let mut result = PipelineStages::TOP_OF_PIPE
            | PipelineStages::BOTTOM_OF_PIPE
            | PipelineStages::HOST
            | PipelineStages::ALL_COMMANDS;

        if val.intersects(QueueFlags::GRAPHICS | QueueFlags::COMPUTE | QueueFlags::TRANSFER) {
            result |= PipelineStages::ALL_TRANSFER
                | PipelineStages::COPY
                | PipelineStages::RESOLVE
                | PipelineStages::BLIT
                | PipelineStages::CLEAR
                | PipelineStages::ACCELERATION_STRUCTURE_COPY;
        }

        if val.intersects(QueueFlags::GRAPHICS) {
            result |= PipelineStages::DRAW_INDIRECT
                | PipelineStages::VERTEX_INPUT
                | PipelineStages::VERTEX_SHADER
                | PipelineStages::TESSELLATION_CONTROL_SHADER
                | PipelineStages::TESSELLATION_EVALUATION_SHADER
                | PipelineStages::GEOMETRY_SHADER
                | PipelineStages::FRAGMENT_SHADER
                | PipelineStages::EARLY_FRAGMENT_TESTS
                | PipelineStages::LATE_FRAGMENT_TESTS
                | PipelineStages::COLOR_ATTACHMENT_OUTPUT
                | PipelineStages::ALL_GRAPHICS
                | PipelineStages::INDEX_INPUT
                | PipelineStages::VERTEX_ATTRIBUTE_INPUT
                | PipelineStages::PRE_RASTERIZATION_SHADERS
                | PipelineStages::CONDITIONAL_RENDERING
                | PipelineStages::TRANSFORM_FEEDBACK
                | PipelineStages::COMMAND_PREPROCESS
                | PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT
                | PipelineStages::TASK_SHADER
                | PipelineStages::MESH_SHADER
                | PipelineStages::FRAGMENT_DENSITY_PROCESS
                | PipelineStages::SUBPASS_SHADER
                | PipelineStages::INVOCATION_MASK;
        }

        if val.intersects(QueueFlags::COMPUTE) {
            result |= PipelineStages::DRAW_INDIRECT
                | PipelineStages::COMPUTE_SHADER
                | PipelineStages::CONDITIONAL_RENDERING
                | PipelineStages::COMMAND_PREPROCESS
                | PipelineStages::ACCELERATION_STRUCTURE_BUILD
                | PipelineStages::RAY_TRACING_SHADER
                | PipelineStages::MICROMAP_BUILD;
        }

        if val.intersects(QueueFlags::VIDEO_DECODE) {
            result |= PipelineStages::VIDEO_DECODE;
        }

        if val.intersects(QueueFlags::VIDEO_ENCODE) {
            result |= PipelineStages::VIDEO_ENCODE;
        }

        if val.intersects(QueueFlags::OPTICAL_FLOW) {
            result |= PipelineStages::OPTICAL_FLOW;
        }

        result
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// A set of memory access types that are included in a memory dependency.
    AccessFlags impl {
        const WRITES: AccessFlags = AccessFlags::SHADER_WRITE
            .union(AccessFlags::COLOR_ATTACHMENT_WRITE)
            .union(AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .union(AccessFlags::TRANSFER_WRITE)
            .union(AccessFlags::HOST_WRITE)
            .union(AccessFlags::MEMORY_WRITE)
            .union(AccessFlags::SHADER_STORAGE_WRITE)
            .union(AccessFlags::VIDEO_DECODE_WRITE)
            .union(AccessFlags::VIDEO_ENCODE_WRITE)
            .union(AccessFlags::TRANSFORM_FEEDBACK_WRITE)
            .union(AccessFlags::TRANSFORM_FEEDBACK_COUNTER_WRITE)
            .union(AccessFlags::COMMAND_PREPROCESS_WRITE)
            .union(AccessFlags::ACCELERATION_STRUCTURE_WRITE);

        /// Returns whether `self` contains any read flags.
        #[inline]
        pub const fn contains_reads(self) -> bool {
            !self.difference(Self::WRITES).is_empty()
        }

        /// Returns whether `self` contains any write flags.
        #[inline]
        pub const fn contains_writes(self) -> bool {
            self.intersects(Self::WRITES)
        }

        /// Returns whether `self` contains flags that are only available in `VkAccessFlagBits2`.
        pub(crate) fn contains_flags2(self) -> bool {
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
        /// This may set flags that are not supported by the device, so this is for internal use
        /// only and should not be passed on to Vulkan.
        #[allow(dead_code)] // TODO: use this function
        pub(crate) fn expand(mut self) -> Self {
            if self.intersects(AccessFlags::SHADER_READ) {
                self -= AccessFlags::SHADER_READ;
                self |= AccessFlags::SHADER_SAMPLED_READ
                    | AccessFlags::SHADER_STORAGE_READ
                    | AccessFlags::SHADER_BINDING_TABLE_READ;
            }

            if self.intersects(AccessFlags::SHADER_WRITE) {
                self -= AccessFlags::SHADER_WRITE;
                self |= AccessFlags::SHADER_STORAGE_WRITE;
            }

            self
        }

        /// Replaces unsupported flags with more general supported ones.
        pub(crate) fn into_supported(mut self, device: &Device) -> Self {
            if !device.enabled_features().synchronization2 {
                if self.intersects(
                    AccessFlags::SHADER_SAMPLED_READ
                    | AccessFlags::SHADER_STORAGE_READ
                ) {
                    self -= AccessFlags::SHADER_SAMPLED_READ | AccessFlags::SHADER_STORAGE_READ;
                    self |= AccessFlags::SHADER_READ;
                }

                if self.intersects(
                    AccessFlags::SHADER_STORAGE_WRITE
                ) {
                    self -= AccessFlags::SHADER_STORAGE_WRITE;
                    self |= AccessFlags::SHADER_WRITE;
                }

                if self.contains_flags2() {
                    return match (self.contains_reads(), self.contains_writes()) {
                        (true, true) => AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                        (true, false) => AccessFlags::MEMORY_READ,
                        (false, true) => AccessFlags::MEMORY_WRITE,
                        (false, false) => unreachable!(),
                    };
                }
            }

            if self.validate_device(device).is_err() {
                return match (self.contains_reads(), self.contains_writes()) {
                    (true, true) => AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                    (true, false) => AccessFlags::MEMORY_READ,
                    (false, true) => AccessFlags::MEMORY_WRITE,
                    (false, false) => unreachable!(),
                };
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
    /// - `shader_binding_table_read`
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
    SHADER_SAMPLED_READ = SHADER_SAMPLED_READ
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),

    /// Read access to a storage buffer, storage texel buffer or storage image in a shader.
    SHADER_STORAGE_READ = SHADER_STORAGE_READ
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),

    /// Write access to a storage buffer, storage texel buffer or storage image in a shader.
    SHADER_STORAGE_WRITE = SHADER_STORAGE_WRITE
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(khr_synchronization2)]),
    ]),

    /// Read access to an image or buffer as part of a video decode operation.
    VIDEO_DECODE_READ = VIDEO_DECODE_READ_KHR
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(khr_video_decode_queue),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(khr_video_decode_queue),
        ]),
    ]),

    /// Write access to an image or buffer as part of a video decode operation.
    VIDEO_DECODE_WRITE = VIDEO_DECODE_WRITE_KHR
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(khr_video_decode_queue),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(khr_video_decode_queue),
        ]),
    ]),

    /// Read access to an image or buffer as part of a video encode operation.
    VIDEO_ENCODE_READ = VIDEO_ENCODE_READ_KHR
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(khr_video_encode_queue),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(khr_video_encode_queue),
        ]),
    ]),

    /// Write access to an image or buffer as part of a video encode operation.
    VIDEO_ENCODE_WRITE = VIDEO_ENCODE_WRITE_KHR
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(khr_video_encode_queue),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(khr_video_encode_queue),
        ]),
    ]),

    /// Write access to a transform feedback buffer during transform feedback operations.
    TRANSFORM_FEEDBACK_WRITE = TRANSFORM_FEEDBACK_WRITE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_transform_feedback)]),
    ]),

    /// Read access to a transform feedback counter buffer during transform feedback operations.
    TRANSFORM_FEEDBACK_COUNTER_READ = TRANSFORM_FEEDBACK_COUNTER_READ_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_transform_feedback)]),
    ]),

    /// Write access to a transform feedback counter buffer during transform feedback operations.
    TRANSFORM_FEEDBACK_COUNTER_WRITE = TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_transform_feedback)]),
    ]),

    /// Read access to a predicate during conditional rendering.
    CONDITIONAL_RENDERING_READ = CONDITIONAL_RENDERING_READ_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_conditional_rendering)]),
    ]),

    /// Read access to preprocess buffers input to `preprocess_generated_commands`.
    COMMAND_PREPROCESS_READ = COMMAND_PREPROCESS_READ_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_device_generated_commands)]),
    ]),

    /// Read access to sequences buffers output by `preprocess_generated_commands`.
    COMMAND_PREPROCESS_WRITE = COMMAND_PREPROCESS_WRITE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_device_generated_commands)]),
    ]),

    /// Read access to a fragment shading rate attachment during rasterization.
    FRAGMENT_SHADING_RATE_ATTACHMENT_READ = FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_fragment_shading_rate)]),
    ]),

    /// Read access to an acceleration structure or acceleration structure scratch buffer during
    /// trace, build or copy commands.
    ACCELERATION_STRUCTURE_READ = ACCELERATION_STRUCTURE_READ_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_acceleration_structure)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    /// Write access to an acceleration structure or acceleration structure scratch buffer during
    /// trace, build or copy commands.
    ACCELERATION_STRUCTURE_WRITE = ACCELERATION_STRUCTURE_WRITE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_acceleration_structure)]),
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),

    /// Read access to a fragment density map attachment during dynamic fragment density map
    /// operations.
    FRAGMENT_DENSITY_MAP_READ = FRAGMENT_DENSITY_MAP_READ_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_fragment_density_map)]),
    ]),

    /// Read access to color attachments when performing advanced blend operations.
    COLOR_ATTACHMENT_READ_NONCOHERENT = COLOR_ATTACHMENT_READ_NONCOHERENT_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_blend_operation_advanced)]),
    ]),

    /// Read access to an invocation mask image.
    INVOCATION_MASK_READ = INVOCATION_MASK_READ_HUAWEI
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(huawei_invocation_mask),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(huawei_invocation_mask),
        ]),
    ]),

    /// Read access to a shader binding table.
    SHADER_BINDING_TABLE_READ = SHADER_BINDING_TABLE_READ_KHR
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(khr_ray_tracing_pipeline),
            DeviceExtension(khr_ray_tracing_maintenance1),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(khr_ray_tracing_pipeline),
            DeviceExtension(khr_ray_tracing_maintenance1),
        ]),
    ]),

    /// Read access to a micromap object.
    MICROMAP_READ = MICROMAP_READ_EXT
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(ext_opacity_micromap),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(ext_opacity_micromap),
        ]),
    ]),

    /// Write access to a micromap object.
    MICROMAP_WRITE = MICROMAP_WRITE_EXT
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(ext_opacity_micromap),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(ext_opacity_micromap),
        ]),
    ]),

    /// Read access to a buffer or image during optical flow operations.
    OPTICAL_FLOW_READ = OPTICAL_FLOW_READ_NV
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(nv_optical_flow),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(nv_optical_flow),
        ]),
    ]),

    /// Write access to a buffer or image during optical flow operations.
    OPTICAL_FLOW_WRITE = OPTICAL_FLOW_WRITE_NV
    RequiresOneOf([
        RequiresAllOf([
            APIVersion(V1_3),
            DeviceExtension(nv_optical_flow),
        ]),
        RequiresAllOf([
            DeviceExtension(khr_synchronization2),
            DeviceExtension(nv_optical_flow),
        ]),
    ]),
}

impl From<AccessFlags> for vk::AccessFlags {
    #[inline]
    fn from(val: AccessFlags) -> Self {
        Self::from_raw(vk::AccessFlags2::from(val).as_raw() as u32)
    }
}

impl From<PipelineStages> for AccessFlags {
    /// Corresponds to the table "[Supported access types]" in the Vulkan specification.
    ///
    /// [Supported access types]: https://registry.khronos.org/vulkan/specs/1.3-extensions/html/chap7.html#synchronization-access-types-supported
    #[inline]
    fn from(mut val: PipelineStages) -> Self {
        if val.is_empty() {
            return AccessFlags::empty();
        }

        val = val.expand(QueueFlags::GRAPHICS | QueueFlags::COMPUTE | QueueFlags::TRANSFER);
        let mut result = AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE;

        if val.intersects(PipelineStages::DRAW_INDIRECT) {
            result |=
                AccessFlags::INDIRECT_COMMAND_READ | AccessFlags::TRANSFORM_FEEDBACK_COUNTER_READ;
        }

        if val.intersects(
            PipelineStages::VERTEX_SHADER
                | PipelineStages::TESSELLATION_CONTROL_SHADER
                | PipelineStages::TESSELLATION_EVALUATION_SHADER
                | PipelineStages::GEOMETRY_SHADER
                | PipelineStages::FRAGMENT_SHADER
                | PipelineStages::COMPUTE_SHADER
                | PipelineStages::RAY_TRACING_SHADER
                | PipelineStages::TASK_SHADER
                | PipelineStages::MESH_SHADER,
        ) {
            result |= AccessFlags::SHADER_READ
                | AccessFlags::UNIFORM_READ
                | AccessFlags::SHADER_SAMPLED_READ
                | AccessFlags::SHADER_STORAGE_READ
                | AccessFlags::SHADER_WRITE
                | AccessFlags::SHADER_STORAGE_WRITE
                | AccessFlags::ACCELERATION_STRUCTURE_READ;
        }

        if val.intersects(PipelineStages::FRAGMENT_SHADER | PipelineStages::SUBPASS_SHADER) {
            result |= AccessFlags::INPUT_ATTACHMENT_READ;
        }

        if val
            .intersects(PipelineStages::EARLY_FRAGMENT_TESTS | PipelineStages::LATE_FRAGMENT_TESTS)
        {
            result |= AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
        }

        if val.intersects(PipelineStages::COLOR_ATTACHMENT_OUTPUT) {
            result |= AccessFlags::COLOR_ATTACHMENT_READ
                | AccessFlags::COLOR_ATTACHMENT_WRITE
                | AccessFlags::COLOR_ATTACHMENT_READ_NONCOHERENT;
        }

        if val.intersects(PipelineStages::HOST) {
            result |= AccessFlags::HOST_READ | AccessFlags::HOST_WRITE;
        }

        if val.intersects(
            PipelineStages::COPY
                | PipelineStages::RESOLVE
                | PipelineStages::BLIT
                | PipelineStages::ACCELERATION_STRUCTURE_COPY,
        ) {
            result |= AccessFlags::TRANSFER_READ | AccessFlags::TRANSFER_WRITE;
        }

        if val.intersects(PipelineStages::CLEAR) {
            result |= AccessFlags::TRANSFER_WRITE;
        }

        if val.intersects(PipelineStages::INDEX_INPUT) {
            result |= AccessFlags::INDEX_READ;
        }

        if val.intersects(PipelineStages::VERTEX_ATTRIBUTE_INPUT) {
            result |= AccessFlags::VERTEX_ATTRIBUTE_READ;
        }

        if val.intersects(PipelineStages::VIDEO_DECODE) {
            result |= AccessFlags::VIDEO_DECODE_READ | AccessFlags::VIDEO_DECODE_WRITE;
        }

        if val.intersects(PipelineStages::VIDEO_ENCODE) {
            result |= AccessFlags::VIDEO_ENCODE_READ | AccessFlags::VIDEO_ENCODE_WRITE;
        }

        if val.intersects(PipelineStages::TRANSFORM_FEEDBACK) {
            result |= AccessFlags::TRANSFORM_FEEDBACK_WRITE
                | AccessFlags::TRANSFORM_FEEDBACK_COUNTER_WRITE
                | AccessFlags::TRANSFORM_FEEDBACK_COUNTER_READ;
        }

        if val.intersects(PipelineStages::CONDITIONAL_RENDERING) {
            result |= AccessFlags::CONDITIONAL_RENDERING_READ;
        }

        if val.intersects(PipelineStages::ACCELERATION_STRUCTURE_BUILD) {
            result |= AccessFlags::INDIRECT_COMMAND_READ
                | AccessFlags::SHADER_READ
                | AccessFlags::SHADER_SAMPLED_READ
                | AccessFlags::SHADER_STORAGE_READ
                | AccessFlags::SHADER_STORAGE_WRITE
                | AccessFlags::TRANSFER_READ
                | AccessFlags::TRANSFER_WRITE
                | AccessFlags::ACCELERATION_STRUCTURE_READ
                | AccessFlags::ACCELERATION_STRUCTURE_WRITE
                | AccessFlags::MICROMAP_READ;
        }

        if val.intersects(PipelineStages::RAY_TRACING_SHADER) {
            result |= AccessFlags::SHADER_BINDING_TABLE_READ;
        }

        if val.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS) {
            result |= AccessFlags::FRAGMENT_DENSITY_MAP_READ;
        }

        if val.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT) {
            result |= AccessFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_READ;
        }

        if val.intersects(PipelineStages::COMMAND_PREPROCESS) {
            result |= AccessFlags::COMMAND_PREPROCESS_READ | AccessFlags::COMMAND_PREPROCESS_WRITE;
        }

        if val.intersects(PipelineStages::INVOCATION_MASK) {
            result |= AccessFlags::INVOCATION_MASK_READ;
        }

        if val.intersects(PipelineStages::MICROMAP_BUILD) {
            result |= AccessFlags::MICROMAP_READ | AccessFlags::MICROMAP_WRITE;
        }

        if val.intersects(PipelineStages::OPTICAL_FLOW) {
            result |= AccessFlags::OPTICAL_FLOW_READ | AccessFlags::OPTICAL_FLOW_WRITE;
        }

        result
    }
}

macro_rules! pipeline_stage_access {
    (
        $(
            $val:ident, $stage:ident, $access:ident
        );*
        $(;)?
    ) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[allow(non_camel_case_types, dead_code)]
        #[repr(u8)]
        pub(crate) enum PipelineStageAccess {
            $($val,)*

            // If there are ever more than 128 preceding values, then there will be a compile error:
            // "discriminant value `128` assigned more than once"
            __MAX_VALUE__ = 128,
        }

        impl From<PipelineStageAccess> for PipelineStages {
            #[inline]
            fn from(val: PipelineStageAccess) -> Self {
                match val {
                    $(
                        PipelineStageAccess::$val => PipelineStages::$stage,
                    )*
                    PipelineStageAccess::__MAX_VALUE__ => unreachable!(),
                }
            }
        }

        impl From<PipelineStageAccess> for AccessFlags {
            #[inline]
            fn from(val: PipelineStageAccess) -> Self {
                match val {
                    $(
                        PipelineStageAccess::$val => AccessFlags::$access,
                    )*
                    PipelineStageAccess::__MAX_VALUE__ => unreachable!(),
                }
            }
        }

        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        pub(crate) struct PipelineStageAccessFlags(u128);

        #[allow(non_upper_case_globals, dead_code)]
        impl PipelineStageAccessFlags {
            $(
                pub(crate) const $val: Self = Self(1u128 << PipelineStageAccess::$val as u8);
            )*

            #[inline]
            pub(crate) const fn empty() -> Self {
                Self(0)
            }

            #[inline]
            pub(crate) const fn count(self) -> u32 {
                self.0.count_ones()
            }

            #[inline]
            pub(crate) const fn is_empty(self) -> bool {
                self.0 == 0
            }

            #[inline]
            pub(crate) const fn intersects(self, other: Self) -> bool {
                self.0 & other.0 != 0
            }

            #[inline]
            pub(crate) const fn contains(self, other: Self) -> bool {
                self.0 & other.0 == other.0
            }

            #[inline]
            pub(crate) const fn union(self, other: Self) -> Self {
                Self(self.0 | other.0)
            }

            #[inline]
            pub(crate) const fn intersection(self, other: Self) -> Self {
                Self(self.0 & other.0)
            }

            #[inline]
            pub(crate) const fn difference(self, other: Self) -> Self {
                Self(self.0 & !other.0)
            }

            #[inline]
            pub(crate) const fn symmetric_difference(self, other: Self) -> Self {
                Self(self.0 ^ other.0)
            }

            #[inline]
            pub(crate) fn contains_enum(self, val: PipelineStageAccess) -> bool {
                self.intersects(val.into())
            }
        }

        impl std::ops::BitAnd for PipelineStageAccessFlags {
            type Output = Self;

            #[inline]
            fn bitand(self, rhs: Self) -> Self {
                self.intersection(rhs)
            }
        }

        impl std::ops::BitAndAssign for PipelineStageAccessFlags {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                *self = self.intersection(rhs);
            }
        }

        impl std::ops::BitOr for PipelineStageAccessFlags {
            type Output = Self;

            #[inline]
            fn bitor(self, rhs: Self) -> Self {
                self.union(rhs)
            }
        }

        impl std::ops::BitOrAssign for PipelineStageAccessFlags {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                *self = self.union(rhs);
            }
        }

        impl std::ops::BitXor for PipelineStageAccessFlags {
            type Output = Self;

            #[inline]
            fn bitxor(self, rhs: Self) -> Self {
                self.symmetric_difference(rhs)
            }
        }

        impl std::ops::BitXorAssign for PipelineStageAccessFlags {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) {
                *self = self.symmetric_difference(rhs);
            }
        }

        impl std::ops::Sub for PipelineStageAccessFlags {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: Self) -> Self {
                self.difference(rhs)
            }
        }

        impl std::ops::SubAssign for PipelineStageAccessFlags {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = self.difference(rhs);
            }
        }

        impl From<PipelineStageAccess> for PipelineStageAccessFlags {
            #[inline]
            fn from(val: PipelineStageAccess) -> Self {
                debug_assert!(val != PipelineStageAccess::__MAX_VALUE__); // You did something very dumb...
                Self(1u128 << val as u8)
            }
        }

        impl From<PipelineStageAccessFlags> for PipelineStages {
            #[inline]
            fn from(val: PipelineStageAccessFlags) -> Self {
                let mut result = PipelineStages::empty();

                $(
                    if val.intersects(PipelineStageAccessFlags::$val) {
                        result |= PipelineStages::$stage;
                    }
                )*

                result
            }
        }

        impl From<PipelineStageAccessFlags> for AccessFlags {
            #[inline]
            fn from(val: PipelineStageAccessFlags) -> Self {
                let mut result = AccessFlags::empty();

                $(
                    if val.intersects(PipelineStageAccessFlags::$val) {
                        result |= AccessFlags::$access;
                    }
                )*

                result
            }
        }
    };
}

pipeline_stage_access! {
    DrawIndirect_IndirectCommandRead, DRAW_INDIRECT, INDIRECT_COMMAND_READ;
    DrawIndirect_TransformFeedbackCounterRead, DRAW_INDIRECT, TRANSFORM_FEEDBACK_COUNTER_READ;
    VertexShader_UniformRead, VERTEX_SHADER, UNIFORM_READ;
    VertexShader_ShaderSampledRead, VERTEX_SHADER, SHADER_SAMPLED_READ;
    VertexShader_ShaderStorageRead, VERTEX_SHADER, SHADER_STORAGE_READ;
    VertexShader_ShaderStorageWrite, VERTEX_SHADER, SHADER_STORAGE_WRITE;
    VertexShader_AccelerationStructureRead, VERTEX_SHADER, ACCELERATION_STRUCTURE_READ;
    TessellationControlShader_UniformRead, TESSELLATION_CONTROL_SHADER, UNIFORM_READ;
    TessellationControlShader_ShaderSampledRead, TESSELLATION_CONTROL_SHADER, SHADER_SAMPLED_READ;
    TessellationControlShader_ShaderStorageRead, TESSELLATION_CONTROL_SHADER, SHADER_STORAGE_READ;
    TessellationControlShader_ShaderStorageWrite, TESSELLATION_CONTROL_SHADER, SHADER_STORAGE_WRITE;
    TessellationControlShader_AccelerationStructureRead, TESSELLATION_CONTROL_SHADER, ACCELERATION_STRUCTURE_READ;
    TessellationEvaluationShader_UniformRead, TESSELLATION_EVALUATION_SHADER, UNIFORM_READ;
    TessellationEvaluationShader_ShaderSampledRead, TESSELLATION_EVALUATION_SHADER, SHADER_SAMPLED_READ;
    TessellationEvaluationShader_ShaderStorageRead, TESSELLATION_EVALUATION_SHADER, SHADER_STORAGE_READ;
    TessellationEvaluationShader_ShaderStorageWrite, TESSELLATION_EVALUATION_SHADER, SHADER_STORAGE_WRITE;
    TessellationEvaluationShader_AccelerationStructureRead, TESSELLATION_EVALUATION_SHADER, ACCELERATION_STRUCTURE_READ;
    GeometryShader_UniformRead, GEOMETRY_SHADER, UNIFORM_READ;
    GeometryShader_ShaderSampledRead, GEOMETRY_SHADER, SHADER_SAMPLED_READ;
    GeometryShader_ShaderStorageRead, GEOMETRY_SHADER, SHADER_STORAGE_READ;
    GeometryShader_ShaderStorageWrite, GEOMETRY_SHADER, SHADER_STORAGE_WRITE;
    GeometryShader_AccelerationStructureRead, GEOMETRY_SHADER, ACCELERATION_STRUCTURE_READ;
    FragmentShader_UniformRead, FRAGMENT_SHADER, UNIFORM_READ;
    FragmentShader_InputAttachmentRead, FRAGMENT_SHADER, INPUT_ATTACHMENT_READ;
    FragmentShader_ShaderSampledRead, FRAGMENT_SHADER, SHADER_SAMPLED_READ;
    FragmentShader_ShaderStorageRead, FRAGMENT_SHADER, SHADER_STORAGE_READ;
    FragmentShader_ShaderStorageWrite, FRAGMENT_SHADER, SHADER_STORAGE_WRITE;
    FragmentShader_AccelerationStructureRead, FRAGMENT_SHADER, ACCELERATION_STRUCTURE_READ;
    EarlyFragmentTests_DepthStencilAttachmentRead, EARLY_FRAGMENT_TESTS, DEPTH_STENCIL_ATTACHMENT_READ;
    EarlyFragmentTests_DepthStencilAttachmentWrite, EARLY_FRAGMENT_TESTS, DEPTH_STENCIL_ATTACHMENT_WRITE;
    LateFragmentTests_DepthStencilAttachmentRead, LATE_FRAGMENT_TESTS, DEPTH_STENCIL_ATTACHMENT_READ;
    LateFragmentTests_DepthStencilAttachmentWrite, LATE_FRAGMENT_TESTS, DEPTH_STENCIL_ATTACHMENT_WRITE;
    ColorAttachmentOutput_ColorAttachmentRead, COLOR_ATTACHMENT_OUTPUT, COLOR_ATTACHMENT_READ;
    ColorAttachmentOutput_ColorAttachmentWrite, COLOR_ATTACHMENT_OUTPUT, COLOR_ATTACHMENT_WRITE;
    ColorAttachmentOutput_ColorAttachmentReadNoncoherent, COLOR_ATTACHMENT_OUTPUT, COLOR_ATTACHMENT_READ_NONCOHERENT;
    ComputeShader_UniformRead, COMPUTE_SHADER, UNIFORM_READ;
    ComputeShader_ShaderSampledRead, COMPUTE_SHADER, SHADER_SAMPLED_READ;
    ComputeShader_ShaderStorageRead, COMPUTE_SHADER, SHADER_STORAGE_READ;
    ComputeShader_ShaderStorageWrite, COMPUTE_SHADER, SHADER_STORAGE_WRITE;
    ComputeShader_AccelerationStructureRead, COMPUTE_SHADER, ACCELERATION_STRUCTURE_READ;
    Host_HostRead, HOST, HOST_READ;
    Host_HostWrite, HOST, HOST_WRITE;
    Copy_TransferRead, COPY, TRANSFER_READ;
    Copy_TransferWrite, COPY, TRANSFER_WRITE;
    Resolve_TransferRead, RESOLVE, TRANSFER_READ;
    Resolve_TransferWrite, RESOLVE, TRANSFER_WRITE;
    Blit_TransferRead, BLIT, TRANSFER_READ;
    Blit_TransferWrite, BLIT, TRANSFER_WRITE;
    Clear_TransferWrite, CLEAR, TRANSFER_WRITE;
    IndexInput_IndexRead, INDEX_INPUT, INDEX_READ;
    VertexAttributeInput_VertexAttributeRead, VERTEX_ATTRIBUTE_INPUT, VERTEX_ATTRIBUTE_READ;
    VideoDecode_VideoDecodeRead, VIDEO_DECODE, VIDEO_DECODE_READ;
    VideoDecode_VideoDecodeWrite, VIDEO_DECODE, VIDEO_DECODE_WRITE;
    VideoEncode_VideoEncodeRead, VIDEO_ENCODE, VIDEO_ENCODE_READ;
    VideoEncode_VideoEncodeWrite, VIDEO_ENCODE, VIDEO_ENCODE_WRITE;
    TransformFeedback_TransformFeedbackWrite, TRANSFORM_FEEDBACK, TRANSFORM_FEEDBACK_WRITE;
    TransformFeedback_TransformFeedbackCounterRead, TRANSFORM_FEEDBACK, TRANSFORM_FEEDBACK_COUNTER_READ;
    TransformFeedback_TransformFeedbackCounterWrite, TRANSFORM_FEEDBACK, TRANSFORM_FEEDBACK_COUNTER_WRITE;
    ConditionalRendering_ConditionalRenderingRead, CONDITIONAL_RENDERING, CONDITIONAL_RENDERING_READ;
    AccelerationStructureBuild_IndirectCommandRead, ACCELERATION_STRUCTURE_BUILD, INDIRECT_COMMAND_READ;
    AccelerationStructureBuild_UniformRead, ACCELERATION_STRUCTURE_BUILD, UNIFORM_READ;
    AccelerationStructureBuild_TransferRead, ACCELERATION_STRUCTURE_BUILD, TRANSFER_READ;
    AccelerationStructureBuild_TransferWrite, ACCELERATION_STRUCTURE_BUILD, TRANSFER_WRITE;
    AccelerationStructureBuild_ShaderSampledRead, ACCELERATION_STRUCTURE_BUILD, SHADER_SAMPLED_READ;
    AccelerationStructureBuild_ShaderStorageRead, ACCELERATION_STRUCTURE_BUILD, SHADER_STORAGE_READ;
    AccelerationStructureBuild_AccelerationStructureRead, ACCELERATION_STRUCTURE_BUILD, ACCELERATION_STRUCTURE_READ;
    AccelerationStructureBuild_AccelerationStructureWrite, ACCELERATION_STRUCTURE_BUILD, ACCELERATION_STRUCTURE_WRITE;
    AccelerationStructureBuild_MicromapRead, ACCELERATION_STRUCTURE_BUILD, MICROMAP_READ;
    RayTracingShader_UniformRead, RAY_TRACING_SHADER, UNIFORM_READ;
    RayTracingShader_ShaderSampledRead, RAY_TRACING_SHADER, SHADER_SAMPLED_READ;
    RayTracingShader_ShaderStorageRead, RAY_TRACING_SHADER, SHADER_STORAGE_READ;
    RayTracingShader_ShaderStorageWrite, RAY_TRACING_SHADER, SHADER_STORAGE_WRITE;
    RayTracingShader_AccelerationStructureRead, RAY_TRACING_SHADER, ACCELERATION_STRUCTURE_READ;
    RayTracingShader_ShaderBindingTableRead, RAY_TRACING_SHADER, SHADER_BINDING_TABLE_READ;
    FragmentDensityProcess_FragmentDensityMapRead, FRAGMENT_DENSITY_PROCESS, FRAGMENT_DENSITY_MAP_READ;
    FragmentShadingRateAttachment_FragmentShadingRateAttachmentRead, FRAGMENT_SHADING_RATE_ATTACHMENT, FRAGMENT_SHADING_RATE_ATTACHMENT_READ;
    CommandPreprocess_CommandPreprocessRead, COMMAND_PREPROCESS, COMMAND_PREPROCESS_READ;
    CommandPreprocess_CommandPreprocessWrite, COMMAND_PREPROCESS, COMMAND_PREPROCESS_WRITE;
    TaskShader_UniformRead, TASK_SHADER, UNIFORM_READ;
    TaskShader_ShaderSampledRead, TASK_SHADER, SHADER_SAMPLED_READ;
    TaskShader_ShaderStorageRead, TASK_SHADER, SHADER_STORAGE_READ;
    TaskShader_ShaderStorageWrite, TASK_SHADER, SHADER_STORAGE_WRITE;
    TaskShader_AccelerationStructureRead, TASK_SHADER, ACCELERATION_STRUCTURE_READ;
    MeshShader_UniformRead, MESH_SHADER, UNIFORM_READ;
    MeshShader_ShaderSampledRead, MESH_SHADER, SHADER_SAMPLED_READ;
    MeshShader_ShaderStorageRead, MESH_SHADER, SHADER_STORAGE_READ;
    MeshShader_ShaderStorageWrite, MESH_SHADER, SHADER_STORAGE_WRITE;
    MeshShader_AccelerationStructureRead, MESH_SHADER, ACCELERATION_STRUCTURE_READ;
    SubpassShading_InputAttachmentRead, SUBPASS_SHADER, INPUT_ATTACHMENT_READ;
    InvocationMask_InvocationMaskRead, INVOCATION_MASK, INVOCATION_MASK_READ;
    AccelerationStructureCopy_AccelerationStructureRead, ACCELERATION_STRUCTURE_COPY, ACCELERATION_STRUCTURE_READ;
    AccelerationStructureCopy_AccelerationStructureWrite, ACCELERATION_STRUCTURE_COPY, ACCELERATION_STRUCTURE_WRITE;
    AccelerationStructureCopy_TransferRead, ACCELERATION_STRUCTURE_COPY, TRANSFER_READ;
    AccelerationStructureCopy_TransferWrite, ACCELERATION_STRUCTURE_COPY, TRANSFER_WRITE;
    OpticalFlow_OpticalFlowRead, OPTICAL_FLOW, OPTICAL_FLOW_READ;
    OpticalFlow_OpticalFlowWrite, OPTICAL_FLOW, OPTICAL_FLOW_WRITE;
    MicromapBuild_MicromapRead, MICROMAP_BUILD, MICROMAP_READ;
    MicromapBuild_MicromapWrite, MICROMAP_BUILD, MICROMAP_WRITE;
}

impl PipelineStageAccess {
    pub(crate) fn iter_descriptor_stages(
        descriptor_type: DescriptorType,
        stages_read: ShaderStages,
        stages_write: ShaderStages,
    ) -> impl Iterator<Item = Self> + 'static {
        static MAP_READ: LazyLock<
            HashMap<DescriptorType, HashMap<PipelineStage, PipelineStageAccess>>,
        > = LazyLock::new(|| {
            let uniform_read = [
                DescriptorType::UniformBuffer,
                DescriptorType::UniformBufferDynamic,
            ]
            .into_iter()
            .map(|descriptor_type| {
                (
                    descriptor_type,
                    [
                        (
                            PipelineStage::VertexShader,
                            PipelineStageAccess::VertexShader_UniformRead,
                        ),
                        (
                            PipelineStage::TessellationControlShader,
                            PipelineStageAccess::TessellationControlShader_UniformRead,
                        ),
                        (
                            PipelineStage::TessellationEvaluationShader,
                            PipelineStageAccess::TessellationControlShader_UniformRead,
                        ),
                        (
                            PipelineStage::GeometryShader,
                            PipelineStageAccess::GeometryShader_UniformRead,
                        ),
                        (
                            PipelineStage::FragmentShader,
                            PipelineStageAccess::FragmentShader_UniformRead,
                        ),
                        (
                            PipelineStage::ComputeShader,
                            PipelineStageAccess::ComputeShader_UniformRead,
                        ),
                        (
                            PipelineStage::RayTracingShader,
                            PipelineStageAccess::RayTracingShader_UniformRead,
                        ),
                        (
                            PipelineStage::TaskShader,
                            PipelineStageAccess::TaskShader_UniformRead,
                        ),
                        (
                            PipelineStage::MeshShader,
                            PipelineStageAccess::MeshShader_UniformRead,
                        ),
                    ]
                    .into_iter()
                    .collect(),
                )
            });

            let shader_sampled_read = [
                DescriptorType::CombinedImageSampler,
                DescriptorType::SampledImage,
                DescriptorType::UniformTexelBuffer,
            ]
            .into_iter()
            .map(|descriptor_type| {
                (
                    descriptor_type,
                    [
                        (
                            PipelineStage::VertexShader,
                            PipelineStageAccess::VertexShader_ShaderSampledRead,
                        ),
                        (
                            PipelineStage::TessellationControlShader,
                            PipelineStageAccess::TessellationControlShader_ShaderSampledRead,
                        ),
                        (
                            PipelineStage::TessellationEvaluationShader,
                            PipelineStageAccess::TessellationControlShader_ShaderSampledRead,
                        ),
                        (
                            PipelineStage::GeometryShader,
                            PipelineStageAccess::GeometryShader_ShaderSampledRead,
                        ),
                        (
                            PipelineStage::FragmentShader,
                            PipelineStageAccess::FragmentShader_ShaderSampledRead,
                        ),
                        (
                            PipelineStage::ComputeShader,
                            PipelineStageAccess::ComputeShader_ShaderSampledRead,
                        ),
                        (
                            PipelineStage::RayTracingShader,
                            PipelineStageAccess::RayTracingShader_ShaderSampledRead,
                        ),
                        (
                            PipelineStage::TaskShader,
                            PipelineStageAccess::TaskShader_ShaderSampledRead,
                        ),
                        (
                            PipelineStage::MeshShader,
                            PipelineStageAccess::MeshShader_ShaderSampledRead,
                        ),
                    ]
                    .into_iter()
                    .collect(),
                )
            });

            let shader_storage_read = [
                DescriptorType::StorageImage,
                DescriptorType::StorageTexelBuffer,
                DescriptorType::StorageBuffer,
                DescriptorType::StorageBufferDynamic,
            ]
            .into_iter()
            .map(|descriptor_type| {
                (
                    descriptor_type,
                    [
                        (
                            PipelineStage::VertexShader,
                            PipelineStageAccess::VertexShader_ShaderStorageRead,
                        ),
                        (
                            PipelineStage::TessellationControlShader,
                            PipelineStageAccess::TessellationControlShader_ShaderStorageRead,
                        ),
                        (
                            PipelineStage::TessellationEvaluationShader,
                            PipelineStageAccess::TessellationControlShader_ShaderStorageRead,
                        ),
                        (
                            PipelineStage::GeometryShader,
                            PipelineStageAccess::GeometryShader_ShaderStorageRead,
                        ),
                        (
                            PipelineStage::FragmentShader,
                            PipelineStageAccess::FragmentShader_ShaderStorageRead,
                        ),
                        (
                            PipelineStage::ComputeShader,
                            PipelineStageAccess::ComputeShader_ShaderStorageRead,
                        ),
                        (
                            PipelineStage::RayTracingShader,
                            PipelineStageAccess::RayTracingShader_ShaderStorageRead,
                        ),
                        (
                            PipelineStage::TaskShader,
                            PipelineStageAccess::TaskShader_ShaderStorageRead,
                        ),
                        (
                            PipelineStage::MeshShader,
                            PipelineStageAccess::MeshShader_ShaderStorageRead,
                        ),
                    ]
                    .into_iter()
                    .collect(),
                )
            });

            let input_attachment_read =
                [DescriptorType::InputAttachment]
                    .into_iter()
                    .map(|descriptor_type| {
                        (
                            descriptor_type,
                            [(
                                PipelineStage::FragmentShader,
                                PipelineStageAccess::FragmentShader_InputAttachmentRead,
                            )]
                            .into_iter()
                            .collect(),
                        )
                    });

            uniform_read
                .chain(shader_sampled_read)
                .chain(shader_storage_read)
                .chain(input_attachment_read)
                .collect()
        });
        static MAP_WRITE: LazyLock<
            HashMap<DescriptorType, HashMap<PipelineStage, PipelineStageAccess>>,
        > = LazyLock::new(|| {
            let shader_storage_write = [
                DescriptorType::StorageImage,
                DescriptorType::StorageTexelBuffer,
                DescriptorType::StorageBuffer,
                DescriptorType::StorageBufferDynamic,
            ]
            .into_iter()
            .map(|descriptor_type| {
                (
                    descriptor_type,
                    [
                        (
                            PipelineStage::VertexShader,
                            PipelineStageAccess::VertexShader_ShaderStorageWrite,
                        ),
                        (
                            PipelineStage::TessellationControlShader,
                            PipelineStageAccess::TessellationControlShader_ShaderStorageWrite,
                        ),
                        (
                            PipelineStage::TessellationEvaluationShader,
                            PipelineStageAccess::TessellationControlShader_ShaderStorageWrite,
                        ),
                        (
                            PipelineStage::GeometryShader,
                            PipelineStageAccess::GeometryShader_ShaderStorageWrite,
                        ),
                        (
                            PipelineStage::FragmentShader,
                            PipelineStageAccess::FragmentShader_ShaderStorageWrite,
                        ),
                        (
                            PipelineStage::ComputeShader,
                            PipelineStageAccess::ComputeShader_ShaderStorageWrite,
                        ),
                        (
                            PipelineStage::RayTracingShader,
                            PipelineStageAccess::RayTracingShader_ShaderStorageWrite,
                        ),
                        (
                            PipelineStage::TaskShader,
                            PipelineStageAccess::TaskShader_ShaderStorageWrite,
                        ),
                        (
                            PipelineStage::MeshShader,
                            PipelineStageAccess::MeshShader_ShaderStorageWrite,
                        ),
                    ]
                    .into_iter()
                    .collect(),
                )
            });

            shader_storage_write.collect()
        });

        [
            (stages_read, &*MAP_READ, "read"),
            (stages_write, &*MAP_WRITE, "write"),
        ]
        .into_iter()
        .filter(|(stages, _, _)| !stages.is_empty())
        .flat_map(move |(stages, descriptor_map, access)| {
            let stages_map = descriptor_map.get(&descriptor_type).unwrap_or_else(|| {
                panic!(
                    "DescriptorType::{:?} does not {} memory",
                    descriptor_type, access,
                )
            });

            PipelineStages::from(stages).into_iter().map(move |stage| {
                *stages_map.get(&stage).unwrap_or_else(|| {
                    panic!(
                        "DescriptorType::{:?} does not {} memory in PipelineStage::{:?}",
                        descriptor_type, access, stage,
                    )
                })
            })
        })
    }
}

impl PipelineStageAccessFlags {
    #[inline]
    pub(crate) fn contains_write(self) -> bool {
        self.intersects(
            PipelineStageAccessFlags::VertexShader_ShaderStorageWrite
                | PipelineStageAccessFlags::TessellationControlShader_ShaderStorageWrite
                | PipelineStageAccessFlags::TessellationEvaluationShader_ShaderStorageWrite
                | PipelineStageAccessFlags::GeometryShader_ShaderStorageWrite
                | PipelineStageAccessFlags::FragmentShader_ShaderStorageWrite
                | PipelineStageAccessFlags::EarlyFragmentTests_DepthStencilAttachmentWrite
                | PipelineStageAccessFlags::LateFragmentTests_DepthStencilAttachmentWrite
                | PipelineStageAccessFlags::ColorAttachmentOutput_ColorAttachmentWrite
                | PipelineStageAccessFlags::ComputeShader_ShaderStorageWrite
                | PipelineStageAccessFlags::Host_HostWrite
                | PipelineStageAccessFlags::Copy_TransferWrite
                | PipelineStageAccessFlags::Resolve_TransferWrite
                | PipelineStageAccessFlags::Blit_TransferWrite
                | PipelineStageAccessFlags::Clear_TransferWrite
                | PipelineStageAccessFlags::VideoDecode_VideoDecodeWrite
                | PipelineStageAccessFlags::VideoEncode_VideoEncodeWrite
                | PipelineStageAccessFlags::TransformFeedback_TransformFeedbackWrite
                | PipelineStageAccessFlags::TransformFeedback_TransformFeedbackCounterWrite
                | PipelineStageAccessFlags::AccelerationStructureBuild_TransferWrite
                | PipelineStageAccessFlags::AccelerationStructureBuild_AccelerationStructureWrite
                | PipelineStageAccessFlags::RayTracingShader_ShaderStorageWrite
                | PipelineStageAccessFlags::CommandPreprocess_CommandPreprocessWrite
                | PipelineStageAccessFlags::TaskShader_ShaderStorageWrite
                | PipelineStageAccessFlags::MeshShader_ShaderStorageWrite
                | PipelineStageAccessFlags::AccelerationStructureCopy_AccelerationStructureWrite
                | PipelineStageAccessFlags::AccelerationStructureCopy_TransferWrite
                | PipelineStageAccessFlags::OpticalFlow_OpticalFlowWrite
                | PipelineStageAccessFlags::MicromapBuild_MicromapWrite,
        )
    }
}

/// Dependency info for barriers in a pipeline barrier or event command.
///
/// A pipeline barrier creates a dependency between commands submitted before the barrier (the
/// source scope) and commands submitted after it (the destination scope). An event command acts
/// like a split pipeline barrier: the source scope and destination scope are defined
/// relative to different commands. Each `DependencyInfo` consists of multiple individual barriers
/// that concern a either single resource or operate globally.
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
    /// Flags to modify how the execution and memory dependencies are formed.
    ///
    /// The default value is empty.
    pub dependency_flags: DependencyFlags,

    /// Memory barriers for global operations and accesses, not limited to a single resource.
    ///
    /// The default value is empty.
    pub memory_barriers: SmallVec<[MemoryBarrier; 2]>,

    /// Memory barriers for individual buffers.
    ///
    /// The default value is empty.
    pub buffer_memory_barriers: SmallVec<[BufferMemoryBarrier; 8]>,

    /// Memory barriers for individual images.
    ///
    /// The default value is empty.
    pub image_memory_barriers: SmallVec<[ImageMemoryBarrier; 8]>,

    pub _ne: crate::NonExhaustive<'static>,
}

impl Default for DependencyInfo {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl DependencyInfo {
    /// Returns a default `DependencyInfo`.
    // TODO: make const
    #[inline]
    pub fn new() -> Self {
        Self {
            dependency_flags: DependencyFlags::empty(),
            memory_barriers: SmallVec::new(),
            buffer_memory_barriers: SmallVec::new(),
            image_memory_barriers: SmallVec::new(),
            _ne: crate::NE,
        }
    }

    /// Returns whether `self` contains any barriers.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.memory_barriers.is_empty()
            && self.buffer_memory_barriers.is_empty()
            && self.image_memory_barriers.is_empty()
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            dependency_flags,
            ref memory_barriers,
            ref buffer_memory_barriers,
            ref image_memory_barriers,
            _ne: _,
        } = self;

        dependency_flags.validate_device(device).map_err(|err| {
            err.add_context("dependency_flags")
                .set_vuids(&["VUID-VkDependencyInfo-dependencyFlags-parameter"])
        })?;

        for (barrier_index, memory_barrier) in memory_barriers.iter().enumerate() {
            memory_barrier
                .validate(device)
                .map_err(|err| err.add_context(format!("memory_barriers[{}]", barrier_index)))?;
        }

        for (barrier_index, buffer_memory_barrier) in buffer_memory_barriers.iter().enumerate() {
            buffer_memory_barrier.validate(device).map_err(|err| {
                err.add_context(format!("buffer_memory_barriers[{}]", barrier_index))
            })?;
        }

        for (barrier_index, image_memory_barrier) in image_memory_barriers.iter().enumerate() {
            image_memory_barrier.validate(device).map_err(|err| {
                err.add_context(format!("image_memory_barriers[{}]", barrier_index))
            })?;
        }

        Ok(())
    }

    pub(crate) fn to_vk2<'a>(
        &self,
        memory_barriers_vk: &'a [vk::MemoryBarrier2<'_>],
        buffer_memory_barriers_vk: &'a [vk::BufferMemoryBarrier2<'_>],
        image_memory_barriers_vk: &'a [vk::ImageMemoryBarrier2<'_>],
    ) -> vk::DependencyInfo<'a> {
        let &Self {
            dependency_flags,
            memory_barriers: _,
            buffer_memory_barriers: _,
            image_memory_barriers: _,
            _ne: _,
        } = self;

        vk::DependencyInfo::default()
            .dependency_flags(dependency_flags.into())
            .memory_barriers(memory_barriers_vk)
            .buffer_memory_barriers(buffer_memory_barriers_vk)
            .image_memory_barriers(image_memory_barriers_vk)
    }

    pub(crate) fn to_vk2_fields1(&self) -> DependencyInfo2Fields1Vk {
        let &Self {
            dependency_flags: _,
            ref memory_barriers,
            ref buffer_memory_barriers,
            ref image_memory_barriers,
            _ne: _,
        } = self;

        let memory_barriers_vk = memory_barriers.iter().map(MemoryBarrier::to_vk2).collect();
        let buffer_memory_barriers_vk = buffer_memory_barriers
            .iter()
            .map(BufferMemoryBarrier::to_vk2)
            .collect();
        let image_memory_barriers_vk = image_memory_barriers
            .iter()
            .map(ImageMemoryBarrier::to_vk2)
            .collect();

        DependencyInfo2Fields1Vk {
            memory_barriers_vk,
            buffer_memory_barriers_vk,
            image_memory_barriers_vk,
        }
    }

    pub(crate) fn to_vk_dependency_flags(&self) -> vk::DependencyFlags {
        self.dependency_flags.into()
    }

    pub(crate) fn to_vk_fields1(&self) -> DependencyInfoFields1Vk {
        let &Self {
            dependency_flags: _,
            ref memory_barriers,
            ref buffer_memory_barriers,
            ref image_memory_barriers,
            _ne: _,
        } = self;

        let mut src_stage_mask_vk = vk::PipelineStageFlags::empty();
        let mut dst_stage_mask_vk = vk::PipelineStageFlags::empty();

        let memory_barriers_vk = memory_barriers
            .iter()
            .inspect(|barrier| {
                src_stage_mask_vk |= barrier.src_stages.into();
                dst_stage_mask_vk |= barrier.dst_stages.into();
            })
            .map(MemoryBarrier::to_vk)
            .collect();
        let buffer_memory_barriers_vk = buffer_memory_barriers
            .iter()
            .inspect(|barrier| {
                src_stage_mask_vk |= barrier.src_stages.into();
                dst_stage_mask_vk |= barrier.dst_stages.into();
            })
            .map(BufferMemoryBarrier::to_vk)
            .collect();
        let image_memory_barriers_vk = image_memory_barriers
            .iter()
            .inspect(|barrier| {
                src_stage_mask_vk |= barrier.src_stages.into();
                dst_stage_mask_vk |= barrier.dst_stages.into();
            })
            .map(ImageMemoryBarrier::to_vk)
            .collect();

        if src_stage_mask_vk.is_empty() {
            // "VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT is [...] equivalent to
            // VK_PIPELINE_STAGE_2_NONE in the first scope."
            src_stage_mask_vk |= vk::PipelineStageFlags::TOP_OF_PIPE;
        }

        if dst_stage_mask_vk.is_empty() {
            // "VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT is [...] equivalent to
            // VK_PIPELINE_STAGE_2_NONE in the second scope."
            dst_stage_mask_vk |= vk::PipelineStageFlags::BOTTOM_OF_PIPE;
        }

        DependencyInfoFields1Vk {
            memory_barriers_vk,
            buffer_memory_barriers_vk,
            image_memory_barriers_vk,
            src_stage_mask_vk,
            dst_stage_mask_vk,
        }
    }

    pub(crate) fn to_vk_src_stage_mask(&self) -> vk::PipelineStageFlags {
        let &Self {
            dependency_flags: _,
            ref memory_barriers,
            ref buffer_memory_barriers,
            ref image_memory_barriers,
            _ne: _,
        } = self;

        let mut src_stage_mask_vk = vk::PipelineStageFlags::empty();

        for barrier in memory_barriers {
            src_stage_mask_vk |= barrier.src_stages.into();
        }

        for barrier in buffer_memory_barriers {
            src_stage_mask_vk |= barrier.src_stages.into();
        }

        for barrier in image_memory_barriers {
            src_stage_mask_vk |= barrier.src_stages.into();
        }

        if src_stage_mask_vk.is_empty() {
            // "VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT is [...] equivalent to
            // VK_PIPELINE_STAGE_2_NONE in the first scope."
            src_stage_mask_vk |= vk::PipelineStageFlags::TOP_OF_PIPE;
        }

        src_stage_mask_vk
    }
}

pub(crate) struct DependencyInfo2Fields1Vk {
    pub(crate) memory_barriers_vk: SmallVec<[vk::MemoryBarrier2<'static>; 2]>,
    pub(crate) buffer_memory_barriers_vk: SmallVec<[vk::BufferMemoryBarrier2<'static>; 8]>,
    pub(crate) image_memory_barriers_vk: SmallVec<[vk::ImageMemoryBarrier2<'static>; 8]>,
}

pub(crate) struct DependencyInfoFields1Vk {
    pub(crate) memory_barriers_vk: SmallVec<[vk::MemoryBarrier<'static>; 2]>,
    pub(crate) buffer_memory_barriers_vk: SmallVec<[vk::BufferMemoryBarrier<'static>; 8]>,
    pub(crate) image_memory_barriers_vk: SmallVec<[vk::ImageMemoryBarrier<'static>; 8]>,
    pub(crate) src_stage_mask_vk: vk::PipelineStageFlags,
    pub(crate) dst_stage_mask_vk: vk::PipelineStageFlags,
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags that modify how execution and memory dependencies are formed.
    DependencyFlags = DependencyFlags(u32);

    /// For framebuffer-space pipeline stages, specifies that the dependency is framebuffer-local.
    /// The implementation can start the destination operation for some given pixels as long as the
    /// source operation is finished for these given pixels.
    ///
    /// Framebuffer-local dependencies are usually more efficient, especially on tile-based
    /// architectures.
    BY_REGION = BY_REGION,

    /// For devices that consist of multiple physical devices, specifies that the dependency is
    /// device-local. The dependency will only apply to the operations on each physical device
    /// individually, rather than applying to all physical devices as a whole. This allows each
    /// physical device to operate independently of the others.
    ///
    /// The device API version must be at least 1.1, or the [`khr_device_group`] extension must be
    /// enabled on the device.
    ///
    /// [`khr_device_group`]: crate::device::DeviceExtensions::khr_device_group
    DEVICE_GROUP = DEVICE_GROUP
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_device_group)]),
    ]),


    /// For subpass dependencies, and pipeline barriers executing within a render pass instance,
    /// if the render pass uses multiview rendering, specifies that the dependency is view-local.
    /// Each view in the destination subpass will only depend on a single view in the destination
    /// subpass, instead of all views.
    ///
    /// The device API version must be at least 1.1, or the [`khr_multiview`] extension must be
    /// enabled on the device.
    ///
    /// [`khr_multiview`]: crate::device::DeviceExtensions::khr_multiview
    VIEW_LOCAL = VIEW_LOCAL
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
        RequiresAllOf([DeviceExtension(khr_multiview)]),
    ]),
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

    pub _ne: crate::NonExhaustive<'static>,
}

impl Default for MemoryBarrier {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryBarrier {
    /// Returns a default `MemoryBarrier`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            src_stages: PipelineStages::empty(),
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStages::empty(),
            dst_access: AccessFlags::empty(),
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            _ne: _,
        } = self;

        src_stages.validate_device(device).map_err(|err| {
            err.add_context("src_stages")
                .set_vuids(&["VUID-VkMemoryBarrier2-srcStageMask-parameter"])
        })?;

        dst_stages.validate_device(device).map_err(|err| {
            err.add_context("dst_stages")
                .set_vuids(&["VUID-VkMemoryBarrier2-dstStageMask-parameter"])
        })?;

        src_access.validate_device(device).map_err(|err| {
            err.add_context("src_access")
                .set_vuids(&["VUID-VkMemoryBarrier2-srcAccessMask-parameter"])
        })?;

        dst_access.validate_device(device).map_err(|err| {
            err.add_context("dst_access")
                .set_vuids(&["VUID-VkMemoryBarrier2-dstAccessMask-parameter"])
        })?;

        if !device.enabled_features().synchronization2 {
            if src_stages.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains flags from `VkPipelineStageFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }

            if dst_stages.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains flags from `VkPipelineStageFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }

            if src_access.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "src_access".into(),
                    problem: "contains flags from `VkAccessFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }

            if dst_access.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "dst_access".into(),
                    problem: "contains flags from `VkAccessFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }
        }

        if !device.enabled_features().geometry_shader {
            if src_stages.intersects(PipelineStages::GEOMETRY_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::GEOMETRY_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "geometry_shader",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-srcStageMask-03929"],
                }));
            }

            if dst_stages.intersects(PipelineStages::GEOMETRY_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::GEOMETRY_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "geometry_shader",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-dstStageMask-03929"],
                }));
            }
        }

        if !device.enabled_features().tessellation_shader {
            if src_stages.intersects(
                PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER,
            ) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::TESSELLATION_CONTROL_SHADER` or \
                        `PipelineStages::TESSELLATION_EVALUATION_SHADER`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "tessellation_shader",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-srcStageMask-03930"],
                }));
            }

            if dst_stages.intersects(
                PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER,
            ) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::TESSELLATION_CONTROL_SHADER` or \
                        `PipelineStages::TESSELLATION_EVALUATION_SHADER`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "tessellation_shader",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-dstStageMask-03930"],
                }));
            }
        }

        if !device.enabled_features().conditional_rendering {
            if src_stages.intersects(PipelineStages::CONDITIONAL_RENDERING) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::CONDITIONAL_RENDERING`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "conditional_rendering",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-srcStageMask-03931"],
                }));
            }

            if dst_stages.intersects(PipelineStages::CONDITIONAL_RENDERING) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::CONDITIONAL_RENDERING`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "conditional_rendering",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-dstStageMask-03931"],
                }));
            }
        }

        if !device.enabled_features().fragment_density_map {
            if src_stages.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_DENSITY_PROCESS`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "fragment_density_map",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-srcStageMask-03932"],
                }));
            }

            if dst_stages.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_DENSITY_PROCESS`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "fragment_density_map",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-dstStageMask-03932"],
                }));
            }
        }

        if !device.enabled_features().transform_feedback {
            if src_stages.intersects(PipelineStages::TRANSFORM_FEEDBACK) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::TRANSFORM_FEEDBACK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "transform_feedback",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-srcStageMask-03933"],
                }));
            }

            if dst_stages.intersects(PipelineStages::TRANSFORM_FEEDBACK) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::TRANSFORM_FEEDBACK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "transform_feedback",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-dstStageMask-03933"],
                }));
            }
        }

        if !device.enabled_features().mesh_shader {
            if src_stages.intersects(PipelineStages::MESH_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::MESH_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "mesh_shader",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-srcStageMask-03934"],
                }));
            }

            if dst_stages.intersects(PipelineStages::MESH_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::MESH_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "mesh_shader",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-dstStageMask-03934"],
                }));
            }
        }

        if !device.enabled_features().task_shader {
            if src_stages.intersects(PipelineStages::TASK_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::TASK_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "task_shader",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-srcStageMask-03935"],
                }));
            }

            if dst_stages.intersects(PipelineStages::TASK_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::TASK_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "task_shader",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-dstStageMask-03935"],
                }));
            }
        }

        if !(device.enabled_features().attachment_fragment_shading_rate
            || device.enabled_features().shading_rate_image)
        {
            if src_stages.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::DeviceFeature(
                            "attachment_fragment_shading_rate",
                        )]),
                        RequiresAllOf(&[Requires::DeviceFeature("shading_rate_image")]),
                    ]),
                    vuids: &["VUID-VkMemoryBarrier2-shadingRateImage-07316"],
                }));
            }

            if dst_stages.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::DeviceFeature(
                            "attachment_fragment_shading_rate",
                        )]),
                        RequiresAllOf(&[Requires::DeviceFeature("shading_rate_image")]),
                    ]),
                    vuids: &["VUID-VkMemoryBarrier2-shadingRateImage-07316"],
                }));
            }
        }

        if !device.enabled_features().subpass_shading {
            if src_stages.intersects(PipelineStages::SUBPASS_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::SUBPASS_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "subpass_shading",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-srcStageMask-04957"],
                }));
            }

            if dst_stages.intersects(PipelineStages::SUBPASS_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::SUBPASS_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "subpass_shading",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-dstStageMask-04957"],
                }));
            }
        }

        if !device.enabled_features().invocation_mask {
            if src_stages.intersects(PipelineStages::INVOCATION_MASK) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::INVOCATION_MASK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "invocation_mask",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-srcStageMask-04995"],
                }));
            }

            if dst_stages.intersects(PipelineStages::INVOCATION_MASK) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::INVOCATION_MASK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "invocation_mask",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-dstStageMask-04995"],
                }));
            }
        }

        if !(device.enabled_extensions().nv_ray_tracing
            || device.enabled_features().ray_tracing_pipeline)
        {
            if src_stages.intersects(PipelineStages::RAY_TRACING_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::RAY_TRACING_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "ray_tracing_pipeline",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-srcStageMask-07946"],
                }));
            }

            if dst_stages.intersects(PipelineStages::RAY_TRACING_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::RAY_TRACING_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "ray_tracing_pipeline",
                    )])]),
                    vuids: &["VUID-VkMemoryBarrier2-dstStageMask-07946"],
                }));
            }
        }

        if !AccessFlags::from(src_stages).contains(src_access) {
            return Err(Box::new(ValidationError {
                problem: "`src_access` contains one or more access types that are not performed \
                    by any stage in `src_stages`"
                    .into(),
                vuids: &[
                    "VUID-VkMemoryBarrier2-srcAccessMask-03900",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03901",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03902",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03903",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03904",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03905",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03906",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03907",
                    "VUID-VkMemoryBarrier2-srcAccessMask-07454",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03909",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03910",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03911",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03912",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03913",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03914",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03915",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03916",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03917",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03918",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03919",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03920",
                    "VUID-VkMemoryBarrier2-srcAccessMask-04747",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03922",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03923",
                    "VUID-VkMemoryBarrier2-srcAccessMask-04994",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03924",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03925",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03926",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03927",
                    "VUID-VkMemoryBarrier2-srcAccessMask-03928",
                    "VUID-VkMemoryBarrier2-srcAccessMask-06256",
                    "VUID-VkMemoryBarrier2-srcAccessMask-07272",
                    "VUID-VkMemoryBarrier2-srcAccessMask-04858",
                    "VUID-VkMemoryBarrier2-srcAccessMask-04859",
                    "VUID-VkMemoryBarrier2-srcAccessMask-04860",
                    "VUID-VkMemoryBarrier2-srcAccessMask-04861",
                    "VUID-VkMemoryBarrier2-srcAccessMask-07455",
                    "VUID-VkMemoryBarrier2-srcAccessMask-07456",
                    "VUID-VkMemoryBarrier2-srcAccessMask-07457",
                    "VUID-VkMemoryBarrier2-srcAccessMask-07458",
                    "VUID-VkMemoryBarrier2-srcAccessMask-08118",
                ],
                ..Default::default()
            }));
        }

        if !AccessFlags::from(dst_stages).contains(dst_access) {
            return Err(Box::new(ValidationError {
                problem: "`dst_access` contains one or more access types that are not performed \
                    by any stage in `dst_stages`"
                    .into(),
                vuids: &[
                    "VUID-VkMemoryBarrier2-dstAccessMask-03900",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03901",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03902",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03903",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03904",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03905",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03906",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03907",
                    "VUID-VkMemoryBarrier2-dstAccessMask-07454",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03909",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03910",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03911",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03912",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03913",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03914",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03915",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03916",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03917",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03918",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03919",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03920",
                    "VUID-VkMemoryBarrier2-dstAccessMask-04747",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03922",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03923",
                    "VUID-VkMemoryBarrier2-dstAccessMask-04994",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03924",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03925",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03926",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03927",
                    "VUID-VkMemoryBarrier2-dstAccessMask-03928",
                    "VUID-VkMemoryBarrier2-dstAccessMask-06256",
                    "VUID-VkMemoryBarrier2-dstAccessMask-07272",
                    "VUID-VkMemoryBarrier2-dstAccessMask-04858",
                    "VUID-VkMemoryBarrier2-dstAccessMask-04859",
                    "VUID-VkMemoryBarrier2-dstAccessMask-04860",
                    "VUID-VkMemoryBarrier2-dstAccessMask-04861",
                    "VUID-VkMemoryBarrier2-dstAccessMask-07455",
                    "VUID-VkMemoryBarrier2-dstAccessMask-07456",
                    "VUID-VkMemoryBarrier2-dstAccessMask-07457",
                    "VUID-VkMemoryBarrier2-dstAccessMask-07458",
                    "VUID-VkMemoryBarrier2-dstAccessMask-08118",
                ],
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk2(&self) -> vk::MemoryBarrier2<'static> {
        let &Self {
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            _ne: _,
        } = self;

        vk::MemoryBarrier2::default()
            .src_stage_mask(src_stages.into())
            .src_access_mask(src_access.into())
            .dst_stage_mask(dst_stages.into())
            .dst_access_mask(dst_access.into())
    }

    pub(crate) fn to_vk(&self) -> vk::MemoryBarrier<'static> {
        let &Self {
            src_stages: _,
            src_access,
            dst_stages: _,
            dst_access,
            _ne: _,
        } = self;

        vk::MemoryBarrier::default()
            .src_access_mask(src_access.into())
            .dst_access_mask(dst_access.into())
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
    pub queue_family_ownership_transfer: Option<QueueFamilyOwnershipTransfer>,

    /// The buffer to apply the barrier to.
    pub buffer: Arc<Buffer>,

    /// The byte range of `buffer` to apply the barrier to.
    pub range: Range<DeviceSize>,

    pub _ne: crate::NonExhaustive<'static>,
}

impl BufferMemoryBarrier {
    #[inline]
    pub fn buffer(buffer: Arc<Buffer>) -> Self {
        Self {
            src_stages: PipelineStages::empty(),
            src_access: AccessFlags::empty(),
            dst_stages: PipelineStages::empty(),
            dst_access: AccessFlags::empty(),
            queue_family_ownership_transfer: None,
            buffer,
            range: 0..0,
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            ref queue_family_ownership_transfer,
            ref buffer,
            ref range,
            _ne,
        } = self;

        src_stages.validate_device(device).map_err(|err| {
            err.add_context("src_stages")
                .set_vuids(&["VUID-VkBufferMemoryBarrier2-srcStageMask-parameter"])
        })?;

        dst_stages.validate_device(device).map_err(|err| {
            err.add_context("dst_stages")
                .set_vuids(&["VUID-VkBufferMemoryBarrier2-dstStageMask-parameter"])
        })?;

        src_access.validate_device(device).map_err(|err| {
            err.add_context("src_access")
                .set_vuids(&["VUID-VkBufferMemoryBarrier2-srcAccessMask-parameter"])
        })?;

        dst_access.validate_device(device).map_err(|err| {
            err.add_context("dst_access")
                .set_vuids(&["VUID-VkBufferMemoryBarrier2-dstAccessMask-parameter"])
        })?;

        if !device.enabled_features().synchronization2 {
            if src_stages.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains flags from `VkPipelineStageFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }

            if dst_stages.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains flags from `VkPipelineStageFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }

            if src_access.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "src_access".into(),
                    problem: "contains flags from `VkAccessFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }

            if dst_access.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "dst_access".into(),
                    problem: "contains flags from `VkAccessFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }
        }

        if !device.enabled_features().geometry_shader {
            if src_stages.intersects(PipelineStages::GEOMETRY_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::GEOMETRY_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "geometry_shader",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-03929"],
                }));
            }

            if dst_stages.intersects(PipelineStages::GEOMETRY_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::GEOMETRY_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "geometry_shader",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-dstStageMask-03929"],
                }));
            }
        }

        if !device.enabled_features().tessellation_shader {
            if src_stages.intersects(
                PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER,
            ) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::TESSELLATION_CONTROL_SHADER` or \
                        `PipelineStages::TESSELLATION_EVALUATION_SHADER`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "tessellation_shader",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-03930"],
                }));
            }

            if dst_stages.intersects(
                PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER,
            ) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::TESSELLATION_CONTROL_SHADER` or \
                        `PipelineStages::TESSELLATION_EVALUATION_SHADER`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "tessellation_shader",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-dstStageMask-03930"],
                }));
            }
        }

        if !device.enabled_features().conditional_rendering {
            if src_stages.intersects(PipelineStages::CONDITIONAL_RENDERING) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::CONDITIONAL_RENDERING`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "conditional_rendering",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-03931"],
                }));
            }

            if dst_stages.intersects(PipelineStages::CONDITIONAL_RENDERING) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::CONDITIONAL_RENDERING`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "conditional_rendering",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-dstStageMask-03931"],
                }));
            }
        }

        if !device.enabled_features().fragment_density_map {
            if src_stages.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_DENSITY_PROCESS`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "fragment_density_map",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-03932"],
                }));
            }

            if dst_stages.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_DENSITY_PROCESS`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "fragment_density_map",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-dstStageMask-03932"],
                }));
            }
        }

        if !device.enabled_features().transform_feedback {
            if src_stages.intersects(PipelineStages::TRANSFORM_FEEDBACK) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::TRANSFORM_FEEDBACK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "transform_feedback",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-03933"],
                }));
            }

            if dst_stages.intersects(PipelineStages::TRANSFORM_FEEDBACK) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::TRANSFORM_FEEDBACK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "transform_feedback",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-dstStageMask-03933"],
                }));
            }
        }

        if !device.enabled_features().mesh_shader {
            if src_stages.intersects(PipelineStages::MESH_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::MESH_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "mesh_shader",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-03934"],
                }));
            }

            if dst_stages.intersects(PipelineStages::MESH_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::MESH_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "mesh_shader",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-dstStageMask-03934"],
                }));
            }
        }

        if !device.enabled_features().task_shader {
            if src_stages.intersects(PipelineStages::TASK_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::TASK_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "task_shader",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-03935"],
                }));
            }

            if dst_stages.intersects(PipelineStages::TASK_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::TASK_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "task_shader",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-dstStageMask-03935"],
                }));
            }
        }

        if !(device.enabled_features().attachment_fragment_shading_rate
            || device.enabled_features().shading_rate_image)
        {
            if src_stages.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::DeviceFeature(
                            "attachment_fragment_shading_rate",
                        )]),
                        RequiresAllOf(&[Requires::DeviceFeature("shading_rate_image")]),
                    ]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-shadingRateImage-07316"],
                }));
            }

            if dst_stages.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::DeviceFeature(
                            "attachment_fragment_shading_rate",
                        )]),
                        RequiresAllOf(&[Requires::DeviceFeature("shading_rate_image")]),
                    ]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-shadingRateImage-07316"],
                }));
            }
        }

        if !device.enabled_features().subpass_shading {
            if src_stages.intersects(PipelineStages::SUBPASS_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::SUBPASS_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "subpass_shading",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-04957"],
                }));
            }

            if dst_stages.intersects(PipelineStages::SUBPASS_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::SUBPASS_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "subpass_shading",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-dstStageMask-04957"],
                }));
            }
        }

        if !device.enabled_features().invocation_mask {
            if src_stages.intersects(PipelineStages::INVOCATION_MASK) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::INVOCATION_MASK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "invocation_mask",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-04995"],
                }));
            }

            if dst_stages.intersects(PipelineStages::INVOCATION_MASK) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::INVOCATION_MASK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "invocation_mask",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-dstStageMask-04995"],
                }));
            }
        }

        if !(device.enabled_extensions().nv_ray_tracing
            || device.enabled_features().ray_tracing_pipeline)
        {
            if src_stages.intersects(PipelineStages::RAY_TRACING_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::RAY_TRACING_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "ray_tracing_pipeline",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-07946"],
                }));
            }

            if dst_stages.intersects(PipelineStages::RAY_TRACING_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::RAY_TRACING_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "ray_tracing_pipeline",
                    )])]),
                    vuids: &["VUID-VkBufferMemoryBarrier2-dstStageMask-07946"],
                }));
            }
        }

        if !AccessFlags::from(src_stages).contains(src_access) {
            return Err(Box::new(ValidationError {
                problem: "`src_access` contains one or more access types that are not performed \
                    by any stage in `src_stages`"
                    .into(),
                vuids: &[
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03900",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03901",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03902",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03903",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03904",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03905",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03906",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03907",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-07454",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03909",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03910",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03911",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03912",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03913",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03914",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03915",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03916",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03917",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03918",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03919",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03920",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-04747",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03922",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03923",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-04994",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03924",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03925",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03926",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03927",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-03928",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-06256",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-07272",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-04858",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-04859",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-04860",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-04861",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-07455",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-07456",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-07457",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-07458",
                    "VUID-VkBufferMemoryBarrier2-srcAccessMask-08118",
                ],
                ..Default::default()
            }));
        }

        if !AccessFlags::from(dst_stages).contains(dst_access) {
            return Err(Box::new(ValidationError {
                problem: "`dst_access` contains one or more access types that are not performed \
                    by any stage in `dst_stages`"
                    .into(),
                vuids: &[
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03900",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03901",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03902",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03903",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03904",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03905",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03906",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03907",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-07454",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03909",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03910",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03911",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03912",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03913",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03914",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03915",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03916",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03917",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03918",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03919",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03920",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-04747",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03922",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03923",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-04994",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03924",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03925",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03926",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03927",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-03928",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-06256",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-07272",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-04858",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-04859",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-04860",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-04861",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-07455",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-07456",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-07457",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-07458",
                    "VUID-VkBufferMemoryBarrier2-dstAccessMask-08118",
                ],
                ..Default::default()
            }));
        }

        if range.is_empty() {
            return Err(Box::new(ValidationError {
                context: "range".into(),
                problem: "is empty".into(),
                vuids: &["VUID-VkBufferMemoryBarrier2-size-01188"],
                ..Default::default()
            }));
        }

        if range.end > buffer.size() {
            return Err(Box::new(ValidationError {
                problem: "`range.end` is greater than `buffer.size()`".into(),
                vuids: &[
                    "VUID-VkBufferMemoryBarrier2-offset-01187",
                    "VUID-VkBufferMemoryBarrier2-size-01189",
                ],
                ..Default::default()
            }));
        }

        if let Some(queue_family_ownership_transfer) = queue_family_ownership_transfer {
            if src_stages.intersects(PipelineStages::HOST) {
                return Err(Box::new(ValidationError {
                    problem: "`src_stages` contains `PipelineStages::HOST`, but \
                        `queue_family_ownership_transfer` is `Some`"
                        .into(),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-03851"],
                    ..Default::default()
                }));
            }

            if dst_stages.intersects(PipelineStages::HOST) {
                return Err(Box::new(ValidationError {
                    problem: "`dst_stages` contains `PipelineStages::HOST`, but \
                        `queue_family_ownership_transfer` is `Some`"
                        .into(),
                    vuids: &["VUID-VkBufferMemoryBarrier2-srcStageMask-03851"],
                    ..Default::default()
                }));
            }

            // VUID-VkBufferMemoryBarrier2-srcQueueFamilyIndex-04087
            // VUID-VkBufferMemoryBarrier2-buffer-04088
            // Ensured by the use of an enum.

            let queue_family_count =
                device.physical_device().queue_family_properties().len() as u32;

            match queue_family_ownership_transfer {
                &QueueFamilyOwnershipTransfer::ExclusiveBetweenLocal {
                    src_index,
                    dst_index,
                } => {
                    if src_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.src_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkBufferMemoryBarrier2-buffer-04089"],
                            ..Default::default()
                        }));
                    }

                    if dst_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.dst_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkBufferMemoryBarrier2-buffer-04089"],
                            ..Default::default()
                        }));
                    }

                    if src_index == dst_index {
                        return Err(Box::new(ValidationError {
                            problem: "`queue_family_ownership_transfer.src_index` is equal to \
                                `queue_family_ownership_transfer.dst_index`"
                                .into(),
                            // just for sanity, so that the value is `Some`
                            // if and only if there is a transfer
                            ..Default::default()
                        }));
                    }
                }
                &QueueFamilyOwnershipTransfer::ExclusiveToExternal { src_index } => {
                    if src_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.src_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkBufferMemoryBarrier2-buffer-04089"],
                            ..Default::default()
                        }));
                    }
                }
                &QueueFamilyOwnershipTransfer::ExclusiveFromExternal { dst_index } => {
                    if dst_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.dst_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkBufferMemoryBarrier2-buffer-04089"],
                            ..Default::default()
                        }));
                    }
                }
                &QueueFamilyOwnershipTransfer::ExclusiveToForeign { src_index } => {
                    if src_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.src_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkBufferMemoryBarrier2-buffer-04089"],
                            ..Default::default()
                        }));
                    }
                }
                &QueueFamilyOwnershipTransfer::ExclusiveFromForeign { dst_index } => {
                    if dst_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.dst_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkBufferMemoryBarrier2-buffer-04089"],
                            ..Default::default()
                        }));
                    }
                }
                QueueFamilyOwnershipTransfer::ConcurrentToExternal
                | QueueFamilyOwnershipTransfer::ConcurrentFromExternal
                | QueueFamilyOwnershipTransfer::ConcurrentToForeign
                | QueueFamilyOwnershipTransfer::ConcurrentFromForeign => (),
            }
        }

        Ok(())
    }

    pub(crate) fn to_vk2(&self) -> vk::BufferMemoryBarrier2<'static> {
        let &Self {
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            ref queue_family_ownership_transfer,
            ref buffer,
            ref range,
            _ne: _,
        } = self;

        let (src_queue_family_index, dst_queue_family_index) =
            queue_family_ownership_transfer.as_ref().map_or(
                (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED),
                QueueFamilyOwnershipTransfer::to_vk,
            );

        vk::BufferMemoryBarrier2::default()
            .src_stage_mask(src_stages.into())
            .src_access_mask(src_access.into())
            .dst_stage_mask(dst_stages.into())
            .dst_access_mask(dst_access.into())
            .src_queue_family_index(src_queue_family_index)
            .dst_queue_family_index(dst_queue_family_index)
            .buffer(buffer.handle())
            .offset(range.start)
            .size(range.end - range.start)
    }

    pub(crate) fn to_vk(&self) -> vk::BufferMemoryBarrier<'static> {
        let &Self {
            src_stages: _,
            src_access,
            dst_stages: _,
            dst_access,
            queue_family_ownership_transfer,
            ref buffer,
            ref range,
            _ne: _,
        } = self;

        let (src_queue_family_index, dst_queue_family_index) =
            queue_family_ownership_transfer.as_ref().map_or(
                (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED),
                QueueFamilyOwnershipTransfer::to_vk,
            );

        vk::BufferMemoryBarrier::default()
            .src_access_mask(src_access.into())
            .dst_access_mask(dst_access.into())
            .src_queue_family_index(src_queue_family_index)
            .dst_queue_family_index(dst_queue_family_index)
            .buffer(buffer.handle())
            .offset(range.start)
            .size(range.end - range.start)
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
    pub queue_family_ownership_transfer: Option<QueueFamilyOwnershipTransfer>,

    /// The image to apply the barrier to.
    pub image: Arc<Image>,

    /// The subresource range of `image` to apply the barrier to.
    pub subresource_range: ImageSubresourceRange,

    pub _ne: crate::NonExhaustive<'static>,
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
            queue_family_ownership_transfer: None,
            image,
            subresource_range: ImageSubresourceRange {
                // Can't use image format aspects because `color` can't be specified with `planeN`.
                aspects: ImageAspects::empty(),
                mip_levels: 0..0,
                array_layers: 0..0,
            },
            _ne: crate::NE,
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            old_layout,
            new_layout,
            ref queue_family_ownership_transfer,
            ref image,
            ref subresource_range,
            _ne,
        } = self;

        src_stages.validate_device(device).map_err(|err| {
            err.add_context("src_stages")
                .set_vuids(&["VUID-VkImageMemoryBarrier2-srcStageMask-parameter"])
        })?;

        dst_stages.validate_device(device).map_err(|err| {
            err.add_context("dst_stages")
                .set_vuids(&["VUID-VkImageMemoryBarrier2-dstStageMask-parameter"])
        })?;

        src_access.validate_device(device).map_err(|err| {
            err.add_context("src_access")
                .set_vuids(&["VUID-VkImageMemoryBarrier2-srcAccessMask-parameter"])
        })?;

        dst_access.validate_device(device).map_err(|err| {
            err.add_context("dst_access")
                .set_vuids(&["VUID-VkImageMemoryBarrier2-dstAccessMask-parameter"])
        })?;

        if !device.enabled_features().synchronization2 {
            if src_stages.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains flags from `VkPipelineStageFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }

            if dst_stages.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains flags from `VkPipelineStageFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }

            if src_access.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "src_access".into(),
                    problem: "contains flags from `VkAccessFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }

            if dst_access.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "dst_access".into(),
                    problem: "contains flags from `VkAccessFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }
        }

        if !device.enabled_features().geometry_shader {
            if src_stages.intersects(PipelineStages::GEOMETRY_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::GEOMETRY_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "geometry_shader",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-03929"],
                }));
            }

            if dst_stages.intersects(PipelineStages::GEOMETRY_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::GEOMETRY_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "geometry_shader",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-dstStageMask-03929"],
                }));
            }
        }

        if !device.enabled_features().tessellation_shader {
            if src_stages.intersects(
                PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER,
            ) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::TESSELLATION_CONTROL_SHADER` or \
                        `PipelineStages::TESSELLATION_EVALUATION_SHADER`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "tessellation_shader",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-03930"],
                }));
            }

            if dst_stages.intersects(
                PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER,
            ) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::TESSELLATION_CONTROL_SHADER` or \
                        `PipelineStages::TESSELLATION_EVALUATION_SHADER`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "tessellation_shader",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-dstStageMask-03930"],
                }));
            }
        }

        if !device.enabled_features().conditional_rendering {
            if src_stages.intersects(PipelineStages::CONDITIONAL_RENDERING) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::CONDITIONAL_RENDERING`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "conditional_rendering",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-03931"],
                }));
            }

            if dst_stages.intersects(PipelineStages::CONDITIONAL_RENDERING) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::CONDITIONAL_RENDERING`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "conditional_rendering",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-dstStageMask-03931"],
                }));
            }
        }

        if !device.enabled_features().fragment_density_map {
            if src_stages.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_DENSITY_PROCESS`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "fragment_density_map",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-03932"],
                }));
            }

            if dst_stages.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_DENSITY_PROCESS`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "fragment_density_map",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-dstStageMask-03932"],
                }));
            }
        }

        if !device.enabled_features().transform_feedback {
            if src_stages.intersects(PipelineStages::TRANSFORM_FEEDBACK) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::TRANSFORM_FEEDBACK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "transform_feedback",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-03933"],
                }));
            }

            if dst_stages.intersects(PipelineStages::TRANSFORM_FEEDBACK) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::TRANSFORM_FEEDBACK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "transform_feedback",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-dstStageMask-03933"],
                }));
            }
        }

        if !device.enabled_features().mesh_shader {
            if src_stages.intersects(PipelineStages::MESH_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::MESH_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "mesh_shader",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-03934"],
                }));
            }

            if dst_stages.intersects(PipelineStages::MESH_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::MESH_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "mesh_shader",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-dstStageMask-03934"],
                }));
            }
        }

        if !device.enabled_features().task_shader {
            if src_stages.intersects(PipelineStages::TASK_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::TASK_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "task_shader",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-03935"],
                }));
            }

            if dst_stages.intersects(PipelineStages::TASK_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::TASK_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "task_shader",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-dstStageMask-03935"],
                }));
            }
        }

        if !(device.enabled_features().attachment_fragment_shading_rate
            || device.enabled_features().shading_rate_image)
        {
            if src_stages.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::DeviceFeature(
                            "attachment_fragment_shading_rate",
                        )]),
                        RequiresAllOf(&[Requires::DeviceFeature("shading_rate_image")]),
                    ]),
                    vuids: &["VUID-VkImageMemoryBarrier2-shadingRateImage-07316"],
                }));
            }

            if dst_stages.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::DeviceFeature(
                            "attachment_fragment_shading_rate",
                        )]),
                        RequiresAllOf(&[Requires::DeviceFeature("shading_rate_image")]),
                    ]),
                    vuids: &["VUID-VkImageMemoryBarrier2-shadingRateImage-07316"],
                }));
            }
        }

        if !device.enabled_features().subpass_shading {
            if src_stages.intersects(PipelineStages::SUBPASS_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::SUBPASS_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "subpass_shading",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-04957"],
                }));
            }

            if dst_stages.intersects(PipelineStages::SUBPASS_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::SUBPASS_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "subpass_shading",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-dstStageMask-04957"],
                }));
            }
        }

        if !device.enabled_features().invocation_mask {
            if src_stages.intersects(PipelineStages::INVOCATION_MASK) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::INVOCATION_MASK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "invocation_mask",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-04995"],
                }));
            }

            if dst_stages.intersects(PipelineStages::INVOCATION_MASK) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::INVOCATION_MASK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "invocation_mask",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-dstStageMask-04995"],
                }));
            }
        }

        if !(device.enabled_extensions().nv_ray_tracing
            || device.enabled_features().ray_tracing_pipeline)
        {
            if src_stages.intersects(PipelineStages::RAY_TRACING_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "src_stages".into(),
                    problem: "contains `PipelineStages::RAY_TRACING_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "ray_tracing_pipeline",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-07946"],
                }));
            }

            if dst_stages.intersects(PipelineStages::RAY_TRACING_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "dst_stages".into(),
                    problem: "contains `PipelineStages::RAY_TRACING_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "ray_tracing_pipeline",
                    )])]),
                    vuids: &["VUID-VkImageMemoryBarrier2-dstStageMask-07946"],
                }));
            }
        }

        if !AccessFlags::from(src_stages).contains(src_access) {
            return Err(Box::new(ValidationError {
                problem: "`src_access` contains one or more access types that are not performed \
                    by any stage in `src_stages`"
                    .into(),
                vuids: &[
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03900",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03901",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03902",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03903",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03904",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03905",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03906",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03907",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-07454",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03909",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03910",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03911",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03912",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03913",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03914",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03915",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03916",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03917",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03918",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03919",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03920",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-04747",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03922",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03923",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-04994",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03924",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03925",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03926",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03927",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-03928",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-06256",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-07272",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-04858",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-04859",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-04860",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-04861",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-07455",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-07456",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-07457",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-07458",
                    "VUID-VkImageMemoryBarrier2-srcAccessMask-08118",
                ],
                ..Default::default()
            }));
        }

        if !AccessFlags::from(dst_stages).contains(dst_access) {
            return Err(Box::new(ValidationError {
                problem: "`dst_access` contains one or more access types that are not performed \
                    by any stage in `dst_stages`"
                    .into(),
                vuids: &[
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03900",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03901",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03902",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03903",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03904",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03905",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03906",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03907",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-07454",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03909",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03910",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03911",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03912",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03913",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03914",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03915",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03916",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03917",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03918",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03919",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03920",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-04747",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03922",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03923",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-04994",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03924",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03925",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03926",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03927",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-03928",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-06256",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-07272",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-04858",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-04859",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-04860",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-04861",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-07455",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-07456",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-07457",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-07458",
                    "VUID-VkImageMemoryBarrier2-dstAccessMask-08118",
                ],
                ..Default::default()
            }));
        }

        // VUID-VkImageMemoryBarrier2-synchronization2-07793
        // If the synchronization2 feature is not enabled, oldLayout must not be
        // VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR or VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR

        // VUID-VkImageMemoryBarrier2-synchronization2-07794
        // If the synchronization2 feature is not enabled, newLayout must not be
        // VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR or VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL_KHR

        // VUID-VkImageMemoryBarrier2-attachmentFeedbackLoopLayout-07313
        // If the attachmentFeedbackLoopLayout feature is not enabled, newLayout must not be
        // VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT

        subresource_range
            .validate(device)
            .map_err(|err| err.add_context("subresource_range"))?;

        if subresource_range.mip_levels.end > image.mip_levels() {
            return Err(Box::new(ValidationError {
                problem: "`subresource_range.mip_levels.end` is greater than \
                    `image.mip_levels()`"
                    .into(),
                vuids: &[
                    "VUID-VkImageMemoryBarrier2-subresourceRange-01486",
                    "VUID-VkImageMemoryBarrier2-subresourceRange-01724",
                ],
                ..Default::default()
            }));
        }

        if subresource_range.array_layers.end > image.array_layers() {
            return Err(Box::new(ValidationError {
                problem: "`subresource_range.array_layers.end` is greater than \
                    `image.array_layers()`"
                    .into(),
                vuids: &[
                    "VUID-VkImageMemoryBarrier2-subresourceRange-01488",
                    "VUID-VkImageMemoryBarrier2-subresourceRange-01725",
                ],
                ..Default::default()
            }));
        }

        let image_format_aspects = image.format().aspects();

        if !image_format_aspects.contains(subresource_range.aspects) {
            return Err(Box::new(ValidationError {
                problem: "`subresource_range.aspects` is not a subset of \
                    `image.format().aspects()`"
                    .into(),
                vuids: &[
                    "VUID-VkImageMemoryBarrier2-image-01672",
                    "VUID-VkImageMemoryBarrier2-image-03319",
                ],
                ..Default::default()
            }));
        }

        if image_format_aspects.intersects(ImageAspects::COLOR)
            && !image.flags().intersects(ImageCreateFlags::DISJOINT)
            && subresource_range.aspects != ImageAspects::COLOR
        {
            return Err(Box::new(ValidationError {
                problem: "`image.format()` is a color format, and \
                    `image.flags()` does not contain `ImageCreateFlags::DISJOINT`, but \
                    `subresource_range.aspects` is not `ImageAspects::COLOR`"
                    .into(),
                vuids: &["VUID-VkImageMemoryBarrier2-image-01671"],
                ..Default::default()
            }));
        }

        if image_format_aspects.contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
            && !subresource_range
                .aspects
                .contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
            && !device.enabled_features().separate_depth_stencil_layouts
        {
            return Err(Box::new(ValidationError {
                problem: "`image.format()` has both a depth and a stencil component, and \
                    `subresource_range.aspects` does not contain both \
                    `ImageAspects::DEPTH` and `ImageAspects::STENCIL`"
                    .into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "separate_depth_stencil_layouts",
                )])]),
                vuids: &["VUID-VkImageMemoryBarrier2-image-03320"],
                ..Default::default()
            }));
        }

        if subresource_range.aspects.intersects(ImageAspects::DEPTH) {
            if matches!(
                old_layout,
                ImageLayout::StencilAttachmentOptimal | ImageLayout::StencilReadOnlyOptimal
            ) {
                return Err(Box::new(ValidationError {
                    problem: "`subresource_range.aspects` contains `ImageAspects::DEPTH`, but \
                        `old_layout` is `ImageLayout::StencilAttachmentOptimal` or \
                        `ImageLayout::StencilReadOnlyOptimal`"
                        .into(),
                    vuids: &["VUID-VkImageMemoryBarrier2-aspectMask-08702"],
                    ..Default::default()
                }));
            }

            if matches!(
                new_layout,
                ImageLayout::StencilAttachmentOptimal | ImageLayout::StencilReadOnlyOptimal
            ) {
                return Err(Box::new(ValidationError {
                    problem: "`subresource_range.aspects` contains `ImageAspects::DEPTH`, but \
                        `new_layout` is `ImageLayout::StencilAttachmentOptimal` or \
                        `ImageLayout::StencilReadOnlyOptimal`"
                        .into(),
                    vuids: &["VUID-VkImageMemoryBarrier2-aspectMask-08702"],
                    ..Default::default()
                }));
            }
        }

        if subresource_range.aspects.intersects(ImageAspects::STENCIL) {
            if matches!(
                old_layout,
                ImageLayout::StencilAttachmentOptimal | ImageLayout::StencilReadOnlyOptimal
            ) {
                return Err(Box::new(ValidationError {
                    problem: "`subresource_range.aspects` contains `ImageAspects::STENCIL`, but \
                        `old_layout` is `ImageLayout::DepthAttachmentOptimal` or \
                        `ImageLayout::DepthReadOnlyOptimal`"
                        .into(),
                    vuids: &["VUID-VkImageMemoryBarrier2-aspectMask-08703"],
                    ..Default::default()
                }));
            }

            if matches!(
                new_layout,
                ImageLayout::DepthAttachmentOptimal | ImageLayout::DepthReadOnlyOptimal
            ) {
                return Err(Box::new(ValidationError {
                    problem: "`subresource_range.aspects` contains `ImageAspects::STENCIL`, but \
                        `new_layout` is `ImageLayout::DepthAttachmentOptimal` or \
                        `ImageLayout::DepthReadOnlyOptimal`"
                        .into(),
                    vuids: &["VUID-VkImageMemoryBarrier2-aspectMask-08703"],
                    ..Default::default()
                }));
            }
        }

        if let Some(queue_family_ownership_transfer) = queue_family_ownership_transfer {
            if src_stages.intersects(PipelineStages::HOST) {
                return Err(Box::new(ValidationError {
                    problem: "`src_stages` contains `PipelineStages::HOST`, but \
                        `queue_family_ownership_transfer` is `Some`"
                        .into(),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-03854"],
                    ..Default::default()
                }));
            }

            if dst_stages.intersects(PipelineStages::HOST) {
                return Err(Box::new(ValidationError {
                    problem: "`dst_stages` contains `PipelineStages::HOST`, but \
                        `queue_family_ownership_transfer` is `Some`"
                        .into(),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-03854"],
                    ..Default::default()
                }));
            }

            // VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04070
            // VUID-VkImageMemoryBarrier2-image-04071
            // Ensured by the use of an enum.

            let queue_family_count =
                device.physical_device().queue_family_properties().len() as u32;

            match queue_family_ownership_transfer {
                &QueueFamilyOwnershipTransfer::ExclusiveBetweenLocal {
                    src_index,
                    dst_index,
                } => {
                    if src_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.src_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-image-04072"],
                            ..Default::default()
                        }));
                    }

                    if dst_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.dst_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-image-04072"],
                            ..Default::default()
                        }));
                    }

                    if src_index == dst_index {
                        return Err(Box::new(ValidationError {
                            problem: "`queue_family_ownership_transfer.src_index` is equal to \
                                `queue_family_ownership_transfer.dst_index`"
                                .into(),
                            // just for sanity, so that the value is `Some`
                            // if and only if there is a transfer
                            ..Default::default()
                        }));
                    }
                }
                &QueueFamilyOwnershipTransfer::ExclusiveToExternal { src_index } => {
                    if src_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.src_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-image-04072"],
                            ..Default::default()
                        }));
                    }
                }
                &QueueFamilyOwnershipTransfer::ExclusiveFromExternal { dst_index } => {
                    if dst_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.dst_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-image-04072"],
                            ..Default::default()
                        }));
                    }
                }
                &QueueFamilyOwnershipTransfer::ExclusiveToForeign { src_index } => {
                    if src_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.src_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-image-04072"],
                            ..Default::default()
                        }));
                    }
                }
                &QueueFamilyOwnershipTransfer::ExclusiveFromForeign { dst_index } => {
                    if dst_index >= queue_family_count {
                        return Err(Box::new(ValidationError {
                            context: "queue_family_ownership_transfer.dst_index".into(),
                            problem: "is not less than the number of queue families in the \
                                physical device"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-image-04072"],
                            ..Default::default()
                        }));
                    }
                }
                QueueFamilyOwnershipTransfer::ConcurrentToExternal
                | QueueFamilyOwnershipTransfer::ConcurrentFromExternal
                | QueueFamilyOwnershipTransfer::ConcurrentToForeign
                | QueueFamilyOwnershipTransfer::ConcurrentFromForeign => (),
            }
        }

        let is_image_layout_transition =
            !device.enabled_features().synchronization2 || old_layout != new_layout;
        let is_queue_family_ownership_transfer = queue_family_ownership_transfer.is_some();

        if is_image_layout_transition || is_queue_family_ownership_transfer {
            match old_layout {
                ImageLayout::ColorAttachmentOptimal => {
                    if !image.usage().intersects(ImageUsage::COLOR_ATTACHMENT) {
                        return Err(Box::new(ValidationError {
                            problem: "`old_layout` is `ImageLayout::ColorAttachmentOptimal`, but \
                                `image.usage()` does not contain `ImageUsage::COLOR_ATTACHMENT`"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-oldLayout-01208"],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::DepthStencilAttachmentOptimal
                | ImageLayout::DepthStencilReadOnlyOptimal
                | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                | ImageLayout::DepthAttachmentOptimal
                | ImageLayout::StencilAttachmentOptimal => {
                    if !image
                        .usage()
                        .intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`old_layout` is \
                                `ImageLayout::DepthStencilAttachmentOptimal`, \
                                `ImageLayout::DepthStencilReadOnlyOptimal`, \
                                `ImageLayout::DepthReadOnlyStencilAttachmentOptimal`, \
                                `ImageLayout::DepthAttachmentStencilReadOnlyOptimal`, \
                                `ImageLayout::DepthAttachmentOptimal` or \
                                `ImageLayout::StencilAttachmentOptimal`, but \
                                `image.usage()` does not contain \
                                `ImageUsage::DEPTH_STENCIL_ATTACHMENT`"
                                .into(),
                            vuids: &[
                                "VUID-VkImageMemoryBarrier2-oldLayout-01209",
                                "VUID-VkImageMemoryBarrier2-oldLayout-01210",
                                "VUID-VkImageMemoryBarrier2-oldLayout-01658",
                                "VUID-VkImageMemoryBarrier2-oldLayout-01659",
                                "VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04066",
                                "VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04068",
                            ],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::ShaderReadOnlyOptimal => {
                    if !image
                        .usage()
                        .intersects(ImageUsage::SAMPLED | ImageUsage::INPUT_ATTACHMENT)
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`old_layout` is `ImageLayout::ShaderReadOnlyOptimal`, but \
                                `image.usage()` does not contain `ImageUsage::SAMPLED` or \
                                `ImageUsage::INPUT_ATTACHMENT`"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-oldLayout-01211"],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::TransferSrcOptimal => {
                    if !image.usage().intersects(ImageUsage::TRANSFER_SRC) {
                        return Err(Box::new(ValidationError {
                            problem: "`old_layout` is `ImageLayout::TransferSrcOptimal`, but \
                                `image.usage()` does not contain `ImageUsage::TRANSFER_SRC`"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-oldLayout-01212"],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::TransferDstOptimal => {
                    if !image.usage().intersects(ImageUsage::TRANSFER_DST) {
                        return Err(Box::new(ValidationError {
                            problem: "`old_layout` is `ImageLayout::TransferDstOptimal`, but \
                                `image.usage()` does not contain `ImageUsage::TRANSFER_DST`"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-oldLayout-01213"],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::Preinitialized => todo!(),
                ImageLayout::DepthReadOnlyOptimal | ImageLayout::StencilReadOnlyOptimal => {
                    if !image.usage().intersects(
                        ImageUsage::DEPTH_STENCIL_ATTACHMENT
                            | ImageUsage::SAMPLED
                            | ImageUsage::INPUT_ATTACHMENT,
                    ) {
                        return Err(Box::new(ValidationError {
                            problem: "`old_layout` is `ImageLayout::DepthReadOnlyOptimal` or \
                                `ImageLayout::StencilReadOnlyOptimal`, but \
                                `image.usage()` does not contain \
                                `ImageUsage::DEPTH_STENCIL_ATTACHMENT`, `ImageUsage::SAMPLED` or \
                                `ImageUsage::INPUT_ATTACHMENT`"
                                .into(),
                            vuids: &[
                                "VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04065",
                                "VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04067",
                            ],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::Undefined | ImageLayout::General | ImageLayout::PresentSrc => (),
            }

            match new_layout {
                ImageLayout::Undefined | ImageLayout::Preinitialized => {
                    return Err(Box::new(ValidationError {
                        context: "new_layout".into(),
                        problem: "is `ImageLayout::Undefined` or `ImageLayout::Preinitialized`"
                            .into(),
                        vuids: &["VUID-VkImageMemoryBarrier2-newLayout-01198"],
                        ..Default::default()
                    }));
                }
                ImageLayout::ColorAttachmentOptimal => {
                    if !image.usage().intersects(ImageUsage::COLOR_ATTACHMENT) {
                        return Err(Box::new(ValidationError {
                            problem: "`new_layout` is `ImageLayout::ColorAttachmentOptimal`, but \
                                `image.usage()` does not contain `ImageUsage::COLOR_ATTACHMENT`"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-oldLayout-01208"],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::DepthStencilAttachmentOptimal
                | ImageLayout::DepthStencilReadOnlyOptimal
                | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                | ImageLayout::DepthAttachmentStencilReadOnlyOptimal
                | ImageLayout::DepthAttachmentOptimal
                | ImageLayout::StencilAttachmentOptimal => {
                    if !image
                        .usage()
                        .intersects(ImageUsage::DEPTH_STENCIL_ATTACHMENT)
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`new_layout` is \
                                `ImageLayout::DepthStencilAttachmentOptimal`, \
                                `ImageLayout::DepthStencilReadOnlyOptimal`, \
                                `ImageLayout::DepthReadOnlyStencilAttachmentOptimal`, \
                                `ImageLayout::DepthAttachmentStencilReadOnlyOptimal`, \
                                `ImageLayout::DepthAttachmentOptimal` or \
                                `ImageLayout::StencilAttachmentOptimal`, but \
                                `image.usage()` does not contain \
                                `ImageUsage::DEPTH_STENCIL_ATTACHMENT`"
                                .into(),
                            vuids: &[
                                "VUID-VkImageMemoryBarrier2-oldLayout-01209",
                                "VUID-VkImageMemoryBarrier2-oldLayout-01210",
                                "VUID-VkImageMemoryBarrier2-oldLayout-01658",
                                "VUID-VkImageMemoryBarrier2-oldLayout-01659",
                                "VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04066",
                                "VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04068",
                            ],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::ShaderReadOnlyOptimal => {
                    if !image
                        .usage()
                        .intersects(ImageUsage::SAMPLED | ImageUsage::INPUT_ATTACHMENT)
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`new_layout` is `ImageLayout::ShaderReadOnlyOptimal`, but \
                                `image.usage()` does not contain `ImageUsage::SAMPLED` or \
                                `ImageUsage::INPUT_ATTACHMENT`"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-oldLayout-01211"],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::TransferSrcOptimal => {
                    if !image.usage().intersects(ImageUsage::TRANSFER_SRC) {
                        return Err(Box::new(ValidationError {
                            problem: "`new_layout` is `ImageLayout::TransferSrcOptimal`, but \
                                `image.usage()` does not contain `ImageUsage::TRANSFER_SRC`"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-oldLayout-01212"],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::TransferDstOptimal => {
                    if !image.usage().intersects(ImageUsage::TRANSFER_DST) {
                        return Err(Box::new(ValidationError {
                            problem: "`new_layout` is `ImageLayout::TransferDstOptimal`, but \
                                `image.usage()` does not contain `ImageUsage::TRANSFER_DST`"
                                .into(),
                            vuids: &["VUID-VkImageMemoryBarrier2-oldLayout-01213"],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::DepthReadOnlyOptimal | ImageLayout::StencilReadOnlyOptimal => {
                    if !image.usage().intersects(
                        ImageUsage::DEPTH_STENCIL_ATTACHMENT
                            | ImageUsage::SAMPLED
                            | ImageUsage::INPUT_ATTACHMENT,
                    ) {
                        return Err(Box::new(ValidationError {
                            problem: "`new_layout` is `ImageLayout::DepthReadOnlyOptimal` or \
                                `ImageLayout::StencilReadOnlyOptimal`, but \
                                `image.usage()` does not contain \
                                `ImageUsage::DEPTH_STENCIL_ATTACHMENT`, `ImageUsage::SAMPLED` or \
                                `ImageUsage::INPUT_ATTACHMENT`"
                                .into(),
                            vuids: &[
                                "VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04065",
                                "VUID-VkImageMemoryBarrier2-srcQueueFamilyIndex-04067",
                            ],
                            ..Default::default()
                        }));
                    }
                }
                ImageLayout::General | ImageLayout::PresentSrc => (),
            }

            if src_stages.intersects(PipelineStages::HOST)
                && !matches!(
                    old_layout,
                    ImageLayout::Preinitialized | ImageLayout::Undefined | ImageLayout::General
                )
            {
                return Err(Box::new(ValidationError {
                    problem: "`src_stages` contains `PipelineStages::HOST`, but \
                        `old_layout` is not `ImageLayout::Preinitialized`, \
                        `ImageLayout::Undefined` or `ImageLayout::General`"
                        .into(),
                    vuids: &["VUID-VkImageMemoryBarrier2-srcStageMask-03855"],
                    ..Default::default()
                }));
            }

            // VUID-VkImageMemoryBarrier2-oldLayout-01197
            // Unsafe
        }

        Ok(())
    }

    pub(crate) fn to_vk2(&self) -> vk::ImageMemoryBarrier2<'static> {
        let &Self {
            src_stages,
            src_access,
            dst_stages,
            dst_access,
            old_layout,
            new_layout,
            ref queue_family_ownership_transfer,
            ref image,
            ref subresource_range,
            _ne: _,
        } = self;

        let (src_queue_family_index, dst_queue_family_index) =
            queue_family_ownership_transfer.as_ref().map_or(
                (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED),
                QueueFamilyOwnershipTransfer::to_vk,
            );

        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stages.into())
            .src_access_mask(src_access.into())
            .dst_stage_mask(dst_stages.into())
            .dst_access_mask(dst_access.into())
            .old_layout(old_layout.into())
            .new_layout(new_layout.into())
            .src_queue_family_index(src_queue_family_index)
            .dst_queue_family_index(dst_queue_family_index)
            .image(image.handle())
            .subresource_range(subresource_range.to_vk())
    }

    pub(crate) fn to_vk(&self) -> vk::ImageMemoryBarrier<'static> {
        let &Self {
            src_stages: _,
            src_access,
            dst_stages: _,
            dst_access,
            old_layout,
            new_layout,
            ref queue_family_ownership_transfer,
            ref image,
            ref subresource_range,
            _ne: _,
        } = self;

        let (src_queue_family_index, dst_queue_family_index) =
            queue_family_ownership_transfer.as_ref().map_or(
                (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED),
                QueueFamilyOwnershipTransfer::to_vk,
            );

        vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access.into())
            .dst_access_mask(dst_access.into())
            .old_layout(old_layout.into())
            .new_layout(new_layout.into())
            .src_queue_family_index(src_queue_family_index)
            .dst_queue_family_index(dst_queue_family_index)
            .image(image.handle())
            .subresource_range(subresource_range.to_vk())
    }
}

/// Specifies a queue family ownership transfer for a resource.
///
/// There are three classes of queues that can be used in an ownership transfer:
/// - A **local** queue exists on the current [`Instance`] and [`Device`].
/// - An **external** queue does not exist on the current [`Instance`], but has the same
///   [`device_uuid`] and [`driver_uuid`] as the current [`Device`].
/// - A **foreign** queue can be an external queue, or any queue on another device for which the
///   mentioned parameters do not match.
///
/// [`Instance`]: crate::instance::Instance
/// [`device_uuid`]: crate::device::DeviceProperties::device_uuid
/// [`driver_uuid`]: crate::device::DeviceProperties::driver_uuid
#[derive(Clone, Copy, Debug)]
pub enum QueueFamilyOwnershipTransfer {
    /// For a resource with [`Sharing::Exclusive`], transfers ownership between two local queues.
    ///
    /// [`Sharing::Exclusive`]: crate::sync::Sharing::Exclusive
    ExclusiveBetweenLocal {
        /// The queue family that currently owns the resource.
        src_index: u32,

        /// The queue family to transfer ownership to.
        dst_index: u32,
    },

    /// For a resource with [`Sharing::Exclusive`], transfers ownership from a local queue to an
    /// external queue.
    ///
    /// The device API version must be at least 1.1, or the [`khr_external_memory`] extension must
    /// be enabled on the device.
    ///
    /// [`Sharing::Exclusive`]: crate::sync::Sharing::Exclusive
    /// [`khr_external_memory`]: crate::device::DeviceExtensions::khr_external_memory
    ExclusiveToExternal {
        /// The queue family that currently owns the resource.
        src_index: u32,
    },

    /// For a resource with [`Sharing::Exclusive`], transfers ownership from an external queue to a
    /// local queue.
    ///
    /// The device API version must be at least 1.1, or the [`khr_external_memory`] extension must
    /// be enabled on the device.
    ///
    /// [`Sharing::Exclusive`]: crate::sync::Sharing::Exclusive
    /// [`khr_external_memory`]: crate::device::DeviceExtensions::khr_external_memory
    ExclusiveFromExternal {
        /// The queue family to transfer ownership to.
        dst_index: u32,
    },

    /// For a resource with [`Sharing::Exclusive`], transfers ownership from a local queue to a
    /// foreign queue.
    ///
    /// The [`ext_queue_family_foreign`] extension must be enabled on the device.
    ///
    /// [`Sharing::Exclusive`]: crate::sync::Sharing::Exclusive
    /// [`ext_queue_family_foreign`]: crate::device::DeviceExtensions::ext_queue_family_foreign
    ExclusiveToForeign {
        /// The queue family that currently owns the resource.
        src_index: u32,
    },

    /// For a resource with [`Sharing::Exclusive`], transfers ownership from a foreign queue to a
    /// local queue.
    ///
    /// The [`ext_queue_family_foreign`] extension must be enabled on the device.
    ///
    /// [`Sharing::Exclusive`]: crate::sync::Sharing::Exclusive
    /// [`ext_queue_family_foreign`]: crate::device::DeviceExtensions::ext_queue_family_foreign
    ExclusiveFromForeign {
        /// The queue family to transfer ownership to.
        dst_index: u32,
    },

    /// For a resource with [`Sharing::Concurrent`], transfers ownership from its local queues to
    /// an external queue.
    ///
    /// The device API version must be at least 1.1, or the [`khr_external_memory`] extension must
    /// be enabled on the device.
    ///
    /// [`Sharing::Concurrent`]: crate::sync::Sharing::Concurrent
    /// [`khr_external_memory`]: crate::device::DeviceExtensions::khr_external_memory
    ConcurrentToExternal,

    /// For a resource with [`Sharing::Concurrent`], transfers ownership from an external queue to
    /// its local queues.
    ///
    /// The device API version must be at least 1.1, or the [`khr_external_memory`] extension must
    /// be enabled on the device.
    ///
    /// [`Sharing::Concurrent`]: crate::sync::Sharing::Concurrent
    /// [`khr_external_memory`]: crate::device::DeviceExtensions::khr_external_memory
    ConcurrentFromExternal,

    /// For a resource with [`Sharing::Concurrent`], transfers ownership from its local queues to
    /// a foreign queue.
    ///
    /// The [`ext_queue_family_foreign`] extension must be enabled on the device.
    ///
    /// [`Sharing::Concurrent`]: crate::sync::Sharing::Concurrent
    /// [`ext_queue_family_foreign`]: crate::device::DeviceExtensions::ext_queue_family_foreign
    ConcurrentToForeign,

    /// For a resource with [`Sharing::Concurrent`], transfers ownership from a foreign queue to
    /// its local queues.
    ///
    /// The [`ext_queue_family_foreign`] extension must be enabled on the device.
    ///
    /// [`Sharing::Concurrent`]: crate::sync::Sharing::Concurrent
    /// [`ext_queue_family_foreign`]: crate::device::DeviceExtensions::ext_queue_family_foreign
    ConcurrentFromForeign,
}

impl QueueFamilyOwnershipTransfer {
    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn to_vk(&self) -> (u32, u32) {
        match *self {
            Self::ExclusiveBetweenLocal {
                src_index,
                dst_index,
            } => (src_index, dst_index),
            Self::ExclusiveToExternal { src_index } => (src_index, vk::QUEUE_FAMILY_EXTERNAL),
            Self::ExclusiveFromExternal { dst_index } => (vk::QUEUE_FAMILY_EXTERNAL, dst_index),
            Self::ExclusiveToForeign { src_index } => (src_index, vk::QUEUE_FAMILY_FOREIGN_EXT),
            Self::ExclusiveFromForeign { dst_index } => (vk::QUEUE_FAMILY_FOREIGN_EXT, dst_index),
            Self::ConcurrentToExternal => (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_EXTERNAL),
            Self::ConcurrentFromExternal => (vk::QUEUE_FAMILY_EXTERNAL, vk::QUEUE_FAMILY_IGNORED),
            Self::ConcurrentToForeign => (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_FOREIGN_EXT),
            Self::ConcurrentFromForeign => (vk::QUEUE_FAMILY_FOREIGN_EXT, vk::QUEUE_FAMILY_IGNORED),
        }
    }
}
