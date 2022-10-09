// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::macros::vulkan_bitflags;

vulkan_bitflags! {
    /// Describes how a buffer is going to be used. This is **not** just an optimization.
    ///
    /// If you try to use a buffer in a way that you didn't declare, an error will be returned.
    #[non_exhaustive]
    BufferUsage = BufferUsageFlags(u32);

    /// The buffer can be used as a source for transfer, blit, resolve and clear commands.
    transfer_src = TRANSFER_SRC,

    /// The buffer can be used as a destination for transfer, blit, resolve and clear commands.
    transfer_dst = TRANSFER_DST,

    /// The buffer can be used as a uniform texel buffer in a descriptor set.
    uniform_texel_buffer = UNIFORM_TEXEL_BUFFER,

    /// The buffer can be used as a storage texel buffer in a descriptor set.
    storage_texel_buffer = STORAGE_TEXEL_BUFFER,

    /// The buffer can be used as a uniform buffer in a descriptor set.
    uniform_buffer = UNIFORM_BUFFER,

    /// The buffer can be used as a storage buffer in a descriptor set.
    storage_buffer = STORAGE_BUFFER,

    /// The buffer can be used as an index buffer.
    index_buffer = INDEX_BUFFER,

    /// The buffer can be used as a vertex or instance buffer.
    vertex_buffer = VERTEX_BUFFER,

    /// The buffer can be used as an indirect buffer.
    indirect_buffer = INDIRECT_BUFFER,

    /// The buffer's device address can be retrieved.
    ///
    /// A buffer created with this usage can only be bound to device memory allocated with the
    /// [`device_address`] flag set unless the [`ext_buffer_device_address`] extension is used.
    ///
    /// [`device_address`]: crate::memory::MemoryAllocateFlags::device_address
    /// [`ext_buffer_device_address`]: crate::device::DeviceExtensions::ext_buffer_device_address
    shader_device_address = SHADER_DEVICE_ADDRESS {
        api_version: V1_2,
        device_extensions: [khr_buffer_device_address, ext_buffer_device_address],
    },

    /*
    // TODO: document
    video_decode_src = VIDEO_DECODE_SRC_KHR {
        device_extensions: [khr_video_decode_queue],
    },

    // TODO: document
    video_decode_dst = VIDEO_DECODE_DST_KHR {
        device_extensions: [khr_video_decode_queue],
    },

    // TODO: document
    transform_feedback_buffer = TRANSFORM_FEEDBACK_BUFFER_EXT {
        device_extensions: [ext_transform_feedback],
    },

    // TODO: document
    transform_feedback_counter_buffer = TRANSFORM_FEEDBACK_COUNTER_BUFFER_EXT {
        device_extensions: [ext_transform_feedback],
    },

    // TODO: document
    conditional_rendering = CONDITIONAL_RENDERING_EXT {
        device_extensions: [ext_conditional_rendering],
    },

    // TODO: document
    acceleration_structure_build_input_read_only = ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR {
        device_extensions: [khr_acceleration_structure],
    },

    // TODO: document
    acceleration_structure_storage = ACCELERATION_STRUCTURE_STORAGE_KHR {
        device_extensions: [khr_acceleration_structure],
    },

    // TODO: document
    shader_binding_table = SHADER_BINDING_TABLE_KHR {
        device_extensions: [khr_ray_tracing_pipeline, nv_ray_tracing],
    },

    // TODO: document
    video_encode_dst = VIDEO_ENCODE_DST_KHR {
        device_extensions: [khr_video_encode_queue],
    },

    // TODO: document
    video_encode_src = VIDEO_ENCODE_SRC_KHR {
        device_extensions: [khr_video_encode_queue],
    },
     */
}
