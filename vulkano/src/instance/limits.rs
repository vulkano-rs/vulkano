// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! The `Limits` struct provides a nicer API around `vkPhysicalDeviceLimits`.

use vk;

/// Limits of a physical device.
pub struct Limits<'a> {
    limits: &'a vk::PhysicalDeviceLimits,
}

macro_rules! limits_impl {
    ($($name:ident: $t:ty => $target:ident,)*) => (
        impl<'a> Limits<'a> {
            /// Builds the `Limits` object.
            #[inline]
            pub(crate) fn from_vk_limits(limits: &'a vk::PhysicalDeviceLimits) -> Limits<'a> {
                Limits {
                    limits
                }
            }
            
            $(
                #[inline]
                pub fn $name(&self) -> $t {
                    self.limits.$target
                }
            )*
        }
    )
}

limits_impl!{
    max_image_dimension_1d: u32 => maxImageDimension1D,
    max_image_dimension_2d: u32 => maxImageDimension2D,
    max_image_dimension_3d: u32 => maxImageDimension3D,
    max_image_dimension_cube: u32 => maxImageDimensionCube,
    max_image_array_layers: u32 => maxImageArrayLayers,
    max_texel_buffer_elements: u32 => maxTexelBufferElements,
    max_uniform_buffer_range: u32 => maxUniformBufferRange,
    max_storage_buffer_range: u32 => maxStorageBufferRange,
    max_push_constants_size: u32 => maxPushConstantsSize,
    max_memory_allocation_count: u32 => maxMemoryAllocationCount,
    max_sampler_allocation_count: u32 => maxSamplerAllocationCount,
    buffer_image_granularity: u64 => bufferImageGranularity,
    sparse_address_space_size: u64 => sparseAddressSpaceSize,
    max_bound_descriptor_sets: u32 => maxBoundDescriptorSets,
    max_per_stage_descriptor_samplers: u32 => maxPerStageDescriptorSamplers,
    max_per_stage_descriptor_uniform_buffers: u32 => maxPerStageDescriptorUniformBuffers,
    max_per_stage_descriptor_storage_buffers: u32 => maxPerStageDescriptorStorageBuffers,
    max_per_stage_descriptor_sampled_images: u32 => maxPerStageDescriptorSampledImages,
    max_per_stage_descriptor_storage_images: u32 => maxPerStageDescriptorStorageImages,
    max_per_stage_descriptor_input_attachments: u32 => maxPerStageDescriptorInputAttachments,
    max_per_stage_resources: u32 => maxPerStageResources,
    max_descriptor_set_samplers: u32 => maxDescriptorSetSamplers,
    max_descriptor_set_uniform_buffers: u32 => maxDescriptorSetUniformBuffers,
    max_descriptor_set_uniform_buffers_dynamic: u32 => maxDescriptorSetUniformBuffersDynamic,
    max_descriptor_set_storage_buffers: u32 => maxDescriptorSetStorageBuffers,
    max_descriptor_set_storage_buffers_dynamic: u32 => maxDescriptorSetStorageBuffersDynamic,
    max_descriptor_set_sampled_images: u32 => maxDescriptorSetSampledImages,
    max_descriptor_set_storage_images: u32 => maxDescriptorSetStorageImages,
    max_descriptor_set_input_attachments: u32 => maxDescriptorSetInputAttachments,
    max_vertex_input_attributes: u32 => maxVertexInputAttributes,
    max_vertex_input_bindings: u32 => maxVertexInputBindings,
    max_vertex_input_attribute_offset: u32 => maxVertexInputAttributeOffset,
    max_vertex_input_binding_stride: u32 => maxVertexInputBindingStride,
    max_vertex_output_components: u32 => maxVertexOutputComponents,
    max_tessellation_generation_level: u32 => maxTessellationGenerationLevel,
    max_tessellation_patch_size: u32 => maxTessellationPatchSize,
    max_tessellation_control_per_vertex_input_components: u32 => maxTessellationControlPerVertexInputComponents,
    max_tessellation_control_per_vertex_output_components: u32 => maxTessellationControlPerVertexOutputComponents,
    max_tessellation_control_per_patch_output_components: u32 => maxTessellationControlPerPatchOutputComponents,
    max_tessellation_control_total_output_components: u32 => maxTessellationControlTotalOutputComponents,
    max_tessellation_evaluation_input_components: u32 => maxTessellationEvaluationInputComponents,
    max_tessellation_evaluation_output_components: u32 => maxTessellationEvaluationOutputComponents,
    max_geometry_shader_invocations: u32 => maxGeometryShaderInvocations,
    max_geometry_input_components: u32 => maxGeometryInputComponents,
    max_geometry_output_components: u32 => maxGeometryOutputComponents,
    max_geometry_output_vertices: u32 => maxGeometryOutputVertices,
    max_geometry_total_output_components: u32 => maxGeometryTotalOutputComponents,
    max_fragment_input_components: u32 => maxFragmentInputComponents,
    max_fragment_output_attachments: u32 => maxFragmentOutputAttachments,
    max_fragment_dual_src_attachments: u32 => maxFragmentDualSrcAttachments,
    max_fragment_combined_output_resources: u32 => maxFragmentCombinedOutputResources,
    max_compute_shared_memory_size: u32 => maxComputeSharedMemorySize,
    max_compute_work_group_count: [u32; 3] => maxComputeWorkGroupCount,
    max_compute_work_group_invocations: u32 => maxComputeWorkGroupInvocations,
    max_compute_work_group_size: [u32; 3] => maxComputeWorkGroupSize,
    sub_pixel_precision_bits: u32 => subPixelPrecisionBits,
    sub_texel_precision_bits: u32 => subTexelPrecisionBits,
    mipmap_precision_bits: u32 => mipmapPrecisionBits,
    max_draw_indexed_index_value: u32 => maxDrawIndexedIndexValue,
    max_draw_indirect_count: u32 => maxDrawIndirectCount,
    max_sampler_lod_bias: f32 => maxSamplerLodBias,
    max_sampler_anisotropy: f32 => maxSamplerAnisotropy,
    max_viewports: u32 => maxViewports,
    max_viewport_dimensions: [u32; 2] => maxViewportDimensions,
    viewport_bounds_range: [f32; 2] => viewportBoundsRange,
    viewport_sub_pixel_bits: u32 => viewportSubPixelBits,
    min_memory_map_alignment: usize => minMemoryMapAlignment,
    min_texel_buffer_offset_alignment: u64 => minTexelBufferOffsetAlignment,
    min_uniform_buffer_offset_alignment: u64 => minUniformBufferOffsetAlignment,
    min_storage_buffer_offset_alignment: u64 => minStorageBufferOffsetAlignment,
    min_texel_offset: i32 => minTexelOffset,
    max_texel_offset: u32 => maxTexelOffset,
    min_texel_gather_offset: i32 => minTexelGatherOffset,
    max_texel_gather_offset: u32 => maxTexelGatherOffset,
    min_interpolation_offset: f32 => minInterpolationOffset,
    max_interpolation_offset: f32 => maxInterpolationOffset,
    sub_pixel_interpolation_offset_bits: u32 => subPixelInterpolationOffsetBits,
    max_framebuffer_width: u32 => maxFramebufferWidth,
    max_framebuffer_height: u32 => maxFramebufferHeight,
    max_framebuffer_layers: u32 => maxFramebufferLayers,
    framebuffer_color_sample_counts: u32 => framebufferColorSampleCounts,      // FIXME: SampleCountFlag
    framebuffer_depth_sample_counts: u32 => framebufferDepthSampleCounts,      // FIXME: SampleCountFlag
    framebuffer_stencil_sample_counts: u32 => framebufferStencilSampleCounts,      // FIXME: SampleCountFlag
    framebuffer_no_attachments_sample_counts: u32 => framebufferNoAttachmentsSampleCounts,      // FIXME: SampleCountFlag
    max_color_attachments: u32 => maxColorAttachments,
    sampled_image_color_sample_counts: u32 => sampledImageColorSampleCounts,        // FIXME: SampleCountFlag
    sampled_image_integer_sample_counts: u32 => sampledImageIntegerSampleCounts,        // FIXME: SampleCountFlag
    sampled_image_depth_sample_counts: u32 => sampledImageDepthSampleCounts,        // FIXME: SampleCountFlag
    sampled_image_stencil_sample_counts: u32 => sampledImageStencilSampleCounts,        // FIXME: SampleCountFlag
    storage_image_sample_counts: u32 => storageImageSampleCounts,      // FIXME: SampleCountFlag
    max_sample_mask_words: u32 => maxSampleMaskWords,
    timestamp_compute_and_graphics: u32 => timestampComputeAndGraphics,        // TODO: these are booleans
    timestamp_period: f32 => timestampPeriod,
    max_clip_distances: u32 => maxClipDistances,
    max_cull_distances: u32 => maxCullDistances,
    max_combined_clip_and_cull_distances: u32 => maxCombinedClipAndCullDistances,
    discrete_queue_priorities: u32 => discreteQueuePriorities,
    point_size_range: [f32; 2] => pointSizeRange,
    line_width_range: [f32; 2] => lineWidthRange,
    point_size_granularity: f32 => pointSizeGranularity,
    line_width_granularity: f32 => lineWidthGranularity,
    strict_lines: u32 => strictLines,        // TODO: these are booleans
    standard_sample_locations: u32 => standardSampleLocations,        // TODO: these are booleans
    optimal_buffer_copy_offset_alignment: u64 => optimalBufferCopyOffsetAlignment,
    optimal_buffer_copy_row_pitch_alignment: u64 => optimalBufferCopyRowPitchAlignment,
    non_coherent_atom_size: u64 => nonCoherentAtomSize,
}
