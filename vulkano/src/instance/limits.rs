// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! The `Limits` struct provides a nicer API around `vkPhysicalDeviceLimits`.

use crate::image::SampleCounts;

/// Limits of a physical device.
pub struct Limits<'a> {
    limits: &'a ash::vk::PhysicalDeviceLimits,
}

macro_rules! limits_impl {
    ($($name:ident: $t:ty => $target:ident,)*) => (
        impl<'a> Limits<'a> {
            /// Builds the `Limits` object.
            #[inline]
            pub(crate) fn from_vk_limits(limits: &'a ash::vk::PhysicalDeviceLimits) -> Limits<'a> {
                Limits {
                    limits
                }
            }

            $(
                #[inline]
                pub fn $name(&self) -> $t {
                    <$t>::from(self.limits.$target)
                }
            )*
        }
    )
}

limits_impl! {
    max_image_dimension_1d: u32 => max_image_dimension1_d,
    max_image_dimension_2d: u32 => max_image_dimension2_d,
    max_image_dimension_3d: u32 => max_image_dimension3_d,
    max_image_dimension_cube: u32 => max_image_dimension_cube,
    max_image_array_layers: u32 => max_image_array_layers,
    max_texel_buffer_elements: u32 => max_texel_buffer_elements,
    max_uniform_buffer_range: u32 => max_uniform_buffer_range,
    max_storage_buffer_range: u32 => max_storage_buffer_range,
    max_push_constants_size: u32 => max_push_constants_size,
    max_memory_allocation_count: u32 => max_memory_allocation_count,
    max_sampler_allocation_count: u32 => max_sampler_allocation_count,
    buffer_image_granularity: u64 => buffer_image_granularity,
    sparse_address_space_size: u64 => sparse_address_space_size,
    max_bound_descriptor_sets: u32 => max_bound_descriptor_sets,
    max_per_stage_descriptor_samplers: u32 => max_per_stage_descriptor_samplers,
    max_per_stage_descriptor_uniform_buffers: u32 => max_per_stage_descriptor_uniform_buffers,
    max_per_stage_descriptor_storage_buffers: u32 => max_per_stage_descriptor_storage_buffers,
    max_per_stage_descriptor_sampled_images: u32 => max_per_stage_descriptor_sampled_images,
    max_per_stage_descriptor_storage_images: u32 => max_per_stage_descriptor_storage_images,
    max_per_stage_descriptor_input_attachments: u32 => max_per_stage_descriptor_input_attachments,
    max_per_stage_resources: u32 => max_per_stage_resources,
    max_descriptor_set_samplers: u32 => max_descriptor_set_samplers,
    max_descriptor_set_uniform_buffers: u32 => max_descriptor_set_uniform_buffers,
    max_descriptor_set_uniform_buffers_dynamic: u32 => max_descriptor_set_uniform_buffers_dynamic,
    max_descriptor_set_storage_buffers: u32 => max_descriptor_set_storage_buffers,
    max_descriptor_set_storage_buffers_dynamic: u32 => max_descriptor_set_storage_buffers_dynamic,
    max_descriptor_set_sampled_images: u32 => max_descriptor_set_sampled_images,
    max_descriptor_set_storage_images: u32 => max_descriptor_set_storage_images,
    max_descriptor_set_input_attachments: u32 => max_descriptor_set_input_attachments,
    max_vertex_input_attributes: u32 => max_vertex_input_attributes,
    max_vertex_input_bindings: u32 => max_vertex_input_bindings,
    max_vertex_input_attribute_offset: u32 => max_vertex_input_attribute_offset,
    max_vertex_input_binding_stride: u32 => max_vertex_input_binding_stride,
    max_vertex_output_components: u32 => max_vertex_output_components,
    max_tessellation_generation_level: u32 => max_tessellation_generation_level,
    max_tessellation_patch_size: u32 => max_tessellation_patch_size,
    max_tessellation_control_per_vertex_input_components: u32 => max_tessellation_control_per_vertex_input_components,
    max_tessellation_control_per_vertex_output_components: u32 => max_tessellation_control_per_vertex_output_components,
    max_tessellation_control_per_patch_output_components: u32 => max_tessellation_control_per_patch_output_components,
    max_tessellation_control_total_output_components: u32 => max_tessellation_control_total_output_components,
    max_tessellation_evaluation_input_components: u32 => max_tessellation_evaluation_input_components,
    max_tessellation_evaluation_output_components: u32 => max_tessellation_evaluation_output_components,
    max_geometry_shader_invocations: u32 => max_geometry_shader_invocations,
    max_geometry_input_components: u32 => max_geometry_input_components,
    max_geometry_output_components: u32 => max_geometry_output_components,
    max_geometry_output_vertices: u32 => max_geometry_output_vertices,
    max_geometry_total_output_components: u32 => max_geometry_total_output_components,
    max_fragment_input_components: u32 => max_fragment_input_components,
    max_fragment_output_attachments: u32 => max_fragment_output_attachments,
    max_fragment_dual_src_attachments: u32 => max_fragment_dual_src_attachments,
    max_fragment_combined_output_resources: u32 => max_fragment_combined_output_resources,
    max_compute_shared_memory_size: u32 => max_compute_shared_memory_size,
    max_compute_work_group_count: [u32; 3] => max_compute_work_group_count,
    max_compute_work_group_invocations: u32 => max_compute_work_group_invocations,
    max_compute_work_group_size: [u32; 3] => max_compute_work_group_size,
    sub_pixel_precision_bits: u32 => sub_pixel_precision_bits,
    sub_texel_precision_bits: u32 => sub_texel_precision_bits,
    mipmap_precision_bits: u32 => mipmap_precision_bits,
    max_draw_indexed_index_value: u32 => max_draw_indexed_index_value,
    max_draw_indirect_count: u32 => max_draw_indirect_count,
    max_sampler_lod_bias: f32 => max_sampler_lod_bias,
    max_sampler_anisotropy: f32 => max_sampler_anisotropy,
    max_viewports: u32 => max_viewports,
    max_viewport_dimensions: [u32; 2] => max_viewport_dimensions,
    viewport_bounds_range: [f32; 2] => viewport_bounds_range,
    viewport_sub_pixel_bits: u32 => viewport_sub_pixel_bits,
    min_memory_map_alignment: usize => min_memory_map_alignment,
    min_texel_buffer_offset_alignment: u64 => min_texel_buffer_offset_alignment,
    min_uniform_buffer_offset_alignment: u64 => min_uniform_buffer_offset_alignment,
    min_storage_buffer_offset_alignment: u64 => min_storage_buffer_offset_alignment,
    min_texel_offset: i32 => min_texel_offset,
    max_texel_offset: u32 => max_texel_offset,
    min_texel_gather_offset: i32 => min_texel_gather_offset,
    max_texel_gather_offset: u32 => max_texel_gather_offset,
    min_interpolation_offset: f32 => min_interpolation_offset,
    max_interpolation_offset: f32 => max_interpolation_offset,
    sub_pixel_interpolation_offset_bits: u32 => sub_pixel_interpolation_offset_bits,
    max_framebuffer_width: u32 => max_framebuffer_width,
    max_framebuffer_height: u32 => max_framebuffer_height,
    max_framebuffer_layers: u32 => max_framebuffer_layers,
    framebuffer_color_sample_counts: SampleCounts => framebuffer_color_sample_counts,      // FIXME: SampleCountFlag
    framebuffer_depth_sample_counts: SampleCounts => framebuffer_depth_sample_counts,      // FIXME: SampleCountFlag
    framebuffer_stencil_sample_counts: SampleCounts => framebuffer_stencil_sample_counts,      // FIXME: SampleCountFlag
    framebuffer_no_attachments_sample_counts: SampleCounts => framebuffer_no_attachments_sample_counts,      // FIXME: SampleCountFlag
    max_color_attachments: u32 => max_color_attachments,
    sampled_image_color_sample_counts: SampleCounts => sampled_image_color_sample_counts,        // FIXME: SampleCountFlag
    sampled_image_integer_sample_counts: SampleCounts => sampled_image_integer_sample_counts,        // FIXME: SampleCountFlag
    sampled_image_depth_sample_counts: SampleCounts => sampled_image_depth_sample_counts,        // FIXME: SampleCountFlag
    sampled_image_stencil_sample_counts: SampleCounts => sampled_image_stencil_sample_counts,        // FIXME: SampleCountFlag
    storage_image_sample_counts: SampleCounts => storage_image_sample_counts,      // FIXME: SampleCountFlag
    max_sample_mask_words: u32 => max_sample_mask_words,
    timestamp_compute_and_graphics: u32 => timestamp_compute_and_graphics,        // TODO: these are booleans
    timestamp_period: f32 => timestamp_period,
    max_clip_distances: u32 => max_clip_distances,
    max_cull_distances: u32 => max_cull_distances,
    max_combined_clip_and_cull_distances: u32 => max_combined_clip_and_cull_distances,
    discrete_queue_priorities: u32 => discrete_queue_priorities,
    point_size_range: [f32; 2] => point_size_range,
    line_width_range: [f32; 2] => line_width_range,
    point_size_granularity: f32 => point_size_granularity,
    line_width_granularity: f32 => line_width_granularity,
    strict_lines: u32 => strict_lines,        // TODO: these are booleans
    standard_sample_locations: u32 => standard_sample_locations,        // TODO: these are booleans
    optimal_buffer_copy_offset_alignment: u64 => optimal_buffer_copy_offset_alignment,
    optimal_buffer_copy_row_pitch_alignment: u64 => optimal_buffer_copy_row_pitch_alignment,
    non_coherent_atom_size: u64 => non_coherent_atom_size,
}
