// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::Version;
use std::marker::PhantomPinned;
use std::ptr::addr_of_mut;

macro_rules! features {
    {
        $({ $member:ident => $($ffi_struct:ident.$($ffi_field:ident).+)|+ },)*
    } => {
        /// Represents all the features that are available on a physical device or enabled on
        /// a logical device.
        ///
        /// Note that the `robust_buffer_access` is guaranteed to be supported by all Vulkan
        /// implementations.
        ///
        /// # Example
        ///
        /// ```
        /// use vulkano::device::Features;
        /// # let physical_device: vulkano::instance::PhysicalDevice = return;
        /// let minimal_features = Features {
        ///     geometry_shader: true,
        ///     .. Features::none()
        /// };
        ///
        /// let optimal_features = vulkano::device::Features {
        ///     geometry_shader: true,
        ///     tessellation_shader: true,
        ///     .. Features::none()
        /// };
        ///
        /// if !physical_device.supported_features().superset_of(&minimal_features) {
        ///     panic!("The physical device is not good enough for this application.");
        /// }
        ///
        /// assert!(optimal_features.superset_of(&minimal_features));
        /// let features_to_request = optimal_features.intersection(physical_device.supported_features());
        /// ```
        ///
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
        #[allow(missing_docs)]
        pub struct Features {
            $(pub $member: bool,)*
        }

        impl Features {
            /// Builds a `Features` object with all values to false.
            pub fn none() -> Features {
                Features {
                    $($member: false,)*
                }
            }

            /// Builds a `Features` object with all values to true.
            ///
            /// > **Note**: This function is used for testing purposes, and is probably useless in
            /// > a real code.
            pub fn all() -> Features {
                Features {
                    $($member: true,)*
                }
            }

            /// Returns true if `self` is a superset of the parameter.
            ///
            /// That is, for each feature of the parameter that is true, the corresponding value
            /// in self is true as well.
            pub fn superset_of(&self, other: &Features) -> bool {
                $((self.$member == true || other.$member == false))&&+
            }

            /// Builds a `Features` that is the intersection of `self` and another `Features`
            /// object.
            ///
            /// The result's field will be true if it is also true in both `self` and `other`.
            pub fn intersection(&self, other: &Features) -> Features {
                Features {
                    $($member: self.$member && other.$member,)*
                }
            }

            /// Builds a `Features` that is the difference of another `Features` object from `self`.
            ///
            /// The result's field will be true if it is true in `self` but not `other`.
            pub fn difference(&self, other: &Features) -> Features {
                Features {
                    $($member: self.$member && !other.$member,)*
                }
            }
        }

        impl From<&Features> for FeaturesFfi {
            #[inline(always)]
            fn from(features: &Features) -> Self {
                let mut features_ffi = FeaturesFfi::default();
                $(
                    $(features_ffi.$ffi_struct.$($ffi_field).+ |= features.$member as ash::vk::Bool32;)*
                )+
                features_ffi
            }
        }

        impl From<&FeaturesFfi> for Features {
            #[inline(always)]
            fn from(features_ffi: &FeaturesFfi) -> Self {
                Features {
                    $($member: $(features_ffi.$ffi_struct.$($ffi_field).+ != 0)||+,)*
                }
            }
        }
    };
}

features! {
    // Vulkan 1.0
    {robust_buffer_access => vulkan_1_0.features.robust_buffer_access},
    {full_draw_index_uint32 => vulkan_1_0.features.full_draw_index_uint32},
    {image_cube_array => vulkan_1_0.features.image_cube_array},
    {independent_blend => vulkan_1_0.features.independent_blend},
    {geometry_shader => vulkan_1_0.features.geometry_shader},
    {tessellation_shader => vulkan_1_0.features.tessellation_shader},
    {sample_rate_shading => vulkan_1_0.features.sample_rate_shading},
    {dual_src_blend => vulkan_1_0.features.dual_src_blend},
    {logic_op => vulkan_1_0.features.logic_op},
    {multi_draw_indirect => vulkan_1_0.features.multi_draw_indirect},
    {draw_indirect_first_instance => vulkan_1_0.features.draw_indirect_first_instance},
    {depth_clamp => vulkan_1_0.features.depth_clamp},
    {depth_bias_clamp => vulkan_1_0.features.depth_bias_clamp},
    {fill_mode_non_solid => vulkan_1_0.features.fill_mode_non_solid},
    {depth_bounds => vulkan_1_0.features.depth_bounds},
    {wide_lines => vulkan_1_0.features.wide_lines},
    {large_points => vulkan_1_0.features.large_points},
    {alpha_to_one => vulkan_1_0.features.alpha_to_one},
    {multi_viewport => vulkan_1_0.features.multi_viewport},
    {sampler_anisotropy => vulkan_1_0.features.sampler_anisotropy},
    {texture_compression_etc2 => vulkan_1_0.features.texture_compression_etc2},
    {texture_compression_astc_ldr => vulkan_1_0.features.texture_compression_astc_ldr},
    {texture_compression_bc => vulkan_1_0.features.texture_compression_bc},
    {occlusion_query_precise => vulkan_1_0.features.occlusion_query_precise},
    {pipeline_statistics_query => vulkan_1_0.features.pipeline_statistics_query},
    {vertex_pipeline_stores_and_atomics => vulkan_1_0.features.vertex_pipeline_stores_and_atomics},
    {fragment_stores_and_atomics => vulkan_1_0.features.fragment_stores_and_atomics},
    {shader_tessellation_and_geometry_point_size => vulkan_1_0.features.shader_tessellation_and_geometry_point_size},
    {shader_image_gather_extended => vulkan_1_0.features.shader_image_gather_extended},
    {shader_storage_image_extended_formats => vulkan_1_0.features.shader_storage_image_extended_formats},
    {shader_storage_image_multisample => vulkan_1_0.features.shader_storage_image_multisample},
    {shader_storage_image_read_without_format => vulkan_1_0.features.shader_storage_image_read_without_format},
    {shader_storage_image_write_without_format => vulkan_1_0.features.shader_storage_image_write_without_format},
    {shader_uniform_buffer_array_dynamic_indexing => vulkan_1_0.features.shader_uniform_buffer_array_dynamic_indexing},
    {shader_sampled_image_array_dynamic_indexing => vulkan_1_0.features.shader_sampled_image_array_dynamic_indexing},
    {shader_storage_buffer_array_dynamic_indexing => vulkan_1_0.features.shader_storage_buffer_array_dynamic_indexing},
    {shader_storage_image_array_dynamic_indexing => vulkan_1_0.features.shader_storage_image_array_dynamic_indexing},
    {shader_clip_distance => vulkan_1_0.features.shader_clip_distance},
    {shader_cull_distance => vulkan_1_0.features.shader_cull_distance},
    {shader_float64 => vulkan_1_0.features.shader_float64},
    {shader_int64 => vulkan_1_0.features.shader_int64},
    {shader_int16 => vulkan_1_0.features.shader_int16},
    {shader_resource_residency => vulkan_1_0.features.shader_resource_residency},
    {shader_resource_min_lod => vulkan_1_0.features.shader_resource_min_lod},
    {sparse_binding => vulkan_1_0.features.sparse_binding},
    {sparse_residency_buffer => vulkan_1_0.features.sparse_residency_buffer},
    {sparse_residency_image2d => vulkan_1_0.features.sparse_residency_image2_d},
    {sparse_residency_image3d => vulkan_1_0.features.sparse_residency_image3_d},
    {sparse_residency2_samples => vulkan_1_0.features.sparse_residency2_samples},
    {sparse_residency4_samples => vulkan_1_0.features.sparse_residency4_samples},
    {sparse_residency8_samples => vulkan_1_0.features.sparse_residency8_samples},
    {sparse_residency16_samples => vulkan_1_0.features.sparse_residency16_samples},
    {sparse_residency_aliased => vulkan_1_0.features.sparse_residency_aliased},
    {variable_multisample_rate => vulkan_1_0.features.variable_multisample_rate},
    {inherited_queries => vulkan_1_0.features.inherited_queries},

    // Vulkan 1.1
    {storage_buffer_16bit => vulkan_1_1.storage_buffer16_bit_access | khr_16bit_storage.storage_buffer16_bit_access},
    {storage_uniform_16bit => vulkan_1_1.uniform_and_storage_buffer16_bit_access | khr_16bit_storage.uniform_and_storage_buffer16_bit_access},
    {storage_push_constant_16bit => vulkan_1_1.storage_push_constant16 | khr_16bit_storage.storage_push_constant16},
    {storage_input_output_16bit => vulkan_1_1.storage_input_output16 | khr_16bit_storage.storage_input_output16},
    {multiview => vulkan_1_1.multiview | khr_multiview.multiview},
    {multiview_geometry_shader => vulkan_1_1.multiview_geometry_shader | khr_multiview.multiview_geometry_shader},
    {multiview_tessellation_shader => vulkan_1_1.multiview_tessellation_shader | khr_multiview.multiview_tessellation_shader},
    {variable_pointers_storage_buffer => vulkan_1_1.variable_pointers_storage_buffer | khr_variable_pointers.variable_pointers_storage_buffer},
    {variable_pointers => vulkan_1_1.variable_pointers | khr_variable_pointers.variable_pointers},
    {protected_memory => vulkan_1_1.protected_memory | protected_memory.protected_memory},
    {sampler_ycbcr_conversion => vulkan_1_1.sampler_ycbcr_conversion | khr_sampler_ycbcr_conversion.sampler_ycbcr_conversion},
    {shader_draw_parameters => vulkan_1_1.shader_draw_parameters | shader_draw_parameters.shader_draw_parameters},

    // Vulkan 1.2
    {sampler_mirror_clamp_to_edge => vulkan_1_2.sampler_mirror_clamp_to_edge},
    {draw_indirect_count => vulkan_1_2.draw_indirect_count},
    {storage_buffer_8bit => vulkan_1_2.storage_buffer8_bit_access | khr_8bit_storage.storage_buffer8_bit_access},
    {storage_uniform_8bit => vulkan_1_2.uniform_and_storage_buffer8_bit_access | khr_8bit_storage.uniform_and_storage_buffer8_bit_access},
    {storage_push_constant_8bit => vulkan_1_2.storage_push_constant8 | khr_8bit_storage.storage_push_constant8},
    {shader_buffer_int64_atomics => vulkan_1_2.shader_buffer_int64_atomics | khr_shader_atomic_int64.shader_buffer_int64_atomics},
    {shader_shared_int64_atomics => vulkan_1_2.shader_shared_int64_atomics | khr_shader_atomic_int64.shader_shared_int64_atomics},
    {shader_float16 => vulkan_1_2.shader_float16 | khr_shader_float16_int8.shader_float16},
    {shader_int8 => vulkan_1_2.shader_int8 | khr_shader_float16_int8.shader_int8},
    {descriptor_indexing => vulkan_1_2.descriptor_indexing},
    {shader_input_attachment_array_dynamic_indexing => vulkan_1_2.shader_input_attachment_array_dynamic_indexing | ext_descriptor_indexing.shader_input_attachment_array_dynamic_indexing},
    {shader_uniform_texel_buffer_array_dynamic_indexing => vulkan_1_2.shader_uniform_texel_buffer_array_dynamic_indexing | ext_descriptor_indexing.shader_uniform_texel_buffer_array_dynamic_indexing},
    {shader_storage_texel_buffer_array_dynamic_indexing => vulkan_1_2.shader_storage_texel_buffer_array_dynamic_indexing | ext_descriptor_indexing.shader_storage_texel_buffer_array_dynamic_indexing},
    {shader_uniform_buffer_array_non_uniform_indexing => vulkan_1_2.shader_uniform_buffer_array_non_uniform_indexing | ext_descriptor_indexing.shader_uniform_buffer_array_non_uniform_indexing},
    {shader_sampled_image_array_non_uniform_indexing => vulkan_1_2.shader_sampled_image_array_non_uniform_indexing | ext_descriptor_indexing.shader_sampled_image_array_non_uniform_indexing},
    {shader_storage_buffer_array_non_uniform_indexing => vulkan_1_2.shader_storage_buffer_array_non_uniform_indexing | ext_descriptor_indexing.shader_storage_buffer_array_non_uniform_indexing},
    {shader_storage_image_array_non_uniform_indexing => vulkan_1_2.shader_storage_image_array_non_uniform_indexing | ext_descriptor_indexing.shader_storage_image_array_non_uniform_indexing},
    {shader_input_attachment_array_non_uniform_indexing => vulkan_1_2.shader_input_attachment_array_non_uniform_indexing | ext_descriptor_indexing.shader_input_attachment_array_non_uniform_indexing},
    {shader_uniform_texel_buffer_array_non_uniform_indexing => vulkan_1_2.shader_uniform_texel_buffer_array_non_uniform_indexing | ext_descriptor_indexing.shader_uniform_texel_buffer_array_non_uniform_indexing},
    {shader_storage_texel_buffer_array_non_uniform_indexing => vulkan_1_2.shader_storage_texel_buffer_array_non_uniform_indexing | ext_descriptor_indexing.shader_storage_texel_buffer_array_non_uniform_indexing},
    {descriptor_binding_uniform_buffer_update_after_bind => vulkan_1_2.descriptor_binding_uniform_buffer_update_after_bind | ext_descriptor_indexing.descriptor_binding_uniform_buffer_update_after_bind},
    {descriptor_binding_sampled_image_update_after_bind => vulkan_1_2.descriptor_binding_sampled_image_update_after_bind | ext_descriptor_indexing.descriptor_binding_sampled_image_update_after_bind},
    {descriptor_binding_storage_image_update_after_bind => vulkan_1_2.descriptor_binding_storage_image_update_after_bind | ext_descriptor_indexing.descriptor_binding_storage_image_update_after_bind},
    {descriptor_binding_storage_buffer_update_after_bind => vulkan_1_2.descriptor_binding_storage_buffer_update_after_bind | ext_descriptor_indexing.descriptor_binding_storage_buffer_update_after_bind},
    {descriptor_binding_uniform_texel_buffer_update_after_bind => vulkan_1_2.descriptor_binding_uniform_texel_buffer_update_after_bind | ext_descriptor_indexing.descriptor_binding_uniform_texel_buffer_update_after_bind},
    {descriptor_binding_storage_texel_buffer_update_after_bind => vulkan_1_2.descriptor_binding_storage_texel_buffer_update_after_bind | ext_descriptor_indexing.descriptor_binding_storage_texel_buffer_update_after_bind},
    {descriptor_binding_update_unused_while_pending => vulkan_1_2.descriptor_binding_update_unused_while_pending | ext_descriptor_indexing.descriptor_binding_update_unused_while_pending},
    {descriptor_binding_partially_bound => vulkan_1_2.descriptor_binding_partially_bound | ext_descriptor_indexing.descriptor_binding_partially_bound},
    {descriptor_binding_variable_descriptor_count => vulkan_1_2.descriptor_binding_variable_descriptor_count | ext_descriptor_indexing.descriptor_binding_variable_descriptor_count},
    {runtime_descriptor_array => vulkan_1_2.runtime_descriptor_array | ext_descriptor_indexing.runtime_descriptor_array},
    {sampler_filter_minmax => vulkan_1_2.sampler_filter_minmax},
    {scalar_block_layout => vulkan_1_2.scalar_block_layout | ext_scalar_block_layout.scalar_block_layout},
    {imageless_framebuffer => vulkan_1_2.imageless_framebuffer | khr_imageless_framebuffer.imageless_framebuffer},
    {uniform_buffer_standard_layout => vulkan_1_2.uniform_buffer_standard_layout | khr_uniform_buffer_standard_layout.uniform_buffer_standard_layout},
    {shader_subgroup_extended_types => vulkan_1_2.shader_subgroup_extended_types | khr_shader_subgroup_extended_types.shader_subgroup_extended_types},
    {separate_depth_stencil_layouts => vulkan_1_2.separate_depth_stencil_layouts | khr_separate_depth_stencil_layouts.separate_depth_stencil_layouts},
    {host_query_reset => vulkan_1_2.host_query_reset | ext_host_query_reset.host_query_reset},
    {timeline_semaphore => vulkan_1_2.timeline_semaphore | khr_timeline_semaphore.timeline_semaphore},
    {buffer_device_address => vulkan_1_2.buffer_device_address | khr_buffer_device_address.buffer_device_address},
    {buffer_device_address_capture_replay => vulkan_1_2.buffer_device_address_capture_replay | khr_buffer_device_address.buffer_device_address_capture_replay},
    {buffer_device_address_multi_device => vulkan_1_2.buffer_device_address_multi_device | khr_buffer_device_address.buffer_device_address_multi_device},
    {vulkan_memory_model => vulkan_1_2.vulkan_memory_model | khr_vulkan_memory_model.vulkan_memory_model},
    {vulkan_memory_model_device_scope => vulkan_1_2.vulkan_memory_model_device_scope | khr_vulkan_memory_model.vulkan_memory_model_device_scope},
    {vulkan_memory_model_availability_visibility_chains => vulkan_1_2.vulkan_memory_model_availability_visibility_chains | khr_vulkan_memory_model.vulkan_memory_model_availability_visibility_chains},
    {shader_output_viewport_index => vulkan_1_2.shader_output_viewport_index},
    {shader_output_layer => vulkan_1_2.shader_output_layer},
    {subgroup_broadcast_dynamic_id => vulkan_1_2.subgroup_broadcast_dynamic_id},

    // Extensions
    {ext_buffer_device_address => ext_buffer_address.buffer_device_address},
    {ext_buffer_device_address_capture_replay => ext_buffer_address.buffer_device_address_capture_replay},
    {ext_buffer_device_address_multi_device => ext_buffer_address.buffer_device_address_multi_device},
}

#[derive(Default)]
pub(crate) struct FeaturesFfi {
    _pinned: PhantomPinned,

    vulkan_1_0: ash::vk::PhysicalDeviceFeatures2KHR,
    vulkan_1_1: ash::vk::PhysicalDeviceVulkan11Features,
    vulkan_1_2: ash::vk::PhysicalDeviceVulkan12Features,

    protected_memory: ash::vk::PhysicalDeviceProtectedMemoryFeatures,
    shader_draw_parameters: ash::vk::PhysicalDeviceShaderDrawParametersFeatures,

    khr_16bit_storage: ash::vk::PhysicalDevice16BitStorageFeaturesKHR,
    khr_8bit_storage: ash::vk::PhysicalDevice8BitStorageFeaturesKHR,
    khr_buffer_device_address: ash::vk::PhysicalDeviceBufferDeviceAddressFeaturesKHR,
    khr_imageless_framebuffer: ash::vk::PhysicalDeviceImagelessFramebufferFeaturesKHR,
    khr_multiview: ash::vk::PhysicalDeviceMultiviewFeaturesKHR,
    khr_sampler_ycbcr_conversion: ash::vk::PhysicalDeviceSamplerYcbcrConversionFeaturesKHR,
    khr_separate_depth_stencil_layouts:
        ash::vk::PhysicalDeviceSeparateDepthStencilLayoutsFeaturesKHR,
    khr_shader_atomic_int64: ash::vk::PhysicalDeviceShaderAtomicInt64FeaturesKHR,
    khr_shader_float16_int8: ash::vk::PhysicalDeviceShaderFloat16Int8FeaturesKHR,
    khr_shader_subgroup_extended_types:
        ash::vk::PhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR,
    khr_timeline_semaphore: ash::vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR,
    khr_uniform_buffer_standard_layout:
        ash::vk::PhysicalDeviceUniformBufferStandardLayoutFeaturesKHR,
    khr_variable_pointers: ash::vk::PhysicalDeviceVariablePointersFeaturesKHR,
    khr_vulkan_memory_model: ash::vk::PhysicalDeviceVulkanMemoryModelFeaturesKHR,

    ext_buffer_address: ash::vk::PhysicalDeviceBufferAddressFeaturesEXT,
    ext_descriptor_indexing: ash::vk::PhysicalDeviceDescriptorIndexingFeaturesEXT,
    ext_host_query_reset: ash::vk::PhysicalDeviceHostQueryResetFeaturesEXT,
    ext_scalar_block_layout: ash::vk::PhysicalDeviceScalarBlockLayoutFeaturesEXT,
}

macro_rules! push_struct {
    ($self:ident, $struct:ident) => {
        $self.$struct.p_next = $self.vulkan_1_0.p_next;
        $self.vulkan_1_0.p_next = addr_of_mut!($self.$struct) as _;
    };
}

impl FeaturesFfi {
    pub(crate) fn make_chain(&mut self, api_version: Version) {
        if api_version >= Version::V1_2 {
            push_struct!(self, vulkan_1_1);
            push_struct!(self, vulkan_1_2);
        } else {
            if api_version >= Version::V1_1 {
                push_struct!(self, protected_memory);
                push_struct!(self, shader_draw_parameters);
            }

            push_struct!(self, khr_16bit_storage);
            push_struct!(self, khr_8bit_storage);
            push_struct!(self, khr_buffer_device_address);
            push_struct!(self, khr_imageless_framebuffer);
            push_struct!(self, khr_multiview);
            push_struct!(self, khr_sampler_ycbcr_conversion);
            push_struct!(self, khr_separate_depth_stencil_layouts);
            push_struct!(self, khr_shader_atomic_int64);
            push_struct!(self, khr_shader_float16_int8);
            push_struct!(self, khr_shader_subgroup_extended_types);
            push_struct!(self, khr_timeline_semaphore);
            push_struct!(self, khr_uniform_buffer_standard_layout);
            push_struct!(self, khr_variable_pointers);
            push_struct!(self, khr_vulkan_memory_model);
            push_struct!(self, ext_descriptor_indexing);
            push_struct!(self, ext_host_query_reset);
            push_struct!(self, ext_scalar_block_layout);
        }

        push_struct!(self, ext_buffer_address);
    }

    pub(crate) fn head_as_ref(&self) -> &ash::vk::PhysicalDeviceFeatures2KHR {
        &self.vulkan_1_0
    }

    pub(crate) fn head_as_mut(&mut self) -> &mut ash::vk::PhysicalDeviceFeatures2KHR {
        &mut self.vulkan_1_0
    }
}
