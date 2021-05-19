// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::vk;
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
                    $(features_ffi.$ffi_struct.$($ffi_field).+ |= features.$member as vk::Bool32;)*
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
    {robust_buffer_access => vulkan_1_0.features.robustBufferAccess},
    {full_draw_index_uint32 => vulkan_1_0.features.fullDrawIndexUint32},
    {image_cube_array => vulkan_1_0.features.imageCubeArray},
    {independent_blend => vulkan_1_0.features.independentBlend},
    {geometry_shader => vulkan_1_0.features.geometryShader},
    {tessellation_shader => vulkan_1_0.features.tessellationShader},
    {sample_rate_shading => vulkan_1_0.features.sampleRateShading},
    {dual_src_blend => vulkan_1_0.features.dualSrcBlend},
    {logic_op => vulkan_1_0.features.logicOp},
    {multi_draw_indirect => vulkan_1_0.features.multiDrawIndirect},
    {draw_indirect_first_instance => vulkan_1_0.features.drawIndirectFirstInstance},
    {depth_clamp => vulkan_1_0.features.depthClamp},
    {depth_bias_clamp => vulkan_1_0.features.depthBiasClamp},
    {fill_mode_non_solid => vulkan_1_0.features.fillModeNonSolid},
    {depth_bounds => vulkan_1_0.features.depthBounds},
    {wide_lines => vulkan_1_0.features.wideLines},
    {large_points => vulkan_1_0.features.largePoints},
    {alpha_to_one => vulkan_1_0.features.alphaToOne},
    {multi_viewport => vulkan_1_0.features.multiViewport},
    {sampler_anisotropy => vulkan_1_0.features.samplerAnisotropy},
    {texture_compression_etc2 => vulkan_1_0.features.textureCompressionETC2},
    {texture_compression_astc_ldr => vulkan_1_0.features.textureCompressionASTC_LDR},
    {texture_compression_bc => vulkan_1_0.features.textureCompressionBC},
    {occlusion_query_precise => vulkan_1_0.features.occlusionQueryPrecise},
    {pipeline_statistics_query => vulkan_1_0.features.pipelineStatisticsQuery},
    {vertex_pipeline_stores_and_atomics => vulkan_1_0.features.vertexPipelineStoresAndAtomics},
    {fragment_stores_and_atomics => vulkan_1_0.features.fragmentStoresAndAtomics},
    {shader_tessellation_and_geometry_point_size => vulkan_1_0.features.shaderTessellationAndGeometryPointSize},
    {shader_image_gather_extended => vulkan_1_0.features.shaderImageGatherExtended},
    {shader_storage_image_extended_formats => vulkan_1_0.features.shaderStorageImageExtendedFormats},
    {shader_storage_image_multisample => vulkan_1_0.features.shaderStorageImageMultisample},
    {shader_storage_image_read_without_format => vulkan_1_0.features.shaderStorageImageReadWithoutFormat},
    {shader_storage_image_write_without_format => vulkan_1_0.features.shaderStorageImageWriteWithoutFormat},
    {shader_uniform_buffer_array_dynamic_indexing => vulkan_1_0.features.shaderUniformBufferArrayDynamicIndexing},
    {shader_sampled_image_array_dynamic_indexing => vulkan_1_0.features.shaderSampledImageArrayDynamicIndexing},
    {shader_storage_buffer_array_dynamic_indexing => vulkan_1_0.features.shaderStorageBufferArrayDynamicIndexing},
    {shader_storage_image_array_dynamic_indexing => vulkan_1_0.features.shaderStorageImageArrayDynamicIndexing},
    {shader_clip_distance => vulkan_1_0.features.shaderClipDistance},
    {shader_cull_distance => vulkan_1_0.features.shaderCullDistance},
    {shader_float64 => vulkan_1_0.features.shaderFloat64},
    {shader_int64 => vulkan_1_0.features.shaderInt64},
    {shader_int16 => vulkan_1_0.features.shaderInt16},
    {shader_resource_residency => vulkan_1_0.features.shaderResourceResidency},
    {shader_resource_min_lod => vulkan_1_0.features.shaderResourceMinLod},
    {sparse_binding => vulkan_1_0.features.sparseBinding},
    {sparse_residency_buffer => vulkan_1_0.features.sparseResidencyBuffer},
    {sparse_residency_image2d => vulkan_1_0.features.sparseResidencyImage2D},
    {sparse_residency_image3d => vulkan_1_0.features.sparseResidencyImage3D},
    {sparse_residency2_samples => vulkan_1_0.features.sparseResidency2Samples},
    {sparse_residency4_samples => vulkan_1_0.features.sparseResidency4Samples},
    {sparse_residency8_samples => vulkan_1_0.features.sparseResidency8Samples},
    {sparse_residency16_samples => vulkan_1_0.features.sparseResidency16Samples},
    {sparse_residency_aliased => vulkan_1_0.features.sparseResidencyAliased},
    {variable_multisample_rate => vulkan_1_0.features.variableMultisampleRate},
    {inherited_queries => vulkan_1_0.features.inheritedQueries},

    {storage_buffer_16bit => vulkan_1_1.storageBuffer16BitAccess | khr_16bit_storage.storageBuffer16BitAccess},
    {storage_uniform_16bit => vulkan_1_1.uniformAndStorageBuffer16BitAccess | khr_16bit_storage.uniformAndStorageBuffer16BitAccess},
    {storage_push_constant_16bit => vulkan_1_1.storagePushConstant16 | khr_16bit_storage.storagePushConstant16},
    {storage_input_output_16bit => vulkan_1_1.storageInputOutput16 | khr_16bit_storage.storageInputOutput16},
    {multiview => vulkan_1_1.multiview | khr_multiview.multiview},
    {multiview_geometry_shader => vulkan_1_1.multiviewGeometryShader | khr_multiview.multiviewGeometryShader},
    {multiview_tessellation_shader => vulkan_1_1.multiviewTessellationShader | khr_multiview.multiviewTessellationShader},
    {variable_pointers_storage_buffer => vulkan_1_1.variablePointersStorageBuffer | khr_variable_pointers.variablePointersStorageBuffer},
    {variable_pointers => vulkan_1_1.variablePointers | khr_variable_pointers.variablePointers},
    {protected_memory => vulkan_1_1.protectedMemory | khr_protected_memory.protectedMemory},
    {sampler_ycbcr_conversion => vulkan_1_1.samplerYcbcrConversion | khr_sampler_ycbcr_conversion.samplerYcbcrConversion},
    {shader_draw_parameters => vulkan_1_1.shaderDrawParameters | shader_draw_parameters.shaderDrawParameters},

    {sampler_mirror_clamp_to_edge => vulkan_1_2.samplerMirrorClampToEdge},
    {draw_indirect_count => vulkan_1_2.drawIndirectCount},
    {storage_buffer_8bit => vulkan_1_2.storageBuffer8BitAccess | khr_8bit_storage.storageBuffer8BitAccess},
    {storage_uniform_8bit => vulkan_1_2.uniformAndStorageBuffer8BitAccess | khr_8bit_storage.uniformAndStorageBuffer8BitAccess},
    {storage_push_constant_8bit => vulkan_1_2.storagePushConstant8 | khr_8bit_storage.storagePushConstant8},
    {shader_buffer_int64_atomics => vulkan_1_2.shaderBufferInt64Atomics | khr_shader_atomic_int64.shaderBufferInt64Atomics},
    {shader_shared_int64_atomics => vulkan_1_2.shaderSharedInt64Atomics | khr_shader_atomic_int64.shaderSharedInt64Atomics},
    {shader_float16 => vulkan_1_2.shaderFloat16 | khr_shader_float16_int8.shaderFloat16},
    {shader_int8 => vulkan_1_2.shaderInt8 | khr_shader_float16_int8.shaderInt8},
    {descriptor_indexing => vulkan_1_2.descriptorIndexing},
    {shader_input_attachment_array_dynamic_indexing => vulkan_1_2.shaderInputAttachmentArrayDynamicIndexing | ext_descriptor_indexing.shaderInputAttachmentArrayDynamicIndexing},
    {shader_uniform_texel_buffer_array_dynamic_indexing => vulkan_1_2.shaderUniformTexelBufferArrayDynamicIndexing | ext_descriptor_indexing.shaderUniformTexelBufferArrayDynamicIndexing},
    {shader_storage_texel_buffer_array_dynamic_indexing => vulkan_1_2.shaderStorageTexelBufferArrayDynamicIndexing | ext_descriptor_indexing.shaderStorageTexelBufferArrayDynamicIndexing},
    {shader_uniform_buffer_array_non_uniform_indexing => vulkan_1_2.shaderUniformBufferArrayNonUniformIndexing | ext_descriptor_indexing.shaderUniformBufferArrayNonUniformIndexing},
    {shader_sampled_image_array_non_uniform_indexing => vulkan_1_2.shaderSampledImageArrayNonUniformIndexing | ext_descriptor_indexing.shaderSampledImageArrayNonUniformIndexing},
    {shader_storage_buffer_array_non_uniform_indexing => vulkan_1_2.shaderStorageBufferArrayNonUniformIndexing | ext_descriptor_indexing.shaderStorageBufferArrayNonUniformIndexing},
    {shader_storage_image_array_non_uniform_indexing => vulkan_1_2.shaderStorageImageArrayNonUniformIndexing | ext_descriptor_indexing.shaderStorageImageArrayNonUniformIndexing},
    {shader_input_attachment_array_non_uniform_indexing => vulkan_1_2.shaderInputAttachmentArrayNonUniformIndexing | ext_descriptor_indexing.shaderInputAttachmentArrayNonUniformIndexing},
    {shader_uniform_texel_buffer_array_non_uniform_indexing => vulkan_1_2.shaderUniformTexelBufferArrayNonUniformIndexing | ext_descriptor_indexing.shaderUniformTexelBufferArrayNonUniformIndexing},
    {shader_storage_texel_buffer_array_non_uniform_indexing => vulkan_1_2.shaderStorageTexelBufferArrayNonUniformIndexing | ext_descriptor_indexing.shaderStorageTexelBufferArrayNonUniformIndexing},
    {descriptor_binding_uniform_buffer_update_after_bind => vulkan_1_2.descriptorBindingUniformBufferUpdateAfterBind | ext_descriptor_indexing.descriptorBindingUniformBufferUpdateAfterBind},
    {descriptor_binding_sampled_image_update_after_bind => vulkan_1_2.descriptorBindingSampledImageUpdateAfterBind | ext_descriptor_indexing.descriptorBindingSampledImageUpdateAfterBind},
    {descriptor_binding_storage_image_update_after_bind => vulkan_1_2.descriptorBindingStorageImageUpdateAfterBind | ext_descriptor_indexing.descriptorBindingStorageImageUpdateAfterBind},
    {descriptor_binding_storage_buffer_update_after_bind => vulkan_1_2.descriptorBindingStorageBufferUpdateAfterBind | ext_descriptor_indexing.descriptorBindingStorageBufferUpdateAfterBind},
    {descriptor_binding_uniform_texel_buffer_update_after_bind => vulkan_1_2.descriptorBindingUniformTexelBufferUpdateAfterBind | ext_descriptor_indexing.descriptorBindingUniformTexelBufferUpdateAfterBind},
    {descriptor_binding_storage_texel_buffer_update_after_bind => vulkan_1_2.descriptorBindingStorageTexelBufferUpdateAfterBind | ext_descriptor_indexing.descriptorBindingStorageTexelBufferUpdateAfterBind},
    {descriptor_binding_update_unused_while_pending => vulkan_1_2.descriptorBindingUpdateUnusedWhilePending | ext_descriptor_indexing.descriptorBindingUpdateUnusedWhilePending},
    {descriptor_binding_partially_bound => vulkan_1_2.descriptorBindingPartiallyBound | ext_descriptor_indexing.descriptorBindingPartiallyBound},
    {descriptor_binding_variable_descriptor_count => vulkan_1_2.descriptorBindingVariableDescriptorCount | ext_descriptor_indexing.descriptorBindingVariableDescriptorCount},
    {runtime_descriptor_array => vulkan_1_2.runtimeDescriptorArray | ext_descriptor_indexing.runtimeDescriptorArray},
    {sampler_filter_minmax => vulkan_1_2.samplerFilterMinmax},
    {scalar_block_layout => vulkan_1_2.scalarBlockLayout | ext_scalar_block_layout.scalarBlockLayout},
    {imageless_framebuffer => vulkan_1_2.imagelessFramebuffer | khr_imageless_framebuffer.imagelessFramebuffer},
    {uniform_buffer_standard_layout => vulkan_1_2.uniformBufferStandardLayout | khr_uniform_buffer_standard_layout.uniformBufferStandardLayout},
    {shader_subgroup_extended_types => vulkan_1_2.shaderSubgroupExtendedTypes | khr_shader_subgroup_extended_types.shaderSubgroupExtendedTypes},
    {separate_depth_stencil_layouts => vulkan_1_2.separateDepthStencilLayouts | khr_separate_depth_stencil_layouts.separateDepthStencilLayouts},
    {host_query_reset => vulkan_1_2.hostQueryReset | ext_host_query_reset.hostQueryReset},
    {timeline_semaphore => vulkan_1_2.timelineSemaphore | khr_timeline_semaphore.timelineSemaphore},
    {buffer_device_address => vulkan_1_2.bufferDeviceAddress | khr_buffer_device_address.bufferDeviceAddress},
    {buffer_device_address_capture_replay => vulkan_1_2.bufferDeviceAddressCaptureReplay | khr_buffer_device_address.bufferDeviceAddressCaptureReplay},
    {buffer_device_address_multi_device => vulkan_1_2.bufferDeviceAddressMultiDevice | khr_buffer_device_address.bufferDeviceAddressMultiDevice},
    {vulkan_memory_model => vulkan_1_2.vulkanMemoryModel | khr_vulkan_memory_model.vulkanMemoryModel},
    {vulkan_memory_model_device_scope => vulkan_1_2.vulkanMemoryModelDeviceScope | khr_vulkan_memory_model.vulkanMemoryModelDeviceScope},
    {vulkan_memory_model_availability_visibility_chains => vulkan_1_2.vulkanMemoryModelAvailabilityVisibilityChains | khr_vulkan_memory_model.vulkanMemoryModelAvailabilityVisibilityChains},
    {shader_output_viewport_index => vulkan_1_2.shaderOutputViewportIndex},
    {shader_output_layer => vulkan_1_2.shaderOutputLayer},
    {subgroup_broadcast_dynamic_id => vulkan_1_2.subgroupBroadcastDynamicId},

    {ext_buffer_device_address => ext_buffer_address.bufferDeviceAddress},
    {ext_buffer_device_address_capture_replay => ext_buffer_address.bufferDeviceAddressCaptureReplay},
    {ext_buffer_device_address_multi_device => ext_buffer_address.bufferDeviceAddressMultiDevice},
}

#[derive(Default)]
pub(crate) struct FeaturesFfi {
    _pinned: PhantomPinned,

    vulkan_1_0: vk::PhysicalDeviceFeatures2KHR,
    vulkan_1_1: vk::PhysicalDeviceVulkan11Features,
    vulkan_1_2: vk::PhysicalDeviceVulkan12Features,

    shader_draw_parameters: vk::PhysicalDeviceShaderDrawParametersFeatures,

    khr_16bit_storage: vk::PhysicalDevice16BitStorageFeaturesKHR,
    khr_8bit_storage: vk::PhysicalDevice8BitStorageFeaturesKHR,
    khr_buffer_device_address: vk::PhysicalDeviceBufferDeviceAddressFeaturesKHR,
    khr_imageless_framebuffer: vk::PhysicalDeviceImagelessFramebufferFeaturesKHR,
    khr_multiview: vk::PhysicalDeviceMultiviewFeaturesKHR,
    khr_protected_memory: vk::PhysicalDeviceProtectedMemoryFeaturesKHR,
    khr_sampler_ycbcr_conversion: vk::PhysicalDeviceSamplerYcbcrConversionFeaturesKHR,
    khr_separate_depth_stencil_layouts: vk::PhysicalDeviceSeparateDepthStencilLayoutsFeaturesKHR,
    khr_shader_atomic_int64: vk::PhysicalDeviceShaderAtomicInt64FeaturesKHR,
    khr_shader_float16_int8: vk::PhysicalDeviceShaderFloat16Int8FeaturesKHR,
    khr_shader_subgroup_extended_types: vk::PhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR,
    khr_timeline_semaphore: vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR,
    khr_uniform_buffer_standard_layout: vk::PhysicalDeviceUniformBufferStandardLayoutFeaturesKHR,
    khr_variable_pointers: vk::PhysicalDeviceVariablePointersFeaturesKHR,
    khr_vulkan_memory_model: vk::PhysicalDeviceVulkanMemoryModelFeaturesKHR,

    ext_buffer_address: vk::PhysicalDeviceBufferAddressFeaturesEXT,
    ext_descriptor_indexing: vk::PhysicalDeviceDescriptorIndexingFeaturesEXT,
    ext_host_query_reset: vk::PhysicalDeviceHostQueryResetFeaturesEXT,
    ext_scalar_block_layout: vk::PhysicalDeviceScalarBlockLayoutFeaturesEXT,
}

macro_rules! push_struct {
    ($self:ident, $struct:ident) => {
        $self.$struct.pNext = $self.vulkan_1_0.pNext;
        $self.vulkan_1_0.pNext = addr_of_mut!($self.$struct) as _;
    };
}

impl FeaturesFfi {
    pub(crate) fn make_chain(&mut self, api_version: Version) {
        if api_version >= Version::major_minor(1, 2) {
            push_struct!(self, vulkan_1_1);
            push_struct!(self, vulkan_1_2);
        } else {
            if api_version >= Version::major_minor(1, 1) {
                push_struct!(self, shader_draw_parameters);
            }

            push_struct!(self, khr_16bit_storage);
            push_struct!(self, khr_8bit_storage);
            push_struct!(self, khr_buffer_device_address);
            push_struct!(self, khr_imageless_framebuffer);
            push_struct!(self, khr_multiview);
            push_struct!(self, khr_protected_memory);
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

    pub(crate) fn head_as_ref(&self) -> &vk::PhysicalDeviceFeatures2KHR {
        &self.vulkan_1_0
    }

    pub(crate) fn head_as_mut(&mut self) -> &mut vk::PhysicalDeviceFeatures2KHR {
        &mut self.vulkan_1_0
    }
}
