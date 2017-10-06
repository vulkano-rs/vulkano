// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use vk;

macro_rules! features {
    ($($name:ident => $vk:ident,)+) => (
        /// Represents all the features that are available on a physical device or enabled on
        /// a logical device.
        ///
        /// Note that the `robust_buffer_access` is guaranteed to be supported by all Vulkan
        /// implementations.
        ///
        /// # Example
        ///
        /// ```
        /// # let physical_device: vulkano::instance::PhysicalDevice = return;
        /// let minimal_features = vulkano::instance::Features {
        ///     geometry_shader: true,
        ///     .. vulkano::instance::Features::none()
        /// };
        ///
        /// let optimal_features = vulkano::instance::Features {
        ///     geometry_shader: true,
        ///     tessellation_shader: true,
        ///     .. vulkano::instance::Features::none()
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
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[allow(missing_docs)]
        pub struct Features {
            $(
                pub $name: bool,
            )+
        }

        impl Features {
            /// Builds a `Features` object with all values to false.
            pub fn none() -> Features {
                Features {
                    $(
                        $name: false,
                    )+
                }
            }

            /// Builds a `Features` object with all values to true.
            ///
            /// > **Note**: This function is used for testing purposes, and is probably useless in
            /// > a real code.
            pub fn all() -> Features {
                Features {
                    $(
                        $name: true,
                    )+
                }
            }

            /// Returns true if `self` is a superset of the parameter.
            ///
            /// That is, for each feature of the parameter that is true, the corresponding value
            /// in self is true as well.
            pub fn superset_of(&self, other: &Features) -> bool {
                $((self.$name == true || other.$name == false))&&+
            }

            /// Builds a `Features` that is the intersection of `self` and another `Features`
            /// object.
            ///
            /// The result's field will be true if it is also true in both `self` and `other`.
            pub fn intersection(&self, other: &Features) -> Features {
                Features {
                    $(
                        $name: self.$name && other.$name,
                    )+
                }
            }

            /// Builds a `Features` that is the difference of another `Features` object from `self`.
            ///
            /// The result's field will be true if it is true in `self` but not `other`.
            pub fn difference(&self, other: &Features) -> Features {
                Features {
                    $(
                        $name: self.$name && !other.$name,
                    )+
                }
            }

            pub(crate) fn from_vulkan_features(features: vk::PhysicalDeviceFeatures) -> Features {
                Features {
                    $(
                        $name: features.$vk != 0,
                    )+
                }
            }

            pub(crate) fn into_vulkan_features(self) -> vk::PhysicalDeviceFeatures {
                vk::PhysicalDeviceFeatures {
                    $(
                        $vk: if self.$name { vk::TRUE } else { vk::FALSE },
                    )+
                }
            }
        }
    )
}

features!{
    robust_buffer_access => robustBufferAccess,
    full_draw_index_uint32 => fullDrawIndexUint32,
    image_cube_array => imageCubeArray,
    independent_blend => independentBlend,
    geometry_shader => geometryShader,
    tessellation_shader => tessellationShader,
    sample_rate_shading => sampleRateShading,
    dual_src_blend => dualSrcBlend,
    logic_op => logicOp,
    multi_draw_indirect => multiDrawIndirect,
    draw_indirect_first_instance => drawIndirectFirstInstance,
    depth_clamp => depthClamp,
    depth_bias_clamp => depthBiasClamp,
    fill_mode_non_solid => fillModeNonSolid,
    depth_bounds => depthBounds,
    wide_lines => wideLines,
    large_points => largePoints,
    alpha_to_one => alphaToOne,
    multi_viewport => multiViewport,
    sampler_anisotropy => samplerAnisotropy,
    texture_compression_etc2 => textureCompressionETC2,
    texture_compression_astc_ldr => textureCompressionASTC_LDR,
    texture_compression_bc => textureCompressionBC,
    occlusion_query_precise => occlusionQueryPrecise,
    pipeline_statistics_query => pipelineStatisticsQuery,
    vertex_pipeline_stores_and_atomics => vertexPipelineStoresAndAtomics,
    fragment_stores_and_atomics => fragmentStoresAndAtomics,
    shader_tessellation_and_geometry_point_size => shaderTessellationAndGeometryPointSize,
    shader_image_gather_extended => shaderImageGatherExtended,
    shader_storage_image_extended_formats => shaderStorageImageExtendedFormats,
    shader_storage_image_multisample => shaderStorageImageMultisample,
    shader_storage_image_read_without_format => shaderStorageImageReadWithoutFormat,
    shader_storage_image_write_without_format => shaderStorageImageWriteWithoutFormat,
    shader_uniform_buffer_array_dynamic_indexing => shaderUniformBufferArrayDynamicIndexing,
    shader_sampled_image_array_dynamic_indexing => shaderSampledImageArrayDynamicIndexing,
    shader_storage_buffer_array_dynamic_indexing => shaderStorageBufferArrayDynamicIndexing,
    shader_storage_image_array_dynamic_indexing => shaderStorageImageArrayDynamicIndexing,
    shader_clip_distance => shaderClipDistance,
    shader_cull_distance => shaderCullDistance,
    shader_f3264 => shaderf3264,
    shader_int64 => shaderInt64,
    shader_int16 => shaderInt16,
    shader_resource_residency => shaderResourceResidency,
    shader_resource_min_lod => shaderResourceMinLod,
    sparse_binding => sparseBinding,
    sparse_residency_buffer => sparseResidencyBuffer,
    sparse_residency_image2d => sparseResidencyImage2D,
    sparse_residency_image3d => sparseResidencyImage3D,
    sparse_residency2_samples => sparseResidency2Samples,
    sparse_residency4_samples => sparseResidency4Samples,
    sparse_residency8_samples => sparseResidency8Samples,
    sparse_residency16_samples => sparseResidency16Samples,
    sparse_residency_aliased => sparseResidencyAliased,
    variable_multisample_rate => variableMultisampleRate,
    inherited_queries => inheritedQueries,
}
