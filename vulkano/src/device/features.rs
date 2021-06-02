// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::device::DeviceExtensions;
use crate::Version;
use std::error;
use std::fmt;

macro_rules! features {
    {
        $($member:ident => {
			ffi_name: $ffi_field:ident,
            ffi_members: [$($ffi_struct:ident $(.$ffi_struct_field:ident)?),+],
            requires_features: [$($requires_feature:ident),*],
            conflicts_features: [$($conflicts_feature:ident),*],
            required_by_extensions: [$($required_by_extension:ident),*],
        },)*
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
            /// Checks enabled features against the device version, device extensions and each other.
            pub(super) fn check_requirements(&self, supported: &Features, api_version: Version, extensions: &DeviceExtensions) -> Result<(), FeatureRestrictionError> {
                $(
                    if self.$member {
                        if !supported.$member {
                            return Err(FeatureRestrictionError {
                                feature: stringify!($member),
                                restriction: FeatureRestriction::NotSupported,
                            });
                        }

                        $(
                            if !self.$requires_feature {
                                return Err(FeatureRestrictionError {
                                    feature: stringify!($member),
                                    restriction: FeatureRestriction::RequiresFeature(stringify!($requires_feature)),
                                });
                            }
                        )*

                        $(
                            if self.$conflicts_feature {
                                return Err(FeatureRestrictionError {
                                    feature: stringify!($member),
                                    restriction: FeatureRestriction::ConflictsFeature(stringify!($conflicts_feature)),
                                });
                            }
                        )*
                    } else {
                        $(
                            if extensions.$required_by_extension {
                                return Err(FeatureRestrictionError {
                                    feature: stringify!($member),
                                    restriction: FeatureRestriction::RequiredByExtension(stringify!($required_by_extension)),
                                });
                            }
                        )*
                    }
                )*
                Ok(())
            }

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

        impl FeaturesFfi {
            pub(crate) fn write(&mut self, features: &Features) {
                $(
                    std::array::IntoIter::new([
                        $(self.$ffi_struct.as_mut().map(|s| &mut s$(.$ffi_struct_field)?.$ffi_field)),+
                    ]).flatten().next().map(|f| *f = features.$member as ash::vk::Bool32);
                )*
            }
        }

        impl From<&FeaturesFfi> for Features {
            fn from(features_ffi: &FeaturesFfi) -> Self {
                Features {
                    $(
                        $member: std::array::IntoIter::new([
                            $(features_ffi.$ffi_struct.map(|s| s$(.$ffi_struct_field)?.$ffi_field)),+
                        ]).flatten().next().unwrap_or(0) != 0,
                    )*
                }
            }
        }
    };
}

/// An error that can happen when enabling a feature on a device.
#[derive(Clone, Copy, Debug)]
pub struct FeatureRestrictionError {
    /// The feature in question.
    pub feature: &'static str,
    /// The restriction that was not met.
    pub restriction: FeatureRestriction,
}

impl error::Error for FeatureRestrictionError {}

impl fmt::Display for FeatureRestrictionError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "a restriction for the feature {} was not met: {}",
            self.feature, self.restriction,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub enum FeatureRestriction {
    /// Not supported by the physical device.
    NotSupported,
    /// Requires a feature to be enabled.
    RequiresFeature(&'static str),
    /// Requires a feature to be disabled.
    ConflictsFeature(&'static str),
    /// An extension requires this feature to be enabled.
    RequiredByExtension(&'static str),
}

impl fmt::Display for FeatureRestriction {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            FeatureRestriction::NotSupported => {
                write!(fmt, "not supported by the physical device")
            }
            FeatureRestriction::RequiresFeature(feat) => {
                write!(fmt, "requires feature {} to be enabled", feat)
            }
            FeatureRestriction::ConflictsFeature(feat) => {
                write!(fmt, "requires feature {} to be disabled", feat)
            }
            FeatureRestriction::RequiredByExtension(ext) => {
                write!(fmt, "required to be enabled by extension {}", ext)
            }
        }
    }
}

macro_rules! features_ffi {
    {
        $api_version:ident,
        $extensions:ident,
        $($member:ident => {
            ty: $ty:ident,
            provided_by: [$($provided_by:expr),+],
            conflicts: [$($conflicts:ident),*],
        },)+
    } => {
        #[derive(Default)]
        pub(crate) struct FeaturesFfi {
            features_vulkan10: Option<ash::vk::PhysicalDeviceFeatures2KHR>,

            $(
                $member: Option<ash::vk::$ty>,
            )+
        }

        impl FeaturesFfi {
            pub(crate) fn make_chain(&mut self, $api_version: Version, $extensions: &DeviceExtensions) {
                self.features_vulkan10 = Some(Default::default());
                $(
                    if std::array::IntoIter::new([$($provided_by),+]).any(|x| x) &&
                        std::array::IntoIter::new([$(self.$conflicts.is_none()),*]).all(|x| x) {
                        self.$member = Some(Default::default());
                        self.$member.unwrap().p_next = self.features_vulkan10.unwrap().p_next;
                        self.features_vulkan10.unwrap().p_next = self.$member.as_mut().unwrap() as *mut _ as _;
                    }
                )+
            }
        }
    };
}

impl FeaturesFfi {
    pub(crate) fn head_as_ref(&self) -> &ash::vk::PhysicalDeviceFeatures2KHR {
        self.features_vulkan10.as_ref().unwrap()
    }

    pub(crate) fn head_as_mut(&mut self) -> &mut ash::vk::PhysicalDeviceFeatures2KHR {
        self.features_vulkan10.as_mut().unwrap()
    }
}

// Auto-generated from vk.xml header version 168
features! {
    acceleration_structure => {
        ffi_name: acceleration_structure,
        ffi_members: [features_acceleration_structure_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    acceleration_structure_capture_replay => {
        ffi_name: acceleration_structure_capture_replay,
        ffi_members: [features_acceleration_structure_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    acceleration_structure_host_commands => {
        ffi_name: acceleration_structure_host_commands,
        ffi_members: [features_acceleration_structure_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    acceleration_structure_indirect_build => {
        ffi_name: acceleration_structure_indirect_build,
        ffi_members: [features_acceleration_structure_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    advanced_blend_coherent_operations => {
        ffi_name: advanced_blend_coherent_operations,
        ffi_members: [features_blend_operation_advanced_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    alpha_to_one => {
        ffi_name: alpha_to_one,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    attachment_fragment_shading_rate => {
        ffi_name: attachment_fragment_shading_rate,
        ffi_members: [features_fragment_shading_rate_khr],
        requires_features: [],
        conflicts_features: [shading_rate_image, fragment_density_map],
        required_by_extensions: [],
    },
    bresenham_lines => {
        ffi_name: bresenham_lines,
        ffi_members: [features_line_rasterization_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    buffer_device_address => {
        ffi_name: buffer_device_address,
        ffi_members: [features_vulkan12, features_buffer_device_address, features_buffer_device_address_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    buffer_device_address_capture_replay => {
        ffi_name: buffer_device_address_capture_replay,
        ffi_members: [features_vulkan12, features_buffer_device_address, features_buffer_device_address_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    buffer_device_address_multi_device => {
        ffi_name: buffer_device_address_multi_device,
        ffi_members: [features_vulkan12, features_buffer_device_address, features_buffer_device_address_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    compute_derivative_group_linear => {
        ffi_name: compute_derivative_group_linear,
        ffi_members: [features_compute_shader_derivatives_nv],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    compute_derivative_group_quads => {
        ffi_name: compute_derivative_group_quads,
        ffi_members: [features_compute_shader_derivatives_nv],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    compute_full_subgroups => {
        ffi_name: compute_full_subgroups,
        ffi_members: [features_subgroup_size_control_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    conditional_rendering => {
        ffi_name: conditional_rendering,
        ffi_members: [features_conditional_rendering_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    constant_alpha_color_blend_factors => {
        ffi_name: constant_alpha_color_blend_factors,
        ffi_members: [features_portability_subset_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    cooperative_matrix => {
        ffi_name: cooperative_matrix,
        ffi_members: [features_cooperative_matrix_nv],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    cooperative_matrix_robust_buffer_access => {
        ffi_name: cooperative_matrix_robust_buffer_access,
        ffi_members: [features_cooperative_matrix_nv],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    corner_sampled_image => {
        ffi_name: corner_sampled_image,
        ffi_members: [features_corner_sampled_image_nv],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    coverage_reduction_mode => {
        ffi_name: coverage_reduction_mode,
        ffi_members: [features_coverage_reduction_mode_nv],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    custom_border_color_without_format => {
        ffi_name: custom_border_color_without_format,
        ffi_members: [features_custom_border_color_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    custom_border_colors => {
        ffi_name: custom_border_colors,
        ffi_members: [features_custom_border_color_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    decode_mode_shared_exponent => {
        ffi_name: decode_mode_shared_exponent,
        ffi_members: [features_astc_decode_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    dedicated_allocation_image_aliasing => {
        ffi_name: dedicated_allocation_image_aliasing,
        ffi_members: [features_dedicated_allocation_image_aliasing_nv],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    depth_bias_clamp => {
        ffi_name: depth_bias_clamp,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    depth_bounds => {
        ffi_name: depth_bounds,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    depth_clamp => {
        ffi_name: depth_clamp,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    depth_clip_enable => {
        ffi_name: depth_clip_enable,
        ffi_members: [features_depth_clip_enable_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    descriptor_binding_acceleration_structure_update_after_bind => {
        ffi_name: descriptor_binding_acceleration_structure_update_after_bind,
        ffi_members: [features_acceleration_structure_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    descriptor_binding_inline_uniform_block_update_after_bind => {
        ffi_name: descriptor_binding_inline_uniform_block_update_after_bind,
        ffi_members: [features_inline_uniform_block_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    descriptor_binding_partially_bound => {
        ffi_name: descriptor_binding_partially_bound,
        ffi_members: [features_vulkan12, features_descriptor_indexing],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    descriptor_binding_sampled_image_update_after_bind => {
        ffi_name: descriptor_binding_sampled_image_update_after_bind,
        ffi_members: [features_vulkan12, features_descriptor_indexing],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    descriptor_binding_storage_buffer_update_after_bind => {
        ffi_name: descriptor_binding_storage_buffer_update_after_bind,
        ffi_members: [features_vulkan12, features_descriptor_indexing],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    descriptor_binding_storage_image_update_after_bind => {
        ffi_name: descriptor_binding_storage_image_update_after_bind,
        ffi_members: [features_vulkan12, features_descriptor_indexing],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    descriptor_binding_storage_texel_buffer_update_after_bind => {
        ffi_name: descriptor_binding_storage_texel_buffer_update_after_bind,
        ffi_members: [features_vulkan12, features_descriptor_indexing],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    descriptor_binding_uniform_buffer_update_after_bind => {
        ffi_name: descriptor_binding_uniform_buffer_update_after_bind,
        ffi_members: [features_vulkan12, features_descriptor_indexing],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    descriptor_binding_uniform_texel_buffer_update_after_bind => {
        ffi_name: descriptor_binding_uniform_texel_buffer_update_after_bind,
        ffi_members: [features_vulkan12, features_descriptor_indexing],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    descriptor_binding_update_unused_while_pending => {
        ffi_name: descriptor_binding_update_unused_while_pending,
        ffi_members: [features_vulkan12, features_descriptor_indexing],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    descriptor_binding_variable_descriptor_count => {
        ffi_name: descriptor_binding_variable_descriptor_count,
        ffi_members: [features_vulkan12, features_descriptor_indexing],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    descriptor_indexing => {
        ffi_name: descriptor_indexing,
        ffi_members: [features_vulkan12],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [ext_descriptor_indexing],
    },
    device_coherent_memory => {
        ffi_name: device_coherent_memory,
        ffi_members: [features_coherent_memory_amd],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    device_generated_commands => {
        ffi_name: device_generated_commands,
        ffi_members: [features_device_generated_commands_nv],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    device_memory_report => {
        ffi_name: device_memory_report,
        ffi_members: [features_device_memory_report_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    diagnostics_config => {
        ffi_name: diagnostics_config,
        ffi_members: [features_diagnostics_config_nv],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    draw_indirect_count => {
        ffi_name: draw_indirect_count,
        ffi_members: [features_vulkan12],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [khr_draw_indirect_count],
    },
    draw_indirect_first_instance => {
        ffi_name: draw_indirect_first_instance,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    dual_src_blend => {
        ffi_name: dual_src_blend,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    events => {
        ffi_name: events,
        ffi_members: [features_portability_subset_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    exclusive_scissor => {
        ffi_name: exclusive_scissor,
        ffi_members: [features_exclusive_scissor_nv],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    extended_dynamic_state => {
        ffi_name: extended_dynamic_state,
        ffi_members: [features_extended_dynamic_state_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    fill_mode_non_solid => {
        ffi_name: fill_mode_non_solid,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    format_a4b4g4r4 => {
        ffi_name: format_a4b4g4r4,
        ffi_members: [features_4444formats_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    format_a4r4g4b4 => {
        ffi_name: format_a4r4g4b4,
        ffi_members: [features_4444formats_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    fragment_density_map => {
        ffi_name: fragment_density_map,
        ffi_members: [features_fragment_density_map_ext],
        requires_features: [],
        conflicts_features: [pipeline_fragment_shading_rate, primitive_fragment_shading_rate, attachment_fragment_shading_rate],
        required_by_extensions: [],
    },
    fragment_density_map_deferred => {
        ffi_name: fragment_density_map_deferred,
        ffi_members: [features_fragment_density_map2_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    fragment_density_map_dynamic => {
        ffi_name: fragment_density_map_dynamic,
        ffi_members: [features_fragment_density_map_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    fragment_density_map_non_subsampled_images => {
        ffi_name: fragment_density_map_non_subsampled_images,
        ffi_members: [features_fragment_density_map_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    fragment_shader_barycentric => {
        ffi_name: fragment_shader_barycentric,
        ffi_members: [features_fragment_shader_barycentric_nv],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    fragment_shader_pixel_interlock => {
        ffi_name: fragment_shader_pixel_interlock,
        ffi_members: [features_fragment_shader_interlock_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    fragment_shader_sample_interlock => {
        ffi_name: fragment_shader_sample_interlock,
        ffi_members: [features_fragment_shader_interlock_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    fragment_shader_shading_rate_interlock => {
        ffi_name: fragment_shader_shading_rate_interlock,
        ffi_members: [features_fragment_shader_interlock_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    fragment_shading_rate_enums => {
        ffi_name: fragment_shading_rate_enums,
        ffi_members: [features_fragment_shading_rate_enums_nv],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    fragment_stores_and_atomics => {
        ffi_name: fragment_stores_and_atomics,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    full_draw_index_uint32 => {
        ffi_name: full_draw_index_uint32,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    geometry_shader => {
        ffi_name: geometry_shader,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    geometry_streams => {
        ffi_name: geometry_streams,
        ffi_members: [features_transform_feedback_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    host_query_reset => {
        ffi_name: host_query_reset,
        ffi_members: [features_vulkan12, features_host_query_reset],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    image_cube_array => {
        ffi_name: image_cube_array,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    image_footprint => {
        ffi_name: image_footprint,
        ffi_members: [features_shader_image_footprint_nv],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    image_view2_d_on3_d_image => {
        ffi_name: image_view2_d_on3_d_image,
        ffi_members: [features_portability_subset_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    image_view_format_reinterpretation => {
        ffi_name: image_view_format_reinterpretation,
        ffi_members: [features_portability_subset_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    image_view_format_swizzle => {
        ffi_name: image_view_format_swizzle,
        ffi_members: [features_portability_subset_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    imageless_framebuffer => {
        ffi_name: imageless_framebuffer,
        ffi_members: [features_vulkan12, features_imageless_framebuffer],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    independent_blend => {
        ffi_name: independent_blend,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    index_type_uint8 => {
        ffi_name: index_type_uint8,
        ffi_members: [features_index_type_uint8_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    inherited_conditional_rendering => {
        ffi_name: inherited_conditional_rendering,
        ffi_members: [features_conditional_rendering_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    inherited_queries => {
        ffi_name: inherited_queries,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    inline_uniform_block => {
        ffi_name: inline_uniform_block,
        ffi_members: [features_inline_uniform_block_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    large_points => {
        ffi_name: large_points,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    logic_op => {
        ffi_name: logic_op,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    memory_priority => {
        ffi_name: memory_priority,
        ffi_members: [features_memory_priority_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    mesh_shader => {
        ffi_name: mesh_shader,
        ffi_members: [features_mesh_shader_nv],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    multi_draw_indirect => {
        ffi_name: multi_draw_indirect,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    multi_viewport => {
        ffi_name: multi_viewport,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    multisample_array_image => {
        ffi_name: multisample_array_image,
        ffi_members: [features_portability_subset_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    multiview => {
        ffi_name: multiview,
        ffi_members: [features_multiview, features_vulkan11],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    multiview_geometry_shader => {
        ffi_name: multiview_geometry_shader,
        ffi_members: [features_multiview, features_vulkan11],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    multiview_tessellation_shader => {
        ffi_name: multiview_tessellation_shader,
        ffi_members: [features_multiview, features_vulkan11],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    mutable_comparison_samplers => {
        ffi_name: mutable_comparison_samplers,
        ffi_members: [features_portability_subset_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    mutable_descriptor_type => {
        ffi_name: mutable_descriptor_type,
        ffi_members: [features_mutable_descriptor_type_valve],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    no_invocation_fragment_shading_rates => {
        ffi_name: no_invocation_fragment_shading_rates,
        ffi_members: [features_fragment_shading_rate_enums_nv],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    null_descriptor => {
        ffi_name: null_descriptor,
        ffi_members: [features_robustness2_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    occlusion_query_precise => {
        ffi_name: occlusion_query_precise,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    performance_counter_multiple_query_pools => {
        ffi_name: performance_counter_multiple_query_pools,
        ffi_members: [features_performance_query_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    performance_counter_query_pools => {
        ffi_name: performance_counter_query_pools,
        ffi_members: [features_performance_query_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    pipeline_creation_cache_control => {
        ffi_name: pipeline_creation_cache_control,
        ffi_members: [features_pipeline_creation_cache_control_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    pipeline_executable_info => {
        ffi_name: pipeline_executable_info,
        ffi_members: [features_pipeline_executable_properties_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    pipeline_fragment_shading_rate => {
        ffi_name: pipeline_fragment_shading_rate,
        ffi_members: [features_fragment_shading_rate_khr],
        requires_features: [],
        conflicts_features: [shading_rate_image, fragment_density_map],
        required_by_extensions: [],
    },
    pipeline_statistics_query => {
        ffi_name: pipeline_statistics_query,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    point_polygons => {
        ffi_name: point_polygons,
        ffi_members: [features_portability_subset_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    primitive_fragment_shading_rate => {
        ffi_name: primitive_fragment_shading_rate,
        ffi_members: [features_fragment_shading_rate_khr],
        requires_features: [],
        conflicts_features: [shading_rate_image, fragment_density_map],
        required_by_extensions: [],
    },
    private_data => {
        ffi_name: private_data,
        ffi_members: [features_private_data_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    protected_memory => {
        ffi_name: protected_memory,
        ffi_members: [features_vulkan11, features_protected_memory],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    ray_query => {
        ffi_name: ray_query,
        ffi_members: [features_ray_query_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    ray_tracing_pipeline => {
        ffi_name: ray_tracing_pipeline,
        ffi_members: [features_ray_tracing_pipeline_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    ray_tracing_pipeline_shader_group_handle_capture_replay => {
        ffi_name: ray_tracing_pipeline_shader_group_handle_capture_replay,
        ffi_members: [features_ray_tracing_pipeline_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    ray_tracing_pipeline_shader_group_handle_capture_replay_mixed => {
        ffi_name: ray_tracing_pipeline_shader_group_handle_capture_replay_mixed,
        ffi_members: [features_ray_tracing_pipeline_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    ray_tracing_pipeline_trace_rays_indirect => {
        ffi_name: ray_tracing_pipeline_trace_rays_indirect,
        ffi_members: [features_ray_tracing_pipeline_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    ray_traversal_primitive_culling => {
        ffi_name: ray_traversal_primitive_culling,
        ffi_members: [features_ray_tracing_pipeline_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    rectangular_lines => {
        ffi_name: rectangular_lines,
        ffi_members: [features_line_rasterization_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    representative_fragment_test => {
        ffi_name: representative_fragment_test,
        ffi_members: [features_representative_fragment_test_nv],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    robust_buffer_access => {
        ffi_name: robust_buffer_access,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    robust_buffer_access2 => {
        ffi_name: robust_buffer_access2,
        ffi_members: [features_robustness2_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    robust_image_access => {
        ffi_name: robust_image_access,
        ffi_members: [features_image_robustness_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    robust_image_access2 => {
        ffi_name: robust_image_access2,
        ffi_members: [features_robustness2_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    runtime_descriptor_array => {
        ffi_name: runtime_descriptor_array,
        ffi_members: [features_vulkan12, features_descriptor_indexing],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    sample_rate_shading => {
        ffi_name: sample_rate_shading,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    sampler_anisotropy => {
        ffi_name: sampler_anisotropy,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    sampler_filter_minmax => {
        ffi_name: sampler_filter_minmax,
        ffi_members: [features_vulkan12],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [ext_sampler_filter_minmax],
    },
    sampler_mip_lod_bias => {
        ffi_name: sampler_mip_lod_bias,
        ffi_members: [features_portability_subset_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    sampler_mirror_clamp_to_edge => {
        ffi_name: sampler_mirror_clamp_to_edge,
        ffi_members: [features_vulkan12],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [khr_sampler_mirror_clamp_to_edge],
    },
    sampler_ycbcr_conversion => {
        ffi_name: sampler_ycbcr_conversion,
        ffi_members: [features_sampler_ycbcr_conversion, features_vulkan11],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    scalar_block_layout => {
        ffi_name: scalar_block_layout,
        ffi_members: [features_vulkan12, features_scalar_block_layout],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    separate_depth_stencil_layouts => {
        ffi_name: separate_depth_stencil_layouts,
        ffi_members: [features_vulkan12, features_separate_depth_stencil_layouts],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    separate_stencil_mask_ref => {
        ffi_name: separate_stencil_mask_ref,
        ffi_members: [features_portability_subset_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_buffer_float32_atomic_add => {
        ffi_name: shader_buffer_float32_atomic_add,
        ffi_members: [features_shader_atomic_float_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_buffer_float32_atomics => {
        ffi_name: shader_buffer_float32_atomics,
        ffi_members: [features_shader_atomic_float_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_buffer_float64_atomic_add => {
        ffi_name: shader_buffer_float64_atomic_add,
        ffi_members: [features_shader_atomic_float_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_buffer_float64_atomics => {
        ffi_name: shader_buffer_float64_atomics,
        ffi_members: [features_shader_atomic_float_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_buffer_int64_atomics => {
        ffi_name: shader_buffer_int64_atomics,
        ffi_members: [features_vulkan12, features_shader_atomic_int64],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_clip_distance => {
        ffi_name: shader_clip_distance,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_cull_distance => {
        ffi_name: shader_cull_distance,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_demote_to_helper_invocation => {
        ffi_name: shader_demote_to_helper_invocation,
        ffi_members: [features_shader_demote_to_helper_invocation_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_device_clock => {
        ffi_name: shader_device_clock,
        ffi_members: [features_shader_clock_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_draw_parameters => {
        ffi_name: shader_draw_parameters,
        ffi_members: [features_vulkan11, features_shader_draw_parameters],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [khr_shader_draw_parameters],
    },
    shader_float16 => {
        ffi_name: shader_float16,
        ffi_members: [features_vulkan12, features_shader_float16_int8],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_float64 => {
        ffi_name: shader_float64,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_image_float32_atomic_add => {
        ffi_name: shader_image_float32_atomic_add,
        ffi_members: [features_shader_atomic_float_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_image_float32_atomics => {
        ffi_name: shader_image_float32_atomics,
        ffi_members: [features_shader_atomic_float_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_image_gather_extended => {
        ffi_name: shader_image_gather_extended,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_image_int64_atomics => {
        ffi_name: shader_image_int64_atomics,
        ffi_members: [features_shader_image_atomic_int64_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_input_attachment_array_dynamic_indexing => {
        ffi_name: shader_input_attachment_array_dynamic_indexing,
        ffi_members: [features_vulkan12, features_descriptor_indexing],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_input_attachment_array_non_uniform_indexing => {
        ffi_name: shader_input_attachment_array_non_uniform_indexing,
        ffi_members: [features_vulkan12, features_descriptor_indexing],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_int16 => {
        ffi_name: shader_int16,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_int64 => {
        ffi_name: shader_int64,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_int8 => {
        ffi_name: shader_int8,
        ffi_members: [features_vulkan12, features_shader_float16_int8],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_integer_functions2 => {
        ffi_name: shader_integer_functions2,
        ffi_members: [features_shader_integer_functions2_intel],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_output_layer => {
        ffi_name: shader_output_layer,
        ffi_members: [features_vulkan12],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [ext_shader_viewport_index_layer],
    },
    shader_output_viewport_index => {
        ffi_name: shader_output_viewport_index,
        ffi_members: [features_vulkan12],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [ext_shader_viewport_index_layer],
    },
    shader_resource_min_lod => {
        ffi_name: shader_resource_min_lod,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_resource_residency => {
        ffi_name: shader_resource_residency,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_sample_rate_interpolation_functions => {
        ffi_name: shader_sample_rate_interpolation_functions,
        ffi_members: [features_portability_subset_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_sampled_image_array_dynamic_indexing => {
        ffi_name: shader_sampled_image_array_dynamic_indexing,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_sampled_image_array_non_uniform_indexing => {
        ffi_name: shader_sampled_image_array_non_uniform_indexing,
        ffi_members: [features_vulkan12, features_descriptor_indexing],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_shared_float32_atomic_add => {
        ffi_name: shader_shared_float32_atomic_add,
        ffi_members: [features_shader_atomic_float_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_shared_float32_atomics => {
        ffi_name: shader_shared_float32_atomics,
        ffi_members: [features_shader_atomic_float_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_shared_float64_atomic_add => {
        ffi_name: shader_shared_float64_atomic_add,
        ffi_members: [features_shader_atomic_float_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_shared_float64_atomics => {
        ffi_name: shader_shared_float64_atomics,
        ffi_members: [features_shader_atomic_float_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_shared_int64_atomics => {
        ffi_name: shader_shared_int64_atomics,
        ffi_members: [features_vulkan12, features_shader_atomic_int64],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_sm_builtins => {
        ffi_name: shader_sm_builtins,
        ffi_members: [features_shader_sm_builtins_nv],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_storage_buffer_array_dynamic_indexing => {
        ffi_name: shader_storage_buffer_array_dynamic_indexing,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_storage_buffer_array_non_uniform_indexing => {
        ffi_name: shader_storage_buffer_array_non_uniform_indexing,
        ffi_members: [features_vulkan12, features_descriptor_indexing],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_storage_image_array_dynamic_indexing => {
        ffi_name: shader_storage_image_array_dynamic_indexing,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_storage_image_array_non_uniform_indexing => {
        ffi_name: shader_storage_image_array_non_uniform_indexing,
        ffi_members: [features_vulkan12, features_descriptor_indexing],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_storage_image_extended_formats => {
        ffi_name: shader_storage_image_extended_formats,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_storage_image_multisample => {
        ffi_name: shader_storage_image_multisample,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_storage_image_read_without_format => {
        ffi_name: shader_storage_image_read_without_format,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_storage_image_write_without_format => {
        ffi_name: shader_storage_image_write_without_format,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_storage_texel_buffer_array_dynamic_indexing => {
        ffi_name: shader_storage_texel_buffer_array_dynamic_indexing,
        ffi_members: [features_vulkan12, features_descriptor_indexing],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_storage_texel_buffer_array_non_uniform_indexing => {
        ffi_name: shader_storage_texel_buffer_array_non_uniform_indexing,
        ffi_members: [features_vulkan12, features_descriptor_indexing],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_subgroup_clock => {
        ffi_name: shader_subgroup_clock,
        ffi_members: [features_shader_clock_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_subgroup_extended_types => {
        ffi_name: shader_subgroup_extended_types,
        ffi_members: [features_vulkan12, features_shader_subgroup_extended_types],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_terminate_invocation => {
        ffi_name: shader_terminate_invocation,
        ffi_members: [features_shader_terminate_invocation_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_tessellation_and_geometry_point_size => {
        ffi_name: shader_tessellation_and_geometry_point_size,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_uniform_buffer_array_dynamic_indexing => {
        ffi_name: shader_uniform_buffer_array_dynamic_indexing,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_uniform_buffer_array_non_uniform_indexing => {
        ffi_name: shader_uniform_buffer_array_non_uniform_indexing,
        ffi_members: [features_vulkan12, features_descriptor_indexing],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_uniform_texel_buffer_array_dynamic_indexing => {
        ffi_name: shader_uniform_texel_buffer_array_dynamic_indexing,
        ffi_members: [features_vulkan12, features_descriptor_indexing],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_uniform_texel_buffer_array_non_uniform_indexing => {
        ffi_name: shader_uniform_texel_buffer_array_non_uniform_indexing,
        ffi_members: [features_vulkan12, features_descriptor_indexing],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shader_zero_initialize_workgroup_memory => {
        ffi_name: shader_zero_initialize_workgroup_memory,
        ffi_members: [features_zero_initialize_workgroup_memory_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shading_rate_coarse_sample_order => {
        ffi_name: shading_rate_coarse_sample_order,
        ffi_members: [features_shading_rate_image_nv],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    shading_rate_image => {
        ffi_name: shading_rate_image,
        ffi_members: [features_shading_rate_image_nv],
        requires_features: [],
        conflicts_features: [pipeline_fragment_shading_rate, primitive_fragment_shading_rate, attachment_fragment_shading_rate],
        required_by_extensions: [],
    },
    smooth_lines => {
        ffi_name: smooth_lines,
        ffi_members: [features_line_rasterization_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    sparse_binding => {
        ffi_name: sparse_binding,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    sparse_image_float32_atomic_add => {
        ffi_name: sparse_image_float32_atomic_add,
        ffi_members: [features_shader_atomic_float_ext],
        requires_features: [shader_image_float32_atomic_add],
        conflicts_features: [],
        required_by_extensions: [],
    },
    sparse_image_float32_atomics => {
        ffi_name: sparse_image_float32_atomics,
        ffi_members: [features_shader_atomic_float_ext],
        requires_features: [shader_image_float32_atomics],
        conflicts_features: [],
        required_by_extensions: [],
    },
    sparse_image_int64_atomics => {
        ffi_name: sparse_image_int64_atomics,
        ffi_members: [features_shader_image_atomic_int64_ext],
        requires_features: [shader_image_int64_atomics],
        conflicts_features: [],
        required_by_extensions: [],
    },
    sparse_residency16_samples => {
        ffi_name: sparse_residency16_samples,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    sparse_residency2_samples => {
        ffi_name: sparse_residency2_samples,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    sparse_residency4_samples => {
        ffi_name: sparse_residency4_samples,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    sparse_residency8_samples => {
        ffi_name: sparse_residency8_samples,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    sparse_residency_aliased => {
        ffi_name: sparse_residency_aliased,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    sparse_residency_buffer => {
        ffi_name: sparse_residency_buffer,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    sparse_residency_image2_d => {
        ffi_name: sparse_residency_image2_d,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    sparse_residency_image3_d => {
        ffi_name: sparse_residency_image3_d,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    stippled_bresenham_lines => {
        ffi_name: stippled_bresenham_lines,
        ffi_members: [features_line_rasterization_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    stippled_rectangular_lines => {
        ffi_name: stippled_rectangular_lines,
        ffi_members: [features_line_rasterization_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    stippled_smooth_lines => {
        ffi_name: stippled_smooth_lines,
        ffi_members: [features_line_rasterization_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    storage_buffer16_bit_access => {
        ffi_name: storage_buffer16_bit_access,
        ffi_members: [features_vulkan11, features_16bit_storage],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    storage_buffer8_bit_access => {
        ffi_name: storage_buffer8_bit_access,
        ffi_members: [features_vulkan12, features_8bit_storage],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    storage_input_output16 => {
        ffi_name: storage_input_output16,
        ffi_members: [features_vulkan11, features_16bit_storage],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    storage_push_constant16 => {
        ffi_name: storage_push_constant16,
        ffi_members: [features_vulkan11, features_16bit_storage],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    storage_push_constant8 => {
        ffi_name: storage_push_constant8,
        ffi_members: [features_vulkan12, features_8bit_storage],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    subgroup_broadcast_dynamic_id => {
        ffi_name: subgroup_broadcast_dynamic_id,
        ffi_members: [features_vulkan12],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    subgroup_size_control => {
        ffi_name: subgroup_size_control,
        ffi_members: [features_subgroup_size_control_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    supersample_fragment_shading_rates => {
        ffi_name: supersample_fragment_shading_rates,
        ffi_members: [features_fragment_shading_rate_enums_nv],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    task_shader => {
        ffi_name: task_shader,
        ffi_members: [features_mesh_shader_nv],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    tessellation_isolines => {
        ffi_name: tessellation_isolines,
        ffi_members: [features_portability_subset_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    tessellation_point_mode => {
        ffi_name: tessellation_point_mode,
        ffi_members: [features_portability_subset_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    tessellation_shader => {
        ffi_name: tessellation_shader,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    texel_buffer_alignment => {
        ffi_name: texel_buffer_alignment,
        ffi_members: [features_texel_buffer_alignment_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    texture_compression_astc_hdr => {
        ffi_name: texture_compression_astc_hdr,
        ffi_members: [features_texture_compression_astchdr_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    texture_compression_astc_ldr => {
        ffi_name: texture_compression_astc_ldr,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    texture_compression_bc => {
        ffi_name: texture_compression_bc,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    texture_compression_etc2 => {
        ffi_name: texture_compression_etc2,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    timeline_semaphore => {
        ffi_name: timeline_semaphore,
        ffi_members: [features_vulkan12, features_timeline_semaphore],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    transform_feedback => {
        ffi_name: transform_feedback,
        ffi_members: [features_transform_feedback_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    triangle_fans => {
        ffi_name: triangle_fans,
        ffi_members: [features_portability_subset_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    uniform_and_storage_buffer16_bit_access => {
        ffi_name: uniform_and_storage_buffer16_bit_access,
        ffi_members: [features_vulkan11, features_16bit_storage],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    uniform_and_storage_buffer8_bit_access => {
        ffi_name: uniform_and_storage_buffer8_bit_access,
        ffi_members: [features_vulkan12, features_8bit_storage],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    uniform_buffer_standard_layout => {
        ffi_name: uniform_buffer_standard_layout,
        ffi_members: [features_vulkan12, features_uniform_buffer_standard_layout],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    variable_multisample_rate => {
        ffi_name: variable_multisample_rate,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    variable_pointers => {
        ffi_name: variable_pointers,
        ffi_members: [features_variable_pointers, features_vulkan11],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    variable_pointers_storage_buffer => {
        ffi_name: variable_pointers_storage_buffer,
        ffi_members: [features_variable_pointers, features_vulkan11],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    vertex_attribute_access_beyond_stride => {
        ffi_name: vertex_attribute_access_beyond_stride,
        ffi_members: [features_portability_subset_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    vertex_attribute_instance_rate_divisor => {
        ffi_name: vertex_attribute_instance_rate_divisor,
        ffi_members: [features_vertex_attribute_divisor_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    vertex_attribute_instance_rate_zero_divisor => {
        ffi_name: vertex_attribute_instance_rate_zero_divisor,
        ffi_members: [features_vertex_attribute_divisor_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    vertex_pipeline_stores_and_atomics => {
        ffi_name: vertex_pipeline_stores_and_atomics,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    vulkan_memory_model => {
        ffi_name: vulkan_memory_model,
        ffi_members: [features_vulkan12, features_vulkan_memory_model],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    vulkan_memory_model_availability_visibility_chains => {
        ffi_name: vulkan_memory_model_availability_visibility_chains,
        ffi_members: [features_vulkan12, features_vulkan_memory_model],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    vulkan_memory_model_device_scope => {
        ffi_name: vulkan_memory_model_device_scope,
        ffi_members: [features_vulkan12, features_vulkan_memory_model],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    wide_lines => {
        ffi_name: wide_lines,
        ffi_members: [features_vulkan10.features],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    workgroup_memory_explicit_layout => {
        ffi_name: workgroup_memory_explicit_layout,
        ffi_members: [features_workgroup_memory_explicit_layout_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    workgroup_memory_explicit_layout16_bit_access => {
        ffi_name: workgroup_memory_explicit_layout16_bit_access,
        ffi_members: [features_workgroup_memory_explicit_layout_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    workgroup_memory_explicit_layout8_bit_access => {
        ffi_name: workgroup_memory_explicit_layout8_bit_access,
        ffi_members: [features_workgroup_memory_explicit_layout_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    workgroup_memory_explicit_layout_scalar_block_layout => {
        ffi_name: workgroup_memory_explicit_layout_scalar_block_layout,
        ffi_members: [features_workgroup_memory_explicit_layout_khr],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
    ycbcr_image_arrays => {
        ffi_name: ycbcr_image_arrays,
        ffi_members: [features_ycbcr_image_arrays_ext],
        requires_features: [],
        conflicts_features: [],
        required_by_extensions: [],
    },
}

features_ffi! {
    api_version,
    extensions,
    features_vulkan11 => {
        ty: PhysicalDeviceVulkan11Features,
        provided_by: [api_version >= Version::V1_2],
        conflicts: [],
    },
    features_vulkan12 => {
        ty: PhysicalDeviceVulkan12Features,
        provided_by: [api_version >= Version::V1_2],
        conflicts: [],
    },
    features_16bit_storage => {
        ty: PhysicalDevice16BitStorageFeatures,
        provided_by: [api_version >= Version::V1_1, extensions.khr_16bit_storage],
        conflicts: [features_vulkan11],
    },
    features_multiview => {
        ty: PhysicalDeviceMultiviewFeatures,
        provided_by: [api_version >= Version::V1_1, extensions.khr_multiview],
        conflicts: [features_vulkan11],
    },
    features_protected_memory => {
        ty: PhysicalDeviceProtectedMemoryFeatures,
        provided_by: [api_version >= Version::V1_1],
        conflicts: [features_vulkan11],
    },
    features_sampler_ycbcr_conversion => {
        ty: PhysicalDeviceSamplerYcbcrConversionFeatures,
        provided_by: [api_version >= Version::V1_1, extensions.khr_sampler_ycbcr_conversion],
        conflicts: [features_vulkan11],
    },
    features_shader_draw_parameters => {
        ty: PhysicalDeviceShaderDrawParametersFeatures,
        provided_by: [api_version >= Version::V1_1],
        conflicts: [features_vulkan11],
    },
    features_variable_pointers => {
        ty: PhysicalDeviceVariablePointersFeatures,
        provided_by: [api_version >= Version::V1_1, extensions.khr_variable_pointers],
        conflicts: [features_vulkan11],
    },
    features_8bit_storage => {
        ty: PhysicalDevice8BitStorageFeatures,
        provided_by: [api_version >= Version::V1_2, extensions.khr_8bit_storage],
        conflicts: [features_vulkan12],
    },
    features_buffer_device_address => {
        ty: PhysicalDeviceBufferDeviceAddressFeatures,
        provided_by: [api_version >= Version::V1_2, extensions.khr_buffer_device_address],
        conflicts: [features_vulkan12],
    },
    features_descriptor_indexing => {
        ty: PhysicalDeviceDescriptorIndexingFeatures,
        provided_by: [api_version >= Version::V1_2, extensions.ext_descriptor_indexing],
        conflicts: [features_vulkan12],
    },
    features_host_query_reset => {
        ty: PhysicalDeviceHostQueryResetFeatures,
        provided_by: [api_version >= Version::V1_2, extensions.ext_host_query_reset],
        conflicts: [features_vulkan12],
    },
    features_imageless_framebuffer => {
        ty: PhysicalDeviceImagelessFramebufferFeatures,
        provided_by: [api_version >= Version::V1_2, extensions.khr_imageless_framebuffer],
        conflicts: [features_vulkan12],
    },
    features_scalar_block_layout => {
        ty: PhysicalDeviceScalarBlockLayoutFeatures,
        provided_by: [api_version >= Version::V1_2, extensions.ext_scalar_block_layout],
        conflicts: [features_vulkan12],
    },
    features_separate_depth_stencil_layouts => {
        ty: PhysicalDeviceSeparateDepthStencilLayoutsFeatures,
        provided_by: [api_version >= Version::V1_2, extensions.khr_separate_depth_stencil_layouts],
        conflicts: [features_vulkan12],
    },
    features_shader_atomic_int64 => {
        ty: PhysicalDeviceShaderAtomicInt64Features,
        provided_by: [api_version >= Version::V1_2, extensions.khr_shader_atomic_int64],
        conflicts: [features_vulkan12],
    },
    features_shader_float16_int8 => {
        ty: PhysicalDeviceShaderFloat16Int8Features,
        provided_by: [api_version >= Version::V1_2, extensions.khr_shader_float16_int8],
        conflicts: [features_vulkan12],
    },
    features_shader_subgroup_extended_types => {
        ty: PhysicalDeviceShaderSubgroupExtendedTypesFeatures,
        provided_by: [api_version >= Version::V1_2, extensions.khr_shader_subgroup_extended_types],
        conflicts: [features_vulkan12],
    },
    features_timeline_semaphore => {
        ty: PhysicalDeviceTimelineSemaphoreFeatures,
        provided_by: [api_version >= Version::V1_2, extensions.khr_timeline_semaphore],
        conflicts: [features_vulkan12],
    },
    features_uniform_buffer_standard_layout => {
        ty: PhysicalDeviceUniformBufferStandardLayoutFeatures,
        provided_by: [api_version >= Version::V1_2, extensions.khr_uniform_buffer_standard_layout],
        conflicts: [features_vulkan12],
    },
    features_vulkan_memory_model => {
        ty: PhysicalDeviceVulkanMemoryModelFeatures,
        provided_by: [api_version >= Version::V1_2, extensions.khr_vulkan_memory_model],
        conflicts: [features_vulkan12],
    },
    features_acceleration_structure_khr => {
        ty: PhysicalDeviceAccelerationStructureFeaturesKHR,
        provided_by: [extensions.khr_acceleration_structure],
        conflicts: [],
    },
    features_fragment_shading_rate_khr => {
        ty: PhysicalDeviceFragmentShadingRateFeaturesKHR,
        provided_by: [extensions.khr_fragment_shading_rate],
        conflicts: [],
    },
    features_performance_query_khr => {
        ty: PhysicalDevicePerformanceQueryFeaturesKHR,
        provided_by: [extensions.khr_performance_query],
        conflicts: [],
    },
    features_pipeline_executable_properties_khr => {
        ty: PhysicalDevicePipelineExecutablePropertiesFeaturesKHR,
        provided_by: [extensions.khr_pipeline_executable_properties],
        conflicts: [],
    },
    features_portability_subset_khr => {
        ty: PhysicalDevicePortabilitySubsetFeaturesKHR,
        provided_by: [extensions.khr_portability_subset],
        conflicts: [],
    },
    features_ray_query_khr => {
        ty: PhysicalDeviceRayQueryFeaturesKHR,
        provided_by: [extensions.khr_ray_query],
        conflicts: [],
    },
    features_ray_tracing_pipeline_khr => {
        ty: PhysicalDeviceRayTracingPipelineFeaturesKHR,
        provided_by: [extensions.khr_ray_tracing_pipeline],
        conflicts: [],
    },
    features_shader_clock_khr => {
        ty: PhysicalDeviceShaderClockFeaturesKHR,
        provided_by: [extensions.khr_shader_clock],
        conflicts: [],
    },
    features_shader_terminate_invocation_khr => {
        ty: PhysicalDeviceShaderTerminateInvocationFeaturesKHR,
        provided_by: [extensions.khr_shader_terminate_invocation],
        conflicts: [],
    },
    features_workgroup_memory_explicit_layout_khr => {
        ty: PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR,
        provided_by: [extensions.khr_workgroup_memory_explicit_layout],
        conflicts: [],
    },
    features_zero_initialize_workgroup_memory_khr => {
        ty: PhysicalDeviceZeroInitializeWorkgroupMemoryFeaturesKHR,
        provided_by: [extensions.khr_zero_initialize_workgroup_memory],
        conflicts: [],
    },
    features_4444formats_ext => {
        ty: PhysicalDevice4444FormatsFeaturesEXT,
        provided_by: [extensions.ext_4444_formats],
        conflicts: [],
    },
    features_astc_decode_ext => {
        ty: PhysicalDeviceASTCDecodeFeaturesEXT,
        provided_by: [extensions.ext_astc_decode_mode],
        conflicts: [],
    },
    features_blend_operation_advanced_ext => {
        ty: PhysicalDeviceBlendOperationAdvancedFeaturesEXT,
        provided_by: [extensions.ext_blend_operation_advanced],
        conflicts: [],
    },
    features_buffer_device_address_ext => {
        ty: PhysicalDeviceBufferDeviceAddressFeaturesEXT,
        provided_by: [extensions.ext_buffer_device_address],
        conflicts: [features_vulkan12, features_buffer_device_address],
    },
    features_conditional_rendering_ext => {
        ty: PhysicalDeviceConditionalRenderingFeaturesEXT,
        provided_by: [extensions.ext_conditional_rendering],
        conflicts: [],
    },
    features_custom_border_color_ext => {
        ty: PhysicalDeviceCustomBorderColorFeaturesEXT,
        provided_by: [extensions.ext_custom_border_color],
        conflicts: [],
    },
    features_depth_clip_enable_ext => {
        ty: PhysicalDeviceDepthClipEnableFeaturesEXT,
        provided_by: [extensions.ext_depth_clip_enable],
        conflicts: [],
    },
    features_device_memory_report_ext => {
        ty: PhysicalDeviceDeviceMemoryReportFeaturesEXT,
        provided_by: [extensions.ext_device_memory_report],
        conflicts: [],
    },
    features_extended_dynamic_state_ext => {
        ty: PhysicalDeviceExtendedDynamicStateFeaturesEXT,
        provided_by: [extensions.ext_extended_dynamic_state],
        conflicts: [],
    },
    features_fragment_density_map2_ext => {
        ty: PhysicalDeviceFragmentDensityMap2FeaturesEXT,
        provided_by: [extensions.ext_fragment_density_map2],
        conflicts: [],
    },
    features_fragment_density_map_ext => {
        ty: PhysicalDeviceFragmentDensityMapFeaturesEXT,
        provided_by: [extensions.ext_fragment_density_map],
        conflicts: [],
    },
    features_fragment_shader_interlock_ext => {
        ty: PhysicalDeviceFragmentShaderInterlockFeaturesEXT,
        provided_by: [extensions.ext_fragment_shader_interlock],
        conflicts: [],
    },
    features_image_robustness_ext => {
        ty: PhysicalDeviceImageRobustnessFeaturesEXT,
        provided_by: [extensions.ext_image_robustness],
        conflicts: [],
    },
    features_index_type_uint8_ext => {
        ty: PhysicalDeviceIndexTypeUint8FeaturesEXT,
        provided_by: [extensions.ext_index_type_uint8],
        conflicts: [],
    },
    features_inline_uniform_block_ext => {
        ty: PhysicalDeviceInlineUniformBlockFeaturesEXT,
        provided_by: [extensions.ext_inline_uniform_block],
        conflicts: [],
    },
    features_line_rasterization_ext => {
        ty: PhysicalDeviceLineRasterizationFeaturesEXT,
        provided_by: [extensions.ext_line_rasterization],
        conflicts: [],
    },
    features_memory_priority_ext => {
        ty: PhysicalDeviceMemoryPriorityFeaturesEXT,
        provided_by: [extensions.ext_memory_priority],
        conflicts: [],
    },
    features_pipeline_creation_cache_control_ext => {
        ty: PhysicalDevicePipelineCreationCacheControlFeaturesEXT,
        provided_by: [extensions.ext_pipeline_creation_cache_control],
        conflicts: [],
    },
    features_private_data_ext => {
        ty: PhysicalDevicePrivateDataFeaturesEXT,
        provided_by: [extensions.ext_private_data],
        conflicts: [],
    },
    features_robustness2_ext => {
        ty: PhysicalDeviceRobustness2FeaturesEXT,
        provided_by: [extensions.ext_robustness2],
        conflicts: [],
    },
    features_shader_atomic_float_ext => {
        ty: PhysicalDeviceShaderAtomicFloatFeaturesEXT,
        provided_by: [extensions.ext_shader_atomic_float],
        conflicts: [],
    },
    features_shader_demote_to_helper_invocation_ext => {
        ty: PhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT,
        provided_by: [extensions.ext_shader_demote_to_helper_invocation],
        conflicts: [],
    },
    features_shader_image_atomic_int64_ext => {
        ty: PhysicalDeviceShaderImageAtomicInt64FeaturesEXT,
        provided_by: [extensions.ext_shader_image_atomic_int64],
        conflicts: [],
    },
    features_subgroup_size_control_ext => {
        ty: PhysicalDeviceSubgroupSizeControlFeaturesEXT,
        provided_by: [extensions.ext_subgroup_size_control],
        conflicts: [],
    },
    features_texel_buffer_alignment_ext => {
        ty: PhysicalDeviceTexelBufferAlignmentFeaturesEXT,
        provided_by: [extensions.ext_texel_buffer_alignment],
        conflicts: [],
    },
    features_texture_compression_astchdr_ext => {
        ty: PhysicalDeviceTextureCompressionASTCHDRFeaturesEXT,
        provided_by: [extensions.ext_texture_compression_astc_hdr],
        conflicts: [],
    },
    features_transform_feedback_ext => {
        ty: PhysicalDeviceTransformFeedbackFeaturesEXT,
        provided_by: [extensions.ext_transform_feedback],
        conflicts: [],
    },
    features_vertex_attribute_divisor_ext => {
        ty: PhysicalDeviceVertexAttributeDivisorFeaturesEXT,
        provided_by: [extensions.ext_vertex_attribute_divisor],
        conflicts: [],
    },
    features_ycbcr_image_arrays_ext => {
        ty: PhysicalDeviceYcbcrImageArraysFeaturesEXT,
        provided_by: [extensions.ext_ycbcr_image_arrays],
        conflicts: [],
    },
    features_coherent_memory_amd => {
        ty: PhysicalDeviceCoherentMemoryFeaturesAMD,
        provided_by: [extensions.amd_device_coherent_memory],
        conflicts: [],
    },
    features_compute_shader_derivatives_nv => {
        ty: PhysicalDeviceComputeShaderDerivativesFeaturesNV,
        provided_by: [extensions.nv_compute_shader_derivatives],
        conflicts: [],
    },
    features_cooperative_matrix_nv => {
        ty: PhysicalDeviceCooperativeMatrixFeaturesNV,
        provided_by: [extensions.nv_cooperative_matrix],
        conflicts: [],
    },
    features_corner_sampled_image_nv => {
        ty: PhysicalDeviceCornerSampledImageFeaturesNV,
        provided_by: [extensions.nv_corner_sampled_image],
        conflicts: [],
    },
    features_coverage_reduction_mode_nv => {
        ty: PhysicalDeviceCoverageReductionModeFeaturesNV,
        provided_by: [extensions.nv_coverage_reduction_mode],
        conflicts: [],
    },
    features_dedicated_allocation_image_aliasing_nv => {
        ty: PhysicalDeviceDedicatedAllocationImageAliasingFeaturesNV,
        provided_by: [extensions.nv_dedicated_allocation_image_aliasing],
        conflicts: [],
    },
    features_device_generated_commands_nv => {
        ty: PhysicalDeviceDeviceGeneratedCommandsFeaturesNV,
        provided_by: [extensions.nv_device_generated_commands],
        conflicts: [],
    },
    features_diagnostics_config_nv => {
        ty: PhysicalDeviceDiagnosticsConfigFeaturesNV,
        provided_by: [extensions.nv_device_diagnostics_config],
        conflicts: [],
    },
    features_exclusive_scissor_nv => {
        ty: PhysicalDeviceExclusiveScissorFeaturesNV,
        provided_by: [extensions.nv_scissor_exclusive],
        conflicts: [],
    },
    features_fragment_shader_barycentric_nv => {
        ty: PhysicalDeviceFragmentShaderBarycentricFeaturesNV,
        provided_by: [extensions.nv_fragment_shader_barycentric],
        conflicts: [],
    },
    features_fragment_shading_rate_enums_nv => {
        ty: PhysicalDeviceFragmentShadingRateEnumsFeaturesNV,
        provided_by: [extensions.nv_fragment_shading_rate_enums],
        conflicts: [],
    },
    features_mesh_shader_nv => {
        ty: PhysicalDeviceMeshShaderFeaturesNV,
        provided_by: [extensions.nv_mesh_shader],
        conflicts: [],
    },
    features_mutable_descriptor_type_valve => {
        ty: PhysicalDeviceMutableDescriptorTypeFeaturesVALVE,
        provided_by: [extensions.valve_mutable_descriptor_type],
        conflicts: [],
    },
    features_representative_fragment_test_nv => {
        ty: PhysicalDeviceRepresentativeFragmentTestFeaturesNV,
        provided_by: [extensions.nv_representative_fragment_test],
        conflicts: [],
    },
    features_shader_image_footprint_nv => {
        ty: PhysicalDeviceShaderImageFootprintFeaturesNV,
        provided_by: [extensions.nv_shader_image_footprint],
        conflicts: [],
    },
    features_shader_integer_functions2_intel => {
        ty: PhysicalDeviceShaderIntegerFunctions2FeaturesINTEL,
        provided_by: [extensions.intel_shader_integer_functions2],
        conflicts: [],
    },
    features_shader_sm_builtins_nv => {
        ty: PhysicalDeviceShaderSMBuiltinsFeaturesNV,
        provided_by: [extensions.nv_shader_sm_builtins],
        conflicts: [],
    },
    features_shading_rate_image_nv => {
        ty: PhysicalDeviceShadingRateImageFeaturesNV,
        provided_by: [extensions.nv_shading_rate_image],
        conflicts: [],
    },
}
