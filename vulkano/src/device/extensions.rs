// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::check_errors;
use crate::extensions::SupportedExtensionsError;
use crate::extensions::{ExtensionRequirement, ExtensionRequirementError};
use crate::instance::{InstanceExtensions, PhysicalDevice};
use crate::Version;
use crate::VulkanObject;
use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::fmt;
use std::iter::FromIterator;
use std::ptr;
use std::str;

macro_rules! device_extensions {
    (
        $($member:ident => {
            raw: $raw:expr,
            requires_core: $requires_core:ident,
            requires_device_extensions: [$($requires_device_extension:ident),*],
            requires_instance_extensions: [$($requires_instance_extension:ident),*],
        },)*
    ) => (
        extensions! {
            DeviceExtensions, RawDeviceExtensions,
            $( $member => {
                raw: $raw,
                requires_core: $requires_core,
                requires_device_extensions: [$($requires_device_extension),*],
                requires_instance_extensions: [$($requires_instance_extension),*],
            },)*
        }

        impl DeviceExtensions {
            /// Checks enabled extensions against the device version, instance extensions and each other.
            pub(super) fn check_requirements(&self, api_version: Version, instance_extensions: InstanceExtensions) -> Result<(), ExtensionRequirementError> {
                $(
                    if self.$member {
                        if api_version < Version::$requires_core {
                            return Err(ExtensionRequirementError {
                                extension: stringify!($member),
                                requirement: ExtensionRequirement::Core(Version::$requires_core),
                            });
                        } else {
                            $(
                                if !self.$requires_device_extension {
                                    return Err(ExtensionRequirementError {
                                        extension: stringify!($member),
                                        requirement: ExtensionRequirement::DeviceExtension(stringify!($requires_device_extension)),
                                    });
                                }
                            )*

                            $(
                                if !instance_extensions.$requires_instance_extension {
                                    return Err(ExtensionRequirementError {
                                        extension: stringify!($member),
                                        requirement: ExtensionRequirement::InstanceExtension(stringify!($requires_instance_extension)),
                                    });
                                }
                            )*
                        }
                    }
                )*
                Ok(())
            }
        }

        impl From<&[ash::vk::ExtensionProperties]> for DeviceExtensions {
            fn from(properties: &[ash::vk::ExtensionProperties]) -> Self {
                let mut extensions = DeviceExtensions::none();
                for property in properties {
                    let name = unsafe { CStr::from_ptr(property.extension_name.as_ptr()) };
                    $(
                        // TODO: Check specVersion?
                        if name.to_bytes() == &$raw[..] {
                            extensions.$member = true;
                        }
                    )*
                }
                extensions
            }
        }
   );
}

impl DeviceExtensions {
    /// See the docs of supported_by_device().
    pub fn supported_by_device_raw(
        physical_device: PhysicalDevice,
    ) -> Result<Self, SupportedExtensionsError> {
        let fns = physical_device.instance().fns();

        let properties: Vec<ash::vk::ExtensionProperties> = unsafe {
            let mut num = 0;
            check_errors(fns.v1_0.enumerate_device_extension_properties(
                physical_device.internal_object(),
                ptr::null(),
                &mut num,
                ptr::null_mut(),
            ))?;

            let mut properties = Vec::with_capacity(num as usize);
            check_errors(fns.v1_0.enumerate_device_extension_properties(
                physical_device.internal_object(),
                ptr::null(),
                &mut num,
                properties.as_mut_ptr(),
            ))?;
            properties.set_len(num as usize);
            properties
        };

        Ok(DeviceExtensions::from(properties.as_slice()))
    }

    /// Returns an `Extensions` object with extensions supported by the `PhysicalDevice`.
    pub fn supported_by_device(physical_device: PhysicalDevice) -> Self {
        match DeviceExtensions::supported_by_device_raw(physical_device) {
            Ok(l) => l,
            Err(SupportedExtensionsError::LoadingError(_)) => unreachable!(),
            Err(SupportedExtensionsError::OomError(e)) => panic!("{:?}", e),
        }
    }

    /// Returns an `Extensions` object with extensions required as well as supported by the `PhysicalDevice`.
    /// They are needed to be passed to `Device::new(...)`.
    pub fn required_extensions(physical_device: PhysicalDevice) -> Self {
        let supported = Self::supported_by_device(physical_device);
        let required_if_supported = Self::required_if_supported_extensions();

        required_if_supported.intersection(&supported)
    }

    // required if supported extensions
    fn required_if_supported_extensions() -> Self {
        Self {
            // https://vulkan.lunarg.com/doc/view/1.2.162.1/mac/1.2-extensions/vkspec.html#VUID-VkDeviceCreateInfo-pProperties-04451
            khr_portability_subset: true,
            ..Self::none()
        }
    }
}

impl RawDeviceExtensions {
    /// See the docs of supported_by_device().
    pub fn supported_by_device_raw(
        physical_device: PhysicalDevice,
    ) -> Result<Self, SupportedExtensionsError> {
        let fns = physical_device.instance().fns();

        let properties: Vec<ash::vk::ExtensionProperties> = unsafe {
            let mut num = 0;
            check_errors(fns.v1_0.enumerate_device_extension_properties(
                physical_device.internal_object(),
                ptr::null(),
                &mut num,
                ptr::null_mut(),
            ))?;

            let mut properties = Vec::with_capacity(num as usize);
            check_errors(fns.v1_0.enumerate_device_extension_properties(
                physical_device.internal_object(),
                ptr::null(),
                &mut num,
                properties.as_mut_ptr(),
            ))?;
            properties.set_len(num as usize);
            properties
        };
        Ok(RawDeviceExtensions(
            properties
                .iter()
                .map(|x| unsafe { CStr::from_ptr(x.extension_name.as_ptr()) }.to_owned())
                .collect(),
        ))
    }

    /// Returns an `Extensions` object with extensions supported by the `PhysicalDevice`.
    pub fn supported_by_device(physical_device: PhysicalDevice) -> Self {
        match RawDeviceExtensions::supported_by_device_raw(physical_device) {
            Ok(l) => l,
            Err(SupportedExtensionsError::LoadingError(_)) => unreachable!(),
            Err(SupportedExtensionsError::OomError(e)) => panic!("{:?}", e),
        }
    }
}

/// This helper type can only be instantiated inside this module.
/// See `*Extensions::_unbuildable`.
#[doc(hidden)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Unbuildable(());

#[cfg(test)]
mod tests {
    use crate::device::{DeviceExtensions, RawDeviceExtensions};

    #[test]
    fn empty_extensions() {
        let d: RawDeviceExtensions = (&DeviceExtensions::none()).into();
        assert!(d.iter().next().is_none());
    }

    #[test]
    fn required_if_supported_extensions() {
        assert_eq!(
            DeviceExtensions::required_if_supported_extensions(),
            DeviceExtensions {
                khr_portability_subset: true,
                ..DeviceExtensions::none()
            }
        )
    }
}

// Auto-generated from vk.xml
device_extensions! {
    khr_16bit_storage => {
        raw: b"VK_KHR_16bit_storage",
        requires_core: V1_0,
        requires_device_extensions: [khr_storage_buffer_storage_class],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_8bit_storage => {
        raw: b"VK_KHR_8bit_storage",
        requires_core: V1_0,
        requires_device_extensions: [khr_storage_buffer_storage_class],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_acceleration_structure => {
        raw: b"VK_KHR_acceleration_structure",
        requires_core: V1_1,
        requires_device_extensions: [ext_descriptor_indexing, khr_buffer_device_address, khr_deferred_host_operations],
        requires_instance_extensions: [],
    },
    khr_bind_memory2 => {
        raw: b"VK_KHR_bind_memory2",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_buffer_device_address => {
        raw: b"VK_KHR_buffer_device_address",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_copy_commands2 => {
        raw: b"VK_KHR_copy_commands2",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_create_renderpass2 => {
        raw: b"VK_KHR_create_renderpass2",
        requires_core: V1_0,
        requires_device_extensions: [khr_multiview, khr_maintenance2],
        requires_instance_extensions: [],
    },
    khr_dedicated_allocation => {
        raw: b"VK_KHR_dedicated_allocation",
        requires_core: V1_0,
        requires_device_extensions: [khr_get_memory_requirements2],
        requires_instance_extensions: [],
    },
    khr_deferred_host_operations => {
        raw: b"VK_KHR_deferred_host_operations",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_depth_stencil_resolve => {
        raw: b"VK_KHR_depth_stencil_resolve",
        requires_core: V1_0,
        requires_device_extensions: [khr_create_renderpass2],
        requires_instance_extensions: [],
    },
    khr_descriptor_update_template => {
        raw: b"VK_KHR_descriptor_update_template",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_device_group => {
        raw: b"VK_KHR_device_group",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_device_group_creation],
    },
    khr_display_swapchain => {
        raw: b"VK_KHR_display_swapchain",
        requires_core: V1_0,
        requires_device_extensions: [khr_swapchain],
        requires_instance_extensions: [khr_display],
    },
    khr_draw_indirect_count => {
        raw: b"VK_KHR_draw_indirect_count",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_driver_properties => {
        raw: b"VK_KHR_driver_properties",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_external_fence => {
        raw: b"VK_KHR_external_fence",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_external_fence_capabilities],
    },
    khr_external_fence_fd => {
        raw: b"VK_KHR_external_fence_fd",
        requires_core: V1_0,
        requires_device_extensions: [khr_external_fence],
        requires_instance_extensions: [],
    },
    khr_external_fence_win32 => {
        raw: b"VK_KHR_external_fence_win32",
        requires_core: V1_0,
        requires_device_extensions: [khr_external_fence],
        requires_instance_extensions: [],
    },
    khr_external_memory => {
        raw: b"VK_KHR_external_memory",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_external_memory_capabilities],
    },
    khr_external_memory_fd => {
        raw: b"VK_KHR_external_memory_fd",
        requires_core: V1_0,
        requires_device_extensions: [khr_external_memory],
        requires_instance_extensions: [],
    },
    khr_external_memory_win32 => {
        raw: b"VK_KHR_external_memory_win32",
        requires_core: V1_0,
        requires_device_extensions: [khr_external_memory],
        requires_instance_extensions: [],
    },
    khr_external_semaphore => {
        raw: b"VK_KHR_external_semaphore",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_external_semaphore_capabilities],
    },
    khr_external_semaphore_fd => {
        raw: b"VK_KHR_external_semaphore_fd",
        requires_core: V1_0,
        requires_device_extensions: [khr_external_semaphore],
        requires_instance_extensions: [],
    },
    khr_external_semaphore_win32 => {
        raw: b"VK_KHR_external_semaphore_win32",
        requires_core: V1_0,
        requires_device_extensions: [khr_external_semaphore],
        requires_instance_extensions: [],
    },
    khr_fragment_shading_rate => {
        raw: b"VK_KHR_fragment_shading_rate",
        requires_core: V1_0,
        requires_device_extensions: [khr_create_renderpass2],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_get_memory_requirements2 => {
        raw: b"VK_KHR_get_memory_requirements2",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_image_format_list => {
        raw: b"VK_KHR_image_format_list",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_imageless_framebuffer => {
        raw: b"VK_KHR_imageless_framebuffer",
        requires_core: V1_0,
        requires_device_extensions: [khr_maintenance2, khr_image_format_list],
        requires_instance_extensions: [],
    },
    khr_incremental_present => {
        raw: b"VK_KHR_incremental_present",
        requires_core: V1_0,
        requires_device_extensions: [khr_swapchain],
        requires_instance_extensions: [],
    },
    khr_maintenance1 => {
        raw: b"VK_KHR_maintenance1",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_maintenance2 => {
        raw: b"VK_KHR_maintenance2",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_maintenance3 => {
        raw: b"VK_KHR_maintenance3",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_multiview => {
        raw: b"VK_KHR_multiview",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_performance_query => {
        raw: b"VK_KHR_performance_query",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_pipeline_executable_properties => {
        raw: b"VK_KHR_pipeline_executable_properties",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_pipeline_library => {
        raw: b"VK_KHR_pipeline_library",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_portability_subset => {
        raw: b"VK_KHR_portability_subset",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_push_descriptor => {
        raw: b"VK_KHR_push_descriptor",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_ray_query => {
        raw: b"VK_KHR_ray_query",
        requires_core: V1_1,
        requires_device_extensions: [khr_spirv_1_4, khr_acceleration_structure],
        requires_instance_extensions: [],
    },
    khr_ray_tracing_pipeline => {
        raw: b"VK_KHR_ray_tracing_pipeline",
        requires_core: V1_1,
        requires_device_extensions: [khr_spirv_1_4, khr_acceleration_structure],
        requires_instance_extensions: [],
    },
    khr_relaxed_block_layout => {
        raw: b"VK_KHR_relaxed_block_layout",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_sampler_mirror_clamp_to_edge => {
        raw: b"VK_KHR_sampler_mirror_clamp_to_edge",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_sampler_ycbcr_conversion => {
        raw: b"VK_KHR_sampler_ycbcr_conversion",
        requires_core: V1_0,
        requires_device_extensions: [khr_maintenance1, khr_bind_memory2, khr_get_memory_requirements2],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_separate_depth_stencil_layouts => {
        raw: b"VK_KHR_separate_depth_stencil_layouts",
        requires_core: V1_0,
        requires_device_extensions: [khr_create_renderpass2],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_shader_atomic_int64 => {
        raw: b"VK_KHR_shader_atomic_int64",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_shader_clock => {
        raw: b"VK_KHR_shader_clock",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_shader_draw_parameters => {
        raw: b"VK_KHR_shader_draw_parameters",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_shader_float16_int8 => {
        raw: b"VK_KHR_shader_float16_int8",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_shader_float_controls => {
        raw: b"VK_KHR_shader_float_controls",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_shader_non_semantic_info => {
        raw: b"VK_KHR_shader_non_semantic_info",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_shader_subgroup_extended_types => {
        raw: b"VK_KHR_shader_subgroup_extended_types",
        requires_core: V1_1,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_shader_terminate_invocation => {
        raw: b"VK_KHR_shader_terminate_invocation",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_shared_presentable_image => {
        raw: b"VK_KHR_shared_presentable_image",
        requires_core: V1_0,
        requires_device_extensions: [khr_swapchain],
        requires_instance_extensions: [khr_get_physical_device_properties2, khr_get_surface_capabilities2],
    },
    khr_spirv_1_4 => {
        raw: b"VK_KHR_spirv_1_4",
        requires_core: V1_1,
        requires_device_extensions: [khr_shader_float_controls],
        requires_instance_extensions: [],
    },
    khr_storage_buffer_storage_class => {
        raw: b"VK_KHR_storage_buffer_storage_class",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_swapchain => {
        raw: b"VK_KHR_swapchain",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_surface],
    },
    khr_swapchain_mutable_format => {
        raw: b"VK_KHR_swapchain_mutable_format",
        requires_core: V1_0,
        requires_device_extensions: [khr_swapchain, khr_maintenance2, khr_image_format_list],
        requires_instance_extensions: [],
    },
    khr_synchronization2 => {
        raw: b"VK_KHR_synchronization2",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_timeline_semaphore => {
        raw: b"VK_KHR_timeline_semaphore",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_uniform_buffer_standard_layout => {
        raw: b"VK_KHR_uniform_buffer_standard_layout",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_variable_pointers => {
        raw: b"VK_KHR_variable_pointers",
        requires_core: V1_0,
        requires_device_extensions: [khr_storage_buffer_storage_class],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_video_decode_queue => {
        raw: b"VK_KHR_video_decode_queue",
        requires_core: V1_0,
        requires_device_extensions: [khr_video_queue, khr_synchronization2],
        requires_instance_extensions: [],
    },
    khr_video_encode_queue => {
        raw: b"VK_KHR_video_encode_queue",
        requires_core: V1_0,
        requires_device_extensions: [khr_video_queue, khr_synchronization2],
        requires_instance_extensions: [],
    },
    khr_video_queue => {
        raw: b"VK_KHR_video_queue",
        requires_core: V1_0,
        requires_device_extensions: [khr_sampler_ycbcr_conversion],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_vulkan_memory_model => {
        raw: b"VK_KHR_vulkan_memory_model",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_win32_keyed_mutex => {
        raw: b"VK_KHR_win32_keyed_mutex",
        requires_core: V1_0,
        requires_device_extensions: [khr_external_memory_win32],
        requires_instance_extensions: [],
    },
    khr_workgroup_memory_explicit_layout => {
        raw: b"VK_KHR_workgroup_memory_explicit_layout",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_zero_initialize_workgroup_memory => {
        raw: b"VK_KHR_zero_initialize_workgroup_memory",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_4444_formats => {
        raw: b"VK_EXT_4444_formats",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_astc_decode_mode => {
        raw: b"VK_EXT_astc_decode_mode",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_blend_operation_advanced => {
        raw: b"VK_EXT_blend_operation_advanced",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_buffer_device_address => {
        raw: b"VK_EXT_buffer_device_address",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_calibrated_timestamps => {
        raw: b"VK_EXT_calibrated_timestamps",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_color_write_enable => {
        raw: b"VK_EXT_color_write_enable",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_conditional_rendering => {
        raw: b"VK_EXT_conditional_rendering",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_conservative_rasterization => {
        raw: b"VK_EXT_conservative_rasterization",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_custom_border_color => {
        raw: b"VK_EXT_custom_border_color",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_debug_marker => {
        raw: b"VK_EXT_debug_marker",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [ext_debug_report],
    },
    ext_depth_clip_enable => {
        raw: b"VK_EXT_depth_clip_enable",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_depth_range_unrestricted => {
        raw: b"VK_EXT_depth_range_unrestricted",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_descriptor_indexing => {
        raw: b"VK_EXT_descriptor_indexing",
        requires_core: V1_0,
        requires_device_extensions: [khr_maintenance3],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_device_memory_report => {
        raw: b"VK_EXT_device_memory_report",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_discard_rectangles => {
        raw: b"VK_EXT_discard_rectangles",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_display_control => {
        raw: b"VK_EXT_display_control",
        requires_core: V1_0,
        requires_device_extensions: [khr_swapchain],
        requires_instance_extensions: [ext_display_surface_counter],
    },
    ext_extended_dynamic_state => {
        raw: b"VK_EXT_extended_dynamic_state",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_extended_dynamic_state2 => {
        raw: b"VK_EXT_extended_dynamic_state2",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_external_memory_dma_buf => {
        raw: b"VK_EXT_external_memory_dma_buf",
        requires_core: V1_0,
        requires_device_extensions: [khr_external_memory_fd],
        requires_instance_extensions: [],
    },
    ext_external_memory_host => {
        raw: b"VK_EXT_external_memory_host",
        requires_core: V1_0,
        requires_device_extensions: [khr_external_memory],
        requires_instance_extensions: [],
    },
    ext_filter_cubic => {
        raw: b"VK_EXT_filter_cubic",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_fragment_density_map => {
        raw: b"VK_EXT_fragment_density_map",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_fragment_density_map2 => {
        raw: b"VK_EXT_fragment_density_map2",
        requires_core: V1_0,
        requires_device_extensions: [ext_fragment_density_map],
        requires_instance_extensions: [],
    },
    ext_fragment_shader_interlock => {
        raw: b"VK_EXT_fragment_shader_interlock",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_full_screen_exclusive => {
        raw: b"VK_EXT_full_screen_exclusive",
        requires_core: V1_0,
        requires_device_extensions: [khr_swapchain],
        requires_instance_extensions: [khr_get_physical_device_properties2, khr_surface, khr_get_surface_capabilities2],
    },
    ext_global_priority => {
        raw: b"VK_EXT_global_priority",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_hdr_metadata => {
        raw: b"VK_EXT_hdr_metadata",
        requires_core: V1_0,
        requires_device_extensions: [khr_swapchain],
        requires_instance_extensions: [],
    },
    ext_host_query_reset => {
        raw: b"VK_EXT_host_query_reset",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_image_drm_format_modifier => {
        raw: b"VK_EXT_image_drm_format_modifier",
        requires_core: V1_0,
        requires_device_extensions: [khr_bind_memory2, khr_image_format_list, khr_sampler_ycbcr_conversion],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_image_robustness => {
        raw: b"VK_EXT_image_robustness",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_index_type_uint8 => {
        raw: b"VK_EXT_index_type_uint8",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_inline_uniform_block => {
        raw: b"VK_EXT_inline_uniform_block",
        requires_core: V1_0,
        requires_device_extensions: [khr_maintenance1],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_line_rasterization => {
        raw: b"VK_EXT_line_rasterization",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_memory_budget => {
        raw: b"VK_EXT_memory_budget",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_memory_priority => {
        raw: b"VK_EXT_memory_priority",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_pci_bus_info => {
        raw: b"VK_EXT_pci_bus_info",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_pipeline_creation_cache_control => {
        raw: b"VK_EXT_pipeline_creation_cache_control",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_pipeline_creation_feedback => {
        raw: b"VK_EXT_pipeline_creation_feedback",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_post_depth_coverage => {
        raw: b"VK_EXT_post_depth_coverage",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_private_data => {
        raw: b"VK_EXT_private_data",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_provoking_vertex => {
        raw: b"VK_EXT_provoking_vertex",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_queue_family_foreign => {
        raw: b"VK_EXT_queue_family_foreign",
        requires_core: V1_0,
        requires_device_extensions: [khr_external_memory],
        requires_instance_extensions: [],
    },
    ext_robustness2 => {
        raw: b"VK_EXT_robustness2",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_sample_locations => {
        raw: b"VK_EXT_sample_locations",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_sampler_filter_minmax => {
        raw: b"VK_EXT_sampler_filter_minmax",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_scalar_block_layout => {
        raw: b"VK_EXT_scalar_block_layout",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_separate_stencil_usage => {
        raw: b"VK_EXT_separate_stencil_usage",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_shader_atomic_float => {
        raw: b"VK_EXT_shader_atomic_float",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_shader_demote_to_helper_invocation => {
        raw: b"VK_EXT_shader_demote_to_helper_invocation",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_shader_image_atomic_int64 => {
        raw: b"VK_EXT_shader_image_atomic_int64",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_shader_stencil_export => {
        raw: b"VK_EXT_shader_stencil_export",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_shader_subgroup_ballot => {
        raw: b"VK_EXT_shader_subgroup_ballot",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_shader_subgroup_vote => {
        raw: b"VK_EXT_shader_subgroup_vote",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_shader_viewport_index_layer => {
        raw: b"VK_EXT_shader_viewport_index_layer",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_subgroup_size_control => {
        raw: b"VK_EXT_subgroup_size_control",
        requires_core: V1_1,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_texel_buffer_alignment => {
        raw: b"VK_EXT_texel_buffer_alignment",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_texture_compression_astc_hdr => {
        raw: b"VK_EXT_texture_compression_astc_hdr",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_tooling_info => {
        raw: b"VK_EXT_tooling_info",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_transform_feedback => {
        raw: b"VK_EXT_transform_feedback",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_validation_cache => {
        raw: b"VK_EXT_validation_cache",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_vertex_attribute_divisor => {
        raw: b"VK_EXT_vertex_attribute_divisor",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_vertex_input_dynamic_state => {
        raw: b"VK_EXT_vertex_input_dynamic_state",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_video_decode_h264 => {
        raw: b"VK_EXT_video_decode_h264",
        requires_core: V1_0,
        requires_device_extensions: [khr_video_decode_queue],
        requires_instance_extensions: [],
    },
    ext_video_decode_h265 => {
        raw: b"VK_EXT_video_decode_h265",
        requires_core: V1_0,
        requires_device_extensions: [khr_video_decode_queue],
        requires_instance_extensions: [],
    },
    ext_video_encode_h264 => {
        raw: b"VK_EXT_video_encode_h264",
        requires_core: V1_0,
        requires_device_extensions: [khr_video_encode_queue],
        requires_instance_extensions: [],
    },
    ext_ycbcr_2plane_444_formats => {
        raw: b"VK_EXT_ycbcr_2plane_444_formats",
        requires_core: V1_0,
        requires_device_extensions: [khr_sampler_ycbcr_conversion],
        requires_instance_extensions: [],
    },
    ext_ycbcr_image_arrays => {
        raw: b"VK_EXT_ycbcr_image_arrays",
        requires_core: V1_0,
        requires_device_extensions: [khr_sampler_ycbcr_conversion],
        requires_instance_extensions: [],
    },
    amd_buffer_marker => {
        raw: b"VK_AMD_buffer_marker",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_device_coherent_memory => {
        raw: b"VK_AMD_device_coherent_memory",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_display_native_hdr => {
        raw: b"VK_AMD_display_native_hdr",
        requires_core: V1_0,
        requires_device_extensions: [khr_swapchain],
        requires_instance_extensions: [khr_get_physical_device_properties2, khr_get_surface_capabilities2],
    },
    amd_draw_indirect_count => {
        raw: b"VK_AMD_draw_indirect_count",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_gcn_shader => {
        raw: b"VK_AMD_gcn_shader",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_gpu_shader_half_float => {
        raw: b"VK_AMD_gpu_shader_half_float",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_gpu_shader_int16 => {
        raw: b"VK_AMD_gpu_shader_int16",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_memory_overallocation_behavior => {
        raw: b"VK_AMD_memory_overallocation_behavior",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_mixed_attachment_samples => {
        raw: b"VK_AMD_mixed_attachment_samples",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_negative_viewport_height => {
        raw: b"VK_AMD_negative_viewport_height",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_pipeline_compiler_control => {
        raw: b"VK_AMD_pipeline_compiler_control",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_rasterization_order => {
        raw: b"VK_AMD_rasterization_order",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_shader_ballot => {
        raw: b"VK_AMD_shader_ballot",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_shader_core_properties => {
        raw: b"VK_AMD_shader_core_properties",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    amd_shader_core_properties2 => {
        raw: b"VK_AMD_shader_core_properties2",
        requires_core: V1_0,
        requires_device_extensions: [amd_shader_core_properties],
        requires_instance_extensions: [],
    },
    amd_shader_explicit_vertex_parameter => {
        raw: b"VK_AMD_shader_explicit_vertex_parameter",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_shader_fragment_mask => {
        raw: b"VK_AMD_shader_fragment_mask",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_shader_image_load_store_lod => {
        raw: b"VK_AMD_shader_image_load_store_lod",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_shader_info => {
        raw: b"VK_AMD_shader_info",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_shader_trinary_minmax => {
        raw: b"VK_AMD_shader_trinary_minmax",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_texture_gather_bias_lod => {
        raw: b"VK_AMD_texture_gather_bias_lod",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    android_external_memory_android_hardware_buffer => {
        raw: b"VK_ANDROID_external_memory_android_hardware_buffer",
        requires_core: V1_0,
        requires_device_extensions: [khr_sampler_ycbcr_conversion, khr_external_memory, ext_queue_family_foreign, khr_dedicated_allocation],
        requires_instance_extensions: [],
    },
    fuchsia_external_memory => {
        raw: b"VK_FUCHSIA_external_memory",
        requires_core: V1_0,
        requires_device_extensions: [khr_external_memory],
        requires_instance_extensions: [khr_external_memory_capabilities],
    },
    fuchsia_external_semaphore => {
        raw: b"VK_FUCHSIA_external_semaphore",
        requires_core: V1_0,
        requires_device_extensions: [khr_external_semaphore],
        requires_instance_extensions: [khr_external_semaphore_capabilities],
    },
    ggp_frame_token => {
        raw: b"VK_GGP_frame_token",
        requires_core: V1_0,
        requires_device_extensions: [khr_swapchain],
        requires_instance_extensions: [ggp_stream_descriptor_surface],
    },
    google_decorate_string => {
        raw: b"VK_GOOGLE_decorate_string",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    google_display_timing => {
        raw: b"VK_GOOGLE_display_timing",
        requires_core: V1_0,
        requires_device_extensions: [khr_swapchain],
        requires_instance_extensions: [],
    },
    google_hlsl_functionality1 => {
        raw: b"VK_GOOGLE_hlsl_functionality1",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    google_user_type => {
        raw: b"VK_GOOGLE_user_type",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    img_filter_cubic => {
        raw: b"VK_IMG_filter_cubic",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    img_format_pvrtc => {
        raw: b"VK_IMG_format_pvrtc",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    intel_performance_query => {
        raw: b"VK_INTEL_performance_query",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    intel_shader_integer_functions2 => {
        raw: b"VK_INTEL_shader_integer_functions2",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nvx_binary_import => {
        raw: b"VK_NVX_binary_import",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nvx_image_view_handle => {
        raw: b"VK_NVX_image_view_handle",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nvx_multiview_per_view_attributes => {
        raw: b"VK_NVX_multiview_per_view_attributes",
        requires_core: V1_0,
        requires_device_extensions: [khr_multiview],
        requires_instance_extensions: [],
    },
    nv_acquire_winrt_display => {
        raw: b"VK_NV_acquire_winrt_display",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [ext_direct_mode_display],
    },
    nv_clip_space_w_scaling => {
        raw: b"VK_NV_clip_space_w_scaling",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_compute_shader_derivatives => {
        raw: b"VK_NV_compute_shader_derivatives",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_cooperative_matrix => {
        raw: b"VK_NV_cooperative_matrix",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_corner_sampled_image => {
        raw: b"VK_NV_corner_sampled_image",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_coverage_reduction_mode => {
        raw: b"VK_NV_coverage_reduction_mode",
        requires_core: V1_0,
        requires_device_extensions: [nv_framebuffer_mixed_samples],
        requires_instance_extensions: [],
    },
    nv_dedicated_allocation => {
        raw: b"VK_NV_dedicated_allocation",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_dedicated_allocation_image_aliasing => {
        raw: b"VK_NV_dedicated_allocation_image_aliasing",
        requires_core: V1_0,
        requires_device_extensions: [khr_dedicated_allocation],
        requires_instance_extensions: [],
    },
    nv_device_diagnostic_checkpoints => {
        raw: b"VK_NV_device_diagnostic_checkpoints",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_device_diagnostics_config => {
        raw: b"VK_NV_device_diagnostics_config",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_device_generated_commands => {
        raw: b"VK_NV_device_generated_commands",
        requires_core: V1_1,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_external_memory => {
        raw: b"VK_NV_external_memory",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [nv_external_memory_capabilities],
    },
    nv_external_memory_win32 => {
        raw: b"VK_NV_external_memory_win32",
        requires_core: V1_0,
        requires_device_extensions: [nv_external_memory],
        requires_instance_extensions: [],
    },
    nv_fill_rectangle => {
        raw: b"VK_NV_fill_rectangle",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_fragment_coverage_to_color => {
        raw: b"VK_NV_fragment_coverage_to_color",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_fragment_shader_barycentric => {
        raw: b"VK_NV_fragment_shader_barycentric",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_fragment_shading_rate_enums => {
        raw: b"VK_NV_fragment_shading_rate_enums",
        requires_core: V1_0,
        requires_device_extensions: [khr_fragment_shading_rate],
        requires_instance_extensions: [],
    },
    nv_framebuffer_mixed_samples => {
        raw: b"VK_NV_framebuffer_mixed_samples",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_geometry_shader_passthrough => {
        raw: b"VK_NV_geometry_shader_passthrough",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_glsl_shader => {
        raw: b"VK_NV_glsl_shader",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_inherited_viewport_scissor => {
        raw: b"VK_NV_inherited_viewport_scissor",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_mesh_shader => {
        raw: b"VK_NV_mesh_shader",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_ray_tracing => {
        raw: b"VK_NV_ray_tracing",
        requires_core: V1_0,
        requires_device_extensions: [khr_get_memory_requirements2],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_representative_fragment_test => {
        raw: b"VK_NV_representative_fragment_test",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_sample_mask_override_coverage => {
        raw: b"VK_NV_sample_mask_override_coverage",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_scissor_exclusive => {
        raw: b"VK_NV_scissor_exclusive",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_shader_image_footprint => {
        raw: b"VK_NV_shader_image_footprint",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_shader_sm_builtins => {
        raw: b"VK_NV_shader_sm_builtins",
        requires_core: V1_1,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_shader_subgroup_partitioned => {
        raw: b"VK_NV_shader_subgroup_partitioned",
        requires_core: V1_1,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_shading_rate_image => {
        raw: b"VK_NV_shading_rate_image",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_viewport_array2 => {
        raw: b"VK_NV_viewport_array2",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_viewport_swizzle => {
        raw: b"VK_NV_viewport_swizzle",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_win32_keyed_mutex => {
        raw: b"VK_NV_win32_keyed_mutex",
        requires_core: V1_0,
        requires_device_extensions: [nv_external_memory_win32],
        requires_instance_extensions: [],
    },
    qcom_render_pass_shader_resolve => {
        raw: b"VK_QCOM_render_pass_shader_resolve",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    qcom_render_pass_store_ops => {
        raw: b"VK_QCOM_render_pass_store_ops",
        requires_core: V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    qcom_render_pass_transform => {
        raw: b"VK_QCOM_render_pass_transform",
        requires_core: V1_0,
        requires_device_extensions: [khr_swapchain],
        requires_instance_extensions: [khr_surface],
    },
    qcom_rotated_copy_commands => {
        raw: b"VK_QCOM_rotated_copy_commands",
        requires_core: V1_0,
        requires_device_extensions: [khr_swapchain, khr_copy_commands2],
        requires_instance_extensions: [],
    },
    valve_mutable_descriptor_type => {
        raw: b"VK_VALVE_mutable_descriptor_type",
        requires_core: V1_0,
        requires_device_extensions: [khr_maintenance3],
        requires_instance_extensions: [],
    },

}
