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
use std::ffi::{CStr, CString};
use std::fmt;
use std::ptr;
use std::str;

macro_rules! device_extensions {
    (
        $($member:ident => {
            doc: $doc:expr,
            raw: $raw:expr,
            requires_core: $requires_core:expr,
            requires_device_extensions: [$($requires_device_extension:ident),*],
            requires_instance_extensions: [$($requires_instance_extension:ident),*],
        },)*
    ) => (
        extensions! {
            DeviceExtensions,
            $( $member => {
                doc: $doc,
                raw: $raw,
                requires_core: $requires_core,
                requires_device_extensions: [$($requires_device_extension),*],
                requires_instance_extensions: [$($requires_instance_extension),*],
            },)*
        }

        impl DeviceExtensions {
            /// Checks enabled extensions against the device version, instance extensions and each other.
            pub(super) fn check_requirements(&self, api_version: Version, instance_extensions: &InstanceExtensions) -> Result<(), ExtensionRequirementError> {
                $(
                    if self.$member {
                        if api_version < $requires_core {
                            return Err(ExtensionRequirementError {
                                extension: stringify!($member),
                                requirement: ExtensionRequirement::Core($requires_core),
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

        Ok(Self::from(properties.iter().map(|property| unsafe {
            CStr::from_ptr(property.extension_name.as_ptr())
        })))
    }

    /// Returns a `DeviceExtensions` object with extensions supported by the `PhysicalDevice`.
    pub fn supported_by_device(physical_device: PhysicalDevice) -> Self {
        match DeviceExtensions::supported_by_device_raw(physical_device) {
            Ok(l) => l,
            Err(SupportedExtensionsError::LoadingError(_)) => unreachable!(),
            Err(SupportedExtensionsError::OomError(e)) => panic!("{:?}", e),
        }
    }

    /// Returns a `DeviceExtensions` object with extensions required as well as supported by the `PhysicalDevice`.
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

/// This helper type can only be instantiated inside this module.
/// See `*Extensions::_unbuildable`.
#[doc(hidden)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Unbuildable(());

#[cfg(test)]
mod tests {
    use crate::device::DeviceExtensions;
    use std::ffi::CString;

    #[test]
    fn empty_extensions() {
        let d: Vec<CString> = (&DeviceExtensions::none()).into();
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

// Auto-generated from vk.xml header version 168
device_extensions! {
    khr_16bit_storage => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_16bit_storage.html)
			- Requires device extensions: [`khr_storage_buffer_storage_class`](crate::device::DeviceExtensions::khr_storage_buffer_storage_class)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_16bit_storage",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_storage_buffer_storage_class],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_8bit_storage => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_8bit_storage.html)
			- Requires device extensions: [`khr_storage_buffer_storage_class`](crate::device::DeviceExtensions::khr_storage_buffer_storage_class)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_KHR_8bit_storage",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_storage_buffer_storage_class],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_acceleration_structure => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_acceleration_structure.html)
			- Requires Vulkan 1.1
			- Requires device extensions: [`ext_descriptor_indexing`](crate::device::DeviceExtensions::ext_descriptor_indexing), [`khr_buffer_device_address`](crate::device::DeviceExtensions::khr_buffer_device_address), [`khr_deferred_host_operations`](crate::device::DeviceExtensions::khr_deferred_host_operations)
		",
        raw: b"VK_KHR_acceleration_structure",
        requires_core: Version::V1_1,
        requires_device_extensions: [ext_descriptor_indexing, khr_buffer_device_address, khr_deferred_host_operations],
        requires_instance_extensions: [],
    },
    khr_bind_memory2 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_bind_memory2.html)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_bind_memory2",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_buffer_device_address => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_buffer_device_address.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_KHR_buffer_device_address",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_copy_commands2 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_copy_commands2.html)
		",
        raw: b"VK_KHR_copy_commands2",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_create_renderpass2 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_create_renderpass2.html)
			- Requires device extensions: [`khr_multiview`](crate::device::DeviceExtensions::khr_multiview), [`khr_maintenance2`](crate::device::DeviceExtensions::khr_maintenance2)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_KHR_create_renderpass2",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_multiview, khr_maintenance2],
        requires_instance_extensions: [],
    },
    khr_dedicated_allocation => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_dedicated_allocation.html)
			- Requires device extensions: [`khr_get_memory_requirements2`](crate::device::DeviceExtensions::khr_get_memory_requirements2)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_dedicated_allocation",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_get_memory_requirements2],
        requires_instance_extensions: [],
    },
    khr_deferred_host_operations => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_deferred_host_operations.html)
		",
        raw: b"VK_KHR_deferred_host_operations",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_depth_stencil_resolve => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_depth_stencil_resolve.html)
			- Requires device extensions: [`khr_create_renderpass2`](crate::device::DeviceExtensions::khr_create_renderpass2)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_KHR_depth_stencil_resolve",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_create_renderpass2],
        requires_instance_extensions: [],
    },
    khr_descriptor_update_template => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_descriptor_update_template.html)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_descriptor_update_template",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_device_group => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_device_group.html)
			- Requires instance extensions: [`khr_device_group_creation`](crate::instance::InstanceExtensions::khr_device_group_creation)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_device_group",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_device_group_creation],
    },
    khr_display_swapchain => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_display_swapchain.html)
			- Requires device extensions: [`khr_swapchain`](crate::device::DeviceExtensions::khr_swapchain)
			- Requires instance extensions: [`khr_display`](crate::instance::InstanceExtensions::khr_display)
		",
        raw: b"VK_KHR_display_swapchain",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_swapchain],
        requires_instance_extensions: [khr_display],
    },
    khr_draw_indirect_count => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_draw_indirect_count.html)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_KHR_draw_indirect_count",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_driver_properties => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_driver_properties.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_KHR_driver_properties",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_external_fence => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_external_fence.html)
			- Requires instance extensions: [`khr_external_fence_capabilities`](crate::instance::InstanceExtensions::khr_external_fence_capabilities)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_external_fence",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_external_fence_capabilities],
    },
    khr_external_fence_fd => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_external_fence_fd.html)
			- Requires device extensions: [`khr_external_fence`](crate::device::DeviceExtensions::khr_external_fence)
		",
        raw: b"VK_KHR_external_fence_fd",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_external_fence],
        requires_instance_extensions: [],
    },
    khr_external_fence_win32 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_external_fence_win32.html)
			- Requires device extensions: [`khr_external_fence`](crate::device::DeviceExtensions::khr_external_fence)
		",
        raw: b"VK_KHR_external_fence_win32",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_external_fence],
        requires_instance_extensions: [],
    },
    khr_external_memory => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_external_memory.html)
			- Requires instance extensions: [`khr_external_memory_capabilities`](crate::instance::InstanceExtensions::khr_external_memory_capabilities)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_external_memory",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_external_memory_capabilities],
    },
    khr_external_memory_fd => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_external_memory_fd.html)
			- Requires device extensions: [`khr_external_memory`](crate::device::DeviceExtensions::khr_external_memory)
		",
        raw: b"VK_KHR_external_memory_fd",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_external_memory],
        requires_instance_extensions: [],
    },
    khr_external_memory_win32 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_external_memory_win32.html)
			- Requires device extensions: [`khr_external_memory`](crate::device::DeviceExtensions::khr_external_memory)
		",
        raw: b"VK_KHR_external_memory_win32",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_external_memory],
        requires_instance_extensions: [],
    },
    khr_external_semaphore => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_external_semaphore.html)
			- Requires instance extensions: [`khr_external_semaphore_capabilities`](crate::instance::InstanceExtensions::khr_external_semaphore_capabilities)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_external_semaphore",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_external_semaphore_capabilities],
    },
    khr_external_semaphore_fd => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_external_semaphore_fd.html)
			- Requires device extensions: [`khr_external_semaphore`](crate::device::DeviceExtensions::khr_external_semaphore)
		",
        raw: b"VK_KHR_external_semaphore_fd",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_external_semaphore],
        requires_instance_extensions: [],
    },
    khr_external_semaphore_win32 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_external_semaphore_win32.html)
			- Requires device extensions: [`khr_external_semaphore`](crate::device::DeviceExtensions::khr_external_semaphore)
		",
        raw: b"VK_KHR_external_semaphore_win32",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_external_semaphore],
        requires_instance_extensions: [],
    },
    khr_fragment_shading_rate => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_fragment_shading_rate.html)
			- Requires device extensions: [`khr_create_renderpass2`](crate::device::DeviceExtensions::khr_create_renderpass2)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_KHR_fragment_shading_rate",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_create_renderpass2],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_get_memory_requirements2 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_get_memory_requirements2.html)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_get_memory_requirements2",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_image_format_list => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_image_format_list.html)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_KHR_image_format_list",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_imageless_framebuffer => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_imageless_framebuffer.html)
			- Requires device extensions: [`khr_maintenance2`](crate::device::DeviceExtensions::khr_maintenance2), [`khr_image_format_list`](crate::device::DeviceExtensions::khr_image_format_list)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_KHR_imageless_framebuffer",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_maintenance2, khr_image_format_list],
        requires_instance_extensions: [],
    },
    khr_incremental_present => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_incremental_present.html)
			- Requires device extensions: [`khr_swapchain`](crate::device::DeviceExtensions::khr_swapchain)
		",
        raw: b"VK_KHR_incremental_present",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_swapchain],
        requires_instance_extensions: [],
    },
    khr_maintenance1 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_maintenance1.html)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_maintenance1",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_maintenance2 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_maintenance2.html)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_maintenance2",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_maintenance3 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_maintenance3.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_maintenance3",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_multiview => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_multiview.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_multiview",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_performance_query => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_performance_query.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_KHR_performance_query",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_pipeline_executable_properties => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_pipeline_executable_properties.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_KHR_pipeline_executable_properties",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_pipeline_library => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_pipeline_library.html)
		",
        raw: b"VK_KHR_pipeline_library",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_portability_subset => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_portability_subset.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_KHR_portability_subset",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_push_descriptor => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_push_descriptor.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_KHR_push_descriptor",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_ray_query => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_ray_query.html)
			- Requires Vulkan 1.1
			- Requires device extensions: [`khr_spirv_1_4`](crate::device::DeviceExtensions::khr_spirv_1_4), [`khr_acceleration_structure`](crate::device::DeviceExtensions::khr_acceleration_structure)
		",
        raw: b"VK_KHR_ray_query",
        requires_core: Version::V1_1,
        requires_device_extensions: [khr_spirv_1_4, khr_acceleration_structure],
        requires_instance_extensions: [],
    },
    khr_ray_tracing_pipeline => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_ray_tracing_pipeline.html)
			- Requires Vulkan 1.1
			- Requires device extensions: [`khr_spirv_1_4`](crate::device::DeviceExtensions::khr_spirv_1_4), [`khr_acceleration_structure`](crate::device::DeviceExtensions::khr_acceleration_structure)
		",
        raw: b"VK_KHR_ray_tracing_pipeline",
        requires_core: Version::V1_1,
        requires_device_extensions: [khr_spirv_1_4, khr_acceleration_structure],
        requires_instance_extensions: [],
    },
    khr_relaxed_block_layout => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_relaxed_block_layout.html)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_relaxed_block_layout",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_sampler_mirror_clamp_to_edge => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_sampler_mirror_clamp_to_edge.html)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_KHR_sampler_mirror_clamp_to_edge",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_sampler_ycbcr_conversion => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_sampler_ycbcr_conversion.html)
			- Requires device extensions: [`khr_maintenance1`](crate::device::DeviceExtensions::khr_maintenance1), [`khr_bind_memory2`](crate::device::DeviceExtensions::khr_bind_memory2), [`khr_get_memory_requirements2`](crate::device::DeviceExtensions::khr_get_memory_requirements2)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_sampler_ycbcr_conversion",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_maintenance1, khr_bind_memory2, khr_get_memory_requirements2],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_separate_depth_stencil_layouts => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_separate_depth_stencil_layouts.html)
			- Requires device extensions: [`khr_create_renderpass2`](crate::device::DeviceExtensions::khr_create_renderpass2)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_KHR_separate_depth_stencil_layouts",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_create_renderpass2],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_shader_atomic_int64 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_shader_atomic_int64.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_KHR_shader_atomic_int64",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_shader_clock => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_shader_clock.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_KHR_shader_clock",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_shader_draw_parameters => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_shader_draw_parameters.html)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_shader_draw_parameters",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_shader_float16_int8 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_shader_float16_int8.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_KHR_shader_float16_int8",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_shader_float_controls => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_shader_float_controls.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_KHR_shader_float_controls",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_shader_non_semantic_info => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_shader_non_semantic_info.html)
		",
        raw: b"VK_KHR_shader_non_semantic_info",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_shader_subgroup_extended_types => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_shader_subgroup_extended_types.html)
			- Requires Vulkan 1.1
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_KHR_shader_subgroup_extended_types",
        requires_core: Version::V1_1,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_shader_terminate_invocation => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_shader_terminate_invocation.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_KHR_shader_terminate_invocation",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_shared_presentable_image => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_shared_presentable_image.html)
			- Requires device extensions: [`khr_swapchain`](crate::device::DeviceExtensions::khr_swapchain)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2), [`khr_get_surface_capabilities2`](crate::instance::InstanceExtensions::khr_get_surface_capabilities2)
		",
        raw: b"VK_KHR_shared_presentable_image",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_swapchain],
        requires_instance_extensions: [khr_get_physical_device_properties2, khr_get_surface_capabilities2],
    },
    khr_spirv_1_4 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_spirv_1_4.html)
			- Requires Vulkan 1.1
			- Requires device extensions: [`khr_shader_float_controls`](crate::device::DeviceExtensions::khr_shader_float_controls)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_KHR_spirv_1_4",
        requires_core: Version::V1_1,
        requires_device_extensions: [khr_shader_float_controls],
        requires_instance_extensions: [],
    },
    khr_storage_buffer_storage_class => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_storage_buffer_storage_class.html)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_storage_buffer_storage_class",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_swapchain => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_swapchain.html)
			- Requires instance extensions: [`khr_surface`](crate::instance::InstanceExtensions::khr_surface)
		",
        raw: b"VK_KHR_swapchain",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_surface],
    },
    khr_swapchain_mutable_format => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_swapchain_mutable_format.html)
			- Requires device extensions: [`khr_swapchain`](crate::device::DeviceExtensions::khr_swapchain), [`khr_maintenance2`](crate::device::DeviceExtensions::khr_maintenance2), [`khr_image_format_list`](crate::device::DeviceExtensions::khr_image_format_list)
		",
        raw: b"VK_KHR_swapchain_mutable_format",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_swapchain, khr_maintenance2, khr_image_format_list],
        requires_instance_extensions: [],
    },
    khr_timeline_semaphore => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_timeline_semaphore.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_KHR_timeline_semaphore",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_uniform_buffer_standard_layout => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_uniform_buffer_standard_layout.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_KHR_uniform_buffer_standard_layout",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_variable_pointers => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_variable_pointers.html)
			- Requires device extensions: [`khr_storage_buffer_storage_class`](crate::device::DeviceExtensions::khr_storage_buffer_storage_class)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.1
		",
        raw: b"VK_KHR_variable_pointers",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_storage_buffer_storage_class],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_vulkan_memory_model => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_vulkan_memory_model.html)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_KHR_vulkan_memory_model",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    khr_win32_keyed_mutex => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_win32_keyed_mutex.html)
			- Requires device extensions: [`khr_external_memory_win32`](crate::device::DeviceExtensions::khr_external_memory_win32)
		",
        raw: b"VK_KHR_win32_keyed_mutex",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_external_memory_win32],
        requires_instance_extensions: [],
    },
    khr_workgroup_memory_explicit_layout => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_workgroup_memory_explicit_layout.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_KHR_workgroup_memory_explicit_layout",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    khr_zero_initialize_workgroup_memory => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_zero_initialize_workgroup_memory.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_KHR_zero_initialize_workgroup_memory",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_4444_formats => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_4444_formats.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_4444_formats",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_astc_decode_mode => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_astc_decode_mode.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_astc_decode_mode",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_blend_operation_advanced => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_blend_operation_advanced.html)
		",
        raw: b"VK_EXT_blend_operation_advanced",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_buffer_device_address => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_buffer_device_address.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Deprecated by [`khr_buffer_device_address`](crate::device::DeviceExtensions::khr_buffer_device_address)
		",
        raw: b"VK_EXT_buffer_device_address",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_calibrated_timestamps => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_calibrated_timestamps.html)
		",
        raw: b"VK_EXT_calibrated_timestamps",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_conditional_rendering => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_conditional_rendering.html)
		",
        raw: b"VK_EXT_conditional_rendering",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_conservative_rasterization => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_conservative_rasterization.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_conservative_rasterization",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_custom_border_color => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_custom_border_color.html)
		",
        raw: b"VK_EXT_custom_border_color",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_debug_marker => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_debug_marker.html)
			- Requires instance extensions: [`ext_debug_report`](crate::instance::InstanceExtensions::ext_debug_report)
			- Promoted to [`ext_debug_utils`](crate::instance::InstanceExtensions::ext_debug_utils)
		",
        raw: b"VK_EXT_debug_marker",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [ext_debug_report],
    },
    ext_depth_clip_enable => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_depth_clip_enable.html)
		",
        raw: b"VK_EXT_depth_clip_enable",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_depth_range_unrestricted => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_depth_range_unrestricted.html)
		",
        raw: b"VK_EXT_depth_range_unrestricted",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_descriptor_indexing => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_descriptor_indexing.html)
			- Requires device extensions: [`khr_maintenance3`](crate::device::DeviceExtensions::khr_maintenance3)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_EXT_descriptor_indexing",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_maintenance3],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_device_memory_report => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_device_memory_report.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_device_memory_report",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_discard_rectangles => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_discard_rectangles.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_discard_rectangles",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_display_control => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_display_control.html)
			- Requires device extensions: [`khr_swapchain`](crate::device::DeviceExtensions::khr_swapchain)
			- Requires instance extensions: [`ext_display_surface_counter`](crate::instance::InstanceExtensions::ext_display_surface_counter)
		",
        raw: b"VK_EXT_display_control",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_swapchain],
        requires_instance_extensions: [ext_display_surface_counter],
    },
    ext_extended_dynamic_state => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_extended_dynamic_state.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_extended_dynamic_state",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_external_memory_dma_buf => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_external_memory_dma_buf.html)
			- Requires device extensions: [`khr_external_memory_fd`](crate::device::DeviceExtensions::khr_external_memory_fd)
		",
        raw: b"VK_EXT_external_memory_dma_buf",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_external_memory_fd],
        requires_instance_extensions: [],
    },
    ext_external_memory_host => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_external_memory_host.html)
			- Requires device extensions: [`khr_external_memory`](crate::device::DeviceExtensions::khr_external_memory)
		",
        raw: b"VK_EXT_external_memory_host",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_external_memory],
        requires_instance_extensions: [],
    },
    ext_filter_cubic => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_filter_cubic.html)
		",
        raw: b"VK_EXT_filter_cubic",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_fragment_density_map => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_fragment_density_map.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_fragment_density_map",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_fragment_density_map2 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_fragment_density_map2.html)
			- Requires device extensions: [`ext_fragment_density_map`](crate::device::DeviceExtensions::ext_fragment_density_map)
		",
        raw: b"VK_EXT_fragment_density_map2",
        requires_core: Version::V1_0,
        requires_device_extensions: [ext_fragment_density_map],
        requires_instance_extensions: [],
    },
    ext_fragment_shader_interlock => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_fragment_shader_interlock.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_fragment_shader_interlock",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_full_screen_exclusive => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_full_screen_exclusive.html)
			- Requires device extensions: [`khr_swapchain`](crate::device::DeviceExtensions::khr_swapchain)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2), [`khr_surface`](crate::instance::InstanceExtensions::khr_surface), [`khr_get_surface_capabilities2`](crate::instance::InstanceExtensions::khr_get_surface_capabilities2)
		",
        raw: b"VK_EXT_full_screen_exclusive",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_swapchain],
        requires_instance_extensions: [khr_get_physical_device_properties2, khr_surface, khr_get_surface_capabilities2],
    },
    ext_global_priority => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_global_priority.html)
		",
        raw: b"VK_EXT_global_priority",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_hdr_metadata => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_hdr_metadata.html)
			- Requires device extensions: [`khr_swapchain`](crate::device::DeviceExtensions::khr_swapchain)
		",
        raw: b"VK_EXT_hdr_metadata",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_swapchain],
        requires_instance_extensions: [],
    },
    ext_host_query_reset => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_host_query_reset.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_EXT_host_query_reset",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_image_drm_format_modifier => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_image_drm_format_modifier.html)
			- Requires device extensions: [`khr_bind_memory2`](crate::device::DeviceExtensions::khr_bind_memory2), [`khr_image_format_list`](crate::device::DeviceExtensions::khr_image_format_list), [`khr_sampler_ycbcr_conversion`](crate::device::DeviceExtensions::khr_sampler_ycbcr_conversion)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_image_drm_format_modifier",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_bind_memory2, khr_image_format_list, khr_sampler_ycbcr_conversion],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_image_robustness => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_image_robustness.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_image_robustness",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_index_type_uint8 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_index_type_uint8.html)
		",
        raw: b"VK_EXT_index_type_uint8",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_inline_uniform_block => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_inline_uniform_block.html)
			- Requires device extensions: [`khr_maintenance1`](crate::device::DeviceExtensions::khr_maintenance1)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_inline_uniform_block",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_maintenance1],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_line_rasterization => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_line_rasterization.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_line_rasterization",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_memory_budget => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_memory_budget.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_memory_budget",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_memory_priority => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_memory_priority.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_memory_priority",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_pci_bus_info => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_pci_bus_info.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_pci_bus_info",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_pipeline_creation_cache_control => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_pipeline_creation_cache_control.html)
		",
        raw: b"VK_EXT_pipeline_creation_cache_control",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_pipeline_creation_feedback => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_pipeline_creation_feedback.html)
		",
        raw: b"VK_EXT_pipeline_creation_feedback",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_post_depth_coverage => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_post_depth_coverage.html)
		",
        raw: b"VK_EXT_post_depth_coverage",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_private_data => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_private_data.html)
		",
        raw: b"VK_EXT_private_data",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_queue_family_foreign => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_queue_family_foreign.html)
			- Requires device extensions: [`khr_external_memory`](crate::device::DeviceExtensions::khr_external_memory)
		",
        raw: b"VK_EXT_queue_family_foreign",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_external_memory],
        requires_instance_extensions: [],
    },
    ext_robustness2 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_robustness2.html)
		",
        raw: b"VK_EXT_robustness2",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_sample_locations => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_sample_locations.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_sample_locations",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_sampler_filter_minmax => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_sampler_filter_minmax.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_EXT_sampler_filter_minmax",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_scalar_block_layout => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_scalar_block_layout.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_EXT_scalar_block_layout",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_separate_stencil_usage => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_separate_stencil_usage.html)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_EXT_separate_stencil_usage",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_shader_atomic_float => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_shader_atomic_float.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_shader_atomic_float",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_shader_demote_to_helper_invocation => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_shader_demote_to_helper_invocation.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_shader_demote_to_helper_invocation",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_shader_image_atomic_int64 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_shader_image_atomic_int64.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_shader_image_atomic_int64",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_shader_stencil_export => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_shader_stencil_export.html)
		",
        raw: b"VK_EXT_shader_stencil_export",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_shader_subgroup_ballot => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_shader_subgroup_ballot.html)
			- Deprecated by Vulkan 1.2
		",
        raw: b"VK_EXT_shader_subgroup_ballot",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_shader_subgroup_vote => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_shader_subgroup_vote.html)
			- Deprecated by Vulkan 1.1
		",
        raw: b"VK_EXT_shader_subgroup_vote",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_shader_viewport_index_layer => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_shader_viewport_index_layer.html)
			- Promoted to Vulkan 1.2
		",
        raw: b"VK_EXT_shader_viewport_index_layer",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_subgroup_size_control => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_subgroup_size_control.html)
			- Requires Vulkan 1.1
		",
        raw: b"VK_EXT_subgroup_size_control",
        requires_core: Version::V1_1,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_texel_buffer_alignment => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_texel_buffer_alignment.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_texel_buffer_alignment",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_texture_compression_astc_hdr => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_texture_compression_astc_hdr.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_texture_compression_astc_hdr",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_tooling_info => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_tooling_info.html)
		",
        raw: b"VK_EXT_tooling_info",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_transform_feedback => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_transform_feedback.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_transform_feedback",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_validation_cache => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_validation_cache.html)
		",
        raw: b"VK_EXT_validation_cache",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    ext_vertex_attribute_divisor => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_vertex_attribute_divisor.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_EXT_vertex_attribute_divisor",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    ext_ycbcr_image_arrays => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_ycbcr_image_arrays.html)
			- Requires device extensions: [`khr_sampler_ycbcr_conversion`](crate::device::DeviceExtensions::khr_sampler_ycbcr_conversion)
		",
        raw: b"VK_EXT_ycbcr_image_arrays",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_sampler_ycbcr_conversion],
        requires_instance_extensions: [],
    },
    amd_buffer_marker => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_buffer_marker.html)
		",
        raw: b"VK_AMD_buffer_marker",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_device_coherent_memory => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_device_coherent_memory.html)
		",
        raw: b"VK_AMD_device_coherent_memory",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_display_native_hdr => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_display_native_hdr.html)
			- Requires device extensions: [`khr_swapchain`](crate::device::DeviceExtensions::khr_swapchain)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2), [`khr_get_surface_capabilities2`](crate::instance::InstanceExtensions::khr_get_surface_capabilities2)
		",
        raw: b"VK_AMD_display_native_hdr",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_swapchain],
        requires_instance_extensions: [khr_get_physical_device_properties2, khr_get_surface_capabilities2],
    },
    amd_draw_indirect_count => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_draw_indirect_count.html)
			- Promoted to [`khr_draw_indirect_count`](crate::device::DeviceExtensions::khr_draw_indirect_count)
		",
        raw: b"VK_AMD_draw_indirect_count",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_gcn_shader => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_gcn_shader.html)
		",
        raw: b"VK_AMD_gcn_shader",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_gpu_shader_half_float => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_gpu_shader_half_float.html)
			- Deprecated by [`khr_shader_float16_int8`](crate::device::DeviceExtensions::khr_shader_float16_int8)
		",
        raw: b"VK_AMD_gpu_shader_half_float",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_gpu_shader_int16 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_gpu_shader_int16.html)
			- Deprecated by [`khr_shader_float16_int8`](crate::device::DeviceExtensions::khr_shader_float16_int8)
		",
        raw: b"VK_AMD_gpu_shader_int16",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_memory_overallocation_behavior => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_memory_overallocation_behavior.html)
		",
        raw: b"VK_AMD_memory_overallocation_behavior",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_mixed_attachment_samples => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_mixed_attachment_samples.html)
		",
        raw: b"VK_AMD_mixed_attachment_samples",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_pipeline_compiler_control => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_pipeline_compiler_control.html)
		",
        raw: b"VK_AMD_pipeline_compiler_control",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_rasterization_order => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_rasterization_order.html)
		",
        raw: b"VK_AMD_rasterization_order",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_shader_ballot => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_shader_ballot.html)
		",
        raw: b"VK_AMD_shader_ballot",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_shader_core_properties => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_shader_core_properties.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_AMD_shader_core_properties",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    amd_shader_core_properties2 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_shader_core_properties2.html)
			- Requires device extensions: [`amd_shader_core_properties`](crate::device::DeviceExtensions::amd_shader_core_properties)
		",
        raw: b"VK_AMD_shader_core_properties2",
        requires_core: Version::V1_0,
        requires_device_extensions: [amd_shader_core_properties],
        requires_instance_extensions: [],
    },
    amd_shader_explicit_vertex_parameter => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_shader_explicit_vertex_parameter.html)
		",
        raw: b"VK_AMD_shader_explicit_vertex_parameter",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_shader_fragment_mask => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_shader_fragment_mask.html)
		",
        raw: b"VK_AMD_shader_fragment_mask",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_shader_image_load_store_lod => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_shader_image_load_store_lod.html)
		",
        raw: b"VK_AMD_shader_image_load_store_lod",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_shader_info => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_shader_info.html)
		",
        raw: b"VK_AMD_shader_info",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_shader_trinary_minmax => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_shader_trinary_minmax.html)
		",
        raw: b"VK_AMD_shader_trinary_minmax",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    amd_texture_gather_bias_lod => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_AMD_texture_gather_bias_lod.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_AMD_texture_gather_bias_lod",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    android_external_memory_android_hardware_buffer => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_ANDROID_external_memory_android_hardware_buffer.html)
			- Requires device extensions: [`khr_sampler_ycbcr_conversion`](crate::device::DeviceExtensions::khr_sampler_ycbcr_conversion), [`khr_external_memory`](crate::device::DeviceExtensions::khr_external_memory), [`ext_queue_family_foreign`](crate::device::DeviceExtensions::ext_queue_family_foreign), [`khr_dedicated_allocation`](crate::device::DeviceExtensions::khr_dedicated_allocation)
		",
        raw: b"VK_ANDROID_external_memory_android_hardware_buffer",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_sampler_ycbcr_conversion, khr_external_memory, ext_queue_family_foreign, khr_dedicated_allocation],
        requires_instance_extensions: [],
    },
    ggp_frame_token => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_GGP_frame_token.html)
			- Requires device extensions: [`khr_swapchain`](crate::device::DeviceExtensions::khr_swapchain)
			- Requires instance extensions: [`ggp_stream_descriptor_surface`](crate::instance::InstanceExtensions::ggp_stream_descriptor_surface)
		",
        raw: b"VK_GGP_frame_token",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_swapchain],
        requires_instance_extensions: [ggp_stream_descriptor_surface],
    },
    google_decorate_string => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_GOOGLE_decorate_string.html)
		",
        raw: b"VK_GOOGLE_decorate_string",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    google_display_timing => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_GOOGLE_display_timing.html)
			- Requires device extensions: [`khr_swapchain`](crate::device::DeviceExtensions::khr_swapchain)
		",
        raw: b"VK_GOOGLE_display_timing",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_swapchain],
        requires_instance_extensions: [],
    },
    google_hlsl_functionality1 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_GOOGLE_hlsl_functionality1.html)
		",
        raw: b"VK_GOOGLE_hlsl_functionality1",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    google_user_type => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_GOOGLE_user_type.html)
		",
        raw: b"VK_GOOGLE_user_type",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    img_filter_cubic => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_IMG_filter_cubic.html)
		",
        raw: b"VK_IMG_filter_cubic",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    img_format_pvrtc => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_IMG_format_pvrtc.html)
		",
        raw: b"VK_IMG_format_pvrtc",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    intel_performance_query => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_INTEL_performance_query.html)
		",
        raw: b"VK_INTEL_performance_query",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    intel_shader_integer_functions2 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_INTEL_shader_integer_functions2.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_INTEL_shader_integer_functions2",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nvx_image_view_handle => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NVX_image_view_handle.html)
		",
        raw: b"VK_NVX_image_view_handle",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nvx_multiview_per_view_attributes => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NVX_multiview_per_view_attributes.html)
			- Requires device extensions: [`khr_multiview`](crate::device::DeviceExtensions::khr_multiview)
		",
        raw: b"VK_NVX_multiview_per_view_attributes",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_multiview],
        requires_instance_extensions: [],
    },
    nv_acquire_winrt_display => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_acquire_winrt_display.html)
			- Requires instance extensions: [`ext_direct_mode_display`](crate::instance::InstanceExtensions::ext_direct_mode_display)
		",
        raw: b"VK_NV_acquire_winrt_display",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [ext_direct_mode_display],
    },
    nv_clip_space_w_scaling => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_clip_space_w_scaling.html)
		",
        raw: b"VK_NV_clip_space_w_scaling",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_compute_shader_derivatives => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_compute_shader_derivatives.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_NV_compute_shader_derivatives",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_cooperative_matrix => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_cooperative_matrix.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_NV_cooperative_matrix",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_corner_sampled_image => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_corner_sampled_image.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_NV_corner_sampled_image",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_coverage_reduction_mode => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_coverage_reduction_mode.html)
			- Requires device extensions: [`nv_framebuffer_mixed_samples`](crate::device::DeviceExtensions::nv_framebuffer_mixed_samples)
		",
        raw: b"VK_NV_coverage_reduction_mode",
        requires_core: Version::V1_0,
        requires_device_extensions: [nv_framebuffer_mixed_samples],
        requires_instance_extensions: [],
    },
    nv_dedicated_allocation => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_dedicated_allocation.html)
			- Deprecated by [`khr_dedicated_allocation`](crate::device::DeviceExtensions::khr_dedicated_allocation)
		",
        raw: b"VK_NV_dedicated_allocation",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_dedicated_allocation_image_aliasing => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_dedicated_allocation_image_aliasing.html)
			- Requires device extensions: [`khr_dedicated_allocation`](crate::device::DeviceExtensions::khr_dedicated_allocation)
		",
        raw: b"VK_NV_dedicated_allocation_image_aliasing",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_dedicated_allocation],
        requires_instance_extensions: [],
    },
    nv_device_diagnostic_checkpoints => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_device_diagnostic_checkpoints.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_NV_device_diagnostic_checkpoints",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_device_diagnostics_config => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_device_diagnostics_config.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_NV_device_diagnostics_config",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_device_generated_commands => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_device_generated_commands.html)
			- Requires Vulkan 1.1
		",
        raw: b"VK_NV_device_generated_commands",
        requires_core: Version::V1_1,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_external_memory => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_external_memory.html)
			- Requires instance extensions: [`nv_external_memory_capabilities`](crate::instance::InstanceExtensions::nv_external_memory_capabilities)
			- Deprecated by [`khr_external_memory`](crate::device::DeviceExtensions::khr_external_memory)
		",
        raw: b"VK_NV_external_memory",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [nv_external_memory_capabilities],
    },
    nv_external_memory_win32 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_external_memory_win32.html)
			- Requires device extensions: [`nv_external_memory`](crate::device::DeviceExtensions::nv_external_memory)
			- Deprecated by [`khr_external_memory_win32`](crate::device::DeviceExtensions::khr_external_memory_win32)
		",
        raw: b"VK_NV_external_memory_win32",
        requires_core: Version::V1_0,
        requires_device_extensions: [nv_external_memory],
        requires_instance_extensions: [],
    },
    nv_fill_rectangle => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_fill_rectangle.html)
		",
        raw: b"VK_NV_fill_rectangle",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_fragment_coverage_to_color => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_fragment_coverage_to_color.html)
		",
        raw: b"VK_NV_fragment_coverage_to_color",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_fragment_shader_barycentric => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_fragment_shader_barycentric.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_NV_fragment_shader_barycentric",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_fragment_shading_rate_enums => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_fragment_shading_rate_enums.html)
			- Requires device extensions: [`khr_fragment_shading_rate`](crate::device::DeviceExtensions::khr_fragment_shading_rate)
		",
        raw: b"VK_NV_fragment_shading_rate_enums",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_fragment_shading_rate],
        requires_instance_extensions: [],
    },
    nv_framebuffer_mixed_samples => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_framebuffer_mixed_samples.html)
		",
        raw: b"VK_NV_framebuffer_mixed_samples",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_geometry_shader_passthrough => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_geometry_shader_passthrough.html)
		",
        raw: b"VK_NV_geometry_shader_passthrough",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_glsl_shader => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_glsl_shader.html)
			- Deprecated without a replacement
		",
        raw: b"VK_NV_glsl_shader",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_mesh_shader => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_mesh_shader.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_NV_mesh_shader",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_ray_tracing => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_ray_tracing.html)
			- Requires device extensions: [`khr_get_memory_requirements2`](crate::device::DeviceExtensions::khr_get_memory_requirements2)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_NV_ray_tracing",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_get_memory_requirements2],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_representative_fragment_test => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_representative_fragment_test.html)
		",
        raw: b"VK_NV_representative_fragment_test",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_sample_mask_override_coverage => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_sample_mask_override_coverage.html)
		",
        raw: b"VK_NV_sample_mask_override_coverage",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_scissor_exclusive => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_scissor_exclusive.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_NV_scissor_exclusive",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_shader_image_footprint => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_shader_image_footprint.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_NV_shader_image_footprint",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_shader_sm_builtins => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_shader_sm_builtins.html)
			- Requires Vulkan 1.1
		",
        raw: b"VK_NV_shader_sm_builtins",
        requires_core: Version::V1_1,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_shader_subgroup_partitioned => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_shader_subgroup_partitioned.html)
			- Requires Vulkan 1.1
		",
        raw: b"VK_NV_shader_subgroup_partitioned",
        requires_core: Version::V1_1,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_shading_rate_image => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_shading_rate_image.html)
			- Requires instance extensions: [`khr_get_physical_device_properties2`](crate::instance::InstanceExtensions::khr_get_physical_device_properties2)
		",
        raw: b"VK_NV_shading_rate_image",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [khr_get_physical_device_properties2],
    },
    nv_viewport_array2 => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_viewport_array2.html)
		",
        raw: b"VK_NV_viewport_array2",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_viewport_swizzle => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_viewport_swizzle.html)
		",
        raw: b"VK_NV_viewport_swizzle",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    nv_win32_keyed_mutex => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_NV_win32_keyed_mutex.html)
			- Requires device extensions: [`nv_external_memory_win32`](crate::device::DeviceExtensions::nv_external_memory_win32)
			- Promoted to [`khr_win32_keyed_mutex`](crate::device::DeviceExtensions::khr_win32_keyed_mutex)
		",
        raw: b"VK_NV_win32_keyed_mutex",
        requires_core: Version::V1_0,
        requires_device_extensions: [nv_external_memory_win32],
        requires_instance_extensions: [],
    },
    qcom_render_pass_shader_resolve => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_QCOM_render_pass_shader_resolve.html)
		",
        raw: b"VK_QCOM_render_pass_shader_resolve",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    qcom_render_pass_store_ops => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_QCOM_render_pass_store_ops.html)
		",
        raw: b"VK_QCOM_render_pass_store_ops",
        requires_core: Version::V1_0,
        requires_device_extensions: [],
        requires_instance_extensions: [],
    },
    qcom_render_pass_transform => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_QCOM_render_pass_transform.html)
			- Requires device extensions: [`khr_swapchain`](crate::device::DeviceExtensions::khr_swapchain)
			- Requires instance extensions: [`khr_surface`](crate::instance::InstanceExtensions::khr_surface)
		",
        raw: b"VK_QCOM_render_pass_transform",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_swapchain],
        requires_instance_extensions: [khr_surface],
    },
    qcom_rotated_copy_commands => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_QCOM_rotated_copy_commands.html)
			- Requires device extensions: [`khr_swapchain`](crate::device::DeviceExtensions::khr_swapchain), [`khr_copy_commands2`](crate::device::DeviceExtensions::khr_copy_commands2)
		",
        raw: b"VK_QCOM_rotated_copy_commands",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_swapchain, khr_copy_commands2],
        requires_instance_extensions: [],
    },
    valve_mutable_descriptor_type => {
        doc: "
			- [Vulkan documentation](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_VALVE_mutable_descriptor_type.html)
			- Requires device extensions: [`khr_maintenance3`](crate::device::DeviceExtensions::khr_maintenance3)
		",
        raw: b"VK_VALVE_mutable_descriptor_type",
        requires_core: Version::V1_0,
        requires_device_extensions: [khr_maintenance3],
        requires_instance_extensions: [],
    },
}
