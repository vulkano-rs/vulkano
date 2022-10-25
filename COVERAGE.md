# Vulkano's coverage of Vulkan

Coverage of support for Vulkan core features and extensions in Vulkano.

"Partially supported" includes core features and extensions that have some support in Vulkano, but are not fully implemented. A list of unimplemented features is given.

## Vulkan 1.0

### Unsupported

- `vkGetImageSubresourceLayout` (used, but not exposed to the user)
- `VkAllocationCallbacks`
- `VkPipelineCreateFlags`
- Possibly more?

## Vulkan 1.1

### Fully supported

- [`VK_KHR_16bit_storage`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_16bit_storage.html)
- [`VK_KHR_dedicated_allocation`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_dedicated_allocation.html)
- [`VK_KHR_get_memory_requirements2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_get_memory_requirements2.html)
- [`VK_KHR_get_physical_device_properties2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_get_physical_device_properties2.html)
- [`VK_KHR_external_fence`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_fence.html)
- [`VK_KHR_external_fence_capabilities`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_fence_capabilities.html)
- [`VK_KHR_external_memory`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_memory.html)
- [`VK_KHR_external_memory_capabilities`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_memory_capabilities.html)
- [`VK_KHR_external_semaphore`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_semaphore.html)
- [`VK_KHR_external_semaphore_capabilities`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_semaphore_capabilities.html)
- [`VK_KHR_maintenance1`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_maintenance1.html)
- [`VK_KHR_multiview`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_multiview.html)
- [`VK_KHR_relaxed_block_layout`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_relaxed_block_layout.html)
- [`VK_KHR_shader_draw_parameters`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_shader_draw_parameters.html)
- [`VK_KHR_storage_buffer_storage_class`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_storage_buffer_storage_class.html)
- [`VK_KHR_variable_pointers`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_variable_pointers.html)
- `vkEnumerateInstanceVersion`

### Partially supported

- [`VK_KHR_maintenance2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_maintenance2.html)
	- `VkPipelineTessellationDomainOriginStateCreateInfoKHR`
	- `VK_IMAGE_CREATE_EXTENDED_USAGE_BIT`
- [`VK_KHR_sampler_ycbcr_conversion`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_sampler_ycbcr_conversion.html)
	- `VkBindImagePlaneMemoryInfoKHR`
	- `VkSamplerYcbcrConversionImageFormatPropertiesKHR`
	- `VkImagePlaneMemoryRequirementsInfoKHR`

### Unsupported

- [`VK_KHR_bind_memory2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_bind_memory2.html)
- [`VK_KHR_descriptor_update_template`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_descriptor_update_template.html)
- [`VK_KHR_device_group`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_device_group.html)
- [`VK_KHR_device_group_creation`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_device_group_creation.html)
- [`VK_KHR_maintenance3`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_maintenance3.html)
- Group operations, subgroup scope
- Protected memory

## Vulkan 1.2

### Fully supported

- [`VK_KHR_8bit_storage`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_8bit_storage.html)
- [`VK_KHR_driver_properties`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_driver_properties.html)
- [`VK_KHR_sampler_mirror_clamp_to_edge`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_sampler_mirror_clamp_to_edge.html)
- [`VK_KHR_spirv_1_4`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_spirv_1_4.html)
- [`VK_KHR_shader_atomic_int64`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_shader_atomic_int64.html)
- [`VK_KHR_shader_float16_int8`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_shader_float16_int8.html)
- [`VK_KHR_shader_float_controls`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_shader_float_controls.html)
- [`VK_KHR_shader_subgroup_extended_types`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_shader_subgroup_extended_types.html)
- [`VK_KHR_uniform_buffer_standard_layout`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_uniform_buffer_standard_layout.html)
- [`VK_KHR_vulkan_memory_model`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_vulkan_memory_model.html)
- [`VK_EXT_sampler_filter_minmax`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_sampler_filter_minmax.html)
- [`VK_EXT_scalar_block_layout`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_scalar_block_layout.html)
- [`VK_EXT_separate_stencil_usage`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_separate_stencil_usage.html)
- [`VK_EXT_shader_viewport_index_layer`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_shader_viewport_index_layer.html)
- SPIR-V 1.4 and 1.5

### Partially supported

- [`VK_KHR_buffer_device_address`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_buffer_device_address.html)
	- Only the deprecated EXT version is supported.
	- `vkGetBufferOpaqueCaptureAddressKHR` / `VkBufferOpaqueCaptureAddressCreateInfoKHR`
	- `vkGetDeviceMemoryOpaqueCaptureAddressKHR` / `VkMemoryOpaqueCaptureAddressAllocateInfoKHR`
- [`VK_EXT_descriptor_indexing`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_descriptor_indexing.html)
	- `VkDescriptorSetVariableDescriptorCountLayoutSupportEXT`
	- `VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT`
	- `VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT` / `VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT_EXT` / `VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT_EXT`
	- `VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT_EXT`

### Unsupported

- [`VK_KHR_depth_stencil_resolve`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_depth_stencil_resolve.html)
- [`VK_KHR_draw_indirect_count`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_draw_indirect_count.html)
- [`VK_KHR_image_format_list`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_image_format_list.html)
- [`VK_KHR_imageless_framebuffer`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_imageless_framebuffer.html)
- [`VK_KHR_separate_depth_stencil_layouts`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_separate_depth_stencil_layouts.html)
- [`VK_KHR_timeline_semaphore`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_timeline_semaphore.html)
- [`VK_EXT_host_query_reset`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_host_query_reset.html)
- `framebufferIntegerColorSampleCounts`

## Vulkan 1.3

### Fully supported

- [`VK_KHR_copy_commands2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_copy_commands2.html)
- [`VK_KHR_create_renderpass2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_create_renderpass2.html)
- [`VK_KHR_format_feature_flags2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_format_feature_flags2.html)
- [`VK_KHR_shader_integer_dot_product`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_shader_integer_dot_product.html)
- [`VK_KHR_shader_non_semantic_info`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_shader_non_semantic_info.html)
- [`VK_KHR_shader_terminate_invocation`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_shader_terminate_invocation.html)
- [`VK_KHR_zero_initialize_workgroup_memory`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_zero_initialize_workgroup_memory.html)
- [`VK_EXT_4444_formats`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_4444_formats.html)
- [`VK_EXT_extended_dynamic_state`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_extended_dynamic_state.html)
- [`VK_EXT_extended_dynamic_state2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_extended_dynamic_state2.html)
- [`VK_EXT_shader_demote_to_helper_invocation`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_shader_demote_to_helper_invocation.html)
- [`VK_EXT_texel_buffer_alignment`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_texel_buffer_alignment.html)
- [`VK_EXT_texture_compression_astc_hdr`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_texture_compression_astc_hdr.html)
- [`VK_EXT_tooling_info`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_tooling_info.html)
- [`VK_EXT_ycbcr_2plane_444_formats`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_ycbcr_2plane_444_formats.html)
- SPIR-V 1.6

### Partially supported

- [`VK_KHR_dynamic_rendering`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_dynamic_rendering.html)
	- Suspend/resume
- [`VK_KHR_synchronization2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_synchronization2.html)
	- `vkQueueSubmit2KHR` / `VkSemaphoreSubmitInfoKHR` (missing parameters related to device groups)

### Unsupported

- [`VK_KHR_maintenance4`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_maintenance4.html)
- [`VK_EXT_image_robustness`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_image_robustness.html)
- [`VK_EXT_inline_uniform_block`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_inline_uniform_block.html)
- [`VK_EXT_pipeline_creation_cache_control`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_pipeline_creation_cache_control.html)
- [`VK_EXT_pipeline_creation_feedback`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_pipeline_creation_feedback.html)
- [`VK_EXT_private_data`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_private_data.html)
- [`VK_EXT_subgroup_size_control`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_subgroup_size_control.html)

## Extensions not promoted to core

### Fully supported

- [`VK_KHR_android_surface`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_android_surface.html)
- [`VK_KHR_external_fence_fd`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_fence_fd.html)
- [`VK_KHR_external_fence_win32`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_fence_win32.html)
- [`VK_KHR_external_memory_fd`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_memory_fd.html)
- [`VK_KHR_external_semaphore_fd`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_semaphore_fd.html)
- [`VK_KHR_get_surface_capabilities2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_get_surface_capabilities2.html)
- [`VK_KHR_incremental_present`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_incremental_present.html)
- [`VK_KHR_portability_enumeration`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_portability_enumeration.html)
- [`VK_KHR_present_id`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_present_id.html)
- [`VK_KHR_present_wait`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_present_wait.html)
- [`VK_KHR_push_descriptor`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_push_descriptor.html)
- [`VK_KHR_surface`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_surface.html)
- [`VK_KHR_surface_protected_capabilities`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_surface_protected_capabilities.html)
- [`VK_KHR_swapchain`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_swapchain.html)
- [`VK_KHR_wayland_surface`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_wayland_surface.html)
- [`VK_KHR_win32_surface`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_win32_surface.html)
- [`VK_KHR_xcb_surface`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_xcb_surface.html)
- [`VK_KHR_xlib_surface`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_xlib_surface.html)
- [`VK_EXT_color_write_enable`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_color_write_enable.html)
- [`VK_EXT_depth_range_unrestricted`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_depth_range_unrestricted.html)
- [`VK_EXT_directfb_surface`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_directfb_surface.html)
- [`VK_EXT_discard_rectangles`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_discard_rectangles.html)
- [`VK_EXT_external_memory_dma_buf`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_external_memory_dma_buf.html)
- [`VK_EXT_filter_cubic`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_filter_cubic.html)
- [`VK_EXT_headless_surface`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_headless_surface.html)
- [`VK_EXT_index_type_uint8`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_index_type_uint8.html)
- [`VK_EXT_line_rasterization`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_line_rasterization.html)
- [`VK_EXT_metal_surface`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_metal_surface.html)
- [`VK_EXT_primitive_topology_list_restart`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_primitive_topology_list_restart.html)
- [`VK_EXT_robustness2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_robustness2.html)
- [`VK_EXT_swapchain_colorspace`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_swapchain_colorspace.html)
- [`VK_EXT_validation_features`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_validation_features.html)
- [`VK_EXT_vertex_attribute_divisor`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_vertex_attribute_divisor.html)
- [`VK_EXT_ycbcr_image_arrays`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_ycbcr_image_arrays.html)
- [`VK_FUCHSIA_external_semaphore`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_FUCHSIA_external_semaphore.html)
- [`VK_FUCHSIA_imagepipe_surface`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_FUCHSIA_imagepipe_surface.html)
- [`VK_GGP_stream_descriptor_surface`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_GGP_stream_descriptor_surface.html)
- [`VK_MVK_ios_surface`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_FUCHSIA_imagepipe_surface.html) (deprecated)
- [`VK_MVK_macos_surface`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_MVK_macos_surface.html) (deprecated)
- [`VK_QNX_screen_surface`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_QNX_screen_surface.html)

### Partially supported

- [`VK_KHR_display`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_display.html)
	- `vkCreateDisplayModeKHR` / `VkDisplayModeCreateInfoKHR` / `VkDisplayModeParametersKHR`
	- `vkGetDisplayPlaneCapabilitiesKHR` / `VkDisplayPlaneCapabilitiesKHR`
- [`VK_KHR_external_memory_win32`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_memory_win32.html)
	- `vkGetMemoryWin32HandleKHR`
	- `vkGetMemoryWin32HandlePropertiesKHR`
	- `VkExportMemoryWin32HandleInfoKHR`
- [`VK_KHR_external_semaphore_win32`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_semaphore_win32.html)
	- `VkD3D12FenceSubmitInfoKHR`
- [`VK_KHR_portability_subset`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_portability_subset.html) (provisional)
	- Check for `tessellationIsolines`
	- Check for `tessellationPointMode`
- [`VK_EXT_buffer_device_address`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_buffer_device_address.html) (deprecated)
	- `VkBufferDeviceAddressCreateInfoEXT`
- [`VK_EXT_debug_utils`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_debug_utils.html)
	- `vkSetDebugUtilsObjectTagEXT`
	- `vkSubmitDebugUtilsMessageEXT`
	- `VkDebugUtilsMessengerCallbackDataEXT` (not all data exposed to callback)
	- `VkDebugUtilsObjectNameInfoEXT` extending `VkPipelineShaderStageCreateInfo`
- [`VK_EXT_full_screen_exclusive`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_full_screen_exclusive.html)
	- `vkGetPhysicalDeviceSurfacePresentModes2EXT`

### Unsupported

- [`VK_KHR_acceleration_structure`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_acceleration_structure.html)
- [`VK_KHR_deferred_host_operations`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_deferred_host_operations.html)
- [`VK_KHR_display_swapchain`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_display_swapchain.html)
- [`VK_KHR_fragment_shader_barycentric`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_fragment_shader_barycentric.html)
- [`VK_KHR_fragment_shading_rate`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_fragment_shading_rate.html)
- [`VK_KHR_get_display_properties2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_get_display_properties2.html)
- [`VK_KHR_global_priority`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_global_priority.html)
- [`VK_KHR_performance_query`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_performance_query.html)
- [`VK_KHR_pipeline_executable_properties`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_pipeline_executable_properties.html)
- [`VK_KHR_pipeline_library`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_pipeline_library.html)
- [`VK_KHR_ray_query`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_ray_query.html)
- [`VK_KHR_ray_tracing_maintenance1`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_ray_tracing_maintenance1.html)
- [`VK_KHR_ray_tracing_pipeline`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_ray_tracing_pipeline.html)
- [`VK_KHR_shader_clock`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_shader_clock.html)
- [`VK_KHR_shader_subgroup_uniform_control_flow`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_shader_subgroup_uniform_control_flow.html)
- [`VK_KHR_shared_presentable_image`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_shared_presentable_image.html)
- [`VK_KHR_swapchain_mutable_format`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_swapchain_mutable_format.html)
- [`VK_KHR_video_decode_queue`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_video_decode_queue.html) (provisional)
- [`VK_KHR_video_encode_queue`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_video_encode_queue.html) (provisional)
- [`VK_KHR_video_queue`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_video_queue.html) (provisional)
- [`VK_KHR_win32_keyed_mutex`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_win32_keyed_mutex.html)
- [`VK_KHR_workgroup_memory_explicit_layout`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_workgroup_memory_explicit_layout.html)
- [`VK_EXT_acquire_drm_display`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_acquire_drm_display.html)
- [`VK_EXT_acquire_xlib_display`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_acquire_xlib_display.html)
- [`VK_EXT_astc_decode_mode`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_astc_decode_mode.html)
- [`VK_EXT_attachment_feedback_loop_layout`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_attachment_feedback_loop_layout.html)
- [`VK_EXT_blend_operation_advanced`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_blend_operation_advanced.html)
- [`VK_EXT_border_color_swizzle`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_border_color_swizzle.html)
- [`VK_EXT_calibrated_timestamps`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_calibrated_timestamps.html)
- [`VK_EXT_conditional_rendering`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_conditional_rendering.html)
- [`VK_EXT_conservative_rasterization`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_conservative_rasterization.html)
- [`VK_EXT_custom_border_color`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_custom_border_color.html)
- [`VK_EXT_debug_marker`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_debug_marker.html) (promoted)
- [`VK_EXT_debug_report`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_debug_report.html) (deprecated)
- [`VK_EXT_depth_clamp_zero_one`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_depth_clamp_zero_one.html)
- [`VK_EXT_depth_clip_control`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_depth_clip_control.html)
- [`VK_EXT_depth_clip_enable`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_depth_clip_enable.html)
- [`VK_EXT_device_memory_report`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_device_memory_report.html)
- [`VK_EXT_direct_mode_display`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_direct_mode_display.html)
- [`VK_EXT_display_control`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_display_control.html)
- [`VK_EXT_display_surface_counter`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_display_surface_counter.html)
- [`VK_EXT_external_memory_host`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_external_memory_host.html)
- [`VK_EXT_fragment_density_map`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_fragment_density_map.html)
- [`VK_EXT_fragment_density_map2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_fragment_density_map2.html)
- [`VK_EXT_fragment_shader_interlock`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_fragment_shader_interlock.html)
- [`VK_EXT_global_priority`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_global_priority.html) (promoted)
- [`VK_EXT_global_priority_query`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_global_priority_query.html) (promoted)
- [`VK_EXT_graphics_pipeline_library`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_graphics_pipeline_library.html)
- [`VK_EXT_hdr_metadata`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_hdr_metadata.html)
- [`VK_EXT_image_2d_view_of_3d`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_image_2d_view_of_3d.html)
- [`VK_EXT_image_compression_control`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_image_compression_control.html)
- [`VK_EXT_image_compression_control_swapchain`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_image_compression_control_swapchain.html)
- [`VK_EXT_image_drm_format_modifier`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_image_drm_format_modifier.html)
- [`VK_EXT_image_view_min_lod`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_image_view_min_lod.html)
- [`VK_EXT_legacy_dithering`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_legacy_dithering.html)
- [`VK_EXT_load_store_op_none`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_load_store_op_none.html)
- [`VK_EXT_memory_budget`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_memory_budget.html)
- [`VK_EXT_memory_priority`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_memory_priority.html)
- [`VK_EXT_mesh_shader`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_mesh_shader.html)
- [`VK_EXT_metal_objects`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_metal_objects.html)
- [`VK_EXT_multi_draw`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_multi_draw.html)
- [`VK_EXT_multisampled_render_to_single_sampled`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_multisampled_render_to_single_sampled.html)
- [`VK_EXT_mutable_descriptor_type`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_mutable_descriptor_type.html)
- [`VK_EXT_non_seamless_cube_map`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_non_seamless_cube_map.html)
- [`VK_EXT_pageable_device_local_memory`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_pageable_device_local_memory.html)
- [`VK_EXT_pci_bus_info`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_pci_bus_info.html)
- [`VK_EXT_physical_device_drm`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_physical_device_drm.html)
- [`VK_EXT_pipeline_properties`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_pipeline_properties.html)
- [`VK_EXT_pipeline_robustness`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_pipeline_robustness.html)
- [`VK_EXT_post_depth_coverage`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_post_depth_coverage.html)
- [`VK_EXT_primitives_generated_query`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_primitives_generated_query.html)
- [`VK_EXT_provoking_vertex`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_provoking_vertex.html)
- [`VK_EXT_queue_family_foreign`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_queue_family_foreign.html)
- [`VK_EXT_rasterization_order_attachment_access`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_rasterization_order_attachment_access.html)
- [`VK_EXT_rgba10x6_formats`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_rgba10x6_formats.html)
- [`VK_EXT_sample_locations`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_sample_locations.html)
- [`VK_EXT_shader_atomic_float`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_shader_atomic_float.html)
- [`VK_EXT_shader_atomic_float2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_shader_atomic_float2.html)
- [`VK_EXT_shader_image_atomic_int64`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_shader_image_atomic_int64.html)
- [`VK_EXT_shader_module_identifier`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_shader_module_identifier.html)
- [`VK_EXT_shader_stencil_export`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_shader_stencil_export.html)
- [`VK_EXT_shader_subgroup_ballot`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_shader_subgroup_ballot.html) (deprecated)
- [`VK_EXT_shader_subgroup_vote`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_shader_subgroup_vote.html) (deprecated)
- [`VK_EXT_subpass_merge_feedback`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_subpass_merge_feedback.html)
- [`VK_EXT_transform_feedback`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_transform_feedback.html)
- [`VK_EXT_validation_cache`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_validation_cache.html)
- [`VK_EXT_validation_flags`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_validation_flags.html) (deprecated)
- [`VK_EXT_vertex_input_dynamic_state`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_vertex_input_dynamic_state.html)
- [`VK_EXT_video_decode_h264`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_video_decode_h264.html) (provisional)
- [`VK_EXT_video_decode_h265`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_video_decode_h265.html) (provisional)
- [`VK_EXT_video_encode_h264`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_video_encode_h264.html) (provisional)
- [`VK_EXT_video_encode_h265`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_video_encode_h265.html) (provisional)
- [`VK_AMD_buffer_marker`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_buffer_marker.html)
- [`VK_AMD_device_coherent_memory`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_device_coherent_memory.html)
- [`VK_AMD_display_native_hdr`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_display_native_hdr.html)
- [`VK_AMD_draw_indirect_count`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_draw_indirect_count.html) (promoted)
- [`VK_AMD_gcn_shader`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_gcn_shader.html)
- [`VK_AMD_gpu_shader_half_float`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_gpu_shader_half_float.html) (deprecated)
- [`VK_AMD_gpu_shader_int16`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_gpu_shader_int16.html) (deprecated)
- [`VK_AMD_memory_overallocation_behavior`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_memory_overallocation_behavior.html)
- [`VK_AMD_mixed_attachment_samples`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_mixed_attachment_samples.html)
- [`VK_AMD_negative_viewport_height`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_negative_viewport_height.html) (obsoleted)
- [`VK_AMD_pipeline_compiler_control`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_pipeline_compiler_control.html)
- [`VK_AMD_rasterization_order`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_rasterization_order.html)
- [`VK_AMD_shader_ballot`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_shader_ballot.html)
- [`VK_AMD_shader_core_properties`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_shader_core_properties.html)
- [`VK_AMD_shader_core_properties2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_shader_core_properties2.html)
- [`VK_AMD_shader_early_and_late_fragment_tests`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_shader_early_and_late_fragment_tests.html)
- [`VK_AMD_shader_explicit_vertex_parameter`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_shader_explicit_vertex_parameter.html)
- [`VK_AMD_shader_fragment_mask`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_shader_fragment_mask.html)
- [`VK_AMD_shader_image_load_store_lod`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_shader_image_load_store_lod.html)
- [`VK_AMD_shader_info`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_shader_info.html)
- [`VK_AMD_shader_trinary_minmax`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_shader_trinary_minmax.html)
- [`VK_AMD_texture_gather_bias_lod`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_AMD_texture_gather_bias_lod.html)
- [`VK_ANDROID_external_memory_android_hardware_buffer`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_ANDROID_external_memory_android_hardware_buffer.html)
- [`VK_ARM_rasterization_order_attachment_access`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_ARM_rasterization_order_attachment_access.html) (promoted)
- [`VK_FUCHSIA_buffer_collection`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_FUCHSIA_buffer_collection.html)
- [`VK_FUCHSIA_external_memory`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_FUCHSIA_external_memory.html)
- [`VK_GGP_frame_token`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_GGP_frame_token.html)
- [`VK_GOOGLE_decorate_string`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_GOOGLE_decorate_string.html)
- [`VK_GOOGLE_display_timing`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_GOOGLE_display_timing.html)
- [`VK_GOOGLE_hlsl_functionality1`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_GOOGLE_hlsl_functionality1.html)
- [`VK_GOOGLE_surfaceless_query`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_GOOGLE_surfaceless_query.html)
- [`VK_GOOGLE_user_type`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_GOOGLE_user_type.html)
- [`VK_HUAWEI_invocation_mask`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_HUAWEI_invocation_mask.html)
- [`VK_HUAWEI_subpass_shading`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_HUAWEI_subpass_shading.html)
- [`VK_IMG_filter_cubic`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_IMG_filter_cubic.html)
- [`VK_IMG_format_pvrtc`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_IMG_format_pvrtc.html)
- [`VK_INTEL_performance_query`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_INTEL_performance_query.html)
- [`VK_INTEL_shader_integer_functions2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_INTEL_shader_integer_functions2.html)
- [`VK_NV_acquire_winrt_display`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_acquire_winrt_display.html)
- [`VK_NV_clip_space_w_scaling`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_clip_space_w_scaling.html)
- [`VK_NV_compute_shader_derivatives`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_compute_shader_derivatives.html)
- [`VK_NV_cooperative_matrix`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_cooperative_matrix.html)
- [`VK_NV_corner_sampled_image`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_corner_sampled_image.html)
- [`VK_NV_coverage_reduction_mode`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_coverage_reduction_mode.html)
- [`VK_NV_dedicated_allocation`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_dedicated_allocation.html) (deprecated)
- [`VK_NV_dedicated_allocation_image_aliasing`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_dedicated_allocation_image_aliasing.html)
- [`VK_NV_device_diagnostic_checkpoints`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_device_diagnostic_checkpoints.html)
- [`VK_NV_device_diagnostics_config`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_device_diagnostics_config.html)
- [`VK_NV_device_generated_commands`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_device_generated_commands.html)
- [`VK_NV_external_memory`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_external_memory.html) (deprecated)
- [`VK_NV_external_memory_capabilities`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_external_memory_capabilities.html) (deprecated)
- [`VK_NV_external_memory_rdma`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_external_memory_rdma.html)
- [`VK_NV_external_memory_win32`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_external_memory_win32.html) (deprecated)
- [`VK_NV_fill_rectangle`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_fill_rectangle.html)
- [`VK_NV_fragment_coverage_to_color`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_fragment_coverage_to_color.html)
- [`VK_NV_fragment_shader_barycentric`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_fragment_shader_barycentric.html) (promoted)
- [`VK_NV_fragment_shading_rate_enums`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_fragment_shading_rate_enums.html)
- [`VK_NV_framebuffer_mixed_samples`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_framebuffer_mixed_samples.html)
- [`VK_NV_geometry_shader_passthrough`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_geometry_shader_passthrough.html)
- [`VK_NV_glsl_shader`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_glsl_shader.html) (deprecated)
- [`VK_NV_inherited_viewport_scissor`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_inherited_viewport_scissor.html)
- [`VK_NV_linear_color_attachment`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_linear_color_attachment.html)
- [`VK_NV_mesh_shader`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_mesh_shader.html)
- [`VK_NV_ray_tracing`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_ray_tracing.html)
- [`VK_NV_ray_tracing_motion_blur`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_ray_tracing_motion_blur.html)
- [`VK_NV_representative_fragment_test`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_representative_fragment_test.html)
- [`VK_NV_sample_mask_override_coverage`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_sample_mask_override_coverage.html)
- [`VK_NV_scissor_exclusive`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_scissor_exclusive.html)
- [`VK_NV_shader_image_footprint`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_shader_image_footprint.html)
- [`VK_NV_shader_sm_builtins`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_shader_sm_builtins.html)
- [`VK_NV_shader_subgroup_partitioned`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_shader_subgroup_partitioned.html)
- [`VK_NV_shading_rate_image`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_shading_rate_image.html)
- [`VK_NV_viewport_array2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_viewport_array2.html)
- [`VK_NV_viewport_swizzle`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_viewport_swizzle.html)
- [`VK_NV_win32_keyed_mutex`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_win32_keyed_mutex.html) (promoted)
- [`VK_NVX_binary_import`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NVX_binary_import.html)
- [`VK_NVX_image_view_handle`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NVX_image_view_handle.html)
- [`VK_NVX_multiview_per_view_attributes`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NVX_multiview_per_view_attributes.html)
- [`VK_QCOM_fragment_density_map_offset`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_QCOM_fragment_density_map_offset.html)
- [`VK_QCOM_image_processing`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_QCOM_image_processing.html)
- [`VK_QCOM_render_pass_shader_resolve`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_QCOM_render_pass_shader_resolve.html)
- [`VK_QCOM_render_pass_store_ops`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_QCOM_render_pass_store_ops.html)
- [`VK_QCOM_render_pass_transform`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_QCOM_render_pass_transform.html)
- [`VK_QCOM_rotated_copy_commands`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_QCOM_rotated_copy_commands.html)
- [`VK_QCOM_tile_properties`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_QCOM_tile_properties.html)
- [`VK_SEC_amigo_profiling`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_SEC_amigo_profiling.html)
- [`VK_VALVE_descriptor_set_host_mapping`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_VALVE_descriptor_set_host_mapping.html)
- [`VK_VALVE_mutable_descriptor_type`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_VALVE_mutable_descriptor_type.html) (promoted)
