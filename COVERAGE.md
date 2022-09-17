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

- [`VK_KHR_external_memory`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_memory.html)
	- `VkExternalMemoryBufferCreateInfoKHR`
- [`VK_KHR_maintenance2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_maintenance2.html)
	- `VkImageViewUsageCreateInfoKHR`
	- `VkPipelineTessellationDomainOriginStateCreateInfoKHR`
	- `VK_IMAGE_CREATE_EXTENDED_USAGE_BIT`
	- `VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL_KHR`
	- `VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL_KHR`
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
- [`VK_EXT_separate_stencil_usage`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_separate_stencil_usage.html)
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
- [`VK_EXT_extended_dynamic_state`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_extended_dynamic_state.html)
- [`VK_EXT_extended_dynamic_state2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_extended_dynamic_state2.html)
- [`VK_EXT_shader_demote_to_helper_invocation`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_shader_demote_to_helper_invocation.html)
- [`VK_EXT_texel_buffer_alignment`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_texel_buffer_alignment.html)
- [`VK_EXT_texture_compression_astc_hdr`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_texture_compression_astc_hdr.html)
- [`VK_EXT_ycbcr_2plane_444_formats`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_ycbcr_2plane_444_formats.html)
- SPIR-V 1.6

### Partially supported

- [`VK_KHR_dynamic_rendering`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_dynamic_rendering.html)
	- Suspend/resume
- [`VK_KHR_synchronization2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_synchronization2.html)
	- `vkCmdResetEvent2KHR`
	- `vkCmdSetEvent2KHR`
	- `vkCmdWaitEvents2KHR`
	- `vkCmdWriteTimestamp2KHR`
	- `vkQueueSubmit2KHR` / `VkSemaphoreSubmitInfoKHR`
	- `VkMemoryBarrier2KHR` extending `VkSubpassDependency`
	- `VkAccessFlagBits2KHR` / `VkPipelineStageFlagBits2KHR` (only the base Vulkan 1.0 bits are defined)

### Unsupported

- [`VK_KHR_maintenance4`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_maintenance4.html)
- [`VK_EXT_4444_formats`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_4444_formats.html)
- [`VK_EXT_image_robustness`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_image_robustness.html)
- [`VK_EXT_inline_uniform_block`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_inline_uniform_block.html)
- [`VK_EXT_pipeline_creation_cache_control`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_pipeline_creation_cache_control.html)
- [`VK_EXT_pipeline_creation_feedback`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_pipeline_creation_feedback.html)
- [`VK_EXT_private_data`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_private_data.html)
- [`VK_EXT_subgroup_size_control`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_subgroup_size_control.html)
- [`VK_EXT_tooling_info`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_tooling_info.html)

## Unpromoted extensions

### Fully supported

- [`VK_KHR_android_surface`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_android_surface.html)
- [`VK_KHR_external_memory_fd`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_memory_fd.html)
- [`VK_KHR_get_surface_capabilities2`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_get_surface_capabilities2.html)
- [`VK_KHR_incremental_present`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_incremental_present.html)
- [`VK_KHR_push_descriptor`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_push_descriptor.html)
- [`VK_KHR_surface`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_surface.html)
- [`VK_KHR_swapchain`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_swapchain.html)
- [`VK_KHR_present_id`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_present_id.html)
- [`VK_KHR_present_wait`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_present_wait.html)
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
- [`VK_EXT_vertex_attribute_divisor`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_vertex_attribute_divisor.html)
- [`VK_EXT_ycbcr_image_arrays`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_ycbcr_image_arrays.html)
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
- [`VK_KHR_external_semaphore_fd`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_semaphore_fd.html)
	- `vkImportSemaphoreFdKHR` / `VkImportSemaphoreFdInfoKHR`
- [`VK_EXT_buffer_device_address`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_buffer_device_address.html) (deprecated)
	- `VkBufferDeviceAddressCreateInfoEXT`
- [`VK_EXT_debug_utils`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_debug_utils.html)
	- `vkSetDebugUtilsObjectTagEXT`
	- `vkSubmitDebugUtilsMessageEXT`
	- `VkDebugUtilsMessengerCallbackDataEXT` (not all data exposed to callback)
	- `VkDebugUtilsObjectNameInfoEXT` extending `VkPipelineShaderStageCreateInfo`
- [`VK_EXT_full_screen_exclusive`](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_full_screen_exclusive.html)
	- `vkGetPhysicalDeviceSurfacePresentModes2EXT`
