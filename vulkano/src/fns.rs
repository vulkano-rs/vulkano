// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use ash::vk::*;
use std::ffi::CStr;
use std::os::raw::c_void;

macro_rules! fns {
    ($struct_name:ident, { $($member:ident => $fn_struct:ident,)+ }) => {
        pub struct $struct_name {
            $(
                pub $member: $fn_struct,
            )+
        }

        impl $struct_name {
            pub fn load<F>(mut load_fn: F) -> $struct_name
                where F: FnMut(&CStr) -> *const c_void
            {
                $struct_name {
                    $(
                        $member: ash::vk::$fn_struct::load(&mut load_fn),
                    )+
                }
            }
        }
    };
}

// Auto-generated from vk.xml header version 168

fns!(EntryFunctions, {
    v1_0 => EntryFnV1_0,
    v1_1 => EntryFnV1_1,
    v1_2 => EntryFnV1_2,
});

fns!(InstanceFunctions, {
    v1_0 => InstanceFnV1_0,
    v1_1 => InstanceFnV1_1,
    v1_2 => InstanceFnV1_2,
    khr_android_surface => KhrAndroidSurfaceFn,
    khr_device_group_creation => KhrDeviceGroupCreationFn,
    khr_display => KhrDisplayFn,
    khr_external_fence_capabilities => KhrExternalFenceCapabilitiesFn,
    khr_external_memory_capabilities => KhrExternalMemoryCapabilitiesFn,
    khr_external_semaphore_capabilities => KhrExternalSemaphoreCapabilitiesFn,
    khr_get_display_properties2 => KhrGetDisplayProperties2Fn,
    khr_get_physical_device_properties2 => KhrGetPhysicalDeviceProperties2Fn,
    khr_get_surface_capabilities2 => KhrGetSurfaceCapabilities2Fn,
    khr_surface => KhrSurfaceFn,
    khr_wayland_surface => KhrWaylandSurfaceFn,
    khr_win32_surface => KhrWin32SurfaceFn,
    khr_xcb_surface => KhrXcbSurfaceFn,
    khr_xlib_surface => KhrXlibSurfaceFn,
    ext_acquire_xlib_display => ExtAcquireXlibDisplayFn,
    ext_debug_report => ExtDebugReportFn,
    ext_debug_utils => ExtDebugUtilsFn,
    ext_direct_mode_display => ExtDirectModeDisplayFn,
    ext_directfb_surface => ExtDirectfbSurfaceFn,
    ext_display_surface_counter => ExtDisplaySurfaceCounterFn,
    ext_headless_surface => ExtHeadlessSurfaceFn,
    ext_metal_surface => ExtMetalSurfaceFn,
    fuchsia_imagepipe_surface => FuchsiaImagepipeSurfaceFn,
    ggp_stream_descriptor_surface => GgpStreamDescriptorSurfaceFn,
    mvk_ios_surface => MvkIosSurfaceFn,
    mvk_macos_surface => MvkMacosSurfaceFn,
    nn_vi_surface => NnViSurfaceFn,
    nv_external_memory_capabilities => NvExternalMemoryCapabilitiesFn,
});

fns!(DeviceFunctions, {
    v1_0 => DeviceFnV1_0,
    v1_1 => DeviceFnV1_1,
    v1_2 => DeviceFnV1_2,
    khr_acceleration_structure => KhrAccelerationStructureFn,
    khr_bind_memory2 => KhrBindMemory2Fn,
    khr_buffer_device_address => KhrBufferDeviceAddressFn,
    khr_copy_commands2 => KhrCopyCommands2Fn,
    khr_create_renderpass2 => KhrCreateRenderpass2Fn,
    khr_deferred_host_operations => KhrDeferredHostOperationsFn,
    khr_descriptor_update_template => KhrDescriptorUpdateTemplateFn,
    khr_device_group => KhrDeviceGroupFn,
    khr_display_swapchain => KhrDisplaySwapchainFn,
    khr_draw_indirect_count => KhrDrawIndirectCountFn,
    khr_external_fence_fd => KhrExternalFenceFdFn,
    khr_external_fence_win32 => KhrExternalFenceWin32Fn,
    khr_external_memory_fd => KhrExternalMemoryFdFn,
    khr_external_memory_win32 => KhrExternalMemoryWin32Fn,
    khr_external_semaphore_fd => KhrExternalSemaphoreFdFn,
    khr_external_semaphore_win32 => KhrExternalSemaphoreWin32Fn,
    khr_fragment_shading_rate => KhrFragmentShadingRateFn,
    khr_get_memory_requirements2 => KhrGetMemoryRequirements2Fn,
    khr_maintenance1 => KhrMaintenance1Fn,
    khr_maintenance3 => KhrMaintenance3Fn,
    khr_performance_query => KhrPerformanceQueryFn,
    khr_pipeline_executable_properties => KhrPipelineExecutablePropertiesFn,
    khr_push_descriptor => KhrPushDescriptorFn,
    khr_ray_tracing_pipeline => KhrRayTracingPipelineFn,
    khr_sampler_ycbcr_conversion => KhrSamplerYcbcrConversionFn,
    khr_shared_presentable_image => KhrSharedPresentableImageFn,
    khr_swapchain => KhrSwapchainFn,
    khr_timeline_semaphore => KhrTimelineSemaphoreFn,
    ext_buffer_device_address => ExtBufferDeviceAddressFn,
    ext_calibrated_timestamps => ExtCalibratedTimestampsFn,
    ext_conditional_rendering => ExtConditionalRenderingFn,
    ext_debug_marker => ExtDebugMarkerFn,
    ext_discard_rectangles => ExtDiscardRectanglesFn,
    ext_display_control => ExtDisplayControlFn,
    ext_extended_dynamic_state => ExtExtendedDynamicStateFn,
    ext_external_memory_host => ExtExternalMemoryHostFn,
    ext_full_screen_exclusive => ExtFullScreenExclusiveFn,
    ext_hdr_metadata => ExtHdrMetadataFn,
    ext_host_query_reset => ExtHostQueryResetFn,
    ext_image_drm_format_modifier => ExtImageDrmFormatModifierFn,
    ext_line_rasterization => ExtLineRasterizationFn,
    ext_private_data => ExtPrivateDataFn,
    ext_sample_locations => ExtSampleLocationsFn,
    ext_tooling_info => ExtToolingInfoFn,
    ext_transform_feedback => ExtTransformFeedbackFn,
    ext_validation_cache => ExtValidationCacheFn,
    amd_buffer_marker => AmdBufferMarkerFn,
    amd_display_native_hdr => AmdDisplayNativeHdrFn,
    amd_draw_indirect_count => AmdDrawIndirectCountFn,
    amd_shader_info => AmdShaderInfoFn,
    android_external_memory_android_hardware_buffer => AndroidExternalMemoryAndroidHardwareBufferFn,
    google_display_timing => GoogleDisplayTimingFn,
    intel_performance_query => IntelPerformanceQueryFn,
    nvx_image_view_handle => NvxImageViewHandleFn,
    nv_acquire_winrt_display => NvAcquireWinrtDisplayFn,
    nv_clip_space_w_scaling => NvClipSpaceWScalingFn,
    nv_cooperative_matrix => NvCooperativeMatrixFn,
    nv_coverage_reduction_mode => NvCoverageReductionModeFn,
    nv_device_diagnostic_checkpoints => NvDeviceDiagnosticCheckpointsFn,
    nv_device_generated_commands => NvDeviceGeneratedCommandsFn,
    nv_external_memory_win32 => NvExternalMemoryWin32Fn,
    nv_fragment_shading_rate_enums => NvFragmentShadingRateEnumsFn,
    nv_mesh_shader => NvMeshShaderFn,
    nv_ray_tracing => NvRayTracingFn,
    nv_scissor_exclusive => NvScissorExclusiveFn,
    nv_shading_rate_image => NvShadingRateImageFn,
});
