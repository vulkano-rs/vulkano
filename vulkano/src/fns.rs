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

// TODO: would be nice if these could be generated automatically from Vulkano's list of extensions

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
    khr_display => KhrDisplayFn,
    khr_get_physical_device_properties2 => KhrGetPhysicalDeviceProperties2Fn, // Promoted
    khr_surface => KhrSurfaceFn,
    khr_wayland_surface => KhrWaylandSurfaceFn,
    khr_win32_surface => KhrWin32SurfaceFn,
    khr_xcb_surface => KhrXcbSurfaceFn,
    khr_xlib_surface => KhrXlibSurfaceFn,

    // This is an instance extension, so it should be loaded with `vkGetInstanceProcAddr`, despite
    // having device-level functions. This is an unfortunate exception in the spec that even causes
    // the LunarG people headaches:
    // https://github.com/KhronosGroup/Vulkan-Loader/issues/116#issuecomment-580982393
    ext_debug_utils => ExtDebugUtilsFn,

    mvk_ios_surface => MvkIosSurfaceFn,
    mvk_macos_surface => MvkMacosSurfaceFn,

    nn_vi_surface => NnViSurfaceFn,
});

fns!(DeviceFunctions, {
    v1_0 => DeviceFnV1_0,
    v1_1 => DeviceFnV1_1,
    v1_2 => DeviceFnV1_2,

    khr_external_semaphore_fd => KhrExternalSemaphoreFdFn,
    khr_external_memory_fd => KhrExternalMemoryFdFn,
    khr_get_memory_requirements2 => KhrGetMemoryRequirements2Fn, // Promoted
    khr_maintenance1 => KhrMaintenance1Fn, // Promoted
    khr_swapchain => KhrSwapchainFn,

    ext_buffer_device_address => ExtBufferDeviceAddressFn,
    ext_full_screen_exclusive => ExtFullScreenExclusiveFn,
});
