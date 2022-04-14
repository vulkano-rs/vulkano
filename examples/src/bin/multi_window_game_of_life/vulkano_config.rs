// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use vulkano::{
    device::{DeviceExtensions, Features},
    instance::{InstanceExtensions, VulkanLibrary},
};

/// A utility struct to configure vulkano context
pub struct VulkanoConfig {
    pub return_from_run: bool,
    pub add_primary_window: bool,
    pub instance_extensions: InstanceExtensions,
    pub device_extensions: DeviceExtensions,
    pub features: Features,
    pub layers: Vec<String>,
    pub lib: VulkanLibrary,
}

impl Default for VulkanoConfig {
    fn default() -> Self {
        let lib = VulkanLibrary::default();
        VulkanoConfig {
            return_from_run: false,
            add_primary_window: true,
            instance_extensions: InstanceExtensions {
                ext_debug_utils: true,
                ..vulkano_win::required_extensions(&lib)
            },
            device_extensions: DeviceExtensions {
                khr_swapchain: true,
                ..DeviceExtensions::none()
            },
            features: Features::none(),
            layers: vec![],
            lib,
        }
    }
}
