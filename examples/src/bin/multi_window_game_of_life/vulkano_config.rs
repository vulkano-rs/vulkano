use vulkano::{
    device::{DeviceExtensions, Features},
    instance::InstanceExtensions,
};

/// A utility struct to configure vulkano context
pub struct VulkanoConfig {
    pub return_from_run: bool,
    pub add_primary_window: bool,
    pub instance_extensions: InstanceExtensions,
    pub device_extensions: DeviceExtensions,
    pub features: Features,
    pub layers: Vec<&'static str>,
}

impl Default for VulkanoConfig {
    fn default() -> Self {
        VulkanoConfig {
            return_from_run: false,
            add_primary_window: true,
            instance_extensions: InstanceExtensions {
                ext_debug_utils: true,
                ..vulkano_win::required_extensions()
            },
            device_extensions: DeviceExtensions {
                khr_swapchain: true,
                ..DeviceExtensions::none()
            },
            features: Features::none(),
            layers: vec![],
        }
    }
}
