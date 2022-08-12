// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::Version;
use std::ffi::CStr;

/// Properties of a layer.
#[derive(Clone)]
pub struct LayerProperties {
    pub(crate) props: ash::vk::LayerProperties,
}

impl LayerProperties {
    /// Returns the name of the layer.
    ///
    /// If you want to enable this layer on an instance, you need to pass this value to
    /// `Instance::new`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use vulkano::VulkanLibrary;
    ///
    /// let library = VulkanLibrary::new().unwrap();
    ///
    /// for layer in library.layer_properties().unwrap() {
    ///     println!("Layer name: {}", layer.name());
    /// }
    /// ```
    #[inline]
    pub fn name(&self) -> &str {
        unsafe {
            CStr::from_ptr(self.props.layer_name.as_ptr())
                .to_str()
                .unwrap()
        }
    }

    /// Returns a description of the layer.
    ///
    /// This description is chosen by the layer itself.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use vulkano::VulkanLibrary;
    ///
    /// let library = VulkanLibrary::new().unwrap();
    ///
    /// for layer in library.layer_properties().unwrap() {
    ///     println!("Layer description: {}", layer.description());
    /// }
    /// ```
    #[inline]
    pub fn description(&self) -> &str {
        unsafe {
            CStr::from_ptr(self.props.description.as_ptr())
                .to_str()
                .unwrap()
        }
    }

    /// Returns the version of Vulkan supported by this layer.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use vulkano::{Version, VulkanLibrary};
    ///
    /// let library = VulkanLibrary::new().unwrap();
    ///
    /// for layer in library.layer_properties().unwrap() {
    ///     if layer.vulkan_version() >= Version::major_minor(2, 0) {
    ///         println!("Layer {} requires Vulkan 2.0", layer.name());
    ///     }
    /// }
    /// ```
    #[inline]
    pub fn vulkan_version(&self) -> Version {
        Version::from(self.props.spec_version)
    }

    /// Returns an implementation-specific version number for this layer.
    ///
    /// The number is chosen by the layer itself. It can be used for bug reports for example.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use vulkano::VulkanLibrary;
    ///
    /// let library = VulkanLibrary::new().unwrap();
    ///
    /// for layer in library.layer_properties().unwrap() {
    ///     println!("Layer {} - Version: {}", layer.name(), layer.implementation_version());
    /// }
    /// ```
    #[inline]
    pub fn implementation_version(&self) -> u32 {
        self.props.implementation_version
    }
}

#[cfg(test)]
mod tests {
    use crate::VulkanLibrary;

    #[test]
    fn layers_list() {
        let library = match VulkanLibrary::new() {
            Ok(x) => x,
            Err(_) => return,
        };

        let list = match library.layer_properties() {
            Ok(l) => l,
            Err(_) => return,
        };

        for _ in list {}
    }
}
