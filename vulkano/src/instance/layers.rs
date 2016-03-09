use std::ffi::CStr;
use std::ptr;

//use alloc::Alloc;
use check_errors;
use OomError;
use vk;
use VK_ENTRY;
use version::Version;

/// Queries the list of layers that are available when creating an instance.
pub fn layers_list() -> Result<Vec<LayerProperties>, OomError> {
    unsafe {
        let mut num = 0;
        try!(check_errors(VK_ENTRY.EnumerateInstanceLayerProperties(&mut num, ptr::null_mut())));

        let mut layers: Vec<vk::LayerProperties> = Vec::with_capacity(num as usize);
        try!(check_errors(VK_ENTRY.EnumerateInstanceLayerProperties(&mut num, layers.as_mut_ptr())));
        layers.set_len(num as usize);

        Ok(layers.into_iter().map(|layer| {
            LayerProperties { props: layer }
        }).collect())
    }
}

/// Properties of an available layer.
pub struct LayerProperties {
    props: vk::LayerProperties,
}

impl LayerProperties {
    /// Returns the name of the layer.
    #[inline]
    pub fn name(&self) -> &str {
        unsafe { CStr::from_ptr(self.props.layerName.as_ptr()).to_str().unwrap() }
    }

    /// Returns a description of the layer.
    #[inline]
    pub fn description(&self) -> &str {
        unsafe { CStr::from_ptr(self.props.description.as_ptr()).to_str().unwrap() }
    }

    /// Returns the version of Vulkan supported by this layer.
    #[inline]
    pub fn vulkan_version(&self) -> Version {
        Version::from_vulkan_version(self.props.specVersion)
    }

    /// Returns an implementation-specific version number for this layer.
    #[inline]
    pub fn implementation_version(&self) -> u32 {
        self.props.implementationVersion
    }
}

#[cfg(test)]
mod tests {
    use instance;

    #[test]
    fn layers_list() {
        let _ = instance::layers_list();
    }
}
