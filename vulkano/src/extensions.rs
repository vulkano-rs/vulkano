use bytemuck::cast_slice;

/// Properties of an extension in the loader or a physical device.
#[derive(Clone, Debug)]
pub struct ExtensionProperties {
    /// The name of the extension.
    pub extension_name: String,

    /// The version of the extension.
    pub spec_version: u32,
}

impl From<ash::vk::ExtensionProperties> for ExtensionProperties {
    #[inline]
    fn from(val: ash::vk::ExtensionProperties) -> Self {
        Self {
            extension_name: {
                let bytes = cast_slice(val.extension_name.as_slice());
                let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                String::from_utf8_lossy(&bytes[0..end]).into()
            },
            spec_version: val.spec_version,
        }
    }
}
