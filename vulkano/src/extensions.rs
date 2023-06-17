// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

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
