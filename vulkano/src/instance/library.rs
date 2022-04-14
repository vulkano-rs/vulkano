// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

#[cfg(feature = "loaded")]
pub use ash::LoadingError;

/// Wraps the `ash::Entry` type which contains function pointers into the Vulkan library.
pub struct VulkanLibrary {
    entry: ash::Entry,
}

impl VulkanLibrary {
    /// Load entry points from a Vulkan loader linked at compile time.
    #[cfg(feature = "linked")]
    pub fn linked() -> VulkanLibrary {
        let entry = ash::Entry::linked();
        VulkanLibrary { entry }
    }

    /// Load default Vulkan library for the current platform with `libloading`.
    #[cfg(feature = "loaded")]
    pub unsafe fn load() -> Result<VulkanLibrary, LoadingError> {
        let entry = ash::Entry::load()?;
        Ok(VulkanLibrary { entry })
    }

    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }
}

impl Default for VulkanLibrary {
    /// Returns the default vulkan library. The way that the Vulkan library was
    /// loaded depends on the feature flags.
    fn default() -> Self {
        #[cfg(feature = "linked")]
        let entry = VulkanLibrary::linked();

        #[cfg(feature = "loaded")]
        let entry = unsafe { VulkanLibrary::load().unwrap() };

        entry
    }
}
