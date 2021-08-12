// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::check_errors;
pub use crate::extensions::{
    ExtensionRestriction, ExtensionRestrictionError, SupportedExtensionsError,
};
use crate::instance::loader;
use crate::instance::loader::LoadingError;
use std::ffi::CStr;
use std::ptr;

pub use crate::autogen::InstanceExtensions;

impl InstanceExtensions {
    /// See the docs of supported_by_core().
    pub fn supported_by_core_raw() -> Result<Self, SupportedExtensionsError> {
        InstanceExtensions::supported_by_core_raw_with_loader(loader::auto_loader()?)
    }

    /// Returns an `InstanceExtensions` object with extensions supported by the core driver.
    pub fn supported_by_core() -> Result<Self, LoadingError> {
        match InstanceExtensions::supported_by_core_raw() {
            Ok(l) => Ok(l),
            Err(SupportedExtensionsError::LoadingError(e)) => Err(e),
            Err(SupportedExtensionsError::OomError(e)) => panic!("{:?}", e),
        }
    }

    /// Same as `supported_by_core`, but allows specifying a loader.
    pub fn supported_by_core_with_loader<L>(
        ptrs: &loader::FunctionPointers<L>,
    ) -> Result<Self, LoadingError>
    where
        L: loader::Loader,
    {
        match InstanceExtensions::supported_by_core_raw_with_loader(ptrs) {
            Ok(l) => Ok(l),
            Err(SupportedExtensionsError::LoadingError(e)) => Err(e),
            Err(SupportedExtensionsError::OomError(e)) => panic!("{:?}", e),
        }
    }

    /// See the docs of supported_by_core().
    pub fn supported_by_core_raw_with_loader<L>(
        ptrs: &loader::FunctionPointers<L>,
    ) -> Result<Self, SupportedExtensionsError>
    where
        L: loader::Loader,
    {
        let fns = ptrs.fns();

        let properties: Vec<ash::vk::ExtensionProperties> = unsafe {
            let mut num = 0;
            check_errors(fns.v1_0.enumerate_instance_extension_properties(
                ptr::null(),
                &mut num,
                ptr::null_mut(),
            ))?;

            let mut properties = Vec::with_capacity(num as usize);
            check_errors(fns.v1_0.enumerate_instance_extension_properties(
                ptr::null(),
                &mut num,
                properties.as_mut_ptr(),
            ))?;
            properties.set_len(num as usize);
            properties
        };

        Ok(Self::from(properties.iter().map(|property| unsafe {
            CStr::from_ptr(property.extension_name.as_ptr())
        })))
    }
}

#[cfg(test)]
mod tests {
    use crate::instance::InstanceExtensions;
    use std::ffi::CString;

    #[test]
    fn empty_extensions() {
        let i: Vec<CString> = (&InstanceExtensions::none()).into();
        assert!(i.iter().next().is_none());
    }
}
