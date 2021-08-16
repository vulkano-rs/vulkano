// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::instance::loader::LoadingError;
use crate::Error;
use crate::OomError;
use crate::Version;
use std::error;
use std::fmt;

/// Error that can happen when loading the list of layers.
#[derive(Clone, Debug)]
pub enum SupportedExtensionsError {
    /// Failed to load the Vulkan shared library.
    LoadingError(LoadingError),
    /// Not enough memory.
    OomError(OomError),
}

impl error::Error for SupportedExtensionsError {
    #[inline]
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            SupportedExtensionsError::LoadingError(ref err) => Some(err),
            SupportedExtensionsError::OomError(ref err) => Some(err),
        }
    }
}

impl fmt::Display for SupportedExtensionsError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                SupportedExtensionsError::LoadingError(_) =>
                    "failed to load the Vulkan shared library",
                SupportedExtensionsError::OomError(_) => "not enough memory available",
            }
        )
    }
}

impl From<OomError> for SupportedExtensionsError {
    #[inline]
    fn from(err: OomError) -> SupportedExtensionsError {
        SupportedExtensionsError::OomError(err)
    }
}

impl From<LoadingError> for SupportedExtensionsError {
    #[inline]
    fn from(err: LoadingError) -> SupportedExtensionsError {
        SupportedExtensionsError::LoadingError(err)
    }
}

impl From<Error> for SupportedExtensionsError {
    #[inline]
    fn from(err: Error) -> SupportedExtensionsError {
        match err {
            err @ Error::OutOfHostMemory => SupportedExtensionsError::OomError(OomError::from(err)),
            err @ Error::OutOfDeviceMemory => {
                SupportedExtensionsError::OomError(OomError::from(err))
            }
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

/// An error that can happen when enabling an extension on an instance or device.
#[derive(Clone, Copy, Debug)]
pub struct ExtensionRestrictionError {
    /// The extension in question.
    pub extension: &'static str,
    /// The restriction that was not met.
    pub restriction: ExtensionRestriction,
}

impl error::Error for ExtensionRestrictionError {}

impl fmt::Display for ExtensionRestrictionError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "a restriction for the extension {} was not met: {}",
            self.extension, self.restriction,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ExtensionRestriction {
    /// Not supported by the loader or physical device.
    NotSupported,
    /// Requires a minimum Vulkan API version.
    RequiresCore(Version),
    /// Requires a device extension to be enabled.
    RequiresDeviceExtension(&'static str),
    /// Requires an instance extension to be enabled.
    RequiresInstanceExtension(&'static str),
    /// Required to be enabled by the physical device.
    RequiredIfSupported,
    /// Requires a device extension to be disabled.
    ConflictsDeviceExtension(&'static str),
}

impl fmt::Display for ExtensionRestriction {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            ExtensionRestriction::NotSupported => {
                write!(fmt, "not supported by the loader or physical device")
            }
            ExtensionRestriction::RequiresCore(version) => {
                write!(
                    fmt,
                    "requires Vulkan API version {}.{}",
                    version.major, version.minor
                )
            }
            ExtensionRestriction::RequiresDeviceExtension(ext) => {
                write!(fmt, "requires device extension {} to be enabled", ext)
            }
            ExtensionRestriction::RequiresInstanceExtension(ext) => {
                write!(fmt, "requires instance extension {} to be enabled", ext)
            }
            ExtensionRestriction::RequiredIfSupported => {
                write!(fmt, "required to be enabled by the physical device")
            }
            ExtensionRestriction::ConflictsDeviceExtension(ext) => {
                write!(fmt, "requires device extension {} to be disabled", ext)
            }
        }
    }
}

/// This helper type can only be instantiated inside this module.
#[doc(hidden)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Unbuildable(pub(crate) ());
