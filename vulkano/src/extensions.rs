// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::Version;
use std::error;
use std::fmt::{Display, Error as FmtError, Formatter};

/// An error that can happen when enabling an extension on an instance or device.
#[derive(Clone, Copy, Debug)]
pub struct ExtensionRestrictionError {
    /// The extension in question.
    pub extension: &'static str,
    /// The restriction that was not met.
    pub restriction: ExtensionRestriction,
}

impl error::Error for ExtensionRestrictionError {}

impl Display for ExtensionRestrictionError {
    #[inline]
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), FmtError> {
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
    /// Required to be enabled by the physical device.
    RequiredIfSupported,
    /// Requires one of the following.
    Requires(OneOfRequirements),
    /// Requires a device extension to be disabled.
    ConflictsDeviceExtension(&'static str),
}

impl Display for ExtensionRestriction {
    #[inline]
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), FmtError> {
        match *self {
            ExtensionRestriction::NotSupported => {
                write!(fmt, "not supported by the loader or physical device")
            }
            ExtensionRestriction::RequiredIfSupported => {
                write!(fmt, "required to be enabled by the physical device")
            }
            ExtensionRestriction::Requires(requires) => {
                if requires.has_multiple() {
                    write!(fmt, "requires one of: {}", requires)
                } else {
                    write!(fmt, "requires: {}", requires)
                }
            }
            ExtensionRestriction::ConflictsDeviceExtension(ext) => {
                write!(fmt, "requires device extension {} to be disabled", ext)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OneOfRequirements {
    pub api_version: Option<Version>,
    pub device_extensions: &'static [&'static str],
    pub instance_extensions: &'static [&'static str],
}

impl OneOfRequirements {
    /// Returns whether there is more than one possible requirement.
    #[inline]
    pub fn has_multiple(&self) -> bool {
        self.api_version.iter().count()
            + self.device_extensions.len()
            + self.instance_extensions.len()
            > 1
    }
}

impl Display for OneOfRequirements {
    #[inline]
    fn fmt(&self, fmt: &mut Formatter) -> Result<(), FmtError> {
        let mut items_written = 0;

        if let Some(version) = self.api_version {
            write!(
                fmt,
                "Vulkan API version {}.{}",
                version.major, version.minor
            )?;
            items_written += 1;
        }

        for ext in self.instance_extensions.iter() {
            if items_written != 0 {
                write!(fmt, ", ")?;
            }

            write!(fmt, "instance extension {}", ext)?;
            items_written += 1;
        }

        for ext in self.device_extensions.iter() {
            if items_written != 0 {
                write!(fmt, ", ")?;
            }

            write!(fmt, "device extension {}", ext)?;
            items_written += 1;
        }

        Ok(())
    }
}
