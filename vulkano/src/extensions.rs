// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::RequiresOneOf;
use std::{
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
};

/// An error that can happen when enabling an extension on an instance or device.
#[derive(Clone, Copy, Debug)]
pub struct ExtensionRestrictionError {
    /// The extension in question.
    pub extension: &'static str,
    /// The restriction that was not met.
    pub restriction: ExtensionRestriction,
}

impl Error for ExtensionRestrictionError {}

impl Display for ExtensionRestrictionError {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        write!(
            f,
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
    Requires(RequiresOneOf),
    /// Requires a device extension to be disabled.
    ConflictsDeviceExtension(&'static str),
}

impl Display for ExtensionRestriction {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match *self {
            ExtensionRestriction::NotSupported => {
                write!(f, "not supported by the loader or physical device")
            }
            ExtensionRestriction::RequiredIfSupported => {
                write!(f, "required to be enabled by the physical device")
            }
            ExtensionRestriction::Requires(requires) => {
                if requires.len() > 1 {
                    write!(f, "requires one of: {}", requires)
                } else {
                    write!(f, "requires: {}", requires)
                }
            }
            ExtensionRestriction::ConflictsDeviceExtension(ext) => {
                write!(f, "requires device extension {} to be disabled", ext)
            }
        }
    }
}
