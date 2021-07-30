// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// The `Version` object is reexported from the `instance` module.

use std::convert::TryFrom;
use std::fmt;

/// Represents an API version of Vulkan.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    /// Major version number.
    pub major: u32,
    /// Minor version number.
    pub minor: u32,
    /// Patch version number.
    pub patch: u32,
}

impl Version {
    pub const V1_0: Version = Version::major_minor(1, 0);
    pub const V1_1: Version = Version::major_minor(1, 1);
    pub const V1_2: Version = Version::major_minor(1, 2);

    /// Constructs a `Version` from the given major and minor version numbers.
    #[inline]
    pub const fn major_minor(major: u32, minor: u32) -> Version {
        Version {
            major,
            minor,
            patch: 0,
        }
    }
}

impl Default for Version {
  fn default() -> Self {
    Self::V1_0
  }
}

impl From<u32> for Version {
    #[inline]
    fn from(val: u32) -> Self {
        Version {
            major: ash::vk::version_major(val),
            minor: ash::vk::version_minor(val),
            patch: ash::vk::version_patch(val),
        }
    }
}

impl TryFrom<Version> for u32 {
    type Error = ();

    #[inline]
    fn try_from(val: Version) -> Result<Self, Self::Error> {
        if val.major <= 0x3ff && val.minor <= 0x3ff && val.patch <= 0xfff {
            Ok(ash::vk::make_version(val.major, val.minor, val.patch))
        } else {
            Err(())
        }
    }
}

impl fmt::Debug for Version {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl fmt::Display for Version {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self, formatter)
    }
}

#[cfg(test)]
mod tests {
    use super::Version;
    use std::convert::TryFrom;

    #[test]
    fn into_vk_version() {
        let version = Version {
            major: 1,
            minor: 0,
            patch: 0,
        };
        assert_eq!(u32::try_from(version).unwrap(), 0x400000);
    }

    #[test]
    fn greater_major() {
        let v1 = Version {
            major: 1,
            minor: 0,
            patch: 0,
        };
        let v2 = Version {
            major: 2,
            minor: 0,
            patch: 0,
        };
        assert!(v2 > v1);
    }

    #[test]
    fn greater_minor() {
        let v1 = Version {
            major: 1,
            minor: 1,
            patch: 0,
        };
        let v2 = Version {
            major: 1,
            minor: 3,
            patch: 0,
        };
        assert!(v2 > v1);
    }

    #[test]
    fn greater_patch() {
        let v1 = Version {
            major: 1,
            minor: 0,
            patch: 4,
        };
        let v2 = Version {
            major: 1,
            minor: 0,
            patch: 5,
        };
        assert!(v2 > v1);
    }
}
