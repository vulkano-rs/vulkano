// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// The `Version` object is reexported from the `instance` module.

use std::cmp::Ordering;
use std::fmt;

/// Represents an API version of Vulkan.
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Version {
    /// Major version number.
    pub major: u16,
    /// Minor version number.
    pub minor: u16,
    /// Patch version number.
    pub patch: u16,
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

impl PartialOrd for Version {
    #[inline]
    fn partial_cmp(&self, other: &Version) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Version) -> Ordering {
        match self.major.cmp(&other.major) {
            Ordering::Equal => (),
            o => return o,
        };

        match self.minor.cmp(&other.minor) {
            Ordering::Equal => (),
            o => return o,
        };

        self.patch.cmp(&other.patch)
    }
}

impl Version {
    /// Turns a version number given by Vulkan into a `Version` struct.
    #[inline]
    pub fn from_vulkan_version(value: u32) -> Version {
        Version {
            major: ((value & 0xffc00000) >> 22) as u16,
            minor: ((value & 0x003ff000) >> 12) as u16,
            patch: (value & 0x00000fff) as u16,
        }
    }

    /// Turns a `Version` into a version number accepted by Vulkan.
    ///
    /// # Panic
    ///
    /// Panics if the values in the `Version` are out of acceptable range.
    #[inline]
    pub fn into_vulkan_version(&self) -> u32 {
        assert!(self.major <= 0x3ff);
        assert!(self.minor <= 0x3ff);
        assert!(self.patch <= 0xfff);

        (self.major as u32) << 22 | (self.minor as u32) << 12 | (self.patch as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::Version;

    #[test]
    fn into_vk_version() {
        let version = Version {
            major: 1,
            minor: 0,
            patch: 0,
        };
        assert_eq!(version.into_vulkan_version(), 0x400000);
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
