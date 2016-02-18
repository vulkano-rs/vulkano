
/// Represents an API version of Vulkan.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Version {
    /// Major version number.
    pub major: u16,
    /// Minor version number.
    pub minor: u16,
    /// Patch version number.
    pub patch: u16,
}

// TODO: implement PartialOrd & Ord

impl Version {
    /// Turns a version number given by Vulkan into a `Version` struct.
    #[inline]
    pub fn from_vulkan_version(value: u32) -> Version {
        Version {
            major: ((value & 0xffc00000) >> 22) as u16,
            minor: ((value & 0x003ff000) >> 12) as u16,
            patch:  (value & 0x00000fff) as u16,
        }
    }

    /// Turns a `Version` into a version number accepted by Vulkan.
    ///
    /// # Panic
    ///
    /// Panicks if the values in the `Version` are out of acceptable range.
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
    fn test() {
        let version = Version { major: 1, minor: 0, patch: 0 };
        assert_eq!(version.into_vulkan_version(), 0x400000);
    }
}
