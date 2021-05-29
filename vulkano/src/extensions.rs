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

macro_rules! extensions {
    (
        $sname:ident,
        $rawname:ident,
        $($member:ident => {
            raw: $raw:expr,
            requires_core: $requires_core:ident,
            requires_device_extensions: [$($requires_device_extension:ident),*],
            requires_instance_extensions: [$($requires_instance_extension:ident),*]$(,)?
        },)*
    ) => (
        /// List of extensions that are enabled or available.
        #[derive(Copy, Clone, PartialEq, Eq)]
        pub struct $sname {
            $(
                // TODO: Rust 1.54 will allow generating documentation from macro arguments with the
                // #[doc = ..] attribute using concat! and stringify!. Once this is available,
                // generate documentation here with requirements and a link to the Vulkan
                // documentation page?
                pub $member: bool,
            )*

            /// This field ensures that an instance of this `Extensions` struct
            /// can only be created through Vulkano functions and the update
            /// syntax. This way, extensions can be added to Vulkano without
            /// breaking existing code.
            pub _unbuildable: Unbuildable,
        }

        impl $sname {
            /// Returns an `Extensions` object with all members set to `false`.
            #[inline]
            pub const fn none() -> $sname {
                $sname {
                    $($member: false,)*
                    _unbuildable: Unbuildable(())
                }
            }

            /// Returns the union of this list and another list.
            #[inline]
            pub const fn union(&self, other: &$sname) -> $sname {
                $sname {
                    $(
                        $member: self.$member || other.$member,
                    )*
                    _unbuildable: Unbuildable(())
                }
            }

            /// Returns the intersection of this list and another list.
            #[inline]
            pub const fn intersection(&self, other: &$sname) -> $sname {
                $sname {
                    $(
                        $member: self.$member && other.$member,
                    )*
                    _unbuildable: Unbuildable(())
                }
            }

            /// Returns the difference of another list from this list.
            #[inline]
            pub const fn difference(&self, other: &$sname) -> $sname {
                $sname {
                    $(
                        $member: self.$member && !other.$member,
                    )*
                    _unbuildable: Unbuildable(())
                }
            }
        }

        impl fmt::Debug for $sname {
            #[allow(unused_assignments)]
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "[")?;

                let mut first = true;

                $(
                    if self.$member {
                        if !first { write!(f, ", ")? }
                        else { first = false; }
                        f.write_str(str::from_utf8($raw).unwrap())?;
                    }
                )*

                write!(f, "]")
            }
        }

        /// Set of extensions, not restricted to those vulkano knows about.
        ///
        /// This is useful when interacting with external code that has statically-unknown extension
        /// requirements.
        #[derive(Clone, Eq, PartialEq)]
        pub struct $rawname(HashSet<CString>);

        impl $rawname {
            /// Constructs an extension set containing the supplied extensions.
            pub fn new<I>(extensions: I) -> Self
                where I: IntoIterator<Item=CString>
            {
                $rawname(extensions.into_iter().collect())
            }

            /// Constructs an empty extension set.
            pub fn none() -> Self { $rawname(HashSet::new()) }

            /// Adds an extension to the set if it is not already present.
            pub fn insert(&mut self, extension: CString) {
                self.0.insert(extension);
            }

            /// Returns the intersection of this set and another.
            pub fn intersection(&self, other: &Self) -> Self {
                $rawname(self.0.intersection(&other.0).cloned().collect())
            }

            /// Returns the difference of another set from this one.
            pub fn difference(&self, other: &Self) -> Self {
                $rawname(self.0.difference(&other.0).cloned().collect())
            }

            /// Returns the union of both extension sets
            pub fn union(&self, other: &Self) -> Self {
                $rawname(self.0.union(&other.0).cloned().collect())
            }

            pub fn iter(&self) -> impl Iterator<Item = &CString> { self.0.iter() }
        }

        impl fmt::Debug for $rawname {
            #[allow(unused_assignments)]
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                self.0.fmt(f)
            }
        }

        impl FromIterator<CString> for $rawname {
            fn from_iter<T>(iter: T) -> Self
                where T: IntoIterator<Item = CString>
            {
                $rawname(iter.into_iter().collect())
            }
        }

        impl<'a> From<&'a $sname> for $rawname {
            fn from(x: &'a $sname) -> Self {
                let mut data = HashSet::new();
                $(if x.$member { data.insert(CString::new(&$raw[..]).unwrap()); })*
                $rawname(data)
            }
        }

        impl<'a> From<&'a $rawname> for $sname {
            fn from(x: &'a $rawname) -> Self {
                let mut extensions = $sname::none();
                $(
                    if x.0.iter().any(|x| x.as_bytes() == &$raw[..]) {
                        extensions.$member = true;
                    }
                )*
                extensions
            }
        }
    );
}

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
pub struct ExtensionRequirementError {
    /// The extension in question.
    pub extension: &'static str,
    /// The requirement that was not met.
    pub requirement: ExtensionRequirement,
}

impl error::Error for ExtensionRequirementError {}

impl fmt::Display for ExtensionRequirementError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "a requirement for the extension {} was not met: {}",
            self.extension, self.requirement,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ExtensionRequirement {
    /// Requires a minimum Vulkan API version.
    Core(Version),
    /// Requires a device extension to be also enabled.
    DeviceExtension(&'static str),
    /// Requires an instance extension to be also enabled.
    InstanceExtension(&'static str),
}

impl fmt::Display for ExtensionRequirement {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        match *self {
            ExtensionRequirement::Core(version) => {
                write!(
                    fmt,
                    "Vulkan API version {}.{}",
                    version.major, version.minor
                )
            }
            ExtensionRequirement::DeviceExtension(ext) => write!(fmt, "device extension {}", ext),
            ExtensionRequirement::InstanceExtension(ext) => {
                write!(fmt, "instance extension {}", ext)
            }
        }
    }
}
