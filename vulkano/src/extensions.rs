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
        $($member:ident => {
            doc: $doc:expr,
            raw: $raw:expr,
            requires_core: $requires_core:expr,
            requires_device_extensions: [$($requires_device_extension:ident),*],
            requires_instance_extensions: [$($requires_instance_extension:ident),*]$(,)?
        },)*
    ) => (
        /// List of extensions that are enabled or available.
        #[derive(Copy, Clone, PartialEq, Eq)]
        pub struct $sname {
            $(
                #[doc = $doc]
                pub $member: bool,
            )*

            /// This field ensures that an instance of this `Extensions` struct
            /// can only be created through Vulkano functions and the update
            /// syntax. This way, extensions can be added to Vulkano without
            /// breaking existing code.
            pub _unbuildable: crate::extensions::Unbuildable,
        }

        impl $sname {
            /// Returns an `Extensions` object with all members set to `false`.
            #[inline]
            pub const fn none() -> $sname {
                $sname {
                    $($member: false,)*
                    _unbuildable: crate::extensions::Unbuildable(())
                }
            }

            /// Returns true if `self` is a superset of the parameter.
            ///
            /// That is, for each extension of the parameter that is true, the corresponding value
            /// in self is true as well.
            pub fn is_superset_of(&self, other: &$sname) -> bool {
                $((self.$member == true || other.$member == false))&&+
            }

            /// Returns the union of this list and another list.
            #[inline]
            pub const fn union(&self, other: &$sname) -> $sname {
                $sname {
                    $(
                        $member: self.$member || other.$member,
                    )*
                    _unbuildable: crate::extensions::Unbuildable(())
                }
            }

            /// Returns the intersection of this list and another list.
            #[inline]
            pub const fn intersection(&self, other: &$sname) -> $sname {
                $sname {
                    $(
                        $member: self.$member && other.$member,
                    )*
                    _unbuildable: crate::extensions::Unbuildable(())
                }
            }

            /// Returns the difference of another list from this list.
            #[inline]
            pub const fn difference(&self, other: &$sname) -> $sname {
                $sname {
                    $(
                        $member: self.$member && !other.$member,
                    )*
                    _unbuildable: crate::extensions::Unbuildable(())
                }
            }
        }

        impl std::fmt::Debug for $sname {
            #[allow(unused_assignments)]
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "[")?;

                let mut first = true;

                $(
                    if self.$member {
                        if !first { write!(f, ", ")? }
                        else { first = false; }
                        f.write_str(std::str::from_utf8($raw).unwrap())?;
                    }
                )*

                write!(f, "]")
            }
        }

        impl<'a, I> From<I> for $sname where I: IntoIterator<Item = &'a std::ffi::CStr> {
            fn from(names: I) -> Self {
                let mut extensions = Self::none();
                for name in names {
                    match name.to_bytes() {
                        $(
                            $raw => { extensions.$member = true; }
                        )*
                        _ => (),
                    }
                }
                extensions
            }
        }

        impl<'a> From<&'a $sname> for Vec<std::ffi::CString> {
            fn from(x: &'a $sname) -> Self {
                let mut data = Self::new();
                $(if x.$member { data.push(std::ffi::CString::new(&$raw[..]).unwrap()); })*
                data
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
