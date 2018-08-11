// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::str;

use Error;
use OomError;
use instance::loader::LoadingError;

macro_rules! extensions {
    ($sname:ident, $rawname:ident, $($ext:ident => $s:expr,)*) => (
        /// List of extensions that are enabled or available.
        #[derive(Copy, Clone, PartialEq, Eq)]
        #[allow(missing_docs)]
        pub struct $sname {
            $(
                pub $ext: bool,
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
            pub fn none() -> $sname {
                $sname {
                    $($ext: false,)*
                    _unbuildable: Unbuildable(())
                }
            }

            /// Returns the intersection of this list and another list.
            #[inline]
            pub fn intersection(&self, other: &$sname) -> $sname {
                $sname {
                    $(
                        $ext: self.$ext && other.$ext,
                    )*
                    _unbuildable: Unbuildable(())
                }
            }

            /// Returns the difference of another list from this list.
            #[inline]
            pub fn difference(&self, other: &$sname) -> $sname {
                $sname {
                    $(
                        $ext: self.$ext && !other.$ext,
                    )*
                    _unbuildable: Unbuildable(())
                }
            }
        }

        impl fmt::Debug for $sname {
            #[allow(unused_assignments)]
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                try!(write!(f, "["));

                let mut first = true;

                $(
                    if self.$ext {
                        if !first { try!(write!(f, ", ")); }
                        else { first = false; }
                        try!(f.write_str(str::from_utf8($s).unwrap()));
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

            // TODO: impl Iterator
            pub fn iter(&self) -> ::std::collections::hash_set::Iter<CString> { self.0.iter() }
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
                $(if x.$ext { data.insert(CString::new(&$s[..]).unwrap()); })*
                $rawname(data)
            }
        }

        impl<'a> From<&'a $rawname> for $sname {
            fn from(x: &'a $rawname) -> Self {
                let mut extensions = $sname::none();
                $(
                    if x.0.iter().any(|x| x.as_bytes() == &$s[..]) {
                        extensions.$ext = true;
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
    fn description(&self) -> &str {
        match *self {
            SupportedExtensionsError::LoadingError(_) => "failed to load the Vulkan shared library",
            SupportedExtensionsError::OomError(_) => "not enough memory available",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&error::Error> {
        match *self {
            SupportedExtensionsError::LoadingError(ref err) => Some(err),
            SupportedExtensionsError::OomError(ref err) => Some(err),
        }
    }
}

impl fmt::Display for SupportedExtensionsError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
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
            err @ Error::OutOfHostMemory => {
                SupportedExtensionsError::OomError(OomError::from(err))
            },
            err @ Error::OutOfDeviceMemory => {
                SupportedExtensionsError::OomError(OomError::from(err))
            },
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}
