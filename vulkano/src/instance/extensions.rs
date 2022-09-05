// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

pub use crate::{
    extensions::{ExtensionRestriction, ExtensionRestrictionError, OneOfRequirements},
    Version,
};
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Sub, SubAssign};
use std::{
    ffi::{CStr, CString},
    fmt::Formatter,
};

// Generated by build.rs
include!(concat!(env!("OUT_DIR"), "/instance_extensions.rs"));

#[cfg(test)]
mod tests {
    use crate::instance::InstanceExtensions;
    use std::ffi::CString;

    #[test]
    fn empty_extensions() {
        let i: Vec<CString> = (&InstanceExtensions::empty()).into();
        assert!(i.get(0).is_none());
    }
}
