// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

macro_rules! fns {
    ($struct_name:ident, { $($member:ident => $fn_struct:ident,)+ }) => {
        pub struct $struct_name {
            $(
                pub $member: ash::vk::$fn_struct,
            )+
        }

        impl $struct_name {
            pub fn load<F>(mut load_fn: F) -> $struct_name
                where F: FnMut(&CStr) -> *const c_void
            {
                $struct_name {
                    $(
                        $member: ash::vk::$fn_struct::load(&mut load_fn),
                    )+
                }
            }
        }
    };
}

pub use crate::autogen::{DeviceFunctions, EntryFunctions, InstanceFunctions};
pub(crate) use fns;
