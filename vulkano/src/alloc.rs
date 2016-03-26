// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::mem;
use std::os::raw::c_void;

pub unsafe trait Alloc {
    fn alloc(&self, size: usize, alignment: usize) -> Result<*mut c_void, ()>;

    fn realloc(&self, original: *mut c_void, size: usize, alignment: usize) -> Result<*mut c_void, ()>;

    fn free(&self, *mut c_void);

    fn internal_free_notification(&self, size: usize);

    fn internal_allocation_notification(&self, size: usize);
}
