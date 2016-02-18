use std::mem;
use std::os::raw::c_void;

pub unsafe trait Alloc {
    fn alloc(&self, size: usize, alignment: usize) -> Result<*mut c_void, ()>;

    fn realloc(&self, original: *mut c_void, size: usize, alignment: usize) -> Result<*mut c_void, ()>;

    fn free(&self, *mut c_void);

    fn internal_free_notification(&self, size: usize);

    fn internal_allocation_notification(&self, size: usize);
}
