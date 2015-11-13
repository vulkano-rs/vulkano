use lock;

unsafe trait BufferLock {
    unsafe fn get_id(&self) -> u64;
}

impl<T> BufferLock for T where T: lock::Lock<Buffer> {

}

pub struct Buffer {
    id: ,
}

unsafe impl BufferLock for Buffer {
    #[inline]
    unsafe fn get_id(&self) -> u64 {
        self.id
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        error::check_result(unsafe { ffi::grDestroyBuffer(self.id) }).unwrap();
    }
}
