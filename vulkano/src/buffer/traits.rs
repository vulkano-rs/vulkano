use std::ops::Range;
use std::sync::Arc;

use buffer::sys::UnsafeBuffer;
use command_buffer::Submission;
use memory::Content;

pub unsafe trait Buffer {
    /// Returns the inner buffer.
    // TODO: should be named "inner()" after https://github.com/rust-lang/rust/issues/12808 is fixed
    fn inner_buffer(&self) -> &UnsafeBuffer;

    /// Returns whether accessing a range of this buffer should signal a fence.
    fn needs_fence(&self, write: bool, Range<usize>) -> Option<bool>;

    unsafe fn gpu_access(&self, ranges: &mut Iterator<Item = AccessRange>,
                         submission: &Arc<Submission>) -> Vec<Arc<Submission>>;

    #[inline]
    fn size(&self) -> usize {
        self.inner_buffer().size()
    }
}

pub unsafe trait TypedBuffer: Buffer {
    type Content: ?Sized + 'static;

    #[inline]
    fn len(&self) -> usize where Self::Content: Content {
        self.size() / <Self::Content as Content>::indiv_size()
    }
}

#[derive(Debug, Clone)]
pub struct AccessRange {
    pub range: Range<usize>,
    pub write: bool
}
