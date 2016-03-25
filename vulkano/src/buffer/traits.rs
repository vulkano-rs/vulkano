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

    /// Called when a command buffer that uses this buffer is being built.
    ///
    /// Must return true if the command buffer should include a pipeline barrier at the start,
    /// to read from what the host wrote, and a pipeline barrier at the end, to flush caches and
    /// allows the host to read the data.
    fn host_accesses(&self, block: usize) -> bool;

    /// Given a range, returns the list of blocks which each range is contained in.
    ///
    /// Each block must have a unique number. Hint: it can simply be the offset of the start of the
    /// block.
    /// Calling this function multiple times with the same parameter must always return the same
    /// value.
    /// The return value must not be empty.
    fn blocks(&self, range: Range<usize>) -> Vec<usize>;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessRange {
    pub block: usize,
    pub write: bool,
}
