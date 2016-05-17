// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::ops::Range;
use std::sync::Arc;

use command_buffer::Submission;
use format::ClearValue;
use format::Format;
use image::sys::Dimensions;
use image::sys::Layout;
use image::sys::UnsafeImage;
use image::sys::UnsafeImageView;
use sampler::Sampler;
use sync::PipelineBarrier;
use sync::Semaphore;

pub unsafe trait Image: 'static + Send + Sync {
    type CbConstructionState;
    type SyncState;

    /// Returns the inner unsafe image object used by this image.
    // TODO: should be named "inner()" after https://github.com/rust-lang/rust/issues/12808 is fixed
    ///
    /// Two different implementations of the `Image` trait must never return the same unsafe
    /// image.
    fn inner_image(&self) -> &UnsafeImage;

    /// Returns the format of this image.
    #[inline]
    fn format(&self) -> Format {
        self.inner_image().format()
    }

    #[inline]
    fn samples(&self) -> u32 {
        self.inner_image().samples()
    }

    /// Returns the dimensions of the image.
    #[inline]
    fn dimensions(&self) -> Dimensions {
        self.inner_image().dimensions()
    }
}

pub unsafe trait ImageClearValue<T>: Image {
    fn decode(&self, T) -> Option<ClearValue>;
}

pub unsafe trait ImageContent<P>: Image {
    /// Checks whether pixels of type `P` match the format of the image.
    fn matches_format(&self) -> bool;
}

pub unsafe trait ImageView: 'static + Send + Sync {
    /// Returns the inner unsafe image view object used by this image view.
    // TODO: should be named "inner()" after https://github.com/rust-lang/rust/issues/12808 is fixed
    fn inner_view(&self) -> &UnsafeImageView;

    /// Returns the blocks of the parent image this image view overlaps.
    fn blocks(&self) -> Vec<(u32, u32)>;

    /// Returns the format of this view. This can be different from the parent's format.
    #[inline]
    fn format(&self) -> Format {
        self.inner_view().format()
    }

    #[inline]
    fn samples(&self) -> u32 {
        self.parent().samples()
    }

    /// Returns the image layout to use in a descriptor with the given subresource.
    fn descriptor_set_storage_image_layout(&self) -> Layout;
    /// Returns the image layout to use in a descriptor with the given subresource.
    fn descriptor_set_combined_image_sampler_layout(&self) -> Layout;
    /// Returns the image layout to use in a descriptor with the given subresource.
    fn descriptor_set_sampled_image_layout(&self) -> Layout;
    /// Returns the image layout to use in a descriptor with the given subresource.
    fn descriptor_set_input_attachment_layout(&self) -> Layout;

    /// Returns true if the view doesn't use components swizzling.
    ///
    /// Must be true when the view is used as a framebuffer attachment or TODO: I don't remember
    /// the other thing.
    fn identity_swizzle(&self) -> bool;

    /// Returns true if the given sampler can be used with this image view.
    ///
    /// This method should check whether the sampler's configuration can be used with the format
    /// of the view.
    // TODO: return a Result
    fn can_be_sampled(&self, sampler: &Sampler) -> bool { true /* FIXME */ }

    //fn usable_as_render_pass_attachment(&self, ???) -> Result<(), ???>;
}

pub unsafe trait AttachmentImageView: ImageView {
    fn accept(&self, initial_layout: Layout, final_layout: Layout) -> bool;
}

pub struct GpuAccessResult {
    pub dependencies: Vec<Arc<Submission>>,
    pub additional_wait_semaphore: Option<Arc<Semaphore>>,
    pub additional_signal_semaphore: Option<Arc<Semaphore>>,
    pub before_transitions: Vec<Transition>,
    pub after_transitions: Vec<Transition>,
}

pub unsafe trait TransferSourceImage: Image {
    fn command_buffer_transfer_source(&self, range: Range<usize>,
                                      prev_barrier: Option<&mut PipelineBarrier>,
                                      state: &mut Option<(CbConstructionState, SyncState)>)
                                      -> Option<PipelineBarrier>;
}

pub unsafe trait TransferDestinationImage: Image {
    fn command_buffer_transfer_destination(&self, range: Range<usize>,
                                           prev_barrier: Option<&mut PipelineBarrier>,
                                           state: &mut Option<(CbConstructionState, SyncState)>)
                                           -> Option<PipelineBarrier>;
}

pub unsafe trait FramebufferAttachmentImage: Image {
    fn command_buffer_render_pass_enter(&self, range: Range<usize>,
                                        prev_barrier: Option<&mut PipelineBarrier>,
                                        state: &mut Option<(CbConstructionState, SyncState)>)
                                        -> Option<PipelineBarrier>;
}
