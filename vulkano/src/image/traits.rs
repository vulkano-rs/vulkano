// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::any::Any;
use std::sync::Arc;
use std::sync::mpsc::Sender;
use std::sync::mpsc::Receiver;

use buffer::Buffer;
use device::Queue;
use format::ClearValue;
use format::Format;
use image::Dimensions;
use image::ImageDimensions;
use image::sys::Layout;
use image::sys::UnsafeImage;
use image::sys::UnsafeImageView;
use sampler::Sampler;
use sync::AccessFlagBits;
use sync::PipelineStages;
use sync::Fence;
use sync::Semaphore;

use VulkanObject;

/// Trait for types that represent images.
// TODO: remove 'static + Send + Sync
pub unsafe trait Image: 'static + Send + Sync {
    /// Returns the inner unsafe image object used by this image.
    fn inner(&self) -> &UnsafeImage;

    /// Returns the format of this image.
    #[inline]
    fn format(&self) -> Format {
        self.inner().format()
    }

    /// Returns the number of samples of this image.
    #[inline]
    fn samples(&self) -> u32 {
        self.inner().samples()
    }

    /// Returns the dimensions of the image.
    #[inline]
    fn dimensions(&self) -> ImageDimensions {
        self.inner().dimensions()
    }

    /// Returns true if the image can be used as a source for blits.
    #[inline]
    fn supports_blit_source(&self) -> bool {
        self.inner().supports_blit_source()
    }

    /// Returns true if the image can be used as a destination for blits.
    #[inline]
    fn supports_blit_destination(&self) -> bool {
        self.inner().supports_blit_destination()
    }
}

/// Extension trait for `Image`. Types that implement this can be used in a `StdCommandBuffer`.
///
/// Each buffer and image used in a `StdCommandBuffer` have an associated state which is
/// represented by the `CommandListState` associated type of this trait. You can make multiple
/// buffers or images share the same state by making `is_same` return true.
pub unsafe trait TrackedImage: Image {
    /// State of the image in a list of commands.
    ///
    /// The `Any` bound is here for stupid reasons, sorry.
    // TODO: remove Any bound
    type CommandListState: Any + CommandListState<FinishedState = Self::FinishedState>;
    /// State of the buffer in a finished list of commands.
    type FinishedState: CommandBufferState;

    /// Returns true if TODO.
    ///
    /// If `is_same` returns true, then the type of `CommandListState` must be the same as for the
    /// other buffer. Otherwise a panic will occur.
    #[inline]
    fn is_same_buffer<B>(&self, other: &B) -> bool where B: Buffer {
        false
    }

    /// Returns true if TODO.
    ///
    /// If `is_same` returns true, then the type of `CommandListState` must be the same as for the
    /// other image. Otherwise a panic will occur.
    #[inline]
    fn is_same_image<I>(&self, other: &I) -> bool where I: Image {
        self.inner().internal_object() == other.inner().internal_object()
    }

    /// Returns the state of the image when it has not yet been used.
    fn initial_state(&self) -> Self::CommandListState;
}

/// Trait for objects that represent the state of a slice of the image in a list of commands.
pub trait CommandListState {
    type FinishedState: CommandBufferState;

    /// Returns a new state that corresponds to the moment after a slice of the image has been
    /// used in the pipeline. The parameters indicate in which way it has been used.
    ///
    /// If the transition should result in a pipeline barrier, then it must be returned by this
    /// function.
    // TODO: what should be the behavior if `num_command` is equal to the `num_command` of a
    // previous transition?
    fn transition(self, num_command: usize, image: &UnsafeImage, first_mipmap: u32,
                  num_mipmaps: u32, first_layer: u32, num_layers: u32, write: bool, layout: Layout,
                  stage: PipelineStages, access: AccessFlagBits)
                  -> (Self, Option<PipelineBarrierRequest>)
        where Self: Sized;

    /// Function called when the command buffer builder is turned into a real command buffer.
    ///
    /// This function can return an additional pipeline barrier that will be applied at the end
    /// of the command buffer.
    fn finish(self) -> (Self::FinishedState, Option<PipelineBarrierRequest>);
}

/// Requests that a pipeline barrier is created.
pub struct PipelineBarrierRequest {
    /// The number of the command after which the barrier should be placed. Must usually match
    /// the number that was passed to the previous call to `transition`, or 0 if the image hasn't
    /// been used yet.
    pub after_command_num: usize,

    /// The source pipeline stages of the transition.
    pub source_stage: PipelineStages,

    /// The destination pipeline stages of the transition.
    pub destination_stages: PipelineStages,

    /// If true, the pipeliner barrier is by region.
    pub by_region: bool,

    /// An optional memory barrier. See the docs of `PipelineMemoryBarrierRequest`.
    pub memory_barrier: Option<PipelineMemoryBarrierRequest>,
}

/// Requests that a memory barrier is created as part of the pipeline barrier.
///
/// By default, a pipeline barrier only guarantees that the source operations are executed before
/// the destination operations, but it doesn't make memory writes made by source operations visible
/// to the destination operations. In order to make so, you have to add a memory barrier.
///
/// The memory barrier always concerns the image that is currently being processed. You can't add
/// a memory barrier that concerns another resource.
pub struct PipelineMemoryBarrierRequest {
    pub first_mipmap: u32,
    pub num_mipmaps: u32,
    pub first_layer: u32,
    pub num_layers: u32,

    pub old_layout: Layout,
    pub new_layout: Layout,

    /// Source accesses.
    pub source_access: AccessFlagBits,
    /// Destination accesses.
    pub destination_access: AccessFlagBits,
}

/// Trait for objects that represent the state of the image in a command buffer.
pub trait CommandBufferState {
    /// Called right before the command buffer is submitted.
    // TODO: function should be unsafe because it must be guaranteed that a cb is submitted
    fn on_submit<I, F>(&self, image: &I, queue: &Arc<Queue>, fence: F) -> SubmitInfos
        where I: Image, F: FnOnce() -> Arc<Fence>;
}

pub struct SubmitInfos {
    pub pre_semaphore: Option<(Receiver<Arc<Semaphore>>, PipelineStages)>,
    pub post_semaphore: Option<Sender<Arc<Semaphore>>>,
    pub pre_barrier: Option<PipelineBarrierRequest>,
    pub post_barrier: Option<PipelineBarrierRequest>,
}

/// Extension trait for images. Checks whether the value `T` can be used as a clear value for the
/// given image.
// TODO: isn't that for image views instead?
pub unsafe trait ImageClearValue<T>: Image {
    fn decode(&self, T) -> Option<ClearValue>;
}

pub unsafe trait ImageContent<P>: Image {
    /// Checks whether pixels of type `P` match the format of the image.
    fn matches_format(&self) -> bool;
}

/// Trait for types that represent image views.
// TODO: remove 'static + Send + Sync
pub unsafe trait ImageView: 'static + Send + Sync {
    fn parent(&self) -> &Image;

    fn parent_arc(&Arc<Self>) -> Arc<Image> where Self: Sized;

    /// Returns the dimensions of the image view.
    fn dimensions(&self) -> Dimensions;

    /// Returns the inner unsafe image view object used by this image view.
    fn inner(&self) -> &UnsafeImageView;

    /// Returns the blocks of the parent image this image view overlaps.
    fn blocks(&self) -> Vec<(u32, u32)>;

    /// Returns the format of this view. This can be different from the parent's format.
    #[inline]
    fn format(&self) -> Format {
        self.inner().format()
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

unsafe impl<T> ImageView for Arc<T> where T: ImageView {
    #[inline]
    fn parent(&self) -> &Image {
        (**self).parent()
    }

    #[inline]
    fn parent_arc(me: &Arc<Self>) -> Arc<Image> {
        ImageView::parent_arc(&**me)
    }

    #[inline]
    fn inner(&self) -> &UnsafeImageView {
        (**self).inner()
    }

    #[inline]
    fn dimensions(&self) -> Dimensions {
        (**self).dimensions()
    }

    #[inline]
    fn blocks(&self) -> Vec<(u32, u32)> {
        (**self).blocks()
    }

    #[inline]
    fn descriptor_set_storage_image_layout(&self) -> Layout {
        (**self).descriptor_set_storage_image_layout()
    }
    #[inline]
    fn descriptor_set_combined_image_sampler_layout(&self) -> Layout {
        (**self).descriptor_set_combined_image_sampler_layout()
    }
    #[inline]
    fn descriptor_set_sampled_image_layout(&self) -> Layout {
        (**self).descriptor_set_sampled_image_layout()
    }
    #[inline]
    fn descriptor_set_input_attachment_layout(&self) -> Layout {
        (**self).descriptor_set_input_attachment_layout()
    }

    #[inline]
    fn identity_swizzle(&self) -> bool {
        (**self).identity_swizzle()
    }

    #[inline]
    fn can_be_sampled(&self, sampler: &Sampler) -> bool {
        (**self).can_be_sampled(sampler)
    }
}

pub unsafe trait TrackedImageView: ImageView {
    type Image: TrackedImage;

    fn image(&self) -> &Self::Image;
}

unsafe impl<T> TrackedImageView for Arc<T> where T: TrackedImageView {
    type Image = T::Image;

    #[inline]
    fn image(&self) -> &Self::Image {
        (**self).image()
    }
}

pub unsafe trait AttachmentImageView: ImageView {
    fn accept(&self, initial_layout: Layout, final_layout: Layout) -> bool;
}
