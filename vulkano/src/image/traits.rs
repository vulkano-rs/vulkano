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

pub unsafe trait Image {
    /// Returns the inner unsafe image object used by this image.
    // TODO: should be named "inner()" after https://github.com/rust-lang/rust/issues/12808 is fixed
    fn inner_image(&self) -> &UnsafeImage;

    //fn align(&self, subresource_range: ) -> ;

    /// Returns the format of this image.
    #[inline]
    fn format(&self) -> Format {
        self.inner_image().format()
    }

    /// Returns the dimensions of the image.
    #[inline]
    fn dimensions(&self) -> Dimensions {
        self.inner_image().dimensions()
    }

    /// Returns whether accessing a subresource of that image should signal a fence.
    fn needs_fence(&self, access: &mut Iterator<Item = AccessRange>) -> Option<bool>;

    unsafe fn gpu_access(&self, access: &mut Iterator<Item = AccessRange>,
                         submission: &Arc<Submission>) -> Vec<Arc<Submission>>;
}

pub unsafe trait ImageClearValue<T>: Image {
    fn decode(&self, T) -> Option<ClearValue>;
}

pub unsafe trait ImageContent<P>: Image {
    /// Checks whether pixels of type `P` match the format of the image.
    fn matches_format(&self) -> bool;
}

pub unsafe trait ImageView {
    fn parent(&self) -> &Image;

    /// Returns the inner unsafe image view object used by this image view.
    // TODO: should be named "inner()" after https://github.com/rust-lang/rust/issues/12808 is fixed
    fn inner_view(&self) -> &UnsafeImageView;

    /// Returns the format of this view. This can be different from the parent's format.
    #[inline]
    fn format(&self) -> Format {
        self.inner_view().format()
    }

    /// Returns the image layout to use in a descriptor with the given subresource.
    fn descriptor_set_storage_image_layout(&self, AccessRange) -> Layout;
    /// Returns the image layout to use in a descriptor with the given subresource.
    fn descriptor_set_combined_image_sampler_layout(&self, AccessRange) -> Layout;
    /// Returns the image layout to use in a descriptor with the given subresource.
    fn descriptor_set_sampled_image_layout(&self, AccessRange) -> Layout;
    /// Returns the image layout to use in a descriptor with the given subresource.
    fn descriptor_set_input_attachment_layout(&self, AccessRange) -> Layout;

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

pub struct AccessRange {
    pub mipmap_levels_range: Range<u32>,
    pub array_layers_range: Range<u32>,
    pub write: bool,
}
