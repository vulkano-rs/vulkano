use std::ops::Range;
use std::sync::Arc;

use command_buffer::Submission;
use format::ClearValue;
use image::Layout;
use image::sys::UnsafeImage;
use image::sys::UnsafeImageView;
use sampler::Sampler;

pub unsafe trait Image {
    /// Returns the inner unsafe image object used by this image.
    // TODO: should be named "inner()" after https://github.com/rust-lang/rust/issues/12808 is fixed
    fn inner_image(&self) -> &UnsafeImage;

    //fn align(&self, subresource_range: ) -> ;

    /// Returns whether accessing a subresource of that image should signal a fence.
    fn needs_fence(&self, access: &mut Iterator<Item = AccessRange>) -> Option<bool>;

    unsafe fn gpu_access(&self, access: &mut Iterator<Item = AccessRange>,
                         submission: &Arc<Submission>) -> Vec<Arc<Submission>>;
}

pub unsafe trait ImageClearValue<T>: Image {
    fn decode(&self, T) -> ClearValue;
}

pub unsafe trait ImageView {
    fn parent(&self) -> &Image;

    fn inner(&self) -> &UnsafeImageView;

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
