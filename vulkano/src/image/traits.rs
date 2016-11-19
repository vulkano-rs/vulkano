// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use format::ClearValue;
use format::Format;
use format::PossibleFloatFormatDesc;
use format::PossibleUintFormatDesc;
use format::PossibleSintFormatDesc;
use format::PossibleDepthFormatDesc;
use format::PossibleStencilFormatDesc;
use format::PossibleDepthStencilFormatDesc;
use image::Dimensions;
use image::ImageDimensions;
use image::sys::Layout;
use image::sys::UnsafeImage;
use image::sys::UnsafeImageView;
use sampler::Sampler;

/// Trait for types that represent images.
pub unsafe trait Image {
    /// Returns the inner unsafe image object used by this image.
    fn inner(&self) -> &UnsafeImage;

    /// Returns the format of this image.
    #[inline]
    fn format(&self) -> Format {
        self.inner().format()
    }

    /// Returns true if the image is a color image.
    #[inline]
    fn has_color(&self) -> bool {
        let format = self.format();
        format.is_float() || format.is_uint() || format.is_sint()
    }

    /// Returns true if the image has a depth component. In other words, if it is a depth or a
    /// depth-stencil format. 
    #[inline]
    fn has_depth(&self) -> bool {
        let format = self.format();
        format.is_depth() || format.is_depth_stencil()
    }

    /// Returns true if the image has a stencil component. In other words, if it is a stencil or a
    /// depth-stencil format. 
    #[inline]
    fn has_stencil(&self) -> bool {
        let format = self.format();
        format.is_stencil() || format.is_depth_stencil()
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

unsafe impl<I: ?Sized> Image for Arc<I> where I: Image {
    #[inline]
    fn inner(&self) -> &UnsafeImage {
        (**self).inner()
    }
}

unsafe impl<'a, I: ?Sized + 'a> Image for &'a I where I: Image {
    #[inline]
    fn inner(&self) -> &UnsafeImage {
        (**self).inner()
    }
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
pub unsafe trait ImageView {
    fn parent(&self) -> &Image;

    /// Returns the dimensions of the image view.
    fn dimensions(&self) -> Dimensions;

    /// Returns the inner unsafe image view object used by this image view.
    fn inner(&self) -> &UnsafeImageView;

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

unsafe impl<'a, T: ?Sized + 'a> ImageView for &'a T where T: ImageView {
    #[inline]
    fn parent(&self) -> &Image {
        (**self).parent()
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

unsafe impl<T: ?Sized> ImageView for Arc<T> where T: ImageView {
    #[inline]
    fn parent(&self) -> &Image {
        (**self).parent()
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

pub unsafe trait AttachmentImageView: ImageView {
    fn accept(&self, initial_layout: Layout, final_layout: Layout) -> bool;
}
