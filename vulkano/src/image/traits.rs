// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use buffer::BufferAccess;
use device::Queue;
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

use SafeDeref;
use VulkanObject;

/// Utility trait.
pub unsafe trait Image {
    type Target: ImageAccess;

    fn into_image(self) -> Self::Target;
}

/// Trait for types that represent images.
pub unsafe trait ImageAccess {
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

    /// Returns the layout that the image has when it is first used in a primary command buffer,
    /// and the layout it must be returned to before the end of the command buffer.
    fn default_layout(&self) -> Layout;

    /// Returns true if an access to `self` (as defined by `self_first_layer`, `self_num_layers`,
    /// `self_first_mipmap` and `self_num_mipmaps`) potentially overlaps the same memory as an
    /// access to `other` (as defined by `other_offset` and `other_size`).
    ///
    /// If this function returns `false`, this means that we are allowed to access the offset/size
    /// of `self` at the same time as the offset/size of `other` without causing a data race.
    fn conflicts_buffer(&self, self_first_layer: u32, self_num_layers: u32, self_first_mipmap: u32,
                        self_num_mipmaps: u32, other: &BufferAccess, other_offset: usize,
                        other_size: usize) -> bool
    {
        // TODO: should we really provide a default implementation?
        false
    }

    /// Returns true if an access to `self` (as defined by `self_first_layer`, `self_num_layers`,
    /// `self_first_mipmap` and `self_num_mipmaps`) potentially overlaps the same memory as an
    /// access to `other` (as defined by `other_first_layer`, `other_num_layers`,
    /// `other_first_mipmap` and `other_num_mipmaps`).
    ///
    /// If this function returns `false`, this means that we are allowed to access the offset/size
    /// of `self` at the same time as the offset/size of `other` without causing a data race.
    fn conflicts_image(&self, self_first_layer: u32, self_num_layers: u32, self_first_mipmap: u32,
                       self_num_mipmaps: u32, other: &ImageAccess,
                       other_first_layer: u32, other_num_layers: u32, other_first_mipmap: u32,
                       other_num_mipmaps: u32) -> bool
    {
        // TODO: should we really provide a default implementation?

        // TODO: debug asserts to check for ranges

        if self.inner().internal_object() != other.inner().internal_object() {
            return false;
        }

        true
    }

    /// Returns a key that uniquely identifies the range given by
    /// first_layer/num_layers/first_mipmap/num_mipmaps.
    ///
    /// Two ranges that potentially overlap in memory should return the same key.
    ///
    /// The key is shared amongst all buffers and images, which means that you can make several
    /// different image objects share the same memory, or make some image objects share memory
    /// with buffers, as long as they return the same key.
    ///
    /// Since it is possible to accidentally return the same key for memory ranges that don't
    /// overlap, the `conflicts_image` or `conflicts_buffer` function should always be called to
    /// verify whether they actually overlap.
    fn conflict_key(&self, first_layer: u32, num_layers: u32, first_mipmap: u32, num_mipmaps: u32)
                    -> u64;

    /// Locks the resource for usage on the GPU. Returns `false` if the lock was already acquired.
    ///
    /// This function implementation should remember that it has been called and return `false` if
    /// it gets called a second time.
    ///
    /// The only way to know that the GPU has stopped accessing a queue is when the image object
    /// gets destroyed. Therefore you are encouraged to use temporary objects or handles (similar
    /// to a lock) in order to represent a GPU access.
    fn try_gpu_lock(&self, exclusive_access: bool, queue: &Queue) -> bool;

    /// Locks the resource for usage on the GPU. Supposes that the resource is already locked, and
    /// simply increases the lock by one.
    ///
    /// Must only be called after `try_gpu_lock()` succeeded.
    unsafe fn increase_gpu_lock(&self);
}

unsafe impl<T> ImageAccess for T where T: SafeDeref, T::Target: ImageAccess {
    #[inline]
    fn inner(&self) -> &UnsafeImage {
        (**self).inner()
    }

    #[inline]
    fn default_layout(&self) -> Layout {
        (**self).default_layout()
    }

    #[inline]
    fn conflict_key(&self, first_layer: u32, num_layers: u32, first_mipmap: u32, num_mipmaps: u32)
                    -> u64
    {
        (**self).conflict_key(first_layer, num_layers, first_mipmap, num_mipmaps)
    }

    #[inline]
    fn try_gpu_lock(&self, exclusive_access: bool, queue: &Queue) -> bool {
        (**self).try_gpu_lock(exclusive_access, queue)
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
        (**self).increase_gpu_lock()
    }
}

/// Extension trait for images. Checks whether the value `T` can be used as a clear value for the
/// given image.
// TODO: isn't that for image views instead?
pub unsafe trait ImageClearValue<T>: ImageAccess {
    fn decode(&self, T) -> Option<ClearValue>;
}

pub unsafe trait ImageContent<P>: ImageAccess {
    /// Checks whether pixels of type `P` match the format of the image.
    fn matches_format(&self) -> bool;
}

/// Utility trait.
pub unsafe trait IntoImageView {
    type Target: ImageView;

    fn into_image_view(self) -> Self::Target;
}

/// Trait for types that represent image views.
pub unsafe trait ImageView {
    fn parent(&self) -> &ImageAccess;

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

unsafe impl<T> ImageView for T where T: SafeDeref, T::Target: ImageView {
    #[inline]
    fn parent(&self) -> &ImageAccess {
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
