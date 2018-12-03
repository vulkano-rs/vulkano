// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use buffer::BufferAccess;
use format::ClearValue;
use format::Format;
use format::PossibleDepthFormatDesc;
use format::PossibleDepthStencilFormatDesc;
use format::PossibleFloatFormatDesc;
use format::PossibleSintFormatDesc;
use format::PossibleStencilFormatDesc;
use format::PossibleUintFormatDesc;
use image::Dimensions;
use image::ImageDimensions;
use image::ImageLayout;
use image::sys::UnsafeImage;
use image::sys::UnsafeImageView;
use sampler::Sampler;
use sync::AccessError;

use SafeDeref;

/// Trait for types that represent the way a GPU can access an image.
pub unsafe trait ImageAccess {
    /// Returns the inner unsafe image object used by this image.
    fn inner(&self) -> ImageInner;

    /// Returns the format of this image.
    #[inline]
    fn format(&self) -> Format {
        self.inner().image.format()
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

    /// Returns the number of mipmap levels of this image.
    #[inline]
    fn mipmap_levels(&self) -> u32 {
        // TODO: not necessarily correct because of the new inner() design?
        self.inner().image.mipmap_levels()
    }

    /// Returns the number of samples of this image.
    #[inline]
    fn samples(&self) -> u32 {
        self.inner().image.samples()
    }

    /// Returns the dimensions of the image.
    #[inline]
    fn dimensions(&self) -> ImageDimensions {
        // TODO: not necessarily correct because of the new inner() design?
        self.inner().image.dimensions()
    }

    /// Returns true if the image can be used as a source for blits.
    #[inline]
    fn supports_blit_source(&self) -> bool {
        self.inner().image.supports_blit_source()
    }

    /// Returns true if the image can be used as a destination for blits.
    #[inline]
    fn supports_blit_destination(&self) -> bool {
        self.inner().image.supports_blit_destination()
    }

    /// Returns the layout that the image has when it is first used in a primary command buffer.
    ///
    /// The first time you use an image in an `AutoCommandBufferBuilder`, vulkano will suppose that
    /// the image is in the layout returned by this function. Later when the command buffer is
    /// submitted vulkano will check whether the image is actually in this layout, and if it is not
    /// the case then an error will be returned.
    /// TODO: ^ that check is not yet implemented
    fn initial_layout_requirement(&self) -> ImageLayout;

    /// Returns the layout that the image must be returned to before the end of the command buffer.
    ///
    /// When an image is used in an `AutoCommandBufferBuilder` vulkano will automatically
    /// transition this image to the layout returned by this function at the end of the command
    /// buffer, if necessary.
    ///
    /// Except for special cases, this value should likely be the same as the one returned by
    /// `initial_layout_requirement` so that the user can submit multiple command buffers that use
    /// this image one after the other.
    fn final_layout_requirement(&self) -> ImageLayout;

    /// Wraps around this `ImageAccess` and returns an identical `ImageAccess` but whose initial
    /// layout requirement is either `Undefined` or `Preinitialized`.
    #[inline]
    unsafe fn forced_undefined_initial_layout(self, preinitialized: bool)
                                              -> ImageAccessFromUndefinedLayout<Self>
        where Self: Sized
    {
        ImageAccessFromUndefinedLayout {
            image: self,
            preinitialized: preinitialized,
        }
    }

    /// Returns true if an access to `self` potentially overlaps the same memory as an
    /// access to `other`.
    ///
    /// If this function returns `false`, this means that we are allowed to access the content
    /// of `self` at the same time as the content of `other` without causing a data race.
    ///
    /// Note that the function must be transitive. In other words if `conflicts(a, b)` is true and
    /// `conflicts(b, c)` is true, then `conflicts(a, c)` must be true as well.
    fn conflicts_buffer(&self, other: &BufferAccess) -> bool;

    /// Returns true if an access to `self` potentially overlaps the same memory as an
    /// access to `other`.
    ///
    /// If this function returns `false`, this means that we are allowed to access the content
    /// of `self` at the same time as the content of `other` without causing a data race.
    ///
    /// Note that the function must be transitive. In other words if `conflicts(a, b)` is true and
    /// `conflicts(b, c)` is true, then `conflicts(a, c)` must be true as well.
    fn conflicts_image(&self, other: &ImageAccess) -> bool;

    /// Returns a key that uniquely identifies the memory content of the image.
    /// Two ranges that potentially overlap in memory must return the same key.
    ///
    /// The key is shared amongst all buffers and images, which means that you can make several
    /// different image objects share the same memory, or make some image objects share memory
    /// with buffers, as long as they return the same key.
    ///
    /// Since it is possible to accidentally return the same key for memory ranges that don't
    /// overlap, the `conflicts_image` or `conflicts_buffer` function should always be called to
    /// verify whether they actually overlap.
    fn conflict_key(&self) -> u64;

    /// Locks the resource for usage on the GPU. Returns an error if the lock can't be acquired.
    ///
    /// After this function returns `Ok`, you are authorized to use the image on the GPU. If the
    /// GPU operation requires an exclusive access to the image (which includes image layout
    /// transitions) then `exclusive_access` should be true.
    ///
    /// The `expected_layout` is the layout we expect the image to be in when we lock it. If the
    /// actual layout doesn't match this expected layout, then an error should be returned. If
    /// `Undefined` is passed, that means that the caller doesn't care about the actual layout,
    /// and that a layout mismatch shouldn't return an error.
    ///
    /// This function exists to prevent the user from causing a data race by reading and writing
    /// to the same resource at the same time.
    ///
    /// If you call this function, you should call `unlock()` once the resource is no longer in use
    /// by the GPU. The implementation is not expected to automatically perform any unlocking and
    /// can rely on the fact that `unlock()` is going to be called.
    fn try_gpu_lock(&self, exclusive_access: bool, expected_layout: ImageLayout)
                    -> Result<(), AccessError>;

    /// Locks the resource for usage on the GPU. Supposes that the resource is already locked, and
    /// simply increases the lock by one.
    ///
    /// Must only be called after `try_gpu_lock()` succeeded.
    ///
    /// If you call this function, you should call `unlock()` once the resource is no longer in use
    /// by the GPU. The implementation is not expected to automatically perform any unlocking and
    /// can rely on the fact that `unlock()` is going to be called.
    unsafe fn increase_gpu_lock(&self);

    /// Unlocks the resource previously acquired with `try_gpu_lock` or `increase_gpu_lock`.
    ///
    /// If the GPU operation that we unlock from transitioned the image to another layout, then
    /// it should be passed as parameter.
    ///
    /// A layout transition requires exclusive access to the image, which means two things:
    ///
    /// - The implementation can panic if it finds out that the layout is not the same as it
    ///   currently is and that it is not locked in exclusive mode.
    /// - There shouldn't be any possible race between `unlock` and `try_gpu_lock`, since
    ///   `try_gpu_lock` should fail if the image is already locked in exclusive mode.
    ///
    /// # Safety
    ///
    /// - Must only be called once per previous lock.
    /// - The transitioned layout must be supported by the image (eg. the layout shouldn't be
    ///   `ColorAttachmentOptimal` if the image wasn't created with the `color_attachment` usage).
    /// - The transitioned layout must not be `Undefined`.
    ///
    unsafe fn unlock(&self, transitioned_layout: Option<ImageLayout>);
}

/// Inner information about an image.
#[derive(Copy, Clone, Debug)]
pub struct ImageInner<'a> {
    /// The underlying image object.
    pub image: &'a UnsafeImage,

    /// The first layer of `image` to consider.
    pub first_layer: usize,

    /// The number of layers of `image` to consider.
    pub num_layers: usize,

    /// The first mipmap level of `image` to consider.
    pub first_mipmap_level: usize,

    /// The number of mipmap levels of `image` to consider.
    pub num_mipmap_levels: usize,
}

unsafe impl<T> ImageAccess for T
    where T: SafeDeref,
          T::Target: ImageAccess
{
    #[inline]
    fn inner(&self) -> ImageInner {
        (**self).inner()
    }

    #[inline]
    fn initial_layout_requirement(&self) -> ImageLayout {
        (**self).initial_layout_requirement()
    }

    #[inline]
    fn final_layout_requirement(&self) -> ImageLayout {
        (**self).final_layout_requirement()
    }

    #[inline]
    fn conflicts_buffer(&self, other: &BufferAccess) -> bool {
        (**self).conflicts_buffer(other)
    }

    #[inline]
    fn conflicts_image(&self, other: &ImageAccess) -> bool {
        (**self).conflicts_image(other)
    }

    #[inline]
    fn conflict_key(&self) -> u64 {
        (**self).conflict_key()
    }

    #[inline]
    fn try_gpu_lock(&self, exclusive_access: bool, expected_layout: ImageLayout)
                    -> Result<(), AccessError> {
        (**self).try_gpu_lock(exclusive_access, expected_layout)
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
        (**self).increase_gpu_lock()
    }

    #[inline]
    unsafe fn unlock(&self, transitioned_layout: Option<ImageLayout>) {
        (**self).unlock(transitioned_layout)
    }
}

/// Wraps around an object that implements `ImageAccess` and modifies the initial layout
/// requirement to be either `Undefined` or `Preinitialized`.
#[derive(Debug, Copy, Clone)]
pub struct ImageAccessFromUndefinedLayout<I> {
    image: I,
    preinitialized: bool,
}

unsafe impl<I> ImageAccess for ImageAccessFromUndefinedLayout<I>
    where I: ImageAccess
{
    #[inline]
    fn inner(&self) -> ImageInner {
        self.image.inner()
    }

    #[inline]
    fn initial_layout_requirement(&self) -> ImageLayout {
        if self.preinitialized {
            ImageLayout::Preinitialized
        } else {
            ImageLayout::Undefined
        }
    }

    #[inline]
    fn final_layout_requirement(&self) -> ImageLayout {
        self.image.final_layout_requirement()
    }

    #[inline]
    fn conflicts_buffer(&self, other: &BufferAccess) -> bool {
        self.image.conflicts_buffer(other)
    }

    #[inline]
    fn conflicts_image(&self, other: &ImageAccess) -> bool {
        self.image.conflicts_image(other)
    }

    #[inline]
    fn conflict_key(&self) -> u64 {
        self.image.conflict_key()
    }

    #[inline]
    fn try_gpu_lock(&self, exclusive_access: bool, expected_layout: ImageLayout)
                    -> Result<(), AccessError> {
        self.image.try_gpu_lock(exclusive_access, expected_layout)
    }

    #[inline]
    unsafe fn increase_gpu_lock(&self) {
        self.image.increase_gpu_lock()
    }

    #[inline]
    unsafe fn unlock(&self, new_layout: Option<ImageLayout>) {
        self.image.unlock(new_layout)
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

/// Trait for types that represent the GPU can access an image view.
pub unsafe trait ImageViewAccess {
    fn parent(&self) -> &ImageAccess;

    /// Returns the dimensions of the image view.
    fn dimensions(&self) -> Dimensions;

    /// Returns the inner unsafe image view object used by this image view.
    fn inner(&self) -> &UnsafeImageView;

    /// Returns the format of this view. This can be different from the parent's format.
    #[inline]
    fn format(&self) -> Format {
        // TODO: remove this default impl
        self.inner().format()
    }

    #[inline]
    fn samples(&self) -> u32 {
        self.parent().samples()
    }

    /// Returns the image layout to use in a descriptor with the given subresource.
    fn descriptor_set_storage_image_layout(&self) -> ImageLayout;
    /// Returns the image layout to use in a descriptor with the given subresource.
    fn descriptor_set_combined_image_sampler_layout(&self) -> ImageLayout;
    /// Returns the image layout to use in a descriptor with the given subresource.
    fn descriptor_set_sampled_image_layout(&self) -> ImageLayout;
    /// Returns the image layout to use in a descriptor with the given subresource.
    fn descriptor_set_input_attachment_layout(&self) -> ImageLayout;

    /// Returns true if the view doesn't use components swizzling.
    ///
    /// Must be true when the view is used as a framebuffer attachment or TODO: I don't remember
    /// the other thing.
    fn identity_swizzle(&self) -> bool;

    /// Returns true if the given sampler can be used with this image view.
    ///
    /// This method should check whether the sampler's configuration can be used with the format
    /// of the view.
    // TODO: return a Result and propagate it when binding to a descriptor set
    fn can_be_sampled(&self, _sampler: &Sampler) -> bool {
        true /* FIXME */
    }

    //fn usable_as_render_pass_attachment(&self, ???) -> Result<(), ???>;
}

unsafe impl<T> ImageViewAccess for T
    where T: SafeDeref,
          T::Target: ImageViewAccess
{
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
    fn descriptor_set_storage_image_layout(&self) -> ImageLayout {
        (**self).descriptor_set_storage_image_layout()
    }
    #[inline]
    fn descriptor_set_combined_image_sampler_layout(&self) -> ImageLayout {
        (**self).descriptor_set_combined_image_sampler_layout()
    }
    #[inline]
    fn descriptor_set_sampled_image_layout(&self) -> ImageLayout {
        (**self).descriptor_set_sampled_image_layout()
    }
    #[inline]
    fn descriptor_set_input_attachment_layout(&self) -> ImageLayout {
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

pub unsafe trait AttachmentImageView: ImageViewAccess {
    fn accept(&self, initial_layout: ImageLayout, final_layout: ImageLayout) -> bool;
}
