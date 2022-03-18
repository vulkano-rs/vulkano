// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{sys::UnsafeImage, ImageDescriptorLayouts, ImageDimensions, ImageLayout, SampleCount};
use crate::{
    format::{ClearValue, Format, FormatFeatures},
    SafeDeref,
};
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

/// Trait for types that represent the way a GPU can access an image.
pub unsafe trait ImageAccess: Send + Sync {
    /// Returns the inner unsafe image object used by this image.
    fn inner(&self) -> ImageInner;

    /// Returns the format of this image.
    #[inline]
    fn format(&self) -> Format {
        self.inner().image.format().unwrap()
    }

    /// Returns the number of mipmap levels of this image.
    #[inline]
    fn mip_levels(&self) -> u32 {
        // TODO: not necessarily correct because of the new inner() design?
        self.inner().image.mip_levels()
    }

    /// Returns the number of samples of this image.
    #[inline]
    fn samples(&self) -> SampleCount {
        self.inner().image.samples()
    }

    /// Returns the dimensions of the image.
    #[inline]
    fn dimensions(&self) -> ImageDimensions {
        // TODO: not necessarily correct because of the new inner() design?
        self.inner().image.dimensions()
    }

    /// Returns the features supported by the image's format.
    #[inline]
    fn format_features(&self) -> &FormatFeatures {
        self.inner().image.format_features()
    }

    /// When images are created their memory layout is initially `Undefined` or `Preinitialized`.
    /// This method allows the image memory barrier creation process to signal when an image
    /// has been transitioned out of its initial `Undefined` or `Preinitialized` state. This
    /// allows vulkano to avoid creating unnecessary image memory barriers between future
    /// uses of the image.
    ///
    /// ## Unsafe
    ///
    /// If a user calls this method outside of the intended context and signals that the layout
    /// is no longer `Undefined` or `Preinitialized` when it is still in an `Undefined` or
    /// `Preinitialized` state, this may result in the vulkan implementation attempting to use
    /// an image in an invalid layout. The same problem must be considered by the implementer
    /// of the method.
    unsafe fn layout_initialized(&self) {}

    fn is_layout_initialized(&self) -> bool {
        false
    }

    unsafe fn initial_layout(&self) -> ImageLayout {
        self.inner().image.initial_layout()
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
    unsafe fn forced_undefined_initial_layout(
        self,
        preinitialized: bool,
    ) -> Arc<ImageAccessFromUndefinedLayout<Self>>
    where
        Self: Sized,
    {
        Arc::new(ImageAccessFromUndefinedLayout {
            image: self,
            preinitialized,
        })
    }

    /// Returns an [`ImageDescriptorLayouts`] structure specifying the image layout to use
    /// in descriptors of various kinds.
    ///
    /// This must return `Some` if the image is to be used to create an image view.
    fn descriptor_layouts(&self) -> Option<ImageDescriptorLayouts>;

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

    /// Returns the current mip level that is accessed by the gpu
    fn current_mip_levels_access(&self) -> std::ops::Range<u32>;

    /// Returns the current array layer that is accessed by the gpu
    fn current_array_layers_access(&self) -> std::ops::Range<u32>;
}

/// Inner information about an image.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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

impl PartialEq for dyn ImageAccess {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner()
    }
}

impl Eq for dyn ImageAccess {}

impl Hash for dyn ImageAccess {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
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
where
    I: ImageAccess,
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
    fn descriptor_layouts(&self) -> Option<ImageDescriptorLayouts> {
        self.image.descriptor_layouts()
    }

    #[inline]
    fn conflict_key(&self) -> u64 {
        self.image.conflict_key()
    }

    fn current_mip_levels_access(&self) -> std::ops::Range<u32> {
        self.image.current_mip_levels_access()
    }

    fn current_array_layers_access(&self) -> std::ops::Range<u32> {
        self.image.current_array_layers_access()
    }
}

impl<I> PartialEq for ImageAccessFromUndefinedLayout<I>
where
    I: ImageAccess,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner() == other.inner()
    }
}

impl<I> Eq for ImageAccessFromUndefinedLayout<I> where I: ImageAccess {}

impl<I> Hash for ImageAccessFromUndefinedLayout<I>
where
    I: ImageAccess,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner().hash(state);
    }
}

/// Extension trait for images. Checks whether the value `T` can be used as a clear value for the
/// given image.
// TODO: isn't that for image views instead?
pub unsafe trait ImageClearValue<T>: ImageAccess {
    fn decode(&self, value: T) -> Option<ClearValue>;
}

pub unsafe trait ImageContent<P>: ImageAccess {
    /// Checks whether pixels of type `P` match the format of the image.
    fn matches_format(&self) -> bool;
}

unsafe impl<T> ImageAccess for T
where
    T: SafeDeref + Send + Sync,
    T::Target: ImageAccess,
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
    fn descriptor_layouts(&self) -> Option<ImageDescriptorLayouts> {
        (**self).descriptor_layouts()
    }

    #[inline]
    fn conflict_key(&self) -> u64 {
        (**self).conflict_key()
    }

    #[inline]
    unsafe fn layout_initialized(&self) {
        (**self).layout_initialized();
    }

    #[inline]
    fn is_layout_initialized(&self) -> bool {
        (**self).is_layout_initialized()
    }

    fn current_mip_levels_access(&self) -> std::ops::Range<u32> {
        (**self).current_mip_levels_access()
    }

    fn current_array_layers_access(&self) -> std::ops::Range<u32> {
        (**self).current_array_layers_access()
    }
}
