// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::iter::Empty;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Weak;

use command_buffer::Submission;
use device::Device;
use format::ClearValue;
use format::FormatDesc;
use format::FormatTy;
use image::Dimensions;
use image::ImageDimensions;
use image::ViewType;
use image::sys::ImageCreationError;
use image::sys::Layout;
use image::sys::UnsafeImage;
use image::sys::UnsafeImageView;
use image::sys::Usage;
use image::traits::Image;
use image::traits::ImageClearValue;
use image::traits::ImageContent;
use image::traits::ImageView;
use memory::pool::AllocLayout;
use memory::pool::MemoryPool;
use memory::pool::MemoryPoolAlloc;
use memory::pool::StdMemoryPool;
use sync::Sharing;

/// Image whose purpose is to be used as a framebuffer attachment.
///
/// The image is always two-dimensional and has only one mipmap, but it can have any kind of
/// format. Trying to use a format that the backend doesn't support for rendering will result in
/// an error being returned when creating the image. Once you have an `AttachmentImage`, you are
/// guaranteed that you will be able to draw on it.
///
/// The template parameter of `AttachmentImage` is a type that describes the format of the image.
///
/// # Regular vs transient
///
/// Calling `AttachmentImage::new` will create a regular image, while calling
/// `AttachmentImage::transient` will create a *transient* image. Transient image are only
/// relevant for images that serve as attachments, so `AttachmentImage` is the only type of
/// image in vulkano that provides a shortcut for this.
///
/// A transient image is a special kind of image whose content is undefined outside of render
/// passes. Once you finish drawing, reading from it will returned undefined data (which can be
/// either valid or garbage, depending on the implementation).
///
/// This gives a hint to the Vulkan implementation that it is possible for the image's content to
/// live exclusively in some cache memory, and that no real memory has to be allocated for it.
///
/// In other words, if you are going to read from the image after drawing to it, use a regular
/// image. If you don't need to read from it (for example if it's some kind of intermediary color,
/// or a depth buffer that is only used once) then use a transient image as it may improve
/// performances.
///
// TODO: forbid reading transient images outside render passes?
#[derive(Debug)]
pub struct AttachmentImage<F, A = Arc<StdMemoryPool>> where A: MemoryPool {
    // Inner implementation.
    image: UnsafeImage,

    // We maintain a view of the whole image since we will need it when rendering.
    view: UnsafeImageView,

    // Memory used to back the image.
    memory: A::Alloc,

    // Format.
    format: F,

    // Layout to use when the image is used as a framebuffer attachment.
    // Must be either "depth-stencil optimal" or "color optimal".
    attachment_layout: Layout,

    // Additional info behind a mutex.
    guarded: Mutex<Guarded>,
}

#[derive(Debug)]
struct Guarded {
    // If false, the image is still in the undefined layout.
    correct_layout: bool,

    // The latest submission that used the image. Used for synchronization purposes.
    latest_submission: Option<Weak<Submission>>,    // TODO: can use `Weak::new()` once it's stabilized
}

impl<F> AttachmentImage<F> {
    /// Creates a new image with the given dimensions and format.
    ///
    /// Returns an error if the dimensions are too large or if the backend doesn't support this
    /// format as a framebuffer attachment.
    pub fn new(device: &Arc<Device>, dimensions: [u32; 2], format: F)
               -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
        where F: FormatDesc
    {
        let base_usage = Usage {
            transfer_source: true,
            transfer_dest: true,
            sampled: true,
            .. Usage::none()
        };

        AttachmentImage::new_impl(device, dimensions, format, base_usage)
    }

    /// Same as `new`, except that the image will be transient.
    ///
    /// A transient image is special because its content is undefined outside of a render pass.
    /// This means that the implementation has the possibility to not allocate any memory for it.
    pub fn transient(device: &Arc<Device>, dimensions: [u32; 2], format: F)
                     -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
        where F: FormatDesc
    {
        let base_usage = Usage {
            transient_attachment: true,
            .. Usage::none()
        };

        AttachmentImage::new_impl(device, dimensions, format, base_usage)
    }

    fn new_impl(device: &Arc<Device>, dimensions: [u32; 2], format: F, base_usage: Usage)
                -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
        where F: FormatDesc
    {
        let is_depth = match format.format().ty() {
            FormatTy::Depth => true,
            FormatTy::DepthStencil => true,
            FormatTy::Stencil => true,
            FormatTy::Compressed => panic!(),
            _ => false
        };

        let usage = Usage {
            color_attachment: !is_depth,
            depth_stencil_attachment: is_depth,
            input_attachment: true,
            .. base_usage
        };

        let (image, mem_reqs) = unsafe {
            try!(UnsafeImage::new(device, &usage, format.format(),
                                  ImageDimensions::Dim2d { width: dimensions[0], height: dimensions[1], array_layers: 1, cubemap_compatible: false },
                                  1, 1, Sharing::Exclusive::<Empty<u32>>, false, false))
        };

        let mem_ty = {
            let device_local = device.physical_device().memory_types()
                                     .filter(|t| (mem_reqs.memory_type_bits & (1 << t.id())) != 0)
                                     .filter(|t| t.is_device_local());
            let any = device.physical_device().memory_types()
                            .filter(|t| (mem_reqs.memory_type_bits & (1 << t.id())) != 0);
            device_local.chain(any).next().unwrap()
        };

        let mem = try!(MemoryPool::alloc(&Device::standard_pool(device), mem_ty,
                                         mem_reqs.size, mem_reqs.alignment, AllocLayout::Optimal));
        debug_assert!((mem.offset() % mem_reqs.alignment) == 0);
        unsafe { try!(image.bind_memory(mem.memory(), mem.offset())); }

        let view = unsafe {
            try!(UnsafeImageView::raw(&image, ViewType::Dim2d, 0 .. 1, 0 .. 1))
        };

        Ok(Arc::new(AttachmentImage {
            image: image,
            view: view,
            memory: mem,
            format: format,
            attachment_layout: if is_depth { Layout::DepthStencilAttachmentOptimal }
                               else { Layout::ColorAttachmentOptimal },
            guarded: Mutex::new(Guarded {
                correct_layout: false,
                latest_submission: None,
            }),
        }))
    }
}

impl<F, A> AttachmentImage<F, A> where A: MemoryPool {
    /// Returns the dimensions of the image.
    #[inline]
    pub fn dimensions(&self) -> [u32; 2] {
        let dims = self.image.dimensions();
        [dims.width(), dims.height()]
    }
}

unsafe impl<F, A> Image for AttachmentImage<F, A> where F: 'static + Send + Sync, A: MemoryPool {
    #[inline]
    fn inner(&self) -> &UnsafeImage {
        &self.image
    }
}

unsafe impl<F, A> ImageClearValue<F::ClearValue> for AttachmentImage<F, A>
    where F: FormatDesc + 'static + Send + Sync, A: MemoryPool
{
    #[inline]
    fn decode(&self, value: F::ClearValue) -> Option<ClearValue> {
        Some(self.format.decode_clear_value(value))
    }
}

unsafe impl<P, F, A> ImageContent<P> for AttachmentImage<F, A>
    where F: 'static + Send + Sync, A: MemoryPool
{
    #[inline]
    fn matches_format(&self) -> bool {
        true        // FIXME:
    }
}

unsafe impl<F, A> ImageView for AttachmentImage<F, A>
    where F: 'static + Send + Sync, A: MemoryPool
{
    #[inline]
    fn parent(&self) -> &Image {
        self
    }

    #[inline]
    fn parent_arc(me: &Arc<Self>) -> Arc<Image> where Self: Sized {
        me.clone() as Arc<_>
    }

    #[inline]
    fn dimensions(&self) -> Dimensions {
        let dims = self.image.dimensions();
        Dimensions::Dim2d { width: dims.width(), height: dims.height() }
    }

    #[inline]
    fn blocks(&self) -> Vec<(u32, u32)> {
        vec![(0, 0)]
    }

    #[inline]
    fn inner(&self) -> &UnsafeImageView {
        &self.view
    }

    #[inline]
    fn descriptor_set_storage_image_layout(&self) -> Layout {
        Layout::ShaderReadOnlyOptimal
    }

    #[inline]
    fn descriptor_set_combined_image_sampler_layout(&self) -> Layout {
        Layout::ShaderReadOnlyOptimal
    }

    #[inline]
    fn descriptor_set_sampled_image_layout(&self) -> Layout {
        Layout::ShaderReadOnlyOptimal
    }

    #[inline]
    fn descriptor_set_input_attachment_layout(&self) -> Layout {
        Layout::ShaderReadOnlyOptimal
    }

    #[inline]
    fn identity_swizzle(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::AttachmentImage;
    use format::Format;

    #[test]
    fn create_regular() {
        let (device, _) = gfx_dev_and_queue!();
        let _img = AttachmentImage::new(&device, [32, 32], Format::R8G8B8A8Unorm).unwrap();
    }

    #[test]
    fn create_transient() {
        let (device, _) = gfx_dev_and_queue!();
        let _img = AttachmentImage::transient(&device, [32, 32], Format::R8G8B8A8Unorm).unwrap();
    }
}
