// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Image whose purpose is to be used as a framebuffer attachment.
//! 
//! This module declares the `AttachmentImage` type. It is a safe wrapper around `UnsafeImage`
//! and implements all the relevant image traits.
//! 
//! The image is always two-dimensional and has only one mipmap, but it can have any kind of
//! format. Trying to use a format that the backend doesn't support for rendering will result in
//! an error being returned when creating the image. Once you have an `AttachmentImage`, you are
//! guaranteed that you will be able to draw on it.
//! 
//! The template parameter of `AttachmentImage` is a type that describes the format of the image.
//! 
//! # Regular vs transient
//! 
//! Calling `AttachmentImage::new` will create a regular image, while calling
//! `AttachmentImage::transient` will create a *transient* image.
//! 
//! A transient image is a special kind of image whose content is undefined outside of render
//! passes. Once you finish drawing, you can't read from it anymore.
//! 
//! This gives a hint to the Vulkan implementation that it is possible for the image's content to
//! live exclusively in some cache memory, and that no real memory has to be allocated for it.
//! 
//! In other words, if you are going to read from the image after drawing to it, use a regular
//! image. If you don't need to read from it (for example if it's some kind of intermediary color,
//! or a depth buffer that is only used once) then use a transient image.
//!
use std::mem;
use std::iter::Empty;
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Weak;

use command_buffer::Submission;
use device::Device;
use format::ClearValue;
use format::FormatDesc;
use format::FormatTy;
use image::sys::Dimensions;
use image::sys::ImageCreationError;
use image::sys::Layout;
use image::sys::UnsafeImage;
use image::sys::UnsafeImageView;
use image::sys::Usage;
use image::traits::AccessRange;
use image::traits::GpuAccessResult;
use image::traits::Image;
use image::traits::ImageClearValue;
use image::traits::ImageContent;
use image::traits::ImageView;
use image::traits::Transition;
use memory::DeviceMemory;
use sync::Sharing;

/// Image whose purpose is to be used as a framebuffer attachment.
#[derive(Debug)]
pub struct AttachmentImage<F> {
    // Inner implementation.
    image: UnsafeImage,

    // We maintain a view of the whole image since we will need it when rendering.
    view: UnsafeImageView,

    // Memory used to back the image.
    memory: DeviceMemory,

    // Format.
    format: F,

    // Layout to use when the image is used as a framebuffer attachment.
    // Should be either "depth-stencil optimal" or "color optimal".
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
        let usage = Usage {
            transfer_source: true,
            sampled: true,
            .. Usage::none()
        };

        AttachmentImage::new_impl(device, dimensions, format, usage)
    }

    /// Same as `new`, except that the image will be transient.
    ///
    /// A transient image is special because its content is undefined outside of a render pass.
    /// This means that the implementation has the possibility to not allocate any memory for it.
    pub fn transient(device: &Arc<Device>, dimensions: [u32; 2], format: F)
                     -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
        where F: FormatDesc
    {
        let usage = Usage {
            transient_attachment: true,
            .. Usage::none()
        };

        AttachmentImage::new_impl(device, dimensions, format, usage)
    }

    fn new_impl(device: &Arc<Device>, dimensions: [u32; 2], format: F, usage: Usage)
                -> Result<Arc<AttachmentImage<F>>, ImageCreationError>
        where F: FormatDesc
    {
        let is_depth = match format.format().ty() {
            FormatTy::Depth => true,
            FormatTy::DepthStencil => true,
            FormatTy::Compressed => panic!(),
            _ => false
        };

        let usage = Usage {
            color_attachment: !is_depth,
            depth_stencil_attachment: is_depth,
            input_attachment: true,
            .. usage
        };

        let (image, mem_reqs) = unsafe {
            try!(UnsafeImage::new(device, &usage, format.format(),
                                  Dimensions::Dim2d { width: dimensions[0], height: dimensions[1] },
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

        // note: alignment doesn't need to be checked because allocating memory is guaranteed to
        //       fulfill any alignment requirement

        let mem = try!(DeviceMemory::alloc(device, &mem_ty, mem_reqs.size));
        unsafe { try!(image.bind_memory(&mem, 0)); }

        let view = unsafe {
            try!(UnsafeImageView::new(&image, 0 .. 1, 0 .. 1))
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

    /// Returns the dimensions of the image.
    #[inline]
    pub fn dimensions(&self) -> [u32; 2] {
        let dims = self.image.dimensions();
        [dims.width(), dims.height()]
    }
}

unsafe impl<F> Image for AttachmentImage<F> where F: 'static + Send + Sync {
    #[inline]
    fn inner_image(&self) -> &UnsafeImage {
        &self.image
    }

    #[inline]
    fn blocks(&self, _: Range<u32>, _: Range<u32>) -> Vec<(u32, u32)> {
        vec![(0, 0)]
    }

    #[inline]
    fn block_mipmap_levels_range(&self, block: (u32, u32)) -> Range<u32> {
        0 .. 1
    }

    #[inline]
    fn block_array_layers_range(&self, block: (u32, u32)) -> Range<u32> {
        0 .. 1
    }

    #[inline]
    fn initial_layout(&self, _: (u32, u32), _: Layout) -> (Layout, bool, bool) {
        (self.attachment_layout, false, false)
    }

    #[inline]
    fn final_layout(&self, _: (u32, u32), _: Layout) -> (Layout, bool, bool) {
        (self.attachment_layout, false, false)
    }

    fn needs_fence(&self, access: &mut Iterator<Item = AccessRange>) -> Option<bool> {
        Some(false)
    }

    unsafe fn gpu_access(&self, _: &mut Iterator<Item = AccessRange>,
                         submission: &Arc<Submission>) -> GpuAccessResult
    {
        let mut guarded = self.guarded.lock().unwrap();

        let dependency = mem::replace(&mut guarded.latest_submission, Some(Arc::downgrade(submission)));
        let dependency = dependency.and_then(|d| d.upgrade());

        let transition = if !guarded.correct_layout {
            vec![Transition {
                block: (0, 0),
                from: Layout::Undefined,
                to: self.attachment_layout,
            }]
        } else {
            vec![]
        };

        guarded.correct_layout = true;

        GpuAccessResult {
            dependencies: if let Some(dependency) = dependency {
                vec![dependency]
            } else {
                vec![]
            },
            additional_wait_semaphore: None,
            additional_signal_semaphore: None,
            before_transitions: transition,
            after_transitions: vec![],
        }
    }
}

unsafe impl<F> ImageClearValue<F::ClearValue> for AttachmentImage<F>
    where F: FormatDesc + 'static + Send + Sync
{
    #[inline]
    fn decode(&self, value: F::ClearValue) -> Option<ClearValue> {
        Some(self.format.decode_clear_value(value))
    }
}

unsafe impl<P, F> ImageContent<P> for AttachmentImage<F> where F: 'static + Send + Sync {
    #[inline]
    fn matches_format(&self) -> bool {
        true        // FIXME:
    }
}

unsafe impl<F: 'static> ImageView for AttachmentImage<F> where F: 'static + Send + Sync {
    #[inline]
    fn parent(&self) -> &Image {
        self
    }

    #[inline]
    fn parent_arc(me: &Arc<Self>) -> Arc<Image> where Self: Sized {
        me.clone() as Arc<_>
    }

    #[inline]
    fn blocks(&self) -> Vec<(u32, u32)> {
        vec![(0, 0)]
    }

    #[inline]
    fn inner_view(&self) -> &UnsafeImageView {
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
