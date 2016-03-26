// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::mem;
use std::iter::Empty;
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;

use command_buffer::Submission;
use device::Device;
use format::FormatDesc;
use format::FormatTy;
use image::sys::Dimensions;
use image::sys::Layout;
use image::sys::UnsafeImage;
use image::sys::UnsafeImageView;
use image::sys::Usage;
use image::traits::AccessRange;
use image::traits::GpuAccessResult;
use image::traits::Image;
use image::traits::ImageContent;
use image::traits::ImageView;
use image::traits::Transition;
use memory::DeviceMemory;
use sync::Sharing;

use OomError;

pub struct AttachmentImage<F> {
    image: UnsafeImage,
    view: UnsafeImageView,
    memory: DeviceMemory,
    format: F,

    // Layout to use when the image is used as a framebuffer attachment.
    // Should be either "depth-stencil optimal" or "color optimal".
    attachment_layout: Layout,

    guarded: Mutex<Guarded>,
}

struct Guarded {
    correct_layout: bool,
    latest_submission: Option<Arc<Submission>>,
}

impl<F> AttachmentImage<F> {
    pub fn new(device: &Arc<Device>, dimensions: [u32; 2], format: F)
               -> Result<Arc<AttachmentImage<F>>, OomError>
        where F: FormatDesc
    {
        let is_depth = match format.format().ty() {
            FormatTy::Depth => true,
            FormatTy::DepthStencil => true,
            FormatTy::Compressed => panic!(),
            _ => false
        };

        let usage = Usage {
            transfer_source: true,
            sampled: true,
            color_attachment: !is_depth,
            depth_stencil_attachment: is_depth,
            input_attachment: true,
            .. Usage::none()
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
        unsafe { try!(image.bind_memory(&mem, 0 .. mem_reqs.size)); }

        let view = unsafe {
            try!(UnsafeImageView::new(&image))
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

    #[inline]
    pub fn dimensions(&self) -> Dimensions {
        self.image.dimensions()
    }
}

unsafe impl<F> Image for AttachmentImage<F> {
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

        let dependency = mem::replace(&mut guarded.latest_submission, Some(submission.clone()));

        let transition = if guarded.correct_layout {
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

unsafe impl<P, F> ImageContent<P> for AttachmentImage<F> {
    #[inline]
    fn matches_format(&self) -> bool {
        true        // FIXME:
    }
}

unsafe impl<F: 'static> ImageView for AttachmentImage<F> {
    #[inline]
    fn parent(&self) -> &Image {
        self
    }

    #[inline]
    fn parent_arc(me: &Arc<Self>) -> Arc<Image> where Self: Sized {
        me.clone() as Arc<_>
    }

    #[inline]
    fn inner_view(&self) -> &UnsafeImageView {
        &self.view
    }

    #[inline]
    fn descriptor_set_storage_image_layout(&self, _: AccessRange) -> Layout {
        Layout::ColorAttachmentOptimal
    }

    #[inline]
    fn descriptor_set_combined_image_sampler_layout(&self, _: AccessRange) -> Layout {
        Layout::ColorAttachmentOptimal
    }

    #[inline]
    fn descriptor_set_sampled_image_layout(&self, _: AccessRange) -> Layout {
        Layout::ColorAttachmentOptimal
    }

    #[inline]
    fn descriptor_set_input_attachment_layout(&self, _: AccessRange) -> Layout {
        Layout::ColorAttachmentOptimal
    }

    #[inline]
    fn identity_swizzle(&self) -> bool {
        true
    }
}
