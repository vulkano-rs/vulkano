// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::mem;
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;

use command_buffer::Submission;
use format::Format;
use image::traits::AccessRange;
use image::traits::GpuAccessResult;
use image::traits::Image;
use image::traits::ImageContent;
use image::traits::ImageView;
use image::traits::Transition;
use image::sys::Dimensions;
use image::sys::Layout;
use image::sys::UnsafeImage;
use image::sys::UnsafeImageView;
use swapchain::Swapchain;

use OomError;

pub struct SwapchainImage {
    image: UnsafeImage,
    view: UnsafeImageView,
    format: Format,
    swapchain: Arc<Swapchain>,
    id: u32,
    guarded: Mutex<Guarded>,
}

struct Guarded {
    present_layout: bool,
    latest_submission: Option<Arc<Submission>>,
}

impl SwapchainImage {
    pub unsafe fn from_raw(image: UnsafeImage, format: Format, swapchain: &Arc<Swapchain>, id: u32)
                           -> Result<Arc<SwapchainImage>, OomError>
    {
        let view = try!(UnsafeImageView::new(&image));

        Ok(Arc::new(SwapchainImage {
            image: image,
            view: view,
            format: format,
            swapchain: swapchain.clone(),
            id: id,
            guarded: Mutex::new(Guarded {
                present_layout: false,
                latest_submission: None,
            }),
        }))
    }

    #[inline]
    pub fn dimensions(&self) -> Dimensions {
        self.image.dimensions()
    }
}

unsafe impl Image for SwapchainImage {
    #[inline]
    fn inner_image(&self) -> &UnsafeImage {
        &self.image
    }

    #[inline]
    fn blocks(&self, _: Range<u32>, _: Range<u32>) -> Vec<(u32, u32)> {
        vec![(0, 0)]
    }

    #[inline]
    fn initial_layout(&self, _: (u32, u32), _: Layout) -> (Layout, bool, bool) {
        (Layout::PresentSrc, false, true)
    }

    #[inline]
    fn final_layout(&self, _: (u32, u32), _: Layout) -> (Layout, bool, bool) {
        (Layout::PresentSrc, false, true)
    }

    fn needs_fence(&self, access: &mut Iterator<Item = AccessRange>) -> Option<bool> {
        Some(false)
    }

    unsafe fn gpu_access(&self, access: &mut Iterator<Item = AccessRange>,
                         submission: &Arc<Submission>) -> GpuAccessResult
    {
        let mut guarded = self.guarded.lock().unwrap();

        let dependency = mem::replace(&mut guarded.latest_submission, Some(submission.clone()));
        let semaphore = self.swapchain.image_semaphore(self.id).expect("Try to render to a swapchain image that was not acquired first");

        if guarded.present_layout {
            return GpuAccessResult {
                dependencies: if let Some(dependency) = dependency {
                    vec![dependency]
                } else {
                    vec![]
                },
                additional_wait_semaphore: Some(semaphore.clone()),
                additional_signal_semaphore: Some(semaphore),
                before_transitions: vec![],
                after_transitions: vec![],
            };
        }

        guarded.present_layout = true;

        GpuAccessResult {
            dependencies: if let Some(dependency) = dependency {
                vec![dependency]
            } else {
                vec![]
            },
            additional_wait_semaphore: Some(semaphore.clone()),
            additional_signal_semaphore: Some(semaphore),
            before_transitions: vec![Transition {
                block: (0, 0),
                from: Layout::Undefined,
                to: Layout::PresentSrc,
            }],
            after_transitions: vec![],
        }
    }
}

unsafe impl<P> ImageContent<P> for SwapchainImage {
    #[inline]
    fn matches_format(&self) -> bool {
        true        // FIXME:
    }
}

unsafe impl ImageView for SwapchainImage {
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
