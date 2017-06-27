// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use SafeDeref;
use image::ImageViewAccess;
use std::sync::Arc;
//use sync::AccessFlagBits;
//use sync::PipelineStages;

/// A list of attachments.
// TODO: rework this trait
pub unsafe trait AttachmentsList {
    // TODO: meh for API
    fn as_image_view_accesses(&self) -> Vec<&ImageViewAccess>;
}

unsafe impl<T> AttachmentsList for T
    where T: SafeDeref,
          T::Target: AttachmentsList
{
    #[inline]
    fn as_image_view_accesses(&self) -> Vec<&ImageViewAccess> {
        (**self).as_image_view_accesses()
    }
}

unsafe impl AttachmentsList for () {
    #[inline]
    fn as_image_view_accesses(&self) -> Vec<&ImageViewAccess> {
        vec![]
    }
}

unsafe impl AttachmentsList for Vec<Arc<ImageViewAccess + Send + Sync>> {
    #[inline]
    fn as_image_view_accesses(&self) -> Vec<&ImageViewAccess> {
        self.iter().map(|p| &**p as &ImageViewAccess).collect()
    }
}

unsafe impl<A, B> AttachmentsList for (A, B)
    where A: AttachmentsList,
          B: ImageViewAccess
{
    #[inline]
    fn as_image_view_accesses(&self) -> Vec<&ImageViewAccess> {
        let mut list = self.0.as_image_view_accesses();
        list.push(&self.1);
        list
    }
}
