// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::image::view::ImageViewAbstract;
use crate::SafeDeref;
use std::sync::Arc;
//use sync::AccessFlags;
//use sync::PipelineStages;

/// A list of attachments.
// TODO: rework this trait
pub unsafe trait AttachmentsList {
    fn num_attachments(&self) -> usize;

    fn as_image_view_access(&self, index: usize) -> Option<&dyn ImageViewAbstract>;
}

unsafe impl<T> AttachmentsList for T
where
    T: SafeDeref,
    T::Target: AttachmentsList,
{
    #[inline]
    fn num_attachments(&self) -> usize {
        (**self).num_attachments()
    }

    #[inline]
    fn as_image_view_access(&self, index: usize) -> Option<&dyn ImageViewAbstract> {
        (**self).as_image_view_access(index)
    }
}

unsafe impl AttachmentsList for () {
    #[inline]
    fn num_attachments(&self) -> usize {
        0
    }

    #[inline]
    fn as_image_view_access(&self, _: usize) -> Option<&dyn ImageViewAbstract> {
        None
    }
}

unsafe impl AttachmentsList for Vec<Arc<dyn ImageViewAbstract + Send + Sync>> {
    #[inline]
    fn num_attachments(&self) -> usize {
        self.len()
    }

    #[inline]
    fn as_image_view_access(&self, index: usize) -> Option<&dyn ImageViewAbstract> {
        self.get(index).map(|v| &**v as &_)
    }
}

unsafe impl<A, B> AttachmentsList for (A, B)
where
    A: AttachmentsList,
    B: ImageViewAbstract,
{
    #[inline]
    fn num_attachments(&self) -> usize {
        self.0.num_attachments() + 1
    }

    #[inline]
    fn as_image_view_access(&self, index: usize) -> Option<&dyn ImageViewAbstract> {
        if index == self.0.num_attachments() {
            Some(&self.1)
        } else {
            self.0.as_image_view_access(index)
        }
    }
}
