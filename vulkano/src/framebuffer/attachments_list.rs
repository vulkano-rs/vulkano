// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::cmp;
use command_buffer::cmd::CommandsListSink;
use image::ImageView;
use image::sys::UnsafeImageView;
//use sync::AccessFlagBits;
//use sync::PipelineStages;

/// A list of attachments.
// TODO: rework this trait
pub unsafe trait AttachmentsList {
    /// Returns the image views of this list.
    // TODO: better return type
    fn raw_image_view_handles(&self) -> Vec<&UnsafeImageView>;

    /// Returns the dimensions of the intersection of the views. Returns `None` if the list is
    /// empty.
    ///
    /// For example if one view is 256x256x2 and another one is 128x512x3, then this function
    /// should return 128x256x2.
    fn intersection_dimensions(&self) -> Option<[u32; 3]>;

    fn add_transition<'a>(&'a self, sink: &mut CommandsListSink<'a>);
}

unsafe impl AttachmentsList for () {
    #[inline]
    fn raw_image_view_handles(&self) -> Vec<&UnsafeImageView> {
        vec![]
    }

    #[inline]
    fn intersection_dimensions(&self) -> Option<[u32; 3]> {
        None
    }

    #[inline]
    fn add_transition<'a>(&'a self, sink: &mut CommandsListSink<'a>) {
    }
}

macro_rules! impl_into_atch_list {
    ($first:ident $(, $rest:ident)*) => (
        unsafe impl<$first $(, $rest)*> AttachmentsList for ($first, $($rest),*)
            where $first: ImageView,
                  $($rest: ImageView,)*
        {
            #[inline]
            #[allow(non_snake_case)]
            fn raw_image_view_handles(&self) -> Vec<&UnsafeImageView> {
                let &(ref $first, $(ref $rest,)*) = self;
                
                vec![
                    &$first.inner(),
                    $(
                        &$rest.inner(),
                    )*
                ]
            }

            #[inline]
            #[allow(non_snake_case)]
            fn intersection_dimensions(&self) -> Option<[u32; 3]> {
                let &(ref $first, $(ref $rest,)*) = self;

                let dims = {
                    let d = $first.dimensions();
                    debug_assert_eq!(d.depth(), 1);
                    [d.width(), d.height(), d.array_layers()]
                };

                $(
                    let dims = {
                        let d = $rest.dimensions();
                        debug_assert_eq!(d.depth(), 1);
                        [
                            cmp::min(d.width(), dims[0]),
                            cmp::min(d.height(), dims[1]),
                            cmp::min(d.array_layers(), dims[2])
                        ]
                    };
                )*

                Some(dims)
            }

            #[inline]
            fn add_transition<'a>(&'a self, sink: &mut CommandsListSink<'a>) {
                // FIXME: implement
                /*// TODO: "wrong" values
                let stages = PipelineStages {
                    color_attachment_output: true,
                    late_fragment_tests: true,
                    .. PipelineStages::none()
                };
                
                let access = AccessFlagBits {
                    color_attachment_read: true,
                    color_attachment_write: true,
                    depth_stencil_attachment_read: true,
                    depth_stencil_attachment_write: true,
                    .. AccessFlagBits::none()
                };

                // FIXME: adjust layers & mipmaps with the view's parameters
                sink.add_image_transition(self.first.parent(), 0, 1, 0, 1, true, Layout::General /* FIXME: wrong */,
                                        stages, access);
                self.rest.add_transition(sink);*/
            }
        }

        impl_into_atch_list!($($rest),*);
    );
    
    () => ();
}

impl_into_atch_list!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z);
