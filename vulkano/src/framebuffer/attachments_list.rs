// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use command_buffer::cmd::CommandsListSink;
use image::traits::ImageView;
//use sync::AccessFlagBits;
//use sync::PipelineStages;
use VulkanObject;
use vk;

/// A list of attachments.
// TODO: rework this trait
pub unsafe trait AttachmentsList {
    /// Returns the raw handles of the image views of this list.
    // TODO: better return type
    fn raw_image_view_handles(&self) -> Vec<vk::ImageView>;

    /// Returns the minimal dimensions of the views. Returns `None` if the list is empty.
    ///
    /// Must be done for each component individually.
    ///
    /// For example if one view is 256x256x1 and another one is 128x512x2, then this function
    /// should return 128x256x1.
    fn min_dimensions(&self) -> Option<[u32; 3]>;

    fn add_transition<'a>(&'a self, sink: &mut CommandsListSink<'a>);
}

unsafe impl AttachmentsList for () {
    #[inline]
    fn raw_image_view_handles(&self) -> Vec<vk::ImageView> {
        vec![]
    }

    #[inline]
    fn min_dimensions(&self) -> Option<[u32; 3]> {
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
            fn raw_image_view_handles(&self) -> Vec<vk::ImageView> {
                let &(ref $first, $(ref $rest,)*) = self;
                
                vec![
                    $first.inner().internal_object(),
                    $(
                        $rest.inner().internal_object(),
                    )*
                ]
            }

            #[inline]
            fn min_dimensions(&self) -> Option<[u32; 3]> {
                unimplemented!()
                /*let my_view_dims = self.first.parent().dimensions();
                debug_assert_eq!(my_view_dims.depth(), 1);
                let my_view_dims = [my_view_dims.width(), my_view_dims.height(),
                                    my_view_dims.array_layers()];       // FIXME: should be the view's layers, not the image's

                match self.rest.min_dimensions() {
                    Some(r_dims) => {
                        Some([
                            cmp::min(r_dims[0], my_view_dims[0]),
                            cmp::min(r_dims[1], my_view_dims[1]),
                            cmp::min(r_dims[2], my_view_dims[2])
                        ])
                    },
                    None => Some(my_view_dims),
                }*/
            }

            #[inline]
            fn add_transition<'a>(&'a self, sink: &mut CommandsListSink<'a>) {
                unimplemented!()
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
