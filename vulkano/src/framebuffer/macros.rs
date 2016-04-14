// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

/// Builds a `CustomRenderPass` object that provides a safe wrapper around `UnsafeRenderPass`.
#[macro_export]
macro_rules! single_pass_renderpass {
    (
        attachments: { $($a:tt)* },
        pass: {
            color: [$($color_atch:ident),*],
            depth_stencil: {$($depth_atch:ident)*}
        }
    ) => {
        ordered_passes_renderpass!{
            attachments: { $($a)* },
            passes: [
                {
                    color: [$($color_atch),*],
                    depth_stencil: {$($depth_atch)*},
                    input: []
                }
            ]
        }
    }
}

/// Builds a `CustomRenderPass` object that provides a safe wrapper around `UnsafeRenderPass`.
#[macro_export]
macro_rules! ordered_passes_renderpass {
    (
        attachments: {
            $(
                $atch_name:ident: {
                    load: $load:ident,
                    store: $store:ident,
                    format: $format:ty,
                }
            ),*
        },
        passes: [
            $(
                {
                    color: [$($color_atch:ident),*],
                    depth_stencil: {$($depth_atch:ident)*},
                    input: [$($input_atch:ident),*]
                }
            ),*
        ]
    ) => {
        use std::vec::IntoIter as VecIntoIter;
        use std::sync::Arc;
        use $crate::OomError;
        use $crate::device::Device;
        use $crate::format::ClearValue;
        use $crate::framebuffer::UnsafeRenderPass;
        use $crate::framebuffer::RenderPass;
        use $crate::framebuffer::RenderPassDesc;
        use $crate::framebuffer::RenderPassClearValues;
        use $crate::framebuffer::RenderPassAttachmentsList;
        use $crate::framebuffer::LayoutAttachmentDescription;
        use $crate::framebuffer::LayoutPassDescription;
        use $crate::framebuffer::LayoutPassDependencyDescription;
        use $crate::framebuffer::FramebufferCreationError;
        use $crate::image::Layout;
        use $crate::image::traits::Image;
        use $crate::image::traits::ImageView;

        #[derive(Debug, Clone)]
        pub struct Formats {
            $(
                pub $atch_name: ($format, u32),
            )*
        }

        pub struct CustomRenderPass {
            render_pass: UnsafeRenderPass,
            formats: Formats,
        }

        impl CustomRenderPass {
            pub fn raw(device: &Arc<Device>, formats: &Formats)
                       -> Result<CustomRenderPass, OomError>
            {
                #![allow(unsafe_code)]

                let rp = try!(unsafe {
                    UnsafeRenderPass::raw(device, attachments(formats), passes(), dependencies())
                });

                Ok(CustomRenderPass {
                    render_pass: rp,
                    formats: formats.clone(),
                })
            }
            
            #[inline]
            pub fn new(device: &Arc<Device>, formats: &Formats)
                       -> Arc<CustomRenderPass>
            {
                Arc::new(CustomRenderPass::raw(device, formats).unwrap())
            }
        }

        unsafe impl RenderPass for CustomRenderPass {
            #[inline]
            fn render_pass(&self) -> &UnsafeRenderPass {
                &self.render_pass
            }
        }

        unsafe impl RenderPassDesc for CustomRenderPass {
            type AttachmentsIter = VecIntoIter<LayoutAttachmentDescription>;
            type PassesIter = VecIntoIter<LayoutPassDescription>;
            type DependenciesIter = VecIntoIter<LayoutPassDependencyDescription>;

            #[inline]
            fn attachments(&self) -> Self::AttachmentsIter {
                attachments(&self.formats)
            }

            #[inline]
            fn passes(&self) -> Self::PassesIter {
                passes()
            }

            #[inline]
            fn dependencies(&self) -> Self::DependenciesIter {
                dependencies()
            }
        }

        /// Returns an iterator to the list of attachments of this render pass.
        #[inline]
        fn attachments(formats: &Formats) -> VecIntoIter<LayoutAttachmentDescription> {
            #![allow(unused_assignments)]
            #![allow(unused_mut)]

            let mut num = 0;
            vec![$({
                let (initial_layout, final_layout) = attachment_layouts(num);
                num += 1;

                $crate::framebuffer::LayoutAttachmentDescription {
                    format: $crate::format::FormatDesc::format(&formats.$atch_name.0),
                    samples: formats.$atch_name.1,
                    load: $crate::framebuffer::LoadOp::$load,
                    store: $crate::framebuffer::StoreOp::$store,
                    initial_layout: initial_layout,
                    final_layout: final_layout,
                }
            }),*].into_iter()
        }

        /// Returns an iterator to the list of passes of this render pass.
        #[inline]
        fn passes() -> VecIntoIter<LayoutPassDescription> {
            #![allow(unused_assignments)]
            #![allow(unused_mut)]

            let mut attachment_num = 0;
            $(
                let $atch_name = attachment_num;
                attachment_num += 1;
            )*

            vec![
                $({
                    let mut depth = None;
                    $(
                        depth = Some(($depth_atch, Layout::DepthStencilAttachmentOptimal));
                    )*

                    $crate::framebuffer::LayoutPassDescription {
                        color_attachments: vec![
                            $(
                                ($color_atch, Layout::ColorAttachmentOptimal)
                            ),*
                        ],
                        depth_stencil: depth,
                        input_attachments: vec![
                            $(
                                ($input_atch, Layout::ShaderReadOnlyOptimal)
                            ),*
                        ],
                        resolve_attachments: vec![],
                        preserve_attachments: (0 .. attachment_num).filter(|&a| {
                            $(if a == $color_atch { return false; })*
                            $(if a == $depth_atch { return false; })*
                            $(if a == $input_atch { return false; })*
                            true
                        }).collect()
                    }
                }),*
            ].into_iter()
        }

        /// Returns an iterator to the list of pass dependencies of this render pass.
        #[inline]
        fn dependencies() -> VecIntoIter<LayoutPassDependencyDescription> {
            #![allow(unused_assignments)]
            #![allow(unused_mut)]

            (1 .. passes().len()).flat_map(|p2| {
                (0 .. p2.clone()).map(move |p1| {
                    LayoutPassDependencyDescription {
                        source_subpass: p1,
                        destination_subpass: p2,
                        by_region: false,
                    }
                })
            }).collect::<Vec<_>>().into_iter()
        }

        /// Returns the initial and final layout of an attachment, given its num.
        fn attachment_layouts(num: u32) -> (Layout, Layout) {
            #![allow(unused_assignments)]
            #![allow(unused_mut)]

            let mut attachment_num = 0;
            $(
                let $atch_name = attachment_num;
                attachment_num += 1;
            )*

            let mut initial_layout = None;
            let mut final_layout = None;

            $({
                $(
                    if $depth_atch == num {
                        if initial_layout.is_none() {
                            initial_layout = Some(Layout::DepthStencilAttachmentOptimal);
                        }
                        final_layout = Some(Layout::DepthStencilAttachmentOptimal);
                    }
                )*

                $(
                    if $color_atch == num {
                        if initial_layout.is_none() {
                            initial_layout = Some(Layout::ColorAttachmentOptimal);
                        }
                        final_layout = Some(Layout::ColorAttachmentOptimal);
                    }
                )*

                $(
                    if $input_atch == num {
                        if initial_layout.is_none() {
                            initial_layout = Some(Layout::ShaderReadOnlyOptimal);
                        }
                        final_layout = Some(Layout::ShaderReadOnlyOptimal);
                    }
                )*
            })*

            (initial_layout.unwrap(), final_layout.unwrap())
        }

        #[allow(non_camel_case_types)]
        pub struct AList<'a, $($atch_name: 'a),*> {
            $(
                pub $atch_name: &'a Arc<$atch_name>,
            )*
        }

        #[allow(non_camel_case_types)]
        unsafe impl<'a, $($atch_name: 'static + ImageView),*> RenderPassAttachmentsList<AList<'a, $($atch_name),*>> for CustomRenderPass {
            // TODO: shouldn't build a Vec
            type AttachmentsIter = VecIntoIter<(Arc<ImageView>, Arc<Image>, Layout, Layout)>;

            #[inline]
            fn convert_attachments_list(&self, l: AList<'a, $($atch_name),*>) -> Result<Self::AttachmentsIter, FramebufferCreationError> {
                #![allow(unused_assignments)]

                let mut result = Vec::new();

                let mut num = 0;
                $({
                    if !l.$atch_name.identity_swizzle() {
                        return Err(FramebufferCreationError::AttachmentNotIdentitySwizzled);
                    }

                    // FIXME: lots of checks missing (format, samples, layout, etc.)

                    let (initial_layout, final_layout) = attachment_layouts(num);
                    num += 1;
                    result.push((l.$atch_name.clone() as Arc<_>, ImageView::parent_arc(&l.$atch_name), initial_layout, final_layout));
                })*

                Ok(result.into_iter())
            }
        }

        ordered_passes_renderpass!{__impl_clear_values__ [0] [] [$($atch_name $format, $load,)*] }

        unsafe impl RenderPassClearValues<ClearValues> for CustomRenderPass {
            type ClearValuesIter = ClearValuesIter;

            #[inline]
            fn convert_clear_values(&self, val: ClearValues) -> ClearValuesIter {
                ClearValuesIter(val, 0)
            }
        }
    };

    (__impl_clear_values__ [$num:expr] [$($s:tt)*] [$atch_name:ident $format:ty, Clear, $($rest:tt)*]) => {
        ordered_passes_renderpass!{__impl_clear_values__ [$num+1] [$($s)* $atch_name [$num] $format,] [$($rest)*] }
    };

    (__impl_clear_values__ [$num:expr] [$($s:tt)*] [$atch_name:ident $format:ty, $misc:ident, $($rest:tt)*]) => {
        ordered_passes_renderpass!{__impl_clear_values__ [$num+1] [$($s)*] [$($rest)*] }
    };

    (__impl_clear_values__ [$total:expr] [$($atch:ident [$num:expr] $format:ty,)+] []) => {
        pub struct ClearValues {
            $(
                pub $atch: <$format as $crate::format::FormatDesc>::ClearValue,
            )+
        }

        pub struct ClearValuesIter(ClearValues, usize);

        impl Iterator for ClearValuesIter {
            type Item = $crate::format::ClearValue;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                $(
                    if self.1 == $num {
                        self.1 += 1;
                        return Some(ClearValue::from((self.0).$atch));        // FIXME: should use Format::decode_clear_value instead
                    }
                )+

                if self.1 >= $total {
                    None
                } else {
                    Some(ClearValue::None)
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = $total - self.1;
                (len, Some(len))
            }
        }

        impl ExactSizeIterator for ClearValuesIter {}
    };

    (__impl_clear_values__ [$total:expr] [] []) => {
        pub type ClearValues = ();

        pub struct ClearValuesIter((), usize);

        impl Iterator for ClearValuesIter {
            type Item = $crate::format::ClearValue;
            #[inline]
            fn next(&mut self) -> Option<Self::Item> { None }
            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) { (0, Some(0)) }
        }

        impl ExactSizeIterator for ClearValuesIter {}
    };
}
