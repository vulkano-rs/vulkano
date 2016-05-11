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
                    $(initial_layout: $init_layout:expr,)*
                    $(final_layout: $final_layout:expr,)*
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
        use $crate::framebuffer::RenderPassCreationError;
        use $crate::image::Layout;
        use $crate::image::traits::Image;
        use $crate::image::traits::ImageView;
        use $crate::sync::AccessFlagBits;
        use $crate::sync::PipelineStages;

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
                       -> Result<CustomRenderPass, RenderPassCreationError>
            {
                #![allow(unsafe_code)]

                let rp = try!(unsafe {
                    UnsafeRenderPass::new(device, AttachmentsIter(formats.clone(), 0),
                                          PassesIter(0), DependenciesIter(0, 0))
                });

                Ok(CustomRenderPass {
                    render_pass: rp,
                    formats: formats.clone(),
                })
            }

            #[inline]
            pub fn new(device: &Arc<Device>, formats: &Formats)
                       -> Result<Arc<CustomRenderPass>, RenderPassCreationError>
            {
                Ok(Arc::new(try!(CustomRenderPass::raw(device, formats))))
            }
        }

        unsafe impl RenderPass for CustomRenderPass {
            #[inline]
            fn render_pass(&self) -> &UnsafeRenderPass {
                &self.render_pass
            }
        }

        unsafe impl RenderPassDesc for CustomRenderPass {
            type AttachmentsIter = AttachmentsIter;
            type PassesIter = PassesIter;
            type DependenciesIter = DependenciesIter;

            #[inline]
            fn attachments(&self) -> Self::AttachmentsIter {
                AttachmentsIter(self.formats.clone(), 0)
            }

            #[inline]
            fn passes(&self) -> Self::PassesIter {
                PassesIter(0)
            }

            #[inline]
            fn dependencies(&self) -> Self::DependenciesIter {
                DependenciesIter(0, 0)
            }
        }

        #[derive(Debug, Clone)]
        pub struct AttachmentsIter(Formats, usize);
        impl ExactSizeIterator for AttachmentsIter {}
        impl Iterator for AttachmentsIter {
            type Item = LayoutAttachmentDescription;

            #[inline]
            fn next(&mut self) -> Option<LayoutAttachmentDescription> {
                #![allow(unused_assignments)]
                #![allow(unused_mut)]

                let mut num = 0;

                $({
                    if self.1 == num {
                        self.1 += 1;

                        let (initial_layout, final_layout) = attachment_layouts(num);

                        return Some($crate::framebuffer::LayoutAttachmentDescription {
                            format: $crate::format::FormatDesc::format(&(self.0).$atch_name.0),
                            samples: (self.0).$atch_name.1,
                            load: $crate::framebuffer::LoadOp::$load,
                            store: $crate::framebuffer::StoreOp::$store,
                            initial_layout: initial_layout,
                            final_layout: final_layout,
                        });
                    }

                    num += 1;
                })*

                None
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                #![allow(unused_assignments)]
                #![allow(unused_mut)]
                let mut num = 0;
                $(let $atch_name = num; num += 1;)*
                num -= self.1;
                (num, Some(num))
            }
        }

        #[derive(Debug, Clone)]
        pub struct PassesIter(usize);
        impl ExactSizeIterator for PassesIter {}
        impl Iterator for PassesIter {
            type Item = LayoutPassDescription;

            #[inline]
            fn next(&mut self) -> Option<LayoutPassDescription> {
                #![allow(unused_assignments)]
                #![allow(unused_mut)]

                let mut attachment_num = 0;
                $(
                    let $atch_name = attachment_num;
                    attachment_num += 1;
                )*

                let mut cur_pass_num = 0;

                $({
                    if self.0 == cur_pass_num {
                        self.0 += 1;

                        let mut depth = None;
                        $(
                            depth = Some(($depth_atch, Layout::DepthStencilAttachmentOptimal));
                        )*

                        return Some(LayoutPassDescription {
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
                        });
                    }

                    cur_pass_num += 1;
                })*

                None
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                #![allow(unused_assignments)]
                #![allow(unused_mut)]
                let mut num = 0;
                $($(let $color_atch = num;)* num += 1;)*
                num -= self.0;
                (num, Some(num))
            }
        }

        #[derive(Debug, Clone)]
        pub struct DependenciesIter(usize, usize);
        impl ExactSizeIterator for DependenciesIter {}
        impl Iterator for DependenciesIter {
            type Item = LayoutPassDependencyDescription;

            #[inline]
            fn next(&mut self) -> Option<LayoutPassDependencyDescription> {
                let num_passes = PassesIter(0).len();

                self.1 += 1;
                if self.1 >= num_passes {
                    self.0 += 1;
                    self.1 = self.0 + 1;
                }

                if self.0 >= num_passes || self.1 >= num_passes {
                    return None;
                }

                Some(LayoutPassDependencyDescription {
                    source_subpass: self.0,
                    destination_subpass: self.1,
                    src_stages: PipelineStages { all_graphics: true, .. PipelineStages::none() },         // TODO: correct values
                    dst_stages: PipelineStages { all_graphics: true, .. PipelineStages::none() },         // TODO: correct values
                    src_access: AccessFlagBits::all(),         // TODO: correct values
                    dst_access: AccessFlagBits::all(),         // TODO: correct values
                    by_region: true,            // TODO: correct values
                })
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let num_passes = PassesIter(0).len();

                let out = (num_passes - self.1 - 1) +
                          ((self.0 + 1) .. num_passes).map(|p| p * (num_passes - p - 1))
                                                      .fold(0, |a, b| a + b);
                (out, Some(out))
            }
        }

        /// Returns the initial and final layout of an attachment, given its num.
        ///
        /// The value always correspond to the first and last usages of an attachment.
        fn attachment_layouts(num: usize) -> (Layout, Layout) {
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

            $(if $atch_name == num {
                $(initial_layout = Some($init_layout);)*
                $(final_layout = Some($final_layout);)*
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
    };

    (__impl_clear_values__ [$num:expr] [$($s:tt)*] [$atch_name:ident $format:ty, Clear, $($rest:tt)*]) => {
        ordered_passes_renderpass!{__impl_clear_values__ [$num+1] [$($s)* $atch_name [$num] $format,] [$($rest)*] }
    };

    (__impl_clear_values__ [$num:expr] [$($s:tt)*] [$atch_name:ident $format:ty, $misc:ident, $($rest:tt)*]) => {
        ordered_passes_renderpass!{__impl_clear_values__ [$num+1] [$($s)*] [$($rest)*] }
    };

    (__impl_clear_values__ [$total:expr] [$($atch:ident [$num:expr] $format:ty,)+] []) => {
        #[allow(non_camel_case_types)]
        pub struct ClearValues<$($atch),+> {
            $(
                pub $atch: $atch,
            )+
        }

        #[allow(non_camel_case_types)]
        unsafe impl<$($atch: Clone + Into<<$format as $crate::format::FormatDesc>::ClearValue>),*>
            RenderPassClearValues<ClearValues<$($atch),*>> for CustomRenderPass
        {
            type ClearValuesIter = ClearValuesIter<$($atch),+>;

            #[inline]
            fn convert_clear_values(&self, val: ClearValues<$($atch),+>)
                                    -> ClearValuesIter<$($atch),+>
            {
                ClearValuesIter(self.formats.clone(), val, 0)
            }
        }

        #[allow(non_camel_case_types)]
        pub struct ClearValuesIter<$($atch),*>(Formats, ClearValues<$($atch),+>, usize);

        #[allow(non_camel_case_types)]
        impl<$($atch: Clone + Into<<$format as $crate::format::FormatDesc>::ClearValue>),+>
            Iterator for ClearValuesIter<$($atch),+>
        {
            type Item = $crate::format::ClearValue;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                use $crate::format::FormatDesc;

                $(
                    if self.2 == $num {
                        self.2 += 1;
                        return Some((self.0).$atch.0.decode_clear_value((self.1).$atch.clone().into()));
                    }
                )+

                if self.2 >= $total {
                    None
                } else {
                    Some(ClearValue::None)
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = $total - self.2;
                (len, Some(len))
            }
        }

        #[allow(non_camel_case_types)]
        impl<$($atch: Clone + Into<<$format as $crate::format::FormatDesc>::ClearValue>),+>
            ExactSizeIterator for ClearValuesIter<$($atch),+> {}
    };

    (__impl_clear_values__ [$total:expr] [] []) => {
        pub type ClearValues = ();

        unsafe impl RenderPassClearValues<()> for CustomRenderPass {
            type ClearValuesIter = ClearValuesIter;

            #[inline]
            fn convert_clear_values(&self, val: ()) -> ClearValuesIter {
                ClearValuesIter
            }
        }

        pub struct ClearValuesIter;

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
