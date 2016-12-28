// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

/// Builds a `RenderPass` object whose template parameter is of undeterminate type.
#[macro_export]
macro_rules! single_pass_renderpass {
    (
        $device:expr,
        attachments: { $($a:tt)* },
        pass: {
            color: [$($color_atch:ident),*],
            depth_stencil: {$($depth_atch:ident)*}
        }
    ) => (
        ordered_passes_renderpass!(
            $device,
            attachments: { $($a)* },
            passes: [
                {
                    color: [$($color_atch),*],
                    depth_stencil: {$($depth_atch)*},
                    input: []
                }
            ]
        )
    )
}

/// Builds a `RenderPass` object whose template parameter is of undeterminate type.
#[macro_export]
macro_rules! ordered_passes_renderpass {
    (
        $device:expr,
        attachments: {
            $(
                $atch_name:ident: {
                    load: $load:ident,
                    store: $store:ident,
                    format: $format:expr,
                    samples: $samples:expr,
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
    ) => ({
        use std::sync::Arc;
        use $crate::device::Device;
        use $crate::format::ClearValue;
        use $crate::format::Format;
        use $crate::framebuffer::RenderPass;
        use $crate::framebuffer::RenderPassRef;
        use $crate::framebuffer::RenderPassDesc;
        use $crate::framebuffer::RenderPassClearValues;
        use $crate::framebuffer::RenderPassDescAttachmentsList;
        use $crate::framebuffer::LayoutAttachmentDescription;
        use $crate::framebuffer::LayoutPassDescription;
        use $crate::framebuffer::LayoutPassDependencyDescription;
        use $crate::framebuffer::FramebufferCreationError;
        use $crate::framebuffer::RenderPassCreationError;
        use $crate::framebuffer::IntoAttachmentsList;
        use $crate::image::Layout;
        use $crate::image::traits::ImageView;
        use $crate::sync::AccessFlagBits;
        use $crate::sync::PipelineStages;

        pub struct CustomRenderPassDesc {
            $(
                $atch_name: (Format, u32),
            )*
        }

        impl CustomRenderPassDesc {
            #[inline]
            pub fn start_attachments(&self) -> AttachmentsStart {
                AttachmentsStart
            }
        }

        pub struct AttachmentsStart;

        ordered_passes_renderpass!{[] __impl_attachments__ [] [] [$($atch_name),*] [A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z]}

        #[allow(unsafe_code)]
        unsafe impl RenderPassDesc for CustomRenderPassDesc {
            #[inline]
            fn num_attachments(&self) -> usize {
                num_attachments()
            }

            #[inline]
            fn attachment(&self, id: usize) -> Option<LayoutAttachmentDescription> {
                attachment(self, id)
            }

            #[inline]
            fn num_subpasses(&self) -> usize {
                num_subpasses()
            }

            #[inline]
            fn subpass(&self, id: usize) -> Option<LayoutPassDescription> {
                subpass(id)
            }

            #[inline]
            fn num_dependencies(&self) -> usize {
                num_dependencies()
            }

            #[inline]
            fn dependency(&self, id: usize) -> Option<LayoutPassDependencyDescription> {
                dependency(id)
            }
        }

        #[inline]
        fn num_attachments() -> usize {
            #![allow(unused_assignments)]
            #![allow(unused_mut)]
            #![allow(unused_variables)]
            let mut num = 0;
            $(let $atch_name = num; num += 1;)*
            num
        }

        #[inline]
        fn attachment(desc: &CustomRenderPassDesc, id: usize) -> Option<LayoutAttachmentDescription> {
            #![allow(unused_assignments)]
            #![allow(unused_mut)]

            let mut num = 0;

            $({
                if id == num {
                    let (initial_layout, final_layout) = attachment_layouts(num);

                    return Some($crate::framebuffer::LayoutAttachmentDescription {
                        format: desc.$atch_name.0,
                        samples: desc.$atch_name.1,
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
        fn num_subpasses() -> usize {
            #![allow(unused_assignments)]
            #![allow(unused_mut)]
            #![allow(unused_variables)]
            let mut num = 0;
            $($(let $color_atch = num;)* num += 1;)*
            num
        }

        #[inline]
        fn subpass(id: usize) -> Option<LayoutPassDescription> {
            #![allow(unused_assignments)]
            #![allow(unused_mut)]
            #![allow(unused_variables)]

            let mut attachment_num = 0;
            $(
                let $atch_name = attachment_num;
                attachment_num += 1;
            )*

            let mut cur_pass_num = 0;

            $({
                if id == cur_pass_num {
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
        fn num_dependencies() -> usize {
            num_subpasses().saturating_sub(1)
        }

        #[inline]
        fn dependency(id: usize) -> Option<LayoutPassDependencyDescription> {
            let num_passes = num_subpasses();

            if id + 1 >= num_passes {
                return None;
            }

            Some(LayoutPassDependencyDescription {
                source_subpass: id,
                destination_subpass: id + 1,
                src_stages: PipelineStages { all_graphics: true, .. PipelineStages::none() },         // TODO: correct values
                dst_stages: PipelineStages { all_graphics: true, .. PipelineStages::none() },         // TODO: correct values
                src_access: AccessFlagBits::all(),         // TODO: correct values
                dst_access: AccessFlagBits::all(),         // TODO: correct values
                by_region: true,            // TODO: correct values
            })
        }

        /// Returns the initial and final layout of an attachment, given its num.
        ///
        /// The value always correspond to the first and last usages of an attachment.
        fn attachment_layouts(num: usize) -> (Layout, Layout) {
            #![allow(unused_assignments)]
            #![allow(unused_mut)]
            #![allow(unused_variables)]

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

        /*#[allow(non_camel_case_types)]
        pub struct AList<$($atch_name),*> {
            $(
                pub $atch_name: $atch_name,
            )*
        }

        impl<$($atch_name: ImageView),*> IntoAttachmentsList for AList<$($atch_name),*> {
            type List = <($($atch_name,)*) as IntoAttachmentsList>::List;

            fn into_attachments_list(self) -> Self::List {
                IntoAttachmentsList::into_attachments_list(($(self.$atch_name,)*))
            }
        }

        #[allow(non_camel_case_types)]
        #[allow(unsafe_code)]
        unsafe impl<$($atch_name: ImageView),*> RenderPassDescAttachmentsList<AList<$($atch_name),*>> for CustomRenderPassDesc {
            #[inline]
            fn check_attachments_list(&self, l: &AList<$($atch_name),*>) -> Result<(), FramebufferCreationError> {
                #![allow(unused_assignments)]

                $({
                    if !l.$atch_name.identity_swizzle() {
                        return Err(FramebufferCreationError::AttachmentNotIdentitySwizzled);
                    }
                })*

                Ok(())
            }
        }*/

        // FIXME: restore
        //ordered_passes_renderpass!{[] __impl_clear_values__ [0] [] [$($atch_name $format, $load,)*] }

        CustomRenderPassDesc {
            $(
                $atch_name: ($format, $samples),
            )*
        }.build_render_pass($device)
    });

    ([] __impl_attachments__ [] [] [] [$($params:ident),*]) => {
        unsafe impl RenderPassDescAttachmentsList<AttachmentsStart> for CustomRenderPassDesc {
            fn check_attachments_list(&self, attachments: &AttachmentsStart) -> Result<(), FramebufferCreationError> {
                Ok(())        // FIXME:
            }
        }
    };

    ([] __impl_attachments__ [] [] [$next:ident $(, $rest:ident)*] [$first_param:ident, $($rest_params:ident),*]) => {
        pub struct $next<$first_param> {
            current: $first_param,
        }

        impl AttachmentsStart {
            pub fn $next<$first_param>(self, next: $first_param) -> $next<$first_param> {
                $next {
                    current: next,
                }
            }
        }

        ordered_passes_renderpass!{[] __impl_attachments__ [$next] [$first_param] [$($rest),*] [$($rest_params),*]}
    };

    ([] __impl_attachments__ [$prev:ident] [$($prev_params:ident),*] [] [$($params:ident),*]) => {
        unsafe impl<$($prev_params,)*> RenderPassDescAttachmentsList<$prev<$($prev_params,)*>> for CustomRenderPassDesc {
            fn check_attachments_list(&self, attachments: &$prev<$($prev_params,)*>) -> Result<(), FramebufferCreationError> {
                Ok(())        // FIXME:
            }
        }
    };

    ([] __impl_attachments__ [$prev:ident] [$($prev_params:ident),*] [$next:ident $(, $rest:ident)*] [$first_param:ident, $($rest_params:ident),*]) => {
        pub struct $next<$($prev_params,)* $first_param> {
            prev: $prev,
            current: $first_param,
        }

        impl<$($prev_params,)*> $prev<$($prev_params,)*> {
            pub fn $next<$first_param>(self, next: $first_param) -> $next<$($prev_params,)* $first_param> {
                $next {
                    prev: self,
                    current: next,
                }
            }
        }

        ordered_passes_renderpass!{[] __impl_attachments__ [$next] [$($prev_params,)* $first_param] [$($rest),*] [$($rest_params),*]}
    };

    ([] __impl_clear_values__ [$num:expr] [$($s:tt)*] [$atch_name:ident $format:ty, Clear, $($rest:tt)*]) => {
        ordered_passes_renderpass!{[] __impl_clear_values__ [$num+1] [$($s)* $atch_name [$num] $format,] [$($rest)*] }
    };

    ([] __impl_clear_values__ [$num:expr] [$($s:tt)*] [$atch_name:ident $format:ty, $misc:ident, $($rest:tt)*]) => {
        ordered_passes_renderpass!{[] __impl_clear_values__ [$num+1] [$($s)*] [$($rest)*] }
    };

    ([] __impl_clear_values__ [$total:expr] [$($atch:ident [$num:expr] $format:ty,)+] []) => {
        #[allow(non_camel_case_types)]
        pub struct ClearValues<$($atch),+> {
            $(
                pub $atch: $atch,
            )+
        }

        #[allow(non_camel_case_types)]
        #[allow(unsafe_code)]
        unsafe impl<$($atch: Clone + Into<<$format as $crate::format::FormatDesc>::ClearValue>),*>
            RenderPassClearValues<ClearValues<$($atch),*>> for CustomRenderPassDesc
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

    ([] __impl_clear_values__ [$total:expr] [] []) => {
        pub type ClearValues = ();

        #[allow(unsafe_code)]
        unsafe impl RenderPassClearValues<()> for CustomRenderPassDesc {
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
