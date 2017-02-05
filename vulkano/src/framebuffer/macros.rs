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
        use $crate::framebuffer::RenderPassDesc;

        mod scope {
            #![allow(non_camel_case_types)]

            use $crate::format::ClearValue;
            use $crate::format::Format;
            use $crate::framebuffer::RenderPassDesc;
            use $crate::framebuffer::RenderPassDescClearValues;
            use $crate::framebuffer::LayoutAttachmentDescription;
            use $crate::framebuffer::LayoutPassDescription;
            use $crate::framebuffer::LayoutPassDependencyDescription;
            use $crate::image::Layout;
            use $crate::sync::AccessFlagBits;
            use $crate::sync::PipelineStages;

            pub struct CustomRenderPassDesc {
                $(
                    pub $atch_name: (Format, u32),
                )*
            }

            impl CustomRenderPassDesc {
                #[inline]
                pub fn start_attachments(&self) -> atch::AttachmentsStart {
                    atch::AttachmentsStart
                }

                #[inline]
                pub fn start_clear_values(&self) -> cv::ClearValuesStart {
                    cv::ClearValuesStart
                }
            }

            pub mod atch {
                use $crate::framebuffer::FramebufferCreationError;
                use $crate::framebuffer::RenderPassDescAttachmentsList;
                use $crate::image::traits::ImageView;
                use super::CustomRenderPassDesc;
                pub struct AttachmentsStart;
                ordered_passes_renderpass!{[] __impl_attachments__ [] [] [$($atch_name),*] [A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z]}
            }
    
            pub mod cv {
                use std::iter;
                use $crate::format::ClearValue;
                use $crate::framebuffer::RenderPassDescClearValues;
                use super::CustomRenderPassDesc;
                pub struct ClearValuesStart;
                ordered_passes_renderpass!{[] __impl_clear_values__ [] [] [$($atch_name: $load),*] [A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z]}
            }

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

            unsafe impl RenderPassDescClearValues<Vec<ClearValue>> for CustomRenderPassDesc {
                fn convert_clear_values(&self, values: Vec<ClearValue>) -> Box<Iterator<Item = ClearValue>> {
                    // FIXME: safety checks
                    Box::new(values.into_iter())
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
        }

        scope::CustomRenderPassDesc {
            $(
                $atch_name: ($format, $samples),
            )*
        }.build_render_pass($device)
    });







    ([] __impl_attachments__ [] [] [] [$($params:ident),*]) => {
        unsafe impl RenderPassDescAttachmentsList<AttachmentsStart> for CustomRenderPassDesc {
            type List = ();

            fn check_attachments_list(&self, attachments: AttachmentsStart) -> Result<(), FramebufferCreationError> {
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

        impl<$first_param> $next<$first_param> {
            fn check_attachments_list(self) -> Result<($first_param,), FramebufferCreationError> {
                Ok((self.current,))     // FIXME: check attachment
            }
        }

        ordered_passes_renderpass!{[] __impl_attachments__ [$next] [$first_param] [$($rest),*] [$($rest_params),*]}
    };

    ([] __impl_attachments__ [$prev:ident] [$($prev_params:ident),*] [] [$($params:ident),*]) => {
        unsafe impl<$($prev_params,)*> RenderPassDescAttachmentsList<$prev<$($prev_params,)*>> for CustomRenderPassDesc
            where $($prev_params: ImageView,)*
        {
            type List = ($($prev_params,)*);

            fn check_attachments_list(&self, attachments: $prev<$($prev_params,)*>) -> Result<Self::List, FramebufferCreationError> {
                attachments.check_attachments_list()
                
                // FIXME:
                /*$({
                    if !l.$atch_name.identity_swizzle() {
                        return Err(FramebufferCreationError::AttachmentNotIdentitySwizzled);
                    }
                })**/
            }
        }
    };

    ([] __impl_attachments__ [$prev:ident] [$($prev_params:ident),*] [$next:ident $(, $rest:ident)*] [$first_param:ident, $($rest_params:ident),*]) => {
        pub struct $next<$($prev_params,)* $first_param> {
            prev: $prev<$($prev_params,)*>,
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

        impl<$($prev_params,)* $first_param> $next<$($prev_params,)* $first_param> {
            fn check_attachments_list(self) -> Result<($first_param, $($prev_params,)*), FramebufferCreationError> {
                let ($($prev_params,)*) = try!(self.prev.check_attachments_list());
                // FIXME: check attachment
                (self.current, $($prev_params,)*)
            }
        }

        ordered_passes_renderpass!{[] __impl_attachments__ [$next] [$($prev_params,)* $first_param] [$($rest),*] [$($rest_params),*]}
    };







    ([] __impl_clear_values__ [] [] [] [$($params:ident),*]) => {
        unsafe impl RenderPassDescClearValues<ClearValuesStart> for CustomRenderPassDesc {
            #[inline]
            fn convert_clear_values(&self, values: ClearValuesStart) -> Box<Iterator<Item = ClearValue>> {
                Box::new(iter::empty())
            }
        }
    };

    ([] __impl_clear_values__ [] [] [$next:ident: Clear $(, $rest:ident: $rest_load:ident)*] [$first_param:ident, $($rest_params:ident),*]) => {
        pub struct $next<$first_param> {
            current: $first_param,
        }

        impl ClearValuesStart {
            pub fn $next<$first_param>(self, next: $first_param) -> $next<$first_param> {
                $next {
                    current: next,
                }
            }
        }

        impl<$first_param> $next<$first_param>
            where $first_param: Into<ClearValue>
        {
            #[inline]
            fn convert_clear_values(self) -> iter::Once<ClearValue> {
                // FIXME: check format
                iter::once(self.current.into())
            }
        }

        ordered_passes_renderpass!{[] __impl_clear_values__ [$next] [$first_param] [$($rest: $rest_load),*] [$($rest_params),*]}
    };

    ([] __impl_clear_values__ [] [] [$next:ident: $other:ident $(, $rest:ident: $rest_load:ident)*] [$first_param:ident, $($rest_params:ident),*]) => {
        ordered_passes_renderpass!{[] __impl_clear_values__ [] [] [$($rest: $rest_load),*] [$first_param, $($rest_params),*]}
    };

    ([] __impl_clear_values__ [$prev:ident] [$($prev_params:ident),*] [] [$($params:ident),*]) => {
        unsafe impl<$($prev_params,)*> RenderPassDescClearValues<$prev<$($prev_params,)*>> for CustomRenderPassDesc
            where $($prev_params: Into<ClearValue>)*
        {
            #[inline]
            fn convert_clear_values(&self, values: $prev<$($prev_params,)*>) -> Box<Iterator<Item = ClearValue>> {
                Box::new(values.convert_clear_values())
            }
        }
    };

    ([] __impl_clear_values__ [$prev:ident] [$($prev_params:ident),*] [$next:ident: Clear $(, $rest:ident: $rest_load:ident)*] [$first_param:ident, $($rest_params:ident),*]) => {
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

        impl<$($prev_params,)* $first_param> $next<$($prev_params,)* $first_param>
            where $first_param: Into<ClearValue>,
                  $($prev_params: Into<ClearValue>,)*
        {
            #[inline]
            fn convert_clear_values(self) -> Box<Iterator<Item = ClearValue>> {
                // TODO: subopptimal iterator
                let prev = self.prev.convert_clear_values();
                // FIXME: check format
                prev.chain(iter::once(self.current.into()))
            }
        }

        ordered_passes_renderpass!{[] __impl_clear_values__ [$next] [$($prev_params,)* $first_param] [$($rest: $rest_load),*] [$($rest_params),*]}
    };

    ([] __impl_clear_values__ [$prev:ident] [$($prev_params:ident),*] [$next:ident: $other:ident $(, $rest:ident: $rest_load:ident)*] [$first_param:ident, $($rest_params:ident),*]) => {
        ordered_passes_renderpass!{[] __impl_clear_values__ [$prev] [$($prev_params,)*] [$($rest: $rest_load),*] [$first_param, $($rest_params),*]}
    };
}
