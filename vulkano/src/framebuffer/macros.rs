// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

/// Builds a `RenderPass` object whose template parameter is of indeterminate type.
#[macro_export]
macro_rules! single_pass_renderpass {
    (
        $device:expr,
        attachments: { $($a:tt)* },
        pass: {
            color: [$($color_atch:ident),*],
            depth_stencil: {$($depth_atch:ident)*}$(,)*
            $(resolve: [$($resolve_atch:ident),*])*$(,)*
        }
    ) => (
        $crate::ordered_passes_renderpass!(
            $device,
            attachments: { $($a)* },
            passes: [
                {
                    color: [$($color_atch),*],
                    depth_stencil: {$($depth_atch)*},
                    input: [],
                    resolve: [$($($resolve_atch),*)*]
                }
            ]
        )
    )
}

/// Builds a `RenderPass` object whose template parameter is of indeterminate type.
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
                    input: [$($input_atch:ident),*]$(,)*
                    $(resolve: [$($resolve_atch:ident),*])*$(,)*
                }
            ),*
        ]
    ) => ({
        use $crate::framebuffer::RenderPassDesc;

        mod scope {
            #![allow(non_camel_case_types)]
            #![allow(non_snake_case)]

            use $crate::format::ClearValue;
            use $crate::format::Format;
            use $crate::framebuffer::RenderPassDesc;
            use $crate::framebuffer::RenderPassDescClearValues;
            use $crate::framebuffer::AttachmentDescription;
            use $crate::framebuffer::PassDescription;
            use $crate::framebuffer::PassDependencyDescription;
            use $crate::image::ImageLayout;
            use $crate::sync::AccessFlagBits;
            use $crate::sync::PipelineStages;

            pub struct CustomRenderPassDesc {
                $(
                    pub $atch_name: (Format, u32),
                )*
            }

            #[allow(unsafe_code)]
            unsafe impl RenderPassDesc for CustomRenderPassDesc {
                #[inline]
                fn num_attachments(&self) -> usize {
                    num_attachments()
                }

                #[inline]
                fn attachment_desc(&self, id: usize) -> Option<AttachmentDescription> {
                    attachment(self, id)
                }

                #[inline]
                fn num_subpasses(&self) -> usize {
                    num_subpasses()
                }

                #[inline]
                fn subpass_desc(&self, id: usize) -> Option<PassDescription> {
                    subpass(id)
                }

                #[inline]
                fn num_dependencies(&self) -> usize {
                    num_dependencies()
                }

                #[inline]
                fn dependency_desc(&self, id: usize) -> Option<PassDependencyDescription> {
                    dependency(id)
                }
            }

            unsafe impl RenderPassDescClearValues<Vec<ClearValue>> for CustomRenderPassDesc {
                fn convert_clear_values(&self, values: Vec<ClearValue>) -> Box<dyn Iterator<Item = ClearValue>> {
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
            fn attachment(desc: &CustomRenderPassDesc, id: usize) -> Option<AttachmentDescription> {
                #![allow(unused_assignments)]
                #![allow(unused_mut)]

                let mut num = 0;

                $({
                    if id == num {
                        let (initial_layout, final_layout) = attachment_layouts(num);

                        return Some($crate::framebuffer::AttachmentDescription {
                            format: desc.$atch_name.0,
                            samples: desc.$atch_name.1,
                            load: $crate::framebuffer::LoadOp::$load,
                            store: $crate::framebuffer::StoreOp::$store,
                            stencil_load: $crate::framebuffer::LoadOp::$load,
                            stencil_store: $crate::framebuffer::StoreOp::$store,
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
            fn subpass(id: usize) -> Option<PassDescription> {
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
                            depth = Some(($depth_atch, ImageLayout::DepthStencilAttachmentOptimal));
                        )*

                        let mut desc = PassDescription {
                            color_attachments: vec![
                                $(
                                    ($color_atch, ImageLayout::ColorAttachmentOptimal)
                                ),*
                            ],
                            depth_stencil: depth,
                            input_attachments: vec![
                                $(
                                    ($input_atch, ImageLayout::ShaderReadOnlyOptimal)
                                ),*
                            ],
                            resolve_attachments: vec![
                                $($(
                                    ($resolve_atch, ImageLayout::TransferDstOptimal)
                                ),*)*
                            ],
                            preserve_attachments: (0 .. attachment_num).filter(|&a| {
                                $(if a == $color_atch { return false; })*
                                $(if a == $depth_atch { return false; })*
                                $(if a == $input_atch { return false; })*
                                $($(if a == $resolve_atch { return false; })*)*
                                true
                            }).collect()
                        };

                        assert!(desc.resolve_attachments.is_empty() ||
                                desc.resolve_attachments.len() == desc.color_attachments.len());
                        return Some(desc);
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
            fn dependency(id: usize) -> Option<PassDependencyDescription> {
                let num_passes = num_subpasses();

                if id + 1 >= num_passes {
                    return None;
                }

                Some(PassDependencyDescription {
                    source_subpass: id,
                    destination_subpass: id + 1,
                    source_stages: PipelineStages { all_graphics: true, .. PipelineStages::none() },         // TODO: correct values
                    destination_stages: PipelineStages { all_graphics: true, .. PipelineStages::none() },         // TODO: correct values
                    source_access: AccessFlagBits::all(),         // TODO: correct values
                    destination_access: AccessFlagBits::all(),         // TODO: correct values
                    by_region: true,            // TODO: correct values
                })
            }

            /// Returns the initial and final layout of an attachment, given its num.
            ///
            /// The value always correspond to the first and last usages of an attachment.
            fn attachment_layouts(num: usize) -> (ImageLayout, ImageLayout) {
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
                                initial_layout = Some(ImageLayout::DepthStencilAttachmentOptimal);
                            }
                            final_layout = Some(ImageLayout::DepthStencilAttachmentOptimal);
                        }
                    )*

                    $(
                        if $color_atch == num {
                            if initial_layout.is_none() {
                                initial_layout = Some(ImageLayout::ColorAttachmentOptimal);
                            }
                            final_layout = Some(ImageLayout::ColorAttachmentOptimal);
                        }
                    )*

                    $($(
                        if $resolve_atch == num {
                            if initial_layout.is_none() {
                                initial_layout = Some(ImageLayout::TransferDstOptimal);
                            }
                            final_layout = Some(ImageLayout::TransferDstOptimal);
                        }
                    )*)*

                    $(
                        if $input_atch == num {
                            if initial_layout.is_none() {
                                initial_layout = Some(ImageLayout::ShaderReadOnlyOptimal);
                            }
                            final_layout = Some(ImageLayout::ShaderReadOnlyOptimal);
                        }
                    )*
                })*

                $(if $atch_name == num {
                    $(initial_layout = Some($init_layout);)*
                    $(final_layout = Some($final_layout);)*
                })*
                (
                    initial_layout.expect(format!("Attachment {} is missing initial_layout, this is normally \
                        automatically determined but you can manually specify it for an individual \
                        attachment in the single_pass_renderpass! macro", attachment_num).as_ref()),
                    final_layout.expect(format!("Attachment {} is missing final_layout, this is normally \
                        automatically determined but you can manually specify it for an individual \
                        attachment in the single_pass_renderpass! macro", attachment_num).as_ref())
                )
            }
        }

        scope::CustomRenderPassDesc {
            $(
                $atch_name: ($format, $samples),
            )*
        }.build_render_pass($device)
    });
}

#[cfg(test)]
mod tests {
    use format::Format;

    #[test]
    fn single_pass_resolve() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = single_pass_renderpass!(device.clone(),
            attachments: {
                a: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R8G8B8A8Unorm,
                    samples: 4,
                },
                b: {
                    load: DontCare,
                    store: Store,
                    format: Format::R8G8B8A8Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [a],
                depth_stencil: {},
                resolve: [b],
            }
        ).unwrap();
    }
}
