// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
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
        use $crate::render_pass::RenderPass;

        let desc = {
            use $crate::render_pass::AttachmentDesc;
            use $crate::render_pass::RenderPassDesc;
            use $crate::render_pass::SubpassDependencyDesc;
            use $crate::render_pass::SubpassDesc;
            use $crate::image::ImageLayout;
            use $crate::sync::AccessFlags;
            use $crate::sync::PipelineStages;
            use std::convert::TryInto;

            let mut attachment_num = 0;
            $(
                let $atch_name = attachment_num;
                attachment_num += 1;
            )*

            let mut layouts: Vec<(Option<ImageLayout>, Option<ImageLayout>)> = vec![(None, None); attachment_num];

            let subpasses = vec![
                $({
                    let desc = SubpassDesc {
                        color_attachments: vec![
                            $({
                                let layout = &mut layouts[$color_atch];
                                layout.0 = layout.0.or(Some(ImageLayout::ColorAttachmentOptimal));
                                layout.1 = Some(ImageLayout::ColorAttachmentOptimal);

                                ($color_atch, ImageLayout::ColorAttachmentOptimal)
                            }),*
                        ],
                        depth_stencil: {
                            let depth: Option<(usize, ImageLayout)> = None;
                            $(
                                let layout = &mut layouts[$depth_atch];
                                layout.1 = Some(ImageLayout::DepthStencilAttachmentOptimal);
                                layout.0 = layout.0.or(layout.1);

                                let depth = Some(($depth_atch, ImageLayout::DepthStencilAttachmentOptimal));
                            )*
                            depth
                        },
                        input_attachments: vec![
                            $({
                                let layout = &mut layouts[$input_atch];
                                layout.1 = Some(ImageLayout::ShaderReadOnlyOptimal);
                                layout.0 = layout.0.or(layout.1);

                                ($input_atch, ImageLayout::ShaderReadOnlyOptimal)
                            }),*
                        ],
                        resolve_attachments: vec![
                            $($({
                                let layout = &mut layouts[$resolve_atch];
                                layout.1 = Some(ImageLayout::TransferDstOptimal);
                                layout.0 = layout.0.or(layout.1);

                                ($resolve_atch, ImageLayout::TransferDstOptimal)
                            }),*)*
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
                    desc
                }),*
            ];

            let dependencies = (0..subpasses.len().saturating_sub(1))
                .map(|id| {
                    SubpassDependencyDesc {
                        source_subpass: id,
                        destination_subpass: id + 1,
                        source_stages: PipelineStages {
                            all_graphics: true,
                            ..PipelineStages::none()
                        }, // TODO: correct values
                        destination_stages: PipelineStages {
                            all_graphics: true,
                            ..PipelineStages::none()
                        }, // TODO: correct values
                        source_access: AccessFlags::all(), // TODO: correct values
                        destination_access: AccessFlags::all(), // TODO: correct values
                        by_region: true,                      // TODO: correct values
                    }
                })
                .collect();

            let attachments = vec![
                $({
                    let layout = &mut layouts[$atch_name];
                    $(layout.0 = Some($init_layout);)*
                    $(layout.1 = Some($final_layout);)*

                    AttachmentDesc {
                        format: $format,
                        samples: $samples.try_into().unwrap(),
                        load: $crate::render_pass::LoadOp::$load,
                        store: $crate::render_pass::StoreOp::$store,
                        stencil_load: $crate::render_pass::LoadOp::$load,
                        stencil_store: $crate::render_pass::StoreOp::$store,
                        initial_layout: layout.0.expect(
                            format!(
                                "Attachment {} is missing initial_layout, this is normally \
                                automatically determined but you can manually specify it for an individual \
                                attachment in the single_pass_renderpass! macro",
                                attachment_num
                            )
                            .as_ref(),
                        ),
                        final_layout: layout.1.expect(
                            format!(
                                "Attachment {} is missing final_layout, this is normally \
                                automatically determined but you can manually specify it for an individual \
                                attachment in the single_pass_renderpass! macro",
                                attachment_num
                            )
                            .as_ref(),
                        ),
                    }
                }),*
            ];

            RenderPassDesc::new(
                attachments,
                subpasses,
                dependencies,
            )
        };

        RenderPass::new($device, desc)
    });
}

#[cfg(test)]
mod tests {
    use crate::format::Format;

    #[test]
    fn single_pass_resolve() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = single_pass_renderpass!(device.clone(),
            attachments: {
                a: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R8G8B8A8_UNORM,
                    samples: 4,
                },
                b: {
                    load: DontCare,
                    store: Store,
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                }
            },
            pass: {
                color: [a],
                depth_stencil: {},
                resolve: [b],
            }
        )
        .unwrap();
    }
}
