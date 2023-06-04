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
            color: [
                $(
                    $color_atch:ident
                    $(-> $color_resolve_atch:ident:$color_resolve_mode:ident)?
                ),* $(,)?
            ],
            depth: {
                $(
                    $depth_atch:ident
                    $(-> $depth_resolve_atch:ident:$depth_resolve_mode:ident)?
                )?
            },
            stencil: {
                $(
                    $stencil_atch:ident
                    $(-> $stencil_resolve_atch:ident:$stencil_resolve_mode:ident)?
                )?
            }
            $(,)*
        } $(,)?
    ) => (
        $crate::ordered_passes_renderpass!(
            $device,
            attachments: { $($a)* },
            passes: [
                {
                    color: [
                        $(
                            $color_atch
                            $(-> $color_resolve_atch:$color_resolve_mode)?
                        ),*
                    ],
                    depth: {
                        $(
                            $depth_atch
                            $(-> $depth_resolve_atch:$depth_resolve_mode)?
                        )?
                    },
                    stencil: {
                        $(
                            $stencil_atch
                            $(-> $stencil_resolve_atch:$stencil_resolve_mode)?
                        )?
                    },
                    input: [],
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
                    samples: $samples:expr
                    $(,initial_layout: $init_layout:expr)?
                    $(,final_layout: $final_layout:expr)?
                    $(,)?
                }
            ),* $(,)?
        },
        passes: [
            $(
                {
                    color: [
                        $(
                            $color_atch:ident
                            $(-> $color_resolve_atch:ident:$color_resolve_mode:ident)?
                        ),* $(,)?
                    ],
                    depth: {
                        $(
                            $depth_atch:ident
                            $(-> $depth_resolve_atch:ident:$depth_resolve_mode:ident)?
                        )?
                    },
                    stencil: {
                        $(
                            $stencil_atch:ident
                            $(-> $stencil_resolve_atch:ident:$stencil_resolve_mode:ident)?
                        )?
                    },
                    input: [$($input_atch:ident),* $(,)?]
                    $(,)*
                }
            ),* $(,)?
        ] $(,)?
    ) => ({
        use $crate::render_pass::RenderPass;

        let create_info = {
            #[allow(unused)]
            let mut attachment_num = 0;
            $(
                let $atch_name = attachment_num;
                attachment_num += 1;
            )*

            #[allow(unused)]
            let mut layouts: Vec<(
                Option<$crate::image::ImageLayout>,
                Option<$crate::image::ImageLayout>
            )> = vec![(None, None); attachment_num as usize];

            let subpasses = vec![
                $({
                    let desc = $crate::render_pass::SubpassDescription {
                        color_attachments: vec![
                            $({
                                let layout = &mut layouts[$color_atch as usize];
                                layout.0 = layout.0.or(Some($crate::image::ImageLayout::ColorAttachmentOptimal));
                                layout.1 = Some($crate::image::ImageLayout::ColorAttachmentOptimal);

                                Some($crate::render_pass::ResolvableAttachmentReference{
                                    attachment_ref: $crate::render_pass::AttachmentReference {
                                        attachment: $color_atch,
                                        layout: $crate::image::ImageLayout::ColorAttachmentOptimal,
                                        ..Default::default()
                                    },
                                    resolve: None $(.or({
                                        let layout = &mut layouts[$color_resolve_atch as usize];
                                        layout.1 = Some($crate::image::ImageLayout::TransferDstOptimal);
                                        layout.0 = layout.0.or(layout.1);

                                        Some($crate::render_pass::ResolveAttachmentReference {
                                            attachment_ref: $crate::render_pass::AttachmentReference {
                                                attachment: $color_resolve_atch,
                                                layout: $crate::image::ImageLayout::TransferDstOptimal,
                                                ..Default::default()
                                            },
                                            mode: $crate::render_pass::ResolveMode::$color_resolve_mode,
                                        })
                                    }))?,
                                })
                            }),*
                        ],
                        depth_attachment: {
                            None $(.or({
                                let layout = &mut layouts[$depth_atch as usize];
                                layout.1 = Some($crate::image::ImageLayout::DepthStencilAttachmentOptimal);
                                layout.0 = layout.0.or(layout.1);

                                Some($crate::render_pass::ResolvableAttachmentReference{
                                    attachment_ref: $crate::render_pass::AttachmentReference {
                                        attachment: $depth_atch,
                                        layout: $crate::image::ImageLayout::DepthStencilAttachmentOptimal,
                                        ..Default::default()
                                    },
                                    resolve: None $(.or({
                                        let layout = &mut layouts[$depth_resolve_atch as usize];
                                        layout.1 = Some($crate::image::ImageLayout::TransferDstOptimal);
                                        layout.0 = layout.0.or(layout.1);

                                        Some($crate::render_pass::ResolveAttachmentReference {
                                            attachment_ref: $crate::render_pass::AttachmentReference {
                                                attachment: $depth_resolve_atch,
                                                layout: $crate::image::ImageLayout::TransferDstOptimal,
                                                ..Default::default()
                                            },
                                            mode: $crate::render_pass::ResolveMode::$depth_resolve_mode,
                                        })
                                    }))?,
                                })
                            }))?
                        },
                        stencil_attachment: {
                            None $(.or({
                                let layout = &mut layouts[$stencil_atch as usize];
                                layout.1 = Some($crate::image::ImageLayout::DepthStencilAttachmentOptimal);
                                layout.0 = layout.0.or(layout.1);

                                Some($crate::render_pass::ResolvableAttachmentReference{
                                    attachment_ref: $crate::render_pass::AttachmentReference {
                                        attachment: $stencil_atch,
                                        layout: $crate::image::ImageLayout::DepthStencilAttachmentOptimal,
                                        ..Default::default()
                                    },
                                    resolve: None $(.or({
                                        let layout = &mut layouts[$stencil_resolve_atch as usize];
                                        layout.1 = Some($crate::image::ImageLayout::TransferDstOptimal);
                                        layout.0 = layout.0.or(layout.1);

                                        Some($crate::render_pass::ResolveAttachmentReference {
                                            attachment_ref: $crate::render_pass::AttachmentReference {
                                                attachment: $stencil_resolve_atch,
                                                layout: $crate::image::ImageLayout::TransferDstOptimal,
                                                ..Default::default()
                                            },
                                            mode: $crate::render_pass::ResolveMode::$stencil_resolve_mode,
                                        })
                                    }))?,
                                })
                            }))?
                        },
                        input_attachments: vec![
                            $({
                                let layout = &mut layouts[$input_atch as usize];
                                layout.1 = Some($crate::image::ImageLayout::ShaderReadOnlyOptimal);
                                layout.0 = layout.0.or(layout.1);

                                Some($crate::render_pass::InputAttachmentReference {
                                    attachment_ref: $crate::render_pass::AttachmentReference {
                                        attachment: $input_atch,
                                        layout: $crate::image::ImageLayout::ShaderReadOnlyOptimal,
                                        ..Default::default()
                                    },
                                    ..Default::default()
                                })
                            }),*
                        ],
                        preserve_attachments: (0 .. attachment_num).filter(|&a| {
                            ![
                                $($color_atch, $($color_resolve_atch,)?)*
                                $($depth_atch,)*
                                $($input_atch,)*
                            ].contains(&a)
                        }).collect(),
                        ..Default::default()
                    };

                    desc
                }),*
            ];

            let dependencies: Vec<_> = (0..subpasses.len().saturating_sub(1) as u32)
                .map(|id| {
                    // TODO: correct values
                    let src_stages = $crate::sync::PipelineStages::ALL_GRAPHICS;
                    let dst_stages = $crate::sync::PipelineStages::ALL_GRAPHICS;
                    let src_access = $crate::sync::AccessFlags::MEMORY_READ
                        | $crate::sync::AccessFlags::MEMORY_WRITE;
                    let dst_access = $crate::sync::AccessFlags::MEMORY_READ
                        | $crate::sync::AccessFlags::MEMORY_WRITE;

                    $crate::render_pass::SubpassDependency {
                        src_subpass: id.into(),
                        dst_subpass: (id + 1).into(),
                        src_stages,
                        dst_stages,
                        src_access,
                        dst_access,
                        // TODO: correct values
                        dependency_flags: $crate::sync::DependencyFlags::BY_REGION,
                        ..Default::default()
                    }
                })
                .collect();

            let attachments = vec![
                $({
                    let layout = &mut layouts[$atch_name as usize];
                    $(layout.0 = Some($init_layout);)*
                    $(layout.1 = Some($final_layout);)*

                    $crate::render_pass::AttachmentDescription {
                        format: Some($format),
                        samples: $crate::image::SampleCount::try_from($samples).unwrap(),
                        load_op: $crate::render_pass::LoadOp::$load,
                        store_op: $crate::render_pass::StoreOp::$store,
                        stencil_load_op: $crate::render_pass::LoadOp::$load,
                        stencil_store_op: $crate::render_pass::StoreOp::$store,
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
                        ..Default::default()
                    }
                }),*
            ];

            $crate::render_pass::RenderPassCreateInfo {
                attachments,
                subpasses,
                dependencies,
                ..Default::default()
            }
        };

        RenderPass::new($device, create_info)
    });
}

#[cfg(test)]
mod tests {
    use crate::format::Format;

    #[test]
    fn single_pass_resolve() {
        let (device, _) = gfx_dev_and_queue!();
        let _ = single_pass_renderpass!(
            device,
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
                },
            },
            pass: {
                color: [a -> b:Average],
                depth: {},
                stencil: {},
            },
        )
        .unwrap();
    }
}
