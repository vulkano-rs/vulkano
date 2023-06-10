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
            color: [$($color_atch:ident),* $(,)?]
            $(, color_resolve: [$($color_resolve_atch:ident),* $(,)?])?
            , depth_stencil: {$($depth_stencil_atch:ident)?}
            $(, depth_stencil_resolve: {$depth_stencil_resolve_atch:ident})?
            $(,)?
        } $(,)?
    ) => (
        $crate::ordered_passes_renderpass!(
            $device,
            attachments: { $($a)* },
            passes: [
                {
                    color: [$($color_atch),*]
                    $(, color_resolve: [$($color_resolve_atch),*])?
                    , depth_stencil: {$($depth_stencil_atch)?}
                    $(, depth_stencil_resolve: {$depth_stencil_resolve_atch})?
                    , input: [],
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
                    format: $format:expr,
                    samples: $samples:expr,
                    load_op: $load_op:ident,
                    store_op: $store_op:ident
                    $(,initial_layout: $init_layout:expr)?
                    $(,final_layout: $final_layout:expr)?
                    $(,)?
                }
            ),* $(,)?
        },
        passes: [
            $(
                {
                    color: [$($color_atch:ident),* $(,)?]
                    $(, color_resolve: [$($color_resolve_atch:ident),* $(,)?])?
                    , depth_stencil: {$($depth_stencil_atch:ident)?}
                    $(, depth_stencil_resolve: {$depth_stencil_resolve_atch:ident})?
                    , input: [$($input_atch:ident),* $(,)?]
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
            #[derive(Clone, Copy, Default)]
            struct Layouts {
                initial_layout: Option<$crate::image::ImageLayout>,
                final_layout: Option<$crate::image::ImageLayout>,
            }

            #[allow(unused)]
            let mut layouts: Vec<Layouts> = vec![Layouts::default(); attachment_num as usize];

            let subpasses = vec![
                $({
                    let desc = $crate::render_pass::SubpassDescription {
                        color_attachments: vec![
                            $({
                                let layouts = &mut layouts[$color_atch as usize];
                                layouts.initial_layout = layouts.initial_layout.or(Some($crate::image::ImageLayout::ColorAttachmentOptimal));
                                layouts.final_layout = Some($crate::image::ImageLayout::ColorAttachmentOptimal);

                                Some($crate::render_pass::AttachmentReference {
                                    attachment: $color_atch,
                                    layout: $crate::image::ImageLayout::ColorAttachmentOptimal,
                                    ..Default::default()
                                })
                            }),*
                        ],
                        color_resolve_attachments: vec![$(
                            $({
                                let layouts = &mut layouts[$color_resolve_atch as usize];
                                layouts.final_layout = Some($crate::image::ImageLayout::TransferDstOptimal);
                                layouts.initial_layout = layouts.initial_layout.or(layouts.final_layout);

                                Some($crate::render_pass::AttachmentReference {
                                    attachment: $color_resolve_atch,
                                    layout: $crate::image::ImageLayout::TransferDstOptimal,
                                    ..Default::default()
                                })
                            }),*
                        )?],
                        depth_stencil_attachment: {
                            None $(.or({
                                let layouts = &mut layouts[$depth_stencil_atch as usize];
                                layouts.final_layout = Some($crate::image::ImageLayout::DepthStencilAttachmentOptimal);
                                layouts.initial_layout = layouts.initial_layout.or(layouts.final_layout);

                                Some($crate::render_pass::AttachmentReference {
                                    attachment: $depth_stencil_atch,
                                    layout: $crate::image::ImageLayout::DepthStencilAttachmentOptimal,
                                    ..Default::default()
                                })
                            }))?
                        },
                        depth_stencil_resolve_attachment: {
                            None $(.or({
                                let layouts = &mut layouts[$depth_stencil_resolve_atch as usize];
                                layouts.final_layout = Some($crate::image::ImageLayout::TransferDstOptimal);
                                layouts.initial_layout = layouts.initial_layout.or(layouts.final_layout);

                                Some($crate::render_pass::AttachmentReference {
                                    attachment: $depth_stencil_resolve_atch,
                                    layout: $crate::image::ImageLayout::TransferDstOptimal,
                                    ..Default::default()
                                })
                            }))?
                        },
                        depth_resolve_mode: None, // TODO:
                        stencil_resolve_mode: None, // TODO:
                        input_attachments: vec![
                            $({
                                let layouts = &mut layouts[$input_atch as usize];
                                layouts.final_layout = Some($crate::image::ImageLayout::ShaderReadOnlyOptimal);
                                layouts.initial_layout = layouts.initial_layout.or(layouts.final_layout);

                                Some($crate::render_pass::AttachmentReference {
                                    attachment: $input_atch,
                                    layout: $crate::image::ImageLayout::ShaderReadOnlyOptimal,
                                    ..Default::default()
                                })
                            }),*
                        ],
                        preserve_attachments: (0 .. attachment_num).filter(|&a| {
                            ![
                                $($color_atch,)*
                                $($($color_resolve_atch,)*)?
                                $($depth_stencil_atch,)*
                                $($depth_stencil_resolve_atch,)*
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
                    let layouts = &mut layouts[$atch_name as usize];
                    $(layouts.initial_layout = Some($init_layout);)*
                    $(layouts.final_layout = Some($final_layout);)*

                    $crate::render_pass::AttachmentDescription {
                        format: Some($format),
                        samples: $crate::image::SampleCount::try_from($samples).unwrap(),
                        load_op: $crate::render_pass::AttachmentLoadOp::$load_op,
                        store_op: $crate::render_pass::AttachmentStoreOp::$store_op,
                        initial_layout: layouts.initial_layout.expect(
                            format!(
                                "Attachment {} is missing initial_layout, this is normally \
                                automatically determined but you can manually specify it for an individual \
                                attachment in the single_pass_renderpass! macro",
                                attachment_num
                            )
                            .as_ref(),
                        ),
                        final_layout: layouts.final_layout.expect(
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
                    format: Format::R8G8B8A8_UNORM,
                    samples: 4,
                    load_op: Clear,
                    store_op: DontCare,
                },
                b: {
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                    load_op: DontCare,
                    store_op: Store,
                },
            },
            pass: {
                color: [a],
                color_resolve: [b],
                depth_stencil: {},
            },
        )
        .unwrap();
    }
}
