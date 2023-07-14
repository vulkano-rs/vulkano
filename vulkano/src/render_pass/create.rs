// Copyright (c) 2022 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{AttachmentDescription, AttachmentReference, RenderPass, RenderPassCreateInfo};
use crate::{
    device::Device,
    render_pass::{SubpassDependency, SubpassDescription},
    Version, VulkanError, VulkanObject,
};
use smallvec::SmallVec;
use std::{mem::MaybeUninit, ptr};

impl RenderPass {
    pub(super) unsafe fn create_v2(
        device: &Device,
        create_info: &RenderPassCreateInfo,
    ) -> Result<ash::vk::RenderPass, VulkanError> {
        let &RenderPassCreateInfo {
            flags,
            ref attachments,
            ref subpasses,
            ref dependencies,
            ref correlated_view_masks,
            _ne: _,
        } = create_info;

        struct PerAttachment {
            stencil_layout_vk: Option<ash::vk::AttachmentDescriptionStencilLayout>,
        }

        let (mut attachments_vk, mut per_attachment_vk): (SmallVec<[_; 4]>, SmallVec<[_; 4]>) =
            attachments
                .iter()
                .map(|attachment| {
                    let &AttachmentDescription {
                        flags,
                        format,
                        samples,
                        load_op,
                        store_op,
                        initial_layout,
                        final_layout,
                        stencil_load_op,
                        stencil_store_op,
                        stencil_initial_layout,
                        stencil_final_layout,
                        _ne: _,
                    } = attachment;

                    (
                        ash::vk::AttachmentDescription2 {
                            flags: flags.into(),
                            format: format.into(),
                            samples: samples.into(),
                            load_op: load_op.into(),
                            store_op: store_op.into(),
                            stencil_load_op: stencil_load_op.unwrap_or(load_op).into(),
                            stencil_store_op: stencil_store_op.unwrap_or(store_op).into(),
                            initial_layout: initial_layout.into(),
                            final_layout: final_layout.into(),
                            ..Default::default()
                        },
                        PerAttachment {
                            stencil_layout_vk: stencil_initial_layout
                                .zip(stencil_final_layout)
                                .map(|(stencil_initial_layout, stencil_final_layout)| {
                                    ash::vk::AttachmentDescriptionStencilLayout {
                                        stencil_initial_layout: stencil_initial_layout.into(),
                                        stencil_final_layout: stencil_final_layout.into(),
                                        ..Default::default()
                                    }
                                }),
                        },
                    )
                })
                .unzip();

        for (attachment_vk, per_attachment_vk) in
            attachments_vk.iter_mut().zip(per_attachment_vk.iter_mut())
        {
            let PerAttachment { stencil_layout_vk } = per_attachment_vk;

            if let Some(next) = stencil_layout_vk {
                next.p_next = attachment_vk.p_next as *mut _;
                attachment_vk.p_next = next as *const _ as *const _;
            }
        }

        struct PerSubpassDescriptionVk {
            input_attachments_vk: SmallVec<[ash::vk::AttachmentReference2; 4]>,
            per_input_attachments_vk: SmallVec<[PerAttachmentReferenceVk; 4]>,
            color_attachments_vk: SmallVec<[ash::vk::AttachmentReference2; 4]>,
            resolve_attachments_vk: SmallVec<[ash::vk::AttachmentReference2; 4]>,
            depth_stencil_attachment_vk: ash::vk::AttachmentReference2,
            per_depth_stencil_attachment_vk: PerAttachmentReferenceVk,
            depth_stencil_resolve_attachment_vk: ash::vk::AttachmentReference2,
            per_depth_stencil_resolve_attachment_vk: PerAttachmentReferenceVk,
            depth_stencil_resolve_vk: Option<ash::vk::SubpassDescriptionDepthStencilResolve>,
        }

        #[derive(Default)]
        struct PerAttachmentReferenceVk {
            stencil_layout_vk: Option<ash::vk::AttachmentReferenceStencilLayout>,
        }

        let (mut subpasses_vk, mut per_subpass_vk): (SmallVec<[_; 4]>, SmallVec<[_; 4]>) =
            subpasses
                .iter()
                .map(|subpass| {
                    let &SubpassDescription {
                        flags,
                        view_mask,
                        ref input_attachments,
                        ref color_attachments,
                        ref color_resolve_attachments,
                        ref depth_stencil_attachment,
                        ref depth_stencil_resolve_attachment,
                        depth_resolve_mode,
                        stencil_resolve_mode,
                        ref preserve_attachments,
                        _ne: _,
                    } = subpass;

                    let (input_attachments_vk, per_input_attachments_vk) = input_attachments
                        .iter()
                        .map(|input_attachment| {
                            if let Some(input_attachment) = input_attachment {
                                let &AttachmentReference {
                                    attachment,
                                    layout,
                                    stencil_layout,
                                    aspects,
                                    _ne: _,
                                } = input_attachment;

                                (
                                    ash::vk::AttachmentReference2 {
                                        attachment,
                                        layout: layout.into(),
                                        aspect_mask: aspects.into(),
                                        ..Default::default()
                                    },
                                    PerAttachmentReferenceVk {
                                        stencil_layout_vk: stencil_layout.map(|stencil_layout| {
                                            ash::vk::AttachmentReferenceStencilLayout {
                                                stencil_layout: stencil_layout.into(),
                                                ..Default::default()
                                            }
                                        }),
                                    },
                                )
                            } else {
                                (
                                    ash::vk::AttachmentReference2 {
                                        attachment: ash::vk::ATTACHMENT_UNUSED,
                                        ..Default::default()
                                    },
                                    PerAttachmentReferenceVk::default(),
                                )
                            }
                        })
                        .unzip();

                    let color_attachments_vk = color_attachments
                        .iter()
                        .map(|color_attachment| {
                            if let Some(color_attachment) = color_attachment {
                                let &AttachmentReference {
                                    attachment,
                                    layout,
                                    stencil_layout: _,
                                    aspects: _,
                                    _ne: _,
                                } = color_attachment;

                                ash::vk::AttachmentReference2 {
                                    attachment,
                                    layout: layout.into(),
                                    ..Default::default()
                                }
                            } else {
                                ash::vk::AttachmentReference2 {
                                    attachment: ash::vk::ATTACHMENT_UNUSED,
                                    ..Default::default()
                                }
                            }
                        })
                        .collect();

                    let resolve_attachments_vk = color_resolve_attachments
                        .iter()
                        .map(|color_resolve_attachment| {
                            if let Some(color_resolve_attachment) = color_resolve_attachment {
                                let &AttachmentReference {
                                    attachment,
                                    layout,
                                    stencil_layout: _,
                                    aspects: _,
                                    _ne: _,
                                } = color_resolve_attachment;

                                ash::vk::AttachmentReference2 {
                                    attachment,
                                    layout: layout.into(),
                                    ..Default::default()
                                }
                            } else {
                                ash::vk::AttachmentReference2 {
                                    attachment: ash::vk::ATTACHMENT_UNUSED,
                                    ..Default::default()
                                }
                            }
                        })
                        .collect();

                    let (depth_stencil_attachment_vk, per_depth_stencil_attachment_vk) =
                        if let Some(depth_stencil_attachment) = depth_stencil_attachment {
                            let &AttachmentReference {
                                attachment,
                                layout,
                                stencil_layout,
                                aspects: _,
                                _ne: _,
                            } = depth_stencil_attachment;

                            (
                                ash::vk::AttachmentReference2 {
                                    attachment,
                                    layout: layout.into(),
                                    ..Default::default()
                                },
                                PerAttachmentReferenceVk {
                                    stencil_layout_vk: stencil_layout.map(|stencil_layout| {
                                        ash::vk::AttachmentReferenceStencilLayout {
                                            stencil_layout: stencil_layout.into(),
                                            ..Default::default()
                                        }
                                    }),
                                },
                            )
                        } else {
                            (
                                ash::vk::AttachmentReference2 {
                                    attachment: ash::vk::ATTACHMENT_UNUSED,
                                    ..Default::default()
                                },
                                PerAttachmentReferenceVk::default(),
                            )
                        };

                    let (
                        depth_stencil_resolve_attachment_vk,
                        per_depth_stencil_resolve_attachment_vk,
                    ) = if let Some(depth_stencil_resolve_attachment) =
                        depth_stencil_resolve_attachment
                    {
                        let &AttachmentReference {
                            attachment,
                            layout,
                            stencil_layout,
                            aspects: _,
                            _ne: _,
                        } = depth_stencil_resolve_attachment;

                        (
                            ash::vk::AttachmentReference2 {
                                attachment,
                                layout: layout.into(),
                                ..Default::default()
                            },
                            PerAttachmentReferenceVk {
                                stencil_layout_vk: stencil_layout.map(|stencil_layout| {
                                    ash::vk::AttachmentReferenceStencilLayout {
                                        stencil_layout: stencil_layout.into(),
                                        ..Default::default()
                                    }
                                }),
                            },
                        )
                    } else {
                        (
                            ash::vk::AttachmentReference2 {
                                attachment: ash::vk::ATTACHMENT_UNUSED,
                                ..Default::default()
                            },
                            PerAttachmentReferenceVk::default(),
                        )
                    };

                    let depth_stencil_resolve_vk = depth_stencil_resolve_attachment
                        .is_some()
                        .then_some(ash::vk::SubpassDescriptionDepthStencilResolve {
                            depth_resolve_mode: depth_resolve_mode
                                .map_or(ash::vk::ResolveModeFlags::NONE, Into::into),
                            stencil_resolve_mode: stencil_resolve_mode
                                .map_or(ash::vk::ResolveModeFlags::NONE, Into::into),
                            p_depth_stencil_resolve_attachment: ptr::null(),
                            ..Default::default()
                        });

                    (
                        ash::vk::SubpassDescription2 {
                            flags: flags.into(),
                            pipeline_bind_point: ash::vk::PipelineBindPoint::GRAPHICS, // TODO: any need to make this user-specifiable?
                            view_mask,
                            input_attachment_count: 0,
                            p_input_attachments: ptr::null(),
                            color_attachment_count: 0,
                            p_color_attachments: ptr::null(),
                            p_resolve_attachments: ptr::null(),
                            p_depth_stencil_attachment: ptr::null(),
                            preserve_attachment_count: preserve_attachments.len() as u32,
                            p_preserve_attachments: if preserve_attachments.is_empty() {
                                ptr::null()
                            } else {
                                preserve_attachments.as_ptr()
                            },
                            ..Default::default()
                        },
                        PerSubpassDescriptionVk {
                            input_attachments_vk,
                            per_input_attachments_vk,
                            color_attachments_vk,
                            resolve_attachments_vk,
                            depth_stencil_attachment_vk,
                            per_depth_stencil_attachment_vk,
                            depth_stencil_resolve_attachment_vk,
                            per_depth_stencil_resolve_attachment_vk,
                            depth_stencil_resolve_vk,
                        },
                    )
                })
                .unzip();

        for (subpass_vk, per_subpass_vk) in subpasses_vk.iter_mut().zip(per_subpass_vk.iter_mut()) {
            let PerSubpassDescriptionVk {
                input_attachments_vk,
                per_input_attachments_vk,
                color_attachments_vk,
                resolve_attachments_vk,
                depth_stencil_attachment_vk,
                per_depth_stencil_attachment_vk,
                depth_stencil_resolve_attachment_vk,
                per_depth_stencil_resolve_attachment_vk,
                depth_stencil_resolve_vk,
            } = per_subpass_vk;

            for (input_attachment_vk, per_input_attachment_vk) in input_attachments_vk
                .iter_mut()
                .zip(per_input_attachments_vk)
            {
                let PerAttachmentReferenceVk { stencil_layout_vk } = per_input_attachment_vk;

                if let Some(stencil_layout_vk) = stencil_layout_vk {
                    stencil_layout_vk.p_next = input_attachment_vk.p_next as *mut _;
                    input_attachment_vk.p_next = stencil_layout_vk as *const _ as *const _;
                }
            }

            {
                let PerAttachmentReferenceVk { stencil_layout_vk } =
                    per_depth_stencil_attachment_vk;

                if let Some(stencil_layout_vk) = stencil_layout_vk {
                    stencil_layout_vk.p_next = depth_stencil_attachment_vk.p_next as *mut _;
                    depth_stencil_attachment_vk.p_next = stencil_layout_vk as *const _ as *const _;
                }
            }

            {
                let PerAttachmentReferenceVk { stencil_layout_vk } =
                    per_depth_stencil_resolve_attachment_vk;

                if let Some(stencil_layout_vk) = stencil_layout_vk {
                    stencil_layout_vk.p_next = depth_stencil_resolve_attachment_vk.p_next as *mut _;
                    depth_stencil_resolve_attachment_vk.p_next =
                        stencil_layout_vk as *const _ as *const _;
                }
            }

            *subpass_vk = ash::vk::SubpassDescription2 {
                input_attachment_count: input_attachments_vk.len() as u32,
                p_input_attachments: if input_attachments_vk.is_empty() {
                    ptr::null()
                } else {
                    input_attachments_vk.as_ptr()
                },
                color_attachment_count: color_attachments_vk.len() as u32,
                p_color_attachments: if color_attachments_vk.is_empty() {
                    ptr::null()
                } else {
                    color_attachments_vk.as_ptr()
                },
                p_resolve_attachments: if resolve_attachments_vk.is_empty() {
                    ptr::null()
                } else {
                    resolve_attachments_vk.as_ptr()
                },
                p_depth_stencil_attachment: depth_stencil_attachment_vk,
                ..*subpass_vk
            };

            if let Some(depth_stencil_resolve_vk) = depth_stencil_resolve_vk {
                *depth_stencil_resolve_vk = ash::vk::SubpassDescriptionDepthStencilResolve {
                    p_depth_stencil_resolve_attachment: depth_stencil_resolve_attachment_vk,
                    ..*depth_stencil_resolve_vk
                };

                depth_stencil_resolve_vk.p_next = subpass_vk.p_next;
                subpass_vk.p_next = depth_stencil_resolve_vk as *const _ as *const _;
            }
        }

        struct PerSubpassDependencyVk {
            memory_barrier_vk: Option<ash::vk::MemoryBarrier2>,
        }

        let (mut dependencies_vk, mut per_dependency_vk): (SmallVec<[_; 4]>, SmallVec<[_; 4]>) =
            dependencies
                .iter()
                .map(|dependency| {
                    let &SubpassDependency {
                        src_subpass,
                        dst_subpass,
                        src_stages,
                        dst_stages,
                        src_access,
                        dst_access,
                        dependency_flags,
                        view_offset,
                        _ne: _,
                    } = dependency;

                    (
                        ash::vk::SubpassDependency2 {
                            src_subpass: src_subpass.unwrap_or(ash::vk::SUBPASS_EXTERNAL),
                            dst_subpass: dst_subpass.unwrap_or(ash::vk::SUBPASS_EXTERNAL),
                            src_stage_mask: src_stages.into(),
                            dst_stage_mask: dst_stages.into(),
                            src_access_mask: src_access.into(),
                            dst_access_mask: dst_access.into(),
                            dependency_flags: dependency_flags.into(),
                            // VUID-VkSubpassDependency2-dependencyFlags-03092
                            view_offset,
                            ..Default::default()
                        },
                        PerSubpassDependencyVk {
                            memory_barrier_vk: device
                                .enabled_features()
                                .synchronization2
                                .then_some(ash::vk::MemoryBarrier2 {
                                    src_stage_mask: src_stages.into(),
                                    src_access_mask: src_access.into(),
                                    dst_stage_mask: dst_stages.into(),
                                    dst_access_mask: dst_access.into(),
                                    ..Default::default()
                                }),
                        },
                    )
                })
                .unzip();

        for (dependency_vk, per_dependency_vk) in
            dependencies_vk.iter_mut().zip(&mut per_dependency_vk)
        {
            let PerSubpassDependencyVk { memory_barrier_vk } = per_dependency_vk;

            if let Some(next) = memory_barrier_vk {
                next.p_next = dependency_vk.p_next;
                dependency_vk.p_next = next as *const _ as *const _;
            }
        }

        let create_info = ash::vk::RenderPassCreateInfo2 {
            flags: flags.into(),
            attachment_count: attachments_vk.len() as u32,
            p_attachments: if attachments_vk.is_empty() {
                ptr::null()
            } else {
                attachments_vk.as_ptr()
            },
            subpass_count: subpasses_vk.len() as u32,
            p_subpasses: if subpasses_vk.is_empty() {
                ptr::null()
            } else {
                subpasses_vk.as_ptr()
            },
            dependency_count: dependencies_vk.len() as u32,
            p_dependencies: if dependencies_vk.is_empty() {
                ptr::null()
            } else {
                dependencies_vk.as_ptr()
            },
            correlated_view_mask_count: correlated_view_masks.len() as u32,
            p_correlated_view_masks: correlated_view_masks.as_ptr(),
            ..Default::default()
        };

        Ok({
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();

            if device.api_version() >= Version::V1_2 {
                (fns.v1_2.create_render_pass2)(
                    device.handle(),
                    &create_info,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            } else {
                (fns.khr_create_renderpass2.create_render_pass2_khr)(
                    device.handle(),
                    &create_info,
                    ptr::null(),
                    output.as_mut_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;

            output.assume_init()
        })
    }

    pub(super) unsafe fn create_v1(
        device: &Device,
        create_info: &RenderPassCreateInfo,
    ) -> Result<ash::vk::RenderPass, VulkanError> {
        let &RenderPassCreateInfo {
            flags,
            ref attachments,
            ref subpasses,
            ref dependencies,
            ref correlated_view_masks,
            _ne: _,
        } = create_info;

        let attachments_vk = attachments
            .iter()
            .map(|attachment| {
                let &AttachmentDescription {
                    flags,
                    format,
                    samples,
                    load_op,
                    store_op,
                    initial_layout,
                    final_layout,
                    stencil_load_op,
                    stencil_store_op,
                    stencil_initial_layout: _,
                    stencil_final_layout: _,
                    _ne: _,
                } = attachment;

                ash::vk::AttachmentDescription {
                    flags: flags.into(),
                    format: format.into(),
                    samples: samples.into(),
                    load_op: load_op.into(),
                    store_op: store_op.into(),
                    stencil_load_op: stencil_load_op.unwrap_or(load_op).into(),
                    stencil_store_op: stencil_store_op.unwrap_or(store_op).into(),
                    initial_layout: initial_layout.into(),
                    final_layout: final_layout.into(),
                }
            })
            .collect::<SmallVec<[_; 4]>>();

        struct PerSubpassDescriptionVk {
            input_attachments_vk: SmallVec<[ash::vk::AttachmentReference; 4]>,
            color_attachments_vk: SmallVec<[ash::vk::AttachmentReference; 4]>,
            resolve_attachments_vk: SmallVec<[ash::vk::AttachmentReference; 4]>,
            depth_stencil_attachment_vk: ash::vk::AttachmentReference,
        }

        let (mut subpasses_vk, per_subpass_vk): (SmallVec<[_; 4]>, SmallVec<[_; 4]>) = subpasses
            .iter()
            .map(|subpass| {
                let &SubpassDescription {
                    flags,
                    view_mask: _,
                    ref input_attachments,
                    ref color_attachments,
                    ref color_resolve_attachments,
                    ref depth_stencil_attachment,
                    depth_stencil_resolve_attachment: _,
                    depth_resolve_mode: _,
                    stencil_resolve_mode: _,
                    ref preserve_attachments,
                    _ne: _,
                } = subpass;

                let input_attachments_vk = input_attachments
                    .iter()
                    .map(|input_attachment| {
                        if let Some(input_attachment) = input_attachment {
                            let &AttachmentReference {
                                attachment,
                                layout,
                                stencil_layout: _,
                                aspects: _,
                                _ne: _,
                            } = input_attachment;

                            ash::vk::AttachmentReference {
                                attachment,
                                layout: layout.into(),
                            }
                        } else {
                            ash::vk::AttachmentReference {
                                attachment: ash::vk::ATTACHMENT_UNUSED,
                                ..Default::default()
                            }
                        }
                    })
                    .collect();

                let color_attachments_vk = color_attachments
                    .iter()
                    .map(|color_attachment| {
                        if let Some(color_attachment) = color_attachment {
                            let &AttachmentReference {
                                attachment,
                                layout,
                                stencil_layout: _,
                                aspects: _,
                                _ne: _,
                            } = color_attachment;

                            ash::vk::AttachmentReference {
                                attachment,
                                layout: layout.into(),
                            }
                        } else {
                            ash::vk::AttachmentReference {
                                attachment: ash::vk::ATTACHMENT_UNUSED,
                                ..Default::default()
                            }
                        }
                    })
                    .collect();

                let resolve_attachments_vk = color_resolve_attachments
                    .iter()
                    .map(|color_resolve_attachment| {
                        if let Some(color_resolve_attachment) = color_resolve_attachment {
                            let &AttachmentReference {
                                attachment,
                                layout,
                                stencil_layout: _,
                                aspects: _,
                                _ne: _,
                            } = color_resolve_attachment;

                            ash::vk::AttachmentReference {
                                attachment,
                                layout: layout.into(),
                            }
                        } else {
                            ash::vk::AttachmentReference {
                                attachment: ash::vk::ATTACHMENT_UNUSED,
                                ..Default::default()
                            }
                        }
                    })
                    .collect();

                let depth_stencil_attachment_vk =
                    if let Some(depth_stencil_attachment) = depth_stencil_attachment {
                        let &AttachmentReference {
                            attachment,
                            layout,
                            stencil_layout: _,
                            aspects: _,
                            _ne: _,
                        } = depth_stencil_attachment;

                        ash::vk::AttachmentReference {
                            attachment,
                            layout: layout.into(),
                        }
                    } else {
                        ash::vk::AttachmentReference {
                            attachment: ash::vk::ATTACHMENT_UNUSED,
                            ..Default::default()
                        }
                    };

                (
                    ash::vk::SubpassDescription {
                        flags: flags.into(),
                        pipeline_bind_point: ash::vk::PipelineBindPoint::GRAPHICS,
                        input_attachment_count: 0,
                        p_input_attachments: ptr::null(),
                        color_attachment_count: 0,
                        p_color_attachments: ptr::null(),
                        p_resolve_attachments: ptr::null(),
                        p_depth_stencil_attachment: ptr::null(),
                        preserve_attachment_count: preserve_attachments.len() as u32,
                        p_preserve_attachments: if preserve_attachments.is_empty() {
                            ptr::null()
                        } else {
                            preserve_attachments.as_ptr()
                        },
                    },
                    PerSubpassDescriptionVk {
                        input_attachments_vk,
                        color_attachments_vk,
                        resolve_attachments_vk,
                        depth_stencil_attachment_vk,
                    },
                )
            })
            .unzip();

        for (subpass_vk, per_subpass_vk) in subpasses_vk.iter_mut().zip(&per_subpass_vk) {
            let PerSubpassDescriptionVk {
                input_attachments_vk,
                color_attachments_vk,
                resolve_attachments_vk,
                depth_stencil_attachment_vk,
            } = per_subpass_vk;

            *subpass_vk = ash::vk::SubpassDescription {
                input_attachment_count: input_attachments_vk.len() as u32,
                p_input_attachments: if input_attachments_vk.is_empty() {
                    ptr::null()
                } else {
                    input_attachments_vk.as_ptr()
                },
                color_attachment_count: color_attachments_vk.len() as u32,
                p_color_attachments: if color_attachments_vk.is_empty() {
                    ptr::null()
                } else {
                    color_attachments_vk.as_ptr()
                },
                p_resolve_attachments: if resolve_attachments_vk.is_empty() {
                    ptr::null()
                } else {
                    resolve_attachments_vk.as_ptr()
                },
                p_depth_stencil_attachment: depth_stencil_attachment_vk,
                ..*subpass_vk
            };
        }

        let dependencies_vk = dependencies
            .iter()
            .map(|dependency| {
                let &SubpassDependency {
                    src_subpass,
                    dst_subpass,
                    src_stages,
                    dst_stages,
                    src_access,
                    dst_access,
                    dependency_flags,
                    view_offset: _,
                    _ne: _,
                } = dependency;

                ash::vk::SubpassDependency {
                    src_subpass: src_subpass.unwrap_or(ash::vk::SUBPASS_EXTERNAL),
                    dst_subpass: dst_subpass.unwrap_or(ash::vk::SUBPASS_EXTERNAL),
                    src_stage_mask: src_stages.into(),
                    dst_stage_mask: dst_stages.into(),
                    src_access_mask: src_access.into(),
                    dst_access_mask: dst_access.into(),
                    dependency_flags: dependency_flags.into(),
                }
            })
            .collect::<SmallVec<[_; 4]>>();

        /* Create */

        let mut create_info_vk = ash::vk::RenderPassCreateInfo {
            flags: flags.into(),
            attachment_count: attachments_vk.len() as u32,
            p_attachments: if attachments_vk.is_empty() {
                ptr::null()
            } else {
                attachments_vk.as_ptr()
            },
            subpass_count: subpasses_vk.len() as u32,
            p_subpasses: if subpasses_vk.is_empty() {
                ptr::null()
            } else {
                subpasses_vk.as_ptr()
            },
            dependency_count: dependencies_vk.len() as u32,
            p_dependencies: if dependencies_vk.is_empty() {
                ptr::null()
            } else {
                dependencies_vk.as_ptr()
            },
            ..Default::default()
        };

        /* Input attachment aspect */

        let input_attachment_aspect_references_vk: SmallVec<[_; 8]>;
        let mut input_attachment_aspect_create_info_vk = None;

        if device.api_version() >= Version::V1_1 || device.enabled_extensions().khr_maintenance2 {
            input_attachment_aspect_references_vk = subpasses
                .iter()
                .enumerate()
                .flat_map(|(subpass_num, subpass)| {
                    subpass.input_attachments.iter().enumerate().flat_map(
                        move |(atch_num, input_attachment)| {
                            input_attachment.as_ref().map(|input_attachment| {
                                let &AttachmentReference {
                                    attachment: _,
                                    layout: _,
                                    stencil_layout: _,
                                    aspects,
                                    _ne,
                                } = input_attachment;

                                ash::vk::InputAttachmentAspectReference {
                                    subpass: subpass_num as u32,
                                    input_attachment_index: atch_num as u32,
                                    aspect_mask: aspects.into(),
                                }
                            })
                        },
                    )
                })
                .collect();

            if !input_attachment_aspect_references_vk.is_empty() {
                let next = input_attachment_aspect_create_info_vk.insert(
                    ash::vk::RenderPassInputAttachmentAspectCreateInfo {
                        aspect_reference_count: input_attachment_aspect_references_vk.len() as u32,
                        p_aspect_references: input_attachment_aspect_references_vk.as_ptr(),
                        ..Default::default()
                    },
                );

                next.p_next = create_info_vk.p_next;
                create_info_vk.p_next = next as *const _ as *const _;
            }
        }

        /* Multiview */

        let mut multiview_create_info_vk = None;
        let multiview_view_masks_vk: SmallVec<[_; 4]>;
        let multiview_view_offsets_vk: SmallVec<[_; 4]>;

        let is_multiview = subpasses[0].view_mask != 0;

        if is_multiview {
            multiview_view_masks_vk = subpasses.iter().map(|subpass| subpass.view_mask).collect();
            multiview_view_offsets_vk = dependencies
                .iter()
                .map(|dependency| dependency.view_offset)
                .collect();

            debug_assert!(multiview_view_masks_vk.len() == subpasses.len());
            debug_assert!(multiview_view_offsets_vk.len() == dependencies.len());

            let next = multiview_create_info_vk.insert(ash::vk::RenderPassMultiviewCreateInfo {
                subpass_count: multiview_view_masks_vk.len() as u32,
                p_view_masks: multiview_view_masks_vk.as_ptr(),
                dependency_count: multiview_view_offsets_vk.len() as u32,
                p_view_offsets: multiview_view_offsets_vk.as_ptr(),
                correlation_mask_count: correlated_view_masks.len() as u32,
                p_correlation_masks: correlated_view_masks.as_ptr(),
                ..Default::default()
            });

            next.p_next = create_info_vk.p_next;
            create_info_vk.p_next = next as *const _ as *const _;
        }

        Ok({
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_render_pass)(
                device.handle(),
                &create_info_vk,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        })
    }
}
