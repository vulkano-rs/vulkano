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
    image::ImageAspects,
    render_pass::{
        InputAttachmentReference, ResolvableAttachmentReference, ResolveAttachmentReference,
        SubpassDependency, SubpassDescription,
    },
    RuntimeError, Version, VulkanObject,
};
use smallvec::SmallVec;
use std::{mem::MaybeUninit, ptr};

impl RenderPass {
    pub(super) unsafe fn create_v2(
        device: &Device,
        create_info: &RenderPassCreateInfo,
    ) -> Result<ash::vk::RenderPass, RuntimeError> {
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

                    let aspects = format.unwrap().aspects();

                    let (initial_layout_vk, final_layout_vk, stencil_layout_vk) = if aspects
                        .contains(ImageAspects::DEPTH | ImageAspects::STENCIL)
                        && (initial_layout != stencil_initial_layout
                            || final_layout != stencil_final_layout)
                    {
                        (
                            initial_layout,
                            final_layout,
                            Some(ash::vk::AttachmentDescriptionStencilLayout {
                                stencil_initial_layout: stencil_initial_layout.into(),
                                stencil_final_layout: stencil_final_layout.into(),
                                ..Default::default()
                            }),
                        )
                    } else if aspects.intersects(ImageAspects::STENCIL) {
                        (stencil_initial_layout, stencil_final_layout, None)
                    } else {
                        (initial_layout, final_layout, None)
                    };

                    (
                        ash::vk::AttachmentDescription2 {
                            flags: flags.into(),
                            format: format.map_or(ash::vk::Format::UNDEFINED, |f| f.into()),
                            samples: samples.into(),
                            load_op: load_op.into(),
                            store_op: store_op.into(),
                            stencil_load_op: stencil_load_op.into(),
                            stencil_store_op: stencil_store_op.into(),
                            initial_layout: initial_layout_vk.into(),
                            final_layout: final_layout_vk.into(),
                            ..Default::default()
                        },
                        PerAttachment { stencil_layout_vk },
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

        struct PerSubpass {
            input_attachments_vk: SmallVec<[ash::vk::AttachmentReference2; 4]>,
            color_attachments_vk: SmallVec<[ash::vk::AttachmentReference2; 4]>,
            resolve_attachments_vk: SmallVec<[ash::vk::AttachmentReference2; 4]>,
            depth_stencil_attachment_vk: DepthStencilAttachment,
            depth_stencil_resolve_vk: Option<ash::vk::SubpassDescriptionDepthStencilResolve>,
            depth_stencil_resolve_attachment_vk: DepthStencilAttachment,
        }

        struct DepthStencilAttachment {
            attachment_reference_vk: ash::vk::AttachmentReference2,
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
                        ref depth_attachment,
                        ref stencil_attachment,
                        ref preserve_attachments,
                        _ne: _,
                    } = subpass;

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
                        PerSubpass {
                            input_attachments_vk: input_attachments
                                .iter()
                                .map(|input_attachment| {
                                    if let Some(input_attachment) = input_attachment {
                                        let &InputAttachmentReference {
                                            attachment_ref:
                                                AttachmentReference {
                                                    attachment,
                                                    layout,
                                                    _ne: _,
                                                },
                                            aspects,
                                        } = input_attachment;

                                        ash::vk::AttachmentReference2 {
                                            attachment,
                                            layout: layout.into(),
                                            aspect_mask: aspects.into(),
                                            ..Default::default()
                                        }
                                    } else {
                                        ash::vk::AttachmentReference2 {
                                            attachment: ash::vk::ATTACHMENT_UNUSED,
                                            ..Default::default()
                                        }
                                    }
                                })
                                .collect(),
                            color_attachments_vk: color_attachments
                                .iter()
                                .map(|color_attachment| {
                                    if let Some(color_attachment) = color_attachment {
                                        let &ResolvableAttachmentReference {
                                            attachment_ref:
                                                AttachmentReference {
                                                    attachment,
                                                    layout,
                                                    _ne: _,
                                                },
                                            resolve: _,
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
                                .collect(),
                            resolve_attachments_vk: color_attachments
                                .iter()
                                .map(|color_attachment| {
                                    if let Some(&ResolvableAttachmentReference {
                                        attachment_ref: _,
                                        resolve:
                                            Some(ResolveAttachmentReference {
                                                attachment_ref:
                                                    AttachmentReference {
                                                        attachment,
                                                        layout,
                                                        _ne: _,
                                                    },
                                                mode: _,
                                            }),
                                    }) = color_attachment.as_ref()
                                    {
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
                                .collect(),
                            depth_stencil_attachment_vk: {
                                match (depth_attachment.as_ref(), stencil_attachment.as_ref()) {
                                    (
                                        Some(&ResolvableAttachmentReference {
                                            attachment_ref:
                                                AttachmentReference {
                                                    attachment,
                                                    layout: depth_layout,
                                                    _ne: _,
                                                },
                                            resolve: _,
                                        }),
                                        Some(&ResolvableAttachmentReference {
                                            attachment_ref:
                                                AttachmentReference {
                                                    attachment: _,
                                                    layout: stencil_layout,
                                                    _ne: _,
                                                },
                                            resolve: _,
                                        }),
                                    ) => DepthStencilAttachment {
                                        attachment_reference_vk: ash::vk::AttachmentReference2 {
                                            attachment,
                                            layout: depth_layout.into(),
                                            ..Default::default()
                                        },
                                        stencil_layout_vk: (depth_layout != stencil_layout)
                                            .then_some(ash::vk::AttachmentReferenceStencilLayout {
                                                stencil_layout: stencil_layout.into(),
                                                ..Default::default()
                                            }),
                                    },
                                    (
                                        Some(&ResolvableAttachmentReference {
                                            attachment_ref:
                                                AttachmentReference {
                                                    attachment,
                                                    layout,
                                                    _ne: _,
                                                },
                                            resolve: _,
                                        }),
                                        None,
                                    )
                                    | (
                                        None,
                                        Some(&ResolvableAttachmentReference {
                                            attachment_ref:
                                                AttachmentReference {
                                                    attachment,
                                                    layout,
                                                    _ne: _,
                                                },
                                            resolve: _,
                                        }),
                                    ) => DepthStencilAttachment {
                                        attachment_reference_vk: ash::vk::AttachmentReference2 {
                                            attachment,
                                            layout: layout.into(),
                                            ..Default::default()
                                        },
                                        stencil_layout_vk: None,
                                    },
                                    _ => DepthStencilAttachment {
                                        attachment_reference_vk: ash::vk::AttachmentReference2 {
                                            attachment: ash::vk::ATTACHMENT_UNUSED,
                                            ..Default::default()
                                        },
                                        stencil_layout_vk: None,
                                    },
                                }
                            },
                            depth_stencil_resolve_vk: {
                                let depth_resolve_mode = depth_attachment
                                    .as_ref()
                                    .and_then(|attachment| attachment.resolve.as_ref())
                                    .map(|resolve| resolve.mode);
                                let stencil_resolve_mode = stencil_attachment
                                    .as_ref()
                                    .and_then(|attachment| attachment.resolve.as_ref())
                                    .map(|resolve| resolve.mode);

                                // VUID-VkSubpassDescriptionDepthStencilResolve-pDepthStencilResolveAttachment-03178
                                (depth_resolve_mode.is_some() || stencil_resolve_mode.is_some())
                                    .then_some(ash::vk::SubpassDescriptionDepthStencilResolve {
                                        depth_resolve_mode: depth_resolve_mode
                                            .map_or(ash::vk::ResolveModeFlags::NONE, Into::into),
                                        stencil_resolve_mode: stencil_resolve_mode
                                            .map_or(ash::vk::ResolveModeFlags::NONE, Into::into),
                                        p_depth_stencil_resolve_attachment: ptr::null(),
                                        ..Default::default()
                                    })
                            },
                            depth_stencil_resolve_attachment_vk: {
                                match (depth_attachment.as_ref(), stencil_attachment.as_ref()) {
                                    (
                                        Some(&ResolvableAttachmentReference {
                                            attachment_ref: _,
                                            resolve:
                                                Some(ResolveAttachmentReference {
                                                    attachment_ref:
                                                        AttachmentReference {
                                                            attachment,
                                                            layout: depth_layout,
                                                            _ne: _,
                                                        },
                                                    mode: _,
                                                }),
                                        }),
                                        Some(&ResolvableAttachmentReference {
                                            attachment_ref: _,
                                            resolve:
                                                Some(ResolveAttachmentReference {
                                                    attachment_ref:
                                                        AttachmentReference {
                                                            attachment: _,
                                                            layout: stencil_layout,
                                                            _ne: _,
                                                        },
                                                    mode: _,
                                                }),
                                        }),
                                    ) => DepthStencilAttachment {
                                        attachment_reference_vk: ash::vk::AttachmentReference2 {
                                            attachment,
                                            layout: depth_layout.into(),
                                            ..Default::default()
                                        },
                                        stencil_layout_vk: (depth_layout != stencil_layout)
                                            .then_some(ash::vk::AttachmentReferenceStencilLayout {
                                                stencil_layout: stencil_layout.into(),
                                                ..Default::default()
                                            }),
                                    },
                                    (
                                        Some(&ResolvableAttachmentReference {
                                            attachment_ref: _,
                                            resolve:
                                                Some(ResolveAttachmentReference {
                                                    attachment_ref:
                                                        AttachmentReference {
                                                            attachment,
                                                            layout,
                                                            _ne: _,
                                                        },
                                                    mode: _,
                                                }),
                                        }),
                                        None
                                        | Some(&ResolvableAttachmentReference {
                                            attachment_ref: _,
                                            resolve: None,
                                        }),
                                    )
                                    | (
                                        None
                                        | Some(&ResolvableAttachmentReference {
                                            attachment_ref: _,
                                            resolve: None,
                                        }),
                                        Some(&ResolvableAttachmentReference {
                                            attachment_ref: _,
                                            resolve:
                                                Some(ResolveAttachmentReference {
                                                    attachment_ref:
                                                        AttachmentReference {
                                                            attachment,
                                                            layout,
                                                            _ne: _,
                                                        },
                                                    mode: _,
                                                }),
                                        }),
                                    ) => DepthStencilAttachment {
                                        attachment_reference_vk: ash::vk::AttachmentReference2 {
                                            attachment,
                                            layout: layout.into(),
                                            ..Default::default()
                                        },
                                        stencil_layout_vk: None,
                                    },
                                    _ => DepthStencilAttachment {
                                        attachment_reference_vk: ash::vk::AttachmentReference2 {
                                            attachment: ash::vk::ATTACHMENT_UNUSED,
                                            ..Default::default()
                                        },
                                        stencil_layout_vk: None,
                                    },
                                }
                            },
                        },
                    )
                })
                .unzip();

        for (subpass_vk, per_subpass_vk) in subpasses_vk.iter_mut().zip(per_subpass_vk.iter_mut()) {
            let PerSubpass {
                input_attachments_vk,
                color_attachments_vk,
                resolve_attachments_vk,
                depth_stencil_attachment_vk,
                depth_stencil_resolve_attachment_vk,
                depth_stencil_resolve_vk,
            } = per_subpass_vk;

            if let Some(next) = &mut depth_stencil_attachment_vk.stencil_layout_vk {
                next.p_next = depth_stencil_attachment_vk.attachment_reference_vk.p_next as *mut _;
                depth_stencil_attachment_vk.attachment_reference_vk.p_next =
                    next as *const _ as *const _;
            }

            *subpass_vk = ash::vk::SubpassDescription2 {
                input_attachment_count: input_attachments_vk.len() as u32,
                p_input_attachments: input_attachments_vk.as_ptr(),
                color_attachment_count: color_attachments_vk.len() as u32,
                p_color_attachments: color_attachments_vk.as_ptr(),
                p_resolve_attachments: resolve_attachments_vk.as_ptr(),
                p_depth_stencil_attachment: &depth_stencil_attachment_vk.attachment_reference_vk,
                ..*subpass_vk
            };

            if let Some(next) = depth_stencil_resolve_vk {
                *next = ash::vk::SubpassDescriptionDepthStencilResolve {
                    p_depth_stencil_resolve_attachment: &depth_stencil_resolve_attachment_vk
                        .attachment_reference_vk,
                    ..*next
                };

                next.p_next = subpass_vk.p_next;
                subpass_vk.p_next = next as *const _ as *const _;

                if let Some(next) = &mut depth_stencil_resolve_attachment_vk.stencil_layout_vk {
                    next.p_next = depth_stencil_resolve_attachment_vk
                        .attachment_reference_vk
                        .p_next as *mut _;
                    depth_stencil_resolve_attachment_vk
                        .attachment_reference_vk
                        .p_next = next as *const _ as *const _;
                }
            }
        }

        struct PerDependency {
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
                        PerDependency {
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
            let PerDependency { memory_barrier_vk } = per_dependency_vk;

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
            .map_err(RuntimeError::from)?;

            output.assume_init()
        })
    }

    pub(super) unsafe fn create_v1(
        device: &Device,
        create_info: &RenderPassCreateInfo,
    ) -> Result<ash::vk::RenderPass, RuntimeError> {
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
                    stencil_initial_layout,
                    stencil_final_layout,
                    _ne: _,
                } = attachment;

                let aspects = format.unwrap().aspects();

                let (initial_layout_vk, final_layout_vk) = if aspects
                    .intersects(ImageAspects::STENCIL)
                    && !aspects.intersects(ImageAspects::DEPTH)
                {
                    (stencil_initial_layout, stencil_final_layout)
                } else {
                    (initial_layout, final_layout)
                };

                ash::vk::AttachmentDescription {
                    flags: flags.into(),
                    format: format.map_or(ash::vk::Format::UNDEFINED, |f| f.into()),
                    samples: samples.into(),
                    load_op: load_op.into(),
                    store_op: store_op.into(),
                    stencil_load_op: stencil_load_op.into(),
                    stencil_store_op: stencil_store_op.into(),
                    initial_layout: initial_layout_vk.into(),
                    final_layout: final_layout_vk.into(),
                }
            })
            .collect::<SmallVec<[_; 4]>>();

        struct PerSubpass {
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
                    ref depth_attachment,
                    ref stencil_attachment,
                    ref preserve_attachments,
                    _ne: _,
                } = subpass;

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
                    PerSubpass {
                        input_attachments_vk: input_attachments
                            .iter()
                            .map(|input_attachment| {
                                if let Some(input_attachment) = input_attachment {
                                    let &InputAttachmentReference {
                                        attachment_ref:
                                            AttachmentReference {
                                                attachment,
                                                layout,
                                                _ne: _,
                                            },
                                        aspects: _,
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
                            .collect(),
                        color_attachments_vk: color_attachments
                            .iter()
                            .map(|color_attachment| {
                                if let Some(color_attachment) = color_attachment {
                                    let &ResolvableAttachmentReference {
                                        attachment_ref:
                                            AttachmentReference {
                                                attachment,
                                                layout,
                                                _ne: _,
                                            },
                                        resolve: _,
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
                            .collect(),
                        resolve_attachments_vk: color_attachments
                            .iter()
                            .map(|color_attachment| {
                                if let Some(&ResolvableAttachmentReference {
                                    attachment_ref: _,
                                    resolve:
                                        Some(ResolveAttachmentReference {
                                            attachment_ref:
                                                AttachmentReference {
                                                    attachment,
                                                    layout,
                                                    _ne: _,
                                                },
                                            mode: _,
                                        }),
                                }) = color_attachment.as_ref()
                                {
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
                            .collect(),
                        depth_stencil_attachment_vk: if let Some(depth_stencil_attachment) =
                            depth_attachment.as_ref().or(stencil_attachment.as_ref())
                        {
                            let &ResolvableAttachmentReference {
                                attachment_ref:
                                    AttachmentReference {
                                        attachment,
                                        layout,
                                        _ne: _,
                                    },
                                resolve: _,
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
                        },
                    },
                )
            })
            .unzip();

        for (subpass_vk, per_subpass_vk) in subpasses_vk.iter_mut().zip(&per_subpass_vk) {
            let PerSubpass {
                input_attachments_vk,
                color_attachments_vk,
                resolve_attachments_vk,
                depth_stencil_attachment_vk,
            } = per_subpass_vk;

            *subpass_vk = ash::vk::SubpassDescription {
                input_attachment_count: input_attachments_vk.len() as u32,
                p_input_attachments: input_attachments_vk.as_ptr(),
                color_attachment_count: color_attachments_vk.len() as u32,
                p_color_attachments: color_attachments_vk.as_ptr(),
                p_resolve_attachments: resolve_attachments_vk.as_ptr(),
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
                                let &InputAttachmentReference {
                                    attachment_ref: _,
                                    aspects,
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
            .map_err(RuntimeError::from)?;
            output.assume_init()
        })
    }
}
