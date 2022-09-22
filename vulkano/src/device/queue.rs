// Copyright (c) 2022 The Vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{Device, DeviceOwned};
use crate::{
    buffer::BufferAccess,
    command_buffer::{SemaphoreSubmitInfo, SubmitInfo},
    image::ImageAccess,
    instance::debug::DebugUtilsLabel,
    macros::vulkan_bitflags,
    memory::{
        BindSparseInfo, SparseBufferMemoryBind, SparseImageMemoryBind, SparseImageOpaqueMemoryBind,
    },
    swapchain::{PresentInfo, SwapchainPresentInfo},
    sync::{Fence, PipelineStage},
    OomError, RequirementNotMet, RequiresOneOf, SynchronizedVulkanObject, Version, VulkanError,
    VulkanObject,
};
use parking_lot::{Mutex, MutexGuard};
use smallvec::SmallVec;
use std::{
    error::Error,
    ffi::CString,
    fmt::{Display, Error as FmtError, Formatter},
    hash::{Hash, Hasher},
    ptr,
    sync::Arc,
};

/// Represents a queue where commands can be submitted.
// TODO: should use internal synchronization?
#[derive(Debug)]
pub struct Queue {
    handle: Mutex<ash::vk::Queue>,
    device: Arc<Device>,
    queue_family_index: u32,
    id: u32, // id within family
}

impl Queue {
    #[inline]
    pub(super) fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::Queue,
        queue_family_index: u32,
        id: u32,
    ) -> Arc<Self> {
        Arc::new(Queue {
            handle: Mutex::new(handle),
            device,
            queue_family_index,
            id,
        })
    }

    /// Returns the device that this queue belongs to.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the index of the queue family that this queue belongs to.
    #[inline]
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    /// Returns the index of this queue within its queue family.
    #[inline]
    pub fn id_within_family(&self) -> u32 {
        self.id
    }

    /// Locks the queue, making it possible to perform operations on the queue, such as submissions.
    #[inline]
    pub fn lock(&self) -> QueueGuard {
        QueueGuard {
            queue: self,
            handle: self.handle.lock(),
        }
    }
}

unsafe impl SynchronizedVulkanObject for Queue {
    type Object = ash::vk::Queue;

    #[inline]
    fn internal_object_guard(&self) -> MutexGuard<Self::Object> {
        self.handle.lock()
    }
}

unsafe impl DeviceOwned for Queue {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl PartialEq for Queue {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.queue_family_index == other.queue_family_index
            && self.device == other.device
    }
}

impl Eq for Queue {}

impl Hash for Queue {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.queue_family_index.hash(state);
        self.device.hash(state);
    }
}

pub struct QueueGuard<'a> {
    queue: &'a Queue,
    handle: MutexGuard<'a, ash::vk::Queue>,
}

impl<'a> QueueGuard<'a> {
    /// Waits until all work on this queue has finished.
    ///
    /// Just like [`Device::wait_idle`], you shouldn't have to call this function in a typical
    /// program.
    #[inline]
    pub fn wait_idle(&mut self) -> Result<(), OomError> {
        unsafe {
            let fns = self.queue.device.fns();
            (fns.v1_0.queue_wait_idle)(*self.handle)
                .result()
                .map_err(VulkanError::from)?;
            Ok(())
        }
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub(crate) unsafe fn bind_sparse_unchecked(
        &mut self,
        bind_infos: impl IntoIterator<Item = BindSparseInfo>,
        fence: Option<Arc<Fence>>,
    ) -> Result<(), VulkanError> {
        let bind_infos = bind_infos.into_iter();

        #[allow(unused)]
        struct PerBindSparseInfo {
            wait_semaphores_vk: SmallVec<[ash::vk::Semaphore; 4]>,
            buffer_bind_infos_vk: SmallVec<[ash::vk::SparseBufferMemoryBindInfo; 4]>,
            buffer_binds_vk: SmallVec<[SmallVec<[ash::vk::SparseMemoryBind; 4]>; 4]>,
            image_opaque_bind_infos_vk: SmallVec<[ash::vk::SparseImageOpaqueMemoryBindInfo; 4]>,
            image_opaque_binds_vk: SmallVec<[SmallVec<[ash::vk::SparseMemoryBind; 4]>; 4]>,
            image_bind_infos_vk: SmallVec<[ash::vk::SparseImageMemoryBindInfo; 4]>,
            image_binds_vk: SmallVec<[SmallVec<[ash::vk::SparseImageMemoryBind; 4]>; 4]>,
            signal_semaphores_vk: SmallVec<[ash::vk::Semaphore; 4]>,
        }

        let (mut bind_infos_vk, mut per_bind_vk): (SmallVec<[_; 4]>, SmallVec<[_; 4]>) = bind_infos
            .map(|bind_info| {
                let &BindSparseInfo {
                    ref wait_semaphores,
                    ref buffer_binds,
                    ref image_opaque_binds,
                    ref image_binds,
                    ref signal_semaphores,
                    _ne: _,
                } = &bind_info;

                let wait_semaphores_vk: SmallVec<[_; 4]> = wait_semaphores
                    .iter()
                    .map(|semaphore| semaphore.internal_object())
                    .collect();

                let (buffer_bind_infos_vk, buffer_binds_vk): (SmallVec<[_; 4]>, SmallVec<[_; 4]>) =
                    buffer_binds
                        .iter()
                        .map(|(buffer, memory_binds)| {
                            (
                                ash::vk::SparseBufferMemoryBindInfo {
                                    buffer: buffer.inner().buffer.internal_object(),
                                    bind_count: 0,
                                    p_binds: ptr::null(),
                                },
                                memory_binds
                                    .iter()
                                    .map(|memory_bind| {
                                        let &SparseBufferMemoryBind {
                                            resource_offset,
                                            size,
                                            ref memory,
                                        } = memory_bind;

                                        let (memory, memory_offset) = memory.as_ref().map_or_else(
                                            Default::default,
                                            |(memory, memory_offset)| {
                                                (memory.internal_object(), *memory_offset)
                                            },
                                        );

                                        ash::vk::SparseMemoryBind {
                                            resource_offset,
                                            size,
                                            memory,
                                            memory_offset,
                                            flags: ash::vk::SparseMemoryBindFlags::empty(),
                                        }
                                    })
                                    .collect::<SmallVec<[_; 4]>>(),
                            )
                        })
                        .unzip();

                let (image_opaque_bind_infos_vk, image_opaque_binds_vk): (
                    SmallVec<[_; 4]>,
                    SmallVec<[_; 4]>,
                ) = image_opaque_binds
                    .iter()
                    .map(|(image, memory_binds)| {
                        (
                            ash::vk::SparseImageOpaqueMemoryBindInfo {
                                image: image.inner().image.internal_object(),
                                bind_count: 0,
                                p_binds: ptr::null(),
                            },
                            memory_binds
                                .iter()
                                .map(|memory_bind| {
                                    let &SparseImageOpaqueMemoryBind {
                                        resource_offset,
                                        size,
                                        ref memory,
                                        metadata,
                                    } = memory_bind;

                                    let (memory, memory_offset) = memory.as_ref().map_or_else(
                                        Default::default,
                                        |(memory, memory_offset)| {
                                            (memory.internal_object(), *memory_offset)
                                        },
                                    );

                                    ash::vk::SparseMemoryBind {
                                        resource_offset,
                                        size,
                                        memory,
                                        memory_offset,
                                        flags: if metadata {
                                            ash::vk::SparseMemoryBindFlags::METADATA
                                        } else {
                                            ash::vk::SparseMemoryBindFlags::empty()
                                        },
                                    }
                                })
                                .collect::<SmallVec<[_; 4]>>(),
                        )
                    })
                    .unzip();

                let (image_bind_infos_vk, image_binds_vk): (SmallVec<[_; 4]>, SmallVec<[_; 4]>) =
                    image_binds
                        .iter()
                        .map(|(image, memory_binds)| {
                            (
                                ash::vk::SparseImageMemoryBindInfo {
                                    image: image.inner().image.internal_object(),
                                    bind_count: 0,
                                    p_binds: ptr::null(),
                                },
                                memory_binds
                                    .iter()
                                    .map(|memory_bind| {
                                        let &SparseImageMemoryBind {
                                            aspects,
                                            mip_level,
                                            array_layer,
                                            offset,
                                            extent,
                                            ref memory,
                                        } = memory_bind;

                                        let (memory, memory_offset) = memory.as_ref().map_or_else(
                                            Default::default,
                                            |(memory, memory_offset)| {
                                                (memory.internal_object(), *memory_offset)
                                            },
                                        );

                                        ash::vk::SparseImageMemoryBind {
                                            subresource: ash::vk::ImageSubresource {
                                                aspect_mask: aspects.into(),
                                                mip_level,
                                                array_layer,
                                            },
                                            offset: ash::vk::Offset3D {
                                                x: offset[0] as i32,
                                                y: offset[1] as i32,
                                                z: offset[2] as i32,
                                            },
                                            extent: ash::vk::Extent3D {
                                                width: extent[0],
                                                height: extent[1],
                                                depth: extent[2],
                                            },
                                            memory,
                                            memory_offset,
                                            flags: ash::vk::SparseMemoryBindFlags::empty(),
                                        }
                                    })
                                    .collect::<SmallVec<[_; 4]>>(),
                            )
                        })
                        .unzip();

                let signal_semaphores_vk: SmallVec<[_; 4]> = signal_semaphores
                    .iter()
                    .map(|semaphore| semaphore.internal_object())
                    .collect();

                (
                    ash::vk::BindSparseInfo::default(),
                    PerBindSparseInfo {
                        wait_semaphores_vk,
                        buffer_bind_infos_vk,
                        buffer_binds_vk,
                        image_opaque_bind_infos_vk,
                        image_opaque_binds_vk,
                        image_bind_infos_vk,
                        image_binds_vk,
                        signal_semaphores_vk,
                    },
                )
            })
            .unzip();

        for (
            bind_info_vk,
            PerBindSparseInfo {
                wait_semaphores_vk,
                buffer_bind_infos_vk,
                buffer_binds_vk,
                image_opaque_bind_infos_vk,
                image_opaque_binds_vk,
                image_bind_infos_vk,
                image_binds_vk,
                signal_semaphores_vk,
            },
        ) in (bind_infos_vk.iter_mut()).zip(per_bind_vk.iter_mut())
        {
            for (buffer_bind_infos_vk, buffer_binds_vk) in
                (buffer_bind_infos_vk.iter_mut()).zip(buffer_binds_vk.iter())
            {
                *buffer_bind_infos_vk = ash::vk::SparseBufferMemoryBindInfo {
                    bind_count: buffer_binds_vk.len() as u32,
                    p_binds: buffer_binds_vk.as_ptr(),
                    ..*buffer_bind_infos_vk
                };
            }

            for (image_opaque_bind_infos_vk, image_opaque_binds_vk) in
                (image_opaque_bind_infos_vk.iter_mut()).zip(image_opaque_binds_vk.iter())
            {
                *image_opaque_bind_infos_vk = ash::vk::SparseImageOpaqueMemoryBindInfo {
                    bind_count: image_opaque_binds_vk.len() as u32,
                    p_binds: image_opaque_binds_vk.as_ptr(),
                    ..*image_opaque_bind_infos_vk
                };
            }

            for (image_bind_infos_vk, image_binds_vk) in
                (image_bind_infos_vk.iter_mut()).zip(image_binds_vk.iter())
            {
                *image_bind_infos_vk = ash::vk::SparseImageMemoryBindInfo {
                    bind_count: image_binds_vk.len() as u32,
                    p_binds: image_binds_vk.as_ptr(),
                    ..*image_bind_infos_vk
                };
            }

            *bind_info_vk = ash::vk::BindSparseInfo {
                wait_semaphore_count: wait_semaphores_vk.len() as u32,
                p_wait_semaphores: wait_semaphores_vk.as_ptr(),
                buffer_bind_count: buffer_bind_infos_vk.len() as u32,
                p_buffer_binds: buffer_bind_infos_vk.as_ptr(),
                image_opaque_bind_count: image_opaque_bind_infos_vk.len() as u32,
                p_image_opaque_binds: image_opaque_bind_infos_vk.as_ptr(),
                image_bind_count: image_bind_infos_vk.len() as u32,
                p_image_binds: image_bind_infos_vk.as_ptr(),
                signal_semaphore_count: signal_semaphores_vk.len() as u32,
                p_signal_semaphores: signal_semaphores_vk.as_ptr(),
                ..*bind_info_vk
            }
        }

        let fns = self.queue.device.fns();
        (fns.v1_0.queue_bind_sparse)(
            *self.handle,
            bind_infos_vk.len() as u32,
            bind_infos_vk.as_ptr(),
            fence.map_or_else(Default::default, |fence| fence.internal_object()),
        )
        .result()
        .map_err(VulkanError::from)?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub(crate) unsafe fn present_unchecked(
        &mut self,
        present_info: PresentInfo,
    ) -> impl ExactSizeIterator<Item = Result<(), VulkanError>> {
        let &PresentInfo {
            ref wait_semaphores,
            ref swapchain_infos,
            _ne: _,
        } = &present_info;

        let wait_semaphores_vk: SmallVec<[_; 4]> = wait_semaphores
            .iter()
            .map(|semaphore| semaphore.internal_object())
            .collect();

        let mut swapchains_vk: SmallVec<[_; 4]> = SmallVec::with_capacity(swapchain_infos.len());
        let mut image_indices_vk: SmallVec<[_; 4]> = SmallVec::with_capacity(swapchain_infos.len());
        let mut present_ids_vk: SmallVec<[_; 4]> = SmallVec::with_capacity(swapchain_infos.len());
        let mut present_regions_vk: SmallVec<[_; 4]> =
            SmallVec::with_capacity(swapchain_infos.len());
        let mut rectangles_vk: SmallVec<[_; 4]> = SmallVec::with_capacity(swapchain_infos.len());

        let mut has_present_ids = false;
        let mut has_present_regions = false;

        for swapchain_info in swapchain_infos {
            let &SwapchainPresentInfo {
                ref swapchain,
                image_index,
                present_id,
                ref present_regions,
                _ne: _,
            } = swapchain_info;

            swapchains_vk.push(swapchain.internal_object());
            image_indices_vk.push(image_index);
            present_ids_vk.push(present_id.map_or(0, u64::from));
            present_regions_vk.push(ash::vk::PresentRegionKHR::default());
            rectangles_vk.push(
                present_regions
                    .iter()
                    .map(ash::vk::RectLayerKHR::from)
                    .collect::<SmallVec<[_; 4]>>(),
            );

            if present_id.is_some() {
                has_present_ids = true;
            }

            if !present_regions.is_empty() {
                has_present_regions = true;
            }
        }

        let mut results = vec![ash::vk::Result::SUCCESS; swapchain_infos.len()];
        let mut info_vk = ash::vk::PresentInfoKHR {
            wait_semaphore_count: wait_semaphores_vk.len() as u32,
            p_wait_semaphores: wait_semaphores_vk.as_ptr(),
            swapchain_count: swapchains_vk.len() as u32,
            p_swapchains: swapchains_vk.as_ptr(),
            p_image_indices: image_indices_vk.as_ptr(),
            p_results: results.as_mut_ptr(),
            ..Default::default()
        };
        let mut present_id_info_vk = None;
        let mut present_region_info_vk = None;

        if has_present_ids {
            let next = present_id_info_vk.insert(ash::vk::PresentIdKHR {
                swapchain_count: present_ids_vk.len() as u32,
                p_present_ids: present_ids_vk.as_ptr(),
                ..Default::default()
            });

            next.p_next = info_vk.p_next;
            info_vk.p_next = next as *const _ as *const _;
        }

        if has_present_regions {
            for (present_regions_vk, rectangles_vk) in
                (present_regions_vk.iter_mut()).zip(rectangles_vk.iter())
            {
                *present_regions_vk = ash::vk::PresentRegionKHR {
                    rectangle_count: rectangles_vk.len() as u32,
                    p_rectangles: rectangles_vk.as_ptr(),
                };
            }

            let next = present_region_info_vk.insert(ash::vk::PresentRegionsKHR {
                swapchain_count: present_regions_vk.len() as u32,
                p_regions: present_regions_vk.as_ptr(),
                ..Default::default()
            });

            next.p_next = info_vk.p_next;
            info_vk.p_next = next as *const _ as *const _;
        }

        let fns = self.queue.device().fns();
        let _ = (fns.khr_swapchain.queue_present_khr)(*self.handle, &info_vk);

        results
            .into_iter()
            .map(|result| result.result().map_err(VulkanError::from))
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub(crate) unsafe fn submit_unchecked(
        &mut self,
        submits: impl IntoIterator<Item = SubmitInfo>,
        fence: Option<Arc<Fence>>,
    ) -> Result<(), VulkanError> {
        let submits = submits.into_iter();

        if self.queue.device.enabled_features().synchronization2 {
            struct PerSubmitInfo {
                wait_semaphore_infos_vk: SmallVec<[ash::vk::SemaphoreSubmitInfo; 4]>,
                command_buffer_infos_vk: SmallVec<[ash::vk::CommandBufferSubmitInfo; 4]>,
                signal_semaphore_infos_vk: SmallVec<[ash::vk::SemaphoreSubmitInfo; 4]>,
            }

            let (mut submit_info_vk, per_submit_vk): (SmallVec<[_; 4]>, SmallVec<[_; 4]>) = submits
                .map(|submit_info| {
                    let SubmitInfo {
                        wait_semaphores,
                        command_buffers,
                        signal_semaphores,
                        _ne: _,
                    } = submit_info;

                    let wait_semaphore_infos_vk = wait_semaphores
                        .into_iter()
                        .map(|semaphore_submit_info| {
                            let SemaphoreSubmitInfo {
                                semaphore,
                                stages,
                                _ne: _,
                            } = semaphore_submit_info;

                            ash::vk::SemaphoreSubmitInfo {
                                semaphore: semaphore.internal_object(),
                                value: 0, // TODO:
                                stage_mask: stages.into(),
                                device_index: 0, // TODO:
                                ..Default::default()
                            }
                        })
                        .collect();

                    let command_buffer_infos_vk = command_buffers
                        .into_iter()
                        .map(|cb| ash::vk::CommandBufferSubmitInfo {
                            command_buffer: cb.inner().internal_object(),
                            device_mask: 0, // TODO:
                            ..Default::default()
                        })
                        .collect();

                    let signal_semaphore_infos_vk = signal_semaphores
                        .into_iter()
                        .map(|semaphore_submit_info| {
                            let SemaphoreSubmitInfo {
                                semaphore,
                                stages,
                                _ne: _,
                            } = semaphore_submit_info;

                            ash::vk::SemaphoreSubmitInfo {
                                semaphore: semaphore.internal_object(),
                                value: 0, // TODO:
                                stage_mask: stages.into(),
                                device_index: 0, // TODO:
                                ..Default::default()
                            }
                        })
                        .collect();

                    (
                        ash::vk::SubmitInfo2 {
                            flags: ash::vk::SubmitFlags::empty(), // TODO:
                            wait_semaphore_info_count: 0,
                            p_wait_semaphore_infos: ptr::null(),
                            command_buffer_info_count: 0,
                            p_command_buffer_infos: ptr::null(),
                            signal_semaphore_info_count: 0,
                            p_signal_semaphore_infos: ptr::null(),
                            ..Default::default()
                        },
                        PerSubmitInfo {
                            wait_semaphore_infos_vk,
                            command_buffer_infos_vk,
                            signal_semaphore_infos_vk,
                        },
                    )
                })
                .unzip();

            for (
                submit_info_vk,
                PerSubmitInfo {
                    wait_semaphore_infos_vk,
                    command_buffer_infos_vk,
                    signal_semaphore_infos_vk,
                },
            ) in (submit_info_vk.iter_mut()).zip(per_submit_vk.iter())
            {
                *submit_info_vk = ash::vk::SubmitInfo2 {
                    wait_semaphore_info_count: wait_semaphore_infos_vk.len() as u32,
                    p_wait_semaphore_infos: wait_semaphore_infos_vk.as_ptr(),
                    command_buffer_info_count: command_buffer_infos_vk.len() as u32,
                    p_command_buffer_infos: command_buffer_infos_vk.as_ptr(),
                    signal_semaphore_info_count: signal_semaphore_infos_vk.len() as u32,
                    p_signal_semaphore_infos: signal_semaphore_infos_vk.as_ptr(),
                    ..*submit_info_vk
                };
            }

            let fns = self.queue.device.fns();

            if self.queue.device.api_version() >= Version::V1_3 {
                (fns.v1_3.queue_submit2)(
                    *self.handle,
                    submit_info_vk.len() as u32,
                    submit_info_vk.as_ptr(),
                    fence.map_or_else(Default::default, |fence| fence.internal_object()),
                )
            } else {
                debug_assert!(self.queue.device.enabled_extensions().khr_synchronization2);
                (fns.khr_synchronization2.queue_submit2_khr)(
                    *self.handle,
                    submit_info_vk.len() as u32,
                    submit_info_vk.as_ptr(),
                    fence.map_or_else(Default::default, |fence| fence.internal_object()),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
        } else {
            struct PerSubmitInfo {
                wait_semaphores_vk: SmallVec<[ash::vk::Semaphore; 4]>,
                wait_dst_stage_mask_vk: SmallVec<[ash::vk::PipelineStageFlags; 4]>,
                command_buffers_vk: SmallVec<[ash::vk::CommandBuffer; 4]>,
                signal_semaphores_vk: SmallVec<[ash::vk::Semaphore; 4]>,
            }

            let (mut submit_info_vk, per_submit_vk): (SmallVec<[_; 4]>, SmallVec<[_; 4]>) = submits
                .map(|submit_info| {
                    let SubmitInfo {
                        wait_semaphores,
                        command_buffers,
                        signal_semaphores,
                        _ne: _,
                    } = submit_info;

                    let (wait_semaphores_vk, wait_dst_stage_mask_vk) = wait_semaphores
                        .into_iter()
                        .map(|semaphore_submit_info| {
                            let SemaphoreSubmitInfo {
                                semaphore,
                                stages,
                                _ne: _,
                            } = semaphore_submit_info;

                            (semaphore.internal_object(), stages.into())
                        })
                        .unzip();

                    let command_buffers_vk = command_buffers
                        .into_iter()
                        .map(|cb| cb.inner().internal_object())
                        .collect();

                    let signal_semaphores_vk = signal_semaphores
                        .into_iter()
                        .map(|semaphore_submit_info| {
                            let SemaphoreSubmitInfo {
                                semaphore,
                                stages: _,
                                _ne: _,
                            } = semaphore_submit_info;

                            semaphore.internal_object()
                        })
                        .collect();

                    (
                        ash::vk::SubmitInfo {
                            wait_semaphore_count: 0,
                            p_wait_semaphores: ptr::null(),
                            p_wait_dst_stage_mask: ptr::null(),
                            command_buffer_count: 0,
                            p_command_buffers: ptr::null(),
                            signal_semaphore_count: 0,
                            p_signal_semaphores: ptr::null(),
                            ..Default::default()
                        },
                        PerSubmitInfo {
                            wait_semaphores_vk,
                            wait_dst_stage_mask_vk,
                            command_buffers_vk,
                            signal_semaphores_vk,
                        },
                    )
                })
                .unzip();

            for (
                submit_info_vk,
                PerSubmitInfo {
                    wait_semaphores_vk,
                    wait_dst_stage_mask_vk,
                    command_buffers_vk,
                    signal_semaphores_vk,
                },
            ) in (submit_info_vk.iter_mut()).zip(per_submit_vk.iter())
            {
                *submit_info_vk = ash::vk::SubmitInfo {
                    wait_semaphore_count: wait_semaphores_vk.len() as u32,
                    p_wait_semaphores: wait_semaphores_vk.as_ptr(),
                    p_wait_dst_stage_mask: wait_dst_stage_mask_vk.as_ptr(),
                    command_buffer_count: command_buffers_vk.len() as u32,
                    p_command_buffers: command_buffers_vk.as_ptr(),
                    signal_semaphore_count: signal_semaphores_vk.len() as u32,
                    p_signal_semaphores: signal_semaphores_vk.as_ptr(),
                    ..*submit_info_vk
                };
            }

            let fns = self.queue.device.fns();
            (fns.v1_0.queue_submit)(
                *self.handle,
                submit_info_vk.len() as u32,
                submit_info_vk.as_ptr(),
                fence.map_or_else(Default::default, |fence| fence.internal_object()),
            )
            .result()
            .map_err(VulkanError::from)?;
        }

        Ok(())
    }

    /// Opens a queue debug label region.
    ///
    /// The [`ext_debug_utils`](crate::instance::InstanceExtensions::ext_debug_utils) must be
    /// enabled on the instance.
    #[inline]
    pub fn begin_debug_utils_label(
        &mut self,
        label_info: DebugUtilsLabel,
    ) -> Result<(), QueueError> {
        self.validate_begin_debug_utils_label(&label_info)?;

        unsafe {
            self.begin_debug_utils_label_unchecked(label_info);
            Ok(())
        }
    }

    fn validate_begin_debug_utils_label(
        &self,
        _label_info: &DebugUtilsLabel,
    ) -> Result<(), QueueError> {
        if !self
            .queue
            .device
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(QueueError::RequirementNotMet {
                required_for: "`begin_debug_utils_label`",
                requires_one_of: RequiresOneOf {
                    instance_extensions: &["ext_debug_utils"],
                    ..Default::default()
                },
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn begin_debug_utils_label_unchecked(&mut self, label_info: DebugUtilsLabel) {
        let DebugUtilsLabel {
            label_name,
            color,
            _ne: _,
        } = label_info;

        let label_name_vk = CString::new(label_name.as_str()).unwrap();
        let label_info = ash::vk::DebugUtilsLabelEXT {
            p_label_name: label_name_vk.as_ptr(),
            color,
            ..Default::default()
        };

        let fns = self.queue.device.instance().fns();
        (fns.ext_debug_utils.queue_begin_debug_utils_label_ext)(*self.handle, &label_info);
    }

    /// Closes a queue debug label region.
    ///
    /// The [`ext_debug_utils`](crate::instance::InstanceExtensions::ext_debug_utils) must be
    /// enabled on the instance.
    ///
    /// # Safety
    ///
    /// - There must be an outstanding queue label region begun with `begin_debug_utils_label` in
    ///   the queue.
    #[inline]
    pub unsafe fn end_debug_utils_label(&mut self) -> Result<(), QueueError> {
        self.validate_end_debug_utils_label()?;

        self.end_debug_utils_label_unchecked();
        Ok(())
    }

    fn validate_end_debug_utils_label(&self) -> Result<(), QueueError> {
        if !self
            .queue
            .device
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(QueueError::RequirementNotMet {
                required_for: "`end_debug_utils_label`",
                requires_one_of: RequiresOneOf {
                    instance_extensions: &["ext_debug_utils"],
                    ..Default::default()
                },
            });
        }

        // VUID-vkQueueEndDebugUtilsLabelEXT-None-01911
        // TODO: not checked, so unsafe for now

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn end_debug_utils_label_unchecked(&mut self) {
        let fns = self.queue.device.instance().fns();
        (fns.ext_debug_utils.queue_end_debug_utils_label_ext)(*self.handle);
    }

    /// Inserts a queue debug label.
    ///
    /// The [`ext_debug_utils`](crate::instance::InstanceExtensions::ext_debug_utils) must be
    /// enabled on the instance.
    #[inline]
    pub fn insert_debug_utils_label(
        &mut self,
        label_info: DebugUtilsLabel,
    ) -> Result<(), QueueError> {
        self.validate_insert_debug_utils_label(&label_info)?;

        unsafe {
            self.insert_debug_utils_label_unchecked(label_info);
            Ok(())
        }
    }

    fn validate_insert_debug_utils_label(
        &self,
        _label_info: &DebugUtilsLabel,
    ) -> Result<(), QueueError> {
        if !self
            .queue
            .device
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(QueueError::RequirementNotMet {
                required_for: "`insert_debug_utils_label`",
                requires_one_of: RequiresOneOf {
                    instance_extensions: &["ext_debug_utils"],
                    ..Default::default()
                },
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn insert_debug_utils_label_unchecked(&mut self, label_info: DebugUtilsLabel) {
        let DebugUtilsLabel {
            label_name,
            color,
            _ne: _,
        } = label_info;

        let label_name_vk = CString::new(label_name.as_str()).unwrap();
        let label_info = ash::vk::DebugUtilsLabelEXT {
            p_label_name: label_name_vk.as_ptr(),
            color,
            ..Default::default()
        };

        let fns = self.queue.device.instance().fns();
        (fns.ext_debug_utils.queue_insert_debug_utils_label_ext)(*self.handle, &label_info);
    }
}

/// Properties of a queue family in a physical device.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct QueueFamilyProperties {
    /// Attributes of the queue family.
    pub queue_flags: QueueFlags,

    /// The number of queues available in this family.
    ///
    /// This guaranteed to be at least 1 (or else that family wouldn't exist).
    pub queue_count: u32,

    /// If timestamps are supported, the number of bits supported by timestamp operations.
    /// The returned value will be in the range 36..64.
    ///
    /// If timestamps are not supported, this is `None`.
    pub timestamp_valid_bits: Option<u32>,

    /// The minimum granularity supported for image transfers, in terms of `[width, height, depth]`.
    pub min_image_transfer_granularity: [u32; 3],
}

impl QueueFamilyProperties {
    /// Returns whether the queues of this family support a particular pipeline stage.
    #[inline]
    pub fn supports_stage(&self, stage: PipelineStage) -> bool {
        ash::vk::QueueFlags::from(self.queue_flags).contains(stage.required_queue_flags())
    }
}

impl From<ash::vk::QueueFamilyProperties> for QueueFamilyProperties {
    #[inline]
    fn from(val: ash::vk::QueueFamilyProperties) -> Self {
        Self {
            queue_flags: val.queue_flags.into(),
            queue_count: val.queue_count,
            timestamp_valid_bits: (val.timestamp_valid_bits != 0)
                .then_some(val.timestamp_valid_bits),
            min_image_transfer_granularity: [
                val.min_image_transfer_granularity.width,
                val.min_image_transfer_granularity.height,
                val.min_image_transfer_granularity.depth,
            ],
        }
    }
}

vulkan_bitflags! {
    /// Attributes of a queue or queue family.
    #[non_exhaustive]
    QueueFlags = QueueFlags(u32);

    /// Queues of this family can execute graphics operations.
    graphics = GRAPHICS,

    /// Queues of this family can execute compute operations.
    compute = COMPUTE,

    /// Queues of this family can execute transfer operations.
    transfer = TRANSFER,

    /// Queues of this family can execute sparse memory management operations.
    sparse_binding = SPARSE_BINDING,

    /// Queues of this family can be created using the `protected` flag.
    protected = PROTECTED {
        api_version: V1_1,
    },

    /// Queues of this family can execute video decode operations.
    video_decode = VIDEO_DECODE_KHR {
        device_extensions: [khr_video_decode_queue],
    },

    /// Queues of this family can execute video encode operations.
    video_encode = VIDEO_ENCODE_KHR {
        device_extensions: [khr_video_encode_queue],
    },
}

/// Error that can happen when submitting work to a queue.
#[derive(Clone, Debug)]
pub enum QueueError {
    VulkanError(VulkanError),

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },
}

impl Error for QueueError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            QueueError::VulkanError(err) => Some(err),
            _ => None,
        }
    }
}

impl Display for QueueError {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
        match self {
            Self::VulkanError(_) => write!(f, "a runtime error occurred",),

            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
        }
    }
}

impl From<VulkanError> for QueueError {
    #[inline]
    fn from(err: VulkanError) -> Self {
        Self::VulkanError(err)
    }
}

impl From<RequirementNotMet> for QueueError {
    #[inline]
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}
