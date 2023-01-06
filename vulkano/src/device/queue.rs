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
    buffer::BufferState,
    command_buffer::{
        CommandBufferResourcesUsage, CommandBufferState, CommandBufferUsage, SemaphoreSubmitInfo,
        SubmitInfo,
    },
    image::{sys::ImageState, ImageAccess},
    instance::debug::DebugUtilsLabel,
    macros::vulkan_bitflags,
    memory::{
        BindSparseInfo, SparseBufferMemoryBind, SparseImageMemoryBind, SparseImageOpaqueMemoryBind,
    },
    swapchain::{PresentInfo, SwapchainPresentInfo},
    sync::{
        fence::{Fence, FenceState},
        future::{AccessCheckError, FlushError, GpuFuture},
        semaphore::SemaphoreState,
    },
    OomError, RequirementNotMet, RequiresOneOf, Version, VulkanError, VulkanObject,
};
use ahash::HashMap;
use parking_lot::{Mutex, MutexGuard};
use smallvec::{smallvec, SmallVec};
use std::{
    collections::VecDeque,
    error::Error,
    ffi::CString,
    fmt::{Display, Error as FmtError, Formatter},
    hash::{Hash, Hasher},
    mem::take,
    ptr,
    sync::{atomic::Ordering, Arc},
};

/// Represents a queue where commands can be submitted.
// TODO: should use internal synchronization?
#[derive(Debug)]
pub struct Queue {
    handle: ash::vk::Queue,
    device: Arc<Device>,
    queue_family_index: u32,
    id: u32, // id within family

    state: Mutex<QueueState>,
}

impl Queue {
    // TODO: Make public
    #[inline]
    pub(super) fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::Queue,
        queue_family_index: u32,
        id: u32,
    ) -> Arc<Self> {
        Arc::new(Queue {
            handle,
            device,
            queue_family_index,
            id,
            state: Mutex::new(Default::default()),
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

    /// Locks the queue and then calls the provided closure, providing it with an object that
    /// can be used to perform operations on the queue, such as command buffer submissions.
    #[inline]
    pub fn with<'a, R>(self: &'a Arc<Self>, func: impl FnOnce(QueueGuard<'a>) -> R) -> R {
        func(QueueGuard {
            queue: self,
            state: self.state.lock(),
        })
    }
}

impl Drop for Queue {
    #[inline]
    fn drop(&mut self) {
        let state = self.state.get_mut();
        let _ = state.wait_idle(&self.device, self.handle);
    }
}

unsafe impl VulkanObject for Queue {
    type Handle = ash::vk::Queue;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for Queue {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl PartialEq for Queue {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.queue_family_index == other.queue_family_index
            && self.device == other.device
    }
}

impl Eq for Queue {}

impl Hash for Queue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.queue_family_index.hash(state);
        self.device.hash(state);
    }
}

pub struct QueueGuard<'a> {
    queue: &'a Arc<Queue>,
    state: MutexGuard<'a, QueueState>,
}

impl<'a> QueueGuard<'a> {
    pub(crate) unsafe fn fence_signaled(&mut self, fence: &Fence) {
        self.state.fence_signaled(fence)
    }

    /// Waits until all work on this queue has finished, then releases ownership of all resources
    /// that were in use by the queue.
    ///
    /// This is equivalent to submitting a fence to the queue, waiting on it, and then calling
    /// `cleanup_finished`.
    ///
    /// Just like [`Device::wait_idle`], you shouldn't have to call this function in a typical
    /// program.
    #[inline]
    pub fn wait_idle(&mut self) -> Result<(), OomError> {
        self.state.wait_idle(&self.queue.device, self.queue.handle)
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub(crate) unsafe fn bind_sparse_unchecked(
        &mut self,
        bind_infos: impl IntoIterator<Item = BindSparseInfo>,
        fence: Option<Arc<Fence>>,
    ) -> Result<(), VulkanError> {
        let bind_infos: SmallVec<[_; 4]> = bind_infos.into_iter().collect();
        let mut states = States::from_bind_infos(&bind_infos);

        self.bind_sparse_unchecked_locked(
            &bind_infos,
            fence.as_ref().map(|fence| {
                let state = fence.state();
                (fence, state)
            }),
            &mut states,
        )
    }

    unsafe fn bind_sparse_unchecked_locked(
        &mut self,
        bind_infos: &SmallVec<[BindSparseInfo; 4]>,
        fence: Option<(&Arc<Fence>, MutexGuard<'_, FenceState>)>,
        states: &mut States<'_>,
    ) -> Result<(), VulkanError> {
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
            .iter()
            .map(|bind_info| {
                let &BindSparseInfo {
                    ref wait_semaphores,
                    ref buffer_binds,
                    ref image_opaque_binds,
                    ref image_binds,
                    ref signal_semaphores,
                    _ne: _,
                } = bind_info;

                let wait_semaphores_vk: SmallVec<[_; 4]> = wait_semaphores
                    .iter()
                    .map(|semaphore| semaphore.handle())
                    .collect();

                let (buffer_bind_infos_vk, buffer_binds_vk): (SmallVec<[_; 4]>, SmallVec<[_; 4]>) =
                    buffer_binds
                        .iter()
                        .map(|(buffer, memory_binds)| {
                            (
                                ash::vk::SparseBufferMemoryBindInfo {
                                    buffer: buffer.buffer().handle(),
                                    bind_count: 0,
                                    p_binds: ptr::null(),
                                },
                                memory_binds
                                    .iter()
                                    .map(|memory_bind| {
                                        let &SparseBufferMemoryBind {
                                            offset,
                                            size,
                                            ref memory,
                                        } = memory_bind;

                                        let (memory, memory_offset) = memory.as_ref().map_or_else(
                                            Default::default,
                                            |(memory, memory_offset)| {
                                                (memory.handle(), *memory_offset)
                                            },
                                        );

                                        ash::vk::SparseMemoryBind {
                                            resource_offset: offset,
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
                                image: image.inner().image.handle(),
                                bind_count: 0,
                                p_binds: ptr::null(),
                            },
                            memory_binds
                                .iter()
                                .map(|memory_bind| {
                                    let &SparseImageOpaqueMemoryBind {
                                        offset,
                                        size,
                                        ref memory,
                                        metadata,
                                    } = memory_bind;

                                    let (memory, memory_offset) = memory.as_ref().map_or_else(
                                        Default::default,
                                        |(memory, memory_offset)| (memory.handle(), *memory_offset),
                                    );

                                    ash::vk::SparseMemoryBind {
                                        resource_offset: offset,
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
                                    image: image.inner().image.handle(),
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
                                                (memory.handle(), *memory_offset)
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
                    .map(|semaphore| semaphore.handle())
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
            self.queue.handle,
            bind_infos_vk.len() as u32,
            bind_infos_vk.as_ptr(),
            fence
                .as_ref()
                .map_or_else(Default::default, |(fence, _)| fence.handle()),
        )
        .result()
        .map_err(VulkanError::from)?;

        for bind_info in bind_infos {
            let BindSparseInfo {
                wait_semaphores,
                buffer_binds: _,
                image_opaque_binds: _,
                image_binds: _,
                signal_semaphores,
                _ne: _,
            } = bind_info;

            for semaphore in wait_semaphores {
                let state = states.semaphores.get_mut(&semaphore.handle()).unwrap();
                state.add_queue_wait(self.queue);
            }

            for semaphore in signal_semaphores {
                let state = states.semaphores.get_mut(&semaphore.handle()).unwrap();
                state.add_queue_wait(self.queue);
            }
        }

        let fence = fence.map(|(fence, mut state)| {
            state.add_queue_signal(self.queue);
            fence.clone()
        });

        self.state
            .operations
            .push_back((bind_infos.clone().into(), fence));

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn present_unchecked(
        &mut self,
        present_info: PresentInfo,
    ) -> Result<impl ExactSizeIterator<Item = Result<bool, VulkanError>>, VulkanError> {
        let mut states = States::from_present_info(&present_info);
        self.present_unchecked_locked(&present_info, &mut states)
    }

    unsafe fn present_unchecked_locked(
        &mut self,
        present_info: &PresentInfo,
        states: &mut States<'_>,
    ) -> Result<impl ExactSizeIterator<Item = Result<bool, VulkanError>>, VulkanError> {
        let PresentInfo {
            ref wait_semaphores,
            ref swapchain_infos,
            _ne: _,
        } = present_info;

        let wait_semaphores_vk: SmallVec<[_; 4]> = wait_semaphores
            .iter()
            .map(|semaphore| semaphore.handle())
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

            swapchains_vk.push(swapchain.handle());
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
        let result = (fns.khr_swapchain.queue_present_khr)(self.queue.handle, &info_vk);

        // Per the documentation of `vkQueuePresentKHR`, certain results indicate that the whole
        // operation has failed, while others only indicate failure of a particular present.
        // If we got a result that is not one of these per-present ones, we return it directly.
        // Otherwise, we consider the present to be enqueued.
        if !matches!(
            result,
            ash::vk::Result::SUCCESS
                | ash::vk::Result::SUBOPTIMAL_KHR
                | ash::vk::Result::ERROR_OUT_OF_DATE_KHR
                | ash::vk::Result::ERROR_SURFACE_LOST_KHR
                | ash::vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT,
        ) {
            return Err(VulkanError::from(result));
        }

        for semaphore in wait_semaphores {
            let state = states.semaphores.get_mut(&semaphore.handle()).unwrap();
            state.add_queue_wait(self.queue);
        }

        self.state
            .operations
            .push_back((present_info.clone().into(), None));

        // If a presentation results in a loss of full-screen exclusive mode,
        // signal that to the relevant swapchain.
        for (&result, swapchain_info) in results.iter().zip(&present_info.swapchain_infos) {
            if result == ash::vk::Result::ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT {
                swapchain_info
                    .swapchain
                    .full_screen_exclusive_held()
                    .store(false, Ordering::SeqCst);
            }
        }

        Ok(results.into_iter().map(|result| match result {
            ash::vk::Result::SUCCESS => Ok(false),
            ash::vk::Result::SUBOPTIMAL_KHR => Ok(true),
            err => Err(VulkanError::from(err)),
        }))
    }

    // Temporary function to keep futures working.
    pub(crate) unsafe fn submit_with_future(
        &mut self,
        submit_info: SubmitInfo,
        fence: Option<Arc<Fence>>,
        future: &dyn GpuFuture,
        queue: &Queue,
    ) -> Result<(), FlushError> {
        let submit_infos: SmallVec<[_; 4]> = smallvec![submit_info];
        let mut states = States::from_submit_infos(&submit_infos);

        for submit_info in &submit_infos {
            for command_buffer in &submit_info.command_buffers {
                let state = states
                    .command_buffers
                    .get(&command_buffer.handle())
                    .unwrap();

                match command_buffer.usage() {
                    CommandBufferUsage::OneTimeSubmit => {
                        // VUID-vkQueueSubmit2-commandBuffer-03874
                        if state.has_been_submitted() {
                            return Err(FlushError::OneTimeSubmitAlreadySubmitted);
                        }
                    }
                    CommandBufferUsage::MultipleSubmit => {
                        // VUID-vkQueueSubmit2-commandBuffer-03875
                        if state.is_submit_pending() {
                            return Err(FlushError::ExclusiveAlreadyInUse);
                        }
                    }
                    CommandBufferUsage::SimultaneousUse => (),
                }

                let CommandBufferResourcesUsage {
                    buffers,
                    images,
                    buffer_indices: _,
                    image_indices: _,
                } = command_buffer.resources_usage();

                for usage in buffers {
                    let state = states.buffers.get_mut(&usage.buffer.handle()).unwrap();

                    for (range, range_usage) in usage.ranges.iter() {
                        match future.check_buffer_access(
                            &usage.buffer,
                            range.clone(),
                            range_usage.mutable,
                            queue,
                        ) {
                            Err(AccessCheckError::Denied(error)) => {
                                return Err(FlushError::ResourceAccessError {
                                    error,
                                    use_ref: range_usage.first_use,
                                });
                            }
                            Err(AccessCheckError::Unknown) => {
                                let result = if range_usage.mutable {
                                    state.check_gpu_write(range.clone())
                                } else {
                                    state.check_gpu_read(range.clone())
                                };

                                if let Err(error) = result {
                                    return Err(FlushError::ResourceAccessError {
                                        error,
                                        use_ref: range_usage.first_use,
                                    });
                                }
                            }
                            _ => (),
                        }
                    }
                }

                for usage in images {
                    let state = states.images.get_mut(&usage.image.handle()).unwrap();

                    for (range, range_usage) in usage.ranges.iter() {
                        match future.check_image_access(
                            &usage.image,
                            range.clone(),
                            range_usage.mutable,
                            range_usage.expected_layout,
                            queue,
                        ) {
                            Err(AccessCheckError::Denied(error)) => {
                                return Err(FlushError::ResourceAccessError {
                                    error,
                                    use_ref: range_usage.first_use,
                                });
                            }
                            Err(AccessCheckError::Unknown) => {
                                let result = if range_usage.mutable {
                                    state
                                        .check_gpu_write(range.clone(), range_usage.expected_layout)
                                } else {
                                    state.check_gpu_read(range.clone(), range_usage.expected_layout)
                                };

                                if let Err(error) = result {
                                    return Err(FlushError::ResourceAccessError {
                                        error,
                                        use_ref: range_usage.first_use,
                                    });
                                }
                            }
                            _ => (),
                        };
                    }
                }
            }
        }

        Ok(self.submit_unchecked_locked(
            &submit_infos,
            fence.as_ref().map(|fence| {
                let state = fence.state();
                (fence, state)
            }),
            &mut states,
        )?)
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn submit_unchecked(
        &mut self,
        submit_infos: impl IntoIterator<Item = SubmitInfo>,
        fence: Option<Arc<Fence>>,
    ) -> Result<(), VulkanError> {
        let submit_infos: SmallVec<[_; 4]> = submit_infos.into_iter().collect();
        let mut states = States::from_submit_infos(&submit_infos);

        self.submit_unchecked_locked(
            &submit_infos,
            fence.as_ref().map(|fence| {
                let state = fence.state();
                (fence, state)
            }),
            &mut states,
        )
    }

    unsafe fn submit_unchecked_locked(
        &mut self,
        submit_infos: &SmallVec<[SubmitInfo; 4]>,
        fence: Option<(&Arc<Fence>, MutexGuard<'_, FenceState>)>,
        states: &mut States<'_>,
    ) -> Result<(), VulkanError> {
        if self.queue.device.enabled_features().synchronization2 {
            struct PerSubmitInfo {
                wait_semaphore_infos_vk: SmallVec<[ash::vk::SemaphoreSubmitInfo; 4]>,
                command_buffer_infos_vk: SmallVec<[ash::vk::CommandBufferSubmitInfo; 4]>,
                signal_semaphore_infos_vk: SmallVec<[ash::vk::SemaphoreSubmitInfo; 4]>,
            }

            let (mut submit_info_vk, per_submit_vk): (SmallVec<[_; 4]>, SmallVec<[_; 4]>) =
                submit_infos
                    .iter()
                    .map(|submit_info| {
                        let &SubmitInfo {
                            ref wait_semaphores,
                            ref command_buffers,
                            ref signal_semaphores,
                            _ne: _,
                        } = submit_info;

                        let wait_semaphore_infos_vk = wait_semaphores
                            .iter()
                            .map(|semaphore_submit_info| {
                                let &SemaphoreSubmitInfo {
                                    ref semaphore,
                                    stages,
                                    _ne: _,
                                } = semaphore_submit_info;

                                ash::vk::SemaphoreSubmitInfo {
                                    semaphore: semaphore.handle(),
                                    value: 0, // TODO:
                                    stage_mask: stages.into(),
                                    device_index: 0, // TODO:
                                    ..Default::default()
                                }
                            })
                            .collect();

                        let command_buffer_infos_vk = command_buffers
                            .iter()
                            .map(|cb| ash::vk::CommandBufferSubmitInfo {
                                command_buffer: cb.handle(),
                                device_mask: 0, // TODO:
                                ..Default::default()
                            })
                            .collect();

                        let signal_semaphore_infos_vk = signal_semaphores
                            .iter()
                            .map(|semaphore_submit_info| {
                                let &SemaphoreSubmitInfo {
                                    ref semaphore,
                                    stages,
                                    _ne: _,
                                } = semaphore_submit_info;

                                ash::vk::SemaphoreSubmitInfo {
                                    semaphore: semaphore.handle(),
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
                    self.queue.handle,
                    submit_info_vk.len() as u32,
                    submit_info_vk.as_ptr(),
                    fence
                        .as_ref()
                        .map_or_else(Default::default, |(fence, _)| fence.handle()),
                )
            } else {
                debug_assert!(self.queue.device.enabled_extensions().khr_synchronization2);
                (fns.khr_synchronization2.queue_submit2_khr)(
                    self.queue.handle,
                    submit_info_vk.len() as u32,
                    submit_info_vk.as_ptr(),
                    fence
                        .as_ref()
                        .map_or_else(Default::default, |(fence, _)| fence.handle()),
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

            let (mut submit_info_vk, per_submit_vk): (SmallVec<[_; 4]>, SmallVec<[_; 4]>) =
                submit_infos
                    .iter()
                    .map(|submit_info| {
                        let &SubmitInfo {
                            ref wait_semaphores,
                            ref command_buffers,
                            ref signal_semaphores,
                            _ne: _,
                        } = submit_info;

                        let (wait_semaphores_vk, wait_dst_stage_mask_vk) = wait_semaphores
                            .iter()
                            .map(|semaphore_submit_info| {
                                let &SemaphoreSubmitInfo {
                                    ref semaphore,
                                    stages,
                                    _ne: _,
                                } = semaphore_submit_info;

                                (semaphore.handle(), stages.into())
                            })
                            .unzip();

                        let command_buffers_vk =
                            command_buffers.iter().map(|cb| cb.handle()).collect();

                        let signal_semaphores_vk = signal_semaphores
                            .iter()
                            .map(|semaphore_submit_info| {
                                let &SemaphoreSubmitInfo {
                                    ref semaphore,
                                    stages: _,
                                    _ne: _,
                                } = semaphore_submit_info;

                                semaphore.handle()
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
                self.queue.handle,
                submit_info_vk.len() as u32,
                submit_info_vk.as_ptr(),
                fence
                    .as_ref()
                    .map_or_else(Default::default, |(fence, _)| fence.handle()),
            )
            .result()
            .map_err(VulkanError::from)?;
        }

        for submit_info in submit_infos {
            let SubmitInfo {
                wait_semaphores,
                command_buffers,
                signal_semaphores,
                _ne: _,
            } = submit_info;

            for semaphore_submit_info in wait_semaphores {
                let state = states
                    .semaphores
                    .get_mut(&semaphore_submit_info.semaphore.handle())
                    .unwrap();
                state.add_queue_wait(self.queue);
            }

            for command_buffer in command_buffers {
                let state = states
                    .command_buffers
                    .get_mut(&command_buffer.handle())
                    .unwrap();
                state.add_queue_submit();

                let CommandBufferResourcesUsage {
                    buffers,
                    images,
                    buffer_indices: _,
                    image_indices: _,
                } = command_buffer.resources_usage();

                for usage in buffers {
                    let state = states.buffers.get_mut(&usage.buffer.handle()).unwrap();

                    for (range, range_usage) in usage.ranges.iter() {
                        if range_usage.mutable {
                            state.gpu_write_lock(range.clone());
                        } else {
                            state.gpu_read_lock(range.clone());
                        }
                    }
                }

                for usage in images {
                    let state = states.images.get_mut(&usage.image.handle()).unwrap();

                    for (range, range_usage) in usage.ranges.iter() {
                        if range_usage.mutable {
                            state.gpu_write_lock(range.clone(), range_usage.final_layout);
                        } else {
                            state.gpu_read_lock(range.clone());
                        }
                    }
                }
            }

            for semaphore_submit_info in signal_semaphores {
                let state = states
                    .semaphores
                    .get_mut(&semaphore_submit_info.semaphore.handle())
                    .unwrap();
                state.add_queue_signal(self.queue);
            }
        }

        let fence = fence.map(|(fence, mut state)| {
            state.add_queue_signal(self.queue);
            fence.clone()
        });

        self.state
            .operations
            .push_back((submit_infos.clone().into(), fence));

        Ok(())
    }

    /// Opens a queue debug label region.
    ///
    /// The [`ext_debug_utils`] extension must be enabled on the instance.
    ///
    /// [`ext_debug_utils`]: crate::instance::InstanceExtensions::ext_debug_utils
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
                required_for: "`QueueGuard::begin_debug_utils_label`",
                requires_one_of: RequiresOneOf {
                    instance_extensions: &["ext_debug_utils"],
                    ..Default::default()
                },
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
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
        (fns.ext_debug_utils.queue_begin_debug_utils_label_ext)(self.queue.handle, &label_info);
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
                required_for: "`QueueGuard::end_debug_utils_label`",
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
    #[inline]
    pub unsafe fn end_debug_utils_label_unchecked(&mut self) {
        let fns = self.queue.device.instance().fns();
        (fns.ext_debug_utils.queue_end_debug_utils_label_ext)(self.queue.handle);
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
                required_for: "`QueueGuard::insert_debug_utils_label`",
                requires_one_of: RequiresOneOf {
                    instance_extensions: &["ext_debug_utils"],
                    ..Default::default()
                },
            });
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
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
        (fns.ext_debug_utils.queue_insert_debug_utils_label_ext)(self.queue.handle, &label_info);
    }
}

#[derive(Debug, Default)]
struct QueueState {
    operations: VecDeque<(QueueOperation, Option<Arc<Fence>>)>,
}

impl QueueState {
    fn wait_idle(&mut self, device: &Device, handle: ash::vk::Queue) -> Result<(), OomError> {
        unsafe {
            let fns = device.fns();
            (fns.v1_0.queue_wait_idle)(handle)
                .result()
                .map_err(VulkanError::from)?;

            // Since we now know that the queue is finished with all work,
            // we can safely release all resources.
            for (operation, _) in take(&mut self.operations) {
                operation.set_finished();
            }

            Ok(())
        }
    }

    /// Called by `fence` when it finds that it is signaled.
    fn fence_signaled(&mut self, fence: &Fence) {
        // Find the most recent operation that uses `fence`.
        let fence_index = self
            .operations
            .iter()
            .enumerate()
            .rev()
            .find_map(|(index, (_, f))| {
                f.as_ref().map_or(false, |f| **f == *fence).then_some(index)
            });

        if let Some(index) = fence_index {
            // Remove all operations up to this index, and perform cleanup if needed.
            for (operation, fence) in self.operations.drain(..index + 1) {
                unsafe {
                    operation.set_finished();

                    if let Some(fence) = fence {
                        fence.state().set_signal_finished();
                    }
                }
            }
        }
    }
}

#[derive(Debug)]
enum QueueOperation {
    BindSparse(SmallVec<[BindSparseInfo; 4]>),
    Present(PresentInfo),
    Submit(SmallVec<[SubmitInfo; 4]>),
}

impl QueueOperation {
    unsafe fn set_finished(self) {
        match self {
            QueueOperation::BindSparse(bind_infos) => {
                for bind_info in bind_infos {
                    for semaphore in bind_info.wait_semaphores {
                        semaphore.state().set_wait_finished();
                    }

                    for semaphore in bind_info.signal_semaphores {
                        semaphore.state().set_signal_finished();
                    }
                }

                // TODO: Do we need to unlock buffers and images here?
            }
            QueueOperation::Present(present_info) => {
                for semaphore in present_info.wait_semaphores {
                    semaphore.state().set_wait_finished();
                }
            }
            QueueOperation::Submit(submit_infos) => {
                for submit_info in submit_infos {
                    for semaphore_submit_info in submit_info.wait_semaphores {
                        semaphore_submit_info.semaphore.state().set_wait_finished();
                    }

                    for semaphore_submit_info in submit_info.signal_semaphores {
                        semaphore_submit_info
                            .semaphore
                            .state()
                            .set_signal_finished();
                    }

                    for command_buffer in submit_info.command_buffers {
                        let resource_usage = command_buffer.resources_usage();

                        for usage in &resource_usage.buffers {
                            let mut state = usage.buffer.state();

                            for (range, range_usage) in usage.ranges.iter() {
                                if range_usage.mutable {
                                    state.gpu_write_unlock(range.clone());
                                } else {
                                    state.gpu_read_unlock(range.clone());
                                }
                            }
                        }

                        for usage in &resource_usage.images {
                            let mut state = usage.image.state();

                            for (range, range_usage) in usage.ranges.iter() {
                                if range_usage.mutable {
                                    state.gpu_write_unlock(range.clone());
                                } else {
                                    state.gpu_read_unlock(range.clone());
                                }
                            }
                        }

                        command_buffer.state().set_submit_finished();
                    }
                }
            }
        }
    }
}

impl From<SmallVec<[BindSparseInfo; 4]>> for QueueOperation {
    #[inline]
    fn from(val: SmallVec<[BindSparseInfo; 4]>) -> Self {
        Self::BindSparse(val)
    }
}

impl From<PresentInfo> for QueueOperation {
    #[inline]
    fn from(val: PresentInfo) -> Self {
        Self::Present(val)
    }
}

impl From<SmallVec<[SubmitInfo; 4]>> for QueueOperation {
    #[inline]
    fn from(val: SmallVec<[SubmitInfo; 4]>) -> Self {
        Self::Submit(val)
    }
}

// This struct exists to ensure that every object gets locked exactly once.
// Otherwise we get deadlocks.
#[derive(Debug)]
struct States<'a> {
    buffers: HashMap<ash::vk::Buffer, MutexGuard<'a, BufferState>>,
    command_buffers: HashMap<ash::vk::CommandBuffer, MutexGuard<'a, CommandBufferState>>,
    images: HashMap<ash::vk::Image, MutexGuard<'a, ImageState>>,
    semaphores: HashMap<ash::vk::Semaphore, MutexGuard<'a, SemaphoreState>>,
}

impl<'a> States<'a> {
    fn from_bind_infos(bind_infos: &'a [BindSparseInfo]) -> Self {
        let mut buffers = HashMap::default();
        let mut images = HashMap::default();
        let mut semaphores = HashMap::default();

        for bind_info in bind_infos {
            let BindSparseInfo {
                wait_semaphores,
                buffer_binds,
                image_opaque_binds,
                image_binds,
                signal_semaphores,
                _ne: _,
            } = bind_info;

            for semaphore in wait_semaphores {
                semaphores
                    .entry(semaphore.handle())
                    .or_insert_with(|| semaphore.state());
            }

            for (buffer, _) in buffer_binds {
                let buffer = buffer.buffer();
                buffers
                    .entry(buffer.handle())
                    .or_insert_with(|| buffer.state());
            }

            for (image, _) in image_opaque_binds {
                let image = &image.inner().image;
                images
                    .entry(image.handle())
                    .or_insert_with(|| image.state());
            }

            for (image, _) in image_binds {
                let image = &image.inner().image;
                images
                    .entry(image.handle())
                    .or_insert_with(|| image.state());
            }

            for semaphore in signal_semaphores {
                semaphores
                    .entry(semaphore.handle())
                    .or_insert_with(|| semaphore.state());
            }
        }

        Self {
            buffers,
            command_buffers: HashMap::default(),
            images,
            semaphores,
        }
    }

    fn from_present_info(present_info: &'a PresentInfo) -> Self {
        let mut semaphores = HashMap::default();

        let PresentInfo {
            wait_semaphores,
            swapchain_infos: _,
            _ne: _,
        } = present_info;

        for semaphore in wait_semaphores {
            semaphores
                .entry(semaphore.handle())
                .or_insert_with(|| semaphore.state());
        }

        Self {
            buffers: HashMap::default(),
            command_buffers: HashMap::default(),
            images: HashMap::default(),
            semaphores,
        }
    }

    fn from_submit_infos(submit_infos: &'a [SubmitInfo]) -> Self {
        let mut buffers = HashMap::default();
        let mut command_buffers = HashMap::default();
        let mut images = HashMap::default();
        let mut semaphores = HashMap::default();

        for submit_info in submit_infos {
            let SubmitInfo {
                wait_semaphores,
                command_buffers: info_command_buffers,
                signal_semaphores,
                _ne: _,
            } = submit_info;

            for semaphore_submit_info in wait_semaphores {
                let semaphore = &semaphore_submit_info.semaphore;
                semaphores
                    .entry(semaphore.handle())
                    .or_insert_with(|| semaphore.state());
            }

            for command_buffer in info_command_buffers {
                command_buffers
                    .entry(command_buffer.handle())
                    .or_insert_with(|| command_buffer.state());

                let CommandBufferResourcesUsage {
                    buffers: buffers_usage,
                    images: images_usage,
                    buffer_indices: _,
                    image_indices: _,
                } = command_buffer.resources_usage();

                for usage in buffers_usage {
                    let buffer = &usage.buffer;
                    buffers
                        .entry(buffer.handle())
                        .or_insert_with(|| buffer.state());
                }

                for usage in images_usage {
                    let image = &usage.image;
                    images
                        .entry(image.handle())
                        .or_insert_with(|| image.state());
                }
            }

            for semaphore_submit_info in signal_semaphores {
                let semaphore = &semaphore_submit_info.semaphore;
                semaphores
                    .entry(semaphore.handle())
                    .or_insert_with(|| semaphore.state());
            }
        }

        Self {
            buffers,
            command_buffers,
            images,
            semaphores,
        }
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
    #[non_exhaustive]

    /// Attributes of a queue or queue family.
    QueueFlags = QueueFlags(u32);

    /// Queues of this family can execute graphics operations.
    GRAPHICS = GRAPHICS,

    /// Queues of this family can execute compute operations.
    COMPUTE = COMPUTE,

    /// Queues of this family can execute transfer operations.
    TRANSFER = TRANSFER,

    /// Queues of this family can execute sparse memory management operations.
    SPARSE_BINDING = SPARSE_BINDING,

    /// Queues of this family can be created using the `protected` flag.
    PROTECTED = PROTECTED {
        api_version: V1_1,
    },

    /// Queues of this family can execute video decode operations.
    VIDEO_DECODE = VIDEO_DECODE_KHR {
        device_extensions: [khr_video_decode_queue],
    },

    /// Queues of this family can execute video encode operations.
    VIDEO_ENCODE = VIDEO_ENCODE_KHR {
        device_extensions: [khr_video_encode_queue],
    },

    /// Queues of this family can execute optical flow operations.
    OPTICAL_FLOW = OPTICAL_FLOW_NV {
        device_extensions: [nv_optical_flow],
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
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::VulkanError(_) => write!(f, "a runtime error occurred"),
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
    fn from(err: VulkanError) -> Self {
        Self::VulkanError(err)
    }
}

impl From<RequirementNotMet> for QueueError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::sync::fence::Fence;
    use std::{sync::Arc, time::Duration};

    #[test]
    fn empty_submit() {
        let (_device, queue) = gfx_dev_and_queue!();

        queue
            .with(|mut q| unsafe { q.submit_unchecked([Default::default()], None) })
            .unwrap();
    }

    #[test]
    fn signal_fence() {
        unsafe {
            let (device, queue) = gfx_dev_and_queue!();

            let fence = Arc::new(Fence::new(device, Default::default()).unwrap());
            assert!(!fence.is_signaled().unwrap());

            queue
                .with(|mut q| q.submit_unchecked([Default::default()], Some(fence.clone())))
                .unwrap();

            fence.wait(Some(Duration::from_secs(5))).unwrap();
            assert!(fence.is_signaled().unwrap());
        }
    }
}
