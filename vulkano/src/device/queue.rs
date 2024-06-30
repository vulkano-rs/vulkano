use super::{Device, DeviceOwned, QueueCreateFlags};
use crate::{
    command_buffer::{CommandBufferSubmitInfo, SemaphoreSubmitInfo, SubmitInfo},
    instance::{debug::DebugUtilsLabel, InstanceOwnedDebugWrapper},
    macros::vulkan_bitflags,
    memory::{
        BindSparseInfo, SparseBufferMemoryBind, SparseImageMemoryBind, SparseImageOpaqueMemoryBind,
    },
    swapchain::{PresentInfo, SemaphorePresentInfo, SwapchainPresentInfo},
    sync::{fence::Fence, semaphore::SemaphoreType, PipelineStages},
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version, VulkanError,
    VulkanObject,
};
use parking_lot::{Mutex, MutexGuard};
use smallvec::SmallVec;
use std::{
    ffi::CString,
    fmt::{Display, Formatter},
    hash::{Hash, Hasher},
    mem::MaybeUninit,
    ops::Deref,
    ptr,
    sync::Arc,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct QueueFamilyIndex(pub u32);

impl Deref for QueueFamilyIndex {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Display for QueueFamilyIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "QueueFamilyIndex {}", self.0)
    }
}

impl From<QueueFamilyIndex> for u32 {
    fn from(value: QueueFamilyIndex) -> Self {
        value.0
    }
}

impl From<QueueFamilyIndex> for usize {
    fn from(value: QueueFamilyIndex) -> Self {
        value.0 as usize
    }
}

/// Represents a queue where commands can be submitted.
// TODO: should use internal synchronization?
#[derive(Debug)]
pub struct Queue {
    handle: ash::vk::Queue,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,

    flags: QueueCreateFlags,
    queue_family_index: QueueFamilyIndex,
    queue_index: u32, // index within family

    state: Mutex<QueueState>,
}

impl Queue {
    pub(super) unsafe fn new(device: Arc<Device>, queue_info: DeviceQueueInfo) -> Arc<Self> {
        let &DeviceQueueInfo {
            flags,
            queue_family_index,
            queue_index,
            _ne: _,
        } = &queue_info;

        let queue_info_vk = ash::vk::DeviceQueueInfo2 {
            flags: flags.into(),
            queue_family_index: queue_family_index.0,
            queue_index,
            ..Default::default()
        };

        let fns = device.fns();
        let mut output = MaybeUninit::uninit();

        if device.api_version() >= Version::V1_1 {
            (fns.v1_1.get_device_queue2)(device.handle(), &queue_info_vk, output.as_mut_ptr());
        } else {
            debug_assert!(queue_info_vk.flags.is_empty());
            debug_assert!(queue_info_vk.p_next.is_null());
            (fns.v1_0.get_device_queue)(
                device.handle(),
                queue_info_vk.queue_family_index,
                queue_info_vk.queue_index,
                output.as_mut_ptr(),
            );
        }

        let handle = output.assume_init();
        Self::from_handle(device, handle, queue_info)
    }

    // TODO: Make public
    #[inline]
    pub(super) unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::Queue,
        queue_info: DeviceQueueInfo,
    ) -> Arc<Self> {
        let DeviceQueueInfo {
            flags,
            queue_family_index,
            queue_index,
            _ne: _,
        } = queue_info;

        Arc::new(Queue {
            handle,
            device: InstanceOwnedDebugWrapper(device),
            flags,
            queue_family_index,
            queue_index,
            state: Mutex::new(Default::default()),
        })
    }

    /// Returns the device that this queue belongs to.
    #[inline]
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the flags that the queue was created with.
    #[inline]
    pub fn flags(&self) -> QueueCreateFlags {
        self.flags
    }

    /// Returns the index of the queue family that this queue belongs to.
    #[inline]
    pub fn queue_family_index(&self) -> QueueFamilyIndex {
        self.queue_family_index
    }

    /// Returns the index of this queue within its queue family.
    #[inline]
    pub fn queue_index(&self) -> u32 {
        self.queue_index
    }

    /// Locks the queue and then calls the provided closure, providing it with an object that
    /// can be used to perform operations on the queue, such as command buffer submissions.
    #[inline]
    pub fn with<'a, R>(self: &'a Arc<Self>, func: impl FnOnce(QueueGuard<'a>) -> R) -> R {
        func(QueueGuard {
            queue: self,
            _state: self.state.lock(),
        })
    }
}

impl Drop for Queue {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            let _ = (fns.v1_0.queue_wait_idle)(self.handle);
        }
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
        self.queue_index == other.queue_index
            && self.queue_family_index == other.queue_family_index
            && self.device == other.device
    }
}

impl Eq for Queue {}

impl Hash for Queue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.queue_index.hash(state);
        self.queue_family_index.hash(state);
        self.device.hash(state);
    }
}

/// Parameters to retrieve a [`Queue`] from the device.
#[derive(Clone, Debug)]
pub(super) struct DeviceQueueInfo {
    pub(super) flags: QueueCreateFlags,
    pub(super) queue_family_index: QueueFamilyIndex,
    pub(super) queue_index: u32,
    pub(super) _ne: crate::NonExhaustive,
}

impl Default for DeviceQueueInfo {
    #[inline]
    fn default() -> Self {
        Self {
            flags: QueueCreateFlags::empty(),
            queue_family_index: QueueFamilyIndex(0),
            queue_index: 0,
            _ne: crate::NonExhaustive(()),
        }
    }
}

pub struct QueueGuard<'a> {
    queue: &'a Arc<Queue>,
    _state: MutexGuard<'a, QueueState>,
}

impl<'a> QueueGuard<'a> {
    /// Waits until all work on this queue has finished, then releases ownership of all resources
    /// that were in use by the queue.
    ///
    /// This is equivalent to submitting a fence to the queue, waiting on it, and then calling
    /// `cleanup_finished`.
    ///
    /// Just like [`Device::wait_idle`], you shouldn't have to call this function in a typical
    /// program.
    #[inline]
    pub fn wait_idle(&mut self) -> Result<(), VulkanError> {
        unsafe {
            let fns = self.queue.device.fns();
            (fns.v1_0.queue_wait_idle)(self.queue.handle)
                .result()
                .map_err(VulkanError::from)
        }
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn bind_sparse_unchecked(
        &mut self,
        bind_infos: &[BindSparseInfo],
        fence: Option<&Arc<Fence>>,
    ) -> Result<(), VulkanError> {
        struct PerBindSparseInfo {
            wait_semaphores_vk: SmallVec<[ash::vk::Semaphore; 4]>,
            buffer_bind_infos_vk: SmallVec<[ash::vk::SparseBufferMemoryBindInfo<'static>; 4]>,
            buffer_binds_vk: SmallVec<[SmallVec<[ash::vk::SparseMemoryBind; 4]>; 4]>,
            image_opaque_bind_infos_vk:
                SmallVec<[ash::vk::SparseImageOpaqueMemoryBindInfo<'static>; 4]>,
            image_opaque_binds_vk: SmallVec<[SmallVec<[ash::vk::SparseMemoryBind; 4]>; 4]>,
            image_bind_infos_vk: SmallVec<[ash::vk::SparseImageMemoryBindInfo<'static>; 4]>,
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
                                ash::vk::SparseBufferMemoryBindInfo::default()
                                    .buffer(buffer.buffer().handle()),
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
                            ash::vk::SparseImageOpaqueMemoryBindInfo::default()
                                .image(image.handle()),
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
                                ash::vk::SparseImageMemoryBindInfo::default().image(image.handle()),
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
        ) in bind_infos_vk.iter_mut().zip(per_bind_vk.iter_mut())
        {
            for (buffer_bind_infos_vk, buffer_binds_vk) in
                buffer_bind_infos_vk.iter_mut().zip(buffer_binds_vk.iter())
            {
                *buffer_bind_infos_vk = ash::vk::SparseBufferMemoryBindInfo {
                    bind_count: buffer_binds_vk.len() as u32,
                    p_binds: buffer_binds_vk.as_ptr(),
                    ..*buffer_bind_infos_vk
                };
            }

            for (image_opaque_bind_infos_vk, image_opaque_binds_vk) in image_opaque_bind_infos_vk
                .iter_mut()
                .zip(image_opaque_binds_vk.iter())
            {
                *image_opaque_bind_infos_vk = ash::vk::SparseImageOpaqueMemoryBindInfo {
                    bind_count: image_opaque_binds_vk.len() as u32,
                    p_binds: image_opaque_binds_vk.as_ptr(),
                    ..*image_opaque_bind_infos_vk
                };
            }

            for (image_bind_infos_vk, image_binds_vk) in
                image_bind_infos_vk.iter_mut().zip(image_binds_vk.iter())
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
                .map_or_else(Default::default, VulkanObject::handle),
        )
        .result()
        .map_err(VulkanError::from)
    }

    /// Queues swapchain images for presentation to the surface.
    ///
    /// # Safety
    ///
    /// For every semaphore in the `wait_semaphores` elements of `present_info`:
    /// - The semaphore must be kept alive while the command is being executed.
    /// - The semaphore must be already in the signaled state, or there must be a previously
    ///   submitted operation that will signal it.
    /// - When the wait operation is executed, no other queue must be waiting on the same
    ///   semaphore.
    ///
    /// For every element of `present_info.swapchain_infos`:
    /// - `swapchain` must be kept alive while the command is being executed.
    /// - `image_index` must be an index previously acquired from the swapchain, and the present
    ///   operation must happen-after the acquire operation.
    /// - The swapchain image indicated by `swapchain` and `image_index` must be in the
    ///   [`ImageLayout::PresentSrc`] layout when the presentation operation is executed.
    /// - The swapchain image indicated by `swapchain` and `image_index` must not be accessed after
    ///   this function is called, until it is acquired again.
    /// - If `present_id` is `Some`, then it must be greater than any present ID previously used
    ///   for the same swapchain.
    ///
    /// [`ImageLayout::PresentSrc`]: crate::image::ImageLayout::PresentSrc
    #[inline]
    pub unsafe fn present(
        &mut self,
        present_info: &PresentInfo,
    ) -> Result<impl ExactSizeIterator<Item = Result<bool, VulkanError>>, Validated<VulkanError>>
    {
        self.validate_present(present_info)?;

        Ok(self.present_unchecked(present_info)?)
    }

    fn validate_present(&self, present_info: &PresentInfo) -> Result<(), Box<ValidationError>> {
        let device = self.queue.device();

        if !device.enabled_extensions().khr_swapchain {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_swapchain",
                )])]),
                ..Default::default()
            }));
        }

        present_info
            .validate(device)
            .map_err(|err| err.add_context("present_info"))?;

        let &PresentInfo {
            wait_semaphores: _,
            ref swapchains,
            _ne: _,
        } = present_info;

        for (index, swapchain_info) in swapchains.iter().enumerate() {
            let &SwapchainPresentInfo {
                ref swapchain,
                image_index: _,
                present_id: _,
                present_mode: _,
                present_regions: _,
                _ne: _,
            } = swapchain_info;

            if unsafe {
                !device
                    .physical_device
                    .surface_support_unchecked(self.queue.queue_family_index, swapchain.surface())
                    .unwrap_or_default()
            } {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "present_info.swapchain_infos[{}].swapchain.surface()",
                        index
                    )
                    .into(),
                    problem: "the queue family of this queue does not support presenting to \
                        the surface"
                        .into(),
                    vuids: &["VUID-vkQueuePresentKHR-pSwapchains-01292"],
                    ..Default::default()
                }));
            }
        }

        // unsafe
        // VUID-vkQueuePresentKHR-pWaitSemaphores-01294
        // VUID-vkQueuePresentKHR-pWaitSemaphores-03268

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn present_unchecked(
        &mut self,
        present_info: &PresentInfo,
    ) -> Result<impl ExactSizeIterator<Item = Result<bool, VulkanError>>, VulkanError> {
        let PresentInfo {
            wait_semaphores,
            swapchains,
            _ne: _,
        } = present_info;

        let wait_semaphores_vk: SmallVec<[_; 4]> = wait_semaphores
            .iter()
            .map(|semaphore_present_info| {
                let &SemaphorePresentInfo {
                    ref semaphore,
                    _ne: _,
                } = semaphore_present_info;

                semaphore.handle()
            })
            .collect();

        let mut swapchains_vk: SmallVec<[_; 4]> = SmallVec::with_capacity(swapchains.len());
        let mut image_indices_vk: SmallVec<[_; 4]> = SmallVec::with_capacity(swapchains.len());
        let mut present_ids_vk: SmallVec<[_; 4]> = SmallVec::with_capacity(swapchains.len());
        let mut present_modes_vk: SmallVec<[_; 4]> = SmallVec::with_capacity(swapchains.len());
        let mut rectangles_vk: SmallVec<[_; 4]> = SmallVec::with_capacity(swapchains.len());
        let mut present_regions_vk: SmallVec<[_; 4]> = SmallVec::with_capacity(swapchains.len());

        let mut has_present_ids = false;
        let mut has_present_modes = false;
        let mut has_present_regions = false;

        for swapchain_info in swapchains {
            let &SwapchainPresentInfo {
                ref swapchain,
                image_index,
                present_id,
                present_mode,
                ref present_regions,
                _ne: _,
            } = swapchain_info;

            swapchains_vk.push(swapchain.handle());
            image_indices_vk.push(image_index);
            present_ids_vk.push(present_id.map_or(0, u64::from));
            present_modes_vk.push(present_mode.map_or_else(Default::default, Into::into));
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

            if present_mode.is_some() {
                has_present_modes = true;
            }

            if !present_regions.is_empty() {
                has_present_regions = true;
            }
        }

        let mut results = vec![ash::vk::Result::SUCCESS; swapchains.len()];
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
        let mut present_mode_info_vk = None;
        let mut present_region_info_vk = None;

        if has_present_ids {
            let next = present_id_info_vk.insert(ash::vk::PresentIdKHR {
                swapchain_count: present_ids_vk.len() as u32,
                p_present_ids: present_ids_vk.as_ptr(),
                ..Default::default()
            });

            next.p_next = info_vk.p_next;
            info_vk.p_next = <*const _>::cast(next);
        }

        if has_present_modes {
            let next = present_mode_info_vk.insert(ash::vk::SwapchainPresentModeInfoEXT {
                swapchain_count: present_modes_vk.len() as u32,
                p_present_modes: present_modes_vk.as_ptr(),
                ..Default::default()
            });

            next.p_next = info_vk.p_next.cast();
            info_vk.p_next = <*const _>::cast(next);
        }

        if has_present_regions {
            for (present_regions_vk, rectangles_vk) in
                present_regions_vk.iter_mut().zip(rectangles_vk.iter())
            {
                *present_regions_vk =
                    ash::vk::PresentRegionKHR::default().rectangles(rectangles_vk);
            }

            let next = present_region_info_vk.insert(ash::vk::PresentRegionsKHR {
                swapchain_count: present_regions_vk.len() as u32,
                p_regions: present_regions_vk.as_ptr(),
                ..Default::default()
            });

            next.p_next = info_vk.p_next;
            info_vk.p_next = <*const _>::cast(next);
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

        Ok(results.into_iter().map(|result| match result {
            ash::vk::Result::SUCCESS => Ok(false),
            ash::vk::Result::SUBOPTIMAL_KHR => Ok(true),
            err => Err(VulkanError::from(err)),
        }))
    }

    /// Submits command buffers to a queue to be executed.
    ///
    /// # Safety
    ///
    /// For every semaphore in the `wait_semaphores` elements of every `submit_infos` element:
    /// - The semaphore must be kept alive while the command is being executed.
    /// - The safety requirements for semaphores, as detailed in the
    ///   [`semaphore`](crate::sync::semaphore#Safety) module documentation, must be followed.
    ///
    /// For every command buffer in the `command_buffers` elements of every `submit_infos` element,
    /// as well as any secondary command buffers recorded within it:
    /// - The command buffer, and any resources it uses, must be kept alive while the command is
    ///   being executed.
    /// - Any mutable resources used by the command buffer must be in the state (image layout,
    ///   query reset, event signal etc.) expected by the command buffer at the time it begins
    ///   executing, and must be synchronized appropriately.
    /// - If the command buffer's `usage` is [`CommandBufferUsage::OneTimeSubmit`], then it must
    ///   not have been previously submitted.
    /// - If the command buffer's `usage` is [`CommandBufferUsage::MultipleSubmit`], then it must
    ///   not be currently submitted and not yet completed.
    /// - If a recorded command performs a queue family transfer acquire operation, then a
    ///   corresponding queue family transfer release operation with matching parameters must have
    ///   been previously submitted, and must happen-before it.
    /// - If a recorded command references an [`Event`], then that `Event` must not be referenced
    ///   by a command that is currently executing on another queue.
    ///
    /// For every semaphore in the `signal_semaphores` elements of every `submit_infos` element:
    /// - The semaphore must be kept alive while the command is being executed.
    /// - The safety requirements for semaphores, as detailed in the
    ///   [`semaphore`](crate::sync::semaphore#Safety) module documentation, must be followed.
    ///
    /// If `fence` is `Some`:
    /// - The fence must be kept alive while the command is being executed.
    /// - The safety requirements for fences, as detailed in the
    ///   [`fence`](crate::sync::fence#Safety) module documentation, must be followed.
    ///
    /// [`CommandBufferUsage::OneTimeSubmit`]: crate::command_buffer::CommandBufferUsage::OneTimeSubmit
    /// [`CommandBufferUsage::MultipleSubmit`]: crate::command_buffer::CommandBufferUsage::MultipleSubmit
    /// [`Event`]: crate::sync::event::Event
    #[inline]
    pub unsafe fn submit(
        &mut self,
        submit_infos: &[SubmitInfo],
        fence: Option<&Arc<Fence>>,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_submit(submit_infos, fence)?;

        Ok(self.submit_unchecked(submit_infos, fence)?)
    }

    fn validate_submit(
        &self,
        submit_infos: &[SubmitInfo],
        fence: Option<&Arc<Fence>>,
    ) -> Result<(), Box<ValidationError>> {
        let device = self.queue.device();
        let queue_family_properties = &device.physical_device().queue_family_properties()
            [self.queue.queue_family_index.0 as usize];

        if let Some(fence) = fence {
            // VUID-vkQueueSubmit2-commonparent
            assert_eq!(device, fence.device());
        }

        let supported_stages = PipelineStages::from(queue_family_properties.queue_flags);

        for (index, submit_info) in submit_infos.iter().enumerate() {
            submit_info
                .validate(device)
                .map_err(|err| err.add_context(format!("submit_infos[{}]", index)))?;

            let &SubmitInfo {
                ref wait_semaphores,
                ref command_buffers,
                ref signal_semaphores,
                _ne: _,
            } = submit_info;

            for (semaphore_index, semaphore_submit_info) in wait_semaphores.iter().enumerate() {
                let &SemaphoreSubmitInfo {
                    semaphore: _,
                    value: _,
                    stages,
                    _ne: _,
                } = semaphore_submit_info;

                if !supported_stages.contains(stages) {
                    return Err(Box::new(ValidationError {
                        context: format!(
                            "submit_infos[{}].wait_semaphores[{}].stages",
                            index, semaphore_index
                        )
                        .into(),
                        problem: "contains stages that are not supported by the queue family of \
                            the queue"
                            .into(),
                        vuids: &["VUID-vkQueueSubmit2-stageMask-03870"],
                        ..Default::default()
                    }));
                }
            }

            for (command_buffer_index, command_buffer_submit_info) in
                command_buffers.iter().enumerate()
            {
                let &CommandBufferSubmitInfo {
                    ref command_buffer,
                    _ne: _,
                } = command_buffer_submit_info;

                if command_buffer.queue_family_index() != self.queue.queue_family_index {
                    return Err(Box::new(ValidationError {
                        context: format!(
                            "submit_infos[{}].command_buffers[{}]\
                            .command_buffer.queue_family_index()",
                            index, command_buffer_index
                        )
                        .into(),
                        problem: "does not equal the queue family index of this queue".into(),
                        vuids: &["VUID-vkQueueSubmit2-commandBuffer-03878"],
                        ..Default::default()
                    }));
                }
            }

            for (semaphore_index, semaphore_submit_info) in signal_semaphores.iter().enumerate() {
                let &SemaphoreSubmitInfo {
                    semaphore: _,
                    value: _,
                    stages,
                    _ne: _,
                } = semaphore_submit_info;

                if !supported_stages.contains(stages) {
                    return Err(Box::new(ValidationError {
                        context: format!(
                            "submit_infos[{}].signal_semaphores[{}].stages",
                            index, semaphore_index
                        )
                        .into(),
                        problem: "contains stages that are not supported by the queue family of \
                            the queue"
                            .into(),
                        vuids: &["VUID-vkQueueSubmit2-stageMask-03870"],
                        ..Default::default()
                    }));
                }
            }
        }

        // unsafe
        // VUID-VkSubmitInfo2-semaphore-03882
        // VUID-VkSubmitInfo2-semaphore-03883
        // VUID-VkSubmitInfo2-semaphore-03884
        // VUID-vkQueueSubmit2-fence-04894
        // VUID-vkQueueSubmit2-fence-04895
        // VUID-vkQueueSubmit2-commandBuffer-03867
        // VUID-vkQueueSubmit2-semaphore-03868
        // VUID-vkQueueSubmit2-semaphore-03871
        // VUID-vkQueueSubmit2-semaphore-03873
        // VUID-vkQueueSubmit2-commandBuffer-03874
        // VUID-vkQueueSubmit2-commandBuffer-03875
        // VUID-vkQueueSubmit2-commandBuffer-03876
        // VUID-vkQueueSubmit2-commandBuffer-03877
        // VUID-vkQueueSubmit2-commandBuffer-03879

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn submit_unchecked(
        &mut self,
        submit_infos: &[SubmitInfo],
        fence: Option<&Arc<Fence>>,
    ) -> Result<(), VulkanError> {
        if self.queue.device.enabled_features().synchronization2 {
            struct PerSubmitInfo {
                wait_semaphore_infos_vk: SmallVec<[ash::vk::SemaphoreSubmitInfo<'static>; 4]>,
                command_buffer_infos_vk: SmallVec<[ash::vk::CommandBufferSubmitInfo<'static>; 4]>,
                signal_semaphore_infos_vk: SmallVec<[ash::vk::SemaphoreSubmitInfo<'static>; 4]>,
            }

            let (mut submit_info_vk, mut per_submit_vk): (SmallVec<[_; 4]>, SmallVec<[_; 4]>) =
                submit_infos
                    .iter()
                    .map(|submit_info| {
                        let &SubmitInfo {
                            ref wait_semaphores,
                            ref command_buffers,
                            ref signal_semaphores,
                            _ne: _,
                        } = submit_info;

                        let mut per_submit_vk = PerSubmitInfo {
                            wait_semaphore_infos_vk: SmallVec::with_capacity(wait_semaphores.len()),
                            command_buffer_infos_vk: SmallVec::with_capacity(command_buffers.len()),
                            signal_semaphore_infos_vk: SmallVec::with_capacity(
                                signal_semaphores.len(),
                            ),
                        };
                        let PerSubmitInfo {
                            wait_semaphore_infos_vk,
                            command_buffer_infos_vk,
                            signal_semaphore_infos_vk,
                        } = &mut per_submit_vk;

                        for semaphore_submit_info in wait_semaphores {
                            let &SemaphoreSubmitInfo {
                                ref semaphore,
                                value,
                                stages,
                                _ne: _,
                            } = semaphore_submit_info;

                            wait_semaphore_infos_vk.push(ash::vk::SemaphoreSubmitInfo {
                                semaphore: semaphore.handle(),
                                value,
                                stage_mask: stages.into(),
                                device_index: 0, // TODO:
                                ..Default::default()
                            });
                        }

                        for command_buffer_submit_info in command_buffers {
                            let &CommandBufferSubmitInfo {
                                ref command_buffer,
                                _ne: _,
                            } = command_buffer_submit_info;

                            command_buffer_infos_vk.push(ash::vk::CommandBufferSubmitInfo {
                                command_buffer: command_buffer.handle(),
                                device_mask: 0, // TODO:
                                ..Default::default()
                            });
                        }

                        for semaphore_submit_info in signal_semaphores {
                            let &SemaphoreSubmitInfo {
                                ref semaphore,
                                value,
                                stages,
                                _ne: _,
                            } = semaphore_submit_info;

                            signal_semaphore_infos_vk.push(ash::vk::SemaphoreSubmitInfo {
                                semaphore: semaphore.handle(),
                                value,
                                stage_mask: stages.into(),
                                device_index: 0, // TODO:
                                ..Default::default()
                            });
                        }

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
                            per_submit_vk,
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
            ) in submit_info_vk.iter_mut().zip(per_submit_vk.iter_mut())
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
                        .map_or_else(Default::default, VulkanObject::handle),
                )
            } else {
                debug_assert!(self.queue.device.enabled_extensions().khr_synchronization2);
                (fns.khr_synchronization2.queue_submit2_khr)(
                    self.queue.handle,
                    submit_info_vk.len() as u32,
                    submit_info_vk.as_ptr(),
                    fence
                        .as_ref()
                        .map_or_else(Default::default, VulkanObject::handle),
                )
            }
            .result()
            .map_err(VulkanError::from)
        } else {
            struct PerSubmitInfo {
                timeline_semaphore_submit_info_vk:
                    Option<ash::vk::TimelineSemaphoreSubmitInfo<'static>>,
                wait_semaphores_vk: SmallVec<[ash::vk::Semaphore; 4]>,
                wait_semaphore_values_vk: SmallVec<[u64; 4]>,
                wait_dst_stage_mask_vk: SmallVec<[ash::vk::PipelineStageFlags; 4]>,
                command_buffers_vk: SmallVec<[ash::vk::CommandBuffer; 4]>,
                signal_semaphores_vk: SmallVec<[ash::vk::Semaphore; 4]>,
                signal_semaphore_values_vk: SmallVec<[u64; 4]>,
            }

            let (mut submit_info_vk, mut per_submit_vk): (SmallVec<[_; 4]>, SmallVec<[_; 4]>) =
                submit_infos
                    .iter()
                    .map(|submit_info| {
                        let &SubmitInfo {
                            ref wait_semaphores,
                            ref command_buffers,
                            ref signal_semaphores,
                            _ne: _,
                        } = submit_info;

                        let mut per_submit_vk = PerSubmitInfo {
                            timeline_semaphore_submit_info_vk: None,
                            wait_semaphores_vk: SmallVec::with_capacity(wait_semaphores.len()),
                            wait_semaphore_values_vk: SmallVec::with_capacity(
                                wait_semaphores.len(),
                            ),
                            wait_dst_stage_mask_vk: SmallVec::with_capacity(wait_semaphores.len()),
                            command_buffers_vk: SmallVec::with_capacity(command_buffers.len()),
                            signal_semaphores_vk: SmallVec::with_capacity(signal_semaphores.len()),
                            signal_semaphore_values_vk: SmallVec::with_capacity(
                                signal_semaphores.len(),
                            ),
                        };
                        let PerSubmitInfo {
                            timeline_semaphore_submit_info_vk,
                            wait_semaphores_vk,
                            wait_semaphore_values_vk,
                            wait_dst_stage_mask_vk,
                            command_buffers_vk,
                            signal_semaphores_vk,
                            signal_semaphore_values_vk,
                        } = &mut per_submit_vk;

                        let mut has_timeline_semaphores = false;

                        for semaphore_submit_info in wait_semaphores {
                            let &SemaphoreSubmitInfo {
                                ref semaphore,
                                value,
                                stages,
                                _ne: _,
                            } = semaphore_submit_info;

                            if semaphore.semaphore_type() == SemaphoreType::Timeline {
                                has_timeline_semaphores = true;
                            }

                            wait_semaphores_vk.push(semaphore.handle());
                            wait_semaphore_values_vk.push(value);
                            wait_dst_stage_mask_vk.push(stages.into());
                        }

                        for command_buffer_submit_info in command_buffers {
                            let &CommandBufferSubmitInfo {
                                ref command_buffer,
                                _ne: _,
                            } = command_buffer_submit_info;

                            command_buffers_vk.push(command_buffer.handle());
                        }

                        for semaphore_submit_info in signal_semaphores {
                            let &SemaphoreSubmitInfo {
                                ref semaphore,
                                value,
                                stages: _,
                                _ne: _,
                            } = semaphore_submit_info;

                            if semaphore.semaphore_type() == SemaphoreType::Timeline {
                                has_timeline_semaphores = true;
                            }

                            signal_semaphores_vk.push(semaphore.handle());
                            signal_semaphore_values_vk.push(value);
                        }

                        if has_timeline_semaphores {
                            *timeline_semaphore_submit_info_vk =
                                Some(ash::vk::TimelineSemaphoreSubmitInfo::default());
                        }

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
                            per_submit_vk,
                        )
                    })
                    .unzip();

            for (
                submit_info_vk,
                PerSubmitInfo {
                    timeline_semaphore_submit_info_vk,
                    wait_semaphores_vk,
                    wait_semaphore_values_vk,
                    wait_dst_stage_mask_vk,
                    command_buffers_vk,
                    signal_semaphores_vk,
                    signal_semaphore_values_vk,
                },
            ) in submit_info_vk.iter_mut().zip(per_submit_vk.iter_mut())
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

                if let Some(timeline_semaphore_submit_info_vk) = timeline_semaphore_submit_info_vk {
                    *timeline_semaphore_submit_info_vk = ash::vk::TimelineSemaphoreSubmitInfo {
                        wait_semaphore_value_count: wait_semaphore_values_vk.len() as u32,
                        p_wait_semaphore_values: wait_semaphore_values_vk.as_ptr(),
                        signal_semaphore_value_count: signal_semaphore_values_vk.len() as u32,
                        p_signal_semaphore_values: signal_semaphore_values_vk.as_ptr(),
                        ..*timeline_semaphore_submit_info_vk
                    };

                    timeline_semaphore_submit_info_vk.p_next = submit_info_vk.p_next;
                    submit_info_vk.p_next = <*mut _>::cast(timeline_semaphore_submit_info_vk);
                }
            }

            let fns = self.queue.device.fns();
            (fns.v1_0.queue_submit)(
                self.queue.handle,
                submit_info_vk.len() as u32,
                submit_info_vk.as_ptr(),
                fence
                    .as_ref()
                    .map_or_else(Default::default, VulkanObject::handle),
            )
            .result()
            .map_err(VulkanError::from)
        }
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
    ) -> Result<(), Box<ValidationError>> {
        self.validate_begin_debug_utils_label(&label_info)?;

        unsafe {
            self.begin_debug_utils_label_unchecked(label_info);
            Ok(())
        }
    }

    fn validate_begin_debug_utils_label(
        &self,
        _label_info: &DebugUtilsLabel,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue
            .device
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_debug_utils",
                )])]),
                ..Default::default()
            }));
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

        let fns = self.queue.device.fns();
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
    pub unsafe fn end_debug_utils_label(&mut self) -> Result<(), Box<ValidationError>> {
        self.validate_end_debug_utils_label()?;
        self.end_debug_utils_label_unchecked();

        Ok(())
    }

    fn validate_end_debug_utils_label(&self) -> Result<(), Box<ValidationError>> {
        if !self
            .queue
            .device
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_debug_utils",
                )])]),
                ..Default::default()
            }));
        }

        // VUID-vkQueueEndDebugUtilsLabelEXT-None-01911
        // TODO: not checked, so unsafe for now

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    #[inline]
    pub unsafe fn end_debug_utils_label_unchecked(&mut self) {
        let fns = self.queue.device.fns();
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
    ) -> Result<(), Box<ValidationError>> {
        self.validate_insert_debug_utils_label(&label_info)?;

        unsafe {
            self.insert_debug_utils_label_unchecked(label_info);
            Ok(())
        }
    }

    fn validate_insert_debug_utils_label(
        &self,
        _label_info: &DebugUtilsLabel,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue
            .device
            .instance()
            .enabled_extensions()
            .ext_debug_utils
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::InstanceExtension(
                    "ext_debug_utils",
                )])]),
                ..Default::default()
            }));
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

        let fns = self.queue.device.fns();
        (fns.ext_debug_utils.queue_insert_debug_utils_label_ext)(self.queue.handle, &label_info);
    }
}

#[derive(Debug, Default)]
struct QueueState {}

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

    /// The minimum granularity supported for image transfers, in terms of `[width, height,
    /// depth]`.
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
    PROTECTED = PROTECTED
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_1)]),
    ]),

    /// Queues of this family can execute video decode operations.
    VIDEO_DECODE = VIDEO_DECODE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_decode_queue)]),
    ]),

    /// Queues of this family can execute video encode operations.
    VIDEO_ENCODE = VIDEO_ENCODE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_video_encode_queue)]),
    ]),

    /// Queues of this family can execute optical flow operations.
    OPTICAL_FLOW = OPTICAL_FLOW_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_optical_flow)]),
    ]),
}

#[cfg(test)]
mod tests {
    use crate::sync::fence::Fence;
    use std::{sync::Arc, time::Duration};

    #[test]
    fn empty_submit() {
        let (_device, queue) = gfx_dev_and_queue!();

        queue
            .with(|mut q| unsafe { q.submit(&[Default::default()], None) })
            .unwrap();
    }

    #[test]
    fn signal_fence() {
        unsafe {
            let (device, queue) = gfx_dev_and_queue!();

            let fence = Arc::new(Fence::new(device, Default::default()).unwrap());
            assert!(!fence.is_signaled().unwrap());

            queue
                .with(|mut q| q.submit(&[Default::default()], Some(&fence)))
                .unwrap();

            fence.wait(Some(Duration::from_secs(5))).unwrap();
            assert!(fence.is_signaled().unwrap());
        }
    }
}
