use super::{
    BarrierIndex, ExecutableTaskGraph, Instruction, NodeIndex, ResourceAccess, SemaphoreIndex,
};
use crate::{
    command_buffer::RecordingCommandBuffer,
    resource::{
        BufferAccess, BufferState, DeathRow, ImageAccess, ImageState, Resources, SwapchainState,
    },
    Id, InvalidSlotError, ObjectType, TaskContext, TaskError,
};
use ash::vk;
use concurrent_slotmap::epoch;
use smallvec::SmallVec;
use std::{
    error::Error,
    fmt, mem,
    ops::Range,
    ptr,
    sync::{atomic::Ordering, Arc},
};
use vulkano::{
    buffer::{Buffer, BufferMemory},
    command_buffer as raw,
    device::{Device, DeviceOwned, Queue},
    image::Image,
    swapchain::{AcquireNextImageInfo, AcquiredImage, Swapchain},
    sync::{
        fence::{Fence, FenceCreateFlags, FenceCreateInfo},
        semaphore::Semaphore,
        AccessFlags, PipelineStages,
    },
    Validated, Version, VulkanError, VulkanObject,
};

impl<W: ?Sized + 'static> ExecutableTaskGraph<W> {
    /// Executes the next frame of the [flight] given by `flight_id`.
    ///
    /// # Safety
    ///
    /// - There must be no other task graphs executing that access any of the same subresources as
    ///   `self`.
    /// - A subresource in flight must not be accessed in more than one frame in flight.
    ///
    /// # Panics
    ///
    /// - Panics if `resource_map` doesn't map the virtual resources of `self` exhaustively.
    /// - Panics if `self.flight_id()` is invalid.
    /// - Panics if another thread is already executing a task graph using the flight.
    /// - Panics if `resource_map` maps to any swapchain that isn't owned by the flight.
    /// - Panics if the oldest frame of the flight wasn't [waited] on.
    ///
    /// [waited]: crate::resource::Flight::wait
    pub unsafe fn execute(
        &self,
        resource_map: ResourceMap<'_>,
        world: &W,
        pre_present_notify: impl FnOnce(),
    ) -> Result {
        assert!(ptr::eq(
            resource_map.virtual_resources,
            &self.graph.resources,
        ));
        assert!(resource_map.is_exhaustive());

        let flight_id = self.flight_id;

        // SAFETY: `resource_map` owns an `epoch::Guard`.
        let flight = unsafe {
            resource_map
                .physical_resources
                .flight_unprotected(flight_id)
        }
        .expect("invalid flight");

        let mut flight_state = flight.state.try_lock().unwrap_or_else(|| {
            panic!(
                "another thread is already executing a task graph using the flight {flight_id:?}",
            );
        });

        // TODO: This call is quite expensive.
        assert!(
            flight.current_fence().read().is_signaled()?,
            "you must wait on the fence for the current frame before submitting more work",
        );

        for &swapchain_id in &self.swapchains {
            // SAFETY: We checked that `resource_map` maps the virtual IDs exhaustively.
            let swapchain_state = unsafe { resource_map.swapchain_unchecked(swapchain_id) };

            assert_eq!(
                swapchain_state.flight_id(),
                flight_id,
                "`resource_map` must not map to any swapchain not owned by the flight \
                corresponding to `flight_id`",
            );
        }

        let current_frame_index = flight.current_frame_index();
        let death_row = &mut flight_state.death_rows[current_frame_index as usize];

        for object in death_row.drain(..) {
            // FIXME:
            drop(object);
        }

        // SAFETY: We checked that `resource_map` maps the virtual IDs exhaustively.
        unsafe { self.acquire_images_khr(&resource_map, current_frame_index) }?;

        let mut current_fence = flight.current_fence().write();

        // SAFETY: We checked that the fence has been signalled.
        unsafe { current_fence.reset_unchecked() }?;

        // SAFETY: We checked that `resource_map` maps the virtual IDs exhaustively.
        unsafe { self.invalidate_mapped_memory_ranges(&resource_map) }?;

        let mut state_guard = StateGuard {
            executable: self,
            resource_map: &resource_map,
            current_fence: &mut current_fence,
            submission_count: 0,
        };

        let execute_instructions = if self.device().enabled_features().synchronization2 {
            Self::execute_instructions2
        } else {
            Self::execute_instructions
        };

        // SAFETY: We checked that `resource_map` maps the virtual IDs exhaustively.
        unsafe {
            execute_instructions(
                self,
                &resource_map,
                death_row,
                current_frame_index,
                state_guard.current_fence,
                &mut state_guard.submission_count,
                world,
            )
        }?;

        mem::forget(state_guard);

        for semaphore in self.semaphores.borrow().iter() {
            death_row.push(semaphore.clone());
        }

        unsafe { flight.next_frame() };

        pre_present_notify();

        // SAFETY: We checked that `resource_map` maps the virtual IDs exhaustively.
        let res = unsafe { self.present_images_khr(&resource_map, current_frame_index) };

        // SAFETY: We checked that `resource_map` maps the virtual IDs exhaustively.
        unsafe { self.update_resource_state(&resource_map, &self.last_accesses) };

        resource_map
            .physical_resources
            .try_advance_global_and_collect(&resource_map.guard);

        res
    }

    unsafe fn acquire_images_khr(
        &self,
        resource_map: &ResourceMap<'_>,
        current_frame_index: u32,
    ) -> Result {
        for &swapchain_id in &self.swapchains {
            // SAFETY: The caller must ensure that `resource_map` maps the virtual IDs exhaustively.
            let swapchain_state = unsafe { resource_map.swapchain_unchecked(swapchain_id) };
            let semaphore =
                &swapchain_state.semaphores[current_frame_index as usize].image_available_semaphore;

            // Make sure to not acquire another image index if we already acquired one. This can
            // happen when using multiple swapchains, if one acquire succeeds and another fails, or
            // when executing a submission or presenting an image fails.
            if swapchain_state.current_image_index.load(Ordering::Relaxed) != u32::MAX {
                continue;
            }

            let res = unsafe {
                swapchain_state
                    .swapchain()
                    .acquire_next_image(&AcquireNextImageInfo {
                        semaphore: Some(semaphore.clone()),
                        ..Default::default()
                    })
            };

            match res {
                Ok(AcquiredImage { image_index, .. }) => {
                    swapchain_state
                        .current_image_index
                        .store(image_index, Ordering::Relaxed);
                }
                Err(error) => {
                    swapchain_state
                        .current_image_index
                        .store(u32::MAX, Ordering::Relaxed);
                    return Err(ExecuteError::Swapchain {
                        swapchain_id,
                        error,
                    });
                }
            }
        }

        Ok(())
    }

    unsafe fn invalidate_mapped_memory_ranges(&self, resource_map: &ResourceMap<'_>) -> Result {
        let mut mapped_memory_ranges = Vec::new();

        for &buffer_id in &self.graph.resources.host_reads {
            // SAFETY: The caller must ensure that `resource_map` maps the virtual IDs exhaustively.
            let buffer = unsafe { resource_map.buffer_unchecked(buffer_id) }.buffer();

            let allocation = match buffer.memory() {
                BufferMemory::Normal(a) => a,
                BufferMemory::Sparse => todo!("`TaskGraph` doesn't support sparse binding yet"),
                BufferMemory::External => continue,
                _ => unreachable!(),
            };

            if allocation.atom_size().is_none() {
                continue;
            }

            if unsafe { allocation.mapped_slice_unchecked(..) }.is_err() {
                continue;
            }

            // This works because the memory allocator must align allocations to the non-coherent
            // atom size when the memory is host-visible but not host-coherent.
            mapped_memory_ranges.push(
                vk::MappedMemoryRange::default()
                    .memory(allocation.device_memory().handle())
                    .offset(allocation.offset())
                    .size(allocation.size()),
            );
        }

        if !mapped_memory_ranges.is_empty() {
            let fns = self.device().fns();
            unsafe {
                (fns.v1_0.invalidate_mapped_memory_ranges)(
                    self.device().handle(),
                    mapped_memory_ranges.len() as u32,
                    mapped_memory_ranges.as_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
        }

        Ok(())
    }

    unsafe fn execute_instructions2(
        &self,
        resource_map: &ResourceMap<'_>,
        death_row: &mut DeathRow,
        current_frame_index: u32,
        current_fence: &Fence,
        submission_count: &mut usize,
        world: &W,
    ) -> Result {
        let mut state = ExecuteState2::new(
            self,
            resource_map,
            death_row,
            current_frame_index,
            current_fence,
            submission_count,
            world,
        )?;
        let mut execute_initial_barriers = true;

        for instruction in self.instructions.iter().cloned() {
            if execute_initial_barriers {
                let submission = current_submission!(state);
                state.initial_pipeline_barrier(
                    submission.initial_buffer_barrier_range.clone(),
                    submission.initial_image_barrier_range.clone(),
                );
                execute_initial_barriers = false;
            }

            match instruction {
                Instruction::WaitAcquire {
                    swapchain_id,
                    stage_mask,
                } => {
                    state.wait_acquire(swapchain_id, stage_mask);
                }
                Instruction::WaitSemaphore {
                    semaphore_index,
                    stage_mask,
                } => {
                    state.wait_semaphore(semaphore_index, stage_mask);
                }
                Instruction::ExecuteTask { node_index } => {
                    state.execute_task(node_index)?;
                }
                Instruction::PipelineBarrier {
                    buffer_barrier_range,
                    image_barrier_range,
                } => {
                    state.pipeline_barrier(buffer_barrier_range, image_barrier_range)?;
                }
                Instruction::SignalSemaphore {
                    semaphore_index,
                    stage_mask,
                } => {
                    state.signal_semaphore(semaphore_index, stage_mask);
                }
                Instruction::SignalPrePresent {
                    swapchain_id,
                    stage_mask,
                } => {
                    state.signal_pre_present(swapchain_id, stage_mask);
                }
                Instruction::WaitPrePresent {
                    swapchain_id,
                    stage_mask,
                } => {
                    state.wait_pre_present(swapchain_id, stage_mask);
                }
                Instruction::SignalPresent {
                    swapchain_id,
                    stage_mask,
                } => {
                    state.signal_present(swapchain_id, stage_mask);
                }
                Instruction::FlushSubmit => {
                    state.flush_submit()?;
                }
                Instruction::Submit => {
                    state.submit()?;
                    execute_initial_barriers = true;
                }
            }
        }

        Ok(())
    }

    unsafe fn execute_instructions(
        &self,
        resource_map: &ResourceMap<'_>,
        death_row: &mut DeathRow,
        current_frame_index: u32,
        current_fence: &Fence,
        submission_count: &mut usize,
        world: &W,
    ) -> Result {
        let mut state = ExecuteState::new(
            self,
            resource_map,
            death_row,
            current_frame_index,
            current_fence,
            submission_count,
            world,
        )?;
        let mut execute_initial_barriers = true;

        for instruction in self.instructions.iter().cloned() {
            if execute_initial_barriers {
                let submission = current_submission!(state);
                state.initial_pipeline_barrier(
                    submission.initial_buffer_barrier_range.clone(),
                    submission.initial_image_barrier_range.clone(),
                );
                execute_initial_barriers = false;
            }

            match instruction {
                Instruction::WaitAcquire {
                    swapchain_id,
                    stage_mask,
                } => {
                    state.wait_acquire(swapchain_id, stage_mask);
                }
                Instruction::WaitSemaphore {
                    semaphore_index,
                    stage_mask,
                } => {
                    state.wait_semaphore(semaphore_index, stage_mask);
                }
                Instruction::ExecuteTask { node_index } => {
                    state.execute_task(node_index)?;
                }
                Instruction::PipelineBarrier {
                    buffer_barrier_range,
                    image_barrier_range,
                } => {
                    state.pipeline_barrier(buffer_barrier_range, image_barrier_range)?;
                }
                Instruction::SignalSemaphore {
                    semaphore_index,
                    stage_mask,
                } => {
                    state.signal_semaphore(semaphore_index, stage_mask);
                }
                Instruction::SignalPrePresent {
                    swapchain_id,
                    stage_mask,
                } => {
                    state.signal_pre_present(swapchain_id, stage_mask);
                }
                Instruction::WaitPrePresent {
                    swapchain_id,
                    stage_mask,
                } => {
                    state.wait_pre_present(swapchain_id, stage_mask);
                }
                Instruction::SignalPresent {
                    swapchain_id,
                    stage_mask,
                } => {
                    state.signal_present(swapchain_id, stage_mask);
                }
                Instruction::FlushSubmit => {
                    state.flush_submit()?;
                }
                Instruction::Submit => {
                    state.submit()?;
                    execute_initial_barriers = true;
                }
            }
        }

        Ok(())
    }

    unsafe fn flush_mapped_memory_ranges(&self, resource_map: &ResourceMap<'_>) -> Result {
        let mut mapped_memory_ranges = Vec::new();

        for &buffer_id in &self.graph.resources.host_writes {
            // SAFETY: The caller must ensure that `resource_map` maps the virtual IDs exhaustively.
            let buffer = unsafe { resource_map.buffer_unchecked(buffer_id) }.buffer();

            let allocation = match buffer.memory() {
                BufferMemory::Normal(a) => a,
                BufferMemory::Sparse => todo!("`TaskGraph` doesn't support sparse binding yet"),
                BufferMemory::External => continue,
                _ => unreachable!(),
            };

            if allocation.atom_size().is_none() {
                continue;
            }

            if unsafe { allocation.mapped_slice_unchecked(..) }.is_err() {
                continue;
            }

            // This works because the memory allocator must align allocations to the non-coherent
            // atom size when the memory is host-visible but not host-coherent.
            mapped_memory_ranges.push(
                vk::MappedMemoryRange::default()
                    .memory(allocation.device_memory().handle())
                    .offset(allocation.offset())
                    .size(allocation.size()),
            );
        }

        if !mapped_memory_ranges.is_empty() {
            let fns = self.device().fns();
            unsafe {
                (fns.v1_0.flush_mapped_memory_ranges)(
                    self.device().handle(),
                    mapped_memory_ranges.len() as u32,
                    mapped_memory_ranges.as_ptr(),
                )
            }
            .result()
            .map_err(VulkanError::from)?;
        }

        Ok(())
    }

    unsafe fn present_images_khr(
        &self,
        resource_map: &ResourceMap<'_>,
        current_frame_index: u32,
    ) -> Result {
        let Some(present_queue) = &self.present_queue else {
            return Ok(());
        };

        let swapchain_count = self.swapchains.len();
        let mut semaphores = SmallVec::<[_; 1]>::with_capacity(swapchain_count);
        let mut swapchains = SmallVec::<[_; 1]>::with_capacity(swapchain_count);
        let mut image_indices = SmallVec::<[_; 1]>::with_capacity(swapchain_count);
        let mut results = SmallVec::<[_; 1]>::with_capacity(swapchain_count);

        for &swapchain_id in &self.swapchains {
            // SAFETY: The caller must ensure that `resource_map` maps the virtual IDs exhaustively.
            let swapchain_state = unsafe { resource_map.swapchain_unchecked(swapchain_id) };
            semaphores.push(
                swapchain_state.semaphores[current_frame_index as usize]
                    .tasks_complete_semaphore
                    .handle(),
            );
            swapchains.push(swapchain_state.swapchain().handle());
            image_indices.push(swapchain_state.current_image_index().unwrap());
            results.push(vk::Result::SUCCESS);
        }

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices)
            .results(&mut results);

        let fns = self.device().fns();
        let queue_present_khr = fns.khr_swapchain.queue_present_khr;
        let _ = unsafe { queue_present_khr(present_queue.handle(), &present_info) };

        let mut res = Ok(());

        for (&result, &swapchain_id) in results.iter().zip(&self.swapchains) {
            // SAFETY: The caller must ensure that `resource_map` maps the virtual IDs exhaustively.
            let swapchain_state = unsafe { resource_map.swapchain_unchecked(swapchain_id) };

            // TODO: Could there be a use case for keeping the old image contents?
            unsafe { swapchain_state.set_access(ImageAccess::NONE) };

            // In case of these error codes, the semaphore wait operation is not executed.
            if !matches!(
                result,
                vk::Result::ERROR_OUT_OF_HOST_MEMORY
                    | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                    | vk::Result::ERROR_DEVICE_LOST
            ) {
                swapchain_state
                    .current_image_index
                    .store(u32::MAX, Ordering::Relaxed);
            }

            if !matches!(result, vk::Result::SUCCESS | vk::Result::SUBOPTIMAL_KHR) {
                // Return the first error for consistency with the acquisition logic.
                if res.is_ok() {
                    res = Err(ExecuteError::Swapchain {
                        swapchain_id,
                        error: Validated::Error(result.into()),
                    });
                }
            }
        }

        res
    }

    unsafe fn update_resource_state(
        &self,
        resource_map: &ResourceMap<'_>,
        last_accesses: &[ResourceAccess],
    ) {
        for (id, _) in self.graph.resources.iter() {
            let access = last_accesses[id.index() as usize];

            match id.object_type() {
                ObjectType::Buffer => {
                    // SAFETY: The caller must ensure that `resource_map` maps the virtual IDs
                    // exhaustively.
                    let id_p = unsafe { id.parametrize() };
                    let state = unsafe { resource_map.buffer_unchecked(id_p) };
                    let access = BufferAccess::from_masks(
                        access.stage_mask,
                        access.access_mask,
                        access.queue_family_index,
                    );
                    unsafe { state.set_access(access) };
                }
                ObjectType::Image => {
                    // SAFETY: The caller must ensure that `resource_map` maps the virtual IDs
                    // exhaustively.
                    let id_p = unsafe { id.parametrize() };
                    let state = unsafe { resource_map.image_unchecked(id_p) };
                    let access = ImageAccess::from_masks(
                        access.stage_mask,
                        access.access_mask,
                        access.image_layout,
                        access.queue_family_index,
                    );
                    unsafe { state.set_access(access) };
                }
                ObjectType::Swapchain => {
                    // SAFETY: The caller must ensure that `resource_map` maps the virtual IDs
                    // exhaustively.
                    let id_p = unsafe { id.parametrize() };
                    let state = unsafe { resource_map.swapchain_unchecked(id_p) };
                    let access = ImageAccess::from_masks(
                        access.stage_mask,
                        access.access_mask,
                        access.image_layout,
                        access.queue_family_index,
                    );
                    unsafe { state.set_access(access) };
                }
                _ => unreachable!(),
            }
        }
    }
}

struct ExecuteState2<'a, W: ?Sized + 'static> {
    executable: &'a ExecutableTaskGraph<W>,
    resource_map: &'a ResourceMap<'a>,
    death_row: &'a mut DeathRow,
    current_frame_index: u32,
    current_fence: &'a Fence,
    submission_count: &'a mut usize,
    world: &'a W,
    cmd_pipeline_barrier2: vk::PFN_vkCmdPipelineBarrier2,
    queue_submit2: vk::PFN_vkQueueSubmit2,
    per_submits: SmallVec<[PerSubmitInfo2; 4]>,
    current_per_submit: PerSubmitInfo2,
    current_command_buffer: Option<raw::RecordingCommandBuffer>,
    command_buffers: Vec<Arc<raw::CommandBuffer>>,
    current_buffer_barriers: Vec<vk::BufferMemoryBarrier2<'static>>,
    current_image_barriers: Vec<vk::ImageMemoryBarrier2<'static>>,
}

#[derive(Default)]
struct PerSubmitInfo2 {
    wait_semaphore_infos: SmallVec<[vk::SemaphoreSubmitInfo<'static>; 4]>,
    command_buffer_infos: SmallVec<[vk::CommandBufferSubmitInfo<'static>; 1]>,
    signal_semaphore_infos: SmallVec<[vk::SemaphoreSubmitInfo<'static>; 4]>,
}

impl<'a, W: ?Sized + 'static> ExecuteState2<'a, W> {
    fn new(
        executable: &'a ExecutableTaskGraph<W>,
        resource_map: &'a ResourceMap<'a>,
        death_row: &'a mut DeathRow,
        current_frame_index: u32,
        current_fence: &'a Fence,
        submission_count: &'a mut usize,
        world: &'a W,
    ) -> Result<Self> {
        let fns = executable.device().fns();
        let (cmd_pipeline_barrier2, queue_submit2);

        if executable.device().api_version() >= Version::V1_3 {
            cmd_pipeline_barrier2 = fns.v1_3.cmd_pipeline_barrier2;
            queue_submit2 = fns.v1_3.queue_submit2;
        } else {
            cmd_pipeline_barrier2 = fns.khr_synchronization2.cmd_pipeline_barrier2_khr;
            queue_submit2 = fns.khr_synchronization2.queue_submit2_khr;
        }

        Ok(ExecuteState2 {
            executable,
            resource_map,
            death_row,
            current_frame_index,
            current_fence,
            submission_count,
            world,
            cmd_pipeline_barrier2,
            queue_submit2,
            per_submits: SmallVec::new(),
            current_per_submit: PerSubmitInfo2::default(),
            current_command_buffer: None,
            command_buffers: Vec::new(),
            current_buffer_barriers: Vec::new(),
            current_image_barriers: Vec::new(),
        })
    }

    fn initial_pipeline_barrier(
        &mut self,
        buffer_barrier_range: Range<BarrierIndex>,
        image_barrier_range: Range<BarrierIndex>,
    ) {
        self.convert_initial_buffer_barriers(buffer_barrier_range);
        self.convert_initial_image_barriers(image_barrier_range);
    }

    fn convert_initial_buffer_barriers(&mut self, barrier_range: Range<BarrierIndex>) {
        let barrier_range = barrier_range.start as usize..barrier_range.end as usize;
        let queue_family_index = current_submission!(self).queue.queue_family_index();

        for barrier in &self.executable.buffer_barriers[barrier_range] {
            let state = unsafe { self.resource_map.buffer_unchecked(barrier.buffer) };
            let buffer = state.buffer();
            let access = state.access();
            let mut src_stage_mask = PipelineStages::empty();
            let mut src_access_mask = AccessFlags::empty();
            let dst_stage_mask = barrier.dst_stage_mask;
            let mut dst_access_mask = barrier.dst_access_mask;

            if access.queue_family_index() == queue_family_index {
                src_stage_mask = access.stage_mask();
                src_access_mask = access.access_mask();
            }

            if src_access_mask.contains_writes() && dst_access_mask.contains_reads() {
            } else if dst_access_mask.contains_writes() {
                src_access_mask = AccessFlags::empty();
                dst_access_mask = AccessFlags::empty();
            } else {
                continue;
            }

            self.current_buffer_barriers.push(
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(src_stage_mask.into())
                    .src_access_mask(src_access_mask.into())
                    .dst_stage_mask(dst_stage_mask.into())
                    .dst_access_mask(dst_access_mask.into())
                    // FIXME:
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(buffer.handle())
                    .offset(0)
                    .size(buffer.size()),
            );
        }
    }

    fn convert_initial_image_barriers(&mut self, barrier_range: Range<BarrierIndex>) {
        let barrier_range = barrier_range.start as usize..barrier_range.end as usize;
        let queue_family_index = current_submission!(self).queue.queue_family_index();

        for barrier in &self.executable.image_barriers[barrier_range] {
            let (image, access) = match barrier.image.object_type() {
                ObjectType::Image => {
                    let image_id = unsafe { barrier.image.parametrize() };
                    let state = unsafe { self.resource_map.image_unchecked(image_id) };

                    (state.image(), state.access())
                }
                ObjectType::Swapchain => {
                    let swapchain_id = unsafe { barrier.image.parametrize() };
                    let state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };

                    (state.current_image(), state.access())
                }
                _ => unreachable!(),
            };

            let mut src_stage_mask = PipelineStages::empty();
            let mut src_access_mask = AccessFlags::empty();
            let dst_stage_mask = barrier.dst_stage_mask;
            let mut dst_access_mask = barrier.dst_access_mask;

            if access.queue_family_index() == queue_family_index {
                src_stage_mask = access.stage_mask();
                src_access_mask = access.access_mask();
            }

            #[allow(clippy::if_same_then_else)]
            if access.image_layout() != barrier.new_layout {
            } else if src_access_mask.contains_writes() && dst_access_mask.contains_reads() {
            } else if dst_access_mask.contains_writes() {
                src_access_mask = AccessFlags::empty();
                dst_access_mask = AccessFlags::empty();
            } else {
                continue;
            }

            self.current_image_barriers.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(src_stage_mask.into())
                    .src_access_mask(src_access_mask.into())
                    .dst_stage_mask(dst_stage_mask.into())
                    .dst_access_mask(dst_access_mask.into())
                    .old_layout(access.image_layout().into())
                    .new_layout(barrier.new_layout.into())
                    // FIXME:
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image.handle())
                    .subresource_range(image.subresource_range().to_vk()),
            );
        }
    }

    fn wait_acquire(&mut self, swapchain_id: Id<Swapchain>, stage_mask: PipelineStages) {
        let swapchain_state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };
        let semaphore = &swapchain_state.semaphores[self.current_frame_index as usize]
            .image_available_semaphore;

        self.current_per_submit.wait_semaphore_infos.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore.handle())
                .stage_mask(stage_mask.into()),
        );
    }

    fn wait_semaphore(&mut self, semaphore_index: SemaphoreIndex, stage_mask: PipelineStages) {
        self.current_per_submit.wait_semaphore_infos.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.executable.semaphores.borrow()[semaphore_index].handle())
                .stage_mask(stage_mask.into()),
        );
    }

    fn execute_task(&mut self, node_index: NodeIndex) -> Result {
        if !self.current_buffer_barriers.is_empty() || !self.current_image_barriers.is_empty() {
            self.flush_barriers()?;
        }

        let task_node = unsafe { self.executable.graph.nodes.task_node_unchecked(node_index) };
        let task = &task_node.task;
        let mut current_command_buffer = unsafe {
            RecordingCommandBuffer::new(
                current_command_buffer!(self),
                self.resource_map,
                self.death_row,
            )
        };
        let mut context = TaskContext {
            resource_map: self.resource_map,
            current_frame_index: self.current_frame_index,
            command_buffers: &mut self.command_buffers,
        };

        unsafe { task.execute(&mut current_command_buffer, &mut context, self.world) }
            .map_err(|error| ExecuteError::Task { node_index, error })?;

        if !self.command_buffers.is_empty() {
            unsafe { self.flush_current_command_buffer() }?;

            for command_buffer in self.command_buffers.drain(..) {
                self.current_per_submit.command_buffer_infos.push(
                    vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer.handle()),
                );
                self.death_row.push(command_buffer);
            }
        }

        Ok(())
    }

    fn pipeline_barrier(
        &mut self,
        buffer_barrier_range: Range<BarrierIndex>,
        image_barrier_range: Range<BarrierIndex>,
    ) -> Result {
        self.convert_buffer_barriers(buffer_barrier_range);
        self.convert_image_barriers(image_barrier_range);

        self.flush_barriers()
    }

    fn convert_buffer_barriers(&mut self, barrier_range: Range<BarrierIndex>) {
        let barrier_range = barrier_range.start as usize..barrier_range.end as usize;

        for barrier in &self.executable.buffer_barriers[barrier_range] {
            let state = unsafe { self.resource_map.buffer_unchecked(barrier.buffer) };
            let buffer = state.buffer();

            self.current_buffer_barriers.push(
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(barrier.src_stage_mask.into())
                    .src_access_mask(barrier.src_access_mask.into())
                    .dst_stage_mask(barrier.dst_stage_mask.into())
                    .dst_access_mask(barrier.dst_access_mask.into())
                    .src_queue_family_index(barrier.src_queue_family_index)
                    .dst_queue_family_index(barrier.dst_queue_family_index)
                    .buffer(buffer.handle())
                    .offset(0)
                    .size(buffer.size()),
            );
        }
    }

    fn convert_image_barriers(&mut self, barrier_range: Range<BarrierIndex>) {
        let barrier_range = barrier_range.start as usize..barrier_range.end as usize;

        for barrier in &self.executable.image_barriers[barrier_range] {
            let image = match barrier.image.object_type() {
                ObjectType::Image => {
                    let image_id = unsafe { barrier.image.parametrize() };

                    unsafe { self.resource_map.image_unchecked(image_id) }.image()
                }
                ObjectType::Swapchain => {
                    let swapchain_id = unsafe { barrier.image.parametrize() };

                    unsafe { self.resource_map.swapchain_unchecked(swapchain_id) }.current_image()
                }
                _ => unreachable!(),
            };

            self.current_image_barriers.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(barrier.src_stage_mask.into())
                    .src_access_mask(barrier.src_access_mask.into())
                    .dst_stage_mask(barrier.dst_stage_mask.into())
                    .dst_access_mask(barrier.dst_access_mask.into())
                    .old_layout(barrier.old_layout.into())
                    .new_layout(barrier.new_layout.into())
                    .src_queue_family_index(barrier.src_queue_family_index)
                    .dst_queue_family_index(barrier.dst_queue_family_index)
                    .image(image.handle())
                    .subresource_range(image.subresource_range().to_vk()),
            );
        }
    }

    fn signal_semaphore(&mut self, semaphore_index: SemaphoreIndex, stage_mask: PipelineStages) {
        self.current_per_submit.signal_semaphore_infos.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.executable.semaphores.borrow()[semaphore_index].handle())
                .stage_mask(stage_mask.into()),
        );
    }

    fn signal_pre_present(&mut self, swapchain_id: Id<Swapchain>, stage_mask: PipelineStages) {
        let swapchain_state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };
        let semaphore = &swapchain_state.semaphores[self.current_frame_index as usize]
            .pre_present_complete_semaphore;

        self.current_per_submit.signal_semaphore_infos.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore.handle())
                .stage_mask(stage_mask.into()),
        );
    }

    fn wait_pre_present(&mut self, swapchain_id: Id<Swapchain>, stage_mask: PipelineStages) {
        let swapchain_state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };
        let semaphore = &swapchain_state.semaphores[self.current_frame_index as usize]
            .pre_present_complete_semaphore;

        self.current_per_submit.wait_semaphore_infos.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore.handle())
                .stage_mask(stage_mask.into()),
        );
    }

    fn signal_present(&mut self, swapchain_id: Id<Swapchain>, stage_mask: PipelineStages) {
        let swapchain_state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };
        let semaphore =
            &swapchain_state.semaphores[self.current_frame_index as usize].tasks_complete_semaphore;

        self.current_per_submit.signal_semaphore_infos.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore.handle())
                .stage_mask(stage_mask.into()),
        );
    }

    fn flush_barriers(&mut self) -> Result {
        unsafe {
            (self.cmd_pipeline_barrier2)(
                current_command_buffer!(self).handle(),
                &vk::DependencyInfo::default()
                    .buffer_memory_barriers(&self.current_buffer_barriers)
                    .image_memory_barriers(&self.current_image_barriers),
            )
        };

        self.current_buffer_barriers.clear();
        self.current_image_barriers.clear();

        Ok(())
    }

    fn flush_submit(&mut self) -> Result {
        unsafe { self.flush_current_command_buffer() }?;

        self.per_submits
            .push(mem::take(&mut self.current_per_submit));

        Ok(())
    }

    fn submit(&mut self) -> Result {
        unsafe {
            self.executable
                .flush_mapped_memory_ranges(self.resource_map)
        }?;

        let submission = current_submission!(self);

        let mut submit_infos = SmallVec::<[_; 4]>::with_capacity(self.per_submits.len());
        submit_infos.extend(self.per_submits.iter().map(|per_submit| {
            vk::SubmitInfo2::default()
                .wait_semaphore_infos(&per_submit.wait_semaphore_infos)
                .command_buffer_infos(&per_submit.command_buffer_infos)
                .signal_semaphore_infos(&per_submit.signal_semaphore_infos)
        }));

        let max_submission_index = self.executable.submissions.len() - 1;
        let fence_handle = if *self.submission_count == max_submission_index {
            self.current_fence.handle()
        } else {
            vk::Fence::null()
        };

        submission.queue.with(|_guard| {
            unsafe {
                (self.queue_submit2)(
                    submission.queue.handle(),
                    submit_infos.len() as u32,
                    submit_infos.as_ptr(),
                    fence_handle,
                )
            }
            .result()
            .map_err(VulkanError::from)
        })?;

        drop(submit_infos);
        self.per_submits.clear();

        *self.submission_count += 1;

        Ok(())
    }

    unsafe fn flush_current_command_buffer(&mut self) -> Result {
        let current_command_buffer = self.current_command_buffer.take().unwrap();
        let command_buffer = unsafe { current_command_buffer.end() }?;
        self.current_per_submit
            .command_buffer_infos
            .push(vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer.handle()));
        self.death_row.push(Arc::new(command_buffer));

        Ok(())
    }
}

struct ExecuteState<'a, W: ?Sized + 'static> {
    executable: &'a ExecutableTaskGraph<W>,
    resource_map: &'a ResourceMap<'a>,
    death_row: &'a mut DeathRow,
    current_frame_index: u32,
    current_fence: &'a Fence,
    submission_count: &'a mut usize,
    world: &'a W,
    cmd_pipeline_barrier: vk::PFN_vkCmdPipelineBarrier,
    queue_submit: vk::PFN_vkQueueSubmit,
    per_submits: SmallVec<[PerSubmitInfo; 4]>,
    current_per_submit: PerSubmitInfo,
    current_command_buffer: Option<raw::RecordingCommandBuffer>,
    command_buffers: Vec<Arc<raw::CommandBuffer>>,
    current_buffer_barriers: Vec<vk::BufferMemoryBarrier<'static>>,
    current_image_barriers: Vec<vk::ImageMemoryBarrier<'static>>,
    current_src_stage_mask: vk::PipelineStageFlags,
    current_dst_stage_mask: vk::PipelineStageFlags,
}

#[derive(Default)]
struct PerSubmitInfo {
    wait_semaphores: SmallVec<[vk::Semaphore; 4]>,
    wait_dst_stage_mask: SmallVec<[vk::PipelineStageFlags; 4]>,
    command_buffers: SmallVec<[vk::CommandBuffer; 1]>,
    signal_semaphores: SmallVec<[vk::Semaphore; 4]>,
}

impl<'a, W: ?Sized + 'static> ExecuteState<'a, W> {
    fn new(
        executable: &'a ExecutableTaskGraph<W>,
        resource_map: &'a ResourceMap<'a>,
        death_row: &'a mut DeathRow,
        current_frame_index: u32,
        current_fence: &'a Fence,
        submission_count: &'a mut usize,
        world: &'a W,
    ) -> Result<Self> {
        let fns = executable.device().fns();
        let cmd_pipeline_barrier = fns.v1_0.cmd_pipeline_barrier;
        let queue_submit = fns.v1_0.queue_submit;

        Ok(ExecuteState {
            executable,
            resource_map,
            death_row,
            current_frame_index,
            current_fence,
            submission_count,
            world,
            cmd_pipeline_barrier,
            queue_submit,
            per_submits: SmallVec::new(),
            current_per_submit: PerSubmitInfo::default(),
            current_command_buffer: None,
            command_buffers: Vec::new(),
            current_buffer_barriers: Vec::new(),
            current_image_barriers: Vec::new(),
            current_src_stage_mask: vk::PipelineStageFlags::empty(),
            current_dst_stage_mask: vk::PipelineStageFlags::empty(),
        })
    }

    fn initial_pipeline_barrier(
        &mut self,
        buffer_barrier_range: Range<BarrierIndex>,
        image_barrier_range: Range<BarrierIndex>,
    ) {
        self.convert_initial_buffer_barriers(buffer_barrier_range);
        self.convert_initial_image_barriers(image_barrier_range);
    }

    fn convert_initial_buffer_barriers(&mut self, barrier_range: Range<BarrierIndex>) {
        let barrier_range = barrier_range.start as usize..barrier_range.end as usize;
        let queue_family_index = current_submission!(self).queue.queue_family_index();

        for barrier in &self.executable.buffer_barriers[barrier_range] {
            let state = unsafe { self.resource_map.buffer_unchecked(barrier.buffer) };
            let buffer = state.buffer();
            let access = state.access();
            let mut src_stage_mask = PipelineStages::empty();
            let mut src_access_mask = AccessFlags::empty();
            let dst_stage_mask = barrier.dst_stage_mask;
            let mut dst_access_mask = barrier.dst_access_mask;

            if access.queue_family_index() == queue_family_index {
                src_stage_mask = access.stage_mask();
                src_access_mask = access.access_mask();
            }

            if src_access_mask.contains_writes() && dst_access_mask.contains_reads() {
            } else if dst_access_mask.contains_writes() {
                src_access_mask = AccessFlags::empty();
                dst_access_mask = AccessFlags::empty();
            } else {
                continue;
            }

            self.current_buffer_barriers.push(
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(convert_access_mask(src_access_mask))
                    .dst_access_mask(convert_access_mask(dst_access_mask))
                    // FIXME:
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(buffer.handle())
                    .offset(0)
                    .size(buffer.size()),
            );

            self.current_src_stage_mask |= convert_stage_mask(src_stage_mask);
            self.current_dst_stage_mask |= convert_stage_mask(dst_stage_mask);
        }
    }

    fn convert_initial_image_barriers(&mut self, barrier_range: Range<BarrierIndex>) {
        let barrier_range = barrier_range.start as usize..barrier_range.end as usize;
        let queue_family_index = current_submission!(self).queue.queue_family_index();

        for barrier in &self.executable.image_barriers[barrier_range] {
            let (image, access) = match barrier.image.object_type() {
                ObjectType::Image => {
                    let image_id = unsafe { barrier.image.parametrize() };
                    let state = unsafe { self.resource_map.image_unchecked(image_id) };

                    (state.image(), state.access())
                }
                ObjectType::Swapchain => {
                    let swapchain_id = unsafe { barrier.image.parametrize() };
                    let state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };

                    (state.current_image(), state.access())
                }
                _ => unreachable!(),
            };

            let mut src_stage_mask = PipelineStages::empty();
            let mut src_access_mask = AccessFlags::empty();
            let dst_stage_mask = barrier.dst_stage_mask;
            let mut dst_access_mask = barrier.dst_access_mask;

            if access.queue_family_index() == queue_family_index {
                src_stage_mask = access.stage_mask();
                src_access_mask = access.access_mask();
            }

            #[allow(clippy::if_same_then_else)]
            if access.image_layout() != barrier.new_layout {
            } else if src_access_mask.contains_writes() && dst_access_mask.contains_reads() {
            } else if dst_access_mask.contains_writes() {
                src_access_mask = AccessFlags::empty();
                dst_access_mask = AccessFlags::empty();
            } else {
                continue;
            }

            self.current_image_barriers.push(
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(convert_access_mask(src_access_mask))
                    .dst_access_mask(convert_access_mask(dst_access_mask))
                    .old_layout(access.image_layout().into())
                    .new_layout(barrier.new_layout.into())
                    // FIXME:
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image.handle())
                    .subresource_range(image.subresource_range().to_vk()),
            );

            self.current_src_stage_mask |= convert_stage_mask(src_stage_mask);
            self.current_dst_stage_mask |= convert_stage_mask(dst_stage_mask);
        }
    }

    fn wait_acquire(&mut self, swapchain_id: Id<Swapchain>, stage_mask: PipelineStages) {
        let swapchain_state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };
        let semaphore = &swapchain_state.semaphores[self.current_frame_index as usize]
            .image_available_semaphore;

        self.current_per_submit
            .wait_semaphores
            .push(semaphore.handle());
        self.current_per_submit
            .wait_dst_stage_mask
            .push(convert_stage_mask(stage_mask));
    }

    fn wait_semaphore(&mut self, semaphore_index: SemaphoreIndex, stage_mask: PipelineStages) {
        self.current_per_submit
            .wait_semaphores
            .push(self.executable.semaphores.borrow()[semaphore_index].handle());
        self.current_per_submit
            .wait_dst_stage_mask
            .push(convert_stage_mask(stage_mask));
    }

    fn execute_task(&mut self, node_index: NodeIndex) -> Result {
        if !self.current_buffer_barriers.is_empty() || !self.current_image_barriers.is_empty() {
            self.flush_barriers()?;
        }

        let task_node = unsafe { self.executable.graph.nodes.task_node_unchecked(node_index) };
        let task = &task_node.task;
        let mut current_command_buffer = unsafe {
            RecordingCommandBuffer::new(
                current_command_buffer!(self),
                self.resource_map,
                self.death_row,
            )
        };
        let mut context = TaskContext {
            resource_map: self.resource_map,
            current_frame_index: self.current_frame_index,
            command_buffers: &mut self.command_buffers,
        };

        unsafe { task.execute(&mut current_command_buffer, &mut context, self.world) }
            .map_err(|error| ExecuteError::Task { node_index, error })?;

        if !self.command_buffers.is_empty() {
            unsafe { self.flush_current_command_buffer() }?;

            for command_buffer in self.command_buffers.drain(..) {
                self.current_per_submit
                    .command_buffers
                    .push(command_buffer.handle());
                self.death_row.push(command_buffer);
            }
        }

        Ok(())
    }

    fn pipeline_barrier(
        &mut self,
        buffer_barrier_range: Range<BarrierIndex>,
        image_barrier_range: Range<BarrierIndex>,
    ) -> Result {
        self.convert_buffer_barriers(buffer_barrier_range);
        self.convert_image_barriers(image_barrier_range);

        self.flush_barriers()
    }

    fn convert_buffer_barriers(&mut self, barrier_range: Range<BarrierIndex>) {
        let barrier_range = barrier_range.start as usize..barrier_range.end as usize;

        for barrier in &self.executable.buffer_barriers[barrier_range] {
            let state = unsafe { self.resource_map.buffer_unchecked(barrier.buffer) };
            let buffer = state.buffer();

            self.current_buffer_barriers.push(
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(convert_access_mask(barrier.src_access_mask))
                    .dst_access_mask(convert_access_mask(barrier.dst_access_mask))
                    .src_queue_family_index(barrier.src_queue_family_index)
                    .dst_queue_family_index(barrier.dst_queue_family_index)
                    .buffer(state.buffer().handle())
                    .offset(0)
                    .size(buffer.size()),
            );

            self.current_src_stage_mask |= convert_stage_mask(barrier.src_stage_mask);
            self.current_dst_stage_mask |= convert_stage_mask(barrier.dst_stage_mask);
        }
    }

    fn convert_image_barriers(&mut self, barrier_range: Range<BarrierIndex>) {
        let barrier_range = barrier_range.start as usize..barrier_range.end as usize;

        for barrier in &self.executable.image_barriers[barrier_range] {
            let image = match barrier.image.object_type() {
                ObjectType::Image => {
                    let image_id = unsafe { barrier.image.parametrize() };

                    unsafe { self.resource_map.image_unchecked(image_id) }.image()
                }
                ObjectType::Swapchain => {
                    let swapchain_id = unsafe { barrier.image.parametrize() };

                    unsafe { self.resource_map.swapchain_unchecked(swapchain_id) }.current_image()
                }
                _ => unreachable!(),
            };

            self.current_image_barriers.push(
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(convert_access_mask(barrier.src_access_mask))
                    .dst_access_mask(convert_access_mask(barrier.dst_access_mask))
                    .old_layout(barrier.old_layout.into())
                    .new_layout(barrier.new_layout.into())
                    .src_queue_family_index(barrier.src_queue_family_index)
                    .dst_queue_family_index(barrier.dst_queue_family_index)
                    .image(image.handle())
                    .subresource_range(image.subresource_range().to_vk()),
            );

            self.current_src_stage_mask |= convert_stage_mask(barrier.src_stage_mask);
            self.current_dst_stage_mask |= convert_stage_mask(barrier.dst_stage_mask);
        }
    }

    fn signal_semaphore(&mut self, semaphore_index: SemaphoreIndex, _stage_mask: PipelineStages) {
        self.current_per_submit
            .signal_semaphores
            .push(self.executable.semaphores.borrow()[semaphore_index].handle());
    }

    fn signal_pre_present(&mut self, swapchain_id: Id<Swapchain>, _stage_mask: PipelineStages) {
        let swapchain_state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };
        let semaphore = &swapchain_state.semaphores[self.current_frame_index as usize]
            .pre_present_complete_semaphore;

        self.current_per_submit
            .signal_semaphores
            .push(semaphore.handle());
    }

    fn wait_pre_present(&mut self, swapchain_id: Id<Swapchain>, stage_mask: PipelineStages) {
        let swapchain_state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };
        let semaphore = &swapchain_state.semaphores[self.current_frame_index as usize]
            .pre_present_complete_semaphore;

        self.current_per_submit
            .wait_semaphores
            .push(semaphore.handle());
        self.current_per_submit
            .wait_dst_stage_mask
            .push(convert_stage_mask(stage_mask));
    }

    fn signal_present(&mut self, swapchain_id: Id<Swapchain>, _stage_mask: PipelineStages) {
        let swapchain_state = unsafe { self.resource_map.swapchain_unchecked(swapchain_id) };
        let semaphore =
            &swapchain_state.semaphores[self.current_frame_index as usize].tasks_complete_semaphore;

        self.current_per_submit
            .signal_semaphores
            .push(semaphore.handle());
    }

    fn flush_submit(&mut self) -> Result {
        unsafe { self.flush_current_command_buffer() }?;

        self.per_submits
            .push(mem::take(&mut self.current_per_submit));

        Ok(())
    }

    fn flush_barriers(&mut self) -> Result {
        if self.current_src_stage_mask.is_empty() {
            self.current_src_stage_mask = vk::PipelineStageFlags::TOP_OF_PIPE;
        }

        if self.current_dst_stage_mask.is_empty() {
            self.current_dst_stage_mask = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
        }

        unsafe {
            (self.cmd_pipeline_barrier)(
                current_command_buffer!(self).handle(),
                self.current_src_stage_mask,
                self.current_dst_stage_mask,
                vk::DependencyFlags::empty(),
                0,
                ptr::null(),
                self.current_buffer_barriers.len() as u32,
                self.current_buffer_barriers.as_ptr(),
                self.current_image_barriers.len() as u32,
                self.current_image_barriers.as_ptr(),
            )
        };

        self.current_buffer_barriers.clear();
        self.current_image_barriers.clear();
        self.current_src_stage_mask = vk::PipelineStageFlags::empty();
        self.current_dst_stage_mask = vk::PipelineStageFlags::empty();

        Ok(())
    }

    fn submit(&mut self) -> Result {
        unsafe {
            self.executable
                .flush_mapped_memory_ranges(self.resource_map)
        }?;

        let submission = current_submission!(self);

        let mut submit_infos = SmallVec::<[_; 4]>::with_capacity(self.per_submits.len());
        submit_infos.extend(self.per_submits.iter().map(|per_submit| {
            vk::SubmitInfo::default()
                .wait_semaphores(&per_submit.wait_semaphores)
                .wait_dst_stage_mask(&per_submit.wait_dst_stage_mask)
                .command_buffers(&per_submit.command_buffers)
                .signal_semaphores(&per_submit.signal_semaphores)
        }));

        let max_submission_index = self.executable.submissions.len() - 1;
        let fence_handle = if *self.submission_count == max_submission_index {
            self.current_fence.handle()
        } else {
            vk::Fence::null()
        };

        submission.queue.with(|_guard| {
            unsafe {
                (self.queue_submit)(
                    submission.queue.handle(),
                    submit_infos.len() as u32,
                    submit_infos.as_ptr(),
                    fence_handle,
                )
            }
            .result()
            .map_err(VulkanError::from)
        })?;

        drop(submit_infos);
        self.per_submits.clear();

        *self.submission_count += 1;

        Ok(())
    }

    unsafe fn flush_current_command_buffer(&mut self) -> Result {
        let current_command_buffer = self.current_command_buffer.take().unwrap();
        let command_buffer = unsafe { current_command_buffer.end() }?;
        self.current_per_submit
            .command_buffers
            .push(command_buffer.handle());
        self.death_row.push(Arc::new(command_buffer));

        Ok(())
    }
}

macro_rules! current_submission {
    ($state:expr) => {
        &$state.executable.submissions[*$state.submission_count]
    };
}
use current_submission;

macro_rules! current_command_buffer {
    ($state:expr) => {{
        if $state.current_command_buffer.is_none() {
            $state.current_command_buffer = Some(create_command_buffer(
                $state.resource_map,
                &current_submission!($state).queue,
            )?);
        }

        $state.current_command_buffer.as_mut().unwrap()
    }};
}
use current_command_buffer;

fn create_command_buffer(
    resource_map: &ResourceMap<'_>,
    queue: &Queue,
) -> Result<raw::RecordingCommandBuffer, VulkanError> {
    let allocator = resource_map.physical_resources.command_buffer_allocator();

    // SAFETY: The parameters are valid.
    unsafe {
        raw::RecordingCommandBuffer::new_unchecked(
            allocator.clone(),
            queue.queue_family_index(),
            raw::CommandBufferLevel::Primary,
            raw::CommandBufferBeginInfo {
                usage: raw::CommandBufferUsage::OneTimeSubmit,
                inheritance_info: None,
                ..Default::default()
            },
        )
    }
    // This can't panic because we know that the queue family index is active on the device,
    // otherwise we wouldn't have a reference to the `Queue`.
    .map_err(Validated::unwrap)
}

fn convert_stage_mask(mut stage_mask: PipelineStages) -> vk::PipelineStageFlags {
    const VERTEX_INPUT_FLAGS: PipelineStages =
        PipelineStages::INDEX_INPUT.union(PipelineStages::VERTEX_ATTRIBUTE_INPUT);
    const TRANSFER_FLAGS: PipelineStages = PipelineStages::COPY
        .union(PipelineStages::RESOLVE)
        .union(PipelineStages::BLIT)
        .union(PipelineStages::CLEAR);

    if stage_mask.intersects(VERTEX_INPUT_FLAGS) {
        stage_mask -= VERTEX_INPUT_FLAGS;
        stage_mask |= PipelineStages::VERTEX_INPUT;
    }

    if stage_mask.intersects(TRANSFER_FLAGS) {
        stage_mask -= TRANSFER_FLAGS;
        stage_mask |= PipelineStages::ALL_TRANSFER;
    }

    stage_mask.into()
}

fn convert_access_mask(mut access_mask: AccessFlags) -> vk::AccessFlags {
    const READ_FLAGS: AccessFlags =
        AccessFlags::SHADER_SAMPLED_READ.union(AccessFlags::SHADER_STORAGE_READ);
    const WRITE_FLAGS: AccessFlags = AccessFlags::SHADER_STORAGE_WRITE;

    if access_mask.intersects(READ_FLAGS) {
        access_mask -= READ_FLAGS;
        access_mask |= AccessFlags::SHADER_READ;
    }

    if access_mask.intersects(WRITE_FLAGS) {
        access_mask -= WRITE_FLAGS;
        access_mask |= AccessFlags::SHADER_WRITE;
    }

    access_mask.into()
}

struct StateGuard<'a, W: ?Sized + 'static> {
    executable: &'a ExecutableTaskGraph<W>,
    resource_map: &'a ResourceMap<'a>,
    current_fence: &'a mut Fence,
    submission_count: usize,
}

impl<W: ?Sized + 'static> Drop for StateGuard<'_, W> {
    #[cold]
    fn drop(&mut self) {
        let device = self.executable.device();

        // SAFETY: The parameters are valid.
        match unsafe {
            Fence::new_unchecked(
                device.clone(),
                FenceCreateInfo {
                    flags: FenceCreateFlags::SIGNALED,
                    ..Default::default()
                },
            )
        } {
            Ok(new_fence) => {
                drop(mem::replace(self.current_fence, new_fence));
            }
            Err(err) => {
                // Device loss is already a form of poisoning built into Vulkan. There's no
                // invalid state that can be observed by design.
                if err == VulkanError::DeviceLost {
                    return;
                }

                eprintln!(
                    "failed to recreate the current fence after failed execution rendering \
                    recovery impossible: {err}; aborting",
                );
                std::process::abort();
            }
        }

        if self.submission_count == 0 {
            return;
        }

        let submissions = &self.executable.submissions;

        // We must make sure that invalid state cannot be observed, because if at least one
        // submission succeeded while one failed, that means that there are pending semaphore
        // signal operations.
        for submission in &submissions[0..self.submission_count] {
            if let Err(err) = submission.queue.with(|mut guard| guard.wait_idle()) {
                if err == VulkanError::DeviceLost {
                    return;
                }

                eprintln!(
                    "failed to wait on queue idle after partly failed submissions rendering \
                    recovery impossible: {err}; aborting",
                );
                std::process::abort();
            }
        }

        // But even after waiting for idle, the state of the graph is invalid because some
        // semaphores are still signalled, so we have to recreate them.
        for semaphore in self.executable.semaphores.borrow_mut().iter_mut() {
            // SAFETY: The parameters are valid.
            match unsafe { Semaphore::new_unchecked(device.clone(), Default::default()) } {
                Ok(new_semaphore) => {
                    let _ = mem::replace(semaphore, Arc::new(new_semaphore));
                }
                Err(err) => {
                    if err == VulkanError::DeviceLost {
                        return;
                    }

                    eprintln!(
                        "failed to recreate semaphores after partly failed submissions rendering \
                        recovery impossible: {err}; aborting",
                    );
                    std::process::abort();
                }
            }
        }

        let mut last_accesses =
            vec![ResourceAccess::default(); self.executable.graph.resources.capacity() as usize];
        let instruction_range = 0..submissions[self.submission_count - 1].instruction_range.end;

        // Determine the last accesses of resources up until before the failed submission.
        for instruction in &self.executable.instructions[instruction_range] {
            let Instruction::ExecuteTask { node_index } = instruction else {
                continue;
            };
            let task_node = unsafe { self.executable.graph.nodes.task_node_unchecked(*node_index) };

            for (id, access) in task_node.accesses.iter() {
                let prev_access = &mut last_accesses[id.index() as usize];
                let access = ResourceAccess {
                    queue_family_index: task_node.queue_family_index,
                    ..*access
                };

                if prev_access.queue_family_index != access.queue_family_index
                    || prev_access.image_layout != access.image_layout
                    || prev_access.access_mask.contains_writes()
                    || access.access_mask.contains_writes()
                {
                    *prev_access = access;
                } else {
                    prev_access.stage_mask |= access.stage_mask;
                    prev_access.access_mask |= access.access_mask;
                }
            }
        }

        // Update the resource state with the correct last accesses.
        unsafe {
            self.executable
                .update_resource_state(self.resource_map, &last_accesses)
        };
    }
}

/// Maps [virtual resources] to physical resources.
pub struct ResourceMap<'a> {
    virtual_resources: &'a super::Resources,
    physical_resources: Arc<Resources>,
    map: Vec<*const ()>,
    len: u32,
    guard: epoch::Guard<'a>,
}

impl<'a> ResourceMap<'a> {
    /// Creates a new `ResourceMap` mapping the virtual resources of the given `executable`.
    pub fn new(executable: &'a ExecutableTaskGraph<impl ?Sized>) -> Result<Self, InvalidSlotError> {
        let virtual_resources = &executable.graph.resources;
        let physical_resources = virtual_resources.physical_resources.clone();
        let mut map = vec![ptr::null(); virtual_resources.capacity() as usize];
        let guard = virtual_resources.physical_resources.pin();

        for (&physical_id, &virtual_id) in &virtual_resources.physical_map {
            // SAFETY: Virtual IDs inside the `physical_map` are always valid.
            let slot = unsafe { map.get_unchecked_mut(virtual_id.index() as usize) };

            *slot = match physical_id.object_type() {
                // SAFETY: We own an `epoch::Guard`.
                ObjectType::Buffer => {
                    let physical_id_p = unsafe { physical_id.parametrize() };
                    <*const _>::cast(unsafe {
                        physical_resources.buffer_unprotected(physical_id_p)
                    }?)
                }
                // SAFETY: We own an `epoch::Guard`.
                ObjectType::Image => {
                    let physical_id_p = unsafe { physical_id.parametrize() };
                    <*const _>::cast(unsafe {
                        physical_resources.image_unprotected(physical_id_p)
                    }?)
                }
                // SAFETY: We own an `epoch::Guard`.
                ObjectType::Swapchain => {
                    let physical_id_p = unsafe { physical_id.parametrize() };
                    <*const _>::cast(unsafe {
                        physical_resources.swapchain_unprotected(physical_id_p)
                    }?)
                }
                _ => unreachable!(),
            };
        }

        let len = virtual_resources.physical_map.len() as u32;

        Ok(ResourceMap {
            virtual_resources,
            physical_resources,
            map,
            len,
            guard,
        })
    }

    #[doc(hidden)]
    #[inline]
    pub fn insert<R: Resource>(
        &mut self,
        virtual_id: Id<R>,
        physical_id: Id<R>,
    ) -> Result<(), InvalidSlotError> {
        R::insert(self, virtual_id, physical_id)
    }

    /// Inserts a mapping from the [virtual buffer resource] corresponding to `virtual_id` to the
    /// physical resource corresponding to `physical_id`.
    ///
    /// # Panics
    ///
    /// - Panics if the physical resource doesn't match the virtual resource.
    /// - Panics if the physical resource already has a mapping from another virtual resource.
    #[inline]
    pub fn insert_buffer(
        &mut self,
        virtual_id: Id<Buffer>,
        physical_id: Id<Buffer>,
    ) -> Result<(), InvalidSlotError> {
        self.virtual_resources.get(virtual_id.erase())?;

        // SAFETY: We own an `epoch::Guard`.
        let state = unsafe { self.physical_resources.buffer_unprotected(physical_id) }?;

        assert_eq!(
            state.buffer().sharing().is_exclusive(),
            virtual_id.is_exclusive(),
        );

        let ptr = <*const _>::cast(state);
        let is_duplicate = self.map.iter().any(|&p| p == ptr);

        // SAFETY: We checked that `virtual_id` is present in `self.virtual_resources` above, and
        // since we initialized `self.map` with a length at least that of `self.virtual_resources`,
        // the index must be in bounds.
        let slot = unsafe { self.map.get_unchecked_mut(virtual_id.index() as usize) };

        if *slot != ptr {
            assert!(!is_duplicate);
        }

        if slot.is_null() {
            self.len += 1;
        }

        *slot = ptr;

        Ok(())
    }

    /// Inserts a mapping from the [virtual buffer resource] corresponding to `virtual_id` to the
    /// physical resource corresponding to `physical_id` without doing any checks.
    ///
    /// # Safety
    ///
    /// - `virtual_id` must be a valid virtual resource ID.
    /// - `physical_id` must be a valid physical resource ID.
    /// - The physical resource must match the virtual resource.
    /// - The physical resource must not have a mapping from another virtual resource.
    #[inline]
    pub unsafe fn insert_buffer_unchecked(
        &mut self,
        virtual_id: Id<Buffer>,
        physical_id: Id<Buffer>,
    ) {
        // SAFETY:
        // * The caller must ensure that `physical_id` is a valid ID.
        // * We own an `epoch::Guard`.
        let state = unsafe {
            self.physical_resources
                .buffer_unchecked_unprotected(physical_id)
        };

        // SAFETY: The caller must ensure that `virtual_id` is a valid virtual ID, and since we
        // initialized `self.map` with a length at least that of `self.virtual_resources`, the
        // index must be in bounds.
        let slot = unsafe { self.map.get_unchecked_mut(virtual_id.index() as usize) };

        if slot.is_null() {
            self.len += 1;
        }

        *slot = <*const _>::cast(state);
    }

    /// Inserts a mapping from the [virtual image resource] corresponding to `virtual_id` to the
    /// physical resource corresponding to `physical_id`.
    ///
    /// # Panics
    ///
    /// - Panics if the physical resource doesn't match the virtual resource.
    /// - Panics if the physical resource already has a mapping from another virtual resource.
    /// - Panics if `virtual_id` refers to a swapchain image.
    #[inline]
    pub fn insert_image(
        &mut self,
        virtual_id: Id<Image>,
        physical_id: Id<Image>,
    ) -> Result<(), InvalidSlotError> {
        assert_ne!(virtual_id.object_type(), ObjectType::Swapchain);

        self.virtual_resources.get(virtual_id.erase())?;

        // SAFETY: We own an `epoch::Guard`.
        let state = unsafe { self.physical_resources.image_unprotected(physical_id) }?;

        assert_eq!(
            state.image().sharing().is_exclusive(),
            virtual_id.is_exclusive(),
        );

        let ptr = <*const _>::cast(state);
        let is_duplicate = self.map.iter().any(|&p| p == ptr);

        // SAFETY: We checked that `virtual_id` is present in `self.virtual_resources` above, and
        // since we initialized `self.map` with a length at least that of `self.virtual_resources`,
        // the index must be in bounds.
        let slot = unsafe { self.map.get_unchecked_mut(virtual_id.index() as usize) };

        if *slot != ptr {
            assert!(!is_duplicate);
        }

        if slot.is_null() {
            self.len += 1;
        }

        *slot = ptr;

        Ok(())
    }

    /// Inserts a mapping from the [virtual image resource] corresponding to `virtual_id` to the
    /// physical resource corresponding to `physical_id` without doing any checks.
    ///
    /// # Safety
    ///
    /// - `virtual_id` must be a valid virtual resource ID.
    /// - `physical_id` must be a valid physical resource ID.
    /// - The physical resource must match the virtual resource.
    /// - The physical resource must not have a mapping from another virtual resource.
    #[inline]
    pub unsafe fn insert_image_unchecked(&mut self, virtual_id: Id<Image>, physical_id: Id<Image>) {
        // SAFETY:
        // * The caller must ensure that `physical_id` is a valid ID.
        // * We own an `epoch::Guard`.
        let state = unsafe {
            self.physical_resources
                .image_unchecked_unprotected(physical_id)
        };

        // SAFETY: The caller must ensure that `virtual_id` is a valid virtual ID, and since we
        // initialized `self.map` with a length at least that of `self.virtual_resources`, the
        // index must be in bounds.
        let slot = unsafe { self.map.get_unchecked_mut(virtual_id.index() as usize) };

        if slot.is_null() {
            self.len += 1;
        }

        *slot = <*const _>::cast(state);
    }

    /// Inserts a mapping from the [virtual swapchain resource] corresponding to `virtual_id` to
    /// the physical resource corresponding to `physical_id`.
    ///
    /// # Panics
    ///
    /// - Panics if the physical resource doesn't match the virtual resource.
    /// - Panics if the physical resource already has a mapping from another virtual resource.
    #[inline]
    pub fn insert_swapchain(
        &mut self,
        virtual_id: Id<Swapchain>,
        physical_id: Id<Swapchain>,
    ) -> Result<(), InvalidSlotError> {
        self.virtual_resources.get(virtual_id.erase())?;

        // SAFETY: We own an `epoch::Guard`.
        let state = unsafe { self.physical_resources.swapchain_unprotected(physical_id) }?;

        assert_eq!(
            state.swapchain().image_sharing().is_exclusive(),
            virtual_id.is_exclusive(),
        );

        let ptr = <*const _>::cast(state);
        let is_duplicate = self.map.iter().any(|&p| p == ptr);

        // SAFETY: We checked that `virtual_id` is present in `self.virtual_resources` above, and
        // since we initialized `self.map` with a length at least that of `self.virtual_resources`,
        // the index must be in bounds.
        let slot = unsafe { self.map.get_unchecked_mut(virtual_id.index() as usize) };

        if *slot != ptr {
            assert!(!is_duplicate);
        }

        if slot.is_null() {
            self.len += 1;
        }

        *slot = ptr;

        Ok(())
    }

    /// Inserts a mapping from the [virtual swapchain resource] corresponding to `virtual_id` to
    /// the physical resource corresponding to `physical_id` without doing any checks.
    ///
    /// # Safety
    ///
    /// - `virtual_id` must be a valid virtual resource ID.
    /// - `physical_id` must be a valid physical resource ID.
    /// - The physical resource must match the virtual resource.
    /// - The physical resource must not have a mapping from another virtual resource.
    #[inline]
    pub unsafe fn insert_swapchain_unchecked(
        &mut self,
        virtual_id: Id<Swapchain>,
        physical_id: Id<Swapchain>,
    ) {
        // SAFETY:
        // * The caller must ensure that `physical_id` is a valid ID.
        // * We own an `epoch::Guard`.
        let state = unsafe {
            self.physical_resources
                .swapchain_unchecked_unprotected(physical_id)
        };

        // SAFETY: The caller must ensure that `virtual_id` is a valid virtual ID, and since we
        // initialized `self.map` with a length at least that of `self.virtual_resources`, the
        // index must be in bounds.
        let slot = unsafe { self.map.get_unchecked_mut(virtual_id.index() as usize) };

        if slot.is_null() {
            self.len += 1;
        }

        *slot = <*const _>::cast(state);
    }

    pub(crate) fn virtual_resources(&self) -> &super::Resources {
        self.virtual_resources
    }

    /// Returns the `Resources` collection.
    #[inline]
    #[must_use]
    pub fn resources(&self) -> &Arc<Resources> {
        &self.physical_resources
    }

    /// Returns the number of mappings in the map.
    #[inline]
    #[must_use]
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Returns `true` if the map maps every virtual resource.
    #[inline]
    #[must_use]
    pub fn is_exhaustive(&self) -> bool {
        // By our own invariant, the map can only contain mappings for virtual resources that are
        // present in `self.virtual_resources`. It follows then, that when the length of `self` is
        // that of `self.virtual_resources`, that the virtual resources are mapped exhaustively.
        self.len() == self.virtual_resources.len()
    }

    pub(crate) unsafe fn buffer(&self, id: Id<Buffer>) -> Result<&BufferState, InvalidSlotError> {
        self.virtual_resources.get(id.erase())?;

        // SAFETY: The caller must ensure that a mapping for `id` has been inserted.
        Ok(unsafe { self.buffer_unchecked(id) })
    }

    pub(crate) unsafe fn buffer_unchecked(&self, id: Id<Buffer>) -> &BufferState {
        #[cfg(debug_assertions)]
        if self.virtual_resources.get(id.erase()).is_err() {
            std::process::abort();
        }

        // SAFETY: The caller must ensure that `id` is a valid virtual ID.
        let &slot = unsafe { self.map.get_unchecked(id.index() as usize) };

        // SAFETY: The caller must ensure that a mapping for `id` has been inserted.
        unsafe { &*slot.cast::<BufferState>() }
    }

    pub(crate) unsafe fn image(&self, id: Id<Image>) -> Result<&ImageState, InvalidSlotError> {
        self.virtual_resources.get(id.erase())?;

        // SAFETY: The caller must ensure that a mapping for `id` has been inserted.
        Ok(unsafe { self.image_unchecked(id) })
    }

    pub(crate) unsafe fn image_unchecked(&self, id: Id<Image>) -> &ImageState {
        #[cfg(debug_assertions)]
        if self.virtual_resources.get(id.erase()).is_err() {
            std::process::abort();
        }

        // SAFETY: The caller must ensure that `id` is a valid virtual ID.
        let &slot = unsafe { self.map.get_unchecked(id.index() as usize) };

        // SAFETY: The caller must ensure that a mapping for `id` has been inserted.
        unsafe { &*slot.cast::<ImageState>() }
    }

    pub(crate) unsafe fn swapchain(
        &self,
        id: Id<Swapchain>,
    ) -> Result<&SwapchainState, InvalidSlotError> {
        self.virtual_resources.get(id.erase())?;

        // SAFETY: The caller must ensure that a mapping for `id` has been inserted.
        Ok(unsafe { self.swapchain_unchecked(id) })
    }

    pub(crate) unsafe fn swapchain_unchecked(&self, id: Id<Swapchain>) -> &SwapchainState {
        #[cfg(debug_assertions)]
        if self.virtual_resources.get(id.erase()).is_err() {
            std::process::abort();
        }

        // SAFETY: The caller must ensure that `id` is a valid virtual ID.
        let &slot = unsafe { self.map.get_unchecked(id.index() as usize) };

        // SAFETY: The caller must ensure that a mapping for `id` has been inserted.
        unsafe { &*slot.cast::<SwapchainState>() }
    }
}

unsafe impl DeviceOwned for ResourceMap<'_> {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.physical_resources.device()
    }
}

pub trait Resource: Sized {
    fn insert(
        map: &mut ResourceMap<'_>,
        virtual_id: Id<Self>,
        physical_id: Id<Self>,
    ) -> Result<(), InvalidSlotError>;
}

impl Resource for Buffer {
    fn insert(
        map: &mut ResourceMap<'_>,
        virtual_id: Id<Self>,
        physical_id: Id<Self>,
    ) -> Result<(), InvalidSlotError> {
        map.insert_buffer(virtual_id, physical_id)
    }
}

impl Resource for Image {
    fn insert(
        map: &mut ResourceMap<'_>,
        virtual_id: Id<Self>,
        physical_id: Id<Self>,
    ) -> Result<(), InvalidSlotError> {
        map.insert_image(virtual_id, physical_id)
    }
}

impl Resource for Swapchain {
    fn insert(
        map: &mut ResourceMap<'_>,
        virtual_id: Id<Self>,
        physical_id: Id<Self>,
    ) -> Result<(), InvalidSlotError> {
        map.insert_swapchain(virtual_id, physical_id)
    }
}

/// Creates a [`ResourceMap`] containing the given mappings.
#[macro_export]
macro_rules! resource_map {
    ($executable:expr $(, $virtual_id:expr => $physical_id:expr)* $(,)?) => {
        match $crate::graph::ResourceMap::new($executable) {
            ::std::result::Result::Ok(mut map) => {
                $(if let ::std::result::Result::Err(err) = map.insert($virtual_id, $physical_id) {
                    ::std::result::Result::Err(err)
                } else)* {
                    ::std::result::Result::Ok::<_, $crate::InvalidSlotError>(map)
                }
            }
            ::std::result::Result::Err(err) => ::std::result::Result::Err(err),
        }
    };
}

type Result<T = (), E = ExecuteError> = ::std::result::Result<T, E>;

/// Error that can happen when [executing] an [`ExecutableTaskGraph`].
///
/// [executing]: ExecutableTaskGraph::execute
#[derive(Debug)]
pub enum ExecuteError {
    Task {
        node_index: NodeIndex,
        error: TaskError,
    },
    Swapchain {
        swapchain_id: Id<Swapchain>,
        error: Validated<VulkanError>,
    },
    VulkanError(VulkanError),
}

impl From<VulkanError> for ExecuteError {
    fn from(err: VulkanError) -> Self {
        Self::VulkanError(err)
    }
}

impl fmt::Display for ExecuteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Task { node_index, .. } => {
                write!(f, "an error occurred while executing task {node_index:?}")
            }
            Self::Swapchain { swapchain_id, .. } => write!(
                f,
                "an error occurred while using swapchain {swapchain_id:?}",
            ),
            Self::VulkanError(_) => f.write_str("a runtime error occurred"),
        }
    }
}

impl Error for ExecuteError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Task { error, .. } => Some(error),
            Self::Swapchain { error, .. } => Some(error),
            Self::VulkanError(err) => Some(err),
        }
    }
}
