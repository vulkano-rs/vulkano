use crate::{
    command_buffer::sys::RecordingCommandBuffer,
    device::{DeviceOwned, QueueFlags},
    sync::{
        event::Event, BufferMemoryBarrier, DependencyInfo, DependencyInfo2Fields1Vk,
        DependencyInfoFields1Vk, ImageMemoryBarrier, MemoryBarrier, PipelineStages,
    },
    Requires, RequiresAllOf, RequiresOneOf, ValidationError, Version, VulkanObject,
};
use smallvec::SmallVec;
use std::sync::Arc;

impl RecordingCommandBuffer {
    #[inline]
    pub unsafe fn pipeline_barrier(
        &mut self,
        dependency_info: &DependencyInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_pipeline_barrier(dependency_info)?;

        Ok(unsafe { self.pipeline_barrier_unchecked(dependency_info) })
    }

    fn validate_pipeline_barrier(
        &self,
        dependency_info: &DependencyInfo,
    ) -> Result<(), Box<ValidationError>> {
        let queue_family_properties = self.queue_family_properties();

        if !queue_family_properties.queue_flags.intersects(
            QueueFlags::TRANSFER
                | QueueFlags::GRAPHICS
                | QueueFlags::COMPUTE
                | QueueFlags::VIDEO_DECODE
                | QueueFlags::VIDEO_ENCODE,
        ) {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    transfer, graphics, compute, video decode or video encode operations"
                    .into(),
                vuids: &["VUID-vkCmdPipelineBarrier2-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        dependency_info
            .validate(self.device())
            .map_err(|err| err.add_context("dependency_info"))?;

        let &DependencyInfo {
            dependency_flags: _,
            ref memory_barriers,
            ref buffer_memory_barriers,
            ref image_memory_barriers,
            _ne,
        } = dependency_info;

        let supported_pipeline_stages = PipelineStages::from(queue_family_properties.queue_flags);

        for (barrier_index, memory_barrier) in memory_barriers.iter().enumerate() {
            let &MemoryBarrier {
                src_stages,
                src_access: _,
                dst_stages,
                dst_access: _,
                _ne: _,
            } = memory_barrier;

            if !supported_pipeline_stages.contains(src_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.memory_barriers[{}].src_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdPipelineBarrier2-srcStageMask-03849"],
                    ..Default::default()
                }));
            }

            if !supported_pipeline_stages.contains(dst_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.memory_barriers[{}].dst_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdPipelineBarrier2-dstStageMask-03850"],
                    ..Default::default()
                }));
            }
        }

        for (barrier_index, buffer_memory_barrier) in buffer_memory_barriers.iter().enumerate() {
            let &BufferMemoryBarrier {
                src_stages,
                src_access: _,
                dst_stages,
                dst_access: _,
                queue_family_ownership_transfer: _,
                buffer: _,
                offset: _,
                size: _,
                _ne: _,
            } = buffer_memory_barrier;

            if !supported_pipeline_stages.contains(src_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.buffer_memory_barriers[{}].src_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdPipelineBarrier2-srcStageMask-03849"],
                    ..Default::default()
                }));
            }

            if !supported_pipeline_stages.contains(dst_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.buffer_memory_barriers[{}].dst_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdPipelineBarrier2-dstStageMask-03850"],
                    ..Default::default()
                }));
            }
        }

        for (barrier_index, image_memory_barrier) in image_memory_barriers.iter().enumerate() {
            let &ImageMemoryBarrier {
                src_stages,
                src_access: _,
                dst_stages,
                dst_access: _,
                old_layout: _,
                new_layout: _,
                queue_family_ownership_transfer: _,
                image: _,
                subresource_range: _,
                _ne: _,
            } = image_memory_barrier;

            if !supported_pipeline_stages.contains(src_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.image_memory_barriers[{}].src_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdPipelineBarrier2-srcStageMask-03849"],
                    ..Default::default()
                }));
            }

            if !supported_pipeline_stages.contains(dst_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.image_memory_barriers[{}].dst_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdPipelineBarrier2-dstStageMask-03850"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn pipeline_barrier_unchecked(
        &mut self,
        dependency_info: &DependencyInfo,
    ) -> &mut Self {
        if dependency_info.is_empty() {
            return self;
        }

        if self.device().enabled_features().synchronization2 {
            let DependencyInfo2Fields1Vk {
                memory_barriers_vk,
                buffer_memory_barriers_vk,
                image_memory_barriers_vk,
            } = dependency_info.to_vk2_fields1();
            let dependency_info_vk = dependency_info.to_vk2(
                &memory_barriers_vk,
                &buffer_memory_barriers_vk,
                &image_memory_barriers_vk,
            );

            let fns = self.device().fns();

            if self.device().api_version() >= Version::V1_3 {
                unsafe { (fns.v1_3.cmd_pipeline_barrier2)(self.handle(), &dependency_info_vk) };
            } else {
                unsafe {
                    (fns.khr_synchronization2.cmd_pipeline_barrier2_khr)(
                        self.handle(),
                        &dependency_info_vk,
                    )
                };
            }
        } else {
            let DependencyInfoFields1Vk {
                memory_barriers_vk,
                buffer_memory_barriers_vk,
                image_memory_barriers_vk,
                src_stage_mask_vk,
                dst_stage_mask_vk,
            } = dependency_info.to_vk_fields1();
            let dependency_flags_vk = dependency_info.to_vk_dependency_flags();

            let fns = self.device().fns();
            unsafe {
                (fns.v1_0.cmd_pipeline_barrier)(
                    self.handle(),
                    src_stage_mask_vk,
                    dst_stage_mask_vk,
                    dependency_flags_vk,
                    memory_barriers_vk.len() as u32,
                    memory_barriers_vk.as_ptr(),
                    buffer_memory_barriers_vk.len() as u32,
                    buffer_memory_barriers_vk.as_ptr(),
                    image_memory_barriers_vk.len() as u32,
                    image_memory_barriers_vk.as_ptr(),
                )
            };
        }

        self
    }

    #[inline]
    pub unsafe fn set_event(
        &mut self,
        event: &Event,
        dependency_info: &DependencyInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_set_event(event, dependency_info)?;

        Ok(unsafe { self.set_event_unchecked(event, dependency_info) })
    }

    fn validate_set_event(
        &self,
        event: &Event,
        dependency_info: &DependencyInfo,
    ) -> Result<(), Box<ValidationError>> {
        let queue_family_properties = self.queue_family_properties();

        if !queue_family_properties.queue_flags.intersects(
            QueueFlags::GRAPHICS
                | QueueFlags::COMPUTE
                | QueueFlags::VIDEO_DECODE
                | QueueFlags::VIDEO_ENCODE,
        ) {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics, compute, video decode or video encode operations"
                    .into(),
                vuids: &["VUID-vkCmdSetEvent2-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdSetEvent2-commonparent
        assert_eq!(self.device(), event.device());

        dependency_info
            .validate(self.device())
            .map_err(|err| err.add_context("dependency_info"))?;

        let &DependencyInfo {
            dependency_flags,
            ref memory_barriers,
            ref buffer_memory_barriers,
            ref image_memory_barriers,
            _ne,
        } = dependency_info;

        if !dependency_flags.is_empty() {
            return Err(Box::new(ValidationError {
                context: "dependency_info.dependency_flags".into(),
                problem: "is not empty".into(),
                vuids: &["VUID-vkCmdSetEvent2-dependencyFlags-03825"],
                ..Default::default()
            }));
        }

        let supported_pipeline_stages = PipelineStages::from(queue_family_properties.queue_flags);

        for (barrier_index, memory_barrier) in memory_barriers.iter().enumerate() {
            let &MemoryBarrier {
                src_stages,
                src_access: _,
                dst_stages,
                dst_access: _,
                _ne: _,
            } = memory_barrier;

            if !supported_pipeline_stages.contains(src_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.memory_barriers[{}].src_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdSetEvent2-srcStageMask-03827"],
                    ..Default::default()
                }));
            }

            if !supported_pipeline_stages.contains(dst_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.memory_barriers[{}].dst_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdSetEvent2-dstStageMask-03828"],
                    ..Default::default()
                }));
            }
        }

        for (barrier_index, buffer_memory_barrier) in buffer_memory_barriers.iter().enumerate() {
            let &BufferMemoryBarrier {
                src_stages,
                src_access: _,
                dst_stages,
                dst_access: _,
                queue_family_ownership_transfer: _,
                buffer: _,
                offset: _,
                size: _,
                _ne: _,
            } = buffer_memory_barrier;

            if !supported_pipeline_stages.contains(src_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.buffer_memory_barriers[{}].src_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdSetEvent2-srcStageMask-03827"],
                    ..Default::default()
                }));
            }

            if !supported_pipeline_stages.contains(dst_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.buffer_memory_barriers[{}].dst_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdSetEvent2-dstStageMask-03828"],
                    ..Default::default()
                }));
            }
        }

        for (barrier_index, image_memory_barrier) in image_memory_barriers.iter().enumerate() {
            let &ImageMemoryBarrier {
                src_stages,
                src_access: _,
                dst_stages,
                dst_access: _,
                old_layout: _,
                new_layout: _,
                queue_family_ownership_transfer: _,
                image: _,
                subresource_range: _,
                _ne: _,
            } = image_memory_barrier;

            if !supported_pipeline_stages.contains(src_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.image_memory_barriers[{}].src_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdSetEvent2-srcStageMask-03827"],
                    ..Default::default()
                }));
            }

            if !supported_pipeline_stages.contains(dst_stages) {
                return Err(Box::new(ValidationError {
                    context: format!(
                        "dependency_info.image_memory_barriers[{}].dst_stages",
                        barrier_index
                    )
                    .into(),
                    problem: "contains stages that are not supported by the queue family of the \
                        command buffer"
                        .into(),
                    vuids: &["VUID-vkCmdSetEvent2-dstStageMask-03828"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn set_event_unchecked(
        &mut self,
        event: &Event,
        dependency_info: &DependencyInfo,
    ) -> &mut Self {
        let fns = self.device().fns();

        if self.device().enabled_features().synchronization2 {
            let DependencyInfo2Fields1Vk {
                memory_barriers_vk,
                buffer_memory_barriers_vk,
                image_memory_barriers_vk,
            } = dependency_info.to_vk2_fields1();
            let dependency_info_vk = dependency_info.to_vk2(
                &memory_barriers_vk,
                &buffer_memory_barriers_vk,
                &image_memory_barriers_vk,
            );

            if self.device().api_version() >= Version::V1_3 {
                unsafe {
                    (fns.v1_3.cmd_set_event2)(self.handle(), event.handle(), &dependency_info_vk)
                };
            } else {
                unsafe {
                    (fns.khr_synchronization2.cmd_set_event2_khr)(
                        self.handle(),
                        event.handle(),
                        &dependency_info_vk,
                    )
                };
            }
        } else {
            // The original function only takes a source stage mask; the rest of the info is
            // provided with `wait_events` instead. Therefore, we condense the source stages
            // here and ignore the rest.
            let stage_mask_vk = dependency_info.to_vk_src_stage_mask();

            unsafe { (fns.v1_0.cmd_set_event)(self.handle(), event.handle(), stage_mask_vk) };
        }

        self
    }

    #[inline]
    pub unsafe fn wait_events(
        &mut self,
        events: &[(Arc<Event>, DependencyInfo)],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_wait_events(events)?;

        Ok(unsafe { self.wait_events_unchecked(events) })
    }

    fn validate_wait_events(
        &self,
        events: &[(Arc<Event>, DependencyInfo)],
    ) -> Result<(), Box<ValidationError>> {
        let queue_family_properties = self.queue_family_properties();

        if !queue_family_properties.queue_flags.intersects(
            QueueFlags::GRAPHICS
                | QueueFlags::COMPUTE
                | QueueFlags::VIDEO_DECODE
                | QueueFlags::VIDEO_ENCODE,
        ) {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics, compute, video decode or video encode operations"
                    .into(),
                vuids: &["VUID-vkCmdWaitEvents2-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        for (event_index, (event, dependency_info)) in events.iter().enumerate() {
            // VUID-vkCmdWaitEvents2-commandBuffer-cmdpool
            assert_eq!(self.device(), event.device());

            dependency_info
                .validate(self.device())
                .map_err(|err| err.add_context(format!("events[{}].1", event_index)))?;

            let &DependencyInfo {
                dependency_flags: _,
                ref memory_barriers,
                ref buffer_memory_barriers,
                ref image_memory_barriers,
                _ne,
            } = dependency_info;

            let supported_pipeline_stages =
                PipelineStages::from(queue_family_properties.queue_flags);

            for (barrier_index, memory_barrier) in memory_barriers.iter().enumerate() {
                let &MemoryBarrier {
                    src_stages,
                    src_access: _,
                    dst_stages,
                    dst_access: _,
                    _ne: _,
                } = memory_barrier;

                if !supported_pipeline_stages.contains(src_stages) {
                    return Err(Box::new(ValidationError {
                        context: format!(
                            "events[{}].1.memory_barriers[{}].src_stages",
                            event_index, barrier_index
                        )
                        .into(),
                        problem: "contains stages that are not supported by the queue family of \
                            the command buffer"
                            .into(),
                        vuids: &["VUID-vkCmdSetEvent2-srcStageMask-03827"],
                        ..Default::default()
                    }));
                }

                if !supported_pipeline_stages.contains(dst_stages) {
                    return Err(Box::new(ValidationError {
                        context: format!(
                            "events[{}].1.memory_barriers[{}].dst_stages",
                            event_index, barrier_index
                        )
                        .into(),
                        problem: "contains stages that are not supported by the queue family of \
                            the command buffer"
                            .into(),
                        vuids: &["VUID-vkCmdSetEvent2-dstStageMask-03828"],
                        ..Default::default()
                    }));
                }
            }

            for (barrier_index, buffer_memory_barrier) in buffer_memory_barriers.iter().enumerate()
            {
                let &BufferMemoryBarrier {
                    src_stages,
                    src_access: _,
                    dst_stages,
                    dst_access: _,
                    queue_family_ownership_transfer: _,
                    buffer: _,
                    offset: _,
                    size: _,
                    _ne: _,
                } = buffer_memory_barrier;

                if !supported_pipeline_stages.contains(src_stages) {
                    return Err(Box::new(ValidationError {
                        context: format!(
                            "events[{}].1.buffer_memory_barriers[{}].src_stages",
                            event_index, barrier_index
                        )
                        .into(),
                        problem: "contains stages that are not supported by the queue family of \
                            the command buffer"
                            .into(),
                        vuids: &["VUID-vkCmdSetEvent2-srcStageMask-03827"],
                        ..Default::default()
                    }));
                }

                if !supported_pipeline_stages.contains(dst_stages) {
                    return Err(Box::new(ValidationError {
                        context: format!(
                            "events[{}].1.buffer_memory_barriers[{}].dst_stages",
                            event_index, barrier_index
                        )
                        .into(),
                        problem: "contains stages that are not supported by the queue family of \
                            the command buffer"
                            .into(),
                        vuids: &["VUID-vkCmdSetEvent2-dstStageMask-03828"],
                        ..Default::default()
                    }));
                }
            }

            for (barrier_index, image_memory_barrier) in image_memory_barriers.iter().enumerate() {
                let &ImageMemoryBarrier {
                    src_stages,
                    src_access: _,
                    dst_stages,
                    dst_access: _,
                    old_layout: _,
                    new_layout: _,
                    queue_family_ownership_transfer: _,
                    image: _,
                    subresource_range: _,
                    _ne: _,
                } = image_memory_barrier;

                if !supported_pipeline_stages.contains(src_stages) {
                    return Err(Box::new(ValidationError {
                        context: format!(
                            "events[{}].1.image_memory_barriers[{}].src_stages",
                            event_index, barrier_index
                        )
                        .into(),
                        problem: "contains stages that are not supported by the queue family of \
                            the command buffer"
                            .into(),
                        vuids: &["VUID-vkCmdSetEvent2-srcStageMask-03827"],
                        ..Default::default()
                    }));
                }

                if !supported_pipeline_stages.contains(dst_stages) {
                    return Err(Box::new(ValidationError {
                        context: format!(
                            "events[{}].1.image_memory_barriers[{}].dst_stages",
                            event_index, barrier_index
                        )
                        .into(),
                        problem: "contains stages that are not supported by the queue family of \
                            the command buffer"
                            .into(),
                        vuids: &["VUID-vkCmdSetEvent2-dstStageMask-03828"],
                        ..Default::default()
                    }));
                }
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn wait_events_unchecked(
        &mut self,
        events: &[(Arc<Event>, DependencyInfo)],
    ) -> &mut Self {
        let fns = self.device().fns();

        if self.device().enabled_features().synchronization2 {
            let events_vk: SmallVec<[_; 4]> =
                events.iter().map(|(event, _)| event.handle()).collect();
            let dependency_infos_fields1_vk: SmallVec<[_; 4]> = events
                .iter()
                .map(|(_, dependency_info)| dependency_info.to_vk2_fields1())
                .collect();
            let dependency_infos_vk: SmallVec<[_; 4]> = events
                .iter()
                .zip(&dependency_infos_fields1_vk)
                .map(|((_, dependency_info), dependency_info_fields1_vk)| {
                    let DependencyInfo2Fields1Vk {
                        memory_barriers_vk,
                        buffer_memory_barriers_vk,
                        image_memory_barriers_vk,
                    } = dependency_info_fields1_vk;

                    dependency_info.to_vk2(
                        memory_barriers_vk,
                        buffer_memory_barriers_vk,
                        image_memory_barriers_vk,
                    )
                })
                .collect();

            if self.device().api_version() >= Version::V1_3 {
                unsafe {
                    (fns.v1_3.cmd_wait_events2)(
                        self.handle(),
                        events_vk.len() as u32,
                        events_vk.as_ptr(),
                        dependency_infos_vk.as_ptr(),
                    )
                };
            } else {
                unsafe {
                    (fns.khr_synchronization2.cmd_wait_events2_khr)(
                        self.handle(),
                        events_vk.len() as u32,
                        events_vk.as_ptr(),
                        dependency_infos_vk.as_ptr(),
                    )
                };
            }
        } else {
            // With the original function, you can only specify a single dependency info for all
            // events at once, rather than separately for each event. Therefore, to achieve the
            // same behaviour as the "2" function, we split it up into multiple Vulkan API calls,
            // one per event.

            for (event, dependency_info) in events {
                let events_vk = [event.handle()];
                let DependencyInfoFields1Vk {
                    memory_barriers_vk,
                    buffer_memory_barriers_vk,
                    image_memory_barriers_vk,
                    src_stage_mask_vk,
                    dst_stage_mask_vk,
                } = dependency_info.to_vk_fields1();

                unsafe {
                    (fns.v1_0.cmd_wait_events)(
                        self.handle(),
                        1,
                        events_vk.as_ptr(),
                        src_stage_mask_vk,
                        dst_stage_mask_vk,
                        memory_barriers_vk.len() as u32,
                        memory_barriers_vk.as_ptr(),
                        buffer_memory_barriers_vk.len() as u32,
                        buffer_memory_barriers_vk.as_ptr(),
                        image_memory_barriers_vk.len() as u32,
                        image_memory_barriers_vk.as_ptr(),
                    )
                };
            }
        }

        self
    }

    #[inline]
    pub unsafe fn reset_event(
        &mut self,
        event: &Event,
        stages: PipelineStages,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_reset_event(event, stages)?;

        Ok(unsafe { self.reset_event_unchecked(event, stages) })
    }

    fn validate_reset_event(
        &self,
        event: &Event,
        stages: PipelineStages,
    ) -> Result<(), Box<ValidationError>> {
        if !self.queue_family_properties().queue_flags.intersects(
            QueueFlags::GRAPHICS
                | QueueFlags::COMPUTE
                | QueueFlags::VIDEO_DECODE
                | QueueFlags::VIDEO_ENCODE,
        ) {
            return Err(Box::new(ValidationError {
                problem: "the queue family of the command buffer does not support \
                    graphics, compute, video decode or video encode operations"
                    .into(),
                vuids: &["VUID-vkCmdResetEvent2-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        let device = self.device();

        // VUID-vkCmdResetEvent2-commonparent
        assert_eq!(device, event.device());

        stages.validate_device(device).map_err(|err| {
            err.add_context("stages")
                .set_vuids(&["VUID-vkCmdResetEvent2-stageMask-parameter"])
        })?;

        if !device.enabled_features().synchronization2 {
            if stages.contains_flags2() {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains flags from `VkPipelineStageFlagBits2`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "synchronization2",
                    )])]),
                    ..Default::default()
                }));
            }
        }

        if !device.enabled_features().geometry_shader {
            if stages.intersects(PipelineStages::GEOMETRY_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::GEOMETRY_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "geometry_shader",
                    )])]),
                    vuids: &["VUID-vkCmdResetEvent2-stageMask-03929"],
                }));
            }
        }

        if !device.enabled_features().tessellation_shader {
            if stages.intersects(
                PipelineStages::TESSELLATION_CONTROL_SHADER
                    | PipelineStages::TESSELLATION_EVALUATION_SHADER,
            ) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::TESSELLATION_CONTROL_SHADER` or \
                        `PipelineStages::TESSELLATION_EVALUATION_SHADER`"
                        .into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "tessellation_shader",
                    )])]),
                    vuids: &["VUID-vkCmdResetEvent2-stageMask-03930"],
                }));
            }
        }

        if !device.enabled_features().conditional_rendering {
            if stages.intersects(PipelineStages::CONDITIONAL_RENDERING) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::CONDITIONAL_RENDERING`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "conditional_rendering",
                    )])]),
                    vuids: &["VUID-vkCmdResetEvent2-stageMask-03931"],
                }));
            }
        }

        if !device.enabled_features().fragment_density_map {
            if stages.intersects(PipelineStages::FRAGMENT_DENSITY_PROCESS) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_DENSITY_PROCESS`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "fragment_density_map",
                    )])]),
                    vuids: &["VUID-vkCmdResetEvent2-stageMask-03932"],
                }));
            }
        }

        if !device.enabled_features().transform_feedback {
            if stages.intersects(PipelineStages::TRANSFORM_FEEDBACK) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::TRANSFORM_FEEDBACK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "transform_feedback",
                    )])]),
                    vuids: &["VUID-vkCmdResetEvent2-stageMask-03933"],
                }));
            }
        }

        if !device.enabled_features().mesh_shader {
            if stages.intersects(PipelineStages::MESH_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::MESH_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "mesh_shader",
                    )])]),
                    vuids: &["VUID-vkCmdResetEvent2-stageMask-03934"],
                }));
            }
        }

        if !device.enabled_features().task_shader {
            if stages.intersects(PipelineStages::TASK_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::TASK_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "task_shader",
                    )])]),
                    vuids: &["VUID-vkCmdResetEvent2-stageMask-03935"],
                }));
            }
        }

        if !(device.enabled_features().attachment_fragment_shading_rate
            || device.enabled_features().shading_rate_image)
        {
            if stages.intersects(PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::FRAGMENT_SHADING_RATE_ATTACHMENT`".into(),
                    requires_one_of: RequiresOneOf(&[
                        RequiresAllOf(&[Requires::DeviceFeature(
                            "attachment_fragment_shading_rate",
                        )]),
                        RequiresAllOf(&[Requires::DeviceFeature("shading_rate_image")]),
                    ]),
                    vuids: &["VUID-VkImageMemoryBarrier2-shadingRateImage-07316"],
                }));
            }
        }

        if !device.enabled_features().subpass_shading {
            if stages.intersects(PipelineStages::SUBPASS_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::SUBPASS_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "subpass_shading",
                    )])]),
                    vuids: &["VUID-vkCmdResetEvent2-stageMask-04957"],
                }));
            }
        }

        if !device.enabled_features().invocation_mask {
            if stages.intersects(PipelineStages::INVOCATION_MASK) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::INVOCATION_MASK`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "invocation_mask",
                    )])]),
                    vuids: &["VUID-vkCmdResetEvent2-stageMask-04995"],
                }));
            }
        }

        if !(device.enabled_extensions().nv_ray_tracing
            || device.enabled_features().ray_tracing_pipeline)
        {
            if stages.intersects(PipelineStages::RAY_TRACING_SHADER) {
                return Err(Box::new(ValidationError {
                    context: "stages".into(),
                    problem: "contains `PipelineStages::RAY_TRACING_SHADER`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "ray_tracing_pipeline",
                    )])]),
                    vuids: &["VUID-vkCmdResetEvent2-stageMask-07946"],
                }));
            }
        }

        if stages.intersects(PipelineStages::HOST) {
            return Err(Box::new(ValidationError {
                context: "stages".into(),
                problem: "contains `PipelineStages::HOST`".into(),
                vuids: &["VUID-vkCmdResetEvent2-stageMask-03830"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn reset_event_unchecked(
        &mut self,
        event: &Event,
        stages: PipelineStages,
    ) -> &mut Self {
        let fns = self.device().fns();

        if self.device().enabled_features().synchronization2 {
            if self.device().api_version() >= Version::V1_3 {
                unsafe {
                    (fns.v1_3.cmd_reset_event2)(self.handle(), event.handle(), stages.into())
                };
            } else {
                unsafe {
                    (fns.khr_synchronization2.cmd_reset_event2_khr)(
                        self.handle(),
                        event.handle(),
                        stages.into(),
                    )
                };
            }
        } else {
            unsafe { (fns.v1_0.cmd_reset_event)(self.handle(), event.handle(), stages.into()) };
        }

        self
    }
}
