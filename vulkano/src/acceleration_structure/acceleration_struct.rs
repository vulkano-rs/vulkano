// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::DeviceLocalBuffer;
use crate::buffer::{BufferAccess, BufferUsage};
use crate::check_errors;
use crate::device::Device;
use crate::VulkanObject;
use ash::vk::Handle;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

pub(crate) struct AccelerationStructure {
    pub(crate) inner: ash::vk::AccelerationStructureKHR,

    /// Buffer in which the acceleration struct is stored
    buffer: Arc<DeviceLocalBuffer<[u8]>>,
}

impl AccelerationStructure {
    pub(crate) fn new(
        device: Arc<Device>,
        geometries: &[ash::vk::AccelerationStructureGeometryKHR],
        primitives_count: impl Iterator<Item = u32>,
        ty: ash::vk::AccelerationStructureTypeKHR,
    ) -> AccelerationStructure {
        let fns = device.fns();

        let build_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::builder()
            .ty(ty)
            .flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(geometries);

        let size = unsafe {
            let mut output = MaybeUninit::uninit();
            fns.khr_acceleration_structure
                .get_acceleration_structure_build_sizes_khr(
                    device.internal_object(),
                    ash::vk::AccelerationStructureBuildTypeKHR::HOST,
                    &*build_info as *const _,
                    &(geometries.len() as u32) as *const _,
                    output.as_mut_ptr(),
                );
            output.assume_init()
        };

        let buffer_usage = BufferUsage {
            device_address: true,
            acceleration_structure_storage: true,
            ..BufferUsage::none()
        };

        let buffer = unsafe {
            DeviceLocalBuffer::raw(
                device.clone(),
                size.acceleration_structure_size,
                buffer_usage,
                device.active_queue_families(),
            )
            .unwrap()
        };

        let create_info = ash::vk::AccelerationStructureCreateInfoKHR::builder()
            .buffer(buffer.inner().buffer.internal_object())
            .size(size.acceleration_structure_size)
            .ty(ty)
            .build();

        let acceleration_structure = unsafe {
            let mut output = MaybeUninit::uninit();
            let result = fns
                .khr_acceleration_structure
                .create_acceleration_structure_khr(
                    device.internal_object(),
                    &create_info as *const _,
                    ptr::null(),
                    output.as_mut_ptr(),
                );
            check_errors(result).unwrap();
            output.assume_init()
        };

        let scratch_buffer_usage = BufferUsage {
            storage_buffer: true,
            device_address: true,
            ..BufferUsage::none()
        };

        let scratch_buffer = unsafe {
            DeviceLocalBuffer::<[u8]>::raw(
                device.clone(),
                size.build_scratch_size,
                scratch_buffer_usage,
                device.active_queue_families(),
            )
            .unwrap()
        };

        let scratch_buffer_address = ash::vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer.inner().buffer.internal_object().as_raw(),
        };

        let build_info = build_info
            .mode(ash::vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(acceleration_structure)
            .scratch_data(scratch_buffer_address)
            .build();

        let build_ranges: Vec<_> = primitives_count
            .map(|count| {
                ash::vk::AccelerationStructureBuildRangeInfoKHR::builder()
                    .primitive_count(count)
                    .build()
            })
            .collect();

        assert_eq!(build_ranges.len(), geometries.len());

        let build_ranges_ptr = build_ranges.as_ptr();

        unsafe {
            let result = fns
                .khr_acceleration_structure
                .build_acceleration_structures_khr(
                    device.internal_object(),
                    ash::vk::DeferredOperationKHR::null(),
                    1,
                    &build_info as *const _,
                    &build_ranges_ptr as *const *const _,
                );
            check_errors(result).unwrap();
        };

        AccelerationStructure {
            inner: acceleration_structure,
            buffer,
        }
    }
}
