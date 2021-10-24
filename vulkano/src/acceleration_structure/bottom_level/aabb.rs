// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::{AccelerationStructure, BottomLevelAccelerationStructure, BottomLevelData};
use crate::buffer::BufferAccess;
use crate::buffer::TypedBufferAccess;
use crate::device::Device;
use crate::VulkanObject;
use ash::vk::Handle;
use std::mem::size_of;
use std::sync::Arc;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct AabbPosition {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

fn make_aabb_data(aabb: &dyn BufferAccess) -> ash::vk::AccelerationStructureGeometryAabbsDataKHR {
    let data = ash::vk::DeviceOrHostAddressConstKHR {
        device_address: aabb.inner().buffer.internal_object().as_raw(),
    };

    debug_assert_eq!(size_of::<AabbPosition>(), size_of::<ash::vk::AabbPositionsKHR>());

    let stride = size_of::<AabbPosition>() as u64;

    ash::vk::AccelerationStructureGeometryAabbsDataKHR::builder()
        .data(data)
        .stride(stride)
        .build()
}

impl BottomLevelAccelerationStructure {
    pub fn new_aabb(
        device: Arc<Device>,
        aabbs_buffer: Arc<dyn TypedBufferAccess<Content = [AabbPosition]>>,
    ) -> Self {
        let geometry_data = ash::vk::AccelerationStructureGeometryDataKHR {
            aabbs: make_aabb_data(&aabbs_buffer),
        };

        let geometry = ash::vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(ash::vk::GeometryTypeKHR::AABBS)
            .geometry(geometry_data)
            .build();

        let acceleration_structure = AccelerationStructure::new(
            device,
            std::slice::from_ref(&geometry),
            std::iter::once(aabbs_buffer.len() as u32),
            ash::vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        );

        let data = BottomLevelData::Aabb {
            buffer: aabbs_buffer,
        };

        Self {
            acceleration_structure,
            data,
        }
    }
}
