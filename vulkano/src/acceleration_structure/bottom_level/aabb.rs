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
use std::fmt;
use std::mem::size_of;
use std::sync::Arc;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct AabbPosition {
    min: [f32; 3],
    max: [f32; 3],
}

#[derive(Debug)]
pub struct IncorrectAabb;

impl fmt::Display for IncorrectAabb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AABB must not have max that's smaller than min in any dimension"
        )
    }
}

impl std::error::Error for IncorrectAabb {}

impl AabbPosition {
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Result<AabbPosition, IncorrectAabb> {
        let [min_x, min_y, min_z] = min;
        let [max_x, max_y, max_z] = max;
        if max_x < min_x || max_y < min_y || max_z < min_z {
            return Err(IncorrectAabb);
        }
        Ok(Self { min, max })
    }

    pub fn min(&self) -> [f32; 3] {
        self.min
    }

    pub fn max(&self) -> [f32; 3] {
        self.max
    }
}

fn make_aabb_data(aabb: &dyn BufferAccess) -> ash::vk::AccelerationStructureGeometryAabbsDataKHR {
    let data = ash::vk::DeviceOrHostAddressConstKHR {
        device_address: aabb.inner().buffer.internal_object().as_raw(),
    };

    debug_assert_eq!(
        size_of::<AabbPosition>(),
        size_of::<ash::vk::AabbPositionsKHR>()
    );

    let stride = size_of::<AabbPosition>() as u64;

    ash::vk::AccelerationStructureGeometryAabbsDataKHR::builder()
        .data(data)
        .stride(stride)
        .build()
}

impl BottomLevelAccelerationStructure {
    pub fn new_aabbs(
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

        let data = BottomLevelData::Aabbs {
            buffer: aabbs_buffer,
        };

        Self {
            acceleration_structure,
            data,
        }
    }
}
