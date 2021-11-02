// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::acceleration_struct::AccelerationStructure;
use super::BottomLevelAccelerationStructure;
use crate::device::Device;
use std::mem::size_of;
use std::sync::Arc;

pub struct TopLevelAccelerationStructure {
    acceleration_structure: AccelerationStructure,
    /// Here we keep strong refs to bottom level acceleration structures
    /// to prevent them from being dropped
    bottom_acceleration_structures: Box<[Arc<BottomLevelAccelerationStructure>]>,
    /// Here are stored instances of bottom level acceleration structures
    instances: Box<[ash::vk::AccelerationStructureInstanceKHR]>,
}

fn make_instance(
    transform: [[f32; 4]; 3],
    bottom: ash::vk::AccelerationStructureKHR,
    custom_index: u32,
    shader_record_offset: u32,
) -> ash::vk::AccelerationStructureInstanceKHR {
    let matrix: [f32; 12] = unsafe { std::mem::transmute(transform) };

    let transform = ash::vk::TransformMatrixKHR { matrix };

    let mask = 0xFF;
    let flags = 0;

    let bottom_ref = ash::vk::AccelerationStructureReferenceKHR {
        host_handle: bottom,
    };

    ash::vk::AccelerationStructureInstanceKHR {
        transform,
        instance_custom_index_and_mask: (mask << 24) | custom_index,
        instance_shader_binding_table_record_offset_and_flags: (flags << 24) | shader_record_offset,
        acceleration_structure_reference: bottom_ref,
    }
}

unsafe fn make_instances_data(
    instances: *const ash::vk::AccelerationStructureInstanceKHR,
) -> ash::vk::AccelerationStructureGeometryInstancesDataKHR {
    let data = ash::vk::DeviceOrHostAddressConstKHR {
        host_address: instances as *const _,
    };

    let stride = size_of::<ash::vk::AccelerationStructureGeometryInstancesDataKHR>() as u64;

    ash::vk::AccelerationStructureGeometryInstancesDataKHR::builder()
        .array_of_pointers(false)
        .data(data)
        .build()
}

impl TopLevelAccelerationStructure {
    pub fn builder(device: Arc<Device>) -> TopLevelAccelerationStructureBuilder {
        TopLevelAccelerationStructureBuilder {
            device,
            bottom_acceleration_structures: Vec::new(),
            instances: Vec::new(),
        }
    }
}

pub struct TopLevelAccelerationStructureBuilder {
    device: Arc<Device>,
    bottom_acceleration_structures: Vec<Arc<BottomLevelAccelerationStructure>>,
    instances: Vec<ash::vk::AccelerationStructureInstanceKHR>,
}

impl TopLevelAccelerationStructureBuilder {
    pub fn add_instance(
        &mut self,
        bottom_structure: Arc<BottomLevelAccelerationStructure>,
        transform: [[f32; 4]; 3],
        instance_custom_index: u32,
        instance_shader_binding_table_record_offset: u32,
    ) {
        let instance = make_instance(
            transform,
            bottom_structure.acceleration_structure.inner,
            instance_custom_index,
            instance_shader_binding_table_record_offset,
        );

        self.bottom_acceleration_structures.push(bottom_structure);
        self.instances.push(instance);
    }

    pub fn build(self) -> TopLevelAccelerationStructure {
        let bottom_acceleration_structures = self.bottom_acceleration_structures.into_boxed_slice();
        let instances = self.instances.into_boxed_slice();

        // SAFETY
        //
        // Instances are stored in the boxed slice
        // and dropped only when `TopLevelAccelerationStructure` is dropped
        //
        let instances_data = unsafe { make_instances_data(instances.as_ptr()) };

        let geometry_data = ash::vk::AccelerationStructureGeometryDataKHR {
            instances: instances_data,
        };

        let geometry = ash::vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(ash::vk::GeometryTypeKHR::INSTANCES)
            .geometry(geometry_data)
            .build();

        let acceleration_structure = AccelerationStructure::new(
            self.device,
            std::slice::from_ref(&geometry),
            std::iter::once(instances.len() as u32),
            ash::vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        );

        TopLevelAccelerationStructure {
            acceleration_structure,
            bottom_acceleration_structures,
            instances,
        }
    }
}
