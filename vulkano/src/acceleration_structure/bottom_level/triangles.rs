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
use crate::pipeline::input_assembly::Index;
use crate::format::Format;
use ash::vk::Handle;
use std::sync::Arc;
use crate::DeviceSize;

/// Return data contrains references to input buffers
unsafe fn make_triangles_data<I: Index>(
    vertex_buffer: &dyn BufferAccess,
    vertex_format: Format,
    vertex_stride: DeviceSize,
    index_buffer: &dyn BufferAccess,
) -> ash::vk::AccelerationStructureGeometryTrianglesDataKHR {
    let vertex_data = ash::vk::DeviceOrHostAddressConstKHR {
        device_address: vertex_buffer.inner().buffer.internal_object().as_raw(),
    };

    let index_data = ash::vk::DeviceOrHostAddressConstKHR {
        device_address: index_buffer.inner().buffer.internal_object().as_raw(),
    };

    ash::vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
        .vertex_format(vertex_format.into())
        .vertex_data(vertex_data)
        .vertex_stride(vertex_stride)
        .max_vertex((vertex_buffer.size() / vertex_stride) as u32)
        .index_type(I::ty().into())
        .index_data(index_data)
        // .transform_data(transform_data) // TODO
        .build()
}

impl BottomLevelAccelerationStructure {
    pub fn new_triangles<Ib, I>(
        device: Arc<Device>,
        vertex_buffer: Arc<dyn BufferAccess>,
        vertex_format: Format,
        index_buffer: Arc<Ib>, 
    ) -> Self
    where
        Ib: BufferAccess + TypedBufferAccess<Content = [I]> + Send + Sync + 'static,
        I: Index + 'static,
    {
        // SAFETY: we prevent buffers from deallocate by storing them `BottomLevelData`
        let triangles = unsafe {
            make_triangles_data::<I>(
                &vertex_buffer,
                vertex_format,
                vertex_format.size().unwrap(),
                &index_buffer,
            )
        };
        
        let geometry_data = ash::vk::AccelerationStructureGeometryDataKHR {
            triangles,
        };

        let geometry = ash::vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(ash::vk::GeometryTypeKHR::TRIANGLES)
            .geometry(geometry_data)
            .build();

        let triangles_size = index_buffer.len() / 3;

        let acceleration_structure = AccelerationStructure::new(
            device,
            std::slice::from_ref(&geometry),
            std::iter::once(triangles_size as u32),
            ash::vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        );

        let data = BottomLevelData::Triangles {
            vertex_buffer,
            index_buffer,
        };

        Self {
            acceleration_structure,
            data,
        }
    }
}
