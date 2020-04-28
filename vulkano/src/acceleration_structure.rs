// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::error;
use std::fmt;
use std::mem::size_of;
use std::mem::MaybeUninit;
use std::ops;
use std::ptr;

use buffer::{BufferAccess, BufferUsage, ImmutableBuffer};
use device::Queue;
use device::{Device, DeviceOwned};
use format::Format;
use memory::pool::{
    AllocFromRequirementsFilter, AllocLayout, MappingRequirement, MemoryPoolAlloc,
    PotentialDedicatedAllocation, StdMemoryPoolAlloc,
};
use memory::{DedicatedAlloc, MemoryPool, MemoryRequirements};
use pipeline::input_assembly::IndexType;
use smallvec::alloc::sync::Arc;
use std::os::raw::c_void;
use sync::GpuFuture;

use check_errors;
use vk;
use Error;
use OomError;
use VulkanObject;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct AabbPositions {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

#[derive(Clone)]
pub struct Level {
    pub inner_object: vk::AccelerationStructureKHR,
    pub type_: vk::AccelerationStructureTypeKHR,
    pub flags: vk::BuildAccelerationStructureFlagsKHR,
    pub instance_count: u32,
    pub geometries: Vec<Geometry>,
}

/// Structure used by rays to find geometry in a scene
pub struct AccelerationStructure {
    device: Arc<Device>,
    nv_extension: bool,
    top_level: Level,
    bottom_level: Level,

    // Memory allocated for the acceleration structure.
    memory: PotentialDedicatedAllocation<StdMemoryPoolAlloc>,

    instance_buffer: Arc<ImmutableBuffer<vk::AccelerationStructureInstanceKHR>>,
    scratch_buffer: Arc<ImmutableBuffer<u8>>,
}

impl AccelerationStructure {
    /// Creates a new `AccelerationStructure` using the `nv_ray_tracing` extension
    #[inline]
    pub fn nv() -> AccelerationStructureBuilder {
        AccelerationStructureBuilder::nv()
    }

    /// Creates a new `AccelerationStructure` using the `khr_ray_tracing` extension
    #[inline]
    pub fn khr() -> AccelerationStructureBuilder {
        AccelerationStructureBuilder::khr()
    }

    #[inline]
    pub fn nv_extension(&self) -> bool {
        self.nv_extension
    }

    #[inline]
    pub fn top_level(&self) -> Level {
        self.top_level.clone()
    }

    #[inline]
    pub fn bottom_level(&self) -> Level {
        self.bottom_level.clone()
    }

    #[inline]
    pub fn instance_buffer(&self) -> &Arc<ImmutableBuffer<vk::AccelerationStructureInstanceNV>> {
        &self.instance_buffer
    }

    #[inline]
    pub fn scratch_buffer(&self) -> &Arc<ImmutableBuffer<u8>> {
        &self.scratch_buffer
    }
}

struct AccelerationStructureBuilderTriangles {
    vertex_buffer: Arc<dyn BufferAccess + Send + Sync>,
    vertex_stride: vk::DeviceSize,
    vertex_format: Format,
    index_buffer: Arc<dyn BufferAccess + Send + Sync>,
    index_type: IndexType,
}

/// Prototype of a `AccelerationStructure`.
///
/// See the docs of `AccelerationStructure` for an example.
pub struct AccelerationStructureBuilder {
    nv_extension: bool,
    // TODO: Associate `BuildAccelerationStructureFlags`
    triangles: Vec<AccelerationStructureBuilderTriangles>,
    aabbs: Vec<Box<dyn BufferAccess + Send + Sync>>,
}

impl AccelerationStructureBuilder {
    #[inline]
    pub fn nv() -> Self {
        AccelerationStructureBuilder {
            nv_extension: true,
            triangles: vec![],
            aabbs: vec![],
        }
    }

    #[inline]
    pub fn khr() -> Self {
        AccelerationStructureBuilder {
            nv_extension: false,
            triangles: vec![],
            aabbs: vec![],
        }
    }

    /// Builds an `AccelerationStructure` from the builder
    #[inline]
    pub fn build(
        self,
        device: Arc<Device>,
        queue: Arc<Queue>,
    ) -> Result<AccelerationStructure, AccelerationStructureCreationError> {
        if self.nv_extension {
            self.build_nv(device, queue)
        } else {
            self.build_khr(device, queue)
        }
    }

    /// Builds an `AccelerationStructure` from the builder using the
    /// `nv_ray_tracing` extension
    #[inline]
    fn build_nv(
        self,
        device: Arc<Device>,
        queue: Arc<Queue>,
    ) -> Result<AccelerationStructure, AccelerationStructureCreationError> {
        let vk = device.pointers();

        // TODO: Expose to user
        let build_flag = vk::BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_NV;

        let geometries: Vec<vk::GeometryNV> = self
            .aabbs
            .into_iter()
            .map(|aabb| {
                let geometry = vk::GeometryAABBNV {
                    sType: vk::STRUCTURE_TYPE_GEOMETRY_AABB_NV,
                    pNext: ptr::null(),
                    aabbData: aabb.inner().buffer.internal_object(),
                    numAABBs: (aabb.size() / size_of::<AabbPositions>()) as u32,
                    stride: size_of::<AabbPositions>() as u32, // TODO:
                    offset: aabb.inner().offset as vk::DeviceSize,
                };
                vk::GeometryNV {
                    sType: vk::STRUCTURE_TYPE_GEOMETRY_NV,
                    pNext: ptr::null(),
                    geometryType: vk::GEOMETRY_TYPE_AABBS_NV,
                    geometry: vk::GeometryDataNV {
                        triangles: vk::GeometryTrianglesNV {
                            sType: vk::STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV,
                            pNext: ptr::null(),
                            vertexData: 0,
                            vertexOffset: 0,
                            vertexCount: 0,
                            vertexStride: 0,
                            vertexFormat: 0,
                            indexData: 0,
                            indexOffset: 0,
                            indexCount: 0,
                            indexType: 0,
                            transformData: 0,
                            transformOffset: 0,
                        },
                        aabbs: geometry,
                    },
                    flags: vk::GEOMETRY_OPAQUE_BIT_NV, // TODO
                }
            })
            .chain(self.triangles.into_iter().map(|ref triangle| {
                let index_stride = match triangle.index_type {
                    IndexType::U16 => size_of::<u16>(),
                    IndexType::U32 => size_of::<u32>(),
                };
                let geometry = vk::GeometryTrianglesNV {
                    sType: vk::STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV,
                    pNext: ptr::null(),
                    vertexData: triangle.vertex_buffer.inner().buffer.internal_object(),
                    vertexOffset: triangle.vertex_buffer.inner().offset as vk::DeviceSize,
                    vertexCount: (triangle.vertex_buffer.size() / triangle.vertex_stride as usize)
                        as u32,
                    vertexStride: triangle.vertex_stride,
                    vertexFormat: triangle.vertex_format as u32,
                    indexData: triangle.index_buffer.inner().buffer.internal_object(),
                    indexOffset: triangle.index_buffer.inner().offset as vk::DeviceSize,
                    indexCount: (triangle.index_buffer.size() / index_stride as usize) as u32,
                    indexType: triangle.index_type as u32,
                    transformData: vk::NULL_HANDLE,
                    transformOffset: 0,
                };
                vk::GeometryNV {
                    sType: vk::STRUCTURE_TYPE_GEOMETRY_NV,
                    pNext: ptr::null(),
                    geometryType: vk::GEOMETRY_TYPE_TRIANGLES_NV,
                    geometry: vk::GeometryDataNV {
                        triangles: geometry,
                        aabbs: vk::GeometryAABBNV {
                            sType: vk::STRUCTURE_TYPE_GEOMETRY_AABB_NV,
                            pNext: ptr::null(),
                            aabbData: 0,
                            numAABBs: 0,
                            stride: 0,
                            offset: 0,
                        },
                    },
                    flags: vk::GEOMETRY_OPAQUE_BIT_NV, // TODO
                }
            }))
            .collect();

        let bottom_level = {
            let info = vk::AccelerationStructureInfoNV {
                sType: vk::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV,
                pNext: ptr::null(),
                type_: vk::ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV,
                flags: build_flag,
                instanceCount: 0, // must be 0 for bottom level
                geometryCount: geometries.len() as u32,
                pGeometries: geometries.as_ptr(),
            };
            let create_info = vk::AccelerationStructureCreateInfoNV {
                sType: vk::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV,
                pNext: ptr::null(),
                compactedSize: 0, // TODO:
                info,
            };
            let structure = unsafe {
                let mut output = MaybeUninit::uninit();
                check_errors(vk.CreateAccelerationStructureNV(
                    device.internal_object(),
                    &create_info,
                    ptr::null(),
                    output.as_mut_ptr(),
                ))?;
                output.assume_init()
            };
            Level {
                inner_object: structure,
                type_: create_info.info.type_,
                flags: create_info.info.flags,
                instance_count: create_info.info.instanceCount,
                geometries: geometries.iter().map(|g| Geometry::from(g)).collect(),
            }
        };

        let top_level = {
            let info = vk::AccelerationStructureInfoNV {
                sType: vk::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV,
                pNext: ptr::null(),
                type_: vk::ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV,
                flags: 0,
                instanceCount: 1, // number bottom level acceleration structure
                geometryCount: 0, // must be 0 for top level
                pGeometries: ptr::null(),
            };
            let create_info = vk::AccelerationStructureCreateInfoNV {
                sType: vk::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV,
                pNext: ptr::null(),
                compactedSize: 0, // TODO:
                info,
            };
            let structure = unsafe {
                let mut output = MaybeUninit::uninit();
                check_errors(vk.CreateAccelerationStructureNV(
                    device.internal_object(),
                    &create_info,
                    ptr::null(),
                    output.as_mut_ptr(),
                ))?;
                output.assume_init()
            };
            Level {
                inner_object: structure,
                type_: create_info.info.type_,
                flags: create_info.info.flags,
                instance_count: create_info.info.instanceCount,
                geometries: vec![],
            }
        };

        let structures = [&top_level, &bottom_level];

        // Get requirements
        let memory_requirements: Vec<vk::MemoryRequirements2> = structures
            .iter()
            .map(|structure| unsafe {
                let info = vk::AccelerationStructureMemoryRequirementsInfoNV {
                    sType: vk::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV,
                    pNext: ptr::null(),
                    type_: vk::ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV,
                    accelerationStructure: structure.inner_object,
                };
                let mut requirements = vk::MemoryRequirements2 {
                    sType: vk::STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR,
                    pNext: ptr::null_mut(),
                    memoryRequirements: vk::MemoryRequirements {
                        size: 0,
                        alignment: 0,
                        memoryTypeBits: 0,
                    },
                };
                vk.GetAccelerationStructureMemoryRequirementsNV(
                    device.internal_object(),
                    &info,
                    &mut requirements,
                );
                requirements
            })
            .collect();

        // Allocate structure memory
        let memory = {
            let requirements = MemoryRequirements {
                size: memory_requirements
                    .iter()
                    .map(|requirements| requirements.memoryRequirements.size)
                    .sum::<vk::DeviceSize>() as usize,
                alignment: memory_requirements
                    .first()
                    .unwrap()
                    .memoryRequirements
                    .alignment as usize,
                memory_type_bits: memory_requirements
                    .first()
                    .unwrap()
                    .memoryRequirements
                    .memoryTypeBits as u32,
                prefer_dedicated: true,
            };
            MemoryPool::alloc_from_requirements(
                &Device::standard_pool(&device),
                &requirements,
                AllocLayout::Optimal,
                MappingRequirement::DoNotMap,
                DedicatedAlloc::None,
                |t| AllocFromRequirementsFilter::Preferred,
            )
            .unwrap()
        };

        // Bind memory to structure
        let mut memory_offset = 0;
        let bind_infos: Vec<vk::BindAccelerationStructureMemoryInfoNV> = structures
            .iter()
            .zip(memory_requirements.iter())
            .map(|(structure, requirements)| {
                let info = vk::BindAccelerationStructureMemoryInfoNV {
                    sType: vk::STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV,
                    pNext: ptr::null(),
                    accelerationStructure: structure.inner_object,
                    memory: memory.memory().internal_object(),
                    memoryOffset: memory_offset,
                    deviceIndexCount: 0,         // TODO:
                    pDeviceIndices: ptr::null(), // TODO:
                };
                memory_offset += requirements.memoryRequirements.size;
                info
            })
            .collect();
        unsafe {
            check_errors(vk.BindAccelerationStructureMemoryNV(
                device.internal_object(),
                bind_infos.len() as u32,
                bind_infos.as_ptr(),
            ))?;
        };

        let (instance_buffer, instance_buffer_future) = {
            // Get handles for lower level structures
            let bottom_level_handle = unsafe {
                let mut output: u64 = 0;
                check_errors(vk.GetAccelerationStructureHandleNV(
                    device.internal_object(),
                    bottom_level.inner_object.clone(),
                    size_of::<u64>(),
                    &mut output as *mut u64 as *mut c_void,
                ))?;
                output
            };

            let instance = vk::AccelerationStructureInstanceNV {
                transform: vk::TransformMatrixNV {
                    matrix: [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ],
                },
                instanceCustomIndex: [0, 0, 0],
                mask: 0xFF,
                instanceShaderBindingTableRecordOffset: [0, 0, 0],
                flags: 0,
                accelerationStructureReference: bottom_level_handle,
            };
            ImmutableBuffer::from_data(instance, BufferUsage::ray_tracing(), queue.clone()).unwrap()
        };

        // Allocate Scratch of size of the max of any level
        let (scratch_buffer, scratch_buffer_initialization) = unsafe {
            let size = structures
                .iter()
                .map(|structure| unsafe {
                    let info = vk::AccelerationStructureMemoryRequirementsInfoNV {
                        sType:
                            vk::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV,
                        pNext: ptr::null(),
                        type_: vk::ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV,
                        accelerationStructure: structure.inner_object,
                    };
                    let mut requirements = vk::MemoryRequirements2 {
                        sType: vk::STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR,
                        pNext: ptr::null_mut(),
                        memoryRequirements: vk::MemoryRequirements {
                            size: 0,
                            alignment: 0,
                            memoryTypeBits: 0,
                        },
                    };
                    vk.GetAccelerationStructureMemoryRequirementsNV(
                        device.internal_object(),
                        &info,
                        &mut requirements,
                    );
                    requirements.memoryRequirements.size
                })
                .max()
                .unwrap();

            ImmutableBuffer::<u8>::raw(
                device.clone(),
                size as usize,
                BufferUsage::ray_tracing(),
                device.active_queue_families(),
            )
        }
        .unwrap();

        instance_buffer_future
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        Ok(AccelerationStructure {
            device: device.clone(),
            nv_extension: true,
            top_level,
            bottom_level,
            memory,
            instance_buffer,
            scratch_buffer,
        })
    }

    /// Builds an `AccelerationStructure` from the builder using the
    /// `khr_ray_tracing` extension
    #[inline]
    fn build_khr(
        self,
        device: Arc<Device>,
        queue: Arc<Queue>,
    ) -> Result<AccelerationStructure, AccelerationStructureCreationError> {
        let vk = device.pointers();

        // TODO: Expose to user
        let build_flag = vk::BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

        let bottom_level = {
            let geometry_infos: Vec<vk::AccelerationStructureCreateGeometryTypeInfoKHR> = self
                .aabbs
                .iter()
                .map(|aabb| {
                    vk::AccelerationStructureCreateGeometryTypeInfoKHR {
                        sType: vk::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_GEOMETRY_TYPE_INFO_KHR,
                        pNext: ptr::null(),
                        geometryType: vk::GEOMETRY_TYPE_AABBS_KHR,
                        maxPrimitiveCount: (aabb.size() / size_of::<AabbPositions>()) as u32,
                        indexType: 0,
                        maxVertexCount: 0,
                        vertexFormat: 0,
                        allowsTransforms: vk::FALSE,
                    }
                })
                .chain(self.triangles.iter().map(|ref triangle| {
                    let index_stride = match triangle.index_type {
                        IndexType::U16 => size_of::<u16>(),
                        IndexType::U32 => size_of::<u32>(),
                    };
                    vk::AccelerationStructureCreateGeometryTypeInfoKHR {
                        sType: vk::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_GEOMETRY_TYPE_INFO_KHR,
                        pNext: ptr::null(),
                        geometryType: vk::GEOMETRY_TYPE_TRIANGLES_KHR,
                        maxPrimitiveCount: (triangle.index_buffer.size() / index_stride as usize) as u32 / 3,
                        indexType: triangle.index_type as u32,
                        maxVertexCount: (triangle.vertex_buffer.size() / triangle.vertex_stride as usize) as u32,
                        vertexFormat: triangle.vertex_format as u32,
                        allowsTransforms: vk::FALSE, // TODO
                    }
                }))
            .collect();
            let create_info = vk::AccelerationStructureCreateInfoKHR {
                sType: vk::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                pNext: ptr::null(),
                compactedSize: 0, // TODO:
                type_: vk::ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                flags: build_flag,
                maxGeometryCount: geometry_infos.len() as u32,
                pGeometryInfos: geometry_infos.as_ptr(),
                deviceAddress: 0, // TODO: if the rayTracingAccelerationStructureCaptureReplay feature is being used
            };
            let structure = unsafe {
                let mut output = MaybeUninit::uninit();
                check_errors(vk.CreateAccelerationStructureKHR(
                    device.internal_object(),
                    &create_info,
                    ptr::null(),
                    output.as_mut_ptr(),
                ))?;
                output.assume_init()
            };
            Level {
                inner_object: structure,
                type_: create_info.type_,
                flags: create_info.flags,
                instance_count: create_info.maxGeometryCount,
                geometries: self
                    .aabbs
                    .into_iter()
                    .map(|aabb| {
                        Geometry {
                            geometry_type: GeometryType::Aabbs,
                            geometry: GeometryData {
                                aabbs: GeometryAabbsData {
                                    data: aabb.inner().buffer.internal_object() as u64,
                                    stride: size_of::<AabbPositions>(), // TODO:
                                    count: (aabb.size() / size_of::<AabbPositions>()) as u32,
                                },
                            },
                            flags: GeometryFlags {
                                opaque: true,
                                no_duplicate_any_hit_invocation: false,
                            },
                        }
                    })
                    .chain(self.triangles.into_iter().map(|triangle|{
                        let index_stride = match triangle.index_type {
                            IndexType::U16 => size_of::<u16>(),
                            IndexType::U32 => size_of::<u32>(),
                        };
                        Geometry {
                            geometry_type: GeometryType::Triangles,
                            geometry: GeometryData {
                                triangles: GeometryTrianglesData {
                                    vertex_data: triangle.vertex_buffer.inner().buffer.internal_object(),
                                    vertex_offset: triangle.vertex_buffer.inner().offset as vk::DeviceSize,
                                    vertex_count: (triangle.vertex_buffer.size() / triangle.vertex_stride as usize) as u32,
                                    vertex_stride: triangle.vertex_stride as usize,
                                    vertex_format: triangle.vertex_format,
                                    index_data: triangle.index_buffer.inner().buffer.internal_object(),
                                    index_offset: triangle.index_buffer.inner().offset as vk::DeviceSize,
                                    index_count: (triangle.index_buffer.size() / index_stride as usize) as u32,
                                    index_type: triangle.index_type,
                                    transform_data: vk::NULL_HANDLE,
                                    transform_offset: 0,
                                },
                            },
                            flags: GeometryFlags {
                                opaque: true,
                                no_duplicate_any_hit_invocation: false,
                            },
                        }
                    }))
                    .collect(),
            }
        };

        let top_level = {
            let info = vk::AccelerationStructureCreateGeometryTypeInfoKHR {
                sType: vk::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_GEOMETRY_TYPE_INFO_KHR,
                pNext: ptr::null(),
                geometryType: vk::GEOMETRY_TYPE_INSTANCES_KHR,
                maxPrimitiveCount: 1, // number bottom level acceleration structure
                indexType: 0,
                maxVertexCount: 0,
                vertexFormat: 0,
                allowsTransforms: 0
            };
            let create_info = vk::AccelerationStructureCreateInfoKHR {
                sType: vk::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                pNext: ptr::null(),
                compactedSize: 0, // TODO:
                type_: vk::ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                flags: build_flag,
                maxGeometryCount: 1, // if compactedSize is 0, maxGeometryCount must be 1 for top level
                pGeometryInfos: &info,
                deviceAddress: 0, // TODO: if the rayTracingAccelerationStructureCaptureReplay feature is being used
            };
            let structure = unsafe {
                let mut output = MaybeUninit::uninit();
                check_errors(vk.CreateAccelerationStructureKHR(
                    device.internal_object(),
                    &create_info,
                    ptr::null(),
                    output.as_mut_ptr(),
                ))?;
                output.assume_init()
            };
            Level {
                inner_object: structure,
                type_: create_info.type_,
                flags: create_info.flags,
                instance_count: create_info.maxGeometryCount,
                geometries: vec![],
            }
        };

        let structures = [&top_level, &bottom_level];

        // Get requirements
        let memory_requirements: Vec<vk::MemoryRequirements2> = structures
            .iter()
            .map(|structure| unsafe {
                let info = vk::AccelerationStructureMemoryRequirementsInfoKHR {
                    sType: vk::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_KHR,
                    pNext: ptr::null(),
                    type_: vk::ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_KHR,
                    buildType: vk::ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_OR_DEVICE_KHR, // TODO
                    accelerationStructure: structure.inner_object,
                };
                let mut requirements = vk::MemoryRequirements2 {
                    sType: vk::STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR,
                    pNext: ptr::null_mut(),
                    memoryRequirements: vk::MemoryRequirements {
                        size: 0,
                        alignment: 0,
                        memoryTypeBits: 0,
                    },
                };
                vk.GetAccelerationStructureMemoryRequirementsKHR(
                    device.internal_object(),
                    &info,
                    &mut requirements,
                );
                requirements
            })
            .collect();

        // Allocate structure memory
        let memory = {
            let requirements = MemoryRequirements {
                size: memory_requirements
                    .iter()
                    .map(|requirements| requirements.memoryRequirements.size)
                    .sum::<vk::DeviceSize>() as usize,
                alignment: memory_requirements
                    .first()
                    .unwrap()
                    .memoryRequirements
                    .alignment as usize,
                memory_type_bits: memory_requirements
                    .first()
                    .unwrap()
                    .memoryRequirements
                    .memoryTypeBits as u32,
                prefer_dedicated: true,
            };
            MemoryPool::alloc_from_requirements(
                &Device::standard_pool(&device),
                &requirements,
                AllocLayout::Optimal,
                MappingRequirement::DoNotMap,
                DedicatedAlloc::None,
                |t| AllocFromRequirementsFilter::Preferred,
            )
            .unwrap()
        };

        // Bind memory to structure
        let mut memory_offset = 0;
        let bind_infos: Vec<vk::BindAccelerationStructureMemoryInfoNV> = structures
            .iter()
            .zip(memory_requirements.iter())
            .map(|(structure, requirements)| {
                let info = vk::BindAccelerationStructureMemoryInfoKHR {
                    sType: vk::STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_KHR,
                    pNext: ptr::null(),
                    accelerationStructure: structure.inner_object,
                    memory: memory.memory().internal_object(),
                    memoryOffset: memory_offset,
                    deviceIndexCount: 0,         // TODO:
                    pDeviceIndices: ptr::null(), // TODO:
                };
                memory_offset += requirements.memoryRequirements.size;
                info
            })
            .collect();
        unsafe {
            check_errors(vk.BindAccelerationStructureMemoryKHR(
                device.internal_object(),
                bind_infos.len() as u32,
                bind_infos.as_ptr(),
            ))?;
        };

        let (instance_buffer, instance_buffer_future) = {
            // Get handles for lower level structures
            let bottom_level_address = unsafe {
                let info = vk::AccelerationStructureDeviceAddressInfoKHR {
                    sType: vk::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
                    pNext: ptr::null(),
                    accelerationStructure: bottom_level.inner_object.clone(),
                };
                vk.GetAccelerationStructureDeviceAddressKHR(device.internal_object(), &info)
            };

            let instance = vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR {
                    matrix: [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                    ],
                },
                instanceCustomIndex: [0, 0, 0],
                mask: 0xFF,
                instanceShaderBindingTableRecordOffset: [0, 0, 0],
                flags: 0,
                accelerationStructureReference: bottom_level_address,
            };
            ImmutableBuffer::from_data(instance, BufferUsage::ray_tracing(), queue.clone()).unwrap()
        };

        // Allocate Scratch of size of the max of any level
        let (scratch_buffer, scratch_buffer_initialization) = unsafe {
            let size = structures
                .iter()
                .map(|structure| unsafe {
                    let info = vk::AccelerationStructureMemoryRequirementsInfoKHR {
                        sType: vk::STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_KHR,
                        pNext: ptr::null(),
                        type_: vk::ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_KHR,
                        buildType: vk::ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_OR_DEVICE_KHR, // TODO
                        accelerationStructure: structure.inner_object,
                    };
                    let mut requirements = vk::MemoryRequirements2 {
                        sType: vk::STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR,
                        pNext: ptr::null_mut(),
                        memoryRequirements: vk::MemoryRequirements {
                            size: 0,
                            alignment: 0,
                            memoryTypeBits: 0,
                        },
                    };
                    vk.GetAccelerationStructureMemoryRequirementsKHR(
                        device.internal_object(),
                        &info,
                        &mut requirements,
                    );
                    requirements.memoryRequirements.size
                })
                .max()
                .unwrap();

            ImmutableBuffer::<u8>::raw(
                device.clone(),
                size as usize,
                BufferUsage{ray_tracing: true, shader_device_address: true, ..BufferUsage::none()},
                device.active_queue_families(),
            )
        }
        .unwrap();

        instance_buffer_future
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        Ok(AccelerationStructure {
            device: device.clone(),
            nv_extension: false,
            top_level,
            bottom_level,
            memory,
            instance_buffer,
            scratch_buffer,
        })
    }

    /// Add a Triangle Mesh to the acceleration structure
    #[inline]
    pub fn add_triangles<V, I>(
        mut self,
        vertex_buffer: Arc<V>,
        vertex_stride: vk::DeviceSize,
        vertex_format: Format,
        index_buffer: Arc<I>,
        index_type: IndexType,
    ) -> Result<AccelerationStructureBuilder, AccelerationStructureCreationError>
    where
        V: BufferAccess + Send + Sync + 'static,
        I: BufferAccess + Send + Sync + 'static,
    {
        self.triangles.push(AccelerationStructureBuilderTriangles {
            vertex_buffer,
            vertex_stride,
            vertex_format,
            index_buffer,
            index_type,
        });
        Ok(self)
    }

    /// Add Custom Intersection Geometry to the acceleration structure
    #[inline]
    pub fn add_aabbs<T>(
        mut self, buffer: T,
    ) -> Result<AccelerationStructureBuilder, AccelerationStructureCreationError>
    where
        T: BufferAccess + Send + Sync + 'static,
    {
        self.aabbs.push(Box::new(buffer));
        Ok(self)
    }
}

macro_rules! build_acceleration_struct_flags {
    ($($elem:ident => $val:expr,)+) => (
        #[derive(Debug, Copy, Clone)]
        #[allow(missing_docs)]
        pub struct BuildAccelerationStructureFlags {
            $(
                pub $elem: bool,
            )+
        }

        impl BuildAccelerationStructureFlags {
            /// Builds an `BuildAccelerationStructureFlag` struct with all bits set.
            pub fn all() -> BuildAccelerationStructureFlags {
                BuildAccelerationStructureFlags {
                    $(
                        $elem: true,
                    )+
                }
            }

            /// Builds an `BuildAccelerationStructureFlag` struct with none of the bits set.
            pub fn none() -> BuildAccelerationStructureFlags {
                BuildAccelerationStructureFlags {
                    $(
                        $elem: false,
                    )+
                }
            }

            #[inline]
            pub(crate) fn into_vulkan_bits(self) -> vk::BuildAccelerationStructureFlagBitsKHR {
                let mut result = 0;
                $(
                    if self.$elem { result |= $val }
                )+
                result
            }
        }

        impl ops::BitOr for BuildAccelerationStructureFlags {
            type Output = BuildAccelerationStructureFlags;

            #[inline]
            fn bitor(self, rhs: BuildAccelerationStructureFlags) -> BuildAccelerationStructureFlags {
                BuildAccelerationStructureFlags {
                    $(
                        $elem: self.$elem || rhs.$elem,
                    )+
                }
            }
        }

        impl ops::BitOrAssign for BuildAccelerationStructureFlags {
            #[inline]
            fn bitor_assign(&mut self, rhs: BuildAccelerationStructureFlags) {
                $(
                    self.$elem = self.$elem || rhs.$elem;
                )+
            }
        }
    );
}

build_acceleration_struct_flags! {
    allow_update => vk::BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
    allow_compaction => vk::BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR,
    prefer_fast_trace => vk::BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
    prefer_fast_build => vk::BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
    low_memory => vk::BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR,
}

macro_rules! geometry_flags {
    ($($elem:ident => $val:expr,)+) => (
        #[derive(Debug, Copy, Clone)]
        #[allow(missing_docs)]
        pub struct GeometryFlags {
            $(
                pub $elem: bool,
            )+
        }

        impl GeometryFlags {
            /// Builds an `GeometryFlags` struct with all bits set.
            pub fn all() -> GeometryFlags {
                GeometryFlags {
                    $(
                        $elem: true,
                    )+
                }
            }

            /// Builds an `GeometryFlags` struct with none of the bits set.
            pub fn none() -> GeometryFlags {
                GeometryFlags {
                    $(
                        $elem: false,
                    )+
                }
            }

            #[inline]
            pub(crate) fn into_vulkan_bits(self) -> vk::GeometryFlagBitsKHR {
                let mut result = 0;
                $(
                    if self.$elem { result |= $val }
                )+
                result
            }
        }

        impl ops::BitOr for GeometryFlags {
            type Output = GeometryFlags;

            #[inline]
            fn bitor(self, rhs: GeometryFlags) -> GeometryFlags {
                GeometryFlags {
                    $(
                        $elem: self.$elem || rhs.$elem,
                    )+
                }
            }
        }

        impl ops::BitOrAssign for GeometryFlags {
            #[inline]
            fn bitor_assign(&mut self, rhs: GeometryFlags) {
                $(
                    self.$elem = self.$elem || rhs.$elem;
                )+
            }
        }
    );
}

geometry_flags! {
    opaque => vk::GEOMETRY_OPAQUE_BIT_KHR,
    no_duplicate_any_hit_invocation => vk::GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR,
}

/// An enumeration of all valid geometry types.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
#[repr(u32)]
pub enum GeometryType {
    Triangles = vk::GEOMETRY_TYPE_TRIANGLES_KHR,
    Aabbs = vk::GEOMETRY_TYPE_AABBS_KHR,
    Instances = vk::GEOMETRY_TYPE_INSTANCES_KHR,
}

#[derive(Copy, Clone)]
pub struct GeometryTrianglesData {
    pub vertex_data: u64,
    pub vertex_offset: vk::DeviceSize,
    pub vertex_count: u32,
    pub vertex_stride: usize,
    pub vertex_format: Format,
    pub index_data: u64,
    pub index_offset: vk::DeviceSize,
    pub index_count: u32,
    pub index_type: IndexType,
    pub transform_data: u64,
    pub transform_offset: vk::DeviceSize,
}

#[derive(Copy, Clone)]
pub struct GeometryAabbsData {
    pub data: u64,
    pub stride: usize,
    pub count: u32,
}

#[derive(Copy, Clone)]
pub struct GeometryInstancesData {
    pub array_of_pointers: bool,
    pub data: u64,
}

#[derive(Copy, Clone)]
pub union GeometryData {
    pub triangles: GeometryTrianglesData,
    pub aabbs: GeometryAabbsData,
    pub instances: GeometryInstancesData,
}

#[derive(Copy, Clone)]
pub struct Geometry {
    pub geometry_type: GeometryType,
    pub geometry: GeometryData,
    pub flags: GeometryFlags,
}

impl<'a> From<&'a vk::GeometryNV> for Geometry {
    fn from(geometry_nv: &'a vk::GeometryNV) -> Self {
        let geometry_type = match geometry_nv.geometryType {
            vk::GEOMETRY_TYPE_TRIANGLES_KHR => GeometryType::Triangles,
            vk::GEOMETRY_TYPE_AABBS_KHR => GeometryType::Aabbs,
            _ => unreachable!(),
        };

        let geometry = match geometry_nv.geometryType {
            vk::GEOMETRY_TYPE_TRIANGLES_KHR => GeometryData {
                triangles: GeometryTrianglesData {
                    vertex_data: geometry_nv.geometry.triangles.vertexData,
                    vertex_offset: geometry_nv.geometry.triangles.vertexOffset,
                    vertex_count: geometry_nv.geometry.triangles.vertexCount,
                    vertex_stride: geometry_nv.geometry.triangles.vertexStride as usize,
                    vertex_format: Format::from_vulkan_num(
                        geometry_nv.geometry.triangles.vertexFormat,
                    )
                    .unwrap(),
                    index_data: geometry_nv.geometry.triangles.indexData,
                    index_offset: geometry_nv.geometry.triangles.indexOffset,
                    index_count: geometry_nv.geometry.triangles.indexCount,
                    index_type: match geometry_nv.geometry.triangles.indexType {
                        vk::INDEX_TYPE_UINT16 => IndexType::U16,
                        vk::INDEX_TYPE_UINT32 => IndexType::U32,
                        _ => unreachable!(),
                    },
                    transform_data: geometry_nv.geometry.triangles.transformData,
                    transform_offset: geometry_nv.geometry.triangles.transformOffset,
                },
            },
            vk::GEOMETRY_TYPE_AABBS_KHR => GeometryData {
                aabbs: GeometryAabbsData {
                    data: geometry_nv.geometry.aabbs.aabbData,
                    stride: geometry_nv.geometry.aabbs.stride as usize,
                    count: geometry_nv.geometry.aabbs.numAABBs,
                },
            },
            _ => unreachable!(),
        };

        let flags = match geometry_nv.flags {
            vk::GEOMETRY_OPAQUE_BIT_KHR => GeometryFlags {
                opaque: true,
                no_duplicate_any_hit_invocation: false,
            },
            vk::GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR => GeometryFlags {
                opaque: false,
                no_duplicate_any_hit_invocation: true,
            },
            _ => unreachable!(),
        };

        Geometry {
            geometry_type,
            geometry,
            flags,
        }
    }
}

impl<'a> From<&'a Geometry> for vk::GeometryNV {
    fn from(geometry: &'a Geometry) -> Self {
        let geometry_type = geometry.geometry_type as vk::GeometryTypeNV;

        let geometry_nv = unsafe {
            match geometry.geometry_type {
                GeometryType::Triangles => vk::GeometryDataNV {
                    triangles: vk::GeometryTrianglesNV {
                        sType: vk::STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV,
                        pNext: ptr::null(),
                        vertexData: geometry.geometry.triangles.vertex_data,
                        vertexOffset: geometry.geometry.triangles.vertex_offset,
                        vertexCount: geometry.geometry.triangles.vertex_count,
                        vertexStride: geometry.geometry.triangles.vertex_stride as vk::DeviceSize,
                        vertexFormat: geometry.geometry.triangles.vertex_format as u32,
                        indexData: geometry.geometry.triangles.index_data,
                        indexOffset: geometry.geometry.triangles.index_offset,
                        indexCount: geometry.geometry.triangles.index_count,
                        indexType: match geometry.geometry.triangles.index_type {
                            IndexType::U16 => vk::INDEX_TYPE_UINT16,
                            IndexType::U32 => vk::INDEX_TYPE_UINT32,
                        },
                        transformData: geometry.geometry.triangles.transform_data,
                        transformOffset: geometry.geometry.triangles.transform_offset,
                    },
                    aabbs: vk::GeometryAABBNV {
                        sType: vk::STRUCTURE_TYPE_GEOMETRY_AABB_NV,
                        pNext: ptr::null(),
                        aabbData: 0,
                        numAABBs: 0,
                        stride: 0,
                        offset: 0,
                    },
                },
                GeometryType::Aabbs => vk::GeometryDataNV {
                    triangles: vk::GeometryTrianglesNV {
                        sType: vk::STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV,
                        pNext: ptr::null(),
                        vertexData: 0,
                        vertexOffset: 0,
                        vertexCount: 0,
                        vertexStride: 0,
                        vertexFormat: 0,
                        indexData: 0,
                        indexOffset: 0,
                        indexCount: 0,
                        indexType: 0,
                        transformData: 0,
                        transformOffset: 0,
                    },
                    aabbs: vk::GeometryAABBNV {
                        sType: vk::STRUCTURE_TYPE_GEOMETRY_AABB_NV,
                        pNext: ptr::null(),
                        aabbData: geometry.geometry.aabbs.data,
                        numAABBs: geometry.geometry.aabbs.count,
                        stride: geometry.geometry.aabbs.stride as u32,
                        offset: 0, // TODO:
                    },
                },
                _ => unreachable!(),
            }
        };

        let flags = if geometry.flags.opaque {
            vk::GEOMETRY_OPAQUE_BIT_KHR
        } else {
            vk::GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR
        };

        vk::GeometryNV {
            sType: vk::STRUCTURE_TYPE_GEOMETRY_NV,
            pNext: ptr::null(),
            geometryType: geometry_type,
            geometry: geometry_nv,
            flags,
        }
    }
}

unsafe impl DeviceOwned for AccelerationStructure {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl VulkanObject for AccelerationStructure {
    type Object = vk::AccelerationStructureKHR;

    const TYPE: vk::ObjectType = vk::OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR;

    #[inline]
    fn internal_object(&self) -> vk::AccelerationStructureKHR {
        self.top_level.inner_object
    }
}

impl fmt::Debug for AccelerationStructure {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(
            fmt,
            "<Vulkan acceleration structure {:?}>",
            self.top_level.inner_object
        )
    }
}

impl Drop for AccelerationStructure {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            if self.nv_extension {
                vk.DestroyAccelerationStructureNV(
                    self.device.internal_object(),
                    self.top_level.inner_object,
                    ptr::null(),
                );
                vk.DestroyAccelerationStructureNV(
                    self.device.internal_object(),
                    self.bottom_level.inner_object,
                    ptr::null(),
                );
            } else {
                vk.DestroyAccelerationStructureKHR(
                    self.device.internal_object(),
                    self.top_level.inner_object,
                    ptr::null(),
                );
                vk.DestroyAccelerationStructureKHR(
                    self.device.internal_object(),
                    self.bottom_level.inner_object,
                    ptr::null(),
                );
            }
        }
    }
}

/// Error when building a persistent descriptor set.
#[derive(Debug, Clone)]
pub enum AccelerationStructureCreationError {
    /// Out of memory.
    OomError(OomError),
}

impl error::Error for AccelerationStructureCreationError {
    #[inline]
    fn description(&self) -> &str {
        match *self {
            AccelerationStructureCreationError::OomError(_) => "not enough memory available",
        }
    }

    #[inline]
    fn cause(&self) -> Option<&dyn error::Error> {
        match *self {
            AccelerationStructureCreationError::OomError(ref err) => Some(err),
        }
    }
}

impl From<OomError> for AccelerationStructureCreationError {
    #[inline]
    fn from(err: OomError) -> AccelerationStructureCreationError {
        AccelerationStructureCreationError::OomError(err)
    }
}

impl From<Error> for AccelerationStructureCreationError {
    #[inline]
    fn from(err: Error) -> AccelerationStructureCreationError {
        match err {
            err @ Error::OutOfHostMemory => {
                AccelerationStructureCreationError::OomError(OomError::from(err))
            }
            err @ Error::OutOfDeviceMemory => {
                AccelerationStructureCreationError::OomError(OomError::from(err))
            }
            _ => panic!("unexpected error: {:?}", err),
        }
    }
}

impl fmt::Display for AccelerationStructureCreationError {
    #[inline]
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", error::Error::description(self))
    }
}
