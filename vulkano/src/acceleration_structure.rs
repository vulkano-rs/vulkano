// Copyright (c) 2023 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! An opaque data structure that is used to accelerate spatial queries on geometry data.
//!
//! Acceleration structures contain geometry data, arranged in such a way that the device can
//! easily search through the data and check for intersections between the geometry and rays
//! (lines). The geometry data can consist of either triangles, or axis-aligned bounding boxes
//! (AABBs).
//!
//! Acceleration structures come in two forms: top-level and bottom-level. A bottom-level
//! acceleration structure holds the actual geometry data, while a top-level structure contains
//! instances of (references to) one or more bottom-level structures. A top-level structure is
//! intended to contain the whole rendered scene (or the relevant parts of it), while a
//! bottom-level structure may contain individual objects within the scene. This two-level
//! arrangement allows you to easily rearrange the scene, adding and removing parts of it as needed.
//!
//! # Building an acceleration structure
//!
//! When an acceleration structure object is created, it is in an uninitialized state and contains
//! garbage data. To be able to use it for anything, you must first *build* the structure on the
//! device, using the [`build_acceleration_structure`] or [`build_acceleration_structure_indirect`]
//! command buffer commands. You use [`BuildAccelerationStructureMode::Build`] to build a structure
//! from scratch.
//!
//! When specifying geometry data for an acceleration structure build, certain triangle, AABB or
//! instance values mark that item as *inactive*. An inactive item is completely ignored within
//! the acceleration structure, and acts as if it's not there. The following special values
//! make an item inactive:
//! - For triangles, if the vertex format is a floating-point type, then the triangle is inactive
//!   if any of its vertices have NaN as their first coordinate. For integer vertex formats, it is
//!   not possible to mark triangles as inactive.
//! - For AABBs, if [`AabbPositions::min[0]`](AabbPositions::min) is NaN.
//! - For instances, if [`AccelerationStructureInstance::acceleration_structure_reference`] is 0.
//!
//! # Updating an acceleration structure
//!
//! Once an acceleration structure is built, if it was built previously with the
//! [`BuildAccelerationStructureFlags::ALLOW_UPDATE`] flag, then it is possible to update the
//! structure with new data. You use [`BuildAccelerationStructureMode::Update`] for this, which
//! specifies the source acceleration structure to use as a starting point for the update.
//! This can be the same as the destination structure (the update will happen in-place), or a
//! different one.
//!
//! An update operation is limited in which parts of the data it may change. You may change most
//! buffers and their contents, as well as strides and offsets:
//! - [`AccelerationStructureBuildGeometryInfo::scratch_data`]
//! - [`AccelerationStructureGeometryTrianglesData::vertex_data`]
//! - [`AccelerationStructureGeometryTrianglesData::vertex_stride`]
//! - [`AccelerationStructureGeometryTrianglesData::transform_data`]
//!   (but the variant of `Option` must not change)
//! - [`AccelerationStructureGeometryAabbsData::data`]
//! - [`AccelerationStructureGeometryAabbsData::stride`]
//! - [`AccelerationStructureGeometryInstancesData::data`]
//! - [`AccelerationStructureBuildRangeInfo::primitive_offset`]
//! - [`AccelerationStructureBuildRangeInfo::first_vertex`] if no index buffer is used
//! - [`AccelerationStructureBuildRangeInfo::transform_offset`]
//!
//! No other values may be changed, including in particular the variant or number of elements in
//! [`AccelerationStructureBuildGeometryInfo::geometries`], and the value of
//! [`AccelerationStructureBuildRangeInfo::primitive_count`].
//! The enum variants and data in [`AccelerationStructureGeometryTrianglesData::index_data`] must
//! not be changed, but it is allowed to specify a new index buffer, as long as it contains the
//! exact same indices as the old one.
//!
//! An update operation may not change the inactive status of an item: active items must remain
//! active in the update and inactive items must remain inactive.
//!
//! # Accessing an acceleration structure in a shader
//!
//! Acceleration structures can be bound to and accessed in any shader type. They are accessed
//! as descriptors, like buffers and images, and are declared in GLSL with
//! ```glsl
//! layout (set = N, binding = N) uniform accelerationStructureEXT nameOfTheVariable;
//! ```
//! You must enable either the `GL_EXT_ray_query` or the `GL_EXT_ray_tracing` GLSL extensions in
//! the shader to use this.
//!
//! On the Vulkano side, you can then create a descriptor set layout using
//! [`DescriptorType::AccelerationStructure`] as a descriptor type, and write the
//! acceleration structure to a descriptor set using [`WriteDescriptorSet::acceleration_structure`].
//!
//! [`build_acceleration_structure`]: crate::command_buffer::AutoCommandBufferBuilder::build_acceleration_structure
//! [`build_acceleration_structure_indirect`]: crate::command_buffer::AutoCommandBufferBuilder::build_acceleration_structure_indirect
//! [`DescriptorType::AccelerationStructure`]: crate::descriptor_set::layout::DescriptorType::AccelerationStructure
//! [`WriteDescriptorSet::acceleration_structure`]: crate::descriptor_set::WriteDescriptorSet::acceleration_structure

use crate::{
    buffer::{BufferUsage, IndexBuffer, Subbuffer},
    device::{Device, DeviceOwned},
    format::{Format, FormatFeatures},
    macros::{impl_id_counter, vulkan_bitflags, vulkan_enum},
    DeviceSize, NonZeroDeviceSize, Packed24_8, Requires, RequiresAllOf, RequiresOneOf,
    RuntimeError, ValidationError, VulkanError, VulkanObject,
};
use bytemuck::{Pod, Zeroable};
use std::{fmt::Debug, hash::Hash, mem::MaybeUninit, num::NonZeroU64, ptr, sync::Arc};

/// An opaque data structure that is used to accelerate spatial queries on geometry data.
#[derive(Debug)]
pub struct AccelerationStructure {
    device: Arc<Device>,
    handle: ash::vk::AccelerationStructureKHR,
    id: NonZeroU64,

    create_flags: AccelerationStructureCreateFlags,
    buffer: Subbuffer<[u8]>,
    ty: AccelerationStructureType,
}

impl AccelerationStructure {
    /// Creates a new `AccelerationStructure`.
    ///
    /// The [`acceleration_structure`] feature must be enabled on the device.
    ///
    /// # Safety
    ///
    /// - `create_info.buffer` (and any subbuffer it overlaps with) must not be accessed
    ///   while it is bound to the acceleration structure.
    ///
    /// [`acceleration_structure`]: crate::device::Features::acceleration_structure
    #[inline]
    pub unsafe fn new(
        device: Arc<Device>,
        create_info: AccelerationStructureCreateInfo,
    ) -> Result<Arc<Self>, VulkanError> {
        Self::validate_new(&device, &create_info)?;

        Ok(Self::new_unchecked(device, create_info)?)
    }

    fn validate_new(
        device: &Device,
        create_info: &AccelerationStructureCreateInfo,
    ) -> Result<(), ValidationError> {
        if !device.enabled_extensions().khr_acceleration_structure {
            return Err(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceExtension(
                    "khr_acceleration_structure",
                )])]),
                ..Default::default()
            });
        }

        if !device.enabled_features().acceleration_structure {
            return Err(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::Feature(
                    "acceleration_structure",
                )])]),
                vuids: &["VUID-vkCreateAccelerationStructureKHR-accelerationStructure-03611"],
                ..Default::default()
            });
        }

        // VUID-vkCreateAccelerationStructureKHR-pCreateInfo-parameter
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        create_info: AccelerationStructureCreateInfo,
    ) -> Result<Arc<Self>, RuntimeError> {
        let &AccelerationStructureCreateInfo {
            create_flags,
            ref buffer,
            ty,
            _ne: _,
        } = &create_info;

        let create_info_vk = ash::vk::AccelerationStructureCreateInfoKHR {
            create_flags: create_flags.into(),
            buffer: buffer.buffer().handle(),
            offset: buffer.offset(),
            size: buffer.size(),
            ty: ty.into(),
            device_address: 0, // TODO: allow user to specify
            ..Default::default()
        };

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.khr_acceleration_structure
                .create_acceleration_structure_khr)(
                device.handle(),
                &create_info_vk,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(RuntimeError::from)?;
            output.assume_init()
        };

        Ok(Self::from_handle(device, handle, create_info))
    }

    /// Creates a new `AccelerationStructure` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::AccelerationStructureKHR,
        create_info: AccelerationStructureCreateInfo,
    ) -> Arc<Self> {
        let AccelerationStructureCreateInfo {
            create_flags,
            buffer,
            ty,
            _ne: _,
        } = create_info;

        Arc::new(Self {
            device,
            handle,
            id: Self::next_id(),

            create_flags,
            buffer,
            ty,
        })
    }

    /// Returns the flags the acceleration structure was created with.
    #[inline]
    pub fn create_flags(&self) -> AccelerationStructureCreateFlags {
        self.create_flags
    }

    /// Returns the subbuffer that the acceleration structure is stored on.
    #[inline]
    pub fn buffer(&self) -> &Subbuffer<[u8]> {
        &self.buffer
    }

    /// Returns the size of the acceleration structure.
    #[inline]
    pub fn size(&self) -> DeviceSize {
        self.buffer.size()
    }

    /// Returns the type of the acceleration structure.
    #[inline]
    pub fn ty(&self) -> AccelerationStructureType {
        self.ty
    }

    /// Returns the device address of the acceleration structure.
    ///
    /// The device address of the acceleration structure may be different from the device address
    /// of the underlying buffer.
    pub fn device_address(&self) -> NonZeroDeviceSize {
        let info_vk = ash::vk::AccelerationStructureDeviceAddressInfoKHR {
            acceleration_structure: self.handle,
            ..Default::default()
        };
        let ptr = unsafe {
            let fns = self.device.fns();
            (fns.khr_acceleration_structure
                .get_acceleration_structure_device_address_khr)(
                self.device.handle(), &info_vk
            )
        };

        NonZeroDeviceSize::new(ptr).unwrap()
    }
}

impl Drop for AccelerationStructure {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            (fns.khr_acceleration_structure
                .destroy_acceleration_structure_khr)(
                self.device.handle(), self.handle, ptr::null()
            )
        }
    }
}

unsafe impl VulkanObject for AccelerationStructure {
    type Handle = ash::vk::AccelerationStructureKHR;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for AccelerationStructure {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(AccelerationStructure);

vulkan_enum! {
    #[non_exhaustive]

    /// The type of an acceleration structure.
    AccelerationStructureType = AccelerationStructureTypeKHR(i32);

    /// Refers to bottom-level acceleration structures. This type can be bound to a descriptor.
    TopLevel = TOP_LEVEL,

    /// Contains AABBs or geometry to be intersected.
    BottomLevel = BOTTOM_LEVEL,

    /// The type is determined at build time.
    ///
    /// Use of this type is discouraged, it is preferred to specify the type at create time.
    Generic = GENERIC,
}

/// Parameters to create a new `AccelerationStructure`.
#[derive(Clone, Debug)]
pub struct AccelerationStructureCreateInfo {
    /// Specifies how to create the acceleration structure.
    ///
    /// The default value is empty.
    pub create_flags: AccelerationStructureCreateFlags,

    /// The subbuffer to store the acceleration structure on.
    ///
    /// The subbuffer must have an `offset` that is a multiple of 256, and its `usage` must include
    /// [`BufferUsage::ACCELERATION_STRUCTURE_STORAGE`]. It must not be accessed while it is bound
    /// to the acceleration structure.
    ///
    /// There is no default value.
    pub buffer: Subbuffer<[u8]>,

    /// The type of acceleration structure to create.
    ///
    /// The default value is [`AccelerationStructureType::Generic`].
    pub ty: AccelerationStructureType,

    /* TODO: enable
    // TODO: document
    pub device_address: DeviceAddress, */
    pub _ne: crate::NonExhaustive,
}

impl AccelerationStructureCreateInfo {
    /// Returns a `AccelerationStructureCreateInfo` with the specified `buffer`.
    #[inline]
    pub fn new(buffer: Subbuffer<[u8]>) -> Self {
        Self {
            create_flags: AccelerationStructureCreateFlags::empty(),
            buffer,
            ty: AccelerationStructureType::Generic,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            create_flags,
            ref buffer,
            ty,
            _ne: _,
        } = self;

        create_flags
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "create_flags".into(),
                vuids: &["VUID-VkAccelerationStructureCreateInfoKHR-createFlags-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        ty.validate_device(device).map_err(|err| ValidationError {
            context: "ty".into(),
            vuids: &["VUID-VkAccelerationStructureCreateInfoKHR-type-parameter"],
            ..ValidationError::from_requirement(err)
        })?;

        if !buffer
            .buffer()
            .usage()
            .intersects(BufferUsage::ACCELERATION_STRUCTURE_STORAGE)
        {
            return Err(ValidationError {
                context: "buffer".into(),
                problem: "the buffer was not created with the `ACCELERATION_STRUCTURE_STORAGE` \
                    usage"
                    .into(),
                vuids: &["VUID-VkAccelerationStructureCreateInfoKHR-buffer-03614"],
                ..Default::default()
            });
        }

        // VUID-VkAccelerationStructureCreateInfoKHR-offset-03616
        // Ensured by the definition of `Subbuffer`.

        if buffer.offset() % 256 != 0 {
            return Err(ValidationError {
                context: "buffer".into(),
                problem: "the offset of the buffer is not a multiple of 256".into(),
                vuids: &["VUID-VkAccelerationStructureCreateInfoKHR-offset-03734"],
                ..Default::default()
            });
        }

        Ok(())
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags that control how an acceleration structure is created.
    AccelerationStructureCreateFlags = AccelerationStructureCreateFlagsKHR(u32);

    /* TODO: enable
    // TODO: document
    DEVICE_ADDRESS_CAPTURE_REPLAY = DEVICE_ADDRESS_CAPTURE_REPLAY_KHR, */

    /* TODO: enable
    // TODO: document
    DESCRIPTOR_BUFFER_CAPTURE_REPLAY = DESCRIPTOR_BUFFER_CAPTURE_REPLAY_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_descriptor_buffer)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    MOTION = MOTION_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_ray_tracing_motion_blur)]),
    ]),*/
}

/// Geometries and other parameters for an acceleration structure build operation.
#[derive(Clone, Debug)]
pub struct AccelerationStructureBuildGeometryInfo {
    /// Specifies how to build the acceleration structure.
    ///
    /// The default value is empty.
    pub flags: BuildAccelerationStructureFlags,

    /// The mode that the build command should operate in.
    ///
    /// The default value is [`BuildAccelerationStructureMode::Build`].
    pub mode: BuildAccelerationStructureMode,

    /// The acceleration structure to build or update.
    ///
    /// There is no default value.
    pub dst_acceleration_structure: Arc<AccelerationStructure>,

    /// The geometries that will be built into `dst_acceleration_structure`.
    ///
    /// The geometry type must match the `ty` that was specified when the acceleration structure
    /// was created:
    /// - `Instances` must be used with `TopLevel` or `Generic`.
    /// - `Triangles` and `Aabbs` must be used with `BottomLevel` or `Generic`.
    ///
    /// There is no default value.
    pub geometries: AccelerationStructureGeometries,

    /// Scratch memory to be used for the build.
    ///
    /// There is no default value.
    pub scratch_data: Subbuffer<[u8]>,

    pub _ne: crate::NonExhaustive,
}

impl AccelerationStructureBuildGeometryInfo {
    /// Returns a `AccelerationStructureBuildGeometryInfo` with the specified
    /// `dst_acceleration_structure`, `geometries` and `scratch_data`.
    #[inline]
    pub fn new(
        dst_acceleration_structure: Arc<AccelerationStructure>,
        geometries: AccelerationStructureGeometries,
        scratch_data: Subbuffer<[u8]>,
    ) -> Self {
        Self {
            flags: BuildAccelerationStructureFlags::empty(),
            mode: BuildAccelerationStructureMode::Build,
            dst_acceleration_structure,
            geometries,
            scratch_data,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            flags,
            ref mode,
            ref dst_acceleration_structure,
            ref geometries,
            scratch_data: _,
            _ne: _,
        } = self;

        flags
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "flags".into(),
                vuids: &["VUID-VkAccelerationStructureBuildGeometryInfoKHR-flags-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        let max_geometry_count = device
            .physical_device()
            .properties()
            .max_geometry_count
            .unwrap();

        match geometries {
            // VUID-VkAccelerationStructureGeometryKHR-triangles-parameter
            AccelerationStructureGeometries::Triangles(geometries) => {
                for (index, triangles_data) in geometries.iter().enumerate() {
                    triangles_data
                        .validate(device)
                        .map_err(|err| err.add_context(format!("geometries[{}]", index)))?;
                }

                if geometries.len() as u64 > max_geometry_count {
                    return Err(ValidationError {
                        context: "geometries".into(),
                        problem: "the length exceeds the `max_geometry_count` limit".into(),
                        vuids: &["VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03793"],
                        ..Default::default()
                    });
                }

                // VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03795
                // Is checked in the top-level functions.
            }

            // VUID-VkAccelerationStructureGeometryKHR-aabbs-parameter
            AccelerationStructureGeometries::Aabbs(geometries) => {
                for (index, aabbs_data) in geometries.iter().enumerate() {
                    aabbs_data
                        .validate(device)
                        .map_err(|err| err.add_context(format!("geometries[{}]", index)))?;
                }

                if geometries.len() as u64 > max_geometry_count {
                    return Err(ValidationError {
                        context: "geometries".into(),
                        problem: "the length exceeds the `max_geometry_count` limit".into(),
                        vuids: &["VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03793"],
                        ..Default::default()
                    });
                }

                // VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03794
                // Is checked in the top-level functions.
            }

            // VUID-VkAccelerationStructureGeometryKHR-instances-parameter
            AccelerationStructureGeometries::Instances(instances_data) => {
                instances_data
                    .validate(device)
                    .map_err(|err| err.add_context("geometries"))?;
            }
        }

        // VUID-VkAccelerationStructureBuildGeometryInfoKHR-commonparent
        assert_eq!(device, dst_acceleration_structure.device().as_ref());

        if let BuildAccelerationStructureMode::Update(src_acceleration_structure) = mode {
            assert_eq!(device, src_acceleration_structure.device().as_ref());
        }

        if flags.contains(
            BuildAccelerationStructureFlags::PREFER_FAST_TRACE
                | BuildAccelerationStructureFlags::PREFER_FAST_BUILD,
        ) {
            return Err(ValidationError {
                context: "flags".into(),
                problem: "contains both `BuildAccelerationStructureFlags::PREFER_FAST_TRACE` and \
                    `BuildAccelerationStructureFlags::PREFER_FAST_BUILD`"
                    .into(),
                vuids: &["VUID-VkAccelerationStructureBuildGeometryInfoKHR-flags-03796"],
                ..Default::default()
            });
        }

        Ok(())
    }

    pub(crate) fn to_vulkan(
        &self,
    ) -> (
        ash::vk::AccelerationStructureBuildGeometryInfoKHR,
        Vec<ash::vk::AccelerationStructureGeometryKHR>,
    ) {
        let &Self {
            flags,
            ref mode,
            ref dst_acceleration_structure,
            ref geometries,
            ref scratch_data,
            _ne: _,
        } = self;

        let (ty, geometries_vk): (_, Vec<_>) = match geometries {
            AccelerationStructureGeometries::Triangles(geometries) => (
                ash::vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                geometries
                    .iter()
                    .map(|triangles_data| {
                        let &AccelerationStructureGeometryTrianglesData {
                            flags,
                            vertex_format,
                            ref vertex_data,
                            vertex_stride,
                            max_vertex,
                            ref index_data,
                            ref transform_data,
                            _ne,
                        } = triangles_data;

                        ash::vk::AccelerationStructureGeometryKHR {
                            geometry_type: ash::vk::GeometryTypeKHR::TRIANGLES,
                            geometry: ash::vk::AccelerationStructureGeometryDataKHR {
                                triangles: ash::vk::AccelerationStructureGeometryTrianglesDataKHR {
                                    vertex_format: vertex_format.into(),
                                    vertex_data: ash::vk::DeviceOrHostAddressConstKHR {
                                        device_address: vertex_data
                                            .device_address()
                                            .unwrap()
                                            .into(),
                                    },
                                    vertex_stride: vertex_stride as DeviceSize,
                                    max_vertex,
                                    index_type: index_data
                                        .as_ref()
                                        .map_or(ash::vk::IndexType::NONE_KHR, |index_data| {
                                            index_data.index_type().into()
                                        }),
                                    index_data: ash::vk::DeviceOrHostAddressConstKHR {
                                        device_address: index_data.as_ref().map_or(
                                            0,
                                            |index_data| {
                                                index_data
                                                    .as_bytes()
                                                    .device_address()
                                                    .unwrap()
                                                    .get()
                                            },
                                        ),
                                    },
                                    transform_data: ash::vk::DeviceOrHostAddressConstKHR {
                                        device_address: transform_data.as_ref().map_or(
                                            0,
                                            |transform_data| {
                                                transform_data.device_address().unwrap().get()
                                            },
                                        ),
                                    },
                                    ..Default::default()
                                },
                            },
                            flags: flags.into(),
                            ..Default::default()
                        }
                    })
                    .collect(),
            ),
            AccelerationStructureGeometries::Aabbs(geometries) => (
                ash::vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                geometries
                    .iter()
                    .map(|aabbs_data| {
                        let &AccelerationStructureGeometryAabbsData {
                            flags,
                            ref data,
                            stride,
                            _ne: _,
                        } = aabbs_data;

                        ash::vk::AccelerationStructureGeometryKHR {
                            geometry_type: ash::vk::GeometryTypeKHR::AABBS,
                            geometry: ash::vk::AccelerationStructureGeometryDataKHR {
                                aabbs: ash::vk::AccelerationStructureGeometryAabbsDataKHR {
                                    data: ash::vk::DeviceOrHostAddressConstKHR {
                                        device_address: data.device_address().unwrap().get(),
                                    },
                                    stride: stride as DeviceSize,
                                    ..Default::default()
                                },
                            },
                            flags: flags.into(),
                            ..Default::default()
                        }
                    })
                    .collect(),
            ),
            AccelerationStructureGeometries::Instances(instances_data) => {
                (ash::vk::AccelerationStructureTypeKHR::TOP_LEVEL, {
                    let &AccelerationStructureGeometryInstancesData {
                        flags,
                        ref data,
                        _ne: _,
                    } = instances_data;

                    let (array_of_pointers, data) = match data {
                        AccelerationStructureGeometryInstancesDataType::Values(data) => (
                            ash::vk::FALSE,
                            ash::vk::DeviceOrHostAddressConstKHR {
                                device_address: data.device_address().unwrap().get(),
                            },
                        ),
                        AccelerationStructureGeometryInstancesDataType::Pointers(data) => (
                            ash::vk::TRUE,
                            ash::vk::DeviceOrHostAddressConstKHR {
                                device_address: data.device_address().unwrap().get(),
                            },
                        ),
                    };

                    [ash::vk::AccelerationStructureGeometryKHR {
                        geometry_type: ash::vk::GeometryTypeKHR::INSTANCES,
                        geometry: ash::vk::AccelerationStructureGeometryDataKHR {
                            instances: ash::vk::AccelerationStructureGeometryInstancesDataKHR {
                                array_of_pointers,
                                data,
                                ..Default::default()
                            },
                        },
                        flags: flags.into(),
                        ..Default::default()
                    }]
                    .into_iter()
                    .collect()
                })
            }
        };

        (
            ash::vk::AccelerationStructureBuildGeometryInfoKHR {
                ty,
                flags: flags.into(),
                mode: mode.into(),
                src_acceleration_structure: match mode {
                    BuildAccelerationStructureMode::Build => Default::default(),
                    BuildAccelerationStructureMode::Update(src_acceleration_structure) => {
                        src_acceleration_structure.handle()
                    }
                },
                dst_acceleration_structure: dst_acceleration_structure.handle(),
                geometry_count: 0,
                p_geometries: ptr::null(),
                pp_geometries: ptr::null(),
                scratch_data: ash::vk::DeviceOrHostAddressKHR {
                    device_address: scratch_data.device_address().unwrap().get(),
                },
                ..Default::default()
            },
            geometries_vk,
        )
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags to control how an acceleration structure should be built.
    BuildAccelerationStructureFlags = BuildAccelerationStructureFlagsKHR(u32);

    /// The built acceleration structure can be updated later by building it again with
    /// [`BuildAccelerationStructureMode::Update`].
    ///
    /// The building process may take more time and memory than normal.
    ALLOW_UPDATE = ALLOW_UPDATE,

    /// The built acceleration structure can be used later as the source in a copy operation with
    /// [`CopyAccelerationStructureMode::Compact`].
    ///
    /// The building process may take more time and memory than normal.
    ALLOW_COMPACTION = ALLOW_COMPACTION,

    /// Prioritize for best trace performance, with possibly longer build times.
    PREFER_FAST_TRACE = PREFER_FAST_TRACE,

    /// Prioritize for shorter build time, with possibly suboptimal trace performance.
    PREFER_FAST_BUILD = PREFER_FAST_BUILD,

    /// Prioritize low acceleration structure and scratch memory size, with possibly longer build
    /// times or suboptimal trace performance.
    LOW_MEMORY = LOW_MEMORY,

    /* TODO: enable
    // TODO: document
    MOTION = MOTION_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_ray_tracing_motion_blur)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ALLOW_OPACITY_MICROMAP_UPDATE = ALLOW_OPACITY_MICROMAP_UPDATE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_opacity_micromap)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ALLOW_DISABLE_OPACITY_MICROMAPS = ALLOW_DISABLE_OPACITY_MICROMAPS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_opacity_micromap)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ALLOW_OPACITY_MICROMAP_DATA_UPDATE = ALLOW_OPACITY_MICROMAP_DATA_UPDATE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_opacity_micromap)]),
    ]), */

    /* TODO: enable
    // TODO: document
    ALLOW_DISPLACEMENT_MICROMAP_UPDATE = ALLOW_DISPLACEMENT_MICROMAP_UPDATE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_displacement_micromap)]),
    ]), */
}

/// What mode an acceleration structure build command should operate in.
#[derive(Clone, Debug)]
#[repr(i32)]
pub enum BuildAccelerationStructureMode {
    /// Build a new acceleration structure from scratch.
    Build = ash::vk::BuildAccelerationStructureModeKHR::BUILD.as_raw(),

    /// Update a previously built source acceleration structure with new data, storing the
    /// updated structure in the destination. The source and destination acceleration structures
    /// may be the same, which will do the update in-place.
    ///
    /// The destination acceleration structure must have been built with the
    /// [`BuildAccelerationStructureFlags::ALLOW_UPDATE`] flag.
    Update(Arc<AccelerationStructure>) =
        ash::vk::BuildAccelerationStructureModeKHR::UPDATE.as_raw(),
}

impl From<&BuildAccelerationStructureMode> for ash::vk::BuildAccelerationStructureModeKHR {
    #[inline]
    fn from(val: &BuildAccelerationStructureMode) -> Self {
        match val {
            BuildAccelerationStructureMode::Build => {
                ash::vk::BuildAccelerationStructureModeKHR::BUILD
            }
            BuildAccelerationStructureMode::Update(_) => {
                ash::vk::BuildAccelerationStructureModeKHR::UPDATE
            }
        }
    }
}

/// The type of geometry data in an acceleration structure.
#[derive(Clone, Debug)]
pub enum AccelerationStructureGeometries {
    /// The geometries consist of bottom-level triangles data.
    Triangles(Vec<AccelerationStructureGeometryTrianglesData>),

    /// The geometries consist of bottom-level axis-aligned bounding box data.
    Aabbs(Vec<AccelerationStructureGeometryAabbsData>),

    /// The geometries consist of top-level instance data.
    Instances(AccelerationStructureGeometryInstancesData),
}

impl AccelerationStructureGeometries {
    /// Returns the number of geometries.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            AccelerationStructureGeometries::Triangles(geometries) => geometries.len(),
            AccelerationStructureGeometries::Aabbs(geometries) => geometries.len(),
            AccelerationStructureGeometries::Instances(_) => 1,
        }
    }
}

impl From<Vec<AccelerationStructureGeometryTrianglesData>> for AccelerationStructureGeometries {
    #[inline]
    fn from(value: Vec<AccelerationStructureGeometryTrianglesData>) -> Self {
        Self::Triangles(value)
    }
}

impl From<Vec<AccelerationStructureGeometryAabbsData>> for AccelerationStructureGeometries {
    #[inline]
    fn from(value: Vec<AccelerationStructureGeometryAabbsData>) -> Self {
        Self::Aabbs(value)
    }
}

impl From<AccelerationStructureGeometryInstancesData> for AccelerationStructureGeometries {
    #[inline]
    fn from(value: AccelerationStructureGeometryInstancesData) -> Self {
        Self::Instances(value)
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags to control how an acceleration structure geometry should be built.
    GeometryFlags = GeometryFlagsKHR(u32);

    /// The geometry does not invoke the any-hit shaders, even if it is present in a hit group.
    OPAQUE = OPAQUE,

    /// The any-hit shader will never be called more than once for each primitive in the geometry.
    NO_DUPLICATE_ANY_HIT_INVOCATION = NO_DUPLICATE_ANY_HIT_INVOCATION,
}

/// A bottom-level geometry consisting of triangles.
#[derive(Clone, Debug)]
pub struct AccelerationStructureGeometryTrianglesData {
    /// Specifies how the geometry should be built.
    ///
    /// The default value is empty.
    pub flags: GeometryFlags,

    /// The format of each vertex in `vertex_data`.
    ///
    /// This works in the same way as formats for vertex buffers.
    ///
    /// There is no default value.
    pub vertex_format: Format,

    /// The vertex data itself, consisting of an array of `vertex_format` values.
    ///
    /// There is no default value.
    pub vertex_data: Subbuffer<[u8]>,

    /// The number of bytes between the start of successive elements in `vertex_data`.
    ///
    /// This must be a multiple of the smallest component size (in bytes) of `vertex_format`.
    ///
    /// The default value is 0, which must be overridden.
    pub vertex_stride: u32,

    /// The highest vertex index that may be read from `vertex_data`.
    ///
    /// The default value is 0, which must be overridden.
    pub max_vertex: u32,

    /// If indices are to be used, the buffer holding the index data.
    ///
    /// The indices will be used to index into the elements of `vertex_data`.
    ///
    /// The default value is `None`.
    pub index_data: Option<IndexBuffer>,

    /// Optionally, a 3x4 matrix that will be used to transform the vertices in
    /// `vertex_data` to the space in which the acceleration structure is defined.
    ///
    /// The first three columns must be a 3x3 invertible matrix.
    ///
    /// The default value is `None`.
    pub transform_data: Option<Subbuffer<TransformMatrix>>,

    pub _ne: crate::NonExhaustive,
}

impl AccelerationStructureGeometryTrianglesData {
    /// Returns a `AccelerationStructureGeometryTrianglesData` with the specified
    /// `vertex_format` and `vertex_data`.
    #[inline]
    pub fn new(vertex_format: Format, vertex_data: Subbuffer<[u8]>) -> Self {
        Self {
            flags: GeometryFlags::empty(),
            vertex_format,
            vertex_data,
            vertex_stride: 0,
            max_vertex: 0,
            index_data: None,
            transform_data: None,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            flags,
            vertex_format,
            vertex_data: _,
            vertex_stride,
            max_vertex: _,
            ref index_data,
            transform_data: _,
            _ne: _,
        } = self;

        flags
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "flags".into(),
                vuids: &["VUID-VkAccelerationStructureGeometryKHR-flags-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        vertex_format
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "vertex_format".into(),
                vuids: &[
                    "VUID-VkAccelerationStructureGeometryTrianglesDataKHR-vertexFormat-parameter",
                ],
                ..ValidationError::from_requirement(err)
            })?;

        if unsafe {
            !device
                .physical_device()
                .format_properties_unchecked(vertex_format)
                .buffer_features
                .intersects(FormatFeatures::ACCELERATION_STRUCTURE_VERTEX_BUFFER)
        } {
            return Err(ValidationError {
                context: "vertex_format".into(),
                problem: "format features do not contain \
                    `FormatFeature::ACCELERATION_STRUCTURE_VERTEX_BUFFER`"
                    .into(),
                vuids: &["VUID-VkAccelerationStructureGeometryTrianglesDataKHR-vertexFormat-03797"],
                ..Default::default()
            });
        }

        let smallest_component_bits = vertex_format
            .components()
            .into_iter()
            .filter(|&c| c != 0)
            .min()
            .unwrap() as u32;
        let smallest_component_bytes = (smallest_component_bits + 7) & !7;

        if vertex_stride % smallest_component_bytes != 0 {
            return Err(ValidationError {
                problem: "`vertex_stride` is not a multiple of the byte size of the \
                    smallest component of `vertex_format`"
                    .into(),
                vuids: &["VUID-VkAccelerationStructureGeometryTrianglesDataKHR-vertexStride-03735"],
                ..Default::default()
            });
        }

        if let Some(index_data) = index_data.as_ref() {
            if !matches!(index_data, IndexBuffer::U16(_) | IndexBuffer::U32(_)) {
                return Err(ValidationError {
                    context: "index_data".into(),
                    problem: "is not `IndexBuffer::U16` or `IndexBuffer::U32`".into(),
                    vuids: &[
                        "VUID-VkAccelerationStructureGeometryTrianglesDataKHR-indexType-03798",
                    ],
                    ..Default::default()
                });
            }
        }

        Ok(())
    }
}

/// A 3x4 transformation matrix.
///
/// The first three columns must be a 3x3 invertible matrix.
pub type TransformMatrix = [[f32; 4]; 3];

/// A bottom-level geometry consisting of axis-aligned bounding boxes.
#[derive(Clone, Debug)]
pub struct AccelerationStructureGeometryAabbsData {
    /// Specifies how the geometry should be built.
    ///
    /// The default value is empty.
    pub flags: GeometryFlags,

    /// The AABB data itself, consisting of an array of [`AabbPositions`] structs.
    ///
    /// There is no default value.
    pub data: Subbuffer<[u8]>,

    /// The number of bytes between the start of successive elements in `data`.
    ///
    /// This must be a multiple of 8.
    ///
    /// The default value is 0, which must be overridden.
    pub stride: u32,

    pub _ne: crate::NonExhaustive,
}

impl AccelerationStructureGeometryAabbsData {
    /// Returns a `AccelerationStructureGeometryAabbsData` with the specified `data`.
    #[inline]
    pub fn new(data: Subbuffer<[u8]>) -> Self {
        Self {
            flags: GeometryFlags::empty(),
            data,
            stride: 0,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            flags,
            data: _,
            stride,
            _ne: _,
        } = self;

        flags
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "flags".into(),
                vuids: &["VUID-VkAccelerationStructureGeometryKHR-flags-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        if stride % 8 != 0 {
            return Err(ValidationError {
                context: "stride".into(),
                problem: "is not a multiple of 8".into(),
                vuids: &["VUID-VkAccelerationStructureGeometryAabbsDataKHR-stride-03545"],
                ..Default::default()
            });
        }

        Ok(())
    }
}

/// Specifies two opposing corners of an axis-aligned bounding box.
///
/// Each value in `min` must be less than or equal to the corresponding value in `max`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Zeroable, Pod)]
#[repr(C)]
pub struct AabbPositions {
    /// The minimum of the corner coordinates of the bounding box.
    ///
    /// The default value is `[0.0; 3]`.
    pub min: [f32; 3],

    /// The maximum of the corner coordinates of the bounding box.
    ///
    /// The default value is `[0.0; 3]`.
    pub max: [f32; 3],
}

/// A top-level geometry consisting of instances of bottom-level acceleration structures.
#[derive(Clone, Debug)]
pub struct AccelerationStructureGeometryInstancesData {
    /// Specifies how the geometry should be built.
    ///
    /// The default value is empty.
    pub flags: GeometryFlags,

    /// The instance data itself.
    ///
    /// There is no default value.
    pub data: AccelerationStructureGeometryInstancesDataType,

    pub _ne: crate::NonExhaustive,
}

impl AccelerationStructureGeometryInstancesData {
    /// Returns a `AccelerationStructureGeometryInstancesData` with the specified `data`.
    #[inline]
    pub fn new(data: AccelerationStructureGeometryInstancesDataType) -> Self {
        Self {
            flags: GeometryFlags::empty(),
            data,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            flags,
            data: _,
            _ne: _,
        } = self;

        flags
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "flags".into(),
                vuids: &["VUID-VkAccelerationStructureGeometryKHR-flags-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        Ok(())
    }
}

/// The data type of an instances geometry.
#[derive(Clone, Debug)]
pub enum AccelerationStructureGeometryInstancesDataType {
    /// The data buffer contains an array of [`AccelerationStructureInstance`] structures directly.
    Values(Subbuffer<[AccelerationStructureInstance]>),

    /// The data buffer contains an array of pointers to [`AccelerationStructureInstance`]
    /// structures.
    Pointers(Subbuffer<[DeviceSize]>),
}

impl From<Subbuffer<[AccelerationStructureInstance]>>
    for AccelerationStructureGeometryInstancesDataType
{
    #[inline]
    fn from(value: Subbuffer<[AccelerationStructureInstance]>) -> Self {
        Self::Values(value)
    }
}

impl From<Subbuffer<[DeviceSize]>> for AccelerationStructureGeometryInstancesDataType {
    #[inline]
    fn from(value: Subbuffer<[DeviceSize]>) -> Self {
        Self::Pointers(value)
    }
}

/// Specifies a bottom-level acceleration structure instance when
/// building a top-level structure.
#[derive(Clone, Copy, Debug, PartialEq, Zeroable, Pod)]
#[repr(C)]
pub struct AccelerationStructureInstance {
    /// A 3x4 transformation matrix to be applied to the bottom-level acceleration structure.
    ///
    /// The first three columns must be a 3x3 invertible matrix.
    ///
    /// The default value is a 3x3 identity matrix, with the fourth column filled with zeroes.
    pub transform: TransformMatrix,

    /// Low 24 bits: A custom index value to be accessible via the `InstanceCustomIndexKHR`
    /// built-in variable in ray shaders. The default value is 0.
    ///
    /// High 8 bits: A visibility mask for the geometry. The instance will not be hit if the
    /// cull mask ANDed with this mask is zero. The default value is 0xFF.
    pub instance_custom_index_and_mask: Packed24_8,

    /// Low 24 bits: An offset used in calculating the binding table index of the hit shader.
    /// The default value is 0.
    ///
    /// High 8 bits: [`GeometryInstanceFlags`] to apply to the instance. The `From` trait can be
    /// used to convert the flags into a `u8` value. The default value is empty.
    pub instance_shader_binding_table_record_offset_and_flags: Packed24_8,

    /// The device address of the bottom-level acceleration structure in this instance.
    ///
    /// The default value is 0 (null).
    pub acceleration_structure_reference: DeviceSize,
}

impl Default for AccelerationStructureInstance {
    #[inline]
    fn default() -> Self {
        Self {
            transform: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            instance_custom_index_and_mask: Packed24_8::new(0, 0xff),
            instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(0, 0),
            acceleration_structure_reference: 0,
        }
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags for an instance in a top-level acceleration structure.
    GeometryInstanceFlags = GeometryInstanceFlagsKHR(u32);

    /// Disable face culling for the instance.
    TRIANGLE_FACING_CULL_DISABLE = TRIANGLE_FACING_CULL_DISABLE,

    /// Flip the facing (front vs back) of triangles.
    TRIANGLE_FLIP_FACING = TRIANGLE_FLIP_FACING,

    /// Geometries in this instance will act as if [`GeometryFlags::OPAQUE`] were specified.
    FORCE_OPAQUE = FORCE_OPAQUE,

    /// Geometries in this instance will act as if [`GeometryFlags::OPAQUE`] were not specified.
    FORCE_NO_OPAQUE = FORCE_NO_OPAQUE,

    /* TODO: enable
    // TODO: document
    FORCE_OPACITY_MICROMAP_2_STATE = FORCE_OPACITY_MICROMAP_2_STATE_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_opacity_micromap)]),
    ]), */

    /* TODO: enable
    // TODO: document
    DISABLE_OPACITY_MICROMAPS = DISABLE_OPACITY_MICROMAPS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_opacity_micromap)]),
    ]), */
}

impl From<GeometryInstanceFlags> for u8 {
    #[inline]
    fn from(value: GeometryInstanceFlags) -> Self {
        value.0 as u8
    }
}

/// Counts and offsets for an acceleration structure build operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Zeroable, Pod)]
#[repr(C)]
pub struct AccelerationStructureBuildRangeInfo {
    /// The number of primitives.
    ///
    /// The default value is 0.
    pub primitive_count: u32,

    /// The offset (in bytes) into the buffer holding geometry data,
    /// to where the first primitive is stored.
    ///
    /// The default value is 0.
    pub primitive_offset: u32,

    /// The index of the first vertex to build from.
    ///
    /// This is used only for triangle geometries.
    ///
    /// The default value is 0.
    pub first_vertex: u32,

    /// The offset (in bytes) into the buffer holding transform matrices,
    /// to where the matrix is stored.
    ///
    /// This is used only for triangle geometries.
    ///
    /// The default value is 0.
    pub transform_offset: u32,
}

/// Parameters for copying an acceleration structure.
#[derive(Clone, Debug)]
pub struct CopyAccelerationStructureInfo {
    /// The acceleration structure to copy from.
    ///
    /// There is no default value.
    pub src: Arc<AccelerationStructure>,

    /// The acceleration structure to copy into.
    ///
    /// There is no default value.
    pub dst: Arc<AccelerationStructure>,

    /// Additional operations to perform during the copy.
    ///
    /// The default value is [`CopyAccelerationStructureMode::Clone`].
    pub mode: CopyAccelerationStructureMode,

    pub _ne: crate::NonExhaustive,
}

impl CopyAccelerationStructureInfo {
    /// Returns a `CopyAccelerationStructureInfo` with the specified `src` and `dst`.
    #[inline]
    pub fn new(src: Arc<AccelerationStructure>, dst: Arc<AccelerationStructure>) -> Self {
        Self {
            src,
            dst,
            mode: CopyAccelerationStructureMode::Clone,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            ref src,
            ref dst,
            mode,
            _ne: _,
        } = self;

        // VUID-VkCopyAccelerationStructureInfoKHR-commonparent
        assert_eq!(device, src.device().as_ref());
        assert_eq!(device, dst.device().as_ref());

        mode.validate_device(device)
            .map_err(|err| ValidationError {
                context: "mode".into(),
                vuids: &["VUID-VkCopyAccelerationStructureInfoKHR-mode-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        if !matches!(
            mode,
            CopyAccelerationStructureMode::Compact | CopyAccelerationStructureMode::Clone
        ) {
            return Err(ValidationError {
                context: "mode".into(),
                problem: "is not `CopyAccelerationStructureMode::Compact` or \
                    `CopyAccelerationStructureMode::Clone`"
                    .into(),
                vuids: &["VUID-VkCopyAccelerationStructureInfoKHR-mode-03410"],
                ..Default::default()
            });
        }

        if src.buffer() == dst.buffer() {
            return Err(ValidationError {
                problem: "`src` and `dst` share the same buffer".into(),
                vuids: &["VUID-VkCopyAccelerationStructureInfoKHR-dst-07791"],
                ..Default::default()
            });
        }

        // VUID-VkCopyAccelerationStructureInfoKHR-src-04963
        // TODO: unsafe

        // VUID-VkCopyAccelerationStructureInfoKHR-src-03411
        // TODO: unsafe

        Ok(())
    }
}

/// Parameters for copying from an acceleration structure into memory.
#[derive(Clone, Debug)]
pub struct CopyAccelerationStructureToMemoryInfo {
    /// The acceleration structure to copy from.
    ///
    /// There is no default value.
    pub src: Arc<AccelerationStructure>,

    /// The memory to copy the structure to.
    ///
    /// There is no default value.
    pub dst: Subbuffer<[u8]>,

    /// Additional operations to perform during the copy.
    ///
    /// The default value is [`CopyAccelerationStructureMode::Serialize`].
    pub mode: CopyAccelerationStructureMode,

    pub _ne: crate::NonExhaustive,
}

impl CopyAccelerationStructureToMemoryInfo {
    /// Returns a `CopyAccelerationStructureToMemoryInfo` with the specified `src` and `dst`.
    #[inline]
    pub fn new(src: Arc<AccelerationStructure>, dst: Subbuffer<[u8]>) -> Self {
        Self {
            src,
            dst,
            mode: CopyAccelerationStructureMode::Serialize,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            ref src,
            ref dst,
            mode,
            _ne: _,
        } = self;

        assert_eq!(device, src.device().as_ref());
        assert_eq!(device, dst.device().as_ref());

        mode.validate_device(device)
            .map_err(|err| ValidationError {
                context: "mode".into(),
                vuids: &["VUID-VkCopyAccelerationStructureToMemoryInfoKHR-mode-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        if !matches!(mode, CopyAccelerationStructureMode::Serialize) {
            return Err(ValidationError {
                context: "mode".into(),
                problem: "is not `CopyAccelerationStructureMode::Serialize`".into(),
                vuids: &["VUID-VkCopyAccelerationStructureToMemoryInfoKHR-mode-03412"],
                ..Default::default()
            });
        }

        // VUID-VkCopyAccelerationStructureToMemoryInfoKHR-src-04959
        // TODO: unsafe

        // VUID-VkCopyAccelerationStructureToMemoryInfoKHR-dst-03561
        // TODO: unsafe

        Ok(())
    }
}

/// Parameters for copying from memory into an acceleration structure.
#[derive(Clone, Debug)]
pub struct CopyMemoryToAccelerationStructureInfo {
    /// The memory to copy the structure from.
    ///
    /// There is no default value.
    pub src: Subbuffer<[u8]>,

    /// The acceleration structure to copy into.
    ///
    /// There is no default value.
    pub dst: Arc<AccelerationStructure>,

    /// Additional operations to perform during the copy.
    ///
    /// The default value is [`CopyAccelerationStructureMode::Deserialize`].
    pub mode: CopyAccelerationStructureMode,

    pub _ne: crate::NonExhaustive,
}

impl CopyMemoryToAccelerationStructureInfo {
    /// Returns a `CopyMemoryToAccelerationStructureInfo` with the specified `src` and `dst`.
    #[inline]
    pub fn new(src: Subbuffer<[u8]>, dst: Arc<AccelerationStructure>) -> Self {
        Self {
            src,
            dst,
            mode: CopyAccelerationStructureMode::Deserialize,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            ref src,
            ref dst,
            mode,
            _ne: _,
        } = self;

        assert_eq!(device, src.device().as_ref());
        assert_eq!(device, dst.device().as_ref());

        mode.validate_device(device)
            .map_err(|err| ValidationError {
                context: "mode".into(),
                vuids: &["VUID-VkCopyMemoryToAccelerationStructureInfoKHR-mode-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        if !matches!(mode, CopyAccelerationStructureMode::Deserialize) {
            return Err(ValidationError {
                context: "mode".into(),
                problem: "is not `CopyAccelerationStructureMode::Deserialize`".into(),
                vuids: &["VUID-VkCopyMemoryToAccelerationStructureInfoKHR-mode-03413"],
                ..Default::default()
            });
        }

        // VUID-VkCopyMemoryToAccelerationStructureInfoKHR-src-04960
        // TODO: unsafe

        // VUID-VkCopyMemoryToAccelerationStructureInfoKHR-pInfo-03414
        // TODO: unsafe

        // VUID-VkCopyMemoryToAccelerationStructureInfoKHR-dst-03746
        // TODO: unsafe

        Ok(())
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// What mode an acceleration structure copy command should operate in.
    CopyAccelerationStructureMode = CopyAccelerationStructureModeKHR(i32);

    /// Copy the source into the destination.
    /// This is a shallow copy: if the source holds references to other acceleration structures,
    /// only the references are copied, not the other acceleration structures.
    ///
    /// Both source and destination must have been created with the same
    /// [`AccelerationStructureCreateInfo`].
    Clone = CLONE,

    /// Create a more compact version of the source in the destination.
    /// This is a shallow copy: if the source holds references to other acceleration structures,
    /// only the references are copied, not the other acceleration structures.
    ///
    /// The source acceleration structure must have been built with the
    /// [`BuildAccelerationStructureFlags::ALLOW_COMPACTION`] flag.
    Compact = COMPACT,

    /// Serialize the acceleration structure into data in a semi-opaque format,
    /// that can be deserialized by a compatible Vulkan implementation.
    Serialize = SERIALIZE,

    /// Deserialize data back into an acceleration structure.
    Deserialize = DESERIALIZE,
}

vulkan_enum! {
    #[non_exhaustive]

    /// Where the building of an acceleration structure will take place.
    AccelerationStructureBuildType = AccelerationStructureBuildTypeKHR(i32);

    /// Building will take place on the host.
    Host = HOST,

    /// Building will take place on the device.
    Device = DEVICE,

    /// Building will take place on either the host or the device.
    HostOrDevice = HOST_OR_DEVICE,
}

/// The minimum sizes needed for various resources during an acceleration structure build operation.
#[derive(Clone, Debug)]
pub struct AccelerationStructureBuildSizesInfo {
    /// The minimum required size of the acceleration structure for a build or update operation.
    pub acceleration_structure_size: DeviceSize,

    /// The minimum required size of the scratch data buffer for an update operation.
    pub update_scratch_size: DeviceSize,

    /// The minimum required size of the scratch data buffer for a build operation.
    pub build_scratch_size: DeviceSize,

    pub _ne: crate::NonExhaustive,
}
