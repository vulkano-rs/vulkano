use crate::{
    acceleration_structure::{
        AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureBuildType,
        AccelerationStructureGeometries, AccelerationStructureGeometryAabbsData,
        AccelerationStructureGeometryInstancesData, AccelerationStructureGeometryInstancesDataType,
        AccelerationStructureGeometryTrianglesData, AccelerationStructureInstance,
        AccelerationStructureType, BuildAccelerationStructureMode, CopyAccelerationStructureInfo,
        CopyAccelerationStructureToMemoryInfo, CopyMemoryToAccelerationStructureInfo,
        TransformMatrix,
    },
    buffer::{BufferUsage, Subbuffer},
    command_buffer::{
        auto::{Resource, ResourceUseRef2},
        sys::RecordingCommandBuffer,
        AutoCommandBufferBuilder, ResourceInCommand,
    },
    device::{DeviceOwned, QueueFlags},
    query::{QueryPool, QueryType},
    sync::PipelineStageAccessFlags,
    DeviceSize, Requires, RequiresAllOf, RequiresOneOf, ValidationError, VulkanObject,
};
use smallvec::SmallVec;
use std::{mem::size_of, sync::Arc};

/// # Commands to do operations on acceleration structures.
impl<L> AutoCommandBufferBuilder<L> {
    /// Builds or updates an acceleration structure.
    ///
    /// # Safety
    ///
    /// If `info.mode` is [`BuildAccelerationStructureMode::Update`], then the rest of `info` must
    /// be valid for an update operation, as follows:
    /// - The source acceleration structure must have been previously built, with
    ///   [`BuildAccelerationStructureFlags::ALLOW_UPDATE`] included in
    ///   [`AccelerationStructureBuildGeometryInfo::flags`].
    /// - `info` must only differ from the `info` used to build the source acceleration structure,
    ///   according to the allowed changes listed in the [`acceleration_structure`] module.
    ///
    /// If `info.geometries` is [`AccelerationStructureGeometries::Triangles`], then for each
    /// geometry and the corresponding element in `build_range_infos`:
    /// - If [`index_data`] is `Some`, then if `index_max` is the highest index value in the index
    ///   buffer that is accessed, then the size of [`vertex_data`] must be at least<br/>
    ///   [`vertex_stride`] * ([`first_vertex`] + `index_max` + 1).
    /// - If [`transform_data`] is `Some`, then for the 3x4 matrix in the buffer, the first three
    ///   columns must be a 3x3 invertible matrix.
    ///
    /// If `info.geometries` is [`AccelerationStructureGeometries::Aabbs`], then for each geometry:
    /// - For each accessed [`AabbPositions`] element in
    ///   [`data`](AccelerationStructureGeometryAabbsData::data), each value in `min` must not be
    ///   greater than the corresponding value in `max`.
    ///
    /// If `info.geometries` is [`AccelerationStructureGeometries::Instances`], then the contents
    /// of the buffer in [`data`](AccelerationStructureGeometryInstancesData::data) must be valid,
    /// as follows:
    /// - Any [`AccelerationStructureInstance::acceleration_structure_reference`] address contained
    ///   in or referenced by [`data`](AccelerationStructureGeometryInstancesData::data) must be
    ///   either 0, or a device address that was returned from calling [`device_address`] on a
    ///   bottom-level acceleration structure.
    /// - If an [`AccelerationStructureInstance::acceleration_structure_reference`] address is not
    ///   0, then the corresponding acceleration structure object must be kept alive and not be
    ///   dropped while it is bound to the top-level acceleration structure.
    /// - If [`data`](AccelerationStructureGeometryInstancesData::data) is
    ///   [`AccelerationStructureGeometryInstancesDataType::Pointers`], then the addresses in the
    ///   buffer must be a multiple of 16.
    ///
    /// [`BuildAccelerationStructureFlags::ALLOW_UPDATE`]: crate::acceleration_structure::BuildAccelerationStructureFlags::ALLOW_UPDATE
    /// [`acceleration_structure`]: crate::acceleration_structure#updating-an-acceleration-structure
    /// [`index_data`]: AccelerationStructureGeometryTrianglesData::index_data
    /// [`vertex_data`]: AccelerationStructureGeometryTrianglesData::vertex_data
    /// [`vertex_stride`]: AccelerationStructureGeometryTrianglesData::vertex_stride
    /// [`first_vertex`]: AccelerationStructureBuildRangeInfo::first_vertex
    /// [`transform_data`]: AccelerationStructureGeometryTrianglesData::transform_data
    /// [`AabbPositions`]: crate::acceleration_structure::AabbPositions
    /// [`AccelerationStructureInstance::acceleration_structure_reference`]: crate::acceleration_structure::AccelerationStructureInstance::acceleration_structure_reference
    /// [`AccelerationStructureGeometryInstancesData::data`]: crate::acceleration_structure::AccelerationStructureGeometryInstancesData::data
    /// [`device_address`]: AccelerationStructure::device_address
    #[inline]
    pub unsafe fn build_acceleration_structure(
        &mut self,
        info: AccelerationStructureBuildGeometryInfo,
        build_range_infos: SmallVec<[AccelerationStructureBuildRangeInfo; 8]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_build_acceleration_structure(&info, &build_range_infos)?;

        Ok(unsafe { self.build_acceleration_structure_unchecked(info, build_range_infos) })
    }

    fn validate_build_acceleration_structure(
        &self,
        info: &AccelerationStructureBuildGeometryInfo,
        build_range_infos: &[AccelerationStructureBuildRangeInfo],
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_build_acceleration_structure(info, build_range_infos)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                context: "self".into(),
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn build_acceleration_structure_unchecked(
        &mut self,
        info: AccelerationStructureBuildGeometryInfo,
        build_range_infos: SmallVec<[AccelerationStructureBuildRangeInfo; 8]>,
    ) -> &mut Self {
        let mut used_resources = Vec::new();
        add_build_geometry_resources(&mut used_resources, &info);

        self.add_command(
            "build_acceleration_structure",
            used_resources,
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.build_acceleration_structure_unchecked(&info, &build_range_infos) };
            },
        );

        self
    }

    /// Builds or updates an acceleration structure, using [`AccelerationStructureBuildRangeInfo`]
    /// elements stored in an indirect buffer.
    ///
    /// # Safety
    ///
    /// The same requirements as for [`build_acceleration_structure`]. In addition, the following
    /// requirements apply for each [`AccelerationStructureBuildRangeInfo`] element contained in
    /// `indirect_buffer`:
    /// - [`primitive_count`] must not be greater than the corresponding element of
    ///   `max_primitive_counts`.
    /// - If `info.geometries` is [`AccelerationStructureGeometries::Instances`], then
    ///   [`primitive_count`] must not be greater than the [`max_instance_count`] limit. Otherwise,
    ///   it must not be greater than the [`max_primitive_count`] limit.
    ///
    /// If `info.geometries` is [`AccelerationStructureGeometries::Triangles`], then:
    /// - [`primitive_offset`] must be a multiple of:
    ///   - [`index_data.index_type().size()`] if [`index_data`] is `Some`.
    ///   - The byte size of the smallest component of [`vertex_format`] if [`index_data`] is
    ///     `None`.
    /// - [`transform_offset`] must be a multiple of 16.
    /// - The size of [`vertex_data`] must be at least<br/> [`primitive_offset`] +
    ///   ([`first_vertex`] + 3 * [`primitive_count`]) * [`vertex_stride`] <br/>if [`index_data`]
    ///   is `None`, and as in [`build_acceleration_structure`] if [`index_data`] is `Some`.
    /// - The size of [`index_data`] must be at least<br/> [`primitive_offset`] + 3 *
    ///   [`primitive_count`] * [`index_data.index_type().size()`].
    /// - The size of [`transform_data`] must be at least<br/> [`transform_offset`] +
    ///   `size_of::<TransformMatrix>()`.
    ///
    /// If `info.geometries` is [`AccelerationStructureGeometries::Aabbs`], then:
    /// - [`primitive_offset`] must be a multiple of 8.
    /// - The size of [`data`](AccelerationStructureGeometryAabbsData::data) must be at least<br/>
    ///   [`primitive_offset`] + [`primitive_count`] *
    ///   [`stride`](AccelerationStructureGeometryAabbsData::stride).
    ///
    /// If `info.geometries` is [`AccelerationStructureGeometries::Instances`], then:
    /// - [`primitive_offset`] must be a multiple of 16.
    /// - The size of [`data`](AccelerationStructureGeometryInstancesData::data) must be at least:
    ///   - [`primitive_offset`] + [`primitive_count`] *
    ///     `size_of::<AccelerationStructureInstance>()`<br/> if
    ///     [`data`](AccelerationStructureGeometryInstancesData::data) is
    ///     [`AccelerationStructureGeometryInstancesDataType::Values`].
    ///   - [`primitive_offset`] + [`primitive_count`] * `size_of::<DeviceSize>()`<br/> if
    ///     [`data`](AccelerationStructureGeometryInstancesData::data) is
    ///     [`AccelerationStructureGeometryInstancesDataType::Pointers`].
    ///
    /// [`build_acceleration_structure`]: Self::build_acceleration_structure
    /// [`primitive_count`]: AccelerationStructureBuildRangeInfo::primitive_count
    /// [`max_instance_count`]: crate::device::DeviceProperties::max_instance_count
    /// [`max_primitive_count`]: crate::device::DeviceProperties::max_primitive_count
    /// [`primitive_offset`]: AccelerationStructureBuildRangeInfo::primitive_offset
    /// [`index_data.index_type().size()`]: AccelerationStructureGeometryTrianglesData::index_data
    /// [`index_data`]: AccelerationStructureGeometryTrianglesData::index_data
    /// [`vertex_format`]: AccelerationStructureGeometryTrianglesData::vertex_format
    /// [`transform_offset`]: AccelerationStructureBuildRangeInfo::transform_offset
    /// [`vertex_data`]: AccelerationStructureGeometryTrianglesData::vertex_data
    /// [`first_vertex`]: AccelerationStructureBuildRangeInfo::first_vertex
    /// [`vertex_stride`]: AccelerationStructureGeometryTrianglesData::vertex_stride
    /// [`transform_data`]: AccelerationStructureGeometryTrianglesData::transform_data
    pub unsafe fn build_acceleration_structure_indirect(
        &mut self,
        info: AccelerationStructureBuildGeometryInfo,
        indirect_buffer: Subbuffer<[u8]>,
        stride: u32,
        max_primitive_counts: SmallVec<[u32; 8]>,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_build_acceleration_structure_indirect(
            &info,
            &indirect_buffer,
            stride,
            &max_primitive_counts,
        )?;

        Ok(unsafe {
            self.build_acceleration_structure_indirect_unchecked(
                info,
                indirect_buffer,
                stride,
                max_primitive_counts,
            )
        })
    }

    fn validate_build_acceleration_structure_indirect(
        &self,
        info: &AccelerationStructureBuildGeometryInfo,
        indirect_buffer: &Subbuffer<[u8]>,
        stride: u32,
        max_primitive_counts: &[u32],
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_build_acceleration_structure_indirect(
            info,
            indirect_buffer,
            stride,
            max_primitive_counts,
        )?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                context: "self".into(),
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn build_acceleration_structure_indirect_unchecked(
        &mut self,
        info: AccelerationStructureBuildGeometryInfo,
        indirect_buffer: Subbuffer<[u8]>,
        stride: u32,
        max_primitive_counts: SmallVec<[u32; 8]>,
    ) -> &mut Self {
        let mut used_resources = Vec::new();
        add_build_geometry_resources(&mut used_resources, &info);
        add_indirect_buffer_resources(&mut used_resources, &indirect_buffer);

        self.add_command(
            "build_acceleration_structure_indirect",
            used_resources,
            move |out: &mut RecordingCommandBuffer| {
                unsafe {
                    out.build_acceleration_structure_indirect_unchecked(
                        &info,
                        &indirect_buffer,
                        stride,
                        &max_primitive_counts,
                    )
                };
            },
        );

        self
    }

    /// Copies the data of one acceleration structure to another.
    ///
    /// # Safety
    ///
    /// - `info.src` must have been built when this command is executed.
    /// - If `info.mode` is [`CopyAccelerationStructureMode::Compact`], then `info.src` must have
    ///   been built with [`BuildAccelerationStructureFlags::ALLOW_COMPACTION`].
    ///
    /// [`CopyAccelerationStructureMode::Compact`]: crate::acceleration_structure::CopyAccelerationStructureMode::Compact
    /// [`BuildAccelerationStructureFlags::ALLOW_COMPACTION`]: crate::acceleration_structure::BuildAccelerationStructureFlags::ALLOW_COMPACTION
    #[inline]
    pub unsafe fn copy_acceleration_structure(
        &mut self,
        info: CopyAccelerationStructureInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_copy_acceleration_structure(&info)?;

        Ok(unsafe { self.copy_acceleration_structure_unchecked(info) })
    }

    fn validate_copy_acceleration_structure(
        &self,
        info: &CopyAccelerationStructureInfo,
    ) -> Result<(), Box<ValidationError>> {
        self.inner.validate_copy_acceleration_structure(info)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                context: "self".into(),
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdCopyAccelerationStructureKHR-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_acceleration_structure_unchecked(
        &mut self,
        info: CopyAccelerationStructureInfo,
    ) -> &mut Self {
        let &CopyAccelerationStructureInfo {
            ref src,
            ref dst,
            mode: _,
            _ne: _,
        } = &info;

        let src_buffer = src.buffer();
        let dst_buffer = dst.buffer();
        self.add_command(
            "copy_acceleration_structure",
            [
                (
                    ResourceInCommand::Source.into(),
                    Resource::Buffer {
                        buffer: src_buffer.clone(),
                        range: 0..src_buffer.size(), // TODO:
                        memory_access:
                            PipelineStageAccessFlags::AccelerationStructureCopy_AccelerationStructureRead,
                    },
                ),
                (
                    ResourceInCommand::Destination.into(),
                    Resource::Buffer {
                        buffer: dst_buffer.clone(),
                        range: 0..dst_buffer.size(), // TODO:
                        memory_access:
                            PipelineStageAccessFlags::AccelerationStructureCopy_AccelerationStructureWrite,
                    },
                ),
            ]
            .into_iter()
            .collect(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.copy_acceleration_structure_unchecked(&info) };
            },
        );

        self
    }

    /// Serializes the data of an acceleration structure and writes it to a buffer.
    ///
    /// # Safety
    ///
    /// - `info.src` must have been built when this command is executed.
    /// - `info.dst` must be large enough to hold the serialized form of `info.src`. This can be
    ///   queried using [`write_acceleration_structures_properties`] with a query pool whose type
    ///   is [`QueryType::AccelerationStructureSerializationSize`].
    ///
    /// [`write_acceleration_structures_properties`]: Self::write_acceleration_structures_properties
    #[inline]
    pub unsafe fn copy_acceleration_structure_to_memory(
        &mut self,
        info: CopyAccelerationStructureToMemoryInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_copy_acceleration_structure_to_memory(&info)?;

        Ok(unsafe { self.copy_acceleration_structure_to_memory_unchecked(info) })
    }

    fn validate_copy_acceleration_structure_to_memory(
        &self,
        info: &CopyAccelerationStructureToMemoryInfo,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_copy_acceleration_structure_to_memory(info)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                context: "self".into(),
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdCopyAccelerationStructureToMemoryKHR-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_acceleration_structure_to_memory_unchecked(
        &mut self,
        info: CopyAccelerationStructureToMemoryInfo,
    ) -> &mut Self {
        let &CopyAccelerationStructureToMemoryInfo {
            ref src,
            ref dst,
            mode: _,
            _ne: _,
        } = &info;

        let src_buffer = src.buffer();
        self.add_command(
            "copy_acceleration_structure_to_memory",
            [
                (
                    ResourceInCommand::Source.into(),
                    Resource::Buffer {
                        buffer: src_buffer.clone(),
                        range: 0..src_buffer.size(), // TODO:
                        memory_access:
                            PipelineStageAccessFlags::AccelerationStructureCopy_AccelerationStructureRead,
                    },
                ),
                (
                    ResourceInCommand::Destination.into(),
                    Resource::Buffer {
                        buffer: dst.clone(),
                        range: 0..dst.size(), // TODO:
                        memory_access:
                            PipelineStageAccessFlags::AccelerationStructureCopy_TransferWrite,
                    },
                ),
            ]
            .into_iter()
            .collect(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.copy_acceleration_structure_to_memory_unchecked(&info) };
            },
        );

        self
    }

    /// Reads data of a previously serialized acceleration structure from a buffer, and
    /// deserializes it back into an acceleration structure.
    ///
    /// # Safety
    ///
    /// - `info.src` must contain data previously serialized using
    ///   [`copy_acceleration_structure_to_memory`], and must have a format compatible with the
    ///   device (as queried by [`Device::acceleration_structure_is_compatible`]).
    /// - `info.dst.size()` must be at least the size that the structure in `info.src` had before
    ///   it was serialized.
    ///
    /// [`copy_acceleration_structure_to_memory`]: Self::copy_acceleration_structure_to_memory
    /// [`Device::acceleration_structure_is_compatible`]: crate::device::Device::acceleration_structure_is_compatible
    #[inline]
    pub unsafe fn copy_memory_to_acceleration_structure(
        &mut self,
        info: CopyMemoryToAccelerationStructureInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_copy_memory_to_acceleration_structure(&info)?;

        Ok(unsafe { self.copy_memory_to_acceleration_structure_unchecked(info) })
    }

    fn validate_copy_memory_to_acceleration_structure(
        &self,
        info: &CopyMemoryToAccelerationStructureInfo,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_copy_memory_to_acceleration_structure(info)?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                context: "self".into(),
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdCopyMemoryToAccelerationStructureKHR-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_memory_to_acceleration_structure_unchecked(
        &mut self,
        info: CopyMemoryToAccelerationStructureInfo,
    ) -> &mut Self {
        let &CopyMemoryToAccelerationStructureInfo {
            ref src,
            ref dst,
            mode: _,
            _ne: _,
        } = &info;

        let dst_buffer = dst.buffer();
        self.add_command(
            "copy_memory_to_acceleration_structure",
            [
                (
                    ResourceInCommand::Source.into(),
                    Resource::Buffer {
                        buffer: src.clone(),
                        range: 0..src.size(), // TODO:
                        memory_access:
                            PipelineStageAccessFlags::AccelerationStructureCopy_TransferRead,
                    },
                ),
                (
                    ResourceInCommand::Destination.into(),
                    Resource::Buffer {
                        buffer: dst_buffer.clone(),
                        range: 0..dst_buffer.size(), // TODO:
                        memory_access:
                            PipelineStageAccessFlags::AccelerationStructureCopy_AccelerationStructureWrite,
                    },
                ),
            ]
            .into_iter()
            .collect(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.copy_memory_to_acceleration_structure_unchecked(&info) };
            },
        );

        self
    }

    /// Writes the properties of one or more acceleration structures to a query.
    ///
    /// For each element in `acceleration_structures`, one query is written, in numeric order
    /// starting at `first_query`.
    ///
    /// # Safety
    ///
    /// - All elements of `acceleration_structures` must have been built when this command is
    ///   executed.
    /// - If `query_pool.query_type()` is [`QueryType::AccelerationStructureCompactedSize`], all
    ///   elements of `acceleration_structures` must have been built with
    ///   [`BuildAccelerationStructureFlags::ALLOW_COMPACTION`].
    /// - The queries must be unavailable, ensured by calling [`reset_query_pool`].
    ///
    /// [`BuildAccelerationStructureFlags::ALLOW_COMPACTION`]: crate::acceleration_structure::BuildAccelerationStructureFlags::ALLOW_COMPACTION
    /// [`reset_query_pool`]: Self::reset_query_pool
    #[inline]
    pub unsafe fn write_acceleration_structures_properties(
        &mut self,
        acceleration_structures: SmallVec<[Arc<AccelerationStructure>; 4]>,
        query_pool: Arc<QueryPool>,
        first_query: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_write_acceleration_structures_properties(
            &acceleration_structures,
            &query_pool,
            first_query,
        )?;

        Ok(unsafe {
            self.write_acceleration_structures_properties_unchecked(
                acceleration_structures,
                query_pool,
                first_query,
            )
        })
    }

    fn validate_write_acceleration_structures_properties(
        &self,
        acceleration_structures: &[Arc<AccelerationStructure>],
        query_pool: &QueryPool,
        first_query: u32,
    ) -> Result<(), Box<ValidationError>> {
        self.inner
            .validate_write_acceleration_structures_properties(
                acceleration_structures,
                query_pool,
                first_query,
            )?;

        if self.builder_state.render_pass.is_some() {
            return Err(Box::new(ValidationError {
                context: "self".into(),
                problem: "a render pass instance is active".into(),
                vuids: &["VUID-vkCmdWriteAccelerationStructuresPropertiesKHR-renderpass"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn write_acceleration_structures_properties_unchecked(
        &mut self,
        acceleration_structures: SmallVec<[Arc<AccelerationStructure>; 4]>,
        query_pool: Arc<QueryPool>,
        first_query: u32,
    ) -> &mut Self {
        if acceleration_structures.is_empty() {
            return self;
        }

        self.add_command(
            "write_acceleration_structures_properties",
            acceleration_structures.iter().enumerate().map(|(index, acs)| {
                let index = index as u32;
                let buffer = acs.buffer();

                (
                    ResourceInCommand::AccelerationStructure { index }.into(),
                    Resource::Buffer {
                        buffer: buffer.clone(),
                        range: 0..buffer.size(), // TODO:
                        memory_access:
                            PipelineStageAccessFlags::AccelerationStructureCopy_AccelerationStructureRead,
                    },
                )
            }).collect(),
            move |out: &mut RecordingCommandBuffer| {
                unsafe { out.write_acceleration_structures_properties_unchecked(
                    &acceleration_structures,
                    &query_pool,
                    first_query,
                ) };
            },
        );

        self
    }
}

fn add_build_geometry_resources(
    used_resources: &mut Vec<(ResourceUseRef2, Resource)>,
    info: &AccelerationStructureBuildGeometryInfo,
) {
    let &AccelerationStructureBuildGeometryInfo {
        flags: _,
        ref mode,
        ref dst_acceleration_structure,
        ref geometries,
        ref scratch_data,
        _ne: _,
    } = &info;

    match geometries {
        AccelerationStructureGeometries::Triangles(geometries) => {
            used_resources.extend(
                geometries.iter().enumerate().flat_map(|(index, triangles_data)| {
                    let index = index as u32;
                    let &AccelerationStructureGeometryTrianglesData {
                        flags: _,
                        vertex_format: _,
                        ref vertex_data,
                        vertex_stride: _,
                        max_vertex: _,
                        ref index_data,
                        ref transform_data,
                        _ne,
                    } = triangles_data;

                    let vertex_data = vertex_data.as_ref().unwrap();

                    [
                        (
                            ResourceInCommand::GeometryTrianglesVertexData { index }.into(),
                            Resource::Buffer {
                                buffer: vertex_data.clone(),
                                range: 0..vertex_data.size(), // TODO:
                                memory_access: PipelineStageAccessFlags::AccelerationStructureBuild_ShaderSampledRead
                                    | PipelineStageAccessFlags::AccelerationStructureBuild_ShaderStorageRead,
                            },
                        ),
                    ].into_iter()
                    .chain(index_data.as_ref().map(|index_data| {
                        let index_data_bytes = index_data.as_bytes();

                        (
                            ResourceInCommand::GeometryTrianglesIndexData { index }.into(),
                            Resource::Buffer {
                                buffer: index_data_bytes.clone(),
                                range: 0..index_data_bytes.size(), // TODO:
                                memory_access: PipelineStageAccessFlags::AccelerationStructureBuild_ShaderSampledRead
                                    | PipelineStageAccessFlags::AccelerationStructureBuild_ShaderStorageRead,
                            },
                        )
                    }))
                    .chain(transform_data.as_ref().map(|transform_data| {
                        (
                            ResourceInCommand::GeometryTrianglesTransformData { index }.into(),
                            Resource::Buffer {
                                buffer: transform_data.as_bytes().clone(),
                                range: 0..transform_data.size(), // TODO:
                                memory_access: PipelineStageAccessFlags::AccelerationStructureBuild_ShaderSampledRead
                                    | PipelineStageAccessFlags::AccelerationStructureBuild_ShaderStorageRead,
                            },
                        )
                    }))
                })
            );
        }
        AccelerationStructureGeometries::Aabbs(geometries) => {
            used_resources.extend(geometries.iter().enumerate().map(|(index, aabbs_data)| {
                let index = index as u32;
                let &AccelerationStructureGeometryAabbsData {
                    flags: _,
                    ref data,
                    stride: _,
                    _ne: _,
                } = aabbs_data;

                let data = data.as_ref().unwrap();

                (
                    ResourceInCommand::GeometryAabbsData { index }.into(),
                    Resource::Buffer {
                        buffer: data.as_bytes().clone(),
                        range: 0..data.size(), // TODO:
                        memory_access: PipelineStageAccessFlags::AccelerationStructureBuild_ShaderSampledRead
                            | PipelineStageAccessFlags::AccelerationStructureBuild_ShaderStorageRead,
                    },
                )
            }));
        }
        AccelerationStructureGeometries::Instances(instances_data) => {
            let &AccelerationStructureGeometryInstancesData {
                flags: _,
                ref data,
                _ne: _,
            } = instances_data;

            let data = match data {
                AccelerationStructureGeometryInstancesDataType::Values(data) => {
                    let data = data.as_ref().unwrap();
                    data.as_bytes()
                }
                AccelerationStructureGeometryInstancesDataType::Pointers(data) => {
                    let data = data.as_ref().unwrap();
                    data.as_bytes()
                }
            };
            let size = data.size();

            used_resources.push((
                ResourceInCommand::GeometryInstancesData.into(),
                Resource::Buffer {
                    buffer: data.clone(),
                    range: 0..size, // TODO:
                    memory_access: PipelineStageAccessFlags::AccelerationStructureBuild_ShaderSampledRead
                        | PipelineStageAccessFlags::AccelerationStructureBuild_ShaderStorageRead,
                },
            ));
        }
    };

    if let BuildAccelerationStructureMode::Update(src_acceleration_structure) = mode {
        let src_buffer = src_acceleration_structure.buffer();
        used_resources.push((
            ResourceInCommand::Source.into(),
            Resource::Buffer {
                buffer: src_buffer.clone(),
                range: 0..src_buffer.size(), // TODO:
                memory_access:
                    PipelineStageAccessFlags::AccelerationStructureBuild_AccelerationStructureRead,
            },
        ));
    }

    let dst_acceleration_structure = dst_acceleration_structure.as_ref().unwrap();
    let dst_buffer = dst_acceleration_structure.buffer();
    used_resources.push((
        ResourceInCommand::Destination.into(),
        Resource::Buffer {
            buffer: dst_buffer.clone(),
            range: 0..dst_buffer.size(), // TODO:
            memory_access:
                PipelineStageAccessFlags::AccelerationStructureBuild_AccelerationStructureWrite,
        },
    ));

    let scratch_data = scratch_data.as_ref().unwrap();
    used_resources.push((
        ResourceInCommand::ScratchData.into(),
        Resource::Buffer {
            buffer: scratch_data.clone(),
            range: 0..scratch_data.size(), // TODO:
            memory_access: PipelineStageAccessFlags::AccelerationStructureBuild_AccelerationStructureRead
                | PipelineStageAccessFlags::AccelerationStructureBuild_AccelerationStructureWrite,
        },
    ));
}

fn add_indirect_buffer_resources(
    used_resources: &mut Vec<(ResourceUseRef2, Resource)>,
    indirect_buffer: &Subbuffer<[u8]>,
) {
    used_resources.push((
        ResourceInCommand::IndirectBuffer.into(),
        Resource::Buffer {
            buffer: indirect_buffer.as_bytes().clone(),
            range: 0..indirect_buffer.size(), // TODO:
            memory_access: PipelineStageAccessFlags::AccelerationStructureBuild_IndirectCommandRead,
        },
    ));
}

impl RecordingCommandBuffer {
    #[inline]
    pub unsafe fn build_acceleration_structure(
        &mut self,
        info: &AccelerationStructureBuildGeometryInfo,
        build_range_infos: &[AccelerationStructureBuildRangeInfo],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_build_acceleration_structure(info, build_range_infos)?;

        Ok(unsafe { self.build_acceleration_structure_unchecked(info, build_range_infos) })
    }

    fn validate_build_acceleration_structure(
        &self,
        info: &AccelerationStructureBuildGeometryInfo,
        build_range_infos: &[AccelerationStructureBuildRangeInfo],
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                context: "self".into(),
                problem: "queue family does not support compute operations".into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-parameter
        info.validate(self.device())
            .map_err(|err| err.add_context("info"))?;

        let &AccelerationStructureBuildGeometryInfo {
            flags: _,
            ref mode,
            ref dst_acceleration_structure,
            ref geometries,
            ref scratch_data,
            _ne,
        } = info;

        let dst_acceleration_structure = dst_acceleration_structure.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                context: "info.dst_acceleration_structure".into(),
                problem: "is `None`".into(),
                // vuids?
                ..Default::default()
            })
        })?;
        let scratch_data = scratch_data.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                context: "info.scratch_data".into(),
                problem: "is `None`".into(),
                // vuids?
                ..Default::default()
            })
        })?;

        // VUID-vkCmdBuildAccelerationStructuresKHR-mode-04628
        // Ensured as long as `BuildAccelerationStructureMode` is exhaustive.

        // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-04630
        // Ensured by the definition of `BuildAccelerationStructureMode`.

        // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03403
        // VUID-vkCmdBuildAccelerationStructuresKHR-None-03407
        // VUID-vkCmdBuildAccelerationStructuresKHR-dstAccelerationStructure-03698
        // VUID-vkCmdBuildAccelerationStructuresKHR-dstAccelerationStructure-03702
        // VUID-vkCmdBuildAccelerationStructuresKHR-scratchData-03704
        // Ensured as long as only one element is provided in `info`.

        // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03668
        // VUID-vkCmdBuildAccelerationStructuresKHR-dstAccelerationStructure-03701
        // VUID-vkCmdBuildAccelerationStructuresKHR-dstAccelerationStructure-03703
        // VUID-vkCmdBuildAccelerationStructuresKHR-dstAccelerationStructure-03706
        // VUID-vkCmdBuildAccelerationStructuresKHR-scratchData-03705
        // Ensured by unsafe on `AccelerationStructure::new`.

        if !scratch_data
            .buffer()
            .usage()
            .intersects(BufferUsage::STORAGE_BUFFER)
        {
            return Err(Box::new(ValidationError {
                context: "info.scratch_data".into(),
                problem: "the buffer was not created with the `BufferUsage::STORAGE_BUFFER` usage"
                    .into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03674"],
                ..Default::default()
            }));
        }

        let min_acceleration_structure_scratch_offset_alignment = self
            .device()
            .physical_device()
            .properties()
            .min_acceleration_structure_scratch_offset_alignment
            .unwrap();

        if scratch_data.device_address().unwrap().get()
            % min_acceleration_structure_scratch_offset_alignment as u64
            != 0
        {
            return Err(Box::new(ValidationError {
                context: "info.scratch_data".into(),
                problem: "the device address of the buffer is not a multiple of the \
                    `min_acceleration_structure_scratch_offset_alignment` device property"
                    .into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03710"],
                ..Default::default()
            }));
        }

        match dst_acceleration_structure.ty() {
            AccelerationStructureType::TopLevel => {
                if !matches!(geometries, AccelerationStructureGeometries::Instances(_)) {
                    return Err(Box::new(ValidationError {
                        context: "info".into(),
                        problem: "`dst_acceleration_structure` is a top-level \
                            acceleration structure, but `geometries` is not \
                            `AccelerationStructureGeometries::Instances`"
                            .into(),
                        vuids: &[
                            "VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03789",
                            "VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03699",
                        ],
                        ..Default::default()
                    }));
                }
            }
            AccelerationStructureType::BottomLevel => {
                if matches!(geometries, AccelerationStructureGeometries::Instances(_)) {
                    return Err(Box::new(ValidationError {
                        context: "info".into(),
                        problem: "`dst_acceleration_structure` is a bottom-level \
                            acceleration structure, but `geometries` is \
                            `AccelerationStructureGeometries::Instances`"
                            .into(),
                        vuids: &[
                            "VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03791",
                            "VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03700",
                        ],
                        ..Default::default()
                    }));
                }
            }
            AccelerationStructureType::Generic => (),
        }

        if geometries.len() != build_range_infos.len() {
            return Err(Box::new(ValidationError {
                problem: "`info.geometries` and `build_range_infos` do not have the same length"
                    .into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-ppBuildRangeInfos-03676"],
                ..Default::default()
            }));
        }

        let max_primitive_count = self
            .device()
            .physical_device()
            .properties()
            .max_primitive_count
            .unwrap();
        let max_instance_count = self
            .device()
            .physical_device()
            .properties()
            .max_instance_count
            .unwrap();

        match geometries {
            AccelerationStructureGeometries::Triangles(geometries) => {
                for (geometry_index, (triangles_data, build_range_info)) in
                    geometries.iter().zip(build_range_infos).enumerate()
                {
                    let &AccelerationStructureGeometryTrianglesData {
                        flags: _,
                        vertex_format,
                        ref vertex_data,
                        vertex_stride,
                        max_vertex: _,
                        ref index_data,
                        ref transform_data,
                        _ne,
                    } = triangles_data;

                    let vertex_data = vertex_data.as_ref().ok_or_else(|| {
                        Box::new(ValidationError {
                            context: format!("info.geometries[{}].vertex_data", geometry_index)
                                .into(),
                            problem: "is `None`".into(),
                            // vuids?
                            ..Default::default()
                        })
                    })?;

                    let &AccelerationStructureBuildRangeInfo {
                        primitive_count,
                        primitive_offset,
                        first_vertex,
                        transform_offset,
                    } = build_range_info;

                    if primitive_count as u64 > max_primitive_count {
                        return Err(Box::new(ValidationError {
                            context: format!(
                                "build_range_infos[{}].primitive_count",
                                geometry_index
                            )
                            .into(),
                            problem: "exceeds the `max_primitive_count` limit".into(),
                            vuids: &["VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03795"],
                            ..Default::default()
                        }));
                    }

                    if !vertex_data
                        .buffer()
                        .usage()
                        .intersects(BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY)
                    {
                        return Err(Box::new(ValidationError {
                            context: format!("info.geometries[{}].vertex_data", geometry_index)
                                .into(),
                            problem: "the buffer was not created with the \
                                `BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY` usage"
                                .into(),
                            vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673"],
                            ..Default::default()
                        }));
                    }

                    let smallest_component_bits = vertex_format
                        .components()
                        .into_iter()
                        .filter(|&c| c != 0)
                        .min()
                        .unwrap() as u32;
                    let smallest_component_bytes = ((smallest_component_bits + 7) & !7) / 8;

                    if vertex_data.device_address().unwrap().get() % smallest_component_bytes as u64
                        != 0
                    {
                        return Err(Box::new(ValidationError {
                            context: format!("info.geometries[{}].vertex_data", geometry_index)
                                .into(),
                            problem: "the buffer's device address is not a multiple of the byte \
                                size of the smallest component of `vertex_format`"
                                .into(),
                            vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03711"],
                            ..Default::default()
                        }));
                    }

                    if let Some(index_data) = index_data {
                        if !index_data
                            .as_bytes()
                            .buffer()
                            .usage()
                            .intersects(BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY)
                        {
                            return Err(Box::new(ValidationError {
                                context: format!("info.geometries[{}].index_data", geometry_index)
                                    .into(),
                                problem: "the buffer was not created with the \
                                    `BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY` \
                                    usage"
                                    .into(),
                                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673"],
                                ..Default::default()
                            }));
                        }

                        if index_data.as_bytes().device_address().unwrap().get()
                            % index_data.index_type().size()
                            != 0
                        {
                            return Err(Box::new(ValidationError {
                                context: format!("info.geometries[{}].index_data", geometry_index)
                                    .into(),
                                problem: "the buffer's device address is not a multiple \
                                    of the size of the index type"
                                    .into(),
                                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03712"],
                                ..Default::default()
                            }));
                        }

                        if primitive_offset as u64 % index_data.index_type().size() != 0 {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`info.geometries` is \
                                    `AccelerationStructureGeometries::Triangles`, and \
                                    `build_range_infos[{}].primitive_offset` is not a multiple of \
                                    the size of the index type of \
                                    `info.geometries[{0}].index_data`",
                                    geometry_index,
                                )
                                .into(),
                                vuids: &["VUID-VkAccelerationStructureBuildRangeInfoKHR-primitiveOffset-03656"],
                                ..Default::default()
                            }));
                        }

                        if primitive_offset as DeviceSize
                            + 3 * primitive_count as DeviceSize
                                * index_data.index_type().size() as DeviceSize
                            > index_data.as_bytes().size()
                        {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`infos.geometries` is \
                                    `AccelerationStructureGeometries::Triangles`, \
                                    `info.geometries[{0}].index_data` is `Some`, and \
                                    `build_range_infos[{0}].primitive_offset` + \
                                    3 * `build_range_infos[{0}].primitive_count` * \
                                    `info.geometries[{0}].index_data.index_type().size` is \
                                    greater than the size of `infos.geometries[{0}].index_data`",
                                    geometry_index,
                                )
                                .into(),
                                ..Default::default()
                            }));
                        }
                    } else {
                        if primitive_offset % smallest_component_bytes != 0 {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`info.geometries` is \
                                    `AccelerationStructureGeometries::Triangles`, and
                                    `build_range_infos[{}].primitive_offset` is not a multiple of \
                                    the byte size of the smallest component of \
                                    `info.geometries[{0}].vertex_format`",
                                    geometry_index,
                                )
                                .into(),
                                vuids: &["VUID-VkAccelerationStructureBuildRangeInfoKHR-primitiveOffset-03657"],
                                ..Default::default()
                            }));
                        }

                        if primitive_offset as DeviceSize
                            + (first_vertex as DeviceSize + 3 * primitive_count as DeviceSize)
                                * vertex_stride as DeviceSize
                            > vertex_data.size()
                        {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`infos.geometries` is \
                                    `AccelerationStructureGeometries::Triangles`, \
                                    `info.geometries[{0}].index_data` is `None`, \
                                    and `build_range_infos[{0}].primitive_offset` + \
                                    (`build_range_infos[{0}].first_vertex` + 3 * \
                                    `build_range_infos[{0}].primitive_count`) * \
                                    `info.geometries[{0}].vertex_stride` is greater than the size \
                                    of `infos.geometries[{0}].vertex_data`",
                                    geometry_index,
                                )
                                .into(),
                                ..Default::default()
                            }));
                        }
                    }

                    if let Some(transform_data) = transform_data {
                        if !transform_data
                            .buffer()
                            .usage()
                            .intersects(BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY)
                        {
                            return Err(Box::new(ValidationError {
                                context: format!(
                                    "info.geometries[{}].transform_data",
                                    geometry_index
                                )
                                .into(),
                                problem: "the buffer was not created with the \
                                    `BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY` \
                                    usage"
                                    .into(),
                                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673"],
                                ..Default::default()
                            }));
                        }

                        if transform_data.device_address().unwrap().get() % 16 != 0 {
                            return Err(Box::new(ValidationError {
                                context: format!(
                                    "info.geometries[{}].transform_data",
                                    geometry_index
                                )
                                .into(),
                                problem: "the buffer's device address is not a multiple of 16"
                                    .into(),
                                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03810"],
                                ..Default::default()
                            }));
                        }

                        if transform_offset % 16 != 0 {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`info.geometries` is \
                                    `AccelerationStructureGeometries::Triangles`, and \
                                    `build_range_infos[{}].transform_offset` is not a multiple of \
                                    16",
                                    geometry_index,
                                )
                                .into(),
                                vuids: &["VUID-VkAccelerationStructureBuildRangeInfoKHR-transformOffset-03658"],
                                ..Default::default()
                            }));
                        }

                        if transform_offset as DeviceSize
                            + size_of::<TransformMatrix>() as DeviceSize
                            > transform_data.size()
                        {
                            return Err(Box::new(ValidationError {
                                problem: format!(
                                    "`infos.geometries` is \
                                    `AccelerationStructureGeometries::Triangles`, and \
                                    `build_range_infos[{0}].transform_offset` + \
                                    `size_of::<TransformMatrix>` is greater than the size of \
                                    `infos.geometries[{0}].transform_data`",
                                    geometry_index,
                                )
                                .into(),
                                ..Default::default()
                            }));
                        }
                    }
                }
            }
            AccelerationStructureGeometries::Aabbs(geometries) => {
                for (geometry_index, (aabbs_data, build_range_info)) in
                    geometries.iter().zip(build_range_infos).enumerate()
                {
                    let &AccelerationStructureGeometryAabbsData {
                        flags: _,
                        ref data,
                        stride,
                        _ne,
                    } = aabbs_data;

                    let data = data.as_ref().ok_or_else(|| {
                        Box::new(ValidationError {
                            context: format!("info.geometries[{}].data", geometry_index).into(),
                            problem: "is `None`".into(),
                            // vuids?
                            ..Default::default()
                        })
                    })?;

                    let &AccelerationStructureBuildRangeInfo {
                        primitive_count,
                        primitive_offset,
                        first_vertex: _,
                        transform_offset: _,
                    } = build_range_info;

                    if primitive_count as u64 > max_primitive_count {
                        return Err(Box::new(ValidationError {
                            context: format!("build_range_infos[{}]", geometry_index).into(),
                            problem: "exceeds the `max_primitive_count` limit".into(),
                            vuids: &["VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03794"],
                            ..Default::default()
                        }));
                    }

                    if !data
                        .buffer()
                        .usage()
                        .intersects(BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY)
                    {
                        return Err(Box::new(ValidationError {
                            context: format!("info.geometries[{}].data", geometry_index).into(),
                            problem: "the buffer was not created with the \
                                `BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY` \
                                usage"
                                .into(),
                            vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673"],
                            ..Default::default()
                        }));
                    }

                    if data.device_address().unwrap().get() % 8 != 0 {
                        return Err(Box::new(ValidationError {
                            context: format!("info.geometries[{}].data", geometry_index).into(),
                            problem: "the buffer's device address is not a multiple of 8".into(),
                            vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03714"],
                            ..Default::default()
                        }));
                    }

                    if primitive_offset % 8 != 0 {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`info.geometries` is \
                                `AccelerationStructureGeometries::Aabbs`, and \
                                `build_range_infos[{}].primitive_offset` is not a multiple of 8",
                                geometry_index,
                            )
                            .into(),
                            vuids: &["VUID-VkAccelerationStructureBuildRangeInfoKHR-primitiveOffset-03659"],
                            ..Default::default()
                        }));
                    }

                    if primitive_offset as DeviceSize
                        + primitive_count as DeviceSize * stride as DeviceSize
                        > data.size()
                    {
                        return Err(Box::new(ValidationError {
                            problem: format!(
                                "`infos.geometries` is `AccelerationStructureGeometries::Aabbs`,
                                and `build_range_infos[{0}].primitive_offset` + \
                                `build_range_infos[{0}].primitive_count` * \
                                `info.geometries[{0}].stride` is greater than the size of \
                                `infos.geometries[{0}].data`",
                                geometry_index,
                            )
                            .into(),
                            ..Default::default()
                        }));
                    }
                }
            }
            AccelerationStructureGeometries::Instances(instances_data) => {
                let &AccelerationStructureGeometryInstancesData {
                    flags: _,
                    ref data,
                    _ne,
                } = instances_data;

                let &AccelerationStructureBuildRangeInfo {
                    primitive_count,
                    primitive_offset,
                    first_vertex: _,
                    transform_offset: _,
                } = &build_range_infos[0];

                if primitive_count as u64 > max_instance_count {
                    return Err(Box::new(ValidationError {
                        context: "build_range_infos[0]".into(),
                        problem: "exceeds the the `max_instance_count` limit".into(),
                        vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03801"],
                        ..Default::default()
                    }));
                }

                if primitive_offset % 16 != 0 {
                    return Err(Box::new(ValidationError {
                        problem: "`info.geometries is` \
                            `AccelerationStructureGeometries::Instances`, and \
                            `build_range_infos[0].primitive_offset` is not a multiple of 16"
                            .into(),
                        vuids: &[
                            "VUID-VkAccelerationStructureBuildRangeInfoKHR-primitiveOffset-03660",
                        ],
                        ..Default::default()
                    }));
                }

                let data_buffer = match data {
                    AccelerationStructureGeometryInstancesDataType::Values(data) => {
                        let data = data.as_ref().ok_or_else(|| {
                            Box::new(ValidationError {
                                context: "info.geometries.data".into(),
                                problem: "is `None`".into(),
                                // vuids?
                                ..Default::default()
                            })
                        })?;

                        if data.device_address().unwrap().get() % 16 != 0 {
                            return Err(Box::new(ValidationError {
                                context: "info.geometries.data".into(),
                                problem: "is `AccelerationStructureGeometryInstancesDataType::\
                                    Values`, and the buffer's device address is not a multiple of \
                                    16"
                                .into(),
                                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03715"],
                                ..Default::default()
                            }));
                        }

                        if primitive_offset as DeviceSize
                            + primitive_count as DeviceSize
                                * size_of::<AccelerationStructureInstance>() as DeviceSize
                            > data.size()
                        {
                            return Err(Box::new(ValidationError {
                                problem: "`infos.geometries` is \
                                    `AccelerationStructureGeometries::Instances`, \
                                    `infos.geometries.data` is \
                                    `AccelerationStructureGeometryInstancesDataType::Values`, and \
                                    `build_range_infos[0].primitive_offset` + \
                                    `build_range_infos[0].primitive_count` * \
                                    `size_of::<AccelerationStructureInstance>()` is greater than \
                                    the size of `infos.geometries.data`"
                                    .into(),
                                ..Default::default()
                            }));
                        }

                        data.buffer()
                    }
                    AccelerationStructureGeometryInstancesDataType::Pointers(data) => {
                        let data = data.as_ref().ok_or_else(|| {
                            Box::new(ValidationError {
                                context: "info.geometries.data".into(),
                                problem: "is `None`".into(),
                                // vuids?
                                ..Default::default()
                            })
                        })?;

                        if !data
                            .buffer()
                            .usage()
                            .intersects(BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY)
                        {
                            return Err(Box::new(ValidationError {
                                context: "info.geometries.data".into(),
                                problem: "the buffer was not created with the \
                                    `BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY` \
                                    usage"
                                    .into(),
                                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673"],
                                ..Default::default()
                            }));
                        }

                        if data.device_address().unwrap().get() % 8 != 0 {
                            return Err(Box::new(ValidationError {
                                context: "info.geometries.data".into(),
                                problem: "is `AccelerationStructureGeometryInstancesDataType::\
                                    Pointers` and the buffer's device address is not a multiple \
                                    of 8"
                                    .into(),
                                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03716"],
                                ..Default::default()
                            }));
                        }

                        if primitive_offset as DeviceSize
                            + primitive_count as DeviceSize * size_of::<DeviceSize>() as DeviceSize
                            > data.size()
                        {
                            return Err(Box::new(ValidationError {
                                problem: "`infos.geometries` is \
                                    `AccelerationStructureGeometries::Instances`, \
                                    `infos.geometries.data` is \
                                    `AccelerationStructureGeometryInstancesDataType::Pointers`, \
                                    and `build_range_infos[0].primitive_offset` + \
                                    `build_range_infos[0].primitive_count` * \
                                    `size_of::<DeviceSize>()` is greater than the \
                                    size of `infos.geometries.data`"
                                    .into(),
                                ..Default::default()
                            }));
                        }

                        // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03717
                        // unsafe

                        data.buffer()
                    }
                };

                if !data_buffer
                    .usage()
                    .intersects(BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY)
                {
                    return Err(Box::new(ValidationError {
                        context: "info.geometries.data".into(),
                        problem: "the buffer was not created with the \
                            `BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY` usage"
                            .into(),
                        vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-geometry-03673"],
                        ..Default::default()
                    }));
                }

                // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-06707
                // unsafe
            }
        }

        let build_size_info = unsafe {
            self.device().acceleration_structure_build_sizes_unchecked(
                AccelerationStructureBuildType::Device,
                info,
                &build_range_infos
                    .iter()
                    .map(|info| info.primitive_count)
                    .collect::<SmallVec<[_; 8]>>(),
            )
        };

        if dst_acceleration_structure.size() < build_size_info.acceleration_structure_size {
            return Err(Box::new(ValidationError {
                context: "info.dst_acceleration_structure".into(),
                problem: "size is too small to hold the resulting acceleration structure data"
                    .into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03675"],
                ..Default::default()
            }));
        }

        match mode {
            BuildAccelerationStructureMode::Build => {
                if scratch_data.size() < build_size_info.build_scratch_size {
                    return Err(Box::new(ValidationError {
                        context: "info.scratch_data".into(),
                        problem: "size is too small for the build operation".into(),
                        vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03671"],
                        ..Default::default()
                    }));
                }
            }
            BuildAccelerationStructureMode::Update(_src_acceleration_structure) => {
                if scratch_data.size() < build_size_info.update_scratch_size {
                    return Err(Box::new(ValidationError {
                        context: "info.scratch_data".into(),
                        problem: "size is too small for the update operation".into(),
                        vuids: &["VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03672"],
                        ..Default::default()
                    }));
                }

                // VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03667
                // unsafe
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn build_acceleration_structure_unchecked(
        &mut self,
        info: &AccelerationStructureBuildGeometryInfo,
        build_range_infos: &[AccelerationStructureBuildRangeInfo],
    ) -> &mut Self {
        let info_fields1_vk = info.to_vk_fields1();
        let info_vk = info.to_vk(&info_fields1_vk);

        let build_range_info_elements_vk: SmallVec<[_; 8]> = build_range_infos
            .iter()
            .map(AccelerationStructureBuildRangeInfo::to_vk)
            .collect();
        let build_range_info_pointers_vk: SmallVec<[_; 8]> = build_range_info_elements_vk
            .iter()
            .map(|p| -> *const _ { p })
            .collect();

        let fns = self.device().fns();
        unsafe {
            (fns.khr_acceleration_structure
                .cmd_build_acceleration_structures_khr)(
                self.handle(),
                1,
                &info_vk,
                build_range_info_pointers_vk.as_ptr(),
            )
        };

        self
    }

    #[inline]
    pub unsafe fn build_acceleration_structure_indirect(
        &mut self,
        info: &AccelerationStructureBuildGeometryInfo,
        indirect_buffer: &Subbuffer<[u8]>,
        stride: u32,
        max_primitive_counts: &[u32],
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_build_acceleration_structure_indirect(
            info,
            indirect_buffer,
            stride,
            max_primitive_counts,
        )?;

        Ok(unsafe {
            self.build_acceleration_structure_indirect_unchecked(
                info,
                indirect_buffer,
                stride,
                max_primitive_counts,
            )
        })
    }

    fn validate_build_acceleration_structure_indirect(
        &self,
        info: &AccelerationStructureBuildGeometryInfo,
        indirect_buffer: &Subbuffer<[u8]>,
        stride: u32,
        max_primitive_counts: &[u32],
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .device()
            .enabled_features()
            .acceleration_structure_indirect_build
        {
            return Err(Box::new(ValidationError {
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature("acceleration_structure_indirect_build")])]),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-accelerationStructureIndirectBuild-03650"],
                ..Default::default()
            }));
        }

        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                context: "self".into(),
                problem: "queue family does not support compute operations".into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-parameter
        info.validate(self.device())
            .map_err(|err| err.add_context("info"))?;

        let &AccelerationStructureBuildGeometryInfo {
            flags: _,
            ref mode,
            ref dst_acceleration_structure,
            ref geometries,
            ref scratch_data,
            _ne,
        } = info;

        let dst_acceleration_structure = dst_acceleration_structure.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                context: "info.dst_acceleration_structure".into(),
                problem: "is `None`".into(),
                // vuids?
                ..Default::default()
            })
        })?;
        let scratch_data = scratch_data.as_ref().ok_or_else(|| {
            Box::new(ValidationError {
                context: "info.scratch_data".into(),
                problem: "is `None`".into(),
                // vuids?
                ..Default::default()
            })
        })?;

        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-mode-04628
        // Ensured as long as `BuildAccelerationStructureMode` is exhaustive.

        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-04630
        // Ensured by the definition of `BuildAccelerationStructureMode`.

        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03403
        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-None-03407
        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-dstAccelerationStructure-03698
        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-dstAccelerationStructure-03702
        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-scratchData-03704
        // Ensured as long as only one element is provided in `info`.

        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03668
        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-dstAccelerationStructure-03701
        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-dstAccelerationStructure-03703
        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-dstAccelerationStructure-03706
        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-scratchData-03705
        // Ensured by unsafe on `AccelerationStructure::new`.

        if !scratch_data
            .buffer()
            .usage()
            .intersects(BufferUsage::STORAGE_BUFFER)
        {
            return Err(Box::new(ValidationError {
                context: "info.scratch_data".into(),
                problem: "the buffer was not created with the `BufferUsage::STORAGE_BUFFER` usage"
                    .into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03674"],
                ..Default::default()
            }));
        }

        let min_acceleration_structure_scratch_offset_alignment = self
            .device()
            .physical_device()
            .properties()
            .min_acceleration_structure_scratch_offset_alignment
            .unwrap();

        if scratch_data.device_address().unwrap().get()
            % min_acceleration_structure_scratch_offset_alignment as u64
            != 0
        {
            return Err(Box::new(ValidationError {
                context: "info.scratch_data".into(),
                problem: "the device address of the buffer is not a multiple of the \
                    `min_acceleration_structure_scratch_offset_alignment` device property"
                    .into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03710"],
                ..Default::default()
            }));
        }

        match dst_acceleration_structure.ty() {
            AccelerationStructureType::TopLevel => {
                if !matches!(geometries, AccelerationStructureGeometries::Instances(_)) {
                    return Err(Box::new(ValidationError {
                        context: "info".into(),
                        problem: "`dst_acceleration_structure` is a top-level \
                            acceleration structure, but `geometries` is not \
                            `AccelerationStructureGeometries::Instances`"
                            .into(),
                        vuids: &[
                            "VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03789",
                            "VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03699",
                        ],
                        ..Default::default()
                    }));
                }
            }
            AccelerationStructureType::BottomLevel => {
                if matches!(geometries, AccelerationStructureGeometries::Instances(_)) {
                    return Err(Box::new(ValidationError {
                        context: "info".into(),
                        problem: "`dst_acceleration_structure` is a bottom-level \
                            acceleration structure, but `geometries` is \
                            `AccelerationStructureGeometries::Instances`"
                            .into(),
                        vuids: &[
                            "VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03791",
                            "VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03700",
                        ],
                        ..Default::default()
                    }));
                }
            }
            AccelerationStructureType::Generic => (),
        }

        if geometries.len() != max_primitive_counts.len() {
            return Err(Box::new(ValidationError {
                problem: "`info.geometries` and `max_primitive_counts` do not have the same length"
                    .into(),
                vuids: &[
                    "VUID-vkCmdBuildAccelerationStructuresIndirectKHR-ppMaxPrimitiveCounts-parameter",
                ],
                ..Default::default()
            }));
        }

        match geometries {
            AccelerationStructureGeometries::Triangles(geometries) => {
                for (geometry_index, triangles_data) in geometries.iter().enumerate() {
                    let &AccelerationStructureGeometryTrianglesData {
                        flags: _,
                        vertex_format,
                        ref vertex_data,
                        vertex_stride: _,
                        max_vertex: _,
                        ref index_data,
                        ref transform_data,
                        _ne,
                    } = triangles_data;

                    let vertex_data = vertex_data.as_ref().ok_or_else(|| {
                        Box::new(ValidationError {
                            context: format!("info.geometries[{}].vertex_data", geometry_index)
                                .into(),
                            problem: "is `None`".into(),
                            // vuids?
                            ..Default::default()
                        })
                    })?;

                    // VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03795
                    // unsafe

                    if !vertex_data
                        .buffer()
                        .usage()
                        .intersects(BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY)
                    {
                        return Err(Box::new(ValidationError {
                            context: format!("info.geometries[{}].vertex_data", geometry_index)
                                .into(),
                            problem: "the buffer was not created with the \
                                `BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY` usage"
                                .into(),
                            vuids: &[
                                "VUID-vkCmdBuildAccelerationStructuresIndirectKHR-geometry-03673",
                            ],
                            ..Default::default()
                        }));
                    }

                    let smallest_component_bits = vertex_format
                        .components()
                        .into_iter()
                        .filter(|&c| c != 0)
                        .min()
                        .unwrap() as u32;
                    let smallest_component_bytes = ((smallest_component_bits + 7) & !7) / 8;

                    if vertex_data.device_address().unwrap().get() % smallest_component_bytes as u64
                        != 0
                    {
                        return Err(Box::new(ValidationError {
                            context: format!("info.geometries[{}].vertex_data", geometry_index)
                                .into(),
                            problem: "the buffer's device address is not a multiple of the byte \
                                size of the smallest component of `vertex_format`"
                                .into(),
                            vuids: &[
                                "VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03711",
                            ],
                            ..Default::default()
                        }));
                    }

                    if let Some(index_data) = index_data {
                        if !index_data
                            .as_bytes()
                            .buffer()
                            .usage()
                            .intersects(BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY)
                        {
                            return Err(Box::new(ValidationError {
                                context: format!("info.geometries[{}].index_data", geometry_index)
                                    .into(),
                                problem: "the buffer was not created with the \
                                    `BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY` \
                                    usage"
                                    .into(),
                                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-geometry-03673"],
                                ..Default::default()
                            }));
                        }

                        if index_data.as_bytes().device_address().unwrap().get()
                            % index_data.index_type().size()
                            != 0
                        {
                            return Err(Box::new(ValidationError {
                                context: format!("info.geometries[{}].index_data", geometry_index)
                                    .into(),
                                problem: "the buffer's device address is not a multiple \
                                    of the size of the index type"
                                    .into(),
                                vuids: &[
                                    "VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03712",
                                ],
                                ..Default::default()
                            }));
                        }

                        // VUID-VkAccelerationStructureBuildRangeInfoKHR-primitiveOffset-03656
                        // unsafe
                    } else {
                        // VUID-VkAccelerationStructureBuildRangeInfoKHR-primitiveOffset-03657
                        // unsafe
                    }

                    if let Some(transform_data) = transform_data {
                        if !transform_data
                            .buffer()
                            .usage()
                            .intersects(BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY)
                        {
                            return Err(Box::new(ValidationError {
                                context: format!(
                                    "info.geometries[{}].transform_data",
                                    geometry_index
                                )
                                .into(),
                                problem: "the buffer was not created with the \
                                    `BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY` \
                                    usage"
                                    .into(),
                                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-geometry-03673"],
                                ..Default::default()
                            }));
                        }

                        if transform_data.device_address().unwrap().get() % 16 != 0 {
                            return Err(Box::new(ValidationError {
                                context: format!(
                                    "info.geometries[{}].transform_data",
                                    geometry_index
                                )
                                .into(),
                                problem: "the buffer's device address is not a multiple of 16"
                                    .into(),
                                vuids: &[
                                    "VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03810",
                                ],
                                ..Default::default()
                            }));
                        }

                        // VUID-VkAccelerationStructureBuildRangeInfoKHR-transformOffset-03658
                        // unsafe
                    }
                }
            }
            AccelerationStructureGeometries::Aabbs(geometries) => {
                for (geometry_index, aabbs_data) in geometries.iter().enumerate() {
                    let &AccelerationStructureGeometryAabbsData {
                        flags: _,
                        ref data,
                        stride: _,
                        _ne,
                    } = aabbs_data;

                    let data = data.as_ref().ok_or_else(|| {
                        Box::new(ValidationError {
                            context: format!("info.geometries[{}].data", geometry_index).into(),
                            problem: "is `None`".into(),
                            // vuids?
                            ..Default::default()
                        })
                    })?;

                    // VUID-VkAccelerationStructureBuildGeometryInfoKHR-type-03794
                    // unsafe

                    if !data
                        .buffer()
                        .usage()
                        .intersects(BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY)
                    {
                        return Err(Box::new(ValidationError {
                            context: format!("info.geometries[{}].data", geometry_index).into(),
                            problem: "the buffer was not created with the \
                                `BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY` usage"
                                .into(),
                            vuids: &[
                                "VUID-vkCmdBuildAccelerationStructuresIndirectKHR-geometry-03673",
                            ],
                            ..Default::default()
                        }));
                    }

                    if data.device_address().unwrap().get() % 8 != 0 {
                        return Err(Box::new(ValidationError {
                            context: format!("info.geometries[{}].data", geometry_index).into(),
                            problem: "the buffer's device address is not a multiple of 8".into(),
                            vuids: &[
                                "VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03714",
                            ],
                            ..Default::default()
                        }));
                    }

                    // VUID-VkAccelerationStructureBuildRangeInfoKHR-primitiveOffset-03659
                    // unsafe
                }
            }
            AccelerationStructureGeometries::Instances(instances_data) => {
                let &AccelerationStructureGeometryInstancesData {
                    flags: _,
                    ref data,
                    _ne,
                } = instances_data;

                // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03801
                // unsafe

                let data_buffer = match data {
                    AccelerationStructureGeometryInstancesDataType::Values(data) => {
                        let data = data.as_ref().ok_or_else(|| {
                            Box::new(ValidationError {
                                context: "info.geometries.data".into(),
                                problem: "is `None`".into(),
                                // vuids?
                                ..Default::default()
                            })
                        })?;

                        if data.device_address().unwrap().get() % 16 != 0 {
                            return Err(Box::new(ValidationError {
                                context: "info.geometries.data".into(),
                                problem: "is `AccelerationStructureGeometryInstancesDataType::\
                                    Values` and the buffer's device address is not a multiple of \
                                    16"
                                .into(),
                                vuids: &[
                                    "VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03715",
                                ],
                                ..Default::default()
                            }));
                        }

                        data.buffer()
                    }
                    AccelerationStructureGeometryInstancesDataType::Pointers(data) => {
                        let data = data.as_ref().ok_or_else(|| {
                            Box::new(ValidationError {
                                context: "info.geometries.data".into(),
                                problem: "is `None`".into(),
                                // vuids?
                                ..Default::default()
                            })
                        })?;

                        if !data
                            .buffer()
                            .usage()
                            .intersects(BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY)
                        {
                            return Err(Box::new(ValidationError {
                                context: "info.geometries.data".into(),
                                problem: "the buffer was not created with the \
                                    `BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY` \
                                    usage"
                                    .into(),
                                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-geometry-03673"],
                                ..Default::default()
                            }));
                        }

                        if data.device_address().unwrap().get() % 8 != 0 {
                            return Err(Box::new(ValidationError {
                                context: "info.geometries.data".into(),
                                problem: "is `AccelerationStructureGeometryInstancesDataType::\
                                    Pointers` and the buffer's device address is not a multiple \
                                    of 8"
                                    .into(),
                                vuids: &[
                                    "VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03716",
                                ],
                                ..Default::default()
                            }));
                        }

                        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03717
                        // unsafe

                        data.buffer()
                    }
                };

                if !data_buffer
                    .usage()
                    .intersects(BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY)
                {
                    return Err(Box::new(ValidationError {
                        context: "info.geometries.data".into(),
                        problem: "the buffer was not created with the \
                            `BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY` usage"
                            .into(),
                        vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-geometry-03673"],
                        ..Default::default()
                    }));
                }

                // VUID-VkAccelerationStructureBuildRangeInfoKHR-primitiveOffset-03660
                // unsafe

                // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-06707
                // unsafe
            }
        }

        let build_size_info = unsafe {
            self.device().acceleration_structure_build_sizes_unchecked(
                AccelerationStructureBuildType::Device,
                info,
                max_primitive_counts,
            )
        };

        if dst_acceleration_structure.size() < build_size_info.acceleration_structure_size {
            return Err(Box::new(ValidationError {
                context: "info.dst_acceleration_structure".into(),
                problem: "size is too small to hold the resulting acceleration structure data"
                    .into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03652"],
                ..Default::default()
            }));
        }

        match mode {
            BuildAccelerationStructureMode::Build => {
                if scratch_data.size() < build_size_info.build_scratch_size {
                    return Err(Box::new(ValidationError {
                        context: "info.scratch_data".into(),
                        problem: "size is too small for the build operation".into(),
                        vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03671"],
                        ..Default::default()
                    }));
                }
            }
            BuildAccelerationStructureMode::Update(_src_acceleration_structure) => {
                if scratch_data.size() < build_size_info.update_scratch_size {
                    return Err(Box::new(ValidationError {
                        context: "info.scratch_data".into(),
                        problem: "size is too small for the update operation".into(),
                        vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03672"],
                        ..Default::default()
                    }));
                }

                // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pInfos-03667
                // unsafe
            }
        }

        if geometries.len() as DeviceSize * stride as DeviceSize > indirect_buffer.size() {
            return Err(Box::new(ValidationError {
                problem: "`info.geometries.len()` * `stride` is greater than the size of \
                    `indirect_buffer`".into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pIndirectDeviceAddresses-03646"],
                ..Default::default()
            }));
        }

        if !indirect_buffer
            .buffer()
            .usage()
            .intersects(BufferUsage::INDIRECT_BUFFER)
        {
            return Err(Box::new(ValidationError {
                context: "indirect_buffer".into(),
                problem: "the buffer was not created with the `BufferUsage::INDIRECT_BUFFER` usage".into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pIndirectDeviceAddresses-03647"],
                ..Default::default()
            }));
        }

        if indirect_buffer.device_address().unwrap().get() % 4 != 0 {
            return Err(Box::new(ValidationError {
                context: "indirect_buffer".into(),
                problem: "the buffer's device address is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pIndirectDeviceAddresses-03648"],
                ..Default::default()
            }));
        }

        if stride % 4 != 0 {
            return Err(Box::new(ValidationError {
                context: "stride".into(),
                problem: "is not a multiple of 4".into(),
                vuids: &["VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pIndirectStrides-03787"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-pIndirectDeviceAddresses-03651
        // unsafe

        // VUID-vkCmdBuildAccelerationStructuresIndirectKHR-ppMaxPrimitiveCounts-03653
        // unsafe

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn build_acceleration_structure_indirect_unchecked(
        &mut self,
        info: &AccelerationStructureBuildGeometryInfo,
        indirect_buffer: &Subbuffer<[u8]>,
        stride: u32,
        max_primitive_counts: &[u32],
    ) -> &mut Self {
        let info_fields1_vk = info.to_vk_fields1();
        let info_vk = info.to_vk(&info_fields1_vk);

        let fns = self.device().fns();
        unsafe {
            (fns.khr_acceleration_structure
                .cmd_build_acceleration_structures_indirect_khr)(
                self.handle(),
                1,
                &info_vk,
                &indirect_buffer.device_address().unwrap().get(),
                &stride,
                &max_primitive_counts.as_ptr(),
            )
        };

        self
    }

    #[inline]
    pub unsafe fn copy_acceleration_structure(
        &mut self,
        info: &CopyAccelerationStructureInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_copy_acceleration_structure(info)?;

        Ok(unsafe { self.copy_acceleration_structure_unchecked(info) })
    }

    fn validate_copy_acceleration_structure(
        &self,
        info: &CopyAccelerationStructureInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                context: "self".into(),
                problem: "queue family does not support compute operations".into(),
                vuids: &["VUID-vkCmdCopyAccelerationStructureKHR-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdCopyAccelerationStructureKHR-pInfo-parameter
        info.validate(self.device())
            .map_err(|err| err.add_context("info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_acceleration_structure_unchecked(
        &mut self,
        info: &CopyAccelerationStructureInfo,
    ) -> &mut Self {
        let info_vk = info.to_vk();

        let fns = self.device().fns();
        unsafe {
            (fns.khr_acceleration_structure
                .cmd_copy_acceleration_structure_khr)(self.handle(), &info_vk)
        };

        self
    }

    #[inline]
    pub unsafe fn copy_acceleration_structure_to_memory(
        &mut self,
        info: &CopyAccelerationStructureToMemoryInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_copy_acceleration_structure_to_memory(info)?;

        Ok(unsafe { self.copy_acceleration_structure_to_memory_unchecked(info) })
    }

    fn validate_copy_acceleration_structure_to_memory(
        &self,
        info: &CopyAccelerationStructureToMemoryInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                context: "self".into(),
                problem: "queue family does not support compute operations".into(),
                vuids: &["VUID-vkCmdCopyAccelerationStructureToMemoryKHR-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        if info.dst.device_address().unwrap().get() % 256 != 0 {
            return Err(Box::new(ValidationError {
                context: "info.dst".into(),
                problem: "the device address of the buffer is not a multiple of 256".into(),
                vuids: &["VUID-vkCmdCopyAccelerationStructureToMemoryKHR-pInfo-03740"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdCopyAccelerationStructureToMemoryKHR-pInfo-parameter
        info.validate(self.device())
            .map_err(|err| err.add_context("info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_acceleration_structure_to_memory_unchecked(
        &mut self,
        info: &CopyAccelerationStructureToMemoryInfo,
    ) -> &mut Self {
        let info_vk = info.to_vk();

        let fns = self.device().fns();
        unsafe {
            (fns.khr_acceleration_structure
                .cmd_copy_acceleration_structure_to_memory_khr)(self.handle(), &info_vk)
        };

        self
    }

    #[inline]
    pub unsafe fn copy_memory_to_acceleration_structure(
        &mut self,
        info: &CopyMemoryToAccelerationStructureInfo,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_copy_memory_to_acceleration_structure(info)?;

        Ok(unsafe { self.copy_memory_to_acceleration_structure_unchecked(info) })
    }

    fn validate_copy_memory_to_acceleration_structure(
        &self,
        info: &CopyMemoryToAccelerationStructureInfo,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                context: "self".into(),
                problem: "queue family does not support compute operations".into(),
                vuids: &["VUID-vkCmdCopyMemoryToAccelerationStructureKHR-commandBuffer-cmdpool"],
                ..Default::default()
            }));
        }

        if info.src.device_address().unwrap().get() % 256 != 0 {
            return Err(Box::new(ValidationError {
                context: "info.src".into(),
                problem: "the device address of the buffer is not a multiple of 256".into(),
                vuids: &["VUID-vkCmdCopyMemoryToAccelerationStructureKHR-pInfo-03743"],
                ..Default::default()
            }));
        }

        // VUID-vkCmdCopyMemoryToAccelerationStructureKHR-pInfo-parameter
        info.validate(self.device())
            .map_err(|err| err.add_context("info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn copy_memory_to_acceleration_structure_unchecked(
        &mut self,
        info: &CopyMemoryToAccelerationStructureInfo,
    ) -> &mut Self {
        let info_vk = info.to_vk();

        let fns = self.device().fns();
        unsafe {
            (fns.khr_acceleration_structure
                .cmd_copy_memory_to_acceleration_structure_khr)(self.handle(), &info_vk)
        };

        self
    }

    #[inline]
    pub unsafe fn write_acceleration_structures_properties(
        &mut self,
        acceleration_structures: &[Arc<AccelerationStructure>],
        query_pool: &QueryPool,
        first_query: u32,
    ) -> Result<&mut Self, Box<ValidationError>> {
        self.validate_write_acceleration_structures_properties(
            acceleration_structures,
            query_pool,
            first_query,
        )?;

        Ok(unsafe {
            self.write_acceleration_structures_properties_unchecked(
                acceleration_structures,
                query_pool,
                first_query,
            )
        })
    }

    fn validate_write_acceleration_structures_properties(
        &self,
        acceleration_structures: &[Arc<AccelerationStructure>],
        query_pool: &QueryPool,
        first_query: u32,
    ) -> Result<(), Box<ValidationError>> {
        if !self
            .queue_family_properties()
            .queue_flags
            .intersects(QueueFlags::COMPUTE)
        {
            return Err(Box::new(ValidationError {
                context: "self".into(),
                problem: "queue family does not support compute operations".into(),
                vuids: &[
                    "VUID-vkCmdWriteAccelerationStructuresPropertiesKHR-commandBuffer-cmdpool",
                ],
                ..Default::default()
            }));
        }

        for acs in acceleration_structures {
            // VUID-vkCmdWriteAccelerationStructuresPropertiesKHR-commonparent
            assert_eq!(self.device(), acs.device());

            // VUID-vkCmdWriteAccelerationStructuresPropertiesKHR-pAccelerationStructures-04964
            // unsafe

            // VUID-vkCmdWriteAccelerationStructuresPropertiesKHR-accelerationStructures-03431
            // unsafe
        }

        // VUID-vkCmdWriteAccelerationStructuresPropertiesKHR-commonparent
        assert_eq!(self.device(), query_pool.device());

        // VUID-vkCmdWriteAccelerationStructuresPropertiesKHR-queryPool-02494
        // unsafe

        if first_query as usize + acceleration_structures.len() > query_pool.query_count() as usize
        {
            return Err(Box::new(ValidationError {
                problem: "`first_query` + `acceleration_structures.len()` is greater than \
                    `query_pool.query_count`"
                    .into(),
                vuids: &["VUID-vkCmdWriteAccelerationStructuresPropertiesKHR-query-04880"],
                ..Default::default()
            }));
        }

        if !matches!(
            query_pool.query_type(),
            QueryType::AccelerationStructureSize
                | QueryType::AccelerationStructureSerializationBottomLevelPointers
                | QueryType::AccelerationStructureCompactedSize
                | QueryType::AccelerationStructureSerializationSize,
        ) {
            return Err(Box::new(ValidationError {
                context: "query_pool.query_type()".into(),
                problem: "is not an acceleration structure query".into(),
                vuids: &["VUID-vkCmdWriteAccelerationStructuresPropertiesKHR-queryType-06742"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn write_acceleration_structures_properties_unchecked(
        &mut self,
        acceleration_structures: &[Arc<AccelerationStructure>],
        query_pool: &QueryPool,
        first_query: u32,
    ) -> &mut Self {
        if acceleration_structures.is_empty() {
            return self;
        }

        let acceleration_structures_vk: SmallVec<[_; 4]> = acceleration_structures
            .iter()
            .map(VulkanObject::handle)
            .collect();

        let fns = self.device().fns();
        unsafe {
            (fns.khr_acceleration_structure
                .cmd_write_acceleration_structures_properties_khr)(
                self.handle(),
                acceleration_structures_vk.len() as u32,
                acceleration_structures_vk.as_ptr(),
                query_pool.query_type().into(),
                query_pool.handle(),
                first_query,
            )
        };

        self
    }
}
