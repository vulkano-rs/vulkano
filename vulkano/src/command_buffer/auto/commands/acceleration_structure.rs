use crate::{
    acceleration_structure::{
        AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureGeometries,
        AccelerationStructureGeometryAabbsData, AccelerationStructureGeometryInstancesData,
        AccelerationStructureGeometryInstancesDataType, AccelerationStructureGeometryTrianglesData,
        BuildAccelerationStructureMode, CopyAccelerationStructureInfo,
        CopyAccelerationStructureToMemoryInfo, CopyMemoryToAccelerationStructureInfo,
    },
    buffer::Subbuffer,
    command_buffer::{
        auto::{Resource, ResourceUseRef2},
        sys::RecordingCommandBuffer,
        AutoCommandBufferBuilder, ResourceInCommand,
    },
    query::QueryPool,
    sync::PipelineStageAccessFlags,
    ValidationError,
};
use smallvec::SmallVec;
use std::sync::Arc;

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
        let CopyAccelerationStructureInfo {
            src,
            dst,
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
        let CopyAccelerationStructureToMemoryInfo {
            src,
            dst,
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
        let CopyMemoryToAccelerationStructureInfo {
            src,
            dst,
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
    let AccelerationStructureBuildGeometryInfo {
        flags: _,
        mode,
        dst_acceleration_structure,
        geometries,
        scratch_data,
        _ne: _,
    } = info;

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
                let AccelerationStructureGeometryAabbsData {
                    flags: _,
                    data,
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
            let AccelerationStructureGeometryInstancesData {
                flags: _,
                data,
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
