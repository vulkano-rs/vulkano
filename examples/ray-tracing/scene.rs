use crate::{App, RenderContext};
use glam::{Mat4, Vec3};
use std::{iter, mem::size_of, sync::Arc};
use vulkano::{
    acceleration_structure::{
        AccelerationStructure, AccelerationStructureBuildGeometryInfo,
        AccelerationStructureBuildRangeInfo, AccelerationStructureBuildType,
        AccelerationStructureCreateInfo, AccelerationStructureGeometries,
        AccelerationStructureGeometryInstancesData, AccelerationStructureGeometryInstancesDataType,
        AccelerationStructureGeometryTrianglesData, AccelerationStructureInstance,
        AccelerationStructureType, BuildAccelerationStructureFlags, BuildAccelerationStructureMode,
    },
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::CommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryCommandBufferAbstract,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, sys::RawDescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    format::Format,
    image::view::ImageView,
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    pipeline::{
        graphics::vertex_input::Vertex,
        ray_tracing::{
            RayTracingPipeline, RayTracingPipelineCreateInfo, RayTracingShaderGroupCreateInfo,
            ShaderBindingTable,
        },
        PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    swapchain::Swapchain,
    sync::GpuFuture,
};
use vulkano_taskgraph::{
    command_buffer::RecordingCommandBuffer, resource::Resources, Id, Task, TaskContext, TaskResult,
};

mod raygen {
    vulkano_shaders::shader! {
        ty: "raygen",
        path: "raytrace.rgen",
        vulkan_version: "1.2"
    }
}

mod closest_hit {
    vulkano_shaders::shader! {
        ty: "closesthit",
        path: "raytrace.rchit",
        vulkan_version: "1.2"
    }
}

mod miss {
    vulkano_shaders::shader! {
        ty: "miss",
        path: "raytrace.miss",
        vulkan_version: "1.2"
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}

pub struct SceneTask {
    descriptor_set_0: Arc<RawDescriptorSet>,
    swapchain_image_sets: Vec<(Arc<ImageView>, Arc<RawDescriptorSet>)>,
    pipeline_layout: Arc<PipelineLayout>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    virtual_swapchain_id: Id<Swapchain>,
    shader_binding_table: ShaderBindingTable,
    pipeline: Arc<RayTracingPipeline>,
    blas: Arc<AccelerationStructure>,
    tlas: Arc<AccelerationStructure>,
    uniform_buffer: Subbuffer<raygen::Camera>,
}

impl SceneTask {
    pub fn new(
        app: &App,
        pipeline_layout: Arc<PipelineLayout>,
        swapchain_id: Id<Swapchain>,
        virtual_swapchain_id: Id<Swapchain>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        memory_allocator: Arc<dyn MemoryAllocator>,
        command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    ) -> Self {
        let pipeline = {
            let raygen = raygen::load(app.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();
            let closest_hit = closest_hit::load(app.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let miss = miss::load(app.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            // Make a list of the shader stages that the pipeline will have.
            let stages = [
                PipelineShaderStageCreateInfo::new(raygen),
                PipelineShaderStageCreateInfo::new(miss),
                PipelineShaderStageCreateInfo::new(closest_hit),
            ];

            let groups = [
                RayTracingShaderGroupCreateInfo::General { general_shader: 0 },
                RayTracingShaderGroupCreateInfo::General { general_shader: 1 },
                RayTracingShaderGroupCreateInfo::TrianglesHit {
                    closest_hit_shader: Some(2),
                    any_hit_shader: None,
                },
            ];

            RayTracingPipeline::new(
                app.device.clone(),
                None,
                RayTracingPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    groups: groups.into_iter().collect(),
                    max_pipeline_ray_recursion_depth: 1,

                    ..RayTracingPipelineCreateInfo::layout(pipeline_layout.clone())
                },
            )
            .unwrap()
        };

        let vertices = [
            MyVertex {
                position: [-0.5, -0.25, 0.0],
            },
            MyVertex {
                position: [0.0, 0.5, 0.0],
            },
            MyVertex {
                position: [0.25, -0.1, 0.0],
            },
        ];
        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER
                    | BufferUsage::SHADER_DEVICE_ADDRESS
                    | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();

        let blas = unsafe {
            build_acceleration_structure_triangles(
                vertex_buffer,
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                app.device.clone(),
                app.queue.clone(),
            )
        };

        let tlas = unsafe {
            build_top_level_acceleration_structure(
                vec![AccelerationStructureInstance {
                    acceleration_structure_reference: blas.device_address().into(),
                    ..Default::default()
                }],
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                app.device.clone(),
                app.queue.clone(),
            )
        };

        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 4.0 / 3.0, 0.01, 100.0);
        let view = Mat4::look_at_rh(
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        );

        let uniform_buffer = Buffer::from_data(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            raygen::Camera {
                viewInverse: view.inverse().to_cols_array_2d(),
                projInverse: proj.inverse().to_cols_array_2d(),
                viewProj: (proj * view).to_cols_array_2d(),
            },
        )
        .unwrap();

        let descriptor_set_0 = RawDescriptorSet::new(
            descriptor_set_allocator.clone(),
            &pipeline_layout.set_layouts()[0],
            0,
        )
        .unwrap();

        unsafe {
            let writes = &[
                WriteDescriptorSet::acceleration_structure(0, tlas.clone()),
                WriteDescriptorSet::buffer(1, uniform_buffer.clone()),
            ];
            descriptor_set_0.update(writes, &[]).unwrap();
        }

        let swapchain_image_sets = window_size_dependent_setup(
            &app.resources,
            swapchain_id,
            &pipeline_layout,
            &descriptor_set_allocator,
        );

        let shader_binding_table =
            ShaderBindingTable::new(memory_allocator.clone(), &pipeline).unwrap();

        SceneTask {
            descriptor_set_0: Arc::new(descriptor_set_0),
            swapchain_image_sets,
            descriptor_set_allocator,
            pipeline_layout,
            virtual_swapchain_id,
            shader_binding_table,
            pipeline,
            blas,
            tlas,
            uniform_buffer,
        }
    }

    pub fn handle_resize(&mut self, resources: &Resources, swapchain_id: Id<Swapchain>) {
        self.swapchain_image_sets = window_size_dependent_setup(
            resources,
            swapchain_id,
            &self.pipeline_layout,
            &self.descriptor_set_allocator,
        );
    }
}

impl Task for SceneTask {
    type World = RenderContext;

    unsafe fn execute(
        &self,
        cbf: &mut RecordingCommandBuffer<'_>,
        tcx: &mut TaskContext<'_>,
        _rcx: &Self::World,
    ) -> TaskResult {
        let swapchain_state = tcx.swapchain(self.virtual_swapchain_id)?;
        let image_index = swapchain_state.current_image_index().unwrap();

        cbf.as_raw().bind_descriptor_sets(
            PipelineBindPoint::RayTracing,
            &self.pipeline_layout,
            0,
            &[
                &self.descriptor_set_0,
                &self.swapchain_image_sets[image_index as usize].1,
            ],
            &[],
        )?;

        cbf.bind_pipeline_ray_tracing(&self.pipeline)?;

        let extent = self.swapchain_image_sets[0].0.image().extent();

        unsafe {
            cbf.trace_rays(
                self.shader_binding_table.addresses(),
                extent[0],
                extent[1],
                1,
            )
        }?;

        for (image_view, descriptor_set) in self.swapchain_image_sets.iter() {
            cbf.destroy_object(descriptor_set.clone());
            cbf.destroy_object(image_view.clone());
        }
        cbf.destroy_object(self.blas.clone());
        cbf.destroy_object(self.tlas.clone());
        cbf.destroy_object(self.uniform_buffer.clone().into());
        cbf.destroy_object(self.descriptor_set_0.clone());

        Ok(())
    }
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    resources: &Resources,
    swapchain_id: Id<Swapchain>,
    pipeline_layout: &Arc<PipelineLayout>,
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
) -> Vec<(Arc<ImageView>, Arc<RawDescriptorSet>)> {
    let swapchain_state = resources.swapchain(swapchain_id).unwrap();
    let images = swapchain_state.images();

    let swapchain_image_sets = images
        .iter()
        .map(|image| {
            let descriptor_set = RawDescriptorSet::new(
                descriptor_set_allocator.clone(),
                &pipeline_layout.set_layouts()[1],
                0,
            )
            .unwrap();
            let image_view = ImageView::new_default(image.clone()).unwrap();
            let writes = &[WriteDescriptorSet::image_view(0, image_view.clone())];
            unsafe { descriptor_set.update(writes, &[]) }.unwrap();
            (image_view, Arc::new(descriptor_set))
        })
        .collect();

    swapchain_image_sets
}

unsafe fn build_acceleration_structure_common(
    geometries: AccelerationStructureGeometries,
    primitive_count: u32,
    ty: AccelerationStructureType,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
) -> Arc<AccelerationStructure> {
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let mut as_build_geometry_info = AccelerationStructureBuildGeometryInfo {
        mode: BuildAccelerationStructureMode::Build,
        flags: BuildAccelerationStructureFlags::PREFER_FAST_TRACE,
        ..AccelerationStructureBuildGeometryInfo::new(geometries)
    };

    let as_build_sizes_info = device
        .acceleration_structure_build_sizes(
            AccelerationStructureBuildType::Device,
            &as_build_geometry_info,
            &[primitive_count],
        )
        .unwrap();

    let scratch_buffer = Buffer::new_slice::<u8>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS | BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
        as_build_sizes_info.build_scratch_size,
    )
    .unwrap();

    let as_create_info = AccelerationStructureCreateInfo {
        ty,
        ..AccelerationStructureCreateInfo::new(
            Buffer::new_slice::<u8>(
                memory_allocator,
                BufferCreateInfo {
                    usage: BufferUsage::ACCELERATION_STRUCTURE_STORAGE
                        | BufferUsage::SHADER_DEVICE_ADDRESS,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
                as_build_sizes_info.acceleration_structure_size,
            )
            .unwrap(),
        )
    };

    let acceleration = unsafe { AccelerationStructure::new(device, as_create_info).unwrap() };

    as_build_geometry_info.dst_acceleration_structure = Some(acceleration.clone());
    as_build_geometry_info.scratch_data = Some(scratch_buffer);

    let as_build_range_info = AccelerationStructureBuildRangeInfo {
        primitive_count,
        ..Default::default()
    };

    builder
        .build_acceleration_structure(
            as_build_geometry_info,
            iter::once(as_build_range_info).collect(),
        )
        .unwrap();

    builder
        .build()
        .unwrap()
        .execute(queue)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    acceleration
}

unsafe fn build_acceleration_structure_triangles(
    vertex_buffer: Subbuffer<[MyVertex]>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
) -> Arc<AccelerationStructure> {
    let primitive_count = (vertex_buffer.len() / 3) as u32;
    let as_geometry_triangles_data = AccelerationStructureGeometryTrianglesData {
        max_vertex: vertex_buffer.len() as _,
        vertex_data: Some(vertex_buffer.into_bytes()),
        vertex_stride: size_of::<MyVertex>() as _,
        ..AccelerationStructureGeometryTrianglesData::new(Format::R32G32B32_SFLOAT)
    };

    let geometries = AccelerationStructureGeometries::Triangles(vec![as_geometry_triangles_data]);

    build_acceleration_structure_common(
        geometries,
        primitive_count,
        AccelerationStructureType::BottomLevel,
        memory_allocator,
        command_buffer_allocator,
        device,
        queue,
    )
}

unsafe fn build_top_level_acceleration_structure(
    as_instances: Vec<AccelerationStructureInstance>,
    allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
) -> Arc<AccelerationStructure> {
    let primitive_count = as_instances.len() as u32;

    let instance_buffer = Buffer::from_iter(
        allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::SHADER_DEVICE_ADDRESS
                | BufferUsage::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        as_instances,
    )
    .unwrap();

    let as_geometry_instances_data = AccelerationStructureGeometryInstancesData::new(
        AccelerationStructureGeometryInstancesDataType::Values(Some(instance_buffer)),
    );

    let geometries = AccelerationStructureGeometries::Instances(as_geometry_instances_data);

    build_acceleration_structure_common(
        geometries,
        primitive_count,
        AccelerationStructureType::TopLevel,
        allocator,
        command_buffer_allocator,
        device,
        queue,
    )
}
