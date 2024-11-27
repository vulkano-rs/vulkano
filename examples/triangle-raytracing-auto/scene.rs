use std::sync::Arc;
use std::{iter, mem::size_of};

use crate::App;
use glam::{Mat4, Vec3};
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::descriptor_set::DescriptorSet;
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
    descriptor_set::{allocator::StandardDescriptorSetAllocator, WriteDescriptorSet},
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, Image},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    pipeline::{
        graphics::vertex_input::Vertex,
        ray_tracing::{
            RayTracingPipeline, RayTracingPipelineCreateInfo, RayTracingShaderGroupCreateInfo,
            ShaderBindingTable,
        },
        PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    sync::GpuFuture,
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
        path: "raytrace.rmiss",
        vulkan_version: "1.2"
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}

pub struct Scene {
    descriptor_set_0: Arc<DescriptorSet>,
    swapchain_image_sets: Vec<(Arc<ImageView>, Arc<DescriptorSet>)>,
    pipeline_layout: Arc<PipelineLayout>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    shader_binding_table: ShaderBindingTable,
    pipeline: Arc<RayTracingPipeline>,
    // The bottom-level acceleration structure is required to be kept alive
    // as we reference it in the top-level acceleration structure.
    _blas: Arc<AccelerationStructure>,
    _tlas: Arc<AccelerationStructure>,
}

impl Scene {
    pub fn new(
        app: &App,
        images: &[Arc<Image>],
        pipeline_layout: Arc<PipelineLayout>,
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

            // Define the shader groups that will eventually turn into the shader binding table.
            // The numbers are the indices of the stages in the `stages` array.
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

        // Build the bottom-level acceleration structure and then the top-level acceleration structure.
        // Acceleration structures are used to accelerate ray tracing.
        // The bottom-level acceleration structure contains the geometry data.
        // The top-level acceleration structure contains the instances of the bottom-level acceleration structures.
        // In our shader, we will trace rays against the top-level acceleration structure.
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
                blas.clone(),
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                app.device.clone(),
                app.queue.clone(),
            )
        };

        let proj = Mat4::perspective_rh_gl(std::f32::consts::FRAC_PI_2, 4.0 / 3.0, 0.01, 100.0);
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

        let descriptor_set_0 = DescriptorSet::new(
            descriptor_set_allocator.clone(),
            pipeline_layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::acceleration_structure(0, tlas.clone()),
                WriteDescriptorSet::buffer(1, uniform_buffer.clone()),
            ],
            [],
        )
        .unwrap();

        let swapchain_image_sets =
            window_size_dependent_setup(images, &pipeline_layout, &descriptor_set_allocator);

        let shader_binding_table =
            ShaderBindingTable::new(memory_allocator.clone(), &pipeline).unwrap();

        Scene {
            descriptor_set_0,
            swapchain_image_sets,
            descriptor_set_allocator,
            pipeline_layout,
            shader_binding_table,
            pipeline,
            _blas: blas,
            _tlas: tlas,
        }
    }

    pub fn handle_resize(&mut self, images: &[Arc<Image>]) {
        self.swapchain_image_sets = window_size_dependent_setup(
            images,
            &self.pipeline_layout,
            &self.descriptor_set_allocator,
        );
    }

    pub fn record_commands(
        &self,
        image_index: u32,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) {
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::RayTracing,
                self.pipeline_layout.clone(),
                0,
                vec![
                    self.descriptor_set_0.clone(),
                    self.swapchain_image_sets[image_index as usize].1.clone(),
                ],
            )
            .unwrap();

        builder
            .bind_pipeline_ray_tracing(self.pipeline.clone())
            .unwrap();

        let extent = self.swapchain_image_sets[0].0.image().extent();

        unsafe {
            builder
                .trace_rays(
                    self.shader_binding_table.addresses().clone(),
                    extent[0],
                    extent[1],
                    1,
                )
                .unwrap();
        }
    }
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    images: &[Arc<Image>],
    pipeline_layout: &Arc<PipelineLayout>,
    descriptor_set_allocator: &Arc<StandardDescriptorSetAllocator>,
) -> Vec<(Arc<ImageView>, Arc<DescriptorSet>)> {
    let swapchain_image_sets = images
        .iter()
        .map(|image| {
            let image_view = ImageView::new_default(image.clone()).unwrap();
            let descriptor_set = DescriptorSet::new(
                descriptor_set_allocator.clone(),
                pipeline_layout.set_layouts()[1].clone(),
                [WriteDescriptorSet::image_view(0, image_view.clone())],
                [],
            )
            .unwrap();
            (image_view, descriptor_set)
        })
        .collect();

    swapchain_image_sets
}

/// A helper function to build a acceleration structure and wait for its completion.
/// # SAFETY
/// - If you are referencing a bottom-level acceleration structure in a top-level acceleration structure,
/// you must ensure that the bottom-level acceleration structure is kept alive.
unsafe fn build_acceleration_structure_common(
    geometries: AccelerationStructureGeometries,
    primitive_count: u32,
    ty: AccelerationStructureType,
    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
) -> Arc<AccelerationStructure> {
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

    // We build a new scratch buffer for each acceleration structure for simplicity.
    // You may want to reuse scratch buffers if you need to build many acceleration structures.
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

    // For simplicity, we build a single command buffer
    // that builds the acceleration structure, then waits
    // for its execution to complete.
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

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
    acceleration_structure: Arc<AccelerationStructure>,
    allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
) -> Arc<AccelerationStructure> {
    let as_instance = AccelerationStructureInstance {
        acceleration_structure_reference: acceleration_structure.device_address().into(),
        ..Default::default()
    };

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
        [as_instance],
    )
    .unwrap();

    let as_geometry_instances_data = AccelerationStructureGeometryInstancesData::new(
        AccelerationStructureGeometryInstancesDataType::Values(Some(instance_buffer)),
    );

    let geometries = AccelerationStructureGeometries::Instances(as_geometry_instances_data);

    build_acceleration_structure_common(
        geometries,
        1,
        AccelerationStructureType::TopLevel,
        allocator,
        command_buffer_allocator,
        device,
        queue,
    )
}
