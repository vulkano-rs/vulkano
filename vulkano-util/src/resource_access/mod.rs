use std::sync::Arc;
use vulkano::{
    acceleration_structure::{AccelerationStructure, AccelerationStructureCreateInfo},
    buffer::{AllocateBufferError, Buffer, BufferContents, BufferCreateInfo, Subbuffer},
    device::{Device, Queue},
    image::{sampler::SamplerCreateInfo, ImageCreateInfo},
    memory::allocator::{AllocationCreateInfo, DeviceLayout},
    pipeline::{PipelineLayout, PipelineShaderStageCreateInfo},
    Validated, VulkanError,
};
use vulkano_taskgraph::{
    descriptor_set::SamplerId,
    graph::TaskGraph,
    resource::{Flight, HostAccessType, Resources},
    Id,
};

mod global_acceleration_structure;
mod global_buffer;
mod global_image;

pub use global_acceleration_structure::GlobalAccelerationStructureTracker;
pub use global_buffer::GlobalBufferTracker;
pub use global_image::{
    GlobalImageCreateError, GlobalImageCreateInfo, GlobalImageTracker, ModifyImageViewCreateInfo,
};

#[derive(Clone)]
pub struct ResourceAccess {
    resources: Arc<Resources>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    flight_id: Id<Flight>,
}

impl ResourceAccess {
    pub fn new(
        resources: Arc<Resources>,
        device: Arc<Device>,
        queue: Arc<Queue>,
        flight_id: Id<Flight>,
    ) -> ResourceAccess {
        Self {
            resources,
            device,
            queue,
            flight_id,
        }
    }

    pub fn resources(&self) -> Arc<Resources> {
        self.resources.clone()
    }

    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }

    pub fn queue(&self) -> Arc<Queue> {
        self.queue.clone()
    }

    pub fn flight_id(&self) -> Id<Flight> {
        self.flight_id
    }

    pub fn pipeline_layout_from_stages(
        &self,
        stages: &[PipelineShaderStageCreateInfo],
    ) -> Result<Arc<PipelineLayout>, Validated<VulkanError>> {
        let bcx = self.resources.bindless_context().unwrap();

        bcx.pipeline_layout_from_stages(stages)
    }

    pub fn create_global_image<W: 'static + ?Sized>(
        &self,
        task_graph: Option<&mut TaskGraph<W>>,
        image_info: ImageCreateInfo,
        allocation_info: AllocationCreateInfo,
        global_image_create_info: GlobalImageCreateInfo,
    ) -> Result<GlobalImageTracker, Validated<GlobalImageCreateError>> {
        let image_id = self
            .resources
            .create_image(image_info, allocation_info)
            .map_err(|err| err.map(GlobalImageCreateError::AllocateImageError))?;

        GlobalImageTracker::new(
            task_graph,
            &self.resources,
            image_id,
            global_image_create_info,
        )
        .map_err(|err| err.map(GlobalImageCreateError::VulkanError))
    }

    pub fn update_global_image(
        &self,
        global_image_tracker: &mut GlobalImageTracker,
    ) -> Result<(), Validated<VulkanError>> {
        global_image_tracker.update_bindless(&self.resources)
    }

    pub fn create_global_sampler(
        &self,
        sampler_create_info: SamplerCreateInfo,
    ) -> Result<SamplerId, Validated<VulkanError>> {
        let bcx = self.resources.bindless_context().unwrap();

        bcx.global_set().create_sampler(sampler_create_info)
    }

    pub fn create_global_buffer(
        &self,
        buffer_info: BufferCreateInfo,
        allocation_info: AllocationCreateInfo,
        device_layout: DeviceLayout,
    ) -> Result<GlobalBufferTracker, Validated<AllocateBufferError>> {
        let buffer_id =
            self.resources
                .create_buffer(buffer_info, allocation_info, device_layout)?;

        self.add_global_buffer(buffer_id)
    }

    pub fn add_global_buffer(
        &self,
        buffer_id: Id<Buffer>,
    ) -> Result<GlobalBufferTracker, Validated<AllocateBufferError>> {
        let bcx = self.resources.bindless_context().unwrap();

        let storage_buffer_id = bcx
            .global_set()
            .create_storage_buffer(
                buffer_id,
                0,
                self.resources.buffer(buffer_id).unwrap().buffer().size(),
            )
            .unwrap();

        Ok(GlobalBufferTracker::new(buffer_id, storage_buffer_id))
    }

    pub fn create_global_acceleration_structure(
        &self,
        acceleration_structure_create_info: AccelerationStructureCreateInfo,
    ) -> Result<GlobalAccelerationStructureTracker, Validated<VulkanError>> {
        let bcx = self.resources.bindless_context().unwrap();

        let acceleration_structure = unsafe {
            AccelerationStructure::new(self.device.clone(), acceleration_structure_create_info)
        }?;

        let acceleration_structure_id = bcx
            .global_set()
            .add_acceleration_structure(acceleration_structure.clone());

        Ok(GlobalAccelerationStructureTracker::new(
            acceleration_structure,
            acceleration_structure_id,
        ))
    }

    pub fn subbuffer_from_id(&self, buffer_id: Id<Buffer>) -> Subbuffer<[u8]> {
        Subbuffer::new(self.resources.buffer(buffer_id).unwrap().buffer().clone())
    }

    pub fn buffer_from_data<T: BufferContents>(
        &self,
        buffer_info: BufferCreateInfo,
        allocation_info: AllocationCreateInfo,
        data: T,
    ) -> Result<Id<Buffer>, Validated<AllocateBufferError>> {
        let buffer_id = self.resources.create_buffer(
            buffer_info,
            allocation_info,
            DeviceLayout::new_sized::<T>(),
        )?;

        unsafe {
            vulkano_taskgraph::execute(
                &self.queue,
                &self.resources,
                self.flight_id,
                |_builder, task_context| {
                    let write_buffer = task_context.write_buffer::<T>(buffer_id, ..)?;
                    *write_buffer = data;

                    Ok(())
                },
                [(buffer_id, HostAccessType::Write)],
                [],
                [],
            )
        }
        .unwrap();

        let flight = self.resources.flight(self.flight_id).unwrap();
        flight.wait(None).unwrap();

        Ok(buffer_id)
    }

    pub fn buffer_from_slice<T: BufferContents + Copy>(
        &self,
        buffer_info: BufferCreateInfo,
        allocation_info: AllocationCreateInfo,
        iter: &[T],
    ) -> Result<Id<Buffer>, Validated<AllocateBufferError>> {
        let buffer_id = self.resources.create_buffer(
            buffer_info,
            allocation_info,
            DeviceLayout::new_unsized::<[T]>(iter.len() as u64).unwrap(),
        )?;

        unsafe {
            vulkano_taskgraph::execute(
                &self.queue,
                &self.resources,
                self.flight_id,
                |_builder, task_context| {
                    let write_buffer = task_context.write_buffer::<[T]>(buffer_id, ..)?;
                    write_buffer.copy_from_slice(iter);

                    Ok(())
                },
                [(buffer_id, HostAccessType::Write)],
                [],
                [],
            )
        }
        .unwrap();

        let flight = self.resources.flight(self.flight_id).unwrap();
        flight.wait(None).unwrap();

        Ok(buffer_id)
    }
}
