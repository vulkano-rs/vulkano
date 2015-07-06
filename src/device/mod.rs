
/// Represents a Vulkan context.
pub struct Device {
    device: ffi::GR_DEVICE,
    queue: ffi::GR_QUEUE,
}

impl Device {
    /// Builds a new Vulkan context for the given GPU.
    pub fn new(gpu: &GpuId) -> Arc<Device> {
        unimplemented!();
    }

    /// Enumerates the list of universal queues. These queues can do anything.
    pub fn universal_queues(&self) -> UniversalQueuesIter {
        unimplemented!();
    }

    /// Enumerates the list of compute queues.
    pub fn compute_queues(&self) -> ComputeQueuesIter {
        unimplemented!();
    }

    /// Enumerates the list of DMA queues. DMA queues can only transfer data.
    pub fn dma_queues(&self) -> DmaQueueIter {
        unimplemented!();
    }
}
