use crate::resource_access::ResourceAccess;
use std::sync::Arc;
use vulkano::{
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    memory::allocator::StandardMemoryAllocator,
};

#[derive(Clone)]
pub struct Allocators {
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl Allocators {
    pub fn new(access: &ResourceAccess) -> Self {
        Self {
            descriptor_set_allocator: Arc::new(StandardDescriptorSetAllocator::new(
                access.device(),
                Default::default(),
            )),
            memory_allocator: Arc::new(StandardMemoryAllocator::new_default(access.device())),
            command_buffer_allocator: Arc::new(StandardCommandBufferAllocator::new(
                access.device(),
                Default::default(),
            )),
        }
    }

    pub fn descriptor_set_allocator(&self) -> Arc<StandardDescriptorSetAllocator> {
        self.descriptor_set_allocator.clone()
    }

    pub fn memory_allocator(&self) -> Arc<StandardMemoryAllocator> {
        self.memory_allocator.clone()
    }

    pub fn command_buffer_allocator(&self) -> Arc<StandardCommandBufferAllocator> {
        self.command_buffer_allocator.clone()
    }
}
