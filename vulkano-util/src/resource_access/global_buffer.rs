use vulkano::buffer::Buffer;
use vulkano_taskgraph::{descriptor_set::StorageBufferId, Id};

#[derive(Clone, Copy)]
pub struct GlobalBufferTracker {
    buffer_id: Id<Buffer>,
    storage_buffer_id: StorageBufferId,
}

impl GlobalBufferTracker {
    pub fn new(buffer_id: Id<Buffer>, storage_buffer_id: StorageBufferId) -> Self {
        Self {
            buffer_id,
            storage_buffer_id,
        }
    }

    pub fn id(&self) -> Id<Buffer> {
        self.buffer_id
    }

    pub fn storage(&self) -> StorageBufferId {
        self.storage_buffer_id
    }
}
