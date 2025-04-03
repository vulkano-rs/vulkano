use std::sync::Arc;
use vulkano::acceleration_structure::AccelerationStructure;
use vulkano_taskgraph::descriptor_set::AccelerationStructureId;

#[derive(Clone)]
pub struct GlobalAccelerationStructureTracker {
    acceleration_structure: Arc<AccelerationStructure>,
    acceleration_structure_id: AccelerationStructureId,
}

impl GlobalAccelerationStructureTracker {
    pub fn new(
        acceleration_structure: Arc<AccelerationStructure>,
        acceleration_structure_id: AccelerationStructureId,
    ) -> Self {
        Self {
            acceleration_structure,
            acceleration_structure_id,
        }
    }

    pub fn as_data(&self) -> Arc<AccelerationStructure> {
        self.acceleration_structure.clone()
    }

    pub fn as_id(&self) -> AccelerationStructureId {
        self.acceleration_structure_id
    }
}
