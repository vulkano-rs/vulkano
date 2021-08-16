use super::builder::RuntimeDescriptorSetBuilder;
use super::RuntimeDescriptorSetError;
use crate::descriptor_set::DescriptorSetLayout;
use std::sync::Arc;

pub struct RuntimePersistentDescriptorSet {}

impl RuntimePersistentDescriptorSet {
    pub fn start(
        layout: Arc<DescriptorSetLayout>,
    ) -> Result<RuntimeDescriptorSetBuilder, RuntimeDescriptorSetError> {
        RuntimeDescriptorSetBuilder::with_rt_desc_capacity(layout, 0)
    }

    pub fn start_with_rt_desc_capacity(
        layout: Arc<DescriptorSetLayout>,
        capacity: usize,
    ) -> Result<RuntimeDescriptorSetBuilder, RuntimeDescriptorSetError> {
        RuntimeDescriptorSetBuilder::with_rt_desc_capacity(layout, capacity)
    }
}
