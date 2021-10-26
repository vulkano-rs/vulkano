// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::buffer::BufferViewAbstract;
use crate::descriptor_set::layout::{DescriptorSetLayout, DescriptorType};
use crate::descriptor_set::sys::{DescriptorWrite, DescriptorWriteElements};
use crate::descriptor_set::BufferAccess;
use crate::image::ImageViewAbstract;
use crate::sampler::Sampler;
use fnv::FnvHashMap;
use smallvec::{smallvec, SmallVec};
use std::sync::Arc;

/// The resources that are bound to a descriptor set.
#[derive(Clone)]
pub struct DescriptorSetResources {
    descriptors: FnvHashMap<u32, DescriptorBindingResources>,
    buffers: Vec<(u32, usize)>,
    images: Vec<(u32, usize)>,
}

impl DescriptorSetResources {
    /// Creates a new `DescriptorSetResources` matching the provided descriptor set layout, and
    /// all descriptors set to `None`.
    pub fn new(layout: &DescriptorSetLayout) -> Self {
        let descriptors = layout
            .desc()
            .bindings()
            .iter()
            .enumerate()
            .filter_map(|(b, d)| d.as_ref().map(|d| (b as u32, d)))
            .map(|(binding_num, binding_desc)| {
                let count = binding_desc.descriptor_count as usize;
                let binding_resources = match binding_desc.ty.ty() {
                    DescriptorType::UniformBuffer
                    | DescriptorType::StorageBuffer
                    | DescriptorType::UniformBufferDynamic
                    | DescriptorType::StorageBufferDynamic => {
                        DescriptorBindingResources::Buffer(smallvec![None; count])
                    }
                    DescriptorType::UniformTexelBuffer | DescriptorType::StorageTexelBuffer => {
                        DescriptorBindingResources::BufferView(smallvec![None; count])
                    }
                    DescriptorType::SampledImage
                    | DescriptorType::StorageImage
                    | DescriptorType::InputAttachment => {
                        DescriptorBindingResources::ImageView(smallvec![None; count])
                    }
                    DescriptorType::CombinedImageSampler => {
                        if binding_desc.ty.immutable_samplers().is_empty() {
                            DescriptorBindingResources::ImageViewSampler(smallvec![None; count])
                        } else {
                            DescriptorBindingResources::ImageView(smallvec![None; count])
                        }
                    }
                    DescriptorType::Sampler => {
                        if binding_desc.ty.immutable_samplers().is_empty() {
                            DescriptorBindingResources::Sampler(smallvec![None; count])
                        } else {
                            DescriptorBindingResources::None
                        }
                    }
                };
                (binding_num, binding_resources)
            })
            .collect();

        Self {
            descriptors,
            buffers: Vec::new(),
            images: Vec::new(),
        }
    }

    /// Applies descriptor writes to the resources.
    ///
    /// # Panics
    ///
    /// - Panics if the binding number of a write does not exist in the resources.
    /// - See also [`DescriptorBindingResources::update`].
    pub fn update<'a>(&mut self, writes: impl IntoIterator<Item = &'a DescriptorWrite>) {
        for write in writes {
            self.descriptors
                .get_mut(&write.binding_num)
                .expect("descriptor write has invalid binding number")
                .update(write)
        }

        self.buffers.clear();
        self.images.clear();

        for (&binding, resources) in self.descriptors.iter() {
            match resources {
                DescriptorBindingResources::None => (),
                DescriptorBindingResources::Buffer(resources) => {
                    self.buffers.extend(resources.iter().enumerate().filter_map(
                        |(index, resource)| resource.as_ref().map(|_| (binding, index)),
                    ))
                }
                DescriptorBindingResources::BufferView(resources) => {
                    self.buffers.extend(resources.iter().enumerate().filter_map(
                        |(index, resource)| resource.as_ref().map(|_| (binding, index)),
                    ))
                }
                DescriptorBindingResources::ImageView(resources) => {
                    self.images.extend(resources.iter().enumerate().filter_map(
                        |(index, resource)| resource.as_ref().map(|_| (binding, index)),
                    ))
                }
                DescriptorBindingResources::ImageViewSampler(resources) => {
                    self.images.extend(resources.iter().enumerate().filter_map(
                        |(index, resource)| resource.as_ref().map(|_| (binding, index)),
                    ))
                }
                DescriptorBindingResources::Sampler(_) => (),
            }
        }
    }

    /// Returns a reference to the bound resources for `binding`. Returns `None` if the binding
    /// doesn't exist.
    #[inline]
    pub fn binding(&self, binding: u32) -> Option<&DescriptorBindingResources> {
        self.descriptors.get(&binding)
    }

    pub(crate) fn num_buffers(&self) -> usize {
        self.buffers.len()
    }

    pub(crate) fn buffer(&self, index: usize) -> Option<(Arc<dyn BufferAccess>, u32)> {
        self.buffers
            .get(index)
            .and_then(|&(binding, index)| match &self.descriptors[&binding] {
                DescriptorBindingResources::Buffer(resources) => {
                    resources[index].as_ref().map(|r| (r.clone(), binding))
                }
                DescriptorBindingResources::BufferView(resources) => {
                    resources[index].as_ref().map(|r| (r.buffer(), binding))
                }
                _ => unreachable!(),
            })
    }

    pub(crate) fn num_images(&self) -> usize {
        self.images.len()
    }

    pub(crate) fn image(&self, index: usize) -> Option<(Arc<dyn ImageViewAbstract>, u32)> {
        self.images
            .get(index)
            .and_then(|&(binding, index)| match &self.descriptors[&binding] {
                DescriptorBindingResources::ImageView(resources) => {
                    resources[index].as_ref().map(|r| (r.clone(), binding))
                }
                DescriptorBindingResources::ImageViewSampler(resources) => {
                    resources[index].as_ref().map(|r| (r.0.clone(), binding))
                }
                _ => unreachable!(),
            })
    }
}

/// The resources that are bound to a single descriptor set binding.
#[derive(Clone)]
pub enum DescriptorBindingResources {
    None,
    Buffer(Elements<Arc<dyn BufferAccess>>),
    BufferView(Elements<Arc<dyn BufferViewAbstract>>),
    ImageView(Elements<Arc<dyn ImageViewAbstract>>),
    ImageViewSampler(Elements<(Arc<dyn ImageViewAbstract>, Arc<Sampler>)>),
    Sampler(Elements<Arc<Sampler>>),
}

type Elements<T> = SmallVec<[Option<T>; 1]>;

impl DescriptorBindingResources {
    /// Applies a descriptor write to the resources.
    ///
    /// # Panics
    ///
    /// - Panics if the resource types do not match.
    /// - Panics if the write goes out of bounds.
    pub fn update(&mut self, write: &DescriptorWrite) {
        fn write_resources<T: Clone>(first: usize, resources: &mut [Option<T>], elements: &[T]) {
            resources
                .get_mut(first..first + elements.len())
                .expect("descriptor write for binding out of bounds")
                .iter_mut()
                .zip(elements)
                .for_each(|(resource, element)| {
                    *resource = Some(element.clone());
                });
        }

        let first = write.first_array_element() as usize;

        match (self, write.elements()) {
            (
                DescriptorBindingResources::Buffer(resources),
                DescriptorWriteElements::Buffer(elements),
            ) => write_resources(first, resources, elements),
            (
                DescriptorBindingResources::BufferView(resources),
                DescriptorWriteElements::BufferView(elements),
            ) => write_resources(first, resources, elements),
            (
                DescriptorBindingResources::ImageView(resources),
                DescriptorWriteElements::ImageView(elements),
            ) => write_resources(first, resources, elements),
            (
                DescriptorBindingResources::ImageViewSampler(resources),
                DescriptorWriteElements::ImageViewSampler(elements),
            ) => write_resources(first, resources, elements),
            (
                DescriptorBindingResources::Sampler(resources),
                DescriptorWriteElements::Sampler(elements),
            ) => write_resources(first, resources, elements),
            _ => panic!(
                "descriptor write for binding {} has wrong resource type",
                write.binding_num,
            ),
        }
    }
}
