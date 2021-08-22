// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use super::bound::BoundResources;
use super::DescriptorSetError;
use super::MissingBufferUsage;
use super::MissingImageUsage;
use crate::buffer::BufferView;
use crate::descriptor_set::layout::DescriptorDesc;
use crate::descriptor_set::layout::DescriptorDescTy;
use crate::descriptor_set::layout::DescriptorImageDesc;
use crate::descriptor_set::layout::DescriptorImageDescArray;
use crate::descriptor_set::layout::DescriptorImageDescDimensions;
use crate::descriptor_set::layout::DescriptorSetDesc;
use crate::descriptor_set::sys::DescriptorWrite;
use crate::descriptor_set::BufferAccess;
use crate::descriptor_set::DescriptorSetLayout;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::image::ImageViewAbstract;
use crate::image::SampleCount;
use crate::sampler::Sampler;
use crate::VulkanObject;
use std::sync::Arc;

struct BuilderDescriptor {
    desc: Option<DescriptorDesc>,
    array_element: u32,
}

pub struct DescriptorSetBuilder {
    device: Arc<Device>,
    in_array: bool,
    descriptors: Vec<BuilderDescriptor>,
    cur_binding: usize,
    bound_resources: BoundResources,
    desc_writes: Vec<DescriptorWrite>,
}

pub struct DescriptorSetBuilderOutput {
    pub layout: Arc<DescriptorSetLayout>,
    pub writes: Vec<DescriptorWrite>,
    pub bound_resources: BoundResources,
}

impl DescriptorSetBuilder {
    pub fn start(
        layout: Arc<DescriptorSetLayout>,
        runtime_array_capacity: usize,
    ) -> Result<Self, DescriptorSetError> {
        let device = layout.device().clone();
        let mut descriptors = Vec::with_capacity(layout.num_bindings());
        let mut desc_writes_capacity = 0;
        let mut bound_resources_capacity = 0;

        for binding_i in 0..layout.num_bindings() {
            let desc = layout.descriptor(binding_i);

            let array_count = if let Some(desc) = &desc {
                if desc.variable_count {
                    if binding_i != layout.num_bindings() - 1 {
                        return Err(DescriptorSetError::RuntimeArrayMustBeLast);
                    }

                    runtime_array_capacity
                } else {
                    desc.array_count as usize
                }
            } else {
                0
            };

            let resources_per_element = match &desc {
                Some(desc) => match desc.ty {
                    DescriptorDescTy::CombinedImageSampler(_) => 2,
                    _ => array_count,
                },
                None => 0,
            };

            desc_writes_capacity += array_count;
            bound_resources_capacity += resources_per_element * array_count;
            descriptors.push(BuilderDescriptor {
                desc,
                array_element: 0,
            });
        }

        Ok(Self {
            device,
            in_array: false,
            cur_binding: 0,
            descriptors,
            bound_resources: BoundResources::new(bound_resources_capacity),
            desc_writes: Vec::with_capacity(desc_writes_capacity),
        })
    }

    pub fn output(self) -> Result<DescriptorSetBuilderOutput, DescriptorSetError> {
        if self.cur_binding != self.descriptors.len() {
            Err(DescriptorSetError::DescriptorsMissing {
                expected: self.descriptors.len(),
                obtained: self.cur_binding,
            })
        } else {
            let layout = Arc::new(DescriptorSetLayout::new(
                self.device,
                DescriptorSetDesc::new(self.descriptors.into_iter().map(|mut desc| {
                    if let Some(inner_desc) = &mut desc.desc {
                        if inner_desc.variable_count {
                            inner_desc.array_count = desc.array_element;
                        }
                    }

                    desc.desc
                })),
            )?);

            Ok(DescriptorSetBuilderOutput {
                layout,
                writes: self.desc_writes,
                bound_resources: self.bound_resources,
            })
        }
    }

    pub fn enter_array(&mut self) -> Result<(), DescriptorSetError> {
        if self.in_array {
            Err(DescriptorSetError::AlreadyInArray)
        } else if self.cur_binding >= self.descriptors.len() {
            Err(DescriptorSetError::TooManyDescriptors)
        } else if self.descriptors[self.cur_binding].desc.is_none() {
            Err(DescriptorSetError::DescriptorIsEmpty)
        } else {
            self.in_array = true;
            Ok(())
        }
    }

    pub fn leave_array(&mut self) -> Result<(), DescriptorSetError> {
        if !self.in_array {
            Err(DescriptorSetError::NotInArray)
        } else {
            let descriptor = &self.descriptors[self.cur_binding];
            let inner_desc = match descriptor.desc.as_ref() {
                Some(some) => some,
                None => unreachable!(),
            };

            if !inner_desc.variable_count && descriptor.array_element != inner_desc.array_count {
                return Err(DescriptorSetError::ArrayLengthMismatch {
                    expected: inner_desc.array_count,
                    obtained: descriptor.array_element,
                });
            }

            self.in_array = false;
            self.cur_binding += 1;
            Ok(())
        }
    }

    pub fn add_empty(&mut self) -> Result<(), DescriptorSetError> {
        // Should be unreahable as enter_array prevents entering an array for empty descriptors.
        assert!(!self.in_array);

        if self.descriptors[self.cur_binding].desc.is_some() {
            return Err(DescriptorSetError::WrongDescriptorType);
        }

        self.cur_binding += 1;
        Ok(())
    }

    pub fn add_buffer(
        &mut self,
        buffer: Arc<dyn BufferAccess + Send + Sync + 'static>,
    ) -> Result<(), DescriptorSetError> {
        if buffer.inner().buffer.device().internal_object() != self.device.internal_object() {
            return Err(DescriptorSetError::ResourceWrongDevice);
        }

        let leave_array = if !self.in_array {
            self.enter_array()?;
            true
        } else {
            false
        };

        let descriptor = &mut self.descriptors[self.cur_binding];
        let inner_desc = match descriptor.desc.as_ref() {
            Some(some) => some,
            None => return Err(DescriptorSetError::WrongDescriptorType),
        };

        match inner_desc.ty {
            DescriptorDescTy::Buffer(ref buffer_desc) => {
                self.desc_writes.push(if buffer_desc.storage {
                    if !buffer.inner().buffer.usage().storage_buffer {
                        return Err(DescriptorSetError::MissingBufferUsage(
                            MissingBufferUsage::StorageBuffer,
                        ));
                    }

                    unsafe {
                        DescriptorWrite::storage_buffer(
                            self.cur_binding as u32,
                            descriptor.array_element,
                            &buffer,
                        )
                    }
                } else {
                    if !buffer.inner().buffer.usage().uniform_buffer {
                        return Err(DescriptorSetError::MissingBufferUsage(
                            MissingBufferUsage::UniformBuffer,
                        ));
                    }

                    if buffer_desc.dynamic.unwrap_or(false) {
                        unsafe {
                            DescriptorWrite::dynamic_uniform_buffer(
                                self.cur_binding as u32,
                                descriptor.array_element,
                                &buffer,
                            )
                        }
                    } else {
                        unsafe {
                            DescriptorWrite::uniform_buffer(
                                self.cur_binding as u32,
                                descriptor.array_element,
                                &buffer,
                            )
                        }
                    }
                });
            }
            _ => return Err(DescriptorSetError::WrongDescriptorType),
        }

        self.bound_resources
            .add_buffer(self.cur_binding as u32, buffer);
        descriptor.array_element += 1;

        if leave_array {
            self.leave_array()
        } else {
            Ok(())
        }
    }

    pub fn add_buffer_view<B>(&mut self, view: Arc<BufferView<B>>) -> Result<(), DescriptorSetError>
    where
        B: BufferAccess + 'static,
    {
        if view.device().internal_object() != self.device.internal_object() {
            return Err(DescriptorSetError::ResourceWrongDevice);
        }

        let leave_array = if !self.in_array {
            self.enter_array()?;
            true
        } else {
            false
        };

        let descriptor = &mut self.descriptors[self.cur_binding];
        let inner_desc = match descriptor.desc.as_ref() {
            Some(some) => some,
            None => return Err(DescriptorSetError::WrongDescriptorType),
        };

        match inner_desc.ty {
            DescriptorDescTy::TexelBuffer { storage, .. } => {
                if storage {
                    // TODO: storage_texel_buffer_atomic

                    if !view.storage_texel_buffer() {
                        return Err(DescriptorSetError::MissingBufferUsage(
                            MissingBufferUsage::StorageTexelBuffer,
                        ));
                    }

                    self.desc_writes.push(DescriptorWrite::storage_texel_buffer(
                        self.cur_binding as u32,
                        descriptor.array_element,
                        &view,
                    ));
                } else {
                    if !view.uniform_texel_buffer() {
                        return Err(DescriptorSetError::MissingBufferUsage(
                            MissingBufferUsage::UniformTexelBuffer,
                        ));
                    }

                    self.desc_writes.push(DescriptorWrite::uniform_texel_buffer(
                        self.cur_binding as u32,
                        descriptor.array_element,
                        &view,
                    ));
                }
            }
            _ => return Err(DescriptorSetError::WrongDescriptorType),
        }

        self.bound_resources
            .add_buffer_view(self.cur_binding as u32, view);

        if leave_array {
            self.leave_array()
        } else {
            Ok(())
        }
    }

    pub fn add_image(
        &mut self,
        image_view: Arc<dyn ImageViewAbstract + Send + Sync + 'static>,
    ) -> Result<(), DescriptorSetError> {
        if image_view.image().inner().image.device().internal_object()
            != self.device.internal_object()
        {
            return Err(DescriptorSetError::ResourceWrongDevice);
        }

        let leave_array = if !self.in_array {
            self.enter_array()?;
            true
        } else {
            false
        };

        let descriptor = &mut self.descriptors[self.cur_binding];
        let inner_desc = match descriptor.desc.as_ref() {
            Some(some) => some,
            None => return Err(DescriptorSetError::WrongDescriptorType),
        };

        match inner_desc.ty {
            DescriptorDescTy::Image(ref desc) => {
                image_match_desc(&image_view, &desc)?;

                self.desc_writes.push(if desc.sampled {
                    DescriptorWrite::sampled_image(
                        self.cur_binding as u32,
                        descriptor.array_element,
                        &image_view,
                    )
                } else if !image_view.component_mapping().is_identity() {
                    return Err(DescriptorSetError::NotIdentitySwizzled);
                } else {
                    DescriptorWrite::storage_image(
                        self.cur_binding as u32,
                        descriptor.array_element,
                        &image_view,
                    )
                });
            }
            DescriptorDescTy::InputAttachment {
                multisampled,
                array_layers,
            } => {
                if !image_view.image().inner().image.usage().input_attachment {
                    return Err(DescriptorSetError::MissingImageUsage(
                        MissingImageUsage::InputAttachment,
                    ));
                }

                if !image_view.component_mapping().is_identity() {
                    return Err(DescriptorSetError::NotIdentitySwizzled);
                }

                if multisampled && image_view.image().samples() == SampleCount::Sample1 {
                    return Err(DescriptorSetError::ExpectedMultisampled);
                } else if !multisampled && image_view.image().samples() != SampleCount::Sample1 {
                    return Err(DescriptorSetError::UnexpectedMultisampled);
                }

                let image_layers = image_view.array_layers();
                let num_layers = image_layers.end - image_layers.start;

                match array_layers {
                    DescriptorImageDescArray::NonArrayed => {
                        if num_layers != 1 {
                            return Err(DescriptorSetError::ArrayLayersMismatch {
                                expected: 1,
                                obtained: num_layers,
                            });
                        }
                    }
                    DescriptorImageDescArray::Arrayed {
                        max_layers: Some(max_layers),
                    } => {
                        if num_layers > max_layers {
                            // TODO: is this correct? "max" layers? or is it in fact min layers?
                            return Err(DescriptorSetError::ArrayLayersMismatch {
                                expected: max_layers,
                                obtained: num_layers,
                            });
                        }
                    }
                    DescriptorImageDescArray::Arrayed { max_layers: None } => {}
                }

                self.desc_writes.push(DescriptorWrite::input_attachment(
                    self.cur_binding as u32,
                    descriptor.array_element,
                    &image_view,
                ));
            }
            _ => return Err(DescriptorSetError::WrongDescriptorType),
        }

        descriptor.array_element += 1;
        self.bound_resources
            .add_image(self.cur_binding as u32, image_view);

        if leave_array {
            self.leave_array()
        } else {
            Ok(())
        }
    }

    pub fn add_sampled_image(
        &mut self,
        image_view: Arc<dyn ImageViewAbstract + Send + Sync + 'static>,
        sampler: Arc<Sampler>,
    ) -> Result<(), DescriptorSetError> {
        if image_view.image().inner().image.device().internal_object()
            != self.device.internal_object()
        {
            return Err(DescriptorSetError::ResourceWrongDevice);
        }

        if sampler.device().internal_object() != self.device.internal_object() {
            return Err(DescriptorSetError::ResourceWrongDevice);
        }

        if !image_view.can_be_sampled(&sampler) {
            return Err(DescriptorSetError::IncompatibleImageViewSampler);
        }

        let leave_array = if !self.in_array {
            self.enter_array()?;
            true
        } else {
            false
        };

        let descriptor = &mut self.descriptors[self.cur_binding];
        let inner_desc = match descriptor.desc.as_ref() {
            Some(some) => some,
            None => return Err(DescriptorSetError::WrongDescriptorType),
        };

        match inner_desc.ty {
            DescriptorDescTy::CombinedImageSampler(ref desc) => {
                image_match_desc(&image_view, &desc)?;

                self.desc_writes
                    .push(DescriptorWrite::combined_image_sampler(
                        self.cur_binding as u32,
                        descriptor.array_element,
                        &sampler,
                        &image_view,
                    ));
            }
            _ => return Err(DescriptorSetError::WrongDescriptorType),
        }

        descriptor.array_element += 1;
        self.bound_resources
            .add_image(self.cur_binding as u32, image_view);
        self.bound_resources
            .add_sampler(self.cur_binding as u32, sampler);

        if leave_array {
            self.leave_array()
        } else {
            Ok(())
        }
    }

    pub fn add_sampler(&mut self, sampler: Arc<Sampler>) -> Result<(), DescriptorSetError> {
        if sampler.device().internal_object() != self.device.internal_object() {
            return Err(DescriptorSetError::ResourceWrongDevice);
        }

        let leave_array = if !self.in_array {
            self.enter_array()?;
            true
        } else {
            false
        };

        let descriptor = &mut self.descriptors[self.cur_binding];
        let inner_desc = match descriptor.desc.as_ref() {
            Some(some) => some,
            None => return Err(DescriptorSetError::WrongDescriptorType),
        };

        match inner_desc.ty {
            DescriptorDescTy::Sampler => {
                self.desc_writes.push(DescriptorWrite::sampler(
                    self.cur_binding as u32,
                    descriptor.array_element,
                    &sampler,
                ));
            }
            _ => return Err(DescriptorSetError::WrongDescriptorType),
        }

        descriptor.array_element += 1;
        self.bound_resources
            .add_sampler(self.cur_binding as u32, sampler);

        if leave_array {
            self.leave_array()
        } else {
            Ok(())
        }
    }
}

unsafe impl DeviceOwned for DescriptorSetBuilder {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

// Checks whether an image view matches the descriptor.
fn image_match_desc<I>(image_view: &I, desc: &DescriptorImageDesc) -> Result<(), DescriptorSetError>
where
    I: ?Sized + ImageViewAbstract,
{
    if desc.sampled && !image_view.image().inner().image.usage().sampled {
        return Err(DescriptorSetError::MissingImageUsage(
            MissingImageUsage::Sampled,
        ));
    } else if !desc.sampled && !image_view.image().inner().image.usage().storage {
        return Err(DescriptorSetError::MissingImageUsage(
            MissingImageUsage::Storage,
        ));
    }

    let image_view_ty = DescriptorImageDescDimensions::from_image_view_type(image_view.ty());

    if image_view_ty != desc.dimensions {
        return Err(DescriptorSetError::ImageViewTypeMismatch {
            expected: desc.dimensions,
            obtained: image_view_ty,
        });
    }

    if let Some(format) = desc.format {
        if image_view.format() != format {
            return Err(DescriptorSetError::ImageViewFormatMismatch {
                expected: format,
                obtained: image_view.format(),
            });
        }
    }

    if desc.multisampled && image_view.image().samples() == SampleCount::Sample1 {
        return Err(DescriptorSetError::ExpectedMultisampled);
    } else if !desc.multisampled && image_view.image().samples() != SampleCount::Sample1 {
        return Err(DescriptorSetError::UnexpectedMultisampled);
    }

    let image_layers = image_view.array_layers();
    let num_layers = image_layers.end - image_layers.start;

    match desc.array_layers {
        DescriptorImageDescArray::NonArrayed => {
            // TODO: when a non-array is expected, can we pass an image view that is in fact an
            // array with one layer? need to check
            let required_layers = if desc.dimensions == DescriptorImageDescDimensions::Cube {
                6
            } else {
                1
            };

            if num_layers != required_layers {
                return Err(DescriptorSetError::ArrayLayersMismatch {
                    expected: 1,
                    obtained: num_layers,
                });
            }
        }
        DescriptorImageDescArray::Arrayed {
            max_layers: Some(max_layers),
        } => {
            let required_layers = if desc.dimensions == DescriptorImageDescDimensions::Cube {
                max_layers * 6
            } else {
                max_layers
            };

            // TODO: is this correct? "max" layers? or is it in fact min layers?
            if num_layers > required_layers {
                return Err(DescriptorSetError::ArrayLayersMismatch {
                    expected: max_layers,
                    obtained: num_layers,
                });
            }
        }
        DescriptorImageDescArray::Arrayed { max_layers: None } => {}
    };

    Ok(())
}
