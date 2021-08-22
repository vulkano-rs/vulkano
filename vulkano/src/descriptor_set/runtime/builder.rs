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
use crate::descriptor_set::layout::DescriptorDescImage;
use crate::descriptor_set::layout::DescriptorDescTy;
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
        let mut t_num_bufs = 0;
        let mut t_num_imgs = 0;
        let mut t_num_samplers = 0;

        for binding_i in 0..layout.num_bindings() {
            let desc = layout.descriptor(binding_i);

            let descriptor_count = if let Some(desc) = &desc {
                if desc.variable_count {
                    if binding_i != layout.num_bindings() - 1 {
                        return Err(DescriptorSetError::RuntimeArrayMustBeLast);
                    }

                    runtime_array_capacity
                } else {
                    desc.descriptor_count as usize
                }
            } else {
                0
            };

            let (num_bufs, num_imgs, num_samplers) = match &desc {
                Some(desc) => match desc.ty {
                    DescriptorDescTy::Sampler => (0, 0, 1),
                    DescriptorDescTy::CombinedImageSampler(_) => (0, 1, 1),
                    DescriptorDescTy::SampledImage(_) => (0, 1, 0),
                    DescriptorDescTy::StorageImage(_) => (0, 1, 0),
                    DescriptorDescTy::UniformTexelBuffer { .. } => (1, 0, 0),
                    DescriptorDescTy::StorageTexelBuffer { .. } => (1, 0, 0),
                    DescriptorDescTy::UniformBuffer => (1, 0, 0),
                    DescriptorDescTy::StorageBuffer => (1, 0, 0),
                    DescriptorDescTy::UniformBufferDynamic => (1, 0, 0),
                    DescriptorDescTy::StorageBufferDynamic => (1, 0, 0),
                    DescriptorDescTy::InputAttachment { .. } => (0, 1, 0),
                },
                None => (0, 0, 0),
            };

            t_num_bufs += num_bufs * descriptor_count;
            t_num_imgs += num_imgs * descriptor_count;
            t_num_samplers += num_samplers * descriptor_count;
            desc_writes_capacity += descriptor_count;

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
            bound_resources: BoundResources::new(t_num_bufs, t_num_imgs, t_num_samplers),
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
                            inner_desc.descriptor_count = desc.array_element;
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

            if !inner_desc.variable_count && descriptor.array_element != inner_desc.descriptor_count
            {
                return Err(DescriptorSetError::ArrayLengthMismatch {
                    expected: inner_desc.descriptor_count,
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

        // Note that the buffer content is not checked. This is technically not unsafe as
        // long as the data in the buffer has no invalid memory representation (ie. no
        // bool, no enum, no pointer, no str) and as long as the robust buffer access
        // feature is enabled.
        // TODO: this is not checked ^

        // TODO: eventually shouldn't be an assert ; for now robust_buffer_access is always
        //       enabled so this assert should never fail in practice, but we put it anyway
        //       in case we forget to adjust this code

        self.desc_writes.push(match inner_desc.ty {
            DescriptorDescTy::StorageBuffer | DescriptorDescTy::StorageBufferDynamic => {
                assert!(self.device.enabled_features().robust_buffer_access);

                if buffer.inner().buffer.usage().storage_buffer {
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
            }
            DescriptorDescTy::UniformBuffer => {
                assert!(self.device.enabled_features().robust_buffer_access);

                if !buffer.inner().buffer.usage().uniform_buffer {
                    return Err(DescriptorSetError::MissingBufferUsage(
                        MissingBufferUsage::UniformBuffer,
                    ));
                }

                unsafe {
                    DescriptorWrite::uniform_buffer(
                        self.cur_binding as u32,
                        descriptor.array_element,
                        &buffer,
                    )
                }
            }
            DescriptorDescTy::UniformBufferDynamic => {
                assert!(self.device.enabled_features().robust_buffer_access);

                if !buffer.inner().buffer.usage().uniform_buffer {
                    return Err(DescriptorSetError::MissingBufferUsage(
                        MissingBufferUsage::UniformBuffer,
                    ));
                }

                unsafe {
                    DescriptorWrite::dynamic_uniform_buffer(
                        self.cur_binding as u32,
                        descriptor.array_element,
                        &buffer,
                    )
                }
            }
            _ => return Err(DescriptorSetError::WrongDescriptorType),
        });

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

        self.desc_writes.push(match inner_desc.ty {
            DescriptorDescTy::StorageTexelBuffer { .. } => {
                // TODO: storage_texel_buffer_atomic

                if !view.storage_texel_buffer() {
                    return Err(DescriptorSetError::MissingBufferUsage(
                        MissingBufferUsage::StorageTexelBuffer,
                    ));
                }

                DescriptorWrite::storage_texel_buffer(
                    self.cur_binding as u32,
                    descriptor.array_element,
                    &view,
                )
            }
            DescriptorDescTy::UniformTexelBuffer { .. } => {
                if !view.uniform_texel_buffer() {
                    return Err(DescriptorSetError::MissingBufferUsage(
                        MissingBufferUsage::UniformTexelBuffer,
                    ));
                }

                DescriptorWrite::uniform_texel_buffer(
                    self.cur_binding as u32,
                    descriptor.array_element,
                    &view,
                )
            }
            _ => return Err(DescriptorSetError::WrongDescriptorType),
        });

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

        self.desc_writes.push(match inner_desc.ty {
            DescriptorDescTy::SampledImage(ref desc) => {
                if !image_view.image().inner().image.usage().sampled {
                    return Err(DescriptorSetError::MissingImageUsage(
                        MissingImageUsage::Sampled,
                    ));
                }

                image_match_desc(&image_view, &desc)?;

                DescriptorWrite::sampled_image(
                    self.cur_binding as u32,
                    descriptor.array_element,
                    &image_view,
                )
            }
            DescriptorDescTy::StorageImage(ref desc) => {
                if !image_view.image().inner().image.usage().storage {
                    return Err(DescriptorSetError::MissingImageUsage(
                        MissingImageUsage::Storage,
                    ));
                }

                image_match_desc(&image_view, &desc)?;

                if !image_view.component_mapping().is_identity() {
                    return Err(DescriptorSetError::NotIdentitySwizzled);
                }

                DescriptorWrite::storage_image(
                    self.cur_binding as u32,
                    descriptor.array_element,
                    &image_view,
                )
            }
            DescriptorDescTy::InputAttachment { multisampled } => {
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

                if image_view.ty().is_arrayed() {
                    return Err(DescriptorSetError::UnexpectedArrayed);
                }

                DescriptorWrite::input_attachment(
                    self.cur_binding as u32,
                    descriptor.array_element,
                    &image_view,
                )
            }
            _ => return Err(DescriptorSetError::WrongDescriptorType),
        });

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

        if !image_view.image().inner().image.usage().sampled {
            return Err(DescriptorSetError::MissingImageUsage(
                MissingImageUsage::Sampled,
            ));
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

        self.desc_writes.push(match inner_desc.ty {
            DescriptorDescTy::CombinedImageSampler(ref desc) => {
                image_match_desc(&image_view, &desc)?;

                DescriptorWrite::combined_image_sampler(
                    self.cur_binding as u32,
                    descriptor.array_element,
                    &sampler,
                    &image_view,
                )
            }
            _ => return Err(DescriptorSetError::WrongDescriptorType),
        });

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

        self.desc_writes.push(match inner_desc.ty {
            DescriptorDescTy::Sampler => DescriptorWrite::sampler(
                self.cur_binding as u32,
                descriptor.array_element,
                &sampler,
            ),
            _ => return Err(DescriptorSetError::WrongDescriptorType),
        });

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
fn image_match_desc<I>(image_view: &I, desc: &DescriptorDescImage) -> Result<(), DescriptorSetError>
where
    I: ?Sized + ImageViewAbstract,
{
    if image_view.ty() != desc.view_type {
        return Err(DescriptorSetError::ImageViewTypeMismatch {
            expected: desc.view_type,
            obtained: image_view.ty(),
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

    Ok(())
}
