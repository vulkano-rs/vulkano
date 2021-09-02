// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::check_errors;
use crate::descriptor_set::layout::DescriptorDesc;
use crate::descriptor_set::layout::DescriptorDescTy;
use crate::descriptor_set::layout::DescriptorSetCompatibilityError;
use crate::descriptor_set::layout::DescriptorSetDesc;
use crate::descriptor_set::pool::DescriptorsCount;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::OomError;
use crate::Version;
use crate::VulkanObject;
use smallvec::SmallVec;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

/// Describes to the Vulkan implementation the layout of all descriptors within a descriptor set.
#[derive(Debug)]
pub struct DescriptorSetLayout {
    // The layout.
    handle: ash::vk::DescriptorSetLayout,
    // The device this layout belongs to.
    device: Arc<Device>,
    // Descriptors.
    desc: DescriptorSetDesc,
    // Number of descriptors.
    descriptors_count: DescriptorsCount,
}

impl DescriptorSetLayout {
    /// Builds a new `DescriptorSetLayout` with the given descriptors.
    ///
    /// The descriptors must be passed in the order of the bindings. In order words, descriptor
    /// at bind point 0 first, then descriptor at bind point 1, and so on. If a binding must remain
    /// empty, you can make the iterator yield `None` for an element.
    pub fn new<D>(
        device: Arc<Device>,
        desc: D,
    ) -> Result<DescriptorSetLayout, DescriptorSetLayoutError>
    where
        D: Into<DescriptorSetDesc>,
    {
        let desc = desc.into();
        let mut descriptors_count = DescriptorsCount::zero();
        let mut variable_descriptor_count = false;
        let bindings = desc.bindings();
        let mut bindings_vk: SmallVec<[_; 32]> = SmallVec::new();

        for (binding, desc) in bindings.iter().enumerate() {
            let desc = match desc {
                Some(d) => d,
                None => continue,
            };

            // FIXME: it is not legal to pass eg. the TESSELLATION_SHADER bit when the device
            //        doesn't have tess shaders enabled

            let ty = desc.ty.ty();
            descriptors_count.add_num(ty, desc.descriptor_count);

            if desc.variable_count {
                if binding != bindings.len() - 1 {
                    return Err(DescriptorSetLayoutError::VariableCountDescMustBeLast);
                }

                if desc.ty == DescriptorDescTy::UniformBufferDynamic
                    || desc.ty == DescriptorDescTy::StorageBufferDynamic
                {
                    return Err(DescriptorSetLayoutError::VariableCountDescMustNotBeDynamic);
                }

                variable_descriptor_count = true;
            }

            bindings_vk.push(ash::vk::DescriptorSetLayoutBinding {
                binding: binding as u32,
                descriptor_type: ty.into(),
                descriptor_count: desc.descriptor_count,
                stage_flags: desc.stages.into(),
                p_immutable_samplers: ptr::null(), // FIXME: not yet implemented
            });
        }

        // Note that it seems legal to have no descriptor at all in the set.

        let handle = unsafe {
            if variable_descriptor_count {
                let enabled_features = device.enabled_features();
                let api_version = device.api_version();
                let enabled_extensions = device.enabled_extensions();

                if api_version.major < 1 {
                    return Err(DescriptorSetLayoutError::VariableCountIncompatibleDevice(
                        IncompatibleDevice::InsufficientApiVersion {
                            required: Version::major_minor(1, 1),
                            obtained: api_version,
                        },
                    ));
                }

                if api_version.minor < 2 && !enabled_extensions.ext_descriptor_indexing {
                    return Err(DescriptorSetLayoutError::VariableCountIncompatibleDevice(
                        IncompatibleDevice::MissingExtension(MissingExtension::DescriptorIndexing),
                    ));
                }

                if !enabled_features.runtime_descriptor_array {
                    return Err(DescriptorSetLayoutError::VariableCountIncompatibleDevice(
                        IncompatibleDevice::MissingFeature(MissingFeature::RuntimeDescriptorArray),
                    ));
                }

                if !enabled_features.descriptor_binding_variable_descriptor_count {
                    return Err(DescriptorSetLayoutError::VariableCountIncompatibleDevice(
                        IncompatibleDevice::MissingFeature(
                            MissingFeature::DescriptorBindingVariableDescriptorCount,
                        ),
                    ));
                }

                if !enabled_features.descriptor_binding_partially_bound {
                    return Err(DescriptorSetLayoutError::VariableCountIncompatibleDevice(
                        IncompatibleDevice::MissingFeature(
                            MissingFeature::DescriptorBindingPartiallyBound,
                        ),
                    ));
                }

                let mut flags = vec![ash::vk::DescriptorBindingFlags::empty(); bindings_vk.len()];
                *flags.last_mut().unwrap() =
                    ash::vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                        | ash::vk::DescriptorBindingFlags::PARTIALLY_BOUND;

                let binding_flags = ash::vk::DescriptorSetLayoutBindingFlagsCreateInfo {
                    binding_count: bindings_vk.len() as u32,
                    p_binding_flags: flags.as_ptr(),
                    ..Default::default()
                };

                let infos = ash::vk::DescriptorSetLayoutCreateInfo {
                    flags: ash::vk::DescriptorSetLayoutCreateFlags::empty(),
                    binding_count: bindings_vk.len() as u32,
                    p_bindings: bindings_vk.as_ptr(),
                    p_next: &binding_flags as *const _ as *const _,
                    ..Default::default()
                };

                let mut output = MaybeUninit::uninit();
                let fns = device.fns();

                check_errors(fns.v1_0.create_descriptor_set_layout(
                    device.internal_object(),
                    &infos,
                    ptr::null(),
                    output.as_mut_ptr(),
                ))
                .map_err(|e| OomError::from(e))?;

                output.assume_init()
            } else {
                let infos = ash::vk::DescriptorSetLayoutCreateInfo {
                    flags: ash::vk::DescriptorSetLayoutCreateFlags::empty(),
                    binding_count: bindings_vk.len() as u32,
                    p_bindings: bindings_vk.as_ptr(),
                    ..Default::default()
                };

                let mut output = MaybeUninit::uninit();
                let fns = device.fns();

                check_errors(fns.v1_0.create_descriptor_set_layout(
                    device.internal_object(),
                    &infos,
                    ptr::null(),
                    output.as_mut_ptr(),
                ))
                .map_err(|e| OomError::from(e))?;

                output.assume_init()
            }
        };

        Ok(DescriptorSetLayout {
            handle,
            device,
            desc,
            descriptors_count,
        })
    }

    pub(crate) fn desc(&self) -> &DescriptorSetDesc {
        &self.desc
    }

    /// Returns the number of descriptors of each type.
    #[inline]
    pub fn descriptors_count(&self) -> &DescriptorsCount {
        &self.descriptors_count
    }

    /// Returns the number of binding slots in the set.
    #[inline]
    pub fn num_bindings(&self) -> u32 {
        self.desc.bindings().len() as u32
    }

    /// Returns a description of a descriptor, or `None` if out of range.
    #[inline]
    pub fn descriptor(&self, binding: u32) -> Option<DescriptorDesc> {
        self.desc
            .bindings()
            .get(binding as usize)
            .cloned()
            .unwrap_or(None)
    }

    /// Returns whether `self` is compatible with `other`.
    ///
    /// "Compatible" in this sense is defined by the Vulkan specification under the section
    /// "Pipeline layout compatibility": either the two are the same descriptor set layout, or they
    /// must be identically defined to the Vulkan API.
    #[inline]
    pub fn is_compatible_with(&self, other: &DescriptorSetLayout) -> bool {
        self.handle == other.handle || self.desc.is_compatible_with(&other.desc)
    }

    /// Checks whether the descriptor of a pipeline layout `self` is compatible with the descriptor
    /// of a descriptor set being bound `other`.
    pub fn ensure_compatible_with_bind(
        &self,
        other: &DescriptorSetLayout,
    ) -> Result<(), DescriptorSetCompatibilityError> {
        if self.handle == other.handle {
            return Ok(());
        }

        self.desc.ensure_compatible_with_bind(&other.desc)
    }
}

unsafe impl DeviceOwned for DescriptorSetLayout {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

unsafe impl VulkanObject for DescriptorSetLayout {
    type Object = ash::vk::DescriptorSetLayout;

    #[inline]
    fn internal_object(&self) -> ash::vk::DescriptorSetLayout {
        self.handle
    }
}

impl Drop for DescriptorSetLayout {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            fns.v1_0.destroy_descriptor_set_layout(
                self.device.internal_object(),
                self.handle,
                ptr::null(),
            );
        }
    }
}

/// Error related to descriptor set layout
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorSetLayoutError {
    /// Out of Memory
    OomError(OomError),

    /// Variable count descriptor must be last binding
    VariableCountDescMustBeLast,

    /// Variable count descriptor must not be a dynamic buffer
    VariableCountDescMustNotBeDynamic,

    /// Device is not compatible with variable count descriptors
    VariableCountIncompatibleDevice(IncompatibleDevice),
}

// Part of the DescriptorSetLayoutError for the case
// of missing features on a device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IncompatibleDevice {
    MissingExtension(MissingExtension),
    MissingFeature(MissingFeature),
    InsufficientApiVersion {
        required: Version,
        obtained: Version,
    },
}

// Part of the IncompatibleDevice for the case
// of missing features on a device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MissingFeature {
    RuntimeDescriptorArray,
    DescriptorBindingVariableDescriptorCount,
    DescriptorBindingPartiallyBound,
}

// Part of the IncompatibleDevice for the case
// of missing extensions on a device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MissingExtension {
    DescriptorIndexing,
}

impl From<OomError> for DescriptorSetLayoutError {
    fn from(error: OomError) -> Self {
        Self::OomError(error)
    }
}

impl std::error::Error for DescriptorSetLayoutError {}

impl std::fmt::Display for DescriptorSetLayoutError {
    #[inline]
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(
            fmt,
            "{}",
            match *self {
                Self::OomError(_) => "out of memory",
                Self::VariableCountDescMustBeLast =>
                    "variable count descriptor must be last binding",
                Self::VariableCountDescMustNotBeDynamic =>
                    "variable count descriptor must not be a dynamic buffer",
                Self::VariableCountIncompatibleDevice(_) =>
                    "device is not compatible with variable count descriptors",
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::descriptor_set::layout::DescriptorDesc;
    use crate::descriptor_set::layout::DescriptorDescTy;
    use crate::descriptor_set::layout::DescriptorSetDesc;
    use crate::descriptor_set::layout::DescriptorSetLayout;
    use crate::descriptor_set::pool::DescriptorsCount;
    use crate::pipeline::shader::ShaderStages;
    use std::iter;

    #[test]
    fn empty() {
        let (device, _) = gfx_dev_and_queue!();
        let _layout = DescriptorSetLayout::new(device, DescriptorSetDesc::empty());
    }

    #[test]
    fn basic_create() {
        let (device, _) = gfx_dev_and_queue!();

        let layout = DescriptorDesc {
            ty: DescriptorDescTy::UniformBuffer,
            descriptor_count: 1,
            stages: ShaderStages::all_graphics(),
            mutable: false,
            variable_count: false,
        };

        let sl = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetDesc::new(iter::once(Some(layout))),
        )
        .unwrap();

        assert_eq!(
            sl.descriptors_count(),
            &DescriptorsCount {
                uniform_buffer: 1,
                ..DescriptorsCount::zero()
            }
        );
    }
}
