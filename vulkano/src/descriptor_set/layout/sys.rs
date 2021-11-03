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
use crate::descriptor_set::layout::DescriptorSetDesc;
use crate::descriptor_set::layout::DescriptorType;
use crate::descriptor_set::pool::DescriptorsCount;
use crate::device::Device;
use crate::device::DeviceOwned;
use crate::OomError;
use crate::Version;
use crate::VulkanObject;
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
    // Number of descriptors of each type.
    descriptors_count: DescriptorsCount,
    // Number of descriptors in a variable count descriptor. Will be zero if no variable count descriptors are present.
    variable_descriptor_count: u32,
}

impl DescriptorSetLayout {
    /// Builds a new `DescriptorSetLayout` with the given descriptors.
    ///
    /// The descriptors must be passed in the order of the bindings. In order words, descriptor
    /// at bind point 0 first, then descriptor at bind point 1, and so on. If a binding must remain
    /// empty, you can make the iterator yield `None` for an element.
    pub fn new<D>(
        device: Arc<Device>,
        set_desc: D,
    ) -> Result<Arc<DescriptorSetLayout>, DescriptorSetLayoutError>
    where
        D: Into<DescriptorSetDesc>,
    {
        let set_desc = set_desc.into();
        let mut descriptors_count = DescriptorsCount::zero();
        let mut variable_descriptor_count = 0;
        let bindings = set_desc.bindings();
        let mut bindings_vk = Vec::with_capacity(bindings.len());
        let mut binding_flags_vk = Vec::with_capacity(bindings.len());
        let mut immutable_samplers_vk: Vec<Box<[ash::vk::Sampler]>> = Vec::new(); // only to keep the arrays of handles alive

        let mut flags = ash::vk::DescriptorSetLayoutCreateFlags::empty();

        if set_desc.is_push_descriptor() {
            if !device.enabled_extensions().khr_push_descriptor {
                return Err(DescriptorSetLayoutError::ExtensionNotEnabled {
                    extension: "khr_push_descriptor",
                    reason: "description was set to be a push descriptor",
                });
            }

            // TODO: VUID-VkDescriptorSetLayoutCreateInfo-flags-04590
            // If flags contains VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, flags must
            // not contain VK_DESCRIPTOR_SET_LAYOUT_CREATE_HOST_ONLY_POOL_BIT_VALVE

            flags |= ash::vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR;
        }

        for (binding, binding_desc) in bindings.iter().enumerate() {
            let binding_desc = match binding_desc {
                Some(d) => d,
                None => continue,
            };

            // FIXME: it is not legal to pass eg. the TESSELLATION_SHADER bit when the device
            //        doesn't have tess shaders enabled

            let ty = binding_desc.ty;

            if set_desc.is_push_descriptor() {
                if matches!(
                    ty,
                    DescriptorType::StorageBufferDynamic | DescriptorType::UniformBufferDynamic
                ) {
                    return Err(DescriptorSetLayoutError::PushDescriptorDynamicBuffer);
                }

                // TODO: VUID-VkDescriptorSetLayoutCreateInfo-flags-02208
                // If flags contains VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, then all
                // elements of pBindings must not have a descriptorType of
                // VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT

                // TODO: VUID-VkDescriptorSetLayoutCreateInfo-flags-04591
                // If flags contains VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                // pBindings must not have a descriptorType of VK_DESCRIPTOR_TYPE_MUTABLE_VALVE
            }

            descriptors_count.add_num(ty, binding_desc.descriptor_count);
            let mut binding_flags = ash::vk::DescriptorBindingFlags::empty();

            let p_immutable_samplers = if !binding_desc.immutable_samplers.is_empty() {
                if !matches!(
                    ty,
                    DescriptorType::Sampler | DescriptorType::CombinedImageSampler
                ) {
                    return Err(DescriptorSetLayoutError::ImmutableSamplersWrongDescriptorType);
                }

                if binding_desc.descriptor_count != binding_desc.immutable_samplers.len() as u32 {
                    return Err(DescriptorSetLayoutError::ImmutableSamplersCountMismatch {
                        descriptor_count: binding_desc.descriptor_count,
                        sampler_count: binding_desc.immutable_samplers.len() as u32,
                    });
                }

                // TODO: VUID-VkDescriptorSetLayoutBinding-pImmutableSamplers-04009
                // The sampler objects indicated by pImmutableSamplers must not have a borderColor
                // with one of the values VK_BORDER_COLOR_FLOAT_CUSTOM_EXT or
                // VK_BORDER_COLOR_INT_CUSTOM_EXT

                let sampler_handles = binding_desc
                    .immutable_samplers
                    .iter()
                    .map(|s| s.internal_object())
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                let p_immutable_samplers = sampler_handles.as_ptr();
                immutable_samplers_vk.push(sampler_handles);
                p_immutable_samplers
            } else {
                ptr::null()
            };

            if binding_desc.variable_count {
                if binding != bindings.len() - 1 {
                    return Err(DescriptorSetLayoutError::VariableCountDescMustBeLast);
                }

                if binding_desc.ty == DescriptorType::UniformBufferDynamic
                    || binding_desc.ty == DescriptorType::StorageBufferDynamic
                {
                    return Err(DescriptorSetLayoutError::VariableCountDescMustNotBeDynamic);
                }

                if !device.enabled_features().runtime_descriptor_array {
                    return Err(DescriptorSetLayoutError::FeatureNotEnabled {
                        feature: "runtime_descriptor_array",
                        reason: "binding has a variable count",
                    });
                }

                if !device
                    .enabled_features()
                    .descriptor_binding_variable_descriptor_count
                {
                    return Err(DescriptorSetLayoutError::FeatureNotEnabled {
                        feature: "descriptor_binding_variable_descriptor_count",
                        reason: "binding has a variable count",
                    });
                }

                if !device.enabled_features().descriptor_binding_partially_bound {
                    return Err(DescriptorSetLayoutError::FeatureNotEnabled {
                        feature: "descriptor_binding_partially_bound",
                        reason: "binding has a variable count",
                    });
                }

                variable_descriptor_count = binding_desc.descriptor_count;
                // TODO: should these be settable separately by the user?
                binding_flags |= ash::vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;
                binding_flags |= ash::vk::DescriptorBindingFlags::PARTIALLY_BOUND;
            }

            bindings_vk.push(ash::vk::DescriptorSetLayoutBinding {
                binding: binding as u32,
                descriptor_type: ty.into(),
                descriptor_count: binding_desc.descriptor_count,
                stage_flags: binding_desc.stages.into(),
                p_immutable_samplers,
            });
            binding_flags_vk.push(binding_flags);
        }

        if set_desc.is_push_descriptor()
            && descriptors_count.total()
                > device
                    .physical_device()
                    .properties()
                    .max_push_descriptors
                    .unwrap_or(0)
        {
            return Err(DescriptorSetLayoutError::MaxPushDescriptorsExceeded {
                max: device
                    .physical_device()
                    .properties()
                    .max_push_descriptors
                    .unwrap_or(0),
                obtained: descriptors_count.total(),
            });
        }

        let handle = unsafe {
            let binding_flags_infos = if device.api_version() >= Version::V1_2
                || device.enabled_extensions().ext_descriptor_indexing
            {
                Some(ash::vk::DescriptorSetLayoutBindingFlagsCreateInfo {
                    // Note that it seems legal to have no descriptor at all in the set.
                    binding_count: binding_flags_vk.len() as u32,
                    p_binding_flags: binding_flags_vk.as_ptr(),
                    ..Default::default()
                })
            } else {
                None
            };

            let infos = ash::vk::DescriptorSetLayoutCreateInfo {
                flags,
                binding_count: bindings_vk.len() as u32,
                p_bindings: bindings_vk.as_ptr(),
                p_next: if let Some(next) = binding_flags_infos.as_ref() {
                    next as *const _ as *const _
                } else {
                    ptr::null()
                },
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
        };

        Ok(Arc::new(DescriptorSetLayout {
            handle,
            device,
            desc: set_desc,
            descriptors_count,
            variable_descriptor_count,
        }))
    }

    pub(crate) fn desc(&self) -> &DescriptorSetDesc {
        &self.desc
    }

    /// Returns the number of descriptors of each type.
    #[inline]
    pub fn descriptors_count(&self) -> &DescriptorsCount {
        &self.descriptors_count
    }

    /// Returns the number of descriptors in a variable count descriptor. This will return zero if there are no variable count descriptors present.
    #[inline]
    pub fn variable_descriptor_count(&self) -> u32 {
        self.variable_descriptor_count
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

/// Error related to descriptor set layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorSetLayoutError {
    ExtensionNotEnabled {
        extension: &'static str,
        reason: &'static str,
    },
    FeatureNotEnabled {
        feature: &'static str,
        reason: &'static str,
    },

    /// The number of immutable samplers does not match the descriptor count.
    ImmutableSamplersCountMismatch {
        descriptor_count: u32,
        sampler_count: u32,
    },

    /// Immutable samplers were included on a descriptor type other than `Sampler` or
    /// `CombinedImageSampler`.
    ImmutableSamplersWrongDescriptorType,

    /// The maximum number of push descriptors has been exceeded.
    MaxPushDescriptorsExceeded {
        /// Maximum allowed value.
        max: u32,
        /// Value that was passed.
        obtained: u32,
    },

    /// Out of Memory.
    OomError(OomError),

    /// The layout was being created for push descriptors, but included a dynamic buffer descriptor.
    PushDescriptorDynamicBuffer,

    /// Variable count descriptor must be last binding.
    VariableCountDescMustBeLast,

    /// Variable count descriptor must not be a dynamic buffer.
    VariableCountDescMustNotBeDynamic,
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
        match *self {
            Self::ExtensionNotEnabled { extension, reason } => {
                write!(
                    fmt,
                    "the extension {} must be enabled: {}",
                    extension, reason
                )
            }
            Self::FeatureNotEnabled { feature, reason } => {
                write!(fmt, "the feature {} must be enabled: {}", feature, reason)
            }
            Self::ImmutableSamplersCountMismatch { .. } => {
                write!(
                    fmt,
                    "the number of immutable samplers does not match the descriptor count"
                )
            }
            Self::ImmutableSamplersWrongDescriptorType => {
                write!(
                    fmt,
                    "immutable samplers were included on a descriptor type other than Sampler or CombinedImageSampler"
                )
            }
            Self::MaxPushDescriptorsExceeded { .. } => {
                write!(
                    fmt,
                    "the maximum number of push descriptors has been exceeded"
                )
            }
            Self::PushDescriptorDynamicBuffer => {
                write!(fmt, "the layout was being created for push descriptors, but included a dynamic buffer descriptor")
            }
            Self::OomError(_) => {
                write!(fmt, "out of memory")
            }
            Self::VariableCountDescMustBeLast => {
                write!(fmt, "variable count descriptor must be last binding")
            }
            Self::VariableCountDescMustNotBeDynamic => {
                write!(
                    fmt,
                    "variable count descriptor must not be a dynamic buffer"
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::descriptor_set::layout::DescriptorDesc;
    use crate::descriptor_set::layout::DescriptorSetDesc;
    use crate::descriptor_set::layout::DescriptorSetLayout;
    use crate::descriptor_set::layout::DescriptorType;
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
            ty: DescriptorType::UniformBuffer,
            descriptor_count: 1,
            variable_count: false,
            stages: ShaderStages::all_graphics(),
            immutable_samplers: Vec::new(),
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
