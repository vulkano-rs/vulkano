// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Describes the layout of all descriptors within a descriptor set.
//!
//! When creating a new descriptor set, you must provide a *layout* object to create it from.

use crate::{
    device::{Device, DeviceOwned},
    macros::{impl_id_counter, vulkan_enum},
    sampler::Sampler,
    shader::{DescriptorBindingRequirements, ShaderStages},
    OomError, RequirementNotMet, RequiresOneOf, Version, VulkanError, VulkanObject,
};
use ahash::HashMap;
use std::{
    collections::BTreeMap,
    error::Error,
    fmt::{Display, Error as FmtError, Formatter},
    mem::MaybeUninit,
    num::NonZeroU64,
    ptr,
    sync::Arc,
};

/// Describes to the Vulkan implementation the layout of all descriptors within a descriptor set.
#[derive(Debug)]
pub struct DescriptorSetLayout {
    handle: ash::vk::DescriptorSetLayout,
    device: Arc<Device>,
    id: NonZeroU64,

    bindings: BTreeMap<u32, DescriptorSetLayoutBinding>,
    push_descriptor: bool,

    descriptor_counts: HashMap<DescriptorType, u32>,
}

impl DescriptorSetLayout {
    /// Creates a new `DescriptorSetLayout`.
    #[inline]
    pub fn new(
        device: Arc<Device>,
        mut create_info: DescriptorSetLayoutCreateInfo,
    ) -> Result<Arc<DescriptorSetLayout>, DescriptorSetLayoutCreationError> {
        let descriptor_counts = Self::validate(&device, &mut create_info)?;
        let handle = unsafe { Self::create(&device, &create_info)? };

        let DescriptorSetLayoutCreateInfo {
            bindings,
            push_descriptor,
            _ne: _,
        } = create_info;

        Ok(Arc::new(DescriptorSetLayout {
            handle,
            device,
            id: Self::next_id(),
            bindings,
            push_descriptor,
            descriptor_counts,
        }))
    }

    /// Creates a new `DescriptorSetLayout` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::DescriptorSetLayout,
        create_info: DescriptorSetLayoutCreateInfo,
    ) -> Arc<DescriptorSetLayout> {
        let DescriptorSetLayoutCreateInfo {
            bindings,
            push_descriptor,
            _ne: _,
        } = create_info;

        let mut descriptor_counts = HashMap::default();
        for binding in bindings.values() {
            if binding.descriptor_count != 0 {
                *descriptor_counts
                    .entry(binding.descriptor_type)
                    .or_default() += binding.descriptor_count;
            }
        }

        Arc::new(DescriptorSetLayout {
            handle,
            device,
            id: Self::next_id(),
            bindings,
            push_descriptor,
            descriptor_counts,
        })
    }

    fn validate(
        device: &Device,
        create_info: &mut DescriptorSetLayoutCreateInfo,
    ) -> Result<HashMap<DescriptorType, u32>, DescriptorSetLayoutCreationError> {
        let &mut DescriptorSetLayoutCreateInfo {
            ref bindings,
            push_descriptor,
            _ne: _,
        } = create_info;

        let mut descriptor_counts = HashMap::default();

        if push_descriptor {
            if !device.enabled_extensions().khr_push_descriptor {
                return Err(DescriptorSetLayoutCreationError::RequirementNotMet {
                    required_for: "`create_info.push_descriptor` is set",
                    requires_one_of: RequiresOneOf {
                        device_extensions: &["khr_push_descriptor"],
                        ..Default::default()
                    },
                });
            }
        }

        let highest_binding_num = bindings.keys().copied().next_back();

        for (&binding_num, binding) in bindings.iter() {
            let &DescriptorSetLayoutBinding {
                descriptor_type,
                descriptor_count,
                variable_descriptor_count,
                stages,
                ref immutable_samplers,
                _ne: _,
            } = binding;

            // VUID-VkDescriptorSetLayoutBinding-descriptorType-parameter
            descriptor_type.validate_device(device)?;

            if descriptor_count != 0 {
                // VUID-VkDescriptorSetLayoutBinding-descriptorCount-00283
                stages.validate_device(device)?;

                *descriptor_counts.entry(descriptor_type).or_default() += descriptor_count;
            }

            if push_descriptor {
                // VUID-VkDescriptorSetLayoutCreateInfo-flags-00280
                if matches!(
                    descriptor_type,
                    DescriptorType::StorageBufferDynamic | DescriptorType::UniformBufferDynamic
                ) {
                    return Err(
                        DescriptorSetLayoutCreationError::PushDescriptorDescriptorTypeIncompatible {
                            binding_num,
                        },
                    );
                }

                // VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-flags-03003
                if variable_descriptor_count {
                    return Err(
                        DescriptorSetLayoutCreationError::PushDescriptorVariableDescriptorCount {
                            binding_num,
                        },
                    );
                }
            }

            if !immutable_samplers.is_empty() {
                if immutable_samplers
                    .iter()
                    .any(|sampler| sampler.sampler_ycbcr_conversion().is_some())
                {
                    if !matches!(descriptor_type, DescriptorType::CombinedImageSampler) {
                        return Err(
                            DescriptorSetLayoutCreationError::ImmutableSamplersDescriptorTypeIncompatible {
                                binding_num,
                            },
                        );
                    }
                } else {
                    if !matches!(
                        descriptor_type,
                        DescriptorType::Sampler | DescriptorType::CombinedImageSampler
                    ) {
                        return Err(
                            DescriptorSetLayoutCreationError::ImmutableSamplersDescriptorTypeIncompatible {
                                binding_num,
                            },
                        );
                    }
                }

                // VUID-VkDescriptorSetLayoutBinding-descriptorType-00282
                if descriptor_count != immutable_samplers.len() as u32 {
                    return Err(
                        DescriptorSetLayoutCreationError::ImmutableSamplersCountMismatch {
                            binding_num,
                            sampler_count: immutable_samplers.len() as u32,
                            descriptor_count,
                        },
                    );
                }
            }

            // VUID-VkDescriptorSetLayoutBinding-descriptorType-01510
            // If descriptorType is VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT and descriptorCount is not 0, then stageFlags must be 0 or VK_SHADER_STAGE_FRAGMENT_BIT

            if variable_descriptor_count {
                // VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBindingVariableDescriptorCount-03014
                if !device
                    .enabled_features()
                    .descriptor_binding_variable_descriptor_count
                {
                    return Err(DescriptorSetLayoutCreationError::RequirementNotMet {
                        required_for: "`create_info.bindings` has an element where \
                            `variable_descriptor_count` is set",
                        requires_one_of: RequiresOneOf {
                            features: &["descriptor_binding_variable_descriptor_count"],
                            ..Default::default()
                        },
                    });
                }

                // VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-pBindingFlags-03004
                if Some(binding_num) != highest_binding_num {
                    return Err(
                        DescriptorSetLayoutCreationError::VariableDescriptorCountBindingNotHighest {
                            binding_num,
                            highest_binding_num: highest_binding_num.unwrap(),
                        },
                    );
                }

                // VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-pBindingFlags-03015
                if matches!(
                    descriptor_type,
                    DescriptorType::UniformBufferDynamic | DescriptorType::StorageBufferDynamic
                ) {
                    return Err(
                        DescriptorSetLayoutCreationError::VariableDescriptorCountDescriptorTypeIncompatible {
                            binding_num,
                        },
                    );
                }
            }
        }

        // VUID-VkDescriptorSetLayoutCreateInfo-flags-00281
        if push_descriptor
            && descriptor_counts.values().copied().sum::<u32>()
                > device
                    .physical_device()
                    .properties()
                    .max_push_descriptors
                    .unwrap_or(0)
        {
            return Err(
                DescriptorSetLayoutCreationError::MaxPushDescriptorsExceeded {
                    provided: descriptor_counts.values().copied().sum(),
                    max_supported: device
                        .physical_device()
                        .properties()
                        .max_push_descriptors
                        .unwrap_or(0),
                },
            );
        }

        Ok(descriptor_counts)
    }

    unsafe fn create(
        device: &Device,
        create_info: &DescriptorSetLayoutCreateInfo,
    ) -> Result<ash::vk::DescriptorSetLayout, DescriptorSetLayoutCreationError> {
        let &DescriptorSetLayoutCreateInfo {
            ref bindings,
            push_descriptor,
            _ne: _,
        } = create_info;

        let mut bindings_vk = Vec::with_capacity(bindings.len());
        let mut binding_flags_vk = Vec::with_capacity(bindings.len());
        let mut immutable_samplers_vk: Vec<Box<[ash::vk::Sampler]>> = Vec::new(); // only to keep the arrays of handles alive
        let mut flags = ash::vk::DescriptorSetLayoutCreateFlags::empty();

        if push_descriptor {
            flags |= ash::vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR;
        }

        for (&binding_num, binding) in bindings.iter() {
            let mut binding_flags = ash::vk::DescriptorBindingFlags::empty();

            let p_immutable_samplers = if !binding.immutable_samplers.is_empty() {
                // VUID-VkDescriptorSetLayoutBinding-descriptorType-00282
                let sampler_handles = binding
                    .immutable_samplers
                    .iter()
                    .map(|s| s.handle())
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                let p_immutable_samplers = sampler_handles.as_ptr();
                immutable_samplers_vk.push(sampler_handles);
                p_immutable_samplers
            } else {
                ptr::null()
            };

            if binding.variable_descriptor_count {
                binding_flags |= ash::vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;
            }

            // VUID-VkDescriptorSetLayoutCreateInfo-binding-00279
            // Guaranteed by BTreeMap
            bindings_vk.push(ash::vk::DescriptorSetLayoutBinding {
                binding: binding_num,
                descriptor_type: binding.descriptor_type.into(),
                descriptor_count: binding.descriptor_count,
                stage_flags: binding.stages.into(),
                p_immutable_samplers,
            });
            binding_flags_vk.push(binding_flags);
        }

        let mut binding_flags_create_info = if device.api_version() >= Version::V1_2
            || device.enabled_extensions().ext_descriptor_indexing
        {
            Some(ash::vk::DescriptorSetLayoutBindingFlagsCreateInfo {
                // VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-bindingCount-03002
                binding_count: binding_flags_vk.len() as u32,
                p_binding_flags: binding_flags_vk.as_ptr(),
                ..Default::default()
            })
        } else {
            None
        };

        let mut create_info = ash::vk::DescriptorSetLayoutCreateInfo {
            flags,
            binding_count: bindings_vk.len() as u32,
            p_bindings: bindings_vk.as_ptr(),
            ..Default::default()
        };

        if let Some(binding_flags_create_info) = binding_flags_create_info.as_mut() {
            binding_flags_create_info.p_next = create_info.p_next;
            create_info.p_next = binding_flags_create_info as *const _ as *const _;
        }

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_descriptor_set_layout)(
                device.handle(),
                &create_info,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(handle)
    }

    pub(crate) fn id(&self) -> NonZeroU64 {
        self.id
    }

    /// Returns the bindings of the descriptor set layout.
    #[inline]
    pub fn bindings(&self) -> &BTreeMap<u32, DescriptorSetLayoutBinding> {
        &self.bindings
    }

    /// Returns whether the descriptor set layout is for push descriptors or regular descriptor
    /// sets.
    #[inline]
    pub fn push_descriptor(&self) -> bool {
        self.push_descriptor
    }

    /// Returns the number of descriptors of each type.
    ///
    /// The map is guaranteed to not contain any elements with a count of `0`.
    #[inline]
    pub fn descriptor_counts(&self) -> &HashMap<DescriptorType, u32> {
        &self.descriptor_counts
    }

    /// If the highest-numbered binding has a variable count, returns its `descriptor_count`.
    /// Otherwise returns `0`.
    #[inline]
    pub fn variable_descriptor_count(&self) -> u32 {
        self.bindings
            .values()
            .next_back()
            .map(|binding| {
                if binding.variable_descriptor_count {
                    binding.descriptor_count
                } else {
                    0
                }
            })
            .unwrap_or(0)
    }

    /// Returns whether `self` is compatible with `other`.
    ///
    /// "Compatible" in this sense is defined by the Vulkan specification under the section
    /// "Pipeline layout compatibility": either the two are the same descriptor set layout object,
    /// or they must be identically defined to the Vulkan API.
    #[inline]
    pub fn is_compatible_with(&self, other: &DescriptorSetLayout) -> bool {
        self == other
            || (self.bindings == other.bindings && self.push_descriptor == other.push_descriptor)
    }
}

impl Drop for DescriptorSetLayout {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.destroy_descriptor_set_layout)(
                self.device.handle(),
                self.handle,
                ptr::null(),
            );
        }
    }
}

unsafe impl VulkanObject for DescriptorSetLayout {
    type Handle = ash::vk::DescriptorSetLayout;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for DescriptorSetLayout {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(DescriptorSetLayout);

/// Error related to descriptor set layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorSetLayoutCreationError {
    /// Out of Memory.
    OomError(OomError),

    RequirementNotMet {
        required_for: &'static str,
        requires_one_of: RequiresOneOf,
    },

    /// A binding includes immutable samplers but their number differs from  `descriptor_count`.
    ImmutableSamplersCountMismatch {
        binding_num: u32,
        sampler_count: u32,
        descriptor_count: u32,
    },

    /// A binding includes immutable samplers but it has an incompatible `descriptor_type`.
    ImmutableSamplersDescriptorTypeIncompatible { binding_num: u32 },

    /// More descriptors were provided in all bindings than the
    /// [`max_push_descriptors`](crate::device::Properties::max_push_descriptors) limit.
    MaxPushDescriptorsExceeded { provided: u32, max_supported: u32 },

    /// `push_descriptor` is enabled, but a binding has an incompatible `descriptor_type`.
    PushDescriptorDescriptorTypeIncompatible { binding_num: u32 },

    /// `push_descriptor` is enabled, but a binding has `variable_descriptor_count` enabled.
    PushDescriptorVariableDescriptorCount { binding_num: u32 },

    /// A binding has `variable_descriptor_count` enabled, but it is not the highest-numbered
    /// binding.
    VariableDescriptorCountBindingNotHighest {
        binding_num: u32,
        highest_binding_num: u32,
    },

    /// A binding has `variable_descriptor_count` enabled, but it has an incompatible
    /// `descriptor_type`.
    VariableDescriptorCountDescriptorTypeIncompatible { binding_num: u32 },
}

impl From<VulkanError> for DescriptorSetLayoutCreationError {
    fn from(error: VulkanError) -> Self {
        Self::OomError(error.into())
    }
}

impl Error for DescriptorSetLayoutCreationError {}

impl Display for DescriptorSetLayoutCreationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::OomError(_) => {
                write!(f, "out of memory")
            }
            Self::RequirementNotMet {
                required_for,
                requires_one_of,
            } => write!(
                f,
                "a requirement was not met for: {}; requires one of: {}",
                required_for, requires_one_of,
            ),
            Self::ImmutableSamplersCountMismatch {
                binding_num,
                sampler_count,
                descriptor_count,
            } => write!(
                f,
                "binding {} includes immutable samplers but their number ({}) differs from \
                `descriptor_count` ({})",
                binding_num, sampler_count, descriptor_count,
            ),
            Self::ImmutableSamplersDescriptorTypeIncompatible { binding_num } => write!(
                f,
                "binding {} includes immutable samplers but it has an incompatible \
                `descriptor_type`",
                binding_num,
            ),
            Self::MaxPushDescriptorsExceeded {
                provided,
                max_supported,
            } => write!(
                f,
                "more descriptors were provided in all bindings ({}) than the \
                `max_push_descriptors` limit ({})",
                provided, max_supported,
            ),
            Self::PushDescriptorDescriptorTypeIncompatible { binding_num } => write!(
                f,
                "`push_descriptor` is enabled, but binding {} has an incompatible \
                `descriptor_type`",
                binding_num,
            ),
            Self::PushDescriptorVariableDescriptorCount { binding_num } => write!(
                f,
                "`push_descriptor` is enabled, but binding {} has `variable_descriptor_count` \
                enabled",
                binding_num,
            ),
            Self::VariableDescriptorCountBindingNotHighest {
                binding_num,
                highest_binding_num,
            } => write!(
                f,
                "binding {} has `variable_descriptor_count` enabled, but it is not the \
                highest-numbered binding ({})",
                binding_num, highest_binding_num,
            ),
            Self::VariableDescriptorCountDescriptorTypeIncompatible { binding_num } => write!(
                f,
                "binding {} has `variable_descriptor_count` enabled, but it has an incompatible \
                `descriptor_type`",
                binding_num,
            ),
        }
    }
}

impl From<RequirementNotMet> for DescriptorSetLayoutCreationError {
    fn from(err: RequirementNotMet) -> Self {
        Self::RequirementNotMet {
            required_for: err.required_for,
            requires_one_of: err.requires_one_of,
        }
    }
}

/// Parameters to create a new `DescriptorSetLayout`.
#[derive(Clone, Debug)]
pub struct DescriptorSetLayoutCreateInfo {
    /// The bindings of the desriptor set layout. These are specified according to binding number.
    ///
    /// It is generally advisable to keep the binding numbers low. Higher binding numbers may
    /// use more memory inside Vulkan.
    ///
    /// The default value is empty.
    pub bindings: BTreeMap<u32, DescriptorSetLayoutBinding>,

    /// Whether the descriptor set layout should be created for push descriptors.
    ///
    /// If `true`, the layout can only be used for push descriptors, and if `false`, it can only
    /// be used for regular descriptor sets.
    ///
    /// If set to `true`, the
    /// [`khr_push_descriptor`](crate::device::DeviceExtensions::khr_push_descriptor) extension must
    /// be enabled on the device, and there are several restrictions:
    /// - There must be no bindings with a type of [`DescriptorType::UniformBufferDynamic`]
    ///   or [`DescriptorType::StorageBufferDynamic`].
    /// - There must be no bindings with `variable_descriptor_count` enabled.
    /// - The total number of descriptors across all bindings must be less than the
    ///   [`max_push_descriptors`](crate::device::Properties::max_push_descriptors) limit.
    ///
    /// The default value is `false`.
    pub push_descriptor: bool,

    pub _ne: crate::NonExhaustive,
}

impl Default for DescriptorSetLayoutCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            bindings: BTreeMap::new(),
            push_descriptor: false,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl DescriptorSetLayoutCreateInfo {
    /// Builds a list of `DescriptorSetLayoutCreateInfo` from an iterator of
    /// `DescriptorBindingRequirements` originating from a shader.
    pub fn from_requirements<'a>(
        descriptor_requirements: impl IntoIterator<
            Item = ((u32, u32), &'a DescriptorBindingRequirements),
        >,
    ) -> Vec<Self> {
        let mut create_infos: Vec<Self> = Vec::new();

        for ((set_num, binding_num), reqs) in descriptor_requirements {
            let set_num = set_num as usize;

            if set_num >= create_infos.len() {
                create_infos.resize(set_num + 1, Self::default());
            }

            let bindings = &mut create_infos[set_num].bindings;
            bindings.insert(binding_num, reqs.into());
        }

        create_infos
    }
}

/// A binding in a descriptor set layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DescriptorSetLayoutBinding {
    /// The content and layout of each array element of a binding.
    ///
    /// There is no default value.
    pub descriptor_type: DescriptorType,

    /// How many descriptors (array elements) this binding is made of.
    ///
    /// If the binding is a single element rather than an array, then you must specify `1`.
    ///
    /// The default value is `1`.
    pub descriptor_count: u32,

    /// Whether the binding has a variable number of descriptors.
    ///
    /// If set to `true`, the [`descriptor_binding_variable_descriptor_count`] feature must be
    /// enabled. The value of `descriptor_count` specifies the maximum number of descriptors
    /// allowed.
    ///
    /// There may only be one binding with a variable count in a descriptor set, and it must be the
    /// binding with the highest binding number. The `descriptor_type` must not be
    /// [`DescriptorType::UniformBufferDynamic`] or [`DescriptorType::StorageBufferDynamic`].
    ///
    /// The default value is `false`.
    ///
    /// [`descriptor_binding_variable_descriptor_count`]: crate::device::Features::descriptor_binding_variable_descriptor_count
    pub variable_descriptor_count: bool,

    /// Which shader stages are going to access the descriptors in this binding.
    ///
    /// The default value is [`ShaderStages::empty()`], which must be overridden.
    pub stages: ShaderStages,

    /// Samplers that are included as a fixed part of the descriptor set layout. Once bound, they
    /// do not need to be provided when creating a descriptor set.
    ///
    /// The list must be either empty, or contain exactly `descriptor_count` samplers. It can only
    /// be non-empty if `descriptor_type` is [`DescriptorType::Sampler`] or
    /// [`DescriptorType::CombinedImageSampler`]. If any of the samplers has an attached sampler
    /// YCbCr conversion, then only [`DescriptorType::CombinedImageSampler`] is allowed.
    ///
    /// The default value is empty.
    pub immutable_samplers: Vec<Arc<Sampler>>,

    pub _ne: crate::NonExhaustive,
}

impl DescriptorSetLayoutBinding {
    /// Returns a `DescriptorSetLayoutBinding` with the given type.
    #[inline]
    pub fn descriptor_type(descriptor_type: DescriptorType) -> Self {
        Self {
            descriptor_type,
            descriptor_count: 1,
            variable_descriptor_count: false,
            stages: ShaderStages::empty(),
            immutable_samplers: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }

    /// Checks whether the descriptor of a pipeline layout `self` is compatible with the
    /// requirements of a shader `other`.
    #[inline]
    pub fn ensure_compatible_with_shader(
        &self,
        binding_requirements: &DescriptorBindingRequirements,
    ) -> Result<(), DescriptorRequirementsNotMet> {
        let &DescriptorBindingRequirements {
            ref descriptor_types,
            descriptor_count,
            image_format: _,
            image_multisampled: _,
            image_scalar_type: _,
            image_view_type: _,
            stages,
            descriptors: _,
        } = binding_requirements;

        if !descriptor_types.contains(&self.descriptor_type) {
            return Err(DescriptorRequirementsNotMet::DescriptorType {
                required: descriptor_types.clone(),
                obtained: self.descriptor_type,
            });
        }

        if let Some(required) = descriptor_count {
            if self.descriptor_count < required {
                return Err(DescriptorRequirementsNotMet::DescriptorCount {
                    required,
                    obtained: self.descriptor_count,
                });
            }
        }

        if !self.stages.contains(stages) {
            return Err(DescriptorRequirementsNotMet::ShaderStages {
                required: stages,
                obtained: self.stages,
            });
        }

        Ok(())
    }
}

impl From<&DescriptorBindingRequirements> for DescriptorSetLayoutBinding {
    #[inline]
    fn from(reqs: &DescriptorBindingRequirements) -> Self {
        Self {
            descriptor_type: reqs.descriptor_types[0],
            descriptor_count: reqs.descriptor_count.unwrap_or(0),
            variable_descriptor_count: false,
            stages: reqs.stages,
            immutable_samplers: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

/// Error when checking whether the requirements for a binding have been met.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DescriptorRequirementsNotMet {
    /// The binding's `descriptor_type` is not one of those required.
    DescriptorType {
        required: Vec<DescriptorType>,
        obtained: DescriptorType,
    },

    /// The binding's `descriptor_count` is less than what is required.
    DescriptorCount { required: u32, obtained: u32 },

    /// The binding's `stages` does not contain the stages that are required.
    ShaderStages {
        required: ShaderStages,
        obtained: ShaderStages,
    },
}

impl Error for DescriptorRequirementsNotMet {}

impl Display for DescriptorRequirementsNotMet {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), FmtError> {
        match self {
            Self::DescriptorType { required, obtained } => write!(
                f,
                "the descriptor's type ({:?}) is not one of those required ({:?})",
                obtained, required,
            ),
            Self::DescriptorCount { required, obtained } => write!(
                f,
                "the descriptor count ({}) is less than what is required ({})",
                obtained, required,
            ),
            Self::ShaderStages { .. } => write!(
                f,
                "the descriptor's shader stages do not contain the stages that are required",
            ),
        }
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// Describes what kind of resource may later be bound to a descriptor.
    DescriptorType = DescriptorType(i32);

    /// Describes how a `SampledImage` descriptor should be read.
    Sampler = SAMPLER,

    /// Combines `SampledImage` and `Sampler` in one descriptor.
    CombinedImageSampler = COMBINED_IMAGE_SAMPLER,

    /// Gives read-only access to an image via a sampler. The image must be combined with a sampler
    /// inside the shader.
    SampledImage = SAMPLED_IMAGE,

    /// Gives read and/or write access to individual pixels in an image. The image cannot be
    /// sampled, so you have exactly specify which pixel to read or write.
    StorageImage = STORAGE_IMAGE,

    /// Gives read-only access to the content of a buffer, interpreted as an array of texel data.
    UniformTexelBuffer = UNIFORM_TEXEL_BUFFER,

    /// Gives read and/or write access to the content of a buffer, interpreted as an array of texel
    /// data. Less restrictive but sometimes slower than a uniform texel buffer.
    StorageTexelBuffer = STORAGE_TEXEL_BUFFER,

    /// Gives read-only access to the content of a buffer, interpreted as a structure.
    UniformBuffer = UNIFORM_BUFFER,

    /// Gives read and/or write access to the content of a buffer, interpreted as a structure. Less
    /// restrictive but sometimes slower than a uniform buffer.
    StorageBuffer = STORAGE_BUFFER,

    /// As `UniformBuffer`, but the offset within the buffer is specified at the time the descriptor
    /// set is bound, rather than when the descriptor set is updated.
    UniformBufferDynamic = UNIFORM_BUFFER_DYNAMIC,

    /// As `StorageBuffer`, but the offset within the buffer is specified at the time the descriptor
    /// set is bound, rather than when the descriptor set is updated.
    StorageBufferDynamic = STORAGE_BUFFER_DYNAMIC,

    /// Gives access to an image inside a fragment shader via a render pass. You can only access the
    /// pixel that is currently being processed by the fragment shader.
    InputAttachment = INPUT_ATTACHMENT,

    /* TODO: enable
    // TODO: document
    InlineUniformBlock = INLINE_UNIFORM_BLOCK {
        api_version: V1_3,
        device_extensions: [ext_inline_uniform_block],
    },*/

    /* TODO: enable
    // TODO: document
    AccelerationStructure = ACCELERATION_STRUCTURE_KHR {
        device_extensions: [khr_acceleration_structure],
    },*/

    /* TODO: enable
    // TODO: document
    AccelerationStructureNV = ACCELERATION_STRUCTURE_NV {
        device_extensions: [nv_ray_tracing],
    },*/

    /* TODO: enable
    // TODO: document
    SampleWeightImage = SAMPLE_WEIGHT_IMAGE_QCOM {
        device_extensions: [qcom_image_processing],
    },*/

    /* TODO: enable
    // TODO: document
    BlockMatchImage = BLOCK_MATCH_IMAGE_QCOM {
        device_extensions: [qcom_image_processing],
    },*/

    /* TODO: enable
    // TODO: document
    Mutable = MUTABLE_VALVE {
        device_extensions: [valve_mutable_descriptor_type],
    },*/
}

#[cfg(test)]
mod tests {
    use crate::{
        descriptor_set::layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
        shader::ShaderStages,
    };
    use ahash::HashMap;

    #[test]
    fn empty() {
        let (device, _) = gfx_dev_and_queue!();
        let _layout = DescriptorSetLayout::new(device, Default::default());
    }

    #[test]
    fn basic_create() {
        let (device, _) = gfx_dev_and_queue!();

        let sl = DescriptorSetLayout::new(
            device,
            DescriptorSetLayoutCreateInfo {
                bindings: [(
                    0,
                    DescriptorSetLayoutBinding {
                        stages: ShaderStages::all_graphics(),
                        ..DescriptorSetLayoutBinding::descriptor_type(DescriptorType::UniformBuffer)
                    },
                )]
                .into(),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(
            sl.descriptor_counts(),
            &[(DescriptorType::UniformBuffer, 1)]
                .into_iter()
                .collect::<HashMap<_, _>>(),
        );
    }
}
