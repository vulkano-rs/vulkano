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
    image::ImageLayout,
    macros::{impl_id_counter, vulkan_bitflags, vulkan_enum},
    sampler::Sampler,
    shader::{DescriptorBindingRequirements, ShaderStages},
    RequiresOneOf, RuntimeError, ValidationError, Version, VulkanError, VulkanObject,
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

    flags: DescriptorSetLayoutCreateFlags,
    bindings: BTreeMap<u32, DescriptorSetLayoutBinding>,

    descriptor_counts: HashMap<DescriptorType, u32>,
}

impl DescriptorSetLayout {
    /// Creates a new `DescriptorSetLayout`.
    #[inline]
    pub fn new(
        device: Arc<Device>,
        create_info: DescriptorSetLayoutCreateInfo,
    ) -> Result<Arc<DescriptorSetLayout>, VulkanError> {
        Self::validate_new(&device, &create_info)?;

        unsafe { Ok(Self::new_unchecked(device, create_info)?) }
    }

    fn validate_new(
        device: &Device,
        create_info: &DescriptorSetLayoutCreateInfo,
    ) -> Result<(), ValidationError> {
        // VUID-vkCreateDescriptorSetLayout-pCreateInfo-parameter
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        if let Some(max_per_set_descriptors) = device
            .physical_device()
            .properties()
            .max_per_set_descriptors
        {
            let total_descriptor_count: u32 = create_info
                .bindings
                .values()
                .map(|binding| binding.descriptor_count)
                .sum();

            // Safety: create_info is validated, and we only enter here if the
            // max_per_set_descriptors property exists (which means this function exists too).
            if total_descriptor_count > max_per_set_descriptors
                && unsafe {
                    device
                        .descriptor_set_layout_support_unchecked(create_info)
                        .is_none()
                }
            {
                return Err(ValidationError {
                    problem: "the total number of descriptors across all bindings is greater than \
                        the `max_per_set_descriptors` limit, and \
                        `device.descriptor_set_layout_support` returned `None`"
                        .into(),
                    ..Default::default()
                });
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        create_info: DescriptorSetLayoutCreateInfo,
    ) -> Result<Arc<DescriptorSetLayout>, RuntimeError> {
        let &DescriptorSetLayoutCreateInfo {
            flags,
            ref bindings,
            _ne: _,
        } = &create_info;

        struct PerBinding {
            immutable_samplers_vk: Vec<ash::vk::Sampler>,
        }

        let mut bindings_vk = Vec::with_capacity(bindings.len());
        let mut per_binding_vk = Vec::with_capacity(bindings.len());
        let mut binding_flags_info_vk = None;
        let mut binding_flags_vk = Vec::with_capacity(bindings.len());

        for (&binding_num, binding) in bindings.iter() {
            let &DescriptorSetLayoutBinding {
                binding_flags,
                descriptor_type,
                descriptor_count,
                stages,
                ref immutable_samplers,
                _ne: _,
            } = binding;

            bindings_vk.push(ash::vk::DescriptorSetLayoutBinding {
                binding: binding_num,
                descriptor_type: descriptor_type.into(),
                descriptor_count,
                stage_flags: stages.into(),
                p_immutable_samplers: ptr::null(),
            });
            per_binding_vk.push(PerBinding {
                immutable_samplers_vk: immutable_samplers
                    .iter()
                    .map(VulkanObject::handle)
                    .collect(),
            });
            binding_flags_vk.push(binding_flags.into());
        }

        for (binding_vk, per_binding_vk) in bindings_vk.iter_mut().zip(per_binding_vk.iter()) {
            let PerBinding {
                immutable_samplers_vk,
            } = per_binding_vk;

            if !immutable_samplers_vk.is_empty() {
                binding_vk.p_immutable_samplers = immutable_samplers_vk.as_ptr();
            }
        }

        let mut create_info_vk = ash::vk::DescriptorSetLayoutCreateInfo {
            flags: flags.into(),
            binding_count: bindings_vk.len() as u32,
            p_bindings: bindings_vk.as_ptr(),
            ..Default::default()
        };

        if device.api_version() >= Version::V1_2
            || device.enabled_extensions().ext_descriptor_indexing
        {
            let next =
                binding_flags_info_vk.insert(ash::vk::DescriptorSetLayoutBindingFlagsCreateInfo {
                    binding_count: binding_flags_vk.len() as u32,
                    p_binding_flags: binding_flags_vk.as_ptr(),
                    ..Default::default()
                });

            next.p_next = create_info_vk.p_next;
            create_info_vk.p_next = next as *const _ as *const _;
        }

        let handle = {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_descriptor_set_layout)(
                device.handle(),
                &create_info_vk,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(RuntimeError::from)?;
            output.assume_init()
        };

        Ok(Self::from_handle(device, handle, create_info))
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
            flags,
            bindings,
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
            flags,
            bindings,
            descriptor_counts,
        })
    }

    pub(crate) fn id(&self) -> NonZeroU64 {
        self.id
    }

    /// Returns the flags that the descriptor set layout was created with.
    #[inline]
    pub fn flags(&self) -> DescriptorSetLayoutCreateFlags {
        self.flags
    }

    /// Returns the bindings of the descriptor set layout.
    #[inline]
    pub fn bindings(&self) -> &BTreeMap<u32, DescriptorSetLayoutBinding> {
        &self.bindings
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
                if binding
                    .binding_flags
                    .intersects(DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT)
                {
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
        self == other || (self.flags == other.flags && self.bindings == other.bindings)
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

/// Parameters to create a new `DescriptorSetLayout`.
#[derive(Clone, Debug)]
pub struct DescriptorSetLayoutCreateInfo {
    /// Specifies how to create the descriptor set layout.
    pub flags: DescriptorSetLayoutCreateFlags,

    /// The bindings of the desriptor set layout. These are specified according to binding number.
    ///
    /// It is generally advisable to keep the binding numbers low. Higher binding numbers may
    /// use more memory inside Vulkan.
    ///
    /// The default value is empty.
    pub bindings: BTreeMap<u32, DescriptorSetLayoutBinding>,

    pub _ne: crate::NonExhaustive,
}

impl DescriptorSetLayoutCreateInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            flags,
            ref bindings,
            _ne: _,
        } = self;

        flags
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "flags".into(),
                vuids: &["VUID-VkDescriptorSetLayoutCreateInfo-flags-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        // VUID-VkDescriptorSetLayoutCreateInfo-binding-00279
        // Ensured because it is a map

        let mut total_descriptor_count = 0;
        let highest_binding_num = bindings.keys().copied().next_back();

        for (&binding_num, binding) in bindings.iter() {
            binding
                .validate(device)
                .map_err(|err| err.add_context(format!("bindings[{}]", binding_num)))?;

            let &DescriptorSetLayoutBinding {
                binding_flags,
                descriptor_type,
                descriptor_count,
                stages: _,
                immutable_samplers: _,
                _ne: _,
            } = binding;

            total_descriptor_count += descriptor_count;

            if flags.intersects(DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR) {
                if matches!(
                    descriptor_type,
                    DescriptorType::UniformBufferDynamic
                        | DescriptorType::StorageBufferDynamic
                        | DescriptorType::InlineUniformBlock
                ) {
                    return Err(ValidationError {
                        problem: format!(
                            "`flags` contains `DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR`, \
                            and `bindings[{}].descriptor_type` is
                            `DescriptorType::UniformBufferDynamic`, \
                            `DescriptorType::StorageBufferDynamic` or \
                            `DescriptorType::InlineUniformBlock`",
                            binding_num
                        )
                        .into(),
                        vuids: &[
                            "VUID-VkDescriptorSetLayoutCreateInfo-flags-00280",
                            "VUID-VkDescriptorSetLayoutCreateInfo-flags-02208",
                        ],
                        ..Default::default()
                    });
                }

                if binding_flags.intersects(DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT) {
                    return Err(ValidationError {
                        problem: format!(
                            "`flags` contains `DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR`, \
                            and `bindings[{}].flags` contains \
                            `DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT`",
                            binding_num
                        )
                        .into(),
                        vuids: &["VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-flags-03003"],
                        ..Default::default()
                    });
                }
            }

            if binding_flags.intersects(DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT)
                && Some(binding_num) != highest_binding_num
            {
                return Err(ValidationError {
                    problem: format!(
                        "`bindings[{}].flags` contains \
                        `DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT`, but {0} is not the
                        highest binding number in `bindings`",
                        binding_num
                    )
                    .into(),
                    vuids: &[
                        "VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-pBindingFlags-03004",
                    ],
                    ..Default::default()
                });
            }
        }

        let max_push_descriptors = device
            .physical_device()
            .properties()
            .max_push_descriptors
            .unwrap_or(0);

        if flags.intersects(DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR)
            && total_descriptor_count > max_push_descriptors
        {
            return Err(ValidationError {
                problem: "`flags` contains `DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR`, and \
                    the total number of descriptors in `bindings` exceeds the
                    `max_push_descriptors` limit"
                    .into(),
                vuids: &["VUID-VkDescriptorSetLayoutCreateInfo-flags-00281"],
                ..Default::default()
            });
        }

        Ok(())
    }
}

impl Default for DescriptorSetLayoutCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            flags: DescriptorSetLayoutCreateFlags::empty(),
            bindings: BTreeMap::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags that control how a descriptor set layout is created.
    DescriptorSetLayoutCreateFlags = DescriptorSetLayoutCreateFlags(u32);

    /* TODO: enable
    // TODO: document
    UPDATE_AFTER_BIND_POOL = UPDATE_AFTER_BIND_POOL {
        api_version: V1_2,
        device_extensions: [ext_descriptor_indexing],
    }, */

    /// Whether the descriptor set layout should be created for push descriptors.
    ///
    /// If set, the layout can only be used for push descriptors, and if not set, it can only
    /// be used for regular descriptor sets.
    ///
    /// If set, there are several restrictions:
    /// - There must be no bindings with a type of [`DescriptorType::UniformBufferDynamic`],
    ///   [`DescriptorType::StorageBufferDynamic`] or [`DescriptorType::InlineUniformBlock`].
    /// - There must be no bindings with `variable_descriptor_count` enabled.
    /// - The total number of descriptors across all bindings must be less than the
    ///   [`max_push_descriptors`](crate::device::Properties::max_push_descriptors) limit.
    PUSH_DESCRIPTOR = PUSH_DESCRIPTOR_KHR {
        device_extensions: [khr_push_descriptor],
    },

    /* TODO: enable
    // TODO: document
    DESCRIPTOR_BUFFER = DESCRIPTOR_BUFFER_EXT {
        device_extensions: [ext_descriptor_buffer],
    }, */

    /* TODO: enable
    // TODO: document
    EMBEDDED_IMMUTABLE_SAMPLERS = EMBEDDED_IMMUTABLE_SAMPLERS_EXT {
        device_extensions: [ext_descriptor_buffer],
    }, */

    /* TODO: enable
    // TODO: document
    HOST_ONLY_POOL = HOST_ONLY_POOL_EXT {
        device_extensions: [ext_mutable_descriptor_type, valve_mutable_descriptor_type],
    }, */
}

/// A binding in a descriptor set layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DescriptorSetLayoutBinding {
    /// Specifies how to create the binding.
    pub binding_flags: DescriptorBindingFlags,

    /// The content and layout of each array element of a binding.
    ///
    /// There is no default value.
    pub descriptor_type: DescriptorType,

    /// How many descriptors (array elements) this binding is made of.
    ///
    /// If the binding is a single element rather than an array, then you must specify `1`.
    ///
    /// If `descriptor_type` is [`DescriptorType::InlineUniformBlock`], then there can be at most
    /// one descriptor in the binding, and this value instead specifies the number of bytes
    /// available in the inline uniform block. The value must then be a multiple of 4.
    ///
    /// The default value is `1`.
    pub descriptor_count: u32,

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
            binding_flags: DescriptorBindingFlags::empty(),
            descriptor_type,
            descriptor_count: 1,
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

    pub(crate) fn validate(&self, device: &Device) -> Result<(), ValidationError> {
        let &Self {
            binding_flags,
            descriptor_type,
            descriptor_count,
            stages,
            ref immutable_samplers,
            _ne: _,
        } = self;

        binding_flags
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "binding_flags".into(),
                vuids: &[
                    "VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-pBindingFlags-parameter",
                ],
                ..ValidationError::from_requirement(err)
            })?;

        descriptor_type
            .validate_device(device)
            .map_err(|err| ValidationError {
                context: "descriptor_type".into(),
                vuids: &["VUID-VkDescriptorSetLayoutBinding-descriptorType-parameter"],
                ..ValidationError::from_requirement(err)
            })?;

        if descriptor_type == DescriptorType::InlineUniformBlock {
            if !device.enabled_features().inline_uniform_block {
                return Err(ValidationError {
                    context: "descriptor_type".into(),
                    problem: "DescriptorType::InlineUniformBlock".into(),
                    requires_one_of: RequiresOneOf {
                        features: &["inline_uniform_block"],
                        ..Default::default()
                    },
                    vuids: &["VUID-VkDescriptorSetLayoutBinding-descriptorType-04604"],
                });
            }

            if descriptor_count % 4 != 0 {
                return Err(ValidationError {
                    problem: "`descriptor_type` is `DescriptorType::InlineUniformBlock`, and
                        `descriptor_count` is not a multiple of 4"
                        .into(),
                    vuids: &["VUID-VkDescriptorSetLayoutBinding-descriptorType-02209"],
                    ..Default::default()
                });
            }

            if descriptor_count
                > device
                    .physical_device()
                    .properties()
                    .max_inline_uniform_block_size
                    .unwrap_or(0)
            {
                return Err(ValidationError {
                    problem: "`descriptor_type` is `DescriptorType::InlineUniformBlock`, and
                        `descriptor_count` is greater than the `max_inline_uniform_block_size`
                        limit"
                        .into(),
                    vuids: &["VUID-VkDescriptorSetLayoutBinding-descriptorType-08004"],
                    ..Default::default()
                });
            }
        }

        if descriptor_count != 0 {
            stages
                .validate_device(device)
                .map_err(|err| ValidationError {
                    context: "stages".into(),
                    vuids: &["VUID-VkDescriptorSetLayoutBinding-descriptorCount-00283"],
                    ..ValidationError::from_requirement(err)
                })?;

            if descriptor_type == DescriptorType::InputAttachment
                && !(stages.is_empty() || stages == ShaderStages::FRAGMENT)
            {
                return Err(ValidationError {
                    problem: "`descriptor_type` is `DescriptorType::InputAttachment`, but \
                        `stages` is not either empty or equal to `ShaderStages::FRAGMENT`"
                        .into(),
                    vuids: &["VUID-VkDescriptorSetLayoutBinding-descriptorType-01510"],
                    ..Default::default()
                });
            }
        }

        if !immutable_samplers.is_empty() {
            if descriptor_count != immutable_samplers.len() as u32 {
                return Err(ValidationError {
                    problem: "`immutable_samplers` is not empty, but its length does not equal \
                        `descriptor_count`"
                        .into(),
                    vuids: &["VUID-VkDescriptorSetLayoutBinding-descriptorType-00282"],
                    ..Default::default()
                });
            }

            let mut has_sampler_ycbcr_conversion = false;

            for sampler in immutable_samplers {
                assert_eq!(device, sampler.device().as_ref());

                has_sampler_ycbcr_conversion |= sampler.sampler_ycbcr_conversion().is_some();
            }

            if has_sampler_ycbcr_conversion {
                if !matches!(descriptor_type, DescriptorType::CombinedImageSampler) {
                    return Err(ValidationError {
                        problem: "`immutable_samplers` contains a sampler with a \
                            sampler YCbCr conversion, but `descriptor_type` is not \
                            `DescriptorType::CombinedImageSampler`"
                            .into(),
                        vuids: &["VUID-VkDescriptorSetLayoutBinding-descriptorType-00282"],
                        ..Default::default()
                    });
                }
            } else {
                if !matches!(
                    descriptor_type,
                    DescriptorType::Sampler | DescriptorType::CombinedImageSampler
                ) {
                    return Err(ValidationError {
                        problem: "`immutable_samplers` is not empty, but `descriptor_type` is not \
                            `DescriptorType::Sampler` or `DescriptorType::CombinedImageSampler`"
                            .into(),
                        vuids: &["VUID-VkDescriptorSetLayoutBinding-descriptorType-00282"],
                        ..Default::default()
                    });
                }
            }
        }

        if binding_flags.intersects(DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT) {
            if !device
                .enabled_features()
                .descriptor_binding_variable_descriptor_count
            {
                return Err(ValidationError {
                    context: "binding_flags".into(),
                    problem: "contains `DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT`".into(),
                    requires_one_of: RequiresOneOf {
                        features: &["descriptor_binding_variable_descriptor_count"],
                        ..Default::default()
                    },
                    vuids: &["VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBindingVariableDescriptorCount-03014"],
                });
            }

            if matches!(
                descriptor_type,
                DescriptorType::UniformBufferDynamic | DescriptorType::StorageBufferDynamic
            ) {
                return Err(ValidationError {
                    problem: "`binding_flags` contains \
                        `DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT`, and \
                        `descriptor_type` is `DescriptorType::UniformBufferDynamic` or \
                        `DescriptorType::StorageBufferDynamic`"
                        .into(),
                    vuids: &[
                        "VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-pBindingFlags-03015",
                    ],
                    ..Default::default()
                });
            }
        }

        Ok(())
    }
}

impl From<&DescriptorBindingRequirements> for DescriptorSetLayoutBinding {
    #[inline]
    fn from(reqs: &DescriptorBindingRequirements) -> Self {
        Self {
            binding_flags: DescriptorBindingFlags::empty(),
            descriptor_type: reqs.descriptor_types[0],
            descriptor_count: reqs.descriptor_count.unwrap_or(0),
            stages: reqs.stages,
            immutable_samplers: Vec::new(),
            _ne: crate::NonExhaustive(()),
        }
    }
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags that control how a binding in a descriptor set layout is created.
    DescriptorBindingFlags = DescriptorBindingFlags(u32);

    /* TODO: enable
    // TODO: document
    UPDATE_AFTER_BIND = UPDATE_AFTER_BIND {
        api_version: V1_2,
        device_extensions: [ext_descriptor_indexing],
    }, */

    /* TODO: enable
    // TODO: document
    UPDATE_UNUSED_WHILE_PENDING = UPDATE_UNUSED_WHILE_PENDING {
        api_version: V1_2,
        device_extensions: [ext_descriptor_indexing],
    }, */

    /* TODO: enable
    // TODO: document
    PARTIALLY_BOUND = PARTIALLY_BOUND {
        api_version: V1_2,
        device_extensions: [ext_descriptor_indexing],
    }, */

    /// Whether the binding has a variable number of descriptors.
    ///
    /// If set, the [`descriptor_binding_variable_descriptor_count`] feature must be
    /// enabled. The value of `descriptor_count` specifies the maximum number of descriptors
    /// allowed.
    ///
    /// There may only be one binding with a variable count in a descriptor set, and it must be the
    /// binding with the highest binding number. The `descriptor_type` must not be
    /// [`DescriptorType::UniformBufferDynamic`] or [`DescriptorType::StorageBufferDynamic`].
    ///
    /// [`descriptor_binding_variable_descriptor_count`]: crate::device::Features::descriptor_binding_variable_descriptor_count
    VARIABLE_DESCRIPTOR_COUNT = VARIABLE_DESCRIPTOR_COUNT {
        api_version: V1_2,
        device_extensions: [ext_descriptor_indexing],
    },
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

    /// Very similar to `UniformBuffer`, but the data is written directly into an inline buffer
    /// inside the descriptor set, instead of writing a reference to a buffer.
    /// This is similar to push constants, but because the data is stored in the descriptor set
    /// rather than the command buffer, it can be bound multiple times and reused, and you can
    /// have more than one of them bound in a single shader.
    ///
    /// It is not possible to have an arrayed binding of inline uniform blocks; at most one inline
    /// uniform block can be bound to one binding.
    ///
    /// Within a shader, an inline uniform block is defined exactly the same as a uniform buffer.
    /// The Vulkan API acts as if every byte in the inline buffer were its own descriptor:
    /// the `descriptor_count` value specifies the number of bytes available for data, and the
    /// `first_array_element` value when writing a descriptor set specifies the byte offset into
    /// the inline buffer. These values must always be a multiple of 4.
    InlineUniformBlock = INLINE_UNIFORM_BLOCK {
        api_version: V1_3,
        device_extensions: [ext_inline_uniform_block],
    },

    /// Gives read access to an acceleration structure, for performing ray queries and ray tracing.
    AccelerationStructure = ACCELERATION_STRUCTURE_KHR {
        device_extensions: [khr_acceleration_structure],
    },

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

impl DescriptorType {
    pub(crate) fn default_image_layout(self) -> ImageLayout {
        match self {
            DescriptorType::CombinedImageSampler
            | DescriptorType::SampledImage
            | DescriptorType::InputAttachment => ImageLayout::ShaderReadOnlyOptimal,
            DescriptorType::StorageImage => ImageLayout::General,
            DescriptorType::Sampler
            | DescriptorType::UniformTexelBuffer
            | DescriptorType::StorageTexelBuffer
            | DescriptorType::UniformBuffer
            | DescriptorType::StorageBuffer
            | DescriptorType::UniformBufferDynamic
            | DescriptorType::StorageBufferDynamic
            | DescriptorType::InlineUniformBlock
            | DescriptorType::AccelerationStructure => ImageLayout::Undefined,
        }
    }
}

/// Contains information about the level of support a device has for a particular descriptor set.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DescriptorSetLayoutSupport {
    /// If the queried descriptor set layout has a binding with the
    /// [`DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT`] flag set, then this indicates the
    /// maximum number of descriptors that binding could have. This is always at least as large
    /// as the descriptor count of the create info that was queried; if the queried descriptor
    /// count is higher than supported, `None` is returned instead of this structure.
    ///
    /// If the queried descriptor set layout does not have such a binding, or if the
    /// [`descriptor_binding_variable_descriptor_count`] feature isn't enabled on the device, this
    /// will be 0.
    ///
    /// [`descriptor_binding_variable_descriptor_count`]: crate::device::Features::descriptor_binding_variable_descriptor_count
    pub max_variable_descriptor_count: u32,
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
