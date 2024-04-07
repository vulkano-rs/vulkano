//! Describes the layout of all descriptors within a descriptor set.
//!
//! When creating a new descriptor set, you must provide a *layout* object to create it from.

use crate::{
    device::{Device, DeviceOwned},
    image::{sampler::Sampler, ImageLayout},
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_bitflags, vulkan_enum},
    shader::{DescriptorBindingRequirements, ShaderStages},
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version, VulkanError,
    VulkanObject,
};
use ahash::HashMap;
use std::{collections::BTreeMap, mem::MaybeUninit, num::NonZeroU64, ptr, sync::Arc};

/// Describes to the Vulkan implementation the layout of all descriptors within a descriptor set.
#[derive(Debug)]
pub struct DescriptorSetLayout {
    handle: ash::vk::DescriptorSetLayout,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
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
    ) -> Result<Arc<DescriptorSetLayout>, Validated<VulkanError>> {
        Self::validate_new(&device, &create_info)?;

        unsafe { Ok(Self::new_unchecked(device, create_info)?) }
    }

    fn validate_new(
        device: &Device,
        create_info: &DescriptorSetLayoutCreateInfo,
    ) -> Result<(), Box<ValidationError>> {
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
                return Err(Box::new(ValidationError {
                    problem: "the total number of descriptors across all bindings is greater than \
                        the `max_per_set_descriptors` limit, and \
                        `device.descriptor_set_layout_support` returned `None`"
                        .into(),
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        create_info: DescriptorSetLayoutCreateInfo,
    ) -> Result<Arc<DescriptorSetLayout>, VulkanError> {
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
                ..Default::default()
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
            create_info_vk.p_next = ptr::from_ref(next).cast();
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
            .map_err(VulkanError::from)?;
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
            device: InstanceOwnedDebugWrapper(device),
            id: Self::next_id(),
            flags,
            bindings,
            descriptor_counts,
        })
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
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            ref bindings,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkDescriptorSetLayoutCreateInfo-flags-parameter"])
        })?;

        // VUID-VkDescriptorSetLayoutCreateInfo-binding-00279
        // Ensured because it is a map

        let mut total_descriptor_count = 0;
        let highest_binding_num = bindings.keys().copied().next_back();
        let mut update_after_bind_binding = None;
        let mut buffer_dynamic_binding = None;

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
                    return Err(Box::new(ValidationError {
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
                    }));
                }

                if binding_flags.intersects(
                    DescriptorBindingFlags::UPDATE_AFTER_BIND
                        | DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING
                        | DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT,
                ) {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`flags` contains `DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR`, \
                            and `bindings[{}].binding_flags` contains \
                            `DescriptorBindingFlags::UPDATE_AFTER_BIND`, \
                            `DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING` or \
                            `DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT`",
                            binding_num
                        )
                        .into(),
                        vuids: &["VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-flags-03003"],
                        ..Default::default()
                    }));
                }
            }

            if binding_flags.intersects(DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT)
                && Some(binding_num) != highest_binding_num
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`bindings[{}].binding_flags` contains \
                        `DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT`, but {0} is not the
                        highest binding number in `bindings`",
                        binding_num
                    )
                    .into(),
                    vuids: &[
                        "VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-pBindingFlags-03004",
                    ],
                    ..Default::default()
                }));
            }

            if binding_flags.intersects(DescriptorBindingFlags::UPDATE_AFTER_BIND) {
                if !flags.intersects(DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL) {
                    return Err(Box::new(ValidationError {
                        problem: format!(
                            "`bindings[{}].binding_flags` contains \
                            `DescriptorBindingFlags::UPDATE_AFTER_BIND`, but \
                            `flags` does not contain \
                            `DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL`",
                            binding_num
                        )
                        .into(),
                        vuids: &["VUID-VkDescriptorSetLayoutCreateInfo-flags-03000"],
                        ..Default::default()
                    }));
                }

                update_after_bind_binding.get_or_insert(binding_num);
            }

            if matches!(
                descriptor_type,
                DescriptorType::UniformBufferDynamic | DescriptorType::StorageBufferDynamic
            ) {
                buffer_dynamic_binding.get_or_insert(binding_num);
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
            return Err(Box::new(ValidationError {
                problem: "`flags` contains `DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR`, and \
                    the total number of descriptors in `bindings` exceeds the \
                    `max_push_descriptors` limit"
                    .into(),
                vuids: &["VUID-VkDescriptorSetLayoutCreateInfo-flags-00281"],
                ..Default::default()
            }));
        }

        if let (Some(update_after_bind_binding), Some(buffer_dynamic_binding)) =
            (update_after_bind_binding, buffer_dynamic_binding)
        {
            return Err(Box::new(ValidationError {
                problem: format!(
                    "`bindings[{}].binding_flags` contains \
                    `DescriptorBindingFlags::UPDATE_AFTER_BIND`, and \
                    `bindings[{}].descriptor_type` is \
                    `DescriptorType::UniformBufferDynamic` or \
                    `DescriptorType::StorageBufferDynamic`",
                    update_after_bind_binding, buffer_dynamic_binding
                )
                .into(),
                vuids: &["VUID-VkDescriptorSetLayoutCreateInfo-descriptorType-03001"],
                ..Default::default()
            }));
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

    /// Whether descriptor sets using this descriptor set layout must be allocated from a
    /// descriptor pool whose flags contain [`DescriptorPoolCreateFlags::UPDATE_AFTER_BIND`].
    /// Descriptor set layouts with this flag use alternative (typically higher) device limits on
    /// per-stage and total descriptor counts, which have `_update_after_bind_` in their names.
    ///
    /// This flag must be specified whenever the layout contains one or more bindings that have
    /// the [`DescriptorBindingFlags::UPDATE_AFTER_BIND`] flag, but can be specified also if none
    /// of the bindings have this flag, purely to use the alternative device limits.
    ///
    /// [`DescriptorPoolCreateFlags::UPDATE_AFTER_BIND`]: crate::descriptor_set::pool::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND
    UPDATE_AFTER_BIND_POOL = UPDATE_AFTER_BIND_POOL
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_2)]),
        RequiresAllOf([DeviceExtension(ext_descriptor_indexing)]),
    ]),

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
    ///   [`max_push_descriptors`](crate::device::DeviceProperties::max_push_descriptors) limit.
    PUSH_DESCRIPTOR = PUSH_DESCRIPTOR_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_push_descriptor)]),
    ]),

    /* TODO: enable
    // TODO: document
    DESCRIPTOR_BUFFER = DESCRIPTOR_BUFFER_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_descriptor_buffer)]),
    ]), */

    /* TODO: enable
    // TODO: document
    EMBEDDED_IMMUTABLE_SAMPLERS = EMBEDDED_IMMUTABLE_SAMPLERS_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_descriptor_buffer)]),
    ]), */

    /* TODO: enable
    // TODO: document
    HOST_ONLY_POOL = HOST_ONLY_POOL_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_mutable_descriptor_type)]),
        RequiresAllOf([DeviceExtension(valve_mutable_descriptor_type)]),
    ]), */
}

/// A binding in a descriptor set layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DescriptorSetLayoutBinding {
    /// Specifies how to create the binding.
    ///
    /// The default value is empty.
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
    pub(crate) fn ensure_compatible_with_shader(
        &self,
        binding_requirements: &DescriptorBindingRequirements,
    ) -> Result<(), Box<ValidationError>> {
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
            return Err(Box::new(ValidationError {
                problem: "the descriptor type is not one of the types allowed by the \
                    descriptor binding requirements"
                    .into(),
                ..Default::default()
            }));
        }

        if let Some(required) = descriptor_count {
            if self.descriptor_count < required {
                return Err(Box::new(ValidationError {
                    problem: "the descriptor count is less than the count required by the \
                        descriptor binding requirements"
                        .into(),
                    ..Default::default()
                }));
            }
        }

        if !self.stages.contains(stages) {
            return Err(Box::new(ValidationError {
                problem: "the stages are not a superset of the stages required by the \
                    descriptor binding requirements"
                    .into(),
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            binding_flags,
            descriptor_type,
            descriptor_count,
            stages,
            ref immutable_samplers,
            _ne: _,
        } = self;

        binding_flags.validate_device(device).map_err(|err| {
            err.add_context("binding_flags").set_vuids(&[
                "VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-pBindingFlags-parameter",
            ])
        })?;

        descriptor_type.validate_device(device).map_err(|err| {
            err.add_context("descriptor_type")
                .set_vuids(&["VUID-VkDescriptorSetLayoutBinding-descriptorType-parameter"])
        })?;

        if descriptor_type == DescriptorType::InlineUniformBlock {
            if !device.enabled_features().inline_uniform_block {
                return Err(Box::new(ValidationError {
                    context: "descriptor_type".into(),
                    problem: "`DescriptorType::InlineUniformBlock`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "inline_uniform_block",
                    )])]),
                    vuids: &["VUID-VkDescriptorSetLayoutBinding-descriptorType-04604"],
                }));
            }

            if descriptor_count % 4 != 0 {
                return Err(Box::new(ValidationError {
                    problem: "`descriptor_type` is `DescriptorType::InlineUniformBlock`, and \
                        `descriptor_count` is not a multiple of 4"
                        .into(),
                    vuids: &["VUID-VkDescriptorSetLayoutBinding-descriptorType-02209"],
                    ..Default::default()
                }));
            }

            if descriptor_count
                > device
                    .physical_device()
                    .properties()
                    .max_inline_uniform_block_size
                    .unwrap_or(0)
            {
                return Err(Box::new(ValidationError {
                    problem: "`descriptor_type` is `DescriptorType::InlineUniformBlock`, and \
                        `descriptor_count` is greater than the `max_inline_uniform_block_size` \
                        limit"
                        .into(),
                    vuids: &["VUID-VkDescriptorSetLayoutBinding-descriptorType-08004"],
                    ..Default::default()
                }));
            }
        }

        if descriptor_count != 0 {
            stages.validate_device(device).map_err(|err| {
                err.add_context("stages")
                    .set_vuids(&["VUID-VkDescriptorSetLayoutBinding-descriptorCount-00283"])
            })?;

            if descriptor_type == DescriptorType::InputAttachment
                && !(stages.is_empty() || stages == ShaderStages::FRAGMENT)
            {
                return Err(Box::new(ValidationError {
                    problem: "`descriptor_type` is `DescriptorType::InputAttachment`, but \
                        `stages` is not either empty or equal to `ShaderStages::FRAGMENT`"
                        .into(),
                    vuids: &["VUID-VkDescriptorSetLayoutBinding-descriptorType-01510"],
                    ..Default::default()
                }));
            }
        }

        if !immutable_samplers.is_empty() {
            if descriptor_count != immutable_samplers.len() as u32 {
                return Err(Box::new(ValidationError {
                    problem: "`immutable_samplers` is not empty, but its length does not equal \
                        `descriptor_count`"
                        .into(),
                    vuids: &["VUID-VkDescriptorSetLayoutBinding-descriptorType-00282"],
                    ..Default::default()
                }));
            }

            let mut has_sampler_ycbcr_conversion = false;

            for sampler in immutable_samplers {
                assert_eq!(device, sampler.device().as_ref());

                has_sampler_ycbcr_conversion |= sampler.sampler_ycbcr_conversion().is_some();
            }

            if has_sampler_ycbcr_conversion {
                if !matches!(descriptor_type, DescriptorType::CombinedImageSampler) {
                    return Err(Box::new(ValidationError {
                        problem: "`immutable_samplers` contains a sampler with a \
                            sampler YCbCr conversion, but `descriptor_type` is not \
                            `DescriptorType::CombinedImageSampler`"
                            .into(),
                        vuids: &["VUID-VkDescriptorSetLayoutBinding-descriptorType-00282"],
                        ..Default::default()
                    }));
                }
            } else {
                if !matches!(
                    descriptor_type,
                    DescriptorType::Sampler | DescriptorType::CombinedImageSampler
                ) {
                    return Err(Box::new(ValidationError {
                        problem: "`immutable_samplers` is not empty, but `descriptor_type` is not \
                            `DescriptorType::Sampler` or `DescriptorType::CombinedImageSampler`"
                            .into(),
                        vuids: &["VUID-VkDescriptorSetLayoutBinding-descriptorType-00282"],
                        ..Default::default()
                    }));
                }
            }
        }

        if binding_flags.intersects(DescriptorBindingFlags::UPDATE_AFTER_BIND) {
            match descriptor_type {
                DescriptorType::UniformBuffer => {
                    if !device
                        .enabled_features()
                        .descriptor_binding_uniform_buffer_update_after_bind
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`binding_flags` contains \
                                `DescriptorBindingFlags::UPDATE_AFTER_BIND`, and \
                                `descriptor_type` is `DescriptorType::UniformBuffer`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                                "descriptor_binding_uniform_buffer_update_after_bind"
                            )])]),
                            vuids: &["VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBindingUniformBufferUpdateAfterBind-03005"],
                            ..Default::default()
                        }));
                    }
                }
                DescriptorType::Sampler
                | DescriptorType::CombinedImageSampler
                | DescriptorType::SampledImage => {
                    if !device
                        .enabled_features()
                        .descriptor_binding_sampled_image_update_after_bind
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`binding_flags` contains \
                                `DescriptorBindingFlags::UPDATE_AFTER_BIND`, and \
                                `descriptor_type` is `DescriptorType::Sampler`, \
                                `DescriptorType::CombinedImageSampler` or \
                                `DescriptorType::SampledImage`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                                "descriptor_binding_sampled_image_update_after_bind"
                            )])]),
                            vuids: &["VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBindingSampledImageUpdateAfterBind-03006"],
                            ..Default::default()
                        }));
                    }
                }
                DescriptorType::StorageImage => {
                    if !device
                        .enabled_features()
                        .descriptor_binding_storage_image_update_after_bind
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`binding_flags` contains \
                                `DescriptorBindingFlags::UPDATE_AFTER_BIND`, and \
                                `descriptor_type` is `DescriptorType::StorageImage`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                                "descriptor_binding_storage_image_update_after_bind"
                            )])]),
                            vuids: &["VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBindingStorageImageUpdateAfterBind-03007"],
                            ..Default::default()
                        }));
                    }
                }
                DescriptorType::StorageBuffer => {
                    if !device
                        .enabled_features()
                        .descriptor_binding_storage_buffer_update_after_bind
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`binding_flags` contains \
                                `DescriptorBindingFlags::UPDATE_AFTER_BIND`, and \
                                `descriptor_type` is `DescriptorType::StorageBuffer`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                                "descriptor_binding_storage_buffer_update_after_bind"
                            )])]),
                            vuids: &["VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBindingStorageBufferUpdateAfterBind-03008"],
                            ..Default::default()
                        }));
                    }
                }
                DescriptorType::UniformTexelBuffer => {
                    if !device
                        .enabled_features()
                        .descriptor_binding_uniform_texel_buffer_update_after_bind
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`binding_flags` contains \
                                `DescriptorBindingFlags::UPDATE_AFTER_BIND`, and \
                                `descriptor_type` is `DescriptorType::UniformTexelBuffer`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                                "descriptor_binding_uniform_texel_buffer_update_after_bind"
                            )])]),
                            vuids: &["VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBindingUniformTexelBufferUpdateAfterBind-03009"],
                            ..Default::default()
                        }));
                    }
                }
                DescriptorType::StorageTexelBuffer => {
                    if !device
                        .enabled_features()
                        .descriptor_binding_storage_texel_buffer_update_after_bind
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`binding_flags` contains \
                                `DescriptorBindingFlags::UPDATE_AFTER_BIND`, and \
                                `descriptor_type` is `DescriptorType::StorageTexelBuffer`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                                "descriptor_binding_storage_texel_buffer_update_after_bind"
                            )])]),
                            vuids: &["VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBindingStorageTexelBufferUpdateAfterBind-03010"],
                            ..Default::default()
                        }));
                    }
                }
                DescriptorType::InlineUniformBlock => {
                    if !device
                        .enabled_features()
                        .descriptor_binding_inline_uniform_block_update_after_bind
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`binding_flags` contains \
                                `DescriptorBindingFlags::UPDATE_AFTER_BIND`, and \
                                `descriptor_type` is `DescriptorType::InlineUniformBlock`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                                "descriptor_binding_inline_uniform_block_update_after_bind"
                            )])]),
                            vuids: &["VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBindingInlineUniformBlockUpdateAfterBind-02211"],
                            ..Default::default()
                        }));
                    }
                }
                DescriptorType::AccelerationStructure => {
                    if !device
                        .enabled_features()
                        .descriptor_binding_acceleration_structure_update_after_bind
                    {
                        return Err(Box::new(ValidationError {
                            problem: "`binding_flags` contains \
                                `DescriptorBindingFlags::UPDATE_AFTER_BIND`, and \
                                `descriptor_type` is `DescriptorType::AccelerationStructure`".into(),
                            requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                                "descriptor_binding_acceleration_structure_update_after_bind"
                            )])]),
                            vuids: &["VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBindingAccelerationStructureUpdateAfterBind-03570"],
                            ..Default::default()
                        }));
                    }
                }
                DescriptorType::InputAttachment
                | DescriptorType::UniformBufferDynamic
                | DescriptorType::StorageBufferDynamic => {
                    return Err(Box::new(ValidationError {
                        problem: "`binding_flags` contains \
                            `DescriptorBindingFlags::UPDATE_AFTER_BIND`, and \
                            `descriptor_type` is `DescriptorType::InputAttachment`, \
                            `DescriptorType::UniformBufferDynamic` or \
                            `DescriptorType::StorageBufferDynamic`"
                            .into(),
                        vuids: &["VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-None-03011"],
                        ..Default::default()
                    }));
                }
            }
        }

        if binding_flags.intersects(DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING)
            && !device
                .enabled_features()
                .descriptor_binding_update_unused_while_pending
        {
            return Err(Box::new(ValidationError {
                context: "binding_flags".into(),
                problem: "contains `DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "descriptor_binding_update_unused_while_pending"
                )])]),
                vuids: &["VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBindingUpdateUnusedWhilePending-03012"],
            }));
        }

        if binding_flags.intersects(DescriptorBindingFlags::PARTIALLY_BOUND)
            && !device.enabled_features().descriptor_binding_partially_bound
        {
            return Err(Box::new(ValidationError {
                context: "binding_flags".into(),
                problem: "contains `DescriptorBindingFlags::PARTIALLY_BOUND`".into(),
                requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                    "descriptor_binding_partially_bound"
                )])]),
                vuids: &["VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBindingPartiallyBound-03013"],
            }));
        }

        if binding_flags.intersects(DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT) {
            if !device
                .enabled_features()
                .descriptor_binding_variable_descriptor_count
            {
                return Err(Box::new(ValidationError {
                    context: "binding_flags".into(),
                    problem: "contains `DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT`".into(),
                    requires_one_of: RequiresOneOf(&[RequiresAllOf(&[Requires::DeviceFeature(
                        "descriptor_binding_variable_descriptor_count"
                    )])]),
                    vuids: &["VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBindingVariableDescriptorCount-03014"],
                }));
            }

            if matches!(
                descriptor_type,
                DescriptorType::UniformBufferDynamic | DescriptorType::StorageBufferDynamic
            ) {
                return Err(Box::new(ValidationError {
                    problem: "`binding_flags` contains \
                        `DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT`, and \
                        `descriptor_type` is `DescriptorType::UniformBufferDynamic` or \
                        `DescriptorType::StorageBufferDynamic`"
                        .into(),
                    vuids: &[
                        "VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-pBindingFlags-03015",
                    ],
                    ..Default::default()
                }));
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

    /// Allows descriptors in this binding to be updated after a command buffer has already
    /// recorded a bind command containing a descriptor set with this layout, as long as the
    /// command buffer is not executing. Each descriptor can also be updated concurrently.
    ///
    /// If a binding has this flag, then the descriptor set layout must be created with the
    /// [`DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL`] flag, and descriptor sets using
    /// it must be allocated from a descriptor pool that has the
    /// [`DescriptorPoolCreateFlags::UPDATE_AFTER_BIND`] flag.
    /// In addition, the `descriptor_binding_*_update_after_bind` feature corresponding to the
    /// descriptor type of the binding must be enabled.
    ///
    /// [`DescriptorPoolCreateFlags::UPDATE_AFTER_BIND`]: crate::descriptor_set::pool::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND
    UPDATE_AFTER_BIND = UPDATE_AFTER_BIND
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_2)]),
        RequiresAllOf([DeviceExtension(ext_descriptor_indexing)]),
    ]),

    /// Allows descriptors in this binding to be updated after a command buffer has already
    /// recorded a bind command containing a descriptor set with this layout, as long as the
    /// command buffer is not executing, and no shader invocation recorded in the command buffer
    /// *uses* the descriptor. Each descriptor can also be updated concurrently.
    ///
    /// This is a subset of what is allowed by [`DescriptorBindingFlags::UPDATE_AFTER_BIND`], but
    /// has much less strict requirements. It does not require any additional flags to be present
    /// on the descriptor set layout or the descriptor pool, and instead requires the
    /// [`descriptor_binding_update_unused_while_pending`] feature.
    ///
    /// What counts as "used" depends on whether the [`DescriptorBindingFlags::PARTIALLY_BOUND`]
    /// flag is also set. If it is set, then only *dynamic use* by a shader invocation counts as
    /// being used, otherwise all *static use* by a shader invocation is considered used.
    ///
    /// [`descriptor_binding_update_unused_while_pending`]: crate::device::DeviceFeatures::descriptor_binding_update_unused_while_pending
    UPDATE_UNUSED_WHILE_PENDING = UPDATE_UNUSED_WHILE_PENDING
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_2)]),
        RequiresAllOf([DeviceExtension(ext_descriptor_indexing)]),
    ]),

    /// Allows descriptors to be left empty or invalid even if they are *statically used* by a
    /// shader invocation, as long as they are not *dynamically used* . Additionally, if
    /// [`DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING`] is set, allows updating descriptors
    /// if they are statically used by a command buffer they are recorded in, as long as are not
    /// dynamically used.
    ///
    /// The [`descriptor_binding_partially_bound`] feature must be enabled on the device.
    ///
    /// [`descriptor_binding_partially_bound`]: crate::device::DeviceFeatures::descriptor_binding_partially_bound
    PARTIALLY_BOUND = PARTIALLY_BOUND
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_2)]),
        RequiresAllOf([DeviceExtension(ext_descriptor_indexing)]),
    ]),

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
    /// [`descriptor_binding_variable_descriptor_count`]: crate::device::DeviceFeatures::descriptor_binding_variable_descriptor_count
    VARIABLE_DESCRIPTOR_COUNT = VARIABLE_DESCRIPTOR_COUNT
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_2)]),
        RequiresAllOf([DeviceExtension(ext_descriptor_indexing)]),
    ]),
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
    /// - The `descriptor_count` value specifies the number of bytes available for data.
    /// - The `variable_descriptor_count` value when allocating a descriptor set specifies a
    ///   variable byte count instead.
    /// - The `first_array_element` value when writing a descriptor set specifies the byte offset
    ///   into the inline buffer.
    /// These values must always be a multiple of 4.
    InlineUniformBlock = INLINE_UNIFORM_BLOCK
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_inline_uniform_block)]),
    ]),

    /* TODO: enable
    // TODO: document
    InlineUniformBlock = INLINE_UNIFORM_BLOCK
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_3)]),
        RequiresAllOf([DeviceExtension(ext_inline_uniform_block)]),
    ]),*/

    /// Gives read access to an acceleration structure, for performing ray queries and ray tracing.
    AccelerationStructure = ACCELERATION_STRUCTURE_KHR
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(khr_acceleration_structure)]),
    ]),

    /* TODO: enable
    // TODO: document
    AccelerationStructureNV = ACCELERATION_STRUCTURE_NV
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(nv_ray_tracing)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    SampleWeightImage = SAMPLE_WEIGHT_IMAGE_QCOM
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(qcom_image_processing)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    BlockMatchImage = BLOCK_MATCH_IMAGE_QCOM
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(qcom_image_processing)]),
    ]),*/

    /* TODO: enable
    // TODO: document
    Mutable = MUTABLE_VALVE
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(valve_mutable_descriptor_type)]),
    ]),*/
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
    /// [`descriptor_binding_variable_descriptor_count`]: crate::device::DeviceFeatures::descriptor_binding_variable_descriptor_count
    pub max_variable_descriptor_count: u32,
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
