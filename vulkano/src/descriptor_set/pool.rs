use crate::{
    descriptor_set::{
        layout::{DescriptorSetLayout, DescriptorSetLayoutCreateFlags, DescriptorType},
        sys::RawDescriptorSet,
    },
    device::{Device, DeviceOwned, DeviceOwnedDebugWrapper},
    instance::InstanceOwnedDebugWrapper,
    macros::{impl_id_counter, vulkan_bitflags},
    Requires, RequiresAllOf, RequiresOneOf, Validated, ValidationError, Version, VulkanError,
    VulkanObject,
};
use ahash::HashMap;
use smallvec::SmallVec;
use std::{cell::Cell, marker::PhantomData, mem::MaybeUninit, num::NonZeroU64, ptr, sync::Arc};

/// Pool that descriptors are allocated from.
///
/// A pool has a maximum number of descriptor sets and a maximum number of descriptors (one value
/// per descriptor type) it can allocate.
#[derive(Debug)]
pub struct DescriptorPool {
    handle: ash::vk::DescriptorPool,
    device: InstanceOwnedDebugWrapper<Arc<Device>>,
    id: NonZeroU64,

    flags: DescriptorPoolCreateFlags,
    max_sets: u32,
    pool_sizes: HashMap<DescriptorType, u32>,
    max_inline_uniform_block_bindings: u32,

    // Unimplement `Sync`, as Vulkan descriptor pools are not thread safe.
    _marker: PhantomData<Cell<ash::vk::DescriptorPool>>,
}

impl DescriptorPool {
    /// Creates a new `DescriptorPool`.
    #[inline]
    pub fn new(
        device: Arc<Device>,
        create_info: DescriptorPoolCreateInfo,
    ) -> Result<DescriptorPool, Validated<VulkanError>> {
        Self::validate_new(&device, &create_info)?;

        unsafe { Ok(Self::new_unchecked(device, create_info)?) }
    }

    fn validate_new(
        device: &Device,
        create_info: &DescriptorPoolCreateInfo,
    ) -> Result<(), Box<ValidationError>> {
        // VUID-vkCreateDescriptorPool-pCreateInfo-parameter
        create_info
            .validate(device)
            .map_err(|err| err.add_context("create_info"))?;

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn new_unchecked(
        device: Arc<Device>,
        create_info: DescriptorPoolCreateInfo,
    ) -> Result<DescriptorPool, VulkanError> {
        let create_info_fields1_vk = create_info.to_vk_fields1();
        let mut create_info_extensions_vk = create_info.to_vk_extensions();
        let create_info_vk =
            create_info.to_vk(&create_info_fields1_vk, &mut create_info_extensions_vk);

        let handle = unsafe {
            let fns = device.fns();
            let mut output = MaybeUninit::uninit();
            (fns.v1_0.create_descriptor_pool)(
                device.handle(),
                &create_info_vk,
                ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        unsafe { Ok(Self::from_handle(device, handle, create_info)) }
    }

    /// Creates a new `DescriptorPool` from a raw object handle.
    ///
    /// # Safety
    ///
    /// - `handle` must be a valid Vulkan object handle created from `device`.
    /// - `create_info` must match the info used to create the object.
    #[inline]
    pub unsafe fn from_handle(
        device: Arc<Device>,
        handle: ash::vk::DescriptorPool,
        create_info: DescriptorPoolCreateInfo,
    ) -> DescriptorPool {
        let DescriptorPoolCreateInfo {
            flags,
            max_sets,
            pool_sizes,
            max_inline_uniform_block_bindings,
            _ne: _,
        } = create_info;

        DescriptorPool {
            handle,
            device: InstanceOwnedDebugWrapper(device),
            id: Self::next_id(),

            flags,
            max_sets,
            pool_sizes,
            max_inline_uniform_block_bindings,

            _marker: PhantomData,
        }
    }

    /// Returns the flags that the descriptor pool was created with.
    #[inline]
    pub fn flags(&self) -> DescriptorPoolCreateFlags {
        self.flags
    }

    /// Returns the maximum number of sets that can be allocated from the pool.
    #[inline]
    pub fn max_sets(&self) -> u32 {
        self.max_sets
    }

    /// Returns the number of descriptors of each type that the pool was created with.
    #[inline]
    pub fn pool_sizes(&self) -> &HashMap<DescriptorType, u32> {
        &self.pool_sizes
    }

    /// Returns the maximum number of [`DescriptorType::InlineUniformBlock`] bindings that can
    /// be allocated from the descriptor pool.
    #[inline]
    pub fn max_inline_uniform_block_bindings(&self) -> u32 {
        self.max_inline_uniform_block_bindings
    }

    /// Allocates descriptor sets from the pool, one for each element in `allocate_info`.
    /// Returns an iterator to the allocated sets, or an error.
    ///
    /// The `FragmentedPool` errors often can't be prevented. If the function returns this error,
    /// you should just create a new pool.
    ///
    /// # Safety
    ///
    /// - When the pool is dropped, the returned descriptor sets must not be in use by either the
    ///   host or device.
    /// - If the device API version is less than 1.1, and the [`khr_maintenance1`] extension is not
    ///   enabled on the device, then the length of `allocate_infos` must not be greater than the
    ///   number of descriptor sets remaining in the pool, and the total number of descriptors of
    ///   each type being allocated must not be greater than the number of descriptors of that type
    ///   remaining in the pool.
    ///
    /// [`khr_maintenance1`]: crate::device::DeviceExtensions::khr_maintenance1
    #[inline]
    pub unsafe fn allocate_descriptor_sets(
        &self,
        allocate_infos: impl IntoIterator<Item = DescriptorSetAllocateInfo>,
    ) -> Result<impl ExactSizeIterator<Item = DescriptorPoolAlloc>, Validated<VulkanError>> {
        let allocate_infos: SmallVec<[_; 1]> = allocate_infos.into_iter().collect();
        self.validate_allocate_descriptor_sets(&allocate_infos)?;

        Ok(self.allocate_descriptor_sets_unchecked(allocate_infos)?)
    }

    fn validate_allocate_descriptor_sets(
        &self,
        allocate_infos: &[DescriptorSetAllocateInfo],
    ) -> Result<(), Box<ValidationError>> {
        for (index, info) in allocate_infos.iter().enumerate() {
            info.validate(self.device())
                .map_err(|err| err.add_context(format!("allocate_infos[{}]", index)))?;

            let &DescriptorSetAllocateInfo {
                ref layout,
                variable_descriptor_count: _,
                _ne: _,
            } = info;

            if layout
                .flags()
                .intersects(DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                && !self
                    .flags
                    .intersects(DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            {
                return Err(Box::new(ValidationError {
                    problem: format!(
                        "`allocate_infos[{}].layout.flags()` contains \
                        `DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL`, but \
                        `self.flags` does not contain \
                        `DescriptorPoolCreateFlags::UPDATE_AFTER_BIND`",
                        index
                    )
                    .into(),
                    vuids: &["VUID-VkDescriptorSetAllocateInfo-pSetLayouts-03044"],
                    ..Default::default()
                }));
            }
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn allocate_descriptor_sets_unchecked(
        &self,
        allocate_infos: impl IntoIterator<Item = DescriptorSetAllocateInfo>,
    ) -> Result<impl ExactSizeIterator<Item = DescriptorPoolAlloc>, VulkanError> {
        let allocate_infos = allocate_infos.into_iter();

        let (lower_size_bound, _) = allocate_infos.size_hint();
        let mut layouts_vk: SmallVec<[_; 1]> = SmallVec::with_capacity(lower_size_bound);
        let mut variable_descriptor_counts: SmallVec<[_; 1]> =
            SmallVec::with_capacity(lower_size_bound);
        let mut layouts: SmallVec<[_; 1]> = SmallVec::with_capacity(lower_size_bound);

        for info in allocate_infos {
            let DescriptorSetAllocateInfo {
                layout,
                variable_descriptor_count,
                _ne: _,
            } = info;

            layouts_vk.push(layout.handle());
            variable_descriptor_counts.push(variable_descriptor_count);
            layouts.push(layout);
        }

        let mut output: SmallVec<[_; 1]> = SmallVec::new();

        if !layouts_vk.is_empty() {
            let mut variable_desc_count_alloc_info = None;

            let mut info_vk = ash::vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.handle)
                .set_layouts(&layouts_vk);

            if (self.device.api_version() >= Version::V1_2
                || self.device.enabled_extensions().ext_descriptor_indexing)
                && variable_descriptor_counts.iter().any(|c| *c != 0)
            {
                let next = variable_desc_count_alloc_info.insert(
                    ash::vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
                        .descriptor_counts(&variable_descriptor_counts),
                );
                info_vk = info_vk.push_next(next);
            }

            output.reserve(layouts_vk.len());

            let fns = self.device.fns();
            (fns.v1_0.allocate_descriptor_sets)(
                self.device.handle(),
                &info_vk,
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)
            .map_err(|err| match err {
                VulkanError::OutOfHostMemory
                | VulkanError::OutOfDeviceMemory
                | VulkanError::OutOfPoolMemory => err,
                // According to the specs, because `VK_ERROR_FRAGMENTED_POOL` was added after
                // version 1.0 of Vulkan, any negative return value except out-of-memory errors
                // must be considered as a fragmented pool error.
                _ => VulkanError::FragmentedPool,
            })?;

            output.set_len(layouts_vk.len());
        }

        Ok(output
            .into_iter()
            .zip(layouts)
            .zip(variable_descriptor_counts)
            .map(
                move |((handle, layout), variable_descriptor_count)| DescriptorPoolAlloc {
                    handle,
                    id: DescriptorPoolAlloc::next_id(),
                    layout: DeviceOwnedDebugWrapper(layout),
                    variable_descriptor_count,
                },
            ))
    }

    /// Frees some descriptor sets.
    ///
    /// Note that it is not mandatory to free sets. Destroying or resetting the pool destroys all
    /// the descriptor sets.
    ///
    /// # Safety
    ///
    /// - All elements of `descriptor_sets` must have been allocated from `self`, and not freed
    ///   previously.
    /// - All elements of `descriptor_sets` must not be in use by the host or device.
    #[inline]
    pub unsafe fn free_descriptor_sets(
        &self,
        descriptor_sets: impl IntoIterator<Item = RawDescriptorSet>,
    ) -> Result<(), Validated<VulkanError>> {
        self.validate_free_descriptor_sets()?;

        Ok(self.free_descriptor_sets_unchecked(descriptor_sets)?)
    }

    fn validate_free_descriptor_sets(&self) -> Result<(), Box<ValidationError>> {
        if !self
            .flags
            .intersects(DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
        {
            return Err(Box::new(ValidationError {
                context: "self.flags()".into(),
                problem: "does not contain `DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET`".into(),
                vuids: &["VUID-vkFreeDescriptorSets-descriptorPool-00312"],
                ..Default::default()
            }));
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "document_unchecked"), doc(hidden))]
    pub unsafe fn free_descriptor_sets_unchecked(
        &self,
        descriptor_sets: impl IntoIterator<Item = RawDescriptorSet>,
    ) -> Result<(), VulkanError> {
        let sets: SmallVec<[_; 8]> = descriptor_sets.into_iter().map(|s| s.handle()).collect();
        if !sets.is_empty() {
            let fns = self.device.fns();
            (fns.v1_0.free_descriptor_sets)(
                self.device.handle(),
                self.handle,
                sets.len() as u32,
                sets.as_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
        }

        Ok(())
    }

    /// Resets the pool.
    ///
    /// This destroys all descriptor sets and empties the pool.
    ///
    /// # Safety
    ///
    /// - All descriptor sets that were previously allocated from `self` must not be in use by the
    ///   host or device.
    #[inline]
    pub unsafe fn reset(&self) -> Result<(), VulkanError> {
        let fns = self.device.fns();
        (fns.v1_0.reset_descriptor_pool)(
            self.device.handle(),
            self.handle,
            ash::vk::DescriptorPoolResetFlags::empty(),
        )
        .result()
        .map_err(VulkanError::from)?;

        Ok(())
    }
}

impl Drop for DescriptorPool {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.device.fns();
            (fns.v1_0.destroy_descriptor_pool)(self.device.handle(), self.handle, ptr::null());
        }
    }
}

unsafe impl VulkanObject for DescriptorPool {
    type Handle = ash::vk::DescriptorPool;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for DescriptorPool {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

impl_id_counter!(DescriptorPool);

/// Parameters to create a new `DescriptorPool`.
#[derive(Clone, Debug)]
pub struct DescriptorPoolCreateInfo {
    /// Additional properties of the descriptor pool.
    ///
    /// The default value is empty.
    pub flags: DescriptorPoolCreateFlags,

    /// The maximum number of descriptor sets that can be allocated from the pool.
    ///
    /// The default value is `0`, which must be overridden.
    pub max_sets: u32,

    /// The number of descriptors of each type to allocate for the pool.
    ///
    /// If the descriptor type is [`DescriptorType::InlineUniformBlock`], then the value is the
    /// number of bytes to allocate for such descriptors. The value must then be a multiple of 4.
    ///
    /// The default value is empty, which must be overridden.
    pub pool_sizes: HashMap<DescriptorType, u32>,

    /// The maximum number of [`DescriptorType::InlineUniformBlock`] bindings that can be allocated
    /// from the descriptor pool.
    ///
    /// If this is not 0, the device API version must be at least 1.3, or the
    /// [`khr_inline_uniform_block`](crate::device::DeviceExtensions::ext_inline_uniform_block)
    /// extension must be enabled on the device.
    ///
    /// The default value is 0.
    pub max_inline_uniform_block_bindings: u32,

    pub _ne: crate::NonExhaustive,
}

impl Default for DescriptorPoolCreateInfo {
    #[inline]
    fn default() -> Self {
        Self {
            flags: DescriptorPoolCreateFlags::empty(),
            max_sets: 0,
            pool_sizes: HashMap::default(),
            max_inline_uniform_block_bindings: 0,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl DescriptorPoolCreateInfo {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            flags,
            max_sets,
            ref pool_sizes,
            max_inline_uniform_block_bindings,
            _ne: _,
        } = self;

        flags.validate_device(device).map_err(|err| {
            err.add_context("flags")
                .set_vuids(&["VUID-VkDescriptorPoolCreateInfo-flags-parameter"])
        })?;

        if max_sets == 0 {
            return Err(Box::new(ValidationError {
                context: "max_sets".into(),
                problem: "is zero".into(),
                vuids: &["VUID-VkDescriptorPoolCreateInfo-maxSets-00301"],
                ..Default::default()
            }));
        }

        if pool_sizes.is_empty() {
            return Err(Box::new(ValidationError {
                context: "pool_sizes".into(),
                problem: "is empty".into(),
                // vuids?
                ..Default::default()
            }));
        }

        // VUID-VkDescriptorPoolCreateInfo-pPoolSizes-parameter
        for (&descriptor_type, &pool_size) in pool_sizes.iter() {
            flags.validate_device(device).map_err(|err| {
                err.add_context("pool_sizes")
                    .set_vuids(&["VUID-VkDescriptorPoolSize-type-parameter"])
            })?;

            if pool_size == 0 {
                return Err(Box::new(ValidationError {
                    context: format!("pool_sizes[DescriptorType::{:?}]", descriptor_type).into(),
                    problem: "is zero".into(),
                    vuids: &["VUID-VkDescriptorPoolSize-descriptorCount-00302"],
                    ..Default::default()
                }));
            }

            if descriptor_type == DescriptorType::InlineUniformBlock {
                return Err(Box::new(ValidationError {
                    context: "pool_sizes[DescriptorType::InlineUniformBlock]".into(),
                    problem: "is not a multiple of 4".into(),
                    vuids: &["VUID-VkDescriptorPoolSize-type-02218"],
                    ..Default::default()
                }));
            }
        }

        if max_inline_uniform_block_bindings != 0
            && !(device.api_version() >= Version::V1_3
                || device.enabled_extensions().ext_inline_uniform_block)
        {
            return Err(Box::new(ValidationError {
                context: "max_inline_uniform_block_bindings".into(),
                problem: "is not zero".into(),
                requires_one_of: RequiresOneOf(&[
                    RequiresAllOf(&[Requires::APIVersion(Version::V1_3)]),
                    RequiresAllOf(&[Requires::DeviceExtension("ext_inline_uniform_block")]),
                ]),
                // vuids?
                ..Default::default()
            }));
        }

        Ok(())
    }

    pub(crate) fn to_vk<'a>(
        &self,
        fields1_vk: &'a DescriptorPoolCreateInfoFields1Vk,
        extensions_vk: &'a mut DescriptorPoolCreateInfoExtensionsVk,
    ) -> ash::vk::DescriptorPoolCreateInfo<'a> {
        let &Self {
            flags,
            max_sets,
            pool_sizes: _,
            max_inline_uniform_block_bindings: _,
            _ne: _,
        } = self;
        let DescriptorPoolCreateInfoFields1Vk { pool_sizes_vk } = fields1_vk;

        let mut val_vk = ash::vk::DescriptorPoolCreateInfo::default()
            .flags(flags.into())
            .max_sets(max_sets)
            .pool_sizes(pool_sizes_vk);

        let DescriptorPoolCreateInfoExtensionsVk {
            inline_uniform_block_vk,
        } = extensions_vk;

        if let Some(next) = inline_uniform_block_vk {
            val_vk = val_vk.push_next(next);
        }

        val_vk
    }

    pub(crate) fn to_vk_fields1(&self) -> DescriptorPoolCreateInfoFields1Vk {
        let pool_sizes_vk = self
            .pool_sizes
            .iter()
            .map(|(&ty, &descriptor_count)| ash::vk::DescriptorPoolSize {
                ty: ty.into(),
                descriptor_count,
            })
            .collect();

        DescriptorPoolCreateInfoFields1Vk { pool_sizes_vk }
    }

    pub(crate) fn to_vk_extensions(&self) -> DescriptorPoolCreateInfoExtensionsVk {
        let inline_uniform_block_vk = (self.max_inline_uniform_block_bindings != 0).then(|| {
            ash::vk::DescriptorPoolInlineUniformBlockCreateInfo::default()
                .max_inline_uniform_block_bindings(self.max_inline_uniform_block_bindings)
        });

        DescriptorPoolCreateInfoExtensionsVk {
            inline_uniform_block_vk,
        }
    }
}

pub(crate) struct DescriptorPoolCreateInfoExtensionsVk {
    pub(crate) inline_uniform_block_vk:
        Option<ash::vk::DescriptorPoolInlineUniformBlockCreateInfo<'static>>,
}

pub(crate) struct DescriptorPoolCreateInfoFields1Vk {
    pub(crate) pool_sizes_vk: SmallVec<[ash::vk::DescriptorPoolSize; 8]>,
}

vulkan_bitflags! {
    #[non_exhaustive]

    /// Flags specifying additional properties of a descriptor pool.
    DescriptorPoolCreateFlags = DescriptorPoolCreateFlags(u32);

    /// Individual descriptor sets can be freed from the pool. Otherwise you must reset or
    /// destroy the whole pool at once.
    FREE_DESCRIPTOR_SET = FREE_DESCRIPTOR_SET,

    /// The pool can allocate descriptor sets with a layout whose flags include
    /// [`DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL`].
    ///
    /// A pool created with this flag can still allocate descriptor sets without the flag.
    /// However, descriptor copy operations are only allowed between pools of the same type;
    /// it is not possible to copy between a descriptor set whose pool has `UPDATE_AFTER_BIND`,
    /// and a descriptor set whose pool does not have this flag.
    UPDATE_AFTER_BIND = UPDATE_AFTER_BIND
    RequiresOneOf([
        RequiresAllOf([APIVersion(V1_2)]),
        RequiresAllOf([DeviceExtension(ext_descriptor_indexing)]),
    ]),

    /* TODO: enable
    // TODO: document
    HOST_ONLY = HOST_ONLY_EXT
    RequiresOneOf([
        RequiresAllOf([DeviceExtension(ext_mutable_descriptor_type)]),
        RequiresAllOf([DeviceExtension(valve_mutable_descriptor_type)]),
    ]), */
}

/// Parameters to allocate a new `DescriptorPoolAlloc` from a `DescriptorPool`.
#[derive(Clone, Debug)]
pub struct DescriptorSetAllocateInfo {
    /// The descriptor set layout to create the set for.
    ///
    /// There is no default value.
    pub layout: Arc<DescriptorSetLayout>,

    /// For layouts with a variable-count binding, the number of descriptors to allocate for that
    /// binding. This should be 0 for layouts that don't have a variable-count binding.
    ///
    /// The default value is 0.
    pub variable_descriptor_count: u32,

    pub _ne: crate::NonExhaustive,
}

impl DescriptorSetAllocateInfo {
    /// Returns a `DescriptorSetAllocateInfo` with the specified `layout`.
    #[inline]
    pub fn new(layout: Arc<DescriptorSetLayout>) -> Self {
        Self {
            layout,
            variable_descriptor_count: 0,
            _ne: crate::NonExhaustive(()),
        }
    }

    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            ref layout,
            variable_descriptor_count,
            _ne,
        } = self;

        // VUID-VkDescriptorSetAllocateInfo-commonparent
        assert_eq!(device, layout.device().as_ref());

        if layout
            .flags()
            .intersects(DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR)
        {
            return Err(Box::new(ValidationError {
                context: "layout.flags()".into(),
                problem: "contains `DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR`".into(),
                vuids: &["VUID-VkDescriptorSetAllocateInfo-pSetLayouts-00308"],
                ..Default::default()
            }));
        }

        if variable_descriptor_count > layout.variable_descriptor_count() {
            return Err(Box::new(ValidationError {
                problem: "`variable_descriptor_count` is greater than
                    `layout.variable_descriptor_count()`"
                    .into(),
                // vuids: https://github.com/KhronosGroup/Vulkan-Docs/issues/2244
                ..Default::default()
            }));
        }

        Ok(())
    }
}

/// Opaque type that represents a descriptor set allocated from a pool.
#[derive(Debug)]
pub struct DescriptorPoolAlloc {
    handle: ash::vk::DescriptorSet,
    id: NonZeroU64,
    layout: DeviceOwnedDebugWrapper<Arc<DescriptorSetLayout>>,
    variable_descriptor_count: u32,
}

impl DescriptorPoolAlloc {
    /// Returns the descriptor set layout of the descriptor set.
    #[inline]
    pub fn layout(&self) -> &Arc<DescriptorSetLayout> {
        &self.layout
    }

    /// Returns the variable descriptor count that this descriptor set was allocated with.
    #[inline]
    pub fn variable_descriptor_count(&self) -> u32 {
        self.variable_descriptor_count
    }
}

unsafe impl VulkanObject for DescriptorPoolAlloc {
    type Handle = ash::vk::DescriptorSet;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.handle
    }
}

unsafe impl DeviceOwned for DescriptorPoolAlloc {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.layout.device()
    }
}

impl_id_counter!(DescriptorPoolAlloc);

#[cfg(test)]
mod tests {
    use super::{DescriptorPool, DescriptorPoolCreateInfo};
    use crate::{
        descriptor_set::{
            layout::{
                DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
                DescriptorType,
            },
            pool::DescriptorSetAllocateInfo,
        },
        shader::ShaderStages,
    };

    #[test]
    fn pool_create() {
        let (device, _) = gfx_dev_and_queue!();

        let _ = DescriptorPool::new(
            device,
            DescriptorPoolCreateInfo {
                max_sets: 10,
                pool_sizes: [(DescriptorType::UniformBuffer, 1)].into_iter().collect(),
                ..Default::default()
            },
        )
        .unwrap();
    }

    #[test]
    fn zero_max_set() {
        let (device, _) = gfx_dev_and_queue!();

        DescriptorPool::new(
            device,
            DescriptorPoolCreateInfo {
                max_sets: 0,
                pool_sizes: [(DescriptorType::UniformBuffer, 1)].into_iter().collect(),
                ..Default::default()
            },
        )
        .unwrap_err();
    }

    #[test]
    fn zero_descriptors() {
        let (device, _) = gfx_dev_and_queue!();

        DescriptorPool::new(
            device,
            DescriptorPoolCreateInfo {
                max_sets: 10,
                ..Default::default()
            },
        )
        .unwrap_err();
    }

    #[test]
    fn basic_alloc() {
        let (device, _) = gfx_dev_and_queue!();

        let set_layout = DescriptorSetLayout::new(
            device.clone(),
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

        let pool = DescriptorPool::new(
            device,
            DescriptorPoolCreateInfo {
                max_sets: 10,
                pool_sizes: [(DescriptorType::UniformBuffer, 10)].into_iter().collect(),
                ..Default::default()
            },
        )
        .unwrap();
        unsafe {
            let sets = pool
                .allocate_descriptor_sets([DescriptorSetAllocateInfo::new(set_layout)])
                .unwrap();
            assert_eq!(sets.count(), 1);
        }
    }

    #[test]
    fn alloc_diff_device() {
        let (device1, _) = gfx_dev_and_queue!();
        let (device2, _) = gfx_dev_and_queue!();

        let set_layout = DescriptorSetLayout::new(
            device1,
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

        assert_should_panic!({
            let pool = DescriptorPool::new(
                device2,
                DescriptorPoolCreateInfo {
                    max_sets: 10,
                    pool_sizes: [(DescriptorType::UniformBuffer, 10)].into_iter().collect(),
                    ..Default::default()
                },
            )
            .unwrap();

            unsafe {
                let _ = pool.allocate_descriptor_sets([DescriptorSetAllocateInfo::new(set_layout)]);
            }
        });
    }

    #[test]
    fn alloc_zero() {
        let (device, _) = gfx_dev_and_queue!();

        let pool = DescriptorPool::new(
            device,
            DescriptorPoolCreateInfo {
                max_sets: 1,
                pool_sizes: [(DescriptorType::UniformBuffer, 1)].into_iter().collect(),
                ..Default::default()
            },
        )
        .unwrap();
        unsafe {
            let sets = pool.allocate_descriptor_sets([]).unwrap();
            assert_eq!(sets.count(), 0);
        }
    }
}
