use crate::{
    resource::{ResourceStorage, Resources},
    Id, Ref,
};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use concurrent_slotmap::{hyaline, SlotMap};
use foldhash::HashMap;
use std::{collections::BTreeMap, iter, mem, sync::Arc};
use vulkano::{
    acceleration_structure::AccelerationStructure,
    buffer::{Buffer, Subbuffer},
    descriptor_set::{
        allocator::{AllocationHandle, DescriptorSetAlloc, DescriptorSetAllocator},
        layout::{
            DescriptorBindingFlags, DescriptorSetLayout, DescriptorSetLayoutBinding,
            DescriptorSetLayoutCreateFlags, DescriptorSetLayoutCreateInfo, DescriptorType,
        },
        pool::{
            DescriptorPool, DescriptorPoolCreateFlags, DescriptorPoolCreateInfo,
            DescriptorSetAllocateInfo,
        },
        sys::RawDescriptorSet,
        DescriptorImageViewInfo, WriteDescriptorSet,
    },
    device::{Device, DeviceExtensions, DeviceFeatures, DeviceOwned},
    image::{
        sampler::{Sampler, SamplerCreateInfo},
        view::{ImageView, ImageViewCreateInfo},
        Image, ImageLayout,
    },
    instance::Instance,
    pipeline::{
        layout::{PipelineLayoutCreateInfo, PushConstantRange},
        PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::Framebuffer,
    shader::ShaderStages,
    DeviceSize, Validated, Version, VulkanError, VulkanObject,
};

// NOTE(Marc): The following constants must match the definitions in include/vulkano.glsl!

/// The set number of the [`GlobalDescriptorSet`].
pub const GLOBAL_SET: u32 = 0;

/// The binding number of samplers in the [`GlobalDescriptorSet`].
pub const SAMPLER_BINDING: u32 = 0;

/// The binding number of sampled images in the [`GlobalDescriptorSet`].
pub const SAMPLED_IMAGE_BINDING: u32 = 1;

/// The binding number of storage images in the [`GlobalDescriptorSet`].
pub const STORAGE_IMAGE_BINDING: u32 = 2;

/// The binding number of storage buffers in the [`GlobalDescriptorSet`].
pub const STORAGE_BUFFER_BINDING: u32 = 3;

/// The binding number of acceleration structures in the [`GlobalDescriptorSet`].
pub const ACCELERATION_STRUCTURE_BINDING: u32 = 4;

/// The set number of the local descriptor set.
pub const LOCAL_SET: u32 = 1;

/// The binding number of input attachments in the local descriptor set.
pub const INPUT_ATTACHMENT_BINDING: u32 = 0;

#[derive(Debug)]
pub struct BindlessContext {
    global_set: GlobalDescriptorSet,
    local_set_layout: Option<Arc<DescriptorSetLayout>>,
}

impl BindlessContext {
    /// Returns the device extensions required to create a bindless context.
    pub fn required_extensions(instance: &Instance) -> DeviceExtensions {
        let mut extensions = DeviceExtensions::default();

        if instance.api_version() < Version::V1_2 {
            extensions.ext_descriptor_indexing = true;
        }

        extensions
    }

    /// Returns the device features required to create a bindless context.
    pub fn required_features(_instance: &Instance) -> DeviceFeatures {
        DeviceFeatures {
            shader_sampled_image_array_dynamic_indexing: true,
            shader_storage_image_array_dynamic_indexing: true,
            shader_storage_buffer_array_dynamic_indexing: true,
            descriptor_binding_sampled_image_update_after_bind: true,
            descriptor_binding_storage_image_update_after_bind: true,
            descriptor_binding_storage_buffer_update_after_bind: true,
            descriptor_binding_update_unused_while_pending: true,
            descriptor_binding_partially_bound: true,
            runtime_descriptor_array: true,
            ..DeviceFeatures::default()
        }
    }

    pub(crate) fn new(
        resources: &Arc<ResourceStorage>,
        create_info: &BindlessContextCreateInfo<'_>,
    ) -> Result<Self, Validated<VulkanError>> {
        let global_set_layout =
            GlobalDescriptorSet::create_layout(resources, create_info.global_set)?;

        let global_set = GlobalDescriptorSet::new(resources, &global_set_layout)?;

        let local_set_layout = create_info
            .local_set
            .map(|local_set| LocalDescriptorSet::create_layout(resources, local_set))
            .transpose()?;

        Ok(BindlessContext {
            global_set,
            local_set_layout,
        })
    }

    /// Returns the layout of the [`GlobalDescriptorSet`].
    #[inline]
    pub fn global_set_layout(&self) -> &Arc<DescriptorSetLayout> {
        self.global_set.inner.layout()
    }

    /// Returns the layout of the local descriptor set.
    ///
    /// Returns `None` if [`BindlessContextCreateInfo::local_set`] was not specified when creating
    /// the bindless context.
    #[inline]
    pub fn local_set_layout(&self) -> Option<&Arc<DescriptorSetLayout>> {
        self.local_set_layout.as_ref()
    }

    /// Creates a new bindless pipeline layout from the union of the push constant requirements of
    /// each stage in `stages` for push constant ranges and the [global descriptor set layout] and
    /// optionally the [local descriptor set layout] for set layouts.
    ///
    /// All pipelines that you bind must have been created with a layout created like this or with
    /// a compatible layout for the bindless system to be able to bind its descriptor sets.
    ///
    /// It is recommended that you share the same pipeline layout object with as many pipelines as
    /// possible in order to reduce the amount of descriptor set (re)binding that is needed.
    ///
    /// See also [`pipeline_layout_create_info_from_stages`].
    ///
    /// [global descriptor set layout]: Self::global_set_layout
    /// [local descriptor set layout]: Self::local_set_layout
    /// [`pipeline_layout_create_info_from_stages`]: Self::pipeline_layout_create_info_from_stages
    pub fn pipeline_layout_from_stages<'a>(
        &self,
        stages: impl IntoIterator<Item = &'a PipelineShaderStageCreateInfo>,
    ) -> Result<Arc<PipelineLayout>, Validated<VulkanError>> {
        PipelineLayout::new(
            self.device().clone(),
            self.pipeline_layout_create_info_from_stages(stages),
        )
    }

    /// Creates a new bindless pipeline layout create info from the union of the push constant
    /// requirements of each stage in `stages` for push constant ranges and the [global descriptor
    /// set layout] and optionally the [local descriptor set layout] for set layouts.
    ///
    /// All pipelines that you bind must have been created with a layout created like this or with
    /// a compatible layout for the bindless system to be able to bind its descriptor sets.
    ///
    /// It is recommended that you share the same pipeline layout object with as many pipelines as
    /// possible in order to reduce the amount of descriptor set (re)binding that is needed.
    ///
    /// See also [`pipeline_layout_from_stages`].
    ///
    /// [global descriptor set layout]: Self::global_set_layout
    /// [local descriptor set layout]: Self::local_set_layout
    /// [`pipeline_layout_from_stages`]: Self::pipeline_layout_from_stages
    pub fn pipeline_layout_create_info_from_stages<'a>(
        &self,
        stages: impl IntoIterator<Item = &'a PipelineShaderStageCreateInfo>,
    ) -> PipelineLayoutCreateInfo {
        let mut push_constant_ranges = Vec::<PushConstantRange>::new();

        for stage in stages {
            let entry_point_info = stage.entry_point.info();

            if let Some(range) = &entry_point_info.push_constant_requirements {
                if let Some(existing_range) =
                    push_constant_ranges.iter_mut().find(|existing_range| {
                        existing_range.offset == range.offset && existing_range.size == range.size
                    })
                {
                    // If this range was already used before, add our stage to it.
                    existing_range.stages |= range.stages;
                } else {
                    // If this range is new, insert it.
                    push_constant_ranges.push(*range);
                }
            }
        }

        let mut set_layouts = vec![self.global_set_layout().clone()];

        if let Some(local_set_layout) = &self.local_set_layout {
            set_layouts.push(local_set_layout.clone());
        }

        PipelineLayoutCreateInfo {
            set_layouts,
            push_constant_ranges,
            ..Default::default()
        }
    }

    /// Returns the `GlobalDescriptorSet`.
    #[inline]
    pub fn global_set(&self) -> &GlobalDescriptorSet {
        &self.global_set
    }
}

unsafe impl DeviceOwned for BindlessContext {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.global_set.device()
    }
}

/// Parameters to create a new [`BindlessContext`].
#[derive(Clone, Debug)]
pub struct BindlessContextCreateInfo<'a> {
    /// Parameters to create the [`GlobalDescriptorSet`].
    ///
    /// The default value is `&GlobalDescriptorSetCreateInfo::new()`.
    pub global_set: &'a GlobalDescriptorSetCreateInfo<'a>,

    /// Parameters to create a local descriptor set.
    ///
    /// If set to `Some`, enables the use of bindless input attachments.
    ///
    /// The default value is `None`.
    pub local_set: Option<&'a LocalDescriptorSetCreateInfo<'a>>,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for BindlessContextCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl BindlessContextCreateInfo<'_> {
    /// Returns a default `BindlessContextCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            global_set: &const { GlobalDescriptorSetCreateInfo::new() },
            local_set: None,
            _ne: crate::NE,
        }
    }
}

#[derive(Debug)]
pub struct GlobalDescriptorSet {
    resources: Arc<ResourceStorage>,
    inner: RawDescriptorSet,

    samplers: SlotMap<SamplerId, SamplerDescriptor>,
    sampled_images: SlotMap<SampledImageId, SampledImageDescriptor>,
    storage_images: SlotMap<StorageImageId, StorageImageDescriptor>,
    storage_buffers: SlotMap<StorageBufferId, StorageBufferDescriptor>,
    acceleration_structures: SlotMap<AccelerationStructureId, AccelerationStructureDescriptor>,
}

#[derive(Debug)]
pub struct SamplerDescriptor {
    sampler: Arc<Sampler>,
}

#[derive(Debug)]
pub struct SampledImageDescriptor {
    image_view: Arc<ImageView>,
    image_layout: ImageLayout,
}

#[derive(Debug)]
pub struct StorageImageDescriptor {
    image_view: Arc<ImageView>,
    image_layout: ImageLayout,
}

#[derive(Debug)]
pub struct StorageBufferDescriptor {
    buffer: Arc<Buffer>,
    offset: DeviceSize,
    size: DeviceSize,
}

#[derive(Debug)]
pub struct AccelerationStructureDescriptor {
    acceleration_structure: Arc<AccelerationStructure>,
}

impl GlobalDescriptorSet {
    fn new(
        resources: &Arc<ResourceStorage>,
        layout: &Arc<DescriptorSetLayout>,
    ) -> Result<Self, VulkanError> {
        let device = resources.device();

        let allocator = Arc::new(GlobalDescriptorSetAllocator::new(device));
        let inner = RawDescriptorSet::new(allocator, layout, 0).map_err(Validated::unwrap)?;

        let hyaline_collector = resources.hyaline_collector();

        let descriptor_count = |n| layout.bindings().get(&n).map_or(0, |b| b.descriptor_count);
        let max_samplers = descriptor_count(SAMPLER_BINDING);
        let max_sampled_images = descriptor_count(SAMPLED_IMAGE_BINDING);
        let max_storage_images = descriptor_count(STORAGE_IMAGE_BINDING);
        let max_storage_buffers = descriptor_count(STORAGE_BUFFER_BINDING);
        let max_acceleration_structures = descriptor_count(ACCELERATION_STRUCTURE_BINDING);

        Ok(GlobalDescriptorSet {
            resources: resources.clone(),
            inner,
            samplers: SlotMap::with_collector_and_key(max_samplers, hyaline_collector.clone()),
            sampled_images: SlotMap::with_collector_and_key(
                max_sampled_images,
                hyaline_collector.clone(),
            ),
            storage_images: SlotMap::with_collector_and_key(
                max_storage_images,
                hyaline_collector.clone(),
            ),
            storage_buffers: SlotMap::with_collector_and_key(
                max_storage_buffers,
                hyaline_collector.clone(),
            ),
            acceleration_structures: SlotMap::with_collector_and_key(
                max_acceleration_structures,
                hyaline_collector.clone(),
            ),
        })
    }

    fn create_layout(
        resources: &Arc<ResourceStorage>,
        create_info: &GlobalDescriptorSetCreateInfo<'_>,
    ) -> Result<Arc<DescriptorSetLayout>, Validated<VulkanError>> {
        let device = resources.device();

        let binding_flags = DescriptorBindingFlags::UPDATE_AFTER_BIND
            | DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING
            | DescriptorBindingFlags::PARTIALLY_BOUND;

        let stages = get_all_supported_shader_stages(device);

        let mut bindings = BTreeMap::from([
            (
                SAMPLER_BINDING,
                DescriptorSetLayoutBinding {
                    binding_flags,
                    descriptor_count: create_info.max_samplers,
                    stages,
                    ..DescriptorSetLayoutBinding::new(DescriptorType::Sampler)
                },
            ),
            (
                SAMPLED_IMAGE_BINDING,
                DescriptorSetLayoutBinding {
                    binding_flags,
                    descriptor_count: create_info.max_sampled_images,
                    stages,
                    ..DescriptorSetLayoutBinding::new(DescriptorType::SampledImage)
                },
            ),
            (
                STORAGE_IMAGE_BINDING,
                DescriptorSetLayoutBinding {
                    binding_flags,
                    descriptor_count: create_info.max_storage_images,
                    stages,
                    ..DescriptorSetLayoutBinding::new(DescriptorType::StorageImage)
                },
            ),
            (
                STORAGE_BUFFER_BINDING,
                DescriptorSetLayoutBinding {
                    binding_flags,
                    descriptor_count: create_info.max_storage_buffers,
                    stages,
                    ..DescriptorSetLayoutBinding::new(DescriptorType::StorageBuffer)
                },
            ),
        ]);

        if device.enabled_features().acceleration_structure {
            bindings.insert(
                ACCELERATION_STRUCTURE_BINDING,
                DescriptorSetLayoutBinding {
                    binding_flags,
                    descriptor_count: create_info.max_acceleration_structures,
                    stages,
                    ..DescriptorSetLayoutBinding::new(DescriptorType::AccelerationStructure)
                },
            );
        }

        let layout = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                flags: DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL,
                bindings,
                ..Default::default()
            },
        )?;

        Ok(layout)
    }

    /// Returns the underlying raw descriptor set.
    #[inline]
    pub fn as_raw(&self) -> &RawDescriptorSet {
        &self.inner
    }

    pub fn create_sampler(
        &self,
        create_info: SamplerCreateInfo,
    ) -> Result<SamplerId, Validated<VulkanError>> {
        let sampler = Sampler::new(self.device().clone(), create_info)?;

        Ok(self.add_sampler(sampler))
    }

    pub fn create_sampled_image(
        &self,
        image_id: Id<Image>,
        create_info: ImageViewCreateInfo,
        image_layout: ImageLayout,
    ) -> Result<SampledImageId, Validated<VulkanError>> {
        let image_state = self.resources.image(image_id).unwrap();
        let image_view = ImageView::new(image_state.image().clone(), create_info)?;

        Ok(self.add_sampled_image(image_view, image_layout))
    }

    pub fn create_storage_image(
        &self,
        image_id: Id<Image>,
        create_info: ImageViewCreateInfo,
        image_layout: ImageLayout,
    ) -> Result<StorageImageId, Validated<VulkanError>> {
        let image_state = self.resources.image(image_id).unwrap();
        let image_view = ImageView::new(image_state.image().clone(), create_info)?;

        Ok(self.add_storage_image(image_view, image_layout))
    }

    pub fn create_storage_buffer(
        &self,
        buffer_id: Id<Buffer>,
        offset: DeviceSize,
        size: DeviceSize,
    ) -> Result<StorageBufferId, Validated<VulkanError>> {
        let buffer_state = self.resources.buffer(buffer_id).unwrap();
        let buffer = buffer_state.buffer().clone();

        Ok(self.add_storage_buffer(buffer, offset, size))
    }

    pub fn add_sampler(&self, sampler: Arc<Sampler>) -> SamplerId {
        let descriptor = SamplerDescriptor {
            sampler: sampler.clone(),
        };
        let id = self.samplers.insert(descriptor, &self.resources.pin());

        let write =
            WriteDescriptorSet::sampler_array(SAMPLER_BINDING, id.index, iter::once(sampler));

        unsafe { self.inner.update_unchecked(&[write], &[]) };

        id
    }

    pub fn add_sampled_image(
        &self,
        image_view: Arc<ImageView>,
        image_layout: ImageLayout,
    ) -> SampledImageId {
        assert!(matches!(
            image_layout,
            ImageLayout::General
                | ImageLayout::DepthStencilReadOnlyOptimal
                | ImageLayout::ShaderReadOnlyOptimal
                | ImageLayout::DepthReadOnlyStencilAttachmentOptimal
                | ImageLayout::DepthAttachmentStencilReadOnlyOptimal,
        ));

        let descriptor = SampledImageDescriptor {
            image_view: image_view.clone(),
            image_layout,
        };
        let id = self
            .sampled_images
            .insert(descriptor, &self.resources.pin());

        let write = WriteDescriptorSet::image_view_with_layout_array(
            SAMPLED_IMAGE_BINDING,
            id.index,
            iter::once(DescriptorImageViewInfo {
                image_view,
                image_layout,
            }),
        );

        unsafe { self.inner.update_unchecked(&[write], &[]) };

        id
    }

    pub fn add_storage_image(
        &self,
        image_view: Arc<ImageView>,
        image_layout: ImageLayout,
    ) -> StorageImageId {
        assert_eq!(image_layout, ImageLayout::General);

        let descriptor = StorageImageDescriptor {
            image_view: image_view.clone(),
            image_layout,
        };
        let id = self
            .storage_images
            .insert(descriptor, &self.resources.pin());

        let write = WriteDescriptorSet::image_view_with_layout_array(
            STORAGE_IMAGE_BINDING,
            id.index,
            iter::once(DescriptorImageViewInfo {
                image_view,
                image_layout,
            }),
        );

        unsafe { self.inner.update_unchecked(&[write], &[]) };

        id
    }

    pub fn add_storage_buffer(
        &self,
        buffer: Arc<Buffer>,
        offset: DeviceSize,
        size: DeviceSize,
    ) -> StorageBufferId {
        let subbuffer = Subbuffer::from(buffer.clone()).slice(offset..offset + size);

        let descriptor = StorageBufferDescriptor {
            buffer,
            offset,
            size,
        };
        let id = self
            .storage_buffers
            .insert(descriptor, &self.resources.pin());

        let write = WriteDescriptorSet::buffer_array(
            STORAGE_BUFFER_BINDING,
            id.index,
            iter::once(subbuffer),
        );

        unsafe { self.inner.update_unchecked(&[write], &[]) };

        id
    }

    pub fn add_acceleration_structure(
        &self,
        acceleration_structure: Arc<AccelerationStructure>,
    ) -> AccelerationStructureId {
        let descriptor = AccelerationStructureDescriptor {
            acceleration_structure: acceleration_structure.clone(),
        };
        let id = self
            .acceleration_structures
            .insert(descriptor, &self.resources.pin());

        let write = WriteDescriptorSet::acceleration_structure_array(
            ACCELERATION_STRUCTURE_BINDING,
            id.index,
            iter::once(acceleration_structure),
        );

        unsafe { self.inner.update_unchecked(&[write], &[]) };

        id
    }

    pub(crate) fn invalidate_sampler<'a>(
        &'a self,
        id: SamplerId,
        guard: &'a hyaline::Guard<'a>,
    ) -> Option<&'a SamplerDescriptor> {
        self.samplers.invalidate(id, guard)
    }

    pub(crate) fn invalidate_sampled_image<'a>(
        &'a self,
        id: SampledImageId,
        guard: &'a hyaline::Guard<'a>,
    ) -> Option<&'a SampledImageDescriptor> {
        self.sampled_images.invalidate(id, guard)
    }

    pub(crate) fn invalidate_storage_image<'a>(
        &'a self,
        id: StorageImageId,
        guard: &'a hyaline::Guard<'a>,
    ) -> Option<&'a StorageImageDescriptor> {
        self.storage_images.invalidate(id, guard)
    }

    pub(crate) fn invalidate_storage_buffer<'a>(
        &'a self,
        id: StorageBufferId,
        guard: &'a hyaline::Guard<'a>,
    ) -> Option<&'a StorageBufferDescriptor> {
        self.storage_buffers.invalidate(id, guard)
    }

    pub(crate) fn invalidate_acceleration_structure<'a>(
        &'a self,
        id: AccelerationStructureId,
        guard: &'a hyaline::Guard<'a>,
    ) -> Option<&'a AccelerationStructureDescriptor> {
        self.acceleration_structures.invalidate(id, guard)
    }

    pub(crate) fn remove_invalidated_sampler(&self, id: SamplerId) -> Option<()> {
        self.samplers.remove_invalidated(id)
    }

    pub(crate) fn remove_invalidated_sampled_image(&self, id: SampledImageId) -> Option<()> {
        self.sampled_images.remove_invalidated(id)
    }

    pub(crate) fn remove_invalidated_storage_image(&self, id: StorageImageId) -> Option<()> {
        self.storage_images.remove_invalidated(id)
    }

    pub(crate) fn remove_invalidated_storage_buffer(&self, id: StorageBufferId) -> Option<()> {
        self.storage_buffers.remove_invalidated(id)
    }

    pub(crate) fn remove_invalidated_acceleration_structure(
        &self,
        id: AccelerationStructureId,
    ) -> Option<()> {
        self.acceleration_structures.remove_invalidated(id)
    }

    #[inline]
    pub fn sampler(&self, id: SamplerId) -> Option<Ref<'_, SamplerDescriptor>> {
        let guard = self.resources.pin();

        // SAFETY: We unbind the lifetime because this would result in E0515 otherwise. This is
        // perfectly safe to do -- none of these methods actually borrow from the guard (there's
        // physically no way for them to; this is encoded in the type system). The lifetime is bound
        // to the returned reference to ensure that the reference doesn't outlive the guard. We
        // enforce that by `Ref` owning the `hyaline::Guard` instead.
        let inner = self.samplers.get(id, unsafe {
            mem::transmute::<&hyaline::Guard<'_>, &hyaline::Guard<'_>>(&guard)
        })?;

        Some(Ref { inner, guard })
    }

    #[inline]
    pub fn sampled_image(&self, id: SampledImageId) -> Option<Ref<'_, SampledImageDescriptor>> {
        let guard = self.resources.pin();

        // SAFETY: Same as in the `sampler` method above.
        let inner = self.sampled_images.get(id, unsafe {
            mem::transmute::<&hyaline::Guard<'_>, &hyaline::Guard<'_>>(&guard)
        })?;

        Some(Ref { inner, guard })
    }

    #[inline]
    pub fn storage_image(&self, id: StorageImageId) -> Option<Ref<'_, StorageImageDescriptor>> {
        let guard = self.resources.pin();

        // SAFETY: Same as in the `sampler` method above.
        let inner = self.storage_images.get(id, unsafe {
            mem::transmute::<&hyaline::Guard<'_>, &hyaline::Guard<'_>>(&guard)
        })?;

        Some(Ref { inner, guard })
    }

    #[inline]
    pub fn storage_buffer(&self, id: StorageBufferId) -> Option<Ref<'_, StorageBufferDescriptor>> {
        let guard = self.resources.pin();

        // SAFETY: Same as in the `sampler` method above.
        let inner = self.storage_buffers.get(id, unsafe {
            mem::transmute::<&hyaline::Guard<'_>, &hyaline::Guard<'_>>(&guard)
        })?;

        Some(Ref { inner, guard })
    }

    #[inline]
    pub fn acceleration_structure(
        &self,
        id: AccelerationStructureId,
    ) -> Option<Ref<'_, AccelerationStructureDescriptor>> {
        let guard = self.resources.pin();

        // SAFETY: Same as in the `sampler` method above.
        let inner = self.acceleration_structures.get(id, unsafe {
            mem::transmute::<&hyaline::Guard<'_>, &hyaline::Guard<'_>>(&guard)
        })?;

        Some(Ref { inner, guard })
    }
}

unsafe impl VulkanObject for GlobalDescriptorSet {
    type Handle = vk::DescriptorSet;

    #[inline]
    fn handle(&self) -> Self::Handle {
        self.inner.handle()
    }
}

unsafe impl DeviceOwned for GlobalDescriptorSet {
    #[inline]
    fn device(&self) -> &Arc<Device> {
        self.inner.device()
    }
}

impl SamplerDescriptor {
    #[inline]
    pub fn sampler(&self) -> &Arc<Sampler> {
        &self.sampler
    }
}

impl SampledImageDescriptor {
    #[inline]
    pub fn image_view(&self) -> &Arc<ImageView> {
        &self.image_view
    }

    #[inline]
    pub fn image_layout(&self) -> ImageLayout {
        self.image_layout
    }
}

impl StorageImageDescriptor {
    #[inline]
    pub fn image_view(&self) -> &Arc<ImageView> {
        &self.image_view
    }

    #[inline]
    pub fn image_layout(&self) -> ImageLayout {
        self.image_layout
    }
}

impl StorageBufferDescriptor {
    #[inline]
    pub fn buffer(&self) -> &Arc<Buffer> {
        &self.buffer
    }

    #[inline]
    pub fn offset(&self) -> DeviceSize {
        self.offset
    }

    #[inline]
    pub fn size(&self) -> DeviceSize {
        self.size
    }
}

impl AccelerationStructureDescriptor {
    #[inline]
    pub fn acceleration_structure(&self) -> &Arc<AccelerationStructure> {
        &self.acceleration_structure
    }
}

/// Parameters to create a new [`GlobalDescriptorSet`].
#[derive(Clone, Debug)]
pub struct GlobalDescriptorSetCreateInfo<'a> {
    /// The maximum number of [`Sampler`] descriptors that the collection can hold at once.
    ///
    /// The default value is `256` (2<sup>8</sup>).
    pub max_samplers: u32,

    /// The maximum number of sampled [`Image`] descriptors that the collection can hold at once.
    ///
    /// The default value is `1048576` (2<sup>20</sup>).
    pub max_sampled_images: u32,

    /// The maximum number of storage [`Image`] descriptors that the collection can hold at once.
    ///
    /// The default value is `1048576` (2<sup>20</sup>).
    pub max_storage_images: u32,

    /// The maximum number of storage [`Buffer`] descriptors that the collection can hold at once.
    ///
    /// The default value is `1048576` (2<sup>20</sup>).
    pub max_storage_buffers: u32,

    /// The maximum number of [`AccelerationStructure`] descriptors that the collection can hold at
    /// once.
    ///
    /// The default value is `1048576` (2<sup>20</sup>).
    pub max_acceleration_structures: u32,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for GlobalDescriptorSetCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalDescriptorSetCreateInfo<'_> {
    /// Returns a default `GlobalDescriptorSetCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            max_samplers: 1 << 8,
            max_sampled_images: 1 << 20,
            max_storage_images: 1 << 20,
            max_storage_buffers: 1 << 20,
            max_acceleration_structures: 1 << 20,
            _ne: crate::NE,
        }
    }
}

macro_rules! declare_key {
    (
        $(#[$meta:meta])*
        pub struct $name:ident $(;)?
    ) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[repr(C)]
        pub struct $name {
            index: u32,
            generation: u32,
        }

        unsafe impl Pod for $name {}
        unsafe impl Zeroable for $name {}

        impl Default for $name {
            #[inline]
            fn default() -> Self {
                Self::INVALID
            }
        }

        impl $name {
            /// An ID that's guaranteed to be invalid.
            pub const INVALID: Self = Self::new(concurrent_slotmap::SlotId::INVALID);

            #[inline(always)]
            const fn new(slot: concurrent_slotmap::SlotId) -> Self {
                Self {
                    index: slot.index(),
                    generation: slot.generation(),
                }
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Debug::fmt(&concurrent_slotmap::Key::as_id(*self), f)
            }
        }

        #[doc(hidden)]
        impl concurrent_slotmap::Key for $name {
            #[inline(always)]
            fn from_id(id: concurrent_slotmap::SlotId) -> Self {
                Self::new(id)
            }

            #[inline(always)]
            fn as_id(self) -> concurrent_slotmap::SlotId {
                concurrent_slotmap::SlotId::new(self.index, self.generation)
            }
        }
    };
}

declare_key! {
    pub struct SamplerId;
}

declare_key! {
    pub struct SampledImageId;
}

declare_key! {
    pub struct StorageImageId;
}

declare_key! {
    pub struct StorageBufferId;
}

declare_key! {
    pub struct AccelerationStructureId;
}

struct GlobalDescriptorSetAllocator {
    device: Arc<Device>,
}

impl GlobalDescriptorSetAllocator {
    fn new(device: &Arc<Device>) -> Self {
        GlobalDescriptorSetAllocator {
            device: device.clone(),
        }
    }
}

unsafe impl DescriptorSetAllocator for GlobalDescriptorSetAllocator {
    fn allocate(
        &self,
        layout: &Arc<DescriptorSetLayout>,
        _variable_count: u32,
    ) -> Result<DescriptorSetAlloc, Validated<VulkanError>> {
        let mut pool_sizes = HashMap::default();

        for binding in layout.bindings().values() {
            *pool_sizes.entry(binding.descriptor_type).or_insert(0) += binding.descriptor_count;
        }

        let pool = Arc::new(DescriptorPool::new(
            layout.device().clone(),
            DescriptorPoolCreateInfo {
                flags: DescriptorPoolCreateFlags::UPDATE_AFTER_BIND,
                max_sets: 1,
                pool_sizes,
                ..Default::default()
            },
        )?);

        let allocate_info = DescriptorSetAllocateInfo::new(layout.clone());

        let inner = unsafe { pool.allocate_descriptor_sets(iter::once(allocate_info)) }?
            .next()
            .unwrap();

        Ok(DescriptorSetAlloc {
            inner,
            pool,
            handle: AllocationHandle::null(),
        })
    }

    unsafe fn deallocate(&self, _allocation: DescriptorSetAlloc) {}
}

unsafe impl DeviceOwned for GlobalDescriptorSetAllocator {
    fn device(&self) -> &Arc<Device> {
        &self.device
    }
}

pub(crate) struct LocalDescriptorSet {
    inner: RawDescriptorSet,
}

impl LocalDescriptorSet {
    // FIXME: allocation
    pub(crate) unsafe fn new(
        resources: &Arc<Resources>,
        layout: &Arc<DescriptorSetLayout>,
        framebuffer: &Framebuffer,
        subpass_index: usize,
    ) -> Result<Arc<Self>, VulkanError> {
        let allocator = resources.descriptor_set_allocator().clone();
        let inner = RawDescriptorSet::new(allocator, layout, 0).map_err(Validated::unwrap)?;

        let render_pass = framebuffer.render_pass();
        let subpass_description = &render_pass.subpasses()[subpass_index];
        let input_attachments = &subpass_description.input_attachments;
        let mut writes = Vec::new();

        for (input_attachment_index, attachment_reference) in input_attachments.iter().enumerate() {
            let Some(attachment_reference) = attachment_reference else {
                continue;
            };
            let attachment = &framebuffer.attachments()[attachment_reference.attachment as usize];

            writes.push(WriteDescriptorSet::image_view_with_layout_array(
                INPUT_ATTACHMENT_BINDING,
                input_attachment_index as u32,
                iter::once(DescriptorImageViewInfo {
                    image_view: attachment.clone(),
                    image_layout: attachment_reference.layout,
                }),
            ));
        }

        unsafe { inner.update_unchecked(&writes, &[]) };

        Ok(Arc::new(LocalDescriptorSet { inner }))
    }

    fn create_layout(
        resources: &Arc<ResourceStorage>,
        create_info: &LocalDescriptorSetCreateInfo<'_>,
    ) -> Result<Arc<DescriptorSetLayout>, Validated<VulkanError>> {
        let device = resources.device();

        let layout = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings: BTreeMap::from([(
                    INPUT_ATTACHMENT_BINDING,
                    DescriptorSetLayoutBinding {
                        binding_flags: DescriptorBindingFlags::PARTIALLY_BOUND,
                        descriptor_count: create_info.max_input_attachments,
                        stages: ShaderStages::FRAGMENT,
                        ..DescriptorSetLayoutBinding::new(DescriptorType::InputAttachment)
                    },
                )]),
                ..Default::default()
            },
        )?;

        Ok(layout)
    }

    pub(crate) fn as_raw(&self) -> &RawDescriptorSet {
        &self.inner
    }
}

/// Parameters to create a local descriptor set.
#[derive(Clone, Debug)]
pub struct LocalDescriptorSetCreateInfo<'a> {
    /// The maximum number of input attachment descriptors that the collection can hold at once.
    ///
    /// This also determines the maximum input attachment index: `max_input_attachments - 1`.
    ///
    /// The default value is `8`.
    pub max_input_attachments: u32,

    pub _ne: crate::NonExhaustive<'a>,
}

impl Default for LocalDescriptorSetCreateInfo<'_> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl LocalDescriptorSetCreateInfo<'_> {
    /// Returns a default `LocalDescriptorSetCreateInfo`.
    #[inline]
    pub const fn new() -> Self {
        Self {
            max_input_attachments: 8,
            _ne: crate::NE,
        }
    }
}

fn get_all_supported_shader_stages(device: &Arc<Device>) -> ShaderStages {
    let mut stages = ShaderStages::all_graphics() | ShaderStages::COMPUTE;

    if device.enabled_extensions().khr_ray_tracing_pipeline
        || device.enabled_extensions().nv_ray_tracing
    {
        stages |= ShaderStages::RAYGEN
            | ShaderStages::ANY_HIT
            | ShaderStages::CLOSEST_HIT
            | ShaderStages::MISS
            | ShaderStages::INTERSECTION
            | ShaderStages::CALLABLE;
    }

    if device.enabled_extensions().ext_mesh_shader || device.enabled_extensions().nv_mesh_shader {
        stages |= ShaderStages::TASK | ShaderStages::MESH;
    }

    if device.enabled_extensions().huawei_subpass_shading {
        stages |= ShaderStages::SUBPASS_SHADING;
    }

    stages
}
