use std::iter;

use device::Device;

/// Trait implemented on structs that describe a buffer layout.
///
/// The API will accept any buffer whose content implements `Layout` where the `RawLayout` matches
/// what is expected. This way you can create multiple structs compatible with each other.
pub unsafe trait Layout {
    type RawLayout;
}

/// Represents the layout of the resources and data that can be binded before drawing and that will
/// be accessible from the shaders.
///
/// The template parameter represents the descriptor sets.
// TODO: push constants.
pub struct PipelineLayout<DescriptorSets> {
    device: Arc<Device>,
    layout: VkPipelineLayout,
}

impl<DescriptorSets> PipelineLayout<DescriptorSets> {
    /// Creates a new `PipelineLayout`.
    pub fn new(device: &Arc<Device>) -> Result<Arc<PipelineLayout<L>>, > {
        let layout = unsafe {
            let infos = VkPipelineLayoutCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                setLayoutCount: ,
                pSetLayouts: ,
                pushConstantRangeCount: ,
                pPushConstantRanges: ,
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreatePipelineLayout(device.internal_object(), &infos,
                                                      ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(PipelineLayout {
            device: device.clone(),
            layout: layout,
        }))
    }
}

pub trait DescriptorDef {
    type BindData;
}

pub struct StorageImageMarker;

pub struct SamplerMarker;

pub struct SampledImageMarker;

pub struct CombinedImageSamplerMarker;

pub struct UniformTexelBufferMarker<T: ?Sized, M>(Buffer<T, M>);

pub struct StorageTexelBufferMarker<T: ?Sized, M>(Buffer<T, M>);

pub struct UniformBufferMarker<T: ?Sized, M>(Buffer<T, M>);

pub struct StorageBufferMarker<T: ?Sized, M>(Buffer<T, M>);

pub struct DynamicUniformBufferMarker<T: ?Sized, M>(Buffer<T, M>);

pub struct DynamicStorageBufferMarker<T: ?Sized, M>(Buffer<T, M>);

pub struct InputAttachmentMarker;

pub trait DescriptorSetDefinition {
    type Raw;

    fn into_raw(self) -> Self::Raw;
}

pub struct DescriptorSetLayout<D> {
    device: Arc<Device>,
    layout: vk::DescriptorSetLayout,
    marker: PhantomData<D>,
}

impl<D> DescriptorSetLayout<D> where D: DescriptorSetDefinition {
    pub fn new(device: &Arc<Device>) -> Result<Arc<DescriptorSetLayout<D>> {
        let vk = device.pointers();

        let bindings = ;

        let layout = unsafe {
            let infos = vk::DescriptorSetLayoutCreateInfo {
                sType: vk::STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                bindingCount: bindings.len() as u32,
                pBindings: bindings.as_ptr(),
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateDescriptorSetLayout(device.internal_object(), &infos,
                                                           ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(DescriptorSetLayout {
            device: device.clone(),
            layout: layout,
            marker: PhantomData,
        }))
    }
}

impl DescriptorSet {
    pub fn new<I>(device: &Arc<Device>, descriptors: I) -> Arc<DescriptorSet>
        where I: IntoIterator<Item = Descriptor>
    {
        let descriptors: Vec<_> = descriptors.into_iter().map(|descriptor| {
            let stage_flags =
                if descriptor.stages.vertex { vk::SHADER_STAGE_VERTEX_BIT } else { 0 } |
                if descriptor.stages.vertex { vk::SHADER_STAGE_TESSELLATION_CONTROL_BIT }
                else { 0 } |
                if descriptor.stages.vertex { vk::SHADER_STAGE_TESSELLATION_EVALUATION_BIT }
                else { 0 } |
                if descriptor.stages.vertex { vk::SHADER_STAGE_GEOMETRY_BIT } else { 0 } |
                if descriptor.stages.vertex { vk::SHADER_STAGE_FRAGMENT_BIT } else { 0 } |
                if descriptor.stages.vertex { vk::SHADER_STAGE_COMPUTE_BIT } else { 0 };

            // TODO: check array size limits

            VkDescriptorSetLayoutBinding {
                descriptorType: ,
                arraySize: descriptor.array_size,
                stageFlags: stage_flags,
                pImmutableSamplers: ,
            }
        }).collect();

        vkCreateDescriptorSetLayout
    }

    #[inline]
    pub fn write(&self) {
        DescriptorSet::update(Some(write), iter::empty())
    }

    #[inline]
    pub fn copy(&self) {
        DescriptorSet::update(iter::empty(), Some(copy))
    }

    #[inline]
    pub fn multi_write<I>(writes: I) {
        DescriptorSet::update(writes, iter::empty())
    }

    #[inline]
    pub fn multi_copy<I>(copies: I) {
        DescriptorSet::update(iter::empty(), copies)
    }

    pub fn update(writes: I, copies: J) {

    }
}

pub struct Descriptor {
    pub ty: DescriptorType,
    pub array_size: u32,
    pub stages: ShaderStages,
}

pub enum DescriptorType {
    Sampler,
    CombinedImageSampler,
    SampledImage,
    StorageImage,
    UniformTexelBuffer,
    StorageTexelBuffer,
    UniformBuffer,
    StorageBuffer,
    UniformBufferDynamic,
    StorageBufferDynamic,
    InputAttachment,
}

pub struct ShaderStages {
    pub vertex: bool,
    pub tessellation_control: bool,
    pub tessellation_evaluation: bool,
    pub geometry: bool,
    pub fragment: bool,
    pub compute: bool,
}
