

How does that work?

The shader analyser determines which descriptors are used by each shader and outputs a struct
that describes this stuff. This struct implements `PipelineLayoutDesc`.

When shaders are grouped together in the pipeline, they are passed through `CompatiblePipeline`
to be merged into a single pipeline layout struct. If the descriptors of the various shaders
are incompatible, it is detected at that moment.

The single struct that represents the layout is stored as a template parameter of the pipeline.
It is also stored in the `PipelineLayout` object (which wraps around vulkan's pipeline layout).
When you draw, you have to pass a collection of descriptor sets that are compatible with the
pipeline layout desc of the pipeline.




use std::iter;

use device::Device;

/// Trait implemented on structs that describe a buffer layout.
///
/// The API will accept any buffer whose content implements `Layout` where the `RawLayout` matches
/// what is expected. This way you can create multiple structs compatible with each other.
pub unsafe trait Layout {
    type RawLayout;
}



pub unsafe trait PipelineLayoutDesc {
    type DescriptorSets;
    type PushConstants;
}

// example: impl PipelineLayoutDesc for (CustomFrag1, CustomFrag2, CustomFrag3) {}



pub unsafe trait CompatiblePipeline<T> { type Out; }
pub unsafe trait CompatibleSet<T> { type Out; }

macro_rules! impl_tuple {
    (($in_first:ident $out_first:ident) $(, ($in_rest:ident $out_rest:ident))*) => {
        unsafe impl<$in_first, $out_first $(, $in_rest, $out_rest)*>
            CompatibleSet<($out_first, $($out_rest,)*)> for ($in_first, $($in_rest,)*)
                where $in_first: CompatibleDescriptor<$out_first> $(, $in_rest: CompatibleDescriptor<$out_rest>)*
        {
            type Out = (
                <$in_first as CompatibleDescriptor<$out_first>>::Out,
                $(
                    <$in_rest as CompatibleDescriptor<$out_rest>>::Out,
                )*
            );
        }

        unsafe impl<$in_first, $out_first $(, $in_rest, $out_rest)*>
            CompatiblePipeline<($out_first, $($out_rest,)*)> for ($in_first, $($in_rest,)*)
                where $in_first: CompatibleSet<$out_first> $(, $in_rest: CompatibleSet<$out_rest>)*
        {
            type Out = (
                <$in_first as CompatibleSet<$out_first>>::Out,
                $(
                    <$in_rest as CompatibleSet<$out_rest>>::Out,
                )*
            );
        }

        impl_tuple!{$(($in_rest $out_rest)),*}
    };
    
    () => ();
}

impl_tuple!( (A N), (B O), (C P), (D Q), (E R), (F S), (G T),
             (H U), (I V), (J W), (K X), (L Y), (M Z) );

/// If a type `A` can be interpreted as a `T`, then `A` will implement `CompatibleDescriptor<T>`.
trait CompatibleDescriptor<T> { type Out; }

impl CompatibleDescriptor<()> for () { type Out = (); }
impl<T> CompatibleDescriptor<()> for T where T: Descriptor { type Out = T; }
impl<T> CompatibleDescriptor<T> for () where T: Descriptor { type Out = T; }
impl<T> CompatibleDescriptor<T> for T where T: Descriptor { type Out = T; }


pub unsafe trait Descriptor {}




/// Represents the layout of the resources and data that can be binded before drawing and that will
/// be accessible from the shaders.
///
/// The template parameter represents the descriptor sets.
// TODO: push constants.
pub struct PipelineLayout<P> {
    device: Arc<Device>,
    layout: VkPipelineLayout,
}

impl<P> PipelineLayout<P> where P: PipelineLayoutDesc {
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

pub unsafe trait DescriptorSetDef {
    type Content;


}

// example:
// impl DescriptorSetDef for MyDescriptorSet { }

pub struct DescriptorSetLayout<D> {
    device: Arc<Device>,
    layout: vk::DescriptorSetLayout,
    marker: PhantomData<D>,
}

impl<D> DescriptorSetLayout<D> where D: DescriptorSetDef {
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
