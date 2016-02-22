//! Collection of resources accessed by the pipeline.
//!
//! The resources accessed by the pipeline must be accessed through what is called a *descriptor*.
//! Descriptors are grouped in what is called *descriptor sets*. Descriptor sets are also grouped
//! in what is called a *pipeline layout*.
//!
//! # Pipeline initialization
//!
//! In order to build a pipeline object (a `GraphicsPipeline` or a `ComputePipeline`), you have to
//! pass a pointer to a `PipelineLayout<T>` struct. This struct is a wrapper around a Vulkan struct
//! that contains all the data about the descriptor sets and descriptors that will be available
//! in the pipeline. The `T` parameter must implement the `PipelineLayoutDesc` trait and describes
//! the descriptor sets and descriptors on vulkano's side.
//!
//! To build a `PipelineLayout`, you need to pass a collection of `DescriptorSetLayout` structs.
//! A `DescriptorSetLayout<T>` if the equivalent of `PipelineLayout` but for a single descriptor
//! set. The `T` parameter must implement the `DescriptorSetDesc` trait.
//!
//! # Binding resources
//! 
//! In parallel of the pipeline initialization, you have to create a `DescriptorSet<T>`. This
//! struct contains the list of actual resources that will be bound when the pipeline is executed.
//! To build a `DescriptorSet<T>`, you need to pass a `DescriptorSetLayout<T>`. The `T` parameter
//! must implement `DescriptorSetDesc` as if the same for both the descriptor set and its layout.
//!
//! TODO: describe descriptor set writes
//!
//! # Shader analyser
//! 
//! While you can manually implement the `PipelineLayoutDesc` and `DescriptorSetDesc` traits on
//! your own types, it is encouraged to use the `vulkano-shaders` crate instead. This crate will
//! automatically parse your SPIR-V code and generate structs that implement these traits and
//! describe the pipeline layout to vulkano.

use std::mem;
use std::option::IntoIter as OptionIntoIter;
use std::ptr;
use std::sync::Arc;

use buffer::BufferResource;
use device::Device;

use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

pub use self::pool::DescriptorPool;
pub use self::runtime_desc::RuntimeDesc;
pub use self::runtime_desc::EmptyPipelineDesc;
pub use self::runtime_desc::RuntimeDescriptorSetDesc;

mod pool;
mod runtime_desc;

/// Types that describe the layout of a pipeline (descriptor sets and push constants).
pub unsafe trait PipelineLayoutDesc {
    /// Represents a collection of `DescriptorSet` structs. A parameter of this type must be
    /// passed when you add a draw command to a command buffer that uses this layout.
    type DescriptorSets;

    /// Represents a collection of `DescriptorSetLayout` structs. A parameter of this type must
    /// be passed when creating a `PipelineLayout` struct.
    type DescriptorSetLayouts;

    /// Not yet implemented. Useless for now.
    type PushConstants;

    /// Turns the `DescriptorSets` associated type into something vulkano can understand.
    fn decode_descriptor_sets(&self, Self::DescriptorSets) -> Vec<Arc<AbstractDescriptorSet>>;  // TODO: vec is slow

    /// Turns the `DescriptorSetLayouts` associated type into something vulkano can understand.
    fn decode_descriptor_set_layouts(&self, Self::DescriptorSetLayouts)
                                     -> Vec<Arc<AbstractDescriptorSetLayout>>;  // TODO: vec is slow

    // FIXME: implement this correctly
    fn is_compatible_with<P>(&self, _: &P) -> bool where P: PipelineLayoutDesc { true }
}

/*
#[macro_export]
macro_rules! pipeline_layout {
    (sets: {$($set_name:ident: { $($name:ident : ),* }),*}) => {
        mod layout {
            use std::sync::Arc;
            use $crate::descriptor_set::DescriptorType;
            use $crate::descriptor_set::DescriptorDesc;
            use $crate::descriptor_set::DescriptorSetDesc;
            use $crate::descriptor_set::DescriptorWrite;
            use $crate::descriptor_set::DescriptorBind;
            use $crate::descriptor_set::PipelineLayout;
            use $crate::descriptor_set::PipelineLayoutDesc;
            use $crate::descriptor_set::ShaderStages;
            use $crate::buffer::BufferResource;

            $(
                pub struct $set_name;
                unsafe impl DescriptorSetDesc for $set_name {
                    type Write = (      // FIXME: variable number of elems
                        Arc<BufferResource>     // FIXME: strong typing
                    );

                    type Init = Self::Write;

                    #[inline]
                    fn descriptors(&self) -> Vec<DescriptorDesc> {
                        let mut binding = 0;
                        let mut result = Vec::new();        // TODO: with_capacity

                        //$(
                            result.push(DescriptorDesc {
                                binding: binding,
                                ty: DescriptorType::UniformBuffer,      // FIXME:
                                array_count: 1,     // FIXME:
                                stages: ShaderStages::all_graphics(),       // FIXME:
                            });

                            binding += 1;
                        //)*        // FIXME: variable number of elems

                        let _ = binding;    // removes a warning

                        result
                    }

                    fn decode_write(&self, data: Self::Write) -> Vec<DescriptorWrite> {
                        let mut binding = 0;
                        let mut result = Vec::new();        // TODO: with_capacity

                        let $($name),* = data;

                        $(
                            result.push(DescriptorWrite {
                                binding: binding,
                                array_element: 0,       // FIXME:
                                content: DescriptorBind::UniformBuffer($name),
                            });

                            binding += 1;
                        )*

                        result
                    }

                    #[inline]
                    fn decode_init(&self, data: Self::Init) -> Vec<DescriptorWrite> {
                        self.decode_write(data)
                    }
                }
            )*

            pub struct Layout;
            unsafe impl PipelineLayoutDesc for Layout {
                type DescriptorSets = ($(Arc<DescriptorSet<$set_name>>),*);
                type DescriptorSetLayouts = ($(Arc<DescriptorSetLayout<$set_name>>),*);
                type PushConstants = ();

                #[inline]
                fn decode_descriptor_sets(&self, sets: Self::DescriptorSets)
                                          -> Vec<Arc<AbstractDescriptorSet>>
                {
                    let $($set_name),* = sets;
                    vec![$($set_name as Arc<_>),*]
                }

                #[inline]
                fn decode_descriptor_set_layouts(&self, layouts: Self::DescriptorSetLayouts)
                                                 -> Vec<Arc<AbstractDescriptorSetLayout>>
                {
                    let $($set_name),* = layouts;
                    vec![$($set_name as Arc<_>),*]
                }
            }
        }
    }
}*/

/// Types that describe a single descriptor set.
pub unsafe trait DescriptorSetDesc {
    /// Represents a modification of a descriptor set. A parameter of this type must be passed
    /// when you modify a descriptor set.
    type Write;

    /// 
    type Init;

    /// Returns the list of descriptors contained in this set.
    fn descriptors(&self) -> Vec<DescriptorDesc>;       // TODO: better perfs

    /// Turns the `Write` associated type into something vulkano can understand.
    fn decode_write(&self, Self::Write) -> Vec<DescriptorWrite>;        // TODO: better perfs

    /// Turns the `Init` associated type into something vulkano can understand.
    fn decode_init(&self, Self::Init) -> Vec<DescriptorWrite>;      // TODO: better perfs

    // FIXME: implement this correctly
    fn is_compatible_with<S>(&self, _: &S) -> bool where S: DescriptorSetDesc { true }
}

// FIXME: shoud allow multiple array binds at once
pub struct DescriptorWrite {
    pub binding: u32,
    pub array_element: u32,
    pub content: DescriptorBind,
}

// FIXME: incomplete
#[derive(Clone)]        // TODO: Debug
pub enum DescriptorBind {
    UniformBuffer(Arc<BufferResource>),
}

/// Describes a single descriptor.
#[derive(Debug, Copy, Clone)]
pub struct DescriptorDesc {
    /// Offset of the binding within the descriptor.
    pub binding: u32,

    /// What kind of resource can later be bind to this descriptor.
    pub ty: DescriptorType,

    /// How many array elements this descriptor is made of.
    pub array_count: u32,

    /// Which shader stages are going to access this descriptor.
    pub stages: ShaderStages,
}

/// Describes what kind of resource may later be bound to a descriptor.
// FIXME: add immutable sampler when relevant
#[derive(Debug, Copy, Clone)]
#[repr(u32)]
pub enum DescriptorType {
    Sampler = vk::DESCRIPTOR_TYPE_SAMPLER,
    CombinedImageSampler = vk::DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    SampledImage = vk::DESCRIPTOR_TYPE_SAMPLED_IMAGE,
    StorageImage = vk::DESCRIPTOR_TYPE_STORAGE_IMAGE,
    UniformTexelBuffer = vk::DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
    StorageTexelBuffer = vk::DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
    UniformBuffer = vk::DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    StorageBuffer = vk::DESCRIPTOR_TYPE_STORAGE_BUFFER,
    UniformBufferDynamic = vk::DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
    StorageBufferDynamic = vk::DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC,
    InputAttachment = vk::DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
}

impl DescriptorType {
    /// Turns the `DescriptorType` into the corresponding Vulkan constant.
    // this function exists because when immutable samplers are added, it will no longer be possible to do `as u32`
    #[inline]
    fn vk_enum(&self) -> u32 {
        *self as u32
    }
}

/// Describes which shader stages have access to a descriptor.
#[derive(Debug, Copy, Clone)]
pub struct ShaderStages {
    pub vertex: bool,
    pub tessellation_control: bool,
    pub tessellation_evaluation: bool,
    pub geometry: bool,
    pub fragment: bool,
    pub compute: bool,
}

impl ShaderStages {
    /// Creates a `ShaderStages` struct will all graphics stages set to `true`.
    #[inline]
    pub fn all_graphics() -> ShaderStages {
        ShaderStages {
            vertex: true,
            tessellation_control: true,
            tessellation_evaluation: true,
            geometry: true,
            fragment: true,
            compute: false,
        }
    }

    /// Creates a `ShaderStages` struct will the compute stage set to `true`.
    #[inline]
    pub fn compute() -> ShaderStages {
        ShaderStages {
            vertex: false,
            tessellation_control: false,
            tessellation_evaluation: false,
            geometry: false,
            fragment: false,
            compute: true,
        }
    }
}

#[doc(hidden)]
impl Into<vk::ShaderStageFlags> for ShaderStages {
    #[inline]
    fn into(self) -> vk::ShaderStageFlags {
        let mut result = 0;
        if self.vertex { result |= vk::SHADER_STAGE_VERTEX_BIT; }
        if self.tessellation_control { result |= vk::SHADER_STAGE_TESSELLATION_CONTROL_BIT; }
        if self.tessellation_evaluation { result |= vk::SHADER_STAGE_TESSELLATION_EVALUATION_BIT; }
        if self.geometry { result |= vk::SHADER_STAGE_GEOMETRY_BIT; }
        if self.fragment { result |= vk::SHADER_STAGE_FRAGMENT_BIT; }
        if self.compute { result |= vk::SHADER_STAGE_COMPUTE_BIT; }
        result
    }
}


/// An actual descriptor set with the resources that are binded to it.
pub struct DescriptorSet<S> {
    set: vk::DescriptorSet,
    pool: Arc<DescriptorPool>,
    layout: Arc<DescriptorSetLayout<S>>,
}

impl<S> DescriptorSet<S> where S: DescriptorSetDesc {
    ///
    /// # Panic
    ///
    /// - Panicks if the pool and the layout were not created from the same `Device`.
    ///
    pub fn new(pool: &Arc<DescriptorPool>, layout: &Arc<DescriptorSetLayout<S>>, init: S::Init)
               -> Result<Arc<DescriptorSet<S>>, OomError>
    {
        unsafe {
            let set = try!(DescriptorSet::uninitialized(pool, layout));
            set.unchecked_write(layout.description().decode_init(init));
            Ok(set)
        }
    }

    ///
    /// # Panic
    ///
    /// - Panicks if the pool and the layout were not created from the same `Device`.
    ///
    // FIXME: this has to check whether there's still enough room in the pool
    pub unsafe fn uninitialized(pool: &Arc<DescriptorPool>, layout: &Arc<DescriptorSetLayout<S>>)
                                -> Result<Arc<DescriptorSet<S>>, OomError>
    {
        assert_eq!(&**pool.device() as *const Device, &*layout.device as *const Device);

        let vk = pool.device().pointers();

        let set = {
            let infos = vk::DescriptorSetAllocateInfo {
                sType: vk::STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                pNext: ptr::null(),
                descriptorPool: pool.internal_object(),
                descriptorSetCount: 1,
                pSetLayouts: &layout.layout,
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.AllocateDescriptorSets(pool.device().internal_object(), &infos,
                                                        &mut output)));
            output
        };

        Ok(Arc::new(DescriptorSet {
            set: set,
            pool: pool.clone(),
            layout: layout.clone(),
        }))
    }

    /// Modifies a descriptor set.
    ///
    /// The parameter depends on your implementation of `DescriptorSetDesc`.
    ///
    /// This function trusts the implementation of `DescriptorSetDesc` when it comes to making sure
    /// that the correct resource type is written to the correct descriptor.
    pub fn write(&self, write: S::Write) {
        let write = self.layout.description().decode_write(write);
        unsafe { self.unchecked_write(write); }
    }

    /// Modifies a descriptor set without checking that the writes are correct.
    pub unsafe fn unchecked_write(&self, write: Vec<DescriptorWrite>) {
        let vk = self.pool.device().pointers();

        // FIXME: store resources in the descriptor set so that they aren't destroyed

        // TODO: the architecture of this function is going to be tricky

        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let buffer_descriptors = write.iter().enumerate().map(|(num, write)| {
            match write.content {
                DescriptorBind::UniformBuffer(ref buffer) => {
                    Some(vk::DescriptorBufferInfo {
                        buffer: buffer.internal_object(),
                        offset: 0,      // FIXME: allow buffer slices
                        range: buffer.size() as u64,       // FIXME: allow buffer slices
                    })
                },
            }
        }).collect::<Vec<_>>();

        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let vk_writes = write.iter().enumerate().map(|(num, write)| {
            vk::WriteDescriptorSet {
                sType: vk::STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                pNext: ptr::null(),
                dstSet: self.set,
                dstBinding: write.binding,
                dstArrayElement: write.array_element,
                descriptorCount: 1,
                descriptorType: vk::DESCRIPTOR_TYPE_UNIFORM_BUFFER,     // FIXME:
                pImageInfo: ptr::null(),        // FIXME:
                pBufferInfo: if let Some(ref b) = buffer_descriptors[num] { b } else { ptr::null() },
                pTexelBufferView: ptr::null(),      // FIXME:
            }
        }).collect::<Vec<_>>();

        if !vk_writes.is_empty() {
            vk.UpdateDescriptorSets(self.pool.device().internal_object(),
                                    vk_writes.len() as u32, vk_writes.as_ptr(), 0, ptr::null());
        }
    }
}

unsafe impl<S> VulkanObject for DescriptorSet<S> {
    type Object = vk::DescriptorSet;

    #[inline]
    fn internal_object(&self) -> vk::DescriptorSet {
        self.set
    }
}

impl<S> Drop for DescriptorSet<S> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.pool.device().pointers();
            vk.FreeDescriptorSets(self.pool.device().internal_object(),
                                  self.pool.internal_object(), 1, &self.set);
        }
    }
}


/// Implemented on all `DescriptorSet` objects. Hides the template parameters.
pub unsafe trait AbstractDescriptorSet: ::VulkanObjectU64 {}
unsafe impl<S> AbstractDescriptorSet for DescriptorSet<S> {}

/// Describes the layout of all descriptors within a descriptor set.
pub struct DescriptorSetLayout<S> {
    layout: vk::DescriptorSetLayout,
    device: Arc<Device>,
    description: S,
}

impl<S> DescriptorSetLayout<S> where S: DescriptorSetDesc {
    pub fn new(device: &Arc<Device>, description: S)
               -> Result<Arc<DescriptorSetLayout<S>>, OomError>
    {
        let vk = device.pointers();

        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let bindings = description.descriptors().into_iter().map(|desc| {
            vk::DescriptorSetLayoutBinding {
                binding: desc.binding,
                descriptorType: desc.ty.vk_enum(),
                descriptorCount: desc.array_count,
                stageFlags: desc.stages.into(),
                pImmutableSamplers: ptr::null(),        // FIXME: not yet implemented
            }
        }).collect::<Vec<_>>();

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
            layout: layout,
            device: device.clone(),
            description: description,
        }))
    }

    #[inline]
    pub fn description(&self) -> &S {
        &self.description
    }
}

unsafe impl<S> VulkanObject for DescriptorSetLayout<S> {
    type Object = vk::DescriptorSetLayout;

    #[inline]
    fn internal_object(&self) -> vk::DescriptorSetLayout {
        self.layout
    }
}

impl<S> Drop for DescriptorSetLayout<S> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyDescriptorSetLayout(self.device.internal_object(), self.layout, ptr::null());
        }
    }
}

/// Implemented on all `DescriptorSetLayout` objects. Hides the template parameters.
pub unsafe trait AbstractDescriptorSetLayout: ::VulkanObjectU64 {}
unsafe impl<S> AbstractDescriptorSetLayout for DescriptorSetLayout<S> {}

/// A collection of `DescriptorSetLayout` structs.
// TODO: push constants.
pub struct PipelineLayout<P> {
    device: Arc<Device>,
    layout: vk::PipelineLayout,
    description: P,
    layouts: Vec<Arc<AbstractDescriptorSetLayout>>,     // TODO: is it necessary to keep the layouts alive? check the specs
}

impl<P> PipelineLayout<P> where P: PipelineLayoutDesc {
    /// Creates a new `PipelineLayout`.
    pub fn new(device: &Arc<Device>, description: P, layouts: P::DescriptorSetLayouts)
               -> Result<Arc<PipelineLayout<P>>, OomError>
    {
        let vk = device.pointers();

        let layouts = description.decode_descriptor_set_layouts(layouts);
        // TODO: allocate on stack instead (https://github.com/rust-lang/rfcs/issues/618)
        let layouts_ids = layouts.iter().map(|l| {
            // FIXME: check that they belong to the same device
            ::VulkanObjectU64::internal_object(&**l)
        }).collect::<Vec<_>>();

        let layout = unsafe {
            let infos = vk::PipelineLayoutCreateInfo {
                sType: vk::STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                setLayoutCount: layouts_ids.len() as u32,
                pSetLayouts: layouts_ids.as_ptr(),
                pushConstantRangeCount: 0,      // TODO: unimplemented
                pPushConstantRanges: ptr::null(),    // TODO: unimplemented
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreatePipelineLayout(device.internal_object(), &infos,
                                                      ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(PipelineLayout {
            device: device.clone(),
            layout: layout,
            description: description,
            layouts: layouts,
        }))
    }

    #[inline]
    pub fn description(&self) -> &P {
        &self.description
    }
}

unsafe impl<P> VulkanObject for PipelineLayout<P> {
    type Object = vk::PipelineLayout;

    #[inline]
    fn internal_object(&self) -> vk::PipelineLayout {
        self.layout
    }
}

impl<P> Drop for PipelineLayout<P> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyDescriptorSetLayout(self.device.internal_object(), self.layout, ptr::null());
        }
    }
}


pub unsafe trait DescriptorSetsCollection {
    type Iter: ExactSizeIterator<Item = Arc<AbstractDescriptorSet>>;

    fn list(&self) -> Self::Iter;

    fn is_compatible_with<P>(&self, pipeline_layout: &Arc<PipelineLayout<P>>) -> bool;
}

unsafe impl<T> DescriptorSetsCollection for Arc<DescriptorSet<T>>
    where T: 'static + DescriptorSetDesc
{
    type Iter = OptionIntoIter<Arc<AbstractDescriptorSet>>;

    #[inline]
    fn list(&self) -> Self::Iter {
        Some(self.clone() as Arc<_>).into_iter()
    }

    #[inline]
    fn is_compatible_with<P>(&self, pipeline_layout: &Arc<PipelineLayout<P>>) -> bool {
        // FIXME:
        true
    }
}
