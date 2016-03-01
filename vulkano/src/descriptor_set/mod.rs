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
//! in the pipeline. The `T` parameter must implement the `Layout` trait and describes
//! the descriptor sets and descriptors on vulkano's side.
//!
//! To build a `PipelineLayout`, you need to pass a collection of `DescriptorSetLayout` structs.
//! A `DescriptorSetLayout<T>` if the equivalent of `PipelineLayout` but for a single descriptor
//! set. The `T` parameter must implement the `SetLayout` trait.
//!
//! # Binding resources
//! 
//! In parallel of the pipeline initialization, you have to create a `DescriptorSet<T>`. This
//! struct contains the list of actual resources that will be bound when the pipeline is executed.
//! To build a `DescriptorSet<T>`, you need to pass a `DescriptorSetLayout<T>`. The `T` parameter
//! must implement `SetLayout` as if the same for both the descriptor set and its layout.
//!
//! TODO: describe descriptor set writes
//!
//! # Shader analyser
//! 
//! While you can manually implement the `Layout` and `SetLayout` traits on
//! your own types, it is encouraged to use the `vulkano-shaders` crate instead. This crate will
//! automatically parse your SPIR-V code and generate structs that implement these traits and
//! describe the pipeline layout to vulkano.

use std::option::IntoIter as OptionIntoIter;
use std::sync::Arc;

pub use self::layout_def::Layout;
pub use self::layout_def::SetLayout;
pub use self::layout_def::SetLayoutWrite;
pub use self::layout_def::SetLayoutInit;
pub use self::layout_def::DescriptorWrite;
pub use self::layout_def::DescriptorBind;
pub use self::layout_def::DescriptorDesc;
pub use self::layout_def::DescriptorType;
pub use self::layout_def::ShaderStages;
pub use self::pool::DescriptorPool;
pub use self::runtime_desc::RuntimeDesc;
pub use self::runtime_desc::EmptyPipelineDesc;
pub use self::runtime_desc::RuntimeDescriptorSetDesc;
pub use self::vk_objects::DescriptorSet;
pub use self::vk_objects::AbstractDescriptorSet;
pub use self::vk_objects::DescriptorSetLayout;
pub use self::vk_objects::AbstractDescriptorSetLayout;
pub use self::vk_objects::PipelineLayout;

mod layout_def;
mod pool;
mod runtime_desc;
mod vk_objects;

/// A collection of descriptor set objects.
pub unsafe trait DescriptorSetsCollection {
    /// An iterator that produces the list of descriptor set objects contained in this collection.
    type Iter: ExactSizeIterator<Item = Arc<AbstractDescriptorSet>>;

    /// Returns the list of descriptor set objects of this collection.
    fn list(&self) -> Self::Iter;

    fn is_compatible_with<P>(&self, pipeline_layout: &Arc<PipelineLayout<P>>) -> bool;
}

unsafe impl DescriptorSetsCollection for () {
    type Iter = OptionIntoIter<Arc<AbstractDescriptorSet>>;

    #[inline]
    fn list(&self) -> Self::Iter {
        None.into_iter()
    }

    #[inline]
    fn is_compatible_with<P>(&self, pipeline_layout: &Arc<PipelineLayout<P>>) -> bool {
        // FIXME:
        true
    }
}

unsafe impl<T> DescriptorSetsCollection for Arc<DescriptorSet<T>>
    where T: 'static + SetLayout
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

/*
#[macro_export]
macro_rules! pipeline_layout {
    (sets: {$($set_name:ident: { $($name:ident : ),* }),*}) => {
        mod layout {
            use std::sync::Arc;
            use $crate::descriptor_set::DescriptorType;
            use $crate::descriptor_set::DescriptorDesc;
            use $crate::descriptor_set::SetLayout;
            use $crate::descriptor_set::DescriptorWrite;
            use $crate::descriptor_set::DescriptorBind;
            use $crate::descriptor_set::PipelineLayout;
            use $crate::descriptor_set::Layout;
            use $crate::descriptor_set::ShaderStages;
            use $crate::buffer::AbstractBuffer;

            $(
                pub struct $set_name;
                unsafe impl SetLayout for $set_name {
                    type Write = (      // FIXME: variable number of elems
                        Arc<AbstractBuffer>     // FIXME: strong typing
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
            unsafe impl Layout for Layout {
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
