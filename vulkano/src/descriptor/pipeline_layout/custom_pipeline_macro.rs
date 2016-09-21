// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::marker::PhantomData;
use std::sync::Arc;

use buffer::Buffer;
use buffer::BufferView;
use buffer::TypedBuffer;
use descriptor::descriptor::DescriptorDescTy;
use descriptor::descriptor::DescriptorBufferDesc;
use descriptor::descriptor::DescriptorImageDesc;
use descriptor::descriptor::DescriptorImageDescDimensions;
use descriptor::descriptor::DescriptorImageDescArray;
use descriptor::descriptor_set::DescriptorWrite;
use descriptor::descriptor_set::resources_collection::Buf;
use descriptor::descriptor_set::resources_collection::ResourcesCollection;
use image::ImageView;
use sampler::Sampler;
use sync::AccessFlagBits;
use sync::PipelineStages;

/// Call this macro with the layout of a pipeline to generate some helper structs that wrap around
/// vulkano's unsafe APIs.
// TODO: more docs
#[macro_export]
macro_rules! pipeline_layout {
    (push_constants: { $($pc_f:ident: $pc_t:ty),* } $(, $name:ident: { $($field:ident: $ty:ty),* })*) => {
        use std::mem;
        use std::sync::Arc;
        use std::vec::IntoIter as VecIntoIter;
        use $crate::device::Device;
        use $crate::descriptor::descriptor::DescriptorDesc;
        use $crate::descriptor::descriptor::ShaderStages;
        use $crate::descriptor::pipeline_layout::PipelineLayout;
        use $crate::descriptor::pipeline_layout::PipelineLayoutDesc;
        use $crate::descriptor::pipeline_layout::UnsafePipelineLayout;
        use $crate::descriptor::pipeline_layout::UnsafePipelineLayoutCreationError;

        #[derive(Debug, Copy, Clone)]
        pub struct PushConstants {
            $(pub $pc_f: $pc_t,)*
        }

        pub struct CustomPipeline {
            inner: UnsafePipelineLayout
        }

        impl CustomPipeline {
            #[allow(unsafe_code)]
            pub fn new(device: &Arc<Device>)
                       -> Result<Arc<CustomPipeline>, UnsafePipelineLayoutCreationError>
            {
                let layouts = vec![
                    $(
                        Arc::new(try!($name::build_set_layout_raw(device)))
                    ),*
                ];

                let push_constants = if mem::size_of::<PushConstants>() >= 1 {
                    Some((0, mem::size_of::<PushConstants>(), ShaderStages::all()))
                } else {
                    None
                };

                let inner = {
                    try!(UnsafePipelineLayout::new(device, layouts.iter(), push_constants))
                };

                Ok(Arc::new(CustomPipeline {
                    inner: inner
                }))
            }
        }

        #[allow(unsafe_code)]
        unsafe impl PipelineLayout for CustomPipeline {
            #[inline]
            fn inner(&self) -> &UnsafePipelineLayout {
                &self.inner
            }
        }

        #[allow(unsafe_code)]
        unsafe impl PipelineLayoutDesc for CustomPipeline {
            type SetsIter = VecIntoIter<Self::DescIter>;
            type DescIter = VecIntoIter<DescriptorDesc>;

            fn descriptors_desc(&self) -> Self::SetsIter {
                // FIXME:
                vec![].into_iter()
            }
        }

        /* TODO: uncomment when specialization lands
        #[allow(unsafe_code)]
        unsafe impl<'a> PipelineLayoutSetsCompatible<($(&'a Arc<$name::Set>),*)> for CustomPipeline {
            #[inline]
            default fn is_compatible(&self, _: &($(&'a Arc<$name::Set>),*)) -> bool {
                true
            }
        }*/

        pipeline_layout!{__inner__ (0) $($name: {$($field: $ty),*})*}
    };

    ($($name:ident: { $($field:ident: $ty:ty),* }),*) => {
        pipeline_layout!{ push_constants: {} $(, $name: {$($field: $ty),*})* }
    };

    (__inner__ ($num:expr) $name:ident: { $($field:ident: $ty:ty),* } $($rest:tt)*) => {
        pub mod $name {
            #![allow(unused_imports)]

            use std::sync::Arc;
            use std::vec::IntoIter as VecIntoIter;
            use super::CustomPipeline;
            use $crate::OomError;
            use $crate::device::Device;
            use $crate::descriptor::descriptor::DescriptorDesc;
            use $crate::descriptor::descriptor::ShaderStages;
            use $crate::descriptor::descriptor_set::DescriptorPool;
            use $crate::descriptor::descriptor_set::DescriptorSet;
            use $crate::descriptor::descriptor_set::DescriptorSetDesc;
            use $crate::descriptor::descriptor_set::UnsafeDescriptorSet;
            use $crate::descriptor::descriptor_set::UnsafeDescriptorSetLayout;
            use $crate::descriptor::descriptor_set::DescriptorWrite;
            use $crate::descriptor::pipeline_layout::PipelineLayout;
            use $crate::descriptor::pipeline_layout::custom_pipeline_macro::UniformTexelBuffer;
            use $crate::descriptor::pipeline_layout::custom_pipeline_macro::StorageTexelBuffer;
            use $crate::descriptor::pipeline_layout::custom_pipeline_macro::CombinedImageSampler;
            use $crate::descriptor::pipeline_layout::custom_pipeline_macro::SampledImage;
            use $crate::descriptor::pipeline_layout::custom_pipeline_macro::DescriptorMarker;
            use $crate::descriptor::pipeline_layout::custom_pipeline_macro::StorageBuffer;
            use $crate::descriptor::pipeline_layout::custom_pipeline_macro::StorageImage;
            use $crate::descriptor::pipeline_layout::custom_pipeline_macro::UniformBuffer;
            use $crate::descriptor::pipeline_layout::custom_pipeline_macro::InputAttachment;
            use $crate::descriptor::pipeline_layout::custom_pipeline_macro::ValidParameter;

            // This constant is part of the API, but Rust sees it as dead code.
            #[allow(dead_code)]
            pub const SET_NUM: u32 = $num;

            #[allow(non_camel_case_types)]
            pub struct Descriptors<$($field),*> {
                $(
                    pub $field: $field
                ),*
            }

            #[allow(non_camel_case_types)]
            #[allow(unused_assignments)]
            #[inline]
            impl<$($field: ValidParameter<$ty>),*> Descriptors<$($field),*> {
                pub fn res(&self) -> ($(<$field as ValidParameter<$ty>>::Resource),*) {
                    (
                        ValidParameter<$ty>::build(&self.$field)
                    )
                }
            }

            pub struct Set {
                inner: UnsafeDescriptorSet
            }

            impl Set {
                #[inline]
                #[allow(non_camel_case_types)]
                pub fn raw<$($field: ValidParameter<$ty>),*>
                          (pool: &Arc<DescriptorPool>, layout: &Arc<CustomPipeline>,
                           descriptors: &Descriptors<$($field),*>)
                           -> Result<Set, OomError>
                {
                    #![allow(unsafe_code)]
                    unsafe {
                        let layout = layout.inner().descriptor_set_layout($num).unwrap();
                        let mut set = try!(UnsafeDescriptorSet::uninitialized_raw(pool, layout));
                        set.write(descriptors.writes());
                        Ok(Set { inner: set })
                    }
                }
                
                #[inline]
                #[allow(non_camel_case_types)]
                pub fn new<$($field: ValidParameter<$ty>),*>
                          (pool: &Arc<DescriptorPool>, layout: &Arc<CustomPipeline>,
                           descriptors: &Descriptors<$($field),*>)
                           -> Arc<Set>
                {
                    Arc::new(Set::raw(pool, layout, descriptors).unwrap())
                }
            }

            #[allow(unsafe_code)]
            unsafe impl DescriptorSet for Set {
                #[inline]
                fn inner(&self) -> &UnsafeDescriptorSet {
                    &self.inner
                }
            }

            #[allow(unsafe_code)]
            unsafe impl DescriptorSetDesc for Set {
                type Iter = VecIntoIter<DescriptorDesc>;

                #[inline]
                fn desc(&self) -> Self::Iter {
                    // FIXME:
                    vec![].into_iter()
                }
            }

            #[allow(unused_assignments)]
            #[allow(dead_code)]
            pub fn build_set_layout_raw(device: &Arc<Device>)
                                        -> Result<UnsafeDescriptorSetLayout, OomError>
            {
                let mut descriptors = Vec::new();
                let mut binding = 0;

                $(
                    descriptors.push(DescriptorDesc {
                        binding: binding,
                        ty: <$ty as DescriptorMarker>::descriptor_type(),
                        array_count: 1,                     // TODO:
                        stages: ShaderStages::all(),        // TODO:
                        readonly: false,                    // TODO:
                    });

                    binding += 1;
                )*

                UnsafeDescriptorSetLayout::raw(device.clone(), descriptors.into_iter())
            }

            #[inline]
            #[allow(dead_code)]
            pub fn build_set_layout(device: &Arc<Device>)
                                    -> Arc<UnsafeDescriptorSetLayout>
            {
                Arc::new(build_set_layout_raw(device).unwrap())
            }
        }

        pipeline_layout!{__inner__ ($num+1) $($rest)*}
    };

    (__inner__ ($num:expr)) => {};
}

pub unsafe trait ValidParameter<Target> {
    type Resource;
    fn build(self) -> Self::Resource;
}

pub unsafe trait DescriptorMarker {
    fn descriptor_type() -> DescriptorDescTy;
}

pub struct UniformBuffer<T: ?Sized>(PhantomData<T>);
unsafe impl<T: ?Sized> DescriptorMarker for UniformBuffer<T> {
    #[inline]
    fn descriptor_type() -> DescriptorDescTy {
        DescriptorDescTy::Buffer(DescriptorBufferDesc {
            dynamic: Some(false),
            storage: false,
        })
    }
}

unsafe impl<'a, B, T: ?Sized + 'static> ValidParameter<UniformBuffer<T>> for B
    where B: TypedBuffer<Content = T>
{
    type Resource = Buf<B>;

    #[inline]
    fn build(self) -> Self::Resource {
        let size = self.size();

        Buf {
            buffer: self,
            offset: 0,
            size: size,
            write: false,
            stage: PipelineStages {       // FIXME:
                all_graphics: true,
                all_commands: true,
                .. PipelineStages::none()
            },
            access: AccessFlagBits::all(),      // FIXME:
        }
    }
}

pub struct StorageBuffer<T: ?Sized>(PhantomData<T>);
unsafe impl<T: ?Sized> DescriptorMarker for StorageBuffer<T> {
    #[inline]
    fn descriptor_type() -> DescriptorDescTy {
        DescriptorDescTy::Buffer(DescriptorBufferDesc {
            dynamic: Some(false),
            storage: true,
        })
    }
}

unsafe impl<'a, B, T: ?Sized + 'static> ValidParameter<StorageBuffer<T>> for B
    where B: TypedBuffer<Content = T>
{
    type Resource = Buf<B>;

    #[inline]
    fn build(self) -> Self::Resource {
        let size = self.size();

        Buf {
            buffer: self,
            offset: 0,
            size: size,
            write: false,
            stage: PipelineStages {       // FIXME:
                all_graphics: true,
                all_commands: true,
                .. PipelineStages::none()
            },
            access: AccessFlagBits::all(),      // FIXME:
        }
    }
}
/*
pub struct UniformTexelBuffer;
unsafe impl DescriptorMarker for UniformTexelBuffer {
    #[inline]
    fn descriptor_type() -> DescriptorDescTy {
        DescriptorDescTy::TexelBuffer {
            storage: false,
            format: None,       // TODO:
        }
    }
}

unsafe impl<'a, B, F> ValidParameter<UniformTexelBuffer> for &'a Arc<BufferView<F, B>>   // TODO: format not checked
    where B: Buffer, F: 'static + Send + Sync
{
    type Resource = Buf<B>;

    #[inline]
    fn build(self) -> Self::Resource {
        let buffer = self.buffer();
        let size = self.size();

        Buf {
            buffer: self,
            offset: 0,
            size: size,
            write: false,
            stage: PipelineStages::all(),       // FIXME:
            access: AccessFlagBits::all(),      // FIXME:
        }
    }
}

pub struct StorageTexelBuffer;
unsafe impl DescriptorMarker for StorageTexelBuffer {
    #[inline]
    fn descriptor_type() -> DescriptorDescTy {
        DescriptorDescTy::TexelBuffer {
            storage: true,
            format: None,       // TODO:
        }
    }
}

unsafe impl<'a, B, F> ValidParameter<StorageTexelBuffer> for &'a Arc<BufferView<F, B>>   // TODO: format not checked
    where B: Buffer, F: 'static + Send + Sync
{
    #[inline]
    fn write(&self, binding: u32) -> DescriptorWrite {
        DescriptorWrite::storage_texel_buffer(binding, *self)
    }
}

pub struct CombinedImageSampler;
unsafe impl DescriptorMarker for CombinedImageSampler {
    #[inline]
    fn descriptor_type() -> DescriptorDescTy {
        DescriptorDescTy::CombinedImageSampler(DescriptorImageDesc {
            sampled: true,
            // FIXME: correct values
            dimensions: DescriptorImageDescDimensions::TwoDimensional,
            multisampled: false,
            array_layers: DescriptorImageDescArray::NonArrayed,
            format: None,
        })
    }
}

unsafe impl<'a, I> ValidParameter<CombinedImageSampler> for (&'a Arc<Sampler>, &'a Arc<I>)
    where I: ImageView + 'static
{
    #[inline]
    fn write(&self, binding: u32) -> DescriptorWrite {
        DescriptorWrite::combined_image_sampler(binding, self.0, self.1)
    }
}

pub struct SampledImage;
unsafe impl DescriptorMarker for SampledImage {
    #[inline]
    fn descriptor_type() -> DescriptorDescTy {
        DescriptorDescTy::Image(DescriptorImageDesc {
            sampled: true,
            // FIXME: correct values
            dimensions: DescriptorImageDescDimensions::TwoDimensional,
            multisampled: false,
            array_layers: DescriptorImageDescArray::NonArrayed,
            format: None,
        })
    }
}

unsafe impl<'a, I> ValidParameter<SampledImage> for &'a Arc<I>
    where I: ImageView + 'static
{
    #[inline]
    fn write(&self, binding: u32) -> DescriptorWrite {
        DescriptorWrite::sampled_image(binding, self)
    }
}

pub struct StorageImage;
unsafe impl DescriptorMarker for StorageImage {
    #[inline]
    fn descriptor_type() -> DescriptorDescTy {
        DescriptorDescTy::Image(DescriptorImageDesc {
            sampled: false,
            // FIXME: correct values
            dimensions: DescriptorImageDescDimensions::TwoDimensional,
            multisampled: false,
            array_layers: DescriptorImageDescArray::NonArrayed,
            format: None,
        })
    }
}

unsafe impl<'a, I> ValidParameter<StorageImage> for &'a Arc<I>
    where I: ImageView + 'static
{
    #[inline]
    fn write(&self, binding: u32) -> DescriptorWrite {
        DescriptorWrite::storage_image(binding, self)
    }
}

pub struct InputAttachment;
unsafe impl DescriptorMarker for InputAttachment {
    #[inline]
    fn descriptor_type() -> DescriptorDescTy {
        // FIXME: correct values
        DescriptorDescTy::InputAttachment { multisampled: false, array_layers: DescriptorImageDescArray::NonArrayed }
    }
}

unsafe impl<'a, I> ValidParameter<InputAttachment> for &'a Arc<I>
    where I: ImageView + 'static
{
    #[inline]
    fn write(&self, binding: u32) -> DescriptorWrite {
        DescriptorWrite::input_attachment(binding, self)
    }
}*/

#[cfg(test)]
mod tests {
    #[test]
    fn no_warning() {
        #![deny(warnings)]
        mod layout {
            pipeline_layout! {
                set0: {
                    field1: UniformBuffer<[u8]>,
                    field2: UniformBuffer<[u8]>
                },
                set1: {
                    field1: UniformBuffer<[u8]>,
                    field2: UniformBuffer<[u8]>
                }
            }
        }
    }
}
