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

use buffer::TypedBuffer;
use descriptor::descriptor::DescriptorType;
use descriptor::descriptor::DescriptorWrite;
use image::ImageView;
use sampler::Sampler;

/// Call this macro with the layout of a pipeline to generate some helper structs that wrap around
/// vulkano's unsafe APIs.
// TODO: more docs
#[macro_export]
macro_rules! pipeline_layout {
    ($($name:ident: { $($field:ident: $ty:ty),* }),*) => {
        use std::sync::Arc;
        use std::vec::IntoIter as VecIntoIter;
        use $crate::OomError;
        use $crate::device::Device;
        use $crate::descriptor::descriptor::DescriptorDesc;
        use $crate::descriptor::pipeline_layout::PipelineLayout;
        use $crate::descriptor::pipeline_layout::PipelineLayoutDesc;
        use $crate::descriptor::pipeline_layout::UnsafePipelineLayout;

        pub struct CustomPipeline {
            inner: UnsafePipelineLayout
        }

        impl CustomPipeline {
            #[allow(unsafe_code)]
            pub fn new(device: &Arc<Device>) -> Result<Arc<CustomPipeline>, OomError> {
                let layouts = vec![
                    $(
                        try!($name::build_set_layout(device))
                    ),*
                ];

                let inner = unsafe {
                    try!(UnsafePipelineLayout::new(device, layouts.iter()))
                };

                Ok(Arc::new(CustomPipeline {
                    inner: inner
                }))
            }
        }

        #[allow(unsafe_code)]
        unsafe impl PipelineLayout for CustomPipeline {
            #[inline]
            fn inner_pipeline_layout(&self) -> &UnsafePipelineLayout {
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

    (__inner__ ($num:expr) $name:ident: { $($field:ident: $ty:ty),* } $($rest:tt)*) => {
        pub mod $name {
            #![allow(unused_imports)]

            use std::sync::Arc;
            use std::vec::IntoIter as VecIntoIter;
            use super::CustomPipeline;
            use $crate::OomError;
            use $crate::device::Device;
            use $crate::descriptor::descriptor::DescriptorDesc;
            use $crate::descriptor::descriptor::DescriptorWrite;
            use $crate::descriptor::descriptor::ShaderStages;
            use $crate::descriptor::descriptor_set::DescriptorPool;
            use $crate::descriptor::descriptor_set::DescriptorSet;
            use $crate::descriptor::descriptor_set::DescriptorSetDesc;
            use $crate::descriptor::descriptor_set::UnsafeDescriptorSet;
            use $crate::descriptor::descriptor_set::UnsafeDescriptorSetLayout;
            use $crate::descriptor::pipeline_layout::PipelineLayout;
            use $crate::descriptor::pipeline_layout::custom_pipeline_macro::CombinedImageSampler;
            use $crate::descriptor::pipeline_layout::custom_pipeline_macro::DescriptorMarker;
            use $crate::descriptor::pipeline_layout::custom_pipeline_macro::StorageBuffer;
            use $crate::descriptor::pipeline_layout::custom_pipeline_macro::UniformBuffer;
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
            impl<$($field: ValidParameter<$ty>),*> Descriptors<$($field),*> {
                pub fn writes(&self) -> Vec<DescriptorWrite> {
                    let mut writes = Vec::new();
                    let mut binding = 0;
                    $(
                        writes.push(self.$field.write(binding));
                        binding += 1;
                    )*
                    writes
                }
            }

            pub struct Set {
                inner: UnsafeDescriptorSet
            }

            impl Set {
                #[inline]
                #[allow(non_camel_case_types)]
                pub fn new<$($field: ValidParameter<$ty>),*>
                          (pool: &Arc<DescriptorPool>, layout: &Arc<CustomPipeline>,
                           descriptors: &Descriptors<$($field),*>)
                           -> Result<Arc<Set>, OomError>
                {
                    #![allow(unsafe_code)]
                    unsafe {
                        let layout = layout.inner_pipeline_layout().descriptor_set_layout($num).unwrap();
                        let mut set = try!(UnsafeDescriptorSet::uninitialized(pool, layout));
                        set.write(descriptors.writes());
                        Ok(Arc::new(Set { inner: set }))
                    }
                }
            }

            #[allow(unsafe_code)]
            unsafe impl DescriptorSet for Set {
                #[inline]
                fn inner_descriptor_set(&self) -> &UnsafeDescriptorSet {
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
            pub fn build_set_layout(device: &Arc<Device>)
                                    -> Result<Arc<UnsafeDescriptorSetLayout>, OomError>
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

                UnsafeDescriptorSetLayout::new(device, descriptors.into_iter())
            }
        }

        pipeline_layout!{__inner__ ($num+1) $($rest)*}
    };

    (__inner__ ($num:expr)) => {};
}

pub unsafe trait ValidParameter<Target> {
    fn write(&self, binding: u32) -> DescriptorWrite;
}

pub unsafe trait DescriptorMarker {
    fn descriptor_type() -> DescriptorType;
}

pub struct UniformBuffer<T: ?Sized>(PhantomData<T>);
unsafe impl<T: ?Sized> DescriptorMarker for UniformBuffer<T> {
    #[inline]
    fn descriptor_type() -> DescriptorType {
        DescriptorType::UniformBuffer
    }
}

unsafe impl<'a, B, T: ?Sized + 'static> ValidParameter<UniformBuffer<T>> for &'a Arc<B>
    where B: TypedBuffer<Content = T>
{
    #[inline]
    fn write(&self, binding: u32) -> DescriptorWrite {
        DescriptorWrite::uniform_buffer(binding, *self)
    }
}

pub struct StorageBuffer<T: ?Sized>(PhantomData<T>);
unsafe impl<T: ?Sized> DescriptorMarker for StorageBuffer<T> {
    #[inline]
    fn descriptor_type() -> DescriptorType {
        DescriptorType::StorageBuffer
    }
}

unsafe impl<'a, B, T: ?Sized + 'static> ValidParameter<StorageBuffer<T>> for &'a Arc<B>
    where B: TypedBuffer<Content = T>
{
    #[inline]
    fn write(&self, binding: u32) -> DescriptorWrite {
        DescriptorWrite::storage_buffer(binding, *self)
    }
}

pub struct CombinedImageSampler;
unsafe impl DescriptorMarker for CombinedImageSampler {
    #[inline]
    fn descriptor_type() -> DescriptorType {
        DescriptorType::CombinedImageSampler
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
