// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

#[macro_export]
macro_rules! pipeline_layout {
    ($($name:ident: { $($field:ident $ty:ty),* }),*) => (
        #![allow(unsafe_code)]

        use std::sync::Arc;
        use $crate::OomError;
        use $crate::descriptor_set::descriptor_set::DescriptorSet;
        use $crate::descriptor_set::descriptor_set::UnsafeDescriptorSet;
        use $crate::descriptor_set::descriptor_set::UnsafeDescriptorSetLayout;
        use $crate::descriptor_set::pipeline_layout::PipelineLayout;
        use $crate::descriptor_set::pipeline_layout::UnsafePipelineLayout;

        pub struct CustomPipeline {
            inner: UnsafePipelineLayout
        }

        impl CustomPipeline {
            pub fn new(device: &Arc<Device>) -> Result<Arc<CustomPipeline>, OomError> {
                let layouts = vec![
                    $(
                        try!($name::layout(device))
                    ),*
                ];

                let inner = try!(UnsafePipelineLayout::new(device, layouts.iter()));

                Ok(CustomPipeline {
                    inner: inner
                })
            }
        }

        unsafe impl PipelineLayout for CustomPipeline {
            #[inline]
            fn inner_pipeline_layout(&self) -> &UnsafePipelineLayout {
                &self.inner
            }
        }

        $(
            pub struct $name {
                inner: UnsafeDescriptorSet
            }

            impl $name {
                fn layout(device: &Arc<Device>)
                          -> Result<Arc<UnsafeDescriptorSetLayout>, OomError>
                {
                    unimplemented!()
                }
            }

            unsafe impl DescriptorSet for $name {
                #[inline]
                fn inner_descriptor_set(&self) -> &UnsafeDescriptorSet {
                    &self.inner
                }
            }

        )*
    );
}
