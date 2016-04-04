// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::mem;
use std::ptr;
use std::sync::Arc;
use smallvec::SmallVec;

use check_errors;
use OomError;
use VulkanObject;
use VulkanPointers;
use vk;

use descriptor::descriptor::DescriptorDesc;
use descriptor::descriptor_set::DescriptorSetsCollection;
use descriptor::descriptor_set::UnsafeDescriptorSetLayout;
use device::Device;

pub mod custom_pipeline_macro;

/// Trait for objects that describe the layout of the descriptors and push constants of a pipeline.
pub unsafe trait PipelineLayout: PipelineLayoutDesc + 'static + Send + Sync {
    /// Returns the inner `UnsafePipelineLayout`.
    // TODO: should be named "inner()" after https://github.com/rust-lang/rust/issues/12808 is fixed
    fn inner_pipeline_layout(&self) -> &UnsafePipelineLayout;
}

/// Trait for objects that describe the layout of the descriptors and push constants of a pipeline.
pub unsafe trait PipelineLayoutDesc {
    type SetsIter: Iterator<Item = Self::DescIter>;
    type DescIter: Iterator<Item = DescriptorDesc>;

    /// Describes the layout of the descriptors of the pipeline.
    fn descriptors_desc(&self) -> Self::SetsIter;

    // TODO: describe push constants
}

/// Traits that allow determining whether a pipeline layout is a superset of another one.
///
/// This trait is automatically implemented on all types that implement `PipelineLayout`.
/// TODO: once specialization lands, we can add implementations that don't perform deep comparisons
pub unsafe trait PipelineLayoutSuperset<Other>: PipelineLayoutDesc
    where Other: PipelineLayoutDesc
{
    /// Returns true if `self` is a superset of `Other`.
    fn is_superset_of(&self, &Other) -> bool;
}

unsafe impl<T, U> PipelineLayoutSuperset<U> for T
    where T: PipelineLayoutDesc, U: PipelineLayoutDesc
{
    fn is_superset_of(&self, other: &U) -> bool {
        let mut other_descriptor_sets = other.descriptors_desc();

        for my_set in self.descriptors_desc() {
            let mut other_set = match other_descriptor_sets.next() {
                None => return false,
                Some(s) => s,
            };

            for my_desc in my_set {
                let other_desc = match other_set.next() {
                    None => return false,
                    Some(d) => d,
                };

                if !my_desc.is_superset_of(&other_desc) {
                    return false;
                }
            }
        }

        true
    }
}

/// Traits that allow determining whether 
pub unsafe trait PipelineLayoutSetsCompatible<Other>: PipelineLayout
    where Other: DescriptorSetsCollection
{
    /// Returns true if `Other` can be used with a pipeline that uses `self` as layout.
    fn is_compatible(&self, &Other) -> bool;
}

unsafe impl<T, U> PipelineLayoutSetsCompatible<U> for T
    where T: PipelineLayout, U: DescriptorSetsCollection
{
    fn is_compatible(&self, _: &U) -> bool {
        // FIXME:
        true
    }
}

/// Traits that allow determining whether 
// TODO: require a trait on Pc
pub unsafe trait PipelineLayoutPushConstantsCompatible<Pc>: PipelineLayout {
    /// Returns true if `Pc` can be used with a pipeline that uses `self` as layout.
    fn is_compatible(&self, &Pc) -> bool;
}

unsafe impl<T, U> PipelineLayoutPushConstantsCompatible<U> for T where T: PipelineLayout {
    fn is_compatible(&self, _: &U) -> bool {
        // FIXME:
        true
    }
}

/// Implemented on types whose exact layout is known at compile-time.
pub unsafe trait CompiletimePipelineLayout: PipelineLayout {
    type RawContent;
}

/// A collection of `DescriptorSetLayout` structs.
// TODO: push constants.
pub struct UnsafePipelineLayout {
    device: Arc<Device>,
    layout: vk::PipelineLayout,
    layouts: SmallVec<[Arc<UnsafeDescriptorSetLayout>; 16]>,
}

impl UnsafePipelineLayout {
    /// Creates a new `UnsafePipelineLayout`.
    // TODO: is this function unsafe?
    #[inline]
    pub unsafe fn new<'a, I>(device: &Arc<Device>, layouts: I)
                             -> Result<UnsafePipelineLayout, OomError>
        where I: IntoIterator<Item = &'a Arc<UnsafeDescriptorSetLayout>>
    {
        UnsafePipelineLayout::new_inner(device, layouts.into_iter().map(|e| e.clone()).collect())
    }

    // TODO: is this function unsafe?
    unsafe fn new_inner(device: &Arc<Device>,
                        layouts: SmallVec<[Arc<UnsafeDescriptorSetLayout>; 16]>)
                        -> Result<UnsafePipelineLayout, OomError>
    {
        let vk = device.pointers();

        // FIXME: check that they belong to the right device
        let layouts_ids = layouts.iter().map(|l| l.internal_object())
                                 .collect::<SmallVec<[_; 16]>>();

        let layout = {
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

        Ok(UnsafePipelineLayout {
            device: device.clone(),
            layout: layout,
            layouts: layouts,
        })
    }

    #[inline]
    pub fn descriptor_set_layout(&self, index: usize) -> Option<&Arc<UnsafeDescriptorSetLayout>> {
        self.layouts.get(index)
    }
}

unsafe impl VulkanObject for UnsafePipelineLayout {
    type Object = vk::PipelineLayout;

    #[inline]
    fn internal_object(&self) -> vk::PipelineLayout {
        self.layout
    }
}

impl Drop for UnsafePipelineLayout {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroyPipelineLayout(self.device.internal_object(), self.layout, ptr::null());
        }
    }
}
