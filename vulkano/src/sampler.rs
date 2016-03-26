// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! How to retreive data from an image within a shader.
//! 
//! This module contains a struct named `Sampler` which describes how to get pixel data from
//! a texture.
//!
use std::mem;
use std::ptr;
use std::sync::Arc;

use device::Device;
use OomError;
use VulkanObject;
use VulkanPointers;
use check_errors;
use vk;

/// Describes how to retreive data from an image within a shader.
pub struct Sampler {
    device: Arc<Device>,
    sampler: vk::Sampler,
}

impl Sampler {
    /// Creates a new `Sampler` with the given behavior.
    ///
    /// # Panic
    ///
    /// - Panicks if `max_anisotropy < 1.0`.
    /// - Panicks if `min_lod > max_lod`.
    ///
    pub fn new(device: &Arc<Device>, mag_filter: Filter, min_filter: Filter,
               mipmap_mode: MipmapMode, address_u: SamplerAddressMode,
               address_v: SamplerAddressMode, address_w: SamplerAddressMode, mip_lod_bias: f32,
               max_anisotropy: f32, min_lod: f32, max_lod: f32) -> Result<Arc<Sampler>, OomError>
    {
        assert!(max_anisotropy >= 1.0);
        assert!(min_lod <= max_lod);

        let vk = device.pointers();

        let sampler = unsafe {
            let infos = vk::SamplerCreateInfo {
                sType: vk::STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,   // reserved
                magFilter: mag_filter as u32,
                minFilter: min_filter as u32,
                mipmapMode: mipmap_mode as u32,
                addressModeU: address_u as u32,
                addressModeV: address_v as u32,
                addressModeW: address_w as u32,
                mipLodBias: mip_lod_bias,
                anisotropyEnable: if max_anisotropy > 1.0 { vk::TRUE } else { vk::FALSE },
                maxAnisotropy: max_anisotropy,
                compareEnable: 0,       // FIXME: 
                compareOp: 0,       // FIXME: 
                minLod: min_lod,
                maxLod: max_lod,
                borderColor: 0,     // FIXME: 
                unnormalizedCoordinates: 0,     // FIXME: 
            };

            let mut output = mem::uninitialized();
            try!(check_errors(vk.CreateSampler(device.internal_object(), &infos,
                                               ptr::null(), &mut output)));
            output
        };

        Ok(Arc::new(Sampler {
            device: device.clone(),
            sampler: sampler,
        }))
    }
}

unsafe impl VulkanObject for Sampler {
    type Object = vk::Sampler;

    #[inline]
    fn internal_object(&self) -> vk::Sampler {
        self.sampler
    }
}

impl Drop for Sampler {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let vk = self.device.pointers();
            vk.DestroySampler(self.device.internal_object(), self.sampler, ptr::null());
        }
    }
}

/// Describes how the color of each pixel should be determined.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Filter {
    /// The four pixels whose center surround the requested coordinates are taken, then their
    /// values are interpolated.
    Linear = vk::FILTER_LINEAR,

    /// The pixel whose center is nearest to the requested coordinates is taken from the source
    /// and its value is returned as-is.
    Nearest = vk::FILTER_NEAREST,
}

/// Describes which mipmap from the source to use.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum MipmapMode {
    /// Use the mipmap whose dimensions are the nearest to the dimensions of the destination.
    Nearest = vk::SAMPLER_MIPMAP_MODE_NEAREST,

    /// Take the two mipmaps whose dimensions are immediately inferior and superior to the
    /// dimensions of the destination, calculate the value for both, and interpolate them.
    Linear = vk::SAMPLER_MIPMAP_MODE_LINEAR,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum SamplerAddressMode {
    Repeat = vk::SAMPLER_ADDRESS_MODE_REPEAT,
    MirroredRepeat = vk::SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
    ClampToEdge = vk::SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    ClampToBorder = vk::SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
    MirrorClampToEdge = vk::SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
}

#[cfg(test)]
mod tests {
    use sampler;

    #[test]
    fn create() {
        let (device, queue) = gfx_dev_and_queue!();

        let _ = sampler::Sampler::new(&device, sampler::Filter::Linear, sampler::Filter::Linear,
                                      sampler::MipmapMode::Nearest,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat, 1.0, 1.0,
                                      0.0, 2.0).unwrap();
    }

    #[test]
    #[should_panic]
    fn min_lod_inferior() {
        let (device, queue) = gfx_dev_and_queue!();

        let _ = sampler::Sampler::new(&device, sampler::Filter::Linear, sampler::Filter::Linear,
                                      sampler::MipmapMode::Nearest,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat, 1.0, 1.0, 5.0, 2.0);
    }

    #[test]
    #[should_panic]
    fn max_anisotropy() {
        let (device, queue) = gfx_dev_and_queue!();

        let _ = sampler::Sampler::new(&device, sampler::Filter::Linear, sampler::Filter::Linear,
                                      sampler::MipmapMode::Nearest,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat,
                                      sampler::SamplerAddressMode::Repeat, 1.0, 0.5, 0.0, 2.0);
    }
}
