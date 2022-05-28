// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Generates multiple fragments per framebuffer pixel when rasterizing. This can be used for
//! anti-aliasing.

use crate::image::SampleCount;

// TODO: handle some weird behaviors with non-floating-point targets

/// State of the multisampling.
#[derive(Copy, Clone, Debug)]
pub struct MultisampleState {
    /// The number of rasterization samples to take per pixel. The GPU will pick this many different
    /// locations within each pixel and assign to each of these locations a different depth value.
    /// The depth and stencil test will then be run for each sample.
    ///
    /// The default value is [`SampleCount::Sample1`].
    pub rasterization_samples: SampleCount,

    /// Controls the proportion (between 0.0 and 1.0) of the samples that will be run through the
    /// fragment shader.
    ///
    /// If the value is 1.0, then all sub-pixel samples will run
    /// through the shader and get a different value. If the value is 0.5, about half of the samples
    /// will run through the shader and the other half will get their values from the ones which
    /// went through the shader.
    ///
    /// If set to `Some`, the [`sample_rate_shading`](crate::device::Features::sample_rate_shading)
    /// feature must be enabled on the device.
    pub sample_shading: Option<f32>,

    /// A mask of bits that is ANDed with the coverage mask of each set of `rasterization_samples`
    /// samples. Only the first `rasterization_samples / 32` bits are used, the rest is ignored.
    ///
    /// The default value is `[0xFFFFFFFF; 2]`.
    pub sample_mask: [u32; 2], // 64 bits for needed for 64 SampleCount

    /// Controls whether the alpha value of the fragment will be used in an implementation-defined
    /// way to determine which samples get disabled or not. For example if the alpha value is 0.5,
    /// then about half of the samples will be discarded. If you render to a multisample image, this
    /// means that the color will end up being mixed with whatever color was underneath, which gives
    /// the same effect as alpha blending.
    pub alpha_to_coverage_enable: bool,

    /// Controls whether the alpha value of all the samples will be forced to 1.0 (or the
    /// maximum possible value) after the effects of `alpha_to_coverage` have been applied.
    ///
    /// If set to `true`, the [`alpha_to_one`](crate::device::Features::alpha_to_one)
    /// feature must be enabled on the device.
    pub alpha_to_one_enable: bool,
}

impl MultisampleState {
    /// Creates a `MultisampleState` with multisampling disabled.
    #[inline]
    pub fn new() -> MultisampleState {
        MultisampleState {
            rasterization_samples: SampleCount::Sample1,
            sample_shading: None,
            sample_mask: [0xFFFFFFFF; 2],
            alpha_to_coverage_enable: false,
            alpha_to_one_enable: false,
        }
    }
}

impl Default for MultisampleState {
    /// Returns [`MultisampleState::new()`].
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
