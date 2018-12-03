// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! State of multisampling.
//!
//! Multisampling allows you to ask the GPU to run the rasterizer to generate more than one
//! sample per pixel.
//!
//! For example, if `rasterization_samples` is 1 then the fragment shader, depth test and stencil
//! test will be run once for each pixel. However if `rasterization_samples` is `n`, then the
//! GPU will pick `n` different locations within each pixel and assign to each of these locations
//! a different depth value. Depth and stencil test will then be run `n` times.
//!
//! In addition to this, the `sample_shading` parameter is the proportion (between 0.0 and 1.0) or
//! the samples that will be run through the fragment shader. For example if you set this to 1.0,
//! then all the sub-pixel samples will run through the shader and get a different value. If you
//! set this to 0.5, about half of the samples will run through the shader and the other half will
//! get their values from the ones which went through the shader.
//!
//! If `alpha_to_coverage` is true, then the alpha value of the fragment will be used in
//! an implementation-defined way to determine which samples get disabled or not. For example if
//! the alpha value is 0.5, then about half of the samples will be discarded. If you render to a
//! multisample image, this means that the color will end up being mixed with whatever color was
//! underneath, which gives the same effect as alpha blending.
//!
//! If `alpha_to_one` is true, the alpha value of all the samples will be forced to 1.0 (or the
//! maximum possible value) after the effects of `alpha_to_coverage` have been applied.

// TODO: handle some weird behaviors with non-floating-point targets

/// State of the multisampling.
///
/// See the documentation in this module.
#[deprecated(note = "No longer needed")]
#[derive(Debug, Copy, Clone)]
pub struct Multisample {
    pub rasterization_samples: u32,
    pub sample_mask: [u32; 4],
    pub sample_shading: Option<f32>,
    pub alpha_to_coverage: bool,
    pub alpha_to_one: bool,
}

impl Multisample {
    #[inline]
    pub fn disabled() -> Multisample {
        Multisample {
            rasterization_samples: 1,
            sample_mask: [0xffffffff; 4],
            sample_shading: None,
            alpha_to_coverage: false,
            alpha_to_one: false,
        }
    }
}
