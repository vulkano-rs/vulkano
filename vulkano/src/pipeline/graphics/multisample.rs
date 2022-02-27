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

use crate::device::Device;
use crate::image::SampleCount;
use crate::pipeline::graphics::GraphicsPipelineCreationError;
use crate::pipeline::DynamicState;
use crate::render_pass::Subpass;
use std::collections::HashMap;
use std::ptr;

// TODO: handle some weird behaviors with non-floating-point targets

/// State of the multisampling.
#[derive(Copy, Clone, Debug)]
pub struct MultisampleState {
    /*
        TODO: enable
        /// The number of rasterization samples to take per pixel. The GPU will pick this many different
        /// locations within each pixel and assign to each of these locations a different depth value.
        /// The depth and stencil test will then be run for each sample.
        pub rasterization_samples: SampleCount,
    */
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

    /*
        TODO: enable
        pub sample_mask: Option<[u32; 2]>, // 64 bits for needed for 64 SampleCount
    */
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
            //rasterization_samples: 1,
            sample_shading: None,
            //sample_mask: [0xffffffff; 4],
            alpha_to_coverage_enable: false,
            alpha_to_one_enable: false,
        }
    }

    pub(crate) fn to_vulkan(
        &self,
        device: &Device,
        dynamic_state_modes: &mut HashMap<DynamicState, bool>,
        subpass: &Subpass,
    ) -> Result<ash::vk::PipelineMultisampleStateCreateInfo, GraphicsPipelineCreationError> {
        let rasterization_samples = subpass.num_samples().unwrap_or(SampleCount::Sample1).into();

        let (sample_shading_enable, min_sample_shading) =
            if let Some(min_sample_shading) = self.sample_shading {
                if !device.enabled_features().sample_rate_shading {
                    return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                        feature: "sample_rate_shading",
                        reason: "MultisampleState::sample_shading was Some",
                    });
                }
                assert!(min_sample_shading >= 0.0 && min_sample_shading <= 1.0); // TODO: return error?
                (ash::vk::TRUE, min_sample_shading)
            } else {
                (ash::vk::FALSE, 0.0)
            };

        let alpha_to_one_enable = {
            if self.alpha_to_one_enable {
                if !device.enabled_features().alpha_to_one {
                    return Err(GraphicsPipelineCreationError::FeatureNotEnabled {
                        feature: "alpha_to_one",
                        reason: "MultisampleState::alpha_to_one was true",
                    });
                }
            }
            self.alpha_to_one_enable as ash::vk::Bool32
        };

        Ok(ash::vk::PipelineMultisampleStateCreateInfo {
            flags: ash::vk::PipelineMultisampleStateCreateFlags::empty(),
            rasterization_samples,
            sample_shading_enable,
            min_sample_shading,
            p_sample_mask: ptr::null(),
            alpha_to_coverage_enable: self.alpha_to_coverage_enable as ash::vk::Bool32,
            alpha_to_one_enable,
            ..Default::default()
        })
    }
}

impl Default for MultisampleState {
    /// Returns [`MultisampleState::new()`].
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}
