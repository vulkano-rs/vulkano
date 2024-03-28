//! A mode of rasterization where the edges of primitives are modified so that fragments are generated
//! if the edge of a primitive touches any part of a pixel, or if a pixel is fully covered by a primitive.

use crate::{
    device::Device, macros::vulkan_enum, ValidationError,
};

/// The state in a graphics pipeline describing how the conservative rasterization mode should behave.
#[derive(Clone, Debug)]
pub struct ConservativeRasterizationState {
    /// Sets the conservative rasterization mode.
    ///
    /// The default value is [`ConservativeRasterizationMode::Disabled`].
    pub mode: ConservativeRasterizationMode,


    /// The extra size in pixels to increase the generating primitive during conservative rasterization.
    /// If the mode is set to anything other than [`ConservativeRasterizationMode::Overestimate`] this
    /// value is ignored.
    /// 
    ///  The default value is 0.0.
    pub overestimation_size: f32,

    pub _ne: crate::NonExhaustive,
}

impl Default for ConservativeRasterizationState {
    #[inline]
    fn default() -> Self {
        Self {
            mode: ConservativeRasterizationMode::Disabled,
            overestimation_size: 0.0,
            _ne: crate::NonExhaustive(()),
        }
    }
}

impl ConservativeRasterizationState {
    pub(crate) fn validate(&self, device: &Device) -> Result<(), Box<ValidationError>> {
        let &Self {
            mode,
            overestimation_size,
            _ne: _,
        } = self;

        let properties = device.physical_device().properties();

        mode.validate_device(device).map_err(|err| {
            err.add_context("mode").set_vuids(&[
                "VUID-VkPipelineRasterizationConservativeStateCreateInfoEXT-conservativeRasterizationMode-parameter",
            ])
        })?;

        if overestimation_size < 0.0 || overestimation_size > properties.max_extra_primitive_overestimation_size.unwrap() {
            return Err(Box::new(ValidationError {
                context: "overestimation size".into(),
                problem: "the overestimation size is not in the range of 0.0 to `max_extra_primitive_overestimation_size` inclusive".into(),
                vuids: &[
                    "VUID-VkPipelineRasterizationConservativeStateCreateInfoEXT-extraPrimitiveOverestimationSize-01769",
                ],
                ..Default::default()
            }));
        }

        Ok(())
    }
}

vulkan_enum! {
    #[non_exhaustive]

    /// Describes how fragments will be generated based on how much is covered by a primitive.
    ConservativeRasterizationMode = ConservativeRasterizationModeEXT(i32);

    /// Conservative rasterization is disabled and rasterization proceeds as normal.
    Disabled = DISABLED,

    /// Fragments will be generated if any part of a primitive touches a pixel.
    Overestimate = OVERESTIMATE,

    /// Fragments will be generated only if a primitive completely covers a pixel.
    Underestimate = UNDERESTIMATE,
}

impl Default for ConservativeRasterizationMode {
    #[inline]
    fn default() -> ConservativeRasterizationMode {
        ConservativeRasterizationMode::Disabled
    }
}