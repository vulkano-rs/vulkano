use std::f32::consts::PI;

use nalgebra::Vector3;

/// Describes a uniform omnidirectional point light source.
pub struct PointLight {
    /// The separate luminance (total energy emitted in all directions) for RGB channels.
    /// Probably not a physically correct model of lights.
    pub luminance: Vector3<f32>,
}

impl PointLight {
    pub fn new(luminous_flux: Vector3<f32>, radius: f32) -> Self {
        let luminance = luminous_flux / (4. * PI * PI * radius * radius);
        Self { luminance }
    }
}
