use crate::scene::camera::Camera;
use crate::scene::light::PointLight;
use nalgebra::{Affine3, Point3};

/// Describes the scene from the renderer's perspective.
pub struct Scene {
    pub objects: Vec<(String, Affine3<f32>)>,
    pub point_lights: Vec<(PointLight, Point3<f32>)>,
    pub camera: Camera,
}
