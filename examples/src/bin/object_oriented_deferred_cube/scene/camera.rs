use nalgebra::{Matrix4, Point3, Vector3};

/// Describes a look-at camera that generates view and projection matrices.
/// TODO Look-at model might not be what you want, and you should hack the depth mapping yourself:
/// https://developer.nvidia.com/content/depth-precision-visualized
#[derive(Clone)]
pub struct Camera {
    /// The position of the pinhole camera's aperture.
    pub position: Point3<f32>,
    /// The target location ("focus") the camera is looking at.
    /// Note it merely decides the direction - depth of field needs separate approximation
    /// (because pinhole cameras don't have something called DoF, neither does the linear rasterizer)
    pub target: Point3<f32>,
    /// The up direction of the camera.
    /// It doesn't have to be equal to the real up unit vector, the matrix builder will use
    /// the closest possible one instead.
    pub up: Vector3<f32>,
    /// Vertical field of view specified in radians.
    /// Fixed vertical field of view is the most popular way of specifying FoV in landscape games,
    /// but you might want to change this to adapt to the shortest dimension.
    /// https://en.wikipedia.org/wiki/Field_of_view_in_video_games
    pub fov_y: f32,
    /// The near plane, i.e. the closest depth that will get into the camera.
    /// Usually > 1 because of how floating point precision works;
    /// Learn more by following the link in the struct's comment.
    pub near: f32,
    /// The far plane, which is the opposite of the near plane.
    /// Value depends on your scene scale (but should be obviously bigger than `near`).
    pub far: f32,
}

impl Camera {
    pub fn view_transform(&self) -> Matrix4<f32> {
        Matrix4::look_at_rh(&self.position, &self.target, &self.up)
    }

    pub fn projection_transform(&self, aspect_ratio: f32) -> Matrix4<f32> {
        // nalgebra's projection matrix generator is designed for OpenGL screen space coordinates,
        // where the up direction is +Y, the opposite of Vulkan;
        // Although we can use hacks
        // https://www.saschawillems.de/blog/2019/03/29/flipping-the-vulkan-viewport/
        // or flip the Y in the shaders,
        // it is arguably better to pre-flip on the CPU than in every shader invocation (?)
        const FLIP_Y: Matrix4<f32> = Matrix4::new(
            1., 0., 0., 0., //
            0., -1., 0., 0., //
            0., 0., 1., 0., //
            0., 0., 0., 1., //
        );
        FLIP_Y * Matrix4::new_perspective(aspect_ratio, self.fov_y, self.near, self.far)
    }
}
