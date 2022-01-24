use nalgebra::{Point3, Vector2, Vector3};
use vulkano::impl_vertex;

/// The structure representing a CPU-readable 3D model not uploaded to the GPU.
pub struct Model {
    /// In indexed models, a "vertex" represents a set of properties
    /// that may be used by a corner of a triangle.
    /// They don't need to be in any particular order (although should),
    /// but shouldn't contain duplicates if properly constructed.
    /// https://gamedev.stackexchange.com/questions/40406/does-the-order-of-vertex-buffer-data-when-rendering-indexed-primitives-matter
    pub vertices: Vec<Vertex>,
    /// And the triangles are generated like this:
    /// ```
    /// a_list_of_u32
    ///     .batch_every_3()
    ///     .map(|(a, b, c)| Triangle {vertices[a], vertiecs[b], vertices[c]})
    /// ```
    /// Note that the *winding order* of the vertex buffer
    /// affects what is considered "back face" by culling.
    /// You could choose to use either counterclockwise or clockwise (as long as it's consistent),
    /// but here we'll use the Blender behavior of counterclockwise == front.
    /// Of course, if you set the primitive topology to something other than `TriangleList`,
    /// the generation will be different. But there's little reason in doing that in indexed drawing.
    /// https://stackoverflow.com/questions/12094607/trianglelist-versus-trianglestrip
    pub indices: Vec<u32>,
    /// The identifier to the base color texture.
    pub base_color: String,
}

/// The structure representing a vertex, both usable by the CPU and the GPU.
#[repr(C)]
// ^-- Needed so the struct's runtime memory structure will be somehow more predictable.
// Learn more about repr(C):
// https://doc.rust-lang.org/nomicon/other-reprs.html
#[derive(Default, Debug, Clone)] // Required by Vulkano
pub struct Vertex {
    /// The position in a Z-up Y-back right-handed coordinate system (to match Blender).
    pub position: Point3<f32>,
    /// The object-space normal in a Z-up Y-back right-handed coordinate system (to match Blender).
    pub normal: Vector3<f32>,
    /// The texture coordinates in a bottom-left origin coordinate system
    /// (also to match Blender, and OpenGL).
    /// You should probably convert to top-left origin uvs before hand
    /// if you have no interest in easier integration with Blender or existing OpenGL code,
    /// so you could delete the Y inversion in renderer/passes/main_pass/geometry_subpass.vert.
    pub uv: Vector2<f32>,
}

// Vulkano's black magic macro. Don't ask how, just remember to call it.
impl_vertex!(Vertex, position, normal, uv);
