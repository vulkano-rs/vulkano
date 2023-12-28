// This module exposes what is needed in order to draw with a deferred rendering system.
//
// The main code is in the `system` module, while the other modules implement the different kinds
// of lighting sources.

use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

pub use self::system::{FrameSystem, Pass};

mod ambient_lighting_system;
mod directional_lighting_system;
mod point_lighting_system;
mod system;

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct LightingVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}
