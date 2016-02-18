
pub use self::graphics_pipeline::GraphicsPipeline;

//mod compute_pipeline;
mod graphics_pipeline;

pub mod blend;
pub mod cache;
//pub mod depth_stencil;
pub mod input_assembly;
pub mod multisample;
pub mod raster;
pub mod vertex;
pub mod viewport;

pub trait GenericPipeline {}
