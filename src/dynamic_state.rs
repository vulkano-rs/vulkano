
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum FillMode {
    Solid,
    Wireframe,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum CullMode {
    None,
    Front,
    Back,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct RasterizerState {
    pub fill_mode: FillMode,
    pub cull_mode: CullMode,
    pub front_face: TODO,
    pub depth_bias: i32,
    pub depth_bias_clamp: f32,
    pub slope_scaled_depth_bias: f32,
}

