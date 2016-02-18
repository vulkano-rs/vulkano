use std::ops::Range;
use vk;

#[derive(Debug, Clone)]
pub enum ViewportsState {
    Fixed {
        data: Vec<(Viewport, Scissor)>,
    },

    DynamicViewports {
        scissors: Vec<Scissor>,
    },

    DynamicScissors {
        viewports: Vec<Viewport>,
    },

    Dynamic {
        num: u32,
    },
}

impl ViewportsState {
    pub fn dynamic_viewports(&self) -> bool {
        match *self {
            ViewportsState::Fixed { .. } => false,
            ViewportsState::DynamicViewports { .. } => true,
            ViewportsState::DynamicScissors { .. } => false,
            ViewportsState::Dynamic { .. } => true,
        }
    }

    pub fn dynamic_scissors(&self) -> bool {
        match *self {
            ViewportsState::Fixed { .. } => false,
            ViewportsState::DynamicViewports { .. } => false,
            ViewportsState::DynamicScissors { .. } => true,
            ViewportsState::Dynamic { .. } => true,
        }
    }

    pub fn num_viewports(&self) -> u32 {
        match *self {
            ViewportsState::Fixed { ref data } => data.len() as u32,
            ViewportsState::DynamicViewports { ref scissors } => scissors.len() as u32,
            ViewportsState::DynamicScissors { ref viewports } => viewports.len() as u32,
            ViewportsState::Dynamic { num } => num,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Viewport {
    pub origin: [f32; 2],
    pub dimensions: [f32; 2],
    pub depth_range: Range<f32>,
}

#[doc(hidden)]
impl Into<vk::Viewport> for Viewport {
    #[inline]
    fn into(self) -> vk::Viewport {
        vk::Viewport {
            x: self.origin[0],
            y: self.origin[1],
            width: self.dimensions[0],
            height: self.dimensions[1],
            minDepth: self.depth_range.start,
            maxDepth: self.depth_range.end,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Scissor {
    pub origin: [i32; 2],
    pub dimensions: [u32; 2],
}

#[doc(hidden)]
impl Into<vk::Rect2D> for Scissor {
    #[inline]
    fn into(self) -> vk::Rect2D {
        vk::Rect2D {
            offset: vk::Offset2D {
                x: self.origin[0],
                y: self.origin[1],
            },
            extent: vk::Extent2D {
                width: self.dimensions[0],
                height: self.dimensions[1],
            },
        }
    }
}
