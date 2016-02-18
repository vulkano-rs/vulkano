use std::ops::Range;
/*
typedef struct {
    VkStructureType                             sType;
    const void*                                 pNext;
    VkPipelineViewportStateCreateFlags          flags;
    uint32_t                                    viewportCount;
    const VkViewport*                           pViewports;
    uint32_t                                    scissorCount;
    const VkRect2D*                             pScissors;
} VkPipelineViewportStateCreateInfo;
*/

pub enum ViewportsState {
    Fixed {

    },

    DynamicViewports {

    },

    DynamicScissors {

    },

    Dynamic,
}

#[derive(Debug, Copy, Clone)]
pub struct Viewport {
    pub origin: [f32; 2],
    pub dimensions: [f32; 2],
    pub depth_range: Range<f32>,
}
