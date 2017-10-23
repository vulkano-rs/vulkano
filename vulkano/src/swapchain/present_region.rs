// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use swapchain::Swapchain;
use vk;

/// Represents a region on an image.
///
/// A region consists of an arbitrary amount of rectangles.
#[derive(Debug, Clone)]
pub struct PresentRegion {
    pub rectangles: Vec<RectangleLayer>,
}

impl PresentRegion {
    /// Returns true if this present region is compatible with swapchain.
    pub fn is_compatible_with<W>(&self, swapchain: &Swapchain<W>) -> bool {
        self.rectangles
            .iter()
            .all(|rect| rect.is_compatible_with(swapchain))
    }
}

/// Represents a rectangular region on an image layer.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct RectangleLayer {
    /// Coordinates in pixels of the top-left hand corner of the rectangle.
    pub offset: [i32; 2],

    /// Dimensions in pixels of the rectangle.
    pub extent: [u32; 2],

    /// The layer of the image. For images with only one layer, the value of layer must be 0.
    pub layer: u32,
}

impl RectangleLayer {
    /// Returns true if this rectangle layer is compatible with swapchain.
    pub fn is_compatible_with<W>(&self, swapchain: &Swapchain<W>) -> bool {
        // FIXME negative offset is not disallowed by spec, but semantically should not be possible
        debug_assert!(self.offset[0] >= 0);
        debug_assert!(self.offset[1] >= 0);
        self.offset[0] as u32 + self.extent[0] <= swapchain.dimensions()[0] &&
            self.offset[1] as u32 + self.extent[1] <= swapchain.dimensions()[1] &&
            self.layer < swapchain.layers()
    }

    pub(crate) fn to_vk(&self) -> vk::RectLayerKHR {
        vk::RectLayerKHR {
            offset: vk::Offset2D {
                x: self.offset[0],
                y: self.offset[1],
            },
            extent: vk::Extent2D {
                width: self.extent[0],
                height: self.extent[1],
            },
            layer: self.layer,
        }
    }
}
