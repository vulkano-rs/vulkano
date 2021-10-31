// Copyright (c) 2021 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

mod aabb;
mod triangles;

use super::acceleration_struct::AccelerationStructure;
use crate::buffer::{BufferAccess, TypedBufferAccess};
use std::sync::Arc;

pub use aabb::AabbPosition;

enum BottomLevelData {
    Aabb {
        buffer: Arc<dyn TypedBufferAccess<Content = [AabbPosition]>>,
    },
    Triangles {
        vertex_buffer: Arc<dyn BufferAccess>,
        index_buffer: Arc<dyn BufferAccess>, 
    },
}

pub struct BottomLevelAccelerationStructure {
    pub(crate) acceleration_structure: AccelerationStructure,
    data: BottomLevelData,
}
