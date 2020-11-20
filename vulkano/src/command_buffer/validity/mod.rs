// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Functions that check the validity of commands.

pub use self::blit_image::{check_blit_image, CheckBlitImageError};
pub use self::clear_color_image::{check_clear_color_image, CheckClearColorImageError};
pub use self::copy_buffer::{check_copy_buffer, CheckCopyBuffer, CheckCopyBufferError};
pub use self::copy_image::{check_copy_image, CheckCopyImageError};
pub use self::copy_image_buffer::{
    check_copy_buffer_image, CheckCopyBufferImageError, CheckCopyBufferImageTy,
};
pub use self::debug_marker::{check_debug_marker_color, CheckColorError};
pub use self::descriptor_sets::{check_descriptor_sets_validity, CheckDescriptorSetsValidityError};
pub use self::dispatch::{check_dispatch, CheckDispatchError};
pub use self::dynamic_state::{check_dynamic_state_validity, CheckDynamicStateValidityError};
pub use self::fill_buffer::{check_fill_buffer, CheckFillBufferError};
pub use self::index_buffer::{check_index_buffer, CheckIndexBuffer, CheckIndexBufferError};
pub use self::push_constants::{check_push_constants_validity, CheckPushConstantsValidityError};
pub use self::update_buffer::{check_update_buffer, CheckUpdateBufferError};
pub use self::vertex_buffers::{check_vertex_buffers, CheckVertexBuffer, CheckVertexBufferError};

mod blit_image;
mod clear_color_image;
mod copy_buffer;
mod copy_image;
mod copy_image_buffer;
mod debug_marker;
mod descriptor_sets;
mod dispatch;
mod dynamic_state;
mod fill_buffer;
mod index_buffer;
mod push_constants;
mod update_buffer;
mod vertex_buffers;
