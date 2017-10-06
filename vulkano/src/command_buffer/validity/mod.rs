// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Functions that check the validity of commands.

pub use self::blit_image::{CheckBlitImageError, check_blit_image};
pub use self::clear_color_image::{CheckClearColorImageError, check_clear_color_image};
pub use self::copy_buffer::{CheckCopyBuffer, CheckCopyBufferError, check_copy_buffer};
pub use self::copy_image_buffer::{CheckCopyBufferImageError, CheckCopyBufferImageTy,
                                  check_copy_buffer_image};
pub use self::descriptor_sets::{CheckDescriptorSetsValidityError, check_descriptor_sets_validity};
pub use self::dispatch::{CheckDispatchError, check_dispatch};
pub use self::dynamic_state::{CheckDynamicStateValidityError, check_dynamic_state_validity};
pub use self::fill_buffer::{CheckFillBufferError, check_fill_buffer};
pub use self::index_buffer::{CheckIndexBuffer, CheckIndexBufferError, check_index_buffer};
pub use self::push_constants::{CheckPushConstantsValidityError, check_push_constants_validity};
pub use self::update_buffer::{CheckUpdateBufferError, check_update_buffer};
pub use self::vertex_buffers::{CheckVertexBuffer, CheckVertexBufferError, check_vertex_buffers};

mod blit_image;
mod clear_color_image;
mod copy_buffer;
mod copy_image_buffer;
mod descriptor_sets;
mod dispatch;
mod dynamic_state;
mod fill_buffer;
mod index_buffer;
mod push_constants;
mod update_buffer;
mod vertex_buffers;
