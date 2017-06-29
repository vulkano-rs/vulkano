// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Functions that check the validity of commands.

pub use self::copy_buffer::{CheckCopyBufferError, check_copy_buffer, CheckCopyBuffer};
pub use self::dynamic_state::{CheckDynamicStateValidityError, check_dynamic_state_validity};
pub use self::fill_buffer::{CheckFillBufferError, check_fill_buffer};
pub use self::index_buffer::{check_index_buffer, CheckIndexBuffer, CheckIndexBufferError};
pub use self::update_buffer::{CheckUpdateBufferError, check_update_buffer};

mod copy_buffer;
mod dynamic_state;
mod fill_buffer;
mod index_buffer;
mod update_buffer;
