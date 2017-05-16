// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

//! Additional commands built on top of core commands.
//!
//! These commands are specific to vulkano and make it easier to perform common operations.

pub use self::dispatch::{CmdDispatch, CmdDispatchError};
//pub use self::dispatch_indirect::{CmdDispatchIndirect, CmdDispatchIndirectError};
pub use self::draw::CmdDraw;
pub use self::draw_indexed::CmdDrawIndexed;
pub use self::draw_indirect::CmdDrawIndirect;

mod dispatch;
//mod dispatch_indirect;
mod draw;
mod draw_indexed;
mod draw_indirect;
