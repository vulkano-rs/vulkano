// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use command_buffer::CommandsList;
use command_buffer::CommandsListSink;

#[inline]
pub fn empty() -> EmptyCommandsList {
    EmptyCommandsList
}

#[derive(Debug, Copy, Clone)]
pub struct EmptyCommandsList;

unsafe impl CommandsList for EmptyCommandsList {
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
    }
}
