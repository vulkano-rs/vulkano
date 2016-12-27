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

/// Wraps around two commands lists and joins them together in one list.
#[derive(Debug, Copy, Clone)]
pub struct CommandsListJoin<A, B>
    where A: CommandsList,
          B: CommandsList
{
    // First commands list.
    first: A,
    // Second commands list.
    second: B,
}

impl<A, B> CommandsListJoin<A, B>
    where A: CommandsList,
          B: CommandsList
{
    #[inline]
    pub fn new(first: A, second: B) -> CommandsListJoin<A, B> {
        CommandsListJoin {
            first: first,
            second: second,
        }
    }
}

unsafe impl<A, B> CommandsList for CommandsListJoin<A, B>
    where A: CommandsList,
          B: CommandsList
{
    #[inline]
    fn append<'a>(&'a self, builder: &mut CommandsListSink<'a>) {
        self.first.append(builder);
        self.second.append(builder);
    }
}
