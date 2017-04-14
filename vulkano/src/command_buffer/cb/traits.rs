// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

/// Adds a command to a command buffer builder.
pub unsafe trait AddCommand<C> {
    /// The new command buffer builder type.
    type Out;

    /// Adds the command. This takes ownership of the builder and returns a new builder with the
    /// command appended at the end of it.
    fn add(self, cmd: C) -> Self::Out;
}
