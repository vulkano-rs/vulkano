// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::io::Write;
use std::io::Error as IoError;

pub fn write_bindings<W>(write: &mut W) -> Result<(), IoError>
    where W: Write
{
    try!(write!(write, "{}", include_str!("functions.rs")));
    try!(write!(write, "{}", include_str!("enums.rs")));
    try!(write!(write, "{}", include_str!("structs.rs")));
    Ok(())
}
