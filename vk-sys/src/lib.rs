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
