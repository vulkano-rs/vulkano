use std::io::Error as IoError;
use std::io::Read;

pub use parse::ParseError;

mod parse;

pub fn reflect<R>(mut spirv: R) -> Result<String, Error>
    where R: Read
{
    let mut data = Vec::new();
    try!(spirv.read_to_end(&mut data));

    // now parsing the document
    let doc = try!(parse::parse_spirv(&data));

    unimplemented!()
}

pub enum Error {
    IoError(IoError),
    ParseError(ParseError),
}

impl From<IoError> for Error {
    #[inline]
    fn from(err: IoError) -> Error {
        Error::IoError(err)
    }
}

impl From<ParseError> for Error {
    #[inline]
    fn from(err: ParseError) -> Error {
        Error::ParseError(err)
    }
}
