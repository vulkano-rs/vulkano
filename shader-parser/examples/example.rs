extern crate shader_parser;

use std::io::Cursor;

fn main() {
    let content = include_bytes!("example.spv");
    let output = shader_parser::reflect(Cursor::new(&content[..])).unwrap();
    println!("{}", output);
}
