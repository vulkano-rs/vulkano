// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

extern crate vulkano_www;

use std::env;

fn main() {
    let addr = env::var("ADDR").unwrap_or("0.0.0.0:8000".to_owned());
    println!("Listening on {}", addr);
    vulkano_www::start(&addr)
}
