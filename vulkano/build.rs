// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::env;

fn main() {
    let target = env::var("TARGET").unwrap();
    if target.contains("apple-ios") {
        println!("cargo:rustc-link-search=framework=/Library/Frameworks/");
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=framework=MoltenVK");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=IOSurface");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
        println!("cargo:rustc-link-lib=framework=UIKit");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }
}
