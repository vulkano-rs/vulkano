use std::env;

mod autogen;

fn main() {
    let target = env::var("TARGET").unwrap();
    if target.contains("apple-ios") || target.contains("apple-tvos") {
        println!("cargo:rustc-link-search=framework=/Library/Frameworks/");
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=framework=MoltenVK");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=IOSurface");
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
        println!("cargo:rustc-link-lib=framework=UIKit");
        println!("cargo:rustc-link-lib=framework=IOKit");
        println!("cargo:rustc-link-lib=framework=Foundation");
    }

    // Run autogen
    println!("cargo:rerun-if-changed=vk.xml");
    println!("cargo:rerun-if-changed=spirv.core.grammar.json");
    autogen::autogen();
}
