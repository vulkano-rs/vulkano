extern crate cmake;

use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build/glslangValidator.exe");

    let target = env::var("TARGET").unwrap();
    let out_file = Path::new(&env::var("OUT_DIR").unwrap()).join("glslang_validator");

    let path = if target.contains("windows") {
        // TODO: check the hash of the file to make sure that it is not altered
        Path::new("build/glslangValidator.exe").to_owned()

    } else {
        // Try to initialize submodules. Don't care if it fails, since this code also runs for
        // the crates.io package.
        let _ = Command::new("git").arg("submodule").arg("update").arg("--init").status();
        cmake::build("glslang");
        Path::new(&env::var("OUT_DIR").unwrap()).join("bin").join("glslangValidator")
    };

    if let Err(_) = fs::hard_link(&path, &out_file) {
        fs::copy(&path, &out_file).expect("failed to copy executable");
    }
}
