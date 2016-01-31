use std::env;
use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build/glslangValidator");
    println!("cargo:rerun-if-changed=build/glslangValidator.exe");

    let target = env::var("TARGET").unwrap();
    let out_file = Path::new(&env::var("OUT_DIR").unwrap()).join("glslang_validator");

    let path = if target.contains("windows") {
        Path::new("build/glslangValidator.exe")
    } else if target.contains("linux") {
        Path::new("build/glslangValidator")
    } else {
        panic!("The platform `{}` is not supported", target);
    };

    if let Err(_) = fs::hard_link(&path, &out_file) {
        fs::copy(&path, &out_file).unwrap();
    }

    //fs::set_permissions(&out_file, std::io::USER_EXEC).unwrap();
}
