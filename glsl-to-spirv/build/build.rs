use std::env;
use std::fs;
use std::fs::Permissions;
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

    // setting permissions of the executable
    {
        #[cfg(linux)] fn permissions() -> Option<Permissions> {
            use std::os::unix::fs::PermissionsExt;
            Some(Permissions::from_mode(755))
        }
        #[cfg(not(linux))] fn permissions() -> Option<Permissions> { None }
        if let Some(permissions) = permissions() {
            fs::set_permissions(&out_file, permissions).unwrap();
        }
    }
}
