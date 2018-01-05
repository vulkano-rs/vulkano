extern crate cmake;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Determine if we are on Windows and cache the result.
    let on_windows = {
        let target = env::var("TARGET").unwrap();
        target.contains("windows")
    };

    // A list of all tools we need to build.
    // Format: (<tool name>, <tool Git subdirectory>, <path to binary file>)
    let tools = [
        ("glslangValidator", "glslang", Path::new("bin/glslangValidator")),
        ("spirv-opt", "spirv-tools", Path::new("bin/spirv-opt")),
    ];

    // Update all Git submodules at once.
    if !on_windows {
        // Try to initialize submodules. Don't care if it fails, since this code also runs for
        // the crates.io package.
        let _ = Command::new("git")
            .arg("submodule")
            .arg("update")
            .arg("--init")
            .status();
    }

    // Cache the output dir, since it's used in multiple places.
    let out_dir = PathBuf::from(&env::var("OUT_DIR").unwrap());

    // Process all tools.
    for tool in tools.iter() {
        // The name of the tool binary.
        let file = tool.0;

        // Path to Windows pre-built binary.
        let exe_file = format!("build/{}.exe", file);

        // Rebuild if tool binary is updated.
        // This will also lead to Git updating the submodules.
        println!("cargo:rerun-if-changed={}", exe_file);

        let path = if on_windows {
            // TODO: check the hash of the file to make sure that it is not altered
            Path::new(&exe_file).to_owned()
        } else {
            // Determine the name of the CMake project to build.
            let project_name = tool.1;
            cmake::build(project_name);

            // Determine the subpath where the tool binary is.
            let bin_path = tool.2;
            out_dir.join(bin_path)
        };

        // Determine the final file to copy to.
        let out_file = out_dir.join(file);

        // Copy the binary to the final destination.
        fs::copy(&path, &out_file).expect("failed to copy executable");
    }
}
