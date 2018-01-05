extern crate cmake;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// A tool used by `glsl-to-spirv`.
struct Tool<'a> {
    name: &'a str,
    builder: &'a Fn(),
    bin_path: &'a Path,
}

fn create_cmake_config(out_dir: &Path, tool_name: &str) -> cmake::Config {
    let mut cfg = cmake::Config::new(tool_name);

    // Create a different output dir for each tool.
    let out_dir = out_dir.join(tool_name);
    fs::create_dir_all(&out_dir).unwrap();

    cfg.out_dir(out_dir);

    // Build all tools in release mode.
    cfg.profile("Release");

    cfg
}

fn main() {
    // Cache the output dir, since it's used in multiple places.
    let out_dir = PathBuf::from(&env::var("OUT_DIR").unwrap());

    let manifest_dir = PathBuf::from(&env::var("CARGO_MANIFEST_DIR").unwrap());

    // Determine if we are on Windows and cache the result.
    let on_windows = {
        let target = env::var("TARGET").unwrap();
        target.contains("windows")
    };

    // A list of all tools we need to build.
    let tools = [
        // The glslangValidator is the reference GLSL compiler.
        Tool {
            name: "glslangValidator",
            builder: &|| { create_cmake_config(&out_dir, "glslang").build(); },
            bin_path: Path::new("glslang/bin/glslangValidator"),
        },
        // The spirv-opt tool is used to optimize the generated shader code.
        Tool {
            name: "spirv-opt",
            builder: &|| {
                create_cmake_config(&out_dir, "spirv-tools")
                    .define("SPIRV-Headers_SOURCE_DIR", manifest_dir.join("spirv-headers"))
                    .build();
            },
            bin_path: Path::new("spirv-tools/bin/spirv-opt"),
        },
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

    // Process all tools.
    for tool in tools.iter() {
        // The name of the tool binary.
        let file = tool.name;

        // Path to Windows pre-built binary.
        let exe_file = format!("build/{}.exe", file);

        // Rebuild if tool binary is updated.
        // This will also lead to Git updating the submodules.
        println!("cargo:rerun-if-changed={}", exe_file);

        let path = if on_windows {
            // TODO: check the hash of the file to make sure that it is not altered
            Path::new(&exe_file).to_owned()
        } else {
            // Build the tool.
            (tool.builder)();

            // Determine the subpath where the tool binary is.
            out_dir.join(tool.bin_path)
        };

        // Determine the final file to copy to.
        let out_file = out_dir.join(file);

        // Copy the binary to the final destination.
        fs::copy(&path, &out_file).expect("failed to copy executable");
    }
}
