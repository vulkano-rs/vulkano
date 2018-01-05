extern crate cmake;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

// A tool used by `glsl-to-spirv`.
trait Tool {
    // Name of binary.
    fn name(&self) -> &str;

    // Directory with source files.
    fn src_dir(&self) -> &Path;
    // Subpath to the binary, relative to build directory.
    fn bin_path(&self) -> &Path;

    // Builds the tool.
    fn build(&self);
}

// Tools which are built using CMake should use this function.
fn cmake_config(src_dir: &Path) -> cmake::Config {
    let mut cfg = cmake::Config::new(src_dir);

    // Create a different output dir for each tool.
    let out_dir = Path::new(&env::var("OUT_DIR").unwrap()).join(src_dir);
    fs::create_dir_all(&out_dir).unwrap();

    cfg.out_dir(out_dir);

    // Build all tools in release mode.
    cfg.profile("Release");

    cfg
}

// The glslangValidator is the reference GLSL compiler.
struct GlslangValidator;

impl Tool for GlslangValidator {
    fn name(&self) -> &str {
        "glslangValidator"
    }

    fn src_dir(&self) -> &Path {
        Path::new("glslang")
    }

    fn bin_path(&self) -> &Path {
        Path::new("bin/glslangValidator")
    }

    fn build(&self) {
        cmake_config(self.src_dir()).build();
    }
}

// The spirv-opt tool is used to optimize the generated shader code.
struct SpirvOpt;

impl Tool for SpirvOpt {
    fn name(&self) -> &str {
        "spirv-opt"
    }

    fn src_dir(&self) -> &Path {
        Path::new("spirv-tools")
    }

    fn bin_path(&self) -> &Path {
        Path::new("bin/spirv-opt")
    }

    fn build(&self) {
        let spirv_headers = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).join("spirv-headers");

        cmake_config(self.src_dir())
            .define("SPIRV-Headers_SOURCE_DIR", spirv_headers)
            .build();
    }
}

fn main() {
    // Cache the output dir, since it's used in multiple places.
    let out_dir = PathBuf::from(&env::var("OUT_DIR").unwrap());

    // Determine if we are on Windows and cache the result.
    let on_windows = {
        let target = env::var("TARGET").unwrap();
        target.contains("windows")
    };

    // A list of all tools we need to build.
    let tools: &[&Tool] = &[
        &GlslangValidator,
        &SpirvOpt,
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
        let file = tool.name();

        // Path to Windows pre-built binary.
        let exe_file = format!("build/{}.exe", file);

        // Rebuild if tool binary is updated.
        // This will also lead to Git updating the submodules.
        println!("cargo:rerun-if-changed={}", exe_file);

        let path = if on_windows {
            // TODO: check the hash of the file to make sure that it is not altered
            Path::new(&exe_file).to_owned()
        } else {
            // Compile the tool.
            tool.build();

            // Determine the subpath where the tool binary is.
            out_dir.join(tool.src_dir()).join(tool.bin_path())
        };

        // Determine the final file to copy to.
        let out_file = out_dir.join(file);

        // Copy the binary to the final destination.
        fs::copy(&path, &out_file).expect("failed to copy executable");
    }
}
