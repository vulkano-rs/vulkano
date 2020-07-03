fn main() {
    if cfg!(not(target_os = "ios")) {
        println!("cargo:rustc-link-lib=vulkan");
    }
}
