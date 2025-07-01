{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs @ { self, nixpkgs, flake-utils, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
      in {
        devShells.default = with pkgs; mkShell rec {
          buildInputs = [
            (rust-bin.stable.latest.minimal.override {
              extensions = [ "clippy" "rust-analyzer" "rust-docs" "rust-src" ];
            })
            # We use nightly rustfmt features.
            (rust-bin.selectLatestNightlyWith (toolchain: toolchain.rustfmt))

            # Vulkan dependencies
            shaderc
            spirv-tools
            vulkan-loader
            vulkan-tools
            vulkan-tools-lunarg
            vulkan-validation-layers

            # winit dependencies
            libxkbcommon
            wayland
            xorg.libX11
            xorg.libXcursor
            xorg.libXi
            xorg.libXrandr
          ];

          LD_LIBRARY_PATH = lib.makeLibraryPath buildInputs;
          SHADERC_LIB_DIR = lib.makeLibraryPath [ shaderc ];
          VK_LAYER_PATH = "${vulkan-validation-layers}/share/vulkan/explicit_layer.d";
        };
        devShells.CI = with pkgs; mkShell rec {
          buildInputs = [
            (rust-bin.stable.latest.minimal.override {
              extensions = [ "clippy" ];
              # Windows CI unfortunately needs to cross-compile from within WSL because Nix doesn't
              # work on Windows.
              targets = [ "x86_64-pc-windows-msvc" ];
            })
            # We use nightly rustfmt features.
            (rust-bin.selectLatestNightlyWith (toolchain: toolchain.rustfmt))

            # Vulkan dependencies
            shaderc

            # winit dependencies
            libxkbcommon
            wayland
            xorg.libX11
            xorg.libXcursor
            xorg.libXi
            xorg.libXrandr
          ];

          LD_LIBRARY_PATH = lib.makeLibraryPath buildInputs;
          SHADERC_LIB_DIR = lib.makeLibraryPath [ shaderc ];
        };
      }
    );
}
