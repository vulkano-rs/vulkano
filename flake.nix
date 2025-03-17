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
            (rust-bin.stable.latest.default.override {
              extensions = [ "rust-analyzer" "rust-src" ];
            })

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

          LD_LIBRARY_PATH = "${lib.makeLibraryPath buildInputs}";
          SHADERC_LIB_DIR = "${shaderc.lib}/lib";
          VK_LAYER_PATH = "${vulkan-validation-layers}/share/vulkan/explicit_layer.d";
        };
      }
    );
}
