#!/bin/bash
set -euo pipefail
shopt -s globstar nullglob

# This script searches the `~/.cargo/target` folder for any `exe` files and sets their executable
# bit. This is a hacky workaround for the fact that rustc's MSVC target doesn't set the bit when
# cross-compiling from within WSL even though the files are stored on the Linux filesystem and can't
# be executed from within WSL otherwise.

for exe in ~/.cargo/target/**/*.exe; do
  chmod +x "$exe"
done
