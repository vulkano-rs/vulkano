#!/usr/bin/env bash
set -euo pipefail

# This script builds and runs all the examples
# It is NOT headless
# Human input is required to close all the windows
# Human monitoring is also required to check for errors in stdout

cargo build --bins
ls -F | grep '/$' | sed 's|/||' | xargs -E '' -I {} cargo run --bin {}
rm -f pipeline-caching/pipeline_cache.bin
