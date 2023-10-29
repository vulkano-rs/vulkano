#!/bin/bash
set -e

# This script builds and runs all the examples
# It is NOT headless
# Human input is required to close all the windows
# Human monitoring is also required to check for errors in stdout

cargo build --bins
exa -F . | rg '/$' | sd '/' '' | rargs cargo run --bin {}
rm -f pipeline-caching/pipeline_cache.bin
