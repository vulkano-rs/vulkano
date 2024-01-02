#!/usr/bin/env bash
set -euo pipefail

exa -F . | rg '/$' | sd '/' '' | rargs timeout --preserve-status 15s cargo run --bin {}
rm -f pipeline-caching/pipeline_cache.bin
