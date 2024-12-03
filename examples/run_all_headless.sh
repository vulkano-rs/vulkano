#!/usr/bin/env sh
set -eu

ls -F | grep '/$' | sed 's|/$||' | xargs -E '' -I {} timeout --preserve-status 15s cargo run --bin {}
rm -f pipeline-caching/pipeline_cache.bin
