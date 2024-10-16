cargo build --bins
@(Get-ChildItem -Directory -Name) | %{cargo run --bin $_}
rm pipeline-caching\pipeline_cache.bin
