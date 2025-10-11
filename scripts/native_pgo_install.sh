#!/bin/bash

# Build optimized & install
cargo clean
uv cache clean
uv run --no-sync maturin develop --release --verbose -- "-Cprofile-use=${PWD}/scripts/pgo-profiles/pgo_native.profdata" "-Ctarget-cpu=native"
