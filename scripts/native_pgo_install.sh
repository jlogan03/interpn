#!/bin/bash

# Build optimized & install
cargo clean
uv cache clean
uv run --no-sync maturin develop --release --verbose -- "-Cprofile-use=${PWD}/scripts/pgo-profiles/interpn.profdata -Ctarget-cpu=native"
