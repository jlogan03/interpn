#!/bin/bash

# Get llvm-profdata from `apt install llvm-20`
# Must match rust's llvm version or it will crash

# Build instrumented
cargo clean
uv cache clean
uv run --no-sync maturin develop --release --verbose -- "-Cprofile-generate=${PWD}/scripts/pgo-profiles/interpn.profraw -Ctarget-cpu=native"

# Run profile
uv run --no-sync ./scripts/profile_workload.py

# Merge profiles
/usr/lib/llvm-20/bin/llvm-profdata merge -o scripts/pgo-profiles/interpn.profdata $(find scripts/pgo-profiles/interpn.profraw -name '*.profraw')
