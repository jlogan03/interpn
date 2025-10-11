#!/bin/bash

# Get llvm-profdata from `apt install llvm-20`
# Must match rust's llvm version or it will crash

# Build instrumented wheel
cargo clean
uv cache clean
uv pip install maturin
UV_NO_BUILD_CACHE=1 uv run --no-sync maturin build --compatibility pypi --out dist --verbose -- "-Cprofile-generate=${PWD}/scripts/pgo-profiles/interpn.profraw"

# Install instrumented wheel
uv pip install $(find dist/ -name 'interpn*.whl') --reinstall

# Run reference workload to generate profile
uv run --no-sync ./scripts/profile_workload.py

# Merge profiles
/usr/lib/llvm-20/bin/llvm-profdata merge -o scripts/pgo-profiles/interpn.profdata $(find scripts/pgo-profiles/interpn.profraw -name '*.profraw')
