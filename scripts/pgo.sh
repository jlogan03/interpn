#!/bin/bash

# Get llvm-profdata from `apt install llvm-20`
# Must match rust's llvm version or it will crash

# Build instrumented
UV_NO_BUILD_CACHE=1 RUSTFLAGS="-Cprofile-generate=${PWD}/scripts/pgo-profiles/interpn.profraw -Cmetadata=interpn_pgo" uv pip install . --reinstall

# Run profile
uv run --no-sync ./scripts/profile_workload.py

# Merge profiles
/usr/lib/llvm-20/bin/llvm-profdata merge -o scripts/pgo-profiles/interpn.profdata $(find scripts/pgo-profiles/interpn.profraw -name '*.profraw')

# Build optimized
UV_NO_BUILD_CACHE=1 RUSTFLAGS="-Cprofile-use=${PWD}/scripts/pgo-profiles/interpn.profdata -Cmetadata=interpn_pgo" uv pip install . --reinstall
