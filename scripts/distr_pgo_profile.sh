#!/bin/bash

# Get llvm-profdata from `apt install llvm-20`
# Must match rust's llvm version or it will crash

# Build instrumented
cargo clean
uv cache clean
UV_NO_BUILD_CACHE=1 RUSTFLAGS="-Cprofile-generate=${PWD}/scripts/pgo-profiles/interpn.profraw -Ctarget-feature=+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b,+avx,+avx2,+fma,+bmi1,+bmi2,+lzcnt,+pclmulqdq,+movbe" uv run --no-sync maturin build --compatibility pypi --out dist --verbose

# Run profile (update wheel name here)
uv pip install dist/interpn-0.6.0-cp39-abi3-manylinux_2_34_x86_64.whl --reinstall
uv run --no-sync ./scripts/profile_workload.py

# Merge profiles
/usr/lib/llvm-20/bin/llvm-profdata merge -o scripts/pgo-profiles/interpn.profdata $(find scripts/pgo-profiles/interpn.profraw -name '*.profraw')
