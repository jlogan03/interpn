#!/bin/bash

# Build optimized & install
cargo clean
uv cache clean
UV_NO_BUILD_CACHE=1 RUSTFLAGS="-Cprofile-use=${PWD}/scripts/pgo-profiles/interpn.profdata -Ctarget-feature=+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b,+avx,+avx2,+fma,+bmi1,+bmi2,+lzcnt,+pclmulqdq,+movbe" uv run --no-sync maturin build --compatibility pypi --out dist --verbose

uv pip install dist/interpn-0.6.0-cp39-abi3-manylinux_2_34_x86_64.whl --reinstall
