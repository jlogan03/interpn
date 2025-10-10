#!/bin/bash

# Build optimized & install
cargo clean
uv cache clean
UV_NO_BUILD_CACHE=1 uv run --no-sync maturin build --compatibility pypi --out dist --verbose -- "-Cprofile-use=${PWD}/scripts/pgo-profiles/interpn.profdata"

uv pip install dist/interpn-0.6.0-cp39-abi3-manylinux_2_34_x86_64.whl --reinstall
