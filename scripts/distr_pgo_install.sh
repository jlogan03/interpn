#!/bin/bash

# Build optimized wheel
cargo clean
uv cache clean
uv pip install maturin
UV_NO_BUILD_CACHE=1 uv run --no-sync maturin build --compatibility pypi --out dist --verbose -- "-Cprofile-use=${PWD}/scripts/pgo-profiles/interpn.profdata"

# Install from wheel
uv pip install $(find dist/ -name 'interpn*.whl') --reinstall
