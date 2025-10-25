#!/bin/bash

# Build optimized wheel
cargo clean
uv cache clean
uv pip install maturin
rm -rf dist/
UV_NO_BUILD_CACHE=1 uv run --no-sync maturin build --compatibility pypi --out dist --verbose -- "-Cprofile-use=${PWD}/scripts/pgo-profiles/pgo.profdata" "-Cllvm-args=-pgo-warn-missing-function"

# Install from wheel
uv pip install $(find dist/ -name '*.whl')[pydantic] --group test --reinstall
