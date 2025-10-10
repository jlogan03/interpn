#!/bin/bash

# Build instrumented, run profile, and merge profile files
bash scripts/native_pgo_profile.sh

# Build optimized & install for testing
bash scripts/native_pgo_install.sh

# Run tests with optimized build
uv run --no-sync pytest .
