#!/bin/bash

# Build and install the fastest version that we can make for this specific device.
# Uses all available instruction sets, the environment's default C runtime, and
# profile-guided optimization with a profile made on this device for this build.
# The profile generated from this run will not be ideal for reuse, because the
# native-optimized install differs slightly from the distribution-optimized install.

# Dependencies
# * Get llvm-profdata from `apt install llvm-20`
#   * Must match rust's llvm version or it will crash
# * rustup component add llvm-tools-preview
#   * Provides PGO profile generation functionality

# Build instrumented, run profile, and merge profile files
bash scripts/native_pgo_profile.sh

# Build optimized & install for testing
bash scripts/native_pgo_install.sh

# Run tests with optimized build
uv run --no-sync pytest .
