#!/bin/bash

# Test-drive a profile-guided optimization workflow for a distributable wheel build.
# This builds a wheel in the way that it will be done in CI, storing the locally-generated
# PGO profile for reuse.

# Reusing the profile is possible because the counter instrumentation for
# generating the profile is inserted at the LLVM IR level, which does not differ between
# target platforms in this case; if cfg(target) directives were used to modify logic in
# the code, separate profiles would be beneficial, but no such cfg directives are used here.

# Dependencies
# * Get llvm-profdata from `apt install llvm-21`
#   * Must match rust's llvm version or it will crash
# * rustup component add llvm-tools-preview
#   * Provides PGO profile generation functionality

# Build instrumented, run profile, and merge profile files
bash scripts/distr_pgo_profile.sh

# Build optimized & install for testing
bash scripts/distr_pgo_install.sh

# Run tests with optimized build
uv run --no-sync pytest .
