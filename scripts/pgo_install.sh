#!/bin/bash

# Build optimized & install
UV_NO_BUILD_CACHE=1 RUSTFLAGS="-Cprofile-use=${PWD}/scripts/pgo-profiles/interpn.profdata -Cmetadata=interpn_pgo" uv pip install .[test,bench] --reinstall --verbose
