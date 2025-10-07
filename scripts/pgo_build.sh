#!/bin/bash

# Build optimized wheel
UV_NO_BUILD_CACHE=1 RUSTFLAGS="-Cprofile-use=${PWD}/scripts/pgo-profiles/interpn.profdata -Cmetadata=interpn_pgo" uv build .
