#!/usr/bin/env pwsh

$ErrorActionPreference = "Stop"

$root = Get-Location
$profilePath = Join-Path -Path $root.Path -ChildPath "scripts/pgo-profiles/interpn.profdata"

$env:UV_NO_BUILD_CACHE = "1"
$env:RUSTFLAGS = "-Cprofile-use=$profilePath -Cmetadata=interpn_pgo"

uv pip install . --reinstall
