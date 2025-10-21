# Changelog

## 0.6.2 2025-10-20

Add optional use of fused multiply-add, enabled for python distributions.
This substantially improves floating-point roundoff; cubic method now shows
O(1e-14) peak roundoff error even under extrapolation of a quadratic function,
and 0-4 epsilon roundoff inside interpolating region.
Overall effect on throughput performance is neutral.

### Added

* Rust
  * Add `fma` feature

### Changed

* Rust
  * Use Horner's method for evaluating normalized cubic hermite spline
  * If `fma` feature is enabled, use FMA in cubic and linear methods where possible
* Python
  * Enable `fma` feature for python distribution
  * Update pgo profile data and benchmark plots

## 0.6.1 2025-10-10

### Changed

* Python
  * Pass PGO profile-use argument for rustc via maturin args instead of RUSTFLAGS to avoid overriding flags set in .cargo/config.toml
  * Split PGO scripts into native and distribution variants
    * Distribution variant tests the exact build configuration used for distribution
    * Native variant builds with target-cpu=native to enable all available instruction sets
  * Update baked PGO profile based on distribution build
  * Only run pypi distribution for single python version, because ABI3 build is portable to later python versions
* Rust
  * Add x64 to platforms where extra instruction sets are enabled in .cargo/config.toml to capture windows 64-bit x86

## 0.6.0 2025-10-05

Combine python bindings project into rust crate to streamline development process.
Implement PGO (profile-guided optimization) for python releases, giving a 1.2-2x speedup for
dimensions 1-4 at the expense of a ~2x slowdown for higher dimensions.

### Changed

* Python
  * Port python project from `interpnpy` repo
  * Implement PGO with hand-tuned profile workload
  * !Set `linearize_extrapolation=True` as default for cubic interpolators
  * Use pytest-cov for coverage testing
  * Update test deps & add linter/formatter configuration
  * Add uv lock and uv cache configuration
  * Use uv for actions
  * Add more vector instruction sets for x86_64 targets reflecting a ~2015-era CPU
* Rust
  * Eliminate some length check error handling that is no longer necessary for const-generic flattened methods
  * Add PyO3 bindings from python project as `python.rs` module behind `python` feature gate

## 0.2.6 2025-09-27

2-5x speedup and fully-analyzable call stack (no recursion) for lower dimensions
(1D-6D linear, 1D-4D cubic). Recursive methods still available for higher dimensions.

### Changed

* Update rust deps
* Reduce boilerplate in bindings
* Update perf plots
* Use abi3-py39 for extension module build

## 0.2.5 2025-03-14

### Changed

* Update interpn rust dep
  * ~2-6x speedup for lower-dimensional inputs
* Update bindings for latest pyo3 and numpy rust deps
* Add .cargo/config.toml with configuration to enable vector instruction sets for x86 targets
  * ~10-30% speedup for regular grid methods on x86 machines (compounded with earlier 2-6x speedup)
* Regenerate CI configuration
* Support python 3.13

## 0.2.3 2024-08-20

### Changed

* Update interpn rust dep to 0.4.3 to capture upgraded linear methods

## 0.2.2 2024-07-08

### Changed

* Update python deps incl. numpy >2
* Update rust deps
* Support python 3.12
