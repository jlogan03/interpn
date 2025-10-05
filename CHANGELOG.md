# Changelog

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
