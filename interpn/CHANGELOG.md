# Changelog

## 0.4.3 - 2024-08-03

## Changed

* Use recursive method to evaluate multilinear interpolation instead of hypercube method
  * This makes extrapolation cost consistent with interpolation cost

## 0.4.2 - 2024-05-12

## Added

* Implement cubic rectilinear method

## 0.4.1 - 2024-05-06

## Fixed

* Fix grid cell index selection to properly center the grid cell s.t. t=0 corresponds to index 1

## Added

* Add test of cubic method against sine function to capture potential problems not visible when testing against linear and quadratic functions

## 0.4.0 - 2024-05-05

## Changes

* Implement cubic interpolation for regular grid
    * Continuous first derivative everywhere
    * Option to clamp to linear extrapolation to prevent unwanted extrapolation of local curvature
        * Test linearized method against linear function in both interpolation and extrapolation
        * Test un-linearized method against quadratic function in both interpolation and extrapolation

## 0.3.0 - 2023-12-17

## Changed

* Remove initial guess for cell index from rectilinear method
* Collapse some loops
* Remove support for negative step sizes for regular grid in favor of reducing number of abs() calls
* Remove some saturating sub calls that are not needed now that degenerate grids are not supported
* Get indexing dimension product in the same way for rectilinear method as for regular grid method
* Use better initial value for folds
* Update docs
* Use optimizations for tests because it's faster overall

## 0.2.0 - 2023-12-06

## Changed

* Propagate Result everywhere that unwrap or assert! was being used
