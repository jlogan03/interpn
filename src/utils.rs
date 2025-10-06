//! Convenience methods for constructing grids in a way that echoes,
//! but does not exactly match, methods common in scripting languages.
use itertools::Itertools;
use num_traits::Float;

/// Generates evenly spaced values from start to stop,
/// including the endpoint.
pub fn linspace<T>(start: T, stop: T, n: usize) -> Vec<T>
where
    T: Float,
{
    let dx: T = (stop - start) / T::from(n - 1).unwrap();
    (0..n).map(|i| start + T::from(i).unwrap() * dx).collect()
}

/// Generates a meshgrid in C ordering (x0, y0, z0, x0, y0, z1, ..., x0, yn, zn)
pub fn meshgrid<T>(x: Vec<&Vec<T>>) -> Vec<Vec<T>>
where
    T: Float,
{
    x.into_iter()
        .multi_cartesian_product()
        .map(|xx| xx.iter().map(|y| **y).collect())
        .collect()
}
