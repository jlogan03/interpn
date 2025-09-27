//! Multilinear interpolation/extrapolation on a rectilinear grid.
//!
//! ```rust
//! use interpn::multilinear::rectilinear;
//!
//! // Define a grid
//! let x = [1.0_f64, 1.2, 2.0];
//! let y = [1.0_f64, 1.3, 1.5];
//!
//! // Grid input for rectilinear method
//! let grids = &[&x[..], &y[..]];
//!
//! // Values at grid points
//! let z = [2.0; 9];
//!
//! // Points to interpolate/extrapolate
//! let xobs = [0.0_f64, 5.0];
//! let yobs = [-1.0, 3.0];
//! let obs = [&xobs[..], &yobs[..]];
//!
//! // Storage for output
//! let mut out = [0.0; 2];
//!
//! // Do interpolation, allocating for the output for convenience
//! rectilinear::interpn_alloc(grids, &z, &obs).unwrap();
//! ```
//!
//! References
//! * https://en.wikipedia.org/wiki/Bilinear_interpolation#Weighted_mean
use crate::index_arr;
use num_traits::Float;

/// Evaluate multicubic interpolation on a regular grid in up to 8 dimensions.
/// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
///
/// This is a convenience function; best performance will be achieved by using the exact right
/// number for the MAXDIMS parameter, as this will slightly reduce compute and storage overhead,
/// and the underlying method can be extended to more than this function's limit of 8 dimensions.
/// The limit of 8 dimensions was chosen for no more specific reason than to reduce unit test times.
///
/// While this method initializes the interpolator struct on every call, the overhead of doing this
/// is minimal even when using it to evaluate one observation point at a time.
pub fn interpn<T: Float>(
    grids: &[&[T]],
    vals: &[T],
    obs: &[&[T]],
    out: &mut [T],
) -> Result<(), &'static str> {
    // Expanding out and using the specialized version for each size
    // gives a substantial speedup for lower dimensionalities
    // (4-5x speedup for 1-dim compared to using MAXDIMS=8)
    let ndims = grids.len();
    match ndims {
        x if x == 1 => {
            MultilinearRectilinearRecursive::<'_, T, 1>::new(grids, vals)?.interp(obs, out)
        }
        x if x == 2 => {
            MultilinearRectilinearRecursive::<'_, T, 2>::new(grids, vals)?.interp(obs, out)
        }
        x if x == 3 => {
            MultilinearRectilinearRecursive::<'_, T, 3>::new(grids, vals)?.interp(obs, out)
        }
        x if x == 4 => {
            MultilinearRectilinearRecursive::<'_, T, 4>::new(grids, vals)?.interp(obs, out)
        }
        x if x == 5 => {
            MultilinearRectilinearRecursive::<'_, T, 5>::new(grids, vals)?.interp(obs, out)
        }
        x if x == 6 => {
            MultilinearRectilinearRecursive::<'_, T, 6>::new(grids, vals)?.interp(obs, out)
        }
        x if x == 7 => {
            MultilinearRectilinearRecursive::<'_, T, 7>::new(grids, vals)?.interp(obs, out)
        }
        x if x == 8 => {
            MultilinearRectilinearRecursive::<'_, T, 8>::new(grids, vals)?.interp(obs, out)
        }
        _ => Err(
            "Dimension exceeds maximum (8). Use interpolator struct directly for higher dimensions.",
        ),
    }?;

    Ok(())
}

/// Evaluate interpolant, allocating a new Vec for the output.
///
/// For best results, use the `interpn` function with preallocated output;
/// allocation has a significant performance cost, and should be used sparingly.
#[cfg(feature = "std")]
pub fn interpn_alloc<T: Float>(
    grids: &[&[T]],
    vals: &[T],
    obs: &[&[T]],
) -> Result<Vec<T>, &'static str> {
    let mut out = vec![T::zero(); obs[0].len()];
    interpn(grids, vals, obs, &mut out)?;
    Ok(out)
}

/// Check whether a list of observation points are inside the grid within some absolute tolerance.
/// Assumes the grid is valid for the rectilinear interpolator (monotonically increasing).
///
/// Output slice entry `i` is set to `false` if no points on that dimension are out of bounds,
/// and set to `true` if there is a bounds violation on that axis.
///
/// # Errors
/// * If the dimensionality of the grid does not match the dimensionality of the observation points
/// * If the output slice length does not match the dimensionality of the grid
pub fn check_bounds<T: Float>(
    grids: &[&[T]],
    obs: &[&[T]],
    atol: T,
    out: &mut [bool],
) -> Result<(), &'static str> {
    let ndims = grids.len();
    if !(obs.len() == ndims && out.len() == ndims && (0..ndims).all(|i| !grids[i].is_empty())) {
        return Err("Dimension mismatch");
    }
    for i in 0..ndims {
        let lo = grids[i][0];
        let hi = grids[i].last();
        match hi {
            Some(&hi) => {
                let bad = obs[i]
                    .iter()
                    .any(|&x| (x - lo) <= -atol || (x - hi) >= atol);

                out[i] = bad;
            }
            None => return Err("Dimension mismatch"),
        }
    }
    Ok(())
}

/// An arbitrary-dimensional multilinear interpolator / extrapolator on a rectilinear grid.
///
/// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
/// Assumes grids are monotonically _increasing_. Checking this is expensive, and is
/// left to the user.
///
/// Operation Complexity
/// * O(2^ndims) for interpolation and extrapolation in all regions.
///
/// Memory Complexity
/// * Peak stack usage is O(MAXDIMS), which is minimally O(ndims).
/// * While evaluation is recursive, the recursion has constant
///   max depth of MAXDIMS, which provides a guarantee on peak
///   memory usage.
///
/// Timing
/// * Timing determinism is very tight, but not guaranteed due to the use of a bisection search.
pub struct MultilinearRectilinearRecursive<'a, T: Float, const MAXDIMS: usize> {
    /// x, y, ... coordinate grids, each entry of size dims[i]
    grids: &'a [&'a [T]],

    /// Size of each dimension
    dims: [usize; MAXDIMS],

    /// Values at each point, size prod(dims)
    vals: &'a [T],
}

impl<'a, T: Float, const MAXDIMS: usize> MultilinearRectilinearRecursive<'a, T, MAXDIMS> {
    /// Build a new interpolator, using O(MAXDIMS) calculations and storage.
    ///
    /// This method does not handle degenerate dimensions; all grids must have at least 4 entries.
    ///
    /// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
    ///
    /// # Errors
    /// * If any input dimensions do not match
    /// * If any dimensions have size < 4
    /// * If any step sizes have zero or negative magnitude
    pub fn new(grids: &'a [&'a [T]], vals: &'a [T]) -> Result<Self, &'static str> {
        // Check dimensions
        let ndims = grids.len();
        let mut dims = [1_usize; MAXDIMS];
        (0..ndims).for_each(|i| dims[i] = grids[i].len());
        let nvals: usize = dims[..ndims].iter().product();
        if !(vals.len() == nvals && ndims > 0 && ndims <= MAXDIMS) {
            return Err("Dimension mismatch");
        };
        // Check if any grids are degenerate
        let degenerate = dims[..ndims].iter().any(|&x| x < 2);
        if degenerate {
            return Err("All grids must have at least 2 entries");
        };
        // Check that at least the first two entries in each grid are monotonic
        let monotonic_maybe = grids.iter().all(|&g| g[1] > g[0]);
        if !monotonic_maybe {
            return Err("All grids must be monotonically increasing");
        };

        Ok(Self { grids, dims, vals })
    }

    /// Interpolate on a contiguous list of observation points.
    ///
    /// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
    ///
    /// # Errors
    ///   * If the dimensionality of the point does not match the data
    ///   * If the dimensionality of point or data does not match the grid
    pub fn interp(&self, x: &[&[T]], out: &mut [T]) -> Result<(), &'static str> {
        let n = out.len();
        let ndims = self.grids.len();

        // Make sure there are enough coordinate inputs for each dimension
        if x.len() != ndims {
            return Err("Dimension mismatch");
        }

        // Make sure the size of inputs and output match
        let size_matches = x.iter().all(|&xx| xx.len() == out.len());
        if !size_matches {
            return Err("Dimension mismatch");
        }

        let tmp = &mut [T::zero(); MAXDIMS][..ndims];
        for i in 0..n {
            (0..ndims).for_each(|j| tmp[j] = x[j][i]);
            out[i] = self.interp_one(tmp)?;
        }

        Ok(())
    }

    /// Interpolate the value at a point,
    /// using fixed-size intermediate storage of O(ndims) and no allocation.
    ///
    /// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
    ///
    /// # Errors
    ///   * If the dimensionality of the point does not match the data
    ///   * If the dimensionality of either one exceeds the fixed maximum
    ///   * If the index along any dimension exceeds the maximum representable
    ///     integer value within the value type `T`
    pub fn interp_one(&self, x: &[T]) -> Result<T, &'static str> {
        // Check sizes
        let ndims = self.grids.len();
        if !(x.len() == ndims && ndims <= MAXDIMS) {
            return Err("Dimension mismatch");
        }

        // Initialize fixed-size intermediate storage.
        // Maybe counterintuitively, initializing this storage here on every usage
        // instead of once with the top level struct is a significant speedup
        // and does not increase peak stack usage.
        //
        // Also notably, storing the index offsets as bool instead of usize
        // reduces memory overhead, but has not effect on throughput rate.
        let origin = &mut [0_usize; MAXDIMS][..ndims]; // Indices of lower corner of hypercub
        let dimprod = &mut [1_usize; MAXDIMS][..ndims];

        // Populate cumulative product of higher dimensions for indexing.
        //
        // Each entry is the cumulative product of the size of dimensions
        // higher than this one, which is the stride between blocks
        // relating to a given index along each dimension.
        let mut acc = 1;
        for i in 0..ndims {
            dimprod[ndims - i - 1] = acc;
            acc *= self.dims[ndims - i - 1];
        }

        // Populate lower corner and saturation flag for each dimension
        for i in 0..ndims {
            origin[i] = self.get_loc(x[i], i)?;
        }

        // Recursive interpolation of one dependency tree at a time
        // let loc = &origin; // Starting location in the tree is the origin
        let dim = ndims; // Start from the end and recurse back to zero
        let loc = &mut [0_usize; MAXDIMS][..ndims];
        loc.copy_from_slice(origin);
        let interped = self.populate(dim, origin, loc, dimprod, x);

        Ok(interped)
    }

    /// Get the lower-corner index along this dimension where `x` is found,
    /// saturating to the bounds at the edges if necessary.
    ///
    /// At the high bound of a given dimension, saturates to the interior.
    #[inline]
    fn get_loc(&self, v: T, dim: usize) -> Result<usize, &'static str> {
        let grid = self.grids[dim];

        // Bisection search to find location on the grid.
        //
        // The search will return `0` if the point is outside-low,
        // and will return `self.dims[dim]` if outside-high.
        //
        // This process accounts for essentially the entire difference in
        // performance between this method and the regular-grid method.
        let iloc: isize = grid.partition_point(|x| *x < v) as isize - 1;

        let n = self.dims[dim] as isize; // Number of grid points on this dimension
        let dimmax = n.saturating_sub(2).max(0); // maximum index for lower corner
        let loc: usize = iloc.max(0).min(dimmax) as usize; // unsigned integer loc clipped to interior

        Ok(loc)
    }

    /// Recursive evaluation of interpolant on each dimension
    #[inline]
    fn populate(
        &self,
        dim: usize,
        origin: &[usize],
        loc: &mut [usize],
        dimprod: &[usize],
        x: &[T],
    ) -> T {
        // Do the calc for this entry
        match dim {
            // If we have arrived at a leaf, index into data
            0 => index_arr(loc, dimprod, self.vals),

            // Otherwise, continue recursion
            _ => {
                // Keep track of where we are in the tree
                // so that we can index into the value array properly
                // when we reach the leaves
                let next_dim = dim - 1;

                // Populate next dim's values
                let mut vals = [T::zero(); 2];
                for i in 0..2 {
                    loc[next_dim] = origin[next_dim] + i;
                    vals[i] = self.populate(next_dim, origin, loc, dimprod, x);
                }
                loc[next_dim] = origin[next_dim]; // Reset for next usage

                // Interpolate on next dim's values to populate an entry in this dim
                let grid_cell = &self.grids[next_dim][origin[next_dim]..origin[next_dim] + 2];
                let step = grid_cell[1] - grid_cell[0];
                let t = (x[next_dim] - grid_cell[0]) / step;
                let dy = vals[1] - vals[0];
                vals[0] + t * dy
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::{MultilinearRectilinearRecursive, interpn};
    use crate::testing::*;
    use crate::utils::*;

    /// Test with one dimension that is minimum size and one that is not
    #[test]
    fn test_interp_extrap_2d_small() {
        let (nx, ny) = (3, 2);
        let x = linspace(-1.0, 1.0, nx);
        let y = Vec::from([0.5, 0.6]);
        let grids = [&x[..], &y[..]];
        let xy = meshgrid(Vec::from([&x, &y]));

        // z = x + y
        let z: Vec<f64> = (0..nx * ny).map(|i| &xy[i][0] + &xy[i][1]).collect();

        // Observation points all over in 2D space
        let xobs = linspace(-10.0_f64, 10.0, 5);
        let yobs = linspace(-10.0_f64, 10.0, 5);
        let xyobs = meshgrid(Vec::from([&xobs, &yobs]));
        let zobs: Vec<f64> = (0..xobs.len() * yobs.len())
            .map(|i| &xyobs[i][0] + &xyobs[i][1])
            .collect(); // Every `z` should match the degenerate `y` value

        let interpolator: MultilinearRectilinearRecursive<'_, _, 2> =
            MultilinearRectilinearRecursive::new(&grids, &z[..]).unwrap();

        // Check values at every incident vertex
        xyobs.iter().zip(zobs.iter()).for_each(|(xyi, zi)| {
            let zii = interpolator.interp_one(&[xyi[0], xyi[1]]).unwrap();
            assert!((*zi - zii).abs() < 1e-12)
        });
    }

    /// Iterate from 1 to 8 dimensions, making a minimum-sized grid for each one
    /// to traverse every combination of interpolating or extrapolating high or low on each dimension.
    /// Each test evaluates at 3^ndims locations, largely extrapolated in corner regions, so it
    /// rapidly becomes prohibitively slow after about ndims=9.
    #[test]
    fn test_interp_extrap_1d_to_8d() {
        let mut rng = rng_fixed_seed();

        for ndims in 1..=8 {
            println!("Testing in {ndims} dims");
            // Interp grid
            let dims: Vec<usize> = vec![2; ndims];
            let xs: Vec<Vec<f64>> = (0..ndims)
                .map(|i| {
                    // Make a linear grid and add noise
                    let mut x = linspace(-5.0 * (i as f64), 5.0 * ((i + 1) as f64), dims[i]);
                    let dx = randn::<f64>(&mut rng, x.len());
                    (0..x.len()).for_each(|i| x[i] += (dx[i] - 0.5) / 10.0);
                    (0..x.len() - 1).for_each(|i| assert!(x[i + 1] > x[i]));
                    x
                })
                .collect();

            let grids: Vec<&[f64]> = xs.iter().map(|x| &x[..]).collect();
            let grid = meshgrid((0..ndims).map(|i| &xs[i]).collect());
            let u: Vec<f64> = grid.iter().map(|x| x.iter().sum()).collect(); // sum is linear in every direction, good for testing

            // Observation points
            let xobs: Vec<Vec<f64>> = (0..ndims)
                .map(|i| linspace(-7.0 * (i as f64), 7.0 * ((i + 1) as f64), 3))
                .collect();
            let gridobs = meshgrid((0..ndims).map(|i| &xobs[i]).collect());
            let gridobs_t: Vec<Vec<f64>> = (0..ndims)
                .map(|i| gridobs.iter().map(|x| x[i]).collect())
                .collect(); // transpose
            let xobsslice: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..]).collect();
            let uobs: Vec<f64> = gridobs.iter().map(|x| x.iter().sum()).collect(); // expected output at observation points
            let mut out = vec![0.0; uobs.len()];

            // Evaluate
            interpn(&grids, &u, &xobsslice, &mut out[..]).unwrap();

            // Check that interpolated values match expectation,
            // using an absolute difference because some points are very close to or exactly at zero,
            // and do not do well under a check on relative difference.
            (0..uobs.len()).for_each(|i| assert!((out[i] - uobs[i]).abs() < 1e-12));
        }
    }

    /// Interpolate on a hat-shaped function to make sure that the grid cell indexing is aligned properly
    #[test]
    fn test_interp_hat_func() {
        fn hat_func(x: f64) -> f64 {
            if x <= 1.0 { x } else { 2.0 - x }
        }

        let x = (0..3).map(|x| x as f64).collect::<Vec<f64>>();
        let grids = [&x[..]];
        let y = (0..3).map(|x| hat_func(x as f64)).collect::<Vec<f64>>();
        let obs = linspace(-2.0, 4.0, 100);

        let interpolator: MultilinearRectilinearRecursive<f64, 1> =
            MultilinearRectilinearRecursive::new(&grids, &y).unwrap();

        (0..obs.len()).for_each(|i| {
            assert_eq!(
                hat_func(obs[i]),
                interpolator.interp_one(&[obs[i]]).unwrap()
            );
        })
    }
}
