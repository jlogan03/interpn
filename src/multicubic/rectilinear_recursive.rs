//! An arbitrary-dimensional multicubic interpolator / extrapolator on a rectilinear grid.
//!
//! ```rust
//! use interpn::multicubic::rectilinear;
//!
//! // Define a grid
//! let x = [1.0_f64, 2.0, 3.0, 4.0];
//! let y = [0.0_f64, 1.0, 2.0, 3.0];
//!
//! // Grid input for rectilinear method
//! let grids = &[&x[..], &y[..]];
//!
//! // Values at grid points
//! let z = [2.0; 16];
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
//! let linearize_extrapolation = false;
//! rectilinear::interpn_alloc(grids, &z, linearize_extrapolation, &obs).unwrap();
//! ```
//!
//! References
//! * A. E. P. Veldman and K. Rinzema, “Playing with nonuniform grids”.
//!   https://pure.rug.nl/ws/portalfiles/portal/3332271/1992JEngMathVeldman.pdf
use super::{Saturation, centered_difference_nonuniform, normalized_hermite_spline};
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
    linearize_extrapolation: bool,
    obs: &[&[T]],
    out: &mut [T],
) -> Result<(), &'static str> {
    // Expanding out and using the specialized version for each size
    // gives a substantial speedup for lower dimensionalities
    // (4-5x speedup for 1-dim compared to using MAXDIMS=8)
    let ndims = grids.len();
    match ndims {
        1 => MulticubicRectilinearRecursive::<'_, T, 1>::new(grids, vals, linearize_extrapolation)?
            .interp(obs, out),
        2 => MulticubicRectilinearRecursive::<'_, T, 2>::new(grids, vals, linearize_extrapolation)?
            .interp(obs, out),
        3 => MulticubicRectilinearRecursive::<'_, T, 3>::new(grids, vals, linearize_extrapolation)?
            .interp(obs, out),
        4 => MulticubicRectilinearRecursive::<'_, T, 4>::new(grids, vals, linearize_extrapolation)?
            .interp(obs, out),
        5 => MulticubicRectilinearRecursive::<'_, T, 5>::new(grids, vals, linearize_extrapolation)?
            .interp(obs, out),
        6 => MulticubicRectilinearRecursive::<'_, T, 6>::new(grids, vals, linearize_extrapolation)?
            .interp(obs, out),
        7 => MulticubicRectilinearRecursive::<'_, T, 7>::new(grids, vals, linearize_extrapolation)?
            .interp(obs, out),
        8 => MulticubicRectilinearRecursive::<'_, T, 8>::new(grids, vals, linearize_extrapolation)?
            .interp(obs, out),
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
    linearize_extrapolation: bool,
    obs: &[&[T]],
) -> Result<Vec<T>, &'static str> {
    let mut out = vec![T::zero(); obs[0].len()];
    interpn(grids, vals, linearize_extrapolation, obs, &mut out)?;
    Ok(out)
}

// We can use the same rectilinear-grid method again
pub use crate::multilinear::rectilinear::check_bounds;

/// An arbitrary-dimensional multicubic interpolator / extrapolator on a regular grid.
///
/// On interior points, a hermite spline is used, with the derivative at each
/// grid point matched to a second-order central difference. This allows the
/// interpolant to reproduce a quadratic function exactly, and to approximate
/// others with minimal overshoot and wobble.
///
/// At the grid boundary, a natural spline boundary condition is applied,
/// meaning the third derivative of the interpolant is constrainted to zero
/// at the last grid point, with the result that the interpolant is quadratic
/// on the last interval before the boundary.
///
/// With "linearize_extrapolation" set, extrapolation is linear on the extrapolated
/// dimensions, holding the same derivative as the natural boundary condition produces
/// at the last grid point. Otherwise, the last grid cell's spline function is continued,
/// producing a quadratic extrapolation.
///
/// This effectively gives a gradual decrease in the order of the interpolant
/// as the observation point approaches then leaves the grid:
///
/// out                     out
/// ---|---|---|---|---|---|--- Grid
///  2   2   3   3   3   2   2  Order of interpolant between grid points
///  1                       1  Extrapolation with linearize_extrapolation
///
/// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
///
/// Operation Complexity
/// * O(4^ndims) for interpolation and extrapolation in all regions.
///
/// Memory Complexity
/// * Peak stack usage is O(MAXDIMS), which is minimally O(ndims).
/// * While evaluation is recursive, the recursion has constant
///   max depth of MAXDIMS, which provides a guarantee on peak
///   memory usage.
///
/// Timing
/// * Timing determinism very tight, but is not exact due to the
///   differences in calculations (but not complexity) between
///   interpolation and extrapolation.
/// * An interpolation-only variant of this algorithm could achieve
///   near-deterministic timing, but would produce incorrect results
///   when evaluated at off-grid points.
pub struct MulticubicRectilinearRecursive<'a, T: Float, const MAXDIMS: usize> {
    /// x, y, ... coordinate grids, each entry of size dims[i]
    grids: &'a [&'a [T]],

    /// Size of each dimension
    dims: [usize; MAXDIMS],

    /// Values at each point, size prod(dims)
    vals: &'a [T],

    /// Whether to extrapolate linearly instead of continuing spline
    linearize_extrapolation: bool,
}

impl<'a, T: Float, const MAXDIMS: usize> MulticubicRectilinearRecursive<'a, T, MAXDIMS> {
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
    pub fn new(
        grids: &'a [&'a [T]],
        vals: &'a [T],
        linearize_extrapolation: bool,
    ) -> Result<Self, &'static str> {
        // Check dimensions
        let ndims = grids.len();
        let mut dims = [1_usize; MAXDIMS];
        (0..ndims).for_each(|i| dims[i] = grids[i].len());
        let nvals: usize = dims[..ndims].iter().product();
        if !(vals.len() == nvals && ndims > 0 && ndims <= MAXDIMS) {
            return Err("Dimension mismatch");
        };
        // Check if any grids are degenerate
        let degenerate = dims[..ndims].iter().any(|&x| x < 4);
        if degenerate {
            return Err("All grids must have at least 4 entries");
        };
        // Check that at least the first two entries in each grid are monotonic
        let monotonic_maybe = grids.iter().all(|&g| g[1] > g[0]);
        if !monotonic_maybe {
            return Err("All grids must be monotonically increasing");
        };

        Ok(Self {
            grids,
            dims,
            vals,
            linearize_extrapolation,
        })
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
        let sat = &mut [Saturation::None; MAXDIMS][..ndims]; // Saturation none/high/low flags for each dim
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
            (origin[i], sat[i]) = self.get_loc(x[i], i)?;
        }

        // Recursive interpolation of one dependency tree at a time
        // let loc = &origin; // Starting location in the tree is the origin
        let dim = ndims; // Start from the end and recurse back to zero
        let loc = &mut [0_usize; MAXDIMS][..ndims];
        loc.copy_from_slice(origin);
        let interped = self.populate(dim, sat, origin, loc, dimprod, x);

        Ok(interped)
    }

    /// Get the two-lower index along this dimension where `x` is found,
    /// saturating to the bounds at the edges if necessary.
    ///
    /// At the high bound of a given dimension, saturates to the fourth internal
    /// point in order to capture a full 4-cube.
    ///
    /// Returned value like (lower_corner_index, saturation_flag).
    #[inline]
    fn get_loc(&self, v: T, dim: usize) -> Result<(usize, Saturation), &'static str> {
        let saturation: Saturation; // What part of the grid cell are we in?
        let grid = self.grids[dim];

        // Bisection search to find location on the grid.
        //
        // The search will return `0` if the point is outside-low,
        // and will return `self.dims[dim]` if outside-high.
        //
        // This process accounts for essentially the entire difference in
        // performance between this method and the regular-grid method.
        let iloc: isize = grid.partition_point(|x| *x < v) as isize - 2;

        let n = self.dims[dim] as isize; // Number of grid points on this dimension
        let dimmax = n.saturating_sub(4).max(0); // maximum index for lower corner
        let loc: usize = iloc.max(0).min(dimmax) as usize; // unsigned integer loc clipped to interior

        // Observation point is outside the grid on the low side
        if iloc == -2 {
            saturation = Saturation::OutsideLow;
        }
        // Observation point is in the lower part of the cell
        // but not outside the grid
        else if iloc == -1 {
            saturation = Saturation::InsideLow;
        }
        // Observation point is in the upper part of the cell
        // but not outside the grid
        else if iloc == n - 2 {
            saturation = Saturation::OutsideHigh;
        }
        // Observation point is in the upper part of the cell
        // but not outside the grid
        else if iloc == n - 3 {
            saturation = Saturation::InsideHigh;
        }
        // Observation point is on the interior
        else {
            saturation = Saturation::None;
        }

        Ok((loc, saturation))
    }

    /// Recursive evaluation of interpolant on each dimension
    #[inline]
    fn populate(
        &self,
        dim: usize,
        sat: &[Saturation],
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
                let mut vals = [T::zero(); 4];
                for i in 0..4 {
                    loc[next_dim] = origin[next_dim] + i;
                    vals[i] = self.populate(next_dim, sat, origin, loc, dimprod, x);
                }
                loc[next_dim] = origin[next_dim]; // Reset for next usage

                // Interpolate on next dim's values to populate an entry in this dim
                let grid_cell = &self.grids[next_dim][origin[next_dim]..origin[next_dim] + 4];
                interp_inner::<T>(
                    vals,
                    grid_cell,
                    x[next_dim],
                    sat[next_dim],
                    self.linearize_extrapolation,
                )
            }
        }
    }
}

/// Calculate slopes and offsets & select evaluation method
#[inline]
fn interp_inner<T: Float>(
    vals: [T; 4],
    grid_cell: &[T],
    x: T,
    sat: Saturation,
    linearize_extrapolation: bool,
) -> T {
    // Construct some constants using generic methods
    let one = T::one();
    let two = one + one;

    // For cases on the interior, use two slopes (from centered difference) and two values
    // as the BCs.
    //
    // For locations falling near and edge, take one centered
    // difference for the inside derivative,
    // then for the derivative at the edge, impose a natural
    // spline constraint, meaning the third derivative q'''(t) = 0
    // at the last grid point, which produces a quadratic in the
    // last cell, reducing wobble that would be cause by enforcing
    // the use of a cubic function where there is not enough information
    // to support it.

    match sat {
        Saturation::None => {
            //       |-> t
            // --|---|---|---|--
            //         x
            //
            // This is the nominal case
            let y0 = vals[1];
            let dy = vals[2] - vals[1];

            let h01 = grid_cell[1] - grid_cell[0];
            let h12 = grid_cell[2] - grid_cell[1];
            let h23 = grid_cell[3] - grid_cell[2];
            let k0 = centered_difference_nonuniform(vals[0], vals[1], vals[2], h01 / h12, T::one());
            let k1 = centered_difference_nonuniform(vals[1], vals[2], vals[3], T::one(), h23 / h12);

            let t = (x - grid_cell[1]) / h12;

            normalized_hermite_spline(t, y0, dy, k0, k1)
        }
        Saturation::InsideLow => {
            //   t <-|
            // --|---|---|---|--
            //     x
            //
            // Flip direction to maintain symmetry
            // with the InsideHigh case.

            let y0 = vals[1]; // Same starting point, opposite direction
            let dy = vals[0] - vals[1];

            let h01 = grid_cell[1] - grid_cell[0];
            let h12 = grid_cell[2] - grid_cell[1];
            let k0 =
                -centered_difference_nonuniform(vals[0], vals[1], vals[2], T::one(), h12 / h01);
            let k1 = two * dy - k0; // Natural spline boundary condition

            let t = -(x - grid_cell[1]) / h01;

            normalized_hermite_spline(t, y0, dy, k0, k1)
        }
        Saturation::OutsideLow => {
            //   t <-|
            // --|---|---|---|--
            // x
            //
            // Flip direction to maintain symmetry
            // with the InsideHigh case.

            let y0 = vals[1];
            let y1 = vals[0];
            let dy = vals[0] - vals[1];

            let h01 = grid_cell[1] - grid_cell[0];
            let h12 = grid_cell[2] - grid_cell[1];
            let k0 =
                -centered_difference_nonuniform(vals[0], vals[1], vals[2], T::one(), h12 / h01);
            let k1 = two * dy - k0; // Natural spline boundary condition

            let t = -(x - grid_cell[1]) / h01;

            // If we are linearizing the interpolant under extrapolation,
            // hold the last slope outside the grid
            if linearize_extrapolation {
                y1 + k1 * (t - one)
            } else {
                normalized_hermite_spline(t, y0, dy, k0, k1)
            }
        }
        Saturation::InsideHigh => {
            //           |-> t
            // --|---|---|---|--
            //             x
            let y0 = vals[2];
            let dy = vals[3] - vals[2];

            let h12 = grid_cell[2] - grid_cell[1];
            let h23 = grid_cell[3] - grid_cell[2];
            let k0 = centered_difference_nonuniform(vals[1], vals[2], vals[3], h12 / h23, T::one());
            let k1 = two * dy - k0; // Natural spline boundary condition

            let t = (x - grid_cell[2]) / h23;

            normalized_hermite_spline(t, y0, dy, k0, k1)
        }
        Saturation::OutsideHigh => {
            //           |-> t
            // --|---|---|---|--
            //                 x
            let y0 = vals[2];
            let y1 = vals[3];
            let dy = vals[3] - vals[2];

            let h12 = grid_cell[2] - grid_cell[1];
            let h23 = grid_cell[3] - grid_cell[2];
            let k0 = centered_difference_nonuniform(vals[1], vals[2], vals[3], h12 / h23, T::one());
            let k1 = two * dy - k0; // Natural spline boundary condition

            let t = (x - grid_cell[2]) / h23;

            // If we are linearizing the interpolant under extrapolation,
            // hold the last slope outside the grid
            if linearize_extrapolation {
                y1 + k1 * (t - one)
            } else {
                normalized_hermite_spline(t, y0, dy, k0, k1)
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::interpn;
    use crate::testing::*;
    use crate::utils::*;

    /// Iterate from 1 to 8 dimensions, making a minimum-sized grid for each one
    /// to traverse every combination of interpolating or extrapolating high or low on each dimension.
    /// Each test evaluates at 5^ndims locations, largely extrapolated in corner regions, so it
    /// rapidly becomes prohibitively slow in higher dimensions.
    #[test]
    fn test_interp_extrap_1d_to_6d_linear() {
        let mut rng = rng_fixed_seed();

        for ndims in 1..=6 {
            println!("Testing in {ndims} dims");
            // Interp grid
            let dims: Vec<usize> = vec![4; ndims];
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
                .map(|i| linspace(-7.0 * (i as f64), 7.0 * ((i + 1) as f64), dims[i] + 2))
                .collect();
            let gridobs = meshgrid((0..ndims).map(|i| &xobs[i]).collect());
            let gridobs_t: Vec<Vec<f64>> = (0..ndims)
                .map(|i| gridobs.iter().map(|x| x[i]).collect())
                .collect(); // transpose
            let xobsslice: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..]).collect();
            let uobs: Vec<f64> = gridobs.iter().map(|x| x.iter().sum()).collect(); // expected output at observation points
            let mut out = vec![0.0; uobs.len()];

            // Evaluate with linearized extrapolation
            interpn(&grids, &u, true, &xobsslice, &mut out[..]).unwrap();

            // Check that interpolated values match expectation,
            // using an absolute difference because some points are very close to or exactly at zero,
            // and do not do well under a check on relative difference.
            (0..uobs.len()).for_each(|i| assert!((out[i] - uobs[i]).abs() < 1e-10));

            // Evaluate and check without linearized extrapolation
            interpn(&grids, &u, false, &xobsslice, &mut out[..]).unwrap();
            (0..uobs.len()).for_each(|i| assert!((out[i] - uobs[i]).abs() < 1e-10));
        }
    }

    /// Under both interpolation and extrapolation, a hermite spline with natural boundary condition
    /// can reproduce an N-dimensional quadratic function exactly
    #[test]
    fn test_interp_extrap_1d_to_6d_quadratic() {
        let mut rng = rng_fixed_seed();

        for ndims in 1..6 {
            println!("Testing in {ndims} dims");
            // Interp grid
            let dims: Vec<usize> = vec![4; ndims];
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
            let u: Vec<f64> = (0..grid.len())
                .map(|i| {
                    let mut v = 0.0;
                    for j in 0..ndims {
                        v += grid[i][j] * grid[i][j];
                    }
                    v
                })
                .collect(); // Quadratic in every directio

            // Observation points
            let xobs: Vec<Vec<f64>> = (0..ndims)
                .map(|i| linspace(-7.0 * (i as f64), 7.0 * ((i + 1) as f64), dims[i] + 2))
                .collect();
            let gridobs = meshgrid((0..ndims).map(|i| &xobs[i]).collect());
            let gridobs_t: Vec<Vec<f64>> = (0..ndims)
                .map(|i| gridobs.iter().map(|x| x[i]).collect())
                .collect(); // transpose
            let xobsslice: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..]).collect();
            let uobs: Vec<f64> = (0..gridobs.len())
                .map(|i| {
                    let mut v = 0.0;
                    for j in 0..ndims {
                        v += gridobs[i][j] * gridobs[i][j];
                    }
                    v
                })
                .collect(); // Quadratic in every direction
            let mut out = vec![0.0; uobs.len()];

            // Evaluate
            interpn(&grids, &u, false, &xobsslice, &mut out[..]).unwrap();

            // Check that interpolated and extrapolated values match expectation,
            // using an absolute difference because some points are very close to or exactly at zero,
            // and do not do well under a check on relative difference.
            (0..uobs.len()).for_each(|i| assert!((out[i] - uobs[i]).abs() < 1e-10));
        }
    }

    /// Under interpolation, a hermite spline with natural boundary condition
    /// can reproduce an N-dimensional sine function fairly closely, but not exactly.
    /// More points are required to capture a sine function, so fewer dimensions are tested
    /// to keep test run times low.
    #[test]
    fn test_interp_1d_to_3d_sine() {
        let mut rng = rng_fixed_seed();

        for ndims in 1..3 {
            println!("Testing in {ndims} dims");
            // Interp grid
            let dims: Vec<usize> = vec![10; ndims];
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
            let u: Vec<f64> = (0..grid.len())
                .map(|i| {
                    let mut v = 0.0;
                    for j in 0..ndims {
                        v += (grid[i][j] * 6.28 / 10.0).sin();
                    }
                    v
                })
                .collect(); // Quadratic in every direction

            // Observation points
            let xobs: Vec<Vec<f64>> = (0..ndims)
                .map(|i| linspace(-5.0 * (i as f64), 5.0 * ((i + 1) as f64), dims[i] + 1))
                .collect();
            let gridobs = meshgrid((0..ndims).map(|i| &xobs[i]).collect());
            let gridobs_t: Vec<Vec<f64>> = (0..ndims)
                .map(|i| gridobs.iter().map(|x| x[i]).collect())
                .collect(); // transpose
            let xobsslice: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..]).collect();
            let uobs: Vec<f64> = (0..gridobs.len())
                .map(|i| {
                    let mut v = 0.0;
                    for j in 0..ndims {
                        v += (gridobs[i][j] * 6.28 / 10.0).sin();
                    }
                    v
                })
                .collect(); // Quadratic in every direction
            let mut out = vec![0.0; uobs.len()];

            // Evaluate
            interpn(&grids, &u, false, &xobsslice, &mut out[..]).unwrap();

            // Use a tolerance that increases with the number of dimensions, since
            // we are effectively summing ndims times the error from each dimension
            let tol = 2e-2 * f64::from(ndims as u32);

            (0..uobs.len()).for_each(|i| {
                let err = out[i] - uobs[i];
                assert!(err.abs() < tol);
            });
        }
    }
}
