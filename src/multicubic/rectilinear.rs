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
use super::Saturation;
use crate::{
    index_arr_fixed_dims,
    interp_math::{dot4, hermite_basis},
    mul_add,
};
use crunchy::unroll;
use num_traits::Float;

/// Evaluate multicubic interpolation on a regular grid in up to 8 dimensions.
/// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
///
/// For 1-4 dimensions with `deep-unroll` enabled (1-3 by default), a fast flattened method is used.
/// For higher dimensions, where that flattening becomes impractical due to compile times and
/// instruction size, evaluation defers to a run-time loop.
///
/// This is a convenience function; best performance will be achieved by using the exact right
/// number for the N parameter, as this will slightly reduce compute and storage overhead,
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
    // Check dimensionality
    let ndims = grids.len();

    // Dispatch to specialized implementation
    crate::dispatch_ndims!(
        ndims,
        "Dimension exceeds maximum (8). Use interpolator struct directly for higher dimensions.",
        [1, 2, 3, 4, 5, 6, 7, 8],
        |N| {
            MulticubicRectilinear::<'_, T, N>::new(
                grids.try_into().unwrap(),
                vals,
                linearize_extrapolation,
            )?
            .interp(obs.try_into().unwrap(), out)
        }
    )
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
/// * Peak stack usage is O(4^ndims) for lower dimensions (unrolled), and O(N) otherwise.
///
/// Timing
/// * Timing determinism very tight, but is not exact due to the
///   differences in calculations (but not complexity) between
///   interpolation and extrapolation.
/// * An interpolation-only variant of this algorithm could achieve
///   near-deterministic timing, but would produce incorrect results
///   when evaluated at off-grid points.
pub struct MulticubicRectilinear<'a, T: Float, const N: usize> {
    /// x, y, ... coordinate grids, each entry of size dims[i]
    grids: &'a [&'a [T]],

    /// Size of each dimension
    dims: [usize; N],

    /// Values at each point, size prod(dims)
    vals: &'a [T],

    /// Whether to extrapolate linearly instead of continuing spline
    linearize_extrapolation: bool,
}

impl<'a, T: Float, const N: usize> MulticubicRectilinear<'a, T, N> {
    /// Build a new interpolator, using O(N) calculations and storage.
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
        grids: &'a [&'a [T]; N],
        vals: &'a [T],
        linearize_extrapolation: bool,
    ) -> Result<Self, &'static str> {
        // Check dimensions
        let mut dims = [1_usize; N];
        (0..N).for_each(|i| dims[i] = grids[i].len());
        let nvals: usize = dims.iter().product();
        if vals.len() != nvals {
            return Err("Dimension mismatch");
        };
        // Check if any grids are degenerate
        let degenerate = dims.iter().any(|&x| x < 4);
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
    pub fn interp(&self, x: &[&[T]; N], out: &mut [T]) -> Result<(), &'static str> {
        // Make sure the size of inputs and output match
        let n = out.len();
        for i in 0..N {
            if x[i].len() != n {
                return Err("Dimension mismatch");
            }
        }

        let mut tmp = [T::zero(); N];
        for i in 0..n {
            (0..N).for_each(|j| tmp[j] = x[j][i]);
            out[i] = self.interp_one(tmp)?;
        }

        Ok(())
    }

    /// Interpolate the value at a point,
    /// using fixed-size intermediate storage of O(N) and no allocation.
    ///
    /// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
    ///
    /// # Errors
    ///   * If the dimensionality of the point does not match the data
    ///   * If the dimensionality of either one exceeds the fixed maximum
    ///   * If the index along any dimension exceeds the maximum representable
    ///     integer value within the value type `T`
    pub fn interp_one(&self, x: [T; N]) -> Result<T, &'static str> {
        // Initialize fixed-size intermediate storage.
        // Maybe counterintuitively, initializing this storage here on every usage
        // instead of once with the top level struct is a significant speedup
        // and does not increase peak stack usage.
        let mut origin = [0_usize; N]; // Indices of lower corner of hypercub
        let mut sat = [Saturation::None; N]; // Saturation none/high/low flags for each dim
        let mut weights = [[T::zero(); FP]; N];
        let mut dimprod = [1_usize; N];
        let mut loc = [0_usize; N];
        let mut store = [[T::zero(); FP]; N];

        let mut acc = 1;
        for i in 0..N {
            // Populate cumulative product of higher dimensions for indexing.
            //
            // Each entry is the cumulative product of the size of dimensions
            // higher than this one, which is the stride between blocks
            // relating to a given index along each dimension.
            if i > 0 {
                acc *= self.dims[N - i];
            }
            dimprod[N - i - 1] = acc;

            // Populate lower corner and saturation flag for each dimension
            (origin[i], sat[i]) = self.get_loc(x[i], i)?;
            let grid_cell = &self.grids[i][origin[i]..origin[i] + 4];
            weights[i] = interp_weights(
                grid_cell.try_into().unwrap(),
                x[i],
                sat[i],
                self.linearize_extrapolation,
            );
        }

        // Recursive interpolation of one dependency tree at a time
        const FP: usize = 4; // Footprint size
        let nverts = const { FP.pow(N as u32) }; // Total number of vertices

        macro_rules! unroll_vertices_body {
            ($i:ident) => {
                // Index, interpolate, or pass on each level of the tree
                for j in 0..N {
                    // Most of these iterations will get optimized out
                    if j == 0 {
                        // const branch
                        // At leaves, index values
                        for k in 0..N {
                            // Bit pattern in an integer matches C-ordered array indexing
                            // so we can just use the vertex index to index into the array
                            // by selecting the appropriate bit from the index.
                            let offset: usize = ($i & (3 << (2 * k))) >> (2 * k);
                            loc[k] = origin[k] + offset;
                        }
                        let store_ind: usize = $i % FP;
                        store[0][store_ind] = index_arr_fixed_dims(loc, dimprod, self.vals);
                    } else {
                        // const branch
                        // For other nodes, interpolate on child values
                        let q: usize = FP.pow(j as u32);
                        let level: bool = ($i + 1).is_multiple_of(q);
                        let p: usize = (($i + 1) / q).saturating_sub(1) % FP;
                        let ind: usize = j.saturating_sub(1);

                        if level {
                            // const branch
                            store[j][p] = dot4(weights[ind], store[ind]);
                        }
                    }
                }
            };
        }

        #[cfg(not(feature = "deep-unroll"))]
        if N <= 3 {
            unroll! {
                for i < 64 in 0..nverts {  // const loop
                    unroll_vertices_body!(i);
                }
            }
        } else {
            for i in 0..nverts {
                unroll_vertices_body!(i);
            }
        }

        #[cfg(feature = "deep-unroll")]
        if N <= 4 {
            unroll! {
                for i < 256 in 0..nverts {  // const loop
                    unroll_vertices_body!(i);
                }
            }
        } else {
            for i in 0..nverts {
                unroll_vertices_body!(i);
            }
        }

        // Interpolate the final value
        Ok(dot4(weights[N - 1], store[N - 1]))
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
}

/// Calculate one-dimensional interpolation weights for a fixed grid cell.
///
/// For cases on the interior, use two slopes from a nonuniform-grid centered
/// difference and two values as the Hermite boundary conditions.
///
/// For locations near an edge, take one centered difference for the inside
/// derivative, then impose a natural spline boundary condition on the
/// derivative at the edge, meaning the third derivative q'''(t) = 0 at the
/// last grid point. This produces a quadratic in the last cell, reducing wobble
/// that would be caused by enforcing the use of a cubic function where there is
/// not enough information to support it.
///
/// The returned weights multiply the four local values directly. This keeps the
/// same interpolant as the direct Hermite evaluation, but lets the N-dimensional
/// reduction reuse the weights for every child group on the same axis.
#[inline]
fn interp_weights<T: Float>(
    grid_cell: &[T; 4],
    x: T,
    sat: Saturation,
    linearize_extrapolation: bool,
) -> [T; 4] {
    let one = T::one();

    match sat {
        Saturation::None => {
            //       |-> t
            // --|---|---|---|--
            //         x
            //
            // This is the nominal case.
            let h01 = grid_cell[1] - grid_cell[0];
            let h12 = grid_cell[2] - grid_cell[1];
            let h23 = grid_cell[3] - grid_cell[2];
            let t = (x - grid_cell[1]) / h12;
            let [h00, h10, h01_basis, h11] = hermite_basis(t);
            let k0 = centered_difference_weights(h01 / h12, one);
            let k1 = centered_difference_weights(one, h23 / h12);

            [
                h10 * k0[0],
                h00 + h10 * k0[1] + h11 * k1[0],
                h01_basis + h10 * k0[2] + h11 * k1[1],
                h11 * k1[2],
            ]
        }
        Saturation::InsideLow => {
            //   t <-|
            // --|---|---|---|--
            //     x
            //
            // Flip direction to maintain symmetry with the InsideHigh case.
            let h01 = grid_cell[1] - grid_cell[0];
            let h12 = grid_cell[2] - grid_cell[1];
            let t = -(x - grid_cell[1]) / h01;

            low_weights(t, h12 / h01, false)
        }
        Saturation::OutsideLow => {
            //   t <-|
            // --|---|---|---|--
            // x
            //
            // Flip direction to maintain symmetry with the OutsideHigh case.
            let h01 = grid_cell[1] - grid_cell[0];
            let h12 = grid_cell[2] - grid_cell[1];
            let t = -(x - grid_cell[1]) / h01;

            low_weights(t, h12 / h01, linearize_extrapolation)
        }
        Saturation::InsideHigh => {
            //           |-> t
            // --|---|---|---|--
            //             x
            let h12 = grid_cell[2] - grid_cell[1];
            let h23 = grid_cell[3] - grid_cell[2];
            let t = (x - grid_cell[2]) / h23;

            high_weights(t, h12 / h23, false)
        }
        Saturation::OutsideHigh => {
            //           |-> t
            // --|---|---|---|--
            //                 x
            let h12 = grid_cell[2] - grid_cell[1];
            let h23 = grid_cell[3] - grid_cell[2];
            let t = (x - grid_cell[2]) / h23;

            high_weights(t, h12 / h23, linearize_extrapolation)
        }
    }
}

#[inline]
fn low_weights<T: Float>(t: T, h12_over_h01: T, linearize_extrapolation: bool) -> [T; 4] {
    let one = T::one();
    let two = one + one;
    let k0 = centered_difference_weights(one, h12_over_h01);

    // If we are linearizing the interpolant under extrapolation, hold the last
    // slope outside the grid. Otherwise, continue the natural-boundary spline.
    if linearize_extrapolation {
        let s = t - one;
        [
            one + s * (two + k0[0]),
            s * (-two + k0[1]),
            s * k0[2],
            T::zero(),
        ]
    } else {
        let [h00, h10, h01, h11] = hermite_basis(t);
        let slope_factor = h11 - h10;
        [
            h01 + two * h11 + slope_factor * k0[0],
            h00 - two * h11 + slope_factor * k0[1],
            slope_factor * k0[2],
            T::zero(),
        ]
    }
}

#[inline]
fn high_weights<T: Float>(t: T, h12_over_h23: T, linearize_extrapolation: bool) -> [T; 4] {
    let one = T::one();
    let two = one + one;
    let k0 = centered_difference_weights(h12_over_h23, one);

    // If we are linearizing the interpolant under extrapolation, hold the last
    // slope outside the grid. Otherwise, continue the natural-boundary spline.
    if linearize_extrapolation {
        let s = t - one;
        [
            T::zero(),
            -s * k0[0],
            s * (-two - k0[1]),
            one + s * (two - k0[2]),
        ]
    } else {
        let [h00, h10, h01, h11] = hermite_basis(t);
        let slope_factor = h10 - h11;
        [
            T::zero(),
            slope_factor * k0[0],
            h00 - two * h11 + slope_factor * k0[1],
            h01 + two * h11 + slope_factor * k0[2],
        ]
    }
}

/// Second-order central difference weights on a nonuniform grid per
///
/// A. E. P. Veldman and K. Rinzema, "Playing with nonuniform grids".
/// https://pure.rug.nl/ws/portalfiles/portal/3332271/1992JEngMathVeldman.pdf
///
/// Method B, which is essentially a distance-weighted average of the forward
/// and backward differences s.t. the closer points have more influence on the
/// derivative estimate.
///
/// The returned weights multiply `[y0, y1, y2]` and produce the same result as:
///
/// ```text
/// a = h01 / (h01 + h12)
/// b = (y2 - y1) / h12
/// c = h12 / (h12 + h01)
/// d = (y1 - y0) / h01
/// derivative = a * b + c * d
/// ```
#[inline]
fn centered_difference_weights<T: Float>(h01: T, h12: T) -> [T; 3] {
    let denom = h01 + h12;
    let a = h01 / denom;
    let c = h12 / denom;

    [-c / h01, mul_add(c, T::one() / h01, -a / h12), a / h12]
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
    fn test_interp_extrap_1d_to_4d_linear() {
        let mut rng = rng_fixed_seed();

        for ndims in 1..=4 {
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
            (0..uobs.len()).for_each(|i| assert!((out[i] - uobs[i]).abs() < 3e-10));
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
