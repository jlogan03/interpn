//! An arbitrary-dimensional multicubic interpolator / extrapolator on a regular grid.
//!
//! ```rust
//! use interpn::multicubic::regular;
//!
//! // Define a grid
//! let x = [1.0_f64, 2.0, 3.0, 4.0];
//! let y = [0.0_f64, 1.0, 2.0, 3.0];
//!
//! // Grid input for rectilinear method
//! let grids = &[&x[..], &y[..]];
//!
//! // Grid input for regular grid method
//! let dims = [x.len(), y.len()];
//! let starts = [x[0], y[0]];
//! let steps = [x[1] - x[0], y[1] - y[0]];
//!
//! // Values at grid points
//! let z = [2.0; 16];
//!
//! // Observation points to interpolate/extrapolate
//! let xobs = [0.0_f64, 5.0];
//! let yobs = [-1.0, 3.0];
//! let obs = [&xobs[..], &yobs[..]];
//!
//! // Storage for output
//! let mut out = [0.0; 2];
//!
//! // Do interpolation, allocating for the output for convenience
//! let linearize_extrapolation = false;
//! regular::interpn_alloc(&dims, &starts, &steps, &z, linearize_extrapolation, &obs).unwrap();
//! ```
use super::{MulticubicRegularRecursive, Saturation, normalized_hermite_spline};
use crunchy::unroll;
use num_traits::{Float, NumCast};

/// Evaluate multicubic interpolation on a regular grid in up to 8 dimensions.
/// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
///
/// For 1-4 dimensions, a fast flattened method is used. For higher dimensions, where that flattening
/// becomes impractical due to compile times and instruction size, evaluation defers to a bounded
/// recursion.
///
/// This is a convenience function; best performance will be achieved by using the exact right
/// number for the N parameter, as this will slightly reduce compute and storage overhead,
/// and the underlying method can be extended to more than this function's limit of 8 dimensions.
/// The limit of 8 dimensions was chosen for no more specific reason than to reduce unit test times.
///
/// While this method initializes the interpolator struct on every call, the overhead of doing this
/// is minimal even when using it to evaluate one observation point at a time.
pub fn interpn<T: Float>(
    dims: &[usize],
    starts: &[T],
    steps: &[T],
    vals: &[T],
    linearize_extrapolation: bool,
    obs: &[&[T]],
    out: &mut [T],
) -> Result<(), &'static str> {
    // Expanding out and using the specialized version for each size
    // gives a substantial speedup for lower dimensionalities
    // (4-5x speedup for 1-dim compared to using N=8)
    let ndims = dims.len();
    match ndims {
        1 => MulticubicRegular::<'_, T, 1>::new(
            dims.try_into().unwrap(),
            starts.try_into().unwrap(),
            steps.try_into().unwrap(),
            vals,
            linearize_extrapolation,
        )?
        .interp(obs.try_into().unwrap(), out),
        2 => MulticubicRegular::<'_, T, 2>::new(
            dims.try_into().unwrap(),
            starts.try_into().unwrap(),
            steps.try_into().unwrap(),
            vals,
            linearize_extrapolation,
        )?
        .interp(obs.try_into().unwrap(), out),
        3 => MulticubicRegular::<'_, T, 3>::new(
            dims.try_into().unwrap(),
            starts.try_into().unwrap(),
            steps.try_into().unwrap(),
            vals,
            linearize_extrapolation,
        )?
        .interp(obs.try_into().unwrap(), out),
        4 => MulticubicRegular::<'_, T, 4>::new(
            dims.try_into().unwrap(),
            starts.try_into().unwrap(),
            steps.try_into().unwrap(),
            vals,
            linearize_extrapolation,
        )?
        .interp(obs.try_into().unwrap(), out),
        5 => MulticubicRegularRecursive::<'_, T, 5>::new(
            dims,
            starts,
            steps,
            vals,
            linearize_extrapolation,
        )?
        .interp(obs, out),
        6 => MulticubicRegularRecursive::<'_, T, 6>::new(
            dims,
            starts,
            steps,
            vals,
            linearize_extrapolation,
        )?
        .interp(obs, out),
        7 => MulticubicRegularRecursive::<'_, T, 7>::new(
            dims,
            starts,
            steps,
            vals,
            linearize_extrapolation,
        )?
        .interp(obs, out),
        8 => MulticubicRegularRecursive::<'_, T, 8>::new(
            dims,
            starts,
            steps,
            vals,
            linearize_extrapolation,
        )?
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
    dims: &[usize],
    starts: &[T],
    steps: &[T],
    vals: &[T],
    linearize_extrapolation: bool,
    obs: &[&[T]],
) -> Result<Vec<T>, &'static str> {
    let mut out = vec![T::zero(); obs[0].len()];
    interpn(
        dims,
        starts,
        steps,
        vals,
        linearize_extrapolation,
        obs,
        &mut out,
    )?;
    Ok(out)
}

// We can use the same regular-grid method again
pub use crate::multilinear::regular::check_bounds;

/// An N-dimensional multicubic interpolator / extrapolator on a regular grid.
/// Available for up to 6D; for higher dimensions, use
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
/// * Peak stack usage is O(N), which is minimally O(ndims).
/// * While evaluation is recursive, the recursion has constant
///   max depth of N, which provides a guarantee on peak
///   memory usage.
///
/// Timing
/// * Timing determinism very tight, but is not exact due to the
///   differences in calculations (but not complexity) between
///   interpolation and extrapolation.
/// * An interpolation-only variant of this algorithm could achieve
///   near-deterministic timing, but would produce incorrect results
///   when evaluated at off-grid points.
pub struct MulticubicRegular<'a, T: Float, const N: usize> {
    /// Size of each dimension
    dims: [usize; N],

    /// Starting point of each dimension, size dims.len()
    starts: [T; N],

    /// Step size for each dimension, size dims.len()
    steps: [T; N],

    /// Values at each point, size prod(dims)
    vals: &'a [T],

    /// Whether to extrapolate linearly instead of continuing spline
    linearize_extrapolation: bool,
}

impl<'a, T: Float, const N: usize> MulticubicRegular<'a, T, N> {
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
        dims: [usize; N],
        starts: [T; N],
        steps: [T; N],
        vals: &'a [T],
        linearize_extrapolation: bool,
    ) -> Result<Self, &'static str> {
        // Check dimensions
        const {
            assert!(
                N > 0 && N < 5,
                "Flattened method defined for 1-5 dimensions. For higher dimensions, use recursive method."
            );
        }
        let nvals: usize = dims.iter().product();
        if !(starts.len() == N && steps.len() == N && vals.len() == nvals && N > 0) {
            return Err("Dimension mismatch");
        }

        // Make sure all dimensions have at least four entries
        let degenerate = dims.iter().any(|&x| x < 4);
        if degenerate {
            return Err("All grids must have at least four entries");
        }
        // Check if any dimensions have zero or negative step size
        let steps_are_positive = steps.iter().all(|&x| x > T::zero());
        if !steps_are_positive {
            return Err("All grids must be monotonically increasing");
        }

        // Copy grid info into struct.
        // Keeping this data local to the struct eliminates an issue where,
        // if the grid info is defined somewhere not local to where the struct
        // is defined and used, the & references to it cost more than the entire
        // rest of the interpolation operation.
        let mut steps_local = [T::zero(); N];
        let mut starts_local = [T::zero(); N];
        let mut dims_local = [0_usize; N];
        steps_local[..N].copy_from_slice(&steps[..N]);
        starts_local[..N].copy_from_slice(&starts[..N]);
        dims_local[..N].copy_from_slice(&dims[..N]);

        Ok(Self {
            dims: dims_local,
            starts: starts_local,
            steps: steps_local,
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
        let n = out.len();
        // Make sure there are enough coordinate inputs for each dimension
        if x.len() != N {
            return Err("Dimension mismatch");
        }
        // Make sure the size of inputs and output match
        let size_matches = x.iter().all(|&xx| xx.len() == out.len());
        if !size_matches {
            return Err("Dimension mismatch");
        }

        let mut tmp = [T::zero(); N];
        for i in 0..n {
            (0..N).for_each(|j| tmp[j] = x[j][i]);
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
    pub fn interp_one(&self, x: [T; N]) -> Result<T, &'static str> {
        // Check sizes
        if x.len() != N {
            return Err("Dimension mismatch");
        }

        // Initialize fixed-size intermediate storage.
        // Maybe counterintuitively, initializing this storage here on every usage
        // instead of once with the top level struct is a significant speedup
        // and does not increase peak stack usage.
        //
        // Also notably, storing the index offsets as bool instead of usize
        // reduces memory overhead, but has not effect on throughput rate.
        let mut origin = [0_usize; N]; // Indices of lower corner of hypercube
        let mut sat = [Saturation::None; N]; // Saturation none/high/low flags for each dim
        let mut dts = [T::zero(); N]; // Normalized coordinate storage
        let mut dimprod = [1_usize; N];
        let mut loc = [0_usize; N];
        let mut store = [[T::zero(); FP]; N];

        let mut acc = 1;
        unroll! {
            for i < 5 in 0..N {
                // Populate cumulative product of higher dimensions for indexing.
                //
                // Each entry is the cumulative product of the size of dimensions
                // higher than this one, which is the stride between blocks
                // relating to a given index along each dimension.
                if const { i > 0 } {
                    acc *= self.dims[N - i];
                }
                dimprod[N - i - 1] = acc;

                // Populate lower corner and saturation flag for each dimension
                (origin[i], sat[i]) = self.get_loc(x[i], i)?;

                // Calculate normalized delta locations
                // For the cubic method, the normalized coordinate `t` is always relative
                // to cube index 1 (out of 0-3)
                let index_one_loc = self.starts[i]
                    + self.steps[i]
                        * <T as NumCast>::from(origin[i] + 1)
                            .ok_or("Unrepresentable coordinate value")?;
                dts[i] = (x[i] - index_one_loc) / self.steps[i];
            }
        }

        // Recursive interpolation of one dependency tree at a time
        const FP: usize = 4; // Footprint size      
        let nverts = const { FP.pow(N as u32) }; // Total number of vertices

        unroll! {
            for i < 256 in 0..nverts {  // const loop
                // Index, interpolate, or pass on each level of the tree
                unroll!{
                    for j < 5 in 0..N {  // const loop

                        // Most of these iterations will get optimized out
                        if const{j == 0} { // const branch
                            // At leaves, index values

                            unroll!{
                                for k < 5 in 0..N {  // const loop
                                    // Bit pattern in an integer matches C-ordered array indexing
                                    // so we can just use the vertex index to index into the array
                                    // by selecting the appropriate bit from the index.
                                    const OFFSET: usize = const{(i & (3 << (2*k))) >> (2*k)};
                                    loc[k] = origin[k] + OFFSET;
                                }
                            }
                            const STORE_IND: usize = i % FP;
                            store[0][STORE_IND] = self.index_arr(loc, dimprod);
                        }
                        else { // const branch
                            // For other nodes, interpolate on child values

                            const Q: usize = const{FP.pow(j as u32)};
                            const LEVEL: bool = const {(i + 1).is_multiple_of(Q)};
                            const P: usize = const{((i + 1) / Q).saturating_sub(1) % FP};
                            const IND: usize = const{j.saturating_sub(1)};

                            if LEVEL { // const branch
                                let interped = interp_inner::<T, N>(
                                    &store[IND],
                                    dts[IND],
                                    sat[IND],
                                    self.linearize_extrapolation
                                );

                                store[j][P] = interped;
                            }
                        }
                    }
                }
            }
        }

        // Interpolate the final value
        // This could use a const index as well, if we were using a fixed number of dims
        let interped = interp_inner::<T, N>(
            &store[N - 1],
            dts[N - 1],
            sat[N - 1],
            self.linearize_extrapolation,
        );
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

        let floc = ((v - self.starts[dim]) / self.steps[dim]).floor(); // float loc
        // Signed integer loc, with the bottom of the cell aligned to place the normalized
        // coordinate t=0 at cell index 1
        let iloc = <isize as NumCast>::from(floc).ok_or("Unrepresentable coordinate value")? - 1;

        let n = self.dims[dim] as isize; // Number of grid points on this dimension
        let dimmax = n.saturating_sub(4).max(0); // maximum index for lower corner
        let loc: usize = iloc.max(0).min(dimmax) as usize; // unsigned integer loc clipped to interior

        // Observation point is outside the grid on the low side
        if iloc < -1 {
            saturation = Saturation::OutsideLow;
        }
        // Observation point is in the lower part of the cell
        // but not outside the grid
        else if iloc == -1 {
            saturation = Saturation::InsideLow;
        }
        // Observation point is in the upper part of the cell
        // but not outside the grid
        else if iloc > (n - 3) {
            saturation = Saturation::OutsideHigh;
        }
        // Observation point is in the upper part of the cell
        // but not outside the grid
        else if iloc == (n - 3) {
            saturation = Saturation::InsideHigh;
        }
        // Observation point is on the interior
        else {
            saturation = Saturation::None;
        }

        Ok((loc, saturation))
    }

    /// Index a single value from an array
    #[inline]
    fn index_arr(&self, loc: [usize; N], dimprod: [usize; N]) -> T {
        let mut i = 0;

        unroll! {
            for j < 5 in 0..N {
                i += loc[j] * dimprod[j];
            }
        }

        self.vals[i]
    }
}

/// Calculate slopes and offsets & select evaluation method
#[inline]
fn interp_inner<T: Float, const N: usize>(
    vals: &[T; 4],
    t: T,
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

            // Take slopes from centered difference
            let k0 = (vals[2] - vals[0]) / two;
            let k1 = (vals[3] - vals[1]) / two;

            normalized_hermite_spline(t, y0, dy, k0, k1)
        }
        Saturation::InsideLow => {
            //   t <-|
            // --|---|---|---|--
            //     x
            //
            // Flip direction to maintain symmetry
            // with the InsideHigh case
            let t = -t; // `t` always w.r.t. index 1 of cube
            let y0 = vals[1]; // Same starting point, opposite direction
            let dy = vals[0] - vals[1];

            let k0 = -(vals[2] - vals[0]) / two;
            let k1 = two * dy - k0; // Natural spline boundary condition

            normalized_hermite_spline(t, y0, dy, k0, k1)
        }
        Saturation::OutsideLow => {
            //   t <-|
            // --|---|---|---|--
            // x
            //
            // Flip direction to maintain symmetry
            // with the InsideHigh case
            let t = -t; // `t` always w.r.t. index 1 of cube
            let y0 = vals[1]; // Same starting point, opposite direction
            let y1 = vals[0];
            let dy = vals[0] - vals[1];

            let k0 = -(vals[2] - vals[0]) / two;
            let k1 = two * dy - k0; // Natural spline boundary condition

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
            //
            // Shift cell up an index
            // and offset `t`, which has value between 1 and 2
            // because it is calculated w.r.t. index 1
            let t = t - one;
            let y0 = vals[2];
            let dy = vals[3] - vals[2];

            let k0 = (vals[3] - vals[1]) / two;
            let k1 = two * dy - k0; // Natural spline boundary condition

            normalized_hermite_spline(t, y0, dy, k0, k1)
        }
        Saturation::OutsideHigh => {
            //           |-> t
            // --|---|---|---|--
            //                 x
            //
            // Shift cell up an index
            // and offset `t`, which has value between 1 and 2
            // because it is calculated w.r.t. index 1
            let t = t - one;
            let y0 = vals[2];
            let y1 = vals[3];
            let dy = vals[3] - vals[2];

            let k0 = (vals[3] - vals[1]) / two;
            let k1 = two * dy - k0; // Natural spline boundary condition

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
    use crate::utils::*;

    /// Iterate from 1 to 6 dimensions, making a minimum-sized grid for each one
    /// to traverse every combination of interpolating or extrapolating high or low on each dimension.
    /// Each test evaluates at 5^ndims locations, largely extrapolated in corner regions, so it
    /// rapidly becomes prohibitively slow above ndims=6.
    #[test]
    fn test_interp_extrap_1d_to_4d_linear() {
        for ndims in 1..=4 {
            println!("Testing in {ndims} dims");
            // Interp grid
            let dims: Vec<usize> = vec![4; ndims];
            let xs: Vec<Vec<f64>> = (0..ndims)
                .map(|i| linspace(-5.0 * (i as f64), 5.0 * ((i + 1) as f64), dims[i]))
                .collect();
            let grid = meshgrid((0..ndims).map(|i| &xs[i]).collect());
            let u: Vec<f64> = grid.iter().map(|x| x.iter().sum()).collect(); // sum is linear in every direction, good for testing
            let starts: Vec<f64> = xs.iter().map(|x| x[0]).collect();
            let steps: Vec<f64> = xs.iter().map(|x| x[1] - x[0]).collect();

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

            // Evaluate with spline extrapolation, which should collapse to linear
            interpn(&dims, &starts, &steps, &u, false, &xobsslice, &mut out[..]).unwrap();

            // Check that interpolated values match expectation,
            // using an absolute difference because some points are very close to or exactly at zero,
            // and do not do well under a check on relative difference.
            (0..uobs.len()).for_each(|i| assert!((out[i] - uobs[i]).abs() < 1e-12));

            // Evaluate with linear extrapolation
            interpn(&dims, &starts, &steps, &u, true, &xobsslice, &mut out[..]).unwrap();

            // Check that interpolated values match expectation,
            // using an absolute difference because some points are very close to or exactly at zero,
            // and do not do well under a check on relative difference.
            (0..uobs.len()).for_each(|i| assert!((out[i] - uobs[i]).abs() < 1e-12));
        }
    }

    /// Under both interpolation and extrapolation, a hermite spline with natural boundary condition
    /// can reproduce an N-dimensional quadratic function exactly
    #[test]
    fn test_interp_extrap_1d_to_4d_quadratic() {
        for ndims in 1..=4 {
            println!("Testing in {ndims} dims");
            // Interp grid
            let dims: Vec<usize> = vec![4; ndims];
            let xs: Vec<Vec<f64>> = (0..ndims)
                .map(|i| linspace(-5.0 * (i as f64), 5.0 * ((i + 1) as f64), dims[i]))
                .collect();
            let grid = meshgrid((0..ndims).map(|i| &xs[i]).collect());
            let u: Vec<f64> = (0..grid.len())
                .map(|i| {
                    let mut v = 0.0;
                    for j in 0..ndims {
                        v += grid[i][j] * grid[i][j];
                    }
                    v
                })
                .collect(); // Quadratic in every direction
            let starts: Vec<f64> = xs.iter().map(|x| x[0]).collect();
            let steps: Vec<f64> = xs.iter().map(|x| x[1] - x[0]).collect();

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
            interpn(&dims, &starts, &steps, &u, false, &xobsslice, &mut out[..]).unwrap();

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
        for ndims in 1..3 {
            println!("Testing in {ndims} dims");
            // Interp grid
            let dims: Vec<usize> = vec![10; ndims];
            let xs: Vec<Vec<f64>> = (0..ndims)
                .map(|i| linspace(-5.0 * (i as f64), 5.0 * ((i + 1) as f64), dims[i]))
                .collect();
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
            let starts: Vec<f64> = xs.iter().map(|x| x[0]).collect();
            let steps: Vec<f64> = xs.iter().map(|x| x[1] - x[0]).collect();

            // Observation points
            let xobs: Vec<Vec<f64>> = (0..ndims)
                .map(|i| linspace(-5.0 * (i as f64), 5.0 * ((i + 1) as f64), dims[i] + 2))
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
            interpn(&dims, &starts, &steps, &u, false, &xobsslice, &mut out[..]).unwrap();

            // Check that interpolated and extrapolated values match expectation,
            // using an absolute difference because some points are very close to or exactly at zero,
            // and do not do well under a check on relative difference.
            let tol = 2e-2 * f64::from(ndims as u32);

            (0..uobs.len()).for_each(|i| {
                let err = out[i] - uobs[i];
                // println!("{err}");
                assert!(err.abs() < tol);
            });
        }
    }
}
