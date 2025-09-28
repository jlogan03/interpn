//! Multilinear interpolation/extrapolation on a regular grid.
//!
//! ```rust
//! use interpn::multilinear::regular;
//!
//! // Define a grid
//! let x = [1.0_f64, 2.0];
//! let y = [1.0_f64, 1.5];
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
//! let z = [2.0; 4];
//!
//! // Observation points to interpolate/extrapolate
//! let xobs = [0.0_f64, 5.0];
//! let yobs = [-1.0, 3.0];
//! let obs = [&xobs[..], &yobs[..]];
//!
//! // Do interpolation, allocating for the output for convenience
//! regular::interpn_alloc(&dims, &starts, &steps, &z, &obs).unwrap();
//! ```
//!
//! References
//! * https://en.wikipedia.org/wiki/Bilinear_interpolation#Repeated_linear_interpolation
use super::MultilinearRegularRecursive;
use crunchy::unroll;
use num_traits::{Float, NumCast};

/// Evaluate multilinear interpolation on a regular grid in up to 8 dimensions.
/// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
///
/// For 1-6 dimensions, a fast flattened method is used. For higher dimensions, where that flattening
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
    obs: &[&[T]],
    out: &mut [T],
) -> Result<(), &'static str> {
    let ndims = dims.len();
    if starts.len() != ndims || steps.len() != ndims || obs.len() != ndims {
        return Err("Dimension mismatch");
    }

    match ndims {
        1 => MultilinearRegular::<'_, T, 1>::new(
            dims.try_into().unwrap(),
            starts.try_into().unwrap(),
            steps.try_into().unwrap(),
            vals,
        )?
        .interp(obs.try_into().unwrap(), out),
        2 => MultilinearRegular::<'_, T, 2>::new(
            dims.try_into().unwrap(),
            starts.try_into().unwrap(),
            steps.try_into().unwrap(),
            vals,
        )?
        .interp(obs.try_into().unwrap(), out),
        3 => MultilinearRegular::<'_, T, 3>::new(
            dims.try_into().unwrap(),
            starts.try_into().unwrap(),
            steps.try_into().unwrap(),
            vals,
        )?
        .interp(obs.try_into().unwrap(), out),
        4 => MultilinearRegular::<'_, T, 4>::new(
            dims.try_into().unwrap(),
            starts.try_into().unwrap(),
            steps.try_into().unwrap(),
            vals,
        )?
        .interp(obs.try_into().unwrap(), out),
        5 => MultilinearRegular::<'_, T, 5>::new(
            dims.try_into().unwrap(),
            starts.try_into().unwrap(),
            steps.try_into().unwrap(),
            vals,
        )?
        .interp(obs.try_into().unwrap(), out),
        6 => MultilinearRegular::<'_, T, 6>::new(
            dims.try_into().unwrap(),
            starts.try_into().unwrap(),
            steps.try_into().unwrap(),
            vals,
        )?
        .interp(obs.try_into().unwrap(), out),
        7 => MultilinearRegularRecursive::<'_, T, 7>::new(dims, starts, steps, vals)?
            .interp(obs, out),
        8 => MultilinearRegularRecursive::<'_, T, 8>::new(dims, starts, steps, vals)?
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
    obs: &[&[T]],
) -> Result<Vec<T>, &'static str> {
    let mut out = vec![T::zero(); obs[0].len()];
    interpn(dims, starts, steps, vals, obs, &mut out)?;
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
    dims: &[usize],
    starts: &[T],
    steps: &[T],
    obs: &[&[T]],
    atol: T,
    out: &mut [bool],
) -> Result<(), &'static str> {
    let n = dims.len();
    if !(obs.len() == n && out.len() == n) {
        return Err("Dimension mismatch");
    }

    for i in 0..n {
        let first = starts[i];
        let last_elem = <T as NumCast>::from(dims[i] - 1); // Last element in grid

        match last_elem {
            Some(last_elem) => {
                let last = starts[i] + steps[i] * last_elem;
                let lo = first.min(last);
                let hi = first.max(last);

                let bad = obs[i]
                    .iter()
                    .any(|&x| (x - lo) <= -atol || (x - hi) >= atol);
                out[i] = bad;
            }
            // Passing an unrepresentable number in isn't, strictly speaking, an error
            // and since an unrepresentable number can't be on the grid,
            // we can just flag it for the bounds check like normal
            None => {
                out[i] = true;
            }
        }
    }
    Ok(())
}

/// An arbitrary-dimensional multilinear interpolator / extrapolator on a regular grid.
///
/// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
///
/// Operation Complexity
/// * O(2^N) for interpolation and extrapolation in all regions.
///
/// Memory Complexity
/// * Peak stack usage is O(N), which is minimally O(N).
/// * While evaluation is recursive, the recursion has constant
///   max depth of N, which provides a guarantee on peak
///   memory usage.
///
/// Timing
/// * Timing determinism is guaranteed to the extent that floating-point calculation timing is consistent.
///   That said, floating-point calculations can take a different number of clock-cycles depending on numerical values.
pub struct MultilinearRegular<'a, T: Float, const N: usize> {
    /// Size of each dimension
    dims: [usize; N],

    /// Starting point of each dimension, size dims.len()
    starts: [T; N],

    /// Step size for each dimension, size dims.len()
    steps: [T; N],

    /// Values at each point, size prod(dims)
    vals: &'a [T],
}

impl<'a, T: Float, const N: usize> MultilinearRegular<'a, T, N> {
    /// Build a new interpolator, using O(N) calculations and storage.
    ///
    /// This method does not handle degenerate dimensions; all grids must have at least 2 entries.
    ///
    /// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
    ///
    /// # Errors
    /// * If any input dimensions do not match
    /// * If any dimensions have size < 2
    /// * If any step sizes have zero or negative magnitude
    pub fn new(
        dims: [usize; N],
        starts: [T; N],
        steps: [T; N],
        vals: &'a [T],
    ) -> Result<Self, &'static str> {
        // Check dimensions
        const {
            assert!(
                N > 0 && N < 7,
                "Flattened method defined for 1-6 dimensions. For higher dimensions, use recursive method."
            );
        }
        let nvals: usize = dims.iter().product();
        if vals.len() != nvals {
            return Err("Dimension mismatch");
        }
        // Make sure all dimensions have at least four entries
        let degenerate = dims[..N].iter().any(|&x| x < 2);
        if degenerate {
            return Err("All grids must have at least two entries");
        }
        // Check if any dimensions have zero or negative step size
        let steps_are_positive = steps.iter().all(|&x| x > T::zero());
        if !steps_are_positive {
            return Err("All grids must be monotonically increasing");
        }

        Ok(Self {
            dims,
            starts,
            steps,
            vals,
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
        // if x.len() != N {
        //     return Err("Dimension mismatch");
        // }
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
    /// using fixed-size intermediate storage of O(N) and no allocation.
    ///
    /// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
    ///
    /// # Errors
    ///   * If the dimensionality of the point does not match the data
    ///   * If the dimensionality of either one exceeds the fixed maximum
    ///   * If the index along any dimension exceeds the maximum representable
    ///     integer value within the value type `T`
    #[inline]
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
        let mut origin = [0_usize; N]; // Indices of lower corner of hypercub
        let mut dts = [T::zero(); N]; // Normalized coordinate storage
        let mut dimprod = [1_usize; N];
        let mut loc = [0_usize; N];
        let mut store = [[T::zero(); FP]; N];

        let mut acc = 1;
        unroll! {
            for i < 7 in 0..N {
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
                origin[i] = self.get_loc(x[i], i)?;


                // Calculate normalized delta locations
                let index_zero_loc = self.starts[i]
                    + self.steps[i]
                        * <T as NumCast>::from(origin[i]).ok_or("Unrepresentable coordinate value")?;
                dts[i] = (x[i] - index_zero_loc) / self.steps[i];
            }
        }

        // Recursive interpolation of one dependency tree at a time
        const FP: usize = 2; // Footprint size
        let nverts = const { FP.pow(N as u32) }; // Total number of vertices

        unroll! {
            for i < 64 in 0..nverts {  // const loop
                // Index, interpolate, or pass on each level of the tree
                unroll!{
                    for j < 7 in 0..N {  // const loop

                        // Most of these iterations will get optimized out
                        if const{j == 0} { // const branch
                            // At leaves, index values

                            unroll!{
                                for k < 7 in 0..N {  // const loop
                                    // Bit pattern in an integer matches C-ordered array indexing
                                    // so we can just use the vertex index to index into the array
                                    // by selecting the appropriate bit from the index.
                                    const OFFSET: usize = const{(i & (1 << k)) >> k};
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
                                let y0 = store[IND][0];
                                let dy = store[IND][1] - y0;
                                let t = dts[IND];
                                let interped = y0 + t * dy;

                                store[j][P] = interped;
                            }
                        }
                    }
                }
            }
        }

        // Interpolate the final value
        let y0 = store[N - 1][0];
        let dy = store[N - 1][1] - y0;
        let t = dts[N - 1];
        let interped = y0 + t * dy;
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
    fn get_loc(&self, v: T, dim: usize) -> Result<usize, &'static str> {
        let floc = ((v - self.starts[dim]) / self.steps[dim]).floor(); // float loc
        // Signed integer loc, with the bottom of the cell aligned to place the normalized
        // coordinate t=0 at cell index 1
        let iloc = <isize as NumCast>::from(floc).ok_or("Unrepresentable coordinate value")?;

        let n = self.dims[dim] as isize; // Number of grid points on this dimension
        let dimmax = n.saturating_sub(2).max(0); // maximum index for lower corner
        let loc: usize = iloc.max(0).min(dimmax) as usize; // unsigned integer loc clipped to interior

        Ok(loc)
    }

    /// Index a single value from an array
    #[inline]
    fn index_arr(&self, loc: [usize; N], dimprod: [usize; N]) -> T {
        let mut i = 0;

        unroll! {
            for j < 7 in 0..N {
                i += loc[j] * dimprod[j];
            }
        }

        self.vals[i]
    }
}

#[cfg(test)]
mod test {
    use super::interpn;
    use crate::{MultilinearRegular, utils::*};

    /// Iterate from 1 to 8 dimensions, making a minimum-sized grid for each one
    /// to traverse every combination of interpolating or extrapolating high or low on each dimension.
    /// Each test evaluates at 3^N locations, largely extrapolated in corner regions, so it
    /// rapidly becomes prohibitively slow after about N=9.
    #[test]
    fn test_interp_extrap_1d_to_6d() {
        for n in 1..=6 {
            println!("Testing in {n} dims");
            // Interp grid
            let dims: Vec<usize> = vec![2; n];
            let xs: Vec<Vec<f64>> = (0..n)
                .map(|i| linspace(-5.0 * (i as f64), 5.0 * ((i + 1) as f64), dims[i]))
                .collect();
            let grid = meshgrid((0..n).map(|i| &xs[i]).collect());
            let u: Vec<f64> = grid.iter().map(|x| x.iter().sum()).collect(); // sum is linear in every direction, good for testing
            let starts: Vec<f64> = xs.iter().map(|x| x[0]).collect();
            let steps: Vec<f64> = xs.iter().map(|x| x[1] - x[0]).collect();

            // Observation points
            let xobs: Vec<Vec<f64>> = (0..n)
                .map(|i| linspace(-7.0 * (i as f64), 7.0 * ((i + 1) as f64), 3))
                .collect();
            let gridobs = meshgrid((0..n).map(|i| &xobs[i]).collect());
            let gridobs_t: Vec<Vec<f64>> = (0..n)
                .map(|i| gridobs.iter().map(|x| x[i]).collect())
                .collect(); // transpose
            let xobsslice: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..]).collect();
            let uobs: Vec<f64> = gridobs.iter().map(|x| x.iter().sum()).collect(); // expected output at observation points
            let mut out = vec![0.0; uobs.len()];

            // Evaluate
            interpn(&dims, &starts, &steps, &u, &xobsslice, &mut out[..]).unwrap();

            // Check that interpolated values match expectation,
            // using an absolute difference because some points are very close to or exactly at zero,
            // and do not do well under a check on relative difference.

            (0..uobs.len()).for_each(|i| {
                let outi = out[i];
                let uobsi = uobs[i];
                println!("{outi} {uobsi}");
                assert!((out[i] - uobs[i]).abs() < 1e-12)
            });
        }
    }

    /// Interpolate on a hat-shaped function to make sure that the grid cell indexing is aligned properly
    #[test]
    fn test_interp_hat_func() {
        fn hat_func(x: f64) -> f64 {
            if x <= 1.0 { x } else { 2.0 - x }
        }

        let y = (0..3).map(|x| hat_func(x as f64)).collect::<Vec<f64>>();
        let obs = linspace(-2.0, 4.0, 100);

        let interpolator: MultilinearRegular<f64, 1> =
            MultilinearRegular::new([3], [0.0], [1.0], &y).unwrap();

        (0..obs.len()).for_each(|i| {
            assert_eq!(hat_func(obs[i]), interpolator.interp_one([obs[i]]).unwrap());
        })
    }
}
