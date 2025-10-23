//! Nearest-neighbor interpolation/extrapolation on a regular grid.
//!
//! ```rust
//! use interpn::nearest::regular;
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
use crate::index_arr_fixed_dims;
use crunchy::unroll;
use num_traits::{Float, NumCast};

/// Evaluate nearest-neighbor interpolation on a regular grid in up to 6 dimensions.
/// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
///
/// This is a convenience function; best performance will be achieved by using the exact right
/// number for the N parameter, as this will slightly reduce compute and compile times.
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
        1 => NearestRegular::<'_, T, 1>::new(
            dims.try_into().unwrap(),
            starts.try_into().unwrap(),
            steps.try_into().unwrap(),
            vals,
        )?
        .interp(obs.try_into().unwrap(), out),
        2 => NearestRegular::<'_, T, 2>::new(
            dims.try_into().unwrap(),
            starts.try_into().unwrap(),
            steps.try_into().unwrap(),
            vals,
        )?
        .interp(obs.try_into().unwrap(), out),
        3 => NearestRegular::<'_, T, 3>::new(
            dims.try_into().unwrap(),
            starts.try_into().unwrap(),
            steps.try_into().unwrap(),
            vals,
        )?
        .interp(obs.try_into().unwrap(), out),
        4 => NearestRegular::<'_, T, 4>::new(
            dims.try_into().unwrap(),
            starts.try_into().unwrap(),
            steps.try_into().unwrap(),
            vals,
        )?
        .interp(obs.try_into().unwrap(), out),
        5 => NearestRegular::<'_, T, 5>::new(
            dims.try_into().unwrap(),
            starts.try_into().unwrap(),
            steps.try_into().unwrap(),
            vals,
        )?
        .interp(obs.try_into().unwrap(), out),
        6 => NearestRegular::<'_, T, 6>::new(
            dims.try_into().unwrap(),
            starts.try_into().unwrap(),
            steps.try_into().unwrap(),
            vals,
        )?
        .interp(obs.try_into().unwrap(), out),
        _ => Err("Dimension exceeds maximum (6)."),
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

pub use crate::multilinear::regular::check_bounds;

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
pub struct NearestRegular<'a, T: Float, const N: usize> {
    /// Size of each dimension
    dims: [usize; N],

    /// Starting point of each dimension, size dims.len()
    starts: [T; N],

    /// Step size for each dimension, size dims.len()
    steps: [T; N],

    /// Values at each point, size prod(dims)
    vals: &'a [T],
}

impl<'a, T: Float, const N: usize> NearestRegular<'a, T, N> {
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
        let mut dimprod = [1_usize; N];
        let mut loc = [0_usize; N];

        // These are done at compile time for primitives like f32, f64
        let two = T::one() + T::one();
        let half = T::one() / two;

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
                let origin = self.get_loc(x[i], i)?;
                let origin_f = <T as NumCast>::from(origin).ok_or("Unrepresentable coordinate value")?;

                // Calculate normalized delta locations
                #[cfg(not(feature = "fma"))]
                let index_zero_loc = self.starts[i] + self.steps[i] * origin_f;
                #[cfg(feature = "fma")]
                let index_zero_loc = self.steps[i].mul_add(origin_f, self.starts[i]);

                let dt = (x[i] - index_zero_loc) / self.steps[i];

                // Determine nearest index for this dimension based on distance.
                // NOTE: This method, despite including a division operation,
                // is about 10-20% faster than just checking the left and right
                // distances against each other directly.
                let offset = if dt <= half {
                    0
                } else {
                    1
                };

                loc[i] = origin + offset;
            }
        }

        let interped = index_arr_fixed_dims(loc, dimprod, self.vals);
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
}

#[cfg(test)]
mod test {
    use super::interpn;
    use crate::{NearestRegular, utils::*};

    fn nearest_regular_index(value: f64, start: f64, step: f64, dim: usize) -> usize {
        let floc = ((value - start) / step).floor();
        let n = dim as isize;
        let dimmax = n.saturating_sub(2).max(0);
        let origin = floc as isize;
        let origin = origin.max(0).min(dimmax) as usize;
        let index_zero = start + step * origin as f64;
        let dt = (value - index_zero) / step;
        if dt <= 0.5 {
            origin
        } else {
            (origin + 1).min(dim - 1)
        }
    }

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
            let expected: Vec<f64> = gridobs
                .iter()
                .map(|point| {
                    (0..n)
                        .map(|dim| {
                            let idx = nearest_regular_index(
                                point[dim],
                                starts[dim],
                                steps[dim],
                                dims[dim],
                            );
                            starts[dim] + steps[dim] * idx as f64
                        })
                        .sum()
                })
                .collect();
            let mut out = vec![0.0; expected.len()];

            // Evaluate
            interpn(&dims, &starts, &steps, &u, &xobsslice, &mut out[..]).unwrap();

            // Check that interpolated values match expectation,
            // using an absolute difference because some points are very close to or exactly at zero,
            // and do not do well under a check on relative difference.

            (0..expected.len()).for_each(|i| {
                let outi = out[i];
                let expecti = expected[i];
                println!("{outi} {expecti}");
                assert!((outi - expecti).abs() < 1e-12)
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

        let interpolator: NearestRegular<f64, 1> =
            NearestRegular::new([3], [0.0], [1.0], &y).unwrap();

        (0..obs.len()).for_each(|i| {
            let idx = nearest_regular_index(obs[i], 0.0, 1.0, y.len());
            assert_eq!(y[idx], interpolator.interp_one([obs[i]]).unwrap());
        })
    }
}
