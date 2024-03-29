//! Multilinear interpolation/extrapolation on a rectilinear grid.
//!
//! This is a fairly literal implementation of the geometric interpretation
//! of multilinear interpolation as an area- or volume- weighted average
//! on the interior of the grid, and extends this metaphor to include
//! extrapolation to off-grid points.
//!
//! While this method does not fully capitalize on vectorization,
//! it results in fairly minimal instantaneous memory usage,
//! and throughput performance is similar to existing methods.
//!
//! Operation Complexity
//! * Interpolating or extrapolating in face regions goes like
//!   * Best case: O(2^ndims * ndims) when evaluating points in neighboring grid cells.
//!   * Worst case: O(ndims * (2^ndims + ndims * log2(gridsize))) when evaluating arbitrary points.
//! * Extrapolating in corner regions goes like O(2^ndims * ndims^2).
//!
//! Memory Complexity
//! * Peak stack usage is O(MAXDIMS), which is minimally O(ndims).
//!
//! Timing
//! * Timing determinism is not guaranteed due to the
//!   difference in complexity between interpolation and extrapolation,
//!   as well as due to the use of a bisection search for the grid index
//!   location (which is itself not timing-deterministic) and the various
//!   methods used to attempt to avoid that bisection search.
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
//! // Do interpolation
//! rectilinear::interpn(grids, &z, &obs, &mut out).unwrap();
//! ```
//!
//! References
//! * https://en.wikipedia.org/wiki/Bilinear_interpolation#Weighted_mean
use num_traits::Float;

/// An arbitrary-dimensional multilinear interpolator / extrapolator on a rectilinear grid.
///
/// Unlike `RegularGridInterpolator`, this method does not handle
/// grids with negative step size; all grids must be monotonically increasing.
///
/// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
/// Assumes grids are monotonically _increasing_. Checking this is expensive, and is
/// left to the user.
///
/// While the worst-case interpolation runtime for this method is somewhat worse than
/// for the regular grid method, in the particular case where the sequence of points
/// being evaluated are within 1 grid cell of each other, it can be significantly
/// faster than the regular grid method due to bypassing the expensive process of
/// finding the location of the relevant grid cell.
///
/// Operation Complexity
/// * Interpolating or extrapolating in face regions goes like
///   * Best case: O(2^ndims * ndims) when evaluating points in neighboring grid cells.
///   * Worst case: O(ndims * (2^ndims + ndims * log2(gridsize))) when evaluating arbitrary points.
/// * Extrapolating in corner regions goes like O(2^ndims * ndims^2).
///
/// Memory Complexity
/// * Peak stack usage is O(MAXDIMS), which is minimally O(ndims).
///
/// Timing
/// * Timing determinism is not guaranteed due to the
///   difference in complexity between interpolation and extrapolation,
///   as well as due to the use of a bisection search for the grid index
///   location (which is itself not timing-deterministic) and the various
///   methods used to attempt to avoid that bisection search.
pub struct MultilinearRectilinear<'a, T: Float, const MAXDIMS: usize> {
    /// x, y, ... coordinate grids, each entry of size dims[i]
    grids: &'a [&'a [T]],

    /// Size of each dimension
    dims: [usize; MAXDIMS],

    /// Values at each point, size prod(dims)
    vals: &'a [T],
}

impl<'a, T: Float, const MAXDIMS: usize> MultilinearRectilinear<'a, T, MAXDIMS> {
    /// Build a new interpolator, using O(MAXDIMS) calculations and storage.
    ///
    /// This method does not handle degenerate dimensions with only a single
    /// grid entry; all grids must have at least 2 entries.
    ///
    /// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
    /// Assumes grids are monotonically _increasing_. Checking this is expensive, and is
    /// left to the user.
    ///
    /// # Errors
    /// * If any input dimensions do not match
    /// * If any dimensions have size < 2
    /// * If any grid's first two entries are not monotonically increasing
    ///   * This is a courtesy to catch _some_, but not all, cases where a non-monotonic
    ///     or reversed-order grid is provided.
    #[inline(always)]
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
    #[inline(always)]
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
    /// `origin` is a required mutable index store of minimum size `ndims`,
    /// which allows bypassing expensive bisection searches in some cases.
    /// It should be initialized to zero.
    ///
    /// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
    ///
    /// # Errors
    ///   * If the dimensionality of the point does not match the data
    ///   * If the dimensionality of either one exceeds the fixed maximum
    ///   * If values in `origin` are initialized to an index outside the grid
    #[inline(always)]
    fn interp_one(&self, x: &[T]) -> Result<T, &'static str> {
        // Check sizes
        let ndims = self.grids.len();
        if x.len() != ndims {
            return Err("Dimension mismatch");
        }

        // Initialize fixed-size intermediate storage.
        // This storage _could_ be initialized with the interpolator struct, but
        // this would then require that every usage of struct be `mut`, which is
        // ergonomically unpleasant. Based on testing with the regular grid method,
        // it would likely also be slower.
        let origin = &mut [0; MAXDIMS][..ndims]; // Indices of lower corner of grid cell
        let ioffs = &mut [false; MAXDIMS][..ndims]; // Offset index for selected vertex
        let sat = &mut [0_u8; MAXDIMS][..ndims]; // Saturated-low flag
        let dxs = &mut [T::zero(); MAXDIMS][..ndims]; // Sub-cell volume storage
        let steps = &mut [T::zero(); MAXDIMS][..ndims]; // Step size on each dimension
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

        // Populate lower corner
        for i in 0..ndims {
            (origin[i], sat[i]) = self.get_loc(x[i], i)
        }

        // Check if any dimension is saturated.
        // This gives a ~15% overall speedup for points on the interior.
        let any_dims_saturated = sat.iter().any(|&x| x != 0);

        // Calculate the total volume of this cell
        let cell_vol = self.get_cell(origin, steps);

        // Traverse vertices, summing contributions to the interpolated value.
        //
        // This visits the 2^ndims elements of the cartesian product
        // of `[0, 1] x ... x [0, 1]` using O(ndims) storage.
        let mut interped = T::zero();
        let nverts = 2_usize.pow(ndims as u32);
        for i in 0..nverts {
            let mut k: usize = 0; // index of the value for this vertex in self.vals

            for j in 0..ndims {
                // Every 2^nth vertex, flip which side of the cube we are examining
                // in the nth dimension.
                //
                // Because i % 2^n has double the period for each sequential n,
                // and their phase is only aligned once every 2^n for the largest
                // n in the set, this is guaranteed to produce a path that visits
                // each vertex exactly once.
                let flip = i % 2_usize.pow(j as u32) == 0;
                if flip {
                    ioffs[j] = !ioffs[j];
                }

                // Accumulate the index into the value array,
                // saturating to the bound if the resulting index would be outside.
                k += dimprod[j] * (origin[j] + ioffs[j] as usize);

                // Find the vector from the opposite vertex to the observation point
                let iloc = origin[j] + !ioffs[j] as usize; // Index of location of opposite vertex
                dxs[j] = (x[j] - self.grids[j][iloc]).abs(); // Loc. of opposite vertex
            }

            // Get the value at this vertex
            let v = self.vals[k];

            // Accumulate contribution from this vertex
            // * Interpolating: just take the volume-weighted value and continue on
            // * Extrapolating
            //   * With opposite vertex on multiple extrapolated dims: return zero
            //   * With opposite vertex on exactly one extrapolated dim
            //     * Negate contribution & clip extrapolated region to maintain linearity
            //   * Otherwise (meaning, corner regions)
            //     * O(ndim^2) operation to accumulate only the linear portions of
            //       the extrapolated volumes.
            //
            // While this section looks nearly identical between the regular grid
            // and rectilinear methods, it is different in a few subtle but important
            // ways, and separating it into shared functions makes it even harder
            // to read than it already is.
            if !any_dims_saturated {
                // Interpolating
                let vol = dxs[1..].iter().fold(dxs[0], |acc, x| acc * *x);
                interped = interped + v * vol;
            } else {
                // Extrapolating requires some special attention.
                let opsat = &mut [false; MAXDIMS][..ndims]; // Whether the opposite vertex is on the saturated bound
                let thissat = &mut [false; MAXDIMS][..ndims]; // Whether the current vertex is on the saturated bound
                let extrapdxs = &mut [T::zero(); MAXDIMS][..ndims]; // Extrapolated distances

                let mut opsatcount = 0;
                for j in 0..ndims {
                    // For which dimensions is the opposite vertex on a saturated bound?
                    opsat[j] = (!ioffs[j] && sat[j] == 2) || (ioffs[j] && sat[j] == 1);
                    // For how many total dimensions is the opposite vertex on a saturated bound?
                    opsatcount += opsat[j] as usize;

                    // For which dimensions is the current vertex on a saturated bound?
                    thissat[j] = sat[j] > 0 && !opsat[j];
                }

                // If the opposite vertex is on _more_ than one saturated bound,
                // it should be clipped on multiple axes which, if the clipping
                // were implemented in a general constructive geometry way, would
                // result in a zero volume. Since we only deal in the difference
                // in position between vertices and the observation point, our
                // clipping method would not properly set this volume to zero,
                // and we need to implement that behavior with explicit logic.
                let zeroed = opsatcount > 1;
                if zeroed {
                    // No contribution from this vertex
                    continue;
                }

                // If the opposite vertex is on exactly one saturated bound, negate its contribution
                // in order to move smoothly from weighted-average on the interior to extrapolation
                // on the exterior.
                //
                // If the opposite vertex is on exactly one saturated bound,
                // allow the dx on that dimension to be as large as needed,
                // but clip the dx on other saturated dimensions so that we
                // don't produce an overlapping partition in outside-corner regions.
                let neg = opsatcount == 1;
                if neg {
                    for j in 0..ndims {
                        if thissat[j] {
                            dxs[j] = dxs[j].min(steps[j]);
                        }
                    }

                    let vol = dxs[1..].iter().fold(dxs[0], |acc, x| acc * *x).neg();
                    interped = interped + v * vol;
                    continue;
                }

                // See `RegularGridInterpolator` for more details about the rationale
                // for this section, which handles extrapolation in corner regions.

                // Get the volume that is inside the cell
                //   Copy forward the original dxs, extrapolated or not,
                //   and clip to the cell boundary
                (0..ndims).for_each(|j| extrapdxs[j] = dxs[j].min(steps[j]));
                //   Get the volume of this region which does not extend outside the cell
                let vinterior = extrapdxs[1..].iter().fold(extrapdxs[0], |acc, x| acc * *x);

                // Find each linear exterior region by, one at a time, taking the volume
                // with one extrapolated dimension masked into the extrapdxs
                // which are otherwise clipped to the interior region.
                let mut vexterior = T::zero();
                for j in 0..ndims {
                    if thissat[j] {
                        let dx_was = extrapdxs[j];
                        extrapdxs[j] = dxs[j] - steps[j];
                        vexterior =
                            vexterior + extrapdxs[1..].iter().fold(extrapdxs[0], |acc, x| acc * *x);
                        extrapdxs[j] = dx_was; // Reset extrapdxs to original state for next calc
                    }
                }

                let vol = vexterior + vinterior;
                interped = interped + v * vol;
            }
        }

        Ok(interped / cell_vol)
    }

    /// Get the next-lower-or-exact index along this dimension where `x` is found,
    /// saturating to the bounds at the edges if the point is outside.
    ///
    /// At the high bound of a given dimension, saturates to the next-most-internal
    /// point in order to capture a full cube, then saturates to 0 if the resulting
    /// index would be off the grid (meaning, if a dimension has size one).
    ///
    /// Returned value like (lower_corner_index, saturation_flag).
    ///
    /// Saturation flag
    /// * 0 => inside
    /// * 1 => low
    /// * 2 => high
    ///
    /// Unfortunately, using a repr(u8) enum for the saturation flag
    /// causes a significant perf hit.
    #[inline(always)]
    fn get_loc(&self, v: T, dim: usize) -> (usize, u8) {
        let grid = self.grids[dim];
        let saturation: u8; // Saturated low/high/not at all
                            // Signed integer index location of this point

        // Bisection search to find location on the grid.
        //
        // The search will return `0` if the point is outside-low,
        // and will return `self.dims[dim]` if outside-high.
        //
        // We still need to convert to a signed integer here, because
        // if the point is extrapolated below the grid,
        // we may need to (briefly) represent a negative index.
        //
        // This process accounts for essentially the entire difference in
        // performance between this method and the regular-grid method.
        let iloc: isize = grid.partition_point(|x| *x < v) as isize - 1;
        let dimmax = self.dims[dim] - 2; // maximum index for lower corner
        let loc: usize = (iloc.max(0) as usize).min(dimmax); // unsigned integer loc clipped to interior

        // Observation point is outside the grid on the low side
        if iloc < 0 {
            saturation = 1;
        }
        // Observation point is outside the grid on the high side
        else if iloc > dimmax as isize {
            saturation = 2;
        }
        // Observation point is on the interior
        else {
            saturation = 0;
        }

        (loc, saturation)
    }

    /// Get the volume of the grid prism with `origin` as its lower corner
    /// and output the step sizes for this cell as well.
    #[inline(always)]
    fn get_cell(&self, origin: &[usize], steps: &mut [T]) -> T {
        let ndims = self.grids.len();
        for i in 0..ndims {
            // Index of upper face (saturating to bounds)
            let j = origin[i] + 1;
            steps[i] = self.grids[i][j] - self.grids[i][origin[i]];
        }

        let cell_vol = steps[1..ndims].iter().fold(steps[0], |acc, x| acc * *x);

        cell_vol
    }
}

/// Evaluate multilinear interpolation on a regular grid in up to 10 dimensions.
/// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
///
/// This is a convenience function; best performance will be achieved by using the exact right
/// number for the MAXDIMS parameter, as this will slightly reduce compute and storage overhead,
/// and the underlying method can be extended to more than this function's limit of 8 dimensions.
/// The limit of 8 dimensions was chosen for no more specific reason than to reduce unit test times.
///
/// While this method initializes the interpolator struct on every call, the overhead of doing this
/// is minimal even when using it to evaluate one observation point at a time.
#[inline(always)]
pub fn interpn<T: Float>(
    grids: &[&[T]],
    vals: &[T],
    obs: &[&[T]],
    out: &mut [T],
) -> Result<(), &'static str> {
    MultilinearRectilinear::<'_, T, 8>::new(grids, vals)?.interp(obs, out)?;
    Ok(())
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
#[inline(always)]
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

#[cfg(test)]
mod test {
    use super::{interpn, MultilinearRectilinear};
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

        let interpolator: MultilinearRectilinear<'_, _, 2> =
            MultilinearRectilinear::new(&grids, &z[..]).unwrap();

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
}
