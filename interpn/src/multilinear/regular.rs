//! Multilinear interpolation/extrapolation on a regular grid.
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
//! * Interpolating or extrapolating in face regions goes like O(2^ndims).
//! * Extrapolating in corner regions goes like O(2^ndims * ndims^2).
//!
//! Memory Complexity
//! * Peak stack usage is O(MAXDIMS), which is minimally O(ndims).
//!
//! Timing
//! * Timing determinism is not guaranteed due to the
//!   difference in complexity between interpolation and extrapolation.
//! * An interpolation-only variant of this algorithm could achieve
//!   near-deterministic timing, but would produce incorrect results
//!   when evaluated at off-grid points.
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
//! // Storage for output
//! let mut out = [0.0; 2];
//!
//! // Do interpolation
//! regular::interpn(&dims, &starts, &steps, &z, &obs, &mut out).unwrap();
//! ```
//!
//! References
//! * https://en.wikipedia.org/wiki/Bilinear_interpolation#Weighted_mean
use num_traits::{Float, NumCast};

/// An arbitrary-dimensional multilinear interpolator / extrapolator on a regular grid.
///
/// Unlike `RectilinearGridInterpolator`, this method can accommodate grids with a
/// negative step size.
///
/// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
///
/// Operation Complexity
/// * Interpolating or extrapolating in face regions goes like O(2^ndims).
/// * Extrapolating in corner regions goes like O(2^ndims * ndims^2).
///
/// Memory Complexity
/// * Peak stack usage is O(MAXDIMS), which is minimally O(ndims).
///
/// Timing
/// * Timing determinism is not guaranteed due to the
///   difference in complexity between interpolation and extrapolation.
/// * An interpolation-only variant of this algorithm could achieve
///   near-deterministic timing, but would produce incorrect results
///   when evaluated at off-grid points.
pub struct RegularGridInterpolator<'a, T: Float, const MAXDIMS: usize> {
    /// Number of dimensions
    ndims: usize,

    /// Size of each dimension
    dims: [usize; MAXDIMS],

    /// Starting point of each dimension, size dims.len()
    starts: [T; MAXDIMS],

    /// Step size for each dimension, size dims.len()
    steps: [T; MAXDIMS],

    /// Values at each point, size prod(dims)
    vals: &'a [T],
}

impl<'a, T: Float, const MAXDIMS: usize> RegularGridInterpolator<'a, T, MAXDIMS> {
    /// Build a new interpolator, using O(MAXDIMS) calculations and storage.
    ///
    /// This method does not handle degenerate dimensions with only a single
    /// grid entry; all grids must have at least 2 entries.
    ///
    /// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
    ///
    /// # Errors
    /// * If any input dimensions do not match
    /// * If any dimensions have size < 2
    /// * If any step sizes have zero or negative magnitude
    #[inline(always)]
    pub fn new(
        dims: &[usize],
        starts: &[T],
        steps: &[T],
        vals: &'a [T],
    ) -> Result<Self, &'static str> {
        // Check dimensions
        let ndims = dims.len();
        let nvals = dims.iter().product();
        if !(starts.len() == ndims && steps.len() == ndims && vals.len() == nvals && ndims > 0) {
            return Err("Dimension mismatch");
        }

        // Make sure all dimensions have at least two entries
        let degenerate = dims[..ndims].iter().any(|&x| x < 2);
        if degenerate {
            return Err("All grids must have at least two entries");
        }
        // Check if any dimensions have zero step size
        let steps_are_positive = steps.iter().all(|&x| x > T::zero());
        if !steps_are_positive {
            return Err("All grids must be monotonically increasing");
        }

        // Copy grid info into struct.
        // Keeping this data local to the struct eliminates an issue where,
        // if the grid info is defined somewhere not local to where the struct
        // is defined and used, the & references to it cost more than the entire
        // rest of the interpolation operation.
        let mut steps_local = [T::zero(); MAXDIMS];
        let mut starts_local = [T::zero(); MAXDIMS];
        let mut dims_local = [0_usize; MAXDIMS];
        steps_local[..ndims].copy_from_slice(&steps[..ndims]);
        starts_local[..ndims].copy_from_slice(&starts[..ndims]);
        dims_local[..ndims].copy_from_slice(&dims[..ndims]);

        Ok(Self {
            ndims,
            dims: dims_local,
            starts: starts_local,
            steps: steps_local,
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
    #[inline(always)]
    pub fn interp(&self, x: &[&[T]], out: &mut [T]) -> Result<(), &'static str> {
        let n = out.len();
        let ndims = self.ndims;
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
    #[inline(always)]
    pub fn interp_one(&self, x: &[T]) -> Result<T, &'static str> {
        // Check sizes
        let ndims = self.ndims;
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
        let steps = &self.steps[..ndims]; // Step size for each dimension
        let origin = &mut [0_usize; MAXDIMS][..ndims]; // Indices of lower corner of hypercube
        let ioffs = &mut [false; MAXDIMS][..ndims]; // Offset index for selected vertex
        let sat = &mut [0_u8; MAXDIMS][..ndims]; // Saturation none/high/low flags for each dim
        let dxs = &mut [T::zero(); MAXDIMS][..ndims]; // Sub-cell volume storage
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

        // Compute volume of reference cell.
        // Maybe counterintuitively, doing this calculation for every call
        // is as fast or faster than doing it once at struct initialization
        // then referring to the stored value.
        let cell_vol = steps[1..].iter().fold(steps[0], |acc, x| acc * *x);

        // Populate lower corner and saturation flag for each dimension
        for i in 0..ndims {
            (origin[i], sat[i]) = self.get_loc(x[i], i)?;
        }

        // Check if any dimension is saturated.
        let any_dims_saturated = sat.iter().any(|&x| x != 0);

        // Traverse vertices, summing contributions to the interpolated value.
        //
        // This visits the 2^ndims elements of the cartesian product
        // of `[0, 1] x ... x [0, 1]` without simultaneously actualizing them in storage.
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
                let floc = T::from(iloc);
                match floc {
                    Some(floc) => {
                        let loc = self.starts[j] + steps[j] * floc; // Loc. of opposite vertex
                        dxs[j] = (x[j] - loc).abs(); // Use dxs[j] as storage for float locs
                    }
                    None => return Err("Unrepresentable coordinate value"),
                }
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

                // If this vertex is on multiple saturated bounds, then the prism formed by the
                // opposite vertex and the observation point will be extrapolated in more than
                // one dimension, which will produce some regions with volume that scales
                // nonlinearly with the position of the observation point.
                // We need to restore linearity without resorting to using the recursive algorithm
                // which would drive us to actualize (2^(n-1) * ndims) float values simultaneously.
                //
                // To do this, we can subtract the nonlinear regions' volume from the total
                // volume of the opposite-to-observation prism for this vertex.
                //
                // Put differently - find the part of the volume that is scaling non-linearly
                // in the coordinates, and bookkeep it to be removed entirely later.
                //
                // For one dimension, there are no such regions. For two dimensions, only the
                // corner region contributes. For higher dimensions, there are increasing
                // numbers of types of regions that appear, so we need a relatively general
                // way of handling this without knowing all of those types of region.
                //
                // One way of circumventing the need to enumerate types of nonlinear region
                // is to capitalize on the observation that the _linear_ regions are all of the
                // same form, even in higher dimensions. We can traverse those instead,
                // subtracting each one from the full extrapolated volume for this vertex
                // until what's left is only the part that we want to remove. Then, we can
                // remove that part and keep the rest.
                //
                // Continuing from that thought, we can skip evaluating the nonlinear portions
                // entirely, by evaluating the interior portion and each linear exterior portion
                // (which we needed to evaluate to remove them from the enclosing volume anyway)
                // then summing the linear portions together directly. This avoids the loss of
                // float precision that can come from addressing the nonlinear regions directly,
                // as this can cause us to add some very large and very small numbers together
                // in an order that is not necessarily favorable.

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
    fn get_loc(&self, v: T, dim: usize) -> Result<(usize, u8), &'static str> {
        let saturation: u8; // Saturated low/high/not at all

        let floc = ((v - self.starts[dim]) / self.steps[dim]).floor(); // float loc
        let iloc = <isize as NumCast>::from(floc); // signed integer loc

        match iloc {
            Some(iloc) => {
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

                Ok((loc, saturation))
            }
            None => Err("Unrepresentable coordinate value"),
        }
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
    dims: &[usize],
    starts: &[T],
    steps: &[T],
    vals: &[T],
    obs: &[&[T]],
    out: &mut [T],
) -> Result<(), &'static str> {
    RegularGridInterpolator::<'_, T, 8>::new(dims, starts, steps, vals)?.interp(obs, out)?;
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
    dims: &[usize],
    starts: &[T],
    steps: &[T],
    obs: &[&[T]],
    atol: T,
    out: &mut [bool],
) -> Result<(), &'static str> {
    let ndims = dims.len();
    if !(obs.len() == ndims && out.len() == ndims) {
        return Err("Dimension mismatch");
    }

    for i in 0..ndims {
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

#[cfg(test)]
mod test {
    use super::interpn;
    use crate::utils::*;

    /// Iterate from 1 to 8 dimensions, making a minimum-sized grid for each one
    /// to traverse every combination of interpolating or extrapolating high or low on each dimension.
    /// Each test evaluates at 3^ndims locations, largely extrapolated in corner regions, so it
    /// rapidly becomes prohibitively slow after about ndims=9.
    #[test]
    fn test_interp_extrap_1d_to_8d() {
        for ndims in 1..=8 {
            println!("Testing in {ndims} dims");
            // Interp grid
            let dims: Vec<usize> = vec![2; ndims];
            let xs: Vec<Vec<f64>> = (0..ndims)
                .map(|i| linspace(-5.0 * (i as f64), 5.0 * ((i + 1) as f64), dims[i]))
                .collect();
            let grid = meshgrid((0..ndims).map(|i| &xs[i]).collect());
            let u: Vec<f64> = grid.iter().map(|x| x.iter().sum()).collect(); // sum is linear in every direction, good for testing
            let starts: Vec<f64> = xs.iter().map(|x| x[0]).collect();
            let steps: Vec<f64> = xs.iter().map(|x| x[1] - x[0]).collect();

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
            interpn(&dims, &starts, &steps, &u, &xobsslice, &mut out[..]).unwrap();

            // Check that interpolated values match expectation,
            // using an absolute difference because some points are very close to or exactly at zero,
            // and do not do well under a check on relative difference.
            (0..uobs.len()).for_each(|i| assert!((out[i] - uobs[i]).abs() < 1e-12));
        }
    }
}
