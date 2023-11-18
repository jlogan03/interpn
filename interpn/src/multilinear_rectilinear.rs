use num_traits::Float;

/// An arbitrary-dimensional multilinear interpolator on a rectilinear grid.
///
/// Unlike `RegularGridInterpolator`, this method does not handle
/// degenerate dimensions with only a single grid entry; all grids
/// must have at least 2 entries.
///
/// Assumes C-style ordering of vals (x0, y0, z0,   x0, y0, z1,   ...,   x0, yn, zn).
/// Assumes grids are monotonically _increasing_. Checking this is expensive, and is
/// left to the user.
pub struct RectilinearGridInterpolator<'a, T: Float, const MAXDIMS: usize> {
    /// x, y, ... coordinate grids, size(dims.len()), each entry of size dims[i]
    grids: &'a [&'a [T]],

    /// Size of each dimension
    dims: [usize; MAXDIMS],

    /// Cumulative products of higher dimensions, used for indexing
    dimprod: [usize; MAXDIMS],

    /// Indices of lower corner of hypercube,
    /// stored with the struct in order to be used as a rolling
    /// initial guess for the index of the observation point,
    /// and mutated on every evaluation of the interpolator.
    origin: [usize; MAXDIMS],

    /// Values at each point, size prod(dims)
    vals: &'a [T],
}

impl<'a, T, const MAXDIMS: usize> RectilinearGridInterpolator<'a, T, MAXDIMS>
where
    T: Float,
{
    /// Build a new interpolator, using O(MAXDIMS) calculations and storage.
    ///
    /// This method does not handle degenerate dimensions with only a single
    /// grid entry; all grids must have at least 2 entries.
    ///
    /// Assumes C-style ordering of vals ([x0, y0], [x0, y1], ..., [x0, yn], [x1, y0], ...).
    /// Assumes grids are monotonically _increasing_. Checking this is expensive, and is
    /// left to the user.
    ///
    /// # Panics
    /// * If any input dimensions do not match
    /// * If any dimensions have size < 2
    /// * If any grid's first two entries are not monotonically increasing
    ///   * This is a courtesy to catch _some_, but not all, cases where a non-monotonic
    ///     or reversed-order grid is provided.
    pub fn new(grids: &'a [&'a [T]], vals: &'a [T]) -> Self {
        // Check dimensions
        let ndims = grids.len();
        let mut dims = [1_usize; MAXDIMS];
        (0..ndims).for_each(|i| dims[i] = grids[i].len());
        let nvals = dims[..ndims].iter().product();
        assert!(
            vals.len() == nvals && ndims > 0 && ndims <= MAXDIMS,
            "Dimension mismatch"
        );
        // Check if any grids are degenerate
        let degenerate = (0..ndims).any(|i| dims[i] < 2);
        assert!(!degenerate, "All grids must have at least 2 entries");
        // Check that at least the first two entries in each grid are monotonic
        let monotonic_maybe = (0..ndims).all(|i| grids[i][1] > grids[i][0]);
        assert!(
            monotonic_maybe,
            "All grids must be monotonically increasing"
        );

        // Populate cumulative product of higher dimensions for indexing.
        //
        // Each entry is the cumulative product of the size of dimensions
        // higher than this one, which is the stride between blocks
        // relating to a given index along each dimension.
        let mut dimprod = [1_usize; MAXDIMS];
        let mut acc = 1;
        (0..ndims).for_each(|i| {
            dimprod[ndims - i - 1] = acc;
            acc *= dims[ndims - i - 1];
        });

        let origin = [0; MAXDIMS];

        Self {
            grids,
            dims,
            dimprod,
            origin,
            vals,
        }
    }

    /// Interpolate on contiguous lists of points.
    #[inline(always)]
    pub fn interp(&mut self, x: &[&'a [T]], out: &mut [T]) {
        let n = out.len();
        let ndims = self.grids.len();
        // Make sure there are enough coordinate inputs for each dimension
        assert!(x.len() == ndims, "Dimension mismatch");
        // Make sure the size of inputs and output match
        let size_matches = (0..ndims).all(|i| x[i].len() == out.len());
        assert!(size_matches, "Dimension mismatch");

        let tmp = &mut [T::zero(); MAXDIMS][..ndims];
        (0..n).for_each(|i| {
            (0..ndims).for_each(|j| tmp[j] = x[j][i]);
            out[i] = self.interp_one(tmp);
        });
    }

    /// Interpolate the value at a point,
    /// using fixed-size intermediate storage of O(ndims) and no allocation.
    /// Assumes C-style ordering of vals ([x0, y0], [x0, y1], ..., [x0, yn], [x1, y0], ...).
    ///
    /// # Panics
    ///   * If the dimensionality of the point does not match the data
    ///   * If the dimensionality of either one exceeds the fixed maximum
    #[inline(always)]
    pub fn interp_one(&mut self, x: &[T]) -> T {
        // Check sizes
        let ndims = self.grids.len();
        assert!(x.len() == ndims, "Dimension mismatch");

        // Initialize fixed-size intermediate storage.
        // This storage _could_ be initialized with the interpolator struct, but
        // this would then require that every usage of struct be `mut`, which is
        // ergonomically unpleasant.
        let ioffs = &mut [false; MAXDIMS][..ndims]; // Offset index for selected vertex
        let sat = &mut [0_u8; MAXDIMS][..ndims]; // Saturated-low flag

        let dxs = &mut [T::zero(); MAXDIMS][..ndims]; // Sub-cell volume storage
        let extrapdxs = &mut [T::zero(); MAXDIMS][..ndims]; // Extrapolated distances
        let steps = &mut [T::zero(); MAXDIMS][..ndims]; // Step size on each dimension

        // Whether the opposite vertex is on the saturated bound
        // on each dimension
        let opsat = &mut [false; MAXDIMS][..ndims];

        // Whether the current vertex is on the saturated bound
        // on each dimension
        let thissat = &mut [false; MAXDIMS][..ndims];

        // Populate lower corner
        for i in 0..ndims {
            (self.origin[i], sat[i]) = self.get_loc(x[i], i, self.origin[i])
        }

        // Check if any dimension is saturated.
        // This gives a ~15% overall speedup for points on the interior.
        let any_dims_saturated = (0..ndims).any(|j| sat[j] != 0);

        // Calculate the total volume of this cell
        let cell_vol = self.get_cell(&self.origin, steps);

        // Traverse vertices, summing contributions to the interpolated value.
        //
        // This visits the 2^ndims elements of the cartesian product
        // of `[0, 1] x ... x [0, 1]` using O(ndims) storage.
        let mut interped = T::zero();
        let nverts = 2_usize.pow(ndims as u32);
        for i in 0..nverts {
            let mut k: usize = 0; // index of the value for this vertex in self.vals
            let mut sign = T::one(); // sign of the contribution from this vertex

            // Every 2^nth vertex, flip which side of the cube we are examining
            // in the nth dimension.
            //
            // Because i % 2^n has double the period for each sequential n,
            // and their phase is only aligned once every 2^n for the largest
            // n in the set, this is guaranteed to produce a path that visits
            // each vertex exactly once.
            for j in 0..ndims {
                let flip = i % 2_usize.pow(j as u32) == 0;
                if flip {
                    ioffs[j] = !ioffs[j];
                }
            }

            // Accumulate the index into the value array,
            // saturating to the bound if the resulting index would be outside.
            for j in 0..ndims {
                k += self.dimprod[j]
                    * (self.origin[j] + ioffs[j] as usize).min(self.dims[j].saturating_sub(1));
            }

            // Get the value at this vertex
            let v = self.vals[k];

            // Find the vector from the opposite vertex to the observation point
            for j in 0..ndims {
                let iloc =
                    (self.origin[j] + !ioffs[j] as usize).min(self.dims[j].saturating_sub(1)); // Index of location of opposite vertex
                let loc = self.grids[j][iloc]; // Loc. of opposite vertex
                dxs[j] = loc;
            }
            (0..ndims).for_each(|j| dxs[j] = x[j] - dxs[j]);
            (0..ndims).for_each(|j| dxs[j] = dxs[j].abs());

            // Accumulate contribution from this vertex
            if !any_dims_saturated {
                // Interpolating
                let vol = dxs.iter().fold(T::one(), |acc, x| acc * *x) * sign;
                interped = interped + v * vol;
            } else {
                // Extrapolating requires some special attention.

                // For which dimensions is the opposite vertex on a saturated bound?
                (0..ndims).for_each(|j| {
                    opsat[j] = (!ioffs[j] && sat[j] == 2) || (ioffs[j] && sat[j] == 1)
                });

                // For how many total dimensions is the opposite vertex on a saturated bound?
                let opsatcount = opsat.iter().fold(0, |acc, x| acc + *x as usize);

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

                // For which dimensions is the current vertex on a saturated bound?
                (0..ndims).for_each(|j| thissat[j] = sat[j] > 0 && !opsat[j]);

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
                    sign = sign.neg();
                    for j in 0..ndims {
                        if thissat[j] {
                            dxs[j] = dxs[j].min(steps[j].abs());
                        }
                    }

                    let vol = dxs.iter().fold(T::one(), |acc, x| acc * *x) * sign;
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
                (0..ndims).for_each(|j| extrapdxs[j] = dxs[j].min(steps[j].abs()));
                //   Get the volume of this region which does not extend outside the cell
                let vinterior = extrapdxs.iter().fold(T::one(), |acc, x| acc * *x);

                // Find each linear exterior region by, one at a time, taking the volume
                // with one extrapolated dimension masked into the extrapdxs
                // which are otherwise clipped to the interior region.
                let mut vexterior = T::zero();
                for j in 0..ndims {
                    if thissat[j] {
                        let dx_was = extrapdxs[j];
                        extrapdxs[j] = dxs[j] - steps[j].abs();
                        vexterior = vexterior + extrapdxs.iter().fold(T::one(), |acc, x| acc * *x);
                        extrapdxs[j] = dx_was; // Reset extrapdxs to original state for next calc
                    }
                }

                let vol = (vexterior + vinterior) * sign;

                interped = interped + v * vol;
            }
        }

        interped / cell_vol
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
    fn get_loc(&self, v: T, dim: usize, guess: usize) -> (usize, u8) {
        let grid = self.grids[dim];
        let saturation: u8; // Saturated low/high/not at all
        let iloc: isize; // Signed integer index location of this point

        // Bisection search to find location on the grid.
        //
        // The search will return `0` if the point is outside-low,
        // and will return `self.dims[dim]` if outside-high.
        //
        // We still need to convert to a signed integer here, because
        // if the grid has less than 2 points in this dimension,
        // we may need to (briefly) represent a negative index.
        //
        // This process accounts for essentially the entire difference in
        // performance between this method and the regular-grid method.
        //
        // First, try hard to avoid doing the bisection search at all
        // by checking for extrapolation and by keeping a rolling
        // initial guess that drastically improves perf for batch runs,
        // and by checking the next and previous index after the guess as well,
        // since usages that traverse from low to high or high to low indices
        // will often move from one index to its immediate neighbor.
        let guess_minus_one = guess.saturating_sub(1);
        let guess_plus_one = (guess + 1).min(grid.len() - 1);
        let guess_plus_two = (guess + 2).min(grid.len() - 1);
        // Check guess
        if grid[guess] < v && grid[guess_plus_one] >= v {
            iloc = guess as isize;
        }
        // Check next cell
        else if grid[guess_plus_one] < v && grid[guess_plus_two] >= v {
            iloc = guess_plus_one as isize;
        }
        // Check previous cell
        else if grid[guess_minus_one] < v && grid[guess] >= v {
            iloc = guess_minus_one as isize;
        }
        // Check for extrapolation below
        else if v < grid[0] {
            iloc = -1;
        }
        // Check for extrapolation above
        else if v > *grid.last().unwrap() {
            iloc = grid.len() as isize - 1
        }
        // If all else fails, do the actual binary search
        else {
            iloc = grid.partition_point(|x| *x < v) as isize - 1;
        }

        let dimmax = self.dims[dim].saturating_sub(2); // maximum index for lower corner

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
            let j = (origin[i] + 1).min(self.dims[i].saturating_sub(1));
            let mut dx = self.grids[i][j] - self.grids[i][origin[i]];

            // Clip degenerate dimensions to one to prevent crashing when a dimension has size one
            if j == 0 {
                dx = T::one();
            }

            steps[i] = dx;
        }

        let vol = steps.iter().fold(T::one(), |acc, x| acc * *x);

        vol
    }
}

/// Evaluate multilinear interpolation on a regular grid in up to 10 dimensions.
/// Assumes C-style ordering of vals ([x0, y0], [x0, y1], ..., [x0, yn], [x1, y0], ...).
///
/// This is a convenience function; best performance will be achieved by using the exact right
/// number for the MAXDIMS parameter, as this will slightly reduce compute and storage overhead,
/// and the underlying method can be extended to more than this function's limit of 8 dimensions.
/// The limit of 8 dimensions was chosen for no more specific reason than to reduce unit test times.
#[inline(always)]
pub fn interpn<'a, T>(grids: &'a [&'a [T]], vals: &'a [T], obs: &'a [&'a [T]], out: &'a mut [T])
where
    T: Float,
{
    RectilinearGridInterpolator::<'_, T, 8>::new(grids, vals).interp(obs, out);
}

#[cfg(test)]
mod test {
    use super::{interpn, RectilinearGridInterpolator};
    use crate::testing::*;
    use crate::utils::*;

    #[test]
    fn test_interp_extrap_2d_small() {
        // Test with one dimension that is minimum size
        let (nx, ny) = (3, 2);
        let x = linspace(-1.0, 1.0, nx);
        let y = Vec::from([0.5, 0.6]);
        let grids = [&x[..], &y[..]];
        let xy = meshgrid(Vec::from([&x, &y]));

        // z = x + y
        let z: Vec<f64> = (0..nx * ny).map(|i| &xy[i][0] + &xy[i][1]).collect();

        // Observation points all over in 2D space
        let xobs = linspace(-10.0_f64, 10.0, 37);
        let yobs = linspace(-10.0_f64, 10.0, 37);
        let xyobs = meshgrid(Vec::from([&xobs, &yobs]));
        let zobs: Vec<f64> = (0..37 * 37).map(|i| &xyobs[i][0] + &xyobs[i][1]).collect(); // Every `z` should match the degenerate `y` value

        let interpolator: &mut RectilinearGridInterpolator<'_, _, 2> =
            &mut RectilinearGridInterpolator::new(&grids, &z[..]);

        // Check values at every incident vertex
        xyobs.iter().zip(zobs.iter()).for_each(|(xyi, zi)| {
            let zii = interpolator.interp_one(&[xyi[0], xyi[1]]);
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
                    (0..x.len()).for_each(|i| x[i] = x[i] + (dx[i] - 0.5) / 1e3);
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
            interpn(&grids, &u, &xobsslice, &mut out[..]);

            // Check that interpolated values match expectation,
            // using an absolute difference because some points are very close to or exactly at zero,
            // and do not do well under a check on relative difference.
            (0..uobs.len()).for_each(|i| assert!((out[i] - uobs[i]).abs() < 1e-12));
        }
    }
}
