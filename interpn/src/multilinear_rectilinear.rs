use num_traits::Float;

/// An arbitrary-dimensional multilinear interpolator on a rectilinear grid.
/// Assumes C-style ordering of vals (x0, y0, z0,   x0, y0, z1,   ...,   x0, yn, zn).
pub struct RectilinearGridInterpolator<'a, T: Float, const MAXDIMS: usize> {
    /// x, y, ... coordinate grids, size(dims.len()), each entry of size dims[i]
    grids: &'a [&'a [T]],

    /// Size of each dimension
    dims: [usize; MAXDIMS],

    /// Cumulative products of higher dimensions, used for indexing
    dimprod: [usize; MAXDIMS],

    /// Values at each point, size prod(dims)
    vals: &'a [T],
}

impl<'a, T, const MAXDIMS: usize> RectilinearGridInterpolator<'a, T, MAXDIMS>
where
    T: Float,
{
    /// Build a new interpolator, using O(MAXDIMS) calculations and storage.
    /// Assumes C-style ordering of vals ([x0, y0], [x0, y1], ..., [x0, yn], [x1, y0], ...).
    #[inline(always)]
    pub fn new(vals: &'a [T], grids: &'a [&'a [T]]) -> Self {
        // Check dimensions
        let ndims = grids.len();
        let mut dims = [1_usize; MAXDIMS];
        (0..ndims).for_each(|i| dims[i] = grids[i].len());
        let nvals = dims[..ndims].iter().fold(1, |acc, x| acc * x);
        assert!(vals.len() == nvals && ndims > 0, "Dimension mismatch");

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

        Self {
            grids,
            dims,
            dimprod,
            vals,
        }
    }

    /// Interpolate on interleaved list of points.
    /// Assumes C-style ordering of points ([x0, y0], [x0, y1], ..., [x0, yn], [x1, y0], ...).
    #[inline(always)]
    pub fn interp(&self, x: &[T], out: &mut [T]) {
        let n = out.len();
        let ndims = self.grids.len();
        assert!(x.len() % ndims == 0, "Dimension mismatch");

        let mut start = 0;
        let mut end = 0;
        (0..n).for_each(|i| {
            end = start + ndims;
            out[i] = self.interp_one(&x[start..end]);
            start = end;
        });
    }

    /// Interpolate the value at a point,
    /// using fixed-size intermediate storage of O(ndims) and no allocation.
    /// Assumes C-style ordering of vals ([x0, y0], [x0, y1], ..., [x0, yn], [x1, y0], ...).
    ///
    /// # Panics
    ///   * If the dimensionality of the point does not match the data
    ///   * If the dimensionality of either one exceeds the fixed maximum
    ///   * If the index along any dimension exceeds the maximum representable
    ///     integer value within the value type `T`
    #[inline(always)]
    pub fn interp_one(&self, x: &[T]) -> T {
        // Check sizes
        let ndims = self.grids.len();
        assert!(x.len() == ndims && ndims <= MAXDIMS, "Dimension mismatch");

        // Initialize fixed-size intermediate storage.
        // This storage _could_ be initialized with the interpolator struct, but
        // this would then require that every usage of struct be `mut`, which is
        // ergonomically unpleasant.
        let inds = &mut [0_usize; MAXDIMS][..ndims]; // Indices of lower corner of hypercube
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
            (inds[i], sat[i]) = self.get_loc(x[i], i)
        }

        // Check if any dimension is saturated.
        // This gives a ~15% overall speedup for points on the interior.
        let any_dims_saturated = (0..ndims).any(|j| sat[j] != 0);

        // Calculate the total volume of this cell
        let cell_vol = self.get_cell(inds, steps);

        // Traverse vertices, summing contributions to the interpolated value.
        //
        // This visits the 2^ndims elements of the cartesian product
        // of `[0, 1] x ... x [0, 1]` using O(ndims) storage.
        let mut interped = T::zero();
        let nverts = 2_usize.pow(ndims as u32);
        for i in 0..nverts {
            let mut k: usize = 0; // index of the value for this vertex in self.vals
            let mut sign = T::one(); // sign of the contribution from this vertex
            let mut extrapvol = T::zero();

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
                k += self.dimprod[j] * (inds[j] + ioffs[j] as usize).min(self.dims[j] - 1);
            }

            // Get the value at this vertex
            let v = self.vals[k];

            // Accumulate the volume of the prism formed by the
            // observation location and the opposing vertex
            for j in 0..ndims {
                let iloc = inds[j] + !ioffs[j] as usize; // Index of location of opposite vertex
                let loc = self.grids[j][iloc]; // Loc. of opposite vertex
                dxs[j] = loc;
            }

            for j in 0..ndims {
                dxs[j] = x[j] - dxs[j]; // Make the actual delta-locs
            }

            for j in 0..ndims {
                dxs[j] = dxs[j].abs();
            }

            // Clip maximum dx for some cases to handle multidimensional extrapolation
            if any_dims_saturated {
                // For which dimensions is the opposite vertex on a saturated bound?
                (0..ndims).for_each(|j| {
                    opsat[j] = (!ioffs[j] && sat[j] == 2) || (ioffs[j] && sat[j] == 1)
                });

                // For how many total dimensions is the opposite vertex on a saturated bound?
                let opsatcount = opsat.iter().fold(0, |acc, x| acc + *x as usize);

                // If the opposite vertex is on exactly one saturated bound, negate its contribution
                let neg = opsatcount == 1;
                if neg {
                    sign = sign.neg();
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
                    sign = T::zero();
                }

                // If the opposite vertex is on exactly one saturated bound,
                // allow the dx on that dimension to be as large as needed,
                // but clip the dx on other saturated dimensions so that we
                // don't produce an overlapping partition in outside-corner regions.
                if neg {
                    for j in 0..ndims {
                        let is_saturated = sat[j] != 0;
                        if is_saturated && !opsat[j] {
                            dxs[j] = dxs[j].min(steps[j]);
                        }
                    }
                }

                // For which dimensions is the current vertex on a saturated bound?
                (0..ndims).for_each(|j| {
                    thissat[j] = (ioffs[j] && sat[j] == 2) || (!ioffs[j] && sat[j] == 1)
                });

                // For how many total dimensions is the current vertex on a saturated bound?
                let thissatcount = thissat.iter().fold(0, |acc, x| acc + *x as usize);

                // Subtract the extrapolated volume from the contribution for this vertex
                // if it is on multiple saturated bounds.
                // Put differently - find the part of the volume that is scaling non-linearly
                // in the coordinates, and bookkeep it to be removed entirely later.
                if thissatcount > 1 {
                    // Copy forward the original dxs, extrapolated or not
                    (0..ndims).for_each(|j| extrapdxs[j] = dxs[j]);
                    // For extrapolated dimensions, take just the extrapolated distance
                    (0..ndims).for_each(|j| {
                        if thissat[j] {
                            extrapdxs[j] = dxs[j] - steps[j]
                        }
                    });
                    // Evaluate the extrapolated corner volume
                    extrapvol = extrapdxs.iter().fold(T::one(), |acc, x| acc * *x);
                }
            }

            let vol = (dxs.iter().fold(T::one(), |acc, x| acc * *x).abs() - extrapvol) * sign;

            // Add contribution from this vertex, leaving the division
            // by the total cell volume for later to save a few flops
            interped = interped + v * vol;
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
    /// Unfortunately, using a repr(u8) enum for the saturation flag is a >10% perf hit.
    #[inline(always)]
    fn get_loc(&self, v: T, dim: usize) -> (usize, u8) {
        let loc: usize; // Lower corner index
        let saturation: u8; // Saturated low/high/not at all

        // Bisection search to find location on the grid
        // The search will return `0` if the point is outside-low,
        // and will return `self.dims[dim]` if outside-high.
        // We still need to convert to a signed integer here, because
        // if the grid has less than 2 points in this dimension,
        // we may need to (briefly) represent a negative index.
        let iloc = self.grids[dim].partition_point(|x| *x < v) as isize - 1;

        let dimmax = self.dims[dim].saturating_sub(2); // maximum index for lower corner

        loc = (iloc.max(0) as usize).min(dimmax); // unsigned integer loc clipped to interior

        // Handle points outside the grid on the low side
        if iloc < 0 {
            saturation = 1;
        }
        // Handle points outside the grid on the high side
        // This is for the lower corner of the cell, so if we saturate high,
        // we have to return the value that is the next-most-interior
        else if iloc > dimmax as isize {
            saturation = 2;
        }
        // Handle points on the interior.
        // These points may still saturate the index, which needs to be
        // farther inside than the last grid point for the lower corner,
        // but clipping to the most-inside point may, in turn, saturate
        // at the lower bound if the
        else {
            saturation = 0;
        }

        (loc, saturation)
    }

    /// Get the volume of the grid prism with `inds` as its lower corner
    /// and output the step sizes for this cell as well.
    #[inline(always)]
    fn get_cell(&self, inds: &[usize], steps: &mut [T]) -> T {
        let ndims = self.grids.len();
        for i in 0..ndims {
            // Index of upper face (saturating to bounds)
            let j = (inds[i] + 1).min(self.dims[i].saturating_sub(1));
            let mut dx = self.grids[i][j] - self.grids[i][inds[i]];

            // Clip degenerate dimensions to one to prevent crashing when a dimension has size one
            if j == inds[i] {
                dx = T::one();
            }

            steps[i] = dx;
        }

        let vol = steps.iter().fold(T::one(), |acc, x| acc * *x);

        return vol;
    }
}

/// Initialize and evaluate multilinear interpolation on a rectilinear grid in up to 10 dimensions.
/// Assumes C-style ordering of vals ([x0, y0], [x0, y1], ..., [x0, yn], [x1, y0], ...).
///
/// This is a convenience function; best performance will be achieved by using the exact right
/// number for the MAXDIMS parameter, as this will slightly reduce compute and storage overhead,
/// and the underlying method can be extended to more than this function's limit of 10 dimensions.
#[inline(always)]
pub fn interpn<'a>(x: &'a [f64], out: &'a mut [f64], vals: &'a [f64], grids: &'a [&'a [f64]]) {
    let interpolator: RectilinearGridInterpolator<'_, _, 10> =
        RectilinearGridInterpolator::new(vals, grids);
    interpolator.interp(x, out);
}

#[cfg(test)]
mod test {
    use super::{interpn, RectilinearGridInterpolator};
    use crate::testing::*;
    use crate::utils::*;

    #[test]
    fn test_interp_one_2d() {
        let mut rng = rng_fixed_seed();

        let (nx, ny) = (3, 4);
        let mut x = linspace(-1.0, 1.0, nx);
        let mut y = linspace(2.0, 4.0, ny);

        // Add noise to the grid
        let dx = randn::<f64>(&mut rng, nx);
        let dy = randn::<f64>(&mut rng, ny);
        (0..nx).for_each(|i| x[i] = x[i] + (dx[i] - 0.5) / 100.0);
        (0..ny).for_each(|i| y[i] = y[i] + (dy[i] - 0.5) / 100.0);

        // Make sure the grid is still monotonic
        (0..nx - 1).for_each(|i| assert!(x[i + 1] > x[i]));
        (0..ny - 1).for_each(|i| assert!(y[i + 1] > y[i]));

        let xy = meshgrid(Vec::from([&x, &y]));

        // z = x * y^2
        let z: Vec<f64> = (0..nx * ny)
            .map(|i| &xy[i][0] * &xy[i][1] * &xy[i][1])
            .collect();

        let grids = [&x[..], &y[..]];

        let interpolator: RectilinearGridInterpolator<'_, _, 2> =
            RectilinearGridInterpolator::new(&z[..], &grids);

        // Check values at every incident vertex
        xy.iter().zip(z.iter()).for_each(|(xyi, zi)| {
            let zii = interpolator.interp_one(&[xyi[0], xyi[1]]);
            assert!((*zi - zii).abs() < 1e-12) // Allow small error at edges
        });
    }

    #[test]
    fn test_interp_interleaved_2d() {
        let mut rng = rng_fixed_seed();
        let m: usize = (100 as f64).sqrt() as usize;
        let nx = m / 2;
        let ny = m * 2;
        let n = nx * ny;

        let mut x = linspace(0.0, 100.0, nx);
        let mut y = linspace(0.0, 100.0, ny);

        // Add noise to the grid
        let dx = randn::<f64>(&mut rng, nx);
        let dy = randn::<f64>(&mut rng, ny);
        (0..nx).for_each(|i| x[i] = x[i] + (dx[i] - 0.5) / 100.0);
        (0..ny).for_each(|i| y[i] = y[i] + (dy[i] - 0.5) / 100.0);

        // Make sure the grid is still monotonic
        (0..nx - 1).for_each(|i| assert!(x[i + 1] > x[i]));
        (0..ny - 1).for_each(|i| assert!(y[i + 1] > y[i]));

        let z = randn::<f64>(&mut rng, n);
        let mut out = vec![0.0; n];

        let grid = meshgrid(Vec::from([&x, &y]));
        let xy: Vec<f64> = grid.iter().flatten().map(|xx| *xx).collect();

        let grids = [&x[..], &y[..]];

        let interpolator: RectilinearGridInterpolator<'_, _, 2> =
            RectilinearGridInterpolator::new(&z[..], &grids);

        interpolator.interp(&xy[..], &mut out);

        (0..n).for_each(|i| assert!((out[i] - z[i]).abs() < 1e-14)); // Allow small error at edges
    }

    #[test]
    fn test_interpn_2d() {
        let mut rng = rng_fixed_seed();
        let m: usize = (100 as f64).sqrt() as usize;
        let nx = m / 2;
        let ny = m * 2;
        let n = nx * ny;

        let mut x = linspace(0.0, 100.0, nx);
        let mut y = linspace(0.0, 100.0, ny);

        // Add noise to the grid
        let dx = randn::<f64>(&mut rng, nx);
        let dy = randn::<f64>(&mut rng, ny);
        (0..nx).for_each(|i| x[i] = x[i] + (dx[i] - 0.5) / 100.0);
        (0..ny).for_each(|i| y[i] = y[i] + (dy[i] - 0.5) / 100.0);

        // Make sure the grid is still monotonic
        (0..nx - 1).for_each(|i| assert!(x[i + 1] > x[i]));
        (0..ny - 1).for_each(|i| assert!(y[i + 1] > y[i]));

        let z = randn::<f64>(&mut rng, n);
        let mut out = vec![0.0; n];

        let grid = meshgrid(Vec::from([&x, &y]));

        let xy: Vec<f64> = grid.iter().flatten().map(|xx| *xx).collect();

        interpn(&xy, &mut out, &z, &[&x, &y]);

        (0..n).for_each(|i| assert!((out[i] - z[i]).abs() < 1e-14)); // Allow small error at edges
    }

    #[test]
    fn test_extrap_2d() {
        let mut rng = rng_fixed_seed();

        let m: usize = (100 as f64).sqrt() as usize;
        let nx = m / 2;
        let ny = m * 2;

        let mut x = linspace(0.0, 10.0, nx);
        let mut y = linspace(-5.0, 5.0, ny);

        // Add noise to the grid
        let dx = randn::<f64>(&mut rng, nx);
        let dy = randn::<f64>(&mut rng, ny);
        (0..nx).for_each(|i| x[i] = x[i] + (dx[i] - 0.5) / 1e1);
        (0..ny).for_each(|i| y[i] = y[i] + (dy[i] - 0.5) / 1e1);

        // Make sure the grid is still monotonic
        (0..nx - 1).for_each(|i| assert!(x[i + 1] > x[i]));
        (0..ny - 1).for_each(|i| assert!(y[i + 1] > y[i]));

        let grid = meshgrid(Vec::from([&x, &y]));
        let grids = &[&x[..], &y[..]];

        //   High/low corners and all over the place
        //   For this one, use a function that is linear in every direction,
        //   z = x + y,
        //   so that it will be extrapolated correctly in the corner regions
        let xw = linspace(-10.0, 11.0, 200);
        let yw = linspace(-7.0, 6.0, 200);
        let xyw: Vec<f64> = meshgrid(vec![&xw, &yw])
            .iter()
            .flatten()
            .map(|xx| *xx)
            .collect();

        let zw: Vec<f64> = (0..xyw.len() / 2)
            .map(|i| xyw[2 * i] + xyw[2 * i + 1])
            .collect();
        let zgrid1: Vec<f64> = grid.iter().map(|xyi| xyi[0] + xyi[1]).collect();

        let mut out = vec![0.0; nx.max(ny).max(zw.len())];

        // Check extrapolating off grid and interpolating between grid points all around
        interpn(&xyw, &mut out[..zw.len()], &zgrid1, grids);
        (0..zw.len()).for_each(|i| assert!((out[i] - zw[i]).abs() < 1e-12));
    }
}
