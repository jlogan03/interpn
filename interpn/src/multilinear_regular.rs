use num_traits::{Float, NumCast};

/// An arbitrary-dimensional multilinear interpolator on a regular grid.
/// Assumes C-style ordering of vals (x0, y0, z0,   x0, y0, z1,   ...,   x0, yn, zn).
pub struct RegularGridInterpolator<'a, T: Float, const MAXDIMS: usize> {
    /// Size of each dimension
    dims: &'a [usize],

    /// Starting point of each dimension, size dims.len()
    starts: &'a [T],

    /// Step size for each dimension, size dims.len()
    steps: &'a [T],

    /// Volume of cell (cumulative product of steps)
    vol: T,

    /// Values at each point, size prod(dims)
    vals: &'a [T],

    /// Cumulative products of higher dimensions, used for indexing
    dimprod: [usize; MAXDIMS],
}

impl<'a, T, const MAXDIMS: usize> RegularGridInterpolator<'a, T, MAXDIMS>
where
    T: Float,
{
    /// Build a new interpolator, using O(MAXDIMS) calculations and storage.
    /// Assumes C-style ordering of vals ([x0, y0], [x0, y1], ..., [x0, yn], [x1, y0], ...).
    #[inline(always)]
    pub fn new(vals: &'a [T], dims: &'a [usize], starts: &'a [T], steps: &'a [T]) -> Self {
        // Check dimensions
        let ndims = dims.len();
        let nvals = dims.iter().fold(1, |acc, x| acc * x);
        assert!(
            starts.len() == ndims && steps.len() == ndims && vals.len() == nvals && ndims > 0,
            "Dimension mismatch"
        );

        // Compute volume of reference cell
        let vol = steps.iter().fold(T::one(), |acc, x| acc * *x).abs();

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
            dims,
            starts,
            steps,
            vals,
            vol,
            dimprod,
        }
    }

    /// Interpolate on interleaved list of points.
    /// Assumes C-style ordering of points ([x0, y0], [x0, y1], ..., [x0, yn], [x1, y0], ...).
    #[inline(always)]
    pub fn interp(&self, x: &[T], out: &mut [T]) {
        let ndims = self.dims.len();
        assert!(
            x.len() % ndims == 0 && x.len() / ndims == out.len(),
            "Dimension mismatch"
        );

        let mut start = 0;
        let mut end = 0;
        (0..out.len()).for_each(|i| {
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
        let ndims = self.dims.len();
        assert!(x.len() == ndims && ndims <= MAXDIMS, "Dimension mismatch");

        // Initialize fixed-size intermediate storage.
        // This storage _could_ be initialized with the interpolator struct, but
        // this would then require that every usage of struct be `mut`, which is
        // ergonomically unpleasant.
        // Also notably, storing the index offsets as bool instead of usize
        // reduces memory overhead, but has not effect on throughput rate.
        let origin = &mut [0_usize; MAXDIMS][..ndims]; // Indices of lower corner of hypercube
        let ioffs = &mut [false; MAXDIMS][..ndims]; // Offset index for selected vertex
        let sat = &mut [0_u8; MAXDIMS][..ndims]; // Saturation none/high/low flags for each dim

        let dxs = &mut [T::zero(); MAXDIMS][..ndims]; // Sub-cell volume storage
        let extrapdxs = &mut [T::zero(); MAXDIMS][..ndims]; // Extrapolated distances

        // Whether the opposite vertex is on the saturated bound
        // on each dimension
        let opsat = &mut [false; MAXDIMS][..ndims];

        // Whether the current vertex is on the saturated bound
        // on each dimension
        let thissat = &mut [false; MAXDIMS][..ndims];

        // Populate lower corner and saturation flag for each dimension
        for i in 0..ndims {
            (origin[i], sat[i]) = self.get_loc(x[i], i)
        }

        // Check if any dimension is saturated.
        // This gives a ~15% overall speedup for points on the interior.
        let any_dims_saturated = (0..ndims).any(|j| sat[j] != 0);

        // Traverse vertices, summing contributions to the interpolated value.
        //
        // This visits the 2^ndims elements of the cartesian product
        // of `[0, 1] x ... x [0, 1]` without simultaneously actualizing them in storage.
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
                k += self.dimprod[j]
                    * (origin[j] + ioffs[j] as usize).min(self.dims[j].saturating_sub(1));
            }

            // Get the value at this vertex
            let v = self.vals[k];

            // Accumulate the volume of the prism formed by the
            // observation location and the opposing vertex
            for j in 0..ndims {
                let iloc = origin[j] + !ioffs[j] as usize; // Index of location of opposite vertex
                let loc = self.starts[j] + self.steps[j] * T::from(iloc).unwrap(); // Loc. of opposite vertex
                dxs[j] = loc; // Use dxs[j] as storage for float locs
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
                            dxs[j] = dxs[j].min(self.steps[j]);
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
                            extrapdxs[j] = dxs[j] - self.steps[j]
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

        interped / self.vol
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

        let floc = ((v - self.starts[dim]) / self.steps[dim]).floor(); // float loc
        let iloc: isize = <isize as NumCast>::from(floc).unwrap(); // signed integer loc

        let dimmax = self.dims[dim].saturating_sub(2); // maximum index for lower corner

        loc = (iloc.max(0) as usize).min(dimmax); // unsigned integer loc clipped to interior

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
}

/// Evaluate multilinear interpolation on a regular grid in up to 10 dimensions.
/// Assumes C-style ordering of vals ([x0, y0], [x0, y1], ..., [x0, yn], [x1, y0], ...).
///
/// This is a convenience function; best performance will be achieved by using the exact right
/// number for the MAXDIMS parameter, as this will slightly reduce compute and storage overhead,
/// and the underlying method can be extended to more than this function's limit of 10 dimensions.
#[inline(always)]
pub fn interpn(
    x: &[f64],
    out: &mut [f64],
    vals: &[f64],
    dims: &[usize],
    starts: &[f64],
    steps: &[f64],
) {
    // Initialization is fairly cheap in most cases (O(ndim) int muls) so unless we're
    // repetitively using this to interpolate single points, we probably won't notice
    // the little bit of extra overhead.
    RegularGridInterpolator::<'_, _, 10>::new(vals, dims, starts, steps).interp(x, out);
}

#[cfg(test)]
mod test {
    use super::{interpn, RegularGridInterpolator};
    use crate::testing::*;
    use crate::utils::*;

    #[test]
    fn test_interp_one_2d() {
        let (nx, ny) = (3, 4);
        let x = linspace(-1.0, 1.0, nx);
        let y = linspace(2.0, 4.0, ny);
        let xy = meshgrid(Vec::from([&x, &y]));

        // z = x * y^2
        let z: Vec<f64> = (0..nx * ny)
            .map(|i| &xy[i][0] * &xy[i][1] * &xy[i][1])
            .collect();

        let dims = [nx, ny];
        let starts = [x[0], y[0]];
        let steps = [x[1] - x[0], y[1] - y[0]];
        let interpolator: RegularGridInterpolator<'_, _, 2> =
            RegularGridInterpolator::new(&z, &dims, &starts, &steps);

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

        let x = linspace(0.0, 100.0, nx);
        let y = linspace(0.0, 100.0, ny);
        let z = randn::<f64>(&mut rng, n);
        let mut out = vec![0.0; n];

        let grid = meshgrid(Vec::from([&x, &y]));
        let xy: Vec<f64> = grid.iter().flatten().map(|xx| *xx).collect();

        let dims = [nx, ny];
        let starts = [x[0], y[0]];
        let steps = [x[1] - x[0], y[1] - y[0]];

        let interpolator: RegularGridInterpolator<'_, _, 2> =
            RegularGridInterpolator::new(&z, &dims, &starts, &steps);

        interpolator.interp(&xy, &mut out);

        (0..n).for_each(|i| assert!((out[i] - z[i]).abs() < 1e-14)); // Allow small error at edges
    }

    #[test]
    fn test_interpn_2d() {
        let mut rng = rng_fixed_seed();
        let m: usize = (100 as f64).sqrt() as usize;
        let nx = m / 2;
        let ny = m * 2;
        let n = nx * ny;

        let x = linspace(0.0, 100.0, nx);
        let y = linspace(0.0, 100.0, ny);
        let z = randn::<f64>(&mut rng, n);
        let mut out = vec![0.0; n];

        let grid = meshgrid(Vec::from([&x, &y]));

        let xy: Vec<f64> = grid.iter().flatten().map(|xx| *xx).collect();

        let dims = [nx, ny];
        let starts = [x[0], y[0]];
        let steps = [x[1] - x[0], y[1] - y[0]];

        interpn(&xy, &mut out, &z, &dims, &starts, &steps);
        (0..n).for_each(|i| assert!((out[i] - z[i]).abs() < 1e-14));
    }

    #[test]
    fn test_extrap_2d() {
        let m: usize = (100 as f64).sqrt() as usize;
        let nx = m / 2;
        let ny = m * 2;

        let x = linspace(0.0, 10.0, nx);
        let y = linspace(-5.0, 5.0, ny);

        let grid = meshgrid(Vec::from([&x, &y]));

        let z: Vec<f64> = grid.iter().map(|xyi| xyi[0] + xyi[1]).collect();

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

        let mut out = vec![0.0; nx.max(ny).max(zw.len())];

        let dims = [nx, ny];
        let starts = [x[0], y[0]];
        let steps = [x[1] - x[0], y[1] - y[0]];

        // Check extrapolating off grid and interpolating between grid points all around
        interpn(&xyw, &mut out[..zw.len()], &z, &dims, &starts, &steps);
        (0..zw.len()).for_each(|i| assert!((out[i] - zw[i]).abs() < 1e-12));
    }
}
