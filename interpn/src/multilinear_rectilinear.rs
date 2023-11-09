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

        // Populate lower corner
        for i in 0..ndims {
            (inds[i], sat[i]) = self.get_loc(x[i], i)
        }

        // Check if any dimension is saturated.
        // This gives a ~15% overall speedup for points on the interior.
        let maybe_neg = (0..ndims).any(|j| sat[j] != 0);

        // Calculate the total volume of this cell
        let cell_vol = self.get_vol(inds);

        // Traverse vertices, summing contributions to the interpolated value.
        //
        // This visits the 2^ndims elements of the cartesian product
        // of `[0, 1] x ... x [0, 1]` using O(ndims) storage.
        let mut interped = T::zero();
        let nverts = 2_usize.pow(ndims as u32);
        for i in 0..nverts {
            let mut k: usize = 0; // index of the value for this vertex in self.vals

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
            let mut vol = T::one();
            for j in 0..ndims {
                let iloc = inds[j] + !ioffs[j] as usize; // Index of location of opposite vertex
                let loc = self.grids[j][iloc]; // Loc. of opposite vertex
                let dx = x[j] - loc; // Delta position from opposite vertex to obs. loc
                vol = vol * dx;
            }

            // Determine the sign of the contribution.
            // For observation points outside the grid, negate the contribution
            // of the inner points on the dimensions that are saturated, in order
            // to naturally transition to extrapolation.
            // If there are any saturated points, check if the current vertex
            // is on a saturated dimension. If it is, return
            if maybe_neg {
                let neg =
                    (0..ndims).any(|j| (((!ioffs[j]) && sat[j] == 2) || (ioffs[j] && sat[j] == 1)));
                if neg {
                    interped = interped + v.neg() * vol.abs();
                    continue;
                }
            }

            // Add contribution from this vertex, leaving the division
            // by the total cell volume for later to save a few flops
            interped = interped + v * vol.abs();
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

        // Handle points outside the grid on the low side
        if iloc < 0 {
            loc = 0;
            saturation = 1;
        }
        // Handle points outside the grid on the high side
        // This is for the lower corner of the cell, so if we saturate high,
        // we have to return the value that is the next-most-interior
        else if iloc > (self.dims[dim] as isize - 2) {
            loc = iloc.min(self.dims[dim] as isize - 2).max(0) as usize;
            saturation = 2;
        }
        // Handle points on the interior.
        // These points may still saturate the index, which needs to be
        // farther inside than the last grid point for the lower corner,
        // but clipping to the most-inside point may, in turn, saturate
        // at the lower bound if the
        else {
            loc = iloc.min(self.dims[dim] as isize - 2).max(0) as usize;
            saturation = 0;
        }

        (loc, saturation)
    }

    /// Get the volume of the grid prism with `inds` as its lower corner
    #[inline(always)]
    fn get_vol(&self, inds: &[usize]) -> T {
        let mut vol = T::one();
        for i in 0..self.grids.len() {
            let j = (inds[i] + 1).min(self.dims[i] - 1).max(0); // Index of upper corner (saturating to bounds)
            let mut dx = self.grids[i][j] - self.grids[i][inds[i]];
            if dx == T::zero() {
                dx = T::one();
            }

            vol = vol * dx;
        }

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
        let (nx, ny) = (3, 4);
        let x = linspace(-1.0, 1.0, nx);
        let y = linspace(2.0, 4.0, ny);
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

        let x = linspace(0.0, 100.0, nx);
        let y = linspace(0.0, 100.0, ny);
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

        let x = linspace(0.0, 100.0, nx);
        let y = linspace(0.0, 100.0, ny);
        let z = randn::<f64>(&mut rng, n);
        let mut out = vec![0.0; n];

        let grid = meshgrid(Vec::from([&x, &y]));

        let xy: Vec<f64> = grid.iter().flatten().map(|xx| *xx).collect();

        interpn(&xy[..], &mut out, &z[..], &[&x, &y]);

        (0..n).for_each(|i| assert!((out[i] - z[i]).abs() < 1e-14)); // Allow small error at edges
    }
}
