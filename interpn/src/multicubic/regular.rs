use num_traits::{Float, NumCast};

#[derive(Clone, Copy, PartialEq)]
enum Saturation {
    None,
    InsideLow,
    OutsideLow,
    InsideHigh,
    OutsideHigh,
}

pub struct MulticubicRegular<'a, T: Float, const MAXDIMS: usize> {
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

impl<'a, T: Float, const MAXDIMS: usize> MulticubicRegular<'a, T, MAXDIMS> {
    /// Build a new interpolator, using O(MAXDIMS) calculations and storage.
    ///
    /// This method does not handle degenerate dimensions with only a single
    /// grid entry; all grids must have at least 4 entries.
    ///
    /// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
    ///
    /// # Errors
    /// * If any input dimensions do not match
    /// * If any dimensions have size < 4
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

        // Make sure all dimensions have at least four entries
        let degenerate = dims[..ndims].iter().any(|&x| x < 4);
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
        let origin = &mut [0_usize; MAXDIMS][..ndims]; // Indices of lower corner of hypercub
        let sat = &mut [Saturation::None; MAXDIMS][..ndims]; // Saturation none/high/low flags for each dim
        let dts = &mut [T::zero(); MAXDIMS][..ndims]; // Normalized coordinate storage
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

        // Populate lower corner and saturation flag for each dimension
        for i in 0..ndims {
            (origin[i], sat[i]) = self.get_loc(x[i], i)?;
        }

        // Calculate normalized delta locations
        for i in 0..ndims {
            let index_one_loc = self.starts[i]
                + self.steps[i]
                    * <T as NumCast>::from(origin[i] + 1).ok_or("Unrepresentable coordinate value")?;
            dts[i] = (x[i] - index_one_loc) / self.steps[i];
        }

        // Recursive interpolation of one dependency tree at a time
        let loc = &origin;  // Starting location in the tree is the origin
        let dim =self.dims[ndims - 1];  // Start from the end and recurse back to zero
        let ind = 0;  // Start from the zero-index
        let interped = self.populate(ind, dim, sat, origin, loc, dimprod, dts);

        Ok(interped)
    }

    /// Get the two-lower index along this dimension where `x` is found,
    /// saturating to the bounds at the edges if necessary.
    ///
    /// At the high bound of a given dimension, saturates to the fourth internal
    /// point in order to capture a full 4-cube.
    ///
    /// Returned value like (lower_corner_index, saturation_flag).
    #[inline(always)]
    fn get_loc(&self, v: T, dim: usize) -> Result<(usize, Saturation), &'static str> {
        let saturation: Saturation; // What part of the grid cell are we in?

        let floc = ((v - self.starts[dim]) / self.steps[dim]).floor(); // float loc
        let iloc = <isize as NumCast>::from(floc); // signed integer loc

        let n = self.dims[dim]; // Number of grid points on this dimension
        match iloc {
            Some(iloc) => {
                let dimmax = n - 4; // maximum index for lower corner
                let loc: usize = (iloc.max(0) as usize).min(dimmax); // unsigned integer loc clipped to interior

                // Observation point is outside the grid on the low side
                if iloc < 0 {
                    saturation = Saturation::OutsideLow;
                }
                // Observation point is in the lower part of the cell
                // but not outside the grid
                else if iloc < 1 {
                    saturation = Saturation::InsideLow;
                }
                // Observation point is in the upper part of the cell
                // but not outside the grid
                else if iloc > (n - 1) as isize {
                    saturation = Saturation::OutsideHigh;
                }
                // Observation point is in the upper part of the cell
                // but not outside the grid
                else if iloc > (n - 2) as isize {
                    saturation = Saturation::InsideHigh;
                }
                // Observation point is on the interior
                else {
                    saturation = Saturation::None;
                }

                Ok((loc, saturation))
            }
            None => Err("Unrepresentable coordinate value"),
        }
    }

    /// Recursive evaluation of interpolant on each dimension
    #[inline]
    fn populate(
        &self,
        ind: usize,
        dim: usize,
        sat: &[Saturation],
        origin: &[usize],
        loc: &[usize],
        dimprod: &[usize],
        dts: &[T],
    ) -> T {
        // Keep track of where we are in the tree
        // so that we can index into the value array properly
        // when we reach the leaves
        let ndims = loc.len();
        let thisloc = &mut [0_usize; MAXDIMS][..ndims];
        thisloc.copy_from_slice(loc);
        thisloc[dim] = origin[dim] + ind;

        // Do the calc for this entry
        if dim == 0 {
            // If we have arrived at a leaf, index into data
            let v = index_arr(thisloc, dimprod, self.vals);
            return v;
        } else {
            // Populate next dim's values
            let next_dim = dim - 1;
            let mut vals = [T::zero(); 4];
            for i in 0..4 {
                vals[i] = self.populate(i, next_dim, sat, origin, thisloc, dimprod, dts);
            }
            // Interpolate on next dim's values to populate an entry in this dim
            let v = interp_inner::<T, MAXDIMS>(vals, dts[dim], sat[dim]);
            return v;
        }
    }
}

/// Calculate slopes and offsets & select evaluation method
#[inline]
fn interp_inner<T: Float, const MAXDIMS: usize>(vals: [T; 4], t: T, sat: Saturation) -> T {
    // Construct some constants using generic methods
    let one = T::one();
    let two = one + one;

    let dx = one; // Normalized coordinates, regular grid
    match sat {
        Saturation::None => {
            // This is the nominal case
            let y0 = vals[1];
            let dy = vals[2] - vals[1];

            // Take slopes from centered difference
            let k0 = (vals[2] - vals[0]) / two;
            let k1 = (vals[3] - vals[1]) / two;

            hermite_spline(t, y0, dx, dy, k0, k1)
        }
        Saturation::InsideLow => {
            // Flip direction to maintain symmetry
            // with the InsideHigh case
            let t = -t; // `t` always w.r.t. index 1 of cube
            let y0 = vals[1]; // Same starting point, opposite direction
            let dy = vals[0] - vals[1];

            // Take one backward difference and one centered,
            // in the opposite direction
            let k0 = -(vals[2] - vals[0]) / two;
            let k1 = -(vals[1] - vals[0]);

            hermite_spline(t, y0, dx, dy, k0, k1)
        }
        Saturation::OutsideLow => {
            // Fall back on linear extrapolation
            // `t` is already negative, since it's calculated
            // w.r.t. index 1, but is off by a normalized cell (1.0)
            let t = t + one;
            let y0 = vals[0];
            let dy = vals[1] - vals[0];

            // Since the grid is regular and `t` is normalized,
            // dx = 1 -> dy/dx = dy
            let k0 = dy; // Just to be explicit

            y0 + k0 * t // `t` is negative
        }
        Saturation::InsideHigh => {
            // Shift cell up an index
            // and offset `t`, which has value between 1 and 2
            // because it is calculated w.r.t. index 1
            let t = t - one;
            let y0 = vals[2];
            let dy = vals[3] - vals[2];

            // Take one backward difference and one centered
            let k0 = (vals[3] - vals[1]) / two;
            let k1 = vals[3] - vals[2];

            hermite_spline(t, y0, dx, dy, k0, k1)
        }
        Saturation::OutsideHigh => {
            // Fall back on linear extrapolation
            // and offset `t`, which has value relative to index 1
            // but will be used relative to index 3
            let t = t - two;
            let y0 = vals[3];
            let dy = vals[3] - vals[2];

            // Since the grid is regular and `t` is normalized,
            // dx = 1 -> dy/dx = dy
            let k0 = dy; // Just to be explicit

            y0 + k0 * t
        }
    }
}

/// Evaluate a hermite spline function on an interval from x0 to x1,
/// with imposed slopes k0 and k1 at the endpoints, and normalized
/// coordinate t = (x - x0) / (x1 - x0)
#[inline]
fn hermite_spline<T: Float>(t: T, y0: T, dx: T, dy: T, k0: T, k1: T) -> T {
    // `a` and `b` are difference between this function and a linear one going
    // forward or backward with the imposed slopes.
    let a = k0 * dx - dy;
    let b = -k1 * dx + dy;

    let t2 = t * t;
    let t3 = t * t * t;

    let c1 = dy + a;
    let c2 = b - (a + a);
    let c3 = a - b;

    y0 + (c1 * t) + (c2 * t2) + (c3 * t3)
}

/// Index a single value from an array
#[inline]
fn index_arr<T: Copy>(loc: &[usize], dimprod: &[usize], data: &[T]) -> T {
    let mut i = 0;
    for j in 0..dimprod.len() {
        i += loc[j] * dimprod[j];
    }

    return data[i];
}
