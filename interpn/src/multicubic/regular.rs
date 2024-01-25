use num_traits::{Float, NumCast};
use rand::seq::index;

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
        let steps = &self.steps[..ndims]; // Step size for each dimension
        let origin = &mut [0_usize; MAXDIMS][..ndims]; // Indices of lower corner of hypercube
        let ioffs = &mut [false; MAXDIMS][..ndims]; // Offset index for selected vertex
        let sat = &mut [Saturation::None; MAXDIMS][..ndims]; // Saturation none/high/low flags for each dim
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

        // Populate lower corner and saturation flag for each dimension
        for i in 0..ndims {
            (origin[i], sat[i]) = self.get_loc(x[i], i)?;
        }

        // Check if any dimension is saturated.
        let any_dims_saturated = sat.iter().any(|&x| x != Saturation::None);

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

        }

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

    #[inline]
    fn populate(&self, ind: usize, dim: usize, sat: &[Saturation], loc: &[usize], dimprod: &[usize], dts: &[T]) -> T {
        let ndims = loc.len();
        if dim == 0 {
            // Index into data
            let thisloc = &mut [0_usize; MAXDIMS][..ndims];
            thisloc[0] += ind;
            let v = index_arr(thisloc, dimprod, self.vals);
            return v
        } else {
            // Populate next dim and interpolate
            let mut vals = [T::zero(); 4];
            for i in 0..4 {
                vals[i] = self.populate(i, dim-1, sat, loc, dimprod, dts);
            }

            let v = interp_inner::<T, MAXDIMS>(vals, dts[dim], sat[dim]);
            return v
        }
    }

}


#[inline]
fn interp_inner<T: Float, const MAXDIMS: usize>(vals: [T; 4], t: T, sat: Saturation) -> T {
    match sat {

    }
}

/// Index a single value from an array
#[inline]
fn index_arr<T: Copy>(loc: &[usize], dimprod: &[usize], data: &[T]) -> T {
    let mut i = 0;
    for j in 0..dimprod.len() {
        i += loc[j] * dimprod[j];
    }

    return data[i]
}