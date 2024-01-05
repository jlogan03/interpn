use num_traits::{Float, NumCast};

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
}
