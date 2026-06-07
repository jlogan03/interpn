//! An arbitrary-dimensional cubic B-spline interpolator / extrapolator on a
//! rectilinear grid.
//!
//! Boundary spans use ghost coefficients: virtual B-spline coefficients just
//! outside the stored coefficient line. A cubic span needs four coefficients,
//! so the low boundary span naturally references `c[-1], c[0], c[1], c[2]`,
//! while the high boundary span references `c[n - 3], c[n - 2], c[n - 1], c[n]`.
//! The ghost coefficients are not stored. They are derived from the boundary
//! condition that the boundary span has zero third derivative.
//!
//! On a rectilinear grid, the ghost relations depend on the local nonuniform
//! knot spacing. The low-side relation is computed from the third derivatives
//! of the fixed-span basis weights:
//!
//! ```text
//! q[-1]*c[-1] + q[0]*c[0] + q[1]*c[1] + q[2]*c[2] = 0
//! c[-1] = p[0]*c[0] + p[1]*c[1] + p[2]*c[2]
//! p[j] = -q[j] / q[-1]
//! ```
//!
//! Evaluation folds the ghost coefficient into the stored coefficient footprint
//! by substitution. On the low side, the raw span expression is
//!
//! ```text
//! S = w0*c[-1] + w1*c[0] + w2*c[1] + w3*c[2]
//! ```
//!
//! Substituting the low ghost relation and collecting terms gives stored-only
//! weights:
//!
//! ```text
//! S = (w1 + w0*p[0])*c[0]
//!   + (w2 + w0*p[1])*c[1]
//!   + (w3 + w0*p[2])*c[2]
//!
//! folded weights = [w1 + w0*p[0], w2 + w0*p[1], w3 + w0*p[2], 0]
//! ```
//!
//! The high side is analogous. If
//! `c[n] = s[0]*c[n - 3] + s[1]*c[n - 2] + s[2]*c[n - 1]`, then
//!
//! ```text
//! folded weights = [
//!     0,
//!     w0 + w3*s[0],
//!     w1 + w3*s[1],
//!     w2 + w3*s[2],
//! ]
//! ```
//!
//! In the uniform-grid limit these relations reduce to the regular-grid
//! formulas `c[-1] = 3c[0] - 3c[1] + c[2]` and
//! `c[n] = c[n - 3] - 3c[n - 2] + 3c[n - 1]`.

use super::Saturation;
#[cfg(feature = "par")]
use super::max_usize;
use crate::{index_arr_fixed_dims, interp_math::dot4, mul_add};
use crunchy::unroll;
use num_traits::Float;

/// Evaluate cubic B-spline interpolation on a rectilinear grid in up to 8 dimensions.
///
/// This function expects precomputed B-spline coefficients. To build
/// coefficients from nodal values without allocation, use
/// [`MultiBsplineRectilinear::from_values_with_workspace`].
pub fn interpn<T: Float>(
    grids: &[&[T]],
    coeffs: &[T],
    linearize_extrapolation: bool,
    obs: &[&[T]],
    out: &mut [T],
) -> Result<(), &'static str> {
    let ndims = grids.len();
    if obs.len() != ndims {
        return Err("Dimension mismatch");
    }

    crate::dispatch_ndims!(
        ndims,
        "Dimension exceeds maximum (8). Use interpolator struct directly for higher dimensions.",
        [1, 2, 3, 4, 5, 6, 7, 8],
        |N| {
            MultiBsplineRectilinear::<'_, T, N>::new(
                grids.try_into().unwrap(),
                coeffs,
                linearize_extrapolation,
            )?
            .interp(obs.try_into().unwrap(), out)
        }
    )
}

/// Evaluate interpolant, allocating a new Vec for the output.
#[cfg(feature = "std")]
pub fn interpn_alloc<T: Float>(
    grids: &[&[T]],
    coeffs: &[T],
    linearize_extrapolation: bool,
    obs: &[&[T]],
) -> Result<Vec<T>, &'static str> {
    let mut out = vec![T::zero(); obs[0].len()];
    interpn(grids, coeffs, linearize_extrapolation, obs, &mut out)?;
    Ok(out)
}

/// Construct cubic B-spline coefficients from nodal values on a rectilinear grid.
///
/// The input values are immutable. The generated coefficients are written into
/// `coeffs`. The `scratch` buffer must have length at least
/// [`MultiBsplineRectilinear::construction_scratch_len`] for the provided
/// dimensions.
pub fn coefficients<T: Float, const N: usize>(
    grids: &[&[T]; N],
    vals: &[T],
    coeffs: &mut [T],
    scratch: &mut [T],
) -> Result<(), &'static str> {
    let dims = dims_from_grids(grids);
    check_dims(grids, coeffs)?;
    if vals.len() != coeffs.len() {
        return Err("Dimension mismatch");
    }

    let scratch_len = MultiBsplineRectilinear::<T, N>::construction_scratch_len(dims);
    if scratch.len() < scratch_len {
        return Err("Scratch buffer is too small");
    }

    coeffs.copy_from_slice(vals);

    let max_dim = max_dim(dims);
    let (upper, rhs) = scratch.split_at_mut(max_dim);

    let mut dimprod = [1_usize; N];
    populate_dimprod(dims, &mut dimprod);

    for axis in 0..N {
        solve_axis(grids, dims, dimprod, axis, coeffs, upper, rhs)?;
    }

    Ok(())
}

/// Construct cubic B-spline coefficients from nodal values on a rectilinear
/// grid, solving independent coefficient slabs in parallel.
///
/// This is the nonallocating parallel construction path. The `scratch` buffer
/// must have length at least
/// [`MultiBsplineRectilinear::parallel_construction_scratch_len`] for the
/// provided dimensions and `max_threads`.
///
/// Parallelism is applied one axis at a time. For each axis, the coefficient
/// tensor is divided into contiguous slabs that can be written independently.
/// In C-style memory order, the leading axis forms a single slab and falls back
/// to the serial solve for that axis.
///
/// `max_threads` is clamped to at least one worker and to the current Rayon
/// pool size. The function does not allocate; each worker borrows its own
/// tridiagonal `upper` and `rhs` slices from `scratch`.
#[cfg(feature = "par")]
pub fn coefficients_par<T: Float + Send + Sync, const N: usize>(
    grids: &[&[T]; N],
    vals: &[T],
    coeffs: &mut [T],
    scratch: &mut [T],
    max_threads: usize,
) -> Result<(), &'static str> {
    let dims = dims_from_grids(grids);
    check_dims(grids, coeffs)?;
    if vals.len() != coeffs.len() {
        return Err("Dimension mismatch");
    }

    let scratch_len =
        MultiBsplineRectilinear::<T, N>::parallel_construction_scratch_len(dims, max_threads);
    if scratch_len == 0 || scratch.len() < scratch_len {
        return Err("Scratch buffer is too small");
    }

    coeffs.copy_from_slice(vals);

    let mut dimprod = [1_usize; N];
    populate_dimprod(dims, &mut dimprod);

    let tasks = max_threads.max(1).min(rayon::current_num_threads()).max(1);
    for axis in 0..N {
        solve_axis_par(grids, dims, dimprod, axis, coeffs, scratch, tasks)?;
    }

    Ok(())
}

/// An N-dimensional cubic B-spline interpolator / extrapolator on a
/// rectilinear grid.
pub struct MultiBsplineRectilinear<'a, T: Float, const N: usize> {
    /// x, y, ... coordinate grids, each entry of size dims[i]
    grids: &'a [&'a [T]],

    /// Size of each dimension
    dims: [usize; N],

    /// B-spline coefficients, size prod(dims)
    coeffs: &'a [T],

    /// Whether to extrapolate linearly instead of continuing the boundary segment
    linearize_extrapolation: bool,
}

impl<'a, T: Float, const N: usize> MultiBsplineRectilinear<'a, T, N> {
    /// Number of coefficients required for `dims`.
    ///
    /// Returns zero if any dimension is invalid for this interpolator, if
    /// `N == 0`, or if the product overflows `usize`.
    pub const fn coeff_storage_len(dims: [usize; N]) -> usize {
        if N == 0 {
            return 0;
        }

        coeff_storage_len_inner(dims, 0, 1)
    }

    /// Number of scratch values required to construct coefficients for `dims`.
    ///
    /// Returns zero if any dimension is invalid for this interpolator, if
    /// `N == 0`, or if the scratch length overflows `usize`.
    pub const fn construction_scratch_len(dims: [usize; N]) -> usize {
        if N == 0 {
            return 0;
        }

        match max_valid_dim(dims, 0, 0).checked_mul(2) {
            Some(v) => v,
            None => 0,
        }
    }

    /// Number of scratch values required to construct coefficients with
    /// [`coefficients_par`] using at most `max_threads` worker tasks.
    ///
    /// The returned length is the maximum per-axis scratch requirement for the
    /// slab decomposition used by [`coefficients_par`], not simply
    /// `construction_scratch_len(dims) * max_threads`. In C-style memory order,
    /// an axis can only use as many independent contiguous slabs as there are
    /// prefix elements before that axis, so early axes may require less scratch
    /// than the requested thread count.
    ///
    /// A `max_threads` value of zero is treated as one worker.
    ///
    /// Returns zero if any dimension is invalid for this interpolator, if
    /// `N == 0`, or if the scratch length overflows `usize`.
    #[cfg(feature = "par")]
    pub const fn parallel_construction_scratch_len(dims: [usize; N], max_threads: usize) -> usize {
        if N == 0 {
            return 0;
        }

        let max_threads = max_usize(max_threads, 1);
        parallel_construction_scratch_len_inner(dims, max_threads, 0, 1, 0)
    }

    /// Build an interpolator from precomputed coefficients.
    ///
    /// # Errors
    /// * If dimensions do not match
    /// * If any dimension has fewer than four entries
    /// * If any grid is not monotonically increasing
    pub fn new(
        grids: &'a [&'a [T]; N],
        coeffs: &'a [T],
        linearize_extrapolation: bool,
    ) -> Result<Self, &'static str> {
        check_dims(grids, coeffs)?;
        let dims = dims_from_grids(grids);

        Ok(Self {
            grids,
            dims,
            coeffs,
            linearize_extrapolation,
        })
    }

    /// Build coefficients from nodal values using caller-provided storage, then
    /// return a borrowed interpolator over those coefficients.
    ///
    /// This is the nonallocating construction path. `vals` is immutable input,
    /// `coeffs` receives generated B-spline coefficients, and `scratch` is used
    /// only during construction.
    pub fn from_values_with_workspace(
        grids: &'a [&'a [T]; N],
        vals: &[T],
        coeffs: &'a mut [T],
        scratch: &mut [T],
        linearize_extrapolation: bool,
    ) -> Result<Self, &'static str> {
        coefficients(grids, vals, coeffs, scratch)?;
        Self::new(grids, coeffs, linearize_extrapolation)
    }

    /// Build coefficients from nodal values using caller-provided storage and a
    /// caller-provided parallel construction scratch buffer, then return a
    /// borrowed interpolator over those coefficients.
    ///
    /// This preserves the borrowed, nonallocating Rust API while allowing
    /// coefficient construction to use Rayon. The caller owns `coeffs` for the
    /// lifetime of the returned interpolator and owns `scratch` only during
    /// construction.
    #[cfg(feature = "par")]
    pub fn from_values_with_workspace_par(
        grids: &'a [&'a [T]; N],
        vals: &[T],
        coeffs: &'a mut [T],
        scratch: &mut [T],
        max_threads: usize,
        linearize_extrapolation: bool,
    ) -> Result<Self, &'static str>
    where
        T: Send + Sync,
    {
        coefficients_par(grids, vals, coeffs, scratch, max_threads)?;
        Self::new(grids, coeffs, linearize_extrapolation)
    }

    /// Interpolate on a contiguous list of observation points.
    pub fn interp(&self, x: &[&[T]; N], out: &mut [T]) -> Result<(), &'static str> {
        let n = out.len();
        for i in 0..N {
            if x[i].len() != n {
                return Err("Dimension mismatch");
            }
        }

        let mut tmp = [T::zero(); N];
        for i in 0..n {
            (0..N).for_each(|j| tmp[j] = x[j][i]);
            out[i] = self.interp_one(tmp)?;
        }

        Ok(())
    }

    /// Interpolate the value at one point.
    pub fn interp_one(&self, x: [T; N]) -> Result<T, &'static str> {
        let mut origin = [0_usize; N];
        let mut span = [0_usize; N];
        let mut sat = [Saturation::None; N];
        let mut weights = [[T::zero(); FP]; N];
        let mut dimprod = [1_usize; N];
        let mut loc = [0_usize; N];
        let mut store = [[T::zero(); FP]; N];

        populate_dimprod(self.dims, &mut dimprod);

        for i in 0..N {
            (origin[i], span[i], sat[i]) = self.get_loc(x[i], i);
            weights[i] = interp_weights(
                self.grids[i],
                span[i],
                x[i],
                sat[i],
                self.linearize_extrapolation,
            );
        }

        let nverts = const { FP.pow(N as u32) };

        macro_rules! unroll_vertices_body {
            ($i:ident) => {
                for j in 0..N {
                    if j == 0 {
                        for k in 0..N {
                            let offset: usize = ($i & (3 << (2 * k))) >> (2 * k);
                            loc[k] = origin[k] + offset;
                        }
                        let store_ind: usize = $i % FP;
                        store[0][store_ind] = index_arr_fixed_dims(loc, dimprod, self.coeffs);
                    } else {
                        let q: usize = FP.pow(j as u32);
                        let level: bool = ($i + 1).is_multiple_of(q);
                        let p: usize = (($i + 1) / q).saturating_sub(1) % FP;
                        let ind: usize = j.saturating_sub(1);

                        if level {
                            store[j][p] = dot4(weights[ind], store[ind]);
                        }
                    }
                }
            };
        }

        #[cfg(not(feature = "deep-unroll"))]
        if N <= 3 {
            unroll! {
                for i < 64 in 0..nverts {
                    unroll_vertices_body!(i);
                }
            }
        } else {
            for i in 0..nverts {
                unroll_vertices_body!(i);
            }
        }

        #[cfg(feature = "deep-unroll")]
        if N <= 4 {
            unroll! {
                for i < 256 in 0..nverts {
                    unroll_vertices_body!(i);
                }
            }
        } else {
            for i in 0..nverts {
                unroll_vertices_body!(i);
            }
        }

        Ok(dot4(weights[N - 1], store[N - 1]))
    }

    /// Return the stored coefficient origin, knot span, and saturation region.
    #[inline]
    fn get_loc(&self, v: T, dim: usize) -> (usize, usize, Saturation) {
        let grid = self.grids[dim];
        let n = self.dims[dim];
        let iloc: isize = grid.partition_point(|x| *x < v) as isize - 2;

        if iloc < -1 {
            (0, 0, Saturation::OutsideLow)
        } else if iloc == -1 {
            (0, 0, Saturation::InsideLow)
        } else if iloc > n as isize - 3 {
            (n - 4, n - 2, Saturation::OutsideHigh)
        } else if iloc == n as isize - 3 {
            (n - 4, n - 2, Saturation::InsideHigh)
        } else {
            let span = (iloc + 1) as usize;
            (span - 1, span, Saturation::None)
        }
    }
}

const FP: usize = 4;

/// Extract dimension lengths from a fixed-size array of grid slices.
fn dims_from_grids<T: Float, const N: usize>(grids: &[&[T]; N]) -> [usize; N] {
    let mut dims = [0_usize; N];
    for i in 0..N {
        dims[i] = grids[i].len();
    }
    dims
}

/// Validate dimensions, coefficient storage length, and grid monotonicity.
fn check_dims<T: Float, const N: usize>(grids: &[&[T]; N], data: &[T]) -> Result<(), &'static str> {
    let dims = dims_from_grids(grids);
    let nvals = MultiBsplineRectilinear::<T, N>::coeff_storage_len(dims);
    if nvals == 0 || data.len() != nvals {
        return Err("Dimension mismatch");
    }
    for grid in grids {
        if !grid.windows(2).all(|w| w[1] > w[0]) {
            return Err("All grids must be monotonically increasing");
        }
    }
    Ok(())
}

/// Recursive implementation for [`MultiBsplineRectilinear::coeff_storage_len`].
///
/// This is recursive rather than a `for` loop so the public sizing function can
/// remain `const fn` on stable Rust.
const fn coeff_storage_len_inner<const N: usize>(
    dims: [usize; N],
    axis: usize,
    out: usize,
) -> usize {
    if axis == N {
        return out;
    }
    if dims[axis] < 4 {
        return 0;
    }

    match out.checked_mul(dims[axis]) {
        Some(v) => coeff_storage_len_inner(dims, axis + 1, v),
        None => 0,
    }
}

/// Return the largest valid dimension, or zero if any dimension is invalid.
///
/// This is recursive rather than a `for` loop so it can be called from `const fn`.
const fn max_valid_dim<const N: usize>(dims: [usize; N], axis: usize, max: usize) -> usize {
    if axis == N {
        return max;
    }
    if dims[axis] < 4 {
        return 0;
    }

    let next_max = if dims[axis] > max { dims[axis] } else { max };
    max_valid_dim(dims, axis + 1, next_max)
}

#[cfg(feature = "par")]
/// Recursive implementation for the parallel construction scratch length.
///
/// Each axis can use at most the number of independent prefix slabs available in
/// C-style memory order, so this tracks both the prefix slab count and the
/// largest per-axis scratch requirement.
const fn parallel_construction_scratch_len_inner<const N: usize>(
    dims: [usize; N],
    max_threads: usize,
    axis: usize,
    prefix_slabs: usize,
    max_scratch: usize,
) -> usize {
    if axis == N {
        return max_scratch;
    }
    if dims[axis] < 4 {
        return 0;
    }

    let tasks = if prefix_slabs < max_threads {
        prefix_slabs
    } else {
        max_threads
    };
    let scratch = match dims[axis].checked_mul(2) {
        Some(v) => match v.checked_mul(tasks) {
            Some(v) => v,
            None => return 0,
        },
        None => return 0,
    };
    let next_max_scratch = if scratch > max_scratch {
        scratch
    } else {
        max_scratch
    };
    let next_prefix_slabs = match prefix_slabs.checked_mul(dims[axis]) {
        Some(v) => v,
        None => return 0,
    };

    parallel_construction_scratch_len_inner(
        dims,
        max_threads,
        axis + 1,
        next_prefix_slabs,
        next_max_scratch,
    )
}

/// Return the largest dimension length for runtime scratch splitting.
fn max_dim<const N: usize>(dims: [usize; N]) -> usize {
    let mut max = 0_usize;
    for dim in dims {
        if dim > max {
            max = dim;
        }
    }
    max
}

/// Populate C-order strides for each dimension.
///
/// `dimprod[axis]` is the distance in the flattened coefficient buffer between
/// neighboring coefficients along `axis`.
fn populate_dimprod<const N: usize>(dims: [usize; N], dimprod: &mut [usize; N]) {
    let mut acc = 1;
    for i in 0..N {
        if i > 0 {
            acc *= dims[N - i];
        }
        dimprod[N - i - 1] = acc;
    }
}

/// Solve every one-dimensional coefficient line along one axis.
fn solve_axis<T: Float, const N: usize>(
    grids: &[&[T]; N],
    dims: [usize; N],
    dimprod: [usize; N],
    axis: usize,
    coeffs: &mut [T],
    upper: &mut [T],
    rhs: &mut [T],
) -> Result<(), &'static str> {
    let n = dims[axis];
    let stride = dimprod[axis];
    let nlines = coeffs.len() / n;

    for line in 0..nlines {
        let base = line_base_index(dims, dimprod, axis, line);
        solve_line(grids[axis], base, stride, coeffs, upper, rhs)?;
    }

    Ok(())
}

#[cfg(feature = "par")]
/// Solve one axis by splitting independent contiguous slabs across workers.
fn solve_axis_par<T: Float + Send + Sync, const N: usize>(
    grids: &[&[T]; N],
    dims: [usize; N],
    dimprod: [usize; N],
    axis: usize,
    coeffs: &mut [T],
    scratch: &mut [T],
    max_tasks: usize,
) -> Result<(), &'static str> {
    let n = dims[axis];
    let stride = dimprod[axis];
    let slab_len = n * stride;
    let nslabs = coeffs.len() / slab_len;
    let tasks = max_tasks.min(nslabs).max(1);
    let scratch_len = 2 * n * tasks;
    let scratch = &mut scratch[..scratch_len];

    if tasks == 1 {
        let (upper, rhs) = scratch.split_at_mut(n);
        solve_axis_slabs(grids[axis], coeffs, slab_len, stride, upper, rhs)
    } else {
        solve_axis_slabs_par(grids[axis], coeffs, slab_len, stride, scratch, tasks)
    }
}

#[cfg(feature = "par")]
/// Recursively split slab ranges so each Rayon branch owns disjoint coeff/scratch slices.
fn solve_axis_slabs_par<T: Float + Send + Sync>(
    grid: &[T],
    coeffs: &mut [T],
    slab_len: usize,
    stride: usize,
    scratch: &mut [T],
    tasks: usize,
) -> Result<(), &'static str> {
    let nslabs = coeffs.len() / slab_len;
    if tasks <= 1 || nslabs <= 1 {
        let n = grid.len();
        let (upper, rhs) = scratch.split_at_mut(n);
        return solve_axis_slabs(grid, coeffs, slab_len, stride, upper, rhs);
    }

    let left_slabs = nslabs / 2;
    let left_tasks = tasks / 2;
    let right_tasks = tasks - left_tasks;

    let n = grid.len();
    let coeff_split = left_slabs * slab_len;
    let scratch_split = 2 * n * left_tasks;
    let (left_coeffs, right_coeffs) = coeffs.split_at_mut(coeff_split);
    let (left_scratch, right_scratch) = scratch.split_at_mut(scratch_split);

    let (left, right) = rayon::join(
        || {
            solve_axis_slabs_par(
                grid,
                left_coeffs,
                slab_len,
                stride,
                left_scratch,
                left_tasks,
            )
        },
        || {
            solve_axis_slabs_par(
                grid,
                right_coeffs,
                slab_len,
                stride,
                right_scratch,
                right_tasks,
            )
        },
    );
    left?;
    right
}

#[cfg(feature = "par")]
/// Solve all coefficient lines contained in a contiguous set of slabs.
fn solve_axis_slabs<T: Float>(
    grid: &[T],
    coeffs: &mut [T],
    slab_len: usize,
    stride: usize,
    upper: &mut [T],
    rhs: &mut [T],
) -> Result<(), &'static str> {
    for slab in coeffs.chunks_mut(slab_len) {
        for base in 0..stride {
            solve_line(grid, base, stride, slab, upper, rhs)?;
        }
    }

    Ok(())
}

/// Convert a line number, excluding one active axis, into a flattened base index.
fn line_base_index<const N: usize>(
    dims: [usize; N],
    dimprod: [usize; N],
    axis: usize,
    line: usize,
) -> usize {
    let mut rem = line;
    let mut base = 0_usize;

    for d in (0..N).rev() {
        if d != axis {
            let coord = rem % dims[d];
            rem /= dims[d];
            base += coord * dimprod[d];
        }
    }

    base
}

/// Solve one nonuniform cubic B-spline coefficient line.
///
/// The direct coefficient system remains tridiagonal with the same coefficient
/// alignment used by `MultiBsplineRegular`. At node `x[i]`, the fixed-span
/// cubic B-spline collocation row has the raw local form:
///
/// ```text
/// l[i] c[i - 1] + m[i] c[i] + r[i] c[i + 1] = y[i]
/// ```
///
/// The low endpoint row initially contains the ghost coefficient:
///
/// ```text
/// l[0] c[-1] + m[0] c[0] + r[0] c[1] = y[0]
/// ```
///
/// On the low boundary interval, the third derivative is constant and linear
/// in active coefficients:
///
/// ```text
/// q[-1] c[-1] + q[0] c[0] + q[1] c[1] + q[2] c[2] = S_0'''
/// ```
///
/// The zero-third-derivative boundary condition gives:
///
/// ```text
/// c[-1] = p[0] c[0] + p[1] c[1] + p[2] c[2]
/// p[j] = -q[j] / q[-1]
/// ```
///
/// Substituting this relation creates a temporary endpoint row involving
/// `c[0]`, `c[1]`, and `c[2]`. Combining it with the adjacent collocation row
/// eliminates `c[2]`, leaving a tridiagonal first row. The high endpoint is
/// symmetric: derive `c[n]` from the last interval's zero third derivative and
/// combine with the adjacent row to eliminate `c[n - 3]`.
///
/// In the uniform-grid limit, the ghost relations reduce to:
///
/// ```text
/// c[-1] = 3c[0] - 3c[1] + c[2]
/// c[n]  = c[n - 3] - 3c[n - 2] + 3c[n - 1]
/// ```
///
/// and the row combinations reduce to the regular-grid rows:
///
/// ```text
/// row 0:       c[0] - c[1] = y[0] - y[1]
/// row 1..n-2:  c[i - 1] + 4c[i] + c[i + 1] = 6y[i]
/// row n-1:    -c[n - 2] + c[n - 1] = y[n - 1] - y[n - 2]
/// ```
fn solve_line<T: Float>(
    grid: &[T],
    base: usize,
    stride: usize,
    coeffs: &mut [T],
    upper: &mut [T],
    rhs: &mut [T],
) -> Result<(), &'static str> {
    let n = grid.len();
    if n < 4 || upper.len() < n || rhs.len() < n {
        return Err("Dimension mismatch");
    }

    let (diag0, upper0, rhs0) = first_row(grid, coeffs[base], coeffs[base + stride]);
    upper[0] = upper0 / diag0;
    rhs[0] = rhs0 / diag0;

    for i in 1..n {
        let (lower, diag, upper_i, y) = if i == n - 1 {
            last_row(
                grid,
                coeffs[base + (n - 2) * stride],
                coeffs[base + (n - 1) * stride],
            )
        } else {
            let w = basis_span_weights(grid, i, grid[i]);
            (w[0], w[1], w[2], coeffs[base + i * stride])
        };

        let denom = diag - lower * upper[i - 1];
        upper[i] = upper_i / denom;
        rhs[i] = (y - lower * rhs[i - 1]) / denom;
    }

    for i in (0..n - 1).rev() {
        rhs[i] = rhs[i] - upper[i] * rhs[i + 1];
    }

    for i in 0..n {
        coeffs[base + i * stride] = rhs[i];
    }

    Ok(())
}

/// Build the first tridiagonal row after folding in the low ghost coefficient.
fn first_row<T: Float>(grid: &[T], y0: T, y1: T) -> (T, T, T) {
    let w0 = basis_span_weights(grid, 0, grid[0]);
    let p = low_ghost_coeffs(grid);
    let e0 = w0[1] + w0[0] * p[0];
    let e1 = w0[2] + w0[0] * p[1];
    let e2 = w0[3] + w0[0] * p[2];

    let w1 = basis_span_weights(grid, 1, grid[1]);
    let factor = e2 / w1[2];
    (e0 - factor * w1[0], e1 - factor * w1[1], y0 - factor * y1)
}

/// Build the last tridiagonal row after folding in the high ghost coefficient.
fn last_row<T: Float>(grid: &[T], y_prev: T, y_last: T) -> (T, T, T, T) {
    let n = grid.len();
    let w = basis_span_weights(grid, n - 2, grid[n - 1]);
    let s = high_ghost_coeffs(grid);

    let e0 = w[0] + w[3] * s[0];
    let e1 = w[1] + w[3] * s[1];
    let e2 = w[2] + w[3] * s[2];

    let wadj = basis_span_weights(grid, n - 2, grid[n - 2]);
    let factor = e0 / wadj[0];
    (
        e1 - factor * wadj[1],
        e2 - factor * wadj[2],
        T::zero(),
        y_last - factor * y_prev,
    )
}

#[inline]
/// Select the appropriate four local B-spline weights for an interpolation region.
fn interp_weights<T: Float>(
    grid: &[T],
    span: usize,
    x: T,
    sat: Saturation,
    linearize_extrapolation: bool,
) -> [T; 4] {
    match sat {
        Saturation::None => basis_span_weights(grid, span, x),
        Saturation::InsideLow => low_boundary_weights(grid, x, false),
        Saturation::OutsideLow => low_boundary_weights(grid, x, linearize_extrapolation),
        Saturation::InsideHigh => high_boundary_weights(grid, x, false),
        Saturation::OutsideHigh => high_boundary_weights(grid, x, linearize_extrapolation),
    }
}

#[inline]
/// Low-boundary weights with the ghost coefficient folded into stored coefficients.
fn low_boundary_weights<T: Float>(grid: &[T], x: T, linearize_extrapolation: bool) -> [T; 4] {
    let raw = if linearize_extrapolation {
        linearized_boundary_weights(grid, 0, grid[0], x)
    } else {
        basis_span_weights(grid, 0, x)
    };
    let p = low_ghost_coeffs(grid);

    [
        mul_add(raw[0], p[0], raw[1]),
        mul_add(raw[0], p[1], raw[2]),
        mul_add(raw[0], p[2], raw[3]),
        T::zero(),
    ]
}

#[inline]
/// High-boundary weights with the ghost coefficient folded into stored coefficients.
fn high_boundary_weights<T: Float>(grid: &[T], x: T, linearize_extrapolation: bool) -> [T; 4] {
    let n = grid.len();
    let raw = if linearize_extrapolation {
        linearized_boundary_weights(grid, n - 2, grid[n - 1], x)
    } else {
        basis_span_weights(grid, n - 2, x)
    };
    let s = high_ghost_coeffs(grid);

    [
        T::zero(),
        mul_add(raw[3], s[0], raw[0]),
        mul_add(raw[3], s[1], raw[1]),
        mul_add(raw[3], s[2], raw[2]),
    ]
}

#[inline]
/// Affine continuation of a boundary span's value and first derivative.
fn linearized_boundary_weights<T: Float>(grid: &[T], span: usize, endpoint: T, x: T) -> [T; 4] {
    let weights = basis_span_weights(grid, span, endpoint);
    let derivs = basis_span_weight_derivatives(grid, span, endpoint);
    let dx = x - endpoint;
    [
        mul_add(dx, derivs[0], weights[0]),
        mul_add(dx, derivs[1], weights[1]),
        mul_add(dx, derivs[2], weights[2]),
        mul_add(dx, derivs[3], weights[3]),
    ]
}

#[cfg(test)]
/// Evaluate the low ghost coefficient relation for tests.
fn low_ghost<T: Float>(grid: &[T], vals: &[T; 4]) -> T {
    let p = low_ghost_coeffs(grid);
    mul_add(p[2], vals[2], mul_add(p[1], vals[1], p[0] * vals[0]))
}

#[cfg(test)]
/// Evaluate the high ghost coefficient relation for tests.
fn high_ghost<T: Float>(grid: &[T], vals: &[T; 4]) -> T {
    let s = high_ghost_coeffs(grid);
    mul_add(s[2], vals[3], mul_add(s[1], vals[2], s[0] * vals[1]))
}

/// Coefficients expressing the low ghost as a linear combination of stored coeffs.
fn low_ghost_coeffs<T: Float>(grid: &[T]) -> [T; 3] {
    let q = span_weight_third_derivatives(grid, 0);
    [-q[1] / q[0], -q[2] / q[0], -q[3] / q[0]]
}

/// Coefficients expressing the high ghost as a linear combination of stored coeffs.
fn high_ghost_coeffs<T: Float>(grid: &[T]) -> [T; 3] {
    let q = span_weight_third_derivatives(grid, grid.len() - 2);
    [-q[0] / q[3], -q[1] / q[3], -q[2] / q[3]]
}

/// Third derivatives of the four fixed-span basis weights on one span.
fn span_weight_third_derivatives<T: Float>(grid: &[T], span: usize) -> [T; 4] {
    let xs = span_samples(grid, span);
    let mut values = [[T::zero(); 4]; 4];
    for i in 0..4 {
        values[i] = basis_span_weights(grid, span, xs[i]);
    }

    let six = T::from(6.0).unwrap();
    let mut out = [T::zero(); 4];
    for j in 0..4 {
        let f = [values[0][j], values[1][j], values[2][j], values[3][j]];
        out[j] = six * third_divided_difference(xs, f);
    }
    out
}

/// First derivatives of the four fixed-span basis weights at `x`.
fn basis_span_weight_derivatives<T: Float>(grid: &[T], span: usize, x: T) -> [T; 4] {
    let xs = span_samples(grid, span);
    let mut values = [[T::zero(); 4]; 4];
    for i in 0..4 {
        values[i] = basis_span_weights(grid, span, xs[i]);
    }

    let mut out = [T::zero(); 4];
    for j in 0..4 {
        let f = [values[0][j], values[1][j], values[2][j], values[3][j]];
        out[j] = lagrange_derivative(xs, f, x);
    }
    out
}

/// Four sample locations spanning one knot interval.
fn span_samples<T: Float>(grid: &[T], span: usize) -> [T; 4] {
    let three = T::from(3.0).unwrap();
    let a = grid[span];
    let h = grid[span + 1] - a;
    [a, a + h / three, a + (h + h) / three, a + h]
}

/// Third divided difference of four samples.
fn third_divided_difference<T: Float>(x: [T; 4], f: [T; 4]) -> T {
    let d01 = (f[1] - f[0]) / (x[1] - x[0]);
    let d12 = (f[2] - f[1]) / (x[2] - x[1]);
    let d23 = (f[3] - f[2]) / (x[3] - x[2]);
    let d012 = (d12 - d01) / (x[2] - x[0]);
    let d123 = (d23 - d12) / (x[3] - x[1]);
    (d123 - d012) / (x[3] - x[0])
}

/// Derivative of the cubic Lagrange interpolant through four samples.
fn lagrange_derivative<T: Float>(x: [T; 4], f: [T; 4], at: T) -> T {
    let mut out = T::zero();
    for j in 0..4 {
        let mut basis_deriv = T::zero();
        for m in 0..4 {
            if m == j {
                continue;
            }
            let mut term = T::one() / (x[j] - x[m]);
            for k in 0..4 {
                if k != j && k != m {
                    term = term * (at - x[k]) / (x[j] - x[k]);
                }
            }
            basis_deriv = basis_deriv + term;
        }
        out = out + f[j] * basis_deriv;
    }
    out
}

/// Fixed-span cubic B-spline basis weights.
///
/// The returned weights multiply coefficients
/// `c[span - 1]..c[span + 2]`, with ghost coefficients allowed at the two
/// boundaries. The span is held fixed, so this also gives the polynomial
/// continuation of the boundary span during extrapolation.
fn basis_span_weights<T: Float>(grid: &[T], span: usize, x: T) -> [T; 4] {
    let i = span as isize;
    let mut n = [T::zero(); 4];
    let mut left = [T::zero(); 4];
    let mut right = [T::zero(); 4];
    n[0] = T::one();

    for j in 1..=3 {
        left[j] = x - knot(grid, i + 1 - j as isize);
        right[j] = knot(grid, i + j as isize) - x;
        let mut saved = T::zero();
        for r in 0..j {
            let denom = right[r + 1] + left[j - r];
            let temp = n[r] / denom;
            n[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        n[j] = saved;
    }

    n
}

/// Return an extrapolated open-uniform knot for boundary spans.
fn knot<T: Float>(grid: &[T], index: isize) -> T {
    let n = grid.len() as isize;
    if index < 0 {
        grid[0] + T::from(index).unwrap() * (grid[1] - grid[0])
    } else if index >= n {
        grid[(n - 1) as usize]
            + T::from(index - n + 1).unwrap() * (grid[(n - 1) as usize] - grid[(n - 2) as usize])
    } else {
        grid[index as usize]
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::multibspline::regular::MultiBsplineRegular;
    use crate::utils::meshgrid;

    fn reconstruct_values<const N: usize>(grids: &[&[f64]; N], coeffs: &[f64]) -> Vec<f64> {
        let dims = dims_from_grids(grids);
        let mut out = coeffs.to_vec();
        let mut tmp = vec![0.0; max_dim(dims)];
        let mut dimprod = [1_usize; N];
        populate_dimprod(dims, &mut dimprod);

        for axis in 0..N {
            let n = dims[axis];
            let stride = dimprod[axis];
            let nlines = out.len() / n;

            for line in 0..nlines {
                let base = line_base_index(dims, dimprod, axis, line);
                for i in 0..n {
                    tmp[i] = out[base + i * stride];
                }

                for i in 0..n {
                    let value = if i == 0 {
                        let ghost = low_ghost(grids[axis], (&tmp[0..4]).try_into().unwrap());
                        let w = basis_span_weights(grids[axis], 0, grids[axis][0]);
                        dot4(w, [ghost, tmp[0], tmp[1], tmp[2]])
                    } else if i == n - 1 {
                        let ghost = high_ghost(grids[axis], (&tmp[n - 4..n]).try_into().unwrap());
                        let w = basis_span_weights(grids[axis], n - 2, grids[axis][n - 1]);
                        dot4(w, [tmp[n - 3], tmp[n - 2], tmp[n - 1], ghost])
                    } else {
                        let w = basis_span_weights(grids[axis], i, grids[axis][i]);
                        dot4(w, [tmp[i - 1], tmp[i], tmp[i + 1], tmp[(i + 2).min(n - 1)]])
                    };
                    out[base + i * stride] = value;
                }
            }
        }

        out
    }

    fn assert_linear_extrapolation(values: [f64; 3]) {
        assert!(
            (values[2] - 2.0 * values[1] + values[0]).abs() < 1e-10,
            "extrapolated values are not linear: {values:?}"
        );
    }

    #[test]
    fn test_uniform_rows_reduce_to_regular_rows() {
        let grid = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let p = low_ghost_coeffs(&grid);
        let s = high_ghost_coeffs(&grid);
        for (got, expected) in p.iter().zip([3.0, -3.0, 1.0]) {
            assert!((got - expected).abs() < 1e-10, "{got} != {expected}");
        }
        for (got, expected) in s.iter().zip([1.0, -3.0, 3.0]) {
            assert!((got - expected).abs() < 1e-10, "{got} != {expected}");
        }

        let (d0, u0, r0) = first_row(&grid, 7.0, 11.0);
        assert!((d0 - 1.0).abs() < 1e-10);
        assert!((u0 + 1.0).abs() < 1e-10);
        assert!((r0 - (7.0 - 11.0)).abs() < 1e-10);

        let (ll, dl, _, rl) = last_row(&grid, 13.0, 17.0);
        assert!((ll + 1.0).abs() < 1e-10);
        assert!((dl - 1.0).abs() < 1e-10);
        assert!((rl - (17.0 - 13.0)).abs() < 1e-10);
    }

    #[test]
    fn test_storage_lengths() {
        assert_eq!(
            MultiBsplineRectilinear::<f64, 2>::coeff_storage_len([4, 5]),
            20
        );
        assert_eq!(
            MultiBsplineRectilinear::<f64, 2>::construction_scratch_len([4, 5]),
            10
        );
        #[cfg(feature = "par")]
        assert_eq!(
            MultiBsplineRectilinear::<f64, 2>::parallel_construction_scratch_len([4, 5], 4),
            40
        );
        assert_eq!(
            MultiBsplineRectilinear::<f64, 2>::coeff_storage_len([3, 5]),
            0
        );
        #[cfg(feature = "par")]
        assert_eq!(
            MultiBsplineRectilinear::<f64, 2>::parallel_construction_scratch_len([4, 5], 0),
            MultiBsplineRectilinear::<f64, 2>::parallel_construction_scratch_len([4, 5], 1)
        );
    }

    #[cfg(feature = "par")]
    #[test]
    fn test_parallel_coefficients_match_serial() {
        let dims = [5_usize, 6, 7];
        let xs: Vec<Vec<f64>> = (0..3)
            .map(|i| {
                (0..dims[i])
                    .map(|j| -1.0 + i as f64 + j as f64 * 0.31 + (j as f64).powi(2) * 0.03)
                    .collect()
            })
            .collect();
        let grids: Vec<&[f64]> = xs.iter().map(|x| &x[..]).collect();
        let grids_ref: &[&[f64]; 3] = grids.as_slice().try_into().unwrap();
        let grid = meshgrid((0..3).map(|i| &xs[i]).collect());
        let vals: Vec<f64> = grid
            .iter()
            .map(|x| {
                x.iter()
                    .enumerate()
                    .map(|(i, v)| (i as f64 + 0.5) * v.sin() + v * v)
                    .sum()
            })
            .collect();

        let nvals = MultiBsplineRectilinear::<f64, 3>::coeff_storage_len(dims);
        let serial_scratch_len = MultiBsplineRectilinear::<f64, 3>::construction_scratch_len(dims);
        let parallel_scratch_len =
            MultiBsplineRectilinear::<f64, 3>::parallel_construction_scratch_len(dims, 4);

        let mut serial_coeffs = vec![0.0; nvals];
        let mut serial_scratch = vec![0.0; serial_scratch_len];
        coefficients(grids_ref, &vals, &mut serial_coeffs, &mut serial_scratch).unwrap();

        let mut parallel_coeffs = vec![0.0; nvals];
        let mut parallel_scratch = vec![0.0; parallel_scratch_len];
        coefficients_par(
            grids_ref,
            &vals,
            &mut parallel_coeffs,
            &mut parallel_scratch,
            4,
        )
        .unwrap();

        for i in 0..nvals {
            assert!(
                (serial_coeffs[i] - parallel_coeffs[i]).abs() < 1e-12,
                "coefficient mismatch at {i}: {} vs {}",
                serial_coeffs[i],
                parallel_coeffs[i]
            );
        }
    }

    #[test]
    fn test_coefficients_reconstruct_values_1d_to_3d() {
        for ndims in 1..=3 {
            crate::dispatch_ndims!(ndims, "bad dims", [1, 2, 3], |N| {
                let dims = [6_usize; N];
                let xs: Vec<Vec<f64>> = (0..N)
                    .map(|i| {
                        let start = -1.5 * (i as f64 + 1.0);
                        (0..dims[i])
                            .map(|j| start + (j as f64).powi(2) * 0.21 + j as f64 * 0.37)
                            .collect()
                    })
                    .collect();
                let grids: Vec<&[f64]> = xs.iter().map(|x| &x[..]).collect();
                let grid = meshgrid((0..N).map(|i| &xs[i]).collect());
                let vals: Vec<f64> = grid
                    .iter()
                    .map(|x| {
                        x.iter()
                            .enumerate()
                            .map(|(i, v)| (i as f64 + 1.0) * v * v + 0.25 * v)
                            .sum()
                    })
                    .collect();

                let nvals = MultiBsplineRectilinear::<f64, N>::coeff_storage_len(dims);
                let scratch_len = MultiBsplineRectilinear::<f64, N>::construction_scratch_len(dims);
                let mut coeffs = vec![0.0; nvals];
                let mut scratch = vec![0.0; scratch_len];
                let grids_ref: &[&[f64]; N] = grids.as_slice().try_into().unwrap();
                coefficients::<f64, N>(grids_ref, &vals, &mut coeffs, &mut scratch).unwrap();
                let reconstructed = reconstruct_values::<N>(grids_ref, &coeffs);

                for i in 0..vals.len() {
                    assert!(
                        (vals[i] - reconstructed[i]).abs() < 1e-9,
                        "{ndims}D mismatch at {i}: {} vs {}",
                        vals[i],
                        reconstructed[i]
                    );
                }
                Ok(())
            })
            .unwrap();
        }
    }

    #[test]
    fn test_interp_reproduces_grid_values_1d_to_3d() {
        for ndims in 1..=3 {
            crate::dispatch_ndims!(ndims, "bad dims", [1, 2, 3], |N| {
                let dims = [6_usize; N];
                let xs: Vec<Vec<f64>> = (0..N)
                    .map(|i| {
                        (0..dims[i])
                            .map(|j| -1.0 + j as f64 * 0.4 + (j as f64).powi(2) * 0.05)
                            .collect()
                    })
                    .collect();
                let grids: Vec<&[f64]> = xs.iter().map(|x| &x[..]).collect();
                let grid = meshgrid((0..N).map(|i| &xs[i]).collect());
                let vals: Vec<f64> = grid
                    .iter()
                    .map(|x| x.iter().map(|v| v * v + 0.3 * v).sum())
                    .collect();
                let obs: Vec<Vec<f64>> = (0..N)
                    .map(|i| grid.iter().map(|x| x[i]).collect())
                    .collect();
                let obs_ref: Vec<&[f64]> = obs.iter().map(|x| &x[..]).collect();

                let nvals = MultiBsplineRectilinear::<f64, N>::coeff_storage_len(dims);
                let scratch_len = MultiBsplineRectilinear::<f64, N>::construction_scratch_len(dims);
                let mut coeffs = vec![0.0; nvals];
                let mut scratch = vec![0.0; scratch_len];
                let grids_ref: &[&[f64]; N] = grids.as_slice().try_into().unwrap();
                let obs_ref: &[&[f64]; N] = obs_ref.as_slice().try_into().unwrap();
                let interp: MultiBsplineRectilinear<'_, f64, N> =
                    MultiBsplineRectilinear::from_values_with_workspace(
                        grids_ref,
                        &vals,
                        &mut coeffs,
                        &mut scratch,
                        false,
                    )
                    .unwrap();

                let mut out = vec![0.0; vals.len()];
                interp.interp(obs_ref, &mut out).unwrap();

                for i in 0..vals.len() {
                    assert!((vals[i] - out[i]).abs() < 1e-9);
                }
                Ok(())
            })
            .unwrap();
        }
    }

    #[test]
    fn test_uniform_grid_matches_regular() {
        let dims = [6_usize, 5];
        let starts = [-0.5_f64, 1.25];
        let steps = [0.4_f64, 0.7];
        let xs: Vec<Vec<f64>> = (0..2)
            .map(|i| {
                (0..dims[i])
                    .map(|j| starts[i] + steps[i] * j as f64)
                    .collect()
            })
            .collect();
        let grids: Vec<&[f64]> = xs.iter().map(|x| &x[..]).collect();
        let grids_ref: &[&[f64]; 2] = grids.as_slice().try_into().unwrap();
        let grid = meshgrid(vec![&xs[0], &xs[1]]);
        let vals: Vec<f64> = grid
            .iter()
            .map(|x| x[0] * x[0] + 0.25 * x[1] * x[1] + x[0] * x[1])
            .collect();

        let nvals = MultiBsplineRectilinear::<f64, 2>::coeff_storage_len(dims);
        let scratch_len = MultiBsplineRectilinear::<f64, 2>::construction_scratch_len(dims);
        let mut rect_coeffs = vec![0.0; nvals];
        let mut rect_scratch = vec![0.0; scratch_len];
        coefficients::<f64, 2>(grids_ref, &vals, &mut rect_coeffs, &mut rect_scratch).unwrap();

        let mut reg_coeffs = vec![0.0; nvals];
        let mut reg_scratch =
            vec![0.0; MultiBsplineRegular::<f64, 2>::construction_scratch_len(dims)];
        crate::multibspline::regular::coefficients(dims, &vals, &mut reg_coeffs, &mut reg_scratch)
            .unwrap();

        for i in 0..nvals {
            assert!((rect_coeffs[i] - reg_coeffs[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn test_quadratic_truth_with_extrapolation() {
        let x = [-1.0, -0.35, 0.2, 0.95, 1.8, 3.0];
        let y = [-2.0, -1.25, -0.1, 0.55, 1.7];
        let grids = [&x[..], &y[..]];
        let dims = [x.len(), y.len()];
        let x_vec = x.to_vec();
        let y_vec = y.to_vec();
        let grid = meshgrid(vec![&x_vec, &y_vec]);
        let truth = |p: &[f64]| 0.5 * p[0] * p[0] + 0.25 * p[1] * p[1] + p[0] * p[1] - 0.3;
        let vals: Vec<f64> = grid.iter().map(|p| truth(p)).collect();

        let nvals = MultiBsplineRectilinear::<f64, 2>::coeff_storage_len(dims);
        let scratch_len = MultiBsplineRectilinear::<f64, 2>::construction_scratch_len(dims);
        let mut coeffs = vec![0.0; nvals];
        let mut scratch = vec![0.0; scratch_len];
        let interp = MultiBsplineRectilinear::from_values_with_workspace(
            &grids,
            &vals,
            &mut coeffs,
            &mut scratch,
            false,
        )
        .unwrap();

        let xobs = [-1.5, -0.7, 0.0, 1.2, 3.4];
        let yobs = [-2.4, -0.9, 0.2, 1.2, 2.2];
        let obs = [&xobs[..], &yobs[..]];
        let mut out = vec![0.0; xobs.len()];
        interp.interp(&obs, &mut out).unwrap();

        for i in 0..out.len() {
            let expected = truth(&[xobs[i], yobs[i]]);
            assert!(
                (out[i] - expected).abs() < 1e-9,
                "{} != {} at {i}",
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_linearized_extrapolation_is_linear_outside_grid() {
        let x = [-1.0_f64, -0.2, 0.35, 1.4, 2.8, 4.0];
        let grids = [&x[..]];
        let dims = [x.len()];
        let vals: Vec<f64> = x
            .iter()
            .map(|&x| x * x * x - 0.5 * x * x + 2.0 * x)
            .collect();
        let mut coeffs = vec![0.0; MultiBsplineRectilinear::<f64, 1>::coeff_storage_len(dims)];
        let mut scratch =
            vec![0.0; MultiBsplineRectilinear::<f64, 1>::construction_scratch_len(dims)];

        let interp = MultiBsplineRectilinear::from_values_with_workspace(
            &grids,
            &vals,
            &mut coeffs,
            &mut scratch,
            true,
        )
        .unwrap();

        assert_linear_extrapolation([
            interp.interp_one([-1.0]).unwrap(),
            interp.interp_one([-1.6]).unwrap(),
            interp.interp_one([-2.2]).unwrap(),
        ]);
        assert_linear_extrapolation([
            interp.interp_one([4.0]).unwrap(),
            interp.interp_one([4.7]).unwrap(),
            interp.interp_one([5.4]).unwrap(),
        ]);
    }
}
