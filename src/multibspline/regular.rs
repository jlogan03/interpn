//! An arbitrary-dimensional cubic B-spline interpolator / extrapolator on a regular grid.

use super::Saturation;
#[cfg(feature = "par")]
use super::max_usize;
use crate::{index_arr_fixed_dims, interp_math::dot4, mul_add};
use crunchy::unroll;
use num_traits::{Float, NumCast};

/// Evaluate cubic B-spline interpolation on a regular grid in up to 8 dimensions.
/// Assumes C-style ordering of coeffs (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
///
/// This function expects precomputed B-spline coefficients. To build coefficients from
/// nodal values without allocation, use [`MultiBsplineRegular::from_values_with_workspace`].
pub fn interpn<T: Float>(
    dims: &[usize],
    starts: &[T],
    steps: &[T],
    coeffs: &[T],
    linearize_extrapolation: bool,
    obs: &[&[T]],
    out: &mut [T],
) -> Result<(), &'static str> {
    let ndims = dims.len();
    if starts.len() != ndims || steps.len() != ndims || obs.len() != ndims {
        return Err("Dimension mismatch");
    }

    crate::dispatch_ndims!(
        ndims,
        "Dimension exceeds maximum (8). Use interpolator struct directly for higher dimensions.",
        [1, 2, 3, 4, 5, 6, 7, 8],
        |N| {
            MultiBsplineRegular::<'_, T, N>::new(
                dims.try_into().unwrap(),
                starts.try_into().unwrap(),
                steps.try_into().unwrap(),
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
    dims: &[usize],
    starts: &[T],
    steps: &[T],
    coeffs: &[T],
    linearize_extrapolation: bool,
    obs: &[&[T]],
) -> Result<Vec<T>, &'static str> {
    let mut out = vec![T::zero(); obs[0].len()];
    interpn(
        dims,
        starts,
        steps,
        coeffs,
        linearize_extrapolation,
        obs,
        &mut out,
    )?;
    Ok(out)
}

/// Construct cubic B-spline coefficients from nodal values on a regular grid.
///
/// The input values are immutable. The generated coefficients are written into `coeffs`.
/// The `scratch` buffer must have length at least
/// [`MultiBsplineRegular::construction_scratch_len`] for the provided dimensions.
pub fn coefficients<T: Float, const N: usize>(
    dims: [usize; N],
    vals: &[T],
    coeffs: &mut [T],
    scratch: &mut [T],
) -> Result<(), &'static str> {
    check_dims(dims, coeffs)?;
    if vals.len() != coeffs.len() {
        return Err("Dimension mismatch");
    }

    let scratch_len = MultiBsplineRegular::<T, N>::construction_scratch_len(dims);
    if scratch.len() < scratch_len {
        return Err("Scratch buffer is too small");
    }

    coeffs.copy_from_slice(vals);

    let max_dim = max_dim(dims);
    let (upper, rhs) = scratch.split_at_mut(max_dim);

    let mut dimprod = [1_usize; N];
    populate_dimprod(dims, &mut dimprod);

    for axis in 0..N {
        solve_axis(dims, dimprod, axis, coeffs, upper, rhs)?;
    }

    Ok(())
}

/// Construct cubic B-spline coefficients from nodal values on a regular grid,
/// solving independent coefficient slabs in parallel.
///
/// This is the nonallocating parallel construction path. The `scratch` buffer
/// must have length at least
/// [`MultiBsplineRegular::parallel_construction_scratch_len`] for the provided
/// dimensions and `max_threads`.
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
    dims: [usize; N],
    vals: &[T],
    coeffs: &mut [T],
    scratch: &mut [T],
    max_threads: usize,
) -> Result<(), &'static str> {
    check_dims(dims, coeffs)?;
    if vals.len() != coeffs.len() {
        return Err("Dimension mismatch");
    }

    let scratch_len =
        MultiBsplineRegular::<T, N>::parallel_construction_scratch_len(dims, max_threads);
    if scratch_len == 0 || scratch.len() < scratch_len {
        return Err("Scratch buffer is too small");
    }

    coeffs.copy_from_slice(vals);

    let mut dimprod = [1_usize; N];
    populate_dimprod(dims, &mut dimprod);

    let tasks = max_threads.max(1).min(rayon::current_num_threads()).max(1);
    for axis in 0..N {
        solve_axis_par(dims, dimprod, axis, coeffs, scratch, tasks)?;
    }

    Ok(())
}

/// An N-dimensional cubic B-spline interpolator / extrapolator on a regular grid.
///
/// Assumes C-style ordering of coeffs (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
pub struct MultiBsplineRegular<'a, T: Float, const N: usize> {
    /// Size of each dimension
    dims: [usize; N],

    /// Starting point of each dimension
    starts: [T; N],

    /// Step size for each dimension
    steps: [T; N],

    /// B-spline coefficients, size prod(dims)
    coeffs: &'a [T],

    /// Whether to extrapolate linearly instead of continuing the boundary segment
    linearize_extrapolation: bool,
}

impl<'a, T: Float, const N: usize> MultiBsplineRegular<'a, T, N> {
    /// Number of coefficients required for `dims`.
    ///
    /// Returns zero if any dimension is invalid for this interpolator, if `N == 0`,
    /// or if the product overflows `usize`.
    pub const fn coeff_storage_len(dims: [usize; N]) -> usize {
        if N == 0 {
            return 0;
        }

        let mut out = 1_usize;
        let mut i = 0;
        while i < N {
            if dims[i] < 4 {
                return 0;
            }
            match out.checked_mul(dims[i]) {
                Some(v) => out = v,
                None => return 0,
            }
            i += 1;
        }

        out
    }

    /// Number of scratch values required to construct coefficients for `dims`.
    ///
    /// Returns zero if any dimension is invalid for this interpolator, if `N == 0`,
    /// or if the scratch length overflows `usize`.
    pub const fn construction_scratch_len(dims: [usize; N]) -> usize {
        if N == 0 {
            return 0;
        }

        let mut max = 0_usize;
        let mut i = 0;
        while i < N {
            if dims[i] < 4 {
                return 0;
            }
            if dims[i] > max {
                max = dims[i];
            }
            i += 1;
        }

        match max.checked_mul(2) {
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
        let mut max_scratch = 0_usize;
        let mut prefix_slabs = 1_usize;
        let mut axis = 0;
        while axis < N {
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
            if scratch > max_scratch {
                max_scratch = scratch;
            }

            match prefix_slabs.checked_mul(dims[axis]) {
                Some(v) => prefix_slabs = v,
                None => return 0,
            }
            axis += 1;
        }

        max_scratch
    }

    /// Build an interpolator from precomputed coefficients.
    ///
    /// # Errors
    /// * If dimensions do not match
    /// * If any dimension has fewer than four entries
    /// * If any step size is zero or negative
    pub fn new(
        dims: [usize; N],
        starts: [T; N],
        steps: [T; N],
        coeffs: &'a [T],
        linearize_extrapolation: bool,
    ) -> Result<Self, &'static str> {
        check_dims(dims, coeffs)?;
        if !steps.iter().all(|&x| x > T::zero()) {
            return Err("All grids must be monotonically increasing");
        }

        let mut steps_local = [T::zero(); N];
        let mut starts_local = [T::zero(); N];
        let mut dims_local = [0_usize; N];
        steps_local[..N].copy_from_slice(&steps[..N]);
        starts_local[..N].copy_from_slice(&starts[..N]);
        dims_local[..N].copy_from_slice(&dims[..N]);

        Ok(Self {
            dims: dims_local,
            starts: starts_local,
            steps: steps_local,
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
        dims: [usize; N],
        starts: [T; N],
        steps: [T; N],
        vals: &[T],
        coeffs: &'a mut [T],
        scratch: &mut [T],
        linearize_extrapolation: bool,
    ) -> Result<Self, &'static str> {
        coefficients(dims, vals, coeffs, scratch)?;
        Self::new(dims, starts, steps, coeffs, linearize_extrapolation)
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
        dims: [usize; N],
        starts: [T; N],
        steps: [T; N],
        vals: &[T],
        coeffs: &'a mut [T],
        scratch: &mut [T],
        max_threads: usize,
        linearize_extrapolation: bool,
    ) -> Result<Self, &'static str>
    where
        T: Send + Sync,
    {
        coefficients_par(dims, vals, coeffs, scratch, max_threads)?;
        Self::new(dims, starts, steps, coeffs, linearize_extrapolation)
    }

    /// Interpolate on a contiguous list of observation points.
    ///
    /// Assumes C-style ordering of coeffs.
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
    ///
    /// Uses fixed-size intermediate storage of O(N) and no allocation.
    pub fn interp_one(&self, x: [T; N]) -> Result<T, &'static str> {
        let mut origin = [0_usize; N];
        let mut sat = [Saturation::None; N];
        let mut weights = [[T::zero(); FP]; N];
        let mut dimprod = [1_usize; N];
        let mut loc = [0_usize; N];
        let mut store = [[T::zero(); FP]; N];

        populate_dimprod(self.dims, &mut dimprod);

        for i in 0..N {
            (origin[i], sat[i]) = self.get_loc(x[i], i)?;
            let origin_f =
                <T as NumCast>::from(origin[i] + 1).ok_or("Unrepresentable coordinate value")?;
            let index_one_loc = mul_add(self.steps[i], origin_f, self.starts[i]);
            let t = (x[i] - index_one_loc) / self.steps[i];
            weights[i] = interp_weights(t, sat[i], self.linearize_extrapolation);
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

    /// Get the two-lower index along this dimension where `x` is found,
    /// saturating to the bounds at the edges if necessary.
    #[inline]
    fn get_loc(&self, v: T, dim: usize) -> Result<(usize, Saturation), &'static str> {
        let floc = ((v - self.starts[dim]) / self.steps[dim]).floor();
        let iloc = <isize as NumCast>::from(floc).ok_or("Unrepresentable coordinate value")? - 1;

        let n = self.dims[dim] as isize;
        let dimmax = n.saturating_sub(4).max(0);
        let loc = iloc.max(0).min(dimmax) as usize;

        let saturation = if iloc < -1 {
            Saturation::OutsideLow
        } else if iloc == -1 {
            Saturation::InsideLow
        } else if iloc > n - 3 {
            Saturation::OutsideHigh
        } else if iloc == n - 3 {
            Saturation::InsideHigh
        } else {
            Saturation::None
        };

        Ok((loc, saturation))
    }
}

const FP: usize = 4;

fn check_dims<T: Float, const N: usize>(dims: [usize; N], data: &[T]) -> Result<(), &'static str> {
    let nvals = MultiBsplineRegular::<T, N>::coeff_storage_len(dims);
    if nvals == 0 || data.len() != nvals {
        return Err("Dimension mismatch");
    }
    Ok(())
}

const fn max_dim<const N: usize>(dims: [usize; N]) -> usize {
    let mut max = 0_usize;
    let mut i = 0;
    while i < N {
        if dims[i] > max {
            max = dims[i];
        }
        i += 1;
    }
    max
}

fn populate_dimprod<const N: usize>(dims: [usize; N], dimprod: &mut [usize; N]) {
    let mut acc = 1;
    for i in 0..N {
        if i > 0 {
            acc *= dims[N - i];
        }
        dimprod[N - i - 1] = acc;
    }
}

fn solve_axis<T: Float, const N: usize>(
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
        solve_line(base, stride, n, coeffs, upper, rhs)?;
    }

    Ok(())
}

#[cfg(feature = "par")]
fn solve_axis_par<T: Float + Send + Sync, const N: usize>(
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
        solve_axis_slabs(coeffs, slab_len, stride, n, upper, rhs)
    } else {
        solve_axis_slabs_par(coeffs, slab_len, stride, n, scratch, tasks)
    }
}

#[cfg(feature = "par")]
fn solve_axis_slabs_par<T: Float + Send + Sync>(
    coeffs: &mut [T],
    slab_len: usize,
    stride: usize,
    n: usize,
    scratch: &mut [T],
    tasks: usize,
) -> Result<(), &'static str> {
    let nslabs = coeffs.len() / slab_len;
    if tasks <= 1 || nslabs <= 1 {
        let (upper, rhs) = scratch.split_at_mut(n);
        return solve_axis_slabs(coeffs, slab_len, stride, n, upper, rhs);
    }

    let left_slabs = nslabs / 2;
    let left_tasks = tasks / 2;
    let right_tasks = tasks - left_tasks;

    let coeff_split = left_slabs * slab_len;
    let scratch_split = 2 * n * left_tasks;
    let (left_coeffs, right_coeffs) = coeffs.split_at_mut(coeff_split);
    let (left_scratch, right_scratch) = scratch.split_at_mut(scratch_split);

    let (left, right) = rayon::join(
        || solve_axis_slabs_par(left_coeffs, slab_len, stride, n, left_scratch, left_tasks),
        || {
            solve_axis_slabs_par(
                right_coeffs,
                slab_len,
                stride,
                n,
                right_scratch,
                right_tasks,
            )
        },
    );
    left?;
    right
}

#[cfg(feature = "par")]
fn solve_axis_slabs<T: Float>(
    coeffs: &mut [T],
    slab_len: usize,
    stride: usize,
    n: usize,
    upper: &mut [T],
    rhs: &mut [T],
) -> Result<(), &'static str> {
    for slab in coeffs.chunks_mut(slab_len) {
        for base in 0..stride {
            solve_line(base, stride, n, slab, upper, rhs)?;
        }
    }

    Ok(())
}

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

/// Solve one regular-grid cubic B-spline coefficient line.
///
/// The cubic cardinal B-spline interpolation equations for interior nodes are:
///
/// ```text
/// (c[i - 1] + 4c[i] + c[i + 1]) / 6 = y[i]
/// ```
///
/// At the endpoints, we use one ghost coefficient on each side and impose zero
/// third derivative at the boundary interval:
///
/// ```text
/// -c[-1] + 3c[0] - 3c[1] + c[2] = 0
/// c[-1] = 3c[0] - 3c[1] + c[2]
///
/// -c[n - 3] + 3c[n - 2] - 3c[n - 1] + c[n] = 0
/// c[n] = c[n - 3] - 3c[n - 2] + 3c[n - 1]
/// ```
///
/// Substituting those ghosts into the endpoint interpolation equations gives:
///
/// ```text
/// 7c[0] - 2c[1] + c[2] = 6y[0]
/// c[n - 3] - 2c[n - 2] + 7c[n - 1] = 6y[n - 1]
/// ```
///
/// These are not tridiagonal as written. Combining each with its adjacent
/// interior row gives equivalent tridiagonal boundary rows:
///
/// ```text
/// c[0] - c[1] = y[0] - y[1]
/// -c[n - 2] + c[n - 1] = y[n - 1] - y[n - 2]
/// ```
///
/// Therefore the Thomas solve uses:
///
/// ```text
/// row 0:      diag =  1, upper = -1, rhs = y[0] - y[1]
/// row 1..n-2: lower = 1, diag = 4, upper = 1, rhs = 6y[i]
/// row n-1:    lower = -1, diag = 1, rhs = y[n - 1] - y[n - 2]
/// ```
fn solve_line<T: Float>(
    base: usize,
    stride: usize,
    n: usize,
    coeffs: &mut [T],
    upper: &mut [T],
    rhs: &mut [T],
) -> Result<(), &'static str> {
    if n < 4 || upper.len() < n || rhs.len() < n {
        return Err("Dimension mismatch");
    }

    let one = T::one();
    let two = one + one;
    let four = two + two;
    let six = four + two;

    let y0 = coeffs[base];
    let y1 = coeffs[base + stride];
    upper[0] = -one;
    rhs[0] = y0 - y1;

    for i in 1..n {
        let last = i == n - 1;
        let lower = if last { -one } else { one };
        let diag = if last { one } else { four };
        let upper_i = if last { T::zero() } else { one };
        let y = if last {
            coeffs[base + i * stride] - coeffs[base + (i - 1) * stride]
        } else {
            six * coeffs[base + i * stride]
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

#[inline]
fn interp_weights<T: Float>(t: T, sat: Saturation, linearize_extrapolation: bool) -> [T; 4] {
    match sat {
        Saturation::None => cubic_bspline_weights(t),
        Saturation::InsideLow => low_boundary_weights(t + T::one(), false),
        Saturation::OutsideLow => low_boundary_weights(t + T::one(), linearize_extrapolation),
        Saturation::InsideHigh => high_boundary_weights(t - T::one(), false),
        Saturation::OutsideHigh => high_boundary_weights(t - T::one(), linearize_extrapolation),
    }
}

#[inline]
fn cubic_bspline_weights<T: Float>(t: T) -> [T; 4] {
    let one = T::one();
    let two = one + one;
    let three = two + one;
    let six = three + three;

    let t2 = t * t;
    let t3 = t2 * t;
    [
        (one - three * t + three * t2 - t3) / six,
        ((two + two) - six * t2 + three * t3) / six,
        (one + three * t + three * t2 - three * t3) / six,
        t3 / six,
    ]
}

#[inline]
fn low_boundary_weights<T: Float>(t: T, linearize_extrapolation: bool) -> [T; 4] {
    let raw = if linearize_extrapolation {
        low_linearized_boundary_weights(t)
    } else {
        cubic_bspline_weights(t)
    };

    // The low-side ghost coefficient is 3*c0 - 3*c1 + c2. Fold the
    // ghost's contribution into the four stored coefficients.
    let three = T::one() + T::one() + T::one();
    [
        mul_add(three, raw[0], raw[1]),
        mul_add(-three, raw[0], raw[2]),
        raw[0] + raw[3],
        T::zero(),
    ]
}

#[inline]
fn high_boundary_weights<T: Float>(t: T, linearize_extrapolation: bool) -> [T; 4] {
    let raw = if linearize_extrapolation {
        high_linearized_boundary_weights(t)
    } else {
        cubic_bspline_weights(t)
    };

    // The high-side ghost coefficient is c1 - 3*c2 + 3*c3. Fold the
    // ghost's contribution into the four stored coefficients.
    let three = T::one() + T::one() + T::one();
    [
        T::zero(),
        raw[0] + raw[3],
        mul_add(-three, raw[3], raw[1]),
        mul_add(three, raw[3], raw[2]),
    ]
}

#[inline]
fn low_linearized_boundary_weights<T: Float>(t: T) -> [T; 4] {
    let one = T::one();
    let two = one + one;
    let three = two + one;
    let six = three + three;

    [
        one / six - t / two,
        (two + two) / six,
        one / six + t / two,
        T::zero(),
    ]
}

#[inline]
fn high_linearized_boundary_weights<T: Float>(t: T) -> [T; 4] {
    let one = T::one();
    let two = one + one;
    let three = two + one;
    let six = three + three;
    let u = t - one;

    [
        T::zero(),
        one / six - u / two,
        (two + two) / six,
        one / six + u / two,
    ]
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::{linspace, meshgrid};

    fn reconstruct_values<const N: usize>(dims: [usize; N], coeffs: &[f64]) -> Vec<f64> {
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
                    out[base + i * stride] = if i == 0 {
                        let ghost = 3.0 * tmp[0] - 3.0 * tmp[1] + tmp[2];
                        (ghost + 4.0 * tmp[0] + tmp[1]) / 6.0
                    } else if i == n - 1 {
                        let ghost = tmp[n - 3] - 3.0 * tmp[n - 2] + 3.0 * tmp[n - 1];
                        (tmp[n - 2] + 4.0 * tmp[n - 1] + ghost) / 6.0
                    } else {
                        (tmp[i - 1] + 4.0 * tmp[i] + tmp[i + 1]) / 6.0
                    };
                }
            }
        }

        out
    }

    #[test]
    fn test_storage_lengths() {
        assert_eq!(MultiBsplineRegular::<f64, 2>::coeff_storage_len([4, 5]), 20);
        assert_eq!(
            MultiBsplineRegular::<f64, 2>::construction_scratch_len([4, 5]),
            10
        );
        #[cfg(feature = "par")]
        assert_eq!(
            MultiBsplineRegular::<f64, 2>::parallel_construction_scratch_len([4, 5], 4),
            40
        );
        assert_eq!(MultiBsplineRegular::<f64, 2>::coeff_storage_len([3, 5]), 0);
        assert_eq!(
            MultiBsplineRegular::<f64, 2>::construction_scratch_len([3, 5]),
            0
        );
        #[cfg(feature = "par")]
        assert_eq!(
            MultiBsplineRegular::<f64, 2>::parallel_construction_scratch_len([4, 5], 0),
            MultiBsplineRegular::<f64, 2>::parallel_construction_scratch_len([4, 5], 1)
        );
    }

    #[cfg(feature = "par")]
    #[test]
    fn test_parallel_coefficients_match_serial() {
        let dims = [5_usize, 6, 7];
        let nvals = MultiBsplineRegular::<f64, 3>::coeff_storage_len(dims);
        let serial_scratch_len = MultiBsplineRegular::<f64, 3>::construction_scratch_len(dims);
        let parallel_scratch_len =
            MultiBsplineRegular::<f64, 3>::parallel_construction_scratch_len(dims, 4);
        let xs: Vec<Vec<f64>> = (0..3)
            .map(|i| linspace(-1.0 + i as f64, 2.0 + i as f64, dims[i]))
            .collect();
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

        let mut serial_coeffs = vec![0.0; nvals];
        let mut serial_scratch = vec![0.0; serial_scratch_len];
        coefficients(dims, &vals, &mut serial_coeffs, &mut serial_scratch).unwrap();

        let mut parallel_coeffs = vec![0.0; nvals];
        let mut parallel_scratch = vec![0.0; parallel_scratch_len];
        coefficients_par(dims, &vals, &mut parallel_coeffs, &mut parallel_scratch, 4).unwrap();

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
                let nvals = MultiBsplineRegular::<f64, N>::coeff_storage_len(dims);
                let scratch_len = MultiBsplineRegular::<f64, N>::construction_scratch_len(dims);

                let xs: Vec<Vec<f64>> = (0..N)
                    .map(|i| linspace(-1.5 * (i as f64 + 1.0), 2.0 * (i as f64 + 1.0), dims[i]))
                    .collect();
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

                let mut coeffs = vec![0.0; nvals];
                let mut scratch = vec![0.0; scratch_len];
                coefficients(dims, &vals, &mut coeffs, &mut scratch).unwrap();
                let reconstructed = reconstruct_values(dims, &coeffs);

                for i in 0..vals.len() {
                    assert!(
                        (vals[i] - reconstructed[i]).abs() < 1e-10,
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
                let nvals = MultiBsplineRegular::<f64, N>::coeff_storage_len(dims);
                let scratch_len = MultiBsplineRegular::<f64, N>::construction_scratch_len(dims);
                let starts = [0.0_f64; N];
                let steps = [0.5_f64; N];

                let xs: Vec<Vec<f64>> = (0..N)
                    .map(|i| {
                        (0..dims[i])
                            .map(|j| starts[i] + steps[i] * j as f64)
                            .collect()
                    })
                    .collect();
                let grid = meshgrid((0..N).map(|i| &xs[i]).collect());
                let vals: Vec<f64> = grid
                    .iter()
                    .map(|x| x.iter().map(|v| v * v + 0.3 * v).sum())
                    .collect();
                let obs: Vec<Vec<f64>> = (0..N)
                    .map(|i| grid.iter().map(|x| x[i]).collect())
                    .collect();
                let obs_ref: Vec<&[f64]> = obs.iter().map(|x| &x[..]).collect();

                let mut coeffs = vec![0.0; nvals];
                let mut scratch = vec![0.0; scratch_len];
                let interp = MultiBsplineRegular::from_values_with_workspace(
                    dims,
                    starts,
                    steps,
                    &vals,
                    &mut coeffs,
                    &mut scratch,
                    false,
                )
                .unwrap();

                let mut out = vec![0.0; vals.len()];
                interp
                    .interp(obs_ref.as_slice().try_into().unwrap(), &mut out)
                    .unwrap();

                for i in 0..vals.len() {
                    assert!((vals[i] - out[i]).abs() < 1e-10);
                }
                Ok(())
            })
            .unwrap();
        }
    }

    #[test]
    fn test_linearized_extrapolation_is_linear() {
        let dims = [6_usize];
        let starts = [0.0_f64];
        let steps = [1.0_f64];
        let vals: Vec<f64> = (0..dims[0]).map(|i| (i as f64) * (i as f64)).collect();
        let mut coeffs = vec![0.0; MultiBsplineRegular::<f64, 1>::coeff_storage_len(dims)];
        let mut scratch = vec![0.0; MultiBsplineRegular::<f64, 1>::construction_scratch_len(dims)];

        let interp = MultiBsplineRegular::from_values_with_workspace(
            dims,
            starts,
            steps,
            &vals,
            &mut coeffs,
            &mut scratch,
            true,
        )
        .unwrap();

        let y_hi = interp.interp_one([5.0]).unwrap();
        let y_hi_far = interp.interp_one([6.0]).unwrap();
        let y_hi_farther = interp.interp_one([7.0]).unwrap();
        assert!(((y_hi_farther - y_hi_far) - (y_hi_far - y_hi)).abs() < 1e-12);
    }
}
