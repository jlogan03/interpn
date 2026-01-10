//! N-dimensional interpolation/extrapolation methods, no-std and no-alloc compatible,
//! prioritizing correctness, performance, and compatiblity with memory-constrained environments.
//!
//! # Performance Scalings
//! Note that for a self-consistent multidimensional linear interpolation, there are 2^ndims grid values that contribute
//! to each observation point, and as such, that is the theoretical floor for performance scaling. That said,
//! depending on the implementation, the constant term can vary by more than an order of magnitude.
//!
//! Cubic interpolations require two more degrees of freedom per dimension, and have a minimal runtime scaling of 4^ndims.
//! Similar to the linear methods, depending on implementation, the constant term can vary by orders of magnitude,
//! as can the RAM usage.
//!
//! Rectilinear methods perform a bisection search to find the relevant grid cell, which takes
//! a worst-case number of iterations of log2(number of grid elements).
//!
//! | Method                        | RAM       | Interp. / Extrap. Cost       |
//! |-------------------------------|-----------|------------------------------|
//! | multilinear::regular          | O(ndims)  | O(2^ndims)                   |
//! | multilinear::rectilinear      | O(ndims)  | O(2^ndims) + log2(gridsize)  |
//! | multicubic::regular           | O(ndims)  | O(4^ndims)                   |
//! | multicubic::rectilinear       | O(ndims)  | O(4^ndims) + log2(gridsize)  |
//!
//! # Example: Multilinear and Multicubic w/ Regular Grid
//! ```rust
//! use interpn::{multilinear, multicubic};
//!
//! // Define a grid
//! let x = [1.0_f64, 2.0, 3.0, 4.0];
//! let y = [0.0_f64, 1.0, 2.0, 3.0];
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
//! let z = [2.0; 16];
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
//! multilinear::regular::interpn(&dims, &starts, &steps, &z, &obs, &mut out);
//! multicubic::regular::interpn(&dims, &starts, &steps, &z, false, &obs, &mut out);
//! ```
//!
//! # Example: Multilinear and Multicubic w/ Rectilinear Grid
//! ```rust
//! use interpn::{multilinear, multicubic};
//!
//! // Define a grid
//! let x = [1.0_f64, 2.0, 3.0, 4.0];
//! let y = [0.0_f64, 1.0, 2.0, 3.0];
//!
//! // Grid input for rectilinear method
//! let grids = &[&x[..], &y[..]];
//!
//! // Values at grid points
//! let z = [2.0; 16];
//!
//! // Points to interpolate/extrapolate
//! let xobs = [0.0_f64, 5.0];
//! let yobs = [-1.0, 3.0];
//! let obs = [&xobs[..], &yobs[..]];
//!
//! // Storage for output
//! let mut out = [0.0; 2];
//!
//! // Do interpolation
//! multilinear::rectilinear::interpn(grids, &z, &obs, &mut out).unwrap();
//! multicubic::rectilinear::interpn(grids, &z, false, &obs, &mut out).unwrap();
//! ```
//!
//! # Development Roadmap
//! * Methods for unstructured triangular and tetrahedral meshes
#![cfg_attr(not(feature = "std"), no_std)]
// These "needless" range loops are a significant speedup
#![allow(clippy::needless_range_loop)]
// Some const loops produce flattened code with unresolvable lints on
// expanded code that is entirely in const.
#![allow(clippy::absurd_extreme_comparisons)]

use num_traits::Float;

pub mod multilinear;
pub use multilinear::{MultilinearRectilinear, MultilinearRegular};

pub mod multicubic;
pub use multicubic::{MulticubicRectilinear, MulticubicRegular};

pub mod linear {
    pub use crate::multilinear::rectilinear;
    pub use crate::multilinear::regular;
}

pub mod cubic {
    pub use crate::multicubic::rectilinear;
    pub use crate::multicubic::regular;
}

pub mod nearest;
pub use nearest::{NearestRectilinear, NearestRegular};

pub mod one_dim;
pub use one_dim::{
    RectilinearGrid1D, RegularGrid1D, hold::Left1D, hold::Nearest1D, hold::Right1D,
    linear::Linear1D, linear::LinearHoldLast1D,
};

#[cfg(feature = "par")]
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

#[cfg(feature = "par")]
use std::sync::{LazyLock, Mutex};

#[cfg(feature = "std")]
pub mod utils;

#[cfg(all(test, feature = "std"))]
pub(crate) mod testing;

#[cfg(feature = "python")]
pub mod python;

/// Interpolant function for multi-dimensional methods.
#[derive(Clone, Copy)]
pub enum GridInterpMethod {
    /// Multi-linear interpolation.
    Linear,
    /// Cubic Hermite spline interpolation.
    Cubic,
    /// Nearest-neighbor interpolation.
    Nearest,
}

/// Grid spacing category for multi-dimensional methods.
#[derive(Clone, Copy)]
pub enum GridKind {
    /// Evenly-spaced points along each axis.
    Regular,
    /// Un-evenly spaced points along each axis.
    Rectilinear,
}

const MAXDIMS: usize = 8;
const MAXDIMS_ERR: &str =
    "Dimension exceeds maximum (8). Use interpolator struct directly for higher dimensions.";
const MIN_CHUNK_SIZE: usize = 1024;

/// The number of physical cores present on the machine;
/// initialized once, then never again, because each call involves some file I/O
/// and allocations that can be slower than the function call that they support.
///
/// On subsequent accesses, each access is an atomic load without any waiting paths.
///
/// This lock can only be contended if multiple threads attempt access
/// before it is initialized; in that case, the waiting threads may park
/// until initialization is complete, which can cause ~20us delays
/// on first access only.
#[cfg(feature = "par")]
static PHYSICAL_CORES: LazyLock<usize> = LazyLock::new(num_cpus::get_physical);

/// Evaluate multidimensional interpolation on a regular grid in up to 8 dimensions.
/// Assumes C-style ordering of vals (z(x0, y0), z(x0, y1), ..., z(x0, yn), z(x1, y0), ...).
///
/// For lower dimensions, a fast flattened method is used. For higher dimensions, where that flattening
/// becomes impractical due to compile times and instruction size, evaluation defers to a bounded
/// recursion.
/// The linear method uses the flattening for 1-6 dimensions, while
/// flattened cubic methods are available up to 3 dimensions by default and up to 4 dimensions
/// with the `deep_unroll` feature enabled.
///
/// This is a convenience function; best performance will be achieved by using the exact right
/// number for the N parameter, as this will slightly reduce compute and storage overhead,
/// and the underlying method can be extended to more than this function's limit of 8 dimensions.
/// The limit of 8 dimensions was chosen for no more specific reason than to reduce unit test times.
///
/// While this method initializes the interpolator struct on every call, the overhead of doing this
/// is minimal even when using it to evaluate one observation point at a time.
///
/// Like most grid search algorithms (including in the standard library), the uniqueness and
/// monotonicity of the grid is the responsibility of the user, because checking it is often much
/// more expensive than the algorithm that we will perform on it. Behavior with ill-posed grids
/// is undefined.
///
/// #### Args:
///
/// * `grids`: `N` slices of each axis' grid coordinates. Must be unique and monotonically increasing.
/// * `vals`:  Flattened `N`-dimensional array of data values at each grid point in C-style order.
///          Must be the same length as the cartesian product of the grids, (n_x * n_y * ...).
/// * `obs`:   `N` slices of Observation points where the interpolant should be evaluated.
///          Must be of equal length.
/// * `out`:   Pre-allocated output buffer to place the resulting values.
///          Must  be the same length as each of the `obs` slices.
/// * `method`: Choice of interpolant function.
/// * `assume_grid_kind`: Whether to assume the grid is regular (evenly-spaced),
///                     rectilinear (un-evenly spaced), or make no assumption.
///                     If an assumption is provided, this bypasses a check of each
///                     grid, which can be a major speedup in some cases.
/// * `linearize_extrapolation`: Whether cubic methods should extrapolate linearly instead of the default
///                            quadratic extrapolation. Linearization is recommended to prevent
///                            the interpolant from diverging to extremes outside the grid.
/// * `check_bounds_with_atol`: If provided, return an error if any observation points are outside the grid
///                           by an amount exceeding the provided tolerance.
/// * `max_threads`: If provided, limit number of threads used to at most this number. Otherwise,
///                use a heuristic to choose the number that will provide the best throughput.
#[cfg(feature = "par")]
pub fn interpn<T: Float + Send + Sync>(
    grids: &[&[T]],
    vals: &[T],
    obs: &[&[T]],
    out: &mut [T],
    method: GridInterpMethod,
    assume_grid_kind: Option<GridKind>,
    linearize_extrapolation: bool,
    check_bounds_with_atol: Option<T>,
    max_threads: Option<usize>,
) -> Result<(), &'static str> {
    let ndims = grids.len();
    if ndims > MAXDIMS {
        return Err(MAXDIMS_ERR);
    }
    let n = out.len();

    // Resolve grid kind, checking the grid if the kind is not provided by the user.
    // We do this once at the top level so that the work is not repeated by each thread.
    let kind = resolve_grid_kind(assume_grid_kind, grids)?;

    // If there are enough points to justify it, run parallel
    if 2 * MIN_CHUNK_SIZE <= n {
        // Chunk for parallelism.
        //
        // By default, use only physical cores, because on most machines as of
        // 2026, only half the available cores represent real compute capability due to
        // the widespread adoption of hyperthreading. If a larger number is requested for
        // max_threads, that value is clamped to the total available threads so that we don't
        // queue chunks unnecessarily.
        //
        // We also use a minimum chunk size of 1024 as a heuristic, because below that limit,
        // single-threaded performance is usually faster due to a combination of thread spawning overhead,
        // memory page sizing, and improved vectorization over larger inputs.
        let num_cores_physical = *PHYSICAL_CORES; // Real cores, populated on first access
        let num_cores_pool = rayon::current_num_threads(); // Available cores from rayon thread pool
        let num_cores_available = num_cores_physical.min(num_cores_pool).max(1); // Real max
        let num_cores = match max_threads {
            Some(num_cores_requested) => num_cores_requested.min(num_cores_available),
            None => num_cores_available,
        };
        let chunk = MIN_CHUNK_SIZE.max(n / num_cores);

        // Make a shared error indicator
        let result: Mutex<Option<&'static str>> = Mutex::new(None);
        let write_err = |msg: &'static str| {
            let mut guard = result.lock().unwrap();
            if guard.is_none() {
                *guard = Some(msg);
            }
        };

        // Run threaded
        out.par_chunks_mut(chunk).enumerate().for_each(|(i, outc)| {
            // Calculate the start and end of observation point chunks
            let start = chunk * i;
            let end = start + outc.len();

            // Chunk observation points
            let mut obs_slices: [&[T]; 8] = [&[]; 8];
            for (j, o) in obs.iter().enumerate() {
                let s = &o.get(start..end);
                match s {
                    Some(s) => obs_slices[j] = s,
                    None => {
                        write_err("Dimension mismatch");
                        return;
                    }
                };
            }

            // Do interpolations
            let res_inner = interpn_serial(
                grids,
                vals,
                &obs_slices[..ndims],
                outc,
                method,
                Some(kind),
                linearize_extrapolation,
                check_bounds_with_atol,
            );

            match res_inner {
                Ok(()) => {}
                Err(msg) => write_err(msg),
            }
        });

        // Handle errors from threads
        match *result.lock().unwrap() {
            Some(msg) => Err(msg),
            None => Ok(()),
        }
    } else {
        // If there are not enough points to justify parallelism, run serial
        interpn_serial(
            grids,
            vals,
            obs,
            out,
            method,
            Some(kind),
            linearize_extrapolation,
            check_bounds_with_atol,
        )
    }
}

/// Allocating variant of [interpn].
/// It is recommended to pre-allocate outputs and use the non-allocating variant
/// whenever possible.
#[cfg(feature = "par")]
pub fn interpn_alloc<T: Float + Send + Sync>(
    grids: &[&[T]],
    vals: &[T],
    obs: &[&[T]],
    out: Option<Vec<T>>,
    method: GridInterpMethod,
    assume_grid_kind: Option<GridKind>,
    linearize_extrapolation: bool,
    check_bounds_with_atol: Option<T>,
    max_threads: Option<usize>,
) -> Result<Vec<T>, &'static str> {
    // Empty input -> empty output
    if obs.len() == 0 {
        return Ok(Vec::with_capacity(0));
    }

    // If output storage was not provided, build it now
    let mut out = out.unwrap_or_else(|| vec![T::zero(); obs[0].len()]);

    interpn(
        grids,
        vals,
        obs,
        &mut out,
        method,
        assume_grid_kind,
        linearize_extrapolation,
        check_bounds_with_atol,
        max_threads,
    )?;

    Ok(out)
}

/// Single-threaded, non-allocating variant of [interpn] available without `par` feature.
pub fn interpn_serial<T: Float>(
    grids: &[&[T]],
    vals: &[T],
    obs: &[&[T]],
    out: &mut [T],
    method: GridInterpMethod,
    assume_grid_kind: Option<GridKind>,
    linearize_extrapolation: bool,
    check_bounds_with_atol: Option<T>,
) -> Result<(), &'static str> {
    let ndims = grids.len();
    if ndims > MAXDIMS {
        return Err(MAXDIMS_ERR);
    }

    // Resolve grid kind, checking the grid if the kind is not provided by the user.
    let kind = resolve_grid_kind(assume_grid_kind, grids)?;

    // Extract regular grid params
    let get_regular_grid = || {
        let mut dims = [0_usize; MAXDIMS];
        let mut starts = [T::zero(); MAXDIMS];
        let mut steps = [T::zero(); MAXDIMS];

        for (i, grid) in grids.iter().enumerate() {
            if grid.len() < 2 {
                return Err("All grids must have at least two entries");
            }
            dims[i] = grid.len();
            starts[i] = grid[0];
            steps[i] = grid[1] - grid[0];
        }

        Ok((dims, starts, steps))
    };

    // Bounds checks for regular grid, if requested
    let maybe_check_bounds_regular = |dims: &[usize], starts: &[T], steps: &[T], obs: &[&[T]]| {
        if let Some(atol) = check_bounds_with_atol {
            let mut bounds = [false; MAXDIMS];
            let out = &mut bounds[..ndims];
            multilinear::regular::check_bounds(
                &dims[..ndims],
                &starts[..ndims],
                &steps[..ndims],
                obs,
                atol,
                out,
            )?;
            if bounds.iter().any(|x| *x) {
                return Err("At least one observation point is outside the grid.");
            }
        }
        Ok(())
    };

    // Bounds checks for rectilinear grid, if requested
    let maybe_check_bounds_rectilinear = |grids, obs| {
        if let Some(atol) = check_bounds_with_atol {
            let mut bounds = [false; MAXDIMS];
            let out = &mut bounds[..ndims];
            multilinear::rectilinear::check_bounds(grids, obs, atol, out)?;
            if bounds.iter().any(|x| *x) {
                return Err("At least one observation point is outside the grid.");
            }
        }
        Ok(())
    };

    // Select lower-level method
    match (method, kind) {
        (GridInterpMethod::Linear, GridKind::Regular) => {
            let (dims, starts, steps) = get_regular_grid()?;
            maybe_check_bounds_regular(&dims, &starts, &steps, obs)?;
            linear::regular::interpn(
                &dims[..ndims],
                &starts[..ndims],
                &steps[..ndims],
                vals,
                obs,
                out,
            )
        }
        (GridInterpMethod::Linear, GridKind::Rectilinear) => {
            maybe_check_bounds_rectilinear(grids, obs)?;
            linear::rectilinear::interpn(grids, vals, obs, out)
        }
        (GridInterpMethod::Cubic, GridKind::Regular) => {
            let (dims, starts, steps) = get_regular_grid()?;
            maybe_check_bounds_regular(&dims, &starts, &steps, obs)?;
            cubic::regular::interpn(
                &dims[..ndims],
                &starts[..ndims],
                &steps[..ndims],
                vals,
                linearize_extrapolation,
                obs,
                out,
            )
        }
        (GridInterpMethod::Cubic, GridKind::Rectilinear) => {
            maybe_check_bounds_rectilinear(grids, obs)?;
            cubic::rectilinear::interpn(grids, vals, linearize_extrapolation, obs, out)
        }
        (GridInterpMethod::Nearest, GridKind::Regular) => {
            let (dims, starts, steps) = get_regular_grid()?;
            maybe_check_bounds_regular(&dims, &starts, &steps, obs)?;
            nearest::regular::interpn(
                &dims[..ndims],
                &starts[..ndims],
                &steps[..ndims],
                vals,
                obs,
                out,
            )
        }
        (GridInterpMethod::Nearest, GridKind::Rectilinear) => {
            maybe_check_bounds_rectilinear(grids, obs)?;
            nearest::rectilinear::interpn(grids, vals, obs, out)
        }
    }
}

/// Figure out whether a grid is regular or rectilinear.
fn resolve_grid_kind<T: Float>(
    assume_grid_kind: Option<GridKind>,
    grids: &[&[T]],
) -> Result<GridKind, &'static str> {
    let kind = match assume_grid_kind {
        Some(GridKind::Regular) => GridKind::Regular,
        Some(GridKind::Rectilinear) => GridKind::Rectilinear,
        None => {
            // Check whether grid is regular
            let mut is_regular = true;

            for grid in grids.iter() {
                if grid.len() < 2 {
                    return Err("All grids must have at least two entries");
                }
                let step = grid[1] - grid[0];

                if !grid.windows(2).all(|pair| pair[1] - pair[0] == step) {
                    is_regular = false;
                    break;
                }
            }

            if is_regular {
                GridKind::Regular
            } else {
                GridKind::Rectilinear
            }
        }
    };

    Ok(kind)
}

/// Index a single value from an array
#[inline]
pub(crate) fn index_arr<T: Copy>(loc: &[usize], dimprod: &[usize], data: &[T]) -> T {
    let mut i = 0;
    for j in 0..dimprod.len() {
        i += loc[j] * dimprod[j];
    }

    data[i]
}

/// Index a single value from an array with a known fixed number of dimensions
#[inline]
pub(crate) fn index_arr_fixed_dims<T: Copy, const N: usize>(
    loc: [usize; N],
    dimprod: [usize; N],
    data: &[T],
) -> T {
    let mut i = 0;

    for j in 0..N {
        i += loc[j] * dimprod[j];
    }

    data[i]
}
