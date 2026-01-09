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
use std::sync::Mutex;

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

    // Chunk for parallelism
    let num_cores = rayon::current_num_threads()
        .max(1)
        .min(max_threads.unwrap_or(usize::MAX));
    let n = out.len();
    let chunk = 1024.max(n / num_cores);

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
                None => write_err("Dimension mismatch"),
            };
        }

        // Do interpolations
        let res_inner = interpn_serial(
            grids,
            vals,
            &obs_slices[..ndims],
            outc,
            method,
            assume_grid_kind,
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
}

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

    // Resolve grid kind, checking the grid if
    // the kind is not provided by the user.
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
                return Err("At least one observation point is outside the grid.")
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
                return Err("At least one observation point is outside the grid.")
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
