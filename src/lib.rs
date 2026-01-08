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

#[cfg(feature = "std")]
pub mod utils;

#[cfg(all(test, feature = "std"))]
pub(crate) mod testing;

#[cfg(feature = "python")]
pub mod python;

pub enum GridInterpMethod {
    Linear,
    Cubic,
}

pub enum GridKind {
    Regular,
    Rectilinear,
}

const MAXDIMS: usize = 8;
const MAXDIMS_ERR: &str =
    "Dimension exceeds maximum (8). Use interpolator struct directly for higher dimensions.";

pub fn interpn<T: Float>(
    grids: &[&[T]],
    vals: &[T],
    obs: &[&[T]],
    out: &mut [T],
    method: GridInterpMethod,
    assume_grid_kind: Option<GridKind>,
    linearize_extrapolation: bool,
) -> Result<(), &'static str> {
    let ndims = grids.len();
    if ndims > MAXDIMS {
        return Err(MAXDIMS_ERR);
    }

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

    // Select lower-level method
    match (method, kind) {
        (GridInterpMethod::Linear, GridKind::Regular) => {
            let (dims, starts, steps) = get_regular_grid()?;
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
            linear::rectilinear::interpn(grids, vals, obs, out)
        }
        (GridInterpMethod::Cubic, GridKind::Regular) => {
            let (dims, starts, steps) = get_regular_grid()?;
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
            cubic::rectilinear::interpn(grids, vals, linearize_extrapolation, obs, out)
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

    // unroll! {
    //     for j < 7 in 0..N {
    for j in 0..N {
        i += loc[j] * dimprod[j];
        // }
    }

    data[i]
}
