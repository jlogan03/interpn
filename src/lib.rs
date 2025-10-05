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

pub mod multilinear;
pub use multilinear::{MultilinearRectilinear, MultilinearRegular};

pub mod multicubic;
pub use multicubic::{MulticubicRectilinear, MulticubicRegular};

pub mod one_dim;
pub use one_dim::{
    RectilinearGrid1D, RegularGrid1D, hold::Left1D, hold::Nearest1D, hold::Right1D,
    linear::Linear1D, linear::LinearHoldLast1D,
};

#[cfg(feature = "std")]
pub mod utils;

#[cfg(all(test, feature = "std"))]
pub(crate) mod testing;

#[cfg(feature="python")]
pub mod python;

/// Index a single value from an array
#[inline]
pub(crate) fn index_arr<T: Copy>(loc: &[usize], dimprod: &[usize], data: &[T]) -> T {
    let mut i = 0;
    for j in 0..dimprod.len() {
        i += loc[j] * dimprod[j];
    }

    data[i]
}
