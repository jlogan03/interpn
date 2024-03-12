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
//! | Method                        | RAM       | Interp. Cost (Best Case) | Interp. Cost (Worst Case)               | Extrap. Cost (Worst Case)                      |
//! |-------------------------------|-----------|--------------------------|-----------------------------------------|------------------------------------------------|
//! | multilinear::regular          | O(ndims)  | O(2^ndims * ndims)       | O(2^ndims * ndims)                      | O(2^ndims + ndims^2)                           |
//! | multilinear::rectilinear      | O(ndims)  | O(2^ndims * ndims)       | O(ndims * (2^ndims + log2(gridsize)))   | O(ndims * (2^ndims + ndims + log2(gridsize)))  |
//! | multicubic::regular           | O(ndims)  | O(4^ndims)               | O(4^ndims)                              | O(4^ndims)                                     |
//! | multicubic::rectilinear       | O(ndims)  | O(4^ndims)               | O(4^ndims) + ndims * log2(gridsize)     | O(4^ndims) + ndims * log2(gridsize)            |
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
//! * Recursive multilinear methods (for better extrapolation speed and timing determinism)
//! * Methods for unstructured triangular and tetrahedral meshes
#![cfg_attr(not(feature = "std"), no_std)]
// These "needless" range loops are a significant speedup
#![allow(clippy::needless_range_loop)]

pub mod multilinear;
pub use multilinear::{MultilinearRectilinear, MultilinearRegular};

pub mod multicubic;
pub use multicubic::{MulticubicRectilinear, MulticubicRegular};

#[cfg(feature = "std")]
pub mod utils;

#[cfg(all(test, feature = "std"))]
pub(crate) mod testing;
