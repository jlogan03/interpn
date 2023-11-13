//! N-dimensional interpolation methods
#![cfg_attr(not(feature = "std"), no_std)]

pub mod multilinear_rectilinear;
pub mod multilinear_regular;

#[cfg(feature = "std")]
pub mod utils;

#[cfg(all(test, feature = "std"))]
pub(crate) mod testing;
