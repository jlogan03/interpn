//! N-dimensional interpolation methods
#![cfg_attr(not(feature = "std"), no_std)]
// These "needless" range loops are a significant speedup
#![allow(clippy::needless_range_loop)]

pub mod multilinear_rectilinear;
pub mod multilinear_regular;

#[cfg(feature = "std")]
pub mod utils;

#[cfg(all(test, feature = "std"))]
pub(crate) mod testing;
