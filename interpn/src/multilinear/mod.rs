//! Multilinear interpolation and extrapolation.
//! See individual modules for more detailed documentation.

pub mod rectilinear;
pub mod regular;

pub use rectilinear::MultilinearRectilinear;
pub use regular::MultilinearRegular;
