//! Multilinear interpolation and extrapolation.

pub mod rectilinear;
pub mod regular;

pub use rectilinear::RectilinearGridInterpolator;
pub use regular::RegularGridInterpolator;