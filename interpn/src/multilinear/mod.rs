//! Multilinear interpolation and extrapolation.
//! See individual modules for more detailed documentation.

pub mod rectilinear;
pub mod rectilinear_recursive;
pub mod regular;
pub mod regular_recursive;

pub use rectilinear::MultilinearRectilinear;
pub use rectilinear_recursive::MultilinearRectilinearRecursive;
pub use regular::MultilinearRegular;
pub use regular_recursive::MultilinearRegularRecursive;
