//! Multicubic interpolation.
//!
//! For interior points, this method gives the same result as
//! a Hermite spline interpolation, with continuous first derivatives
//! across the cell boundaries. Under extrapolation, either quadratic
//! or linear extrapolation is used depending on configuration, and
//! both maintain continuous first derivatives everywhere.
//!
//! The solution on interior points is notably different from B-splines,
//! where the exact values of the first and second derivatives are taken
//! as optimization variables in order to produce an interpolant with
//! both first and second derivatives continuous across cell boundaries.
//! The B-spline method produces physically-meaningful results in
//! the special case of a one-dimensional slender beam under pinned
//! structural constraints (the original meaning of a spline), but has
//! some drawbacks:
//! * Requires solving a large linear system during initialization to
//!   determine knot coefficients
//! * Requires storing solved knot coefficients simultaneously and indexing
//!   into several coefficient arrays during evaluation, which produces
//!   an undesirable memory scaling
//! * Produces a nonzero sensitivity between an interpolant at any
//!   observation point and the value of _every_ node in the grid
//!   through that linear solve, which does not accurately represent
//!   the flow of information in many real systems
//! * Discards local information about the slope of the interpolant
//!   in favor of enforcing the continuity of the second derivative
//!   * This essentially sacrifices the correctness of both the
//!     first and second derivatives of the interpolant in order to
//!     prioritize their continuity
//!
//! By contrast, this method
//! * Does not require storing any coefficients - the grid data is the
//!   only thing required to evaluate the interpolation
//!   * This allows initialization and evaluation entirely on the stack
//!     and in an embedded environment
//! * Produces an interpolant function that depends on a limited number
//!   of grid points (rather than the entire grid)
//! * Does not perform any expensive processing during initialization
//! * Prioritizes the correctness of the (continuous) estimate of the
//!   first derivative at the expense of the continuity of the second
//!   derivative
//!   * This ordering of priorities best supports optimization
//!     algorithms such as quadratic programming, which depend
//!     on the correctness and continuity of the first derivative,
//!     but can tolerate some discontinuity in the second derivative
use num_traits::Float;

pub mod rectilinear;
pub mod regular;

pub use rectilinear::MulticubicRectilinear;
pub use regular::MulticubicRegular;

/// Evaluate a hermite spline function on an interval from x0 to x1,
/// with imposed slopes k0 and k1 at the endpoints, and normalized
/// coordinate t = (x - x0) / (x1 - x0).
#[inline]
pub(crate) fn normalized_hermite_spline<T: Float>(t: T, y0: T, dy: T, k0: T, k1: T) -> T {
    // `a` and `b` are the difference between this function and a linear one going
    // forward or backward with the imposed slopes.
    let a = k0 - dy;
    let b = -k1 + dy;

    let t2 = t * t;
    let t3 = t.powi(3);

    let c1 = dy + a;
    let c2 = b - (a + a);
    let c3 = a - b;

    y0 + (c1 * t) + (c2 * t2) + (c3 * t3)
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Saturation {
    None,
    InsideLow,
    OutsideLow,
    InsideHigh,
    OutsideHigh,
}
