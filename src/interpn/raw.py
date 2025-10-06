"""
Re-exported raw PyO3/Maturin bindings to Rust functions.
Using these can yield some performance benefit at the expense of ergonomics.
"""

from .interpn import (
    interpn_linear_regular_f64,
    interpn_linear_regular_f32,
    interpn_linear_rectilinear_f64,
    interpn_linear_rectilinear_f32,
    interpn_cubic_regular_f64,
    interpn_cubic_regular_f32,
    interpn_cubic_rectilinear_f64,
    interpn_cubic_rectilinear_f32,
    check_bounds_regular_f64,
    check_bounds_regular_f32,
    check_bounds_rectilinear_f64,
    check_bounds_rectilinear_f32,
)

__all__ = [
    "interpn_linear_regular_f64",
    "interpn_linear_regular_f32",
    "interpn_linear_rectilinear_f64",
    "interpn_linear_rectilinear_f32",
    "interpn_cubic_regular_f64",
    "interpn_cubic_regular_f32",
    "interpn_cubic_rectilinear_f64",
    "interpn_cubic_rectilinear_f32",
    "check_bounds_regular_f64",
    "check_bounds_regular_f32",
    "check_bounds_rectilinear_f64",
    "check_bounds_rectilinear_f32",
]
