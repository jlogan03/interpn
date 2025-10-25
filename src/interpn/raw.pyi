from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

NDArrayF64 = NDArray[np.float64]
NDArrayF32 = NDArray[np.float32]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.intp]

__all__ = [
    "interpn_linear_regular_f64",
    "interpn_linear_regular_f32",
    "interpn_linear_rectilinear_f64",
    "interpn_linear_rectilinear_f32",
    "interpn_nearest_regular_f64",
    "interpn_nearest_regular_f32",
    "interpn_nearest_rectilinear_f64",
    "interpn_nearest_rectilinear_f32",
    "interpn_cubic_regular_f64",
    "interpn_cubic_regular_f32",
    "interpn_cubic_rectilinear_f64",
    "interpn_cubic_rectilinear_f32",
    "check_bounds_regular_f64",
    "check_bounds_regular_f32",
    "check_bounds_rectilinear_f64",
    "check_bounds_rectilinear_f32",
]

def interpn_linear_regular_f64(
    dims: IntArray,
    starts: NDArrayF64,
    steps: NDArrayF64,
    vals: NDArrayF64,
    obs: Sequence[NDArrayF64],
    out: NDArrayF64,
) -> None: ...
def interpn_linear_regular_f32(
    dims: IntArray,
    starts: NDArrayF32,
    steps: NDArrayF32,
    vals: NDArrayF32,
    obs: Sequence[NDArrayF32],
    out: NDArrayF32,
) -> None: ...
def interpn_linear_rectilinear_f64(
    grids: Sequence[NDArrayF64],
    vals: NDArrayF64,
    obs: Sequence[NDArrayF64],
    out: NDArrayF64,
) -> None: ...
def interpn_linear_rectilinear_f32(
    grids: Sequence[NDArrayF32],
    vals: NDArrayF32,
    obs: Sequence[NDArrayF32],
    out: NDArrayF32,
) -> None: ...
def interpn_nearest_regular_f64(
    dims: IntArray,
    starts: NDArrayF64,
    steps: NDArrayF64,
    vals: NDArrayF64,
    obs: Sequence[NDArrayF64],
    out: NDArrayF64,
) -> None: ...
def interpn_nearest_regular_f32(
    dims: IntArray,
    starts: NDArrayF32,
    steps: NDArrayF32,
    vals: NDArrayF32,
    obs: Sequence[NDArrayF32],
    out: NDArrayF32,
) -> None: ...
def interpn_nearest_rectilinear_f64(
    grids: Sequence[NDArrayF64],
    vals: NDArrayF64,
    obs: Sequence[NDArrayF64],
    out: NDArrayF64,
) -> None: ...
def interpn_nearest_rectilinear_f32(
    grids: Sequence[NDArrayF32],
    vals: NDArrayF32,
    obs: Sequence[NDArrayF32],
    out: NDArrayF32,
) -> None: ...
def interpn_cubic_regular_f64(
    dims: IntArray,
    starts: NDArrayF64,
    steps: NDArrayF64,
    vals: NDArrayF64,
    linearize_extrapolation: bool,
    obs: Sequence[NDArrayF64],
    out: NDArrayF64,
) -> None: ...
def interpn_cubic_regular_f32(
    dims: IntArray,
    starts: NDArrayF32,
    steps: NDArrayF32,
    vals: NDArrayF32,
    linearize_extrapolation: bool,
    obs: Sequence[NDArrayF32],
    out: NDArrayF32,
) -> None: ...
def interpn_cubic_rectilinear_f64(
    grids: Sequence[NDArrayF64],
    vals: NDArrayF64,
    linearize_extrapolation: bool,
    obs: Sequence[NDArrayF64],
    out: NDArrayF64,
) -> None: ...
def interpn_cubic_rectilinear_f32(
    grids: Sequence[NDArrayF32],
    vals: NDArrayF32,
    linearize_extrapolation: bool,
    obs: Sequence[NDArrayF32],
    out: NDArrayF32,
) -> None: ...
def check_bounds_regular_f64(
    dims: IntArray,
    starts: NDArrayF64,
    steps: NDArrayF64,
    obs: Sequence[NDArrayF64],
    atol: float,
    out: BoolArray,
) -> None: ...
def check_bounds_regular_f32(
    dims: IntArray,
    starts: NDArrayF32,
    steps: NDArrayF32,
    obs: Sequence[NDArrayF32],
    atol: float,
    out: BoolArray,
) -> None: ...
def check_bounds_rectilinear_f64(
    grids: Sequence[NDArrayF64],
    obs: Sequence[NDArrayF64],
    atol: float,
    out: BoolArray,
) -> None: ...
def check_bounds_rectilinear_f32(
    grids: Sequence[NDArrayF32],
    obs: Sequence[NDArrayF32],
    atol: float,
    out: BoolArray,
) -> None: ...
