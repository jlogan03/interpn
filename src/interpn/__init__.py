"""
Python bindings to the `interpn` Rust library
for N-dimensional interpolation and extrapolation.
"""

from __future__ import annotations

from typing import Literal
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec

import numpy as np

from numpy.typing import NDArray

from interpn import raw

_PYDANTIC_FOUND = find_spec("pydantic") is not None

if _PYDANTIC_FOUND:
    from .multilinear_regular import MultilinearRegular
    from .multilinear_rectilinear import MultilinearRectilinear
    from .multicubic_regular import MulticubicRegular
    from .multicubic_rectilinear import MulticubicRectilinear
    from .nearest_regular import NearestRegular
    from .nearest_rectilinear import NearestRectilinear

try:
    __version__ = version("interpn")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "raw",
    "interpn",
]

if _PYDANTIC_FOUND:
    __all__ += [
        "MultilinearRegular",
        "MultilinearRectilinear",
        "MulticubicRegular",
        "MulticubicRectilinear",
        "NearestRegular",
        "NearestRectilinear",
    ]


def interpn(
    obs: Sequence[NDArray],
    grids: Sequence[NDArray],
    vals: NDArray,
    *,
    method: Literal["linear", "cubic", "nearest"] = "linear",
    out: NDArray | None = None,
    linearize_extrapolation: bool = True,
    grid_kind: Literal["regular", "rectilinear"] | None = None,
    check_bounds_with_atol: float | None = None,
    max_threads: int | None = None,
) -> NDArray:
    """
    Evaluate an N-dimensional grid at the supplied observation points.

    Reallocates input arrays if and only if they are not contiguous yet.

    Note: values must be defined in C-order, like made by
    `numpy.meshgrid(*grids, indexing="ij")`. Values on meshgrids defined
    in graphics-order without `indexing="ij"` will not have the desired effect.

    If a pre-allocated output array is provided, the returned array is a
    reference to that array.

    Args:
        obs: Observation coordinates, one array per dimension.
        grids: Grid axis coordinates, one array per dimension.
        vals: Values defined on the full cartesian-product grid.
        method: Interpolation kind, one of ``"linear"``, ``"cubic"``, or ``"nearest"``.
        out: Optional preallocated array that receives the result.
        linearize_extrapolation: Whether cubic extrapolation should fall back to
            linear behaviour outside the grid bounds.
        grid_kind: Optional ``"regular"`` or ``"rectilinear"`` to skip
            grid-shape autodetection.
        check_bounds_with_atol: When set, raise if any observation lies outside
            the grid by more than this absolute tolerance.
        max_threads: Optional upper bound for parallel execution threads.

    Returns:
        Interpolated values
    """
    # Allocate for the output if it is not supplied
    out = out if out is not None else np.zeros_like(obs[0])
    outshape = out.shape
    out = out.ravel()  # Flat view without reallocating

    # Ensure contiguous and flat, reallocating only if necessary
    obs = [np.ascontiguousarray(x.ravel()) for x in obs]
    grids = [np.ascontiguousarray(x.ravel()) for x in grids]
    vals = np.ascontiguousarray(vals.ravel())

    # Check data type
    dtype = vals.dtype
    assert dtype in [np.float64, np.float32], (
        "`interpn` defined only for float32 and float64 data"
    )

    # Do interpolation
    match dtype:
        case np.float32:
            raw.interpn_f32(
                grids,
                vals,
                obs,
                out,
                method=method,
                grid_kind=grid_kind,
                linearize_extrapolation=linearize_extrapolation,
                check_bounds_with_atol=check_bounds_with_atol,
                max_threads=max_threads,
            )
        case np.float64:
            raw.interpn_f64(
                grids,
                vals,
                obs,
                out,
                method=method,
                grid_kind=grid_kind,
                linearize_extrapolation=linearize_extrapolation,
                check_bounds_with_atol=check_bounds_with_atol,
                max_threads=max_threads,
            )
        case _:
            raise ValueError(
                f"Unsupported interpolation configuration: {dtype}, {method}"
            )

    return out.reshape(outshape)
