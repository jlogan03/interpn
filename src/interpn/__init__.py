"""
Python bindings to the `interpn` Rust library
for N-dimensional interpolation and extrapolation.
"""

from __future__ import annotations

from typing import Literal
from collections.abc import Sequence
from importlib.metadata import version
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

__version__ = version("interpn")

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
    assume_regular: bool = False,
    check_bounds: bool = False,
    bounds_atol: float = 1e-8,
) -> NDArray:
    """
    Evaluate an N-dimensional grid at the supplied observation points.

    Performs some small allocations to prepare the inputs and
    performs O(gridsize) checks to determine grid regularity
    unless `assume_regular` is set. To avoid this overhead entirely,
    use the persistent wrapper classes or raw bindings instead.

    Reallocates input arrays if and only if they are not contiguous yet.

    Args:
        obs: Observation coordinates, one array per dimension.
        grids: Grid axis coordinates, one array per dimension.
        vals: Values defined on the full tensor-product grid.
        method: Interpolation kind, one of ``"linear"``, ``"cubic"``, or ``"nearest"``.
        out: Optional preallocated array that receives the result.
        linearize_extrapolation: Whether cubic extrapolation should fall back to
            linear behaviour outside the grid bounds.
        assume_regular: Treat the grid as regular without checking spacing.
        check_bounds: When True, raise if any observation lies outside the grid.
        bounds_atol: Absolute tolerance for bounds checks, to avoid spurious errors

    Returns:
        Interpolated values
    """
    # Allocate for the output if it is not supplied
    out = out or np.zeros_like(obs[0])
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

    # Check regularity
    is_regular = assume_regular or _check_regular(grids)

    if is_regular:
        dims = np.array([len(grid) for grid in grids], dtype=int)
        starts = np.array([grid[0] for grid in grids], dtype=dtype)
        steps = np.array([grid[1] - grid[0] for grid in grids], dtype=dtype)
    else:
        # Pyright doesn't understand match-case
        dims = np.empty((0,), dtype=int)
        starts = np.empty((0,), dtype=dtype)
        steps = starts

    # Check bounds
    if check_bounds:
        outb = np.zeros_like(out.shape, dtype=bool)
        match (dtype, is_regular):
            case (np.float32, True):
                raw.check_bounds_regular_f32(
                    dims, starts, steps, obs, atol=bounds_atol, out=outb
                )
            case (np.float64, True):
                raw.check_bounds_regular_f64(
                    dims, starts, steps, obs, atol=bounds_atol, out=outb
                )
            case (np.float32, False):
                raw.check_bounds_rectilinear_f32(grids, obs, atol=bounds_atol, out=outb)
            case (np.float64, False):
                raw.check_bounds_rectilinear_f64(grids, obs, atol=bounds_atol, out=outb)

        if any(outb):
            raise ValueError("Observation points violate interpolator bounds")

    # Do interpolation
    match (dtype, is_regular, method):
        case (np.float32, True, "linear"):
            raw.interpn_linear_regular_f32(dims, starts, steps, vals, obs, out)
        case (np.float64, True, "linear"):
            raw.interpn_linear_regular_f64(dims, starts, steps, vals, obs, out)
        case (np.float32, False, "linear"):
            raw.interpn_linear_rectilinear_f32(grids, vals, obs, out)
        case (np.float64, False, "linear"):
            raw.interpn_linear_rectilinear_f64(grids, vals, obs, out)
        case (np.float32, True, "nearest"):
            raw.interpn_nearest_regular_f32(dims, starts, steps, vals, obs, out)
        case (np.float64, True, "nearest"):
            raw.interpn_nearest_regular_f64(dims, starts, steps, vals, obs, out)
        case (np.float32, False, "nearest"):
            raw.interpn_nearest_rectilinear_f32(grids, vals, obs, out)
        case (np.float64, False, "nearest"):
            raw.interpn_nearest_rectilinear_f64(grids, vals, obs, out)
        case (np.float32, True, "cubic"):
            raw.interpn_cubic_regular_f32(
                dims,
                starts,
                steps,
                vals,
                linearize_extrapolation,
                obs,
                out,
            )
        case (np.float64, True, "cubic"):
            raw.interpn_cubic_regular_f64(
                dims,
                starts,
                steps,
                vals,
                linearize_extrapolation,
                obs,
                out,
            )
        case (np.float32, False, "cubic"):
            raw.interpn_cubic_rectilinear_f32(
                grids,
                vals,
                linearize_extrapolation,
                obs,
                out,
            )
        case (np.float64, False, "cubic"):
            raw.interpn_cubic_rectilinear_f64(
                grids,
                vals,
                linearize_extrapolation,
                obs,
                out,
            )
        case _:
            raise ValueError(
                "Unsupported interpolation configuration:"
                f" {dtype}, {is_regular}, {method}"
            )

    return out.reshape(outshape)


def _check_regular(grids: Sequence[NDArray]) -> bool:
    """Check if grids are all regularly spaced"""
    is_regular = True
    for grid in grids:
        dgrid = np.diff(grid)
        is_regular = is_regular and np.all(dgrid == dgrid[0])
    return bool(is_regular)
