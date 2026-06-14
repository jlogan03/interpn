#!/usr/bin/env python3
"""Lightweight workload used to gather PGO profiles for interpn."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from interpn import interpn as interpn_fn

_TARGET_COUNT = int(1e4)
_OBSERVATION_COUNTS = (1, 3, 571, 2017, int(1e4))
_MAX_DIMS = 4
_GRID_SIZE = 30


def _observation_points(
    rng: np.random.Generator, ndims: int, nobs: int, dtype: np.dtype
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate observation points inside and outside the grid domain.
    The fraction of points outside the domain here will set the relative weight of
    extrapolation branches.
    """
    m = max(int(float(nobs) ** (1.0 / ndims) + 2.0), 2)
    axes = [rng.uniform(-1.05, 1.05, m).astype(dtype) for _ in range(ndims)]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = [axis.flatten()[:nobs].copy() for axis in mesh]
    return points


def _evaluate(
    *,
    grids: list[np.ndarray],
    vals: np.ndarray,
    points: list[np.ndarray],
    method: str,
    grid_kind: str,
    max_threads: int | None,
) -> None:
    # Run once without preallocated output to exercise allocating path
    interpn_fn(
        obs=points,
        grids=grids,
        vals=vals,
        method=method,
        grid_kind=grid_kind,
        linearize_extrapolation=True,
        max_threads=max_threads,
    )
    # Run again with preallocated output
    out = np.empty_like(points[0])
    interpn_fn(
        obs=points,
        grids=grids,
        vals=vals,
        method=method,
        grid_kind=grid_kind,
        linearize_extrapolation=True,
        out=out,
        max_threads=max_threads,
    )


def main() -> None:
    rng = np.random.default_rng(2394587)

    for dtype in (np.float64, np.float32):
        for ndims in range(1, _MAX_DIMS + 1):
            ngrid = _GRID_SIZE if ndims < 5 else 6
            grids = [np.linspace(-1.0, 1.0, ngrid, dtype=dtype) for _ in range(ndims)]
            grids_rect = [
                np.array(sorted(np.random.uniform(-1.0, 1.0, ngrid).astype(dtype)))
                for _ in range(ndims)
            ]
            mesh = np.meshgrid(*grids, indexing="ij")
            zgrid = rng.uniform(-1.0, 1.0, mesh[0].size).astype(dtype)
            cases = (
                ("linear", "regular", grids),
                ("linear", "rectilinear", grids_rect),
                ("cubic", "regular", grids),
                ("cubic", "rectilinear", grids_rect),
                ("bspline", "regular", grids),
                ("bspline", "rectilinear", grids_rect),
                ("nearest", "regular", grids),
                ("nearest", "rectilinear", grids_rect),
            )

            for nobs in _OBSERVATION_COUNTS:
                nreps = max(int(_TARGET_COUNT / nobs), 1)

                for max_threads in (None, 1):
                    for method, grid_kind, grids_in in cases:
                        # B-spline method has to do the fitting solve,
                        # so it's much slower than the others.
                        if method == "bspline":
                            nreps = max(nreps / 100, 1)

                        for _ in range(nreps):
                            points = _observation_points(rng, ndims, nobs, dtype)
                            _evaluate(
                                grids=grids_in,
                                vals=zgrid,
                                points=points,
                                method=method,
                                grid_kind=grid_kind,
                                max_threads=max_threads,
                            )

                        mode = "parallel" if max_threads is None else "serial"
                        print(
                            f"Completed interpn method={method} grid={grid_kind} "
                            f"dtype={np.dtype(dtype).name} ndims={ndims} nobs={nobs} "
                            f"mode={mode}"
                        )


if __name__ == "__main__":
    main()
    # Run the serial script separately. This will make another
    # profile file that needs to be merged, but since we ran
    # some threaded workloads, we already end up with several
    # files to merge.
    script = Path(__file__).with_name("profile_workload_ser.py")
    subprocess.run([sys.executable, str(script)], check=True)
