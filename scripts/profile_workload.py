#!/usr/bin/env python3
"""Lightweight workload used to gather PGO profiles for interpn."""

from __future__ import annotations

import numpy as np

from interpn import MulticubicRectilinear, MulticubicRegular, MultilinearRectilinear, MultilinearRegular

_OBSERVATION_COUNTS = (1,3)
_MAX_DIMS = 4
_GRID_SIZE = 20


def _observation_points(
    rng: np.random.Generator, ndims: int, nobs: int, dtype: np.dtype
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate observation points inside and outside the grid domain.
    The fraction of points outside the domain here will set the relative weight of
    extrapolation branches.
    """
    m = max(int(float(nobs) ** (1.0 / ndims) + 2.0), 2)
    axes = [rng.uniform(-1.1, 1.1, m).astype(dtype) for _ in range(ndims)]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = [axis.flatten()[:nobs].copy() for axis in mesh]
    return points


def _evaluate(interpolator, points: list[np.ndarray]) -> None:
    # Without preallocated output
    # interpolator.eval(points)
    # With preallocated output - we don't need to profile the allocator
    out = np.empty_like(points[0])
    interpolator.eval(points, out)


def main() -> None:
    rng = np.random.default_rng(2394587)

    for dtype in (np.float64, np.float32):
        for ndims in range(1, _MAX_DIMS + 1):
            grids = [np.linspace(-1.0, 1.0, _GRID_SIZE, dtype=dtype) for _ in range(ndims)]
            mesh = np.meshgrid(*grids, indexing="ij")
            zgrid = rng.uniform(-1.0, 1.0, mesh[0].size).astype(dtype)
            dims = [grid.size for grid in grids]
            starts = np.array([grid[0] for grid in grids], dtype=dtype)
            steps = np.array([grid[1] - grid[0] for grid in grids], dtype=dtype)

            linear_regular = MultilinearRegular.new(dims, starts, steps, zgrid)
            linear_rect = MultilinearRectilinear.new(grids, zgrid)
            cubic_regular = MulticubicRegular.new(
                dims,
                starts,
                steps,
                zgrid,
                linearize_extrapolation=(ndims % 2 == 0),
            )
            cubic_rect = MulticubicRectilinear.new(
                grids,
                zgrid,
                linearize_extrapolation=(ndims % 2 == 1),
            )

            for nobs in _OBSERVATION_COUNTS:
                points = _observation_points(rng, ndims, nobs, dtype)

                for interpolator in (
                    linear_regular,
                    linear_rect,
                    cubic_regular,
                    cubic_rect,
                ):
                    _evaluate(interpolator, points)

                    print(f"Completed {type(interpolator).__name__} dtype={np.dtype(dtype).name} ndims={ndims} nobs={nobs}")


if __name__ == "__main__":
    main()
