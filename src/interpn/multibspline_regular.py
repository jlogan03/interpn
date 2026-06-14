from __future__ import annotations

from functools import reduce

import numpy as np
from numpy.typing import NDArray

from pydantic import (
    model_validator,
    ConfigDict,
    BaseModel,
)

from .serialization import Array, ArrayF32, ArrayF64

from .interpn import (
    coefficients_bspline_regular_f64,
    coefficients_bspline_regular_f32,
    _eval_bspline_regular_f64,
    _eval_bspline_regular_f32,
    check_bounds_regular_f64,
    check_bounds_regular_f32,
)


class MultiBsplineRegular(BaseModel):
    """
    Cubic B-spline interpolation on a regular grid in up to 8 dimensions.

    This class owns generated B-spline coefficients as NumPy arrays. The Rust
    implementation remains a borrowed interface over caller-provided storage.

    All array inputs must be of the same type, either np.float32 or np.float64,
    and must be 1D and contiguous. Each dimension must have size at least 4.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    dims: list[int]
    starts: Array
    steps: Array
    coeffs: Array
    linearize_extrapolation: bool

    @classmethod
    def new(
        cls,
        dims: list[int],
        starts: NDArray,
        steps: NDArray,
        vals: NDArray,
        linearize_extrapolation: bool = True,
    ) -> MultiBsplineRegular:
        """
        Initialize an interpolator by generating B-spline coefficients from
        nodal values.
        """
        dtype = vals.dtype
        arrtype = ArrayF64 if dtype == np.float64 else ArrayF32

        vals_flat = np.ascontiguousarray(vals.flatten())
        coeffs = np.zeros_like(vals_flat)
        scratch = np.zeros(2 * max(dims), dtype=dtype)

        if dtype == np.float64:
            coefficients_bspline_regular_f64(dims, vals_flat, coeffs, scratch)
        elif dtype == np.float32:
            coefficients_bspline_regular_f32(dims, vals_flat, coeffs, scratch)
        else:
            raise TypeError(f"Unexpected data type: {dtype}")

        return cls(
            dims=dims,
            starts=arrtype(data=starts.flatten()),
            steps=arrtype(data=steps.flatten()),
            coeffs=arrtype(data=coeffs),
            linearize_extrapolation=linearize_extrapolation,
        )

    @model_validator(mode="after")
    def _validate_model(self):
        ndims = self.ndims()
        assert ndims <= 8 and ndims >= 1, (
            "Number of dimensions must be at least 1 and no more than 8"
        )
        assert all([x >= 4 for x in self.dims]), (
            "All grid dimensions must have at least 4 entries"
        )
        assert self.starts.data.size == ndims, "Grid dimension mismatch"
        assert self.steps.data.size == ndims, "Grid dimension mismatch"
        assert self.coeffs.data.size == reduce(lambda acc, x: acc * x, self.dims), (
            "Size of coefficient array does not match grid dims"
        )
        assert all([x > 0.0 for x in self.steps.data]), (
            "All grid steps must be positive and nonzero"
        )
        assert all(
            [x.data.dtype == self.coeffs.data.dtype for x in [self.steps, self.coeffs]]
        ), "All grid inputs must be of the same data type (np.float32 or np.float64)"
        assert all(
            [x.data.data.contiguous for x in [self.starts, self.steps, self.coeffs]]
        ), "Grid data must be contiguous"

        return self

    def ndims(self) -> int:
        return len(self.dims)

    def eval(self, obs: list[NDArray], out: NDArray | None = None) -> NDArray:
        """Evaluate the interpolator at a set of observation points."""
        out_inner = out if out is not None else np.zeros_like(obs[0])
        self.eval_unchecked(obs, out_inner)
        return out_inner

    def eval_unchecked(self, obs: list[NDArray], out: NDArray | None = None) -> NDArray:
        """
        Evaluate the interpolator, skipping Python-side contiguity checks.
        """
        dtype = self.coeffs.data.dtype
        out_inner = out if out is not None else np.zeros_like(obs[0])

        if dtype == np.float64:
            _eval_bspline_regular_f64(
                self.dims,
                self.starts.data,
                self.steps.data,
                self.coeffs.data,
                self.linearize_extrapolation,
                obs,
                out_inner,
            )
        elif dtype == np.float32:
            _eval_bspline_regular_f32(
                self.dims,
                self.starts.data,
                self.steps.data,
                self.coeffs.data,
                self.linearize_extrapolation,
                obs,
                out_inner,
            )
        else:
            raise TypeError(f"Unexpected data type: {dtype}")

        return out_inner

    def check_bounds(self, obs: list[NDArray], atol: float) -> NDArray[np.bool_]:
        """
        Check if the observation points violated the bounds on each dimension.
        """
        ndims = self.ndims()
        out = np.array([False] * ndims)

        dtype = self.coeffs.data.dtype
        if dtype == np.float64:
            check_bounds_regular_f64(
                self.dims,
                self.starts.data,
                self.steps.data,
                [x.flatten() for x in obs],
                atol,
                out,
            )
        elif dtype == np.float32:
            check_bounds_regular_f32(
                self.dims,
                self.starts.data,
                self.steps.data,
                [x.flatten() for x in obs],
                atol,
                out,
            )
        else:
            raise TypeError(f"Unexpected data type: {dtype}")

        return out
