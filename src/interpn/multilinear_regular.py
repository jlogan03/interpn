from __future__ import annotations

from typing import Optional
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
    interpn_linear_regular_f64,
    interpn_linear_regular_f32,
    check_bounds_regular_f64,
    check_bounds_regular_f32,
)


class MultilinearRegular(BaseModel):
    """
    Multilinear interpolation on a regular grid in up to 8 dimensions.

    All array inputs must be of the same type, either np.float32 or np.float64
    and must be 1D and contiguous.
    """

    # Immutable after initialization checks
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    dims: list[int]
    starts: Array
    steps: Array
    vals: Array

    @classmethod
    def new(
        cls, dims: list[int], starts: NDArray, steps: NDArray, vals: NDArray
    ) -> MultilinearRegular:
        """
        Initialize interpolator and check types and dimensions, casting other arrays
        to the same type as `vals` if they do not match, and flattening and/or
        reallocating into contiguous storage if necessary.

        This method exists primarily to remove boilerplate introduced by
        mixing pydantic and numpy.

        Args:
            dims: Number of elements on each dimension of the grid
            starts: Starting point of each dimension of the grid
            steps: Step size on each dimension of the grid
            vals: Values at grid points in C-style ordering,
                  as obtained from np.meshgrid(..., indexing="ij")

        Returns:
            A new MultilinearRegular interpolator instance.
        """
        dtype = vals.dtype
        arrtype = ArrayF64 if dtype == np.float64 else ArrayF32
        interpolator = MultilinearRegular(
            dims=dims,
            starts=arrtype(data=starts.flatten()),
            steps=arrtype(data=steps.flatten()),
            vals=arrtype(data=vals.flatten()),
        )

        return interpolator

    @model_validator(mode="after")
    def _validate_model(self):
        """Check that all inputs are contiguous and of the same data type,
        and that the grid dimensions and values make sense."""
        ndims = self.ndims()
        assert (
            ndims <= 8 and ndims >= 1
        ), "Number of dimensions must be at least 1 and no more than 8"
        assert self.starts.data.size == ndims, "Grid dimension mismatch"
        assert self.steps.data.size == ndims, "Grid dimension mismatch"
        assert self.vals.data.size == reduce(
            lambda acc, x: acc * x, self.dims
        ), "Size of value array does not match grid dims"
        assert all(
            [x > 0.0 for x in self.steps.data]
        ), "All grid steps must be positive and nonzero"
        assert all(
            [x.data.dtype == self.vals.data.dtype for x in [self.steps, self.vals]]
        ), "All grid inputs must be of the same data type (np.float32 or np.float64)"
        assert all(
            [x.data.data.contiguous for x in [self.starts, self.steps, self.vals]]
        ), "Grid data must be contiguous"

        return self

    def ndims(self) -> int:
        return len(self.dims)

    def eval(self, obs: list[NDArray], out: Optional[NDArray] = None) -> NDArray:
        """Evaluate the interpolator at a set of observation points,
        optionally writing the output into a preallocated array.

        This function does not reallocate inputs, and will error if the
        inputs are not contiguous or are of the wrong data type.

        Args:
            obs: [x, y, ...] coordinates of observation points.
            out: Optional preallocated array for output. Defaults to None.

        Raises:
            TypeError: If data type is not np.float32 or np.float64
            AssertionError: If input data is not contiguous or dimensions do not match

        Returns:
            Array of evaluated values in the same shape and data type as obs[0]
        """
        # Allocate output if it was not provided
        out_inner = out if out is not None else np.zeros_like(obs[0])
        self.eval_unchecked(obs, out_inner)

        return out_inner

    def eval_unchecked(
        self, obs: list[NDArray], out: Optional[NDArray] = None
    ) -> NDArray:
        """Evaluate the interpolator at a set of observation points,
        optionally writing the output into a preallocated array,
        and skipping checks on the dimensionality and contiguousness
        of the inputs.

        This function does not reallocate inputs, and will error in a lower-level
        function if the inputs are not contiguous or are of the wrong data type.

        Args:
            obs: [x, y, ...] coordinates of observation points.
            out: Optional preallocated array for output. Defaults to None.

        Raises:
            TypeError: If data type is not np.float32 or np.float64

        Returns:
            Array of evaluated values in the same shape and data type as obs[0]
        """
        dtype = self.vals.data.dtype
        out_inner = out if out is not None else np.zeros_like(obs[0])

        if dtype == np.float64:
            interpn_linear_regular_f64(
                self.dims,
                self.starts.data,
                self.steps.data,
                self.vals.data,
                obs,
                out_inner,
            )
        elif dtype == np.float32:
            interpn_linear_regular_f32(
                self.dims,
                self.starts.data,
                self.steps.data,
                self.vals.data,
                obs,
                out_inner,
            )
        else:
            raise TypeError(f"Unexpected data type: {dtype}")

        return out_inner

    def check_bounds(self, obs: list[NDArray], atol: float) -> NDArray[np.bool_]:
        """
        Check if the observation points violated the bounds on each dimension.

        This performs a (small) allocation for the output.

        Args:
            obs: [x, y, ...] coordinates of observation points.
            atol: Absolute tolerance on bounds.

        Raises:
            TypeError: If an unexpected data type is encountered

        Returns:
            An array of flags for each dimension, each True if that dimension's
            bounds were violated.
        """
        ndims = self.ndims()
        out = np.array([False] * ndims)

        dtype = self.vals.data.dtype
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
