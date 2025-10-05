"""
Python bindings to the `interpn` Rust library
for N-dimensional interpolation and extrapolation.
"""
from __future__ import annotations

from importlib.metadata import version

from .multilinear_regular import MultilinearRegular
from .multilinear_rectilinear import MultilinearRectilinear
from .multicubic_regular import MulticubicRegular
from .multicubic_rectilinear import MulticubicRectilinear
from interpn import raw

__version__ = version("interpn")

__all__ = [
    "__version__",
    "MultilinearRegular",
    "MultilinearRectilinear",
    "MulticubicRegular",
    "MulticubicRectilinear",
    "raw",
]
