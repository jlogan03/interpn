"""
Generate a quality-of-fit comparison figure for nearest-neighbor interpolation.

This script compares InterpN's nearest-neighbor interpolator against SciPy's
``griddata`` with the ``nearest`` method on a slightly irregular rectilinear grid.
The resulting SVG is saved into the ``docs`` directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from interpn import NearestRectilinear


def _truth(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Synthetic test function with curvature and mixed terms."""
    return np.sin(x) + 0.5 * np.cos(2.0 * y) + 0.15 * x * y


def _build_rectilinear_grid(
    rng: np.random.Generator, size: int, start: float, stop: float
) -> np.ndarray:
    """Create a slightly irregular rectilinear grid that stays monotonic."""
    base = np.linspace(start, stop, size, dtype=np.float64)
    step = (stop - start) / (size - 1)
    deltas = step + rng.uniform(-0.3 * step, 0.3 * step, size - 1)
    deltas = np.maximum(deltas, 0.1 * step)
    coords = np.concatenate(([base[0]], base[0] + np.cumsum(deltas)))
    scale = (stop - start) / (coords[-1] - coords[0])
    coords = (coords - coords[0]) * scale + start
    coords[-1] = stop
    return coords


if __name__ == "__main__":
    rng = np.random.default_rng(6)

    xdata = _build_rectilinear_grid(rng, size=25, start=-3.0, stop=3.0)
    ydata = _build_rectilinear_grid(rng, size=18, start=-2.5, stop=2.5)
    xmesh, ymesh = np.meshgrid(xdata, ydata, indexing="ij")
    zmesh = _truth(xmesh, ymesh)

    x_eval = np.linspace(-3.6, 3.6, 160)
    y_eval = np.linspace(-3.0, 3.0, 160)
    x_eval_mesh, y_eval_mesh = np.meshgrid(x_eval, y_eval, indexing="ij")
    z_truth = _truth(x_eval_mesh, y_eval_mesh)

    interpn_vals = (
        NearestRectilinear.new([xdata, ydata], zmesh)
        .eval([x_eval_mesh.flatten(), y_eval_mesh.flatten()])
        .reshape(x_eval_mesh.shape)
    )

    points = np.column_stack((xmesh.flatten(), ymesh.flatten()))
    griddata_vals = griddata(
        points, zmesh.flatten(), (x_eval_mesh, y_eval_mesh), method="nearest"
    )

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    plt.suptitle(
        "Nearest-Neighbor Quality of Fit\nInterpN vs. SciPy griddata (nearest)",
        fontsize=14,
    )

    plots = [
        (z_truth, "Truth"),
        (interpn_vals, "InterpN NearestRectilinear"),
        (griddata_vals, "SciPy griddata (nearest)"),
        (interpn_vals - z_truth, "Error: InterpN"),
        (griddata_vals - z_truth, "Error: SciPy"),
        (griddata_vals - interpn_vals, "SciPy - InterpN"),
    ]

    extent = [x_eval.min(), x_eval.max(), y_eval.min(), y_eval.max()]
    for ax, (data, title) in zip(axes, plots, strict=True):
        im = ax.imshow(
            data.T,
            origin="lower",
            extent=extent,
            aspect="auto",
        )
        ax.set_title(title, fontsize=11)
        ax.scatter(
            xmesh.flatten(),
            ymesh.flatten(),
            s=4,
            c="k",
            alpha=0.6,
            label="Data",
        )
        ax.add_patch(
            plt.Rectangle(
                (xdata[0], ydata[0]),
                xdata[-1] - xdata[0],
                ydata[-1] - ydata[0],
                edgecolor="white",
                linewidth=1.0,
                fill=False,
                label="Grid extent",
            )
        )
        ax.legend(loc="lower right", fontsize=7, facecolor="white", framealpha=0.8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.tight_layout()
    fig.show()
    output_path = Path(__file__).parent / "../docs/nearest_quality_of_fit.svg"
    fig.savefig(output_path)
