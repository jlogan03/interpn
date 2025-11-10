"""
Generate a quality-of-fit comparison figure for nearest-neighbor interpolation.

This script compares InterpN's nearest-neighbor interpolator against SciPy's
``griddata`` with the ``nearest`` method on a slightly irregular rectilinear grid.
The resulting SVG is saved into the ``docs`` directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def _add_grid_outline(
    fig: go.Figure,
    row: int,
    col: int,
    xrange: tuple[float, float],
    yrange: tuple[float, float],
) -> None:
    fig.add_shape(
        type="rect",
        x0=xrange[0],
        x1=xrange[1],
        y0=yrange[0],
        y1=yrange[1],
        line=dict(color="white"),
        row=row,
        col=col,
    )


def _axis_name(prefix: str, row: int, col: int, ncols: int) -> str:
    idx = (row - 1) * ncols + col
    return prefix if idx == 1 else f"{prefix}{idx}"


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

    plots = [
        (z_truth, "Truth", False),
        (interpn_vals, "InterpN", False),
        (griddata_vals, "SciPy", False),
        (interpn_vals - z_truth, "Error: InterpN", True),
        (griddata_vals - z_truth, "Error: SciPy", True),
        (griddata_vals - interpn_vals, "SciPy - InterpN", True),
    ]

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[title for _, title, _ in plots],
        horizontal_spacing=0.05,
        vertical_spacing=0.16,
    )

    for idx, (data, title, is_error) in enumerate(plots, start=1):
        row = 1 if idx <= 3 else 2
        col = idx - 3 if idx > 3 else idx
        coloraxis = "coloraxis2" if is_error else "coloraxis1"
        showscale = (row == 1 and col == 3) or (row == 2 and col == 3)
        fig.add_trace(
            go.Heatmap(
                x=x_eval,
                y=y_eval,
                z=data.T,
                coloraxis=coloraxis,
                showscale=showscale,
                name=title,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=xmesh.flatten(),
                y=ymesh.flatten(),
                mode="markers",
                marker=dict(color="white", size=4, line=dict(color="black", width=0.5)),
                name="Grid samples",
                legendgroup="samples",
                showlegend=idx == 1,
            ),
            row=row,
            col=col,
        )
        _add_grid_outline(
            fig, row, col, xrange=(xdata[0], xdata[-1]), yrange=(ydata[0], ydata[-1])
        )

    for row in (1, 2):
        for col in (1, 2, 3):
            fig.update_xaxes(
                showticklabels=False,
                title_text="",
                showgrid=False,
                zeroline=False,
                row=row,
                col=col,
                showline=False,
            )
            fig.update_yaxes(
                showticklabels=False,
                title_text="",
                showgrid=False,
                zeroline=False,
                row=row,
                col=col,
                showline=False,
            )

    fig.update_layout(
        title=dict(
            text="Nearest-Neighbor Quality of Fit —"
            " InterpN vs. SciPy griddata (nearest)",
            y=0.97,
            yanchor="top",
        ),
        height=500,
        margin=dict(t=70, l=60, r=80, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            x=0.0,
            xanchor="left",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis1=dict(
            colorscale=[
                [0.0, "#ffffff"],
                [1.0, "#000000"],
            ],
            colorbar=dict(len=0.55, x=1.2, y=0.78),
        ),
        coloraxis2=dict(
            colorscale=[
                [0.0, "#000000"],
                [0.5, "#ffffff"],
                [1.0, "#000000"],
            ],
            cmid=0.0,
            colorbar=dict(len=0.55, x=1.2, y=0.25),
        ),
        font=dict(color="black"),
    )
    for row in (1, 2):
        for col in (1, 2, 3):
            x_name = _axis_name("x", row, col, 3)
            fig.update_yaxes(
                scaleanchor=x_name,
                scaleratio=1,
                row=row,
                col=col,
            )

    output_path = Path(__file__).parent / "../docs/nearest_quality_of_fit.svg"
    fig.write_image(str(output_path))
    fig.write_html(
        str(output_path.with_suffix(".html")), include_plotlyjs="cdn", full_html=False
    )
    fig.show()
