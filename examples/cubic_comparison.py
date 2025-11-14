from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from interpn import MulticubicRegular, MulticubicRectilinear


def _step(x: np.ndarray) -> np.ndarray:
    y = np.ones_like(x)
    y[np.where(x < 0.0)] = 0.0
    y[np.where(x >= 0.0)] = 1.0
    return y


def _add_interpolating_region(
    fig: go.Figure, row: int, col: int, xmin: float, xmax: float
) -> None:
    fig.add_vrect(
        x0=float(xmin),
        x1=float(xmax),
        row=row,
        col=col,
        fillcolor="green",
        opacity=0.00,
        layer="below",
        line_width=0,
    )


def _axis_name(prefix: str, row: int, col: int, ncols: int) -> str:
    idx = (row - 1) * ncols + col
    if idx == 1:
        return prefix
    return f"{prefix}{idx}"


if __name__ == "__main__":
    rng = np.random.RandomState(42)

    fn_defs = [
        ("Quadratic", lambda x: x**2, 0.5),
        ("Sine", np.sin, 0.5),
        ("Step", _step, 0.5),
    ]

    for kind in ["Regular", "Rectilinear"]:
        fn_titles = [name for name, *_ in fn_defs]
        subplot_titles = fn_titles + [f"Error, {name}" for name in fn_titles]
        fig_1d = make_subplots(
            rows=2,
            cols=3,
            shared_xaxes=True,
            subplot_titles=subplot_titles,
            vertical_spacing=0.2,
            horizontal_spacing=0.07,
        )
        legend_tracker: set[str] = set()

        for i, (_fnname, fn, data_res) in enumerate(fn_defs):
            xdata = np.arange(-2.0, 2.5, data_res)
            if kind == "Rectilinear":
                xdata += rng.uniform(-0.45 * data_res, 0.45 * data_res, xdata.size)
            ydata = fn(xdata)

            xinterp = np.arange(-3.0, 3.05, data_res / 100)

            if kind == "Regular":
                dims = np.asarray([xdata.size])
                starts = np.asarray([-2.0])
                steps = np.asarray([data_res])
                y_interpn = MulticubicRegular.new(
                    dims, starts, steps, ydata, linearize_extrapolation=False
                ).eval([xinterp])
            else:
                y_interpn = MulticubicRectilinear.new(
                    [xdata], ydata, linearize_extrapolation=False
                ).eval([xinterp])

            y_sp = RegularGridInterpolator(
                [xdata], ydata, bounds_error=None, fill_value=None, method="cubic"
            )(xinterp)

            col = i + 1
            _add_interpolating_region(fig_1d, 1, col, xdata.min(), xdata.max())
            _add_interpolating_region(fig_1d, 2, col, xdata.min(), xdata.max())

            fig_1d.add_trace(
                go.Scatter(
                    x=xdata,
                    y=ydata,
                    mode="markers",
                    marker=dict(color="black", size=6),
                    name="Data",
                    legendgroup="data",
                    showlegend="Data" not in legend_tracker,
                ),
                row=1,
                col=col,
            )
            legend_tracker.add("Data")

            fig_1d.add_trace(
                go.Scatter(
                    x=xinterp,
                    y=y_interpn,
                    mode="lines",
                    line=dict(color="black", width=2),
                    name="InterpN",
                    legendgroup="interpn",
                    showlegend="InterpN" not in legend_tracker,
                ),
                row=1,
                col=col,
            )
            legend_tracker.add("InterpN")

            fig_1d.add_trace(
                go.Scatter(
                    x=xinterp,
                    y=y_sp,
                    mode="lines",
                    line=dict(color="black", width=2, dash="dot"),
                    opacity=0.7,
                    name="Scipy",
                    legendgroup="scipy",
                    showlegend="Scipy" not in legend_tracker,
                ),
                row=1,
                col=col,
            )
            legend_tracker.add("Scipy")

            truth = fn(xinterp)
            fig_1d.add_trace(
                go.Scatter(
                    x=xinterp,
                    y=y_interpn - truth,
                    mode="lines",
                    line=dict(color="black", width=2),
                    name="InterpN Error",
                    legendgroup="interpn_err",
                    showlegend="InterpN Error" not in legend_tracker,
                ),
                row=2,
                col=col,
            )
            legend_tracker.add("InterpN Error")
            fig_1d.add_trace(
                go.Scatter(
                    x=xinterp,
                    y=y_sp - truth,
                    mode="lines",
                    line=dict(color="black", width=2, dash="dot"),
                    opacity=0.7,
                    name="Scipy Error",
                    legendgroup="scipy_err",
                    showlegend="Scipy Error" not in legend_tracker,
                ),
                row=2,
                col=col,
            )
            legend_tracker.add("Scipy Error")

        for col in range(1, 4):
            fig_1d.update_xaxes(title_text="x", row=2, col=col)
        fig_1d.update_yaxes(title_text="f(x)", row=1, col=1)
        fig_1d.update_yaxes(title_text="Error", row=2, col=1)
        fig_1d.update_xaxes(
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
            ticks="outside",
            tickcolor="black",
            showgrid=False,
            zeroline=False,
        )
        fig_1d.update_yaxes(
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
            ticks="outside",
            tickcolor="black",
            showgrid=False,
            zeroline=False,
        )
        fig_1d.update_layout(
            title=dict(
                text=(
                    "Comparison — InterpN  vs. Scipy"
                    f" w/ Cubic Interpolant<br>{kind} Grid"
                ),
                y=0.97,
                yanchor="top",
            ),
            height=500,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.0,
                x=1.02,
                xanchor="left",
            ),
            margin=dict(t=80, l=60, r=200, b=80),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="black"),
        )
        output_1d = Path(__file__).parent / f"../docs/1d_quality_of_fit_{kind}.svg"
        fig_1d.write_image(str(output_1d))
        fig_1d.write_html(
            str(output_1d.with_suffix(".html")), include_plotlyjs="cdn", full_html=False
        )

        xdata = np.linspace(-3.0, 3.0, 7, endpoint=True)
        ydata = np.linspace(-3.0, 3.0, 7, endpoint=True)
        data_res = xdata[1] - xdata[0]
        if kind == "Rectilinear":
            xdata[1:-1] += rng.uniform(
                -0.45 * data_res, 0.45 * data_res, xdata.size - 2
            )
            ydata[1:-1] += rng.uniform(
                -0.45 * data_res, 0.45 * data_res, ydata.size - 2
            )
        xmesh, ymesh = np.meshgrid(xdata, ydata, indexing="ij")
        zmesh = xmesh**2 + ymesh**2

        xinterp = np.linspace(-5.0, 5.0, 30, endpoint=True)
        yinterp = np.linspace(-5.0, 5.0, 30, endpoint=True)
        xinterpmesh, yinterpmesh = np.meshgrid(xinterp, yinterp, indexing="ij")
        zinterp = xinterpmesh**2 + yinterpmesh**2

        if kind == "Regular":
            dims = np.asarray([xdata.size, ydata.size])
            starts = np.asarray([-3.0, -3.0])
            steps = np.asarray([xmesh[1, 0] - xmesh[0, 0], ymesh[0, 1] - ymesh[0, 0]])
            z_interpn = (
                MulticubicRegular.new(
                    dims, starts, steps, zmesh, linearize_extrapolation=False
                )
                .eval([xinterpmesh.flatten(), yinterpmesh.flatten()])
                .reshape(xinterpmesh.shape)
            )
        else:
            z_interpn = (
                MulticubicRectilinear.new(
                    [xdata, ydata], zmesh, linearize_extrapolation=False
                )
                .eval([xinterpmesh.flatten(), yinterpmesh.flatten()])
                .reshape(xinterpmesh.shape)
            )

        z_sp = RegularGridInterpolator(
            [xdata, ydata], zmesh, bounds_error=None, fill_value=None, method="cubic"
        )((xinterpmesh, yinterpmesh))

        fig_2d = make_subplots(
            rows=2,
            cols=3,
            specs=[[{"type": "heatmap"}] * 3, [{"type": "heatmap"}] * 3],
            subplot_titles=[
                "Truth",
                "InterpN",
                "Scipy",
                "",
                "Error, InterpN",
                "Error, Scipy",
            ],
            horizontal_spacing=0.06,
            vertical_spacing=0.18,
        )

        colorbar_x_top = {1: 1.02, 2: 1.09, 3: 1.16}
        for col, (z_data, title) in enumerate(
            [
                (zinterp, "Truth"),
                (z_interpn, "InterpN"),
                (z_sp, "Scipy"),
            ],
            start=1,
        ):
            showscale = col == 3
            fig_2d.add_trace(
                go.Heatmap(
                    x=xinterp,
                    y=yinterp,
                    z=z_data.T,
                    coloraxis="coloraxis1",
                    showscale=showscale,
                    name=title,
                ),
                row=1,
                col=col,
            )
            fig_2d.add_trace(
                go.Contour(
                    x=xinterp,
                    y=yinterp,
                    z=z_data.T,
                    showscale=False,
                    line=dict(color="black"),
                    contours=dict(
                        coloring="none", showlabels=False, start=0, end=50, size=10
                    ),
                    showlegend=False,
                ),
                row=1,
                col=col,
            )
            fig_2d.add_trace(
                go.Scatter(
                    x=xmesh.flatten(),
                    y=ymesh.flatten(),
                    mode="markers",
                    marker=dict(
                        color="white", size=5, line=dict(color="black", width=0.5)
                    ),
                    name="Sampled data",
                    legendgroup="samples",
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )
            fig_2d.add_shape(
                type="rect",
                x0=-3.0,
                x1=3.0,
                y0=-3.0,
                y1=3.0,
                line=dict(color="white"),
                row=1,
                col=col,
            )

        fig_2d.add_shape(
            type="rect",
            x0=-3.0,
            x1=3.0,
            y0=-3.0,
            y1=3.0,
            line=dict(color="white"),
            row=2,
            col=2,
        )
        fig_2d.add_shape(
            type="rect",
            x0=-3.0,
            x1=3.0,
            y0=-3.0,
            y1=3.0,
            line=dict(color="white"),
            row=2,
            col=3,
        )

        colorbar_x_bottom = {2: 1.02, 3: 1.09}
        for col, (z_data, name) in enumerate(
            [
                (z_interpn - zinterp, "Error, InterpN"),
                (z_sp - zinterp, "Error, Scipy"),
            ],
            start=2,
        ):
            showscale = col == 3
            fig_2d.add_trace(
                go.Heatmap(
                    x=xinterp,
                    y=yinterp,
                    z=z_data.T,
                    coloraxis="coloraxis2",
                    showscale=showscale,
                    name=name,
                ),
                row=2,
                col=col,
            )

        for row in (1, 2):
            for col in (1, 2, 3):
                fig_2d.update_xaxes(
                    showticklabels=False,
                    title_text="",
                    showgrid=False,
                    zeroline=False,
                    row=row,
                    col=col,
                    showline=False,
                )
                fig_2d.update_yaxes(
                    showticklabels=False,
                    title_text="",
                    showgrid=False,
                    zeroline=False,
                    row=row,
                    col=col,
                    showline=False,
                )
        fig_2d.update_layout(
            title=dict(
                text=f"Quadratic Test Function w/ Cubic Interpolant<br>{kind} Grid",
                y=0.97,
                yanchor="top",
            ),
            height=500,
            margin=dict(t=80, l=60, r=40, b=80),
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
                colorbar=dict(len=0.4, x=1.2, y=0.25),
            ),
            font=dict(color="black"),
        )
        for row, col in [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)]:
            x_name = _axis_name("x", row, col, 3)
            fig_2d.update_yaxes(
                scaleanchor=x_name,
                scaleratio=1,
                row=row,
                col=col,
            )

        output_2d = Path(__file__).parent / f"../docs/2d_quality_of_fit_{kind}.svg"
        fig_2d.write_image(str(output_2d))
        fig_2d.write_html(
            str(output_2d.with_suffix(".html")), include_plotlyjs="cdn", full_html=False
        )

        fig_1d.show()
        fig_2d.show()
