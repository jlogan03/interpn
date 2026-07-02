from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from interpn import (
    MulticubicRegular,
    MulticubicRectilinear,
    MultiBsplineRegular,
    MultiBsplineRectilinear,
)

INTERPN_CUBIC_LABEL = "InterpN Cubic"
INTERPN_BSPLINE_LABEL = "InterpN B-Spline"


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
        ("Quadratic", lambda x: x**2, lambda x: 2.0 * x, 0.5),
        ("Sine", np.sin, np.cos, 0.5),
        ("Step", _step, lambda x: np.zeros_like(x), 0.5),
    ]

    for kind in ["Regular", "Rectilinear"]:
        fn_titles = [name for name, *_ in fn_defs]
        subplot_titles = (
            fn_titles
            + [f"Error, {name}" for name in fn_titles]
            + [f"Gradient, {name}" for name in fn_titles]
            + [f"Gradient Error, {name}" for name in fn_titles]
        )
        fig_1d = make_subplots(
            rows=4,
            cols=3,
            shared_xaxes=True,
            subplot_titles=subplot_titles,
            vertical_spacing=0.09,
            horizontal_spacing=0.07,
        )
        legend_tracker: set[str] = set()

        for i, (_fnname, fn, grad_fn, data_res) in enumerate(fn_defs):
            xdata = np.arange(-2.0, 2.5, data_res)
            if kind == "Rectilinear":
                xdata += rng.uniform(-0.45 * data_res, 0.45 * data_res, xdata.size)
            ydata = fn(xdata)

            xinterp = np.arange(-3.0, 3.05, data_res / 100)

            if kind == "Regular":
                dims = np.asarray([xdata.size])
                starts = np.asarray([-2.0])
                steps = np.asarray([data_res])
                interpn_interp = MulticubicRegular.new(
                    dims, starts, steps, ydata, linearize_extrapolation=False
                )
                bspline_interp = MultiBsplineRegular.new(
                    dims, starts, steps, ydata, linearize_extrapolation=False
                )
            else:
                interpn_interp = MulticubicRectilinear.new(
                    [xdata], ydata, linearize_extrapolation=False
                )
                bspline_interp = MultiBsplineRectilinear.new(
                    [xdata], ydata, linearize_extrapolation=False
                )

            y_interpn = interpn_interp.eval([xinterp])
            y_interpn_grad = interpn_interp.eval_grad([xinterp])[0]
            y_bspline = bspline_interp.eval([xinterp])
            y_bspline_grad = bspline_interp.eval_grad([xinterp])[0]
            scipy_interp = RegularGridInterpolator(
                [xdata], ydata, bounds_error=None, fill_value=None, method="cubic"
            )
            y_sp = scipy_interp(xinterp)
            y_sp_grad = scipy_interp(xinterp[:, None], nu=(1,))

            col = i + 1
            _add_interpolating_region(fig_1d, 1, col, xdata.min(), xdata.max())
            _add_interpolating_region(fig_1d, 2, col, xdata.min(), xdata.max())
            _add_interpolating_region(fig_1d, 3, col, xdata.min(), xdata.max())
            _add_interpolating_region(fig_1d, 4, col, xdata.min(), xdata.max())

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
                    name=INTERPN_CUBIC_LABEL,
                    legendgroup="interpn",
                    showlegend=INTERPN_CUBIC_LABEL not in legend_tracker,
                ),
                row=1,
                col=col,
            )
            legend_tracker.add(INTERPN_CUBIC_LABEL)

            if y_bspline is not None:
                fig_1d.add_trace(
                    go.Scatter(
                        x=xinterp,
                        y=y_bspline,
                        mode="lines",
                        line=dict(color="#1f77b4", width=2, dash="dash"),
                        name=INTERPN_BSPLINE_LABEL,
                        legendgroup="multibspline",
                        showlegend=INTERPN_BSPLINE_LABEL not in legend_tracker,
                    ),
                    row=1,
                    col=col,
                )
                legend_tracker.add(INTERPN_BSPLINE_LABEL)

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
                    name=f"{INTERPN_CUBIC_LABEL} Error",
                    legendgroup="interpn_err",
                    showlegend=f"{INTERPN_CUBIC_LABEL} Error" not in legend_tracker,
                ),
                row=2,
                col=col,
            )
            legend_tracker.add(f"{INTERPN_CUBIC_LABEL} Error")
            if y_bspline is not None:
                fig_1d.add_trace(
                    go.Scatter(
                        x=xinterp,
                        y=y_bspline - truth,
                        mode="lines",
                        line=dict(color="#1f77b4", width=2, dash="dash"),
                        name=f"{INTERPN_BSPLINE_LABEL} Error",
                        legendgroup="multibspline_err",
                        showlegend=f"{INTERPN_BSPLINE_LABEL} Error"
                        not in legend_tracker,
                    ),
                    row=2,
                    col=col,
                )
                legend_tracker.add(f"{INTERPN_BSPLINE_LABEL} Error")
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

            truth_grad = grad_fn(xinterp)
            fig_1d.add_trace(
                go.Scatter(
                    x=xinterp,
                    y=truth_grad,
                    mode="lines",
                    line=dict(color="#2ca02c", width=2),
                    name="Exact Gradient",
                    legendgroup="exact_grad",
                    showlegend="Exact Gradient" not in legend_tracker,
                ),
                row=3,
                col=col,
            )
            legend_tracker.add("Exact Gradient")
            fig_1d.add_trace(
                go.Scatter(
                    x=xinterp,
                    y=y_interpn_grad,
                    mode="lines",
                    line=dict(color="black", width=2),
                    name=f"{INTERPN_CUBIC_LABEL} Gradient",
                    legendgroup="interpn_grad",
                    showlegend=f"{INTERPN_CUBIC_LABEL} Gradient" not in legend_tracker,
                ),
                row=3,
                col=col,
            )
            legend_tracker.add(f"{INTERPN_CUBIC_LABEL} Gradient")
            fig_1d.add_trace(
                go.Scatter(
                    x=xinterp,
                    y=y_bspline_grad,
                    mode="lines",
                    line=dict(color="#1f77b4", width=2, dash="dash"),
                    name=f"{INTERPN_BSPLINE_LABEL} Gradient",
                    legendgroup="multibspline_grad",
                    showlegend=f"{INTERPN_BSPLINE_LABEL} Gradient"
                    not in legend_tracker,
                ),
                row=3,
                col=col,
            )
            legend_tracker.add(f"{INTERPN_BSPLINE_LABEL} Gradient")
            fig_1d.add_trace(
                go.Scatter(
                    x=xinterp,
                    y=y_sp_grad,
                    mode="lines",
                    line=dict(color="black", width=2, dash="dot"),
                    opacity=0.7,
                    name="Scipy Gradient",
                    legendgroup="scipy_grad",
                    showlegend="Scipy Gradient" not in legend_tracker,
                ),
                row=3,
                col=col,
            )
            legend_tracker.add("Scipy Gradient")

            fig_1d.add_trace(
                go.Scatter(
                    x=xinterp,
                    y=y_interpn_grad - truth_grad,
                    mode="lines",
                    line=dict(color="black", width=2),
                    name=f"{INTERPN_CUBIC_LABEL} Gradient Error",
                    legendgroup="interpn_grad_err",
                    showlegend=f"{INTERPN_CUBIC_LABEL} Gradient Error"
                    not in legend_tracker,
                ),
                row=4,
                col=col,
            )
            legend_tracker.add(f"{INTERPN_CUBIC_LABEL} Gradient Error")
            fig_1d.add_trace(
                go.Scatter(
                    x=xinterp,
                    y=y_bspline_grad - truth_grad,
                    mode="lines",
                    line=dict(color="#1f77b4", width=2, dash="dash"),
                    name=f"{INTERPN_BSPLINE_LABEL} Gradient Error",
                    legendgroup="multibspline_grad_err",
                    showlegend=f"{INTERPN_BSPLINE_LABEL} Gradient Error"
                    not in legend_tracker,
                ),
                row=4,
                col=col,
            )
            legend_tracker.add(f"{INTERPN_BSPLINE_LABEL} Gradient Error")
            fig_1d.add_trace(
                go.Scatter(
                    x=xinterp,
                    y=y_sp_grad - truth_grad,
                    mode="lines",
                    line=dict(color="black", width=2, dash="dot"),
                    opacity=0.7,
                    name="Scipy Gradient Error",
                    legendgroup="scipy_grad_err",
                    showlegend="Scipy Gradient Error" not in legend_tracker,
                ),
                row=4,
                col=col,
            )
            legend_tracker.add("Scipy Gradient Error")

        for col in range(1, 4):
            fig_1d.update_xaxes(title_text="x", row=4, col=col)
        fig_1d.update_yaxes(title_text="f(x)", row=1, col=1)
        fig_1d.update_yaxes(title_text="Error", row=2, col=1)
        fig_1d.update_yaxes(title_text="df/dx", row=3, col=1)
        fig_1d.update_yaxes(title_text="Gradient Error", row=4, col=1)
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
        title_methods = f"{INTERPN_CUBIC_LABEL} vs. {INTERPN_BSPLINE_LABEL} vs. Scipy"
        fig_1d.update_layout(
            title=dict(
                text=(
                    f"Comparison — {title_methods} w/ Cubic Interpolant<br>{kind} Grid"
                ),
                y=0.97,
                yanchor="top",
            ),
            height=900,
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
        gx_truth = 2.0 * xinterpmesh
        gy_truth = 2.0 * yinterpmesh
        grad_truth = np.sqrt(gx_truth**2 + gy_truth**2)
        obs_2d = [xinterpmesh.flatten(), yinterpmesh.flatten()]
        scipy_points = np.column_stack(obs_2d)

        if kind == "Regular":
            dims = np.asarray([xdata.size, ydata.size])
            starts = np.asarray([-3.0, -3.0])
            steps = np.asarray([xmesh[1, 0] - xmesh[0, 0], ymesh[0, 1] - ymesh[0, 0]])
            interpn_interp = MulticubicRegular.new(
                dims, starts, steps, zmesh, linearize_extrapolation=False
            )
            bspline_interp = MultiBsplineRegular.new(
                dims, starts, steps, zmesh, linearize_extrapolation=False
            )
        else:
            interpn_interp = MulticubicRectilinear.new(
                [xdata, ydata], zmesh, linearize_extrapolation=False
            )
            bspline_interp = MultiBsplineRectilinear.new(
                [xdata, ydata], zmesh, linearize_extrapolation=False
            )

        z_interpn = interpn_interp.eval(obs_2d).reshape(xinterpmesh.shape)
        grad_interpn_raw = interpn_interp.eval_grad(obs_2d)
        gx_interpn = grad_interpn_raw[0].reshape(xinterpmesh.shape)
        gy_interpn = grad_interpn_raw[1].reshape(xinterpmesh.shape)
        grad_interpn = np.sqrt(gx_interpn**2 + gy_interpn**2)
        grad_interpn_err = np.sqrt(
            (gx_interpn - gx_truth) ** 2 + (gy_interpn - gy_truth) ** 2
        )

        z_bspline = bspline_interp.eval(obs_2d).reshape(xinterpmesh.shape)
        grad_bspline_raw = bspline_interp.eval_grad(obs_2d)
        gx_bspline = grad_bspline_raw[0].reshape(xinterpmesh.shape)
        gy_bspline = grad_bspline_raw[1].reshape(xinterpmesh.shape)
        grad_bspline = np.sqrt(gx_bspline**2 + gy_bspline**2)
        grad_bspline_err = np.sqrt(
            (gx_bspline - gx_truth) ** 2 + (gy_bspline - gy_truth) ** 2
        )

        scipy_interp = RegularGridInterpolator(
            [xdata, ydata], zmesh, bounds_error=None, fill_value=None, method="cubic"
        )
        z_sp = scipy_interp((xinterpmesh, yinterpmesh))
        gx_sp = scipy_interp(scipy_points, nu=(1, 0)).reshape(xinterpmesh.shape)
        gy_sp = scipy_interp(scipy_points, nu=(0, 1)).reshape(xinterpmesh.shape)
        grad_sp = np.sqrt(gx_sp**2 + gy_sp**2)
        grad_sp_err = np.sqrt((gx_sp - gx_truth) ** 2 + (gy_sp - gy_truth) ** 2)

        ncols_2d = 4 if z_bspline is not None else 3
        heatmap_specs = [{"type": "heatmap"}] * ncols_2d
        top_data = [
            (zinterp, "Truth"),
            (z_interpn, INTERPN_CUBIC_LABEL),
        ]
        if z_bspline is not None:
            top_data.append((z_bspline, INTERPN_BSPLINE_LABEL))
        top_data.append((z_sp, "Scipy"))
        bottom_data = [
            (z_interpn - zinterp, f"Error, {INTERPN_CUBIC_LABEL}"),
        ]
        if z_bspline is not None:
            bottom_data.append((z_bspline - zinterp, f"Error, {INTERPN_BSPLINE_LABEL}"))
        bottom_data.append((z_sp - zinterp, "Error, Scipy"))
        gradient_data = [
            (grad_truth, "Gradient Norm, Truth"),
            (grad_interpn, f"Gradient Norm, {INTERPN_CUBIC_LABEL}"),
        ]
        if z_bspline is not None:
            gradient_data.append(
                (grad_bspline, f"Gradient Norm, {INTERPN_BSPLINE_LABEL}")
            )
        gradient_data.append((grad_sp, "Gradient Norm, Scipy"))
        gradient_error_data = [
            (grad_interpn_err, f"Gradient Error Norm, {INTERPN_CUBIC_LABEL}"),
        ]
        if z_bspline is not None:
            gradient_error_data.append(
                (grad_bspline_err, f"Gradient Error Norm, {INTERPN_BSPLINE_LABEL}")
            )
        gradient_error_data.append((grad_sp_err, "Gradient Error Norm, Scipy"))
        subplot_titles_2d = (
            [name for _, name in top_data]
            + ["", *[name for _, name in bottom_data]]
            + [name for _, name in gradient_data]
            + ["", *[name for _, name in gradient_error_data]]
        )

        fig_2d = make_subplots(
            rows=4,
            cols=ncols_2d,
            specs=[heatmap_specs, heatmap_specs, heatmap_specs, heatmap_specs],
            subplot_titles=subplot_titles_2d,
            horizontal_spacing=0.06,
            vertical_spacing=0.09,
        )

        for col, (z_data, title) in enumerate(top_data, start=1):
            showscale = col == ncols_2d
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

        for row in (2, 4):
            for col in range(2, ncols_2d + 1):
                fig_2d.add_shape(
                    type="rect",
                    x0=-3.0,
                    x1=3.0,
                    y0=-3.0,
                    y1=3.0,
                    line=dict(color="white"),
                    row=row,
                    col=col,
                )
        for col in range(1, ncols_2d + 1):
            fig_2d.add_shape(
                type="rect",
                x0=-3.0,
                x1=3.0,
                y0=-3.0,
                y1=3.0,
                line=dict(color="white"),
                row=3,
                col=col,
            )

        for col, (z_data, name) in enumerate(bottom_data, start=2):
            showscale = col == ncols_2d
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

        for col, (z_data, name) in enumerate(gradient_data, start=1):
            showscale = col == ncols_2d
            fig_2d.add_trace(
                go.Heatmap(
                    x=xinterp,
                    y=yinterp,
                    z=z_data.T,
                    coloraxis="coloraxis3",
                    showscale=showscale,
                    name=name,
                ),
                row=3,
                col=col,
            )

        for col, (z_data, name) in enumerate(gradient_error_data, start=2):
            showscale = col == ncols_2d
            fig_2d.add_trace(
                go.Heatmap(
                    x=xinterp,
                    y=yinterp,
                    z=z_data.T,
                    coloraxis="coloraxis4",
                    showscale=showscale,
                    name=name,
                ),
                row=4,
                col=col,
            )

        for row in (1, 2, 3, 4):
            for col in range(1, ncols_2d + 1):
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
            height=900,
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
                colorbar=dict(len=0.18, x=1.05, y=0.88),
            ),
            coloraxis2=dict(
                colorscale=[
                    [0.0, "#000000"],
                    [0.5, "#ffffff"],
                    [1.0, "#000000"],
                ],
                cmid=0.0,
                colorbar=dict(len=0.18, x=1.05, y=0.63),
            ),
            coloraxis3=dict(
                colorscale=[
                    [0.0, "#ffffff"],
                    [1.0, "#000000"],
                ],
                colorbar=dict(len=0.18, x=1.05, y=0.37),
            ),
            coloraxis4=dict(
                colorscale=[
                    [0.0, "#ffffff"],
                    [1.0, "#000000"],
                ],
                colorbar=dict(len=0.18, x=1.05, y=0.12),
            ),
            font=dict(color="black"),
        )
        scale_axes = (
            [(1, col) for col in range(1, ncols_2d + 1)]
            + [(2, col) for col in range(2, ncols_2d + 1)]
            + [(3, col) for col in range(1, ncols_2d + 1)]
            + [(4, col) for col in range(2, ncols_2d + 1)]
        )
        for row, col in scale_axes:
            x_name = _axis_name("x", row, col, ncols_2d)
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
