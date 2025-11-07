# ruff: noqa B023

import gc
import os
from pathlib import Path

from timeit import Timer, timeit
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from interpn import (
    MultilinearRectilinear,
    MultilinearRegular,
    MulticubicRegular,
    MulticubicRectilinear,
    NearestRegular,
    NearestRectilinear,
)

# Toggle SciPy/NumPy baselines via environment for PGO workloads.
RUN_INTERPN_ONLY = os.environ.get("INTERPNPY_INTERPN_ONLY", "").lower() in {
    "1",
    "true",
    "yes",
}

TARGET_SAMPLE_SECONDS = 2.0
MAX_TIMER_LOOPS = 1_000_000_000


def average_call_time(
    func, points, target_seconds: float = TARGET_SAMPLE_SECONDS
) -> float:
    """Measure average execution time for func(points) using ~target_seconds of samples."""
    timer = Timer(lambda: func(points))
    gc.collect()
    calibrated_loops, total = timer.autorange()
    avg = total / calibrated_loops if total else 0.0
    fallback_loops = max(1, min(MAX_TIMER_LOOPS, calibrated_loops))
    if avg == 0.0:
        iterations = fallback_loops
    else:
        iterations = max(1, min(MAX_TIMER_LOOPS, int(target_seconds / avg) or 1))
    gc.collect()
    total = timer.timeit(iterations)
    return total / iterations


DASH_STYLES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]


def _normalized_line_style(index: int) -> str:
    return DASH_STYLES[index % len(DASH_STYLES)]


def fill_between(
    fig: go.Figure,
    *,
    x: NDArray,
    upper: NDArray,
    lower: NDArray,
    row: int,
    col: int,
    fillcolor: str = "rgba(139, 196, 59, 0.25)",
) -> None:
    clamped_upper = np.maximum(upper, lower)
    assert np.all(clamped_upper >= lower)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([clamped_upper, lower[::-1]]),
            mode="lines",
            line=dict(width=0),
            fill="toself",
            fillcolor=fillcolor,
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )


def _plot_normalized_vs_nobs(
    *,
    ns: list[int],
    throughputs: dict[str, list[float]],
    kinds: dict[str, str],
    title: str,
    output_path: Path,
) -> None:
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Linear", "Cubic"],
        horizontal_spacing=0.08,
    )
    for row, kind in enumerate(["Linear", "Cubic"], start=1):
        baseline_vals = throughputs.get(f"Scipy RegularGridInterpolator {kind}")
        if not baseline_vals:
            continue
        baseline_arr = np.array(baseline_vals)
        series = [
            (name, np.array(values))
            for name, values in throughputs.items()
            if kinds.get(name) == kind and values
        ]

        for idx, (label, values) in enumerate(series):
            min_len = min(len(values), len(baseline_arr))
            if min_len == 0:
                continue
            x_vals = np.array(ns[:min_len])
            ratios = values[:min_len] / baseline_arr[:min_len]
            is_baseline = label.startswith("Scipy RegularGridInterpolator")
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=np.ones_like(ratios) if is_baseline else ratios,
                    mode="lines",
                    line=dict(color="black", width=2, dash=_normalized_line_style(idx)),
                    marker=dict(
                        symbol="square"
                        if label.lower().startswith("scipy")
                        else "circle",
                        size=8,
                    ),
                    opacity=1.0,
                    name=label if not is_baseline else f"{label}<br>(baseline)",
                    showlegend=True,
                    legendgroup=kind,
                    legendgrouptitle_text=kind,
                ),
                row=row,
                col=1,
            )
            if not is_baseline and label.startswith("InterpN"):
                ones = np.ones_like(ratios)
                upper = np.maximum(ratios, ones)
                lower = np.minimum(ratios, ones)
                fill_between(
                    fig,
                    x=x_vals,
                    upper=upper,
                    lower=lower,
                    row=row,
                    col=1,
                )

    fig.update_xaxes(
        type="log",
        row=1,
        col=1,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="black",
        showgrid=False,
        zeroline=False,
    )
    fig.update_xaxes(
        type="log",
        title_text="Number of Observation Points",
        row=2,
        col=1,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="black",
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="Speedup vs. Scipy",
        row=1,
        col=1,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="black",
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        row=2,
        col=1,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="black",
        showgrid=False,
        zeroline=False,
    )
    fig.update_layout(
        title=dict(text=title, y=0.97, yanchor="top"),
        height=450,
        margin=dict(t=80, l=60, r=200, b=90),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            x=1.02,
            xanchor="left",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.write_image(str(output_path))
    fig.write_html(
        str(output_path.with_suffix(".html")), include_plotlyjs="cdn", full_html=False
    )
    fig.show()


def _plot_throughput_vs_dims(
    *,
    ndims_to_test: list[int],
    throughputs: dict[str, list[float]],
    kinds: dict[str, str],
    nobs: int,
    output_path: Path,
) -> None:
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Linear", "Cubic"],
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )
    special_x = {
        "Scipy RectBivariateSpline Cubic": [2],
        "Numpy Interp": [1],
    }
    for row, kind in enumerate(["Linear", "Cubic"], start=1):
        series = [
            (name, values)
            for name, values in throughputs.items()
            if kinds.get(name) == kind and values
        ]
        if not series:
            continue
        max_throughput = max(max(values) for _, values in series if values) or 1.0

        # Fill
        baseline_vals = throughputs.get(f"Scipy RegularGridInterpolator {kind}")
        interpn_series = [
            values
            for name, values in throughputs.items()
            if name.startswith("InterpN") and f"Multi{kind.lower()}" in name and values
        ]

        v = max(interpn_series, key=lambda vals: vals[-1])

        baseline_norm = np.array(baseline_vals) / max_throughput
        fill_between(
            fig,
            x=np.array(ndims_to_test),
            upper=np.array(v) / max_throughput,
            lower=baseline_norm,
            row=row,
            col=1,
        )

        # Lines
        for idx, (label, values) in enumerate(series):
            normalized = np.array(values) / max_throughput
            if not len(normalized):
                continue
            x_vals = special_x.get(label, ndims_to_test[: len(normalized)])
            is_baseline = label.lower().startswith("scipy") or label.lower().startswith(
                "numpy"
            )
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=normalized,
                    mode="lines+markers" if is_baseline else "lines",
                    line=dict(color="black", width=2, dash=_normalized_line_style(idx)),
                    marker=dict(symbol="square" if is_baseline else "circle", size=8),
                    opacity=1.0 if is_baseline else 1.0,
                    name=label,
                    showlegend=True,
                    legendgroup=kind,
                    legendgrouptitle_text=kind,
                ),
                row=row,
                col=1,
            )

    fig.update_xaxes(
        # title_text="Number of Dimensions",
        row=1,
        col=1,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="black",
        showgrid=False,
        zeroline=False,
    )
    fig.update_xaxes(
        title_text="Number of Dimensions",
        row=2,
        col=1,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="black",
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        type="log",
        title_text="Normalized Throughput",
        row=1,
        col=1,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="black",
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        type="log",
        row=2,
        col=1,
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        ticks="outside",
        tickcolor="black",
        showgrid=False,
        zeroline=False,
    )
    fig.update_layout(
        title=dict(
            text=f"Interpolation on 4x...x4 N-Dimensional Grid<br>{nobs} Observation Point{'s' if nobs > 1 else ''}",
            y=0.97,
            yanchor="top",
        ),
        height=450,
        margin=dict(t=80, l=60, r=200, b=90),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            x=1.02,
            xanchor="left",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.write_image(str(output_path))
    fig.write_html(
        str(output_path.with_suffix(".html")), include_plotlyjs="cdn", full_html=False
    )
    fig.show()


def _plot_speedup_vs_dims(
    *,
    ndims_to_test: list[int],
    throughputs: dict[str, list[float]],
    kinds: dict[str, str],
    nobs: int,
    output_dir: Path,
) -> None:
    for kind in ["Linear", "Cubic"]:
        fig = make_subplots(
            rows=1,
            cols=1,
            horizontal_spacing=0.05,
        )
        baseline = throughputs[f"Scipy RegularGridInterpolator {kind}"]
        baseline = np.array(baseline)
        for idx, (name, values) in enumerate(throughputs.items()):
            if not name.startswith("InterpN") or kinds.get(name) != kind or not values:
                continue
            if "Nearest" in name:
                continue
            if "Rectilinear" in name:
                continue
            vals = np.array(values)
            min_len = min(len(vals), len(baseline))
            if min_len == 0:
                continue
            x_vals = ndims_to_test[:min_len]
            speedup = vals[:min_len] / baseline[:min_len]
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=speedup,
                    mode="lines",
                    line=dict(color="black", width=3),
                    marker=dict(symbol="circle", size=8),
                    name=name,
                    showlegend=False,
                    # legendgroup=kind,
                    # legendgrouptitle_text=kind,
                ),
                row=1,
                col=1,
            )
            ones = np.ones_like(speedup)
            upper = np.maximum(speedup, ones)
            fill_between(
                fig,
                x=np.array(x_vals),
                upper=upper,
                lower=ones,
                row=1,
                col=1,
            )
        fig.add_hline(
            y=1.0,
            line=dict(color="gray", dash="dot"),
            row=1,
            col=1,
        )
        fig.update_xaxes(
            title_text="Number of Dimensions",
            row=1,
            col=1,
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
            ticks="outside",
            tickcolor="black",
            showgrid=False,
            zeroline=False,
        )
        fig.update_yaxes(
            title_text="Speedup vs. Scipy",
            row=1,
            col=1,
            showline=True,
            linecolor="black",
            linewidth=1,
            mirror=True,
            ticks="outside",
            tickcolor="black",
            showgrid=False,
            zeroline=False,
        )
        fig.update_layout(
            title=dict(
                text=f"InterpN Speedup vs. Scipy<br>{kind}, {nobs} Observation Point{'s' if nobs > 1 else ''}",
                y=0.97,
                yanchor="top",
            ),
            height=450,
            margin=dict(t=60, l=60, r=200, b=90),
            # legend=dict(
            #     orientation="v",
            #     yanchor="top",
            #     y=1.0,
            #     x=1.02,
            #     xanchor="left",
            # ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        output_path = output_dir / f"speedup_vs_dims_{nobs}_obs_{kind.lower()}.svg"
        fig.write_image(str(output_path))
        fig.write_html(
            str(output_path.with_suffix(".html")),
            include_plotlyjs="cdn",
            full_html=False,
        )
        fig.show()


def bench_4_dims_1_obs():
    nbench = 30  # Bench iterations
    preallocate = False  # Whether to preallocate output array for InterpN
    ndims = 4  # Number of grid dimensions
    ngrid = 20  # Size of grid on each dimension
    nobs = 1  # Number of observation points
    m = max(int(float(nobs) ** (1.0 / ndims) + 2), 2)

    grids = [np.linspace(-1.0, 1.0, ngrid) for _ in range(ndims)]
    xgrid = np.meshgrid(*grids, indexing="ij")
    zgrid = np.random.uniform(-1.0, 1.0, xgrid[0].size)

    dims = [x.size for x in grids]
    starts = np.array([x[0] for x in grids])
    steps = np.array([x[1] - x[0] for x in grids])

    # Baseline interpolating on the same domain,
    # keeping the points entirely inside the domain to give a clear
    # cut between interpolation and extrapolation
    obsgrid = np.meshgrid(
        *[np.linspace(-0.99, 0.99, m) for _ in range(ndims)], indexing="ij"
    )
    obsgrid = [x.flatten()[0:nobs] for x in obsgrid]  # Trim to the exact right number

    # Initialize all interpolator methods
    # Scipy RegularGridInterpolator is actually a more general rectilinear method
    rectilinear_sp = RegularGridInterpolator(
        grids, zgrid.reshape(xgrid[0].shape), bounds_error=None
    )
    cubic_rectilinear_sp = RegularGridInterpolator(
        grids, zgrid.reshape(xgrid[0].shape), bounds_error=None, method="cubic"
    )
    rectilinear_interpn = MultilinearRectilinear.new(grids, zgrid)
    regular_interpn = MultilinearRegular.new(dims, starts, steps, zgrid)
    cubic_regular_interpn = MulticubicRegular.new(
        dims, starts, steps, zgrid, linearize_extrapolation=True
    )
    cubic_rectilinear_interpn = MulticubicRectilinear.new(
        grids, zgrid, linearize_extrapolation=True
    )
    nearest_regular_interpn = NearestRegular.new(dims, starts, steps, zgrid)
    nearest_rectilinear_interpn = NearestRectilinear.new(grids, zgrid)

    out = None if not preallocate else np.zeros_like(obsgrid[0].flatten())
    interps = {
        "Scipy RegularGridInterpolator Linear": rectilinear_sp,
        "Scipy RegularGridInterpolator Cubic": cubic_rectilinear_sp,
        "InterpN MultilinearRegular": lambda p: regular_interpn.eval(p, out),
        "InterpN MultilinearRectilinear": lambda p: rectilinear_interpn.eval(p, out),
        "InterpN MulticubicRegular": lambda p: cubic_regular_interpn.eval(p, out),
        "InterpN MulticubicRectilinear": lambda p: cubic_rectilinear_interpn.eval(
            p, out
        ),
        "InterpN NearestRegular": lambda p: nearest_regular_interpn.eval(p, out),
        "InterpN NearestRectilinear": lambda p: nearest_rectilinear_interpn.eval(
            p, out
        ),
        "numpy interp": lambda p: np.interp(p[0], grids[0], zgrid),  # 1D only
    }

    # Interpolation in sequential order
    points_interpn = [x.flatten() for x in obsgrid]
    points_sp = np.array(points_interpn).T
    points = {
        "Scipy RegularGridInterpolator Linear": points_sp,
        "Scipy RegularGridInterpolator Cubic": points_sp,
        "InterpN MultilinearRegular": points_interpn,
        "InterpN MultilinearRectilinear": points_interpn,
        "InterpN MulticubicRegular": points_interpn,
        "InterpN MulticubicRectilinear": points_interpn,
        "InterpN NearestRegular": points_interpn,
        "InterpN NearestRectilinear": points_interpn,
        "numpy interp": points_interpn,
    }

    print("\nInterpolation in sequential order")
    for name, func in interps.items():
        if RUN_INTERPN_ONLY and not name.startswith("InterpN "):
            continue
        if name == "numpy interp" and ndims > 1:
            continue
        p = points[name]
        timeit(lambda: func(p), number=nbench)  # warmup
        t = timeit(lambda: func(p), number=nbench) / nbench
        throughput = nobs / t
        print(f"\n---- {ndims} Dims")
        print(f"Method: {name}")
        print(f"Time {t:.2e} s")
        print(f"Throughput {throughput:.2e} #/s")

    # Interpolation in random order
    points_interpn1 = [np.random.permutation(x.flatten()) for x in obsgrid]
    points_sp1 = np.array(points_interpn1).T
    points1 = {
        "Scipy RegularGridInterpolator Linear": points_sp1,
        "Scipy RegularGridInterpolator Cubic": points_sp1,
        "InterpN MultilinearRegular": points_interpn1,
        "InterpN MultilinearRectilinear": points_interpn1,
        "InterpN MulticubicRegular": points_interpn1,
        "InterpN MulticubicRectilinear": points_interpn1,
        "InterpN NearestRegular": points_interpn1,
        "InterpN NearestRectilinear": points_interpn1,
        "numpy interp": points_interpn1,
    }

    print("\nInterpolation in random order")
    for name, func in interps.items():
        if RUN_INTERPN_ONLY and not name.startswith("InterpN "):
            continue
        if name == "numpy interp" and ndims > 1:
            continue
        p = points1[name]
        timeit(lambda: func(p), number=nbench)  # warmup
        t = timeit(lambda: func(p), number=nbench) / nbench
        throughput = nobs / t
        print(f"\n---- {ndims} Dims")
        print(f"Method: {name}")
        print(f"Time {t:.2e} s")
        print(f"Throughput {throughput:.2e} #/s")

    # Extrapolation in corner region in random order
    points_interpn2 = [np.random.permutation(x.flatten()) + 3.0 for x in obsgrid]
    points_sp2 = np.array(points_interpn2).T
    points2 = {
        "Scipy RegularGridInterpolator Linear": points_sp2,
        "Scipy RegularGridInterpolator Cubic": points_sp2,
        "InterpN MultilinearRegular": points_interpn2,
        "InterpN MultilinearRectilinear": points_interpn2,
        "InterpN MulticubicRegular": points_interpn2,
        "InterpN MulticubicRectilinear": points_interpn2,
        "InterpN NearestRegular": points_interpn2,
        "InterpN NearestRectilinear": points_interpn2,
        "numpy interp": points_interpn2,
    }

    print("\nExtrapolation to corner region in random order")
    for name, func in interps.items():
        if RUN_INTERPN_ONLY and not name.startswith("InterpN "):
            continue
        if name == "numpy interp" and ndims > 1:
            continue
        p = points2[name]
        timeit(lambda: func(p), number=nbench)  # warmup
        t = timeit(lambda: func(p), number=nbench) / nbench
        throughput = nobs / t
        print(f"\n---- {ndims} Dims")
        print(f"Method: {name}")
        print(f"Time {t:.2e} s")
        print(f"Throughput {throughput:.2e} #/s")

    # Extrapolation in side region in random order
    points_interpn3 = [
        np.random.permutation(x.flatten()) + (3.0 if i == 0 else 0.0)
        for i, x in enumerate(obsgrid)
    ]
    points_sp3 = np.array(points_interpn).T
    points3 = {
        "Scipy RegularGridInterpolator Linear": points_sp3,
        "Scipy RegularGridInterpolator Cubic": points_sp3,
        "InterpN MultilinearRegular": points_interpn3,
        "InterpN MultilinearRectilinear": points_interpn3,
        "InterpN MulticubicRegular": points_interpn3,
        "InterpN MulticubicRectilinear": points_interpn3,
        "InterpN NearestRegular": points_interpn3,
        "InterpN NearestRectilinear": points_interpn3,
        "numpy interp": points_interpn3,
    }

    print("\nExtrapolation to side region in random order")
    for name, func in interps.items():
        if RUN_INTERPN_ONLY and not name.startswith("InterpN "):
            continue
        if name == "numpy interp" and ndims > 1:
            continue
        p = points3[name]
        t = timeit(lambda: func(p), number=nbench) / nbench
        throughput = nobs / t
        print(f"\n---- {ndims} Dims")
        print(f"Method: {name}")
        print(f"Time {t:.2e} s")
        print(f"Throughput {throughput:.2e} #/s")


def bench_3_dims_n_obs_unordered():
    for preallocate in [False, True]:
        ndims = 3  # Number of grid dimensions
        ngrid = 20  # Size of grid on each dimension

        grids = [np.linspace(-1.0, 1.0, ngrid) for _ in range(ndims)]
        xgrid = np.meshgrid(*grids, indexing="ij")
        zgrid = np.random.uniform(-1.0, 1.0, xgrid[0].size)

        dims = [x.size for x in grids]
        starts = np.array([x[0] for x in grids])
        steps = np.array([x[1] - x[0] for x in grids])

        # Initialize all interpolator methods
        # Scipy RegularGridInterpolator is actually a more general rectilinear method
        rectilinear_sp = RegularGridInterpolator(
            grids, zgrid.reshape(xgrid[0].shape), bounds_error=None
        )
        cubic_rectilinear_sp = RegularGridInterpolator(
            grids, zgrid.reshape(xgrid[0].shape), bounds_error=None, method="cubic"
        )
        rectilinear_interpn = MultilinearRectilinear.new(grids, zgrid)
        regular_interpn = MultilinearRegular.new(dims, starts, steps, zgrid)
        cubic_regular_interpn = MulticubicRegular.new(
            dims, starts, steps, zgrid, linearize_extrapolation=True
        )
        cubic_rectilinear_interpn = MulticubicRectilinear.new(
            grids, zgrid, linearize_extrapolation=True
        )
        nearest_regular_interpn = NearestRegular.new(dims, starts, steps, zgrid)
        nearest_rectilinear_interpn = NearestRectilinear.new(grids, zgrid)

        throughputs = {
            "Scipy RegularGridInterpolator Linear": [],
            "Scipy RegularGridInterpolator Cubic": [],
            "InterpN MultilinearRegular": [],
            "InterpN MultilinearRectilinear": [],
            "InterpN MulticubicRegular": [],
            "InterpN MulticubicRectilinear": [],
            "InterpN NearestRegular": [],
            "InterpN NearestRectilinear": [],
        }
        # ns = np.logspace(0, 5, 10, base=10)
        # ns = [int(x) for x in ns]
        # ns = sorted(list(set(ns)))
        ns = [1, 10, 50, 100, 500, 1000, 10000]
        # ns = [1, 10, 100, 1000, 10000, 50000, 100000]
        print("\nThroughput plotting")
        print(ns)
        for nobs in ns:
            print(nobs)
            m = max(int(float(nobs) ** (1.0 / ndims) + 2), 2)

            # Baseline interpolating on the same domain,
            # keeping the points entirely inside the domain to give a clear
            # cut between interpolation and extrapolation
            obsgrid = np.meshgrid(
                *[np.linspace(-0.99, 0.99, m) for _ in range(ndims)], indexing="ij"
            )
            obsgrid = [
                x.flatten()[0:nobs] for x in obsgrid
            ]  # Trim to the exact right number

            # Preallocate output for potential perf advantage
            # Allocate at eval for 1:1 comparison with Scipy
            out = None if not preallocate else np.zeros_like(obsgrid[0].flatten())
            interps = {
                "Scipy RegularGridInterpolator Linear": rectilinear_sp,
                "Scipy RegularGridInterpolator Cubic": cubic_rectilinear_sp,
                "InterpN MultilinearRegular": lambda p: regular_interpn.eval(p, out),
                "InterpN MultilinearRectilinear": lambda p: rectilinear_interpn.eval(
                    p, out
                ),
                "InterpN MulticubicRegular": lambda p: cubic_regular_interpn.eval(
                    p, out
                ),
                "InterpN MulticubicRectilinear": lambda p: cubic_rectilinear_interpn.eval(
                    p, out
                ),
                "InterpN NearestRegular": lambda p: nearest_regular_interpn.eval(
                    p, out
                ),
                "InterpN NearestRectilinear": lambda p: nearest_rectilinear_interpn.eval(
                    p, out
                ),
            }

            # Interpolation in random order
            points_interpn = [np.random.permutation(x.flatten()) for x in obsgrid]
            points_sp = np.array(points_interpn).T
            points = {
                "Scipy RegularGridInterpolator Linear": points_sp,
                "Scipy RegularGridInterpolator Cubic": points_sp,
                "InterpN MultilinearRegular": points_interpn,
                "InterpN MultilinearRectilinear": points_interpn,
                "InterpN MulticubicRegular": points_interpn,
                "InterpN MulticubicRectilinear": points_interpn,
                "InterpN NearestRegular": points_interpn,
                "InterpN NearestRectilinear": points_interpn,
            }

            for name, func in interps.items():
                if RUN_INTERPN_ONLY and not name.startswith("InterpN "):
                    continue
                if "cubic" in name.lower() and nobs > 10000:
                    continue
                p = points[name]
                avg_time = average_call_time(func, p)
                throughput = nobs / avg_time
                throughputs[name].append(throughput)

        kinds = {
            "Scipy RegularGridInterpolator Linear": "Linear",
            "Scipy RegularGridInterpolator Cubic": "Cubic",
            "InterpN MultilinearRegular": "Linear",
            "InterpN MultilinearRectilinear": "Linear",
            "InterpN MulticubicRegular": "Cubic",
            "InterpN MulticubicRectilinear": "Cubic",
            "InterpN NearestRegular": "Linear",
            "InterpN NearestRectilinear": "Linear",
        }

        with_alloc_string = "_prealloc" if preallocate else ""
        suffix = (
            "With Preallocated Output" if preallocate else "Without Preallocated Output"
        )
        figure_title = f"Interpolation on 20x20x20 Grid<br>{suffix}"
        output_path = (
            Path(__file__).parent
            / f"../docs/3d_throughput_vs_nobs{with_alloc_string}.svg"
        )
        _plot_normalized_vs_nobs(
            ns=ns,
            throughputs=throughputs,
            kinds=kinds,
            title=figure_title,
            output_path=output_path,
        )


def bench_4_dims_n_obs_unordered():
    for preallocate in [False, True]:
        ndims = 4  # Number of grid dimensions
        ngrid = 20  # Size of grid on each dimension

        grids = [np.linspace(-1.0, 1.0, ngrid) for _ in range(ndims)]
        xgrid = np.meshgrid(*grids, indexing="ij")
        zgrid = np.random.uniform(-1.0, 1.0, xgrid[0].size)

        dims = [x.size for x in grids]
        starts = np.array([x[0] for x in grids])
        steps = np.array([x[1] - x[0] for x in grids])

        # Initialize all interpolator methods
        # Scipy RegularGridInterpolator is actually a more general rectilinear method
        rectilinear_sp = RegularGridInterpolator(
            grids, zgrid.reshape(xgrid[0].shape), bounds_error=None
        )
        cubic_rectilinear_sp = RegularGridInterpolator(
            grids, zgrid.reshape(xgrid[0].shape), bounds_error=None, method="cubic"
        )
        rectilinear_interpn = MultilinearRectilinear.new(grids, zgrid)
        regular_interpn = MultilinearRegular.new(dims, starts, steps, zgrid)
        cubic_regular_interpn = MulticubicRegular.new(
            dims, starts, steps, zgrid, linearize_extrapolation=True
        )
        cubic_rectilinear_interpn = MulticubicRectilinear.new(
            grids, zgrid, linearize_extrapolation=True
        )
        nearest_regular_interpn = NearestRegular.new(dims, starts, steps, zgrid)
        nearest_rectilinear_interpn = NearestRectilinear.new(grids, zgrid)

        throughputs = {
            "Scipy RegularGridInterpolator Linear": [],
            "Scipy RegularGridInterpolator Cubic": [],
            "InterpN MultilinearRegular": [],
            "InterpN MultilinearRectilinear": [],
            "InterpN MulticubicRegular": [],
            "InterpN MulticubicRectilinear": [],
            "InterpN NearestRegular": [],
            "InterpN NearestRectilinear": [],
        }
        # ns = np.logspace(0, 4, 40, base=10)
        # ns = [int(x) for x in ns]
        ns = [1, 10, 50, 100, 500, 1000, 10000]
        print("\nThroughput plotting")
        print(ns)
        for nobs in ns:
            print(nobs)
            m = max(int(float(nobs) ** (1.0 / ndims) + 2), 2)

            # Baseline interpolating on the same domain,
            # keeping the points entirely inside the domain to give a clear
            # cut between interpolation and extrapolation
            obsgrid = np.meshgrid(
                *[np.linspace(-0.99, 0.99, m) for _ in range(ndims)], indexing="ij"
            )
            obsgrid = [
                x.flatten()[0:nobs] for x in obsgrid
            ]  # Trim to the exact right number

            # Preallocate output for potential perf advantage
            # Allocate at eval for 1:1 comparison with Scipy
            out = None if not preallocate else np.zeros_like(obsgrid[0].flatten())
            interps = {
                "Scipy RegularGridInterpolator Linear": rectilinear_sp,
                "Scipy RegularGridInterpolator Cubic": cubic_rectilinear_sp,
                "InterpN MultilinearRegular": lambda p: regular_interpn.eval(p, out),
                "InterpN MultilinearRectilinear": lambda p: rectilinear_interpn.eval(
                    p, out
                ),
                "InterpN MulticubicRegular": lambda p: cubic_regular_interpn.eval(
                    p, out
                ),
                "InterpN MulticubicRectilinear": lambda p: cubic_rectilinear_interpn.eval(
                    p, out
                ),
                "InterpN NearestRegular": lambda p: nearest_regular_interpn.eval(
                    p, out
                ),
                "InterpN NearestRectilinear": lambda p: nearest_rectilinear_interpn.eval(
                    p, out
                ),
            }

            # Interpolation in random order
            points_interpn = [np.random.permutation(x.flatten()) for x in obsgrid]
            points_sp = np.array(points_interpn).T
            points = {
                "Scipy RegularGridInterpolator Linear": points_sp,
                "Scipy RegularGridInterpolator Cubic": points_sp,
                "InterpN MultilinearRegular": points_interpn,
                "InterpN MultilinearRectilinear": points_interpn,
                "InterpN MulticubicRegular": points_interpn,
                "InterpN MulticubicRectilinear": points_interpn,
                "InterpN NearestRegular": points_interpn,
                "InterpN NearestRectilinear": points_interpn,
            }

            for name, func in interps.items():
                if RUN_INTERPN_ONLY and not name.startswith("InterpN "):
                    continue
                p = points[name]
                avg_time = average_call_time(func, p)
                throughput = nobs / avg_time
                throughputs[name].append(throughput)

        kinds = {
            "Scipy RegularGridInterpolator Linear": "Linear",
            "Scipy RegularGridInterpolator Cubic": "Cubic",
            "InterpN MultilinearRegular": "Linear",
            "InterpN MultilinearRectilinear": "Linear",
            "InterpN MulticubicRegular": "Cubic",
            "InterpN MulticubicRectilinear": "Cubic",
            "InterpN NearestRegular": "Linear",
            "InterpN NearestRectilinear": "Linear",
        }

        with_alloc_string = "_prealloc" if preallocate else ""
        suffix = (
            "With Preallocated Output" if preallocate else "Without Preallocated Output"
        )
        figure_title = f"Interpolation on 20x...x20 4D Grid<br>{suffix}"
        output_path = (
            Path(__file__).parent
            / f"../docs/4d_throughput_vs_nobs{with_alloc_string}.svg"
        )
        _plot_normalized_vs_nobs(
            ns=ns,
            throughputs=throughputs,
            kinds=kinds,
            title=figure_title,
            output_path=output_path,
        )


def bench_throughput_vs_dims():
    for nobs, nbench in [(1, 10000), (1000, 100)]:
        throughputs = {
            "Scipy RegularGridInterpolator Linear": [],
            "Scipy RegularGridInterpolator Cubic": [],
            "InterpN MultilinearRegular": [],
            "InterpN MultilinearRectilinear": [],
            "InterpN MulticubicRegular": [],
            "InterpN MulticubicRectilinear": [],
            "InterpN NearestRegular": [],
            "InterpN NearestRectilinear": [],
            "Scipy RectBivariateSpline Cubic": [],  # Move to end to order plots
            "Numpy Interp": [],
        }
        ndims_to_test = [x for x in range(1, 7)]
        for ndims in ndims_to_test:
            ngrid = 4  # Size of grid on each dimension

            grids = [np.linspace(-1.0, 1.0, ngrid) for _ in range(ndims)]
            xgrid = np.meshgrid(*grids, indexing="ij")
            zgrid = np.random.uniform(-1.0, 1.0, xgrid[0].size)
            z = zgrid.reshape(xgrid[0].shape)

            dims = [x.size for x in grids]
            starts = np.array([x[0] for x in grids])
            steps = np.array([x[1] - x[0] for x in grids])

            # Initialize all interpolator methods
            # Scipy RegularGridInterpolator is actually a more general rectilinear method
            rectilinear_sp = RegularGridInterpolator(grids, z.copy(), bounds_error=None)
            cubic_rectilinear_sp = RegularGridInterpolator(
                grids, z.copy(), bounds_error=None, method="cubic"
            )
            rectilinear_interpn = MultilinearRectilinear.new(grids, zgrid)
            regular_interpn = MultilinearRegular.new(dims, starts, steps, zgrid)
            cubic_regular_interpn = MulticubicRegular.new(
                dims, starts, steps, zgrid, linearize_extrapolation=True
            )
            cubic_rectilinear_interpn = MulticubicRectilinear.new(
                grids, zgrid, linearize_extrapolation=True
            )
            nearest_regular_interpn = NearestRegular.new(dims, starts, steps, zgrid)
            nearest_rectilinear_interpn = NearestRectilinear.new(grids, zgrid)

            m = max(int(float(nobs) ** (1.0 / ndims) + 2), 2)

            # Baseline interpolating on the same domain,
            # keeping the points entirely inside the domain to give a clear
            # cut between interpolation and extrapolation
            obsgrid = np.meshgrid(
                *[np.linspace(-0.99, 0.99, m) for _ in range(ndims)], indexing="ij"
            )
            obsgrid = [
                x.flatten()[0:nobs] for x in obsgrid
            ]  # Trim to the exact right number

            # Preallocate output for fair comparison to numpy interp and scipy RectBivariateSpline,
            # which have specialized scalar variants that trigger when evaluating for a single point.
            out = np.zeros((nobs,))
            interps = {
                "Scipy RegularGridInterpolator Linear": rectilinear_sp,
                "Scipy RegularGridInterpolator Cubic": cubic_rectilinear_sp,
                "InterpN MultilinearRegular": lambda p,
                interp=regular_interpn: interp.eval(p, out),
                "InterpN MultilinearRectilinear": (
                    lambda p, interp=rectilinear_interpn: interp.eval(p, out)
                ),
                "InterpN MulticubicRegular": lambda p,
                interp=cubic_regular_interpn: interp.eval(p, out),
                "InterpN MulticubicRectilinear": (
                    lambda p, interp=cubic_rectilinear_interpn: interp.eval(p, out)
                ),
                "InterpN NearestRegular": lambda p,
                interp=nearest_regular_interpn: interp.eval(p, out),
                "InterpN NearestRectilinear": (
                    lambda p, interp=nearest_rectilinear_interpn: interp.eval(p, out)
                ),
            }

            if ndims == 1:
                interps["Numpy Interp"] = lambda p: np.interp(p[0], grids[0], zgrid)

            if ndims == 2:
                cubic_rbs_sp = RectBivariateSpline(
                    grids[0], grids[1], z.copy(), kx=3, ky=3, s=0
                )
                interps["Scipy RectBivariateSpline Cubic"] = lambda p: cubic_rbs_sp(
                    *p, grid=False
                )

            # Interpolation in random order
            points_interpn = [np.random.permutation(x.flatten()) for x in obsgrid]
            points_sp = np.ascontiguousarray(np.array(points_interpn).T)
            points = {
                "Scipy RegularGridInterpolator Linear": points_sp,
                "Scipy RegularGridInterpolator Cubic": points_sp,
                "Scipy RectBivariateSpline Cubic": points_interpn,
                "InterpN MultilinearRegular": points_interpn,
                "InterpN MultilinearRectilinear": points_interpn,
                "InterpN MulticubicRegular": points_interpn,
                "InterpN MulticubicRectilinear": points_interpn,
                "InterpN NearestRegular": points_interpn,
                "InterpN NearestRectilinear": points_interpn,
                "Numpy Interp": points_interpn,
            }

            for name, func in interps.items():
                if RUN_INTERPN_ONLY and not name.startswith("InterpN "):
                    continue
                print(ndims, name)
                p = points[name]
                timeit(
                    lambda: func(p), setup=gc.collect, number=int(nbench / 4)
                )  # warmup
                t = timeit(lambda: func(p), setup=gc.collect, number=nbench) / nbench
                throughput = nobs / t
                throughputs[name].append(throughput)

        kinds = {
            "Scipy RegularGridInterpolator Linear": "Linear",
            "Scipy RegularGridInterpolator Cubic": "Cubic",
            "Scipy RectBivariateSpline Cubic": "Cubic",
            "InterpN MultilinearRegular": "Linear",
            "InterpN MultilinearRectilinear": "Linear",
            "InterpN MulticubicRegular": "Cubic",
            "InterpN MulticubicRectilinear": "Cubic",
            "InterpN NearestRegular": "Linear",
            "InterpN NearestRectilinear": "Linear",
            "Numpy Interp": "Linear",
        }

        output_path = (
            Path(__file__).parent / f"../docs/throughput_vs_dims_{nobs}_obs.svg"
        )
        _plot_throughput_vs_dims(
            ndims_to_test=ndims_to_test,
            throughputs=throughputs,
            kinds=kinds,
            nobs=nobs,
            output_path=output_path,
        )
        _plot_speedup_vs_dims(
            ndims_to_test=ndims_to_test,
            throughputs=throughputs,
            kinds=kinds,
            nobs=nobs,
            output_dir=Path(__file__).parent.parent / "docs",
        )


def main():
    bench_throughput_vs_dims()
    bench_4_dims_1_obs()
    bench_4_dims_n_obs_unordered()
    bench_3_dims_n_obs_unordered()


if __name__ == "__main__":
    main()
