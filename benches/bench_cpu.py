import gc
import os
from pathlib import Path

from timeit import Timer, timeit
import numpy as np
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
import matplotlib.pyplot as plt

from interpn import (
    MultilinearRectilinear,
    MultilinearRegular,
    MulticubicRegular,
    MulticubicRectilinear,
)

# Toggle SciPy/NumPy baselines via environment for PGO workloads.
RUN_INTERPN_ONLY = os.environ.get("INTERPNPY_INTERPN_ONLY", "").lower() in {
    "1",
    "true",
    "yes",
}

TARGET_SAMPLE_SECONDS = 1.0
MAX_TIMER_LOOPS = 1_000_000_000


def average_call_time(func, points, target_seconds: float = TARGET_SAMPLE_SECONDS) -> float:
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


def bench_4_dims_1_obs():
    nbench = 30  # Bench iterations
    preallocate = False  # Whether to preallocate output array for InterpN
    ndims = 4  # Number of grid dimensions
    ngrid = 20  # Size of grid on each dimension
    nobs = int(1)  # Number of observation points
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
    cubic_regular_interpn = MulticubicRegular.new(dims, starts, steps, zgrid)
    cubic_rectilinear_interpn = MulticubicRectilinear.new(grids, zgrid)

    # Preallocate output for potential perf advantage
    # Allocate at eval for 1:1 comparison with Scipy
    out = None if not preallocate else np.zeros_like(obsgrid[0].flatten())
    interps = {
        "Scipy RegularGridInterpolator Linear": rectilinear_sp,
        "Scipy RegularGridInterpolator Cubic": cubic_rectilinear_sp,
        "InterpN MultilinearRegular": lambda p: regular_interpn.eval(p, out),
        "InterpN MultilinearRectilinear": lambda p: rectilinear_interpn.eval(p, out),
        "InterpN MulticubicRegular": lambda p: cubic_regular_interpn.eval(p, out),
        "InterpN MulticubicRectilinear": lambda p: cubic_rectilinear_interpn.eval(p, out),
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
        cubic_regular_interpn = MulticubicRegular.new(dims, starts, steps, zgrid)
        cubic_rectilinear_interpn = MulticubicRectilinear.new(grids, zgrid)

        throughputs = {
            "Scipy RegularGridInterpolator Linear": [],
            "Scipy RegularGridInterpolator Cubic": [],
            "InterpN MultilinearRegular": [],
            "InterpN MultilinearRectilinear": [],
            "InterpN MulticubicRegular": [],
            "InterpN MulticubicRectilinear": [],
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
        }

        linestyles = ["dotted", "-", "--", "-.", (0, (3, 1, 1, 1, 1, 1))]
        alpha = [0.5, 1.0, 1.0, 1.0, 1.0]

        _fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        plt.suptitle("Interpolation on 20x20x20 Grid")
        for i, kind in enumerate(["Linear", "Cubic"]):
            # plt.figure()
            plt.sca(axes[i])
            throughputs_this_kind = [
                (k, v) for k, v in throughputs.items() if kinds[k] == kind
            ]
            all_throughputs_this_kind = sum([v for _, v in throughputs_this_kind], [])
            max_throughput = max(all_throughputs_this_kind)
            for i, (k, v) in enumerate(throughputs_this_kind):
                if RUN_INTERPN_ONLY and not k.startswith("InterpN "):
                    continue
                normalized_throughput = np.array(v) / max_throughput
                plt.loglog(
                    ns[: normalized_throughput.size],
                    normalized_throughput,
                    color="k",
                    linewidth=2,
                    linestyle=linestyles[i],
                    label=k,
                    alpha=alpha[i],
                )
            plt.legend()
            plt.xlabel("Number of Observation Points")
            plt.ylabel("Normalized Throughput")
            with_alloc_string = "\nWith Preallocated Output" if preallocate else ""
            plt.title(f"{kind}" + with_alloc_string)

        plt.tight_layout()
        with_alloc_string = "_prealloc" if preallocate else ""
        plt.savefig(Path(__file__).parent / f"../docs/3d_throughput_vs_nobs{with_alloc_string}.svg")
        plt.show(block=False)


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
        cubic_regular_interpn = MulticubicRegular.new(dims, starts, steps, zgrid)
        cubic_rectilinear_interpn = MulticubicRectilinear.new(grids, zgrid)

        throughputs = {
            "Scipy RegularGridInterpolator Linear": [],
            "Scipy RegularGridInterpolator Cubic": [],
            "InterpN MultilinearRegular": [],
            "InterpN MultilinearRectilinear": [],
            "InterpN MulticubicRegular": [],
            "InterpN MulticubicRectilinear": [],
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
        }

        linestyles = ["dotted", "-", "--", "-.", (0, (3, 1, 1, 1, 1, 1))]
        alpha = [0.5, 1.0, 1.0, 1.0, 1.0]
        _fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        plt.suptitle("Interpolation on 20x...x20 4D Grid")
        for i, kind in enumerate(["Linear", "Cubic"]):
            # plt.figure()
            plt.sca(axes[i])
            throughputs_this_kind = [
                (k, v) for k, v in throughputs.items() if kinds[k] == kind
            ]
            all_throughputs_this_kind = sum([v for _, v in throughputs_this_kind], [])
            max_throughput = max(all_throughputs_this_kind)
            for i, (k, v) in enumerate(throughputs_this_kind):
                if RUN_INTERPN_ONLY and not k.startswith("InterpN "):
                    continue
                normalized_throughput = np.array(v) / max_throughput
                plt.loglog(
                    ns[: normalized_throughput.size],
                    normalized_throughput,
                    color="k",
                    linewidth=2,
                    linestyle=linestyles[i],
                    label=k,
                    alpha=alpha[i],
                )
            plt.legend()
            plt.xlabel("Number of Observation Points")
            plt.ylabel("Normalized Throughput")
            with_alloc_string = "\nWith Preallocated Output" if preallocate else ""
            plt.title(f"{kind}" + with_alloc_string)

        plt.tight_layout()
        with_alloc_string = "_prealloc" if preallocate else ""
        plt.savefig(Path(__file__).parent / f"../docs/4d_throughput_vs_nobs{with_alloc_string}.svg")
        plt.show(block=False)


def bench_throughput_vs_dims():
    for nobs, nbench in [(1, 1000), (1000, 10)]:
        throughputs = {
            "Scipy RegularGridInterpolator Linear": [],
            "Scipy RegularGridInterpolator Cubic": [],
            "InterpN MultilinearRegular": [],
            "InterpN MultilinearRectilinear": [],
            "InterpN MulticubicRegular": [],
            "InterpN MulticubicRectilinear": [],
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
            cubic_regular_interpn = MulticubicRegular.new(dims, starts, steps, zgrid)
            cubic_rectilinear_interpn = MulticubicRectilinear.new(grids, zgrid)

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
            interps = {
                "Scipy RegularGridInterpolator Linear": rectilinear_sp,
                "Scipy RegularGridInterpolator Cubic": cubic_rectilinear_sp,
                "InterpN MultilinearRegular": lambda p: regular_interpn.eval(p),
                "InterpN MultilinearRectilinear": lambda p: rectilinear_interpn.eval(p),
                "InterpN MulticubicRegular": lambda p: cubic_regular_interpn.eval(p),
                "InterpN MulticubicRectilinear": lambda p: cubic_rectilinear_interpn.eval(p),
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
            "Numpy Interp": "Linear",
        }

        linestyles = ["dotted", "-", "--", "-.", (0, (3, 1, 1, 1, 1, 1))]
        alpha = [0.5, 1.0, 1.0, 1.0, 1.0]

        _fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        plt.suptitle(
            f"Interpolation on 4x...x4 N-Dimensional Grid\n{nobs} Observation Point(s)"
        )
        for i, kind in enumerate(["Linear", "Cubic"]):
            plt.sca(axes[i])
            throughputs_this_kind = [
                (k, v) for k, v in throughputs.items() if kinds[k] == kind
            ]
            all_throughputs_this_kind = sum([v for _, v in throughputs_this_kind], [])
            max_throughput = max(all_throughputs_this_kind)
            for i, (k, v) in enumerate(throughputs_this_kind):
                if RUN_INTERPN_ONLY and not k.startswith("InterpN "):
                    continue
                normalized_throughput = np.array(v) / max_throughput
                if k == "Scipy RectBivariateSpline Cubic":
                    plt.semilogy(
                        [2],
                        normalized_throughput,
                        marker="o",
                        markersize=5,
                        color="k",
                        linewidth=2,
                        linestyle=None,
                        label=k,
                        alpha=alpha[i],
                    )
                elif k == "Numpy Interp":
                    plt.semilogy(
                        [1],
                        normalized_throughput,
                        marker="s",
                        markersize=5,
                        color="k",
                        linewidth=2,
                        linestyle=None,
                        label=k,
                        alpha=alpha[i],
                    )
                else:
                    plt.semilogy(
                        ndims_to_test,
                        normalized_throughput,
                        color="k",
                        linewidth=2,
                        linestyle=linestyles[i],
                        label=k,
                        alpha=alpha[i],
                    )
            plt.legend()
            plt.xlabel("Number of Dimensions")
            plt.ylabel("Normalized Throughput")
            plt.title(kind)

        plt.tight_layout()
        plt.savefig(Path(__file__).parent / f"../docs/throughput_vs_dims_{nobs}_obs.svg")
        plt.show(block=False)

def main():
    bench_throughput_vs_dims()
    bench_4_dims_1_obs()
    bench_4_dims_n_obs_unordered()
    bench_3_dims_n_obs_unordered()
    plt.show(block=True)


if __name__ == "__main__":
    main()
