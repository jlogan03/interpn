"""Benchmarks examining memory usage"""
import gc
import time
from pathlib import Path

from memory_profiler import memory_usage

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from interpn import (
    MultilinearRectilinear,
    MultilinearRegular,
    MulticubicRegular,
    MulticubicRectilinear,
)


def bench_eval_mem_vs_dims():
    usages = {
        "Scipy RegularGridInterpolator Linear": [],
        "Scipy RegularGridInterpolator Cubic": [],
        "InterpN MultilinearRegular": [],
        "InterpN MultilinearRectilinear": [],
        "InterpN MulticubicRegular": [],
        "InterpN MulticubicRectilinear": [],
    }
    ndims_to_test = [x for x in range(1, 9)]
    for ndims in ndims_to_test:
        nobs = 10000
        ngrid = 4  # Size of grid on each dimension

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
            "InterpN MulticubicRectilinear": lambda p: cubic_rectilinear_interpn.eval(
                p
            ),
        }

        # Interpolation in random order
        points_interpn = [np.random.permutation(x.flatten()) for x in obsgrid]
        points_sp = np.ascontiguousarray(np.array(points_interpn).T)
        points = {
            "Scipy RegularGridInterpolator Linear": points_sp,
            "Scipy RegularGridInterpolator Cubic": points_sp,
            "InterpN MultilinearRegular": points_interpn,
            "InterpN MultilinearRectilinear": points_interpn,
            "InterpN MulticubicRegular": points_interpn,
            "InterpN MulticubicRectilinear": points_interpn,
        }

        for name, func in interps.items():
            print(ndims, name)
            gc.collect()
            time.sleep(0.1)
            p = points[name]
            mems = memory_usage((func, (p,), {}), interval=1e-9, backend="psutil")
            usages[name].append(max(mems))

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

    _fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    plt.suptitle(
        f"Interpolation on 4x...x4 N-Dimensional Grid\n{nobs} Observation Points"
    )
    for i, kind in enumerate(["Linear", "Cubic"]):
        plt.sca(axes[i])
        usages_this_kind = [(k, v) for k, v in usages.items() if kinds[k] == kind]
        for i, (k, v) in enumerate(usages_this_kind):
            # The memory profiler captures some things not actually involved
            # in the function evaluation, which gives both methods a floor of
            # about 100MB
            plt.semilogy(
                ndims_to_test,
                v,
                color="k",
                linewidth=2,
                linestyle=linestyles[i],
                label=k,
                alpha=alpha[i],
            )
        plt.legend()
        plt.xlabel("Number of Dimensions")
        plt.ylabel("Peak Memory Usage [MB]")
        plt.title(kind)

    plt.tight_layout()
    plt.savefig(Path(__file__).parent / "../docs/ram_vs_dims.svg")
    plt.show(block=False)


if __name__ == "__main__":
    bench_eval_mem_vs_dims()
    plt.show(block=True)
