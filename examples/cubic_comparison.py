import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

from interpn import MulticubicRegular, MulticubicRectilinear

def _step(x):
    y = np.ones_like(x)
    y[np.where(x < 0.0)] = 0.0
    y[np.where(x >= 0.0)] = 1.0
    return y

if __name__ == "__main__":
    rng = np.random.RandomState(42)  # Random number generator with fixed seed

    for kind in ["Regular", "Rectilinear"]:
        # 1D comparison
        _fig, axes = plt.subplots(2, 3, sharex=True, figsize=(14, 6))
        axes = axes.flatten()
        plt.suptitle(f"Comparison\nInterpN Multicubic{kind} vs. Scipy RegularGridInterpolator Cubic")
        for i, (fnname, fn, data_res) in enumerate([("Quadratic", lambda x: x**2, 0.5), ("Sine", lambda x: np.sin(x), 0.5), ("Step", lambda x: _step(x), 0.5)]):
            xdata = np.arange(-2.0, 2.5, data_res)
            if kind == "Rectilinear":
                xdata += rng.uniform(-0.45 * data_res, 0.45 * data_res, xdata.size)
            ydata = fn(xdata)

            xinterp = np.arange(-3.0, 3.05, data_res / 100)

            if kind == "Regular":
                dims = np.asarray([xdata.size])
                starts = np.asarray([-2.0])
                steps = np.asarray([data_res])
                y_interpn = MulticubicRegular.new(dims, starts, steps, ydata).eval([xinterp])
            else:
                y_interpn = MulticubicRectilinear.new([xdata], ydata).eval([xinterp])

            y_sp = RegularGridInterpolator(
                [xdata], ydata, bounds_error=None, fill_value=None, method="cubic"
            )(xinterp)

            plt.sca(axes[i])
            plt.gca().fill_between(xdata, 0, 1,
                    color='green', alpha=0.1, transform=plt.gca().get_xaxis_transform(), label="Interpolating Region")
            plt.scatter(xdata, ydata, marker='o', color='k', s=20, label="Data")
            plt.plot(xinterp, y_interpn, color='k', linewidth=1, linestyle="-", label="InterpN")
            plt.plot(xinterp, y_sp, color='k', linewidth=2, linestyle=(0, (1, 1)), alpha=0.5, label="Scipy")
            plt.title(fnname)
            plt.legend()

            plt.sca(axes[i + 3])
            plt.gca().fill_between(xdata, 0, 1,
                    color='green', alpha=0.1, transform=plt.gca().get_xaxis_transform(), label="Interpolating Region")
            plt.plot(xinterp, (y_interpn - fn(xinterp)), color='k', linewidth=1, linestyle="-", label="InterpN")
            plt.plot(xinterp, (y_sp - fn(xinterp)), color='k', linewidth=2, linestyle=(0, (1, 1)), alpha=0.5, label="Scipy")

            plt.title("Error, " + fnname)
            plt.legend()

        plt.tight_layout()
        plt.savefig(Path(__file__).parent / f"../docs/1d_quality_of_fit_{kind}.svg")

        # 2D comparison
        xdata = np.linspace(-3.0, 3.0, 7, endpoint=True)
        ydata = np.linspace(-3.0, 3.0, 7, endpoint=True)
        if kind == "Rectilinear":
            # Add noise to grid, but leave the outside extent in place
            xdata[1:-1] += rng.uniform(-0.45 * data_res, 0.45 * data_res, xdata.size - 2)
            ydata[1:-1] += rng.uniform(-0.45 * data_res, 0.45 * data_res, ydata.size - 2)
        xmesh, ymesh = np.meshgrid(xdata, ydata, indexing="ij")
        zmesh = xmesh**2 + ymesh**2

        xinterp = np.linspace(-5.0, 5.0, 30, endpoint=True)
        yinterp = np.linspace(-5.0, 5.0, 30, endpoint=True)
        xinterpmesh, yinterpmesh = np.meshgrid(xinterp, yinterp, indexing="ij")
        zinterp = xinterpmesh**2 + yinterpmesh**2

        if kind == "Regular":
            dims = np.asarray([xdata.size, ydata.size])
            starts = np.asarray([-3.0, -3.0])
            steps = np.asarray([xmesh[1, 0] - xmesh[0,0], ymesh[0, 1] - ymesh[0, 0]])
            z_interpn = MulticubicRegular.new(dims, starts, steps, zmesh).eval([xinterpmesh.flatten(), yinterpmesh.flatten()]).reshape(xinterpmesh.shape)
        else:
            z_interpn = MulticubicRectilinear.new([xdata, ydata], zmesh).eval([xinterpmesh.flatten(), yinterpmesh.flatten()]).reshape(xinterpmesh.shape)

        z_sp = RegularGridInterpolator(
            [xdata, ydata], zmesh, bounds_error=None, fill_value=None, method="cubic"
        )((xinterpmesh, yinterpmesh))

        _fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        plt.suptitle("Quadratic Test Function")
        for i, (z, title) in enumerate([(zinterp, "Truth"), (z_interpn, f"InterpN Multicubic{kind}"), (z_sp, "Scipy RegularGridInterp. Cubic")]):
            plt.sca(axes[i])
            plt.imshow(z.T, origin="lower", extent=[-5.0, 5.0, -5.0, 5.0])
            plt.contour(xinterpmesh, yinterpmesh, z.T, colors='k', levels=6)
            plt.gca().add_patch(plt.Rectangle((-3.0, -3.0), 6.0, 6.0, edgecolor='w', fill=False, label="Interpolating Region"))
            plt.scatter(xmesh.flatten(), ymesh.flatten(), marker="o", s=3, color='w', label="Data")
            plt.title(title)
            plt.legend(facecolor='k', labelcolor='w', framealpha=0.5)

            plt.sca(axes[i + 3])
            if title == "Truth":
                plt.axis("off")
                continue
            plt.imshow((z - zinterp).T, origin="lower", extent=[-5.0, 5.0, -5.0, 5.0])
            plt.gca().add_patch(plt.Rectangle((-3.0, -3.0), 6.0, 6.0, edgecolor='w', fill=False, label="Interpolating Region"))
            plt.title("Error, " + title.split()[0])
            plt.axis("off")
            plt.colorbar()
            plt.legend(facecolor='k', labelcolor='w', framealpha=0.5)
    
        plt.tight_layout()
        plt.savefig(Path(__file__).parent / f"../docs/2d_quality_of_fit_{kind}.svg")

    testing = (len(sys.argv) > 1) and (sys.argv[1] == "test")
    if not testing:
        plt.show()
