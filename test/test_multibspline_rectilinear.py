import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

import interpn


@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-12), (np.float32, 1e-5)])
def test_multibspline_rectilinear(dtype, tol):
    x = np.array([-1.0, -0.35, 0.2, 0.95, 1.8, 3.0], dtype=dtype)
    y = np.array([-2.0, -1.25, -0.1, 0.55, 1.7], dtype=dtype)

    xgrid, ygrid = np.meshgrid(x, y, indexing="ij")
    zgrid = (xgrid * xgrid + 0.25 * xgrid * ygrid + ygrid).astype(dtype)
    vals = zgrid.flatten()
    grids = [x, y]
    dims = [x.size, y.size]

    coeffs = np.zeros_like(vals)
    scratch = np.zeros(2 * max(dims), dtype=dtype)

    if dtype == np.float32:
        interpn.raw.coefficients_bspline_rectilinear_f32(grids, vals, coeffs, scratch)
    else:
        interpn.raw.coefficients_bspline_rectilinear_f64(grids, vals, coeffs, scratch)

    obs = [xgrid.flatten().astype(dtype), ygrid.flatten().astype(dtype)]
    out = np.zeros_like(vals)

    if dtype == np.float32:
        interpn.raw.interpn_bspline_rectilinear_f32(
            grids,
            coeffs,
            False,
            obs,
            out,
        )
    else:
        interpn.raw.interpn_bspline_rectilinear_f64(
            grids,
            coeffs,
            False,
            obs,
            out,
        )

    for i in range(out.size):
        assert approx(out[i], vals[i], dtype(tol))

    interpolator = interpn.MultiBsplineRectilinear.new(grids, vals)
    out2 = interpolator.eval(obs)
    for i in range(out2.size):
        assert approx(out2[i], vals[i], dtype(tol))

    definitely_inside = [
        np.array([0.0]).astype(dtype),
        np.array(0.0).astype(dtype),
    ]
    definitely_outside = [
        np.array([-5.0]).astype(dtype),
        np.array(-25.0).astype(dtype),
    ]
    assert not any(interpolator.check_bounds(definitely_inside, dtype(1e-6)))
    assert any(interpolator.check_bounds(definitely_outside, dtype(1e-6)))

    roundtrip_interpolator = interpn.MultiBsplineRectilinear.model_validate_json(
        interpolator.model_dump_json()
    )
    out3 = roundtrip_interpolator.eval(obs)
    for i in range(out3.size):
        assert approx(out3[i], vals[i], dtype(tol))


def test_multibspline_rectilinear_quadratic_extrapolation():
    x = np.array([-1.0, -0.35, 0.2, 0.95, 1.8, 3.0])
    y = np.array([-2.0, -1.25, -0.1, 0.55, 1.7])
    xgrid, ygrid = np.meshgrid(x, y, indexing="ij")
    vals = 0.5 * xgrid * xgrid + 0.25 * ygrid * ygrid + xgrid * ygrid - 0.3
    interp = interpn.MultiBsplineRectilinear.new(
        [x, y], vals.ravel(), linearize_extrapolation=False
    )

    obs = [
        np.array([-1.5, -0.7, 0.0, 1.2, 3.4]),
        np.array([-2.4, -0.9, 0.2, 1.2, 2.2]),
    ]
    out = interp.eval(obs)
    expected = 0.5 * obs[0] * obs[0] + 0.25 * obs[1] * obs[1] + obs[0] * obs[1] - 0.3

    assert np.allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_multibspline_rectilinear_cubic_internal_points_vs_scipy():
    x = np.array([-1, -0.8, -0.55, -0.3, 0.05, 0.4, 0.9, 1.4, 2.0, 2.7, 3.5, 4.4])
    y = np.array([-2, -1.5, -0.9, -0.2, 0.4, 1.0, 1.8, 2.7, 3.7])
    xgrid, ygrid = np.meshgrid(x, y, indexing="ij")
    vals = 0.2 * xgrid**3 - 0.1 * ygrid**3 + 0.5 * xgrid**2 + xgrid * ygrid - 0.2

    interp = interpn.MultiBsplineRectilinear.new([x, y], vals.ravel())
    scipy_interp = RegularGridInterpolator(
        [x, y], vals, method="cubic", bounds_error=False, fill_value=None
    )

    obs = [
        np.array([0.15, 0.65, 1.1, 1.75, 2.3]),
        np.array([-0.5, 0.15, 0.65, 1.2, 2.1]),
    ]
    out = interp.eval(obs)
    scipy_out = scipy_interp(np.array(obs).T)

    assert np.allclose(out, scipy_out, rtol=0.0, atol=1e-2)


def approx(value_is, value_should_be, tol) -> bool:
    delta = abs(value_is - value_should_be)
    norm = max(abs(value_should_be), 1.0)
    return delta / norm < tol
