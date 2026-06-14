import numpy as np
import pytest
import interpn


@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-12), (np.float32, 1e-5)])
def test_multibspline_regular(dtype, tol):
    x = np.linspace(0.0, 10.0, 7).astype(dtype)
    y = np.linspace(20.0, 30.0, 6).astype(dtype)

    xgrid, ygrid = np.meshgrid(x, y, indexing="ij")
    zgrid = (xgrid * xgrid + 0.25 * xgrid * ygrid + ygrid).astype(dtype)

    dims = [x.size, y.size]
    starts = np.array([x[0], y[0]]).astype(dtype)
    steps = np.array([x[1] - x[0], y[1] - y[0]]).astype(dtype)
    vals = zgrid.flatten()

    coeffs = np.zeros_like(vals)
    scratch = np.zeros(2 * max(dims), dtype=dtype)

    if dtype == np.float32:
        interpn.raw.coefficients_bspline_regular_f32(dims, vals, coeffs, scratch)
    else:
        interpn.raw.coefficients_bspline_regular_f64(dims, vals, coeffs, scratch)

    obs = [xgrid.flatten().astype(dtype), ygrid.flatten().astype(dtype)]
    assert coeffs.shape == vals.shape

    interpolator = interpn.MultiBsplineRegular.new(dims, starts, steps, vals)
    out2 = interpolator.eval(obs)
    for i in range(out2.size):
        assert approx(out2[i], vals[i], dtype(tol))

    definitely_inside = [
        np.array([5.0]).astype(dtype),
        np.array(25.0).astype(dtype),
    ]
    definitely_outside = [
        np.array([-5.0]).astype(dtype),
        np.array(-25.0).astype(dtype),
    ]
    assert not any(interpolator.check_bounds(definitely_inside, dtype(1e-6)))
    assert any(interpolator.check_bounds(definitely_outside, dtype(1e-6)))

    roundtrip_interpolator = interpn.MultiBsplineRegular.model_validate_json(
        interpolator.model_dump_json()
    )
    out3 = roundtrip_interpolator.eval(obs)
    for i in range(out3.size):
        assert approx(out3[i], vals[i], dtype(tol))


def approx(value_is, value_should_be, tol) -> bool:
    delta = abs(value_is - value_should_be)
    norm = max(abs(value_should_be), 1.0)
    return delta / norm < tol
