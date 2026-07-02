import numpy as np
import pytest
import interpn


def linear_field(x, y):
    return 1.25 + 2.0 * x - 3.0 * y


def assert_linear_grad(grad, dtype, tol):
    expected = np.array([[2.0], [-3.0]], dtype=dtype)
    np.testing.assert_allclose(grad, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-12), (np.float32, 1e-5)])
@pytest.mark.parametrize(
    "cls,grid_kind",
    [
        (interpn.MultilinearRegular, "regular"),
        (interpn.MultilinearRectilinear, "rectilinear"),
        (interpn.MulticubicRegular, "regular"),
        (interpn.MulticubicRectilinear, "rectilinear"),
        (interpn.MultiBsplineRegular, "regular"),
        (interpn.MultiBsplineRectilinear, "rectilinear"),
    ],
)
def test_eval_grad_linear_field(cls, grid_kind, dtype, tol):
    if grid_kind == "regular":
        x = np.linspace(-1.0, 2.0, 6).astype(dtype)
        y = np.linspace(0.5, 3.0, 7).astype(dtype)
    else:
        x = np.array([-1.0, -0.2, 0.4, 1.1, 2.0, 3.1], dtype=dtype)
        y = np.array([0.5, 0.7, 1.4, 2.2, 3.0, 4.0, 5.2], dtype=dtype)

    xgrid, ygrid = np.meshgrid(x, y, indexing="ij")
    vals = linear_field(xgrid, ygrid).astype(dtype)
    obs = [np.array([0.23], dtype=dtype), np.array([1.07], dtype=dtype)]

    if grid_kind == "regular":
        dims = [x.size, y.size]
        starts = np.array([x[0], y[0]], dtype=dtype)
        steps = np.array([x[1] - x[0], y[1] - y[0]], dtype=dtype)
        interpolator = cls.new(dims, starts, steps, vals)
    else:
        interpolator = cls.new([x, y], vals)

    grad = interpolator.eval_grad(obs)
    assert grad.shape == (2, 1)
    assert grad.dtype == dtype
    assert_linear_grad(grad, dtype, tol)

    out = np.zeros((2, obs[0].size), dtype=dtype)
    returned = interpolator.eval_grad_unchecked(obs, out)
    assert returned is out
    assert_linear_grad(out, dtype, tol)
