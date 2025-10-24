import numpy as np
import pytest

from interpn import interpn


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_interpn_check_bounds_regular(dtype):
    grid = np.linspace(-1.0, 1.0, 5).astype(dtype)
    vals = np.linspace(0.0, 10.0, grid.size).astype(dtype)

    obs_inside = [np.array([-0.5, 0.5], dtype=dtype)]
    obs_outside = [np.array([-0.5, 1.5], dtype=dtype)]

    inside = interpn(
        obs=obs_inside,
        grids=[grid],
        vals=vals,
        method="linear",
        check_bounds=True,
    )
    assert inside.shape == obs_inside[0].shape

    with pytest.raises(ValueError):
        interpn(
            obs=obs_outside,
            grids=[grid],
            vals=vals,
            method="linear",
            check_bounds=True,
        )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_interpn_check_bounds_rectilinear(dtype):
    grid = np.array([-1.0, -0.25, 0.5, 2.0], dtype=dtype)
    vals = np.linspace(0.0, 10.0, grid.size).astype(dtype)

    obs_inside = [np.array([-0.5, 1.0], dtype=dtype)]
    obs_outside = [np.array([-1.5, 0.25], dtype=dtype)]

    inside = interpn(
        obs=obs_inside,
        grids=[grid],
        vals=vals,
        method="linear",
        check_bounds=True,
    )
    assert inside.shape == obs_inside[0].shape

    with pytest.raises(ValueError):
        interpn(
            obs=obs_outside,
            grids=[grid],
            vals=vals,
            method="linear",
            check_bounds=True,
        )
