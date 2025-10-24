import numpy as np
import interpn


def _nearest_rectilinear_index(value: float, grid: np.ndarray) -> int:
    idx = np.searchsorted(grid, value, side="right") - 1
    idx = int(max(0, min(idx, grid.size - 2)))
    lower = grid[idx]
    upper = grid[idx + 1]
    dt = (value - lower) / (upper - lower)
    return idx if dt <= 0.5 else idx + 1


def test_nearest_rectilinear():
    for dtype in [np.float64, np.float32]:
        x = np.array([0.0, 1.0, 3.5, 4.0], dtype=dtype)
        y = np.array([-2.0, -0.5, 0.1], dtype=dtype)

        xgrid, ygrid = np.meshgrid(x, y, indexing="ij")
        zgrid = (xgrid + ygrid**2).astype(dtype)

        grids = [x.astype(dtype), y.astype(dtype)]
        obs = [
            np.array([0.2, 2.8, 3.8], dtype=dtype),
            np.array([-1.5, -0.2, 0.4], dtype=dtype),
        ]
        out = np.zeros(obs[0].shape, dtype=dtype)

        if dtype == np.float32:
            interpn.raw.interpn_nearest_rectilinear_f32(
                grids,
                zgrid.flatten(),
                obs,
                out,
            )
        else:
            interpn.raw.interpn_nearest_rectilinear_f64(
                grids,
                zgrid.flatten(),
                obs,
                out,
            )

        expected = []
        for xi, yi in zip(obs[0], obs[1]):
            ix = _nearest_rectilinear_index(float(xi), grids[0])
            iy = _nearest_rectilinear_index(float(yi), grids[1])
            expected.append(zgrid[ix, iy])
        expected = np.array(expected, dtype=dtype)
        np.testing.assert_array_equal(out, expected)

        interpolator = interpn.NearestRectilinear.new(grids, zgrid.flatten())
        out2 = interpolator.eval(obs)
        np.testing.assert_array_equal(out2, expected)

        definitely_inside = [
            np.array([1.0], dtype=dtype),
            np.array([-1.0], dtype=dtype),
        ]
        definitely_outside = [
            np.array([-5.0], dtype=dtype),
            np.array([5.0], dtype=dtype),
        ]
        assert not any(interpolator.check_bounds(definitely_inside, dtype(1e-6)))
        assert any(interpolator.check_bounds(definitely_outside, dtype(1e-6)))

        roundtrip_interpolator = interpn.NearestRectilinear.model_validate_json(
            interpolator.model_dump_json()
        )
        out3 = roundtrip_interpolator.eval(obs)
        np.testing.assert_array_equal(out3, expected)
