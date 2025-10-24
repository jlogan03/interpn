import numpy as np
import interpn


def _nearest_regular_index(value: float, start: float, step: float, size: int) -> int:
    loc = np.floor((value - start) / step)
    loc = int(max(0, min(loc, size - 2)))
    base = start + step * loc
    dt = (value - base) / step
    return min(loc if dt <= 0.5 else loc + 1, size - 1)


def test_nearest_regular():
    for dtype in [np.float64, np.float32]:
        x = np.linspace(0.0, 6.0, 4).astype(dtype)
        y = np.linspace(-3.0, 3.0, 3).astype(dtype)

        xgrid, ygrid = np.meshgrid(x, y, indexing="ij")
        zgrid = (xgrid - 2.0 * ygrid).astype(dtype)

        dims = [x.size, y.size]
        starts = np.array([x[0], y[0]]).astype(dtype)
        steps = np.array([x[1] - x[0], y[1] - y[0]]).astype(dtype)

        obs = [
            np.array([0.1, 1.6, 2.9, 5.0], dtype=dtype),
            np.array([-3.0, -1.2, 0.4, 2.4], dtype=dtype),
        ]
        out = np.zeros(obs[0].shape, dtype=dtype)

        if dtype == np.float32:
            interpn.raw.interpn_nearest_regular_f32(
                dims,
                starts,
                steps,
                zgrid.flatten(),
                obs,
                out,
            )
        else:
            interpn.raw.interpn_nearest_regular_f64(
                dims,
                starts,
                steps,
                zgrid.flatten(),
                obs,
                out,
            )

        expected = []
        for xi, yi in zip(obs[0], obs[1], strict=True):
            ix = _nearest_regular_index(
                float(xi), float(starts[0]), float(steps[0]), dims[0]
            )
            iy = _nearest_regular_index(
                float(yi), float(starts[1]), float(steps[1]), dims[1]
            )
            expected.append(zgrid[ix, iy])
        expected = np.array(expected, dtype=dtype)
        np.testing.assert_array_equal(out, expected)

        interpolator = interpn.NearestRegular.new(dims, starts, steps, zgrid.flatten())
        out2 = interpolator.eval(obs)
        np.testing.assert_array_equal(out2, expected)

        definitely_inside = [
            np.array([3.0], dtype=dtype),
            np.array([0.0], dtype=dtype),
        ]
        definitely_outside = [
            np.array([-5.0], dtype=dtype),
            np.array([10.0], dtype=dtype),
        ]
        assert not any(interpolator.check_bounds(definitely_inside, dtype(1e-6)))
        assert any(interpolator.check_bounds(definitely_outside, dtype(1e-6)))

        roundtrip_interpolator = interpn.NearestRegular.model_validate_json(
            interpolator.model_dump_json()
        )
        out3 = roundtrip_interpolator.eval(obs)
        np.testing.assert_array_equal(out3, expected)
