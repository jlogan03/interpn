import numpy as np
import interpn


def test_multilinear_regular():
    for dtype in [np.float64, np.float32]:
        x = np.linspace(0.0, 10.0, 5).astype(dtype)
        y = np.linspace(20.0, 30.0, 3).astype(dtype)

        xgrid, ygrid = np.meshgrid(x, y, indexing="ij")
        zgrid = (xgrid + 2.0 * ygrid).astype(dtype)  # Values at grid points

        # Grid inputs
        dims = [x.size, y.size]
        starts = np.array([x[0], y[0]]).astype(dtype)
        steps = np.array([x[1] - x[0], y[1] - y[0]]).astype(dtype)

        # Observation points
        obs = [xgrid.flatten().astype(dtype), ygrid.flatten().astype(dtype)]

        # Output storage
        out = np.zeros_like(zgrid.flatten()).astype(dtype)

        # Do interpolation
        if dtype == np.float32:
            interpn.raw.interpn_linear_regular_f32(
                dims,
                starts,
                steps,
                zgrid.flatten(),
                obs,
                out,
            )
        else:
            interpn.raw.interpn_linear_regular_f64(
                dims,
                starts,
                steps,
                zgrid.flatten(),
                obs,
                out,
            )

        # Check results
        zf = zgrid.flatten()
        for i in range(out.size):
            assert out[i] == zf[i]

        # Do interpolation using class
        interpolator = interpn.MultilinearRegular.new(
            dims, starts, steps, zgrid.flatten()
        )
        out2 = interpolator.eval(obs)

        # Check results
        zf = zgrid.flatten()
        for i in range(out2.size):
            assert out2[i] == zf[i]

        # Exercise check_bounds
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

        # Test roundtrip serialization
        roundtrip_interpolator = interpn.MultilinearRegular.model_validate_json(
            interpolator.model_dump_json()
        )
        out3 = roundtrip_interpolator.eval(obs)

        # Check results
        zf = zgrid.flatten()
        for i in range(out3.size):
            assert out3[i] == zf[i]
