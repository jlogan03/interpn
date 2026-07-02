use crate::{
    MultiBsplineRectilinear, MultiBsplineRegular, MulticubicRectilinear, MulticubicRegular,
    MultilinearRectilinear, MultilinearRegular, multibspline, testing::assert_close,
};

const LINEAR_DX: f64 = 2.0;
const LINEAR_DY: f64 = -3.0;
const LINEAR_ATOL: f64 = 1e-10;
const FINITE_DIFFERENCE_STEP: f64 = 1e-6;
const FINITE_DIFFERENCE_ATOL: f64 = 1e-7;
const NONLINEAR_POINT: [f64; 2] = [0.23, 1.07];

fn linear_2d(x: f64, y: f64) -> f64 {
    1.25 + LINEAR_DX * x + LINEAR_DY * y
}

fn nonlinear_2d(x: f64, y: f64) -> f64 {
    0.8 * x * x - 0.35 * y * y + 0.25 * x * y + (0.7 * x - 0.2 * y).sin()
}

fn regular_values(
    dims: [usize; 2],
    starts: [f64; 2],
    steps: [f64; 2],
    f: fn(f64, f64) -> f64,
) -> Vec<f64> {
    let mut vals = Vec::with_capacity(dims[0] * dims[1]);
    for i in 0..dims[0] {
        let x = starts[0] + steps[0] * i as f64;
        for j in 0..dims[1] {
            let y = starts[1] + steps[1] * j as f64;
            vals.push(f(x, y));
        }
    }
    vals
}

fn rectilinear_values(x: &[f64], y: &[f64], f: fn(f64, f64) -> f64) -> Vec<f64> {
    let mut vals = Vec::with_capacity(x.len() * y.len());
    for &xi in x {
        for &yj in y {
            vals.push(f(xi, yj));
        }
    }
    vals
}

fn assert_linear_gradient(got: [f64; 2]) {
    assert_close(got[0], LINEAR_DX, LINEAR_ATOL);
    assert_close(got[1], LINEAR_DY, LINEAR_ATOL);
}

fn finite_difference_gradient(
    point: [f64; 2],
    f: impl Fn([f64; 2]) -> Result<f64, &'static str>,
) -> [f64; 2] {
    let h = FINITE_DIFFERENCE_STEP;
    let mut out = [0.0; 2];

    for axis in 0..2 {
        let mut lo = point;
        let mut hi = point;
        lo[axis] -= h;
        hi[axis] += h;
        out[axis] = (f(hi).unwrap() - f(lo).unwrap()) / (2.0 * h);
    }

    out
}

fn assert_gradient_matches_finite_difference(
    got: [f64; 2],
    point: [f64; 2],
    f: impl Fn([f64; 2]) -> Result<f64, &'static str>,
) {
    let expected = finite_difference_gradient(point, f);
    assert_close(got[0], expected[0], FINITE_DIFFERENCE_ATOL);
    assert_close(got[1], expected[1], FINITE_DIFFERENCE_ATOL);
}

#[test]
fn multilinear_regular_gradient_matches_linear_field() {
    let dims = [4, 5];
    let starts = [-1.0, 0.5];
    let steps = [0.75, 0.4];
    let vals = regular_values(dims, starts, steps, linear_2d);
    let interp = MultilinearRegular::new(dims, starts, steps, &vals).unwrap();

    assert_linear_gradient(interp.interp_one_grad([0.2, 1.1]).unwrap());

    let x = [0.2, 0.7];
    let y = [1.1, 0.9];
    let obs = [&x[..], &y[..]];
    let mut gx = [0.0; 2];
    let mut gy = [0.0; 2];
    let mut out = [&mut gx[..], &mut gy[..]];
    interp.interp_grad(&obs, &mut out).unwrap();
    assert_linear_gradient([gx[0], gy[0]]);
    assert_linear_gradient([gx[1], gy[1]]);
}

#[test]
fn multilinear_rectilinear_gradient_matches_linear_field() {
    let x = [-1.0, -0.2, 0.4, 2.0];
    let y = [0.5, 0.7, 1.4, 2.2, 3.0];
    let grids = [&x[..], &y[..]];
    let vals = rectilinear_values(&x, &y, linear_2d);
    let interp = MultilinearRectilinear::new(&grids, &vals).unwrap();

    assert_linear_gradient(interp.interp_one_grad([0.2, 1.1]).unwrap());
}

#[test]
fn multicubic_regular_gradient_matches_linear_field() {
    let dims = [5, 6];
    let starts = [-1.0, 0.5];
    let steps = [0.5, 0.3];
    let vals = regular_values(dims, starts, steps, linear_2d);
    let interp = MulticubicRegular::new(dims, starts, steps, &vals, false).unwrap();

    assert_linear_gradient(interp.interp_one_grad([0.2, 1.1]).unwrap());
}

#[test]
fn multicubic_rectilinear_gradient_matches_linear_field() {
    let x = [-1.0, -0.2, 0.4, 1.1, 2.0];
    let y = [0.5, 0.7, 1.4, 2.2, 3.0, 4.0];
    let grids = [&x[..], &y[..]];
    let vals = rectilinear_values(&x, &y, linear_2d);
    let interp = MulticubicRectilinear::new(&grids, &vals, false).unwrap();

    assert_linear_gradient(interp.interp_one_grad([0.2, 1.1]).unwrap());
}

#[test]
fn multibspline_regular_gradient_matches_linear_field() {
    let dims = [5, 6];
    let starts = [-1.0, 0.5];
    let steps = [0.5, 0.3];
    let vals = regular_values(dims, starts, steps, linear_2d);
    let mut coeffs = vec![0.0; MultiBsplineRegular::<f64, 2>::coeff_storage_len(dims)];
    let mut scratch = vec![0.0; MultiBsplineRegular::<f64, 2>::construction_scratch_len(dims)];
    multibspline::regular::coefficients(dims, &vals, &mut coeffs, &mut scratch).unwrap();
    let interp = MultiBsplineRegular::new(dims, starts, steps, &coeffs, false).unwrap();

    assert_linear_gradient(interp.interp_one_grad([0.2, 1.1]).unwrap());
}

#[test]
fn multibspline_rectilinear_gradient_matches_linear_field() {
    let x = [-1.0, -0.2, 0.4, 1.1, 2.0];
    let y = [0.5, 0.7, 1.4, 2.2, 3.0, 4.0];
    let grids = [&x[..], &y[..]];
    let vals = rectilinear_values(&x, &y, linear_2d);
    let dims = [x.len(), y.len()];
    let mut coeffs = vec![0.0; MultiBsplineRectilinear::<f64, 2>::coeff_storage_len(dims)];
    let mut scratch = vec![0.0; MultiBsplineRectilinear::<f64, 2>::construction_scratch_len(dims)];
    multibspline::rectilinear::coefficients(&grids, &vals, &mut coeffs, &mut scratch).unwrap();
    let interp = MultiBsplineRectilinear::new(&grids, &coeffs, false).unwrap();

    assert_linear_gradient(interp.interp_one_grad([0.2, 1.1]).unwrap());
}

#[test]
fn multilinear_regular_gradient_matches_nonlinear_interpolant() {
    let dims = [5, 6];
    let starts = [-1.0, 0.5];
    let steps = [0.5, 0.3];
    let vals = regular_values(dims, starts, steps, nonlinear_2d);
    let interp = MultilinearRegular::new(dims, starts, steps, &vals).unwrap();
    let point = NONLINEAR_POINT;

    assert_gradient_matches_finite_difference(interp.interp_one_grad(point).unwrap(), point, |x| {
        interp.interp_one(x)
    });
}

#[test]
fn multilinear_rectilinear_gradient_matches_nonlinear_interpolant() {
    let x = [-1.0, -0.2, 0.4, 1.1, 2.0];
    let y = [0.5, 0.7, 1.4, 2.2, 3.0, 4.0];
    let grids = [&x[..], &y[..]];
    let vals = rectilinear_values(&x, &y, nonlinear_2d);
    let interp = MultilinearRectilinear::new(&grids, &vals).unwrap();
    let point = NONLINEAR_POINT;

    assert_gradient_matches_finite_difference(interp.interp_one_grad(point).unwrap(), point, |x| {
        interp.interp_one(x)
    });
}

#[test]
fn multicubic_regular_gradient_matches_nonlinear_interpolant() {
    let dims = [6, 7];
    let starts = [-1.0, 0.5];
    let steps = [0.4, 0.25];
    let vals = regular_values(dims, starts, steps, nonlinear_2d);
    let interp = MulticubicRegular::new(dims, starts, steps, &vals, false).unwrap();
    let point = NONLINEAR_POINT;

    assert_gradient_matches_finite_difference(interp.interp_one_grad(point).unwrap(), point, |x| {
        interp.interp_one(x)
    });
}

#[test]
fn multicubic_rectilinear_gradient_matches_nonlinear_interpolant() {
    let x = [-1.0, -0.2, 0.4, 1.1, 2.0, 3.1];
    let y = [0.5, 0.7, 1.4, 2.2, 3.0, 4.0, 5.2];
    let grids = [&x[..], &y[..]];
    let vals = rectilinear_values(&x, &y, nonlinear_2d);
    let interp = MulticubicRectilinear::new(&grids, &vals, false).unwrap();
    let point = NONLINEAR_POINT;

    assert_gradient_matches_finite_difference(interp.interp_one_grad(point).unwrap(), point, |x| {
        interp.interp_one(x)
    });
}

#[test]
fn multibspline_regular_gradient_matches_nonlinear_interpolant() {
    let dims = [6, 7];
    let starts = [-1.0, 0.5];
    let steps = [0.4, 0.25];
    let vals = regular_values(dims, starts, steps, nonlinear_2d);
    let mut coeffs = vec![0.0; MultiBsplineRegular::<f64, 2>::coeff_storage_len(dims)];
    let mut scratch = vec![0.0; MultiBsplineRegular::<f64, 2>::construction_scratch_len(dims)];
    multibspline::regular::coefficients(dims, &vals, &mut coeffs, &mut scratch).unwrap();
    let interp = MultiBsplineRegular::new(dims, starts, steps, &coeffs, false).unwrap();
    let point = NONLINEAR_POINT;

    assert_gradient_matches_finite_difference(interp.interp_one_grad(point).unwrap(), point, |x| {
        interp.interp_one(x)
    });
}

#[test]
fn multibspline_rectilinear_gradient_matches_nonlinear_interpolant() {
    let x = [-1.0, -0.2, 0.4, 1.1, 2.0, 3.1];
    let y = [0.5, 0.7, 1.4, 2.2, 3.0, 4.0, 5.2];
    let grids = [&x[..], &y[..]];
    let vals = rectilinear_values(&x, &y, nonlinear_2d);
    let dims = [x.len(), y.len()];
    let mut coeffs = vec![0.0; MultiBsplineRectilinear::<f64, 2>::coeff_storage_len(dims)];
    let mut scratch = vec![0.0; MultiBsplineRectilinear::<f64, 2>::construction_scratch_len(dims)];
    multibspline::rectilinear::coefficients(&grids, &vals, &mut coeffs, &mut scratch).unwrap();
    let interp = MultiBsplineRectilinear::new(&grids, &coeffs, false).unwrap();
    let point = NONLINEAR_POINT;

    assert_gradient_matches_finite_difference(interp.interp_one_grad(point).unwrap(), point, |x| {
        interp.interp_one(x)
    });
}
