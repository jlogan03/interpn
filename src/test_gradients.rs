use crate::{
    MultiBsplineRectilinear, MultiBsplineRegular, MulticubicRectilinear, MulticubicRegular,
    MultilinearRectilinear, MultilinearRegular, multibspline,
};

fn linear_2d(x: f64, y: f64) -> f64 {
    1.25 + 2.0 * x - 3.0 * y
}

fn regular_values(dims: [usize; 2], starts: [f64; 2], steps: [f64; 2]) -> Vec<f64> {
    let mut vals = Vec::with_capacity(dims[0] * dims[1]);
    for i in 0..dims[0] {
        let x = starts[0] + steps[0] * i as f64;
        for j in 0..dims[1] {
            let y = starts[1] + steps[1] * j as f64;
            vals.push(linear_2d(x, y));
        }
    }
    vals
}

fn rectilinear_values(x: &[f64], y: &[f64]) -> Vec<f64> {
    let mut vals = Vec::with_capacity(x.len() * y.len());
    for &xi in x {
        for &yj in y {
            vals.push(linear_2d(xi, yj));
        }
    }
    vals
}

fn assert_grad(got: [f64; 2]) {
    assert!((got[0] - 2.0).abs() < 1e-10, "bad x gradient: {}", got[0]);
    assert!((got[1] + 3.0).abs() < 1e-10, "bad y gradient: {}", got[1]);
}

#[test]
fn multilinear_regular_gradient_matches_linear_field() {
    let dims = [4, 5];
    let starts = [-1.0, 0.5];
    let steps = [0.75, 0.4];
    let vals = regular_values(dims, starts, steps);
    let interp = MultilinearRegular::new(dims, starts, steps, &vals).unwrap();

    assert_grad(interp.interp_one_grad([0.2, 1.1]).unwrap());

    let x = [0.2, 0.7];
    let y = [1.1, 0.9];
    let obs = [&x[..], &y[..]];
    let mut gx = [0.0; 2];
    let mut gy = [0.0; 2];
    let mut out = [&mut gx[..], &mut gy[..]];
    interp.interp_grad(&obs, &mut out).unwrap();
    assert_grad([gx[0], gy[0]]);
    assert_grad([gx[1], gy[1]]);
}

#[test]
fn multilinear_rectilinear_gradient_matches_linear_field() {
    let x = [-1.0, -0.2, 0.4, 2.0];
    let y = [0.5, 0.7, 1.4, 2.2, 3.0];
    let grids = [&x[..], &y[..]];
    let vals = rectilinear_values(&x, &y);
    let interp = MultilinearRectilinear::new(&grids, &vals).unwrap();

    assert_grad(interp.interp_one_grad([0.2, 1.1]).unwrap());
}

#[test]
fn multicubic_regular_gradient_matches_linear_field() {
    let dims = [5, 6];
    let starts = [-1.0, 0.5];
    let steps = [0.5, 0.3];
    let vals = regular_values(dims, starts, steps);
    let interp = MulticubicRegular::new(dims, starts, steps, &vals, false).unwrap();

    assert_grad(interp.interp_one_grad([0.2, 1.1]).unwrap());
}

#[test]
fn multicubic_rectilinear_gradient_matches_linear_field() {
    let x = [-1.0, -0.2, 0.4, 1.1, 2.0];
    let y = [0.5, 0.7, 1.4, 2.2, 3.0, 4.0];
    let grids = [&x[..], &y[..]];
    let vals = rectilinear_values(&x, &y);
    let interp = MulticubicRectilinear::new(&grids, &vals, false).unwrap();

    assert_grad(interp.interp_one_grad([0.2, 1.1]).unwrap());
}

#[test]
fn multibspline_regular_gradient_matches_linear_field() {
    let dims = [5, 6];
    let starts = [-1.0, 0.5];
    let steps = [0.5, 0.3];
    let vals = regular_values(dims, starts, steps);
    let mut coeffs = vec![0.0; MultiBsplineRegular::<f64, 2>::coeff_storage_len(dims)];
    let mut scratch = vec![0.0; MultiBsplineRegular::<f64, 2>::construction_scratch_len(dims)];
    multibspline::regular::coefficients(dims, &vals, &mut coeffs, &mut scratch).unwrap();
    let interp = MultiBsplineRegular::new(dims, starts, steps, &coeffs, false).unwrap();

    assert_grad(interp.interp_one_grad([0.2, 1.1]).unwrap());
}

#[test]
fn multibspline_rectilinear_gradient_matches_linear_field() {
    let x = [-1.0, -0.2, 0.4, 1.1, 2.0];
    let y = [0.5, 0.7, 1.4, 2.2, 3.0, 4.0];
    let grids = [&x[..], &y[..]];
    let vals = rectilinear_values(&x, &y);
    let dims = [x.len(), y.len()];
    let mut coeffs = vec![0.0; MultiBsplineRectilinear::<f64, 2>::coeff_storage_len(dims)];
    let mut scratch = vec![0.0; MultiBsplineRectilinear::<f64, 2>::construction_scratch_len(dims)];
    multibspline::rectilinear::coefficients(&grids, &vals, &mut coeffs, &mut scratch).unwrap();
    let interp = MultiBsplineRectilinear::new(&grids, &coeffs, false).unwrap();

    assert_grad(interp.interp_one_grad([0.2, 1.1]).unwrap());
}
