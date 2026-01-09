use numpy::borrow::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions;
use pyo3::prelude::*;

use crate::multicubic;
use crate::multilinear;
use crate::nearest;
use crate::{GridInterpMethod, GridKind};

/// Maximum number of dimensions for linear interpn convenience methods
const MAXDIMS: usize = 8;

/// Python bindings for select functions from `interpn`.
#[pymodule]
#[pyo3(name = "interpn")]
fn interpn<'py>(_py: Python, m: &Bound<'py, PyModule>) -> PyResult<()> {
    // Multilinear regular grid
    m.add_function(wrap_pyfunction!(interpn_linear_regular_f64, m)?)?;
    m.add_function(wrap_pyfunction!(interpn_linear_regular_f32, m)?)?;
    m.add_function(wrap_pyfunction!(check_bounds_regular_f64, m)?)?;
    m.add_function(wrap_pyfunction!(check_bounds_regular_f32, m)?)?;
    // Multilinear rectilinear grid
    m.add_function(wrap_pyfunction!(interpn_linear_rectilinear_f64, m)?)?;
    m.add_function(wrap_pyfunction!(interpn_linear_rectilinear_f32, m)?)?;
    m.add_function(wrap_pyfunction!(check_bounds_rectilinear_f64, m)?)?;
    m.add_function(wrap_pyfunction!(check_bounds_rectilinear_f32, m)?)?;
    // Nearest-neighbor regular grid
    m.add_function(wrap_pyfunction!(interpn_nearest_regular_f64, m)?)?;
    m.add_function(wrap_pyfunction!(interpn_nearest_regular_f32, m)?)?;
    // Nearest-neighbor rectilinear grid
    m.add_function(wrap_pyfunction!(interpn_nearest_rectilinear_f64, m)?)?;
    m.add_function(wrap_pyfunction!(interpn_nearest_rectilinear_f32, m)?)?;
    // Multicubic with regular grid
    m.add_function(wrap_pyfunction!(interpn_cubic_regular_f64, m)?)?;
    m.add_function(wrap_pyfunction!(interpn_cubic_regular_f32, m)?)?;
    // Multicubic with rectilinear grid
    m.add_function(wrap_pyfunction!(interpn_cubic_rectilinear_f64, m)?)?;
    m.add_function(wrap_pyfunction!(interpn_cubic_rectilinear_f32, m)?)?;
    // Top-level interpn dispatch
    m.add_function(wrap_pyfunction!(interpn_f64, m)?)?;
    m.add_function(wrap_pyfunction!(interpn_f32, m)?)?;
    Ok(())
}

macro_rules! unpack_vec_of_arr {
    ($inname:ident, $outname:ident, $T:ty) => {
        // We need a mutable slice-of-slice,
        // and it has to start with a reference to something
        let dummy = [0.0; 0];
        let mut _arr: [&[$T]; MAXDIMS] = [&dummy[..]; MAXDIMS];
        let n = $inname.len();
        for i in 0..n {
            _arr[i] = $inname[i].as_slice()?;
        }
        let $outname = &_arr[..n];
    };
}

macro_rules! interpn_linear_regular_impl {
    ($funcname:ident, $T:ty) => {
        #[pyfunction]
        fn $funcname(
            dims: Vec<usize>, // numpy index arrays are signed; this avoids casting
            starts: PyReadonlyArray1<$T>,
            steps: PyReadonlyArray1<$T>,
            vals: PyReadonlyArray1<$T>,
            obs: Vec<PyReadonlyArray1<$T>>,
            mut out: PyReadwriteArray1<$T>,
        ) -> PyResult<()> {
            unpack_vec_of_arr!(obs, obs, $T);

            // Evaluate
            match multilinear::regular::interpn(
                &dims,
                starts.as_slice()?,
                steps.as_slice()?,
                vals.as_slice()?,
                obs,
                out.as_slice_mut()?,
            ) {
                Ok(()) => Ok(()),
                Err(msg) => Err(exceptions::PyAssertionError::new_err(msg)),
            }
        }
    };
}

interpn_linear_regular_impl!(interpn_linear_regular_f64, f64);
interpn_linear_regular_impl!(interpn_linear_regular_f32, f32);

macro_rules! check_bounds_regular_impl {
    ($funcname:ident, $T:ty) => {
        #[pyfunction]
        fn $funcname(
            dims: Vec<usize>, // numpy index arrays are signed; this avoids casting
            starts: PyReadonlyArray1<$T>,
            steps: PyReadonlyArray1<$T>,
            obs: Vec<PyReadonlyArray1<$T>>,
            atol: $T,
            mut out: PyReadwriteArray1<bool>,
        ) -> PyResult<()> {
            unpack_vec_of_arr!(obs, obs, $T);

            // Evaluate
            match multilinear::regular::check_bounds(
                &dims,
                starts.as_slice()?,
                steps.as_slice()?,
                obs,
                atol,
                out.as_slice_mut()?,
            ) {
                Ok(()) => Ok(()),
                Err(msg) => Err(exceptions::PyAssertionError::new_err(msg)),
            }
        }
    };
}

check_bounds_regular_impl!(check_bounds_regular_f64, f64);
check_bounds_regular_impl!(check_bounds_regular_f32, f32);

macro_rules! interpn_linear_rectilinear_impl {
    ($funcname:ident, $T:ty) => {
        #[pyfunction]
        fn $funcname(
            grids: Vec<PyReadonlyArray1<$T>>,
            vals: PyReadonlyArray1<$T>,
            obs: Vec<PyReadonlyArray1<$T>>,
            mut out: PyReadwriteArray1<$T>,
        ) -> PyResult<()> {
            // Unpack inputs
            unpack_vec_of_arr!(grids, grids, $T);
            unpack_vec_of_arr!(obs, obs, $T);

            // Evaluate
            match multilinear::rectilinear::interpn(
                grids,
                vals.as_slice()?,
                obs,
                out.as_slice_mut()?,
            ) {
                Ok(()) => Ok(()),
                Err(msg) => Err(exceptions::PyAssertionError::new_err(msg)),
            }
        }
    };
}

interpn_linear_rectilinear_impl!(interpn_linear_rectilinear_f64, f64);
interpn_linear_rectilinear_impl!(interpn_linear_rectilinear_f32, f32);

macro_rules! interpn_nearest_regular_impl {
    ($funcname:ident, $T:ty) => {
        #[pyfunction]
        fn $funcname(
            dims: Vec<usize>,
            starts: PyReadonlyArray1<$T>,
            steps: PyReadonlyArray1<$T>,
            vals: PyReadonlyArray1<$T>,
            obs: Vec<PyReadonlyArray1<$T>>,
            mut out: PyReadwriteArray1<$T>,
        ) -> PyResult<()> {
            unpack_vec_of_arr!(obs, obs, $T);

            match nearest::regular::interpn(
                &dims,
                starts.as_slice()?,
                steps.as_slice()?,
                vals.as_slice()?,
                obs,
                out.as_slice_mut()?,
            ) {
                Ok(()) => Ok(()),
                Err(msg) => Err(exceptions::PyAssertionError::new_err(msg)),
            }
        }
    };
}

interpn_nearest_regular_impl!(interpn_nearest_regular_f64, f64);
interpn_nearest_regular_impl!(interpn_nearest_regular_f32, f32);

macro_rules! interpn_nearest_rectilinear_impl {
    ($funcname:ident, $T:ty) => {
        #[pyfunction]
        fn $funcname(
            grids: Vec<PyReadonlyArray1<$T>>,
            vals: PyReadonlyArray1<$T>,
            obs: Vec<PyReadonlyArray1<$T>>,
            mut out: PyReadwriteArray1<$T>,
        ) -> PyResult<()> {
            unpack_vec_of_arr!(grids, grids, $T);
            unpack_vec_of_arr!(obs, obs, $T);

            match nearest::rectilinear::interpn(grids, vals.as_slice()?, obs, out.as_slice_mut()?) {
                Ok(()) => Ok(()),
                Err(msg) => Err(exceptions::PyAssertionError::new_err(msg)),
            }
        }
    };
}

interpn_nearest_rectilinear_impl!(interpn_nearest_rectilinear_f64, f64);
interpn_nearest_rectilinear_impl!(interpn_nearest_rectilinear_f32, f32);

macro_rules! check_bounds_rectilinear_impl {
    ($funcname:ident, $T:ty) => {
        #[pyfunction]
        fn $funcname(
            grids: Vec<PyReadonlyArray1<$T>>,
            obs: Vec<PyReadonlyArray1<$T>>,
            atol: $T,
            mut out: PyReadwriteArray1<bool>,
        ) -> PyResult<()> {
            // Unpack inputs
            unpack_vec_of_arr!(grids, grids, $T);
            unpack_vec_of_arr!(obs, obs, $T);

            // Evaluate
            match multilinear::rectilinear::check_bounds(&grids, obs, atol, out.as_slice_mut()?) {
                Ok(()) => Ok(()),
                Err(msg) => Err(exceptions::PyAssertionError::new_err(msg)),
            }
        }
    };
}

check_bounds_rectilinear_impl!(check_bounds_rectilinear_f64, f64);
check_bounds_rectilinear_impl!(check_bounds_rectilinear_f32, f32);

macro_rules! interpn_cubic_regular_impl {
    ($funcname:ident, $T:ty) => {
        #[pyfunction]
        fn $funcname(
            dims: Vec<usize>, // numpy index arrays are signed; this avoids casting
            starts: PyReadonlyArray1<$T>,
            steps: PyReadonlyArray1<$T>,
            vals: PyReadonlyArray1<$T>,
            linearize_extrapolation: bool,
            obs: Vec<PyReadonlyArray1<$T>>,
            mut out: PyReadwriteArray1<$T>,
        ) -> PyResult<()> {
            unpack_vec_of_arr!(obs, obs, $T);

            // Evaluate
            match multicubic::regular::interpn(
                &dims,
                starts.as_slice()?,
                steps.as_slice()?,
                vals.as_slice()?,
                linearize_extrapolation,
                obs,
                out.as_slice_mut()?,
            ) {
                Ok(()) => Ok(()),
                Err(msg) => Err(exceptions::PyAssertionError::new_err(msg)),
            }
        }
    };
}

interpn_cubic_regular_impl!(interpn_cubic_regular_f64, f64);
interpn_cubic_regular_impl!(interpn_cubic_regular_f32, f32);

macro_rules! interpn_cubic_rectilinear_impl {
    ($funcname:ident, $T:ty) => {
        #[pyfunction]
        fn $funcname(
            grids: Vec<PyReadonlyArray1<$T>>,
            vals: PyReadonlyArray1<$T>,
            linearize_extrapolation: bool,
            obs: Vec<PyReadonlyArray1<$T>>,
            mut out: PyReadwriteArray1<$T>,
        ) -> PyResult<()> {
            // Unpack inputs
            unpack_vec_of_arr!(grids, grids, $T);
            unpack_vec_of_arr!(obs, obs, $T);

            // Evaluate
            match multicubic::rectilinear::interpn(
                grids,
                vals.as_slice()?,
                linearize_extrapolation,
                obs,
                out.as_slice_mut()?,
            ) {
                Ok(()) => Ok(()),
                Err(msg) => Err(exceptions::PyAssertionError::new_err(msg)),
            }
        }
    };
}

interpn_cubic_rectilinear_impl!(interpn_cubic_rectilinear_f64, f64);
interpn_cubic_rectilinear_impl!(interpn_cubic_rectilinear_f32, f32);

fn parse_grid_interp_method(method: &str) -> Result<GridInterpMethod, PyErr> {
    match method.to_ascii_lowercase().as_str() {
        "linear" => Ok(GridInterpMethod::Linear),
        "cubic" => Ok(GridInterpMethod::Cubic),
        "nearest" => Ok(GridInterpMethod::Nearest),
        _ => Err(exceptions::PyValueError::new_err(
            "`method` must be 'linear', 'cubic', or 'nearest'",
        )),
    }
}

fn parse_grid_kind(grid_kind: Option<&str>) -> Result<Option<GridKind>, PyErr> {
    match grid_kind.map(|kind| kind.to_ascii_lowercase()) {
        None => Ok(None),
        Some(kind) => match kind.as_str() {
            "regular" => Ok(Some(GridKind::Regular)),
            "rectilinear" => Ok(Some(GridKind::Rectilinear)),
            _ => Err(exceptions::PyValueError::new_err(
                "`grid_kind` must be 'regular', 'rectilinear', or None",
            )),
        },
    }
}

macro_rules! interpn_top_impl {
    ($funcname:ident, $T:ty) => {
        #[pyfunction]
        #[pyo3(signature = (grids, vals, obs, out, method="linear", grid_kind=None, linearize_extrapolation=true, check_bounds_with_atol=None, max_threads=None))]
        fn $funcname(
            grids: Vec<PyReadonlyArray1<$T>>,
            vals: PyReadonlyArray1<$T>,
            obs: Vec<PyReadonlyArray1<$T>>,
            mut out: PyReadwriteArray1<$T>,
            method: &str,
            grid_kind: Option<&str>,
            linearize_extrapolation: bool,
            check_bounds_with_atol: Option<$T>,
            max_threads: Option<usize>,
        ) -> PyResult<()> {
            unpack_vec_of_arr!(grids, grids, $T);
            unpack_vec_of_arr!(obs, obs, $T);

            let method = parse_grid_interp_method(method)?;
            let grid_kind = parse_grid_kind(grid_kind)?;

            match crate::interpn(
                grids,
                vals.as_slice()?,
                obs,
                out.as_slice_mut()?,
                method,
                grid_kind,
                linearize_extrapolation,
                check_bounds_with_atol,
                max_threads,
            ) {
                Ok(()) => Ok(()),
                Err(msg) => Err(exceptions::PyAssertionError::new_err(msg)),
            }
        }
    };
}

interpn_top_impl!(interpn_f64, f64);
interpn_top_impl!(interpn_f32, f32);
