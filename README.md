# interpn
N-dimensional interpolation/extrapolation methods, no-std and no-alloc compatible,
prioritizing correctness, performance, and compatiblity with memory-constrained environments.

# Performance Scalings
Note that for a self-consistent multidimensional interpolation, there are 2^ndims grid values that contribute
to each observation point, and as such, that is the theoretical floor for performance scaling. That said,
depending on the implementation, the constant term can vary by more than an order of magnitude.

| Method                        | RAM       | Interp. Cost (Best Case) | Interp. Cost (Worst Case)           | Extrap. Cost (Worst Case)                      |
|-------------------------------|-----------|--------------------------|-------------------------------------|------------------------------------------------|
| multilinear::regular          | O(ndims)  | O(2^ndims)               | O(2^ndims)                          | O(2^ndims + ndims^2)                           |
| multilinear::rectilinear      | O(ndims)  | O(2^ndims)               | O(2^ndims + ndims * log2(gridsize)) | O(2^ndims + ndims^2 + ndims * log2(gridsize))  |

# Example: Multilinear w/ Regular Grid
```rust
use interpn::multilinear::regular;

// Define a grid
let x = [1.0_f64, 2.0];
let y = [1.0_f64, 1.5];

// Grid input for rectilinear method
let grids = &[&x[..], &y[..]];

// Grid input for regular grid method
let dims = [x.len(), y.len()];
let starts = [x[0], y[0]];
let steps = [x[1] - x[0], y[1] - y[0]];

// Values at grid points
let z = [2.0; 4];

// Observation points to interpolate/extrapolate
let xobs = [0.0_f64, 5.0];
let yobs = [-1.0, 3.0];
let obs = [&xobs[..], &yobs[..]];

// Storage for output
let mut out = [0.0; 2];

// Do interpolation
regular::interpn(&dims, &starts, &steps, &z, &obs, &mut out).unwrap();
```

# Example: Multilinear w/ Rectilinear Grid
```rust
use interpn::multilinear::rectilinear;

// Define a grid
let x = [1.0_f64, 1.2, 2.0];
let y = [1.0_f64, 1.3, 1.5];

// Grid input for rectilinear method
let grids = &[&x[..], &y[..]];

// Values at grid points
let z = [2.0; 9];

// Points to interpolate/extrapolate
let xobs = [0.0_f64, 5.0];
let yobs = [-1.0, 3.0];
let obs = [&xobs[..], &yobs[..]];

// Storage for output
let mut out = [0.0; 2];

// Do interpolation
rectilinear::interpn(grids, &z, &obs, &mut out).unwrap();
```

# Development Roadmap
* Limited-memory multi-cubic interpolation/extrapolation
* Vectorized multilinear interpolation/extrapolation (with less strict memory limits)
* Vectorized multi-cubic interpolation/extrapolation (with less strict memory limits)

# License
Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.