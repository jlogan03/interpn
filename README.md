# interpn
N-dimensional interpolation methods.

```rust
use interpn::{multilinear_rectilinear, multilinear_regular};

let x = [1.0_f64, 2.0];
let y = [1.0_f64, 2.0];
let z = [2.0; 4];

let mut out = [0.0; 4];

// Grid information for regular-grid method
let dims = [x.len(), y.len()];
let starts = [x[0], y[0]];
let steps = [x[1] - x[0], y[1] - y[0]];

// Grid information for rectilinear-grid method
let grids = &[&x[..], &y[..]];

// An observation location
let xy_obs = [0.0, 0.0];

multilinear_regular::interpn(
    &xy[..],
    &mut out[..],
    &z[..],
    &dims[..],
    &starts[..],
    &steps[..],
);

multilinear_rectilinear::interpn(
    &xy[..],
    &mut out[..],
    &z[..],
    grids,
);
```