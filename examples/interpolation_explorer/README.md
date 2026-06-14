# interpolation_explorer

Browser-hosted `interpn` example using Leptos, Trunk, and Plotly.

```bash
rustup target add wasm32-unknown-unknown
cargo install trunk
cd examples/interpolation_explorer
trunk serve
```

The app contains a 1D interpolation comparison page and a B-spline knot-value panel.
