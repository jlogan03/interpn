# Performance

## Quality-of-Fit

The cubic Hermite interpolation method used in InterpN is slightly different from a B-spline, which is part of what allows it to achieve higher throughput.

This fit method, which uses first derivative BCs at each grid point on the interior region and a "natural spline" (zero third derivative) at the grid edges, also produces a similar-or-better quality fit by most metrics. This method prioritizes correctness of values and first derivatives over maintaining a continuous second derivative.

The linear methods' quality of fit, being linear, is not very interesting.

### 1D Cubic Interpolation & Extrapolation
InterpN shows significantly improvements in both numerical error and quality-of-fit, especially where sharp changes or strong higher derivatives are present. 

--8<--
docs/1d_quality_of_fit_Rectilinear.html
--8<--

### 2D Cubic Interpolation & Extrapolation
Both InterpN and Scipy methods can full capture a quadratic function in arbitrary dimensions, including under extrapolation. However, InterpN produces several orders of magnitude less floating point error, despite requiring significantly less run time.

--8<--
docs/2d_quality_of_fit_Rectilinear.html
--8<--

### 2D Nearest-Neighbor Interpolation & Extrapolation
InterpN's regular- and rectilinear- grid nearest-neighbor methods match scipy griddata at all tested conditions.
Midpoint tie-breaking is not guaranteed to match exactly.

--8<--
docs/nearest_quality_of_fit.html
--8<--

----
## Throughput

InterpN methods are quite fast.

More commentary about low-level perf scalings for each method
can be found in the [documentation for the Rust library](https://docs.rs/interpn/latest/).

----
### Throughput vs. Dimensionality
Specialized methods (Scipy `RectBivariateSpline` and numpy `interp`), while not N-dimensional methods,
are shown to highlight that InterpN achieves parity even with specialized low-dimensional methods,
despite not specifically handling low-dimensional special cases.

#### 1 Observation Point

--8<--
docs/throughput_vs_dims_1_obs.html
--8<--

--8<--
docs/speedup_vs_dims_1_obs_linear.html
--8<--

--8<--
docs/speedup_vs_dims_1_obs_cubic.html
--8<--

#### 1000 Observation Points

--8<--
docs/throughput_vs_dims_1000_obs.html
--8<--

--8<--
docs/speedup_vs_dims_1000_obs_linear.html
--8<--

--8<--
docs/speedup_vs_dims_1000_obs_cubic.html
--8<--


### 3D Throughput vs. Input Size
Evaluating points in large batches is substantially faster than one-at-a-time for all tested methods.

--8<--
docs/3d_throughput_vs_nobs_prealloc_linear.html
--8<--

--8<--
docs/3d_throughput_vs_nobs_prealloc_cubic.html
--8<--

----
## Memory Usage

Memory profiling in Python is an odd activity. `memory_profiler` is used here, and may miss some memory usage in extension libraries. Also for this reason, the memory profiler is unable to capture the memory used by scipy to actualize the spline knots during initialization, and a comparison of profiled memory usage during initialization is unenlightening.

Since InterpN's backend library does not have access to an allocator, it's unlikely that there is much hidden behind that interface during evaluation. However, it is possible that some memory used by scipy is not identified by the profiler.

The linear methods all use roughly the same amount of RAM during evaluation. In the case of InterpN, as designed, the instantaneous memory usage of all the methods, both linear and cubic, is the same during evaluation.

The memory profiler picks up a large amount of RAM that is not actually part of the function evaluation, but belongs to the outer process. As a result, all methods show a bogus memory usage floor of about 97MB.

![ND memory usage](./ram_vs_dims.svg)
