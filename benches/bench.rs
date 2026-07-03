#![allow(clippy::all)] // Clippy will attempt to remove black_box() internals

use criterion::*;
use gridgen::*;
use interpn::{
    Linear1D, LinearHoldLast1D, MultiBsplineRectilinear, MultiBsplineRegular,
    MulticubicRectilinear, MulticubicRegular, MultilinearRectilinear, MultilinearRegular,
    RectilinearGrid1D, RegularGrid1D, multibspline, multicubic, multilinear, nearest,
    one_dim::{
        Interp1D,
        hold::{Left1D, Nearest1D},
    },
};

use std::hint::black_box;

macro_rules! with_grad_out {
    (1, $size:expr, $out:ident, $body:block) => {{
        let mut out0 = vec![0.0; $size];
        let mut $out = [&mut out0[..]];
        $body
    }};
    (2, $size:expr, $out:ident, $body:block) => {{
        let mut out0 = vec![0.0; $size];
        let mut out1 = vec![0.0; $size];
        let mut $out = [&mut out0[..], &mut out1[..]];
        $body
    }};
    (3, $size:expr, $out:ident, $body:block) => {{
        let mut out0 = vec![0.0; $size];
        let mut out1 = vec![0.0; $size];
        let mut out2 = vec![0.0; $size];
        let mut $out = [&mut out0[..], &mut out1[..], &mut out2[..]];
        $body
    }};
}

macro_rules! bench_interp_specific {
    ($group:ident, $ndims:tt, $gridsize:expr, $size:expr) => {
        $group.throughput(Throughput::Elements(*$size as u64));
        // $group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
        let scan_or_shuffle = "Shuffled Order";

        // Do some benches with small fixed MAXDIMS to check if the
        // larger default value has any significant effect on perf.
        $group.bench_with_input(
            BenchmarkId::new(
                format!(
                    "Linear Regular {}x{}D MAXDIMS={}, {}",
                    $gridsize, $ndims, $ndims, scan_or_shuffle
                ),
                $size,
            ),
            $size,
            |b, &size| {
                // Interpolation grid
                let (grids, z) = gen_grid($ndims, $gridsize, 0.0);

                // Observation grid
                let m: usize = ((size as f64).powf(1.0 / ($ndims as f64)) + 2.0) as usize;
                let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                let obs: &[&[f64]; $ndims] = (&obs[..]).try_into().unwrap();
                let mut out = vec![0.0; size];

                let dims = [$gridsize; $ndims];
                let mut starts = [0.0; $ndims];
                let mut steps = [0.0; $ndims];
                (0..$ndims).for_each(|i| starts[i] = grids[i][0]);
                (0..$ndims).for_each(|i| steps[i] = grids[i][1] - grids[i][0]);

                b.iter(|| {
                    black_box({
                        let interpolator: MultilinearRegular<'_, _, $ndims> =
                            MultilinearRegular::new(dims, starts, steps, &z).unwrap();
                        interpolator.interp(obs, &mut out).unwrap()
                    })
                });
            },
        );

        $group.bench_with_input(
            BenchmarkId::new(
                format!(
                    "Gradient Linear Regular {}x{}D MAXDIMS={}, {}",
                    $gridsize, $ndims, $ndims, scan_or_shuffle
                ),
                $size,
            ),
            $size,
            |b, &size| {
                let (grids, z) = gen_grid($ndims, $gridsize, 0.0);

                let m: usize = ((size as f64).powf(1.0 / ($ndims as f64)) + 2.0) as usize;
                let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                let obs: &[&[f64]; $ndims] = (&obs[..]).try_into().unwrap();

                let dims = [$gridsize; $ndims];
                let mut starts = [0.0; $ndims];
                let mut steps = [0.0; $ndims];
                (0..$ndims).for_each(|i| starts[i] = grids[i][0]);
                (0..$ndims).for_each(|i| steps[i] = grids[i][1] - grids[i][0]);

                let interpolator: MultilinearRegular<'_, _, $ndims> =
                    MultilinearRegular::new(dims, starts, steps, &z).unwrap();

                with_grad_out!($ndims, size, out, {
                    b.iter(|| black_box({ interpolator.interp_grad(obs, &mut out).unwrap() }));
                });
            },
        );

        $group.bench_with_input(
            BenchmarkId::new(
                format!(
                    "Linear Regular {}x{}D MAXDIMS=8, {}",
                    $gridsize, $ndims, scan_or_shuffle
                ),
                $size,
            ),
            $size,
            |b, &size| {
                // Interpolation grid
                let (grids, z) = gen_grid($ndims, $gridsize, 0.0);

                // Observation grid
                let m: usize = ((size as f64).powf(1.0 / ($ndims as f64)) + 2.0) as usize;
                let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                let obs: &[&[f64]; $ndims] = (&obs[..]).try_into().unwrap();
                let mut out = vec![0.0; size];

                let dims = [$gridsize; $ndims];
                let mut starts = [0.0; $ndims];
                let mut steps = [0.0; $ndims];
                (0..$ndims).for_each(|i| starts[i] = grids[i][0]);
                (0..$ndims).for_each(|i| steps[i] = grids[i][1] - grids[i][0]);

                b.iter(|| {
                    black_box({
                        multilinear::regular::interpn(&dims, &starts, &steps, &z, obs, &mut out)
                            .unwrap()
                    })
                });
            },
        );

        $group.bench_with_input(
            BenchmarkId::new(
                format!(
                    "Bspline Regular {}x{}D Precomputed Coeffs, {}",
                    $gridsize, $ndims, scan_or_shuffle
                ),
                $size,
            ),
            $size,
            |b, &size| {
                // Interpolation grid
                let (grids, z) = gen_grid($ndims, $gridsize, 0.0);

                // Observation grid
                let m: usize = ((size as f64).powf(1.0 / ($ndims as f64)) + 2.0) as usize;
                let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                let obs: &[&[f64]; $ndims] = (&obs[..]).try_into().unwrap();
                let mut out = vec![0.0; size];

                let dims = [$gridsize; $ndims];
                let mut starts = [0.0; $ndims];
                let mut steps = [0.0; $ndims];
                (0..$ndims).for_each(|i| starts[i] = grids[i][0]);
                (0..$ndims).for_each(|i| steps[i] = grids[i][1] - grids[i][0]);

                let coeff_len =
                    multibspline::MultiBsplineRegular::<f64, $ndims>::coeff_storage_len(dims);
                let scratch_len =
                    multibspline::MultiBsplineRegular::<f64, $ndims>::construction_scratch_len(
                        dims,
                    );
                let mut coeffs = vec![0.0; coeff_len];
                let mut scratch = vec![0.0; scratch_len];
                multibspline::regular::coefficients(dims, &z, &mut coeffs, &mut scratch).unwrap();
                let interpolator: multibspline::MultiBsplineRegular<'_, _, $ndims> =
                    multibspline::MultiBsplineRegular::new(dims, starts, steps, &coeffs, false)
                        .unwrap();

                b.iter(|| black_box({ interpolator.interp(obs, &mut out).unwrap() }));
            },
        );

        $group.bench_with_input(
            BenchmarkId::new(
                format!(
                    "Gradient Bspline Regular {}x{}D Precomputed Coeffs, {}",
                    $gridsize, $ndims, scan_or_shuffle
                ),
                $size,
            ),
            $size,
            |b, &size| {
                let (grids, z) = gen_grid($ndims, $gridsize, 0.0);

                let m: usize = ((size as f64).powf(1.0 / ($ndims as f64)) + 2.0) as usize;
                let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                let obs: &[&[f64]; $ndims] = (&obs[..]).try_into().unwrap();

                let dims = [$gridsize; $ndims];
                let mut starts = [0.0; $ndims];
                let mut steps = [0.0; $ndims];
                (0..$ndims).for_each(|i| starts[i] = grids[i][0]);
                (0..$ndims).for_each(|i| steps[i] = grids[i][1] - grids[i][0]);

                let coeff_len = MultiBsplineRegular::<f64, $ndims>::coeff_storage_len(dims);
                let scratch_len =
                    MultiBsplineRegular::<f64, $ndims>::construction_scratch_len(dims);
                let mut coeffs = vec![0.0; coeff_len];
                let mut scratch = vec![0.0; scratch_len];
                multibspline::regular::coefficients(dims, &z, &mut coeffs, &mut scratch).unwrap();
                let interpolator: MultiBsplineRegular<'_, _, $ndims> =
                    MultiBsplineRegular::new(dims, starts, steps, &coeffs, false).unwrap();

                with_grad_out!($ndims, size, out, {
                    b.iter(|| black_box({ interpolator.interp_grad(obs, &mut out).unwrap() }));
                });
            },
        );

        $group.bench_with_input(
            BenchmarkId::new(
                format!(
                    "Bspline Regular {}x{}D With Construction, {}",
                    $gridsize, $ndims, scan_or_shuffle
                ),
                $size,
            ),
            $size,
            |b, &size| {
                // Interpolation grid
                let (grids, z) = gen_grid($ndims, $gridsize, 0.0);

                // Observation grid
                let m: usize = ((size as f64).powf(1.0 / ($ndims as f64)) + 2.0) as usize;
                let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                let obs: &[&[f64]; $ndims] = (&obs[..]).try_into().unwrap();
                let mut out = vec![0.0; size];

                let dims = [$gridsize; $ndims];
                let mut starts = [0.0; $ndims];
                let mut steps = [0.0; $ndims];
                (0..$ndims).for_each(|i| starts[i] = grids[i][0]);
                (0..$ndims).for_each(|i| steps[i] = grids[i][1] - grids[i][0]);

                let coeff_len =
                    multibspline::MultiBsplineRegular::<f64, $ndims>::coeff_storage_len(dims);
                let scratch_len =
                    multibspline::MultiBsplineRegular::<f64, $ndims>::construction_scratch_len(
                        dims,
                    );
                let mut coeffs = vec![0.0; coeff_len];
                let mut scratch = vec![0.0; scratch_len];

                b.iter(|| {
                    black_box({
                        let interpolator: multibspline::MultiBsplineRegular<'_, _, $ndims> =
                            multibspline::MultiBsplineRegular::from_values_with_workspace(
                                dims,
                                starts,
                                steps,
                                &z,
                                &mut coeffs,
                                &mut scratch,
                                false,
                            )
                            .unwrap();
                        interpolator.interp(obs, &mut out).unwrap()
                    })
                });
            },
        );

        $group.bench_with_input(
            BenchmarkId::new(
                format!(
                    "Gradient Bspline Regular {}x{}D With Construction, {}",
                    $gridsize, $ndims, scan_or_shuffle
                ),
                $size,
            ),
            $size,
            |b, &size| {
                let (grids, z) = gen_grid($ndims, $gridsize, 0.0);

                let m: usize = ((size as f64).powf(1.0 / ($ndims as f64)) + 2.0) as usize;
                let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                let obs: &[&[f64]; $ndims] = (&obs[..]).try_into().unwrap();

                let dims = [$gridsize; $ndims];
                let mut starts = [0.0; $ndims];
                let mut steps = [0.0; $ndims];
                (0..$ndims).for_each(|i| starts[i] = grids[i][0]);
                (0..$ndims).for_each(|i| steps[i] = grids[i][1] - grids[i][0]);

                let coeff_len = MultiBsplineRegular::<f64, $ndims>::coeff_storage_len(dims);
                let scratch_len =
                    MultiBsplineRegular::<f64, $ndims>::construction_scratch_len(dims);
                let mut coeffs = vec![0.0; coeff_len];
                let mut scratch = vec![0.0; scratch_len];

                with_grad_out!($ndims, size, out, {
                    b.iter(|| {
                        black_box({
                            let interpolator: MultiBsplineRegular<'_, _, $ndims> =
                                MultiBsplineRegular::from_values_with_workspace(
                                    dims,
                                    starts,
                                    steps,
                                    &z,
                                    &mut coeffs,
                                    &mut scratch,
                                    false,
                                )
                                .unwrap();
                            interpolator.interp_grad(obs, &mut out).unwrap()
                        })
                    });
                });
            },
        );

        $group.bench_with_input(
            BenchmarkId::new(
                format!(
                    "Linear Rectilinear {}x{}D MAXDIMS=8, {}",
                    $gridsize, $ndims, scan_or_shuffle
                ),
                $size,
            ),
            $size,
            |b, &size| {
                // Interpolation grid with noise
                let (grids, z) = gen_grid($ndims, $gridsize, 1e-3);

                // Observation grid
                let m: usize = ((size as f64).powf(1.0 / ($ndims as f64)) + 2.0) as usize;
                let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                let mut out = vec![0.0; size];

                // Interpolator inputs
                let gridslice: Vec<&[f64]> = grids.iter().map(|x| &x[..]).collect();

                b.iter(|| {
                    black_box(
                        multilinear::rectilinear::interpn(&gridslice, &z, &obs, &mut out).unwrap(),
                    )
                });
            },
        );

        $group.bench_with_input(
            BenchmarkId::new(
                format!(
                    "Gradient Linear Rectilinear {}x{}D MAXDIMS=8, {}",
                    $gridsize, $ndims, scan_or_shuffle
                ),
                $size,
            ),
            $size,
            |b, &size| {
                let (grids, z) = gen_grid($ndims, $gridsize, 1e-3);

                let m: usize = ((size as f64).powf(1.0 / ($ndims as f64)) + 2.0) as usize;
                let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                let obs: &[&[f64]; $ndims] = (&obs[..]).try_into().unwrap();

                let gridslice: Vec<&[f64]> = grids.iter().map(|x| &x[..]).collect();
                let grids: &[&[f64]; $ndims] = (&gridslice[..]).try_into().unwrap();
                let interpolator: MultilinearRectilinear<'_, _, $ndims> =
                    MultilinearRectilinear::new(grids, &z).unwrap();

                with_grad_out!($ndims, size, out, {
                    b.iter(|| black_box({ interpolator.interp_grad(obs, &mut out).unwrap() }));
                });
            },
        );

        $group.bench_with_input(
            BenchmarkId::new(
                format!(
                    "Gradient Bspline Rectilinear {}x{}D Precomputed Coeffs, {}",
                    $gridsize, $ndims, scan_or_shuffle
                ),
                $size,
            ),
            $size,
            |b, &size| {
                let (grids, z) = gen_grid($ndims, $gridsize, 1e-3);

                let m: usize = ((size as f64).powf(1.0 / ($ndims as f64)) + 2.0) as usize;
                let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                let obs: &[&[f64]; $ndims] = (&obs[..]).try_into().unwrap();

                let gridslice: Vec<&[f64]> = grids.iter().map(|x| &x[..]).collect();
                let grids: &[&[f64]; $ndims] = (&gridslice[..]).try_into().unwrap();
                let dims = [$gridsize; $ndims];
                let coeff_len = MultiBsplineRectilinear::<f64, $ndims>::coeff_storage_len(dims);
                let scratch_len =
                    MultiBsplineRectilinear::<f64, $ndims>::construction_scratch_len(dims);
                let mut coeffs = vec![0.0; coeff_len];
                let mut scratch = vec![0.0; scratch_len];
                multibspline::rectilinear::coefficients(grids, &z, &mut coeffs, &mut scratch)
                    .unwrap();
                let interpolator: MultiBsplineRectilinear<'_, _, $ndims> =
                    MultiBsplineRectilinear::new(grids, &coeffs, false).unwrap();

                with_grad_out!($ndims, size, out, {
                    b.iter(|| black_box({ interpolator.interp_grad(obs, &mut out).unwrap() }));
                });
            },
        );

        $group.bench_with_input(
            BenchmarkId::new(
                format!(
                    "Nearest Regular {}x{}D, {}",
                    $gridsize, $ndims, scan_or_shuffle
                ),
                $size,
            ),
            $size,
            |b, &size| {
                let (grids, z) = gen_grid($ndims, $gridsize, 0.0);
                let m: usize = ((size as f64).powf(1.0 / ($ndims as f64)) + 2.0) as usize;
                let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                let mut out = vec![0.0; size];

                let dims = [$gridsize; $ndims];
                let mut starts = [0.0; $ndims];
                let mut steps = [0.0; $ndims];
                (0..$ndims).for_each(|i| starts[i] = grids[i][0]);
                (0..$ndims).for_each(|i| steps[i] = grids[i][1] - grids[i][0]);

                b.iter(|| {
                    black_box({
                        nearest::regular::interpn(&dims, &starts, &steps, &z, &obs[..], &mut out)
                            .unwrap()
                    })
                });
            },
        );

        $group.bench_with_input(
            BenchmarkId::new(
                format!(
                    "Gradient Cubic Regular {}x{}D MAXDIMS=8, {}",
                    $gridsize, $ndims, scan_or_shuffle
                ),
                $size,
            ),
            $size,
            |b, &size| {
                let (grids, z) = gen_grid($ndims, $gridsize, 0.0);

                let m: usize = ((size as f64).powf(1.0 / ($ndims as f64)) + 2.0) as usize;
                let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                let obs: &[&[f64]; $ndims] = (&obs[..]).try_into().unwrap();

                let dims = [$gridsize; $ndims];
                let mut starts = [0.0; $ndims];
                let mut steps = [0.0; $ndims];
                (0..$ndims).for_each(|i| starts[i] = grids[i][0]);
                (0..$ndims).for_each(|i| steps[i] = grids[i][1] - grids[i][0]);

                let interpolator: MulticubicRegular<'_, _, $ndims> =
                    MulticubicRegular::new(dims, starts, steps, &z, false).unwrap();

                with_grad_out!($ndims, size, out, {
                    b.iter(|| black_box({ interpolator.interp_grad(obs, &mut out).unwrap() }));
                });
            },
        );

        $group.bench_with_input(
            BenchmarkId::new(
                format!(
                    "Nearest Rectilinear {}x{}D, {}",
                    $gridsize, $ndims, scan_or_shuffle
                ),
                $size,
            ),
            $size,
            |b, &size| {
                let (grids, z) = gen_grid($ndims, $gridsize, 1e-3);

                let m: usize = ((size as f64).powf(1.0 / ($ndims as f64)) + 2.0) as usize;
                let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                let mut out = vec![0.0; size];

                let gridslice: Vec<&[f64]> = grids.iter().map(|x| &x[..]).collect();

                b.iter(|| {
                    black_box(
                        nearest::rectilinear::interpn(&gridslice, &z, &obs, &mut out).unwrap(),
                    )
                });
            },
        );

        $group.bench_with_input(
            BenchmarkId::new(
                format!(
                    "Cubic Regular {}x{}D MAXDIMS=8, {}",
                    $gridsize, $ndims, scan_or_shuffle
                ),
                $size,
            ),
            $size,
            |b, &size| {
                // Interpolation grid
                let (grids, z) = gen_grid($ndims, $gridsize, 0.0);

                // Observation grid
                let m: usize = ((size as f64).powf(1.0 / ($ndims as f64)) + 2.0) as usize;
                let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                let mut out = vec![0.0; size];

                let dims = [$gridsize; $ndims];
                let mut starts = [0.0; $ndims];
                let mut steps = [0.0; $ndims];
                (0..$ndims).for_each(|i| starts[i] = grids[i][0]);
                (0..$ndims).for_each(|i| steps[i] = grids[i][1] - grids[i][0]);

                b.iter(|| {
                    black_box({
                        multicubic::regular::interpn(
                            &dims, &starts, &steps, &z, false, &obs, &mut out,
                        )
                        .unwrap()
                    })
                });
            },
        );

        $group.bench_with_input(
            BenchmarkId::new(
                format!(
                    "Cubic Rectilinear {}x{}D MAXDIMS=8, {}",
                    $gridsize, $ndims, scan_or_shuffle
                ),
                $size,
            ),
            $size,
            |b, &size| {
                // Interpolation grid with noise
                let (grids, z) = gen_grid($ndims, $gridsize, 1e-3);

                // Observation grid
                let m: usize = ((size as f64).powf(1.0 / ($ndims as f64)) + 2.0) as usize;
                let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                let mut out = vec![0.0; size];

                // Interpolator inputs
                let gridslice: Vec<&[f64]> = grids.iter().map(|x| &x[..]).collect();

                b.iter(|| {
                    black_box(
                        multicubic::rectilinear::interpn(&gridslice, &z, false, &obs, &mut out)
                            .unwrap(),
                    )
                });
            },
        );

        $group.bench_with_input(
            BenchmarkId::new(
                format!(
                    "Gradient Cubic Rectilinear {}x{}D MAXDIMS=8, {}",
                    $gridsize, $ndims, scan_or_shuffle
                ),
                $size,
            ),
            $size,
            |b, &size| {
                let (grids, z) = gen_grid($ndims, $gridsize, 1e-3);

                let m: usize = ((size as f64).powf(1.0 / ($ndims as f64)) + 2.0) as usize;
                let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                let obs: &[&[f64]; $ndims] = (&obs[..]).try_into().unwrap();

                let gridslice: Vec<&[f64]> = grids.iter().map(|x| &x[..]).collect();
                let grids: &[&[f64]; $ndims] = (&gridslice[..]).try_into().unwrap();
                let interpolator: MulticubicRectilinear<'_, _, $ndims> =
                    MulticubicRectilinear::new(grids, &z, false).unwrap();

                with_grad_out!($ndims, size, out, {
                    b.iter(|| black_box({ interpolator.interp_grad(obs, &mut out).unwrap() }));
                });
            },
        );
    };
}

fn bench_interp(c: &mut Criterion) {
    //
    // Shuffled (un-ordered observation points)
    //
    for gridsize in [100, 1000] {
        let mut group = c.benchmark_group(format!("Interp_1D_Shuffled_{gridsize}-grid"));
        for size in [1, 100, 1_000_000].iter() {
            group.throughput(Throughput::Elements(*size as u64));
            bench_interp_specific!(group, 1, gridsize, size);
        }
        group.finish();
    }

    for gridsize in [100, 1000] {
        let mut group = c.benchmark_group(format!("Interp_2D_Shuffled_{gridsize}-grid"));
        for size in [1, 100, 1_000_000].iter() {
            group.throughput(Throughput::Elements(*size as u64));
            bench_interp_specific!(group, 2, gridsize, size);
        }
        group.finish();
    }

    for gridsize in [10, 100] {
        let mut group = c.benchmark_group(format!("Interp_3D_Shuffled_{gridsize}-grid"));
        for size in [1, 100, 1_000_000].iter() {
            group.throughput(Throughput::Elements(*size as u64));
            bench_interp_specific!(group, 3, gridsize, size);
        }
        group.finish();
    }

    // 1D specialized linear rectilinear
    for gridsize in [10, 1000] {
        let ndims = 1;
        let mut group = c.benchmark_group(format!("Interp_1D_Special_{gridsize}-grid"));
        for size in [1, 100, 1_000_000].iter() {
            group.throughput(Throughput::Elements(*size as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("Linear1D Rect. {}-grid", gridsize), size),
                size,
                |b, &size| {
                    // Interpolation grid with noise
                    let (grids, z) = gen_grid(ndims, gridsize, 1e-3);
                    let grid = RectilinearGrid1D::new(&grids[0], &z).unwrap();

                    // Observation grid
                    let m: usize = ((size as f64).powf(1.0 / (ndims as f64)) + 2.0) as usize;
                    let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                    let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                    let mut out = vec![0.0; size];

                    b.iter(|| {
                        black_box({
                            let interp = Linear1D::new(grid);
                            interp.eval(&obs[0], &mut out).unwrap()
                        })
                    });
                },
            );
        }
        group.finish();
    }

    // 1D specialized linear regular
    for gridsize in [10, 1000] {
        let ndims = 1;
        let mut group = c.benchmark_group(format!("Interp_1D_Special_{gridsize}-grid"));
        for size in [1, 100, 1_000_000].iter() {
            group.throughput(Throughput::Elements(*size as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("Linear1D Reg. {}-grid", gridsize), size),
                size,
                |b, &size| {
                    // Interpolation grid with noise
                    let (grids, z) = gen_grid(ndims, gridsize, 0.0);
                    let grid =
                        RegularGrid1D::new(grids[0][0], grids[0][1] - grids[0][0], &z).unwrap();

                    // Observation grid
                    let m: usize = ((size as f64).powf(1.0 / (ndims as f64)) + 2.0) as usize;
                    let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                    let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                    let mut out = vec![0.0; size];

                    b.iter(|| {
                        black_box({
                            let interp = Linear1D::new(grid);
                            interp.eval(&obs[0], &mut out).unwrap()
                        })
                    });
                },
            );
        }
        group.finish();
    }

    // 1D specialized linear hold-last regular
    for gridsize in [10, 1000] {
        let ndims = 1;
        let mut group = c.benchmark_group(format!("Interp_1D_Special_{gridsize}-grid"));
        for size in [1, 100, 1_000_000].iter() {
            group.throughput(Throughput::Elements(*size as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("LinearHoldLast1D Reg. {}-grid", gridsize), size),
                size,
                |b, &size| {
                    // Interpolation grid with noise
                    let (grids, z) = gen_grid(ndims, gridsize, 0.0);
                    let grid =
                        RegularGrid1D::new(grids[0][0], grids[0][1] - grids[0][0], &z).unwrap();

                    // Observation grid
                    let m: usize = ((size as f64).powf(1.0 / (ndims as f64)) + 2.0) as usize;
                    let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                    let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                    let mut out = vec![0.0; size];

                    b.iter(|| {
                        black_box({
                            let interp = LinearHoldLast1D::new(grid);
                            interp.eval(&obs[0], &mut out).unwrap()
                        })
                    });
                },
            );
        }
        group.finish();
    }

    // 1D specialized hold-left
    for gridsize in [10, 1000] {
        let ndims = 1;
        let mut group = c.benchmark_group(format!("Interp_1D_Special_{gridsize}-grid"));
        for size in [1, 100, 1_000_000].iter() {
            group.throughput(Throughput::Elements(*size as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("Left1D Reg. {}-grid", gridsize), size),
                size,
                |b, &size| {
                    // Interpolation grid with noise
                    let (grids, z) = gen_grid(ndims, gridsize, 0.0);
                    let grid =
                        RegularGrid1D::new(grids[0][0], grids[0][1] - grids[0][0], &z).unwrap();

                    // Observation grid
                    let m: usize = ((size as f64).powf(1.0 / (ndims as f64)) + 2.0) as usize;
                    let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                    let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                    let mut out = vec![0.0; size];

                    b.iter(|| {
                        black_box({
                            let interp = Left1D::new(grid);
                            interp.eval(&obs[0], &mut out).unwrap()
                        })
                    });
                },
            );
        }
        group.finish();
    }

    // 1D specialized nearest
    for gridsize in [10, 1000] {
        let ndims = 1;
        let mut group = c.benchmark_group(format!("Interp_1D_Special_{gridsize}-grid"));
        for size in [1, 100, 1_000_000].iter() {
            group.throughput(Throughput::Elements(*size as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("Nearest1D Reg. {}-grid", gridsize), size),
                size,
                |b, &size| {
                    // Interpolation grid with noise
                    let (grids, z) = gen_grid(ndims, gridsize, 0.0);
                    let grid =
                        RegularGrid1D::new(grids[0][0], grids[0][1] - grids[0][0], &z).unwrap();

                    // Observation grid
                    let m: usize = ((size as f64).powf(1.0 / (ndims as f64)) + 2.0) as usize;
                    let gridobs_t = gen_interp_obs_grid(&grids, m, true);
                    let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                    let mut out = vec![0.0; size];

                    b.iter(|| {
                        black_box({
                            let interp = Nearest1D::new(grid);
                            interp.eval(&obs[0], &mut out).unwrap()
                        })
                    });
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches_interp, bench_interp);
criterion_main!(benches_interp);

mod randn {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::distr::StandardUniform;
    use rand::rngs::StdRng;

    /// Fixed random seed to support repeatable testing
    const SEED: [u8; 32] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7,
        6, 5, 4, 3, 2, 1,
    ];

    /// Get a random number generator with a const seed for repeatable testing
    pub fn rng_fixed_seed() -> StdRng {
        StdRng::from_seed(SEED)
    }

    /// Generate `n` random numbers using provided generator
    pub fn randn<T>(rng: &mut StdRng, n: usize) -> Vec<T>
    where
        StandardUniform: rand::distr::Distribution<T>,
    {
        std::iter::repeat_with(|| rng.random::<T>())
            .take(n)
            .collect()
    }
}

mod gridgen {
    use super::randn::*;
    use interpn::utils::*;
    use rand::seq::SliceRandom;

    // Generate a (potentially irregular) grid to interpolate on,
    // and some fake data values.
    pub fn gen_grid(ndims: usize, size: usize, noise: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = rng_fixed_seed();
        let n = size.pow(ndims as u32);
        let z = randn::<f64>(&mut rng, n);

        let grids: Vec<Vec<f64>> = (0..ndims)
            .map(|_| {
                let mut x = linspace(0.0, 100.0, size);
                if noise > 0.0 {
                    let dx = randn::<f64>(&mut rng, size);
                    (0..size).for_each(|i| x[i] = x[i] + (dx[i] - 0.5) * noise);
                }
                x
            })
            .collect();

        (grids, z)
    }

    // Generate a set of either sequential (scanning) or shuffled
    // observation points that are entirely inside the interpolation grid.
    //
    // `size` is the size per grid, so the total number of points will be size.pow(ndims).
    pub fn gen_interp_obs_grid(
        grids: &Vec<Vec<f64>>,
        size: usize,
        shuffled: bool,
    ) -> Vec<Vec<f64>> {
        let mut rng = rng_fixed_seed();
        let ndims = grids.len();

        let xobs: Vec<Vec<f64>> = (0..ndims)
            .map(|i| linspace(grids[i][1], grids[i][grids[i].len() - 2], size))
            .collect();
        let gridobs = meshgrid((0..ndims).map(|i| &xobs[i]).collect());
        let mut gridobs_t: Vec<Vec<f64>> = (0..ndims)
            .map(|i| gridobs.iter().map(|x| x[i]).collect())
            .collect(); // transpose
        if shuffled {
            (0..ndims).for_each(|i| gridobs_t[i].shuffle(&mut rng));
        }
        // unpack like:
        // let xobsslice: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
        gridobs_t
    }
}
