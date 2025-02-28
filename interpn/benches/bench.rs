#![allow(clippy::all)] // Clippy will attempt to remove black_box() internals

use criterion::*;
use gridgen::*;
use interpn::{
    multicubic, multilinear, one_dim::Interp1D, Linear1D, MultilinearRegular, RectilinearGrid1D, RegularGrid1D,
};

enum Kind {
    Interp,
    Extrap,
}

macro_rules! bench_interp_specific {
    ($group:ident, $ndims:expr, $gridsize:expr, $size:expr, $kind:expr) => {
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
                let gridobs_t = match $kind {
                    Kind::Interp => gen_interp_obs_grid(&grids, m, true),
                    Kind::Extrap => gen_extrap_obs_grid(&grids, m, true),
                };
                let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                let mut out = vec![0.0; size];

                let dims = [$gridsize; $ndims];
                let mut starts = [0.0; $ndims];
                let mut steps = [0.0; $ndims];
                (0..$ndims).for_each(|i| starts[i] = grids[i][0]);
                (0..$ndims).for_each(|i| steps[i] = grids[i][1] - grids[i][0]);

                b.iter(|| {
                    black_box({
                        let interpolator: MultilinearRegular<'_, _, $ndims> =
                            MultilinearRegular::new(&dims, &starts, &steps, &z).unwrap();
                        interpolator.interp(&obs, &mut out).unwrap()
                    })
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
                let gridobs_t = match $kind {
                    Kind::Interp => gen_interp_obs_grid(&grids, m, true),
                    Kind::Extrap => gen_extrap_obs_grid(&grids, m, true),
                };
                let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                let mut out = vec![0.0; size];

                let dims = [$gridsize; $ndims];
                let mut starts = [0.0; $ndims];
                let mut steps = [0.0; $ndims];
                (0..$ndims).for_each(|i| starts[i] = grids[i][0]);
                (0..$ndims).for_each(|i| steps[i] = grids[i][1] - grids[i][0]);

                b.iter(|| {
                    black_box({
                        multilinear::regular::interpn(&dims, &starts, &steps, &z, &obs, &mut out)
                            .unwrap()
                    })
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
                let gridobs_t = match $kind {
                    Kind::Interp => gen_interp_obs_grid(&grids, m, true),
                    Kind::Extrap => gen_extrap_obs_grid(&grids, m, true),
                };
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
                let gridobs_t = match $kind {
                    Kind::Interp => gen_interp_obs_grid(&grids, m, true),
                    Kind::Extrap => gen_extrap_obs_grid(&grids, m, true),
                };
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
                let gridobs_t = match $kind {
                    Kind::Interp => gen_interp_obs_grid(&grids, m, true),
                    Kind::Extrap => gen_extrap_obs_grid(&grids, m, true),
                };
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
            bench_interp_specific!(group, 1, gridsize, size, Kind::Interp);
        }
        group.finish();
    }

    for gridsize in [100, 1000] {
        let mut group = c.benchmark_group(format!("Interp_2D_Shuffled_{gridsize}-grid"));
        for size in [1, 100, 1_000_000].iter() {
            group.throughput(Throughput::Elements(*size as u64));
            bench_interp_specific!(group, 2, gridsize, size, Kind::Interp);
        }
        group.finish();
    }

    for gridsize in [10, 100] {
        let mut group = c.benchmark_group(format!("Interp_3D_Shuffled_{gridsize}-grid"));
        for size in [1, 100, 1_000_000].iter() {
            group.throughput(Throughput::Elements(*size as u64));
            bench_interp_specific!(group, 3, gridsize, size, Kind::Interp);
        }
        group.finish();
    }

    // 1D specialized methods
    for gridsize in [10, 1000] {
        let kind = Kind::Interp;
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
                    let gridobs_t = match kind {
                        Kind::Interp => gen_interp_obs_grid(&grids, m, true),
                        Kind::Extrap => gen_extrap_obs_grid(&grids, m, true),
                    };
                    let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                    let mut out = vec![0.0; size];

                    // Interpolator inputs
                    // let gridslice: Vec<&[f64]> = grids.iter().map(|x| &x[..]).collect();

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

    for gridsize in [10, 1000] {
        let kind = Kind::Interp;
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
                    let grid = RegularGrid1D::new(grids[0][0], grids[0][1] - grids[0][0], &z).unwrap();

                    // Observation grid
                    let m: usize = ((size as f64).powf(1.0 / (ndims as f64)) + 2.0) as usize;
                    let gridobs_t = match kind {
                        Kind::Interp => gen_interp_obs_grid(&grids, m, true),
                        Kind::Extrap => gen_extrap_obs_grid(&grids, m, true),
                    };
                    let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
                    let mut out = vec![0.0; size];

                    // Interpolator inputs
                    // let gridslice: Vec<&[f64]> = grids.iter().map(|x| &x[..]).collect();

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
}

fn bench_extrap(c: &mut Criterion) {
    //
    // Shuffled (un-ordered observation points)
    //
    for gridsize in [10] {
        let mut group = c.benchmark_group(format!("Extrap_1D_Shuffled_{gridsize}-grid"));
        for size in [1, 100, 1_000_000].iter() {
            group.throughput(Throughput::Elements(*size as u64));
            bench_interp_specific!(group, 1, gridsize, size, Kind::Extrap);
        }
        group.finish();
    }

    for gridsize in [10] {
        let mut group = c.benchmark_group(format!("Extrap_2D_Shuffled_{gridsize}-grid"));
        for size in [1, 100, 1_000_000].iter() {
            group.throughput(Throughput::Elements(*size as u64));
            bench_interp_specific!(group, 2, gridsize, size, Kind::Extrap);
        }
        group.finish();
    }

    for gridsize in [10] {
        let mut group = c.benchmark_group(format!("Extrap_3D_Shuffled_{gridsize}-grid"));
        for size in [1, 100, 1_000_000].iter() {
            group.throughput(Throughput::Elements(*size as u64));
            bench_interp_specific!(group, 3, gridsize, size, Kind::Extrap);
        }
        group.finish();
    }
}

criterion_group!(benches_interp, bench_interp);
criterion_group!(benches_extrap, bench_extrap);
criterion_main!(benches_interp, benches_extrap,);

mod randn {
    use rand::distributions::{Distribution, Standard};
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

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
        Standard: Distribution<T>,
    {
        let out: Vec<T> = (0..n).map(|_| rng.gen::<T>()).collect();
        out
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

    // Generate a set of observation points that are entirely outside
    // the interpolation grid on every axis (meaning, in a corner region,
    // the worst case for perf).
    //
    // `size` is the size per grid, so the total number of points will be size.pow(ndims).
    pub fn gen_extrap_obs_grid(
        grids: &Vec<Vec<f64>>,
        size: usize,
        _shuffled: bool,
    ) -> Vec<Vec<f64>> {
        let ndims = grids.len();

        let xobs: Vec<Vec<f64>> = (0..ndims)
            .map(|i| {
                linspace(
                    grids[i].last().unwrap() + 1.0,
                    grids[i].last().unwrap() + 2.0,
                    size,
                )
            })
            .collect();
        let gridobs = meshgrid((0..ndims).map(|i| &xobs[i]).collect());
        let gridobs_t: Vec<Vec<f64>> = (0..ndims)
            .map(|i| gridobs.iter().map(|x| x[i]).collect())
            .collect(); // transpose

        // unpack like:
        // let obs: Vec<&[f64]> = gridobs_t.iter().map(|x| &x[..size]).collect();
        gridobs_t
    }
}
