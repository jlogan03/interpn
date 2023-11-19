#![allow(clippy::all)] // Clippy will attempt to remove black_box() internals

use criterion::*;
use interpn::utils::*;
use interpn::{multilinear, RegularGridInterpolator};
use randn::*;

fn bench_interp(c: &mut Criterion) {
    for gridsize in [10_usize, 100, 1000] {
        let mut group = c.benchmark_group(format!("bench_interp_scanning_{gridsize}x{gridsize}"));
        for size in [100, 10_000, 1_000_000].iter() {
            group.throughput(Throughput::Elements(*size as u64));

            group.bench_with_input(
                BenchmarkId::new(
                    format!("{gridsize}x{gridsize} Regular 2D MAXDIMS=2, Scanning"),
                    size,
                ),
                size,
                |b, &size| {
                    let mut rng = rng_fixed_seed();
                    let m: usize = (size as f64).sqrt() as usize;
                    let nx = 10;
                    let ny = 10;
                    let n = nx * ny;

                    let x = linspace(0.0, 100.0, nx);
                    let y = linspace(0.0, 100.0, ny);
                    let z = randn::<f64>(&mut rng, n);

                    // Choose observation points to be concretely inside the grid
                    // in order to avoid mixing interpolation perf with extrapolation
                    // for the points that are right on the bound.
                    let xgridobs = linspace(x[1], x[nx - 2], m);
                    let ygridobs = linspace(y[1], y[ny - 2], m);
                    let obsgrid = meshgrid(Vec::from([&xgridobs, &ygridobs]));
                    let xobs: Vec<f64> = obsgrid.iter().map(|xyi| xyi[0]).collect();
                    let yobs: Vec<f64> = obsgrid.iter().map(|xyi| xyi[1]).collect();
                    let obs = &[&xobs[..], &yobs[..]];

                    let mut out = vec![0.0; size];

                    b.iter(|| {
                        black_box({
                            // Initializing the interpolator local to where it will be used
                            // is about a 2x speedup
                            let dims = [nx, ny];
                            let starts = [x[0], y[0]];
                            let steps = [x[1] - x[0], y[1] - y[0]];
                            let interpolator: RegularGridInterpolator<'_, _, 2> =
                                RegularGridInterpolator::new(&dims, &starts, &steps, &z);
                            interpolator.interp(obs, &mut out)
                        })
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    format!("{gridsize}x{gridsize} Regular 2D MAXDIMS=8, Scanning"),
                    size,
                ),
                size,
                |b, &size| {
                    let mut rng = rng_fixed_seed();
                    let m: usize = (size as f64).sqrt() as usize;
                    let nx = 10;
                    let ny = 10;
                    let n = nx * ny;

                    let x = linspace(0.0, 100.0, nx);
                    let y = linspace(0.0, 100.0, ny);
                    let z = randn::<f64>(&mut rng, n);

                    // Choose observation points to be concretely inside the grid
                    // in order to avoid mixing interpolation perf with extrapolation
                    // for the points that are right on the bound.
                    let xgridobs = linspace(x[1], x[nx - 2], m);
                    let ygridobs = linspace(y[1], y[ny - 2], m);
                    let obsgrid = meshgrid(Vec::from([&xgridobs, &ygridobs]));
                    let xobs: Vec<f64> = obsgrid.iter().map(|xyi| xyi[0]).collect();
                    let yobs: Vec<f64> = obsgrid.iter().map(|xyi| xyi[1]).collect();
                    let obs = &[&xobs[..], &yobs[..]];

                    let mut out = vec![0.0; size];

                    b.iter(|| {
                        black_box({
                            let dims = [nx, ny];
                            let starts = [x[0], y[0]];
                            let steps = [x[1] - x[0], y[1] - y[0]];
                            multilinear::regular::interpn(&dims, &starts, &steps, &z, obs, &mut out)
                        })
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(
                    format!("{gridsize}x{gridsize} Rectilinear 2D MAXDIMS=8, Scanning"),
                    size,
                ),
                size,
                |b, &size| {
                    let mut rng = rng_fixed_seed();
                    let m: usize = (size as f64).sqrt() as usize;
                    let nx = 10;
                    let ny = 10;
                    let n = nx * ny;

                    let mut x = linspace(0.0, 100.0, nx);
                    let mut y = linspace(0.0, 100.0, ny);

                    // Add noise to the grid
                    let dx = randn::<f64>(&mut rng, nx);
                    let dy = randn::<f64>(&mut rng, ny);
                    (0..nx).for_each(|i| x[i] = x[i] + (dx[i] - 0.5) / 1e3);
                    (0..ny).for_each(|i| y[i] = y[i] + (dy[i] - 0.5) / 1e3);

                    // Make sure the grid is still monotonic
                    (0..nx - 1).for_each(|i| assert!(x[i + 1] > x[i]));
                    (0..ny - 1).for_each(|i| assert!(y[i + 1] > y[i]));

                    let z = randn::<f64>(&mut rng, n);

                    // Choose observation points to be concretely inside the grid
                    // in order to avoid mixing interpolation perf with extrapolation
                    // for the points that are right on the bound.
                    let xgridobs = linspace(x[1], x[nx - 2], m);
                    let ygridobs = linspace(y[1], y[ny - 2], m);
                    let obsgrid = meshgrid(Vec::from([&xgridobs, &ygridobs]));
                    let xobs: Vec<f64> = obsgrid.iter().map(|xyi| xyi[0]).collect();
                    let yobs: Vec<f64> = obsgrid.iter().map(|xyi| xyi[1]).collect();
                    let obs = &[&xobs[..], &yobs[..]];

                    let mut out = vec![0.0; size];

                    b.iter(|| {
                        black_box(multilinear::rectilinear::interpn(
                            &[&x, &y],
                            &z,
                            obs,
                            &mut out,
                        ))
                    });
                },
            );
        }
        group.finish();
    }
}

fn bench_extrap(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_extrap");
    for size in [100, 10_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::new("hypercube regular interpn 2d max 8d", size),
            size,
            |b, &size| {
                let mut rng = rng_fixed_seed();
                let m: usize = (size as f64).sqrt() as usize;
                let nx = m / 2;
                let ny = m * 2;
                let n = nx * ny;

                let x = linspace(0.0, 100.0, nx);
                let y = linspace(0.0, 100.0, ny);

                // Grid to extrapolate onto,
                // entirely in corner-region for worst-case perf
                let xw = linspace(101.0, 200.0, nx);
                let yw = linspace(-100.0, -1.0, ny);
                let obsgrid = meshgrid(Vec::from([&xw, &yw]));
                let xobs: Vec<f64> = obsgrid.iter().map(|xyi| xyi[0]).collect();
                let yobs: Vec<f64> = obsgrid.iter().map(|xyi| xyi[1]).collect();
                let obs = &[&xobs[..], &yobs[..]];

                let z = randn::<f64>(&mut rng, n);
                let mut out = vec![0.0; n];

                b.iter(|| {
                    black_box({
                        let dims = [nx, ny];
                        let starts = [x[0], y[0]];
                        let steps = [x[1] - x[0], y[1] - y[0]];
                        multilinear::regular::interpn(&dims, &starts, &steps, &z, obs, &mut out)
                    })
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hypercube rectilinear interpn 2d max 8d", size),
            size,
            |b, &size| {
                let mut rng = rng_fixed_seed();
                let m: usize = (size as f64).sqrt() as usize;
                let nx = m / 2;
                let ny = m * 2;
                let n = nx * ny;

                let mut x = linspace(0.0, 100.0, nx);
                let mut y = linspace(0.0, 100.0, ny);

                // Add noise to the grid
                let dx = randn::<f64>(&mut rng, nx);
                let dy = randn::<f64>(&mut rng, ny);
                (0..nx).for_each(|i| x[i] = x[i] + (dx[i] - 0.5) / 1e3);
                (0..ny).for_each(|i| y[i] = y[i] + (dy[i] - 0.5) / 1e3);

                // Make sure the grid is still monotonic
                (0..nx - 1).for_each(|i| assert!(x[i + 1] > x[i]));
                (0..ny - 1).for_each(|i| assert!(y[i + 1] > y[i]));

                // Grid to extrapolate onto,
                // entirely in corner-region for worst-case perf
                let xw = linspace(101.0, 200.0, nx);
                let yw = linspace(-100.0, -1.0, ny);

                let obsgrid = meshgrid(Vec::from([&xw, &yw]));
                let xobs: Vec<f64> = obsgrid.iter().map(|xyi| xyi[0]).collect();
                let yobs: Vec<f64> = obsgrid.iter().map(|xyi| xyi[1]).collect();
                let obs = &[&xobs[..], &yobs[..]];

                let z = randn::<f64>(&mut rng, n);
                let mut out = vec![0.0; n];

                b.iter(|| {
                    black_box(multilinear::rectilinear::interpn(
                        &[&x, &y],
                        &z,
                        obs,
                        &mut out,
                    ))
                });
            },
        );
    }
    group.finish();
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
